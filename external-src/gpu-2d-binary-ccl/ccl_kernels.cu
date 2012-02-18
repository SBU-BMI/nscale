#include "CUDAFunctionsBase.h"

texture<uchar, 2, cudaReadModeElementType> segmentedTex;

#define COMPARE_NEIGHBORING_REGION_FROM_ADDR(segData, nSegData, label, neighLabel, buf)  if(nSegData == segData) {  label = min(label, buf[neighLabel]);}

//----------------------------------------
// FINDROOT and UNION functions
//----------------------------------------

inline
__device__ int findRoot(int* buf, int x) {
	int nextX;
    do {
	  nextX = x;
      x = buf[nextX];
    } while (x < nextX);
    return x;    
}

inline
__device__ void unionF(int* buf, uchar seg1, uchar seg2, int reg1, int reg2, int* changed)
{
	if(seg1 == seg2) {			
		int newReg1 = findRoot(buf, reg1);		
		int newReg2 = findRoot(buf, reg2);	
	
		if(newReg1 > newReg2) {			
			atomicMin(buf+newReg1, newReg2);		
			changed[0] = 1;			
		} else if(newReg2 > newReg1) {		
			atomicMin(buf+newReg2, newReg1);		
			changed[0] = 1;
		}			
	} 	
}

//----------------------------------------
// END FINDROOT and UNION functions
//----------------------------------------


//----------------------------------------
//KERNEL 1
//----------------------------------------

__global__
void solveCCLSharedKernel(		 int* dLabelsOut, const int pitch,			
					  			 uchar* dSegIn, const int segPitch,
								 const int dataWidth, const int sMemSegOff, const int connectivity)
{	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
	int sPitch = blockDim.x+2;	
	//local index of the processed element = address of the elment in the shared memory - used as a label for the element
	int localIndex = threadIdx.x + 1 + (threadIdx.y+1) * sPitch;	
	int newLabel = localIndex;
	int oldLabel = 0;   
	int index = x+ y * pitch;   
	//define address of the segment data in the shared memory for the given element
	int segLocalIndex = localIndex + sMemSegOff;

	//shared memory used to store both the labels and the segment data
	extern __shared__ int sMem[];   
	//shared flag that is used to check for the final solution on the processed tile 
	//if there are any two connected elements with different labels the flag is set to 1
	__shared__ int sChanged[1];
	//load segment data into the shared memory 
	
	//first initialize boundaries around the computed tile
	//as we to compute only a local solution we initialize the segments on boundaries to 0 == background

	if(threadIdx.x == blockDim.x-1) {	
		sMem[localIndex+1] = 0;
		sMem[segLocalIndex+1] = 0;
	}
	if(threadIdx.x == 0) {	
		sMem[localIndex-1] = 0;
		sMem[segLocalIndex-1] = 0;
	}
	if(threadIdx.y == blockDim.y-1) {			
		sMem[localIndex+sPitch] = 0;
		sMem[segLocalIndex+sPitch] = 0;

		if(threadIdx.x == 0) {			
			sMem[localIndex+sPitch-1] = 0;
			sMem[segLocalIndex+sPitch-1] = 0;
		}
		if(threadIdx.x == blockDim.x-1) {			
			sMem[localIndex+sPitch+1] = 0;
			sMem[segLocalIndex+sPitch+1] = 0;
		}	
	}	
	if(threadIdx.y == 0) {			
		sMem[localIndex-sPitch] = 0;
		sMem[segLocalIndex-sPitch] = 0;
		if(threadIdx.x == 0) {			
			sMem[localIndex-sPitch-1] = 0;
			sMem[segLocalIndex-sPitch-1] = 0;
		}
		if(threadIdx.x == blockDim.x-1) {			
			sMem[localIndex-sPitch+1] = 0;
			sMem[segLocalIndex-sPitch+1] = 0;
		}	
	}	
	uchar seg;
	uchar nSeg[8];
	
	//load the segment data for the processed element
	seg = dSegIn[x+y*segPitch];

	sMem[segLocalIndex] = seg;
	__syncthreads();
	//store data about segments into registers so that we don't have to access shared memory
	//(the data are never modified)
	if (connectivity == 8) {
		nSeg[0] = sMem[segLocalIndex-sPitch-1];
		nSeg[2] = sMem[segLocalIndex-sPitch+1];
		nSeg[5] = sMem[segLocalIndex+sPitch-1];
		nSeg[7] = sMem[segLocalIndex+sPitch+1];
	}	
	nSeg[1] = sMem[segLocalIndex-sPitch];
	nSeg[3] = sMem[segLocalIndex-1];
	nSeg[4] = sMem[segLocalIndex+1];
	nSeg[6] = sMem[segLocalIndex+sPitch];
		    		
	while(1)
	{						
		//update the label of the selement
		//in first pass the newLabel is equal to the local address of the element
		sMem[localIndex] = newLabel;
		//reset the check flag 
		if((threadIdx.x | threadIdx.y) == 0) sChanged[0] = 0;
		oldLabel = newLabel;
		__syncthreads();
		//if the element is not a background, compare the element's label with its neighbors
		if(seg != 0) {				
			//compare with all elements in the 8-neighborhood
					
			COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[1], newLabel, localIndex-sPitch, sMem)		
			COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[3], newLabel, localIndex-1, sMem)
			COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[4], newLabel, localIndex+1, sMem)		
			COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[6], newLabel, localIndex+sPitch, sMem)	
	
			if (connectivity == 8) {
				COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[0], newLabel, localIndex-sPitch-1, sMem)			
				COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[2], newLabel, localIndex-sPitch+1, sMem)  
				COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[5], newLabel, localIndex+sPitch-1, sMem)		
				COMPARE_NEIGHBORING_REGION_FROM_ADDR(seg, nSeg[7], newLabel, localIndex+sPitch+1, sMem)
			}
		}							
		__syncthreads();			
		
		if(oldLabel > newLabel) {
			//if there is a neigboring element with a smaller label, update the equivalence tree of the processed element
			//(the tree is always flattened in this stage so there is no need to use findRoot to find the root)
			atomicMin(sMem+oldLabel, newLabel);
			//set the flag to 1 -> it is necessary to perform another iteration of the CCL solver
			sChanged[0] = 1;
		}		
		__syncthreads();
		//if no equivalence tree was updated, the local solution is complete
		if(sChanged[0] == 0) break;		
		//flatten the equivalence tree
		newLabel = findRoot(sMem,newLabel);			
		__syncthreads();

	}
	if(seg == 0) newLabel = -1;	
	else {
		//transfer the label into global coordinates 
		y = newLabel / (blockDim.x+2);
		x = newLabel - y*(blockDim.x+2);
		x = blockIdx.x*blockDim.x + x-1;
		y = blockIdx.y*blockDim.y + y-1;
		newLabel = x+y*dataWidth;	
	}
    dLabelsOut[index] = newLabel;	    
}

//----------------------------------------
// END KERNEL 1
//----------------------------------------

//----------------------------------------
// KERNEL 2
//----------------------------------------

__global__
void mergeEquivalenceTreesOnBordersKernel(   int* dLabelsInOut, const int pitch,											 						  			 
											 int* dChanged,						
											 const int tileDim, const int connectivity
											 )
{
	//local tileX and Y are stored directly in blockIdx.x and blockIdx.x
	//all threads for each block are stored in the z-dir of each block (threadIdx.z)
	int tileX = threadIdx.x + blockIdx.x * blockDim.x;	
	int tileY = threadIdx.y + blockIdx.y * blockDim.y;
	//the number of times each thread has to be used to process one border of the tile
	int threadIterations = tileDim / blockDim.z;
	//dimensions of the tile on the next level of the merging scheme
	int nextTileDim = tileDim * blockDim.x;
	
	uchar seg;
	int offset;
	
	//shared variable that is set to 1 if an equivalence tree was changed
	__shared__ int sChanged[1];
	while(1) {		
		//reset the check variable
		if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
			sChanged[0] = 0;			
		 }		
		__syncthreads();
		//first process horizontal borders that are between merged tiles (so exclude tiles from the last row)
		if(threadIdx.y < blockDim.y-1) {
			//the horizontal border corresponds to the last row of the tile
			uint y = (tileY+1)*tileDim-1;	
			//offset of the element from the left most boundary of the tile
			offset = threadIdx.x*tileDim + threadIdx.z;
			uint x = tileX*tileDim + threadIdx.z;	
			for(int i=0;i<threadIterations;++i) {					
				//load the segment data for the element
				seg = tex2D(segmentedTex, x, y); 
				if(seg != 0) {		
					//address of the element in the global space
					int idx = x+y*pitch;				
					//perform the union operation on neigboring elements from other tiles that are to be merged with the processed tile
					if(connectivity == 8 && offset>0) unionF(dLabelsInOut, seg, tex2D(segmentedTex, x-1, y+1), idx, idx-1+pitch, sChanged);
					unionF(dLabelsInOut, seg, tex2D(segmentedTex, x, y+1), idx, idx+pitch, sChanged);
					if(connectivity == 8 && offset<nextTileDim-1) unionF(dLabelsInOut, seg, tex2D(segmentedTex, x+1, y+1), idx, idx+1+pitch, sChanged);
					
				}
				//set the processed element to the next in line on the same boundary (in case the threads are used for multiple elements on the boundary)
				x += blockDim.z;
				offset += blockDim.z;
			}
		}
		//process vertical borders between merged tiles (exclude the right most tiles)
		if(threadIdx.x < blockDim.x-1) {
			//the vertical border corresponds to the right most column of elements in the tile
			uint x = (tileX+1)*tileDim-1;		
			//offset of the element from the top most boundary of the tile
			offset = threadIdx.y*tileDim + threadIdx.z;
			uint y = tileY*tileDim+threadIdx.z;
			for(int i=0;i<threadIterations;++i) {			
				//load the segment data for the element
				seg = tex2D(segmentedTex, x, y); 
				if(seg != 0) {
					int idx = x+y*pitch;
					//perform the union operation on neigboring elements from other tiles that are to be merged with the processed tile
					if(connectivity == 8 && offset>0) unionF(dLabelsInOut, seg, tex2D(segmentedTex, x+1, y-1), idx, idx+1-pitch, sChanged);
					unionF(dLabelsInOut, seg, tex2D(segmentedTex, x+1, y), idx, idx+1, sChanged);
					if(connectivity == 8 && offset<nextTileDim-1) unionF(dLabelsInOut, seg, tex2D(segmentedTex, x+1, y+1), idx, idx+1+pitch, sChanged);			
				}	
				y += blockDim.z;
				offset += blockDim.z;
			}		
		}		
		__syncthreads();
		//if no equivalence tree was updated then all equivalence trees of the merged tiles are already merged
		if(sChanged[0] == 0) 		
			break;	
		//need to synchronize here because the sChanged variable is changed next
		__syncthreads();
	}	
}

//----------------------------------------
// END KERNEL 2
//----------------------------------------

//----------------------------------------
// KERNEL 3 + 4 Flattening function
//----------------------------------------

inline __device__
void flattenEquivalenceTreesInternal(int x, int y, int* dLabelsOut, int* dLabelsIn, uint pitch, const int dataWidth, const int log2DataWidth)
{
	int index = x+y*pitch;	
	int label = dLabelsIn[index];
	//flatten the tree for all non-background elements whose labels are not roots of the equivalence tree 
	if(label != index && label != -1)
	{
		int newLabel = findRoot(dLabelsIn, label);			
		if(newLabel < label) 
		{		
			//set the label of the root element as the label of the processed element			
			dLabelsOut[index] = newLabel;
		}
	}		
}

//----------------------------------------
// END KERNEL 3 + 4 Flattening function
//----------------------------------------

//----------------------------------------
// KERNEL 3 
//----------------------------------------

__global__
void flattenEquivalenceTreesAfterMergingTilesKernel(
										int* dLabelsOut, int* dLabelsIn, uint pitch, const int dataWidth, const int log2DataWidth,
										const int tileDim, const int blocksPerTile
									)
{
	//multiple thread blocks can be used to update border of a single tile
	int tileX = blockIdx.x / blocksPerTile;
	int tileOffset = blockDim.x*(blockIdx.x & (blocksPerTile-1));
	int tileY = threadIdx.y + (blockIdx.y*blockDim.y);
	int maxTileY = gridDim.y*blockDim.y-1;	
	
	//a single thread is used to update both the horizontal and the verical boundary on both sides of two merged tiles	

	//first process horizontal borders
	if(tileY < maxTileY) {		
		uint y = (tileY+1)*tileDim-1;	
		uint x = tileX*tileDim+threadIdx.x+tileOffset;			
		flattenEquivalenceTreesInternal(x, y, dLabelsOut, dLabelsIn, pitch, dataWidth, log2DataWidth);
		flattenEquivalenceTreesInternal(x, y+1, dLabelsOut, dLabelsIn, pitch, dataWidth, log2DataWidth);
	}
	//process vertical borders
	if(tileX < gridDim.x-1) {		
		uint x = (tileX+1)*tileDim-1;		
		uint y = tileY*tileDim+threadIdx.x+tileOffset;
		flattenEquivalenceTreesInternal(x, y, dLabelsOut, dLabelsIn, pitch, dataWidth, log2DataWidth);
		flattenEquivalenceTreesInternal(x+1, y, dLabelsOut, dLabelsIn, pitch, dataWidth, log2DataWidth);
	}	
}

//----------------------------------------
// END KERNEL 3 
//----------------------------------------

//----------------------------------------
// KERNEL 4
//----------------------------------------

__global__
void flattenEquivalenceTreesKernel(int* dLabelsOut, int* dLabelsIn, uint pitch, const int dataWidth, const int log2DataWidth)												
{
	//flatten the equivalence trees on all elements
	uint x = (blockIdx.x*blockDim.x)+threadIdx.x;
    uint y = (blockIdx.y*blockDim.y)+threadIdx.y;  
	flattenEquivalenceTreesInternal(x, y, dLabelsOut, dLabelsIn, pitch, dataWidth, log2DataWidth);
}													
												    
//----------------------------------------
// END KERNEL 4
//----------------------------------------

//----------------------------------------
// KERNELS used for reindexing of the labels
//----------------------------------------

__global__ void prepareLabelsForScanKernel(int* dOutData,
									  int* dLabelsIn, uint pitch,
									  const int dataWidth )
{
	uint x = (blockIdx.x*blockDim.x)+threadIdx.x;
    uint y = (blockIdx.y*blockDim.y)+threadIdx.y;  
	int index = x+(y*pitch);
	int label = dLabelsIn[index];
	int dataIndex = x + (y*dataWidth);
	int ret = 0;
	if(label == dataIndex) ret = 1;
	dOutData[dataIndex] = ret;
}

__global__ void reindexLabelsKernel(int* dLabelsOut, int* dLabelsIn, uint pitch,
									 int* dScanIn, const int dataWidth, const int numComponents)
{
	uint x = (blockIdx.x*blockDim.x)+threadIdx.x;
    uint y = (blockIdx.y*blockDim.y)+threadIdx.y;  
	int index = x+(y*pitch);
	int dataIndex = x + (y*dataWidth);
	int label = dLabelsIn[index];
	int ret = -1;
	if(label != -1) {
		int scanId = dScanIn[label];		//label is using the same indexing as the scanned data
		ret = numComponents - scanId;
	}
	dLabelsOut[index] = ret;
}
