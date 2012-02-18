#include "ccl_kernels.cu"
#include "cclFunctions.h"

//----------------------------------------
// host code for KERNEL 1
//----------------------------------------

bool cclSolveCCLShared(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inSegBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth, int connectivity)
{
	dim3 block(threadsX, threadsY, 1);
    dim3 grid(imgWidth / block.x, imgHeight / block.y, 1);

	//we need a border of labels around the block so let's allocate enough memory for it
	int labelsShareSize = sizeof(int)*(threadsX+2)*(threadsY+2);
	int segShareSize = sizeof(int)*(threadsX+2)*(threadsY+2);
	solveCCLSharedKernel<<<grid, block, labelsShareSize+segShareSize>>>
		                                       ((int*)outLabelsBuf.m_data, outLabelsBuf.m_pitch/sizeof(int), 
												(uchar*)inSegBuf.m_data, inSegBuf.m_pitch/sizeof(uchar),											
											    dataWidth, labelsShareSize/sizeof(int), connectivity);
	return true;
}

//----------------------------------------
// MERGING SCHEME
//----------------------------------------

bool cclFlattenEquivalenceTreesAfterMergingTiles(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inLabelsBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth, int tileSize);

bool cclMergeEquivalenceTrees(sCudaBuffer2D inOutLabelsBuf, sCudaBuffer2D inSegBuf, int* dChanged, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth, int tileSize, int connectivity)
{
	cudaError_t err;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
	size_t offset;
	err = cudaBindTexture2D(&offset, &segmentedTex, inSegBuf.m_data, &channelDesc, imgWidth, imgHeight,inSegBuf.m_pitch);
	if(err != cudaSuccess) 
		return false;	
		
	while(tileSize < imgWidth || tileSize < imgHeight) {
		//compute the number of tiles that are going to be merged in a singe thread block
		int xTiles = 4;
		int yTiles = 4;
		if(xTiles*tileSize > imgWidth) xTiles = imgWidth / tileSize;
		if(yTiles*tileSize > imgHeight) yTiles = imgHeight / tileSize;
		//the number of threads that is going to be used to merge neigboring tiles
		int threadsPerBlock = 32;
		if(tileSize < threadsPerBlock) threadsPerBlock = tileSize;
		dim3 block(xTiles,yTiles,threadsPerBlock);
		dim3 grid(imgWidth/(block.x*tileSize), imgHeight/(block.x*tileSize), 1);

		//call KERNEL 2
		mergeEquivalenceTreesOnBordersKernel<<<grid, block>>>((int*)inOutLabelsBuf.m_data, inOutLabelsBuf.m_pitch/sizeof(int),
			 									  dChanged,			
												  tileSize, connectivity);

		if(yTiles > xTiles) tileSize = yTiles * tileSize;
		else tileSize = xTiles * tileSize;

		if(tileSize < imgWidth || tileSize < imgHeight) {
			//update borders (KERNEL 3)
			cclFlattenEquivalenceTreesAfterMergingTiles(inOutLabelsBuf, inOutLabelsBuf, threadsX, threadsX, imgWidth, imgHeight, dataWidth, log2DataWidth, tileSize);
		}
	}
	

	err = cudaUnbindTexture(&segmentedTex);
	if(err != cudaSuccess) 
		return false;

	return true;
}


//compute the number of threads and blocks that will be used for updating elements on the borders of merged tiles
bool initTileKernelData(int tileSize, int imgWidth, int imgHeight, int threadsX, int threadsY, dim3& block, dim3& grid, int& blocksPerTile)
{
	int tileX = imgWidth / tileSize;
	int tileY = imgHeight / tileSize;
	int maxThreads = threadsX*threadsY;
	if(tileY < threadsY) {
		threadsY = tileY;
		threadsX = maxThreads/threadsY;
	}
	if(threadsX > tileSize) threadsX = tileSize;
	block = dim3(threadsX, threadsY, 1);	
    grid = dim3(imgWidth / block.x,	
			  (tileY) / block.y,
			  1);
	blocksPerTile = tileSize / block.x;	
	return true;
}

bool cclFlattenEquivalenceTreesAfterMergingTiles(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inLabelsBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth, int tileSize)
{
	dim3 block, grid;
	int blocksPerTile;
	initTileKernelData(tileSize, imgWidth, imgHeight, threadsX, threadsY, block, grid, blocksPerTile);
	// call KERNEL 3
	flattenEquivalenceTreesAfterMergingTilesKernel<<<grid, block>>>(
												(int*)outLabelsBuf.m_data, (int*)inLabelsBuf.m_data, outLabelsBuf.m_pitch/sizeof(int), 
												dataWidth, log2DataWidth, tileSize, blocksPerTile);
	return true;
}

//----------------------------------------
// end MERGING SCHEME
//----------------------------------------

//----------------------------------------
// host code for KERNEL 4
//----------------------------------------

bool cclFlattenEquivalenceTrees(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inLabelsBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth)
{
	dim3 block(threadsX, threadsY, 1);
    dim3 grid(imgWidth / block.x, imgHeight / block.y, 1);
	flattenEquivalenceTreesKernel<<<grid, block>>>((int*)outLabelsBuf.m_data, (int*)inLabelsBuf.m_data, outLabelsBuf.m_pitch/sizeof(int), 
												dataWidth, log2DataWidth);
	return true;
}



//----------------------------------------
// END host code for KERNEL 4
//----------------------------------------


//----------------------------------------
// host code for label reindexing
//----------------------------------------

bool cclPrepareLabelsForScan(void* outBuf, sCudaBuffer2D inLabelsBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth)
{
	dim3 block(threadsX, threadsY, 1);
    dim3 grid(imgWidth / block.x, imgHeight / block.y, 1);
	prepareLabelsForScanKernel<<<grid, block>>>((int*)outBuf, (int*)inLabelsBuf.m_data, inLabelsBuf.m_pitch/sizeof(int), dataWidth);
	return true;
}

bool cclReindexLabels(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inLabelsBuf, void* inScanBuf, int numComponents, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth)
{
	dim3 block(threadsX, threadsY, 1);
    dim3 grid(imgWidth / block.x, imgHeight / block.y, 1);
	reindexLabelsKernel<<<grid, block>>>((int*)outLabelsBuf.m_data,(int*)inLabelsBuf.m_data, inLabelsBuf.m_pitch/sizeof(int), (int*)inScanBuf, dataWidth, numComponents);
	return true;
}
