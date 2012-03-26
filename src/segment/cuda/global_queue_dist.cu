#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "global_queue_dist.cuh"

#define MAX_NUM_BLOCKS_DIST	70
//#include "global_sync.cu"

#define WARP_SIZE_DIST 	32
#define NUM_THREADS_DIST	512
#define NUM_WARPS_DIST (NUM_THREADS_DIST / WARP_SIZE_DIST)
#define LOG_NUM_THREADS_DIST 9
#define LOG_NUM_WARPS_DIST (LOG_NUM_THREADS_DIST - 5)

#define SCAN_STRIDE_DIST (WARP_SIZE_DIST + WARP_SIZE_DIST / 2 + 1)

__device__ volatile int inQueueSizeDist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int *inQueuePtr1Dist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int inQueueHeadDist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int outQueueMaxSizeDist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int outQueueHeadDist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int *outQueuePtr2Dist[MAX_NUM_BLOCKS_DIST];

__device__ volatile int *curInQueueDist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int *curOutQueueDist[MAX_NUM_BLOCKS_DIST];
__device__ volatile int execution_code_dist;


// This variables are used for debugging purposes only
__device__ volatile int totalInserts_dist[MAX_NUM_BLOCKS_DIST];


// Utils...
// http://www.moderngpu.com/intro/scan.html
__device__ void scan_dist(const int* values, int* exclusive) {

	// Reserve a half warp of extra space plus one per warp in the block.
	// This is exactly enough space to avoid comparisons in the multiscan
	// and to avoid bank conflicts.
	__shared__ volatile int scan[NUM_WARPS_DIST * SCAN_STRIDE_DIST];
	int tid = threadIdx.x;
	int warp = tid / WARP_SIZE_DIST;
	int lane = (WARP_SIZE_DIST - 1) & tid;

	volatile int* s = scan + SCAN_STRIDE_DIST * warp + lane + WARP_SIZE_DIST / 2;
	s[-16] = 0;

	// Read from global memory.
	int x = values[tid];
	s[0] = x;

	// Run inclusive scan on each warp's data.
	int sum = x;    

#pragma unroll
	for(int i = 0; i < 5; ++i) {
		int offset = 1<< i;
		sum += s[-offset];
		s[0] = sum;
	}

	// Synchronize to make all the totals available to the reduction code.
	__syncthreads();
	__shared__ volatile int totals[NUM_WARPS_DIST + NUM_WARPS_DIST / 2];
	if(tid < NUM_WARPS_DIST) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		int total = scan[SCAN_STRIDE_DIST * tid + WARP_SIZE_DIST / 2 + WARP_SIZE_DIST - 1];

		totals[tid] = 0;
		volatile int* s2 = totals + NUM_WARPS_DIST / 2 + tid;
		int totalsSum = total;
		s2[0] = total;

#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS_DIST; ++i) {
			int offset = 1<< i;
			totalsSum += s2[-offset];
			s2[0] = totalsSum;  
		}

		// Subtract total from totalsSum for an exclusive scan.
		totals[tid] = totalsSum - total;
	}

	// Synchronize to make the block scan available to all warps.
	__syncthreads();

	// Add the block scan to the inclusive sum for the block.
	sum += totals[warp];

	// Write the inclusive and exclusive scans to global memory.
//	inclusive[tid] = sum;
	exclusive[tid] = sum - x;
}

__device__ int queueElementDist(int *outQueueCurPtr, int *elements){
	int queue_index = atomicAdd((int*)&outQueueHeadDist[blockIdx.x], 1);
	if(queue_index < outQueueMaxSizeDist[blockIdx.x]){
		curOutQueueDist[blockIdx.x][queue_index] = elements[0];
	}else{
		// error: there are items lost in the propagation. should do another pass to correct it.
		execution_code_dist=1;
		queue_index = -1;
	}
	return queue_index;
}


// Assuming that all threads in a block are calling this function
__device__ int queueElementDist(int *elements){
	int queue_index = -1;
#ifdef	PREFIX_SUM
	__shared__ int writeAddr[NUM_THREADS_DIST];
	__shared__ int exclusiveScan[NUM_THREADS_DIST];
	__shared__ int global_queue_index;

	if(threadIdx.x == 0){
		global_queue_index = outQueueHeadDist[blockIdx.x];
	}

	// set to the number of values this threard is writing
	writeAddr[threadIdx.x] = elements[0];

	// run a prefix-sum on threads inserting data to the queue
	scan_dist(writeAddr, exclusiveScan);

	// calculate index into the queue where given thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// write elemets sequentially to shared memory
//	int localIndex = exclusiveScan[threadIdx.x];
//	for(int i = 0; i < elements[0]; i++){
//		localElements[localIndex+i] = elements[i+1];
//	}

//	__syncthreads();
//	for(int i = threadIdx.x; i < exclusiveScan[NUM_THREADS_DIST-1]+writeAddr[NUM_THREADS_DIST-1]; i+=blockDim.x){
//		curOutQueueDist[blockIdx.x][global_queue_index+i] = localElements[i];
//	}

	for(int i = 0; i < elements[0]; i++){
		// If the queue storage has been exceed, than set the execution code to 1. 
		// This will force a second round in the morphological reconstructio.	
		if(queue_index+i >= outQueueMaxSizeDist[blockIdx.x]){
//			printf("List out of bounds\n");
			execution_code_dist=1;
		}else{
			curOutQueueDist[blockIdx.x][queue_index+i] = elements[i+1];
		}
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHeadDist[blockIdx.x]+=exclusiveScan[NUM_THREADS_DIST-1]+writeAddr[NUM_THREADS_DIST-1];
		if(outQueueHeadDist[blockIdx.x] >= outQueueMaxSizeDist[blockIdx.x]){
			outQueueHeadDist[blockIdx.x] = outQueueMaxSizeDist[blockIdx.x];
		}
//		printf("Inserting = %d - outQueueHeadDist = %d\n", exclusiveScan[NUM_THREADS_DIST-1]+writeAddr[NUM_THREADS_DIST-1], outQueueHeadDist[blockIdx.x]);
	}
#else
	if(elements[0] != 0){
		queue_index = atomicAdd((int*)&outQueueHeadDist[blockIdx.x], elements[0]);
		if(queue_index < outQueueMaxSizeDist[blockIdx.x]){
			for(int i = 0; i < elements[0];i++){
				curOutQueueDist[blockIdx.x][queue_index+i] = elements[i+1];
			}
		}else{
			queue_index = -1;
		}
	}
#endif
	return queue_index;
}


// Assuming that all threads in a block are calling this function
__device__ int queueElementDist(int element){
	int queue_index = -1;
#ifdef	PREFIX_SUM
	__shared__ int writeAddr[NUM_THREADS_DIST];
	__shared__ int exclusiveScan[NUM_THREADS_DIST];
	__shared__ int global_queue_index;

	if(threadIdx.x == 0){
		global_queue_index = outQueueHeadDist[blockIdx.x];
	}

	// set to 1 threards that are writing
	writeAddr[threadIdx.x] = ((element) != (-1) ? (1):(0));

	// run a prefix-sum on threads inserting data to the queue
	scan_dist(writeAddr, exclusiveScan);

	// calculate index into the queue where give thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// If there is data to be queued, do it
	if(element != -1){
		curOutQueueDist[blockIdx.x][queue_index] = element;
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHeadDist[blockIdx.x]+=exclusiveScan[NUM_THREADS_DIST-1]+writeAddr[NUM_THREADS_DIST-1];
	}
#else
	if(element != -1){
		queue_index = atomicAdd((int*)&outQueueHeadDist[blockIdx.x], 1);
		if(queue_index < outQueueMaxSizeDist[blockIdx.x]){
			curOutQueueDist[blockIdx.x][queue_index] = element;
		}else{
			queue_index = -1;
		}
	}
#endif
	return queue_index;
}

// Makes queue 1 point to queue 2, and vice-versa
__device__ void swapQueusDist(int loopIt){
	__syncthreads();

	if(loopIt %2 == 0){
		curInQueueDist[blockIdx.x] = outQueuePtr2Dist[blockIdx.x];
		curOutQueueDist[blockIdx.x] = inQueuePtr1Dist[blockIdx.x];
		if(threadIdx.x == 0){
			inQueueSizeDist[blockIdx.x] = outQueueHeadDist[blockIdx.x];
			outQueueHeadDist[blockIdx.x] = 0;
			inQueueHeadDist[blockIdx.x] = 0;
			// This is used for profiling only
			totalInserts_dist[blockIdx.x]+=inQueueSizeDist[blockIdx.x];
		}
	}else{
		curInQueueDist[blockIdx.x] = inQueuePtr1Dist[blockIdx.x];
		curOutQueueDist[blockIdx.x] = outQueuePtr2Dist[blockIdx.x];

		if(threadIdx.x == 0){
			inQueueSizeDist[blockIdx.x] = outQueueHeadDist[blockIdx.x];
			outQueueHeadDist[blockIdx.x] = 0;
			inQueueHeadDist[blockIdx.x] = 0;
			// This is used for profiling only
			totalInserts_dist[blockIdx.x]+=inQueueSizeDist[blockIdx.x];
		}
	}
	__syncthreads();
}



// -2, nothing else to be done at all
__device__ int dequeueElementDist(int *loopIt){
	// did this block got something to do?
	__shared__ volatile int gotWork;

getWork:
	gotWork = 0;


	// Try to get some work.
//	int queue_index = atomicAdd((int*)&inQueueHeadDist, 1);
	int queue_index = inQueueHeadDist[blockIdx.x] + threadIdx.x;
	// I must guarantee that idle threads are set to 0, and no other thread 
	// will come later and set it to 0 again
	__syncthreads();

	if(threadIdx.x == 0){
		inQueueHeadDist[blockIdx.x]+=blockDim.x;
//		if(loopIt[0] < 1){
//			printf("inQueueSizeDist = %d loopIt[0] = %d queue_index = %d outQueueHeadDist = %d\n", inQueueSizeDist[blockIdx.x], loopIt[0], queue_index, outQueueHeadDist[blockIdx.x]);
//		}
	}

	// Nothing to do by default
	int element = -1;
	if(queue_index < inQueueSizeDist[blockIdx.x]){
		element = curInQueueDist[blockIdx.x][queue_index];
		gotWork = 1;
	}
	__syncthreads();


	// This block does not have anything to process
	if(!gotWork){
//		if(loopIt[0] < 20 && threadIdx.x == 0)
//			printf("inQueueSizeDist = %d loopIt[0] = %d\n", inQueueSizeDist[blockIdx.x], loopIt[0]);
		element = -2;
		if(outQueueHeadDist[blockIdx.x] != 0){
			swapQueusDist(loopIt[0]);
			loopIt[0]++;
			goto getWork;
		}
	}
	return element;
}

__global__ void initQueueIdDist(int *inQueueData, int dataElements, int *outQueueData, int outMaxSize, int qId){
	if(threadIdx.x < 1){
		// Simply assign input data pointers/number of elements to the queue
		inQueuePtr1Dist[qId] = inQueueData;

//		printf("initQueueVector: tid - %d dataElements = %d pointer = %p\n", threadIdx.x, dataElements, inQueueData);
		inQueueSizeDist[qId] = dataElements;

		totalInserts_dist[qId] = 0;
		
		// alloc second vector used to queue output elements
		outQueuePtr2Dist[qId] = outQueueData;

		// Maximum number of elements that fit into the queue
		outQueueMaxSizeDist[qId] = outMaxSize;

		// Head of the out queue
		outQueueHeadDist[qId] = 0;

		// Head of the in queue
		inQueueHeadDist[qId] = 0;

		execution_code_dist=0;
	}
}






//
//extern "C" int morphReconSpeedupDist( int *g_InputListPtr, int h_ListSize, int *g_Seed, int *g_Image, int h_ncols, int h_nrows, int connectivity, int nBlocks, float queue_increase_factor){
//	int *d_Result;
//
//	// alloc space to save output elements in the queue for each block
//	int **h_OutQueuePtr = (int **)malloc(sizeof(int*) * nBlocks);
//
//	// at this moment I should partition the INPUT queue
//	int tempNblocks = nBlocks;
//
//	int subListsInit[tempNblocks];
//	int subListsSize[tempNblocks];
//
//	for(int i = 0; i < tempNblocks; i++){
//		int curSubListInit = (h_ListSize/tempNblocks)*i;
//		int curSubListEnd = ((i+1<tempNblocks)?((i+1)*(h_ListSize/tempNblocks)-1):(h_ListSize-1));
//	//	printf("BlockId = %d - init = %d end = %d size=%d\n", i, curSubListInit, curSubListEnd, curSubListEnd-curSubListInit+1);
//		subListsInit[i] = curSubListInit;
////		subListsEnd[i] = curSubListEnd;
//		subListsSize[i]	= curSubListEnd-curSubListInit+1;
//	}
//
//// Adding code
//	// TODO: free data
//	int *blockSubLists[tempNblocks];
//	for(int i = 0; i < tempNblocks; i++){
//		cudaMalloc((void **)&blockSubLists[i], sizeof(int)*(subListsSize[i]) * queue_increase_factor);
//		cudaMemcpy(blockSubLists[i], &g_InputListPtr[subListsInit[i]], subListsSize[i] * sizeof(int), cudaMemcpyDeviceToDevice);
//	}
//
//
//// End adding code
//
////	printf("h_listSize = %d subListsSize[0]=%d\n", h_ListSize, subListsSize[0]);
////	cout << "h_listSize = "<< h_ListSize<< " subListsSize[0]="<< subListsSize[0] <<endl;
//	
//	for(int i = 0; i < tempNblocks;i++){
//		cudaMalloc((void **)&h_OutQueuePtr[i], sizeof(int) * (subListsSize[i]) * queue_increase_factor);
//	}
//	
//	// Init queue for each image. yes, this may not be the most efficient way, but the code is far easier to read. 
//	// Another version, where all pointer are copied at once to the GPU was also built, buit it was only about 1ms 
//	// faster. Thus, we decide to go with this version 
//	for(int i = 0; i < nBlocks;i++)
//		initQueueIdDist<<<1, 1>>>(blockSubLists[i], subListsSize[i], h_OutQueuePtr[i], (subListsSize[i]) *queue_increase_factor, i);
//
//	// This is used by each block to store the number of queue operations performed
//	cudaMalloc((void **)&d_Result, sizeof(int)*(nBlocks+1)) ;
//	cudaMemset((void *)d_Result, 0, sizeof(int)*(nBlocks+1));
//
//
////	printf("Run computation kernel!\n");
//	morphReconKernelSpeedupDist<<<nBlocks, NUM_THREADS_DIST>>>(d_Result, g_Seed, g_Image, h_ncols, h_nrows, connectivity);
//
//	if(cudaGetLastError() != cudaSuccess){
//		cudaError_t errorCode = cudaGetLastError();
//		const char *error = cudaGetErrorString(errorCode);
//		printf("Error after morphRecon = %s\n", error);
//	}
//
//	int *h_Result = (int *) malloc(sizeof(int) * (nBlocks+1));
//	cudaMemcpy(h_Result, d_Result, sizeof(int) * (nBlocks+1), cudaMemcpyDeviceToHost);
//
//	int resutRet = h_Result[nBlocks];
//	free(h_Result);
//
//	cudaFree(d_Result);
//	for(int i = 0; i < nBlocks; i++){
//		cudaFree(h_OutQueuePtr[i]);
//	}
//	free(h_OutQueuePtr);
//	cudaFree(g_InputListPtr);
//
//	return resutRet;
//}

namespace nscale{
namespace gpu{
using namespace cv::gpu;


// find out if current closest 0 of (x,y) is closet to nearest 0 of (x',y') described by propItem (y'*ncols+x').
__device__ int propagateDist(PtrStep_<unsigned char> mask , PtrStep_<int> nearestNeighbors, int x, int y, int ncols, int propItem){
	int returnValue = -1;

	while(true){
		// retrieve the int stoing the nearest 0 of (x,y)
		int x_y_nearest =  nearestNeighbors.ptr(y)[x];

		// calculate cordinates of the current x_y closest 0 
		int x_nearest_cord = x_y_nearest%ncols;
		int y_nearest_cord = x_y_nearest/ncols;

		// current (x,y) distance to closes 0
		int cur_x_y_dist_closest_0 = (x-x_nearest_cord)*(x-x_nearest_cord) + (y-y_nearest_cord)*(y-y_nearest_cord);

		// do the same process for propItem (the item in the propogation frontier)
		int x_propItem = propItem%ncols;
		int y_propItem = propItem/ncols;

		// retrieve the int stoing the nearest 0 of propItem
		int propItem_nearest = nearestNeighbors.ptr(y_propItem)[x_propItem];

		// calculate cordinates of the current x_y closest 0 for propItem
		int x_propItem_nearest_cord = propItem_nearest%ncols;
		int y_propItem_nearest_cord = propItem_nearest/ncols;

		// (x,y) distance to 0 through propItem.
		int cur_x_y_dist_propItem = (x-x_propItem_nearest_cord)*(x-x_propItem_nearest_cord) + (y-y_propItem_nearest_cord)*(y-y_propItem_nearest_cord);

		// if passing through propItem (x,y) will have a closer 0, lets do it.
		if(cur_x_y_dist_closest_0 > cur_x_y_dist_propItem){
			// try to update x,y current nearest 0,
			int cas_result = atomicCAS( (int*)&(nearestNeighbors.ptr(y)[x]), x_y_nearest, propItem_nearest); 

			// If update has succeed.
			if(cas_result == x_y_nearest){
				returnValue = (y*ncols+x);
				break;
			}

			// if update failed, repeat computation because the nearest element 
			// to (x,y) was update since i has been read.

		}else{
			// Okay. Pass through propItem is not the closest path. lets leave the loop.
			break;
		}
	}
	return returnValue;
}


__global__ void distTransformPropagationKernel(int* d_Result, PtrStep_<unsigned char> mask , PtrStep_<int> nearestNeighbors, int ncols, int nrows){
	curInQueueDist[blockIdx.x] = inQueuePtr1Dist[blockIdx.x];
	curOutQueueDist[blockIdx.x] = outQueuePtr2Dist[blockIdx.x];
//	int *seeds = d_Seeds;
//	int *image = d_Image;

	int loopIt = 0;
	int workUnit = -1;
	int tid = threadIdx.x;
	__shared__ int localQueue[NUM_THREADS_DIST][9];

	__syncthreads();
	do{
		int x, y;

		localQueue[tid][0] = 0;
		
		// Try to get some work.
		workUnit = dequeueElementDist(&loopIt);
		y = workUnit/ncols;
		x = workUnit%ncols;	

		int pval = 0;
		if(workUnit >= 0){
//			pval = seeds[workUnit];
			// get my current nearest 0 point which will be compare to my neighbors nearest 0.
			pval = nearestNeighbors.ptr(y)[x];
		}

		int retWork = -1;
		if(workUnit >= 0 && y > 0){
			retWork = propagateDist(mask , nearestNeighbors, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElementDist(retWork);
		if(workUnit >= 0 && y < nrows-1){
			retWork = propagateDist(mask, nearestNeighbors, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElementDist(retWork);

		if(workUnit >= 0 && x > 0){
			retWork = propagateDist(mask, nearestNeighbors, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElementDist(retWork);

		if(workUnit >= 0 && x < ncols-1){
			retWork = propagateDist(mask, nearestNeighbors, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		// if connectivity is 8, four other neighbors have to be verified
//		if(connectivity == 8){
			if(workUnit >= 0 && y > 0 && x >0){
				retWork = propagateDist(mask, nearestNeighbors, x-1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y > 0 && x < ncols-1){
				retWork = propagateDist(mask, nearestNeighbors, x+1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x >0){
				retWork = propagateDist(mask, nearestNeighbors, x-1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x <(ncols-1)){
				retWork = propagateDist(mask, nearestNeighbors, x+1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}

//		}
//		queueElementDist(retWork);
		queueElementDist(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[blockIdx.x]=totalInserts_dist[blockIdx.x];
	if(execution_code_dist!=0){
		d_Result[gridDim.x]=1;
	}

}


int distTransformPropagation( int *g_InputListPtr, int h_ListSize, PtrStep_<unsigned char> mask , PtrStep_<int> nearestNeighbors, int cols, int rows, int queue_increase_factor){
	int *d_Result;

	int tempNblocks = 12;
	// alloc space to save output elements in the queue for each block
	int **h_OutQueuePtr = (int **)malloc(sizeof(int*) * tempNblocks);

	// at this moment I should partition the INPUT queue

	int subListsInit[tempNblocks];
	int subListsSize[tempNblocks];
//	int queue_increase_factor = 2;
	printf("queue_increase_factor=%d\n", queue_increase_factor);

	// divide gpu input list (propagation frontier) among gpu blocks (sms) 
	for(int i = 0; i < tempNblocks; i++){
		int curSubListInit = (h_ListSize/tempNblocks)*i;
		int curSubListEnd = ((i+1<tempNblocks)?((i+1)*(h_ListSize/tempNblocks)-1):(h_ListSize-1));
	//	printf("BlockId = %d - init = %d end = %d size=%d\n", i, curSubListInit, curSubListEnd, curSubListEnd-curSubListInit+1);
		subListsInit[i] = curSubListInit;
//		subListsEnd[i] = curSubListEnd;
		subListsSize[i]	= curSubListEnd-curSubListInit+1;
	}

	// copy input list to sublist computed by different blocks
	int *blockSubLists[tempNblocks];
	for(int i = 0; i < tempNblocks; i++){
		cudaMalloc((void **)&blockSubLists[i], sizeof(int)*(subListsSize[i]) * queue_increase_factor);
		cudaMemcpy(blockSubLists[i], &g_InputListPtr[subListsInit[i]], subListsSize[i] * sizeof(int), cudaMemcpyDeviceToDevice);
	}
	
	// allocate space to store output list counter part of each sub-input list computed by each block
	for(int i = 0; i < tempNblocks;i++){
		cudaMalloc((void **)&h_OutQueuePtr[i], sizeof(int) * (subListsSize[i]) * queue_increase_factor);
	}
	
	// Init queue for each block. yes, this may not be the most efficient way, but the code is far easier to read. 
	// Another version, where all pointer are copied at once to the GPU was also built, buit it was only about <1ms 
	// faster. Thus, we decide to go with this version 
	for(int i = 0; i < tempNblocks;i++)
		initQueueIdDist<<<1, 1>>>(blockSubLists[i], subListsSize[i], h_OutQueuePtr[i], (subListsSize[i]) *queue_increase_factor, i);

	// This is used by each block to store the number of queue operations performed. Profile purposes only.
	cudaMalloc((void **)&d_Result, sizeof(int)*(tempNblocks+1)) ;
	cudaMemset((void *)d_Result, 0, sizeof(int)*(tempNblocks+1));


//	printf("Run computation kernel!\n");
	distTransformPropagationKernel<<<tempNblocks, NUM_THREADS_DIST>>>(d_Result, mask, nearestNeighbors, cols, rows);

	if(cudaGetLastError() != cudaSuccess){
		cudaError_t errorCode = cudaGetLastError();
		const char *error = cudaGetErrorString(errorCode);
		printf("Error after morphRecon = %s\n", error);
	}

	// get return code saying if distance transform calculation was successful or not.
	int *h_Result = (int *) malloc(sizeof(int) * (tempNblocks+1));
	cudaMemcpy(h_Result, d_Result, sizeof(int) * (tempNblocks+1), cudaMemcpyDeviceToHost);

	int resutRet = h_Result[tempNblocks];

	free(h_Result);

	cudaFree(d_Result);
	for(int i = 0; i < tempNblocks; i++){
		cudaFree(h_OutQueuePtr[i]);
	}
	free(h_OutQueuePtr);
	cudaFree(g_InputListPtr);
	cudaDeviceSynchronize();

	return resutRet;
}




// verify if this 0 has a 1 neighbor
__device__ bool checkDistNeighbors8(int x, int y,  int rows, int cols, PtrStep_<unsigned char> mask){
	bool isCandidate = false;

	// upper line
	if(y>0){
		// uppper left corner
		if(x > 0){
			if(mask.ptr(y-1)[x-1] != 0 ){
				isCandidate = true;
			}
		}
		// upper right corner
		if(x < (cols-1)){
			if(mask.ptr(y-1)[x+1] != 0 ){
				isCandidate = true;
			}
		}
		// upper center
		if(mask.ptr(y-1)[x] != 0 ){
			isCandidate = true;
		}
	}

	// lower line
	if(y < (rows-1)){
		// lower left corner
		if(x > 0){
			if(mask.ptr(y+1)[x-1] != 0 ){
				isCandidate = true;
			}
		}
		// lower right corner
		if(x < (cols-1)){
			if(mask.ptr(y+1)[x+1] != 0 ){
				isCandidate = true;
			}
		}
		// lower center
		if(mask.ptr(y+1)[x] != 0 ){
			isCandidate = true;
		}
	}
	// left item
	if(x>0){
		if(mask.ptr(y)[x-1] != 0){
			isCandidate = true;
		}
	}
	// right item
	if(x < (cols-1)){
		if(mask.ptr(y)[x+1] != 0){
			isCandidate = true;
		}
	}
	return isCandidate;
}


__global__ void distBuildQueueKernel(int rows, int cols, PtrStep_<unsigned char> mask, PtrStep_<int> nearestZero, int* g_queue, int*g_queue_size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y < rows && x < cols){
		// set distance to nearest 0 to inf.
		nearestZero.ptr(y)[x] = rows * cols * 3;

		// if pixel is 0
		if(mask.ptr(y)[x] == 0){

			// Distance to nearest 0 is 0 --- myself.
			nearestZero.ptr(y)[x] = y*cols+x;

			bool isCandidate = false;
			// veryfi if this pixels has a non-zero neighbor (propagation frontier)
			isCandidate = checkDistNeighbors8(x, y, rows, cols, mask);
			
			// 
			if(isCandidate){
				int queueIndex = atomicAdd((unsigned int*) g_queue_size, 1);
	//			printf("y=%d and x=%d index=%d\n", y, x, queueIndex);
				g_queue[queueIndex] = y*cols+x; 
			}
		}
	}

}

int *distQueueBuildCaller(int rows, int cols, PtrStep_<unsigned char> mask, PtrStep_<int> nearestNeighbors, int &queue_size, cudaStream_t stream){
	dim3 threads(16,16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	// Pointer to gpu memory array storing ng inital items queued
	int *g_queue;
	cudaMalloc( (void**)&g_queue, sizeof(int) * rows * cols );

	// Used to store the size of the queue created by build queue.
	int *g_queue_size;
	cudaMalloc( (void**)&g_queue_size, sizeof(int) );
	cudaMemset( (void*) g_queue_size, 0, sizeof(int));

	distBuildQueueKernel<<<grid, threads, 0, stream>>>(rows, cols, mask, nearestNeighbors, g_queue, g_queue_size);

	cudaMemcpy(&queue_size, g_queue_size, sizeof(int), cudaMemcpyDeviceToHost );
	
	cudaFree(g_queue_size);

	return g_queue;
}



__global__ void distMapKernel(int rows, int cols, PtrStep_<int> nearestMap, PtrStep_<float> distanceMap)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if(y < rows && x < cols){
		// calculate x and y of the nearest 0
		int x_nearest = nearestMap.ptr(y)[x] % cols;
		int y_nearest = nearestMap.ptr(y)[x] / cols;

		float dist = sqrtf((x-x_nearest)*(x-x_nearest) + (y-y_nearest)*(y-y_nearest));
		distanceMap.ptr(y)[x] = dist;	
	}

}




void distMapCalcCaller(int rows, int cols, PtrStep_<int> nearestNeighbors, PtrStep_<float> distanceMap, cudaStream_t stream){
	dim3 threads(16,16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	distMapKernel<<<grid, threads, 0, stream>>>(rows, cols, nearestNeighbors, distanceMap);
}





}} // close namespaces




