#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NUM_BLOCKS_FLOAT	70
//#include "global_sync.cu"

#define WARP_SIZE_FLOAT 	32
#define NUM_THREADS_FLOAT	512
#define NUM_WARPS_FLOAT (NUM_THREADS_FLOAT / WARP_SIZE_FLOAT)
#define LOG_NUM_THREADS_FLOAT 9
#define LOG_NUM_WARPS_FLOAT (LOG_NUM_THREADS_FLOAT - 5)

#define SCAN_STRIDE_FLOAT (WARP_SIZE_FLOAT + WARP_SIZE_FLOAT / 2 + 1)

__device__ volatile int inQueueSizeFloat[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int *inQueuePtr1Float[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int inQueueHeadFloat[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int outQueueMaxSizeFloat[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int outQueueHeadFloat[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int *outQueuePtr2Float[MAX_NUM_BLOCKS_FLOAT];

__device__ volatile int *curInQueueFloat[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int *curOutQueueFloat[MAX_NUM_BLOCKS_FLOAT];
__device__ volatile int execution_code_float;


// This variables are used for debugging purposes only
__device__ volatile int totalInserts_float[MAX_NUM_BLOCKS_FLOAT];


// Utils...
// http://www.moderngpu.com/intro/scan.html
__device__ void scan_float(const int* values, int* exclusive) {

	// Reserve a half warp of extra space plus one per warp in the block.
	// This is exactly enough space to avoid comparisons in the multiscan
	// and to avoid bank conflicts.
	__shared__ volatile int scan[NUM_WARPS_FLOAT * SCAN_STRIDE_FLOAT];
	int tid = threadIdx.x;
	int warp = tid / WARP_SIZE_FLOAT;
	int lane = (WARP_SIZE_FLOAT - 1) & tid;

	volatile int* s = scan + SCAN_STRIDE_FLOAT * warp + lane + WARP_SIZE_FLOAT / 2;
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
	__shared__ volatile int totals[NUM_WARPS_FLOAT + NUM_WARPS_FLOAT / 2];
	if(tid < NUM_WARPS_FLOAT) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		int total = scan[SCAN_STRIDE_FLOAT * tid + WARP_SIZE_FLOAT / 2 + WARP_SIZE_FLOAT - 1];

		totals[tid] = 0;
		volatile int* s2 = totals + NUM_WARPS_FLOAT / 2 + tid;
		int totalsSum = total;
		s2[0] = total;

#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS_FLOAT; ++i) {
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

__device__ int queueElementFloat(int *outQueueCurPtr, int *elements){
	int queue_index = atomicAdd((int*)&outQueueHeadFloat[blockIdx.x], 1);
	if(queue_index < outQueueMaxSizeFloat[blockIdx.x]){
		curOutQueueFloat[blockIdx.x][queue_index] = elements[0];
	}else{
		queue_index = -1;
	}
	return queue_index;
}


// Assuming that all threads in a block are calling this function
__device__ int queueElementFloat(int *elements){
	int queue_index = -1;
#ifdef	PREFIX_SUM
	__shared__ int writeAddr[NUM_THREADS_FLOAT];
	__shared__ int exclusiveScan[NUM_THREADS_FLOAT];
	__shared__ int global_queue_index;

	if(threadIdx.x == 0){
		global_queue_index = outQueueHeadFloat[blockIdx.x];
	}

	// set to the number of values this threard is writing
	writeAddr[threadIdx.x] = elements[0];

	// run a prefix-sum on threads inserting data to the queue
	scan_float(writeAddr, exclusiveScan);

	// calculate index into the queue where given thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// write elemets sequentially to shared memory
//	int localIndex = exclusiveScan[threadIdx.x];
//	for(int i = 0; i < elements[0]; i++){
//		localElements[localIndex+i] = elements[i+1];
//	}

//	__syncthreads();
//	for(int i = threadIdx.x; i < exclusiveScan[NUM_THREADS_FLOAT-1]+writeAddr[NUM_THREADS_FLOAT-1]; i+=blockDim.x){
//		curOutQueueFloat[blockIdx.x][global_queue_index+i] = localElements[i];
//	}

	for(int i = 0; i < elements[0]; i++){
		// If the queue storage has been exceed, than set the execution code to 1. 
		// This will force a second round in the morphological reconstructio.	
		if(queue_index+i >= outQueueMaxSizeFloat[blockIdx.x]){
//			printf("List out of bounds\n");
			execution_code_float=1;
		}else{
			curOutQueueFloat[blockIdx.x][queue_index+i] = elements[i+1];
		}
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHeadFloat[blockIdx.x]+=exclusiveScan[NUM_THREADS_FLOAT-1]+writeAddr[NUM_THREADS_FLOAT-1];
		if(outQueueHeadFloat[blockIdx.x] >= outQueueMaxSizeFloat[blockIdx.x]){
			outQueueHeadFloat[blockIdx.x] = outQueueMaxSizeFloat[blockIdx.x];
		}
//		printf("Inserting = %d - outQueueHeadFloat = %d\n", exclusiveScan[NUM_THREADS_FLOAT-1]+writeAddr[NUM_THREADS_FLOAT-1], outQueueHeadFloat[blockIdx.x]);
	}
#else
	if(elements[0] != 0){
		queue_index = atomicAdd((int*)&outQueueHeadFloat[blockIdx.x], elements[0]);
		if(queue_index < outQueueMaxSizeFloat[blockIdx.x]){
			for(int i = 0; i < elements[0];i++){
				curOutQueueFloat[blockIdx.x][queue_index+i] = elements[i+1];
			}
		}else{
			queue_index = -1;
		}
	}
#endif
	return queue_index;
}


// Assuming that all threads in a block are calling this function
__device__ int queueElementFloat(int element){
	int queue_index = -1;
#ifdef	PREFIX_SUM
	__shared__ int writeAddr[NUM_THREADS_FLOAT];
	__shared__ int exclusiveScan[NUM_THREADS_FLOAT];
	__shared__ int global_queue_index;

	if(threadIdx.x == 0){
		global_queue_index = outQueueHeadFloat[blockIdx.x];
	}

	// set to 1 threards that are writing
	writeAddr[threadIdx.x] = ((element) != (-1) ? (1):(0));

	// run a prefix-sum on threads inserting data to the queue
	scan_float(writeAddr, exclusiveScan);

	// calculate index into the queue where give thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// If there is data to be queued, do it
	if(element != -1){
		curOutQueueFloat[blockIdx.x][queue_index] = element;
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHeadFloat[blockIdx.x]+=exclusiveScan[NUM_THREADS_FLOAT-1]+writeAddr[NUM_THREADS_FLOAT-1];
	}
#else
	if(element != -1){
		queue_index = atomicAdd((int*)&outQueueHeadFloat[blockIdx.x], 1);
		if(queue_index < outQueueMaxSizeFloat[blockIdx.x]){
			curOutQueueFloat[blockIdx.x][queue_index] = element;
		}else{
			queue_index = -1;
		}
	}
#endif
	return queue_index;
}

// Makes queue 1 point to queue 2, and vice-versa
__device__ void swapQueusFloat(int loopIt){
	__syncthreads();

	if(loopIt %2 == 0){
		curInQueueFloat[blockIdx.x] = outQueuePtr2Float[blockIdx.x];
		curOutQueueFloat[blockIdx.x] = inQueuePtr1Float[blockIdx.x];
		if(threadIdx.x == 0){
			inQueueSizeFloat[blockIdx.x] = outQueueHeadFloat[blockIdx.x];
			outQueueHeadFloat[blockIdx.x] = 0;
			inQueueHeadFloat[blockIdx.x] = 0;
			// This is used for profiling only
			totalInserts_float[blockIdx.x]+=inQueueSizeFloat[blockIdx.x];
		}
	}else{
		curInQueueFloat[blockIdx.x] = inQueuePtr1Float[blockIdx.x];
		curOutQueueFloat[blockIdx.x] = outQueuePtr2Float[blockIdx.x];

		if(threadIdx.x == 0){
			inQueueSizeFloat[blockIdx.x] = outQueueHeadFloat[blockIdx.x];
			outQueueHeadFloat[blockIdx.x] = 0;
			inQueueHeadFloat[blockIdx.x] = 0;
			// This is used for profiling only
			totalInserts_float[blockIdx.x]+=inQueueSizeFloat[blockIdx.x];
		}
	}
	__syncthreads();
}



// -2, nothing else to be done at all
__device__ int dequeueElementFloat(int *loopIt){
	// did this block got something to do?
	__shared__ volatile int gotWork;

getWork:
	gotWork = 0;


	// Try to get some work.
//	int queue_index = atomicAdd((int*)&inQueueHeadFloat, 1);
	int queue_index = inQueueHeadFloat[blockIdx.x] + threadIdx.x;
	// I must guarantee that idle threads are set to 0, and no other thread 
	// will come later and set it to 0 again
	__syncthreads();

	if(threadIdx.x == 0){
		inQueueHeadFloat[blockIdx.x]+=blockDim.x;
//		if(loopIt[0] < 1){
//			printf("inQueueSizeFloat = %d loopIt[0] = %d queue_index = %d outQueueHeadFloat = %d\n", inQueueSizeFloat[blockIdx.x], loopIt[0], queue_index, outQueueHeadFloat[blockIdx.x]);
//		}
	}

	// Nothing to do by default
	int element = -1;
	if(queue_index < inQueueSizeFloat[blockIdx.x]){
		element = curInQueueFloat[blockIdx.x][queue_index];
		gotWork = 1;
	}
	__syncthreads();


	// This block does not have anything to process
	if(!gotWork){
//		if(loopIt[0] < 20 && threadIdx.x == 0)
//			printf("inQueueSizeFloat = %d loopIt[0] = %d\n", inQueueSizeFloat[blockIdx.x], loopIt[0]);
		element = -2;
		if(outQueueHeadFloat[blockIdx.x] != 0){
			swapQueusFloat(loopIt[0]);
			loopIt[0]++;
			goto getWork;
		}
	}
	return element;
}

__global__ void initQueueIdFloat(int *inQueueData, int dataElements, int *outQueueData, int outMaxSize, int qId){
	if(threadIdx.x < 1){
		// Simply assign input data pointers/number of elements to the queue
		inQueuePtr1Float[qId] = inQueueData;

//		printf("initQueueVector: tid - %d dataElements = %d pointer = %p\n", threadIdx.x, dataElements, inQueueData);
		inQueueSizeFloat[qId] = dataElements;

		totalInserts_float[qId] = 0;
		
		// alloc second vector used to queue output elements
		outQueuePtr2Float[qId] = outQueueData;

		// Maximum number of elements that fit into the queue
		outQueueMaxSizeFloat[qId] = outMaxSize;

		// Head of the out queue
		outQueueHeadFloat[qId] = 0;

		// Head of the in queue
		inQueueHeadFloat[qId] = 0;

		execution_code_float=0;
	}
}



// Returns what should be queued
__device__ int propagateFloat(int *seeds, int *image, int x, int y, int ncols, int pval){
	int returnValue = -1;
	int index = y*ncols + x;
	int seedXYval = seeds[index];
	int imageXYval = image[index];

	if((seedXYval < pval) && (imageXYval != seedXYval)){
//		printf("propagation pval=%d", pval);
		int newValue = min(pval, imageXYval);
		//  this should be a max atomic...
		atomicMax(&(seeds[index]), newValue);
		returnValue = index;
	}
	return returnValue;
}


__global__ void morphReconKernelSpeedupFloat(int* d_Result, int *d_Seeds, int*d_Image, int ncols, int nrows, int connectivity=4){
	curInQueueFloat[blockIdx.x] = inQueuePtr1Float[blockIdx.x];
	curOutQueueFloat[blockIdx.x] = outQueuePtr2Float[blockIdx.x];
	int *seeds = d_Seeds;
	int *image = d_Image;

	int loopIt = 0;
	int workUnit = -1;
	int tid = threadIdx.x;
	__shared__ int localQueue[NUM_THREADS_FLOAT][9];

	__syncthreads();
	do{
		int x, y;

		localQueue[tid][0] = 0;
		
		// Try to get some work.
		workUnit = dequeueElementFloat(&loopIt);
		y = workUnit/ncols;
		x = workUnit%ncols;	

		int pval = 0;
		if(workUnit >= 0){
			pval = seeds[workUnit];

		}

		int retWork = -1;
		if(workUnit >= 0 && y > 0){
			retWork = propagateFloat((int*)seeds, image, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElementFloat(retWork);
		if(workUnit >= 0 && y < nrows-1){
			retWork = propagateFloat((int*)seeds, image, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElementFloat(retWork);

		if(workUnit >= 0 && x > 0){
			retWork = propagateFloat((int*)seeds, image, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElementFloat(retWork);

		if(workUnit >= 0 && x < ncols-1){
			retWork = propagateFloat((int*)seeds, image, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		// if connectivity is 8, four other neighbors have to be verified
		if(connectivity == 8){
			if(workUnit >= 0 && y > 0 && x >0){
				retWork = propagateFloat((int*)seeds, image, x-1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y > 0 && x < ncols-1){
				retWork = propagateFloat((int*)seeds, image, x+1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x >0){
				retWork = propagateFloat((int*)seeds, image, x-1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x <(ncols-1)){
				retWork = propagateFloat((int*)seeds, image, x+1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}

		}
//		queueElementFloat(retWork);
		queueElementFloat(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[blockIdx.x]=totalInserts_float[blockIdx.x];
	if(execution_code_float!=0){
		d_Result[gridDim.x]=1;
	}

}



extern "C" int morphReconSpeedupFloat( int *g_InputListPtr, int h_ListSize, int *g_Seed, int *g_Image, int h_ncols, int h_nrows, int connectivity, int nBlocks, float queue_increase_factor){
	int *d_Result;

	// alloc space to save output elements in the queue for each block
	int **h_OutQueuePtr = (int **)malloc(sizeof(int*) * nBlocks);

	// at this moment I should partition the INPUT queue
	int tempNblocks = nBlocks;

	int *subListsInit = (int*)malloc(sizeof(int)* tempNblocks);
	int *subListsSize = (int*)malloc(sizeof(int)* tempNblocks);

	for(int i = 0; i < tempNblocks; i++){
		int curSubListInit = (h_ListSize/tempNblocks)*i;
		int curSubListEnd = ((i+1<tempNblocks)?((i+1)*(h_ListSize/tempNblocks)-1):(h_ListSize-1));
	//	printf("BlockId = %d - init = %d end = %d size=%d\n", i, curSubListInit, curSubListEnd, curSubListEnd-curSubListInit+1);
		subListsInit[i] = curSubListInit;
//		subListsEnd[i] = curSubListEnd;
		subListsSize[i]	= curSubListEnd-curSubListInit+1;
	}

// Adding code
	// TODO: free data
	int **blockSubLists = (int**)malloc(sizeof(int*)* tempNblocks);
	for(int i = 0; i < tempNblocks; i++){
		cudaMalloc((void **)&blockSubLists[i], sizeof(int)*(subListsSize[i]) * queue_increase_factor);
		cudaMemcpy(blockSubLists[i], &g_InputListPtr[subListsInit[i]], subListsSize[i] * sizeof(int), cudaMemcpyDeviceToDevice);
	}


// End adding code

//	printf("h_listSize = %d subListsSize[0]=%d\n", h_ListSize, subListsSize[0]);
//	cout << "h_listSize = "<< h_ListSize<< " subListsSize[0]="<< subListsSize[0] <<endl;
	
	for(int i = 0; i < tempNblocks;i++){
		cudaMalloc((void **)&h_OutQueuePtr[i], sizeof(int) * (subListsSize[i]) * queue_increase_factor);
	}
	
	// Init queue for each image. yes, this may not be the most efficient way, but the code is far easier to read. 
	// Another version, where all pointer are copied at once to the GPU was also built, buit it was only about 1ms 
	// faster. Thus, we decide to go with this version 
	for(int i = 0; i < nBlocks;i++)
		initQueueIdFloat<<<1, 1>>>(blockSubLists[i], subListsSize[i], h_OutQueuePtr[i], (subListsSize[i]) *queue_increase_factor, i);

	// This is used by each block to store the number of queue operations performed
	cudaMalloc((void **)&d_Result, sizeof(int)*(nBlocks+1)) ;
	cudaMemset((void *)d_Result, 0, sizeof(int)*(nBlocks+1));


//	printf("Run computation kernel!\n");
	morphReconKernelSpeedupFloat<<<nBlocks, NUM_THREADS_FLOAT>>>(d_Result, g_Seed, g_Image, h_ncols, h_nrows, connectivity);

	if(cudaGetLastError() != cudaSuccess){
		cudaError_t errorCode = cudaGetLastError();
		const char *error = cudaGetErrorString(errorCode);
		printf("Error after morphRecon = %s\n", error);
	}

	int *h_Result = (int *) malloc(sizeof(int) * (nBlocks+1));
	cudaMemcpy(h_Result, d_Result, sizeof(int) * (nBlocks+1), cudaMemcpyDeviceToHost);

	int resutRet = h_Result[nBlocks];
	free(h_Result);

	cudaFree(d_Result);
	for(int i = 0; i < nBlocks; i++){
		cudaFree(h_OutQueuePtr[i]);
	}
	free(h_OutQueuePtr);
	free(subListsInit);
	free(subListsSize);
	free(blockSubLists);
	cudaFree(g_InputListPtr);

	return resutRet;
}



