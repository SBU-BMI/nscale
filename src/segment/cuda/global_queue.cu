#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NUM_BLOCKS	30
//#include "global_sync.cu"

#define WARP_SIZE 	32
#define NUM_THREADS	256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_THREADS 8
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)

#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

__device__ volatile int inQueueSize[MAX_NUM_BLOCKS];
__device__ volatile int *inQueuePtr1[MAX_NUM_BLOCKS];
__device__ volatile int inQueueHead[MAX_NUM_BLOCKS];
__device__ volatile int outQueueMaxSize[MAX_NUM_BLOCKS];
__device__ volatile int outQueueHead[MAX_NUM_BLOCKS];
__device__ volatile int *outQueuePtr2[MAX_NUM_BLOCKS];

__device__ volatile int *curInQueue[MAX_NUM_BLOCKS];
__device__ volatile int *curOutQueue[MAX_NUM_BLOCKS];

// This variables are used for debugging purposes only
__device__ volatile int totalInserts[MAX_NUM_BLOCKS];

// Utils...
// http://www.moderngpu.com/intro/scan.html
__device__ void scan(const int* values, int* exclusive) {

	// Reserve a half warp of extra space plus one per warp in the block.
	// This is exactly enough space to avoid comparisons in the multiscan
	// and to avoid bank conflicts.
	__shared__ volatile int scan[NUM_WARPS * SCAN_STRIDE];
	int tid = threadIdx.x;
	int warp = tid / WARP_SIZE;
	int lane = (WARP_SIZE - 1) & tid;

	volatile int* s = scan + SCAN_STRIDE * warp + lane + WARP_SIZE / 2;
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
	__shared__ volatile int totals[NUM_WARPS + NUM_WARPS / 2];
	if(tid < NUM_WARPS) {
		// Grab the block total for the tid'th block. This is the last element
		// in the block's scanned sequence. This operation avoids bank 
		// conflicts.
		int total = scan[SCAN_STRIDE * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

		totals[tid] = 0;
		volatile int* s2 = totals + NUM_WARPS / 2 + tid;
		int totalsSum = total;
		s2[0] = total;

#pragma unroll
		for(int i = 0; i < LOG_NUM_WARPS; ++i) {
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

__device__ int queueElement(int *outQueueCurPtr, int *elements){
	int queue_index = atomicAdd((int*)&outQueueHead[blockIdx.x], 1);
	if(queue_index < outQueueMaxSize[blockIdx.x]){
		curOutQueue[blockIdx.x][queue_index] = elements[0];
	}else{
		queue_index = -1;
	}
	return queue_index;
}


// Assuming that all threads in a block are calling this function
__device__ int queueElement(int *elements){
	int queue_index = -1;
#ifdef	PREFIX_SUM
	__shared__ int writeAddr[NUM_THREADS];
	__shared__ int exclusiveScan[NUM_THREADS];
	__shared__ int global_queue_index;

	if(threadIdx.x == 0){
		global_queue_index = outQueueHead[blockIdx.x];
	}

	// set to the number of values this threard is writing
	writeAddr[threadIdx.x] = elements[0];

	// run a prefix-sum on threads inserting data to the queue
	scan(writeAddr, exclusiveScan);

	// calculate index into the queue where given thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// write elemets sequentially to shared memory
//	int localIndex = exclusiveScan[threadIdx.x];
//	for(int i = 0; i < elements[0]; i++){
//		localElements[localIndex+i] = elements[i+1];
//	}

//	__syncthreads();
//	for(int i = threadIdx.x; i < exclusiveScan[NUM_THREADS-1]+writeAddr[NUM_THREADS-1]; i+=blockDim.x){
//		curOutQueue[blockIdx.x][global_queue_index+i] = localElements[i];
//	}

	for(int i = 0; i < elements[0]; i++){
		curOutQueue[blockIdx.x][queue_index+i] = elements[i+1];
		if(queue_index+i > outQueueMaxSize[blockIdx.x])
			printf("List out of bounds\n");
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHead[blockIdx.x]+=exclusiveScan[NUM_THREADS-1]+writeAddr[NUM_THREADS-1];
//		printf("Inserting = %d - outQueueHead = %d\n", exclusiveScan[NUM_THREADS-1]+writeAddr[NUM_THREADS-1], outQueueHead[blockIdx.x]);
	}
#else
	if(elements[0] != 0){
		queue_index = atomicAdd((int*)&outQueueHead[blockIdx.x], elements[0]);
		if(queue_index < outQueueMaxSize[blockIdx.x]){
			for(int i = 0; i < elements[0];i++){
				curOutQueue[blockIdx.x][queue_index+i] = elements[i+1];
			}
		}else{
			queue_index = -1;
		}
	}
#endif
	return queue_index;
}


// Assuming that all threads in a block are calling this function
__device__ int queueElement(int element){
	int queue_index = -1;
#ifdef	PREFIX_SUM
	__shared__ int writeAddr[NUM_THREADS];
	__shared__ int exclusiveScan[NUM_THREADS];
	__shared__ int global_queue_index;

	if(threadIdx.x == 0){
		global_queue_index = outQueueHead[blockIdx.x];
	}

	// set to 1 threards that are writing
	writeAddr[threadIdx.x] = ((element) != (-1) ? (1):(0));

	// run a prefix-sum on threads inserting data to the queue
	scan(writeAddr, exclusiveScan);

	// calculate index into the queue where give thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// If there is data to be queued, do it
	if(element != -1){
		curOutQueue[blockIdx.x][queue_index] = element;
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHead[blockIdx.x]+=exclusiveScan[NUM_THREADS-1]+writeAddr[NUM_THREADS-1];
	}
#else
	if(element != -1){
		queue_index = atomicAdd((int*)&outQueueHead[blockIdx.x], 1);
		if(queue_index < outQueueMaxSize[blockIdx.x]){
			curOutQueue[blockIdx.x][queue_index] = element;
		}else{
			queue_index = -1;
		}
	}
#endif
	return queue_index;
}

// Makes queue 1 point to queue 2, and vice-versa
__device__ void swapQueues(int loopIt){
	__syncthreads();

	if(loopIt %2 == 0){
		curInQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];
		curOutQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
		if(threadIdx.x == 0){
			inQueueSize[blockIdx.x] = outQueueHead[blockIdx.x];
			outQueueHead[blockIdx.x] = 0;
			inQueueHead[blockIdx.x] = 0;
			// This is used for profiling only
			totalInserts[blockIdx.x]+=inQueueSize[blockIdx.x];
		}
	}else{
		curInQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
		curOutQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];

		if(threadIdx.x == 0){
			inQueueSize[blockIdx.x] = outQueueHead[blockIdx.x];
			outQueueHead[blockIdx.x] = 0;
			inQueueHead[blockIdx.x] = 0;
			// This is used for profiling only
			totalInserts[blockIdx.x]+=inQueueSize[blockIdx.x];
		}
	}
	__syncthreads();
}



// -2, nothing else to be done at all
__device__ int dequeueElement(int *loopIt){
	// did this block got something to do?
	__shared__ volatile int gotWork;

getWork:
	gotWork = 0;


	// Try to get some work.
//	int queue_index = atomicAdd((int*)&inQueueHead, 1);
	int queue_index = inQueueHead[blockIdx.x] + threadIdx.x;
	// I must guarantee that idle threads are set to 0, and no other thread 
	// will come later and set it to 0 again
	__syncthreads();

	if(threadIdx.x == 0){
		inQueueHead[blockIdx.x]+=blockDim.x;
//		if(loopIt[0] < 1){
//			printf("inQueueSize = %d loopIt[0] = %d queue_index = %d outQueueHead = %d\n", inQueueSize[blockIdx.x], loopIt[0], queue_index, outQueueHead[blockIdx.x]);
//		}
	}

	// Nothing to do by default
	int element = -1;
	if(queue_index < inQueueSize[blockIdx.x]){
		element = curInQueue[blockIdx.x][queue_index];
		gotWork = 1;
	}
	__syncthreads();


	// This block does not have anything to process
	if(!gotWork){
//		if(loopIt[0] < 20 && threadIdx.x == 0)
//			printf("inQueueSize = %d loopIt[0] = %d\n", inQueueSize[blockIdx.x], loopIt[0]);
		element = -2;
		if(outQueueHead[blockIdx.x] != 0){
			swapQueues(loopIt[0]);
			loopIt[0]++;
			goto getWork;
		}
	}
	return element;
}

// Initialized queue data structures:
// Initial assumptions: this first kernel should be launched with number of threads at least equal
// to the number of block used with the second kernel
// inQueueData ptr size is same as outQueueMaxSize provided. 
__global__ void initQueue(int *inQueueData, int dataElements, int *outQueueData, int outMaxSize){
	if(threadIdx.x < 1){
		// Simply assign input data pointers/number of elements to the queue
		inQueuePtr1[threadIdx.x] = inQueueData;

//		printf("initQueueVector: tid - %d dataElements = %d pointer = %p\n", threadIdx.x, dataElements, inQueueData);
		inQueueSize[threadIdx.x] = dataElements;

		totalInserts[threadIdx.x] = 0;
		
		// alloc second vector used to queue output elements
		outQueuePtr2[threadIdx.x] = outQueueData;

		// Maximum number of elements that fit into the queue
		outQueueMaxSize[threadIdx.x] = outMaxSize;

		// Head of the out queue
		outQueueHead[threadIdx.x] = 0;

		// Head of the in queue
		inQueueHead[threadIdx.x] = 0;
	}
}


__global__ void initQueueId(int *inQueueData, int dataElements, int *outQueueData, int outMaxSize, int qId){
	if(threadIdx.x < 1){
		// Simply assign input data pointers/number of elements to the queue
		inQueuePtr1[qId] = inQueueData;

//		printf("initQueueVector: tid - %d dataElements = %d pointer = %p\n", threadIdx.x, dataElements, inQueueData);
		inQueueSize[qId] = dataElements;

		totalInserts[qId] = 0;
		
		// alloc second vector used to queue output elements
		outQueuePtr2[qId] = outQueueData;

		// Maximum number of elements that fit into the queue
		outQueueMaxSize[qId] = outMaxSize;

		// Head of the out queue
		outQueueHead[qId] = 0;

		// Head of the in queue
		inQueueHead[qId] = 0;
	}
}


__global__ void initQueueVector(int **inQueueData, int *inQueueSizes, int **outQueueData, int numImages){
	if(threadIdx.x < MAX_NUM_BLOCKS && threadIdx.x < numImages){
//		printf("initQueueVector: tid - %d inQueueSize[%d] = %d pointer = %p outPtr = %p\n", threadIdx.x, threadIdx.x, inQueueSizes[threadIdx.x], inQueueData[threadIdx.x], outQueueData[threadIdx.x]);

		// Simply assign input data pointers/number of elements to the queue
		inQueuePtr1[threadIdx.x] = inQueueData[threadIdx.x];
		inQueueSize[threadIdx.x] = inQueueSizes[threadIdx.x];
		totalInserts[threadIdx.x] = 0;
		
		// alloc second vector used to queue output elements
		outQueuePtr2[threadIdx.x] = outQueueData[threadIdx.x];

		// Maximum number of elements that fit into the queue
		outQueueMaxSize[threadIdx.x] = (inQueueSizes[threadIdx.x]+1000) * 2;

		// Head of the out queue
		outQueueHead[threadIdx.x] = 0;

		// Head of the in queue
		inQueueHead[threadIdx.x] = 0;

	}
}



// Returns what should be queued
__device__ int propagate(int *seeds, unsigned char *image, int x, int y, int ncols, unsigned char pval){
	int returnValue = -1;
	int index = y*ncols + x;
	unsigned char seedXYval = seeds[index];
	unsigned char imageXYval = image[index];

	if((seedXYval < pval) && (imageXYval != seedXYval)){
		unsigned char newValue = min(pval, imageXYval);
		//  this should be a max atomic...
		atomicMax(&(seeds[index]), newValue);
		returnValue = index;
	}
	return returnValue;
}

__global__ void listReduceKernel(int* d_Result, int *seeds, unsigned char *image, int ncols, int nrows){
	curInQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
	curOutQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];

	int loopIt = 0;
	int workUnit = -1;
	int tid = threadIdx.x;
	__shared__ int localQueue[NUM_THREADS][5];

	do{
		int x, y;

		localQueue[tid][0] = 0;
		
		// Try to get some work.
		workUnit = dequeueElement(&loopIt);
		y = workUnit/ncols;
		x = workUnit%ncols;	

		unsigned char pval = 0;

		if(workUnit >= 0){
			pval = seeds[workUnit];
		}

		int retWork = -1;
		if(workUnit > 0){
			retWork = propagate((int*)seeds, image, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);
		if(workUnit > 0){
			retWork = propagate((int*)seeds, image, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit > 0){
			retWork = propagate((int*)seeds, image, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit > 0){
			retWork = propagate((int*)seeds, image, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		queueElement(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[0]=totalInserts[blockIdx.x];
}

extern "C" int listComputation(int *h_Data, int dataElements, int *d_seeds, unsigned char *d_image, int ncols, int nrows){
// seeds contais the maker and it is also the output image

//	uint threadsX = 512;
	int blockNum = 1;
	int *d_Result;

	int *d_Data;
	unsigned int dataSize = dataElements * sizeof(int);
	cudaMalloc((void **)&d_Data, dataSize  );
	cudaMemcpy(d_Data, h_Data, dataSize, cudaMemcpyHostToDevice);

	// alloc space to save output elements in the queue
	int *d_OutVector;
	cudaMalloc((void **)&d_OutVector, sizeof(int) * dataElements);
	
//	printf("Init queue data!\n");
	// init values of the __global__ variables used by the queue
	initQueue<<<1, 1>>>(d_Data, dataElements, d_OutVector, dataElements);

//	init_sync<<<1, 1>>>();
	

	cudaMalloc((void **)&d_Result, sizeof(int) ) ;
	cudaMemset((void *)d_Result, 0, sizeof(int));

//	printf("Run computation kernel!\n");
	listReduceKernel<<<blockNum, NUM_THREADS>>>(d_Result, d_seeds, d_image, ncols, nrows);

//	cutilCheckMsg("histogramKernel() execution failed\n");
	int h_Result;
	cudaMemcpy(&h_Result, d_Result, sizeof(int), cudaMemcpyDeviceToHost);

	printf("	#queue entries = %d\n",h_Result);
	cudaFree(d_Data);
	cudaFree(d_Result);
	cudaFree(d_OutVector);

	// TODO: free everyone
	return h_Result;
}

__global__ void morphReconKernel(int* d_Result, int *seeds, unsigned char *image, int ncols, int nrows){
	curInQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
	curOutQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];

	int loopIt = 0;
	int workUnit = -1;
	int tid = threadIdx.x;
	__shared__ int localQueue[NUM_THREADS][5];

//	printf("inQueueSize = %d\n",inQueueSize[blockIdx.x]);
	__syncthreads();
	do{
		int x, y;

		localQueue[tid][0] = 0;
		
		// Try to get some work.
		workUnit = dequeueElement(&loopIt);
		y = workUnit/ncols;
		x = workUnit%ncols;	

		unsigned char pval = 0;
		if(workUnit >=0){
			pval = seeds[workUnit];
		}

		int retWork = -1;
		if(workUnit >= 0 && y > 0){
			retWork = propagate((int*)seeds, image, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);
		if(workUnit >= 0 && y < nrows-1){
			retWork = propagate((int*)seeds, image, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit >= 0 && x > 0){
			retWork = propagate((int*)seeds, image, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit >= 0 && x < ncols-1){
			retWork = propagate((int*)seeds, image, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		queueElement(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[0]=totalInserts[blockIdx.x];
}

__global__ void morphReconKernelVector(int* d_Result, int **d_SeedsList, unsigned char **d_ImageList, int *d_ncols, int *d_nrows, int connectivity=4){
	curInQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
	curOutQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];

//	if(threadIdx.x == 0){
//		printf("inqueue = %p outqueue = %p ncols = %d nrows = %d connectivity=%d\n", inQueuePtr1[blockIdx.x], outQueuePtr2[blockIdx.x], d_ncols[blockIdx.x], d_nrows[blockIdx.x], connectivity);
//	}
	int *seeds = d_SeedsList[blockIdx.x];
	unsigned char *image = d_ImageList[blockIdx.x];
	int ncols = d_ncols[blockIdx.x];
	int nrows = d_nrows[blockIdx.x];


	int loopIt = 0;
	int workUnit = -1;
	int tid = threadIdx.x;
	__shared__ int localQueue[NUM_THREADS][9];

	__syncthreads();
	do{
		int x, y;

		localQueue[tid][0] = 0;
		
		// Try to get some work.
		workUnit = dequeueElement(&loopIt);
		y = workUnit/ncols;
		x = workUnit%ncols;	

		unsigned char pval = 0;
		if(workUnit >= 0){
			pval = seeds[workUnit];
		}

		int retWork = -1;
		if(workUnit >= 0 && y > 0){
			retWork = propagate((int*)seeds, image, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);
		if(workUnit >= 0 && y < nrows-1){
			retWork = propagate((int*)seeds, image, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit >= 0 && x > 0){
			retWork = propagate((int*)seeds, image, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit >= 0 && x < ncols-1){
			retWork = propagate((int*)seeds, image, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		// if connectivity is 8, four other neighbors have to be verified
		if(connectivity == 8){
			if(workUnit >= 0 && y > 0 && x >0){
				retWork = propagate((int*)seeds, image, x-1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y > 0 && x < ncols-1){
				retWork = propagate((int*)seeds, image, x+1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x >0){
				retWork = propagate((int*)seeds, image, x-1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x <(ncols-1)){
				retWork = propagate((int*)seeds, image, x+1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}

		}
		queueElement(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[blockIdx.x]=totalInserts[blockIdx.x];
}

/// This is an old implementation for this function. Presumably about 1ms faster, but quite more ugly
///extern "C" int morphReconVector(int nImages, int **h_InputListPtr, int* h_ListSize, int **h_Seeds, unsigned char **h_Images, int* h_ncols, int* h_nrows){
///// seeds contais the maker and it is also the output image
///	int blockNum = nImages;
///	int *d_Result;
///
///	// alloc space to save output elements in the queue
///	int **h_OutQueuePtr = (int **)malloc(sizeof(int*) * nImages);;
///
///	for(int i = 0; i < nImages;i++){
///		cudaMalloc((void **)&h_OutQueuePtr[i], sizeof(int) * (h_ListSize[i]+1000) * 2);
///	}
///
///	int **d_OutQueuePtr = NULL;
///	cudaMalloc((void **)&d_OutQueuePtr, sizeof(int*) * nImages);
///	cudaMemcpy(d_OutQueuePtr, h_OutQueuePtr, sizeof(int*) * nImages, cudaMemcpyHostToDevice);
///
///
///
///	printf("nImages = %d\n", nImages);
///
///	int **d_InputListPtr = NULL;
///	cudaMalloc((void **)&d_InputListPtr, sizeof(int*) * nImages);
///	cudaMemcpy(d_InputListPtr, h_InputListPtr, sizeof(int*) * nImages, cudaMemcpyHostToDevice);
///
///
///	int *d_ListSize = NULL;
///	cudaMalloc((void **)&d_ListSize, sizeof(int) * nImages);
///	cudaMemcpy(d_ListSize, h_ListSize, sizeof(int) * nImages, cudaMemcpyHostToDevice);
///
///	// init values of the __global__ variables used by the queue
///	initQueueVector<<<1, nImages>>>(d_InputListPtr, d_ListSize, d_OutQueuePtr, nImages);
///
///	cudaMalloc((void **)&d_Result, sizeof(int)*nImages) ;
///	cudaMemset((void *)d_Result, 0, sizeof(int)*nImages);
///
///	int **d_Seeds = NULL;
///	cudaMalloc((void **)&d_Seeds, sizeof(int*) * nImages);
///	cudaMemcpy(d_Seeds, h_Seeds, sizeof(int*) * nImages, cudaMemcpyHostToDevice);
///
///	unsigned char **d_Images = NULL;
///	cudaMalloc((void **)&d_Images, sizeof(unsigned char*) * nImages);
///	cudaMemcpy(d_Images, h_Images, sizeof(unsigned char*) * nImages, cudaMemcpyHostToDevice);
///
///	int *d_ncols = NULL;
///	cudaMalloc((void **)&d_ncols, sizeof(int) * nImages);
///	cudaMemcpy(d_ncols, h_ncols, sizeof(int) * nImages, cudaMemcpyHostToDevice);
///
///	int *d_nrows = NULL;
///	cudaMalloc((void **)&d_nrows, sizeof(int) * nImages);
///	cudaMemcpy(d_nrows, h_nrows, sizeof(int) * nImages, cudaMemcpyHostToDevice);
///
///	printf("Run computation kernel!\n");
///	morphReconKernelVector<<<blockNum, NUM_THREADS>>>(d_Result, d_Seeds, d_Images, d_ncols, d_nrows);
///
///	cudaError_t errorCode = cudaGetLastError();
///	const char *error = cudaGetErrorString(errorCode);
///	printf("Error after morphRecon = %s\n", error);
///
///	int h_Result;
///	cudaMemcpy(&h_Result, d_Result, sizeof(int), cudaMemcpyDeviceToHost);
///
///	printf("	#queue entries = %d\n",h_Result);
///
//////	cudaFree(d_nrows);
//////	cudaFree(d_ncols);
//////	cudaFree(d_Images);
//////	cudaFree(d_Seeds);
//////	cudaFree(d_InputListPtr);
//////	cudaFree(d_ListSize);
//////      	cudaFree(d_Result);
//////	cudaFree(d_OutQueuePtr);
//////	for(int i = 0; i < nImages; i++){
//////		cudaFree(h_OutQueuePtr[i]);
//////		cudaFree(h_InputListPtr[i]);
//////	}
//////	free(h_OutQueuePtr);
///
///	// TODO: free everyone
///	return h_Result;
///}


extern "C" int morphReconVector(int nImages, int **h_InputListPtr, int* h_ListSize, int **h_Seeds, unsigned char **h_Images, int* h_ncols, int* h_nrows, int connectivity){
// seeds contais the maker and it is also the output image
	int blockNum = nImages;
	int *d_Result;

	// alloc space to save output elements in the queue
	int **h_OutQueuePtr = (int **)malloc(sizeof(int*) * nImages);;

	for(int i = 0; i < nImages;i++){
		cudaMalloc((void **)&h_OutQueuePtr[i], sizeof(int) * (h_ListSize[i]+1000) * 2);
	}
	
	// Init queue for each images. yes, this may not be the most efficient way, but the code is far easier to read. 
	// Another version, where all pointer are copied at once to the GPU was also built, buit it was only about 1ms 
	// faster. Thus, we decide to go with this version 
	for(int i = 0; i < nImages;i++)
		initQueueId<<<1, 1>>>(h_InputListPtr[i], h_ListSize[i], h_OutQueuePtr[i], (h_ListSize[i]+1000) *2, i);
	
	cudaMalloc((void **)&d_Result, sizeof(int)*nImages) ;
	cudaMemset((void *)d_Result, 0, sizeof(int)*nImages);

	int **d_Seeds = NULL;
	cudaMalloc((void **)&d_Seeds, sizeof(int*) * nImages);
	cudaMemcpy(d_Seeds, h_Seeds, sizeof(int*) * nImages, cudaMemcpyHostToDevice);

	unsigned char **d_Images = NULL;
	cudaMalloc((void **)&d_Images, sizeof(unsigned char*) * nImages);
	cudaMemcpy(d_Images, h_Images, sizeof(unsigned char*) * nImages, cudaMemcpyHostToDevice);

	int *d_ncols = NULL;
	cudaMalloc((void **)&d_ncols, sizeof(int) * nImages);
	cudaMemcpy(d_ncols, h_ncols, sizeof(int) * nImages, cudaMemcpyHostToDevice);

	int *d_nrows = NULL;
	cudaMalloc((void **)&d_nrows, sizeof(int) * nImages);
	cudaMemcpy(d_nrows, h_nrows, sizeof(int) * nImages, cudaMemcpyHostToDevice);

//	printf("Run computation kernel!\n");
	morphReconKernelVector<<<blockNum, NUM_THREADS>>>(d_Result, d_Seeds, d_Images, d_ncols, d_nrows, connectivity);

	if(cudaGetLastError() != cudaSuccess){
		cudaError_t errorCode = cudaGetLastError();
		const char *error = cudaGetErrorString(errorCode);
		printf("Error after morphRecon = %s\n", error);
	}

	int *h_Result = (int *) malloc(sizeof(int) * blockNum);
	cudaMemcpy(h_Result, d_Result, sizeof(int) * blockNum, cudaMemcpyDeviceToHost);

	int resutRet = h_Result[0];
//	printf("	#queue entries = %d\n",h_Result[0]);
	free(h_Result);

	cudaFree(d_nrows);
	cudaFree(d_ncols);
	cudaFree(d_Images);
	cudaFree(d_Seeds);
	cudaFree(d_Result);
	for(int i = 0; i < nImages; i++){
		cudaFree(h_OutQueuePtr[i]);
		cudaFree(h_InputListPtr[i]);
	}
	free(h_OutQueuePtr);

	return resutRet;
}

__global__ void morphReconKernelSpeedup(int* d_Result, int *d_Seeds, unsigned char *d_Image, int ncols, int nrows, int connectivity=4){
	curInQueue[blockIdx.x] = inQueuePtr1[blockIdx.x];
	curOutQueue[blockIdx.x] = outQueuePtr2[blockIdx.x];
	int *seeds = d_Seeds;
	unsigned char *image = d_Image;

//	if(threadIdx.x == 0){
//		printf("inqueue = %p outqueue = %p ncols = %d nrows = %d connectivity=%d\n", inQueuePtr1[blockIdx.x], outQueuePtr2[blockIdx.x], ncols, nrows, connectivity);
//	}
//	int *seeds = d_SeedsList[blockIdx.x];
//	unsigned char *image = d_ImageList[blockIdx.x];
//	int ncols = d_ncols[blockIdx.x];
//	int nrows = d_nrows[blockIdx.x];


	int loopIt = 0;
	int workUnit = -1;
	int tid = threadIdx.x;
	__shared__ int localQueue[NUM_THREADS][9];

	__syncthreads();
	do{
		int x, y;

		localQueue[tid][0] = 0;
		
		// Try to get some work.
		workUnit = dequeueElement(&loopIt);
		y = workUnit/ncols;
		x = workUnit%ncols;	

		unsigned char pval = 0;
		if(workUnit >= 0){
			pval = seeds[workUnit];
		}

		int retWork = -1;
		if(workUnit >= 0 && y > 0){
			retWork = propagate((int*)seeds, image, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);
		if(workUnit >= 0 && y < nrows-1){
			retWork = propagate((int*)seeds, image, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit >= 0 && x > 0){
			retWork = propagate((int*)seeds, image, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit >= 0 && x < ncols-1){
			retWork = propagate((int*)seeds, image, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		// if connectivity is 8, four other neighbors have to be verified
		if(connectivity == 8){
			if(workUnit >= 0 && y > 0 && x >0){
				retWork = propagate((int*)seeds, image, x-1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y > 0 && x < ncols-1){
				retWork = propagate((int*)seeds, image, x+1, y-1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x >0){
				retWork = propagate((int*)seeds, image, x-1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}
			if(workUnit >= 0 && y < (nrows-1) && x <(ncols-1)){
				retWork = propagate((int*)seeds, image, x+1, y+1, ncols, pval);
				if(retWork > 0){
					localQueue[tid][0]++;
					localQueue[tid][localQueue[tid][0]] = retWork;
				}
			}

		}
		queueElement(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[blockIdx.x]=totalInserts[blockIdx.x];
}



extern "C" int morphReconSpeedup( int *g_InputListPtr, int h_ListSize, int *g_Seed, unsigned char *g_Image, int h_ncols, int h_nrows, int connectivity){
// seeds contais the maker and it is also the output image
	int nImages = 1;
	// TODO: change blockNum to nBlocks
//	int nBlocks = nImages;
	int *d_Result;
	int nBlocks = 16;

	// alloc space to save output elements in the queue for each block
	int **h_OutQueuePtr = (int **)malloc(sizeof(int*) * nBlocks);

	// at this moment I should partition the INPUT queue
	printf("List size = %d\n", h_ListSize);
	int tempNblocks = nBlocks;

	int subListsInit[tempNblocks];
	int subListsEnd[tempNblocks];
	int subListsSize[tempNblocks];

	for(int i = 0; i < tempNblocks; i++){
		int curSubListInit = (h_ListSize/tempNblocks)*i;
		int curSubListEnd = ((i+1<tempNblocks)?((i+1)*(h_ListSize/tempNblocks)-1):(h_ListSize-1));
//		printf("BlockId = %d - init = %d end = %d size=%d\n", i, curSubListInit, curSubListEnd, curSubListEnd-curSubListInit+1);
		subListsInit[i] = curSubListInit;
		subListsEnd[i] = curSubListEnd;
		subListsSize[i]	= curSubListEnd-curSubListInit+1;
	}

// Adding code
	// TODO: free data
	int *blockSubLists[tempNblocks];
	for(int i = 0; i < tempNblocks; i++){
		cudaMalloc((void **)&blockSubLists[i], sizeof(int)*(subListsSize[i]+1000) * 2);
		cudaMemcpy(blockSubLists[i], &g_InputListPtr[subListsInit[i]], subListsSize[i] * sizeof(int), cudaMemcpyDeviceToDevice);
	}


// End adding code

	printf("h_listSize = %d subListsSize[0]=%d\n", h_ListSize, subListsSize[0]);
//	cout << "h_listSize = "<< h_ListSize<< " subListsSize[0]="<< subListsSize[0] <<endl;
	
	for(int i = 0; i < tempNblocks;i++){
		cudaMalloc((void **)&h_OutQueuePtr[i], sizeof(int) * (subListsSize[i]+1000) * 2);
	}
	
	// Init queue for each image. yes, this may not be the most efficient way, but the code is far easier to read. 
	// Another version, where all pointer are copied at once to the GPU was also built, buit it was only about 1ms 
	// faster. Thus, we decide to go with this version 
//	for(int i = 0; i < nBlocks;i++)
//		initQueueId<<<1, 1>>>(h_InputListPtr[i], h_ListSize[i], h_OutQueuePtr[i], (h_ListSize[i]+1000) *2, i);
	for(int i = 0; i < nBlocks;i++)
		initQueueId<<<1, 1>>>(blockSubLists[i], subListsSize[i], h_OutQueuePtr[i], (subListsSize[i]+1000) *2, i);
//		initQueueId<<<1, 1>>>(g_InputListPtr, h_ListSize, h_OutQueuePtr[i], (h_ListSize+1000) *2, i);

	// This is used by each block to store the number of queue operations performed
	cudaMalloc((void **)&d_Result, sizeof(int)*nBlocks) ;
	cudaMemset((void *)d_Result, 0, sizeof(int)*nBlocks);


//	printf("Run computation kernel!\n");
	morphReconKernelSpeedup<<<nBlocks, NUM_THREADS>>>(d_Result, g_Seed, g_Image, h_ncols, h_nrows, connectivity);

	if(cudaGetLastError() != cudaSuccess){
		cudaError_t errorCode = cudaGetLastError();
		const char *error = cudaGetErrorString(errorCode);
		printf("Error after morphRecon = %s\n", error);
	}

	int *h_Result = (int *) malloc(sizeof(int) * nBlocks);
	cudaMemcpy(h_Result, d_Result, sizeof(int) * nBlocks, cudaMemcpyDeviceToHost);

	int resutRet = h_Result[0];
	for(int i = 0; i < nBlocks; i++){
		printf("	block# %d, #entries=%d\n", i, h_Result[i]);
	}
	free(h_Result);

	cudaFree(d_Result);
	for(int i = 0; i < nBlocks; i++){
		cudaFree(h_OutQueuePtr[i]);
	}
	free(h_OutQueuePtr);
	cudaFree(g_InputListPtr);

	return resutRet;
}



extern "C" int morphRecon(int *d_input_list, int dataElements, int *d_seeds, unsigned char *d_image, int ncols, int nrows){
// seeds contais the maker and it is also the output image
	int blockNum = 1;
	int *d_Result;

	// alloc space to save output elements in the queue
	int *d_OutVector;
	cudaMalloc((void **)&d_OutVector, sizeof(int) * (dataElements+1000) * 2 );
	
	// init values of the __global__ variables used by the queue
	initQueue<<<1, 1>>>(d_input_list, dataElements, d_OutVector, (dataElements+1000) * 2);

	cudaMalloc((void **)&d_Result, sizeof(int) ) ;
	cudaMemset((void *)d_Result, 0, sizeof(int));

//	printf("Run computation kernel!\n");
	morphReconKernel<<<blockNum, NUM_THREADS>>>(d_Result, d_seeds, d_image, ncols, nrows);
	cudaError_t errorCode = cudaGetLastError();
	const char *error = cudaGetErrorString(errorCode);
	printf("Error after morphRecon = %s\n", error);

	int h_Result;
	cudaMemcpy(&h_Result, d_Result, sizeof(int), cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

	printf("	#queue entries = %d\n",h_Result);
	cudaFree(d_Result);
	cudaFree(d_OutVector);

	// TODO: free everyone
	return h_Result;
}
