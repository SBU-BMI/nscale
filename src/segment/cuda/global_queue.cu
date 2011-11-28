#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "queue_common.h"
#include "global_sync.cu"
//#include <sm_11_atomic_functions.h>

#define WARP_SIZE 	32
#define NUM_THREADS	512 
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_THREADS 9
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)

#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

__device__ volatile int inQueueSize;
__device__ volatile int *inQueuePtr1;
__device__ volatile int inQueueHead;
__device__ volatile int outQueueMaxSize;
__device__ volatile int outQueueHead;
__device__ volatile int *outQueuePtr2;
__device__ volatile int syncClock;
__device__ volatile int eow;
__device__ volatile int totalInserts;

__device__ volatile int *curInQueue;
__device__ volatile int *curOutQueue;

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










// This variable is used for debugging purposes only
__device__ volatile int nQueueSwaps;
__device__ int swapIt;

__device__ int queueElement(int *outQueueCurPtr, int *elements){
	int queue_index = atomicAdd((int*)&outQueueHead, 1);
	if(queue_index < outQueueMaxSize){
		curOutQueue[queue_index] = elements[0];
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
		global_queue_index = outQueueHead;
	}

	// set to 1 threards that are writing
	writeAddr[threadIdx.x] = elements[0];//((element) != (-1) ? (1):(0));

	// run a prefix-sum on threads inserting data to the queue
	scan(writeAddr, exclusiveScan);

	// calculate index into the queue where given thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	for(int i = 0; i < elements[0]; i++){
		curOutQueue[queue_index+i] = elements[i+1];
	}
	// If there is data to be queued, do it
//	if(element != -1){
//		curOutQueue[queue_index] = element;
//	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHead+=exclusiveScan[NUM_THREADS-1]+writeAddr[NUM_THREADS-1];
	}
#else
	if(element != -1){
		queue_index = atomicAdd((int*)&outQueueHead, 1);
		if(queue_index < outQueueMaxSize){
			curOutQueue[queue_index] = element;
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
		global_queue_index = outQueueHead;
	}

	// set to 1 threards that are writing
	writeAddr[threadIdx.x] = ((element) != (-1) ? (1):(0));

	// run a prefix-sum on threads inserting data to the queue
	scan(writeAddr, exclusiveScan);

	// calculate index into the queue where give thread is writing
	queue_index = global_queue_index+exclusiveScan[threadIdx.x];

	// If there is data to be queued, do it
	if(element != -1){
		curOutQueue[queue_index] = element;
	}

	// thread 0 updates head of the queue
	if(threadIdx.x == 0){
		outQueueHead+=exclusiveScan[NUM_THREADS-1]+writeAddr[NUM_THREADS-1];
	}
#else
	if(element != -1){
		queue_index = atomicAdd((int*)&outQueueHead, 1);
		if(queue_index < outQueueMaxSize){
			curOutQueue[queue_index] = element;
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
		curInQueue = outQueuePtr2;
		curOutQueue = inQueuePtr1;
		if(threadIdx.x == 0 && blockIdx.x == 0){
			inQueueSize = outQueueHead;
			outQueueHead = 0;
			inQueueHead = 0;
			// This is used for profiling only
			totalInserts+=inQueueSize;
		}
	}else{
		curInQueue = inQueuePtr1;
		curOutQueue = outQueuePtr2;

		if(threadIdx.x == 0 && blockIdx.x == 0){
			inQueueSize = outQueueHead;
			outQueueHead = 0;
			inQueueHead = 0;
			// This is used for profiling only
			totalInserts+=inQueueSize;

		}
	}
	__syncthreads();
}



// -2, nothing else to be done at all
__device__ int dequeueElement(int *loopIt){
	// did this block got something to do?
	__shared__ int gotWork;

getWork:
	gotWork = 0;


	// Try to get some work.
//	int queue_index = atomicAdd((int*)&inQueueHead, 1);
	int queue_index = inQueueHead + threadIdx.x;
	// I must guarantee that idle threads are set to 0, and no other thread 
	// will come later and set it to 0 again
	__syncthreads();

	if(threadIdx.x == 0){
		inQueueHead+=blockDim.x;
	}

	// Nothing to do by default
	int element = -1;
	if(queue_index < inQueueSize){
		element = curInQueue[queue_index];
		gotWork = 1;
	}
	__syncthreads();


	// This block does not have anything to process
	if(!gotWork){
		element = -2;
		if(outQueueHead != 0){
			swapQueues(loopIt[0]);
			loopIt[0]++;
			goto getWork;
		}
	}
	return element;
}

// Initialized queue data structures:
// Initial assumptions:
// inQueueData ptr size is same as outQueueMaxSize provided. 
__global__ void initQueue(int *inQueueData, int dataElements, int *outQueueData, int outMaxSize){
	// Simply assign input data pointers/number of elements to the queue
	inQueuePtr1 = inQueueData;
	inQueueSize = dataElements;

	// alloc second vector used to queue output elements
	outQueuePtr2 = outQueueData;

	// Maximum number of elements that fit into the queue
	outQueueMaxSize = outMaxSize;

	// Head of the out queue
	outQueueHead = 0;

	// Head of the in queue
	inQueueHead = 0;

	// It it a of Lamport's logical clock, and it 
	// ticks for each round of syncronization
	syncClock = 1;

	totalInserts = 0;
	eow = 0;
	nQueueSwaps=0;	
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
	curInQueue = inQueuePtr1;
	curOutQueue = outQueuePtr2;

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

		unsigned char pval = seeds[workUnit];

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

	d_Result[0]=totalInserts;
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
	curInQueue = inQueuePtr1;
	curOutQueue = outQueuePtr2;

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

		unsigned char pval = seeds[workUnit];

		int retWork = -1;
		if(workUnit > 0 && y > 0){
			retWork = propagate((int*)seeds, image, x, y-1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);
		if(workUnit > 0 && y < nrows-1){
			retWork = propagate((int*)seeds, image, x, y+1, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit > 0 && x > 0){
			retWork = propagate((int*)seeds, image, x-1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
//		queueElement(retWork);

		if(workUnit > 0 && x < ncols-1){
			retWork = propagate((int*)seeds, image, x+1, y, ncols, pval);
			if(retWork > 0){
				localQueue[tid][0]++;
				localQueue[tid][localQueue[tid][0]] = retWork;
			}
		}
		queueElement(localQueue[tid]);

	}while(workUnit != -2);

	d_Result[0]=totalInserts;
}



extern "C" int morphRecon(int *d_input_list, int dataElements, int *d_seeds, unsigned char *d_image, int ncols, int nrows){
// seeds contais the maker and it is also the output image
	int blockNum = 1;
	int *d_Result;

	// alloc space to save output elements in the queue
	int *d_OutVector;
	cudaMalloc((void **)&d_OutVector, sizeof(int) * dataElements*2);
	
	// init values of the __global__ variables used by the queue
	initQueue<<<1, 1>>>(d_input_list, dataElements, d_OutVector, dataElements);

	cudaMalloc((void **)&d_Result, sizeof(int) ) ;
	cudaMemset((void *)d_Result, 0, sizeof(int));

//	printf("Run computation kernel!\n");
	morphReconKernel<<<blockNum, NUM_THREADS>>>(d_Result, d_seeds, d_image, ncols, nrows);

	int h_Result;
	cudaMemcpy(&h_Result, d_Result, sizeof(int), cudaMemcpyDeviceToHost);

	printf("	#queue entries = %d\n",h_Result);
	cudaFree(d_Result);
	cudaFree(d_OutVector);

	// TODO: free everyone
	return h_Result;
}
