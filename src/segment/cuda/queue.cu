// testing gpu queue (compacted array)

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <stdio.h>

#define WARP_SIZE 32
#define NUM_WARPS 16
// blocksize: threads should be less than 1024.
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)
// also blocks should be arranged in 2D (would be 32 bit at 64k x 64k).  preferrably in a shape that's closest to square.


namespace nscale { namespace gpu {


// does not touch other warps
template<typename T>
inline __device__ void warp_mark(const T* s_in_data, int* s_mark, const int tid) {
	s_mark[tid] = (s_in_data[tid] > 0 ? 1 : 0);
} 

// s_mark and s_scan pointers point to the starting pos of current warp's data
// idx is the id within the warp.
// first WARP_SIZE in s_scan is dummy so to avoid warp divergence in warp_scan.
// second WARP_SIZE in s_scan is the scan area.
// does not touch other warps
inline __device__ void init_warp_scan(int* s_mark, int* s_scan, const int idx, const int tid) {
	s_scan[idx] = 0;  // extra padding for the scan part...
	s_scan[idx + WARP_SIZE] = s_mark[tid];
}

// adapted from CudPP.  inclusive scan.
// does not touch other warps
template<int maxlevel>
inline __device__ int warp_scan(int* s_scan, const int idx) {
	int t = s_scan[idx];
	if (0 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 1]; }
	if (1 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 2]; }
	if (2 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 4]; }
	if (3 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 8]; }
	if (4 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx -16]; }
	return s_scan[WARP_SIZE - 1]; // return the total
} 

// s_out_data points to the beginning of the shared array.
// s_scan points to the scanned position for this warp
// s_scan should be exclusive
// return total selected for the warp
// touches other warps, but can rely on warp execution ordering.
template<typename T>
inline __device__ void warp_select(const T* s_in_data, const int* s_mark, const int* s_scan, T* s_out_data, const int offset, const int idx, const int tid, const int warpId) {
	//if (tid % 32 == 5) printf("%d output is %d\n", tid, s_out_data[tid]);
	if (warpId == 1) {
		printf("%d %d scan %d, mark %d\n", warpId, idx, s_scan[idx-1], s_mark[tid]);
	}
	if (s_mark[tid] > 0) {
		if (idx == 1) printf("%d scan position %d, offset %d\n", tid, s_scan[idx-1], offset);
		s_out_data[s_scan[idx-1] + offset] = s_in_data[tid];
	}
} 

// unordered
template<typename T>
__global__ void unordered_select(const T* in_data, const int dataSize, T* out_data, unsigned int* queue_size) {

	// initialize the variables
	const int x = threadIdx.x + blockDim.x * (blockIdx.y + blockIdx.x * gridDim.y);
	if (x >= dataSize - (dataSize & (WARP_SIZE - 1)) + WARP_SIZE) return;

	const int idx = threadIdx.x & (WARP_SIZE - 1);
	const int warpId = threadIdx.x >> 5;

	//if (threadIdx.x == 0) printf("block %d %d, thread %d, warpid %d, x %d\n", blockIdx.x, blockIdx.y, threadIdx.x, warpId, x);


	__shared__ int offsets[WARP_SIZE + 1];  // avoid divergence - everyone write...  only using NUM_WARPS
	__shared__ int s_mark[BLOCK_SIZE];
	__shared__ int s_scan[NUM_WARPS][WARP_SIZE * 2 + 1];
	__shared__ int s_block_scan[WARP_SIZE * 2];   // warp size is 32, block size is 1024, so at most we have 32 warps.  so top scan would require 1 warp.
	__shared__ T s_in_data[BLOCK_SIZE];
	__shared__ T s_out_data[BLOCK_SIZE];
	__shared__ int curr_pos[1];

	// copy in data
	if (warpId == 0) {
		offsets[idx] = 0;
		offsets[WARP_SIZE] = 0;
	}
	__syncthreads();

	s_out_data[threadIdx.x] = 0;
	s_in_data[threadIdx.x] = 0;
	if (x < dataSize) s_in_data[threadIdx.x] = in_data[x];

	// compact within this block
	warp_mark(s_in_data, s_mark, threadIdx.x);  // mark the data to be processed
	init_warp_scan(s_mark, s_scan[warpId], idx, threadIdx.x);
	offsets[warpId + 1] = warp_scan<5>(s_scan[warpId] + WARP_SIZE, idx);  // perform the in warp scan.

	// now scan the warp offsets - want exclusive scan hence the idx+1.  note that this is done by 1 warp only, so need thread sync before and after.
	__syncthreads();
	if (warpId == 0) {
		init_warp_scan(offsets + 1, s_block_scan, idx, idx);
		warp_scan<5>(s_block_scan + WARP_SIZE, idx);
//		printf("warpId %d offsets: %d, blockscan %d %d\n", idx, offsets[idx+1], s_block_scan[idx], s_block_scan[idx + WARP_SIZE]);
		offsets[idx + 1] = s_block_scan[idx + WARP_SIZE];
		printf("222 warpId %d offsets: %d, blockscan %d %d\n", idx, offsets[idx+1], s_block_scan[idx], s_block_scan[idx + WARP_SIZE]);
	}
	__syncthreads();

	warp_select(s_in_data, s_mark, s_scan[warpId] + WARP_SIZE, s_out_data, offsets[warpId], idx, threadIdx.x, warpId);  // compact the data into the block space.

	//copy the data back out.  this block will get a place to write using atomic add.  resulting queue has the blocks shuffled
	// this part is multiblock?
	int block_len = offsets[WARP_SIZE];
//	if (threadIdx.x == 0)
//		printf("block %d, %d, thread %d, warp %d, total = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, warpId, block_len);
	int curr_p = 0;
	if (block_len > 0) {
		if (threadIdx.x == 0) {
			curr_pos[0] = atomicAdd(queue_size, block_len); // only done by first thread in the block
			printf("before block %d %d curr pos %d, block len %d \n ", blockIdx.x, blockIdx.y, curr_pos[0], block_len);
		}
		curr_p = curr_pos[0];  // move from a single shared memory location to threads' registers
		if (threadIdx.x == 10) printf("after block %d %d curr pos %d, block len %d \n ", blockIdx.x, blockIdx.y, curr_p, block_len);
		if (threadIdx.x < block_len) out_data[curr_p + threadIdx.x] = s_out_data[threadIdx.x];   // dont need to worry about dataSize.  queue size is smaller...
	}
	if (x < dataSize) out_data[x] = s_out_data[threadIdx.x];
}

template<typename T>
__global__ void clear(T* out_data, const int dataSize) {
	const int x = threadIdx.x + blockDim.x * (blockIdx.y + blockIdx.x * gridDim.y);
	if (x >= dataSize - (dataSize & (WARP_SIZE - 1)) + WARP_SIZE) return;

	if (x < dataSize) out_data[x] = 0;
}


// use after the gapped compact.  (global sync for all blocks.
// step 2 of the compacting.  assumes that within each block the values have already been compacted.
// also block_pos is already scanned to produce final starting positions for each threadblock.
template<typename T>
__global__ void compact(const T* in_data, const int dataSize, const int* block_pos, T* out_data, unsigned int* queue_size) {
	const int x = threadIdx.x + blockDim.x * (blockIdx.y + blockIdx.x * gridDim.y);
	if (x >= dataSize - (dataSize & (WARP_SIZE - 1)) + WARP_SIZE) return;

	const int pos = block_pos[(blockIdx.y + blockIdx.x * gridDim.y)];
	const int len = block_pos[(blockIdx.y + blockIdx.x * gridDim.y) + 1] - pos;

	if (threadIdx.x < len) out_data[pos + threadIdx.x] = in_data[x];

	// do a global reduction to get the queue size.
	if (threadIdx.x == 0) atomicAdd(queue_size, len);
}


// gapped.  so need to have another step to remove the gaps...  block_pos stores the lengths of the blcok queue for each block
template<typename T>
__global__ void gapped_select(const T* in_data, const int dataSize, T* out_data, int* block_pos) {
	const int x = threadIdx.x + blockDim.x * (blockIdx.y + blockIdx.x * gridDim.y);
	if (x >= dataSize - (dataSize & (WARP_SIZE - 1)) + WARP_SIZE) return;

	const int idx = threadIdx.x & (WARP_SIZE - 1);
	const int warpId = threadIdx.x >> 5;

	//if (blockIdx.x == 0 && threadIdx.x == 0) printf("block %d %d, thread %d, warpid %d, blockdim %d, gridDim %d, x %d\n", blockIdx.x, blockIdx.y, threadIdx.x, warpId, blockDim.x, gridDim.y, x);

	__shared__ int offsets[WARP_SIZE + 1];  // avoid divergence - everyone write...  only using NUM_WARPS
	__shared__ int s_mark[BLOCK_SIZE];
	__shared__ int s_scan[NUM_WARPS][WARP_SIZE * 2 + 1];
	__shared__ int s_block_scan[WARP_SIZE * 2];   // warp size is 32, block size is 1024, so at most we have 32 warps.  so top scan would require 1 warp.
	__shared__ T s_in_data[BLOCK_SIZE];
	__shared__ T s_out_data[BLOCK_SIZE];

	// copy in data
	if (warpId == 0) {
		offsets[idx] = 0;
		offsets[WARP_SIZE] = 0;
	}
	__syncthreads();

	s_out_data[threadIdx.x] = 0;
	s_in_data[threadIdx.x] = 0;
	if (x < dataSize) s_in_data[threadIdx.x] = in_data[x];

	// scan the warps
	warp_mark(s_in_data, s_mark, threadIdx.x);  // mark the data to be processed
	init_warp_scan(s_mark, s_scan[warpId], idx, threadIdx.x);
	offsets[warpId + 1] = warp_scan<5>(s_scan[warpId] + WARP_SIZE, idx);  // perform the in warp scan.

	// now scan the warp offsets - want exclusive scan hence the idx+1.  note that this is done by 1 warp only, so need thread sync before and after.
	__syncthreads();
	if (warpId == 0) {
		init_warp_scan(offsets + 1, s_block_scan, idx, idx);
		warp_scan<5>(s_block_scan + WARP_SIZE, idx);
		//printf("warpId %d offsets: %d, blockscan %d %d\n", idx, offsets[idx+1], s_block_scan[idx], s_block_scan[idx + WARP_SIZE]);
		offsets[idx + 1] = s_block_scan[idx + WARP_SIZE];
		//printf("222 warpId %d offsets: %d, blockscan %d %d\n", idx, offsets[idx+1], s_block_scan[idx], s_block_scan[idx + WARP_SIZE]);
		block_pos[(blockIdx.y + blockIdx.x * gridDim.y) + 1] = offsets[WARP_SIZE];
	}
	__syncthreads();

	// now do the per warp select
	warp_select(s_in_data, s_mark, s_scan[warpId] + WARP_SIZE - 1, s_out_data, offsets[warpId], idx, threadIdx.x, warpId);  // compact the data into the block space.

	//if (threadIdx.x == 0) printf("tada: %d\n", s_out_data[threadIdx.x]);
	//copy the data back out.  leaving the space between blocks.
	if (x < dataSize) out_data[x] = s_out_data[threadIdx.x];
//	if (threadIdx.x == 0)
//		printf("block %d, %d, thread %d, warp %d, total = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, warpId, block_pos[(blockIdx.y + blockIdx.x * gridDim.y) + 1]);

}



// fermi can have maximum of 65K blocks in one dim.
//1024 threads - warpscan all, then 1 warp to scan, then everyone add.
// s_totals has size of 32.
inline __device__ void scan1024(const int* in_data, const int dataSize, int* out_data, int** s_scan, int* s_scan2, int* block_total) {
	const int x = threadIdx.x + blockDim.x * (blockIdx.y + blockIdx.x * gridDim.y);
	if (x >= dataSize - (dataSize & (WARP_SIZE - 1)) + WARP_SIZE) return;

	const int idx = threadIdx.x & (WARP_SIZE - 1);
	const int warpId = threadIdx.x >> 5;

	// initialize data:
	if (threadIdx.x < WARP_SIZE) {
		s_scan2[idx] = 0;
		s_scan2[idx + WARP_SIZE] = 0;
	}
	s_scan[warpId][idx] = 0;
	s_scan[warpId][idx + WARP_SIZE] = 0;
	if (x < dataSize) s_scan[warpId][idx + WARP_SIZE] = in_data[x];

	// do the scan
	s_scan2[warpId+WARP_SIZE] = warp_scan<5>(s_scan[warpId] + WARP_SIZE, idx);
	__syncthreads();

	// do the second pass - only the first block..
	if (threadIdx.x < WARP_SIZE)
		block_total[(blockIdx.y + blockIdx.x * gridDim.y)] = warp_scan<5>(s_scan2 + WARP_SIZE, idx);  // inclusive scan
	__syncthreads();

	// now add back to the warps
	if (x < dataSize) out_data[x] = s_scan[warpId][idx+WARP_SIZE] + s_scan2[warpId + WARP_SIZE - 1];
}

// to scan a large amount of data, do it in multilevel way....  allocate the summary array, scan the blocks with a kernel call, then scan the summary array.  recurse. then add the results to previous level summary




// connectivity:  need to have border of 0 ,and should be continuous
template <typename T>
unsigned int SelectCPUTesting(const T* in_data, const int size, T* out_data) {

	// cpu
	unsigned int newId = 0;
	for (int i = 0; i < size; i++) {
		if (in_data[i] > 0) {
			out_data[newId] = in_data[i];
			++newId;
		}
	}
	return newId;
}

// this functor returns true if the argument is odd, and false otherwise
template <typename T>
struct GreaterThanConst : public thrust::unary_function<T,bool>
{
	const T k;

	__host__ __device__
	GreaterThanConst(T _k) : k(_k) {}

    __host__ __device__
    bool operator()(T x)
    {
    	return x > k;
    }
};
// connectivity:  need to have border of 0 ,and should be continuous
template <typename T>
unsigned int SelectThrustScanTesting(const T* in_data, const int size, T* out_data, cudaStream_t stream) {

	// get data to GPU
	T *d_in_data, *d_out_data;
	cudaMalloc(&d_in_data, sizeof(T) * size);
	cudaMalloc(&d_out_data, sizeof(T) * size);
	cudaMemcpy(d_in_data, in_data, sizeof(T) * size, cudaMemcpyHostToDevice);
	cudaMemset(d_out_data, 0, sizeof(T) * size);

	// thrust
	thrust::device_ptr<T> queueBegin(d_in_data);
	thrust::device_ptr<T> queueEnd(d_in_data + size);

	// can change into transform_iterator to use in the copy operation.  the only challenge is don't know queue size, and would still need to compact later...
	// count
//	unsigned queueSize = thrust::count_if(queueBegin, queueEnd, GreaterThanConst<T>(0));
	thrust::device_ptr<T> queueBegin2(d_out_data);
	thrust::device_ptr<T> queueEnd2 = thrust::copy_if(queueBegin, queueEnd, queueBegin2, GreaterThanConst<T>(0));

	unsigned int queueSize = queueEnd2.get() - queueBegin2.get();

	cudaMemcpy(out_data, d_out_data, sizeof(T) * size, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();

	cudaFree(d_in_data);
	cudaFree(d_out_data);

	return queueSize;

}




// warp-scan
// connectivity:  need to have border of 0 ,and should be continuous
template <typename T>
unsigned int SelectWarpScanUnorderedTesting(const T* in_data, const int size, T* out_data, cudaStream_t stream) {

	dim3 threads( BLOCK_SIZE, 1);
	unsigned int numBlocks = size / threads.x + (size % threads.x > 0 ? 1 : 0);
	unsigned int minbx = (unsigned int) ceil(sqrt((double)numBlocks));
	unsigned int minby = numBlocks / minbx + (numBlocks % minbx > 0 ? 1 : 0);
	dim3 blocks( minbx, minby );

	// get data to GPU
	T *d_in_data;
	T *d_out_data;
	cudaMalloc(&d_in_data, sizeof(T) * size);
	cudaMalloc(&d_out_data, sizeof(T) * size);
	cudaMemcpy(d_in_data, in_data, sizeof(T) * size, cudaMemcpyHostToDevice);
	cudaMemset(d_out_data, 0, sizeof(T) * size);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();
	unsigned int *d_queue_size;
	cudaMalloc(&d_queue_size, sizeof(unsigned int));
	cudaMemset(d_queue_size, 0, sizeof(unsigned int));

	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();
	printf("blocks: %d, %d, threads: %d\n", blocks.x, blocks.y, threads.x);
	//	::nscale::gpu::unordered_select<<<blocks, threads, 0, stream >>>(d_in_data, size, d_out_data, d_queue_size);
	unordered_select<<<blocks, threads >>>(d_in_data, size, d_out_data, d_queue_size);

	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();

	// get data off gpu
	unsigned int queue_size = 0;
	cudaMemcpy((void*)&queue_size, (void*)d_queue_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_data, d_out_data, sizeof(T) * size, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();

	cudaFree(d_in_data);
	cudaFree(d_out_data);
	cudaFree(d_queue_size);

	return queue_size;

}

// warp-scan
// connectivity:  need to have border of 0 ,and should be continuous
template <typename T>
unsigned int SelectWarpScanOrderedTesting(const T* in_data, const int size, T* out_data, cudaStream_t stream) {

	dim3 threads( BLOCK_SIZE, 1);
	unsigned int numBlocks = size / threads.x + (size % threads.x > 0 ? 1 : 0);
	unsigned int minbx = (unsigned int) ceil(sqrt((double)numBlocks));
	unsigned int minby = numBlocks / minbx + (numBlocks % minbx > 0 ? 1 : 0);
	dim3 blocks( minbx, minby );

	// get data to GPU
	T *d_in_data;
	T *d_out_data;
	T *d_out_data2;
	cudaMalloc((void **)&d_in_data, sizeof(T) * size);
	cudaMalloc((void **)&d_out_data, sizeof(T) * size);
	cudaMalloc((void **)&d_out_data2, sizeof(T) * size);
	cudaMemcpy(d_in_data, in_data, sizeof(T) * size, cudaMemcpyHostToDevice);
	cudaMemset(d_out_data, 0, sizeof(T) * size);
	cudaMemset(d_out_data2, 0, sizeof(T) * size);

	cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();

	int *d_block_pos;
	cudaMalloc(&d_block_pos, sizeof(int) * blocks.x * blocks.y);
	cudaMemset(d_block_pos, 0, sizeof(int) * blocks.x * blocks.y);
	unsigned int *d_queue_size;

	cudaMalloc(&d_queue_size, sizeof(unsigned int));
	cudaMemset(d_queue_size, 0, sizeof(unsigned int));

	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();


	printf("ordered blocks: %d, %d, threads: %d, size %d \n", blocks.x, blocks.y, threads.x, size);

	//	::nscale::gpu::gapped_select<<<blocks, threads, 0, stream >>>(d_in_data, size, d_out_data, d_block_pos);
	//	::nscale::gpu::compact <<<blocks, threads, 0, stream >>>(d_out_data, size, d_block_pos, d_out_data2);
	gapped_select<<<blocks, threads >>>(d_in_data, size, d_out_data, d_block_pos);
	compact <<<blocks, threads >>>(d_out_data, size, d_block_pos, d_out_data2, d_queue_size);



	err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
    }
	cudaThreadSynchronize();

	// get data off gpu
	unsigned int queue_size ;
	cudaMemcpy((void*)&queue_size, (void*)d_queue_size, sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_data, d_out_data2, sizeof(T) * size, cudaMemcpyDeviceToHost);

	err = cudaGetLastError();
	    if (err != cudaSuccess) {
	        printf("ERROR: %s\n", cudaGetErrorString(err));
	    }
	cudaThreadSynchronize();



	cudaFree(d_in_data);
	cudaFree(d_out_data);
	cudaFree(d_out_data2);
	cudaFree(d_block_pos);
	cudaFree(d_queue_size);

	return queue_size;
}

template unsigned int SelectCPUTesting<int>(const int* in_data, const int size, int* out_data);
template unsigned int SelectThrustScanTesting<int>(const int* in_data, const int size, int* out_data, cudaStream_t stream);

template unsigned int SelectWarpScanUnorderedTesting<int>(const int* in_data, const int size, int* out_data, cudaStream_t stream);

template unsigned int SelectWarpScanOrderedTesting<int>(const int* in_data, const int size, int* out_data, cudaStream_t stream);
}}

