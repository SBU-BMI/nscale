// adaptation of Pavel's imreconstruction code for openCV

#include "internal_shared.hpp"
#include "opencv2/gpu/device/vecmath.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/count.h>


#define MAX_THREADS		256
#define XX_THREADS	4
#define XY_THREADS	32
#define NEQ(a,b)    ( (a) != (b) )

#define WARP_SIZE 32

using namespace cv::gpu;
using namespace cv::gpu::device;


namespace nscale { namespace gpu {


// 3 * WARP_SIZE.    third WARP_SIZE is boolean marking the items to copy later.
// does not touch other warps
template<class T> 
__device__ void warp_mark(const T* s_in_data, volatile int* s_mark) {
	s_mark[threadIdx.x] = (s_in_data[threadIdx.x] > 0 ? 1, 0);
} 

// s_mark and s_scan pointers point to the starting pos of current warp's data: s_scan = s_scan + 2*warpId*WARP_SIZE
// idx is the id within the warp.
// first WARP_SIZE is dummy so to avoid warp divergence in warp_scan.
// second WARP_SIZE is the scan area.
// does not touch other warps
template<typename T>
__device__ void init_warp_scan(volatile int* s_mark, volatile int* s_scan, const int idx) {
	s_scan[idx] = 0;  // extra padding for the scan part...
	s_scan[idx + WARP_SIZE] = s_mark[threadIdx.x];
}

// adapted from CudPP.
// does not touch other warps
template<int maxlevel>
__device__ int warp_scan(volatile int* s_scan, const int idx) {
	int t = s_scan[idx];
	if (0 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 1]; }
	if (1 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 2]; }
	if (2 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 4]); }
	if (3 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx - 8]); }
	if (4 <= maxlevel) { s_scan[idx] = t = t + s_scan[idx -16]); }
	return s_scan[WARP - 1]; // return the total
} 

// s_out_data points to the beginning of the shared array.
// s_scan points to the scanned position for this warp
// s_scan should be exclusive
// return total selected for the warp
// touches other warps, but can rely on warp execution ordering.
template<class T> 
__device__ void warp_select(const T* s_in_data, const int* s_mark, const int* s_scan, volatile T* s_out_data, const int offset, const int idx) {
	if (s_mark[threadIdx.x] > 0) { s_out_data[s_scan[idx] + offset] = s_in_data[threadIdx.x]; }
} 

// unordered
template<class T>
__global__ void unordered_select(const T* in_data, const int dataSize, volatile T* out_data, volatile int* queue_size) {
	// initialize the variables
	const int idx = threadIdx.x & (WARP_SIZE - 1);
	const int warpId = threadIdx.x >> 5;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;

	__shared__ volatile int offsets[(BLOCK_SIZE >> 5) + 1];  // avoid divergence - everyone write...
	__shared__ volatile int s_mark[BLOCK_SIZE];
	__shared__ volatile int s_scan[NUM_WARPS][WARP_SIZE * 2 + 1];
	__shared__ volatile T s_in_data[BLOCK_SIZE];
	__shared__ volatile T s_out_data[BLOCK_SIZE];
	int curr_pos;
	int curr_len;

	// copy in data
	if (warpId == 0) offsets[0] = 0;
	offsets[warpId + 1] = 0;
	s_out_data[threadIdx.x] = 0;
	s_in_data[threadIdx.x] = 0;
	if (x < dataSize) s_in_data[threadIdx.x] = in_data[x];
	__syncthreads();

	// compact within this block
	warp_mark(s_in_data, s_mark);  // mark the data to be processed
	init_warp_scan(s_mark, s_scan[warpId], idx)
	curr_len = warp_scan<5>(s_scan[warpId] + WARP_SIZE, idx);  // perform the in warp scan.
	// now update the totals.  note that warpId+1 would only be updated after warpId is updated, because of warp execution order.
	// note the use of curr_len.  want to avoid reading offsets[warpId] before warp scan is executed for that warpId.
	offsets[warpId + 1] = offsets[warpId] + curr_len;
	warp_select(s_in_data, s_mark, s_scan[warpId] + WARP_SIZE, s_out_data, offsets[warpId], idx);  // compact the data into the block space.
	__syncthreads();

	//copy the data back out.  this block will get a place to write using atomic add.  resulting queue has the blocks shuffled
	// this part is multiblock?
	int block_len = offsets[(BLOCK_SIZE >> 5)];
	if (block_len > 0) {
		if (threadIdx.x == 0) curr_pos = atomicAdd(queue_size, block_len); // only done by first thread in the block
		if (threadIdx.x < block_len) out_data[curr_pos + threadIdx.x] = s_out_data[threadIdx.x];   // dont need to worry about dataSize.  queue size is smaller...
	}
}
template<class T>
__global__ void clear(volatile T* out_data, const int dataSize) {
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	if (x < dataSize) out_data[x] = 0;
}

// gapped.  so need to have anothr step to remove the gaps...  block_pos stores the lengths of the blcok queue for each block
template<class T>
__global__ void gapped_select(const T* in_data, const int dataSize, volatile T* out_data, volatile int* block_pos) {
	// initialize the variables
	const int idx = threadIdx.x & (WARP_SIZE - 1);
	const int warpId = threadIdx.x >> 5;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;

	__shared__ volatile int offsets[(BLOCK_SIZE >> 5) + 1];  // avoid divergence - everyone write...
	__shared__ volatile int s_mark[BLOCK_SIZE];
	__shared__ volatile int s_scan[NUM_WARPS][WARP_SIZE * 2 + 1];
	__shared__ volatile T s_in_data[BLOCK_SIZE];
	__shared__ volatile T s_out_data[BLOCK_SIZE];
	int curr_pos;
	int curr_len;

	// copy in data
	if (warpId == 0) offsets[0] = 0;
	offsets[warpId + 1] = 0;
	s_out_data[threadIdx.x] = 0;
	s_in_data[threadIdx.x] = 0;
	if (x < dataSize) s_in_data[threadIdx.x] = in_data[x];
	__syncthreads();

	// compact within this block
	warp_mark(s_in_data, s_mark);  // mark the data to be processed
	init_warp_scan(s_mark, s_scan[warpId], idx)
	curr_len = warp_scan<5>(s_scan[warpId] + WARP_SIZE, idx);  // perform the in warp scan.
	// now update the totals.  note that warpId+1 would only be updated after warpId is updated, because of warp execution order.
	// note the use of curr_len.  want to avoid reading offsets[warpId] before warp scan is executed for that warpId.
	offsets[warpId + 1] = offsets[warpId] + curr_len;
	warp_select(s_in_data, s_mark, s_scan[warpId] + WARP_SIZE, s_out_data, offsets[warpId], idx);  // compact the data into the block space.
	__syncthreads();

	//copy the data back out.  leaving the space between blocks.
	if (x < dataSize) out_data[x] = s_out_data[threadIdx.x];
	block_pos[blockIdx.x + 1] = offsets[(BLOCK_SIZE >> 5)];
}

// fermi can have maximum of 65K blocks in one dim.
//1024 threads - warpscan all, then 1 warp to scan, then everyone add.
// s_totals has size of 32.
__device__ void scan1024(const int* in_data, const int dataSize, volatile int* out_data, volatile int* s_scan, volatile int* s_scan2, volatile int* block_total) {
	const int idx = threadIdx.x & (WARP_SIZE - 1);
	const int warpId = threadIdx.x >> 5;
	const int x = threadIdx.x + blockDim.x * blockIdx.x;

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
		block_total[blockIdx.x] = warp_scan<5>(s_scan2 + WARP_SIZE, idx);  // inclusive scan
	__synthreads();

	// now add back to the warps
	if (x < dataSize) out_data[x] = s_scan[warpId][idx+WARP_SIZE] + s_scan2[warpId + WARP_SIZE - 1];
}

// to scan a large amount of data, do it in multilevel way....  allocate the summary array, scan the blocks with a kernel call, then scan the summary array.  recurse. then add the results to previous level summary

// step 2 of the compacting.  assumes that within each block the values have already been compacted.
// also block_pos is already scanned to produce final starting positions for each threadblock.
template<class T>
__global__ void compact(const T* in_data, const int* block_pos, volatile T* out_data ) {
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int pos = block_pos[blockIdx.x];
	const int len = block_pos[blockIdx.x + 1] - pos;

	if (threadIdx.x < len) out_data[pos + threadIdx.x] = in_data[x];
}
/*
template<typename T, typename TN>
struct InitialImageToQueue : public thrust::unary_function<TN, int>
{
    __host__ __device__
        int operator()(const TN& pixel) const
        {
		T center = thrust::get<1>(pixel);
		T curr;
		int id = thrust::get<0>(pixel);
		curr = thrust::get<2>(pixel);
		if (curr < center && curr < thrust::get<6>(pixel)) return id;
		curr = thrust::get<3>(pixel);
		if (curr < center && curr < thrust::get<7>(pixel)) return id;
		curr = thrust::get<4>(pixel);
		if (curr < center && curr < thrust::get<8>(pixel)) return id;
		curr = thrust::get<5>(pixel);
		if (curr < center && curr < thrust::get<9>(pixel)) return id;
		return -1;
        }
};
*/





// connectivity:  need to have border of 0 ,and should be continuous
template <typename T>
unsigned int SelectTesting(const T* in_data, volatile T* out_data, cudaStream_t stream) {

	dim3 threadsx( XX_THREADS, XY_THREADS );
	dim3 blocksx( divUp(sy, threadsx.y) );
	dim3 threadsy( MAX_THREADS );
	dim3 blocksy( divUp(sx, threadsy.x) );

	// stability detection



	typedef typename thrust::device_ptr<T> PixelIterator;

//		typedef typename thrust::tuple<int, T, T, T, T, T> PixelNeighborhood;
//		typedef typename thrust::tuple<thrust::counting_iterator<int>, PixelIterator, PixelIterator, PixelIterator, PixelIterator, PixelIterator> WindowedImage;
//		typedef typename thrust::zip_iterator<WindowedImage> WindowedPixelIterator;

	typedef typename thrust::tuple<signed int, T, T, T, T, T, T, T, T, T> ReconNeighborhood;
	typedef typename thrust::tuple<signed int, T, T, T> ReconNeighborhood2;
	typedef typename thrust::tuple<thrust::counting_iterator<int>, PixelIterator, PixelIterator, PixelIterator, PixelIterator, PixelIterator, PixelIterator, PixelIterator, PixelIterator, PixelIterator> ReconImage;
	typedef typename thrust::zip_iterator<ReconImage> ReconPixelIterator;

	typedef typename thrust::device_vector<int> Queue;
	typedef typename Queue::iterator QueueIterator;
	typedef typename thrust::tuple<int, int, int, int> QueueElement;


	thrust::counting_iterator<int> ids;
//		WindowedImage markerImg = thrust::make_tuple(ids, q_ym1xm1, q_ym1, q_ym1xp1, q_xm1, q, q_xp1, q_yp1xm1, q_yp1, q_yp1xp1);
//		WindowedImage markerImgEnd = thrust::make_tuple(ids+area, q_ym1xm1+area, q_ym1+area, q_ym1xp1+area, q_xm1+area, q+area, q_xp1+area, q_yp1xm1+area, q_yp1+area, q_yp1xp1+area);
//		WindowedImage maskImg = thrust::make_tuple(ids, p_ym1xm1, p_ym1, p_ym1xp1, p_xm1, p, p_xp1, p_yp1xm1, p_yp1, p_yp1xp1);
//		ReconPixelIterator mask_last = thrust::make_zip_iterator(thrust::make_tuple(p_ym1xm1+area, p_ym1+area, p_ym1xp1+area, p_xm1+area, p+area, p_xp1+area, p_yp1xm1+area, p_yp1+area, p_yp1xp1+area));

	ReconImage markermaskNp = thrust::make_tuple(ids, q, q_xp1, q_yp1xm1, q_yp1, q_yp1xp1, p_xp1, p_yp1xm1, p_yp1, p_yp1xp1);
	ReconImage markermaskNpEnd = thrust::make_tuple(ids+area, q+area, q_xp1+area, q_yp1xm1+area, q_yp1+area, q_yp1xp1+area, p_xp1+area, p_yp1xm1+area, p_yp1+area, p_yp1xp1+area);
	ReconPixelIterator image_first = thrust::make_zip_iterator(markermaskNp);
	ReconPixelIterator image_last = thrust::make_zip_iterator(markermaskNpEnd); 

	// put the candidates into the queue
	int queueSize = area;
	Queue sparseQueue(queueSize, -1);

	// can change into transform_iterator to use in the copy operation.  the only challenge is don't know queue size, and would still need to compact later...
	// mark
	thrust::transform(image_first, image_last, sparseQueue.begin(), InitialImageToQueue<T, ReconNeighborhood>());
	// select
	queueSize = thrust::count_if(sparseQueue.begin(), sparseQueue.end(), GreaterThanConst<int>(-1));

	Queue testQueue(area, -1);

	// compact the queue
	Queue denseQueue(queueSize, 0);
	QueueIterator denseQueue_end = thrust::copy_if(sparseQueue.begin(), sparseQueue.end(), denseQueue.begin(), GreaterThanConst<int>(-1));
	QueueIterator sparseQueue_end;

	thrust::device_vector<bool> dummy(area, false);
	printf("number of entries in sparseQueue: %d, denseQueue: %d \n", queueSize, denseQueue_end - denseQueue.begin());
	int iterations = 0;
	int total = 0;
	while (queueSize > 0 && iterations < 10000) {
		++iterations;
		total += queueSize;

//			printf("here\n");
		// allocate some memory
//			sparseQueue.resize(queueSize * 8);  // 8 neighbors
//			thrust::fill(sparseQueue.begin(), sparseQueue.end(), -1);
		// also set up as 8 devPtrs
//			QueueIterator ym1xm1 = sparseQueue.begin();
//			QueueIterator ym1 = ym1xm1+queueSize;
//			QueueIterator ym1xp1 = ym1+queueSize;
//			QueueIterator xm1 = ym1xp1+queueSize;
//			QueueIterator xp1 = xm1+queueSize;
//			QueueIterator yp1xm1 = xp1+queueSize;
//			QueueIterator yp1 = yp1xm1+queueSize;
//			QueueIterator yp1xp1 = yp1+queueSize;
//						printf("here3\n");
//			dummy.resize(queueSize);

		// sort the queue by the value
		sparseQueue_end = thrust::copy(denseQueue.begin(), denseQueue.end(), sparseQueue.begin());
		thrust::stable_sort_by_key(thrust::make_permutation_iterator(q, sparseQueue.begin()),
				thrust::make_permutation_iterator(q, sparseQueue_end),
				denseQueue.begin());

		thrust::fill(dummy.begin(), dummy.end(), false);
		thrust::for_each(denseQueue.begin(), denseQueue.end(), Propagate<T>(thrust::raw_pointer_cast(q),
				thrust::raw_pointer_cast(p), thrust::raw_pointer_cast(&*dummy.begin()), sx));


		// and prepare the queue for the next iterations.
			//sparseQueue_end = thrust::unique(sparseQueue.begin(), sparseQueue.end());
			queueSize = thrust::count_if(dummy.begin(), dummy.end(), thrust::identity<bool>());
//			printf("here 7 : queueSize =%d \n", queueSize);

		denseQueue.resize(queueSize);
		thrust::fill(denseQueue.begin(), denseQueue.end(), -1);

		denseQueue_end = thrust::copy_if(ids, ids+area, dummy.begin(), denseQueue.begin(), thrust::identity<bool>());
		printf("number of entries in queue: %d \n", denseQueue_end - denseQueue.begin());

	}


	if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
	else cudaSafeCall( cudaStreamSynchronize(stream));
	cudaSafeCall( cudaGetLastError());

	printf("iterations: %d, total: %d\n", iterations, total);
	return total;

}

template unsigned int imreconQueueIntCaller<unsigned char>(unsigned char*, unsigned char*, const int, const int,
	const int, cudaStream_t );
}}
