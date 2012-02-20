// adaptation of Pavel's imreconstruction code for openCV

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

//using namespace cv::gpu;
//using namespace cv::gpu::device;


namespace nscale { namespace gpu {


////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////
/*
 * warp = 32. shared memory in banks of 32, each 32 bits (128 bytes wide) - interleave of 4 for rows? no need.  compute 2 has no conflict for read/write bytes.
 * global memory in partitions of 256 bytes.  1 warp at a time at 1, 2, 4, 8, or 16 bytes. width of array and threadblock = warpsize * c,
 * try to remove syncthreads by making sure warps do not diverge(and use volatile)
 * thread id = x + y * Dx.  so this means if x and y are swapped between mem and compute steps, must have sync...
 * IF 32 x 8 theads, repeat 4 times in y.  read single char from global, then swap x and y to process 32 y at a time, would need to syncthread inside iterations.  can use 1 warp to go through all shared mem iteratively, or have each warp compute 4 bytes 4 columns (warps are ordered)
 * IF 8x4 or 4x8 threads  for a warp, read 1 bytes from global (linearize the warp thread id (e.g. x + y*8 or x+y*4) to read from global sequentially, and repeat 4 or 8 times) then process the memory for this warp 4 y or 8 y iteratively, repeat for all x chunks.  essentially the original algorithm.  then create threadblock that is just multiplied in y to reach 192 or 256.  avoids syncthreads completely. 
 * or alternatively, treat each warp as 4x8, and each x process columns 8 apart.  each warp then do 4 bytes, (8 warps), to generate 8x8 blocks that are completed. - no synthreads needed. - no... would require more kernel iterations


for backward:  thread ids should map to the data - so first thread has the last data....  ( for correctness)
for y, similar to this...

for register usage: use unsigned int where possible.  maybe use 1D shared array would be better too...
 */
template <typename T>
__global__ void
iRec1DForward_X_dilation ( T* marker, const T* mask, const unsigned int sx, const unsigned int sy)
{
	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;
//	printf("(tx, ty) -> (x, y) : (%d, %d)->(%d,%d)\n", threadIdx.x, threadIdx.y, x, y);

	// XY_THREADS should be 32==warpSize, XX_THREADS should be 4 or 8.
	// init to 0...
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	T s_old, s_new;
	unsigned int startx;
	unsigned int start;



	s_marker[threadIdx.y][WARP_SIZE] = 0;  // only need x=0 to be 0

	// the increment allows overlap by 1 between iterations to move the data to next block.
	for (startx = 0; startx < xstop; startx += WARP_SIZE) {
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		s_marker[threadIdx.y][0] = s_marker[threadIdx.y][WARP_SIZE];

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x+1] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x+1] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			if (threadIdx.x == 0) {  // have all threads do the same work
//#pragma unroll
if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
		for (unsigned int i = 1; i <= WARP_SIZE; ++i) {
			s_old = s_marker[threadIdx.y][i];
			s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
			s_marker[threadIdx.y][i] = s_new;
		}
}
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
//			printf("startx: %d, change = %d\n", startx, s_change);

	}

	if (startx < sx) {
		s_marker[threadIdx.y][0] = s_marker[threadIdx.y][sx-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
		// shared mem copy
		startx = sx - WARP_SIZE;
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x+1] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x+1] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//#pragma unroll
if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
		for (unsigned int i = 1; i <= WARP_SIZE; ++i) {
			s_old = s_marker[threadIdx.y][i];
			s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
			s_marker[threadIdx.y][i] = s_new;
		}
}
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
	}
}


template <typename T>
__global__ void
iRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy)
{

	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;
	//	printf("(tx, ty) -> (x, y) : (%d, %d)->(%d,%d)\n", threadIdx.x, threadIdx.y, x, y);

	// XY_THREADS should be 32==warpSize, XX_THREADS should be 4 or 8.
	// init to 0...
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	T s_old, s_new;
	int startx;
	unsigned int start;
	
	s_marker[threadIdx.y][0] = 0;  // only need x=WARPSIZE to be 0

	// the increment allows overlap by 1 between iterations to move the data to next block.
	for (startx = xstop; startx > 0; startx -= WARP_SIZE) {
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		s_marker[threadIdx.y][WARP_SIZE] = s_marker[threadIdx.y][0];

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			if (threadIdx.x == 0) {  // have all threads do the same work
//#pragma unroll
if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
		for (int i = WARP_SIZE - 1; i >= 0; --i) {
			s_old = s_marker[threadIdx.y][i];
			s_new = min( max( s_marker[threadIdx.y][i+1], s_old ), s_mask[threadIdx.y][i] );
			s_marker[threadIdx.y][i] = s_new;
		}
}
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
//			printf("startx: %d, change = %d\n", startx, s_change);
	}

	if (startx <= 0) {
		s_marker[threadIdx.y][WARP_SIZE] = s_marker[threadIdx.y][-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
		// shared mem copy
		startx = 0;
		start = (blockIdx.x * XY_THREADS + y * ychunk) * sx + startx + x;
//			printf("tx: %d, ty: %d, x: %d, y: %d, startx: %d, start: %d", threadIdx.x, threadIdx.y, x, y, startx, start);

		// copy part of marker and mask to shared memory.  works for 1 warp at a time...
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			s_marker[y * ychunk+i][x] = marker[start + i*sx];
			s_mask  [y * ychunk+i][x] = mask[start + i*sx];
		}

		// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
		// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//#pragma unroll
if (threadIdx.y + blockIdx.x * XY_THREADS < sy) {   //require dimension to be perfectly padded.
		for (int i = WARP_SIZE - 1; i >= 0; --i) {
			s_old = s_marker[threadIdx.y][i];
			s_new = min( max( s_marker[threadIdx.y][i+1], s_old ), s_mask[threadIdx.y][i] );
			s_marker[threadIdx.y][i] = s_new;
		}
}
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
	}

}



template <typename T>
__global__ void
iRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy)
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	T s_old, s_new, s_prev;
	
if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = 0; iy < sy; ++iy) {
			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;

		}
}
		


}

template <typename T>
__global__ void
iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	T s_old, s_new, s_prev;

if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = sy - 1; iy >= 0; --iy) {

			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;
		}
}
		

}

template <typename T>
__global__ void
iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy)
{

	// best thing to do is to use linear arrays.  each warp does a column of 32.

	// parallelize along x.
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_B[MAX_THREADS+2];
	volatile T* s_marker = s_marker_B + 1;

	T s_new, s_old, s_prev;

if ( bx + tx < sx ) { // make sure number of threads is a divisor of sx.

	s_prev = 0;

	for (int iy = 0; iy < sy; ++iy) {
		// copy part of marker and mask to shared memory
		if (tx == 0) {
			s_marker_B[0] = (bx == 0) ? 0 : marker[iy*sx + bx - 1];
			s_marker[MAX_THREADS] = (bx + MAX_THREADS >= sx) ? 0 : marker[iy*sx + bx + MAX_THREADS];
		}
		if (tx < WARP_SIZE) {
			// first warp, get extra stuff
			s_marker[tx] = marker[iy*sx + bx + tx];
		}
		if (tx < MAX_THREADS - WARP_SIZE) {
			s_marker[tx + WARP_SIZE] = marker[iy*sx + bx + tx + WARP_SIZE];
		}
		__syncthreads();

		// perform iteration
		s_old = s_marker[tx];
		s_new = min( max( s_prev, s_old ),  mask[iy*sx + bx + tx]);

		// output result back to global memory
		s_marker[tx] = s_new;
		marker[iy*sx + bx + tx] = s_new;
		__syncthreads();

		s_prev = max( max(s_marker[tx-1], s_marker[tx]), s_marker[tx+1]);
	}
}
}


template <typename T>
__global__ void
iRec1DBackward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy)
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_B[MAX_THREADS+2];
	volatile T* s_marker = s_marker_B + 1;

	T s_new, s_old, s_prev;

	if ( bx + tx < sx ) {  //make sure number of threads is a divisor of sx.

		s_prev = 0;

		for (int iy = sy - 1; iy >= 0; --iy) {

			if (tx == 0) {
				s_marker_B[0] = (bx == 0) ? 0 : marker[iy*sx + bx - 1];
				s_marker[MAX_THREADS] = (bx + MAX_THREADS >= sx) ? 0 : marker[iy*sx + bx + MAX_THREADS];
			}
			if (tx < WARP_SIZE) {
				// first warp, get extra stuff
				s_marker[tx] = marker[iy*sx + bx + tx];
			}
			if (tx < MAX_THREADS - WARP_SIZE) {
				s_marker[tx + WARP_SIZE] = marker[iy*sx + bx + tx + WARP_SIZE];
			}
			__syncthreads();


			// perform iteration
			s_old = s_marker[tx];
			s_new = min( max( s_prev, s_old ),  mask[iy*sx + bx + tx]);

			// output result back to global memory
			s_marker[tx] = s_new;
			marker[iy*sx + bx + tx] = s_new;
			__syncthreads();

			s_prev = max( max(s_marker[tx-1], s_marker[tx]), s_marker[tx+1]);

		}
	}
}





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

// this works
//template<typename T, typename TN>
//struct ReconPixel : public thrust::unary_function<TN, int>
//{
//    __host__ __device__
//        int operator()(const TN& pixel)
//        {
//    	thrust::minimum<T> mn;
//		T center = thrust::get<1>(pixel);
//		int id = thrust::get<0>(pixel);
//		T q = thrust::get<2>(pixel);
//		T p = thrust::get<6>(pixel);
//		if (q < center && p != q) {
//			q = mn(center, p);
//			return id - 4098;
//		}
//		return -1;
//        }
//};


// this works too.
//template<typename T, typename TN>
//struct ReconPixel : public thrust::unary_function<TN, T>
//{
//    __host__ __device__
//        T operator()(const TN& pixel)
//        {
//			thrust::minimum<T> mn;
//			int idx1 = -1;
//			T center = thrust::get<1>(pixel);
//			int id = thrust::get<0>(pixel);
//
//			T q = thrust::get<2>(pixel);
//			T p = thrust::get<3>(pixel);
//			if (q < center && p != q) {
//				q = mn(center, p);
//				idx1 = id - 4098;
//			}
//			return q;
//        }
//};

// this works too
//template<typename T, typename TN, typename TN2>
//struct ReconPixel : public thrust::unary_function<TN, T>
//{
//    __host__ __device__
//        T operator()(TN pixel)
//        {
//			thrust::minimum<T> mn;
//			int idx1 = -1;
//			T center = thrust::get<1>(pixel);
//			int id = thrust::get<0>(pixel);
//
//			T q = thrust::get<2>(pixel);
//			T p = thrust::get<3>(pixel);
//			if (q < center && p != q) {
//				q = mn(center, p);
//				idx1 = id - 4098;
//			}
//			thrust::get<2>(pixel) = q;
////			TN2 test= thrust::make_tuple(idx1, q);
//			return q;
//        }
//};

// DOES NOT WORK
//template<typename T, typename TN, typename TN2>
//struct ReconPixel : public thrust::unary_function<TN, TN2>
//{
//    __host__ __device__
//        TN2 operator()(TN pixel)
//        {
//			thrust::minimum<T> mn;
//			int idx1 = -1;
//			T center = thrust::get<1>(pixel);
//			int id = thrust::get<0>(pixel);
//
//			T q = thrust::get<2>(pixel);
//			T p = thrust::get<3>(pixel);
//			if (q < center && p != q) {
//				q = mn(center, p);
//				idx1 = id - 4098;
//			}
////			thrust::get<2>(pixel) = q;
////			TN2 test= thrust::make_tuple(idx1, q);
//			return TN2(idx1, q);
//        }
//};


// DOES NOT UPDATE INPUT
//template<typename T, typename TN, typename TO>
//struct ReconPixel : public thrust::binary_function<TN, TO, bool>
//{
//	int step1, step2, step3, step4;
//
//	__host__ __device__
//	ReconPixel(int _s1, int _s2, int _s3, int _s4) : step1(_s1), step2(_s2), step3(_s3), step4(_s4) {}
//
//    __host__ __device__
//        bool operator()(TN pixel, TO queue)
//        {
//			thrust::minimum<T> mn;
//			int id = thrust::get<0>(pixel);
//			T center = thrust::get<1>(pixel);
//			T p, q;
//			int nextId;
//			bool result = false;
//
//			q = thrust::get<2>(pixel);
//			p = thrust::get<6>(pixel);
//			nextId = -1;
//			if (q < center && q != p) {
//				thrust::get<2>(pixel) = mn(center, p);
//				nextId = id + step1;
//				result = true;
//			}
//			thrust::get<0>(queue) = nextId;
//
//			q = thrust::get<3>(pixel);
//			p = thrust::get<7>(pixel);
//			nextId = -2;
//			if (q < center && q != p) {
//				thrust::get<3>(pixel) = mn(center, p);
//				nextId = id + step2;
//				result = true;
//			}
//			thrust::get<1>(queue) = nextId;
//
//			q = thrust::get<4>(pixel);
//			p = thrust::get<8>(pixel);
//			nextId = -3;
//			if (q < center && q != p) {
//				thrust::get<4>(pixel) = mn(center, p);
//				nextId = id + step3;
//				result = true;
//			}
//			thrust::get<2>(queue) = nextId;
//
//			q = thrust::get<5>(pixel);
//			p = thrust::get<9>(pixel);
//			nextId = -4;
//			if (q < center && q != p) {
//				thrust::get<5>(pixel) = mn(center, p);
//				nextId = id + step4;
//				result = true;
//			}
//			thrust::get<3>(queue) = nextId;
//
//
//			return result;
//        }
//};
//
template<typename T>
struct Propagate
{
	volatile T *marker;
	volatile T *mask;
	bool *flag;
	const int step;

	__host__ __device__
	Propagate(T* _marker, T* _mask, bool* _flag, int _step) : marker(_marker), mask(_mask), flag(_flag), step(_step) {}

	__host__ __device__
		void updateNeighbor(int nId, T center, thrust::minimum<T> mn) {
			T q = marker[nId];
			T p = mask[nId];
			if (q != p && q < center) {
				marker[nId] = mn(center, p);
//				flag[nId] = true;
			}
	}
	__host__ __device__
		void updateAndMarkNeighbor(int nId, T center, thrust::minimum<T> mn) {
			T q = marker[nId];
			T p = mask[nId];
			if (q != p && q < center) {
				marker[nId] = mn(center, p);
				flag[nId] = true;
			}
	}

    __host__ __device__
        void operator()(int id)
        {
			thrust::minimum<T> mn;
			T center = marker[id];
			int nId;

			nId = id - 1;  updateNeighbor(nId, center, mn);
			nId = id + 1;  updateNeighbor(nId, center, mn);
			nId = id - step - 1;  updateNeighbor(nId, center, mn);
			nId = id - step;  updateNeighbor(nId, center, mn);
			nId = id - step + 1;  updateNeighbor(nId, center, mn);
			nId = id + step - 1;  updateNeighbor(nId, center, mn);
			nId = id + step;  updateNeighbor(nId, center, mn);
			nId = id + step + 1;  updateNeighbor(nId, center, mn);

			nId = id - 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - 2 * step - 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - 2 * step - 1;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - 2 * step;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - 2 * step + 1;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - 2 * step + 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - step - 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id - step + 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + step - 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + step + 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + 2 * step - 2;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + 2 * step - 1;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + 2 * step;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + 2 * step + 1;  updateAndMarkNeighbor(nId, center, mn);
			nId = id + 2 * step + 2;  updateAndMarkNeighbor(nId, center, mn);

        }
};


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
	unsigned int imreconQueueIntCaller(T* __restrict__ marker, T* __restrict__ mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream) {

//		printf("entering imrecon int caller with conn=%d\n", connectivity);

		// setup execution parameters

		dim3 threadsx( XX_THREADS, XY_THREADS );
		dim3 blocksx( (sy + threadsx.y - 1) / threadsx.y );
		dim3 threadsy( MAX_THREADS );
		dim3 blocksy( (sx + threadsy.x - 1) / threadsy.x );

		// stability detection


		// dopredny pruchod pres osu X
		iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy);

		// dopredny pruchod pres osu Y
		if (connectivity == 4) {
			// dopredny pruchod pres osu Y
			iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy);
		} else {
			iRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy);
		}

		// zpetny pruchod pres osu X
		iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy);
 
		// dopredny pruchod pres osu Y
		if (connectivity == 4) {
			// dopredny pruchod pres osu Y
			iRec1DBackward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy);
		} else {
			// zpetny pruchod pres osu Y
			iRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy);
		}

		if (stream == 0) cudaDeviceSynchronize();
		else  cudaStreamSynchronize(stream);
//				printf("%d sync \n", iter);


		// set up some types to work with
		thrust::device_ptr<T> p(mask + sx + 1);
		thrust::device_ptr<T> p_ym1(mask + 1);
		thrust::device_ptr<T> p_yp1(mask + 2 * sx + 1);
		thrust::device_ptr<T> p_xm1(mask + sx);
		thrust::device_ptr<T> p_xp1(mask + sx + 2);
		// conn = 8
		thrust::device_ptr<T> p_ym1xm1(mask);
		thrust::device_ptr<T> p_ym1xp1(mask + 2);
		thrust::device_ptr<T> p_yp1xm1(mask + 2 * sx);
		thrust::device_ptr<T> p_yp1xp1(mask + 2 * sx + 2);

		thrust::device_ptr<T> q(marker + sx + 1);
		thrust::device_ptr<T> q_ym1(marker + 1);
		thrust::device_ptr<T> q_yp1(marker + 2 * sx + 1);
		thrust::device_ptr<T> q_xm1(marker + sx);
		thrust::device_ptr<T> q_xp1(marker + sx + 2);
		// conn = 8
		thrust::device_ptr<T> q_ym1xm1(marker);
		thrust::device_ptr<T> q_ym1xp1(marker + 2);
		thrust::device_ptr<T> q_yp1xm1(marker + 2 * sx);
		thrust::device_ptr<T> q_yp1xp1(marker + 2 * sx + 2);
		int area = sx * (sy - 4) - 4;  // actual image area - sx and sy are padded by 1 on each side,


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
		printf("number of entries in sparseQueue: %d, denseQueue: %lu \n", queueSize, denseQueue_end - denseQueue.begin());
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

// does not work...
//				thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
//						thrust::make_permutation_iterator(ids, denseQueue.begin()),
//						thrust::make_permutation_iterator(q, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_ym1, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_xm1, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_xp1, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_yp1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_ym1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_xm1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_xp1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_yp1, denseQueue.begin()))),
//					thrust::make_zip_iterator(thrust::make_tuple(
//						thrust::make_permutation_iterator(ids, denseQueue.end()),
//						thrust::make_permutation_iterator(q, denseQueue.end()),
//						thrust::make_permutation_iterator(q_ym1, denseQueue.end()),
//						thrust::make_permutation_iterator(q_xm1, denseQueue.end()),
//						thrust::make_permutation_iterator(q_xp1, denseQueue.end()),
//						thrust::make_permutation_iterator(q_yp1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_ym1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_xm1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_xp1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_yp1, denseQueue.end()))),
//					thrust::make_zip_iterator(thrust::make_tuple(ym1, xm1, xp1, yp1)),
//					dummy.begin(),
//					ReconPixel<T, ReconNeighborhood, QueueElement>(-sx, (int)-1, (int)1, sx));

//				thrust::fill(testQueue.begin(), testQueue.end(), -1);
//				thrust::transform(image_first, image_last, testQueue.begin(), InitialImageToQueue<T, ReconNeighborhood>());
//				printf("test queue size : %d \n", thrust::count_if(testQueue.begin(), testQueue.end(), GreaterThanConst<int>(-1)));

// does not work...
//				// 8conn
//				thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
//						thrust::make_permutation_iterator(ids, denseQueue.begin()),
//						thrust::make_permutation_iterator(q, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_ym1xm1, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_ym1xp1, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_yp1xm1, denseQueue.begin()),
//						thrust::make_permutation_iterator(q_yp1xp1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_ym1xm1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_ym1xp1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_yp1xm1, denseQueue.begin()),
//						thrust::make_permutation_iterator(p_yp1xp1, denseQueue.begin()))),
//					thrust::make_zip_iterator(thrust::make_tuple(
//						thrust::make_permutation_iterator(ids, denseQueue.end()),
//						thrust::make_permutation_iterator(q, denseQueue.end()),
//						thrust::make_permutation_iterator(q_ym1xm1, denseQueue.end()),
//						thrust::make_permutation_iterator(q_ym1xp1, denseQueue.end()),
//						thrust::make_permutation_iterator(q_yp1xm1, denseQueue.end()),
//						thrust::make_permutation_iterator(q_yp1xp1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_ym1xm1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_ym1xp1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_yp1xm1, denseQueue.end()),
//						thrust::make_permutation_iterator(p_yp1xp1, denseQueue.end()))),
//					thrust::make_zip_iterator(thrust::make_tuple(ym1xm1, ym1xp1, yp1xm1, yp1xp1)),
//					dummy.begin(),
//					ReconPixel<T, ReconNeighborhood, QueueElement>(-sx-1, -sx+1, sx-1, sx+1)); //

//				thrust::fill(testQueue.begin(), testQueue.end(), -1);
//				thrust::transform(image_first, image_last, testQueue.begin(), InitialImageToQueue<T, ReconNeighborhood>());
//				queueSize = thrust::count_if(testQueue.begin(), testQueue.end(), GreaterThanConst<int>(-1));
//				printf("test queue size : %d \n", queueSize);


			// and prepare the queue for the next iterations.
				//sparseQueue_end = thrust::unique(sparseQueue.begin(), sparseQueue.end());
				queueSize = thrust::count_if(dummy.begin(), dummy.end(), thrust::identity<bool>());
//			printf("here 7 : queueSize =%d \n", queueSize);

			denseQueue.resize(queueSize);
			thrust::fill(denseQueue.begin(), denseQueue.end(), -1);

			denseQueue_end = thrust::copy_if(ids, ids+area, dummy.begin(), denseQueue.begin(), thrust::identity<bool>());
			printf("number of entries in queue: %lu \n", denseQueue_end - denseQueue.begin());

		}


		if (stream == 0) cudaDeviceSynchronize();
		else cudaStreamSynchronize(stream);
		 cudaGetLastError();

		printf("iterations: %d, total: %d\n", iterations, total);
		return total;

	}

	template unsigned int imreconQueueIntCaller<unsigned char>(unsigned char*, unsigned char*, const int, const int,
		const int, cudaStream_t );
}}
