// adaptation of Pavel's imreconstruction code for openCV

#include "internal_shared.hpp"
#include "change_kernel.cuh"
#include "opencv2/gpu/device/vecmath.hpp"

#define MAX_THREADS		256
#define YX_THREADS	64
#define YY_THREADS  4
#define X_THREADS			32
#define Y_THREADS			64
#define XX_THREADS	8
#define XY_THREADS	32
#define NEQ(a,b)    ( (a) != (b) )

#define WARP_SIZE 32

using namespace cv::gpu;
using namespace cv::gpu::device;


namespace nscale { namespace gpu {


////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////
/*
 * original code
 */
template <typename T>
__global__ void
iRec1DForward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * blockDim.x;
	
	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
	bool s_change = false;

	
	if (by + ty < sy) {

		int startx, iy, ix;

		T s_old;
		// the increment allows overlap by 1 between iterations to move the data to next block.
		for (startx = 0; startx < sx - Y_THREADS; startx += Y_THREADS - 1) {
			// copy part of marker and mask to shared memory
			for (iy = 0; iy < Y_THREADS; ++iy) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
			}
			__syncthreads();

			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
			for (ix = 1; ix < Y_THREADS; ++ix) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change |= NEQ( s_old, s_marker[ix][ty] );
			}
			__syncthreads();

			// output result back to global memory
			for (iy = 0; iy < Y_THREADS; ++iy) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			}
			__syncthreads();

		}

		startx = sx - Y_THREADS;

		// copy part of marker and mask to shared memory
		for (iy = 0; iy < Y_THREADS; ++iy) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
		}
		__syncthreads();

		// perform iteration
		for (ix = 1; ix < Y_THREADS; ++ix) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
			s_change |= NEQ( s_old, s_marker[ix][ty] );
		}
		__syncthreads();

		// output result back to global memory
		for (iy = 0; iy < Y_THREADS; ++iy) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change) *change = true;
		}
		__syncthreads();

	}

}

template <typename T>
__global__ void
iRec1DBackward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * Y_THREADS;
	// always 0.  const int bz = blockIdx.y;

	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
	bool s_change = false;


	if (by + ty < sy) {

		int startx;

		T s_old;
		for (startx = sx - Y_THREADS; startx > 0; startx -= Y_THREADS - 1) {

			// copy part of marker and mask to shared memory
			for (int iy = 0; iy < Y_THREADS; iy++) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
			}
			__syncthreads();

			// perform iteration
			for (int ix = Y_THREADS - 2; ix >= 0; ix--) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change |= NEQ( s_old, s_marker[ix][ty] );
			}
			__syncthreads();

			// output result back to global memory
			for (int iy = 0; iy < Y_THREADS; iy++) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			}
			__syncthreads();

		}

		startx = 0;

		// copy part of marker and mask to shared memory
		for (int iy = 0; iy < Y_THREADS; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
		}
		__syncthreads();

		// perform iteration
		for (int ix = Y_THREADS - 2; ix >= 0; ix--) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
			s_change |= NEQ( s_old, s_marker[ix][ty] );
		}
		__syncthreads();

		// output result back to global memory
		for (int iy = 0; iy < Y_THREADS; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change) *change = true;
		}
		__syncthreads();


	}

}

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
iRec1DForward_X_dilation ( T* marker, const T* mask, const unsigned int sx, const unsigned int sy, bool* change )
{
	//const int ty = threadIdx.y;
	const unsigned int by = blockIdx.x * XY_THREADS;  // current Y in image coord.
	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int starty = y * WARP_SIZE / XX_THREADS;  // REQUIRE WARP_SIZE >= X_THREADS
//	printf("(tx, ty) -> (x, y) : (%d, %d)->(%d,%d)\n", threadIdx.x, threadIdx.y, x, y);

	// XY_THREADS should be 32==warpSize, XX_THREADS should be 4 or 8.
	// init to 0...
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	volatile unsigned int s_change = 0;
	volatile T s_old, s_new;
	unsigned int startx;
	unsigned int start;



//	if (threadIdx.y + by < sy) {
		s_marker[threadIdx.y][threadIdx.x] = 0;  // only need x=0 to be 0

		// the increment allows overlap by 1 between iterations to move the data to next block.
		for (startx = 0; startx < sx - WARP_SIZE; startx += WARP_SIZE) {
			start = (by + starty) * sx + startx + x;


			// copy part of marker and mask to shared memory.  works for 1 warp at a time...
			for (unsigned int i = 0; i < WARP_SIZE / XX_THREADS; ++i) {
				s_marker[starty+i][x+1] = marker[start + i*sx];
				s_mask  [starty+i][x+1] = mask[start + i*sx];
			}

			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			if (threadIdx.x == 0) {  // have all threads do the same work
				for (unsigned int i = 1; i <= WARP_SIZE; i++) {
					s_old = s_marker[threadIdx.y][i];
					s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
					s_change |= s_new ^ s_old;
					s_marker[threadIdx.y][i] = s_new;
				}
				s_marker[threadIdx.y][0] = s_marker[threadIdx.y][WARP_SIZE];
//			}

			// output result back to global memory and set up for next x chunk
			for (unsigned int i = 0; i < WARP_SIZE / XX_THREADS; ++i) {
				marker[start + i*sx] = s_marker[starty+i][x+1];
			}
//			printf("startx: %d, change = %d\n", startx, s_change);
		
		}

		if (startx < sx) {
			s_marker[threadIdx.y][0] = s_marker[threadIdx.y][sx-startx];  // getting ix-1st entry, which has been offsetted by 1 in s_marker
			// shared mem copy
			startx = sx - WARP_SIZE;
			start = (by + starty) * sx + startx + x;

			// copy part of marker and mask to shared memory.  works for 1 warp at a time...
			for (unsigned int i = 0; i < WARP_SIZE / XX_THREADS; ++i) {
				s_marker[starty+i][x+1] = marker[start + i*sx];
				s_mask  [starty+i][x+1] = mask[start + i*sx];
			}

			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			if (threadIdx.x == 0) {		
				for (unsigned int i = 1; i <= WARP_SIZE; i++) {
					s_old = s_marker[threadIdx.y][i];
					s_new = min( max( s_marker[threadIdx.y][i-1], s_old ), s_mask[threadIdx.y][i] );
					s_change |= s_new ^ s_old;
					s_marker[threadIdx.y][i] = s_new;
				}
//			}
			// output result back to global memory and set up for next x chunk
			for (unsigned int i = 0; i < WARP_SIZE / XX_THREADS; ++i) {
				marker[start + i*sx] = s_marker[starty+i][x+1];
			}
		}


//	}
//	__syncthreads();
	if (s_change > 0) *change = true;
//	__syncthreads();

//	if (threadIdx.x ==0 && threadIdx.y == 0 && s_change) 
//		printf("change = %s\n", *change ? "true" : "false");

}


template <typename T>
__global__ void
iRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * Y_THREADS;
	// always 0.  const int bz = blockIdx.y;
	
	if (by + ty < sy) {

		volatile __shared__ T s_marker[X_THREADS][Y_THREADS+1];
		volatile __shared__ T s_mask  [X_THREADS][Y_THREADS+1];
		volatile __shared__ bool  s_change[X_THREADS][Y_THREADS+1];
		s_change[tx][ty] = false;
		__syncthreads();

		T s_old;
		int startx;
		for (startx = sx - X_THREADS; startx > 0; startx -= X_THREADS - 1) {

			// copy part of marker and mask to shared memory
			s_marker[tx][ty] = marker[(by + ty)*sx + startx + tx];
			s_mask  [tx][ty] = mask  [(by + ty)*sx + startx + tx];
			__syncthreads();

			// perform iteration
			for (int ix = X_THREADS - 2; ix >= 0; ix--) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
				__syncthreads();
			}

			// output result back to global memory
			marker[(by + ty)*sx + startx + tx] = s_marker[tx][ty];
			__syncthreads();

		}

		startx = 0;

		// copy part of marker and mask to shared memory
		s_marker[tx][ty] = marker[(by + ty)*sx + startx + tx];
		s_mask  [tx][ty] = mask  [(by + ty)*sx + startx + tx];
		__syncthreads();

		// perform iteration
		for (int ix = X_THREADS - 2; ix >= 0; ix--) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
			s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
		}
		__syncthreads();

		// output result back to global memory
		marker[(by + ty)*sx + startx + tx] = s_marker[tx][ty];
		__syncthreads();
		
		if (s_change[tx][ty]) *change = true;
		__syncthreads();

	}

}


/*
template <typename T>
__global__ void
iRec1D8ConnectedWindowedMax ( DevMem2D_<T> g_marker_max, DevMem2D_<T> g_marker)
{
	// parallelize along y.
	const int ty = threadIdx.y;
	const int by = blockIdx.y * MAX_THREADS;
	const int sx = g_marker.cols;
	const int sy = g_marker.rows;
	int y = by + ty;
	
	if ( (by + ty) < sy) {
		__shared__ T s_marker[MAX_THREADS][3];
		__shared__ T s_out[MAX_THREADS];
		T temp;
		T* marker = g_marker.ptr(y);
		T* output = g_marker_max.ptr(y);
		s_marker[ty][0] = 0;
		s_marker[ty][1] = *marker;
		__syncthreads();
		
		for (int ix = 0; ix < (sx - 1); ix++) {
			s_marker[ty][2] = marker[ix + 1];
			__syncthreads;
			
			temp = max(s_marker[ty][0], s_marker[ty][1]);
			s_out[ty] = max(temp, s_marker[ty][2]);
						
			s_marker[ty][0] = s_marker[ty][1];
			s_marker[ty][1] = s_marker[ty][2];
			__syncthreads();
			
			output[ix] = s_out[ty];
			__syncthreads();
		}
		
		// do the last one
		s_out[ty] = max(s_marker[ty][0], s_marker[ty][1]);
		__syncthreads();
		
		output[sx-1] = s_out[ty];
		__syncthreads();
	}

} 
*/


template <typename T>
__global__ void
iRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	bool  s_change = false;
	
	if ( (bx + tx) < sx ) {

		s_marker_B[tx] = marker[bx + tx];
		__syncthreads();

		T s_old;
		for (int ty = 1; ty < sy; ty++) {		
			// copy part of marker and mask to shared memory
			s_marker_A[tx] = s_marker_B[tx];
			s_marker_B[tx] = marker[ty * sx + bx + tx];
			s_mask    [tx] = mask[ty * sx + bx + tx];
//			__syncthreads();

			// perform iteration
			s_old = s_marker_B[tx];
			s_marker_B[tx] = max( s_marker_A[tx], s_marker_B[tx] );
			s_marker_B[tx] = min( s_marker_B[tx], s_mask    [tx] );
			s_change |= NEQ( s_old, s_marker_B[tx] );
//			__syncthreads();

			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
//			__syncthreads();

		}
		__syncthreads();
		
		if (s_change) *change = true;
		__syncthreads();

	}

}

template <typename T>
__global__ void
iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	bool  s_change=false;

	if ( (bx + tx) < sx ) {

		s_marker_B[tx] = marker[(sy-1) * sx + bx + tx];
		__syncthreads();

		T s_old;
		for (int ty = sy - 2; ty >= 0; ty--) {

			// copy part of marker and mask to shared memory
			s_marker_A[tx] = s_marker_B[tx];
			s_marker_B[tx] = marker[ty * sx + bx + tx];
			s_mask    [tx] = mask[ty * sx + bx + tx];
//			__syncthreads();

			// perform iteration
			s_old = s_marker_B[tx];
			s_marker_B[tx] = max( s_marker_A[tx], s_marker_B[tx] );
			s_marker_B[tx] = min( s_marker_B[tx], s_mask    [tx] );
			s_change |= NEQ( s_old, s_marker_B[tx] );
//			__syncthreads();

			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
//			__syncthreads();

		}
		__syncthreads();
		
		if (s_change) *change = true;
		__syncthreads();

	}

}

// 8 conn...
//overlap:  tx 0 to 7 maps to -1 to 6, with usable from 0 to 5.  output for 6-11, from 5 - 12
//formula:  bx * (block-2) - 1 + tx = startx in src data.
//formula:  
/*
template <typename T>
__global__ void
iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{


	// parallelize along x.
	const int tx = threadIdx.x;
	const int ty = threadIdx.x;
	const int bx = blockIdx.x * blockDim.x;
	const int by = blockIdx.y * blockDim.y;


//	__shared__ T s_marker_A[MAX_THREADS];
	__shared__ T s_marker_B[YY_THREADS+1][YX_THREADS+2];
	T* t_marker = marker + by * sx + by;
//	__shared__ T s_mask    [MAX_THREADS];
	bool  s_change = false;

	T s_new, s_old, localmax;
//	int offset;

	if ( bx+tx < sx && by + ty < sy) {

//		if (tx < warpSize) {
//			s_marker[tx] = marker[bx + tx];
//			s_marker[-1] = (blockIdx.x == 0) ? 0 : marker[bx-1];
//			s_marker[MAX_THREADS] = (blockIdx.x == gridDim.x - 1) ? 0 : marker[bx + MAX_THREADS];
//		}
//		if ((tx + warpSize < MAX_THREADS) && (bx+tx+warpSize < sx)) {
//			s_marker[tx + warpSize] = marker[bx + tx + warpSize];
//		}
		if (blockIdx.y > 0) {
			s_marker_B[0][tx + 1] = t_marker[(ty-1)*sx + tx];
		}

		s_marker_B[ty+1][tx + 1] = t_marker[ty*sx + tx];
		if (tx < warpSize) {
			s_marker_B[ty+1][0] = (blockIdx.x == 0) ? 0 : t_marker[ty * sx -1];
			s_marker_B[ty+1][YX_THREADS+1] = (blockIdx.x == gridDim.x - 1) ? 0 : t_marker[ty * sx + YX_THREADS];
		}

		__syncthreads();

		for (int iy = 0; iy < YY_THREADS; iy++) {

//			offset = iy*sx;
			// copy part of marker and mask to shared memory

			localmax = max( max(s_marker_B[iy][tx], s_marker_B[iy][tx+1]), s_marker_B[iy][tx+2]);
//			__syncthreads();

			// perform iteration
			s_old = s_marker_B[iy+1][tx+1];
			s_new = min( max( localmax, s_old ),  mask[by * sx + by + iy*sx + tx]);
			s_marker_B[iy+1][tx+1] = s_new;
			s_change |= NEQ( s_old, s_new );
			// output result back to global memory
			__syncthreads();
		}
//		__syncthreads();
		t_marker[ty*sx + tx] = s_marker_B[ty+1][tx + 1];
//		marker[iy * sx + tx] = s_new;

		if (s_change) *change = true;
		__syncthreads();
	}
}
*/

template <typename T>
__global__ void
iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{


	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * blockDim.x;

//	__shared__ T s_marker_A[MAX_THREADS];
	__shared__ T s_marker_B[MAX_THREADS+2];
	T* s_marker = s_marker_B + 1;
//	__shared__ T s_mask    [MAX_THREADS];
	bool  s_change = false;

	T s_new, s_old, localmax;
	int offset;
	s_marker[tx] = 0;

	if ( bx+tx < sx ) {

//		if (tx < warpSize) {
//			s_marker[tx] = marker[bx + tx];
//			s_marker[-1] = (blockIdx.x == 0) ? 0 : marker[bx-1];
//			s_marker[MAX_THREADS] = (blockIdx.x == gridDim.x - 1) ? 0 : marker[bx + MAX_THREADS];
//		}
//		if ((tx + warpSize < MAX_THREADS) && (bx+tx+warpSize < sx)) {
//			s_marker[tx + warpSize] = marker[bx + tx + warpSize];
//		}
		s_marker[tx] = marker[bx + tx];
		if (tx < warpSize) {
			s_marker[-1] = (blockIdx.x == 0) ? 0 : marker[bx -1];
			s_marker[MAX_THREADS] = (blockIdx.x == gridDim.x - 1) ? 0 : marker[bx + MAX_THREADS];
		}

		__syncthreads();

		for (int ty = 1; ty < sy; ty++) {

			offset = ty*sx + bx;
			// copy part of marker and mask to shared memory

			localmax = max( max(s_marker[tx-1], s_marker[tx]), s_marker[tx+1]);
			__syncthreads();

//			if (tx < warpSize) {
//				s_marker[tx] = marker[offset + tx];
//				s_marker[-1] = (blockIdx.x == 0) ? 0 : marker[offset - 1];
//				s_marker[MAX_THREADS] = (blockIdx.x == gridDim.x - 1) ? 0 : marker[offset + MAX_THREADS];
//			}
//			if ((tx + warpSize < MAX_THREADS) && (offset+tx+warpSize < sx)) {
//				s_marker[tx + warpSize] = marker[offset + tx + warpSize];
//			}
			s_marker[tx] = marker[offset + tx];
			if (tx < warpSize) {
				s_marker[-1] = (blockIdx.x == 0) ? 0 : marker[offset -1];
				s_marker[MAX_THREADS] = (blockIdx.x == gridDim.x - 1) ? 0 : marker[offset + MAX_THREADS];
			}

			//__syncthreads();



			// perform iteration
			s_old = s_marker[tx];
			s_new = min( max( localmax, s_old ),  mask[offset + tx]);
			s_change |= NEQ( s_old, s_new );
			// output result back to global memory
			__syncthreads();
			marker[offset + tx] = s_new;
			s_marker[tx] = s_new;

		}

		if (s_change) *change = true;
		__syncthreads();
	}
}


template <typename T>
__global__ void
iRec1DBackward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	bool  s_change=false;

	if ( bx + tx < sx ) {

		
		s_marker_B[tx] = marker[(sy -1) * sx + bx + tx];
		__syncthreads();

		T s_old;
		for (int ty = sy - 2; ty >= 0; ty--) {

			// copy part of marker and mask to shared memory
			s_marker_A[tx] = s_marker_B[tx];
			if (bx + tx > 0) s_marker_A[tx] = max((tx == 0) ? marker[(ty+1) * sx + bx + tx -1] : s_marker_B[tx-1], s_marker_A[tx]);
			if (bx + tx < sx-1) s_marker_A[tx] = max((tx == blockDim.x-1) ? marker[(ty+1) * sx + bx + tx +1] : s_marker_B[tx+1], s_marker_A[tx]);
			s_mask    [tx] = mask[ty * sx + bx + tx];
//			__syncthreads();
			s_old = marker[ty * sx + bx + tx];
//			s_marker_B[tx] = marker[ty * sx + bx + tx];
			__syncthreads();

			// perform iteration
			//s_old = s_marker_B[tx];
			s_marker_B[tx] = max( s_marker_A[tx], s_old );
//			s_marker_B[tx] = max( s_marker_A[tx], s_marker_B[tx] );
			s_marker_B[tx] = min( s_marker_B[tx], s_mask    [tx] );
			s_change |= NEQ( s_old, s_marker_B[tx] );
			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
			__syncthreads();


		}

		if (s_change) *change = true;
		__syncthreads();

	}

}


	// connectivity:  if 8 conn, need to have border.

	template <typename T>
	unsigned int imreconstructIntCaller(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream) {

		// here because we are not using streams inside.
//		if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
//		else cudaSafeCall( cudaStreamSynchronize(stream));


		printf("entering imrecon int caller with conn=%d\n", connectivity);

		// setup execution parameters
		bool conn8 = (connectivity == 8);

		dim3 threadsx( XX_THREADS, XY_THREADS );
		dim3 blocksx( divUp(sy, threadsx.y) );
		dim3 threadsx2( Y_THREADS );
		dim3 blocksx2( divUp(sy, threadsx2.y) );
		dim3 threadsy( MAX_THREADS );
		dim3 blocksy( divUp(sx, threadsy.x) );
		dim3 threadsy2( YX_THREADS, YY_THREADS );
		dim3 blocksy2( divUp(sx, threadsy2.x), divUp(sy, threadsy2.y)  );
		size_t Nsy = (threadsy.x * 3 + 2) * sizeof(uchar4);
		// stability detection
		unsigned int iter = 0;
		bool *h_change, *d_change;
		h_change = (bool*) malloc( sizeof(bool) );
		cudaSafeCall( cudaMalloc( (void**) &d_change, sizeof(bool) ) );
		
		*h_change = true;
		printf("completed setup for imrecon int caller \n");

		if (conn8) {
			while ( (*h_change) && (iter < 100000) )  // repeat until stability
			{
				iter++;
				*h_change = false;
				init_change<<< 1, 1, 0, stream>>>( d_change );

				// dopredny pruchod pres osu X
				iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
//				iRec1DForward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				iRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DBackward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				iRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
				else cudaSafeCall( cudaStreamSynchronize(stream));
//				printf("%d sync \n", iter);

				cudaSafeCall( cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) );
//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

			}
		} else {
			while ( (*h_change) && (iter < 100000) )  // repeat until stability
			{
				iter++;
				*h_change = false;
				init_change<<< 1, 1, 0, stream>>>( d_change );

				// dopredny pruchod pres osu X
//				iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DForward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DBackward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				iRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
				else cudaSafeCall( cudaStreamSynchronize(stream));
//				printf("%d sync \n", iter);

				cudaSafeCall( cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) );
//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

			}
		}

		cudaSafeCall( cudaFree(d_change) );
		free(h_change);

		printf("Number of iterations: %d\n", iter);
		cudaSafeCall( cudaGetLastError());

		return iter;
	}

	template unsigned int imreconstructIntCaller<unsigned char>(unsigned char*, const unsigned char*, const int, const int,
		const int, cudaStream_t );
}}
