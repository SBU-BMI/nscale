// adaptation of Pavel's imreconstruction code for openCV

#include "change_kernel.cuh"
#include <sys/time.h>
#include <stdio.h>


#define MAX_THREADS		256
#define YX_THREADS		64
#define YY_THREADS  		4
#define X_THREADS		32
#define Y_THREADS		64
#define XX_THREADS		4
#define XY_THREADS		64
#define NEQ(a,b)    ( (a) != (b) )

#define WARP_SIZE 32


long ClockGetTime()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
 //   timespec ts;
//    clock_gettime(CLOCK_REALTIME, &ts);
	return (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL;
//    return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
}




namespace nscale { namespace gpu {


////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////
/*
 * fast code
 */
//template <typename T>
//__global__ void
//iRec1DForward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
//{
//
//	const int ty = threadIdx.x;
//	const int by = blockIdx.x * blockDim.x;
//
//	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
//	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
//	bool s_change = false;
//
//
//
//		int startx, iy, ix;
//
//		T s_old;
//		// the increment allows overlap by 1 between iterations to move the data to next block.
//		for (startx = 0; startx < sx - Y_THREADS; startx += Y_THREADS - 1) {
//			// copy part of marker and mask to shared memory
//			for (iy = 0; iy < Y_THREADS && by+iy<sy; ++iy) {
//				// now treat ty as x, and iy as y, so global mem acccess is closer.
//				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
//				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
//			}
//			__syncthreads();
//
//			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
//			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//	if (by + ty < sy) {
//			for (ix = 1; ix < Y_THREADS; ++ix) {
//				s_old = s_marker[ix][ty];
//				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
//				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
//				s_change |= NEQ( s_old, s_marker[ix][ty] );
//			}
//}			__syncthreads();
//
//			// output result back to global memory
//			for (iy = 0; iy < Y_THREADS && by+iy<sy; ++iy) {
//				// now treat ty as x, and iy as y, so global mem acccess is closer.
//				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
//			}
//			__syncthreads();
//
//		}
//
//		startx = sx - Y_THREADS;
//
//		// copy part of marker and mask to shared memory
//		for (iy = 0; iy < Y_THREADS && by+iy<sy; ++iy) {
//			// now treat ty as x, and iy as y, so global mem acccess is closer.
//			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
//			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
//		}
//		__syncthreads();
//
//		// perform iteration
//	if (by + ty < sy) {
//		for (ix = 1; ix < Y_THREADS; ++ix) {
//			s_old = s_marker[ix][ty];
//			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
//			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
//			s_change |= NEQ( s_old, s_marker[ix][ty] );
//		}
//}		__syncthreads();
//
//		// output result back to global memory
//		for (iy = 0; iy < Y_THREADS && by+iy<sy; ++iy) {
//			// now treat ty as x, and iy as y, so global mem acccess is closer.
//			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
//			if (s_change) *change = true;
//		}
//		__syncthreads();
//
//
//}
//
//template <typename T>
//__global__ void
//iRec1DBackward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
//{
//
//	const int ty = threadIdx.x;
//	const int by = blockIdx.x * Y_THREADS;
//	// always 0.  const int bz = blockIdx.y;
//
//	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
//	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
//	bool s_change = false;
//
//
//
//		int startx;
//
//		T s_old;
//		for (startx = sx - Y_THREADS; startx > 0; startx -= Y_THREADS - 1) {
//
//			// copy part of marker and mask to shared memory
//			for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
//				// now treat ty as x, and iy as y, so global mem acccess is closer.
//				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
//				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
//			}
//			__syncthreads();
//
//			// perform iteration
//	if (by + ty < sy) {
//			for (int ix = Y_THREADS - 2; ix >= 0; ix--) {
//				s_old = s_marker[ix][ty];
//				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
//				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
//				s_change |= NEQ( s_old, s_marker[ix][ty] );
//			}
//}			__syncthreads();
//
//			// output result back to global memory
//			for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
//				// now treat ty as x, and iy as y, so global mem acccess is closer.
//				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
//			}
//			__syncthreads();
//
//		}
//
//		startx = 0;
//
//		// copy part of marker and mask to shared memory
//		for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
//			// now treat ty as x, and iy as y, so global mem acccess is closer.
//			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
//			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
//		}
//		__syncthreads();
//
//		// perform iteration
//	if (by + ty < sy) {
//		for (int ix = Y_THREADS - 2; ix >= 0; ix--) {
//			s_old = s_marker[ix][ty];
//			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
//			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
//			s_change |= NEQ( s_old, s_marker[ix][ty] );
//		}
//}		__syncthreads();
//
//		// output result back to global memory
//		for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
//			// now treat ty as x, and iy as y, so global mem acccess is closer.
//			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
//			if (s_change) *change = true;
//		}
//		__syncthreads();
//
//
//
//}

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
	const unsigned int x = (threadIdx.x + threadIdx.y * XX_THREADS) % WARP_SIZE;
	const unsigned int y = (threadIdx.x + threadIdx.y * XX_THREADS) / WARP_SIZE;
	const unsigned int ychunk = WARP_SIZE / XX_THREADS;
	const unsigned int xstop = sx - WARP_SIZE;
//	printf("(tx, ty) -> (x, y) : (%d, %d)->(%d,%d)\n", threadIdx.x, threadIdx.y, x, y);

	// XY_THREADS should be 32==warpSize, XX_THREADS should be 4 or 8.
	// init to 0...
	volatile __shared__ T s_marker[XY_THREADS][WARP_SIZE+1];
	volatile __shared__ T s_mask  [XY_THREADS][WARP_SIZE+1];
	volatile unsigned int s_change = 0;
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
			s_change |= s_new ^ s_old;
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
			s_change |= s_new ^ s_old;
			s_marker[threadIdx.y][i] = s_new;
		}
}
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x+1];
		}
	}


//	__syncthreads();
	if (s_change > 0) *change = true;
//	__syncthreads();

}


template <typename T>
__global__ void
iRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
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
	volatile unsigned int s_change = 0;
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
			s_change |= s_new ^ s_old;
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
			s_change |= s_new ^ s_old;
			s_marker[threadIdx.y][i] = s_new;
		}
}
		// output result back to global memory and set up for next x chunk
//#pragma unroll
		for (unsigned int i = 0; i < ychunk && y*ychunk+i < sy; ++i) {
			marker[start + i*sx] = s_marker[y * ychunk+i][x];
		}
	}

	//	__syncthreads();
	if (s_change > 0) *change = true;
	//	__syncthreads();
}



template <typename T>
__global__ void
iRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	unsigned int  s_change = 0;
	T s_old, s_new, s_prev;
	
if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = 0; iy < sy; ++iy) {
			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_change |= s_old ^ s_new;
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;

		}
}
		
		if (s_change != 0) *change = true;


}

template <typename T>
	__global__ void
iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	unsigned int s_change=0;
	T s_old, s_new, s_prev;

	if ( (bx + tx) < sx ) {

		s_prev = 0;

		for (int iy = sy - 1; iy >= 0; --iy) {

			// copy part of marker and mask to shared memory
			s_old = marker[iy * sx + bx + tx];

			// perform iteration
			s_new = min( max( s_prev, s_old ), mask[iy * sx + bx + tx] );
			s_change |= s_old ^ s_new;
			s_prev = s_new;

			// output result back to global memory
			marker[iy * sx + bx + tx] = s_new;
		}
	}

	if (s_change != 0) *change = true;

}

template <typename T>
__global__ void
iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const unsigned int sx, const unsigned int sy, bool* __restrict__ change )
{

	// best thing to do is to use linear arrays.  each warp does a column of 32.

	// parallelize along x.
	const unsigned int tx = threadIdx.x;
	const unsigned int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_B[MAX_THREADS+2];
	//	volatile T* s_marker = s_marker_B + 1;
	unsigned int s_change = 0;
	int tx1 = tx + 1;

	T s_new, s_old, s_prev;

if ( bx + tx < sx ) { // make sure number of threads is a divisor of sx.

	s_prev = 0;

	for (int iy = 0; iy < sy; ++iy) {
		// copy part of marker and mask to shared memory
		if (tx == 0) {
			s_marker_B[0] = (bx == 0) ? 0 : marker[iy*sx + bx - 1];
			s_marker_B[MAX_THREADS + 1] = (bx + MAX_THREADS >= sx) ? 0 : marker[iy*sx + bx + MAX_THREADS];
		}
		if (tx < WARP_SIZE) {
			// first warp, get extra stuff
			s_marker_B[tx1] = marker[iy*sx + bx + tx];
		}
		if (tx < MAX_THREADS - WARP_SIZE) {
			s_marker_B[tx1 + WARP_SIZE] = marker[iy*sx + bx + tx + WARP_SIZE];
		}
		__syncthreads();

		// perform iteration
		s_old = s_marker_B[tx1];
		s_new = min( max( s_prev, s_old ),  mask[iy*sx + bx + tx]);
		s_change |= s_old ^ s_new;

		// output result back to global memory
		s_marker_B[tx1] = s_new;
		marker[iy*sx + bx + tx] = s_new;
		__syncthreads();

		s_prev = max( max(s_marker_B[tx1-1], s_marker_B[tx1]), s_marker_B[tx1+1]);
	}
}
	if (s_change != 0) *change = true;
}


template <typename T>
__global__ void
iRec1DBackward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_B[MAX_THREADS+2];
//	volatile T* s_marker = s_marker_B + 1;
	unsigned int s_change = 0;
	int tx1 = tx + 1;  // for accessing s_marker_B

	T s_new, s_old, s_prev;

	if ( bx + tx < sx ) {  //make sure number of threads is a divisor of sx.

	s_prev = 0;

	for (int iy = sy - 1; iy >= 0; --iy) {

		if (tx == 0) {
			s_marker_B[0] = (bx == 0) ? 0 : marker[iy*sx + bx - 1];
			s_marker_B[MAX_THREADS+1] = (bx + MAX_THREADS >= sx) ? 0 : marker[iy*sx + bx + MAX_THREADS];
		}
		if (tx < WARP_SIZE) {
			// first warp, get extra stuff
			s_marker_B[tx1] = marker[iy*sx + bx + tx];
		}
		if (tx < MAX_THREADS - WARP_SIZE) {
			s_marker_B[tx1 + WARP_SIZE] = marker[iy*sx + bx + tx + WARP_SIZE];
		}
		__syncthreads();


		// perform iteration
		s_old = s_marker_B[tx1];
		s_new = min( max( s_prev, s_old ),  mask[iy*sx + bx + tx]);
		s_change |= s_old ^ s_new;

		// output result back to global memory
		s_marker_B[tx1] = s_new;
		marker[iy*sx + bx + tx] = s_new;
		__syncthreads();

		s_prev = max( max(s_marker_B[tx1-1], s_marker_B[tx1]), s_marker_B[tx1+1]);

	}
}
	if (s_change != 0) *change = true;


}


// connectivity:  if 8 conn, need to have border.
template <typename T>
unsigned int imreconstructIntCaller(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream) {
	//, unsigned char*h_markerFistPass) {


	// here because we are not using streams inside.
	//		if (stream == 0) cudaDeviceSynchronize();
	//		else  cudaStreamSynchronize(stream);


	//		printf("entering imrecon int caller with conn=%d\n", connectivity);

	// setup execution parameters
	bool conn8 = (connectivity == 8);

	dim3 threadsx( XX_THREADS, XY_THREADS );
	dim3 blocksx( (sy + threadsx.y - 1) / threadsx.y );
	//		dim3 threadsx2( Y_THREADS );
	//		dim3 blocksx2( (sy + threadsx2.y - 1) / threadsx2.y );
	dim3 threadsy( MAX_THREADS );
	dim3 blocksy( (sx + threadsy.x - 1) / threadsy.x );
	//		dim3 threadsy2( YX_THREADS, YY_THREADS );
	//		dim3 blocksy2( (sx + threadsy2.x - 1) / threadsy2.x, (sy + threadsy2.y - 1) / threadsy2.y  );
	//		size_t Nsy = (threadsy.x * 3 + 2) * sizeof(uchar4);

	// stability detection
	unsigned int iter = 0;
	bool *h_change, *d_change;
	h_change = (bool*) malloc( sizeof(bool) );
	 cudaMalloc( (void**) &d_change, sizeof(bool) ) ;

	*h_change = true;
	//		printf("completed setup for imrecon int caller \n");

	//long t1, t2;

	if (conn8) {
		while ( (*h_change) && (iter < 100000) )  // repeat until stability
		{
//			t1 = ClockGetTime();
			iter++;
			*h_change = false;
			init_change<<< 1, 1, 0, stream>>>( d_change );

			// dopredny pruchod pres osu X
			iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
			//				iRec1DForward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// dopredny pruchod pres osu Y
			iRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu X
			iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
			//				iRec1DBackward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu Y
			iRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);
			//				printf("%d sync \n", iter);

			 cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) ;
			//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

//			t2 = ClockGetTime();
//
//			if (iter == 1) {
//
//				 cudaMemcpy( h_markerFistPass, marker, sizeof(unsigned char) * sx * sy, cudaMemcpyDeviceToHost ) ;
//				printf("first pass 8conn == scan, %lu ms\n", t2-t1);
//			}

		}
	} else {
		while ( (*h_change) && (iter < 100000) )  // repeat until stability
		{
//			t1 = ClockGetTime();

			iter++;
			*h_change = false;
			init_change<<< 1, 1, 0, stream>>>( d_change );

			// dopredny pruchod pres osu X
			iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
			//				iRec1DForward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// dopredny pruchod pres osu Y
			iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu X
			iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
			//				iRec1DBackward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu Y
			iRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);
			//				printf("%d sync \n", iter);

			 cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) ;
			//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

//			t2 = ClockGetTime();
//			if (iter == 1) {
//				// cudaMemcpy( h_markerFistPass, marker, sizeof(unsigned char) * sx * sy, cudaMemcpyDeviceToDevice ) ;
//				printf("first pass 4conn == scan, %lu ms\n", t2-t1);
////				break;
//			}


		}
	}

	 cudaFree(d_change) ;
	free(h_change);

	//		printf("Number of iterations: %d\n", iter);
	 cudaGetLastError();

	return iter;
}

__device__ bool checkCandidateNeighbor4(unsigned char *marker, const unsigned char *mask, int x, int y, int ncols, int nrows,unsigned char pval){
	bool isCandidate = false;
	int index = 0;

	unsigned char markerXYval;
	unsigned char maskXYval;
	if(x < (ncols-1)){
		// check right pixel
		index = y * ncols + (x+1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval, maskXYval)) ){
			isCandidate = true;
		}
	}

	if(y < (nrows-1)){
		// check pixel bellow current
		index = (y+1) * ncols + x;

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	// check left pixel
	if(x > 0){
		index = y * ncols + (x-1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	if(y > 0){
		// check up pixel
		index = (y-1) * ncols + x;

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}
	return isCandidate;
}

__device__ bool checkCandidateNeighbor8(unsigned char *marker, const unsigned char *mask, int x, int y, int ncols, int nrows,unsigned char pval){
	int index = 0;
	bool isCandidate = checkCandidateNeighbor4(marker, mask, x, y, ncols, nrows, pval);
//	if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0){
//		printf("checkCandidateNeighbor8\n");
//	}

	unsigned char markerXYval;
	unsigned char maskXYval;

	// check up right corner
	if(x < (ncols-1) && y > 0){
		// check right pixel
		index = (y-1) * ncols + (x+1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval, maskXYval)) ){
			isCandidate = true;
		}
	}

	// check up left corner
	if(x> 0 && y > 0){
		// check pixel bellow current
		index = (y-1) * ncols + (x-1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	// check bottom left pixel
	if(x > 0 && y < (nrows-1)){
		index = (y+1) * ncols + (x-1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}

	// check bottom right
	if(x < (ncols-1) && y < (nrows-1)){
		index = (y+1) * ncols + (x+1);

		markerXYval = marker[index];
		maskXYval = mask[index];
		if( (markerXYval < min(pval,maskXYval)) ){
			isCandidate = true;
		}
	}
	return isCandidate;
}


__global__ void initQueuePixels(unsigned char *marker, const unsigned char *mask, int sx, int sy, bool conn8, int *d_queue, int *d_queue_size){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// if it is inside image without right/bottom borders
	if(y < (sy) && x < (sx)){
		int input_index = y * sy + x;
		unsigned char pval = marker[input_index];
		bool isCandidate = false;
		if(conn8){
			// connectivity 8
			isCandidate = checkCandidateNeighbor8(marker, mask, x, y, sx, sy, pval);
		}else{
			// connectivity 4
			isCandidate = checkCandidateNeighbor4(marker, mask, x, y, sx, sy, pval);
		}
		if(isCandidate){
			int queuePos = atomicAdd((unsigned int*)d_queue_size, 1);
			d_queue[queuePos] = input_index;
		}	
	}
}

// connectivity:  if 8 conn, need to have border.
template <typename T> int *imreconstructIntCallerBuildQueue(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, const int connectivity, int &queueSize, int num_iterations, cudaStream_t stream) {

	// setup execution parameters
	bool conn8 = (connectivity == 8);

	dim3 threadsx( XX_THREADS, XY_THREADS );
	dim3 blocksx( (sy + threadsx.y - 1) / threadsx.y );
	
	dim3 threadsy( MAX_THREADS );
	dim3 blocksy( (sx + threadsy.x - 1) / threadsy.x );
	bool *d_change;
	 cudaMalloc( (void**) &d_change, sizeof(bool) ) ;

	// alloc it with the same size as the input image.
	int *d_queue = NULL;
	 cudaMalloc( (void**) &d_queue, sizeof(int) * sx * sy ) ;

	int *d_queue_size;
	 cudaMalloc( (void**) &d_queue_size, sizeof(int)) ;
	 cudaMemset( (void*) d_queue_size, 0, sizeof(int)) ;

	// stability detection
	unsigned int iter = 0;

	//long t1, t2, t3;

//	t1 = ClockGetTime();
	if (conn8) {
		while(iter < num_iterations){
			iter++;

			// dopredny pruchod pres osu X
			iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// dopredny pruchod pres osu Y
			iRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu X
			iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu Y
			iRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);
		}
	} else {
		while(iter < num_iterations){
			iter++;
			// dopredny pruchod pres osu X
			iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// dopredny pruchod pres osu Y
			iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu X
			iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );

			// zpetny pruchod pres osu Y
			iRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);
		}
	}


//	t2 = ClockGetTime();
//	printf("first pass 4conn == scan, %lu ms\n", t2-t1);
	 cudaFree(d_change) ;

	// This is now a per pixel operation where we build the 
	// first queue of pixels that may propagate their values.
	// Creating a single thread per-pixel in the input image
	dim3 threads(16, 16);
	dim3 grid((sx + threads.x - 1) / threads.x, (sy + threads.y - 1) / threads.y);

	//
	initQueuePixels<<< grid, threads, 0, stream >>>(marker, mask, sx, sy, conn8, d_queue, d_queue_size);
			if (stream == 0) cudaDeviceSynchronize();
			else  cudaStreamSynchronize(stream);

	int h_compact_queue_size;

	 cudaMemcpy( &h_compact_queue_size, d_queue_size, sizeof(int), cudaMemcpyDeviceToHost ) ;
	
	//t3 = ClockGetTime();
//	printf("	compactQueueSize %d, time to generate %lu ms\n", h_compact_queue_size, t3-t2);


	int *d_queue_fit = NULL;
	// alloc current size +1000 (magic number)
	 cudaMalloc( (void**) &d_queue_fit, sizeof(int) * (h_compact_queue_size+1000)*2 ) ;

	// Copy content of the d_queue (which has the size of the image x*y) to a more compact for (d_queue_fit). 
	// This should save a lot of memory, since the compact queue is usually much smaller than the image size
	 cudaMemcpy( d_queue_fit, d_queue, sizeof(int) * h_compact_queue_size, cudaMemcpyDeviceToDevice ) ;

	// This is the int containing the size of the queue
	 cudaFree(d_queue_size) ;

	// Cleanup the temporary memory use to store the queue
	 cudaFree(d_queue) ;

	queueSize = h_compact_queue_size;

	return d_queue_fit;
}


template unsigned int imreconstructIntCaller<unsigned char>(unsigned char*, const unsigned char*, const int, const int,
		const int, cudaStream_t);
//,unsigned char*h_markerFistPass );

template int *imreconstructIntCallerBuildQueue<unsigned char>(unsigned char*, const unsigned char*, const int, const int, const int, int&, int, cudaStream_t);

}}
