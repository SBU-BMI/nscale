// adaptation of Pavel's imreconstruction code for openCV

#include "change_kernel.cuh"
#ifdef _MSC_VER
#include "time_win.h"
#else
#include <sys/time.h>
#endif



#define MAX_THREADS		256
#define X_THREADS			32
#define Y_THREADS			32
#define NEQ(a,b)    ( (a) != (b) )


long ClockGetTimeb()
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
 * original code
 */
template <typename T>
__global__ void
bRec1DForward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * blockDim.x;

	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
	volatile __shared__ bool  s_change[Y_THREADS][Y_THREADS+1];



		int startx, iy, ix;
		for (int ix = 0; ix < Y_THREADS; ++ix) {
			s_change[ix][ty] = false;
		}
		__syncthreads();

		T s_old;
		// the increment allows overlap by 1 between iterations to move the data to next block.
		for (startx = 0; startx < sx - Y_THREADS; startx += Y_THREADS - 1) {
			// copy part of marker and mask to shared memory
			for (iy = 0; iy < Y_THREADS && by+iy < sy; ++iy) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
			}
			__syncthreads();

			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
	if (by + ty < sy) {
			for (ix = 1; ix < Y_THREADS; ++ix) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] |= s_marker[ix-1][ty];
				s_marker[ix][ty] &= s_mask  [ix]  [ty];
				s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
			}
	}
			__syncthreads();

			// output result back to global memory
			for (iy = 0; iy < Y_THREADS && by+iy < sy; ++iy) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			}
			__syncthreads();

		}

		startx = sx - Y_THREADS;

		// copy part of marker and mask to shared memory
		for (iy = 0; iy < Y_THREADS && by+iy < sy; ++iy) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
		}
		__syncthreads();

		// perform iteration
	if (by + ty < sy) {
		for (ix = 1; ix < Y_THREADS; ++ix) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] |= s_marker[ix-1][ty];
			s_marker[ix][ty] &= s_mask  [ix]  [ty];
			s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
		}
	}
		__syncthreads();

		// output result back to global memory
		for (iy = 0; iy < Y_THREADS && by+iy < sy; ++iy) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change[iy][ty]) *change = true;
		}
		__syncthreads();


}

template <typename T>
__global__ void
bRec1DBackward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * Y_THREADS;
	// always 0.  const int bz = blockIdx.y;

	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
	volatile __shared__ bool  s_change[Y_THREADS][Y_THREADS+1];



		int startx;
		for (int ix = 0; ix < Y_THREADS; ix++) {
			s_change[ix][ty] = false;
		}
		__syncthreads();

		T s_old;
		for (startx = sx - Y_THREADS; startx > 0; startx -= Y_THREADS - 1) {

			// copy part of marker and mask to shared memory
			for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
			}
			__syncthreads();

			// perform iteration
	if (by + ty < sy) {
			for (int ix = Y_THREADS - 2; ix >= 0; ix--) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] |= s_marker[ix+1][ty];
				s_marker[ix][ty] &= s_mask  [ix]  [ty];
				s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
			}
}
			__syncthreads();

			// output result back to global memory
			for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			}
			__syncthreads();

		}

		startx = 0;

		// copy part of marker and mask to shared memory
		for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
		}
		__syncthreads();

		// perform iteration
	if (by + ty < sy) {
		for (int ix = Y_THREADS - 2; ix >= 0; ix--) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] |= s_marker[ix+1][ty];
			s_marker[ix][ty] &= s_mask  [ix]  [ty];
			s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
		}
}
		__syncthreads();

		// output result back to global memory
		for (int iy = 0; iy < Y_THREADS && by+iy<sy; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change[iy][ty]) *change = true;
		}
		__syncthreads();



}

////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////
/*
 * original code
 */
/*
 template <typename T>
__global__ void
bRec1DForward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * Y_THREADS;

	if (ty + by < sy) {

		volatile __shared__ T s_marker[X_THREADS][Y_THREADS+1];
		volatile __shared__ T s_mask  [X_THREADS][Y_THREADS+1];
		volatile __shared__ bool  s_change[X_THREADS][Y_THREADS+1];
		s_change[tx][ty] = false;
		__syncthreads();

		T s_old;
		int startx;
		// the increment allows overlap by 1 between iterations to move the data to next block.
		for (startx = 0; startx < sx - X_THREADS; startx += X_THREADS - 1) {

			// copy part of marker and mask to shared memory
			s_marker[tx][ty] = marker[(by + ty)*sx + startx + tx];
			s_mask  [tx][ty] = mask  [(by + ty)*sx + startx + tx];
			__syncthreads();

			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
			for (int ix = 1; ix < X_THREADS; ix++) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] |= s_marker[ix-1][ty];
				s_marker[ix][ty] &= s_mask  [ix]  [ty];
				s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
			}
			__syncthreads();

			// output result back to global memory
			marker[(by + ty)*sx + startx + tx] = s_marker[tx][ty];
			__syncthreads();

		}

		startx = sx - X_THREADS;

		// copy part of marker and mask to shared memory
		s_marker[tx][ty] = marker[(by + ty)*sx + startx + tx];
		s_mask  [tx][ty] = mask  [(by + ty)*sx + startx + tx];
		__syncthreads();

		// perform iteration
		for (int ix = 1; ix < X_THREADS; ix++) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] |= s_marker[ix-1][ty];
			s_marker[ix][ty] &= s_mask  [ix]  [ty];
			s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
		}
		__syncthreads();

		// output result back to global memory
		marker[(by + ty)*sx + startx + tx] = s_marker[tx][ty];
		__syncthreads();

		if (s_change[tx][ty]) *change = true;
		__syncthreads();

	}

}
*/
/*
template <typename T>
__global__ void
bRec1DBackward_X_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
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
				s_marker[ix][ty] |= s_marker[ix+1][ty];
				s_marker[ix][ty] &= s_mask  [ix]  [ty];
				s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
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
			s_marker[ix][ty] |= s_marker[ix+1][ty];
			s_marker[ix][ty] &= s_mask  [ix]  [ty];
			s_change[ix][ty] |= s_old ^ s_marker[ix][ty];
		}
		__syncthreads();

		// output result back to global memory
		marker[(by + ty)*sx + startx + tx] = s_marker[tx][ty];
		__syncthreads();
		
		if (s_change[tx][ty]) *change = true;
		__syncthreads();

	}

}
*/

/*
template <typename T>
__global__ void
bRec1D8ConnectedWindowedMax ( DevMem2D_<T> g_marker_max, DevMem2D_<T> g_marker)
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
bRec1DForward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	volatile __shared__ bool  s_change  [MAX_THREADS];
	
	if ( (bx + tx) < sx ) {

		s_change[tx] = false;
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
			s_marker_B[tx] |= s_marker_A[tx];
			s_marker_B[tx] &= s_mask    [tx];
			s_change[tx] |= s_old ^ s_marker_B[tx];
//			__syncthreads();

			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
//			__syncthreads();

		}
		__syncthreads();

		if (s_change[tx]) *change = true;
		__syncthreads();

	}

}

template <typename T>
__global__ void
bRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	volatile __shared__ bool  s_change  [MAX_THREADS];

	if ( (bx + tx) < sx ) {

		s_change[tx] = false;
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
			s_marker_B[tx] |= s_marker_A[tx];
			s_marker_B[tx] &= s_mask    [tx];
			s_change[tx] |= s_old ^ s_marker_B[tx];
//			__syncthreads();

			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
//			__syncthreads();

		}
		__syncthreads();

		if (s_change[tx]) *change = true;
		__syncthreads();

	}

}

// 8 conn...
//overlap:  tx 0 to 7 maps to -1 to 6, with usable from 0 to 5.  output for 6-11, from 5 - 12
//formula:  bx * (block-2) - 1 + tx = startx in src data.
//formula:  


template <typename T>
__global__ void
bRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;
	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	volatile __shared__ bool  s_change  [MAX_THREADS];

	if ( bx+tx < sx ) {

		s_change[tx] = false;
		s_marker_B[tx] = marker[bx+tx];
		__syncthreads();

		T s_old;
		for (int ty = 1; ty < sy; ty++) {
		
			// copy part of marker and mask to shared memory
			s_marker_A[tx] = s_marker_B[tx];
			if (bx+tx > 0) s_marker_A[tx] = ((tx == 0) ? marker[(ty-1) * sx + bx + tx - 1] : s_marker_B[tx-1]) | s_marker_A[tx];
			if (bx+tx < sx-1) s_marker_A[tx] = ((tx == blockDim.x-1) ? marker[(ty-1) * sx + bx + tx + 1] : s_marker_B[tx+1]) | s_marker_A[tx];
			s_mask    [tx] = mask[ty * sx + bx + tx];
			//__syncthreads();
			s_old = marker[ty * sx + bx + tx];
			//s_marker_B[tx] = marker[ty * sx + bx + tx];
			__syncthreads();

			// perform iteration
			//s_old = s_marker_B[tx];
			//s_marker_B[tx] |= s_marker_A[tx];
			s_marker_B[tx] = s_marker_A[tx] | s_old;
			s_marker_B[tx] &= s_mask    [tx];
			s_change[tx] |= s_old ^ s_marker_B[tx];
			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
			__syncthreads();
		}

		if (s_change[tx]) *change = true;
		__syncthreads();
	}
}

template <typename T>
__global__ void
bRec1DBackward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	volatile __shared__ T s_marker_A[MAX_THREADS];
	volatile __shared__ T s_marker_B[MAX_THREADS];
	volatile __shared__ T s_mask    [MAX_THREADS];
	volatile __shared__ bool  s_change  [MAX_THREADS];

	if ( bx + tx < sx ) {

		
		s_change[tx] = false;
		s_marker_B[tx] = marker[(sy -1) * sx + bx + tx];
		__syncthreads();

		T s_old;
		for (int ty = sy - 2; ty >= 0; ty--) {

			// copy part of marker and mask to shared memory
			s_marker_A[tx] = s_marker_B[tx];
			if (bx + tx > 0) s_marker_A[tx] = ((tx == 0) ? marker[(ty+1) * sx + bx + tx -1] : s_marker_B[tx-1]) | s_marker_A[tx];
			if (bx + tx < sx-1) s_marker_A[tx] = ((tx == blockDim.x-1) ? marker[(ty+1) * sx + bx + tx +1] : s_marker_B[tx+1]) | s_marker_A[tx];
			s_mask    [tx] = mask[ty * sx + bx + tx];
//			__syncthreads();
			s_old = marker[ty * sx + bx + tx];
//			s_marker_B[tx] = marker[ty * sx + bx + tx];
			__syncthreads();

			// perform iteration
			//s_old = s_marker_B[tx];
			s_marker_B[tx] = s_marker_A[tx] | s_old;
//			s_marker_B[tx] |= s_marker_A[tx];
			s_marker_B[tx] &= s_mask    [tx];
			s_change[tx] |= s_old ^ s_marker_B[tx];
			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
			__syncthreads();


		}

		if (s_change[tx]) *change = true;
		__syncthreads();

	}

}
__device__ bool checkCandidateNeighbor4Binary(unsigned char *marker, const unsigned char *mask, int x, int y, int ncols, int nrows,unsigned char pval){
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

__device__ bool checkCandidateNeighbor8Binary(unsigned char *marker, const unsigned char *mask, int x, int y, int ncols, int nrows,unsigned char pval){
	int index = 0;
	bool isCandidate = checkCandidateNeighbor4Binary(marker, mask, x, y, ncols, nrows, pval);
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


__global__ void initQueuePixelsBinary(unsigned char *marker, const unsigned char *mask, int sx, int sy, bool conn8, int *d_queue, int *d_queue_size){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// if it is inside image without right/bottom borders
	if(y < (sy) && x < (sx)){
		int input_index = y * sy + x;
		unsigned char pval = marker[input_index];
		bool isCandidate = false;
		if(conn8){
			// connectivity 8
			isCandidate = checkCandidateNeighbor8Binary(marker, mask, x, y, sx, sy, pval);
		}else{
			// connectivity 4
			isCandidate = checkCandidateNeighbor4Binary(marker, mask, x, y, sx, sy, pval);
		}
		if(isCandidate){
			int queuePos = atomicAdd((unsigned int*)d_queue_size, 1);
			d_queue[queuePos] = input_index;
		}	
	}
}



	// connectivity:  if 8 conn, need to have border.

	template <typename T>
	unsigned int imreconstructBinaryCaller(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream) {

		// here because we are not using streams inside.
//		if (stream == 0) cudaDeviceSynchronize();
//		else  cudaStreamSynchronize(stream);


//		printf("entering imrecon binary caller with conn=%d\n", connectivity);

		// setup execution parameters
		bool conn8 = (connectivity == 8);

		dim3 threadsx( X_THREADS, Y_THREADS );
		dim3 threadsx2( Y_THREADS );
		dim3 blocksx( (sy + threadsx.y - 1) / threadsx.y );
		dim3 threadsy( MAX_THREADS );
		dim3 blocksy( (sx + threadsy.x - 1) / threadsy.x );

		// stability detection
		unsigned int iter = 0;
		bool *h_change, *d_change;
		h_change = (bool*) malloc( sizeof(bool) );
		 cudaMalloc( (void**) &d_change, sizeof(bool) ) ;
		
		*h_change = true;
//		printf("completed setup for imrecon binary caller \n");
		//long t1, t2;

		if (conn8) {
			while ( (*h_change) && (iter < 100000) )  // repeat until stability
			{

//				t1 = ClockGetTimeb();
				iter++;
				*h_change = false;
				init_change<<< 1, 1, 0, stream>>>( d_change );

				// dopredny pruchod pres osu X
				//bRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DForward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				bRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//bRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DBackward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				bRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				if (stream == 0) cudaDeviceSynchronize();
				else  cudaStreamSynchronize(stream);
//				printf("%d sync \n", iter);

				 cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) ;
//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

//				t2 = ClockGetTimeb();
//				if (iter == 1) {
//					printf("first pass 8conn binary== scan, %lu ms\n", t2-t1);
//				}


			}
		} else {
			while ( (*h_change) && (iter < 100000) )  // repeat until stability
			{
//				t1 = ClockGetTimeb();
				iter++;
				*h_change = false;
				init_change<<< 1, 1, 0, stream>>>( d_change );

				// dopredny pruchod pres osu X
				//bRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DForward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				bRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//bRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DBackward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				bRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				if (stream == 0) cudaDeviceSynchronize();
				else  cudaStreamSynchronize(stream);
//				printf("%d sync \n", iter);

				 cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) ;
//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

//				t2 = ClockGetTimeb();
//				if (iter == 1) {
//					printf("first pass 4conn binary == scan, %lu ms\n", t2-t1);
//				}

			}
		}

		 cudaFree(d_change) ;
		free(h_change);

//		printf("Number of iterations: %d\n", iter);
		 cudaGetLastError();

		return iter;
	}

	template <typename T>
	int* imreconstructBinaryCallerBuildQueue(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy,
		const int connectivity, int &queueSize, int num_iterations, cudaStream_t stream) {

		// here because we are not using streams inside.
//		if (stream == 0) cudaDeviceSynchronize();
//		else  cudaStreamSynchronize(stream);


//		printf("entering imrecon binary caller with conn=%d\n", connectivity);

		// setup execution parameters
		bool conn8 = (connectivity == 8);

		dim3 threadsx( X_THREADS, Y_THREADS );
		dim3 threadsx2( Y_THREADS );
		dim3 blocksx( (sy + threadsx.y - 1) / threadsx.y );
		dim3 threadsy( MAX_THREADS );
		dim3 blocksy( (sx + threadsy.x - 1) / threadsy.x );

		// stability detection
		unsigned int iter = 0;
//		bool *h_change, *d_change;
		bool *d_change;
//		h_change = (bool*) malloc( sizeof(bool) );
		 cudaMalloc( (void**) &d_change, sizeof(bool) ) ;
//		
//		*h_change = true;
//		printf("completed setup for imrecon binary caller \n");
		//long t1, t2;

		if (conn8) {
			while ( (iter < num_iterations) )  // repeat until stability
			{

//				t1 = ClockGetTimeb();
				iter++;
//				*h_change = false;
//				init_change<<< 1, 1, 0, stream>>>( d_change );

				// dopredny pruchod pres osu X
				//bRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DForward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				bRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//bRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DBackward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				bRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				if (stream == 0) cudaDeviceSynchronize();
				else  cudaStreamSynchronize(stream);
////				printf("%d sync \n", iter);
//
//				 cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) ;
////				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));
//
////				t2 = ClockGetTimeb();
////				if (iter == 1) {
////					printf("first pass 8conn binary== scan, %lu ms\n", t2-t1);
////				}
//

			}
		} else {
			while ( (iter < num_iterations) )  // repeat until stability
			{
//				t1 = ClockGetTimeb();
				iter++;
//				*h_change = false;
//				init_change<<< 1, 1, 0, stream>>>( d_change );

				// dopredny pruchod pres osu X
				//bRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DForward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				bRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//bRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				bRec1DBackward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				bRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				if (stream == 0) cudaDeviceSynchronize();
				else  cudaStreamSynchronize(stream);
//				printf("%d sync \n", iter);

//				 cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) ;
//				printf("%d read flag : value %s\n", iter, (*h_change ? "true" : "false"));

//				t2 = ClockGetTimeb();
//				if (iter == 1) {
//					printf("first pass 4conn binary == scan, %lu ms\n", t2-t1);
//				}

			}
		}

		cudaFree(d_change) ;

		// This is now a per pixel operation where we build the 
		// first queue of pixels that may propagate their values.
		// Creating a single thread per-pixel in the input image
		dim3 threads(16, 16);
		dim3 grid((sx + threads.x - 1) / threads.x, (sy + threads.y - 1) / threads.y);

		int *d_queue = NULL;
		cudaMalloc( (void**) &d_queue, sizeof(int) * sx * sy ) ;
		int *d_queue_size;
		cudaMalloc( (void**) &d_queue_size, sizeof(int)) ;
		cudaMemset( (void*) d_queue_size, 0, sizeof(int)) ;

		//
		initQueuePixelsBinary<<< grid, threads, 0, stream >>>(marker, mask, sx, sy, conn8, d_queue, d_queue_size);

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

		cudaGetLastError();
		
		return d_queue_fit;
	

		//		free(h_change);

		//		printf("Number of iterations: %d\n", iter);
//		cudaGetLastError();
//
//		return iter;
	}
	template int* imreconstructBinaryCallerBuildQueue<unsigned char>(unsigned char*, const unsigned char*, const int sx, const int sy,
		const int connectivity, int &queueSize, int num_iterations, cudaStream_t stream);

	template unsigned int imreconstructBinaryCaller<unsigned char>(unsigned char*, const unsigned char*, const int, const int,
		const int, cudaStream_t );
	template unsigned int imreconstructBinaryCaller<int>(int*, const int*, const int, const int,
		const int, cudaStream_t );
}}
