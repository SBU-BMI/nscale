// adaptation of Pavel's imreconstruction code for openCV

#include "internal_shared.hpp"
#include "change_kernel.cuh"

#define MAX_THREADS		256
#define X_THREADS			16
#define Y_THREADS			16
#define NEQ(a,b)    ( (a) != (b) )


using namespace cv::gpu;


namespace nscale { namespace gpu {


////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////
/*
 * original code
 */
template <typename T>
__global__ void
iRec1DForward_X_dilation2 (T* marker, const T* mask, const int sx, const int sy, bool* change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * Y_THREADS;
	const int y = by+ty;
	
	if (y < sy) {

		volatile __shared__ T s_marker[Y_THREADS][Y_THREADS];
		volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS];
		volatile __shared__ bool  s_change[Y_THREADS][Y_THREADS];
		
		int startx;
		for (int ix = 0; ix < Y_THREADS; ix++) {
			s_change[ix][ty] = false;
		}
		__syncthreads();

		T s_old;
		// the increment allows overlap by 1 between iterations to move the data to next block.
		for (startx = 0; startx < sx - Y_THREADS; startx += Y_THREADS - 1) {

			// copy part of marker and mask to shared memory
			for (int iy = 0; iy < Y_THREADS; iy++) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
				s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
			}
			__syncthreads();

			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
			for (int ix = 1; ix < Y_THREADS; ix++) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
			}
			__syncthreads();

			// output result back to global memory
			for (int iy = 0; iy < Y_THREADS; iy++) {
				// now treat ty as x, and iy as y, so global mem acccess is closer.
				marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			}
			__syncthreads();

		}

		startx = sx - Y_THREADS;

		// copy part of marker and mask to shared memory
		for (int iy = 0; iy < Y_THREADS; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			s_marker[ty][iy] = marker[(by + iy)*sx + startx + ty];
			s_mask  [ty][iy] = mask  [(by + iy)*sx + startx + ty];
		}
		__syncthreads();

		// perform iteration
		for (int ix = 1; ix < Y_THREADS; ix++) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
			s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
		}
		__syncthreads();

		// output result back to global memory
		for (int iy = 0; iy < Y_THREADS; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change[iy][ty]) *change = true;
		}
		__syncthreads();

	}

}

template <typename T>
__global__ void
iRec1DBackward_X_dilation2 (T* marker, const T* mask, const int sx, const int sy, bool* change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * Y_THREADS;
	// always 0.  const int bz = blockIdx.y;


	if (by + ty < sy) {

		volatile __shared__ T s_marker[Y_THREADS][Y_THREADS];
		volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS];
		volatile __shared__ bool  s_change[Y_THREADS][Y_THREADS];
		
		int startx;
		for (int ix = 0; ix < Y_THREADS; ix++) {
			s_change[ix][ty] = false;
		}
		__syncthreads();

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
				s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
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
			s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
		}
		__syncthreads();

		// output result back to global memory
		for (int iy = 0; iy < Y_THREADS; iy++) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change[iy][ty]) *change = true;
		}
		__syncthreads();


	}

}

////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////
/*
 * original code
 */
template <typename T>
__global__ void
iRec1DForward_X_dilation ( T* marker, const T* mask, const int sx, const int sy, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * Y_THREADS;

	if (ty + by < sy) {

		volatile __shared__ T s_marker[X_THREADS][Y_THREADS];
		volatile __shared__ T s_mask  [X_THREADS][Y_THREADS];
		volatile __shared__ bool  s_change[X_THREADS][Y_THREADS];
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
				s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
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
			s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
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

template <typename T>
__global__ void
iRec1DBackward_X_dilation ( T* marker, const T* mask, const int sx, const int sy, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * Y_THREADS;
	// always 0.  const int bz = blockIdx.y;
	
	if (by + ty < sy) {

		volatile __shared__ T s_marker[X_THREADS][Y_THREADS];
		volatile __shared__ T s_mask  [X_THREADS][Y_THREADS];
		volatile __shared__ bool  s_change[X_THREADS][Y_THREADS];
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
iRec1DForward_Y_dilation ( T* marker, const T* mask, const int sx, const int sy, bool* change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;
	
	if ( (bx + tx) < sx ) {

		volatile __shared__ T s_marker_A[MAX_THREADS];
		volatile __shared__ T s_marker_B[MAX_THREADS];
		volatile __shared__ T s_mask    [MAX_THREADS];
		volatile __shared__ bool  s_change  [MAX_THREADS];
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
			s_marker_B[tx] = max( s_marker_A[tx], s_marker_B[tx] );
			s_marker_B[tx] = min( s_marker_B[tx], s_mask    [tx] );
			s_change[tx] |= NEQ( s_old, s_marker_B[tx] );
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
iRec1DBackward_Y_dilation ( T* marker, const T* mask, const int sx, const int sy, bool* change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	if ( (bx + tx) < sx ) {

		volatile __shared__ T s_marker_A[MAX_THREADS];
		volatile __shared__ T s_marker_B[MAX_THREADS];
		volatile __shared__ T s_mask    [MAX_THREADS];
		volatile __shared__ bool  s_change  [MAX_THREADS];
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
			s_marker_B[tx] = max( s_marker_A[tx], s_marker_B[tx] );
			s_marker_B[tx] = min( s_marker_B[tx], s_mask    [tx] );
			s_change[tx] |= NEQ( s_old, s_marker_B[tx] );
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
iRec1DForward_Y_dilation_8 ( T* marker, const T* mask, const int sx, const int sy, bool* change )
{
	// parallelize along x.
	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	if ( bx+tx < sx ) {
		
		volatile __shared__ T s_marker_A[MAX_THREADS];
		volatile __shared__ T s_marker_B[MAX_THREADS];
		volatile __shared__ T s_mask    [MAX_THREADS];
		volatile __shared__ bool  s_change  [MAX_THREADS];
		s_change[tx] = false;
		s_marker_B[tx] = marker[bx+tx];
		__syncthreads();

		T s_old;
		for (int ty = 1; ty < sy; ty++) {
		
			// copy part of marker and mask to shared memory
			s_marker_A[tx] = s_marker_B[tx];
			if (bx+tx > 0) s_marker_A[tx] = max((tx == 0) ? marker[(ty-1) * sx + bx + tx - 1] : s_marker_B[tx-1], s_marker_A[tx]);
			if (bx+tx < sx-1) s_marker_A[tx] = max((tx == blockDim.x-1) ? marker[(ty-1) * sx + bx + tx + 1] : s_marker_B[tx+1], s_marker_A[tx]);
			s_mask    [tx] = mask[ty * sx + bx + tx];
			//__syncthreads();
			s_old = marker[ty * sx + bx + tx];
			//s_marker_B[tx] = marker[ty * sx + bx + tx];
			__syncthreads();

			// perform iteration
			//s_old = s_marker_B[tx];
			//s_marker_B[tx] = max( s_marker_A[tx], s_marker_B[tx] );
			s_marker_B[tx] = max( s_marker_A[tx], s_old );
			s_marker_B[tx] = min( s_marker_B[tx], s_mask    [tx] );
			s_change[tx] |= NEQ( s_old, s_marker_B[tx] );
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
iRec1DBackward_Y_dilation_8 ( T* marker, const T* mask, const int sx, const int sy, bool* change )
{

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * MAX_THREADS;

	if ( bx + tx < sx ) {

		volatile __shared__ T s_marker_A[MAX_THREADS];
		volatile __shared__ T s_marker_B[MAX_THREADS];
		volatile __shared__ T s_mask    [MAX_THREADS];
		volatile __shared__ bool  s_change  [MAX_THREADS];
		
		s_change[tx] = false;
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
			s_change[tx] |= NEQ( s_old, s_marker_B[tx] );
			// output result back to global memory
			marker[ty * sx + bx + tx] = s_marker_B[tx];
			__syncthreads();


		}

		if (s_change[tx]) *change = true;
		__syncthreads();

	}

}


	// connectivity:  if 8 conn, need to have border.

	template <typename T>
	unsigned int imreconstructIntCaller(T* marker, const T* mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream) {

		// here because we are not using streams inside.
//		if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
//		else cudaSafeCall( cudaStreamSynchronize(stream));


		printf("entering imrecon int caller with conn=%d\n", connectivity);

		// setup execution parameters
		bool conn8 = (connectivity == 8);

		dim3 threadsx( X_THREADS, Y_THREADS );
		dim3 threadsx2( Y_THREADS );
		dim3 blocksx( divUp(sy, threadsx.y) );
		dim3 threadsy( MAX_THREADS );
		dim3 blocksy( divUp(sx, threadsy.x) );

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
				//iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DForward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				iRec1DForward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DBackward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

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
				//iRec1DForward_X_dilation <<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DForward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
				iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DBackward_X_dilation2<<< blocksx, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

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
