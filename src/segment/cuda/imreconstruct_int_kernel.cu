// adaptation of Pavel's imreconstruction code for openCV

#include "internal_shared.hpp"
#include "change_kernel.cuh"

#define MAX_THREADS		256
#define X_THREADS			32
#define Y_THREADS			32
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
iRec1DForward_X_dilation2 (T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{

	const int ty = threadIdx.x;
	const int by = blockIdx.x * blockDim.x;
	
	volatile __shared__ T s_marker[Y_THREADS][Y_THREADS+1];
	volatile __shared__ T s_mask  [Y_THREADS][Y_THREADS+1];
	volatile __shared__ bool  s_change[Y_THREADS][Y_THREADS+1];

	
	if (by + ty < sy) {

		int startx, iy, ix;
		for (int ix = 0; ix < Y_THREADS; ++ix) {
			s_change[ix][ty] = false;
		}
		__syncthreads();

		T s_old, last;
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
			last = s_marker[0][ty];
			for (ix = 1; ix < Y_THREADS; ++ix) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] = max( s_marker[ix][ty], last );
				s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
				last = s_marker[ix][ty];
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
		last = s_marker[0][ty];
		for (ix = 1; ix < Y_THREADS; ++ix) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] = max( s_marker[ix][ty], last );
			s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
			s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
			last = s_marker[0][ty];
		}
		__syncthreads();

		// output result back to global memory
		for (iy = 0; iy < Y_THREADS; ++iy) {
			// now treat ty as x, and iy as y, so global mem acccess is closer.
			marker[(by + iy)*sx + startx + ty] = s_marker[ty][iy];
			if (s_change[iy][ty]) *change = true;
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
	volatile __shared__ bool  s_change[Y_THREADS][Y_THREADS+1];
		

	if (by + ty < sy) {

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
	optimized.  note that sx and sy MUST be multiples of thread block size.
 */
 template <typename T, typename CT>
__global__ void
iRec1DForward_X_dilation ( T* marker, const T* mask, const int sx, const int sy, bool* __restrict__ change )
{
	extern __shared__ char array[];

	const int tx = threadIdx.x;  // unit is CT
	const int ty = threadIdx.y;  // not affect by data type here
	const int by = blockIdx.y * blockDim.y;  // not affected by data type here
	const int numEl = sizeof(CT) / sizeof(T);
	const int tsx = sx / numEl;  // sx unit is T.  tsx unit is CT
	const int blockDim_x = blockDim.x + 1;
	const int s_idx = ty * blockDim_x + tx;
	const int maxx = tsx - blockDim.x;

	// can do 32 y threads, 4 x threads,16 bytes per element, total 4272 bytes, allows 8 threads.
	// do 1 wider because of the overlap
	CT* ts_marker = (CT*)array;  // in units of CT
	CT* ts_mask = ts_marker + blockDim_x * blockDim.y;  // in units of CT
	bool* s_change = (bool*)(ts_mask + blockDim_x * blockDim.y);  // in units of T

	CT* t_marker = (CT*) marker;
	CT* t_mask = (CT*) mask;
	T* s_marker = (T*)ts_marker;
	T* s_mask = (T*)ts_mask;


	if (ty + by < sy) {

//		for (int i = 0; i < numEl; ++i) {
//			s_change[s_idx * numEl + i] = false;
//		}
//		__syncthreads();

		T s_old;
		// the increment allows overlap by 1 between iterations to move the data to next block.
		// because of this overlap, the alignment will be off.  can do some copying - the shared mem needs to be 1 wider.
		// initialize the first column necessary?
		if (tx == 0) {
			for (int i = 0; i < numEl; ++i){
				s_marker[s_idx * numEl + i] = (T)0;
				s_mask[s_idx * numEl + i] = (T)0;
			}
		}

		for (int startx = 0; startx < maxx; startx += blockDim.x) {
			t_marker = ((CT*)marker) + (by * tsx + startx);
			t_mask = ((CT*)mask) + (by * tsx + startx);
			// copy part of marker and mask to shared memory, offset = 1
			ts_marker[s_idx + 1] = t_marker[ty * tsx + tx];
			ts_mask  [s_idx + 1] = t_mask  [ty * tsx + tx];
			__syncthreads();

//			// perform iteration   all X threads do the same operations, so there may be read/write hazards.  but the output is the same.
//			// this is looping for BLOCK_SIZE times, and each iteration the final results are propagated 1 step closer to tx.
//			for (int ix = 1; ix < s_step; ix++) {
//				s_old = s_marker[ty * s_step + ix];
//				s_marker[ty * s_step + ix] = max( s_marker[ty * s_step + ix], s_marker[ty * s_step + ix-1] );
//				s_marker[ty * s_step + ix] = min( s_marker[ty * s_step + ix], s_mask  [ty * s_step + ix] );
//				s_change[ty * s_step + ix] |= NEQ( s_old, s_marker[ty * s_step + ix] );
//			}
//			__syncthreads();

			// output result back to global memory
			t_marker[ty * tsx + tx] = ts_marker[s_idx + 1];
			__syncthreads();

			if (tx == 0) {
				// bank conflict here - no.  read into register, write to target address.
				ts_marker[ty * blockDim_x] = ts_marker[(ty + 1) * blockDim_x - 1 ];
				ts_mask[ty * blockDim_x] = ts_mask[(ty + 1) * blockDim_x - 1 ];

			}
		}

		t_marker = ((CT*)marker) + (by * tsx + maxx);
		t_mask = ((CT*)mask) + (by * tsx + maxx);
		// copy part of marker and mask to shared memory
		ts_marker[s_idx + 1] = t_marker[ty * tsx + tx];
		ts_mask  [s_idx + 1] = t_mask  [ty * tsx + tx];
		__syncthreads();

//		// perform iteration
//		for (int ix = numEl; ix < s_step; ix++) {
//			s_old = s_marker[ty * s_step + ix];
//			s_marker[ty * s_step + ix] = max( s_marker[ty * s_step + ix], s_marker[ty * s_step + ix-1] );
//			s_marker[ty * s_step + ix] = min( s_marker[ty * s_step + ix], s_mask  [ty * s_step + ix] );
//			s_change[ty * s_step + ix] |= NEQ( s_old, s_marker[ty * s_step + ix] );
//		}
//		__syncthreads();

		// output result back to global memory
		t_marker[ty * tsx + tx] = ts_marker[s_idx + 1];
//		for (int i = 0; i < numEl; ++i) {
//			if (s_change[s_idx * numEl + i]) *change = true;
//		}
		__syncthreads();

	}


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
iRec1DBackward_Y_dilation ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
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
iRec1DForward_Y_dilation_8 ( T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy, bool* __restrict__ change )
{
	// parallelize along x.
	extern __shared__ char array[];

	const int tx = threadIdx.x;
	const int bx = blockIdx.x * blockDim.x;
	const int numEl = sizeof(uint2) / sizeof(T);
	const int s_step = blockDim.x * numEl;
	const int tsx = sx / numEl;


	// can do 128 x threads,4 bytes per element, total 4272 bytes, allows 8 threads.
	T* s_marker_A = (T*)array;
	T* s_marker_B = (T*)(array + s_step * sizeof(T));
	T* s_mask = (T*)(array + s_step * sizeof(T) * 2);
	bool* s_change = (bool*)(array + s_step * sizeof(T) * 3);

	uint2* ts_marker_B = (uint2*)s_marker_B;
	uint2* ts_mask = (uint2*)s_mask;
	uint2* t_marker = (uint2*)marker;
	uint2* t_mask = (uint2*)mask;
	
	if ( bx+tx < tsx + 1 ) {
		
		s_change[tx] = false;
		ts_marker_B[tx] = t_marker[bx+tx];
		__syncthreads();

		T s_old;
		for (int ty = 1; ty < sy; ty++) {
		
			// copy part of marker and mask to shared memory
			for ( int i = 0; i < numEl; i ++) {
				s_marker_A[tx*numEl + i] = s_marker_B[tx*numEl + i];
				if ((bx + tx)*numEl+i > 0) s_marker_A[tx*numEl+i] = max((tx*numEl+i == 0) ? marker[(ty-1) * sx + (bx + tx)*numEl + i - 1] : s_marker_B[tx*numEl+i-1], s_marker_A[tx*numEl+i]);
				if ((bx+tx)*numEl+i < sx-1) s_marker_A[tx*numEl+i] = max((tx*numEl+i == s_step-1) ? marker[(ty-1) * sx + (bx + tx)*numEl + i + 1] : s_marker_B[tx*numEl+i+1], s_marker_A[tx*numEl+i]);
			}
			ts_mask    [tx] = t_mask[ty * tsx + bx + tx];
			//__syncthreads();
			//s_old = marker[ty * tsx + bx + tx];
			ts_marker_B[tx] = t_marker[ty * tsx + bx + tx];
			__syncthreads();

			// perform iteration
			for (int i = 0; i < numEl; i++) {
				s_old = s_marker_B[tx*numEl + i];
				s_marker_B[tx*numEl+i] = max( s_marker_A[tx*numEl+i], s_marker_B[tx*numEl+i] );
			//s_marker_B[tx] = max( s_marker_A[tx], s_old );
				s_marker_B[tx*numEl+i] = min( s_marker_B[tx*numEl+i], s_mask    [tx*numEl+i] );
				s_change[tx*numEl+i] |= NEQ( s_old, s_marker_B[tx*numEl+i] );
				// output result back to global memory
			}
			t_marker[ty * tsx + bx + tx] = ts_marker_B[tx];
			__syncthreads();
		}

		for (int i = 0; i < numEl; i++) {
			if (s_change[tx * numEl + i]) *change = true;
		}
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
	volatile __shared__ bool  s_change  [MAX_THREADS];

	if ( bx + tx < sx ) {

		
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
	unsigned int imreconstructIntCaller(T* __restrict__ marker, const T* __restrict__ mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream) {

		// here because we are not using streams inside.
//		if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
//		else cudaSafeCall( cudaStreamSynchronize(stream));


		printf("entering imrecon int caller with conn=%d\n", connectivity);

		// setup execution parameters
		bool conn8 = (connectivity == 8);

		dim3 threadsx2( Y_THREADS );
		dim3 blocksx2( divUp(sy, threadsx2.y) );
		dim3 threadsy( MAX_THREADS );
		dim3 blocksy( divUp(sx, threadsy.x) );
		dim3 threadsy2( 192 );
		dim3 blocksy2( divUp(sx, threadsy2.x * sizeof(uint2)) );
		size_t Nsy = threadsy2.x * (sizeof(uint2) * 3 + sizeof(uint2) * sizeof(bool) / sizeof(T));

		dim3 threadsx( 32, 12 );
		dim3 blocksx( 1, divUp(sy, threadsx.y) );
		size_t Nsx = (threadsx.x+1) * threadsx.y * (sizeof(uchar4) * 2 + sizeof(bool) * sizeof(uchar4) /sizeof(T));

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
				iRec1DForward_X_dilation<T, uchar4> <<< blocksx, threadsx, Nsx, stream >>> ( marker, mask, sx, sy, d_change );
//				iRec1DForward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// dopredny pruchod pres osu Y
//				iRec1DForward_Y_dilation_8<<< blocksy2, threadsy2, Nsy, stream >>> ( marker, mask, sx, sy, d_change );
/*
				// zpetny pruchod pres osu X
				//iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DBackward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				iRec1DBackward_Y_dilation_8<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );
*/
				if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
//				else cudaSafeCall( cudaStreamSynchronize(stream));
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
				iRec1DForward_X_dilation<T, uchar4> <<< blocksx, threadsx, Nsx, stream >>> ( marker, mask, sx, sy, d_change );
				//iRec1DForward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

/*				// dopredny pruchod pres osu Y
				iRec1DForward_Y_dilation <<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu X
				//iRec1DBackward_X_dilation<<< blocksx, threadsx, 0, stream >>> ( marker, mask, sx, sy, d_change );
				iRec1DBackward_X_dilation2<<< blocksx2, threadsx2, 0, stream >>> ( marker, mask, sx, sy, d_change );

				// zpetny pruchod pres osu Y
				iRec1DBackward_Y_dilation<<< blocksy, threadsy, 0, stream >>> ( marker, mask, sx, sy, d_change );
*/
				if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
//				else cudaSafeCall( cudaStreamSynchronize(stream));
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
