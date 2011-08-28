// adaptation of Pavel's imreconstruction code for openCV

#include "internal_shared.hpp"
#include "change_kernel.cuh"

#define BLOCK_SIZE			  8
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
Rec1DForward_X_dilation ( DevMem2D_<T> g_marker, const DevMem2D_<T> g_mask, int connectivity, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * BLOCK_SIZE;
	// always 0.  const int bz = blockIdx.y;
	const int sx = g_marker.cols;
	const int sy = g_marker.rows;

	if (ty + by < sy) {

		__shared__ T s_marker[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ T s_mask  [BLOCK_SIZE][BLOCK_SIZE];
		__shared__ bool  s_change[BLOCK_SIZE][BLOCK_SIZE];
		int startx;
		int offset = (by + ty) * sx + tx;
		T* marker = g_marker.ptr(offset);
		T* mask = g_mask.ptr(offset);
		s_change[tx][ty] = false;
		__syncthreads();

		T s_old;
		// the increment allows overlap by 1 between iterations to move the data to next block.
		for (startx = 0; startx < sx - BLOCK_SIZE; startx += BLOCK_SIZE - 1) {

			// copy part of marker and mask to shared memory
			s_marker[tx][ty] = marker[startx];
			s_mask  [tx][ty] = mask  [startx];
			__syncthreads();

			// perform iteration   // TODO: this is not using tx, only ty.
			for (int ix = 1; ix < BLOCK_SIZE; ix++) {
				s_old = s_marker[ix][ty];
				s_marker[ix][ty] = fmaxf( s_marker[ix][ty], s_marker[ix-1][ty] );
				s_marker[ix][ty] = fminf( s_marker[ix][ty], s_mask  [ix]  [ty] );
				s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
				__syncthreads();
			}

			// output result back to global memory
			marker[startx] = s_marker[tx][ty];
			__syncthreads();

		}

		startx = sx - BLOCK_SIZE;

		// copy part of marker and mask to shared memory
		s_marker[tx][ty] = marker[ startx ];
		s_mask  [tx][ty] = mask  [ startx ];
		__syncthreads();

		// perform iteration
		for (int ix = 1; ix < BLOCK_SIZE; ix++) {
			s_old = s_marker[ix][ty];
			s_marker[ix][ty] = fmaxf( s_marker[ix][ty], s_marker[ix-1][ty] );
			s_marker[ix][ty] = fminf( s_marker[ix][ty], s_mask  [ix]  [ty] );
			s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
			__syncthreads();
		}

		// output result back to global memory
		marker[ startx ] = s_marker[tx][ty];
		__syncthreads();

		if (s_change[tx][ty]) *change = true;
		__syncthreads();

	}

}

__global__ void
Rec1DForward_Y_dilation ( DevMem2D_<T> g_marker, const DevMem2D_<T> g_mask, int connectivity, bool* change )
{

	const int tx = threadIdx.x;
	const int tz = threadIdx.y;  // z should be 1...
	const int bx = blockIdx.x * BLOCK_SIZE;
//	const int bz = blockIdx.y * BLOCK_SIZE;
	const int sx = g_marker.cols;
	const int sy = g_marker.rows;


	if ( ((bx + tx) < sx) && ((bz + tz) < sz) ) {

		__shared__ float s_marker_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float s_marker_B[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float s_mask    [BLOCK_SIZE][BLOCK_SIZE];
		__shared__ bool  s_change  [BLOCK_SIZE][BLOCK_SIZE];
		s_change[tx][tz] = false;

		s_marker_B[tx][tz] = g_marker[ ((bz + tz) * sy + 0) * sx + (bx + tx) ];
		__syncthreads();

		for (int ty = 1; ty < sy; ty++) {

			// copy part of marker and mask to shared memory
			s_marker_A[tx][tz] = s_marker_B[tx][tz];
			s_marker_B[tx][tz] = g_marker[ ((bz + tz) * sy + ty) * sx + (bx + tx) ];
			s_mask    [tx][tz] = g_mask  [ ((bz + tz) * sy + ty) * sx + (bx + tx) ];
			__syncthreads();

			// perform iteration
			float s_old = s_marker_B[tx][tz];
			s_marker_B[tx][tz] = fmaxf( s_marker_A[tx][tz], s_marker_B[tx][tz] );
			s_marker_B[tx][tz] = fminf( s_marker_B[tx][tz], s_mask    [tx][tz] );
			s_change[tx][tz] |= NEQ( s_old, s_marker_B[tx][tz] );
			__syncthreads();

			// output result back to global memory
			g_marker[ ((bz + tz) * sy + ty) * sx + (bx + tx) ] = s_marker_B[tx][tz];
			__syncthreads();

		}

		if (s_change[tx][tz]) *change = true;
		__syncthreads();

	}

}

__global__ void
Rec1DBackward_X_dilation ( DevMem2D_<T> g_marker, const DevMem2D_<T> g_mask, int connectivity, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * BLOCK_SIZE;
	const int bz = blockIdx.y;

		if (by + ty < sy) {

			int startx;
			__shared__ float s_marker[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ float s_mask  [BLOCK_SIZE][BLOCK_SIZE];
			__shared__ bool  s_change[BLOCK_SIZE][BLOCK_SIZE];
			s_change[tx][ty] = false;
			__syncthreads();

			for (int startx = sx - BLOCK_SIZE; startx > 0; startx -= BLOCK_SIZE - 1) {

				// copy part of marker and mask to shared memory
				s_marker[tx][ty] = g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				s_mask  [tx][ty] = g_mask  [ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				__syncthreads();

				// perform iteration
				for (int ix = BLOCK_SIZE - 2; ix >= 0; ix--) {
					float s_old = s_marker[ix][ty];
					s_marker[ix][ty] = fmaxf( s_marker[ix][ty], s_marker[ix+1][ty] );
					s_marker[ix][ty] = fminf( s_marker[ix][ty], s_mask  [ix]  [ty] );
					s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
					__syncthreads();
				}

				// output result back to global memory
				g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ] = s_marker[tx][ty];
				__syncthreads();

			}

				startx = 0;

				// copy part of marker and mask to shared memory
				s_marker[tx][ty] = g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				s_mask  [tx][ty] = g_mask  [ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				__syncthreads();

				// perform iteration
				for (int ix = BLOCK_SIZE - 2; ix >= 0; ix--) {
					float s_old = s_marker[ix][ty];
					s_marker[ix][ty] = fmaxf( s_marker[ix][ty], s_marker[ix+1][ty] );
					s_marker[ix][ty] = fminf( s_marker[ix][ty], s_mask  [ix]  [ty] );
					s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
					__syncthreads();
				}

				// output result back to global memory
				g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ] = s_marker[tx][ty];
				__syncthreads();

				if (s_change[tx][ty]) *change = true;
				__syncthreads();

		}

}

__global__ void
Rec1DBackward_Y_dilation ( DevMem2D_<T> g_marker, const DevMem2D_<T> g_mask, int connectivity, bool* change )
{

	const int tx = threadIdx.x;
	const int tz = threadIdx.y;
	const int bx = blockIdx.x * BLOCK_SIZE;
	const int bz = blockIdx.y * BLOCK_SIZE;

	if ( ((bx + tx) < sx) && ((bz + tz) < sz) ) {

		__shared__ float s_marker_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float s_marker_B[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float s_mask    [BLOCK_SIZE][BLOCK_SIZE];
		__shared__ bool  s_change  [BLOCK_SIZE][BLOCK_SIZE];
		s_change[tx][tz] = false;

		s_marker_B[tx][tz] = g_marker[ ((bz + tz) * sy + sy - 1) * sx + (bx + tx) ];
		__syncthreads();

		for (int ty = sy - 2; ty >= 0; ty--) {

			// copy part of marker and mask to shared memory
			s_marker_A[tx][tz] = s_marker_B[tx][tz];
			s_marker_B[tx][tz] = g_marker[ ((bz + tz) * sy + ty) * sx + (bx + tx) ];
			s_mask    [tx][tz] = g_mask  [ ((bz + tz) * sy + ty) * sx + (bx + tx) ];
			__syncthreads();

			// perform iteration
			float s_old = s_marker_B[tx][tz];
			s_marker_B[tx][tz] = fmaxf( s_marker_A[tx][tz], s_marker_B[tx][tz] );
			s_marker_B[tx][tz] = fminf( s_marker_B[tx][tz], s_mask    [tx][tz] );
			s_change[tx][tz] |= NEQ( s_old, s_marker_B[tx][tz] );
			__syncthreads();

			// output result back to global memory
			g_marker[ ((bz + tz) * sy + ty) * sx + (bx + tx) ] = s_marker_B[tx][tz];
			__syncthreads();

		}

		if (s_change[tx][tz]) *change = true;
		__syncthreads();

	}

}



	template <typename T>
	void imreconstructCaller(DevMem2D_<T> marker, const DevMem2D_<T> mask,
		int connectivity, cudaStream_t stream) {

		// setup execution parameters
		int sx = marker.cols;
		int sy = marker.rows;
		int bx = sx/BLOCK_SIZE + ( (sx % BLOCK_SIZE == 0) ? 0 : 1 );
		int by = sy/BLOCK_SIZE + ( (sy % BLOCK_SIZE == 0) ? 0 : 1 );
		int bz = 1;
		int sz = 1;
		dim3 blocksx( by, sz );
		dim3 blocksy( bx, bz );
		dim3 threads( BLOCK_SIZE, BLOCK_SIZE );

		// stability detection
		unsigned int iter = 0;
		bool *h_change, *d_change;
		h_change = (bool*) malloc( sizeof(bool) );
		cudaSafeCall( cudaMalloc( (void**) &d_change, sizeof(bool) ) );
		
		*h_change = true;

		while ( (*h_change) && (iter < 100000) )  // repeat until stability
		{
			iter++;
			*h_change = false;
			init_change<<< 1, 1, 0, stream>>>( d_change );

			// dopredny pruchod pres osu X
			Rec1DForward_X_dilation <<< blocksx, threads, 0, stream >>> ( marker, mask, connectivity, d_change );

			// dopredny pruchod pres osu Y
			Rec1DForward_Y_dilation <<< blocksy, threads, 0, stream >>> ( marker, mask, connectivity, d_change );

			// zpetny pruchod pres osu X
			Rec1DBackward_X_dilation<<< blocksx, threads, 0, stream >>> ( marker, mask, connectivity, d_change );

			// zpetny pruchod pres osu Y
			Rec1DBackward_Y_dilation<<< blocksy, threads, 0, stream >>> ( marker, mask, connectivity, d_change );

			if (stream == 0) cudaSafeCall(cudaDeviceSynchronize());
			else cudaSafeCall( cudaStreamSynchronize(stream));

			cudaSafeCall( cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) );

		}

		cudaSafeCall( cudaFree(d_change) );
		free(h_change);

		printf("Number of iterations: %d\n", iter);
		cudaSafeCall( cudaGetLastError());

	}

	template void imreconstructCaller<uchar>(const PtrStep, const PtrStep, PtrStep, int, cudaStream_t stream);
	template void imreconstructCaller<float>(const PtrStep, const PtrStep, PtrStep, int, cudaStream_t stream);
}}
