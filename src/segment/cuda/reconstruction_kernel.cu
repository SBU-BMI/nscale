/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
 * and the reconstruction by erosion (reconstruction_by_erosion_kernel)
 *
 * auxiliary kernels (Rec1DForward_X_dilation, Rec1DForward_Y_dilation, etc.)
 * perform particular scans in X, Y, and Z axis in forward and backward direction, respectively
 *
 */
#include "internal_shared.hpp"
#include "change_kernel.cuh"

#define BLOCK_SIZE			  16
#define NEQ(a,b)    ( (a) != (b) )

using namespace cv::gpu;


namespace nscale { namespace gpu {



////////////////////////////////////////////////////////////////////////////////
// RECONSTRUCTION BY DILATION
////////////////////////////////////////////////////////////////////////////////

__global__ void
Rec1DForward_X_dilation ( unsigned char* g_marker, const unsigned char* g_mask, int sx, int sy, int sz, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * BLOCK_SIZE;
	const int bz = blockIdx.y;
	
		if (ty + by < sy) {

			int startx;
			__shared__ unsigned char s_marker[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ unsigned char s_mask  [BLOCK_SIZE][BLOCK_SIZE];
			__shared__ bool  s_change[BLOCK_SIZE][BLOCK_SIZE];
			s_change[tx][ty] = false;
			__syncthreads();
			
			for (startx = 0; startx < sx - BLOCK_SIZE; startx += BLOCK_SIZE - 1) {
		
				// copy part of marker and mask to shared memory	
				s_marker[tx][ty] = g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				s_mask  [tx][ty] = g_mask  [ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				__syncthreads();
	
				// perform iteration
				for (int ix = 1; ix < BLOCK_SIZE; ix++) {
					unsigned char s_old = s_marker[ix][ty];
					s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
					s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
					s_change[ix][ty] |= NEQ( s_old, s_marker[ix][ty] );
					__syncthreads();
				}
	
				// output result back to global memory
				g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ] = s_marker[tx][ty];
				__syncthreads();
			
			}

				startx = sx - BLOCK_SIZE;
			
				// copy part of marker and mask to shared memory	
				s_marker[tx][ty] = g_marker[ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				s_mask  [tx][ty] = g_mask  [ (bz * sy + (by + ty)) * sx + (startx + tx) ];
				__syncthreads();
	
				// perform iteration
				for (int ix = 1; ix < BLOCK_SIZE; ix++) {
					unsigned char s_old = s_marker[ix][ty];
					s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix-1][ty] );
					s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
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
Rec1DForward_Y_dilation ( unsigned char* g_marker, const unsigned char* g_mask, int sx, int sy, int sz, bool* change )
{

	const int tx = threadIdx.x;
	const int tz = threadIdx.y;
	const int bx = blockIdx.x * BLOCK_SIZE;
	const int bz = blockIdx.y * BLOCK_SIZE;
	
	if ( ((bx + tx) < sx) && ((bz + tz) < sz) ) {
	
		__shared__ unsigned char s_marker_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_marker_B[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_mask    [BLOCK_SIZE][BLOCK_SIZE];
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
			unsigned char s_old = s_marker_B[tx][tz];
			s_marker_B[tx][tz] = max( s_marker_A[tx][tz], s_marker_B[tx][tz] );
			s_marker_B[tx][tz] = min( s_marker_B[tx][tz], s_mask    [tx][tz] );
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
Rec1DForward_Z_dilation ( unsigned char* g_marker, const unsigned char* g_mask, int sx, int sy, int sz, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x * BLOCK_SIZE;
	const int by = blockIdx.y * BLOCK_SIZE;
	
	if ( ((bx + tx) < sx) && ((by + ty) < sy) ) {
	
		__shared__ unsigned char s_marker_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_marker_B[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_mask    [BLOCK_SIZE][BLOCK_SIZE];
		__shared__ bool  s_change  [BLOCK_SIZE][BLOCK_SIZE];
		s_change[tx][ty] = false;

		s_marker_B[tx][ty] = g_marker[ (0 * sy + (by + ty)) * sx + (bx + tx) ];
		__syncthreads();

		for (int tz = 1; tz < sz; tz++) {
		
			// copy part of marker and mask to shared memory	
			s_marker_A[tx][ty] = s_marker_B[tx][ty];
			s_marker_B[tx][ty] = g_marker[ (tz * sy + (by + ty)) * sx + (bx + tx) ];
			s_mask    [tx][ty] = g_mask  [ (tz * sy + (by + ty)) * sx + (bx + tx) ];
			__syncthreads();
	
			// perform iteration
			unsigned char s_old = s_marker_B[tx][ty];
			s_marker_B[tx][ty] = max( s_marker_A[tx][ty], s_marker_B[tx][ty] );
			s_marker_B[tx][ty] = min( s_marker_B[tx][ty], s_mask    [tx][ty] );
			s_change[tx][ty] |= NEQ( s_old, s_marker_B[tx][ty] );
			__syncthreads();
	
			// output result back to global memory
			g_marker[ (tz * sy + (by + ty)) * sx + (bx + tx) ] = s_marker_B[tx][ty];
			__syncthreads();
			
		}
		
		if (s_change[tx][ty]) *change = true;
		__syncthreads();
		
	}

}

__global__ void
Rec1DBackward_X_dilation ( unsigned char* g_marker, const unsigned char* g_mask, int sx, int sy, int sz, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int by = blockIdx.x * BLOCK_SIZE;
	const int bz = blockIdx.y;
		
		if (by + ty < sy) {
	
			int startx;
			__shared__ unsigned char s_marker[BLOCK_SIZE][BLOCK_SIZE];
			__shared__ unsigned char s_mask  [BLOCK_SIZE][BLOCK_SIZE];
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
					unsigned char s_old = s_marker[ix][ty];
					s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
					s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
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
					unsigned char s_old = s_marker[ix][ty];
					s_marker[ix][ty] = max( s_marker[ix][ty], s_marker[ix+1][ty] );
					s_marker[ix][ty] = min( s_marker[ix][ty], s_mask  [ix]  [ty] );
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
Rec1DBackward_Y_dilation ( unsigned char* g_marker, const unsigned char* g_mask, int sx, int sy, int sz, bool* change )
{

	const int tx = threadIdx.x;
	const int tz = threadIdx.y;
	const int bx = blockIdx.x * BLOCK_SIZE;
	const int bz = blockIdx.y * BLOCK_SIZE;
	
	if ( ((bx + tx) < sx) && ((bz + tz) < sz) ) {
	
		__shared__ unsigned char s_marker_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_marker_B[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_mask    [BLOCK_SIZE][BLOCK_SIZE];
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
			unsigned char s_old = s_marker_B[tx][tz];
			s_marker_B[tx][tz] = max( s_marker_A[tx][tz], s_marker_B[tx][tz] );
			s_marker_B[tx][tz] = min( s_marker_B[tx][tz], s_mask    [tx][tz] );
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
Rec1DBackward_Z_dilation ( unsigned char* g_marker, const unsigned char* g_mask, int sx, int sy, int sz, bool* change )
{

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x * BLOCK_SIZE;
	const int by = blockIdx.y * BLOCK_SIZE;
	
	if ( ((bx + tx) < sx) && ((by + ty) < sy) ) {
	
		__shared__ unsigned char s_marker_A[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_marker_B[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ unsigned char s_mask    [BLOCK_SIZE][BLOCK_SIZE];
		__shared__ bool  s_change  [BLOCK_SIZE][BLOCK_SIZE];
		s_change[tx][ty] = false;

		s_marker_B[tx][ty] = g_marker[ ((sz - 1) * sy + (by + ty)) * sx + (bx + tx) ];
		__syncthreads();

		for (int tz = sz - 2; tz >= 0; tz--) {
		
			// copy part of marker and mask to shared memory	
			s_marker_A[tx][ty] = s_marker_B[tx][ty];
			s_marker_B[tx][ty] = g_marker[ (tz * sy + (by + ty)) * sx + (bx + tx) ];
			s_mask    [tx][ty] = g_mask  [ (tz * sy + (by + ty)) * sx + (bx + tx) ];
			__syncthreads();
	
			// perform iteration
			unsigned char s_old = s_marker_B[tx][ty];
			s_marker_B[tx][ty] = max( s_marker_A[tx][ty], s_marker_B[tx][ty] );
			s_marker_B[tx][ty] = min( s_marker_B[tx][ty], s_mask    [tx][ty] );
			s_change[tx][ty] |= NEQ( s_old, s_marker_B[tx][ty] );
			__syncthreads();
	
			// output result back to global memory
			g_marker[ (tz * sy + (by + ty)) * sx + (bx + tx) ] = s_marker_B[tx][ty];
			__syncthreads();
			
		}
		
		if (s_change[tx][ty]) *change = true;
		__syncthreads();

	}

}

unsigned int
reconstruction_by_dilation_kernel( unsigned char* g_marker, const unsigned char* g_mask, const int sx, const int sy, const int sz )
{

	// setup execution parameters
	int bx = sx/BLOCK_SIZE + ( (sx % BLOCK_SIZE == 0) ? 0 : 1 );
	int by = sy/BLOCK_SIZE + ( (sy % BLOCK_SIZE == 0) ? 0 : 1 );
	int bz = sz/BLOCK_SIZE + ( (sz % BLOCK_SIZE == 0) ? 0 : 1 );
	dim3 blocksx( by, sz );
	dim3 blocksy( bx, bz );
	dim3 blocksz( bx, by );
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
		init_change<<< 1, 1 >>>( d_change );

		// dopredny pruchod pres osu X
		Rec1DForward_X_dilation <<< blocksx, threads >>> ( g_marker, g_mask, sx, sy, sz, d_change );

		// dopredny pruchod pres osu Y
		Rec1DForward_Y_dilation <<< blocksy, threads >>> ( g_marker, g_mask, sx, sy, sz, d_change );

//		// dopredny pruchod pres osu Z
//		Rec1DForward_Z_dilation <<< blocksz, threads >>> ( g_marker, g_mask, sx, sy, sz, d_change );
		
		// zpetny pruchod pres osu X
		Rec1DBackward_X_dilation<<< blocksx, threads >>> ( g_marker, g_mask, sx, sy, sz, d_change );

		// zpetny pruchod pres osu Y
		Rec1DBackward_Y_dilation<<< blocksy, threads >>> ( g_marker, g_mask, sx, sy, sz, d_change );

//		// zpetny pruchod pres osu Z
//		Rec1DBackward_Z_dilation<<< blocksz, threads >>> ( g_marker, g_mask, sx, sy, sz, d_change );

		cudaSafeCall( cudaMemcpy( h_change, d_change, sizeof(bool), cudaMemcpyDeviceToHost ) );

	}
	
	cudaSafeCall( cudaFree(d_change) );
	free(h_change);

	printf("Number of iterations: %d\n", iter);
	return iter;
}

}}

