// adaptation of Pavel's imreconstruction code for openCV

#include "neighbor-ops.cuh"
#include <limits>
#include "internal_shared.hpp"



namespace nscale {

namespace gpu {

using namespace cv::gpu;


template <typename T>
__global__ void borderKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result, T background)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y > 0 && y < (rows-1) && x > 0 && x < (cols-1))
    {
    	T p = img1.ptr(y)[x];
    	T output = p;
    	// if it's background, won't change data.  so let it run to avoid branching.
		T p_ym1_xm1 = img1.ptr(y-1)[x-1];
		p_ym1_xm1 = (p_ym1_xm1 == background ? p : p_ym1_xm1);  // if neighbor is a background pixel, consider that to be the same as self.
		T p_ym1_x0 = img1.ptr(y-1)[x];
		p_ym1_x0 = (p_ym1_x0 == background ? p : p_ym1_x0);
//		T p_ym1_xp1 = img1.ptr(y-1)[x+1];   // this thins out the border on the right side of shapes a bit.
//		p_ym1_xp1 = (p_ym1_xp1 == background ? p : p_ym1_xp1);
		T p_y0_xm1 = img1.ptr(y)[x-1];
		p_y0_xm1 = (p_y0_xm1 == background ? p : p_y0_xm1);
		//if (p != p_ym1_xm1 || p != p_ym1_x0 || p != p_ym1_xp1 || p != p_y0_xm1)
		if (p != p_ym1_xm1 || p != p_ym1_x0 || p != p_y0_xm1)
			output = background;

    	result.ptr(y)[x] = output;
    }
}

template <typename T>
void borderCaller(int rows, int cols, const PtrStep_<T> img1,
 PtrStep_<T> result, T background, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

    borderKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, background);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

template void borderCaller<int>(int, int, const PtrStep_<int>, PtrStep_<int>, int, cudaStream_t);





}}
