// adaptation of Pavel's imreconstruction code for openCV

#include "neighbor-ops.cuh"


namespace nscale {

namespace gpu {

using namespace cv::gpu;


template <typename T>
__global__ void borderKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result, T background, int connectivity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // treat this as a specialized erosion
    if (y > 0 && y < (rows-1) && x > 0 && x < (cols-1))
    {
    	T p = img1.ptr(y)[x];
    	T output = p;
	// if p is already background, this will not change it so run it through
	T q1, q2;
//	T q3, q4;

	q1 = img1.ptr(y-1)[x];
	q2 = img1.ptr(y)[x-1];
//	q3 = img1.ptr(y)[x+1];
//	q4 = img1.ptr(y+1)[x];
	if ((q1 != background && p!=q1) ||
	 (q2 != background && p!=q2)) // || 
//	 (q3 != background && p!=q3) ||
//	 (q4 != background && p!=q4))
		 output = background;
		
	if (connectivity == 8) {
	
		q1 = img1.ptr(y-1)[x-1];
		q2 = img1.ptr(y-1)[x+1];
//		q3 = img1.ptr(y+1)[x-1];
//		q4 = img1.ptr(y+1)[x+1];
		if ((q1 != background && p!=q1) ||
		 (q2 != background && p!=q2)) // || 
//		 (q3 != background && p!=q3) ||
//		 (q4 != background && p!=q4))
			 output = background;
	}

    	result.ptr(y)[x] = output;
    }
}

template <typename T>
void borderCaller(int rows, int cols, const PtrStep_<T> img1,
 PtrStep_<T> result, T background, int connectivity, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols + threads.x -1) / threads.x, (rows + threads.y - 1) / threads.y);

    borderKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, background, connectivity);
    cudaGetLastError();

    if (stream == 0)
        cudaDeviceSynchronize();
}

template void borderCaller<int>(int, int, const PtrStep_<int>, PtrStep_<int>, int, int, cudaStream_t);





}}
