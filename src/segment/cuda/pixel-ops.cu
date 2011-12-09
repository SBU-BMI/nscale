// adaptation of Pavel's imreconstruction code for openCV

#include "pixel-ops.cuh"
#include <limits>
#include "internal_shared.hpp"



namespace nscale {

namespace gpu {

using namespace cv::gpu;

template <typename T>
__global__ void invertKernelFloat(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
        result.ptr(y)[x] = - img1.ptr(y)[x];
    }
}

template <typename T>
__global__ void invertKernelUInt(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result, T max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
        result.ptr(y)[x] = max - img1.ptr(y)[x];
    }
}

template <typename T>
__global__ void invertKernelInt(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
        result.ptr(y)[x] = ~img1.ptr(y)[x] + 1;
    }
}

template <typename T>
void invertUIntCaller(int rows, int cols, int cn, const PtrStep_<T> img1,
 PtrStep_<T> result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols * cn, threads.x), divUp(rows, threads.y));

    invertKernelUInt<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result, std::numeric_limits<T>::max());
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}
template <typename T>
void invertIntCaller(int rows, int cols, int cn, const PtrStep_<T> img1,
 PtrStep_<T> result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols * cn, threads.x), divUp(rows, threads.y));

    invertKernelInt<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}
template <typename T>
void invertFloatCaller(int rows, int cols, int cn, const PtrStep_<T> img1,
 PtrStep_<T> result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols * cn, threads.x), divUp(rows, threads.y));

    invertKernelFloat<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

template void invertUIntCaller<unsigned char>(int, int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, cudaStream_t);
template void invertIntCaller<int>(int, int, int, const PtrStep_<int>, PtrStep_<int>, cudaStream_t);
template void invertFloatCaller<float>(int, int, int, const PtrStep_<float>, PtrStep_<float>, cudaStream_t);




template <typename T>
__global__ void thresholdKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<unsigned char> result, T lower, T upper)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
    	T p = img1.ptr(y)[x];
        result.ptr(y)[x] = (p < lower || p >= upper) ? 0 : 255;
    }
}

template <typename T>
void thresholdCaller(int rows, int cols, const PtrStep_<T> img1,
 PtrStep_<unsigned char> result, T lower, T upper, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

    thresholdKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, lower, upper);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

template void thresholdCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char, unsigned char, cudaStream_t);

template void thresholdCaller<float>(int, int, const PtrStep_<float>, PtrStep_<unsigned char>, float, float, cudaStream_t);
template void thresholdCaller<int>(int, int, const PtrStep_<int>, PtrStep_<unsigned char>, int, int, cudaStream_t);




template <typename T>
__global__ void modKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result, T mod)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
    	T p = img1.ptr(y)[x];
        result.ptr(y)[x] = p % mod;
    }
}

template <typename T>
void modCaller(int rows, int cols, const PtrStep_<T> img1,
 PtrStep_<T> result, T mod, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

    modKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, mod);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

template void modCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char, cudaStream_t);
template void modCaller<int>(int, int, const PtrStep_<int>, PtrStep_<int>, int, cudaStream_t);


}}
