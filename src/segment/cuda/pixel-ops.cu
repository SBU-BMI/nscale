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
__global__ void convLoop1Kernel(int rows, int cols, const PtrStep_<uchar> img1, PtrStep_<double> result)
{
	__shared__ double precomp[256];
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;

	for(int i=tid_in_block; i < 256; i+= blockDim.x*blockDim.y){
		precomp[i] = -(255.0*log(((double)i +1.0)/255.0))/log(255.0);;
	}
	if (y < rows && x < cols)
	{
		result.ptr(y)[x] = precomp[img1.ptr(y)[x]];
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

void convLoop1(int rows, int cols, int cn, const PtrStep_<uchar> img1,
 	PtrStep_<double> result, cudaStream_t stream)
{
   	dim3 threads(16, 16);
	dim3 grid(divUp(cols * cn, threads.x), divUp(rows, threads.y));

	convLoop1Kernel<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result);

	cudaSafeCall( cudaGetLastError() );

	if (stream == 0)
        	cudaSafeCall(cudaDeviceSynchronize());
}



__global__ void convLoop2Kernel(int rows, int cols, int cn_channels, PtrStep_<double> g_cn,
 	int dn_channels, PtrStep_<double> g_dn, PtrStep_<double> g_Q, int Q_rows, bool BGR2RGB)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < rows && x < cols)
	{
		// Zero g_cn;
		for(int i=0; i <cn_channels; i++ ){
			g_cn.ptr(y)[x *cn_channels + i] = 0.0;
		}
		for(int k=0; k < dn_channels; k++){
			for(int Q_i=0; Q_i < Q_rows; Q_i++){
				if( BGR2RGB ){
					g_cn.ptr(y)[x * cn_channels + Q_i] += g_Q.ptr(Q_i)[k] * g_dn.ptr(y)[ x * dn_channels + dn_channels-1-k];
				}else{
					g_cn.ptr(y)[x * cn_channels + Q_i] += g_Q.ptr(Q_i)[k] * g_dn.ptr(y)[ x * dn_channels + k];
				}
			}
		}
	}
}


void convLoop2(int rows, int cols, int cn_channels, const PtrStep_<double> g_cn,
 	int dn_channels, PtrStep_<double> g_dn, PtrStep_<double> g_Q, int Q_rows, bool BGR2RGB, cudaStream_t stream)
{
   	dim3 threads(16, 16);
	dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

	convLoop2Kernel<<<grid, threads, 0, stream>>>(rows, cols, cn_channels, g_cn, dn_channels, g_dn, g_Q, Q_rows, BGR2RGB);

	cudaSafeCall( cudaGetLastError() );

	if (stream == 0)
        	cudaSafeCall(cudaDeviceSynchronize());
}

__device__ unsigned char double2char(double d){
	double truncate = min( max(d, (double )0.0), (double)255.0);
	double pt;
	double c = modf(truncate, &pt) >= .5?ceil(truncate):floor(truncate);
	return (unsigned char) c;
}


__global__ void convLoop3Kernel(int rows, int cols, int cn_channels, const PtrStep_<double> g_cn,
 	PtrStep_<uchar> g_E, PtrStep_<uchar> g_H)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	double log255div255 = log(255.0)/255.0;

	if (y < rows && x < cols)
	{
		double temp = exp(-(g_cn.ptr(y)[ x * cn_channels]-255.0)*log255div255);
		g_H.ptr(y)[x] = double2char(temp);

		temp = exp(-(g_cn.ptr(y)[ x * cn_channels + 1]-255.0)*log255div255);
		g_E.ptr(y)[x] = double2char(temp);

	}
}

void convLoop3(int rows, int cols, int cn_channels, const PtrStep_<double> g_cn,
 	PtrStep_<uchar> g_E, PtrStep_<uchar> g_H, cudaStream_t stream)
{
   	dim3 threads(16, 16);
	dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

	convLoop3Kernel<<<grid, threads, 0, stream>>>(rows, cols, cn_channels, g_cn, g_E, g_H);

	cudaSafeCall( cudaGetLastError() );

	if (stream == 0)
        	cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void bgr2grayKernel(int rows, int cols, const PtrStep_<uchar> img,
 	PtrStep_<uchar> result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Same constants as used by Matlab
	double r_const = 0.298936021293776;
	double g_const = 0.587043074451121;
	double b_const = 0.114020904255103;

	if (y < rows && x < cols)
	{
		uchar b = img.ptr(y)[ x * 3];
		uchar g = img.ptr(y)[ x * 3 + 1];
		uchar r = img.ptr(y)[ x * 3 + 2];
		double grayPixelValue =  r_const * (double)r + g_const * (double)g + b_const * (double)b;
		result.ptr(y)[x] = double2char(grayPixelValue);
	}
}
void bgr2grayCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> img,  
	cv::gpu::PtrStep_<unsigned char> result, cudaStream_t stream)
{
	dim3 threads(16, 16);
	dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

	bgr2grayKernel<<<grid, threads, 0, stream>>>(rows, cols, img, result);

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




__global__ void intToCharKernel(int rows, int cols, int *input, unsigned char *result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * cols + x;

	if (y < rows && x < cols)
	{
		result[index] = (unsigned char)input[index];
	}
}


void convertIntToChar(int rows, int cols, int *input, unsigned char *result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

    intToCharKernel<<<grid, threads, 0, stream>>>(rows, cols, input, result);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void intToCharBorderKernel(int rows, int cols, int top, int bottom, int left, int right, int *input, unsigned char *result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// only threads within the input image - border
	if (y < (rows-bottom) && y >= top && x < (cols-right) && x >= (left) )
	{
		int input_index = y * cols + x;
		// for the resuting image we must shift the pixels according to the border size
		int result_index = (y-top) * (cols-left-right) + (x-left);
		result[result_index] = (unsigned char)input[input_index];
	}
}


void convertIntToCharAndRemoveBorder(int rows, int cols, int top, int bottom, int left, int right, int *input, unsigned char *result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));

    intToCharBorderKernel<<<grid, threads, 0, stream>>>(rows, cols, top, bottom, left, right, input, result);
    cudaSafeCall( cudaGetLastError() );

    if (stream == 0)
        cudaSafeCall(cudaDeviceSynchronize());
}


template void thresholdCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char, unsigned char, cudaStream_t);
template void thresholdCaller<float>(int, int, const PtrStep_<float>, PtrStep_<unsigned char>, float, float, cudaStream_t);

}}
