#include "hist-ops.cuh"
#include <limits>


#define HIST_BINS				256
#define THREAD_N_BLOCK_INTENSITY		32
#define N_INTENSITY_FEATURES			8

namespace nscale {

namespace gpu {

using namespace cv::gpu;



__global__ void xDerivativeKernel(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> input, cv::gpu::PtrStep_<float> output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < rows && x < cols){
		if(x==0){
			output.ptr(y)[x] = (float)(input.ptr(y)[x+1] - input.ptr(y)[x]);
		}else if(x==cols-1){
			output.ptr(y)[x] = (float)(input.ptr(y)[x] - input.ptr(y)[x-1]);
		}else{
			output.ptr(y)[x] = (float)(input.ptr(y)[x+1] - input.ptr(y)[x-1])/2.0;
		}
		output.ptr(y)[x] = output.ptr(y)[x] * output.ptr(y)[x];
	}
}




// build a histogram for each component in the image described using bbInfo
void xDerivativeSquareCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> input, cv::gpu::PtrStep_<float> output, cudaStream_t stream)
{

	dim3 threads(16, 16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	xDerivativeKernel<<<grid, threads, 0, stream>>>(rows, cols, input, output);

	 cudaGetLastError();

	if (stream == 0)
        	cudaDeviceSynchronize();


}


__global__ void yDerivativeKernel(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> input, cv::gpu::PtrStep_<float> output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float derivative = 0.0;
	if (y < rows && x < cols){
		if(y==0){
			derivative = (float)(input.ptr(y+1)[x] - input.ptr(y)[x]);
		}else if(y==cols-1){
			derivative = (float)(input.ptr(y)[x] - input.ptr(y-1)[x]);
		}else{
			derivative = (float)(input.ptr(y+1)[x] - input.ptr(y-1)[x])/2.0;
		}
		output.ptr(y)[x] += derivative * derivative;
		output.ptr(y)[x] = sqrt(output.ptr(y)[x]);
	}
}

// build a histogram for each component in the image described using bbInfo
void yDerivativeSquareAccCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> input, cv::gpu::PtrStep_<float> output, cudaStream_t stream)
{

	dim3 threads(16, 16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	yDerivativeKernel<<<grid, threads, 0, stream>>>(rows, cols, input, output);

	 cudaGetLastError();

	if (stream == 0)
        	cudaDeviceSynchronize();


}

// Assuming n > 0
int rndint(float n)//round float to the nearest integer
{
	int ret = (int)floor(n);
	float t;
	t=n-floor(n);
	if (t>=0.5)
	{
		ret = (int)floor(n) + 1;
	}
	return ret;
}


__global__ void floatToUcharKernel(int rows, int cols, const cv::gpu::PtrStep_<float> input, cv::gpu::PtrStep_<unsigned char> output)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (y < rows && x < cols){
		int value = (int)floor(input.ptr(y)[x]);
		float t = input.ptr(y)[x] - value;
		if(t>=0.5){
			value += 1;
		}
		output.ptr(y)[x] = (unsigned char)value;
	}
}


// convert float image to unsigned char
void floatToUcharCaller(int rows, int cols, const cv::gpu::PtrStep_<float> input, const cv::gpu::PtrStep_<unsigned char> output, cudaStream_t stream){
	dim3 threads(16, 16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	floatToUcharKernel<<<grid, threads, 0, stream>>>(rows, cols, input, output);

	 cudaGetLastError();

	if (stream == 0)
        	cudaDeviceSynchronize();

}


}}
