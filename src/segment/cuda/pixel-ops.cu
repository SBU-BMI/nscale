// adaptation of Pavel's imreconstruction code for openCV

#include "pixel-ops.cuh"
#include <limits>



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
__global__ void convLoop1Kernel(int rows, int cols, const PtrStep_<unsigned char> img1, PtrStep_<double> result)
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
    dim3 grid((cols * cn + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    invertKernelUInt<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result, std::numeric_limits<T>::max());
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}
template <typename T>
void invertIntCaller(int rows, int cols, int cn, const PtrStep_<T> img1,
 PtrStep_<T> result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols * cn + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    invertKernelInt<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}
template <typename T>
void invertFloatCaller(int rows, int cols, int cn, const PtrStep_<T> img1,
 PtrStep_<T> result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols * cn + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    invertKernelFloat<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

void convLoop1(int rows, int cols, int cn, const PtrStep_<unsigned char> img1,
 	PtrStep_<double> result, cudaStream_t stream)
{
   	dim3 threads(16, 16);
	dim3 grid((cols * cn + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	convLoop1Kernel<<<grid, threads, 0, stream>>>(rows, cols * cn, img1, result);

	 cudaGetLastError() ;

	if (stream == 0)
        	cudaDeviceSynchronize();
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
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	convLoop2Kernel<<<grid, threads, 0, stream>>>(rows, cols, cn_channels, g_cn, dn_channels, g_dn, g_Q, Q_rows, BGR2RGB);

	 cudaGetLastError() ;

	if (stream == 0)
        	cudaDeviceSynchronize();
}

__device__ unsigned char double2char(double d){
	double truncate = min( max(d, (double )0.0), (double)255.0);
	double pt;
	double c = modf(truncate, &pt) >= .5?ceil(truncate):floor(truncate);
	return (unsigned char) c;
}


__global__ void convLoop3Kernel(int rows, int cols, int cn_channels, const PtrStep_<double> g_cn,
 	PtrStep_<unsigned char> g_E, PtrStep_<unsigned char> g_H)
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
 	PtrStep_<unsigned char> g_E, PtrStep_<unsigned char> g_H, cudaStream_t stream)
{
   	dim3 threads(16, 16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	convLoop3Kernel<<<grid, threads, 0, stream>>>(rows, cols, cn_channels, g_cn, g_E, g_H);

	 cudaGetLastError() ;

	if (stream == 0)
        	cudaDeviceSynchronize();
}

__global__ void bgr2grayKernel(int rows, int cols, const PtrStep_<unsigned char> img,
 	PtrStep_<unsigned char> result)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Same constants as used by Matlab
	double r_const = 0.298936021293776;
	double g_const = 0.587043074451121;
	double b_const = 0.114020904255103;

	if (y < rows && x < cols)
	{
		unsigned char b = img.ptr(y)[ x * 3];
		unsigned char g = img.ptr(y)[ x * 3 + 1];
		unsigned char r = img.ptr(y)[ x * 3 + 2];
		double grayPixelValue =  r_const * (double)r + g_const * (double)g + b_const * (double)b;
		result.ptr(y)[x] = double2char(grayPixelValue);
	}
}
void bgr2grayCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> img,  
	cv::gpu::PtrStep_<unsigned char> result, cudaStream_t stream)
{
	dim3 threads(16, 16);
	dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

	bgr2grayKernel<<<grid, threads, 0, stream>>>(rows, cols, img, result);

	 cudaGetLastError();

	if (stream == 0)
        	cudaDeviceSynchronize();

}

template void invertUIntCaller<unsigned char>(int, int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, cudaStream_t);
template void invertIntCaller<int>(int, int, int, const PtrStep_<int>, PtrStep_<int>, cudaStream_t);
template void invertFloatCaller<float>(int, int, int, const PtrStep_<float>, PtrStep_<float>, cudaStream_t);



template <typename T>
__global__ void thresholdKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<unsigned char> result, T lower, bool lower_inclusive, T upper, bool up_inclusive)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
    	T p = img1.ptr(y)[x];
    	bool pb = (p > lower) && (p < upper);
    	if (lower_inclusive) pb = pb || (p == lower);
    	if (up_inclusive) pb = pb || (p == upper);
    	result.ptr(y)[x] = pb ? 255 : 0;
    }
}

template <typename T>
void thresholdCaller(int rows, int cols, const PtrStep_<T> img1,
 PtrStep_<unsigned char> result, T lower, bool lower_inclusive, T upper, bool up_inclusive, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    thresholdKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, lower, lower_inclusive, upper, up_inclusive);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

template void thresholdCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char, bool, unsigned char, bool, cudaStream_t);
template void thresholdCaller<double>(int, int, const PtrStep_<double>, PtrStep_<unsigned char>, double, bool, double, bool, cudaStream_t);
template void thresholdCaller<float>(int, int, const PtrStep_<float>, PtrStep_<unsigned char>, float, bool, float, bool, cudaStream_t);
template void thresholdCaller<int>(int, int, const PtrStep_<int>, PtrStep_<unsigned char>, int, bool, int, bool, cudaStream_t);



template <typename T>
__global__ void replaceKernel(int rows, int cols, const PtrStep_<T> img1, PtrStep_<T> result, T oldval, T newval)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
    	T p = img1.ptr(y)[x];
    	result.ptr(y)[x] = (p == oldval ? newval : p);
    }
}

template <typename T>
void replaceCaller(int rows, int cols, const PtrStep_<T> img1,
 PtrStep_<T> result, T oldval, T newval, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    replaceKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, oldval, newval);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

template void replaceCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char, unsigned char, cudaStream_t);
template void replaceCaller<int>(int, int, const PtrStep_<int>, PtrStep_<int>, int, int, cudaStream_t);


template <typename T>
__global__ void divideKernel(int rows, int cols, const PtrStep_<T> img1, const PtrStep_<T> img2, PtrStep_<T> result)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
    	T p = img1.ptr(y)[x];
    	T q = img2.ptr(y)[x];
        result.ptr(y)[x] = p / q;
    }
}

template <typename T>
void divideCaller(int rows, int cols, const PtrStep_<T> img1,
		const PtrStep_<T> img2, PtrStep_<T> result, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    divideKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, img2, result);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

template void divideCaller<double>(int, int, const PtrStep_<double>, const PtrStep_<double>, PtrStep_<double>, cudaStream_t);


template <typename T>
__global__ void maskKernel(int rows, int cols, const PtrStep_<T> img1, const PtrStep_<unsigned char> img2, PtrStep_<T> result, T background)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < rows && x < cols)
    {
    	T p = img1.ptr(y)[x];
    	unsigned char q = img2.ptr(y)[x];
        result.ptr(y)[x] = (q > 0) ? p : background;
    }
}

template <typename T>
void maskCaller(int rows, int cols, const PtrStep_<T> img1,
		const PtrStep_<unsigned char> img2, PtrStep_<T> result, T background, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    maskKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, img2, result, background);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

template void maskCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char background, cudaStream_t);
template void maskCaller<int>(int, int, const PtrStep_<int>, const PtrStep_<unsigned char>, PtrStep_<int>, int background, cudaStream_t);


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
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    intToCharKernel<<<grid, threads, 0, stream>>>(rows, cols, input, result);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
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
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    intToCharBorderKernel<<<grid, threads, 0, stream>>>(rows, cols, top, bottom, left, right, input, result);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}




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
    dim3 grid((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    modKernel<<<grid, threads, 0, stream>>>(rows, cols, img1, result, mod);
     cudaGetLastError() ;

    if (stream == 0)
        cudaDeviceSynchronize();
}

template void modCaller<unsigned char>(int, int, const PtrStep_<unsigned char>, PtrStep_<unsigned char>, unsigned char, cudaStream_t);
template void modCaller<int>(int, int, const PtrStep_<int>, PtrStep_<int>, int, cudaStream_t);


}}
