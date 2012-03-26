/*
 * kernel for utility functions
  */
#include "opencv2/gpu/devmem2d.hpp"
#include "cutil.h"
#include <cuda_runtime.h>


namespace nscale { 
namespace gpu {
using namespace cv::gpu;

void xDerivativeSquareCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> input, const cv::gpu::PtrStep_<float> result, cudaStream_t stream);

// Calculate derivative, and accumate sqrt of derivate in output
void yDerivativeSquareAccCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> input, const cv::gpu::PtrStep_<float> result, cudaStream_t stream);

// convert float image to unsigned char
void floatToUcharCaller(int rows, int cols, const cv::gpu::PtrStep_<float> input, const cv::gpu::PtrStep_<unsigned char> result, cudaStream_t stream);



}}
