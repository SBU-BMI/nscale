/*
 * kernel for utility functions
  */
#include "opencv2/gpu/devmem2d.hpp"
#include "cutil.h"
#include <cuda_runtime.h>


namespace nscale { 
namespace gpu {
using namespace cv::gpu;

int* calcHist256Caller(const cv::gpu::PtrStep_<int> labeledImg, const cv::gpu::PtrStep_<unsigned char> grayImage, int *bbInfo, int numComponents, cudaStream_t stream);
void calcFeaturesFromHist256Caller(int *hist, int numHists, float *output, cudaStream_t stream);
void calcGradFeaturesFromHist256Caller(int *hist, int numHists, float *output, cudaStream_t stream);
void calcCannyFeaturesFromHist256Caller(int *hist, int numHists, float *output, cudaStream_t stream);

void cudaFreeCaller(void *data_ptr);
void *cudaMallocCaller(int size);
void cudaUploadCaller(void *dest, void *source, int size);
void cudaDownloadCaller(void *dest, void *source, int size);


}}
