/*
 * kernel for utility functions
  */
#include "internal_shared.hpp"

namespace nscale { 
namespace gpu {

template <typename T>
void invertUIntCaller(int rows, int cols, int cn, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);
template <typename T>
void invertIntCaller(int rows, int cols, int cn, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);
template <typename T>
void invertFloatCaller(int rows, int cols, int cn, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);

template <typename T>
void thresholdCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<unsigned char> result, T lower, T upper, cudaStream_t stream);

void convertIntToChar(int rows, int cols, int* input, unsigned char *result, cudaStream_t stream);
void convertIntToCharAndRemoveBorder(int rows, int cols, int top, int bottom, int left, int right, int* input, unsigned char *result, cudaStream_t stream);

}}
