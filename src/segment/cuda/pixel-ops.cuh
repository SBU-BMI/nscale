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
 
template <typename T>
void modCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, T mod, cudaStream_t stream);

}}
