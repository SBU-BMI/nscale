/*
 * kernel for utility functions
  */
#include "internal_shared.hpp"

namespace nscale { 
namespace gpu {

template <typename T>
void borderCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, T background, cudaStream_t stream);

}}
