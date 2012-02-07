/*
 * kernel for utility functions
  */
#include <opencv2/gpu/devmem2d.hpp>

namespace nscale { 
namespace gpu {

template <typename T>
void borderCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, T background, int connectivity, cudaStream_t stream);

}}
