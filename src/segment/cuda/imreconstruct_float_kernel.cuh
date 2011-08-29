/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
  */
#include "internal_shared.hpp"

namespace nscale { namespace gpu {

template <typename T>
void imreconstructFloatCaller(cv::gpu::DevMem2D_<T> marker, const cv::gpu::DevMem2D_<T> mask,
		int connectivity, cudaStream_t stream );


}}