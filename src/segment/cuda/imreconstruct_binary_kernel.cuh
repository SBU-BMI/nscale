/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
  */
#include "internal_shared.hpp"

namespace nscale { namespace gpu {

template <typename T>
unsigned int imreconstructBinaryCaller(T* g_marker, const T* g_mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream );


}}