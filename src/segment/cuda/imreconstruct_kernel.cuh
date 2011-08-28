/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
  */
#include "internal_shared.hpp"

namespace nscale { namespace gpu {

template <typename T>
void imreconstructCaller(const PtrStep_<T> g_marker, const PtrStep_<T> g_mask, PtrStep_<T> g_result, const int connectivity );


}}