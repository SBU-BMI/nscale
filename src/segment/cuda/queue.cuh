/*
 * kernel codes warp center queue
  */
  
#include <cuda.h>
#include <cuda_runtime.h>

namespace nscale { namespace gpu {

template <typename T>
unsigned int SelectCPUTesting(const T* in_data, const int size, T* out_data);

template <typename T>
unsigned int SelectThrustScanTesting(const T* in_data, const int size, T* out_data, cudaStream_t stream);

template <typename T>
unsigned int SelectWarpScanUnorderedTesting(const T* in_data, const int size, T* out_data, cudaStream_t stream);

template <typename T>
unsigned int SelectWarpScanOrderedTesting(const T* in_data, const int size, T* out_data, cudaStream_t stream);

}}