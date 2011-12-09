/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
  */
#include <cuda.h>
#include <cuda_runtime.h>

namespace nscale { namespace gpu { namespace dw {

__host__ void giwatershed( int *hdataOut,
                                      float *hdataIn,
                                      int w,
                                      int h,
                                      int conn,
                                      cudaStream_t stream);
}}}

//__host__ void giwatershed( int *hdataOut,
//                                      unsigned char *hdataIn,
//                                      int w,
//                                      int h,
//                                      int conn);
