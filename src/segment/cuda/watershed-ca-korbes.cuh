/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
  */
#include <cuda.h>
#include <cuda_runtime.h>

namespace nscale { namespace gpu { namespace ca {

__host__ float ws_kauffmann(int *label, // output
                           float *f, // input
                           int *seeds, // seeds (regional minima)
                           int w,  // width
                           int h, //height
                           int conn); // connectivity (4 or 8)
}}}

//__host__ float ws_kauffmann(int *label, // output
//                           unsigned char *f, // input
//                           int *seeds, // seeds (regional minima)
//                           int w,  // width
//                           int h, //height
//                           int conn); // connectivity (4 or 8)
