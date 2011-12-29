/*
 * kernel codes for the reconstruction by dilation (reconstruction_by_dilation_kernel)
  */
#include "internal_shared.hpp"

namespace nscale { namespace gpu {

template <typename T>
unsigned int imreconstructIntCaller(T* g_marker, const T* g_mask, const int sx, const int sy,
		const int connectivity, cudaStream_t stream);
		//, unsigned char*h_markerFistPass );

template <typename T> 
int *imreconstructIntCallerBuildQueue(T* marker, const T* mask, const int sx, const int sy, 
		const int connectivity, int &queueSize, int num_iterations, cudaStream_t stream);

}}
