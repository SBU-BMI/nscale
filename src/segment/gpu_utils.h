/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef GPU_UTILS_H_
#define GPU_UTILS_H_

#include "opencv2/core/core.hpp"

// can't use a namespace here.  the macro from core.hpp uses relative namespace.
	// from openCV gpu precomp.h

#if defined(WITH_CUDA)

    static inline void throw_nogpu() { CV_Error(CV_GpuNotSupported, "The called functionality is disabled for current build or platform"); }

#else /* defined(WITH_CUDA) */

    static inline void throw_nogpu() { CV_Error(CV_GpuNotSupported, "The library is compiled without GPU support"); }

#endif /* defined(WITH_CUDA) */

#endif /* UTILS_H_ */

