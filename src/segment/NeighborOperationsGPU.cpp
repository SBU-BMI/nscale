/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "NeighborOperations.h"
#include <limits>

#include "precomp.hpp"

#if defined (HAVE_CUDA)
#include "cuda/neighbor-ops.cuh"
#endif

namespace nscale {

using namespace cv;

namespace gpu {

using namespace cv::gpu;


#if !defined (HAVE_CUDA)
template <typename T>
GpuMat NeighborOperations::border(const GpuMat& img, T background, Stream& stream) { throw_nogpu(); }

#else


template <typename T>
GpuMat NeighborOperations::border(const GpuMat& img, T background, Stream& stream) {
	// write the raw image
	CV_Assert(img.channels() == 1);
	CV_Assert(std::numeric_limits<T>::is_integer);

	// make border
	GpuMat input = createContinuous(img.rows + 2, img.cols + 2, img.type());
	copyMakeBorder(img, input, 1, 1, 1, 1, Scalar(background), stream);
	stream.waitForCompletion();

    GpuMat result = createContinuous(input.size(), input.type());

    borderCaller<T>(input.rows, input.cols, input, result, background, StreamAccessor::getStream(stream));
    stream.waitForCompletion();

    input.release();

    return result(Rect(1,1, img.cols, img.rows));
}

#endif

template GpuMat NeighborOperations::border<int>(const GpuMat&, int, Stream&);


}

}


