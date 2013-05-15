/*
 * RedBloodCell.cpp
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#include "SCIOHistologicalEntities.h"
#include <iostream>
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "gpu_utils.h"
#include "opencv2/gpu/gpu.hpp"
#include "NeighborOperations.h"


namespace nscale {

using namespace cv;

namespace gpu {

using namespace cv::gpu;





GpuMat SCIOHistologicalEntities::getRBC(const std::vector<GpuMat>& bgr, Stream& stream,
		::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
GpuMat SCIOHistologicalEntities::getBackground(const std::vector<GpuMat>& g_bgr, Stream& stream,
				::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
int SCIOHistologicalEntities::segmentNuclei(const GpuMat& g_img, GpuMat& g_output,
		int &compcount, int *&g_bbox,  cv::gpu::Stream *str,
		::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
int SCIOHistologicalEntities::segmentNuclei(const Mat& img, Mat& output,
		int &compcount, int *&bbox, cv::gpu::Stream *str,
		::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }
int SCIOHistologicalEntities::segmentNuclei(const std::string& input, const std::string& output,
		int &compcount, int *&bbox, cv::gpu::Stream *str,
		::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }


// the following are specific to the task based implementation for HPDC paper.  The pipeline is refactoring into this form so we're maintaining one set of code.
int plFindNucleusCandidates(const cv::gpu::GpuMat& img, cv::gpu::GpuMat& nuclei,
		::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }  // S1
int plSeparateNuclei(const cv::gpu::GpuMat& img, const cv::gpu::GpuMat& seg_open, cv::gpu::GpuMat& seg_nonoverlap,
		::cci::common::LogSession *logsession, ::cciutils::cv::IntermediateResultHandler *iresHandler) { throw_nogpu(); }// A4


}}
