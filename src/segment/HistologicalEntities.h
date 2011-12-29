/*
 * HistologicalEntities.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef HistologicalEntities_H_
#define HistologicalEntities_H_

#include "utils.h"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <string.h>

namespace nscale {

class HistologicalEntities {

public:
	static cv::Mat getRBC(const cv::Mat& img);
	static cv::Mat getRBC(const std::vector<cv::Mat>& bgr);
	static cv::Mat getBackground(const cv::Mat& img);
	static cv::Mat getBackground(const std::vector<cv::Mat>& bgr);


	static int segmentNuclei(const cv::Mat& img, cv::Mat& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1);
	static int segmentNuclei(const std::string& input, const std::string& output, cciutils::SimpleCSVLogger *logger=NULL, int stage=-1);

	static const int BACKGROUND = 1;
	static const int BACKGROUND_LIKELY = 2;
	static const int NO_CANDIDATES_LEFT = 3;
	static const int INVALID_IMAGE = -1;
	static const int SUCCESS = 0;
	static const int CONTINUE = 4;


	// the following are specific to the task based implementation for HPDC paper.  The pipeline is refactoring into this form so we're maintaining one set of code.
	static int plFindNucleusCandidates(const cv::Mat& img, cv::Mat& seg_norbc, cv::Mat& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1);  // S1
	static int plSeparateNuclei(const cv::Mat& img, const cv::Mat& seg_open, cv::Mat& seg_nonoverlap, cv::Mat& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1); // A4

};


namespace gpu {

class HistologicalEntities {

public:
	static cv::gpu::GpuMat getRBC(const std::vector<cv::gpu::GpuMat>& bgr, cv::gpu::Stream& stream);
	static cv::gpu::GpuMat getBackground(const std::vector<cv::gpu::GpuMat>& bgr, cv::gpu::Stream& stream);

	static int segmentNuclei(const cv::Mat& img, cv::Mat& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1);
	static int segmentNuclei(const std::string& input, const std::string& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1);

	// the following are specific to the task based implementation for HPDC paper.  The pipeline is refactoring into this form so we're maintaining one set of code.
	static int plFindNucleusCandidates(cv::gpu::GpuMat& g_img, cv::gpu::GpuMat& g_seg_norbc, cv::gpu::Stream& stream, cv::Mat& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1);  // S1
	static int plSeparateNuclei(cv::gpu::GpuMat& g_img, cv::gpu::GpuMat& g_seg_open, cv::gpu::GpuMat& g_seg_nonoverlap, cv::gpu::Stream& stream, cv::Mat& output, cciutils::SimpleCSVLogger *logger = NULL, int stage=-1); // A4

};

}

}
#endif /* HistologicalEntities_H_ */
