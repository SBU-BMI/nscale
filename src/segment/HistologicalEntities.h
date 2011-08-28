/*
 * HistologicalEntities.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef HistologicalEntities_H_
#define HistologicalEntities_H_

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace nscale {

class HistologicalEntities {

public:
	static cv::Mat getRBC(const cv::Mat& img);
	static cv::Mat getRBC(const std::vector<cv::Mat>& bgr);

	static int segmentNuclei(const cv::Mat& img, cv::Mat& output);

};


namespace gpu {

class HistologicalEntities {

public:
	static cv::gpu::GpuMat getRBC(const std::vector<cv::gpu::GpuMat>& bgr, cv::gpu::Stream& stream);
	static cv::gpu::GpuMat getBackground(const std::vector<cv::gpu::GpuMat>& bgr, cv::gpu::Stream& stream);


	static int segmentNuclei(const cv::Mat& img, cv::Mat& output);

};

}

}
#endif /* HistologicalEntities_H_ */
