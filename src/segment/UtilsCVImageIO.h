/*
 * for outputting intermediate images.
 * since the intermediate results may only be needed for certain executables, instead of relying on include to pickup the #defines,
 * and risk that a library may not have been compiled with the right set, I am creating 2 versions of the class (so as to not to have branch).
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef UTILS_CV_IMAGEIO_H_
#define UTILS_CV_IMAGEIO_H_

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

namespace cciutils {

namespace cv {

#define RAW ".raw"

class IntermediateResultHandler {

public:
	IntermediateResultHandler() {};

	virtual ~IntermediateResultHandler() {};

	// write out with raw
	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage) {};

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {};

};


class IntermediateResultWriter : public IntermediateResultHandler {
private:
    std::string prefix;
    std::string suffix;
    std::vector<int> selectedStages;


	std::string getFileName(const int stage, const std::string &suf, const std::string &type);
	bool selected(const int stage);
	void imwriteRaw(const ::cv::Mat& intermediate, const int stage);

public:

	IntermediateResultWriter(const std::string &pref, const std::string &suf, std::vector<int> &selStages);
	virtual ~IntermediateResultWriter();

	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage);

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage);

};

}

}


#endif /* UTILS_CV_IMAGEIO_H_ */