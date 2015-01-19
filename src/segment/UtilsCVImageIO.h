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


#ifdef _MSC_VER
#define DllExport __declspec(dllexport)
#else
#define DllExport //nothing 
#endif

namespace cciutils {

namespace cv {

#define RAW ".raw"

class DllExport IntermediateResultHandler {

public:
	IntermediateResultHandler() {};

	virtual ~IntermediateResultHandler() {};

	virtual int persist(int iter) = 0;

	// write out with raw
	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL) = 0;

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL) = 0;

};


class DllExport IntermediateResultWriter : public IntermediateResultHandler {
private:
    std::string prefix;
    std::string suffix;
    std::vector<int> selectedStages;


	std::string getFileName(const int stage, const std::string &suf, const std::string &type);
	bool selected(const int stage);
	void imwriteRaw(const ::cv::Mat& intermediate, const int stage);

public:

	IntermediateResultWriter(const std::string &pref, const std::string &suf, const std::vector<int> &selStages);
	virtual ~IntermediateResultWriter();

	virtual int persist(int iter) { return -1; };

	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL);

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL);

};

}

}


#endif /* UTILS_CV_IMAGEIO_H_ */
