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
#include <fstream>
#include <iostream>
#include <sstream>

namespace cciutils {

namespace cv {

#define RAW ".raw"

using ::cv::Exception;
using ::cv::error;

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

	std::string getFileName(const int stage, const std::string &suf, const std::string &type) {
		std::stringstream ss;
		ss << prefix << "-" << stage;
		
		if (!type.empty()) ss << "." << type;

		if (suf[0] == '.') ss << suf;
		else ss << "." << suf;
		return ss.str();
	};

	bool selected(const int stage) {
		std::vector<int>::iterator pos = std::lower_bound(selectedStages.begin(), selectedStages.end(), stage);
		return (pos == selectedStages.end() || stage != *pos ) ? false : true;
	};

	void imwriteRaw(const ::cv::Mat& intermediate, const int stage) {
		// write out as raw
		std::stringstream ss;
		switch (intermediate.depth()) {
		case 0:
			ss << "8UC";
			break;
		case 1:
			ss << "8SC";
			break;
		case 2:
			ss << "16UC";
			break;
		case 3:
			ss << "16SC";
			break;
		case 4:
			ss << "32SC";
			break;
		case 5:
			ss << "32FC";
			break;
		case 6:
			ss << "64FC";
			break;
		default:
			ss << "USRTYPE1C";
		}
		ss << intermediate.channels();


		std::string filename = getFileName(stage, std::string(RAW), ss.str());
		FILE* fid = fopen(filename.c_str(), "wb");
		const unsigned char* imgPtr;
		int elSize = intermediate.elemSize();
		for (int j = 0; j < intermediate.rows; ++j) {
			imgPtr = intermediate.ptr(j);

			fwrite(imgPtr, elSize, intermediate.cols, fid);
		}
		fclose(fid);

	}


#if defined (HAVE_CUDA)
	::cv::Mat download(const ::cv::gpu::GpuMat& intermediate) {
		::cv::Mat output(intermediate.size(), intermediate.type());
		intermediate.download(output);
		return output;
	};
#endif

public:

	IntermediateResultWriter(const std::string &pref, const std::string &suf, std::vector<int> &selStages) {
		prefix.assign(pref);
		suffix.assign(suf);
		selectedStages = selStages;
		std::sort(selectedStages.begin(), selectedStages.end());
	};

	virtual ~IntermediateResultWriter() {
		selectedStages.clear();
	};

	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage) {
		if (!selected(stage)) return;

		if (intermediate.type() == CV_8UC1 || intermediate.type() == CV_8UC3 ||
				intermediate.type() == CV_8SC1 || intermediate.type() == CV_8SC3) {

			// write grayscale or RGB
			std::string filename = getFileName(stage, suffix, std::string());
			::cv::imwrite(filename, intermediate);
		} else {
			imwriteRaw(intermediate, stage);
		}
	};

#if !defined (HAVE_CUDA)
	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) { throw_nogpu(); };
#else

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {
		if (!selected(stage)) return;
		// first download the data
		saveIntermediate(download(intermediate), stage);
	};

#endif


};




}

}


#endif /* UTILS_CV_IMAGEIO_H_ */
