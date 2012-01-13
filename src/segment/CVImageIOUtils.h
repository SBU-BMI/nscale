/*
 * for outputting intermediate images.
 * since the intermediate results may only be needed for certain executables, instead of relying on include to pickup the #defines,
 * and risk that a library may not have been compiled with the right set, I am creating 2 versions of the class (so as to not to have branch).
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef CV_IMAGEIO_UTILS_H_
#define CV_IMAGEIO_UTILS_H_

#include "cv.h"
#include <fstream>
#include <iostream>

namespace cciutils {

namespace cv {

using ::cv::Exception;
using ::cv::error;


class IntermediateResultHandler {

public:
	virtual ~IntermediateResultHandler();

	virtual void imwriteRaw(const ::cv::Mat& intermediate, const int stage) = 0;
	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage) = 0;

	virtual void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) = 0;
	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) = 0;
};

class IntermediateResultWriter : IntermediateResultHandler {
private:
	std::string prefix;
	std::string suffix;
	const std::string RAW;
	std::vector<int> selectedStages;
	::cv::gpu::Stream stream;

	std::string getFileName(const int stage, const std::string &suf) {
		stringstream ss;
		ss << prefix << "-" << stage << "." << suf;
		return ss.str();
	};

	bool selected(const int stage) {
		std::vector<int>::iterator pos = std::lower_bound(selectedStages.begin(), selectedStages.end(), stage);
		return (pos == selectedStages.end() || stage != *pos ) ? false : true;
	};

#if defined (HAVE_CUDA)
	::cv::Mat download(const ::cv::gpu::GpuMat& intermediate) {
		::cv::Mat output(intermediate.size(), intermediate.type());
		stream.waitForCompletion();
		stream.enqueueDownload(intermediate, output);
		stream.waitForCompletion();
		return output;
	};
#endif

public:
	IntermediateResultWriter(const std::string &pref, const std::string &suf, std::vector<int> selStages, ::cv::gpu::Stream &strm) {
		prefix.assign(pref);
		suffix.assign(suf);
		selectedStages = selStages;
		std::sort(selectedStages.begin(), selectedStages.end());
		stream = strm;
	};

	virtual ~IntermediateResultWriter() {
		selectedStages.clear();
	};

	// write out with raw
	virtual void imwriteRaw(const ::cv::Mat& intermediate, const int stage) {
		if (!selected(stage)) return;

		std::string filename = getFileName(stage, RAW);
		FILE* fid = fopen(filename.c_str(), "wb");
		const unsigned char* imgPtr;
		int elSize = intermediate.elemSize();
		for (int j = 0; j < intermediate.rows; ++j) {
			imgPtr = intermediate.ptr(j);

			fwrite(imgPtr, elSize, intermediate.cols, fid);
		}
		fclose(fid);

	};
	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage) {
		if (!selected(stage)) return;

		std::string filename = getFileName(stage, suffix);
		::cv::imwrite(filename, intermediate);
	};

#if !defined (HAVE_CUDA)
	virtual void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) { throw_nogpu(); };
	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) { throw_nogpu(); };
#else
	virtual void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) {
		if (!selected(stage)) return;
		// first download the data
		imwriteRaw(download(intermediate), stage);
	};
	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {
		if (!selected(stage)) return;
		// first download the data
		saveIntermediate(download(intermediate), stage);
	};

#endif


};


class IntermediateResultDiscarder :IntermediateResultHandler {

public:
	IntermediateResultDiscarder() {};

	virtual ~IntermediateResultDiscarder() {};

	// write out with raw
	virtual void imwriteRaw(const ::cv::Mat& intermediate, const int stage) {};
	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage) {};

	virtual void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) {};
	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {};

};


}

}


#endif /* CV_IMAGEIO_UTILS_H_ */
