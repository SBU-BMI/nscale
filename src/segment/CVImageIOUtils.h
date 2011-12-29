/*
 * utils.h
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef CV_IMAGEIO_UTILS_H_
#define CV_IMAGEIO_UTILS_H_

#include "cv.h"
#include <fstream>
#include <iostream>
#include <sys/time.h>

namespace cciutils {

namespace cv {

using ::cv::Exception;
using ::cv::error;

class IntermediateResultWriter {
private:
	std::string prefix;
	std::string suffix;
	const std::string RAW;
	std::vector<int> selectedStages;
	::cv::gpu::Stream stream;
public:
	IntermediateResultWriter(const std::string &pref, const std::string &suf, std::vector<int> selStages, ::cv::gpu::Stream &strm) {
		prefix.assign(pref);
		suffix.assign(suf);
		selectedStages = selStages;
		std::sort(selectedStages.begin(), selectedStages.end());
		stream = strm;
	};

	~IntermediateResultWriter() {
		selectedStages.clear();
	}

	inline std::string getFileName(const int stage, const std::string &suf) {
		stringstream ss;
		ss << prefix << "-" << stage << "." << suf;
		return ss.str();
	}

#define SAVE_INTERMEDIATES 1;


#if !defined (SAVE_INTERMEDIATES)
	// write out with raw
	inline void imwriteRaw(const ::cv::Mat& intermediate, const int stage) {};
	inline void saveIntermediate(const ::cv::Mat& intermediate, const int stage) {};

	inline void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) {};
	inline void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {};

#else

	// write out with raw
	inline void imwriteRaw(const ::cv::Mat& intermediate, const int stage) {
		std::string filename = getFileName(stage, RAW);
		FILE* fid = fopen(filename.c_str(), "wb");
		const uchar* imgPtr;
		int elSize = intermediate.elemSize();
		for (int j = 0; j < intermediate.rows; ++j) {
			imgPtr = intermediate.ptr(j);

			fwrite(imgPtr, elSize, intermediate.cols, fid);
		}
		fclose(fid);

	};
	inline void saveIntermediate(const ::cv::Mat& intermediate, const int stage) {
		std::string filename = getFileName(stage, suffix);
		::cv::imwrite(filename, intermediate);
	};

#if !defined (HAVE_CUDA)
	inline void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) { throw_nogpu(); };
	inline void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) { throw_nogpu(); };
#else
	inline ::cv::Mat download(const ::cv::gpu::GpuMat& intermediate) {
		::cv::Mat output(intermediate.size(), intermediate.type());
		stream.waitForCompletion();
		stream.enqueueDownload(intermediate, output);
		stream.waitForCompletion();
		return output;
	}

	inline void imwriteRaw(const ::cv::gpu::GpuMat& intermediate, const int stage) {
		// first download the data
		imwriteRaw(download(intermediate), stage);
	};
	inline void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {
		// first download the data
		saveIntermediate(download(intermediate), stage);
	};

#endif



#endif

};


}

}


#endif /* CV_IMAGEIO_UTILS_H_ */
