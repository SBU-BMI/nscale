/*
 * for outputting intermediate images.
 * since the intermediate results may only be needed for certain executables, instead of relying on include to pickup the #defines,
 * and risk that a library may not have been compiled with the right set, I am creating 2 versions of the class (so as to not to have branch).
 *
 *  Created on: Jul 15, 2011
 *      Author: tcpan
 */

#ifndef SCIO_UTILS_CV_IMAGEIO_H_
#define SCIO_UTILS_CV_IMAGEIO_H_

#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "UtilsCVImageIO.h"
#include "Logger.h"

namespace cciutils {

namespace cv {


class SCIOIntermediateResultWriter : public IntermediateResultHandler {
private:
    std::string prefix;
    std::string suffix;
    std::vector<int> selectedStages;
    cci::common::LogSession *session;

    bool compression;

	std::string getFileName(const int stage, const std::string &suf, const std::string &type);
	bool selected(const int stage);
	void imwriteRaw(const ::cv::Mat& intermediate, const int stage);

public:

	SCIOIntermediateResultWriter(const std::string &pref, const std::string &suf, const std::vector<int> &selStages, bool _compression);
	virtual ~SCIOIntermediateResultWriter();

	virtual void setLogSession(::cci::common::LogSession *_session) {
		this->session = _session;
	}

	virtual int persist(int iter) {return -1;};

	virtual void saveIntermediate(const ::cv::Mat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL);

	virtual void saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name = NULL, const int _offsetX = 0, const int _offsetY = 0, const char* _source_tile_file_name = NULL);

};

}

}


#endif /* UTILS_CV_IMAGEIO_H_ */
