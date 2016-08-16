/*
 * HistologicalEntities.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef HistologicalEntities_H_
#define HistologicalEntities_H_

#include "UtilsCVImageIO.h"
#include "UtilsLogger.h"
#include "opencv2/opencv.hpp"
#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif
#include <string.h>


#ifdef _MSC_VER
#define DllExport __declspec(dllexport)
#else
#define DllExport //nothing 
#endif

namespace nscale {


class DllExport HistologicalEntities {

public:


	static const int BACKGROUND = 1;
	static const int BACKGROUND_LIKELY = 2;
	static const int NO_CANDIDATES_LEFT = 3;
	static const int INVALID_IMAGE = -1;
	static const int SUCCESS = 0;
	static const int CONTINUE = 4;
	static const int RUNTIME_FAILED = -2;


	static cv::Mat getRBC(const cv::Mat& img, double T1=5.0, double T2=4.0,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static cv::Mat getRBC(const std::vector<cv::Mat>& bgr, double T1=5.0, double T2=4.0,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static cv::Mat getBackground(const cv::Mat& img, unsigned char blue=220, unsigned char green=220, unsigned char red=220,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static cv::Mat getBackground(const std::vector<cv::Mat>& bgr, unsigned char blue=220, unsigned char green=220, unsigned char red=220,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);

	static int segmentNuclei(const cv::Mat& img, cv::Mat& mask,
			int &compcount, int *&bbox, unsigned char blue=220, unsigned char green=220, unsigned char red=220, double T1=5.0, double T2=4.0, unsigned char G1 = 80, int minSize=11, int maxSize=1000, unsigned char G2 = 45, int minSizePl=30, int minSizeSeg=21, int maxSizeSeg=1000,  int fillHolesConnectivity=4, int reconConnectivity=8, int watershedConnectivity=8,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static int segmentNuclei(const std::string& input, const std::string& output,
			int &compcount, int *&bbox, unsigned char blue=220, unsigned char green=220, unsigned char red=220, double T1=5.0, double T2=4.0, unsigned char G1 = 80, int minSize=11, int maxSize=1000, unsigned char G2 = 45, int minSizePl=30, int minSizeSeg=21, int maxSizeSeg=1000,  int fillHolesConnectivity=4, int reconConnectivity=8, int watershedConnectivity=8,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);

	static int segmentNuclei(const cv::Mat& img, cv::Mat& mask, unsigned char blue=220, unsigned char green=220, unsigned char red=220, double T1=5.0, double T2=4.0, unsigned char G1 = 80, int minSize=11, int maxSize=1000, unsigned char G2 = 45, int minSizePl=30, int minSizeSeg=21, int maxSizeSeg=1000, int fillHolesConnectivity=4, int reconConnectivity=8, int watershedConnectivity=8,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);

	// the following are specific to the task based implementation for HPDC paper.  The pipeline is refactoring into this form so we're maintaining one set of code.
	static int plFindNucleusCandidates(const cv::Mat& img, cv::Mat& seg_norbc, unsigned char blue=220, unsigned char green=220, unsigned char red=220, double T1=5.0, double T2=4.0, unsigned char G1 = 80, int minSize=11, int maxSize=1000, unsigned char G2 = 45, int fillHolesConnectivity=4, int reconConnectivity=8,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);  // S1
	static int plSeparateNuclei(const cv::Mat& img, const cv::Mat& seg_open, cv::Mat& seg_nonoverlap, int minSizePl=30, int watershedConnectivity=8,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL); // A4

};


#ifdef WITH_CUDA
namespace gpu {

class DllExport HistologicalEntities {

public:

	static cv::gpu::GpuMat getRBC(const std::vector<cv::gpu::GpuMat>& bgr, cv::gpu::Stream& stream,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static cv::gpu::GpuMat getBackground(const std::vector<cv::gpu::GpuMat>& bgr, cv::gpu::Stream& stream,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);

	static int segmentNuclei(const cv::gpu::GpuMat& img, cv::gpu::GpuMat& mask,
			int &compcount, int *&g_bbox, cv::gpu::Stream *str = NULL,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static int segmentNuclei(const cv::Mat& img, cv::Mat& mask,
			int &compcount, int *&bbox, cv::gpu::Stream *str = NULL,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);
	static int segmentNuclei(const std::string& input, const std::string& output,
			int &compcount, int *&bbox, cv::gpu::Stream *str = NULL,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);

	// the following are specific to the task based implementation for HPDC paper.  The pipeline is refactoring into this form so we're maintaining one set of code.
	static int plFindNucleusCandidates(const cv::gpu::GpuMat& g_img, cv::gpu::GpuMat& g_seg_norbc, cv::gpu::Stream& stream,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL);  // S1
	static int plSeparateNuclei(const cv::gpu::GpuMat& g_img, cv::gpu::GpuMat& g_seg_open, cv::gpu::GpuMat& g_seg_nonoverlap, cv::gpu::Stream& stream,
			::cciutils::SimpleCSVLogger *logger = NULL, ::cciutils::cv::IntermediateResultHandler *iresHandler = NULL); // A4


};

DllExport int* boundingBox2(cv::gpu::GpuMat g_input, cv::gpu::Stream *str);
DllExport int* boundingBox2(cv::gpu::GpuMat g_input, int &compcount, cv::gpu::Stream *str);
DllExport void cudaFreeData(char *dataPtr);
DllExport void cudaDownloadData(char *dest, char *from, int size);

}
#endif

}
#endif /* HistologicalEntities_H_ */
