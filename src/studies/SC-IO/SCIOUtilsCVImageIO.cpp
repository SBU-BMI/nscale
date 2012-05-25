#include "SCIOUtilsCVImageIO.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "gpu_utils.h"

namespace cciutils {

namespace cv {


	std::string SCIOIntermediateResultWriter::getFileName(const int stage, const std::string &suf, const std::string &type) {
		std::stringstream ss;
		ss << prefix << "-" << stage;
		
		if (!type.empty()) ss << "." << type;

		if (suf[0] == '.') ss << suf;
		else ss << "." << suf;
		return ss.str();
	}

	bool SCIOIntermediateResultWriter::selected(const int stage) {
		std::vector<int>::iterator pos = std::lower_bound(selectedStages.begin(), selectedStages.end(), stage);
		return (pos == selectedStages.end() || stage != *pos ) ? false : true;
	}

	void SCIOIntermediateResultWriter::imwriteRaw(const ::cv::Mat& intermediate, const int stage) {
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
		if (!fid) printf("ERROR: can't open %s to write\n", filename.c_str());

		const unsigned char* imgPtr;
		int elSize = intermediate.elemSize();
		for (int j = 0; j < intermediate.rows; ++j) {
			imgPtr = intermediate.ptr(j);

			fwrite(imgPtr, elSize, intermediate.cols, fid);
		}
		fclose(fid);

	};


	SCIOIntermediateResultWriter::SCIOIntermediateResultWriter(const std::string &pref, const std::string &suf, const std::vector<int> &selStages) {
		prefix.assign(pref);
		suffix.assign(suf);
		selectedStages = selStages;
		std::sort(selectedStages.begin(), selectedStages.end());
	}

	SCIOIntermediateResultWriter::~SCIOIntermediateResultWriter() {
		selectedStages.clear();
	}

	void SCIOIntermediateResultWriter::saveIntermediate(const ::cv::Mat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) {
		if (!selected(stage)) return;

		uint64_t t1, t2;

		t1 = cciutils::event::timestampInUS();
		if (intermediate.type() == CV_8UC1 || intermediate.type() == CV_8UC3 ||
				intermediate.type() == CV_8SC1 || intermediate.type() == CV_8SC3) {

			// write grayscale or RGB
			std::string filename = getFileName(stage, suffix, std::string());
			::cv::imwrite(filename, intermediate);
		} else {
			imwriteRaw(intermediate, stage);
		}
		t2 = cciutils::event::timestampInUS();
		char len[21];  // max length of uint64 is 20 digits
		memset(len, 0, 21);
		sprintf(len, "%lu", (long)(intermediate.dataend) - (long)(intermediate.datastart));

		if (session != NULL) session->log(cciutils::event(stage, std::string("save image"), t1, t2, std::string(len), ::cciutils::event::FILE_O));
	}

#if defined (WITH_CUDA)

	void SCIOIntermediateResultWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) {
		if (!selected(stage)) return;
		// first download the data
		::cv::Mat output(intermediate.size(), intermediate.type());
		intermediate.download(output);
		saveIntermediate(output, stage);
		output.release();
	}
#else
	void SCIOIntermediateResultWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) { throw_nogpu(); }
#endif


}




}


