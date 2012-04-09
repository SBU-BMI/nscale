#include "SCIOUtilsADIOS.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "gpu_utils.h"
#include "adios.h"

namespace cciutils {

ADIOSManager::ADIOSManager(const char* configfilename, int _rank, MPI_Comm *_comm) {
	rank = _rank;
	comm = _comm;

	adios_init(configfilename);
	writers.clear();
}

ADIOSManager::~ADIOSManager() {
	// close all the entries in writer
	SCIOADIOSWriter *w;
	while (writers.back() != NULL) {
		w = writers.back();
		writers.pop_back();
		freeWriter(w);
	}

	adios_finalize(rank);
}

SCIOADIOSWriter* ADIOSManager::allocateWriter(const std::string &pref, const std::string &suf,
		const bool _newfile, const std::string &_group_name,
		std::vector<int> &selStages, int _local_rank, MPI_Comm *_local_comm) {

	SCIOADIOSWriter *w = new SCIOADIOSWriter();

	w->selectedStages = selStages;
	std::sort(w->selectedStages.begin(), w->selectedStages.end());
	w->local_rank = _local_rank;
	w->local_comm = _local_comm;
	w->group_name = _group_name;

	std::stringstream ss;
	ss << pref << "." << suf;
	w->filename = ss.str();
	w->newfile = _newfile;
	writers.push_back(w);
	return w;
}


void ADIOSManager::freeWriter(SCIOADIOSWriter *w) {

	w->selectedStages.clear();

	remove(writers.begin(), writers.end(), w);

	MPI_Barrier(*w->local_comm);

	delete w;
}


bool SCIOADIOSWriter::selected(const int stage) {
	std::vector<int>::iterator pos = std::lower_bound(selectedStages.begin(), selectedStages.end(), stage);
	return (pos == selectedStages.end() || stage != *pos ) ? false : true;
}


int SCIOADIOSWriter::open() {
	int err;
	if (newfile) {
		err = adios_open(&adios_handle, group_name.c_str(), filename.c_str(), "w", local_comm);
	} else {
		err = adios_open(&adios_handle, group_name.c_str(), filename.c_str(), "a", local_comm);
	}
	hasData = false;
	return err;
}

int SCIOADIOSWriter::close() {
	int err;
	if (!hasData) {
		uint64_t total_size;
		err = adios_group_size (adios_handle,
				0, &total_size);
	}


	err = adios_close(adios_handle);
	return err;
}


void SCIOADIOSWriter::saveIntermediate(const ::cv::Mat& intermediate, const int stage, const char* image_name, const int offsetX, const int offsetY) {
	if (!selected(stage)) return;

	unsigned char *data;
	if (intermediate.type() == CV_8UC1 || intermediate.type() == CV_8UC3 ||
			intermediate.type() == CV_8SC1 || intermediate.type() == CV_8SC3) {
		::cv::Mat out;
		// get the data as a continuous thing...
		if (intermediate.isContinuous()) {
			out = intermediate;
		} else {
			out = ::cv::Mat::zeros(intermediate.size(), intermediate.type());
			intermediate.copyTo(out);
		}

		// prepare the data.
		data = (unsigned char*)out.data;
		if (out.data != out.datastart) printf("data pointers are different!\n");
		char encoding[strlen("raw") + 1];
		memset(encoding, 0, strlen("raw") + 1);
		strcpy(encoding, "raw");
		char *imageName = (char *)malloc(strlen(image_name) + 1);
		memset(imageName, 0, strlen(image_name) + 1);
		strncpy(imageName, image_name, strlen(image_name));
		int sizeX = out.cols;
		int sizeY = out.rows;
		int offX = offsetX;
		int offY = offsetY;
		int channels = out.channels();
		int elemSize1 = out.elemSize1();
		printf("elem size: %d\n", elemSize1);
		int cvDataType = out.type();
		int dataLength = out.dataend - out.datastart;

		// compute a tileid.
		int total = 0;
		MPI_Allreduce(&local_count, &total, 1, MPI_INT, MPI_SUM, *local_comm);
		int newcount = 1;
		int id;
		MPI_Scan(&newcount, &id, 1, MPI_INT, MPI_SUM, *local_comm);
		id += total;

		printf("worker %d has tile id %d of %d\n", local_rank, id, total);

		uint64_t total_size;
		int err = adios_group_size (adios_handle,
				strlen(imageName) + 4 * 9 + strlen(encoding) + dataLength, &total_size);


		adios_write(adios_handle, "imageName", imageName);
		adios_write(adios_handle, "tileSizeX", &sizeX);
		adios_write(adios_handle, "tileSizeY", &sizeY);
		adios_write(adios_handle, "tileOffsetX", &offX);
		adios_write(adios_handle, "tileOffsetY", &offY);
		adios_write(adios_handle, "channels", &channels);
		adios_write(adios_handle, "elemSize1", &elemSize1);
		adios_write(adios_handle, "cvDataType", &cvDataType);
		adios_write(adios_handle, "dataLength", &dataLength);
		adios_write(adios_handle, "encoding", &encoding);
		adios_write(adios_handle, "id", &id);
		adios_write(adios_handle, "tile", data);

		free(imageName);

		local_count += newcount;
		hasData = true;

	} else {
		printf("ERROR:  NOT SUPPORTED\n");
	}
}

#if defined (HAVE_CUDA)

	void SCIOADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) {
		if (!selected(stage)) return;
		// first download the data
		::cv::Mat output(intermediate.size(), intermediate.type());
		intermediate.download(output);
		saveIntermediate(output, stage);
		output.release();
	}
#else
	void SCIOADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage) { throw_nogpu(); }
#endif


}







