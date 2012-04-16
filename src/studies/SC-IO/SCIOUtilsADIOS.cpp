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
//	printf("cleaning up manager %d\n", rank);
	// close all the entries in writer
	for (std::vector<SCIOADIOSWriter *>::iterator iter = writers.begin();
			iter != writers.end(); ++iter) {
//		printf("%d found a writer\n", rank);
		freeWriter(*iter);
		iter = writers.begin();
	}

//	printf("cleaned up manager %d\n", rank);

	adios_finalize(rank);
//	printf("finished cleaning up %d\n", rank);

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

//	printf("cleaning up writer %d \n", w->local_rank);
	w->selectedStages.clear();

//	printf("%d number of writers before: %d\n", rank, writers.size());
	std::vector<SCIOADIOSWriter *>::iterator newend = remove(writers.begin(), writers.end(), w);
	writers.erase(newend, writers.end());
//	printf("%d number of writers after: %d\n", rank, writers.size());

	MPI_Barrier(*(w->local_comm));

	delete w;
//	printf("cleaned up writer %d \n", w->local_rank);
}


bool SCIOADIOSWriter::selected(const int stage) {
	std::vector<int>::iterator pos = std::lower_bound(selectedStages.begin(), selectedStages.end(), stage);
	return (pos == selectedStages.end() || stage != *pos ) ? false : true;
}


SCIOADIOSWriter::~SCIOADIOSWriter() {
	tile_cache.clear();
}


int SCIOADIOSWriter::open() {
	int err;
	if (newfile) {
		err = adios_open(&adios_handle, group_name.c_str(), filename.c_str(), "w", local_comm);
		newfile = false;
	} else {
		err = adios_open(&adios_handle, group_name.c_str(), filename.c_str(), "a", local_comm);
	}
	return err;
}

int SCIOADIOSWriter::close() {

	int err = adios_close(adios_handle);
	return err;
}



int SCIOADIOSWriter::persist() {
	// prepare the data.
	// all the data should be continuous now.

	open();
	printf("size of tile_cache is %d\n", tile_cache.size());

	int err;
	uint64_t total_size;

	if (tile_cache.size() <= 0) {
		err = adios_group_size (adios_handle, 0, &total_size);
	} else {
		long chunk_size = tile_cache.size();
		long chunk_total = 0;
		long chunk_offset = 0;

		// get the size info from other workers
		MPI_Scan(&chunk_size, &chunk_offset, 1, MPI_LONG, MPI_SUM, *local_comm);
		chunk_offset -= chunk_size;
		MPI_Allreduce(&chunk_size, &chunk_total, 1, MPI_LONG, MPI_SUM, *local_comm);

		printf("chunk size: %ld of total %ld at offset %ld\n", chunk_size, chunk_total, chunk_offset);

		// get the total_count from last iteration.
		long total_count = 0;
		MPI_Allreduce(&local_count, &total_count, 1, MPI_LONG, MPI_SUM, *local_comm);
		local_count += chunk_size;

		printf("total count: %ld\n", total_count);

		int imageNameLen = 12;

		char imageNames[imageNameLen * chunk_size];
		memset(imageNames, 0, imageNameLen * chunk_size);
		int tileOffsetX[chunk_size];
		int tileOffsetY[chunk_size];
		char encoding[3 * chunk_size];
		memset(encoding, 0, 3 * chunk_size);

		int tileSizeX[chunk_size];
		int tileSizeY[chunk_size];
		int channels[chunk_size];
		int elemSize1[chunk_size];
		int cvDataType[chunk_size];

		long tile_size[chunk_size];
		long id[chunk_size];

		long tile_offset[chunk_size];
		tile_offset[0] = 0;

		for (int i = 0; i < chunk_size; ++i) {
			strncpy(imageNames + i * imageNameLen, tile_cache[i].image_name.c_str(), imageNameLen);
			tileOffsetX[i] = tile_cache[i].x_offset;
			tileOffsetY[i] = tile_cache[i].y_offset;
			strncpy(encoding + i * 3, "raw", 3);

			::cv::Mat tm = tile_cache[i].tile;
			tileSizeX[i] = tm.cols;
			tileSizeY[i] = tm.rows;
			channels[i] = tm.channels();
			elemSize1[i] = tm.elemSize1();
			cvDataType[i] = tm.type();
			tile_size[i] = tm.dataend - tm.datastart;
			id[i] = i + chunk_offset + total_count;


			// exclusive scan.
			if (i > 0) tile_offset[i] = tile_offset[i-1] + tile_size[i-1];
			printf("id = %ld, size = %ld at offset %ld\n", id[i], tile_size[i], tile_offset[i]);
		}


		long chunk_data_size = tile_offset[chunk_size - 1] + tile_size[chunk_size - 1];

		long chunk_data_total;
		long chunk_data_offset;
		MPI_Scan(&chunk_data_size, &chunk_data_offset, 1, MPI_LONG, MPI_SUM, *local_comm);
		chunk_data_offset -= chunk_data_size;
		MPI_Allreduce(&chunk_data_size, &chunk_data_total, 1, MPI_LONG, MPI_SUM, *local_comm);

		printf("chunk data: size = %ld of total %ld at offset %ld\n", chunk_data_size, chunk_data_total, chunk_data_offset);


		//copy data into new buffer
		unsigned char *tiles = (unsigned char*)malloc(chunk_data_size);
		for (int i = 0; i < chunk_size; ++i) {
			memcpy(tiles + tile_offset[i], tile_cache[i].tile.datastart, tile_size[i]);
		}


		int err = adios_group_size (adios_handle,
				3*8 + 4 + (imageNameLen + 7*4 + 3 + 3*8) * chunk_size +  3*8 + chunk_data_size,
				&total_size);


		adios_write(adios_handle, "chunk_total", &chunk_total);
		adios_write(adios_handle, "chunk_offset", &chunk_offset);
		adios_write(adios_handle, "chunk_size", &chunk_size);
		adios_write(adios_handle, "imageNameLen", &imageNameLen);
		adios_write(adios_handle, "imageName", imageNames);
		adios_write(adios_handle, "tileSizeX", tileSizeX);
		adios_write(adios_handle, "tileSizeY", tileSizeY);
		adios_write(adios_handle, "tileOffsetX", tileOffsetX);
		adios_write(adios_handle, "tileOffsetY", tileOffsetY);
		adios_write(adios_handle, "channels", channels);
		adios_write(adios_handle, "elemSize1", elemSize1);
		adios_write(adios_handle, "cvDataType", cvDataType);
		adios_write(adios_handle, "encoding", encoding);
		adios_write(adios_handle, "id", id);
		adios_write(adios_handle, "tile_offset", tile_offset);
		adios_write(adios_handle, "tile_size", tile_size);

		adios_write(adios_handle, "chunk_data_total", &chunk_data_total);
		adios_write(adios_handle, "chunk_data_offset", &chunk_data_offset);
		adios_write(adios_handle, "chunk_data_size", &chunk_data_size);

		adios_write(adios_handle, "tiles", tiles);

		free(tiles);



	}
	close();  // uses matcache.size();

	tile_cache.clear();


}


void SCIOADIOSWriter::saveIntermediate(const ::cv::Mat& intermediate, const int stage,
		const char *_image_name, const int _offsetX, const int _offsetY) {
	if (!selected(stage)) return;

	unsigned char *data;

	// make a copy
	::cv::Mat out = ::cv::Mat::zeros(intermediate.size(), intermediate.type());
	intermediate.copyTo(out);

	Tile t;
	t.image_name.assign(_image_name);
	t.x_offset = _offsetX;
	t.y_offset = _offsetY;
	t.tile = out;

	// next add to cache
	tile_cache.push_back(t);

}

#if defined (HAVE_CUDA)

	void SCIOADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY) {
		if (!selected(stage)) return;
		// first download the data
		::cv::Mat output(intermediate.size(), intermediate.type());
		intermediate.download(output);
		saveIntermediate(output, stage, _image_name, _offsetX, _offsetY);
		output.release();
	}
#else
	void SCIOADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY) { throw_nogpu(); }
#endif


}







