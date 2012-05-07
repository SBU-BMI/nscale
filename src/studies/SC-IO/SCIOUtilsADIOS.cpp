#include "SCIOUtilsADIOS.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "gpu_utils.h"
#include "adios.h"
#include "adios_internals.h"

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
		const bool _newfile,
		std::vector<int> &selStages, const long mx_tileinfo_count, const long mx_imagename_bytes, const long mx_tile_bytes,
 int _local_rank, MPI_Comm *_local_comm) {


	SCIOADIOSWriter *w = new SCIOADIOSWriter();

	w->selectedStages = selStages;
	std::sort(w->selectedStages.begin(), w->selectedStages.end());
	w->local_rank = _local_rank;
	w->local_comm = _local_comm;

	w->tile_capacity = mx_tile_bytes;
	w->imageName_capacity = mx_imagename_bytes;
	w->tileInfo_capacity = mx_tileinfo_count;

	printf("INITIALIZED with tileinfo %d, imagename %d, tile %d\n", w->tileInfo_capacity, w->imageName_capacity, w->tile_capacity);

	std::stringstream ss;
	ss << pref << "." << suf;
	w->filename = ss.str();
	printf("creating file: %s\n", w->filename.c_str());
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


int SCIOADIOSWriter::open(const char* groupName) {
	int err;
	if (newfile) {
		err = adios_open(&adios_handle, groupName, filename.c_str(), "w", local_comm);
		newfile = false;
	} else {
		err = adios_open(&adios_handle, groupName, filename.c_str(), "a", local_comm);
	}
	return err;
}

int SCIOADIOSWriter::close(uint32_t time_index) {
	// if time_index is not specified, then let ADIOS handle it.
        	struct adios_file_struct * fd = (struct adios_file_struct *) adios_handle;
        	struct adios_group_struct *gd = (struct adios_group_struct *) fd->group;
		
		printf("rank %d, group name %s, id %u, membercount %u, offset %lu, timeindex %u, proc id %u\n", local_rank, gd->name, gd->id, gd->member_count, gd->group_offset, gd->time_index, gd->process_id);
		printf("rank %d, file datasize %lu, writesizebytes %lu, pgstart %lu, baseoffset %lu, offset %lu, bytewritten %lu, bufsize %lu\n", local_rank, fd->data_size, fd->write_size_bytes, fd->pg_start_in_file, fd->base_offset, fd->offset, fd->bytes_written, fd->buffer_size); 
	if (time_index > 0) {
		gd->time_index = time_index;

		
	}
	int err = adios_close(adios_handle);
	return err;
}



int SCIOADIOSWriter::persist() {
	// prepare the data.
	// all the data should be continuous now.

	printf("size of tile_cache is %lu\n", tile_cache.size());

	int err;
	uint64_t adios_groupsize, adios_totalsize;


	/**
	*  first set up the index variables.
	*/
	long tileInfo_pg_size = tile_cache.size();
	long tile_pg_size = 0;
	long imageName_pg_size = 0;
	// capacity variables already set.


	/**
	* gather specific data for the tile in the process
	*/
	unsigned char *tiles = NULL;
	char *imageNames = NULL;
	int *tileOffsetX, *tileOffsetY, *tileSizeX, *tileSizeY, *channels,
		*elemSize1, *cvDataType, *encoding;
	long *imageName_offset, *imageName_size, *tile_offset, *tile_size;

	if (tile_cache.size() > 0) {

		/** initialize storage
		*/ 
		tileOffsetX = (int *)malloc(tile_cache.size() * sizeof(int));
		tileOffsetY = (int *)malloc(tile_cache.size() * sizeof(int));
		tileSizeX = (int *)malloc(tile_cache.size() * sizeof(int));
		tileSizeY = (int *)malloc(tile_cache.size() * sizeof(int));
		channels = (int *)malloc(tile_cache.size() * sizeof(int));
		elemSize1 = (int *)malloc(tile_cache.size() * sizeof(int));
		cvDataType = (int *)malloc(tile_cache.size() * sizeof(int));
		encoding = (int *)malloc(tile_cache.size() * sizeof(int));
		
		imageName_size = (long*) malloc(tile_cache.size() * sizeof(long));
		imageName_offset = (long*) malloc(tile_cache.size() * sizeof(long));
		tile_size = (long*) malloc(tile_cache.size() * sizeof(long));
		tile_offset = (long*) malloc(tile_cache.size() * sizeof(long));


		/**  get tile metadata
		*/
		for (int i = 0; i < tile_cache.size(); ++i) {
			
			tileOffsetX[i] = tile_cache[i].x_offset;
			tileOffsetY[i] = tile_cache[i].y_offset;
	
			::cv::Mat tm = tile_cache[i].tile;
			tileSizeX[i] = tm.cols;
			tileSizeY[i] = tm.rows;
			channels[i] = tm.channels();
			elemSize1[i] = tm.elemSize1();
			cvDataType[i] = tm.type();
			
			encoding[i] = ENC_RAW;

			imageName_size[i] = (long)(tile_cache[i].image_name.size());
			tile_size[i] = (long)(tm.dataend) - (long)(tm.datastart);
//			tile_size[i] = 0;

			// update the offset (within the group for this time step)
			// to the size so far.
			// need to update to global coord later.
			imageName_offset[i] = imageName_pg_size;
			tile_offset[i] = tile_pg_size;
			
			// update the process group totals
			imageName_pg_size += imageName_size[i];
			tile_pg_size += tile_size[i];


			printf("rank %d tile %d offset %dx%d, size %dx%dx%d, elemSize %d type %d encoding %d, tile bytes %ld at %ld, imagename %ld at %ld\n", local_rank, i, tileOffsetX[i], tileOffsetY[i], tileSizeX[i], tileSizeY[i], channels[i], elemSize1[i], cvDataType[i], encoding[i], tile_size[i], tile_offset[i], imageName_size[i], imageName_offset[i]);
		}


		/** now allocate the space for tile and for imagename
			and then get the data.
		*/
		imageNames = (char *)malloc(imageName_pg_size);
		memset(imageNames, 0, imageName_pg_size);
		tiles = (unsigned char *)malloc(tile_pg_size);
		memset(tiles, 0, tile_pg_size);
		
		for (int i = 0; i < tile_cache.size(); ++i) {
			strncpy(imageNames + imageName_offset[i], tile_cache[i].image_name.c_str(), imageName_size[i]);
			memcpy(tiles + tile_offset[i], tile_cache[i].tile.datastart, tile_size[i]);
		}	

	}

	/**
	* compute the offset for each step, in global array coordinates
	*/
	long pg_sizes[3];
	pg_sizes[0] = tileInfo_pg_size;
	pg_sizes[1] = imageName_pg_size;
	pg_sizes[2] = tile_pg_size;
	long pg_offsets[3];
	pg_offsets[0] = 0;
	pg_offsets[1] = 0;
	pg_offsets[2] = 0;
	MPI_Scan(pg_sizes, pg_offsets, 3, MPI_LONG, MPI_SUM, *local_comm);
	long tileInfo_pg_offset = pg_offsets[0] - tileInfo_pg_size + this->tileInfo_total;
	long imageName_pg_offset = pg_offsets[1] - imageName_pg_size + this->imageName_total;
	long tile_pg_offset = pg_offsets[2] - tile_pg_size + this->tile_total;
	// update the offsets within the process group
	for (int i = 0; i < tile_cache.size(); ++ i) {
		imageName_offset[i] += imageName_pg_offset;
		tile_offset[i] += tile_pg_offset;
			printf("globally shifted.  rank %d tile %d offset %dx%d, size %dx%dx%d, elemSize %d type %d encoding %d, tile bytes %ld at %ld, imagename %ld at %ld\n", local_rank, i, tileOffsetX[i], tileOffsetY[i], tileSizeX[i], tileSizeY[i], channels[i], elemSize1[i], cvDataType[i], encoding[i], tile_size[i], tile_offset[i], imageName_size[i], imageName_offset[i]);
	}


	/**
	* compute the total written out within this step, then update global total
	*/
	long step_totals[3];
	step_totals[0] = 0;
	step_totals[1] = 0;
	step_totals[2] = 0;
	// get the max inclusive scan result from all workers
	MPI_Allreduce(pg_sizes, step_totals, 3, MPI_LONG, MPI_SUM, *local_comm);
	this->tileInfo_total += step_totals[0];
	this->imageName_total += step_totals[1];
	this->tile_total += step_totals[2];


	/** tracking information for this process	
	*/
	this->pg_tileInfo_count += tileInfo_pg_size;
	this->pg_imageName_bytes += imageName_pg_size;
	this->pg_tile_bytes += tile_pg_size;
printf("totals %ld, %ld, %ld, proc %d total %ld, %ld, %ld\n", this->tileInfo_total, this->imageName_total, this->tile_total, local_rank, this->pg_tileInfo_count, this->pg_imageName_bytes, this->pg_tile_bytes);

	printf("chunk size: %ld of total %ld at offset %ld\n", tileInfo_pg_size, tileInfo_capacity, tileInfo_pg_offset);


	/**  write out the TileInfo group 
	*/
	open("tileInfo");
	if (tile_cache.size() <= 0) {
		err = adios_group_size (adios_handle, 0, &adios_totalsize);

	} else {
#include "gwrite_tileInfo.ch"

	}
	close(1);  // uses matcache.size();



	/** now clean up
	*/
	free(tileOffsetX);
	free(tileOffsetY);
	free(tileSizeX);
	free(tileSizeY);
	free(channels);
	free(elemSize1);
	free(cvDataType);
	free(encoding);
	free(imageName_offset);
	free(imageName_size);
	free(tile_offset);
	free(tile_size);
	free(imageNames);
	free(tiles);
	
	tile_cache.clear();

	return 0;
}

int SCIOADIOSWriter::persistCountInfo() {
	/** then write out the tileCount group
	*/
	uint64_t adios_groupsize, adios_totalsize;

	open("tileCount");
#include "gwrite_tileCount.ch"
	close(0);

	return 0;
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







