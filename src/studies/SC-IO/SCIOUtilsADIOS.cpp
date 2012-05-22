#include "SCIOUtilsADIOS.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "gpu_utils.h"
#include "adios.h"
#include "adios_internals.h"

namespace cciutils {

ADIOSManager::ADIOSManager(const char* configfilename, int _rank, MPI_Comm *_comm, cciutils::SCIOLogSession *session ) {
	rank = _rank;
	comm = _comm;
	logsession = session;
	long long t1 = ::cciutils::event::timestampInUS();
	adios_init(configfilename);
	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("adios init"), t1, t2, std::string(), ::cciutils::event::ADIOS_INIT));

	t1 = ::cciutils::event::timestampInUS();
	writers.clear();
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("clear Writers"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

}

ADIOSManager::~ADIOSManager() {
//	printf("cleaning up manager %d\n", rank);
	// close all the entries in writer
	long long t1 = ::cciutils::event::timestampInUS();

	for (std::vector<SCIOADIOSWriter *>::iterator iter = writers.begin();
			iter != writers.end(); ++iter) {
//		printf("%d found a writer\n", rank);
		freeWriter(*iter);
		iter = writers.begin();
	}
	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS Writer Clear"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

//	printf("cleaned up manager %d\n", rank);
	t1 = ::cciutils::event::timestampInUS();
	adios_finalize(rank);
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS finalize"), t1, t2, std::string(), ::cciutils::event::ADIOS_FINALIZE));
//	printf("finished cleaning up %d\n", rank);

}

SCIOADIOSWriter* ADIOSManager::allocateWriter(const std::string &pref, const std::string &suf,
		const bool _newfile,
		std::vector<int> &selStages,
		const long mx_tileinfo_count, const long mx_imagename_bytes,
		const long mx_sourcetilefile_bytes, const long mx_tile_bytes,
 int _local_rank, MPI_Comm *_local_comm) {


	SCIOADIOSWriter *w = new SCIOADIOSWriter();

	w->selectedStages = selStages;
	std::sort(w->selectedStages.begin(), w->selectedStages.end());
	w->local_rank = _local_rank;
	w->local_comm = _local_comm;

	w->tile_capacity = mx_tile_bytes;
	w->imageName_capacity = mx_imagename_bytes;
	w->sourceTileFile_capacity = mx_sourcetilefile_bytes;
	w->tileInfo_capacity = mx_tileinfo_count;


	std::stringstream ss;
	ss << pref << "." << suf;
	w->filename = ss.str();
	w->newfile = _newfile;
	writers.push_back(w);

	if (w->local_rank == 0) printf("INITIALIZED %s with tileinfo %ld, imagename %ld, sourcetile %ld, tile %ld\n", w->filename.c_str(), w->tileInfo_capacity, w->imageName_capacity, w->sourceTileFile_capacity, w->tile_capacity);

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
	long long t1 = ::cciutils::event::timestampInUS();

	if (newfile) {
		err = adios_open(&adios_handle, groupName, filename.c_str(), "w", local_comm);
		newfile = false;
	} else {
		err = adios_open(&adios_handle, groupName, filename.c_str(), "a", local_comm);
	}
	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("adios open"), t1, t2, std::string(), ::cciutils::event::ADIOS_OPEN));

	return err;
}

int SCIOADIOSWriter::close(uint32_t time_index) {
	long long t1 = ::cciutils::event::timestampInUS();

	// if time_index is not specified, then let ADIOS handle it.
    struct adios_file_struct * fd = (struct adios_file_struct *) adios_handle;
    struct adios_group_struct *gd = (struct adios_group_struct *) fd->group;
		
//		printf("rank %d, group name %s, id %u, membercount %u, offset %lu, timeindex %u, proc id %u\n", local_rank, gd->name, gd->id, gd->member_count, gd->group_offset, gd->time_index, gd->process_id);
//		printf("rank %d, file datasize %lu, writesizebytes %lu, pgstart %lu, baseoffset %lu, offset %lu, bytewritten %lu, bufsize %lu\n", local_rank, fd->data_size, fd->write_size_bytes, fd->pg_start_in_file, fd->base_offset, fd->offset, fd->bytes_written, fd->buffer_size);
	if (time_index > 0) {
		gd->time_index = time_index;
	}

	int err = adios_close(adios_handle);

	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("adios close"), t1, t2, std::string(), ::cciutils::event::ADIOS_CLOSE));

	return err;
}



int SCIOADIOSWriter::persist() {
	// prepare the data.
	// all the data should be continuous now.

	//printf("worker %d writing out %lu tiles to ADIOS\n", local_rank, tile_cache.size());
	long long t1 = ::cciutils::event::timestampInUS();

	int err;
	uint64_t adios_groupsize, adios_totalsize;


	/**
	*  first set up the index variables.
	*/
	long tileInfo_pg_size = tile_cache.size();
	long imageName_pg_size = 0;
	long sourceTileFile_pg_size = 0;
	long tile_pg_size = 0;
	// capacity variables already set.


	/**
	* gather specific data for the tile in the process
	*/
	unsigned char *tile = NULL;
	char *imageName = NULL;
	char *sourceTileFile = NULL;
	int *tileOffsetX, *tileOffsetY, *tileSizeX, *tileSizeY, *nChannels,
		*elemSize1, *cvDataType, *encoding;
	long *imageName_offset, *imageName_size,
		*sourceTileFile_offset, *sourceTileFile_size,
		*tile_offset, *tile_size;

	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO define vars"), t1, t2, std::string(), ::cciutils::event::MEM_IO));


	if (tile_cache.size() > 0) {

		t1 = ::cciutils::event::timestampInUS();
		/** initialize storage
		*/ 
		tileOffsetX = (int *)malloc(tile_cache.size() * sizeof(int));
		tileOffsetY = (int *)malloc(tile_cache.size() * sizeof(int));
		tileSizeX = (int *)malloc(tile_cache.size() * sizeof(int));
		tileSizeY = (int *)malloc(tile_cache.size() * sizeof(int));
		nChannels = (int *)malloc(tile_cache.size() * sizeof(int));
		elemSize1 = (int *)malloc(tile_cache.size() * sizeof(int));
		cvDataType = (int *)malloc(tile_cache.size() * sizeof(int));
		encoding = (int *)malloc(tile_cache.size() * sizeof(int));
		
		imageName_size = (long*) malloc(tile_cache.size() * sizeof(long));
		imageName_offset = (long*) malloc(tile_cache.size() * sizeof(long));
		sourceTileFile_size = (long*) malloc(tile_cache.size() * sizeof(long));
		sourceTileFile_offset = (long*) malloc(tile_cache.size() * sizeof(long));
		tile_size = (long*) malloc(tile_cache.size() * sizeof(long));
		tile_offset = (long*) malloc(tile_cache.size() * sizeof(long));

		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO malloc vars"), t1, t2, std::string(), ::cciutils::event::MEM_IO));


		t1 = ::cciutils::event::timestampInUS();

		/**  get tile metadata
		*/
		for (int i = 0; i < tile_cache.size(); ++i) {
			
			tileOffsetX[i] = tile_cache[i].x_offset;
			tileOffsetY[i] = tile_cache[i].y_offset;
	
			::cv::Mat tm = tile_cache[i].tile;
			tileSizeX[i] = tm.cols;
			tileSizeY[i] = tm.rows;
			nChannels[i] = tm.channels();
			elemSize1[i] = tm.elemSize1();
			cvDataType[i] = tm.type();
			
			encoding[i] = ENC_RAW;

			imageName_size[i] = (long)(tile_cache[i].image_name.size());
			sourceTileFile_size[i] = (long)(tile_cache[i].source_tile_file_name.size());
			tile_size[i] = (long)(tm.dataend) - (long)(tm.datastart);
//			tile_size[i] = 0;

			// update the offset (within the group for this time step)
			// to the size so far.
			// need to update to global coord later.
			imageName_offset[i] = imageName_pg_size;
			sourceTileFile_offset[i] = sourceTileFile_pg_size;
			tile_offset[i] = tile_pg_size;
			
			// update the process group totals
			imageName_pg_size += imageName_size[i];
			sourceTileFile_pg_size += sourceTileFile_size[i];
			tile_pg_size += tile_size[i];


//			printf("rank %d tile %d offset %dx%d, size %dx%dx%d, elemSize %d type %d encoding %d, tile bytes %ld at %ld, imagename %ld at %ld\n", local_rank, i, tileOffsetX[i], tileOffsetY[i], tileSizeX[i], tileSizeY[i], nChannels[i], elemSize1[i], cvDataType[i], encoding[i], tile_size[i], tile_offset[i], imageName_size[i], imageName_offset[i]);
		}
		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO tile metadata"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

		t1 = ::cciutils::event::timestampInUS();

		/** now allocate the space for tile and for imagename
			and then get the data.
		*/
		imageName = (char *)malloc(imageName_pg_size + 1);
		memset(imageName, 0, imageName_pg_size + 1);
		sourceTileFile = (char *)malloc(sourceTileFile_pg_size + 1);
		memset(sourceTileFile, 0, sourceTileFile_pg_size + 1);
		tile = (unsigned char *)malloc(tile_pg_size);
		memset(tile, 0, tile_pg_size);
		
		for (int i = 0; i < tile_cache.size(); ++i) {
			strncpy(imageName + imageName_offset[i], tile_cache[i].image_name.c_str(), imageName_size[i]);
			//printf("filename cp'ed %s\n", tile_cache[i].source_tile_file_name.c_str());
			strncpy(sourceTileFile + sourceTileFile_offset[i], tile_cache[i].source_tile_file_name.c_str(), sourceTileFile_size[i]);
			memcpy(tile + tile_offset[i], tile_cache[i].tile.datastart, tile_size[i]);
		}
		//printf("all filenames together %s\n", sourceTileFile);
		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO tile data"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

	}

	/**
	* compute the offset for each step, in global array coordinates
	*/
	t1 = ::cciutils::event::timestampInUS();

	long pg_sizes[4];
	pg_sizes[0] = tileInfo_pg_size;
	pg_sizes[1] = imageName_pg_size;
	pg_sizes[2] = sourceTileFile_pg_size;
	pg_sizes[3] = tile_pg_size;
	long pg_offsets[4];
	pg_offsets[0] = 0;
	pg_offsets[1] = 0;
	pg_offsets[2] = 0;
	pg_offsets[3] = 0;
	MPI_Scan(pg_sizes, pg_offsets, 4, MPI_LONG, MPI_SUM, *local_comm);
	long tileInfo_pg_offset = pg_offsets[0] - tileInfo_pg_size + this->tileInfo_total;
	long imageName_pg_offset = pg_offsets[1] - imageName_pg_size + this->imageName_total;
	long sourceTileFile_pg_offset = pg_offsets[2] - sourceTileFile_pg_size + this->sourceTileFile_total;
	long tile_pg_offset = pg_offsets[3] - tile_pg_size + this->tile_total;
	// update the offsets within the process group
	for (int i = 0; i < tile_cache.size(); ++ i) {
		imageName_offset[i] += imageName_pg_offset;
		sourceTileFile_offset[i] += sourceTileFile_pg_offset;
		tile_offset[i] += tile_pg_offset;
//		printf("globally shifted.  rank %d tile %d offset %dx%d, size %dx%dx%d, elemSize %d type %d encoding %d, tile bytes %ld at %ld, imagename %ld at %ld, sourceTileFile %ld at %ld\n",
//				local_rank, i, tileOffsetX[i], tileOffsetY[i], tileSizeX[i], tileSizeY[i],
//				nChannels[i], elemSize1[i], cvDataType[i], encoding[i], tile_size[i],
//				tile_offset[i], imageName_size[i], imageName_offset[i], sourceTileFile_size[i], sourceTileFile_offset[i]);
	}

	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO MPI scan"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));
	t1 = ::cciutils::event::timestampInUS();

	/**
	* compute the total written out within this step, then update global total
	*/
	long step_totals[4];
	step_totals[0] = 0;
	step_totals[1] = 0;
	step_totals[2] = 0;
	step_totals[3] = 0;
	// get the max inclusive scan result from all workers
	MPI_Allreduce(pg_sizes, step_totals, 4, MPI_LONG, MPI_SUM, *local_comm);
	this->tileInfo_total += step_totals[0];
	this->imageName_total += step_totals[1];
	this->sourceTileFile_total += step_totals[2];
	this->tile_total += step_totals[3];


	/** tracking information for this process	
	*/
	this->pg_tileInfo_count += tileInfo_pg_size;
	this->pg_imageName_bytes += imageName_pg_size;
	this->pg_sourceTileFile_bytes += sourceTileFile_pg_size;
	this->pg_tile_bytes += tile_pg_size;
//	printf("totals %ld, %ld, %ld, %ld, proc %d total %ld, %ld, %ld\n", this->tileInfo_total, this->imageName_total, this->sourceTileFile_total, this->tile_total, local_rank, this->pg_tileInfo_count, this->pg_imageName_bytes, this->pg_sourceTileFile_bytes, this->pg_tile_bytes);

	//printf("chunk size: %ld of total %ld at offset %ld\n", tileInfo_pg_size, tileInfo_capacity, tileInfo_pg_offset);

	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO MPI allreduce"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));


	/**  write out the TileInfo group 
	*/
	open("tileInfo");
	t1 = ::cciutils::event::timestampInUS();

	if (tile_cache.size() <= 0) {
		err = adios_group_size (adios_handle, 0, &adios_totalsize);

	} else {
#include "gwrite_tileInfo.ch"

	}
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS WRITE"), t1, t2, std::string(), ::cciutils::event::ADIOS_WRITE));

	close(1);  // uses matcache.size();



	/** now clean up
	*/
	t1 = ::cciutils::event::timestampInUS();

	if (tile_cache.size() > 0) {
		free(tileOffsetX);
		free(tileOffsetY);
		free(tileSizeX);
		free(tileSizeY);
		free(nChannels);
		free(elemSize1);
		free(cvDataType);
		free(encoding);
		free(imageName_offset);
		free(imageName_size);
		free(sourceTileFile_offset);
		free(sourceTileFile_size);
		free(tile_offset);
		free(tile_size);
		free(imageName);
		free(sourceTileFile);
		free(tile);
	}
	tile_cache.clear();
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO var clear"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

	return 0;
}

int SCIOADIOSWriter::persistCountInfo() {
	/** then write out the tileCount group
	*/
	uint64_t adios_groupsize, adios_totalsize;

	open("tileCount");
	long long t1 = ::cciutils::event::timestampInUS();
#include "gwrite_tileCount.ch"
	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS WRITE Summary"), t1, t2, std::string(), ::cciutils::event::ADIOS_WRITE));

	close(0);

	return 0;
}


void SCIOADIOSWriter::saveIntermediate(const ::cv::Mat& intermediate, const int stage,
		const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) {
	if (!selected(stage)) return;

	long long t1 = ::cciutils::event::timestampInUS();

	unsigned char *data;

	// make a copy just in case "out" is released elsewhere.
	// TODO: this may be made more efficient.
	::cv::Mat out = ::cv::Mat::zeros(intermediate.size(), intermediate.type());
	intermediate.copyTo(out);

	Tile t;
	t.image_name.assign(_image_name);
	t.x_offset = _offsetX;
	t.y_offset = _offsetY;
	t.source_tile_file_name.assign(_source_tile_file_name);
	t.tile = out;

	// next add to cache
	tile_cache.push_back(t);
	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO Buffer"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

}

#if defined (WITH_CUDA)

	void SCIOADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) {
		if (!selected(stage)) return;
		// first download the data
		long long t1 = ::cciutils::event::timestampInUS();

		::cv::Mat output(intermediate.size(), intermediate.type());
		intermediate.download(output);
		long long t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO GPU Download"), t1, t2, std::string(), ::cciutils::event::GPU_MEM_IO));
		saveIntermediate(output, stage, _image_name, _offsetX, _offsetY, _source_tile_file_name);
		t1 = ::cciutils::event::timestampInUS();

		output.release();
		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO GPU clear"), t1, t2, std::string(), ::cciutils::event::GPU_MEM_IO));

	}
#else
	void SCIOADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) { throw_nogpu(); }
#endif


}







