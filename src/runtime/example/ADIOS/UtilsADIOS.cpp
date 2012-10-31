#include "UtilsADIOS.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "gpu_utils.h"
#include "adios.h"
#include "adios_internals.h"
#include "Debug.h"

namespace cci {
namespace rt {
namespace adios {


ADIOSManager::ADIOSManager(const char* configfilename, std::string const &_transport,
		int _rank, MPI_Comm &_comm, cciutils::SCIOLogSession *session, bool _gapped, bool _grouped ) :
		transport(_transport) {
	gapped = _gapped;
	grouped = _grouped;

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

	for (std::vector<ADIOSWriter *>::iterator iter = writers.begin();
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

ADIOSWriter* ADIOSManager::allocateWriter(
	std::string const &pref, std::string const &suf, bool _newfile,
	bool _appendInTime, std::vector<int> const &selStages,
	int max_image_count, int local_image_count,
	int mx_image_bytes, int mx_imagename_bytes, int mx_sourcetilefile_bytes,
	MPI_Comm const &_local_comm, int _local_group) {

	long long t1 = ::cciutils::event::timestampInUS();


	ADIOSWriter *w = new ADIOSWriter(pref, suf, _newfile,
			_appendInTime, selStages,
			max_image_count, local_image_count, this->gapped,
			mx_image_bytes, mx_imagename_bytes, mx_sourcetilefile_bytes,
			_local_comm, this->grouped, _local_group);

	w->transport = this->transport;

	writers.push_back(w);

	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS Writer alloc"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

//	if (w->local_rank == 0) printf("INITIALIZED group %d for %s with tileinfo %ld, imagename %ld, sourcetile %ld, tile %ld\n", w->local_group, pref.c_str(), w->tileInfo_capacity, w->imageName_capacity, w->sourceTileFile_capacity, w->tile_capacity);

	return w;
}


void ADIOSManager::freeWriter(ADIOSWriter *w) {
	long long t1 = ::cciutils::event::timestampInUS();

	std::vector<ADIOSWriter *>::iterator newend = remove(writers.begin(), writers.end(), w);
	writers.erase(newend, writers.end());

	delete w;
//	printf("cleaned up writer %d \n", w->local_rank);
	long long t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS Writer free"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

}



ADIOSWriter::ADIOSWriter(std::string const &_prefix, std::string const &_suffix, bool _newfile,
	bool _appendInTime, std::vector<int> const &_selected_stages,
	int _mx_image_capacity, int _image_buffer_capacity, bool _gapped,
	int _mx_image_bytes, int _mx_imagename_bytes, int _mx_filename_bytes,
	MPI_Comm const &_comm, bool _grouped, int _comm_group) :
		prefix(_prefix), suffix(_suffix), newfile(_newfile),
		appendInTime(_appendInTime), selected_stages(_selected_stages),
		tileInfo_capacity(_mx_image_capacity), tileInfo_buffer_capacity(_image_buffer_capacity), gapped(_gapped),
		mx_image_bytes(_mx_image_bytes), mx_imagename_bytes(_mx_imagename_bytes), mx_filename_bytes( _mx_filename_bytes),
		comm(_comm), grouped(_grouped), comm_group(_comm_group),

		adios_handle(-1),
//		pg_tileInfo_count(0), pg_imageName_bytes(0), pg_filename_bytes(0), pg_image_bytes(0),
		tileInfo_total(0), imageName_total(0), sourceTileFile_total(0), tile_total(0),
		write_session_id(0),
		data_pos(0), imageNames_pos(0), filenames_pos(0),
		logsession(NULL) {

	std::sort(selected_stages.begin(), selected_stages.end());

	MPI_Comm_rank(_comm, &comm_rank);
	MPI_Comm_size(_comm, &comm_size);

	imageName_capacity = mx_imagename_bytes * tileInfo_capacity;
	sourceTileFile_capacity = mx_filename_bytes * tileInfo_capacity;
	tile_capacity = mx_image_bytes * tileInfo_capacity;

	local_tile_capacity = mx_image_bytes * tileInfo_buffer_capacity;
	local_imagename_capacity = mx_imagename_bytes * tileInfo_buffer_capacity;
	local_sourceTileFile_capacity = mx_filename_bytes * tileInfo_buffer_capacity;

	tile = (unsigned char*)calloc(local_tile_capacity, sizeof(unsigned char));
	Debug::print("Local_tile_buffer at %p with capacity %d\n", tile, local_tile_capacity);
	imageName = (char*)calloc(local_imagename_capacity, sizeof(char));
	sourceTileFile = (char*)calloc(local_sourceTileFile_capacity, sizeof(char));
	this->buffer.clear();

}
ADIOSWriter::~ADIOSWriter() {
	MPI_Barrier(comm);

	free(tile);
	free(imageName);
	free(sourceTileFile);

	for (std::vector<CVImage *>::iterator iter = buffer.begin();
			iter!= buffer.end(); ++iter) {
		delete *iter;
	}
	buffer.clear();

	selected_stages.clear();
}


bool ADIOSWriter::selected(const int stage) {
	return std::binary_search(selected_stages.begin(), selected_stages.end(), stage);
}

void ADIOSWriter::clearBuffer() {
	memset(this->tile, 0, this->local_tile_capacity);
	memset(this->imageName, 0, this->local_imagename_capacity);
	memset(this->sourceTileFile, 0, this->local_sourceTileFile_capacity);
	this->data_pos = 0;
	this->imageNames_pos = 0;
	this->filenames_pos = 0;

	for (std::vector<CVImage *>::iterator iter = buffer.begin();
			iter!= buffer.end(); ++iter) {
		delete *iter;
	}
	this->buffer.clear();
}


CVImage *ADIOSWriter::saveCVImage(CVImage const *img) {
	CVImage::MetadataType *meta = CVImage::allocMetadata();

//	printf("tile at %p, data pos=%ld, max data=%d\n", tile, data_pos, mx_image_bytes);

	CVImage *out = new CVImage(meta, tile + data_pos, mx_image_bytes,
			imageName + imageNames_pos, mx_imagename_bytes,
			sourceTileFile + filenames_pos, mx_filename_bytes);

	out->copy(*img);


	int data_size, img_name_size, src_fn_size;
	int dummy, mx_img_name_size, mx_src_fn_size;
	out->compact();

	out->getData(dummy, data_size);
	data_pos += data_size;
	out->getImageName(mx_img_name_size, img_name_size);
	imageNames_pos += mx_img_name_size;
	out->getSourceFileName(mx_img_name_size, src_fn_size);
	filenames_pos += mx_img_name_size;

	buffer.push_back(out);

	CVImage::freeMetadata(meta);

	return out;
}



int ADIOSWriter::open(const char* groupName) {

	int err;

	this->write_session_id++;
	std::stringstream ss;
	if (this->appendInTime == true) {
		if (this->grouped)
			ss << this->prefix << "/" << transport << ".g" << this->comm_group << "." << this->suffix;
		else
			ss << this->prefix << "/" << transport << "." << this->suffix;

//		printf("opening %s for time appending writing\n", ss.str().c_str());

		if (newfile) {
			err = adios_open(&adios_handle, groupName, ss.str().c_str(), "w", &comm);
			newfile = false;
		} else {
			err = adios_open(&adios_handle, groupName, ss.str().c_str(), "a", &comm);
		}
	} else {
		if (this->grouped)
			ss << this->prefix << "/" << transport << ".g" << this->comm_group << "-t"<< this->write_session_id << "." << this->suffix;
		else
			ss << this->prefix << "/" << transport << ".t" << this->write_session_id << "." << this->suffix;

//		printf("opening %s for time separated writing\n", ss.str().c_str());

		err = adios_open(&adios_handle, groupName, ss.str().c_str(), "w", &comm);
	}

	return err;
}

int ADIOSWriter::close(uint32_t time_index) {

	// if time_index is not specified, then let ADIOS handle it.
    struct adios_file_struct * fd = (struct adios_file_struct *) adios_handle;
    struct adios_group_struct *gd = (struct adios_group_struct *) fd->group;
		
//		printf("rank %d, group name %s, id %u, membercount %u, offset %lu, timeindex %u, proc id %u\n", comm_rank, gd->name, gd->id, gd->member_count, gd->group_offset, gd->time_index, gd->process_id);
//		printf("rank %d, file datasize %lu, writesizebytes %lu, pgstart %lu, baseoffset %lu, offset %lu, bytewritten %lu, bufsize %lu\n", comm_rank, fd->data_size, fd->write_size_bytes, fd->pg_start_in_file, fd->base_offset, fd->offset, fd->bytes_written, fd->buffer_size);
	if (time_index > 0) {
		gd->time_index = time_index;
	}

	int err = adios_close(adios_handle);

	return err;
}



int ADIOSWriter::persist(int iter) {

	std::stringstream ss;

	//printf("worker %d writing out %lu tiles to ADIOS\n", comm_rank, tile_cache.size());
	long long t1 = ::cciutils::event::timestampInUS();
	MPI_Barrier(comm);
	long long t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "IO MPI Wait";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));
	ss.str(std::string());


	t1 = ::cciutils::event::timestampInUS();

	int err;
	uint64_t adios_groupsize, adios_totalsize;

	/**
	*  first set up the index variables.
	*/
	long tileInfo_pg_size = buffer.size();
	long imageName_pg_size = 0;
	long sourceTileFile_pg_size = 0;
	long tile_pg_size = 0;
	// capacity variables already set.
	// data already copied

	/**
	* gather specific data for the tile in the process
	*/
	int *tileOffsetX, *tileOffsetY, *tileSizeX, *tileSizeY, *nChannels,
		*elemSize1, *cvDataType, *encoding;
	long *imageName_offset, *imageName_size,
		*sourceTileFile_offset, *sourceTileFile_size,
		*tile_offset, *tile_size;

	t2 = ::cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%lu", tileInfo_pg_size);
	ss << event_name_prefix << "IO define vars";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::MEM_IO));
	ss.str(std::string());


	if (tileInfo_pg_size > 0) {

		t1 = ::cciutils::event::timestampInUS();
		/** initialize storage
		*/ 
		tileOffsetX = new int[tileInfo_pg_size];
		tileOffsetY = new int[tileInfo_pg_size];
		tileSizeX = new int[tileInfo_pg_size];
		tileSizeY = new int[tileInfo_pg_size];
		nChannels = new int[tileInfo_pg_size];
		elemSize1 = new int[tileInfo_pg_size];
		cvDataType = new int[tileInfo_pg_size];
		encoding = new int[tileInfo_pg_size];
		
		imageName_size = new long[tileInfo_pg_size];
		imageName_offset = new long[tileInfo_pg_size];
		sourceTileFile_size = new long[tileInfo_pg_size];
		sourceTileFile_offset = new long[tileInfo_pg_size];
		tile_size = new long[tileInfo_pg_size];
		tile_offset = new long[tileInfo_pg_size];

		t2 = ::cciutils::event::timestampInUS();
		ss << event_name_prefix << "IO malloc vars";
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::MEM_IO));
		ss.str(std::string());

		t1 = ::cciutils::event::timestampInUS();

		/**  get tile metadata
		*/
		for (int i = 0; i < tileInfo_pg_size; ++i) {
			CVImage::MetadataType md = buffer[i]->getMetadata();
			tileOffsetX[i] = md.info.x_offset;
			tileOffsetY[i] = md.info.y_offset;
			tileSizeX[i] = md.info.x_size;
			tileSizeY[i] = md.info.y_size;
			nChannels[i] = md.info.nChannels;
			elemSize1[i] = md.info.elemSize1;
			cvDataType[i] = md.info.cvDataType;
			encoding[i] = md.info.encoding;

			imageName_size[i] = (long)(md.info.image_name_size);
			sourceTileFile_size[i] = (long)(md.info.source_file_name_size);
			tile_size[i] = (long)(md.info.data_size);

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

//			printf("rank %d tile %d offset %dx%d, size %dx%dx%d, elemSize %d type %d encoding %d, tile bytes %ld at %ld, imagename %ld at %ld\n", comm_rank, i, tileOffsetX[i], tileOffsetY[i], tileSizeX[i], tileSizeY[i], nChannels[i], elemSize1[i], cvDataType[i], encoding[i], tile_size[i], tile_offset[i], imageName_size[i], imageName_offset[i]);
		}
		t2 = ::cciutils::event::timestampInUS();
		ss << event_name_prefix << "IO tile metadata";
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::MEM_IO));
		ss.str(std::string());

		// actual data already copied and is in the buffer.
	}

	/**
	* compute the offset for each step, in global array coordinates
	*/
	t1 = ::cciutils::event::timestampInUS();
	long tileInfo_pg_offset, imageName_pg_offset, sourceTileFile_pg_offset, tile_pg_offset;
	long pg_sizes[4], pg_offsets[4], step_totals[4];

	if (gapped) {
		tileInfo_pg_offset = comm_rank * tileInfo_buffer_capacity + tileInfo_total;
		imageName_pg_offset = comm_rank * local_imagename_capacity + imageName_total;
		sourceTileFile_pg_offset = comm_rank * local_sourceTileFile_capacity + sourceTileFile_total;
		tile_pg_offset = comm_rank * local_tile_capacity + tile_total;
	} else {
		// compute offset within step across all processes
		pg_sizes[0] = tileInfo_pg_size;
		pg_sizes[1] = imageName_pg_size;
		pg_sizes[2] = sourceTileFile_pg_size;
		pg_sizes[3] = tile_pg_size;
		pg_offsets[0] = 0;
		pg_offsets[1] = 0;
		pg_offsets[2] = 0;
		pg_offsets[3] = 0;
		MPI_Scan(pg_sizes, pg_offsets, 4, MPI_LONG, MPI_SUM, comm);

		// convert to offset in global coord, across all processes for this time steps, for each process
		tileInfo_pg_offset = pg_offsets[0] - tileInfo_pg_size + this->tileInfo_total;
		imageName_pg_offset = pg_offsets[1] - imageName_pg_size + this->imageName_total;
		sourceTileFile_pg_offset = pg_offsets[2] - sourceTileFile_pg_size + this->sourceTileFile_total;
		tile_pg_offset = pg_offsets[3] - tile_pg_size + this->tile_total;
	}

	// update the offsets within the process group for each data element
	for (int i = 0; i < tileInfo_pg_size; ++ i) {
		imageName_offset[i] += imageName_pg_offset;
		sourceTileFile_offset[i] += sourceTileFile_pg_offset;
		tile_offset[i] += tile_pg_offset;
//		printf("globally shifted.  rank %d tile %d offset %dx%d, size %dx%dx%d, elemSize %d type %d encoding %d, tile bytes %ld at %ld, imagename %ld at %ld, sourceTileFile %ld at %ld\n",
//				comm_rank, i, tileOffsetX[i], tileOffsetY[i], tileSizeX[i], tileSizeY[i],
//				nChannels[i], elemSize1[i], cvDataType[i], encoding[i], tile_size[i],
//				tile_offset[i], imageName_size[i], imageName_offset[i], sourceTileFile_size[i], sourceTileFile_offset[i]);
	}

	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "IO MPI scan";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string("1"), ::cciutils::event::NETWORK_IO));
	ss.str(std::string());
	t1 = ::cciutils::event::timestampInUS();

	/**
	* compute the total written out within this step, then update global total
	*/
	if (gapped) {
		this->tileInfo_total = (write_session_id + 1) * comm_size * tileInfo_buffer_capacity;
		this->imageName_total= (write_session_id + 1) * comm_size * local_imagename_capacity;
		this->sourceTileFile_total = (write_session_id + 1) * comm_size * local_sourceTileFile_capacity;
		this->tile_total = (write_session_id + 1) * comm_size * local_tile_capacity;
	} else {
		step_totals[0] = 0;
		step_totals[1] = 0;
		step_totals[2] = 0;
		step_totals[3] = 0;
		// get the max inclusive scan result from all workers
		MPI_Allreduce(pg_sizes, step_totals, 4, MPI_LONG, MPI_SUM, comm);
		this->tileInfo_total += step_totals[0];
		this->imageName_total += step_totals[1];
		this->sourceTileFile_total += step_totals[2];
		this->tile_total += step_totals[3];
	}

	/** tracking information for this process	
	*/
//	this->pg_tileInfo_count += tileInfo_pg_size;
//	this->pg_imageName_bytes += imageName_pg_size;
//	this->pg_filename_bytes += sourceTileFile_pg_size;
//	this->pg_image_bytes += tile_pg_size;
//	printf("totals %ld, %ld, %ld, %ld, proc %d total %ld, %ld, %ld\n", this->tileInfo_total, this->imageName_total, this->sourceTileFile_total, this->tile_total, comm_rank, this->pg_tileInfo_count, this->pg_imageName_bytes, this->pg_filename_bytes, this->pg_image_bytes);


	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "IO MPI allreduce";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string("1"), ::cciutils::event::NETWORK_IO));
	ss.str(std::string());


	/**  write out the TileInfo group 
	*/
//	printf("data prepared.  writing out.\n");
	t1 = ::cciutils::event::timestampInUS();
	open("tileInfo");  // also increments the write_session_id
	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "adios open";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(), ::cciutils::event::ADIOS_OPEN));
	ss.str(std::string());

	t1 = ::cciutils::event::timestampInUS();

	if (tileInfo_pg_size <= 0) {
		err = adios_group_size (adios_handle, 0, &adios_totalsize);

	} else {
#include "gwrite_tileInfo.ch"
	}
	t2 = ::cciutils::event::timestampInUS();
	memset(len, 0, 21);
	sprintf(len, "%lu", adios_totalsize);
	ss << event_name_prefix << "ADIOS WRITE";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::ADIOS_WRITE));
	ss.str(std::string());

	t1 = ::cciutils::event::timestampInUS();
	close(1);
	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "adios close";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::ADIOS_CLOSE));
	ss.str(std::string());


	/** now clean up
	*/
	t1 = ::cciutils::event::timestampInUS();

	if (tileInfo_pg_size > 0) {
		delete [] tileOffsetX;
		delete [] tileOffsetY;
		delete [] tileSizeX;
		delete [] tileSizeY;
		delete [] nChannels;
		delete [] elemSize1;
		delete [] cvDataType;
		delete [] encoding;
		delete [] imageName_offset;
		delete [] imageName_size;
		delete [] sourceTileFile_offset;
		delete [] sourceTileFile_size;
		delete [] tile_offset;
		delete [] tile_size;
	}
// 	this->clearBuffer();

	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "IO var clear";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::MEM_IO));
	ss.str(std::string());

	return write_session_id;
}

int ADIOSWriter::persistCountInfo() {

	/** then write out the tileCount group
	*/
	uint64_t adios_groupsize, adios_totalsize;

	long long t1, t2;

	t1 = ::cciutils::event::timestampInUS();
	open("tileCount");
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("adios open"), t1, t2, std::string(), ::cciutils::event::ADIOS_OPEN));

	t1 = ::cciutils::event::timestampInUS();
#include "gwrite_tileCount.ch"
	t2 = ::cciutils::event::timestampInUS();

	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%lu", adios_totalsize);
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("ADIOS WRITE Summary"), t1, t2, std::string(len), ::cciutils::event::ADIOS_WRITE));

	t1 = ::cciutils::event::timestampInUS();
	close(1);
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("adios close"), t1, t2, std::string(len), ::cciutils::event::ADIOS_CLOSE));

	return 0;
}


int ADIOSWriter::benchmark(int id) {

	long long t1, t2;
	t1 = ::cciutils::event::timestampInUS();


	std::stringstream ss;
	ss << "BENCH " << id << " ";
	event_name_prefix = ss.str();
	ss.str(std::string());

	// prepare the data.
	cv::Mat img = cv::Mat::eye(4096, 4096, CV_32SC1);
	std::string img_name("testimg");
	ss << "BENCHTest" << comm_rank << ".tif";
	CVImage *cvi = new CVImage(img, img_name, ss.str(), 1024, 2048);
	saveCVImage(cvi);
	delete cvi;
	ss.str(std::string());
	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "DATA prep";
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%lu", (long)(img.dataend) - (long)(img.datastart));
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(len), ::cciutils::event::MEM_IO));
	ss.str(std::string());


	//printf("worker %d writing out GAPPED %lu tiles to ADIOS, tileSize = %lu\n", comm_rank, tileInfo_pg_size, tileSize);
	t1 = ::cciutils::event::timestampInUS();
	MPI_Barrier(comm);
	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "START MPI Wait";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));
	ss.str(std::string());

	// now write
	persist(0);

	persistCountInfo();
	
	clearBuffer();

	//printf("worker %d writing out GAPPED %lu tiles to ADIOS, tileSize = %lu\n", comm_rank, tileInfo_pg_size, tileSize);
	t1 = ::cciutils::event::timestampInUS();
	MPI_Barrier(comm);
	t2 = ::cciutils::event::timestampInUS();
	ss << event_name_prefix << "END MPI Wait";
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, ss.str(), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));
	ss.str(std::string());

	return write_session_id;
}




void ADIOSWriter::saveIntermediate(CVImage const *img, const int stage) {
	//if (!selected(stage)) return;

	long long t1 = ::cciutils::event::timestampInUS();
	CVImage *out = saveCVImage(img);

	long long t2 = ::cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	int data_size, dummy;
	out->getData(dummy, data_size);
	sprintf(len, "%ld", (long)(data_size));

	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO Buffer"), t1, t2, std::string(len), ::cciutils::event::MEM_IO));

}

void ADIOSWriter::saveIntermediate(const ::cv::Mat& intermediate, const int stage,
		const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) {

	CVImage *source = new CVImage(intermediate, _image_name, _source_tile_file_name, _offsetX, _offsetY);
	saveIntermediate(source, stage);
	delete source;
}

#if defined (WITH_CUDA)

	void ADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) {
		if (!selected(stage)) return;
		// first download the data
		long long t1 = ::cciutils::event::timestampInUS();

		::cv::Mat output(intermediate.size(), intermediate.type());
		intermediate.download(output);
		long long t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO GPU Download"), t1, t2, std::string("1"), ::cciutils::event::GPU_MEM_IO));

		saveIntermediate(output, stage, _image_name, _offsetX, _offsetY, _source_tile_file_name);
		t1 = ::cciutils::event::timestampInUS();
		output.release();
		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO GPU clear"), t1, t2, std::string("1"), ::cciutils::event::GPU_MEM_IO));

	}
#else
	void ADIOSWriter::saveIntermediate(const ::cv::gpu::GpuMat& intermediate, const int stage,
			const char *_image_name, const int _offsetX, const int _offsetY, const char* _source_tile_file_name) { throw_nogpu(); }
#endif


}  // ns adios
}  // ns rt
}  // ns cci







