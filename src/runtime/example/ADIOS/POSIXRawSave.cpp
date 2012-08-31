/*
 * POSIXRawSave.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "POSIXRawSave.h"
#include "Debug.h"
#include "mpi.h"

#include "opencv2/opencv.hpp"
#include "CVImage.h"
#include "UtilsADIOS.h"
#include <string>
#include "FileUtils.h"

namespace cci {
namespace rt {
namespace adios {


POSIXRawSave::POSIXRawSave(MPI_Comm const * _parent_comm, int const _gid,
		std::string &outDir, std::string &iocode, int total, int _buffer_max,
		int tile_max, int imagename_max, int filename_max,
		cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _logsession),
		local_iter(0), local_total(0),
		c(0), outdir(outDir),
		buffer_max(_buffer_max) {

	// always overwrite.
	bool overwrite = true;

	// and the stages to capture.
	std::vector<int> stages;
	for (int i = 0; i < 200; i++) {
		stages.push_back(i);
	}

	size_t pos = outdir.find_last_not_of('/');
	if (pos != std::string::npos) {
		outdir.erase(pos+1);
		if (rank == 0) {
			FileUtils fu;
			fu.mkdirs(outdir);
			}
	} else {
		outdir.clear(); // outdir is "/".
	}
}

POSIXRawSave::~POSIXRawSave() {
	Debug::print("%s destructor called.  total written out is %d\n", getClassName(), local_total);

}

int POSIXRawSave::run() {

	long long t1, t2;

	t1 = ::cciutils::event::timestampInUS();

	int max_iter = 0;
	int active = 0;

	int status = input_status;

	int buffer[2], gbuffer[2];

//	if (test_input_status == DONE)
//		Debug::print("TEST start input status = %d\n", input_status);


	// first get the local states - done or not done.
	if (input_status == DONE || input_status == ERROR) {
		buffer[0] = 0;
		c++;
		status = WAIT;  // pre-initialize to WAIT.
	} else {
		buffer[0] = 1;
	}

	// next predict the local iterations.  write either when full, or when done.
	if ((inputSizes.size() >= buffer_max && (input_status == READY || input_status == WAIT)) ||
			(inputSizes.size() > 0 && (input_status == DONE || input_status == ERROR))) {
		// not done and has full buffer, or done and has some data
		// increment self and accumulate
		buffer[1] = local_iter + 1;
	} else {
		buffer[1] = local_iter;
	}

	MPI_Allreduce(buffer, gbuffer, 2, MPI_INT, MPI_MAX, comm);

//	if (test_input_status == DONE)
//		Debug::print("TEST 1 input status = %d\n", input_status);


	if (gbuffer[0] == 0) {
		status = DONE;
	}
	max_iter = gbuffer[1];


	if (input_status == DONE) Debug::print("%s call_count = %ld, input_status = %d, status = %d, max_iter = %d, local_iter = %d, buffer size = %ld\n", getClassName(), c, input_status, status, max_iter, local_iter, inputSizes.size());

//	if (test_input_status == DONE)
//		Debug::print("TEST 2 input status = %d\n", input_status);

	t2 = ::cciutils::event::timestampInUS();
	//if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO MPI update iter"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));


	// local_count > 0 && max_iter <= local_iter;  //new data to write
	// local_count == 0 && max_iter > local_iter; //catch up with empty
	// local_count > 0 && max_iter > local_iter;  //catch up with some writes
	// local_count == 0 && max_iter <= local_iter;  // update remote, nothing to write locally.

	//	Debug::print("%s rank %d max iter = %d, local_iter = %d\n", getClassName(), rank, max_iter, local_iter);

	/**
	 *  catch up.
	 */
	// catch up.  so flush whatever's in buffer.
	while (max_iter > local_iter) {
		Debug::print("%s write out: IO group %d rank %d, write iter %d, max_iter = %d, tile count %d\n", getClassName(), groupid, rank, local_iter, max_iter, inputSizes.size());
		process();
	}

//	if (test_input_status == DONE)
//		Debug::print("TEST end input status = %d\n", input_status);

	//Debug::print("%s rank %d returning input status %d\n", getClassName(), rank, input_status);
	return status;
}

int POSIXRawSave::process() {
	long long t1, t2;
	t1 = ::cciutils::event::timestampInUS();

	// move data from action's buffer to adios' buffer

	int input_size;  // allocate output vars because these are references
	void *input;
	int result = getInput(input_size, input);

	while (input != NULL) {
		CVImage *in_img = new CVImage(input_size, input);

		int datasize;
		int namesize;
		int maxsize;
		const unsigned char * data = in_img->getData(maxsize, datasize);
		const char* imgname = in_img->getImageName(maxsize, namesize);


		// write out as raw
		std::stringstream ss;
		ss << outdir << "/" << imgname << "_tile_";
		ss << in_img->getMetadata().info.x_offset << "x" << in_img->getMetadata().info.y_offset;
		ss << "_type" << in_img->getMetadata().info.cvDataType;
		ss << "_out.raw";

		FILE* fid = fopen(ss.str().c_str(), "wb");
		if (!fid) {
			printf("ERROR: can't open %s to write\n", ss.str().c_str());
		} else {
			fwrite(data, 1, datasize, fid);
			fclose(fid);
		}

		delete in_img;
		free(input);
		input = NULL;

		++local_total;

		result = getInput(input_size, input);
	}

	++local_iter;

	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO POSIX Write"), t1, t2, std::string(), ::cciutils::event::FILE_O));

	return 0;
}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
