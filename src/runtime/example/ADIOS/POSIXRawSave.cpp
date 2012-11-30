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
#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace adios {


POSIXRawSave::POSIXRawSave(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		boost::program_options::variables_map &_vm,
		cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _input, _output, _logsession),
		local_iter(0), local_total(0) {

	assert(_input != NULL);

	outdir = cci::rt::CmdlineParser::getParamValueByName<std::string>(_vm, cci::rt::CmdlineParser::PARAM_OUTPUTDIR);

	// always overwrite.
	bool overwrite = true;

	// and the stages to capture.
	std::vector<int> stages;
	for (int i = 0; i < 200; i++) {
		stages.push_back(i);
	}

	size_t pos = outdir.rfind('/');
	if (pos == outdir.length() - 1) {
		if (outdir.length() == 1) {
			// outdir is "/"
			outdir.clear();
		} else {
			outdir.erase(pos+1);  // last character is '/'.  remove it for good measure.
		}
	} // else "/" is somewhere else, or not present.  nothing to do.


}

POSIXRawSave::~POSIXRawSave() {
	Debug::print("%s destructor:  wrote out %d over %d iters\n", getClassName(), local_total, local_iter);

}

int POSIXRawSave::run() {

	long long t1, t2;

	t1 = ::cciutils::event::timestampInUS();

	int max_iter = 0;

	// status is set to WAIT or READY, since we can be DONE only if everyone is DONE
	int status = (this->inputBuf->canPop() ? Communicator_I::READY : Communicator_I::WAIT );

	int buffer[2], gbuffer[2];

//	if (test_input_status == DONE)
//		Debug::print("TEST start input status = %d\n", input_status);


	// first get the local states - done or not done.
	if (this->inputBuf->isFinished()) {
		buffer[0] = 0;
	} else {
		buffer[0] = 1;
	}

	// next predict the local iterations.  write either when full, or when done.
	if (this->inputBuf->isFull() ||
			(this->inputBuf->canPop() && this->inputBuf->isStopped())) {
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
		status = Communicator_I::DONE;
	}
	max_iter = gbuffer[1];


	//printf("%s call_count = %ld, status = %d, max_iter = %d, local_iter = %d, buffer size = %d\n", getClassName(), c, status, max_iter, local_iter, inputBuf->debugBufferSize());
//	if (status == Communicator_I::DONE) {
//		Debug::print("%s call_count = %ld, status = %d, max_iter = %d, local_iter = %d, buffer size = %d\n", getClassName(), c, status, max_iter, local_iter, inputBuf->debugBufferSize());
//	}



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
		//Debug::print("%s write out: IO group %d rank %d, write iter %d, max_iter = %d, tile count %d\n", getClassName(), groupid, rank, local_iter, max_iter, inputBuf->debugBufferSize());
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
	DataBuffer::DataType data;
	int input_size;  // allocate output vars because these are references
	void *input;
	int result;

	long output_size = 0;
	while (this->inputBuf->canPop()) {
		result = this->inputBuf->pop(data);
		input_size = data.first;
		input = data.second;

		if (input != NULL) {

			CVImage *in_img = new CVImage(input_size, input);

			int datasize;
			int namesize;
			int maxsize;
			const unsigned char * data = in_img->getData(maxsize, datasize);
			std::string sourcefn(in_img->getSourceFileName(maxsize, namesize));
			std::string tmpfn = FileUtils::replaceDir(sourcefn, FileUtils::getDir(sourcefn), outdir);
			std::string outfn = FileUtils::replaceExt(tmpfn, FileUtils::getExt(tmpfn), "out.raw");

//			printf("FILESNAMES: source %s, temp %s, out %s\n", sourcefn.c_str(), tmpfn.c_str(), outfn.c_str());

			// write out as raw

			FILE* fid = fopen(outfn.c_str(), "r");
			if (!fid) {
				//printf("INFO: creating file %s for writing.\n", outfn.c_str());
			} else {
				fclose(fid);
			}

			fid = fopen(outfn.c_str(), "wb");
			if (!fid) {
				printf("ERROR: can't open %s to write\n", outfn.c_str());
			} else {
				fwrite(data, 1, datasize, fid);
				output_size += datasize;
				fclose(fid);
			}

			delete in_img;
			free(input);
			input = NULL;

			++local_total;
		} else {
			Debug::print("ERROR: %s NULL INPUT from buffer!!!\n", getClassName());
		}
	}

	++local_iter;

	t2 = ::cciutils::event::timestampInUS();
	char len[21];
	memset(len, 0, 21);
	sprintf(len, "%ld", output_size);
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO POSIX Write"), t1, t2, std::string(len), ::cciutils::event::FILE_O));

	return 0;
}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
