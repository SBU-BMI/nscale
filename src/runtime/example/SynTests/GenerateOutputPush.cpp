/*
 * GenerateOutputPush.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "GenerateOutputPush.h"
#include "Debug.h"
#include "opencv2/opencv.hpp"
#include "CVImage.h"
#include "FileUtils.h"
#include <string>
#include "utils.h"
#include "SCIOHistologicalEntities.h"
#include <unistd.h>
#include "CVImage.h"

namespace cci {
namespace rt {
namespace syntest {

GenerateOutputPush::GenerateOutputPush(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		std::string &proctype, int imagedim, int imagecount, int gpuid, bool _compress,
		cciutils::SCIOLogSession *_logsession) :
				Action_I(_parent_comm, _gid, _input, _output, _logsession), output_count(0), output_dim(imagedim), count(imagecount), compress(_compress) {

	if (strcmp(proctype.c_str(), "cpu")) proc_code = cciutils::DEVICE_CPU;
	else if (strcmp(proctype.c_str(), "gpu")) {
		proc_code = cciutils::DEVICE_GPU;
	}
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

}

GenerateOutputPush::~GenerateOutputPush() {
	Debug::print("%s destructor called. %d messages\n", getClassName(), output_count);
}

int GenerateOutputPush::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (output_count >= count) return Communicator_I::DONE;

	long long t1, t2;

	char len[21];
	int tilex = 0;
	int tiley = 0;
	char inchars[21];
	char fnchars[21];
	memset(inchars, 0, 21);
	memset(fnchars, 0, 21);
	sprintf(inchars, "%dx%d", world_rank, count);
	sprintf(fnchars, "/tmp/dummy/synthetic_%dx%d_%d.tiff", world_rank, count, output_count);
	string imagename = string(inchars);
	string fn = string(fnchars);

	t1 = ::cciutils::event::timestampInUS();

	// real computation:
	int status = Communicator_I::READY;
	cv::Mat mask = cv::Mat::zeros(output_dim, output_dim, CV_32SC1);
	sleep(1);
	t2 = ::cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(90, std::string("compute"), t1, t2, std::string("1"), ::cciutils::event::COMPUTE));

		t1 = ::cciutils::event::timestampInUS();
		cci::rt::adios::CVImage *img = new cci::rt::adios::CVImage(mask, imagename, fn, tilex, tiley);
		if (compress) img->serialize(output_size, output, cci::rt::adios::CVImage::ENCODE_Z);
		else img->serialize(output_size, output);
		// clean up
		delete img;

		t2 = ::cciutils::event::timestampInUS();
		memset(len, 0, 21);
		sprintf(len, "%lu", (long)output_size);
		if (logsession != NULL) logsession->log(cciutils::event(90, std::string("serialize"), t1, t2, std::string(len), ::cciutils::event::MEM_IO));

	mask.release();
	output_count++;
	return status;
}

int GenerateOutputPush::run() {

	if (outputBuf->isStopped()) {
		Debug::print("%s STOPPED. call count %d \n", getClassName(), call_count);
		return Communicator_I::DONE;
	} else if (!outputBuf->canPush()){
		//Debug::print("%s FULL. call count %d \n", getClassName(), call_count);
		return Communicator_I::WAIT;
	} // else has room, and not stopped, so can push.

	int output_size = 0;
	void *output = NULL;

	int result = compute(-1, NULL, output_size, output);

//	if (output != NULL)
//		Debug::print("%s iter %d output var passed back at address %x, value %s, size %d, result = %d\n", getClassName(), call_count, output, output, output_size, result);
//	else
//		Debug::print("%s iter %d output var passed back at address %x, size %d, result = %d\n", getClassName(), call_count, output, output_size, result);

	int bstat;
	if (result == Communicator_I::READY) {
		++call_count;
		bstat = outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			free(output);
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			Debug::print("ERROR: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
			free(output);
			return Communicator_I::WAIT;
		} else {
			return Communicator_I::READY;
		}

	} else if (result == Communicator_I::DONE) {

		// no more, so done.
		outputBuf->stop();
	}
	return result;

}

}
} /* namespace rt */
} /* namespace cci */
