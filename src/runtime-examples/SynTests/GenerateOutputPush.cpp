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
#include "TypeUtils.h"
#include "SCIOHistologicalEntities.h"
#include <unistd.h>
#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace syntest {

GenerateOutputPush::GenerateOutputPush(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		boost::program_options::variables_map &_vm,
		cci::common::LogSession *_logsession) :
				Action_I(_parent_comm, _gid, _input, _output, _logsession), output_count(0) {

	int size;
	MPI_Comm_size(*_parent_comm, &size);

	count = cci::rt::CmdlineParser::getParamValueByName<int>(_vm, cci::rt::CmdlineParser::PARAM_INPUTCOUNT);
	count = (count + size - 1) / size;

	output_dim = cci::rt::CmdlineParser::getParamValueByName<int>(_vm, cci::rt::CmdlineParser::PARAM_MAXIMGSIZE);
	compress = cci::rt::CmdlineParser::getParamValueByName<bool>(_vm, cci::rt::DataBuffer::PARAM_COMPRESSION);

	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

}

GenerateOutputPush::~GenerateOutputPush() {
	//cci::common::Debug::print("%s destructor called. %d messages\n", getClassName(), output_count);
}

int GenerateOutputPush::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (output_count >= count) return Communicator_I::DONE;

	long long t1, t2;

	char len[21];
	int tilex = 0;
	int tiley = 0;
	char inchars[256];
	char fnchars[1024];
	memset(inchars, 0, 256);
	memset(fnchars, 0, 1024);
	sprintf(inchars, "%dx%d", world_rank, count);
	sprintf(fnchars, "/tmp/dummy/synthetic_%dx%d_%d.tiff", world_rank, count, output_count);
	std::string imagename(inchars);
	std::string fn(fnchars);

	t1 = ::cci::common::event::timestampInUS();

	// real computation:
	int status = Communicator_I::READY;
	cv::Mat mask = cv::Mat::zeros(output_dim, output_dim, CV_32SC1);
	sleep(1);
	t2 = ::cci::common::event::timestampInUS();
	if (logsession != NULL) logsession->log(cci::common::event(90, std::string("compute"), t1, t2, std::string("1"), ::cci::common::event::COMPUTE));

		t1 = ::cci::common::event::timestampInUS();
		cci::rt::adios::CVImage *img = new cci::rt::adios::CVImage(mask, imagename, fn, tilex, tiley);
		if (compress) img->serialize(output_size, output, cci::rt::adios::CVImage::ENCODE_Z);
		else img->serialize(output_size, output);
		// clean up
		delete img;

		t2 = ::cci::common::event::timestampInUS();
		memset(len, 0, 21);
		sprintf(len, "%d", output_size);
		if (logsession != NULL) logsession->log(cci::common::event(90, std::string("serialize"), t1, t2, std::string(len), ::cci::common::event::MEM_IO));

	mask.release();
	output_count++;
	return status;
}

int GenerateOutputPush::run() {

	if (outputBuf->isStopped()) {
		cci::common::Debug::print("%s STOPPED. call count %d \n", getClassName(), call_count);
		return Communicator_I::DONE;
	} else if (!outputBuf->canPush()){
		//cci::common::Debug::print("%s FULL. call count %d \n", getClassName(), call_count);
		return Communicator_I::WAIT;
	} // else has room, and not stopped, so can push.

	int output_size = 0;
	void *output = NULL;

	int result = compute(-1, NULL, output_size, output);

//	if (output != NULL)
//		cci::common::Debug::print("%s iter %d output var passed back at address %x, value %s, size %d, result = %d\n", getClassName(), call_count, output, output, output_size, result);
//	else
//		cci::common::Debug::print("%s iter %d output var passed back at address %x, size %d, result = %d\n", getClassName(), call_count, output, output_size, result);

	int bstat;
	if (result == Communicator_I::READY) {
		++call_count;
		bstat = outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			cci::common::Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			free(output);
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			cci::common::Debug::print("ERROR: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
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
