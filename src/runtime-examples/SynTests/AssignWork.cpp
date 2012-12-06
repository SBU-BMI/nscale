/*
 * AssignWork.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "AssignWork.h"
#include "Debug.h"
#include "FileUtils.h"
#include <dirent.h>
#include <string.h>
#include <algorithm>
#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace adios {

bool AssignWork::initParams() {
	return true;
}

boost::program_options::options_description AssignWork::params("Input Options");
bool AssignWork::param_init = AssignWork::initParams();

AssignWork::AssignWork(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		boost::program_options::variables_map &_vm,
		cci::common::LogSession *_logsession)  :
	Action_I(_parent_comm, _gid, _input, _output, _logsession) {

	assert(_output != NULL);

	long long t1, t2;
	t1 = ::cci::common::event::timestampInUS();

	int count = cci::rt::CmdlineParser::getParamValueByName<int>(_vm, cci::rt::CmdlineParser::PARAM_INPUTCOUNT);

	for (int i = 0; i < count; ++i) {
		filenames.push_back("/randomstr/randomstr/randomstr/randomstr/randomstr/randomstr/randomstr/randomstr-0000067890-0000012345.randomstr");
	}

	cci::common::Debug::print("%s inputcount %d\n", getClassName(), count);

	t2 = ::cci::common::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);
	sprintf(len, "%ld", (long)(count));
	if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("List Files"), t1, t2, std::string(len), ::cci::common::event::FILE_I));
}

AssignWork::~AssignWork() {

	cci::common::Debug::print("%s destructor called.\n", getClassName());
	filenames.clear();
}

/**
 * generate some results.  if no more, set the done flag.
 */
int AssignWork::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (filenames.size() > 0) {
		long long t1, t2;
		t1 = ::cci::common::event::timestampInUS();

		output_size = filenames.back().length() + 1;
		output = malloc(output_size);
		memset(output, 0, output_size);
		memcpy(output, filenames.back().c_str(), output_size - 1);

		filenames.pop_back();

		t2 = ::cci::common::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("Assign"), t1, t2, std::string(), ::cci::common::event::MEM_IO));

		return Communicator_I::READY;
	} else {
		output = NULL;
		output_size = 0;

		return Communicator_I::DONE;
	}

}

int AssignWork::run() {


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
