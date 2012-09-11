/*
 * Assign.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "Assign.h"
#include "Debug.h"

namespace cci {
namespace rt {

Assign::Assign(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		cciutils::SCIOLogSession *_logsession)  :
	Action_I(_parent_comm, _gid, _input, _output, _logsession) {
}

Assign::~Assign() {
	Debug::print("%s destructor called.\n", getClassName());
}

/**
 * generate some results.  if no more, set the done flag.
 */
int Assign::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (call_count >= 20) {

		output_size = 0;
		output = NULL;

		return Communicator_I::DONE;
	} else {

		output_size = sizeof(int);
		output = malloc(output_size);
//		printf("output var allocated at address %x\n", output);
		memcpy(output, (void*)(&call_count), output_size);
		return Communicator_I::READY;
	}
}

int Assign::run() {

	if (outputBuf->isStopped()) {
		Debug::print("%s STOPPED. call count %d \n", getClassName(), call_count);
		return Communicator_I::DONE;
	} else if (outputBuf->isFull()){
		Debug::print("%s FULL. call count %d \n", getClassName(), call_count);
		return Communicator_I::WAIT;
	} // else has room, and not stopped, so can push.


	int output_size = 0;
	void *output = NULL;

//	Debug::print("iter %d output var initialized at address %x, size %d\n", call_count, output, output_size);

	int result = compute(-1, NULL, output_size, output);
//	output_size = sizeof(int);
//	output = new char[sizeof(int)];
//	memcpy(output,(void*)(&call_count), sizeof(int));
//	int result = 1;

	int bstat;
	if (result == Communicator_I::READY) {
		++call_count;
		bstat = outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			Debug::print("ERROR: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
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


} /* namespace rt */
} /* namespace cci */
