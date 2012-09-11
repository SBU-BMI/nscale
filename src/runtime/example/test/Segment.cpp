/*
 * Segment.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "Segment.h"
#include "Debug.h"

namespace cci {
namespace rt {

Segment::Segment(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
				cciutils::SCIOLogSession *_logsession) :
				Action_I(_parent_comm, _gid, _input, _output, _logsession), output_count(0) {
}

Segment::~Segment() {
	Debug::print("%s destructor called.\n", getClassName());
}
int Segment::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	if (input_size == 0 || input == NULL) return -1;

	output_size = input_size;
	output = malloc(output_size);
	memcpy(output, input, input_size);

	return 1;
}

int Segment::run() {

	if (this->inputBuf->isFinished()) {
		Debug::print("%s input DONE.  input count = %d, output count = %d\n", getClassName(), call_count, output_count);
		this->outputBuf->stop();

		return Communicator_I::DONE;
	} else if (this->outputBuf->isStopped()) {
		Debug::print("%s output DONE.  input count = %d, output count = %d\n", getClassName(), call_count, output_count);
		this->inputBuf->stop();

		return Communicator_I::DONE;
	} else if (this->inputBuf->isEmpty() || this->outputBuf->isFull()) {
		return Communicator_I::WAIT;
	}

	DataBuffer::DataType data;
	int output_size, input_size;
	void *output, *input;

	int bstat = this->inputBuf->pop(data);
	if (bstat == DataBuffer::EMPTY) {
		return Communicator_I::WAIT;
	}
	input_size = data.first;
	input = data.second;


	int result = compute(input_size, input, output_size, output);
	call_count++;


	if (result == 1) {
//			Debug::print("%s bufferring output:  call count= %d\n", getClassName(), call_count);
		++output_count;
		bstat = this->outputBuf->push(std::make_pair(output_size, output));

		if (bstat == DataBuffer::STOP) {
			Debug::print("ERROR: %s can't push into buffer.  status STOP.  Should have caught this earlier. \n", getClassName());
			this->inputBuf->push(data);
			this->inputBuf->stop();
			return Communicator_I::DONE;
		} else if (bstat == DataBuffer::FULL) {
			Debug::print("ERROR: %s can't push into buffer.  status FULL.  Should have caught this earlier.\n", getClassName());
			this->inputBuf->push(data);
			return Communicator_I::WAIT;
		} else {
			if (input != NULL) {
				printf("removed input at %p\n", input);
				free(input);
				input = NULL;
			}
			return Communicator_I::READY;
		}
	} else {
		if (input != NULL) {
			printf("removed input at %p\n", input);
			free(input);
			input = NULL;
		}
		return Communicator_I::READY;
	}



}


} /* namespace rt */
} /* namespace cci */
