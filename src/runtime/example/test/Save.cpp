/*
 * Save.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "Save.h"
#include "Debug.h"

namespace cci {
namespace rt {

Save::Save(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _input, _output, _logsession) {
}

Save::~Save() {
	Debug::print("%s destructor called.\n", getClassName());
}

int Save::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (input_size == 0 || input == NULL) return -1;


	int const *i2 = (int const *)input;
	Debug::print("at SAVE: Inputsize = %d, input = %d\n", input_size, *i2);
	return 1;
}


int Save::run() {


	if (this->inputBuf->isFinished()) {
		Debug::print("%s input DONE.  input count = %d\n", getClassName(), call_count);
		return Communicator_I::DONE;
	} else if (this->inputBuf->isEmpty()) {
		return Communicator_I::WAIT;
	}

	DataBuffer::DataType data;
	void *input, *output;
	int output_size;

	call_count++;
	int bstat = this->inputBuf->pop(data);
	if (bstat == DataBuffer::EMPTY) {
		return Communicator_I::WAIT;
	}
	input = data.second;

	int result = compute(data.first, input, output_size, output);

	if (input != NULL) {
		free(input);
		input = NULL;
	}

	return Communicator_I::READY;


}

} /* namespace rt */
} /* namespace cci */
