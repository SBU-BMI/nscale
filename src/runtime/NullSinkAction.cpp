/*
 * NullSinkAction.cpp
 *
 *  Created on: Aug 2, 2012
 *      Author: tcpan
 */

#include "NullSinkAction.h"
#include "Debug.h"

namespace cci {
namespace rt {

NullSinkAction::NullSinkAction(MPI_Comm const * _parent_comm, int const _gid,
		DataBuffer *_input, DataBuffer *_output,
		cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _input, _output, _logsession) {
	// TODO Auto-generated constructor stub
	assert(_input != NULL);
}

NullSinkAction::~NullSinkAction() {
	// TODO Auto-generated destructor stub
}



int NullSinkAction::run() {


	call_count++;
	//if (call_count % 100 == 0) Debug::print("Save compute called %d\n", call_count);

	DataBuffer::DataType data;

	int input_size, output_size;  // allocate output vars because these are references
	void *input, *output;
	output = NULL;
	output_size = -1;

	int status = (this->inputBuf->isEmpty() ? Communicator_I::WAIT : Communicator_I::READY);
	if (status == Communicator_I::READY) {
		int result = this->inputBuf->pop(data);
		input_size = data.first;
		input = data.second;

		//if (call_count % 100 == 0) Debug::print("SAVE READY\n");
		result = compute(input_size, input, output_size, output);
		if (input != NULL) {
			free(input);
			input = NULL;
		}
		if (result >= 0) return Communicator_I::READY;
		else return Communicator_I::WAIT;
	} else if (status == Communicator_I::WAIT) {
//		if (call_count % 10 == 0) Debug::print("SAVE WAIT\n");
		return Communicator_I::WAIT;
	} else {  // done or error //
		Debug::print("%s SAVE DONE/ERROR at call_count %d\n", getClassName(), call_count);
		// output already changed.
		return Communicator_I::DONE;
	}
}

} /* namespace rt */
} /* namespace cci */
