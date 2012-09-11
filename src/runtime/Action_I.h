/*
 * Worker.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef ACTION_I_H_
#define ACTION_I_H_
#include "mpi.h"
#include "Communicator_I.h"
#include <queue>
#include "Debug.h"
#include <stdlib.h>

namespace cci {
namespace rt {

class Action_I : public Communicator_I {
public:
	Action_I(MPI_Comm const * _parent_comm, int const _gid, DataBuffer *_input, DataBuffer *_output, cciutils::SCIOLogSession *_logsession = NULL) :
		Communicator_I(_parent_comm, _gid, _logsession), debug(false), inputBuf(_input), outputBuf(_output) {
		if (inputBuf != NULL) DataBuffer::reference(inputBuf, this);
		if (outputBuf != NULL) DataBuffer::reference(outputBuf, this);
	};
	virtual ~Action_I() {
		if (inputBuf != NULL) DataBuffer::dereference(inputBuf, this);
		if (outputBuf != NULL) DataBuffer::dereference(outputBuf,this);
	};

	virtual const char* getClassName() { return "Action_I"; };

	virtual int run() = 0;

	 void debugOn() {
		 debug = true;
	 };
	 void debugOff() {
		 debug = false;
	 };
	 DataBuffer *getInputBuffer() {return inputBuf;};
	 DataBuffer *getOutputBuffer() {return outputBuf;};

protected:
	virtual int compute(int const &input_size , void * const &input,
			int &output_size, void * &output) = 0;

	bool debug;

	DataBuffer *inputBuf;
	DataBuffer *outputBuf;
};

} /* namespace rt */
} /* namespace cci */
#endif /* ACTION_I_H_ */
