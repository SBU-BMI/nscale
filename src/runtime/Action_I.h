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
	Action_I(MPI_Comm const * _parent_comm, int const _gid) :
		Communicator_I(_parent_comm, _gid), input_status(READY), output_status(READY) {};
	virtual ~Action_I() {
		void * data;
		if (!inputSizes.empty()) Debug::print("%s WARNING: %d entries left in input\n", getClassName(), inputSizes.size());
		for (int i = 0; i < inputSizes.size(); ++i) {
			inputSizes.pop();
			data = inputData.front();
			inputData.pop();
			free(data);
		}
		if (!outputSizes.empty()) Debug::print("%s WARNING: %d entries left in output\n", getClassName(), outputSizes.size());
		for (int i = 0; i < outputSizes.size(); ++i) {
			outputSizes.pop();
			data = outputData.front();
			outputData.pop();
			free(data);
		}
	};

	virtual const char* getClassName() { return "Action_I"; };

	virtual int run() = 0;

	// can add input if worker's compute is not done, and input buffer is ready
	int addInput(int size , void * data) {
		if (this->canAddInput() && size > 0) {  // only receive in READY and WAIT state
			if (data == NULL) Debug::print("%s ERROR:  why is data NULL?\n", getClassName());

//			void *d2 = malloc(size);
//			memcpy(d2, data, size);
//

			inputSizes.push(size);
			inputData.push(data);
//			Debug::print("added %d bytes at address %x , inputSizes size = %d\n", size, data, inputSizes.size());
			this->input_status = READY;
		}
		return this->input_status;
	};


	int getOutput(int &size , void * &data) {
		int stat = getOutputStatus();
		if (stat == READY) {
			size = outputSizes.front();
			data = outputData.front();
			outputSizes.pop();
			outputData.pop();
		} else {
			size = 0;
			data = NULL;
		}
		return stat;
	};

	void markInputDone() {
		this->input_status = DONE;
	}

	virtual int getOutputStatus() {
		if (this->input_status == ERROR || this->output_status == ERROR) {
			this->input_status = ERROR;
			this->output_status = ERROR;
			return output_status;
		}
		if (outputSizes.empty()) {
			if (output_status == READY) return WAIT;
			if (input_status == DONE) output_status = DONE;
			return output_status;
		} else {
			return READY;
		}
	}

	virtual int getInputStatus() {
		if (this->input_status == ERROR || this->output_status == ERROR) {
			this->input_status = ERROR;
			this->output_status = ERROR;
			return this->input_status;
		}
		if (inputSizes.empty()) {
			if (input_status == READY) return WAIT;
			if (input_status == DONE) output_status = DONE;
			return input_status;
		} else {
			return READY;
		}
	}

	virtual bool canAddInput() {
		if (this->input_status == ERROR || this->output_status == ERROR) {
			this->input_status = ERROR;
			this->output_status = ERROR;
		}
		return this->input_status == READY || this->input_status == WAIT;
	}
	virtual bool canReadOutput() {
		int stat = getOutputStatus();
		return stat == READY;
	};

protected:
	virtual int compute(int const &input_size , void * const &input,
			int &output_size, void * &output) = 0;

	virtual bool canAddOutput(){
		if (this->input_status == ERROR || this->output_status == ERROR) {
			this->input_status = ERROR;
			this->output_status = ERROR;
		}
		return this->output_status == READY || this->output_status == WAIT;
	}


	int addOutput(int size , void * data) {

		if (this->canAddOutput() && size > 0) {  // only receive in READY and WAIT state
//			void *d2 = malloc(size);
//			memcpy(d2, data, size);

			outputSizes.push(size);
			outputData.push(data);
//			Debug::print("added %d bytes at address %x , outputSizes size = %d\n", size, data, outputSizes.size());
			this->output_status = READY;
		}
		return this->output_status;

	};

	virtual bool canReadInput() {
		int stat = getInputStatus();
		return stat == READY;
	};

	int getInput(int &size , void * &data) {
		int stat = getInputStatus();
		if (stat == READY) {
			size = inputSizes.front();
			data = inputData.front();
			inputSizes.pop();
			inputData.pop();
		} else {
			size = 0;
			data = NULL;
		}
		return stat;
	};

	std::queue<int> inputSizes;
	std::queue<void *> inputData;
	std::queue<int> outputSizes;
	std::queue<void *> outputData;

	int input_status;  // take on
		// READY (can push and pop),
		// WAIT (or nothing to get.),
		// DONE (no push or pop)
		// (FLUSH is implcit.  DONE and buffer not empty)..
		// FULL will be implemented later.
	int output_status;
	// output status take on
		// READY (can push and pop),
		// WAIT (nothing is available but not done),
		// DONE (no more and buffer is empty)
		// (FLUSH is implicit.  REDAY and full buffer)
		// FULL will be implemented later.

};

} /* namespace rt */
} /* namespace cci */
#endif /* ACTION_I_H_ */
