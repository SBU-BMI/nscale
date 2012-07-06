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

Assign::Assign(MPI_Comm const * _parent_comm, int const _gid)  :
	Action_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

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
		output_status = DONE;
		output_size = 0;
		output = NULL;
	} else {

		output_size = sizeof(int);
		output = malloc(output_size);
//		printf("output var allocated at address %x\n", output);
		memcpy(output, (void*)(&call_count), output_size);
	}
	return 1;
}

int Assign::run() {

	if (!canAddOutput()) {
		Debug::print("Assign is done... at call count %d \n", call_count);
		return output_status;
	}

	call_count++;
	//Debug::print("Assign run called %d\n", call_count);
//	if (call_count > 100) {
//		output_status = DONE;
//		return DONE;
//	}

	int output_size = 0;
	void *output = NULL;

//	Debug::print("iter %d output var initialized at address %x, size %d\n", call_count, output, output_size);

	int result = compute(-1, NULL, output_size, output);
//	output_size = sizeof(int);
//	output = new char[sizeof(int)];
//	memcpy(output,(void*)(&call_count), sizeof(int));
//	int result = 1;

	if (output != NULL)
		Debug::print("iter %d output var passed back at address %x, value %d, size %d, result = %d\n", call_count, output, *((int*)output), output_size, result);
	else
		Debug::print("iter %d output var passed back at address %x, size %d, result = %d\n", call_count, output, output_size, result);


	if (result >= 0) {
		result = addOutput(output_size, output);
	} else result=WAIT;

//	if (output != NULL) free(output);

	return result;
}


} /* namespace rt */
} /* namespace cci */
