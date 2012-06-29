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

Segment::Segment(MPI_Comm const * _parent_comm, int const _gid) :
				Action_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

}

Segment::~Segment() {
	// TODO Auto-generated destructor stub
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

	if (!canAddOutput()) return output_status;

	call_count++;
	//if (call_count % 100 == 0) Debug::print("Segment compute called %d\n", call_count);

	int output_size, input_size;
	void *output, *input;

	int istatus = getInputStatus();

	if (istatus == READY) {
		input = NULL;
		int result = getInput(input_size, input);
		Debug::print("READY and getting input:  call count= %d\n", call_count);

		result = compute(input_size, input, output_size, output);
		if (input != NULL) {
			free(input);
			input = NULL;
		}

		if (result >= 0) {
			Debug::print("saving output:  call count= %d\n", call_count);

			result = addOutput(output_size, output);
//			free(output);
		} else {
			Debug::print("no output.  call count = %d\n", call_count);
			result = WAIT;
		}
		return result;
	} else if (istatus == WAIT) {
		Debug::print("READY and waiting for input: call count= %d\n", call_count);
		return WAIT;
	} else {  // done or error //
		// output already changed.
		Debug::print("Done for input:  call count= %d\n", call_count);
		return output_status;
	}

}


} /* namespace rt */
} /* namespace cci */
