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

Save::Save(MPI_Comm const * _parent_comm, int const _gid) :
		Action_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

}

Save::~Save() {
	// TODO Auto-generated destructor stub
}

int Save::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {

	if (input_size == 0 || input == NULL) return -1;


	int *i2 = (int*)input;
	Debug::print("at SAVE: Inputsize = %d, input = %d\n", input_size, *i2);
	return 1;
}


int Save::run() {


	call_count++;
	//if (call_count % 100 == 0) Debug::print("Save compute called %d\n", call_count);


	int input_size, output_size;  // allocate output vars because these are references
	void *input, *output;
	output = NULL;
	output_size = -1;

	int istatus = getInputStatus();
	if (istatus == READY) {
		int result = getInput(input_size, input);
		//if (call_count % 100 == 0) Debug::print("SAVE READY\n");
		result = compute(input_size, input, output_size, output);
		if (input != NULL) free(input);
		if (result >= 0) return READY;
		else return WAIT;
	} else if (istatus == WAIT) {
		if (call_count % 10 == 0) Debug::print("SAVE WAIT\n");
		return WAIT;
	} else {  // done or error //
		Debug::print("SAVE DONE/ERROR at call_count %d\n", call_count);
		// output already changed.
		return output_status;
	}
}

} /* namespace rt */
} /* namespace cci */
