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

NullSinkAction::NullSinkAction(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _logsession) {
	// TODO Auto-generated constructor stub

}

NullSinkAction::~NullSinkAction() {
	// TODO Auto-generated destructor stub
}



int NullSinkAction::run() {


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
		if (input != NULL) {
			free(input);
			input = NULL;
		}
		if (result >= 0) return READY;
		else return WAIT;
	} else if (istatus == WAIT) {
//		if (call_count % 10 == 0) Debug::print("SAVE WAIT\n");
		return WAIT;
	} else {  // done or error //
		Debug::print("SAVE DONE/ERROR at call_count %d\n", call_count);
		// output already changed.
		return output_status;
	}
}

} /* namespace rt */
} /* namespace cci */
