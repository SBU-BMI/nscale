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
		Worker_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

}

Save::~Save() {
	// TODO Auto-generated destructor stub
}

int Save::compute(int const &input_size , char* const &input,
			int &output_size, char* &output) {
	call_count++;
	if (call_count % 50 == 0) Debug::print("Save compute called %d\n", call_count);

	if (call_count > 100) return -1;
	else return 1;
}


} /* namespace rt */
} /* namespace cci */
