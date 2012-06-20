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
	Worker_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

}

Assign::~Assign() {
	// TODO Auto-generated destructor stub
}

int Assign::compute(int const &input_size , char* const &input,
			int &output_size, char* &output) {
	call_count++;
	if (call_count % 50 == 0) Debug::print("Assign compute called %d\n", call_count);

	if (call_count > 100) return -1;
	else return 1;
}

} /* namespace rt */
} /* namespace cci */
