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
				Worker_I(_parent_comm, _gid) {
	// TODO Auto-generated constructor stub

}

Segment::~Segment() {
	// TODO Auto-generated destructor stub
}
int Segment::compute(int const &input_size , char* const &input,
			int &output_size, char* &output) {
	call_count++;
	if (call_count % 50 == 0) Debug::print("Segment compute called %d\n", call_count);

	if (call_count > 100) return -1;
	else return 1;

}

} /* namespace rt */
} /* namespace cci */
