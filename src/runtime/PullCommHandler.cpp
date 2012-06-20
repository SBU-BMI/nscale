/*
 * PullCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PullCommHandler.h"
#include "Debug.h"

namespace cci {
namespace rt {

PullCommHandler::PullCommHandler(MPI_Comm const * _parent_comm, int const _gid, std::vector<int> _roots)
 : RootedCommHandler_I(_parent_comm, _gid, _roots) {

}

PullCommHandler::~PullCommHandler() {
	//printf("PullCommHandler destructor called\n");

}

int PullCommHandler::exchange(int &size, char* &data) {
	call_count++;
	if (call_count % 50 == 0) Debug::print("PullCommHandler exchange called %d\n", call_count);

	if (comm == MPI_COMM_NULL) return 0;
	else return 1;
}


} /* namespace rt */
} /* namespace cci */
