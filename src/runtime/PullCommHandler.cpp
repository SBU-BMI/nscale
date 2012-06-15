/*
 * PullCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PullCommHandler.h"

namespace cci {
namespace rt {

PullCommHandler::PullCommHandler(MPI_Comm const &_parent_comm, int groupid, std::vector<int> const &_roots)
 : RootedCommHandler_I(_parent_comm, int groupid, _roots) {

}

PullCommHandler::~PullCommHandler() {

}

void PullCommHandler::exchange(int &size, char* &data) {

}

} /* namespace rt */
} /* namespace cci */
