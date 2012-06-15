/*
 * PushCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PushCommHandler.h"

namespace cci {
namespace rt {

PushCommHandler::PushCommHandler(MPI_Comm const &_parent_comm, int groupid, std::vector<int> const &_roots)
: RootedCommHandler_I(_parent_comm, int groupid, _roots) {
}

PushCommHandler::~PushCommHandler() {
}

void PushCommHandler::exchange(int &size, char* &data) {

}

} /* namespace rt */
} /* namespace cci */
