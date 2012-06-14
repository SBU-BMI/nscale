/*
 * PushCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PushCommHandler.h"

namespace cci {
namespace rt {

PushCommHandler::PushCommHandler(MPI_Comm const &_comm, std::vector<int> const &_roots)
: RootedCommHandler_I(_comm, _roots) {
}

PushCommHandler::~PushCommHandler() {
}

} /* namespace rt */
} /* namespace cci */
