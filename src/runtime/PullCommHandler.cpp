/*
 * PullCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PullCommHandler.h"

namespace cci {
namespace rt {

PullCommHandler::PullCommHandler(MPI_Comm const &_comm, std::vector<int> const &_roots)
 : RootedCommHandler_I(_comm, _roots) {

}

PullCommHandler::~PullCommHandler() {

}

} /* namespace rt */
} /* namespace cci */
