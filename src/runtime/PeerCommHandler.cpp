/*
 * PeerCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PeerCommHandler.h"

namespace cci {
namespace rt {

PeerCommHandler::PeerCommHandler(MPI_Comm const &_parent_comm, int groupid) :
		CommHandler_I(_parent_comm, groupid) {
}

PeerCommHandler::~PeerCommHandler() {
}

} /* namespace rt */
} /* namespace cci */
