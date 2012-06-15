/*
 * PeerCommHandler.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PEERCOMMHANDLER_H_
#define PEERCOMMHANDLER_H_

#include "CommHandler_I.h"

namespace cci {
namespace rt {

class PeerCommHandler: public cci::rt::CommHandler_I {
public:
	PeerCommHandler(MPI_Comm const &_parent_comm, int groupid);
	virtual ~PeerCommHandler();
};

} /* namespace rt */
} /* namespace cci */
#endif /* PEERCOMMHANDLER_H_ */
