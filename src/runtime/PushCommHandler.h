/*
 * PushCommHandler.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PUSHCOMMHANDLER_H_
#define PUSHCOMMHANDLER_H_

#include "CommHandler_I.h"

namespace cci {
namespace rt {

class PushCommHandler: public cci::rt::CommHandler_I {
public:
	PushCommHandler(MPI_Comm const * _parent_comm, int const _gid, Scheduler_I * _scheduler, cciutils::SCIOLogSession *_logger = NULL);
	virtual ~PushCommHandler();
	virtual const char* getClassName() { return "PushCommHandler"; };

	virtual int run();
};

} /* namespace rt */
} /* namespace cci */
#endif /* PUSHCOMMHANDLER_H_ */
