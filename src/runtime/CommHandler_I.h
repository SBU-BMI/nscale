/*
 * CommHandler_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef COMMHANDLER_I_H_
#define COMMHANDLER_I_H_

#include "Communicator_I.h"
#include <vector>
#include <algorithm>
#include <tr1/unordered_map>
#include "Scheduler_I.h"
#include "MPIDataBuffer.h"

namespace cci {
namespace rt {

class CommHandler_I : public cci::rt::Communicator_I {
public:
	virtual ~CommHandler_I();

	/**
	 * splits a communicator and allows handling of the communication between roots and children
	 * _parent_comm: the communicator to split
	 * _gid:  the group id with which to split the parent comm
	 * _roots:  the list of roots, specified as ranks in parent_comm.
	 */
	CommHandler_I(MPI_Comm const *_parent_comm, int const _gid,
			MPIDataBuffer *_buffer, Scheduler_I *_scheduler,
			cciutils::SCIOLogSession *_logsession = NULL);

	virtual const char* getClassName() { return "CommHandler_I"; };

	virtual int run() = 0;
	virtual bool isListener() { return (scheduler == NULL ? false : scheduler->isRoot()); };
//	virtual bool isReady() { return buffer != NULL && !buffer->isStopped(); };

//	virtual int getStatus() { return status; };
	DataBuffer *getBuffer() {return buffer; };

protected:
	Scheduler_I *scheduler;
	MPIDataBuffer *buffer;

	int status;

};

} /* namespace rt */
} /* namespace cci */
#endif /* COMMHANDLER_I_H_ */
