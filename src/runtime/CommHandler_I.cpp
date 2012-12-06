/*
 * CommHandler_I.cpp
 *
 *  Created on: Jun 25, 2012
 *      Author: tcpan
 */

#include "CommHandler_I.h"
#include <sstream>
#include "mpi.h"

namespace cci {
namespace rt {



CommHandler_I::CommHandler_I(MPI_Comm const * _parent_comm, int const _gid, MPIDataBuffer *_buffer, Scheduler_I * _scheduler,
		cci::common::LogSession *_logsession) :
	Communicator_I(_parent_comm, _gid, _logsession), status(Communicator_I::READY), buffer(_buffer), scheduler(_scheduler) {

	long long t1 = ::cci::common::event::timestampInUS();

	if (comm == MPI_COMM_NULL) {
		cci::common::Debug::print("%s ERROR: comm member %d must have a MPI communicator.\n", getClassName(), rank);
		status = Communicator_I::ERROR;
	} else {
		if (scheduler != NULL) scheduler->configure(comm);
	}

	if (buffer != NULL) DataBuffer::reference(buffer, this);

	long long t2 = ::cci::common::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("MPI scheduler"), t1, t2, std::string(), ::cci::common::event::NETWORK_IO));
}

CommHandler_I::~CommHandler_I() {
	//printf("CommHandler destructor called\n");


	if (buffer != NULL) DataBuffer::dereference(buffer, this);
	if (scheduler != NULL) delete scheduler;
}

} /* namespace rt */
} /* namespace cci */
