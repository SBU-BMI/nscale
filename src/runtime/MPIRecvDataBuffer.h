/*
 * MPIRecvDataBuffer.h
 *
 * a data buffer coupled to an out-going data buffer for MPI.
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#ifndef MPIRECVDATABUFFER_H_
#define MPIRECVDATABUFFER_H_

#include "mpi.h"
#include <tr1/unordered_map>
#include "MPIDataBuffer.h"


namespace cci {
namespace rt {

class MPIRecvDataBuffer: public cci::rt::MPIDataBuffer {
public:

	MPIRecvDataBuffer(int _capacity, bool _non_blocking = true, cciutils::SCIOLogSession *_logsession = NULL) :
		MPIDataBuffer(_capacity, _non_blocking, _logsession) {};

	// for MPI send/recv.  this takes the place of push.
	virtual int transmit(int node, int tag, MPI_Datatype type, MPI_Comm &comm, int size=-1);
	virtual int canTransmit() { return !isStopped() && (buffer.size() + mpi_buffer.size()) < capacity; };

	// overridden to add flushRequests capability.
	virtual bool canPush() {
		assert(false); return false;
	};
	virtual int push(DataType const data) { assert(false); return UNSUPPORTED_OP; };  // cannot push directly to a recv buffer.
	// pop is standard.

	virtual bool isFull() {
		checkRequests();
		return buffer.size() >= capacity;
	};
	virtual bool canPop() {
		checkRequests();
		return buffer.size() > 0;
	};

	// this is called by getBufferSize(), which is called by most other function, specifically by pop.
	// this function handles checking the requests, and for all the completed, put into regular buffer.
	virtual int checkRequests(bool waitForAll = false);

	virtual ~MPIRecvDataBuffer() {
		if (mpi_buffer.size() > 0) Debug::print("WARNING: clearning MPISendBuffer\n");

		int completed = checkRequests(true);
		if (completed > 0) Debug::print("WARNING: removed %d from MPISendBuffer\n", completed);
	};


};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
