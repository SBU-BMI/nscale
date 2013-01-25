/*
 * MPISendDataBuffer.h
 *
 * a data buffer coupled to an out-going data buffer for MPI.
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#ifndef MPISENDDATABUFFER_H_
#define MPISENDDATABUFFER_H_

#include "MPIDataBuffer.h"

namespace cci {
namespace rt {

class MPISendDataBuffer: public cci::rt::MPIDataBuffer {
public:

	MPISendDataBuffer(int _capacity, bool _non_blocking = true, cci::common::LogSession *_logsession = NULL) :
		MPIDataBuffer(_capacity, _non_blocking, _logsession) {};
	MPISendDataBuffer(boost::program_options::variables_map &_vm, cci::common::LogSession *_logsession = NULL) :
		MPIDataBuffer(_vm, _logsession) {};

	// for MPI send/recv.  call this instead of pop.
	virtual int transmit(int node, int tag, MPI_Datatype type, MPI_Comm &comm, int size=-1);
	virtual int canTransmit() { return buffer.size() > 0; };

	// can pop is used to determine if there is
	virtual bool canPop() { assert(false); return false; };
	virtual int pop(DataType &data) { assert(false); return UNSUPPORTED_OP; };  // cannot pop from a send buffer.
	// push is standard.

	virtual bool canPush() {
		checkRequests();
		return !isStopped() && ((buffer.size() + mpi_buffer.size()) < capacity);
	};

	virtual int checkRequests(bool waitForAll = false);

	virtual ~MPISendDataBuffer() {
		if (buffer.size() > 0) cci::common::Debug::print("ERROR: clearing unsent stuff in MPISendBuffer\n");

		int todo = mpi_buffer.size();
		int completed = checkRequests(true);
		if (completed != todo) cci::common::Debug::print("ERROR: completed %d from send buffer %d\n", completed, todo);

	};


};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
