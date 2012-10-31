/*
 * MPIDataBuffer.h
 *
 * a data buffer coupled to an out-going data buffer for MPI.
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#ifndef MPIDATABUFFER_H_
#define MPIDATABUFFER_H_

#include "DataBuffer.h"
#include "mpi.h"
#include <tr1/unordered_map>
#include <cstdlib>


namespace cci {
namespace rt {

class MPIDataBuffer: public cci::rt::DataBuffer {
public:

	MPIDataBuffer(int _capacity, bool _non_blocking = true, cciutils::SCIOLogSession *_logsession = NULL)
		: DataBuffer(_capacity, _logsession), debug_complete_count(0), non_blocking(_non_blocking) {
		reqs = new MPI_Request[_capacity];
		reqptrs = new MPI_Request*[_capacity];
		completedreqs = new int[_capacity];

		if (!mpi_buffer.empty()) Debug::print("WARNING: constructing.  mpi_buffer is not empty.\n");
		mpi_buffer.clear();
		mpi_req_starttimes.clear();
	};

	virtual int debugBufferSize() { return buffer.size()+ mpi_buffer.size(); };

	virtual bool isFinished() {
		this->checkRequests();
		return isStopped() && buffer.size() <= 0 && mpi_buffer.size() <= 0;
	};

	// canPop need to be overridden in subclass.
	// for MPI send/recv.  this is how we send or receive a message.
	virtual int transmit(int node, int tag, MPI_Datatype type, MPI_Comm &comm, int size=-1) = 0;
	virtual int canTransmit() = 0;

	// either check for some requests tht completed, or wait for all requests to complete.
	// should check to make sure that there are requests
	virtual int checkRequests(bool waitForAll = false) = 0;

	virtual ~MPIDataBuffer() {
		delete [] reqs;
		delete [] reqptrs;
		delete [] completedreqs;
	};

	int debug_complete_count;

protected:
	std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType> mpi_buffer;

	MPI_Request *reqs;
	MPI_Request **reqptrs;
	int *completedreqs;

	bool non_blocking;

	std::tr1::unordered_map<MPI_Request*, long long> mpi_req_starttimes;
};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
