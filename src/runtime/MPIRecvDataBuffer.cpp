/*
 * MPIRecvDataBuffer.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#include "MPIRecvDataBuffer.h"
#include <cstdlib>

namespace cci {
namespace rt {

int MPIRecvDataBuffer::transmit(int node, int tag, MPI_Datatype type, MPI_Comm &comm, int size) {
	// first check the size to receive.
	if (size < 0) return BAD_DATA;

	void *ldata = NULL;

	if (size == 0) {
		// size is zero, receive it and through it away (so the message is received.
		MPI_Recv(ldata, size, type, node, tag, comm, MPI_STATUS_IGNORE);
		return status;
	}

	if (isStopped()) return status;
	if (isFull()) return FULL;

	long long t1 = ::cciutils::event::timestampInUS();

	// else size is greater than 1.
	ldata = malloc(size);
	memset(ldata, 0, size);

	if (non_blocking) {
		MPI_Request *req = (MPI_Request *)malloc(sizeof(MPI_Request));
		MPI_Irecv(ldata, size, type, node, tag, comm, req);

		mpi_buffer[req] = std::make_pair(size, ldata);
		mpi_req_starttimes[req] = t1;

	} else {
		MPI_Status mstat;

		MPI_Recv(ldata, size, type, node, tag, comm, &mstat);
		buffer.push(std::make_pair(size, ldata));

		long long t2 = ::cciutils::event::timestampInUS();

		char len[21];  // max length of uint64 is 20 digits
		memset(len, 0, 21);
		sprintf(len, "%d", size);
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("MPI B RECV"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));


	}
	return status;
}


int MPIRecvDataBuffer::checkRequests(bool waitForAll) {

	if (mpi_buffer.empty()) return 0;

	if (!non_blocking) return 0;  // blocking comm.  should not have any in MPI queue.

	// get all the requests together
	int active = 0;
	for (std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType>::iterator iter = mpi_buffer.begin();
			iter != mpi_buffer.end(); ++iter) {
		reqs[active] = *(iter->first);
		reqptrs[active] = iter->first;
		++active;
	}

	int completed = 0;
	if (waitForAll) {
		MPI_Waitall(active, reqs, MPI_STATUSES_IGNORE);
		completed = active;
	} else {
		MPI_Testsome(active, reqs, &completed, completedreqs, MPI_STATUSES_IGNORE);
	}

	long long t2 = ::cciutils::event::timestampInUS();
	long long t1 = -1;

	int size = 0;
	MPI_Request* reqptr = NULL;

	if (completed == MPI_UNDEFINED) {
		Debug::print("ERROR: testing completion received a complete count of MPI_UNDEFINED\n");
	} else if (completed == 0) {
		// Debug::print("no mpi requests completed\n");
	} else {
		//Debug::print("MPI Recv Buffer active = %d, number completed = %d, total = %ld\n", active, completed, mpi_buffer.size());

		char len[21];  // max length of uint64 is 20 digits

		for (int i = 0; i < completed; ++i) {
			reqptr = reqptrs[completedreqs[i]];

			//printf("recv MPI error status: %d\n", stati[i].MPI_ERROR);
			size = mpi_buffer[reqptr].first;

			buffer.push(mpi_buffer[reqptr]);
			mpi_buffer.erase(reqptr);

			t1 = mpi_req_starttimes[reqptr];
			mpi_req_starttimes.erase(reqptr);

			free(reqptr);

			memset(len, 0, 21);
			sprintf(len, "%d", size);
			if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("MPI NB RECV"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));
		}
		//Debug::print("MPI Recv Buffer new size: %ld\n", mpi_buffer.size());

		debug_complete_count += completed;
	}

//	Debug::print("MPIRecvDataBuffer: popMPI called.  %d load\n", mpi_buffer.size());

	return completed;
}


} /* namespace rt */
} /* namespace cci */
