/*
 * MPISendDataBuffer.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#include "MPISendDataBuffer.h"
#include <cstdlib>

namespace cci {
namespace rt {


int MPISendDataBuffer::transmit(int node, int tag, MPI_Datatype type, MPI_Comm &comm, int size) {
	if (buffer.size() == 0) return EMPTY; // nothing to transmit.

	DataBuffer::DataType ldata = buffer.front();
	buffer.pop();
	if (ldata.first == 0 || ldata.second == NULL) return BAD_DATA;

	long long t1 = ::cci::common::event::timestampInUS();

	if (non_blocking) {
		MPI_Request *reqptr = (MPI_Request *)malloc(sizeof(MPI_Request));
		MPI_Isend(ldata.second, ldata.first, type, node, tag, comm, reqptr);

		mpi_buffer[reqptr] = ldata;
		mpi_req_starttimes[reqptr] = t1;

		//	cci::common::Debug::print("MPISendDataBuffer: pushMPI called.  %d load\n", mpi_buffer.size());
	} else {

		MPI_Send(ldata.second, ldata.first, type, node, tag, comm);
		free(ldata.second);
		long long t2 = ::cci::common::event::timestampInUS();

		char len[21];  // max length of uint64 is 20 digits
		memset(len, 0, 21);
		sprintf(len, "%d", ldata.first);
		if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("MPI B SEND"), t1, t2, std::string(len), ::cci::common::event::NETWORK_IO));

	}
	return status;
}

int MPISendDataBuffer::checkRequests(bool waitForAll) {

	if (mpi_buffer.empty()) return 0;  // nothing to check

	if (!non_blocking) return 0;  // blocking comm.  so should have nothing to check.

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

	long long t2 = ::cci::common::event::timestampInUS();
	long long t1 = -1;

	int size = 0;
	MPI_Request* reqptr = NULL;

	if (completed == MPI_UNDEFINED) {
		cci::common::Debug::print("ERROR: testing completion received a complete count of MPI_UNDEFINED\n");
	} else if (completed == 0) {
		// cci::common::Debug::print("no mpi requests completed\n");
	} else {
		//cci::common::Debug::print("MPI Send Buffer active = %d, number completed = %d, total = %ld\n", active, completed, mpi_buffer.size());

		char len[21];  // max length of uint64 is 20 digits

		for (int i = 0; i < completed; ++i) {
			if (waitForAll) {
				reqptr = reqptrs[i];
			} else {
				reqptr = reqptrs[completedreqs[i]];
			}
			size = mpi_buffer[reqptr].first;

			free(mpi_buffer[reqptr].second);
			mpi_buffer[reqptr].second = NULL;
			mpi_buffer.erase(reqptr);

			t1 = mpi_req_starttimes[reqptr];
			mpi_req_starttimes.erase(reqptr);

			free(reqptr);

			// clear the data itself.
			memset(len, 0, 21);
			sprintf(len, "%d", size);
			if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("MPI NB SEND"), t1, t2, std::string(len), ::cci::common::event::NETWORK_IO_NB));
		}
//		printf("send new size: %ld\n", mpi_buffer.size());

		debug_complete_count += completed;
	}

//	cci::common::Debug::print("MPISendDataBuffer: popMPI called.  %d load\n", mpi_buffer.size());

	return completed;
}


} /* namespace rt */
} /* namespace cci */
