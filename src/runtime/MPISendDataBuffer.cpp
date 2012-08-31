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

MPISendDataBuffer::MPISendDataBuffer(int _capacity) : DataBuffer(_capacity) {
}

MPISendDataBuffer::~MPISendDataBuffer() { }

void MPISendDataBuffer::dumpBuffer() {
	DataBuffer::dumpBuffer();

	// get all the requests together
	MPI_Request *reqs = new MPI_Request[mpi_buffer.size()];
	int active = 0;
	for (std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType>::iterator iter = mpi_buffer.begin();
			iter != mpi_buffer.end(); ++iter) {
		reqs[active] = *(iter->first);
		++active;
	}

	MPI_Status *stati = new MPI_Status[mpi_buffer.size()];

	// wait for everything to finish.
	MPI_Waitall(active, reqs, stati);

	for (std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType>::iterator iter = mpi_buffer.begin();
			iter != mpi_buffer.end(); ++iter) {
		free(iter->second.second);
	}
	mpi_buffer.clear();

}

int MPISendDataBuffer::pushMPI(MPI_Request *req, DataBuffer::DataType const data) {
	if (!canPushMPI()) return FULL;
	if (data.first == 0 || data.second == NULL) return BAD_DATA;

	mpi_buffer[req] = data;

	if (mpi_buffer.size() >= capacity) return FULL;
	else return status;  // should have value READY.
}

int MPISendDataBuffer::popMPI(DataBuffer::DataType* &data) {
	if (mpi_buffer.size() == 0) return -1;
	data = NULL;
//
//	int retcode = 0;
//	data = new DataBuffer::DataType[mpi_buffer.size()];
//
//	for (std::tr1::unordered_map<MPI_Request, DataBuffer::DataType>::iterator iter = mpi_buffer.begin();
//			iter != mpi_buffer.end(); ++iter) {
//		MPI_Request req = iter->first;
//		int completed;
//		MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
//		if (completed) {
//			printf("send request completed\n");
//			data[retcode] = mpi_buffer[req];
//			++retcode;
//			iter = mpi_buffer.erase(mpi_buffer.find(req));
//		}
//	}


	// get all the requests together
	MPI_Request *reqs = new MPI_Request[mpi_buffer.size()];		// MPI_Test uses array of reqs.
	MPI_Request **reqptrs = new MPI_Request*[mpi_buffer.size()];  // unordered map erases with pointer to reqs.

	int active = 0;
	for (std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType>::iterator iter = mpi_buffer.begin();
			iter != mpi_buffer.end(); ++iter) {
		reqs[active] = *(iter->first);
		reqptrs[active] = iter->first;
		++active;
	}

	int completed = 0;
	int *completedreqs = new int[mpi_buffer.size()];
	MPI_Status *stati = new MPI_Status[mpi_buffer.size()];

	MPI_Testsome(active, reqs, &completed, completedreqs, stati);
	if (completed > 0) printf("send active = %d, number completed = %d, total = %ld\n", active, completed, mpi_buffer.size());
	for (int i = 0; i < completed; ++i) {
		printf("send completed: id = %d\n", completedreqs[i]);
	}

	int retcode;
	if (completed == MPI_UNDEFINED) {
		retcode = -1;
	} else if (completed == 0) {
		retcode = 0;
	} else {
		data = new DataBuffer::DataType[completed];
		for (int i = 0; i < completed; ++i) {
			data[i] = mpi_buffer[reqptrs[completedreqs[i]]];
			mpi_buffer.erase(reqptrs[completedreqs[i]]);
		}
		printf("send new size: %ld\n", mpi_buffer.size());
		retcode = completed;
	}
	delete [] reqs;
	delete [] reqptrs;
	delete [] completedreqs;
	delete [] stati;
	return retcode;

}


} /* namespace rt */
} /* namespace cci */
