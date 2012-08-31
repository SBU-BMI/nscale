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

MPIRecvDataBuffer::MPIRecvDataBuffer(int _capacity) : DataBuffer(_capacity) {
}

MPIRecvDataBuffer::~MPIRecvDataBuffer() { }

void MPIRecvDataBuffer::dumpBuffer() {
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

int MPIRecvDataBuffer::pushMPI(MPI_Request *req, DataBuffer::DataType const data) {
	if (!canPushMPI()) {
		if (isFull()) return FULL;
		else return status;
	}
	if (data.first == 0 || data.second == NULL) return BAD_DATA;

	int completed;
	MPI_Test(req, &completed, MPI_STATUS_IGNORE);
	if (completed == 0) mpi_buffer[req] = data;	
	else buffer.push(data);

	if (this->isFull()) return FULL;
	else return status;  // should have value READY.
}

int MPIRecvDataBuffer::popMPI(DataBuffer::DataType* &data) {
	if (mpi_buffer.size() == 0) return -1;
	data = NULL;


	// get all the requests together
	MPI_Request *reqs = new MPI_Request[mpi_buffer.size()];
	MPI_Request **reqptrs = new MPI_Request*[mpi_buffer.size()];
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

	if (completed > 0) printf("recv active = %d, number completed = %d, total = %ld\n", active, completed, mpi_buffer.size());

	int retcode;
	if (completed == MPI_UNDEFINED) {
		retcode = -1;
	} else if (completed == 0) {
		retcode = 0;
	} else {
		data = new DataBuffer::DataType[completed];

		for (int i = 0; i < completed; ++i) {
			//printf("recv MPI error status: %d\n", stati[i].MPI_ERROR);
			data[i] = mpi_buffer[reqptrs[completedreqs[i]]];
			mpi_buffer.erase(reqptrs[completedreqs[i]]);

			// also move into the regular buffer.
			data[i].second = realloc(data[i].second, data[i].first);  // free as much memory as possible...
			buffer.push(data[i]);
		}
		printf("recv new size: %ld\n", mpi_buffer.size());
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
