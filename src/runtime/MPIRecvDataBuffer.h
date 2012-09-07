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

	MPIRecvDataBuffer(int _capacity) ;
	virtual ~MPIRecvDataBuffer();

	// check to see if there is room for more MPI requests
	virtual bool canPushMPI() { return canPush(); };

	// add data to a different buffer during MPI transmission
	virtual int pushMPI(MPI_Request *req, DataBuffer::DataType const data);
	// check to see transfer is completed.  return completed data for further buffering or deletion.
	virtual int popMPI(DataBuffer::DataType* &data);


};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
