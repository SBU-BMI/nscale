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

#include "mpi.h"
#include <tr1/unordered_map>
#include "MPIDataBuffer.h"

namespace cci {
namespace rt {

class MPISendDataBuffer: public cci::rt::MPIDataBuffer {
public:

	MPISendDataBuffer(int _capacity) ;
	virtual ~MPISendDataBuffer();

	// check to see if there is room for more MPI requests
	virtual bool canPushMPI() { return !isFull(); };  // ready or stopped, can push

	// add data to a different buffer during MPI transmission
	virtual int pushMPI(MPI_Request *req, DataBuffer::DataType const data);  // fairly standard.
	// check to see transfer is completed.  return completed data for further buffering or deletion.
	virtual int popMPI(DataBuffer::DataType* &data);  // check all at once.  return the completed.


};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
