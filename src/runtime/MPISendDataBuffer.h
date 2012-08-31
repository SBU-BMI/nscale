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

#include <DataBuffer.h>
#include "mpi.h"
#include <tr1/unordered_map>

namespace cci {
namespace rt {

class MPISendDataBuffer: public cci::rt::DataBuffer {
public:

	MPISendDataBuffer(int _capacity) ;
	virtual ~MPISendDataBuffer();

	// for data addition
	virtual int getBufferSize() { return buffer.size() + mpi_buffer.size(); };
	virtual bool canPop() { return buffer.size() > 0; };

	// for data undergoing MPI transmission
	virtual int getMPIBufferSize() { return mpi_buffer.size(); };
	// check to see if there is room for more MPI requests
	virtual bool canPushMPI() { return !isFull(); };  // ready or stopped, can push
	virtual bool canPopMPI() { return getMPIBufferSize() > 0; };  // ready or stopped, can pop.

	// add data to a different buffer during MPI transmission
	virtual int pushMPI(MPI_Request *req, DataBuffer::DataType const data);  // fairly standard.
	// check to see transfer is completed.  return completed data for further buffering or deletion.
	virtual int popMPI(DataBuffer::DataType* &data);  // check all at once.  return the completed.


protected:
	std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType> mpi_buffer;

	virtual void dumpBuffer();

};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
