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

	MPIDataBuffer(int _capacity) : DataBuffer(_capacity) {};
	virtual ~MPIDataBuffer() {};

	// for data addition
	virtual int getBufferSize() { return buffer.size() + mpi_buffer.size(); };
	virtual bool canPop() { return buffer.size() > 0; };

	// for data undergoing MPI transmission
	virtual int getMPIBufferSize() { return mpi_buffer.size(); };
	// check to see if there is room for more MPI requests
	virtual bool canPushMPI() = 0;
	virtual bool canPopMPI() { return getMPIBufferSize() > 0; };

	// add data to a different buffer during MPI transmission
	virtual int pushMPI(MPI_Request *req, DataBuffer::DataType const data) = 0;
	// check to see transfer is completed.  return completed data for further buffering or deletion.
	virtual int popMPI(DataBuffer::DataType* &data) = 0;


protected:
	std::tr1::unordered_map<MPI_Request*, DataBuffer::DataType> mpi_buffer;

	virtual void dumpBuffer() {
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

	};


};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
