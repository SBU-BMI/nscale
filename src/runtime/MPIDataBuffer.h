/*
 * MPIDataBuffer.h
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#ifndef MPIDATABUFFER_H_
#define MPIDATABUFFER_H_

#include <DataBuffer.h>

namespace cci {
namespace rt {

class MPIDataBuffer: public cci::rt::DataBuffer {
public:
	MPIDataBuffer();
	virtual ~MPIDataBuffer();

	// for data undergoing MPI transmission
	int getMPIBufferSize() = 0;


	// add data to a different buffer during MPI transmission
	int bufferTransfer() = 0;
	// check to see transfer is completed.  return completed data for further buffering or deletion.
	int transferComplete() = 0;

	bool canBufferTransfer() = 0;
};

} /* namespace rt */
} /* namespace cci */
#endif /* MPIDATABUFFER_H_ */
