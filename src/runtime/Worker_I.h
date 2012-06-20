/*
 * Worker.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef WORKER_I_H_
#define WORKER_I_H_
#include "mpi.h"
#include "Communicator_I.h"

namespace cci {
namespace rt {

class Worker_I : public Communicator_I {
public:
	Worker_I(MPI_Comm const * _parent_comm, int const _gid) :
		Communicator_I(_parent_comm, _gid) {};
	virtual ~Worker_I() {};

	virtual int compute(int const &input_size , char* const &input,
			int &output_size, char* &output) = 0;

};

} /* namespace rt */
} /* namespace cci */
#endif /* WORKER_H_ */
