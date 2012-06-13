/*
 * CommHandler_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef COMMHANDLER_I_H_
#define COMMHANDLER_I_H_

#include <stdio.h>
#include <string.h>
#include <mpi.h>

namespace cci {
namespace rt {

/**
 * handles the communication for 1 message exchange.
 * control messages are embedded and fixed for now
 * payload messages come out
 */
class CommHandler_I {
public:
	CommHandler_I(MPI_Comm const &_parent_comm, ) {};
	virtual ~CommHandler_I() {
		MPI_Comm_free(&comm);
	};

	virtual void exchange(int &size, char* &data) = 0;

private:
	MPI_Comm comm;

};

} /* namespace rt */
} /* namespace cci */
#endif /* COMMHANDLER_I_H_ */
