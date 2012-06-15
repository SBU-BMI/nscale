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
	CommHandler_I(MPI_Comm const &_parent_comm, int _gid) :
		parent_comm(_parent_comm), groupid(_gid) {
		int rank;
		MPI_Comm_rank(parent_comm, &rank);
		MPI_Comm_split(parent_comm, groupid, rank, &comm);

	};
	virtual ~CommHandler_I() {
		MPI_Comm_free(&comm);
	};

	virtual void exchange(int &size, char* &data) = 0;

	MPI_Comm &getComm() { return comm; };
private:
	MPI_Comm parent_comm,comm;
	int groupid;

};

} /* namespace rt */
} /* namespace cci */
#endif /* COMMHANDLER_I_H_ */
