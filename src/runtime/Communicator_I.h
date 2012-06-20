/*
 * Communicator_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef COMMUNICATOR_I_H_
#define COMMUNICATOR_I_H_

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
class Communicator_I {
public:
	Communicator_I(MPI_Comm const * _parent_comm, int const _gid) :
		groupid(_gid), parent_comm(_parent_comm), call_count(0) {
		pcomm_rank = -1;
		rank = -1;

		if (groupid == -1) comm = MPI_COMM_NULL;
		else {
			MPI_Comm_rank(*parent_comm, &pcomm_rank);

			MPI_Comm_split(*parent_comm, groupid, pcomm_rank, &comm);
			MPI_Comm_rank(comm, &rank);
		}
	};
	virtual ~Communicator_I() {
		//printf("Communicator destructor called. %d in group %d\n", rank, groupid);
		if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
	};

	MPI_Comm * getComm() { return &comm; };
protected:
	MPI_Comm const * parent_comm;
	MPI_Comm comm;
	int const groupid;
	int rank;
	int pcomm_rank;

	// some basic metadata tracking.
	long call_count;
};

} /* namespace rt */
} /* namespace cci */
#endif /* COMMUNICATOR_I_H_ */
