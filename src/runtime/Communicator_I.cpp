/*
 * Communicator_I.cpp
 *
 *  Created on: Jun 25, 2012
 *      Author: tcpan
 */

#include "Communicator_I.h"
#include "Debug.h"
#include <unistd.h>

#include <iostream>

namespace cci {
namespace rt {


const int Communicator_I::READY = 1;
const int Communicator_I::WAIT = 2;
const int Communicator_I::DONE = 0;
const int Communicator_I::ERROR = -1;

Communicator_I::Communicator_I(MPI_Comm const * _parent_comm, int const _gid, cci::common::LogSession *_logsession) :
	groupid(_gid), call_count(0), logsession(_logsession), waComm(NULL), comm(MPI_COMM_NULL), rank(MPI_UNDEFINED), size(0) {

	memset(hostname, 0, 256);


	long long t1 = ::cci::common::event::timestampInUS();

	int pcomm_rank = MPI_UNDEFINED;
	if (_parent_comm != NULL && *_parent_comm != MPI_COMM_NULL) {
		MPI_Comm_rank(*_parent_comm, &pcomm_rank);

		if (groupid == MPI_UNDEFINED) {
		} else if (groupid >= 0){

			MPI_Comm_split(*_parent_comm, groupid, pcomm_rank, &comm);
			MPI_Comm_rank(comm, &rank);
			MPI_Comm_size(comm, &size);
			waComm = new cci::common::mpi::waMPI(comm);
		} else {
			cci::common::Debug::print("ERROR: parent comm is NULL\n");
		}
	} else {
		printf("ERROR: parent communicator is NULL.\n");
	}
	gethostname(hostname, 255);  // from <iostream>
	long long t2 = ::cci::common::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cci::common::event(0, std::string("MPI setup"), t1, t2, std::string(), ::cci::common::event::NETWORK_IO));
};
Communicator_I::~Communicator_I() {
	if (waComm != NULL) delete waComm;

	if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
};


} /* namespace rt */
} /* namespace cci */
