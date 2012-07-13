/*
 * Communicator_I.cpp
 *
 *  Created on: Jun 25, 2012
 *      Author: tcpan
 */

#include "Communicator_I.h"
#include "Debug.h"

#include <iostream>

namespace cci {
namespace rt {


const int Communicator_I::READY = 1;
const int Communicator_I::WAIT = 2;
const int Communicator_I::DONE = 0;
const int Communicator_I::ERROR = -1;

Communicator_I::Communicator_I(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogSession *_logsession) :
	groupid(_gid), parent_comm(_parent_comm), call_count(0), logsession(_logsession) {

	long long t1, t2;
	t1 = ::cciutils::event::timestampInUS();
	pcomm_rank = -1;
	rank = -1;
	pcomm_size = 0;
	size = 0;

	if (groupid == -1) comm = MPI_COMM_NULL;
	else {
		MPI_Comm_rank(*parent_comm, &pcomm_rank);
		MPI_Comm_size(*parent_comm, &pcomm_size);


		MPI_Comm_split(*parent_comm, groupid, pcomm_rank, &comm);
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
	}
	gethostname(hostname, 255);  // from <iostream>
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("MPI setup"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));
};
Communicator_I::~Communicator_I() {
	if (!reference_sources.empty()) Debug::print("%s ERROR:  still has %d objects referencing it\n", getClassName(), reference_sources.size());

	if (comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
};


} /* namespace rt */
} /* namespace cci */
