/*
 * RootedCommHandler_I.cpp
 *
 *  Created on: Jun 25, 2012
 *      Author: tcpan
 */

#include "RootedCommHandler_I.h"
#include <sstream>
#include "mpi.h"

namespace cci {
namespace rt {


const int RootedCommHandler_I::CONTROL_TAG = 0;
const int RootedCommHandler_I::DATA_TAG = 1;


RootedCommHandler_I::RootedCommHandler_I(MPI_Comm const * _parent_comm, int const _gid, std::vector<int> _roots) :
	Communicator_I(_parent_comm, _gid), action(NULL), msgToManager(true), status(READY) {

	printf("rootedCommHandler: ROOT: %d\n", _roots[0]);
	std::sort(_roots.begin(), _roots.end());
	isRoot = std::binary_search(_roots.begin(), _roots.end(), pcomm_rank);

	int *recvRoots = new int[size];
	int rootRank = -1;
	if (isRoot) rootRank = rank;
	MPI_Allgather(&rootRank, 1, MPI_INT, recvRoots, 1, MPI_INT, comm);
	roots.clear();
	for (int i = 0; i < size; i++) {
		if (recvRoots[i] > -1) roots.push_back(recvRoots[i]);
	}
	delete [] recvRoots;
	Debug::print("rootedCommHandler: ROOT: size %d, %d\n", roots.size(), roots[0]);
//		printf("rank: %d is root? %s\n ", pcomm_rank, (isRoot ? "true" : "false"));

	if (comm == MPI_COMM_NULL) {
		Debug::print("ERROR: comm member %d must have a MPI communicator.\n", rank);
		status = ERROR;
	}

	if (isRoot) {
		int size;
		MPI_Comm_size(comm, &size);

		for (int i = 0; i < size; ++i) {
			activeWorkers[i] = READY;
		}
		for (int i = 0; i < roots.size(); ++i) {
			activeWorkers.erase(roots[i]);
		}
	}

	std::stringstream ss;
	for (int i = 0; i < roots.size(); ++i) {
		ss << roots[i] << ", ";
	}
	Debug::print("localrank: %d, gid: %d, roots: %s.  IsRoot? %s\n", rank, groupid, ss.str().c_str(), (isRoot ? "true":"false"));
	ss.str(std::string());

}

RootedCommHandler_I::~RootedCommHandler_I() {
	//printf("RootedCommHandler destructor called\n");
	if (action->dereference(this) == 0) delete action;

	roots.clear();

}

} /* namespace rt */
} /* namespace cci */
