/*
 * SchedulerI.h
 *
 *  Created on: Jun 29, 2012
 *      Author: tcpan
 */

#ifndef SCHEDULERI_H_
#define SCHEDULERI_H_

#include "mpi.h"
#include <vector>
#include <algorithm>

namespace cci {
namespace rt {

/**
 * scheduler manages the mapping of root and leaf
 *
 */
class Scheduler_I {
public:

	/**
	 * this version of the function takes a list of roots and leafs.  the list contain the ranks
	 * within the specified communicator.
	 */
	Scheduler_I(std::vector<int> &_roots, std::vector<int> &_leaves) :
		configured(false), root(false), leaf(false), comm(MPI_COMM_NULL), roots(_roots), leaves(_leaves), listform(true) {};
	/**
	 * this version of the function takes 2 flags:  isroot and isleaf.  allGather is used to construct
	 * the list.  all gather uses the communicator specified.
	 */
	Scheduler_I(bool _root, bool _leaf) :
		configured(false), root(_root), leaf(_leaf), comm(MPI_COMM_NULL), listform(false) {};
	virtual ~Scheduler_I() {
		roots.clear();
		leaves.clear();
	};

	void configure(MPI_Comm &_comm) {
		comm = _comm;

		if (listform) {
			int rank;
			MPI_Comm_rank(comm, &rank);

			std::sort(roots.begin(), roots.end());
			root = std::binary_search(roots.begin(), roots.end(), rank);
			std::sort(leaves.begin(), leaves.end());
			leaf = std::binary_search(leaves.begin(), leaves.end(), rank);
		} else {
			int size;
			MPI_Comm_size(comm, &size);

			// get the other node's info
			char *recvbuf = new char[size * 2];
			memset(recvbuf, 0, size * 2);
			char *sendbuf = new char[2];
			memset(sendbuf, 0, 2);
			if (root) sendbuf[0] = 1;
			if (leaf) sendbuf[1] = 1;
			MPI_Allgather(sendbuf, 2, MPI_CHAR, recvbuf, 2, MPI_CHAR, comm);
			roots.clear();
			leaves.clear();
			for (int i = 0; i < size; ++i) {
				if (recvbuf[2 * i] > 0) roots.push_back(i);
				if (recvbuf[2 * i + 1] > 0) leaves.push_back(i);
			}

			delete [] recvbuf;
			delete [] sendbuf;
		}

		configured = true;
	};

	bool isRoot() { return this->root; };
	bool isLeaf() { return this->leaf; };

	std::vector<int> &getRoots() { return roots; };
	std::vector<int> &getLeaves() { return leaves; };

	virtual int getRootFromLeave(int leafId) = 0;
	virtual int getLeafFromRoot(int rootId) = 0;

protected:
	std::vector<int> roots;
	std::vector<int> leaves;
	bool root;
	bool leaf;

	bool listform;
	bool configured;
	MPI_Comm comm;
};

} /* namespace rt */
} /* namespace cci */
#endif /* SCHEDULERI_H_ */
