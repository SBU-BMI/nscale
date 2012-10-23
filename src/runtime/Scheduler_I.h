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
#include <string.h>

#include <iterator>
#include <iostream>

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
	Scheduler_I(std::vector<int> const &_roots, std::vector<int> const &_leaves) :
		configured(false), root(false), leaf(false), comm(MPI_COMM_NULL), roots(_roots), leaves(_leaves), listform(true) {

		// remove any duplicates
		sort(roots.begin(), roots.end());
		std::vector<int>::iterator it = unique(roots.begin(), roots.end());
		roots.resize( it - roots.begin() );

		sort(leaves.begin(), leaves.end());
		it = unique(leaves.begin(), leaves.end());
		leaves.resize( it - leaves.begin() );

//		printf("superclass roots.size = %ld\n", roots.size());
//		printf("superclass leaves.size = %ld\n", leaves.size());


	};
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
		MPI_Comm_rank(comm, &rank);

		if (listform) {

			if (binary_search(roots.begin(), roots.end(), rank)) root = true;
			if (binary_search(leaves.begin(), leaves.end(), rank)) leaf = true;
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

			// now populate the list.  guaranteed to be sorted and unique
			roots.clear();
			leaves.clear();
			for (int i = 0; i < size; ++i) {
				if (recvbuf[2 * i] > 0) roots.push_back(i);
				if (recvbuf[2 * i + 1] > 0) leaves.push_back(i);
			}

			delete [] recvbuf;
			delete [] sendbuf;
		}

		postConfigure();

		configured = true;

	};


	virtual int removeRoot(int id) {
//		printf("Remove Scheduler Root called\n");

		std::vector<int>::iterator it = std::find(roots.begin(), roots.end(), id);
		if (it != roots.end()) roots.erase(it);
		if (rank == id) root = false;
		return roots.size();
	}
	virtual int removeLeaf(int id) {
//		printf("Remove Scheduler leaf called\n");
		std::vector<int>::iterator it = std::find(leaves.begin(), leaves.end(), id);
		if (it != leaves.end()) leaves.erase(it);
		if (rank == id) leaf = false;
		return leaves.size();
	}
	virtual int addRoot(int id) {
//		printf("add Scheduler Root called\n");

		if (std::binary_search(roots.begin(), roots.end(), id)) return roots.size();

		roots.push_back(id);
		sort(roots.begin(), roots.end());

		if (rank == id) root = true;
		return roots.size();
	}
	virtual int addLeaf(int id) {
//		printf("add Scheduler leaf called\n");

		if (std::binary_search(leaves.begin(), leaves.end(), id)) return leaves.size();

		leaves.push_back(id);
		sort(leaves.begin(), leaves.end());

		if (rank == id) leaf = true;
		return leaves.size();
	}

	// MPI_ANY_SOURCE means self.
	bool isRoot(int rank = MPI_ANY_SOURCE) {
		if (rank == MPI_ANY_SOURCE) return this->root;
		else return (std::find(roots.begin(), roots.end(), rank) != roots.end());
	};
	bool isLeaf(int rank = MPI_ANY_SOURCE) {
		if (rank == MPI_ANY_SOURCE) return this->leaf;
		else return (std::find(leaves.begin(), leaves.end(), rank) != leaves.end());
	};

	std::vector<int> &getRoots() { return roots; };
	std::vector<int> &getLeaves() { return leaves; };
	virtual bool hasRoots() { return roots.size() > 0; };
	virtual bool hasLeaves() { return leaves.size() > 0; };

	virtual int getRootFromLeaf(int leafId) = 0;
	virtual int getLeafFromRoot(int rootId) = 0;

protected:
	std::vector<int> roots;
	std::vector<int> leaves;
	bool root;
	bool leaf;

	bool listform;
	bool configured;

	MPI_Comm comm;
	int rank;

	virtual void postConfigure() = 0;

};

} /* namespace rt */
} /* namespace cci */
#endif /* SCHEDULERI_H_ */
