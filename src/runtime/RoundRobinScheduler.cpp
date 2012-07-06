/*
 * RoundRobinScheduler.cpp
 *
 *  Created on: Jun 29, 2012
 *      Author: tcpan
 */

#include "RoundRobinScheduler.h"

namespace cci {
namespace rt {

RoundRobinScheduler::RoundRobinScheduler(std::vector<int> &_roots, std::vector<int> &_leaves) :
	Scheduler_I(_roots, _leaves) {
	rootIdx = roots.end();
	leafIdx = leaves.end();
}

RoundRobinScheduler::RoundRobinScheduler(bool _root, bool _leaf) :
	Scheduler_I(_root, _leaf) {
	rootIdx = roots.end();
	leafIdx = leaves.end();
}
int RoundRobinScheduler::getRootFromLeaf(int leafId) {
	int size = roots.size();
	if (size == 0) return -1;
	else if (size == 1) return roots[0];
	else {
		++rootIdx;
		if (rootIdx == roots.end()) rootIdx = roots.begin();
		return *rootIdx;
	}
}
int RoundRobinScheduler::getLeafFromRoot(int rootId) {
	int size = leaves.size();
	if (size == 0) return -1;
	else if (size == 1) return leaves[0];
	else {
		++leafIdx;
		if (leafIdx == leaves.end()) leafIdx = leaves.begin();
		return *leafIdx;
	}
}

int RoundRobinScheduler::removeRoot(int id) {
	// save the value
	if (*rootIdx == id) {
		++rootIdx;
		if (rootIdx == roots.end()) rootIdx = roots.begin();
	}
	int val = *rootIdx;

	int count = Scheduler_I::removeRoot(id);

	// reset the iterators
	rootIdx = find(roots.begin(), roots.end(), val);
	return count;
}
int RoundRobinScheduler::removeLeaf(int id) {
	// save the value
	if (*leafIdx == id) {
		++leafIdx;
		if (leafIdx == leaves.end()) leafIdx = leaves.begin();
	}
	int val = *leafIdx;

	int count = Scheduler_I::removeLeaf(id);

	// reset the iterators
	leafIdx = find(leaves.begin(), leaves.end(), val);
	return count;
}
int RoundRobinScheduler::addRoot(int id) {
	int val = *rootIdx;

	int count = Scheduler_I::addRoot(id);

	// reset the iterators
	rootIdx = find(roots.begin(), roots.end(), val);
	return count;
}
int RoundRobinScheduler::addLeaf(int id) {
	int val = *leafIdx;

	int count = Scheduler_I::addLeaf(id);

	// reset the iterators
	leafIdx = find(leaves.begin(), leaves.end(), val);
	return count;
}


} /* namespace rt */
} /* namespace cci */
