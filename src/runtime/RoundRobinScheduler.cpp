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
	Scheduler_I(_roots, _leaves), rootIdx(0), leafIdx(0) {
}

RoundRobinScheduler::RoundRobinScheduler(bool _root, bool _leaf) :
	Scheduler_I(_root, _leaf), rootIdx(0), leafIdx(0) {
}
int RoundRobinScheduler::getRootFromLeave(int leafId) {
	int size = roots.size();
	if (size == 0) return -1;
	else if (size == 1) return roots[0];
	else {
		int out= roots[this->rootIdx];
		rootIdx  = (rootIdx + 1) % size;
		return out;
	}
}
int RoundRobinScheduler::getLeafFromRoot(int rootId) {
	int size = leaves.size();
	if (size == 0) return -1;
	else if (size == 1) return leaves[0];
	else {
		int out= leaves[this->leafIdx];
		leafIdx  = (leafIdx + 1) % size;
		return out;
	}
}

} /* namespace rt */
} /* namespace cci */
