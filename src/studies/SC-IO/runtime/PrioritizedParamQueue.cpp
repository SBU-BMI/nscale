/*
 * PrioritizedParamQueue.cpp
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#include "PrioritizedParamQueue.h"

namespace cci {

namespace runtime {

bool ParamPrioritizer::operator() (const Parameters * lhs, const Parameters * rhs) {
	return w->estimatePerformance(lhs) > w->estimatePerformance(rhs);
}

PrioritizedParamQueue::PrioritizedParamQueue(const int _capacity, const Work *_work) :
		capacity(_capacity), w(_work), pq(ParamPrioritizer(_work)) {
	staged.clear();
	pq.clear();

}

PrioritizedParamQueue::~PrioritizedParamQueue() {
	clear();
}


void PrioritizedParamQueue::push_private(Parameters *param) {
	pq.push(param);
}

Parameters* PrioritizedParamQueue::pop() {
	// if there is nothing in q (nothing ready) then return nothing
	if (pq.empty()) return NULL;

	// if there is something, then currSize should be > 0.  pop it, and decrement count
	Parameters *p = pq.top();
	pq.pop();
	// also remove from allParams
	staged.erase(p->id);

	return p;
}


void PrioritizedParamQueue::clear_private() {
	pq.clear();
}



}

}
