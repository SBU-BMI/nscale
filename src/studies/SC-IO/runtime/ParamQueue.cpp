/*
 * ParamQueue.cpp
 *
 *  Created on: Apr 11, 2012
 *      Author: tcpan
 */

#include "ParamQueue.h"

namespace cci {

namespace runtime {

ParamQueue::ParamQueue(const int _capacity, const Work *_work) : capacity(_capacity), w(_work) {
	staged.clear();
	q.clear();

}

ParamQueue::~ParamQueue() {
	clear();
}

int ParamQueue::push(Parameters *param) {
	if (capacity == 0) return -1;  //full always

	uint64_t id = param->id;
	std::tr1::unordered_map<uint64_t, Parameters *>::const_iterator pos = staged.find(id);
	if (capacity > 0 ) { // limited size
		// if parameter is not already in the stage, compare size and capacity to see if can insert
		if (pos == staged.end() && staged.size() >= capacity) return -1; // full, can't insert
		// else either param already in the stage, or capacity is not reached yet, continue.
	} // else, unlimited capacity.  continue.

	Parameters *p;
	if (pos == staged.end()) {
		// if not already in the queue, add to stage
		staged.insert(std::make_pair<uint64_t, Parameter *>(id, param));
		p = param;
	} else {
		// if already in the stage, merge the one in staged with current to update its value
		p = pos->second;
		p->merge(*param);
	}

	// then check to see if it can go to queue.  if yes, add to queue
	if (w->hasRequired(p)) this->push_private(p);
	return 0;
}

void ParamQueue::push_private(Parameters *param) {
	q.push(param);
}


Parameters* ParamQueue::pop() {
	// if there is nothing in q (nothing ready) then return nothing
	if (q.empty()) return NULL;

	// if there is something, then currSize should be > 0.  pop it, and decrement count
	Parameters *p = q.front();
	q.pop();
	// also remove from allParams
	staged.erase(p->id);

	return p;
}

int ParamQueue::clear() {
	// don't clean up the Parameters object - done by ParamQueue
	this->clear_private();

	int count = staged.size();
	for (std::tr1::unordered_map<uint64_t, Parameters *>::iterator iter = staged.begin();
			iter < staged.end(); ++iter) {
		delete iter->second;
	}
	staged.clear();
	return count;
}

void ParamQueue::clear_private() {
	q.clear();
}


}

}
