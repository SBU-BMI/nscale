/*
 * PrioritizedParamQueue.h
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#ifndef PRIORITIZEDPARAMQUEUE_H_
#define PRIORITIZEDPARAMQUEUE_H_

#include "ParamQueue.h"
#include <queue>

namespace cci {

namespace runtime {

class ParamPrioritizer {
private:
	Work *w;
public:
	ParamPrioritizer(const Work *_w) : w(_w) {};
	bool operator() (const Parameters * lhs, const Parameters * rhs) const;
};

class PrioritizedParamQueue: public cci::runtime::ParamQueue {
public:
	PrioritizedParamQueue(const int _capacity, const Work *_work);
	virtual ~PrioritizedParamQueue();

	virtual Parameters *pop();

protected:
	virtual int push_private(Parameters *param);
	virtual int clear_private();

private:
	std::priority_queue<Parameters *> pq;
};

}

}

#endif /* PRIORITIZEDPARAMQUEUE_H_ */
