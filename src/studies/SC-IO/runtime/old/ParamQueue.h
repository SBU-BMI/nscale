/*
 * ParamQueue.h
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#ifndef PARAMQUEUE_H_
#define PARAMQUEUE_H_

#include "Parameters.h"
#include "Work.h"
#include <tr1/unordered_map>
#include <queue>

namespace cci {

namespace runtime {

/**
 * base class for a queue of parameters.  specialized so that we have a staging area
 * to allow for partial results to accumulate.
 *
 * the total size is limited.  meaning the number of
 * unique parameters is capacity
 *
 * if queue is full, push completes correctly if the param updates an existing param
 * else it returns an error.
 *
 * pop and clear works as the semantics indicate.
 *
 */
class ParamQueue {
public:
	ParamQueue(const int _capacity, const Work *_work);
	virtual ~ParamQueue();

	/**
	 * push caches the parameter object
	 * implementation merges the input with what's already in staging if needed
	 * when appropriate.
	 *
	 * returns -1 if full/failed.
	 *
	 * this is a possible place where if a task is restarted or if a "competition" is
	 * performed, there may be "dead params" in staged that will never clear and therefore
	 * cause deadlock.
	 */
	virtual int push(Parameters *param);

	/**
	 * pop gets the next parameter object
	 */
	virtual Parameters *pop();

	/**
	 * clears the stage container and release memory for objects inside that.
	 */
	virtual int clear();

protected:
	// all the current parameters objects.
	std::tr1::unordered_map<uint64_t, Parameters *> staged;

	// subclass should have some way of storing the "ready" parameters.

	// maximum capacity.  -1 means no max.
	int capacity;

	// the work associated with this parameter queue
	Work *w;

	virtual int push_private(Parameters *param);
	virtual void clear_private();

private:
	// change for implementation.

	// holds copy of pointers to the objects ready to be processed.
	// - don't delete the objects...
	std::queue<Parameters *> q;
};

}

}

#endif /* PARAMQUEUE_H_ */
