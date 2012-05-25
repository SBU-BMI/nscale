/*
 * Worker.h
 *
 *  Created on: May 25, 2012
 *      Author: tcpan
 */

#ifndef WORKER_H_
#define WORKER_H_

namespace cci {

namespace runtime {

/**
 * represents an activity to perform some computation/work.
 *
 */
class Worker {
public:
	Worker() {};
	virtual ~Worker() {};

	virtual int perform() = 0;
};

}}
#endif /* WORKER_H_ */
