/*
 * Schedule.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tcpan
 */

#ifndef SCHEDULE_H_
#define SCHEDULE_H_

#include "Process.h"
#include <vector>

namespace cci {

namespace runtime {

/**
 * performs scheduling.
 * each node has a set of conceptual input and a set of conceptual outputs
 * each input or output can have a set of instances.
 *
 * this scheduler performs assignment on a set of instances.
 * the parameter should match what the process's work object expects.
 */
class Schedule {
public:
	Schedule() {};
	virtual ~Schedule() {};

	/**
	 * returns is of process that the params was assigned to.
	 * does not actually invoke the process.
	 */
	virtual int assign(std::vector<Process *> &processes) = 0;

};

}

}

#endif /* SCHEDULE_H_ */
