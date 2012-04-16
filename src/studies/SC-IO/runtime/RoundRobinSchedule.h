/*
 * RoundRobinSchedule.h
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#ifndef ROUNDROBINSCHEDULE_H_
#define ROUNDROBINSCHEDULE_H_

#include "Schedule.h"

namespace cci {

namespace runtime {

class RoundRobinSchedule: public cci::runtime::Schedule {
public:
	RoundRobinSchedule() : id(-1) {};
	virtual ~RoundRobinSchedule() {};

	virtual int assign(std::vector<Process *> &processes);

private:
	int id;
};

}

}

#endif /* ROUNDROBINSCHEDULE_H_ */
