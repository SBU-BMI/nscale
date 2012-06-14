/*
 * UniformRandomSchedule.h
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#ifndef UNIFORMRANDOMSCHEDULE_H_
#define UNIFORMRANDOMSCHEDULE_H_

#include "Schedule.h"

namespace cci {

namespace runtime {

class UniformRandomSchedule: public cci::runtime::Schedule {
public:
	UniformRandomSchedule() {};
	virtual ~UniformRandomSchedule() {};

	static void initialize(const unsigned int seed);

	virtual int assign(std::vector<Process *> &processes);
};

}

}

#endif /* UNIFORMRANDOMSCHEDULE_H_ */
