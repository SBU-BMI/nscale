/*
 * FIFOSchedule.h
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#ifndef FIFOSCHEDULE_H_
#define FIFOSCHEDULE_H_

#include "Schedule.h"

namespace cci {

namespace runtime {

class FIFOSchedule: public cci::runtime::Schedule {
public:
	FIFOSchedule(bool _forInput) : forInput(_forInput), lastId(0) {};
	virtual ~FIFOSchedule() {};

	/**
	 * assign by finding the first available.
	 * use the last found id as the new starting point - this avoids bias to the front of vector
	 * -1 means nothing is found.
	 */
	virtual int assign(std::vector<Process *> &processes);


private:
	bool forInput;
	int lastId;
};

}

}

#endif /* FIFOSCHEDULE_H_ */
