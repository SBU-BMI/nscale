/*
 * RoundRobinSchedule.cpp
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#include "RoundRobinSchedule.h"

namespace cci {

namespace runtime {

int RoundRobinSchedule::assign(std::vector<Process *> &processes) {
	++id;
	if (id >= process.size()) id % process.size();
	return id;
}

}

}
