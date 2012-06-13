/*
 * FIFOSchedule.cpp
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#include "FIFOSchedule.h"

namespace cci {

namespace runtime {

int FIFOSchedule::assign(std::vector<Process *> &processes) {
	// check to see who is ready
	bool wrapped = false;
	int i = lastId;
	lastId = -1;
	int status;
	if (forInput) {
		status = Process::DONE;
	} else {
		status = Process::READY;
	}
	// find a process that is "done"
	for (; i < processes.size() && !wrapped;) {
		if (processes[i].getStatus() == status) {
			lastId = i;
			break;
		}
		++i;
		if (i >= process.size()) {
			i -= process.size();
			wrapped = true;
		}
	}
	return lastId;
}

}

}
