/*
 * Activity.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "Activity.h"
#include "Debug.h"

namespace cci {
namespace rt {

Activity::Activity(CommHandler_I * in, CommHandler_I * out, Worker_I * w) :
				input(in), output(out), worker(w) {

}

Activity::~Activity() {
}

int Activity::process() {
	int isize, osize;
	char *idata, *odata;
	int result = 0;

	if (input != NULL) {
		result = input->exchange(isize, idata);
		if (result == -1) return result;
	}
	if (worker != NULL) {
		result = worker->compute(isize, idata, osize, odata);
		if (result == -1) return result;
	}

	if (output != NULL) {
		result = output->exchange(osize, odata);
		if (result == -1) return result;
	}

	return 1;
}

void Activity::register_listener(std::tr1::unordered_map<MPI_Comm *, Activity *> &listeners) {

	if (input != NULL) {
//		if (input->isListener()) {
			Debug::print("registering listener %d for activity %lu\n", *(input->getComm()), this);
			listeners[input->getComm()] = this;
//		}
	}
	if (output != NULL) {
//		if (output->isListener()) {
			Debug::print("registering listener %d for activity %lu\n", *(output->getComm()), this);
			listeners[output->getComm()] = this;
//		} else {
//			printf("registering requester %d for activity %lu\n", *(output->getComm()), this);
//			requesters[output->getComm()] = this;
//		}
	}
}

} /* namespace rt */
} /* namespace cci */
