/*
 * Activity_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef ACTIVITY_I_H_
#define ACTIVITY_I_H_

#include "mpi.h"
#include "Worker_I.h"
#include "CommHandler_I.h"

namespace cci {
namespace rt {

class Activity_I {
public:
	Activity_I(CommHandler_I* in, CommHandler_I* out, Worker_I* w) :
		input(in), output(out), worker(w) {};
	virtual ~Activity_I() {};

	virtual int process() = 0;

private:
	Worker_I *worker;
	CommHandler_I *input, *output;
};

} /* namespace rt */
} /* namespace cci */
#endif /* ACTIVITY_I_H_ */
