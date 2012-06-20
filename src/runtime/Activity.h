/*
 * Activity.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef ACTIVITY_H_
#define ACTIVITY_H_

#include "Communicator_I.h"
#include "Worker_I.h"
#include <tr1/unordered_map>

namespace cci {
namespace rt {

class Activity {
public:
	Activity(Communicator_I * in, Communicator_I * out, Worker_I * w);
	virtual ~Activity();

	virtual int process();
	virtual void register_listener(std::tr1::unordered_map<MPI_Comm *, Activity *> &listeners);

private:
	Worker_I *worker;
	Communicator_I *input;
	Communicator_I *output;
};

} /* namespace rt */
} /* namespace cci */
#endif /* ACTIVITY_H_ */
