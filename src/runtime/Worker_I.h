/*
 * Worker.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef WORKER_I_H_
#define WORKER_I_H_

namespace cci {
namespace rt {

class Worker_I {
public:
	virtual ~Worker_I() {};

	virtual void compute(int const &input_size , char* const &input,
			int &output_size, char* &output) = 0;

};

} /* namespace rt */
} /* namespace cci */
#endif /* WORKER_H_ */
