/*
 * Work.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tcpan
 */

#ifndef WORK_H_
#define WORK_H_

#include "mpi.h"
#include "Parameters.h"

namespace cci {

namespace runtime {

class Work {
public:
	Work() {};
	virtual ~Work() {};

	virtual int perform(const Parameters *_param) = 0;

	// check to see if parameters for a work object are all present.
	virtual bool hasRequired(const Parameters *_param) = 0;
	virtual float estimatePerformance(const Parameters *_param) = 0;

};

}

}

#endif /* WORK_H_ */
