/*
 * SimpleWorker.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef SIMPLEWORKER_H_
#define SIMPLEWORKER_H_

#include "Worker_I.h"

namespace cci {
namespace rt {

class SimpleWorker: public cci::rt::Worker_I {
public:
	SimpleWorker();
	virtual ~SimpleWorker();
};

} /* namespace rt */
} /* namespace cci */
#endif /* SIMPLEWORKER_H_ */
