/*
 * SegConfigurator.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef SEGCONFIGURATOR_H_
#define SEGCONFIGURATOR_H_

#include <ProcessConfigurator_I.h>

namespace cci {
namespace rt {

class SegConfigurator: public cci::rt::ProcessConfigurator_I {
public:
	SegConfigurator();
	virtual ~SegConfigurator();

	virtual bool configure(MPI_Comm &comm, std::vector<Communicator_I *> &handlers);
};

} /* namespace rt */
} /* namespace cci */
#endif /* SEGCONFIGURATOR_H_ */
