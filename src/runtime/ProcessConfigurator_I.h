/*
 * ProcessConfigurator_I.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef PROCESSCONFIGURATOR_I_H_
#define PROCESSCONFIGURATOR_I_H_

#include "Communicator_I.h"
#include "mpi.h"
#include <vector>

namespace cci {
namespace rt {

class ProcessConfigurator_I {
public:
	ProcessConfigurator_I() {};
	virtual ~ProcessConfigurator_I() {};

	virtual bool configure(MPI_Comm &comm, std::vector<Communicator_I *> &handlers) = 0;
};

} /* namespace rt */
} /* namespace cci */
#endif /* PROCESSCONFIGURATOR_I_H_ */
