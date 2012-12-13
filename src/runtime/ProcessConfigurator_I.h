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
#include "Process.h"
#include <vector>
#include "boost/program_options.hpp"
#include "CmdlineParser.h"

namespace cci {
namespace rt {

class Process;

class ProcessConfigurator_I {
public:
	ProcessConfigurator_I() : logger(NULL) {};
	virtual ~ProcessConfigurator_I() {
		if (logger != NULL) delete logger;
	};

	virtual bool init() = 0;
	virtual bool finalize() = 0;

	virtual bool configure(MPI_Comm &comm, Process * proc) = 0;

	virtual cci::common::Logger *getLogger() { return logger; };

protected:
	cci::common::Logger *logger;
	boost::program_options::variables_map params;
	std::string executable;

};

} /* namespace rt */
} /* namespace cci */
#endif /* PROCESSCONFIGURATOR_I_H_ */
