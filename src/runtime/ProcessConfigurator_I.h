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
	ProcessConfigurator_I(cciutils::SCIOLogger *_logger) : logger(_logger) {};
	virtual ~ProcessConfigurator_I() {};

	virtual bool init() = 0;
	virtual bool finalize() = 0;

	virtual bool configure(MPI_Comm &comm, Process * proc) = 0;
	virtual std::string getOutputDir() {  return cci::rt::CmdlineParser::getParamValueByName<std::string>(params, cci::rt::CmdlineParser::PARAM_OUTPUTDIR); };


protected:
	cciutils::SCIOLogger *logger;
	boost::program_options::variables_map params;
	std::string executable;


};

} /* namespace rt */
} /* namespace cci */
#endif /* PROCESSCONFIGURATOR_I_H_ */
