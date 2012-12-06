/*
 * SynDataConfiguratorFull.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef SYNDATACONFIGURATOR_FULL_H_
#define SYNDATACONFIGURATOR_FULL_H_

#include "ProcessConfigurator_I.h"
#include "UtilsADIOS.h"
#include "Logger.h"

#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace syntest {


class SynDataConfiguratorFull : public cci::rt::ProcessConfigurator_I {
public:
	SynDataConfiguratorFull(int argc, char** argv, cci::common::Logger *_logger = NULL);
	virtual ~SynDataConfiguratorFull() {
		if (iomanager != NULL) {
			delete iomanager;
			iomanager = NULL;
		}
	};

	virtual bool init();
	virtual bool finalize();

	virtual bool configure(MPI_Comm &comm, Process *proc);

	static const int UNDEFINED_GROUP;
	static const int COMPUTE_GROUP;
	static const int IO_GROUP;
	static const int COMPUTE_TO_IO_GROUP;
	static const int UNUSED_GROUP;


protected:
	cci::rt::adios::ADIOSManager *iomanager;
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SYNDATACONFIGURATOR_FULL_H_ */
