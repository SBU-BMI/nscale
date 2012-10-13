/*
 * SynDataConfiguratorPush.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef SYNDATACONFIGURATOR_PUSH_H_
#define SYNDATACONFIGURATOR_PUSH_H_

#include <ProcessConfigurator_I.h>
#include "UtilsADIOS.h"
#include "SCIOUtilsLogger.h"
#include "SynDataCmdParser.h"

namespace cci {
namespace rt {
namespace syntest {


class SynDataConfiguratorPush : public cci::rt::ProcessConfigurator_I {
public:
	SynDataConfiguratorPush(SynDataCmdParser::ParamsType &_params, cciutils::SCIOLogger *_logger) :
		ProcessConfigurator_I(_logger), iomanager(NULL), params(_params) {};
	virtual ~SynDataConfiguratorPush() {
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
	SynDataCmdParser::ParamsType params;
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SYNDATACONFIGURATOR_H_ */
