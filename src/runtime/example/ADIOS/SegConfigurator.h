/*
 * SegConfigurator.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef SEGCONFIGURATOR_H_
#define SEGCONFIGURATOR_H_

#include <ProcessConfigurator_I.h>
#include "UtilsADIOS.h"
#include "SCIOUtilsLogger.h"
#include "SegmentCmdParser.h"

namespace cci {
namespace rt {
namespace adios {


class SegConfigurator : public cci::rt::ProcessConfigurator_I {
public:
	SegConfigurator(SegmentCmdParser::ParamsType &_params, cciutils::SCIOLogger *_logger) :
		ProcessConfigurator_I(_logger), iomanager(NULL), params(_params) {};
	virtual ~SegConfigurator() {
		finalize();
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
	ADIOSManager *iomanager;
	SegmentCmdParser::ParamsType params;
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SEGCONFIGURATOR_H_ */
