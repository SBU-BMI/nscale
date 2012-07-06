/*
 * SegConfigurator.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef SEGCONFIGURATOR_H_
#define SEGCONFIGURATOR_H_

#include <ProcessConfigurator_I.h>
#include "SCIOUtilsADIOS.h"
#include "SCIOUtilsLogger.h"


namespace cci {
namespace rt {
namespace adios {


class SegConfigurator : public cci::rt::ProcessConfigurator_I {
public:
	SegConfigurator(std::string &_iocode) : iomanager(NULL), iocode(_iocode), logger(NULL) {};
	virtual ~SegConfigurator() {
		if (iomanager != NULL) {
			delete iomanager;
			iomanager = NULL;
		}
		if (logger != NULL) {
			delete logger;
			logger = NULL;
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
	cciutils::ADIOSManager *iomanager;
	std::string iocode;
	cciutils::SCIOLogger * logger;
	char hostname[256];


};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SEGCONFIGURATOR_H_ */
