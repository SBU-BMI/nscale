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


namespace cci {
namespace rt {
namespace adios {


class SegConfigurator : public cci::rt::ProcessConfigurator_I {
public:
	SegConfigurator(std::string &_iocode) :
		iomanager(NULL), iocode(_iocode) {};
	virtual ~SegConfigurator() {
		if (iomanager != NULL) {
			delete iomanager;
			iomanager = NULL;
		}
	};

	virtual bool init(cciutils::SCIOLogger *_logger);
	virtual bool finalize();

	virtual bool configure(MPI_Comm &comm, Process *proc);

	static const int UNDEFINED_GROUP;
	static const int COMPUTE_GROUP;
	static const int IO_GROUP;
	static const int COMPUTE_TO_IO_GROUP;
	static const int UNUSED_GROUP;

protected:
	ADIOSManager *iomanager;
	std::string iocode;

};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SEGCONFIGURATOR_H_ */
