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
#include "Logger.h"

#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace adios {


class SegConfigurator : public cci::rt::ProcessConfigurator_I {
public:
	SegConfigurator(int argc, char** argv, cci::common::Logger *_logger=NULL);
	virtual ~SegConfigurator() {
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
	ADIOSManager *iomanager;
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SEGCONFIGURATOR_H_ */
