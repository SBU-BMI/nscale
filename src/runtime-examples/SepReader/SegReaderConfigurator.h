/*
 * SegReaderConfigurator.h
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#ifndef SegReaderConfigurator_H_
#define SegReaderConfigurator_H_

#include "ProcessConfigurator_I.h"
#include "UtilsADIOS.h"
#include "Logger.h"

#include "CmdlineParser.h"

namespace cci {
namespace rt {
namespace adios {


class SegReaderConfigurator : public cci::rt::ProcessConfigurator_I {
public:
	SegReaderConfigurator(int argc, char** argv);
	virtual ~SegReaderConfigurator() {
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
	static const int READ_GROUP;
	static const int WRITE_GROUP;
	static const int READ_TO_COMPUTE_GROUP;
	static const int COMPUTE_TO_WRITE_GROUP;
	static const int UNUSED_GROUP;

protected:
	cci::rt::adios::ADIOSManager *iomanager;
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* SegReaderConfigurator_H_ */
