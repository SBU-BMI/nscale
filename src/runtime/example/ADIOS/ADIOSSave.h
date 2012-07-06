/*
 * ADIOSSave.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef ADIOSSAVE_H_
#define ADIOSSAVE_H_

#include <Action_I.h>
#include "SCIOUtilsADIOS.h"
#include "SCIOUtilsLogger.h"
#include "mpi.h"

namespace cci {
namespace rt {
namespace adios {


class ADIOSSave: public cci::rt::Action_I {
public:
	ADIOSSave(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogger *_logger, cciutils::ADIOSManager *_iomanager, std::string &iocode);
	virtual ~ADIOSSave();
	virtual int run();
	virtual const char* getClassName() { return "ADIOSSave"; };


protected:
	int write();
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

	cciutils::ADIOSManager *iomanager;
	cciutils::SCIOADIOSWriter *writer;
	int local_iter;
	int global_iter;
	int local_count;
	MPI_Win iter_win;
	int buffer_max;

	int local_total;

};

}
} /* namespace rt */
} /* namespace cci */
#endif /* ADIOSSAVE_H_ */
