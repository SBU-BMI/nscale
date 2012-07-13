/*
 * ADIOSSave.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef ADIOSSAVE_H_
#define ADIOSSAVE_H_

#include <Action_I.h>
#include "UtilsADIOS.h"
#include "SCIOUtilsLogger.h"
#include "mpi.h"

namespace cci {
namespace rt {
namespace adios {


class ADIOSSave: public cci::rt::Action_I {
public:
	ADIOSSave(MPI_Comm const *_parent_comm, int const _gid,
			std::string &outDir, std::string &iocode, int total, int _buffer_max,
			int tile_max, int imagename_max, int filename_max,
			ADIOSManager *_iomanager, cciutils::SCIOLogSession *_logsession = NULL);
	virtual ~ADIOSSave();
	virtual int run();
	virtual const char* getClassName() { return "ADIOSSave"; };


protected:
	virtual int process(bool catchup);
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output) { return READY; };

	ADIOSManager *iomanager;
	ADIOSWriter *writer;
	int local_iter;
	int global_iter;

	MPI_Win iter_win;
	int buffer_max;

	int local_total;

};

}
} /* namespace rt */
} /* namespace cci */
#endif /* ADIOSSAVE_H_ */
