/*
 * ADIOSSave_Reduce.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef ADIOSSAVE_REDUCE_H_
#define ADIOSSAVE_REDUCE_H_

#include <Action_I.h>
#include "UtilsADIOS.h"
#include "SCIOUtilsLogger.h"
#include "mpi.h"

extern int test_input_status;


namespace cci {
namespace rt {
namespace adios {


class ADIOSSave_Reduce: public cci::rt::Action_I {
public:
	ADIOSSave_Reduce(MPI_Comm const *_parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			const int tile_max, const int imagename_max, const int filename_max,
			ADIOSManager *_iomanager, cciutils::SCIOLogSession *_logsession = NULL);
	virtual ~ADIOSSave_Reduce();
	virtual int run();
	virtual const char* getClassName() { return "ADIOSSave_Reduce"; };

protected:
	virtual int process();
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output) { return READY; };

	ADIOSManager *iomanager;
	ADIOSWriter *writer;

	int local_iter;
	int local_total;

};

}
} /* namespace rt */
} /* namespace cci */
#endif /* ADIOSSAVE_REDUCE_H_ */
