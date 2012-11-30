/*
 * POSIXRawSave.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef POSIXRAWSAVE_H_
#define POSIXRAWSAVE_H_

#include <Action_I.h>
#include "SCIOUtilsLogger.h"
#include "mpi.h"
#include <string>

extern int test_input_status;


namespace cci {
namespace rt {
namespace adios {


class POSIXRawSave: public cci::rt::Action_I {
public:
	POSIXRawSave(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			boost::program_options::variables_map &_vm,
			cciutils::SCIOLogSession *_logsession = NULL);
	virtual ~POSIXRawSave();
	virtual int run();
	virtual const char* getClassName() { return "POSIXRawSave"; };

protected:
	virtual int process();
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output) { return READY; };

	int local_iter;
	int local_total;

	std::string outdir;

};

}
} /* namespace rt */
} /* namespace cci */
#endif /* POSIXRAWSAVE_H_ */
