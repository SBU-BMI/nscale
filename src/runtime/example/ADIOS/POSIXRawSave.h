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
			std::string &outDir, std::string &iocode, int total, int _buffer_max,
			int tile_max, int imagename_max, int filename_max,
			cciutils::SCIOLogSession *_logsession = NULL);
	virtual ~POSIXRawSave();
	virtual int run();
	virtual const char* getClassName() { return "POSIXRawSave"; };


	// differs from base imple:  base imple if input buffer is not empty, return READY.
	// here we return the input status.
	// TODO: move to separate getInput and addInput statuses.
	virtual int getInputStatus() {
		if (this->input_status == ERROR || this->output_status == ERROR) {
			this->input_status = ERROR;
			this->output_status = ERROR;
			return this->input_status;
		}
		if (input_status == READY && inputSizes.empty()) return WAIT;
		if (input_status == DONE) output_status = DONE;
		return input_status;
	}

	// differs from base imple:  base imple gets data if stat is ready.
	// here we also get data if stat is done and there is content.
	// TODO: move to separate getInput and addInput statuses.
	virtual int getInput(int &size , void * &data) {
		int stat = getInputStatus();
		if (stat == READY ||
				(stat== DONE && !inputSizes.empty())) {
			size = inputSizes.front();
			data = inputData.front();
			inputSizes.pop();
			inputData.pop();
			stat = READY;
		} else {
			size = 0;
			data = NULL;
		}
		return stat;
	};
protected:
	virtual int process();
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output) { return READY; };

	int local_iter;

	int buffer_max;

	int local_total;

	long c;
	std::string outdir;
};

}
} /* namespace rt */
} /* namespace cci */
#endif /* POSIXRAWSAVE_H_ */
