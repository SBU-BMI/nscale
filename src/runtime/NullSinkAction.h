/*
 * NullSinkAction.h
 *
 *  Created on: Aug 2, 2012
 *      Author: tcpan
 */

#ifndef NULLSINKACTION_H_
#define NULLSINKACTION_H_

#include <Action_I.h>

namespace cci {
namespace rt {

class NullSinkAction: public cci::rt::Action_I {
public:
	NullSinkAction(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			cci::common::LogSession *_logsession = NULL);
	virtual ~NullSinkAction();

	virtual int run();

	virtual const char* getClassName() { return "NULLSinkAction"; };

protected:
	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output) { return Communicator_I::READY; };

};

} /* namespace rt */
} /* namespace cci */
#endif /* NULLSINKACTION_H_ */
