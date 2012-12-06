/*
 * Save.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef SAVE_H_
#define SAVE_H_

#include <Action_I.h>

namespace cci {
namespace rt {

class Save: public cci::rt::Action_I {
public:
	Save(MPI_Comm const * _parent_comm, int const _gid,
			DataBuffer *_input, DataBuffer *_output,
			cci::common::LogSession *_logsession = NULL);
	virtual ~Save();
	virtual int run();
	virtual const char* getClassName() { return "Save"; };


protected:

	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);

};

} /* namespace rt */
} /* namespace cci */
#endif /* SAVE_H_ */
