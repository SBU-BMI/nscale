/*
 * Save.h
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#ifndef SAVE_H_
#define SAVE_H_

#include <Worker_I.h>

namespace cci {
namespace rt {

class Save: public cci::rt::Worker_I {
public:
	Save(MPI_Comm const * _parent_comm, int const _gid);
	virtual ~Save();

	virtual int compute(int const &input_size , char* const &input,
				int &output_size, char* &output);
};

} /* namespace rt */
} /* namespace cci */
#endif /* SAVE_H_ */
