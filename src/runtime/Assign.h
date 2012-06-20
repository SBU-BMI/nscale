/*
 * Assign.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef ASSIGN_H_
#define ASSIGN_H_

#include <Worker_I.h>

namespace cci {
namespace rt {

class Assign: public cci::rt::Worker_I {
public:
	Assign(MPI_Comm const * _parent_comm, int const _gid);
	virtual ~Assign();

	virtual int compute(int const &input_size , char* const &input,
				int &output_size, char* &output);

};

} /* namespace rt */
} /* namespace cci */
#endif /* ASSIGN_H_ */
