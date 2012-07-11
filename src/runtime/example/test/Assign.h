/*
 * Assign.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef ASSIGN_H_
#define ASSIGN_H_

#include <Action_I.h>

namespace cci {
namespace rt {

class Assign: public cci::rt::Action_I {
public:
	Assign(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogSession *_logger = NULL);
	virtual ~Assign();
	virtual int run();
	virtual const char* getClassName() { return "Assign"; };


protected:

	virtual int compute(int const &input_size , void * const &input,
				int &output_size, void * &output);


};

} /* namespace rt */
} /* namespace cci */
#endif /* ASSIGN_H_ */
