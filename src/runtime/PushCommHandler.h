/*
 * PushCommHandler.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PUSHCOMMHANDLER_H_
#define PUSHCOMMHANDLER_H_

#include "RootedCommHandler_I.h"

namespace cci {
namespace rt {

class PushCommHandler: public cci::rt::RootedCommHandler_I {
public:
	PushCommHandler(MPI_Comm const &_parent_comm, int groupid, std::vector<int> const &_roots);
	virtual ~PushCommHandler();

	virtual void exchange(int &size, char* &data);
};

} /* namespace rt */
} /* namespace cci */
#endif /* PUSHCOMMHANDLER_H_ */
