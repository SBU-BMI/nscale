/*
 * PullCommHandler.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PULLCOMMHANDLER_H_
#define PULLCOMMHANDLER_H_

#include "RootedCommHandler_I.h"

namespace cci {
namespace rt {

class PullCommHandler: public cci::rt::RootedCommHandler_I {
public:
	PullCommHandler(MPI_Comm const *_parent_comm, int const _gid, std::vector<int> _roots);
	virtual ~PullCommHandler();

	virtual char* getClassName() { return "PullCommHandler"; };


	virtual int run();
};

} /* namespace rt */
} /* namespace cci */
#endif /* PULLCOMMHANDLER_H_ */
