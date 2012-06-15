/*
 * RootedCommHandler_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef ROOTEDCOMMHANDLER_I_H_
#define ROOTEDCOMMHANDLER_I_H_

#include "CommHandler_I.h"

namespace cci {
namespace rt {

class RootedCommHandler_I : public cci::rt::CommHandler_I {
public:
	virtual ~RootedCommHandler_I(): public cci::rt::CommHandler_I  {
		roots.clear();
	}
	RootedCommHandler_I(MPI_Comm const &_parent_comm, int groupid, std::vector<int> const &_roots) :
		CommHandler_I(_parent_comm, groupid), roots(_roots) {};

	virtual void exchange(int &size, char* &data) = 0;

private:
	std::vector<int> roots;

};

} /* namespace rt */
} /* namespace cci */
#endif /* ROOTEDCOMMHANDLER_I_H_ */
