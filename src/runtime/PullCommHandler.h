/*
 * PullCommunicator.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PULLCommunicator_H_
#define PULLCommunicator_H_

#include "RootedCommunicator_I.h"

namespace cci {
namespace rt {

class PullCommunicator: public cci::rt::RootedCommunicator_I {
public:
	PullCommunicator(MPI_Comm const *_parent_comm, int const _gid, std::vector<int> _roots);
	virtual ~PullCommunicator();

	virtual int exchange(int &size, char* &data);

};

} /* namespace rt */
} /* namespace cci */
#endif /* PULLCommunicator_H_ */
