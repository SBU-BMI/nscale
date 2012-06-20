/*
 * PushCommunicator.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef PUSHCommunicator_H_
#define PUSHCommunicator_H_

#include "RootedCommunicator_I.h"

namespace cci {
namespace rt {

class PushCommunicator: public cci::rt::RootedCommunicator_I {
public:
	PushCommunicator(MPI_Comm const * _parent_comm, int const _gid, std::vector<int> _roots);
	virtual ~PushCommunicator();

	virtual int exchange(int &size, char* &data);

};

} /* namespace rt */
} /* namespace cci */
#endif /* PUSHCommunicator_H_ */
