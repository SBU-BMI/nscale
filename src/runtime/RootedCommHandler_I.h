/*
 * RootedCommHandler_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef ROOTEDCOMMHANDLER_I_H_
#define ROOTEDCOMMHANDLER_I_H_

#include "Communicator_I.h"
#include <vector>
#include <algorithm>
#include "Action_I.h"
#include <tr1/unordered_map>

namespace cci {
namespace rt {

class RootedCommHandler_I : public cci::rt::Communicator_I {
public:
	virtual ~RootedCommHandler_I();

	/**
	 * splits a communicator and allows handling of the communication between roots and children
	 * _parent_comm: the communicator to split
	 * _gid:  the group id with which to split the parent comm
	 * _roots:  the list of roots, specified as ranks in parent_comm.
	 */
	RootedCommHandler_I(MPI_Comm const * _parent_comm, int const _gid, std::vector<int> _roots);
	virtual char* getClassName() { return "RootedCommHandler_I"; };

	void setAction(Action_I * _action) {
		action = _action;
		action->reference(this);
	};

	virtual int run() = 0;
	virtual bool isListener() { return isRoot; };
	virtual bool isReady() { return action != NULL && status != ERROR && status != DONE; };

	static const int CONTROL_TAG;
	static const int DATA_TAG;

protected:
	std::vector<int> roots;
	bool isRoot;

	bool msgToManager;
	std::tr1::unordered_map<int, int> msgToWorker;
	std::tr1::unordered_map<int, int> activeWorkers;

	Action_I * action;
	int status;
};

} /* namespace rt */
} /* namespace cci */
#endif /* ROOTEDCOMMHANDLER_I_H_ */
