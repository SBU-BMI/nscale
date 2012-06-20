/*
 * RootedCommunicator_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef ROOTEDCommunicator_I_H_
#define ROOTEDCommunicator_I_H_

#include "Communicator_I.h"
#include <vector>
#include <algorithm>

namespace cci {
namespace rt {

class RootedCommunicator_I : public cci::rt::Communicator_I {
public:
	virtual ~RootedCommunicator_I()  {
		//printf("RootedCommunicator destructor called\n");

		roots.clear();
	};
	RootedCommunicator_I(MPI_Comm const * _parent_comm, int const _gid, std::vector<int> _roots) :
		Communicator_I(_parent_comm, _gid), roots(_roots) {
		std::sort(roots.begin(), roots.end());
		isRoot = std::binary_search(roots.begin(), roots.end(), rank);
		printf("rank: %d is root? %s\n ", pcomm_rank, (isRoot ? "true" : "false"));
	};

	virtual int exchange(int &size, char* &data) = 0;
	virtual bool isListener() { return isRoot; };

protected:
	std::vector<int> roots;
	bool isRoot;

};

} /* namespace rt */
} /* namespace cci */
#endif /* ROOTEDCommunicator_I_H_ */
