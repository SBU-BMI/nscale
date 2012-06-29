/*
 * RoundRobinScheduler.h
 *
 *  Created on: Jun 29, 2012
 *      Author: tcpan
 */

#ifndef ROUNDROBINSCHEDULER_H_
#define ROUNDROBINSCHEDULER_H_

#include <Scheduler_I.h>

namespace cci {
namespace rt {

class RoundRobinScheduler: public cci::rt::Scheduler_I {
public:
	RoundRobinScheduler(std::vector<int> &_roots, std::vector<int> &_leaves);
	RoundRobinScheduler(bool _root, bool _leaf);
	virtual ~RoundRobinScheduler() {};

	virtual int getRootFromLeave(int leafId);
	virtual int getLeafFromRoot(int rootId);

private:
	int rootIdx;
	int leafIdx;
};

} /* namespace rt */
} /* namespace cci */
#endif /* ROUNDROBINSCHEDULER_H_ */
