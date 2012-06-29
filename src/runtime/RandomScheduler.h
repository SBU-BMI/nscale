/*
 * RandomScheduler.h
 *
 *  Created on: Jun 29, 2012
 *      Author: tcpan
 */

#ifndef RANDOMSCHEDULER_H_
#define RANDOMSCHEDULER_H_

#include <Scheduler_I.h>

namespace cci {
namespace rt {

class RandomScheduler: public cci::rt::Scheduler_I {
public:
	RandomScheduler(std::vector<int> &_roots, std::vector<int> &_leaves);
	RandomScheduler(bool _root, bool _leaf);
	virtual ~RandomScheduler() {};

	virtual int getRootFromLeave(int leafId);
	virtual int getLeafFromRoot(int rootId);
};

} /* namespace rt */
} /* namespace cci */
#endif /* RANDOMSCHEDULER_H_ */
