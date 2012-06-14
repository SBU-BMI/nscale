/*
 * UniformRandomSchedule.cpp
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#include "UniformRandomSchedule.h"
#include <cstdlib>

namespace cci {

namespace runtime {


void UniformRandomSchedule::initialize(const unsigned int seed) {
	srand(seed);
}

int UniformRandomSchedule::assign(std::vector<Process *> &processes) {
	return rand() % process.size();
}


}

}
