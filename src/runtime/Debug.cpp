/*
 * Debug2.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "Debug.h"

namespace cci {
namespace rt {


char Debug::msg[4096];
bool Debug::checked = false;
int Debug::rank = -1;

} /* namespace rt */
} /* namespace cci */
