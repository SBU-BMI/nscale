/*
 * Debug2.cpp
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#include "Debug.h"

namespace cci {
namespace common {


char Debug::msg[4096];

#if defined (WITH_MPI)
bool Debug::checked = false;
int Debug::rank = MPI_UNDEFINED;
#endif

} /* namespace common */
} /* namespace cci */
