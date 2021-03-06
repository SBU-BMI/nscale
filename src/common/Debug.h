/*
 * Debug.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#if defined (WITH_MPI)
#include "mpi.h"
#endif


#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <limits>
#include <cassert>

namespace cci {
namespace common {

class Debug {
public:
	static void print(const char* fmt, ...)
	{
		memset(msg, 0, 4096);

	    va_list args;
	    va_start(args,fmt);
	    vsprintf(msg, fmt, args);
	    va_end(args);

#if defined (WITH_MPI)
	    if (!checked) {
			int initialized;
			MPI_Initialized(&initialized);
			if (initialized == 1) {
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			} else {
				rank = std::numeric_limits<int>::min();
			}
			checked = true;
	    }
	   	printf("[DEBUG rank %d] %s", rank, msg);
#else
	   	printf("%s", msg);
#endif

	   	fflush(stdout);
	}

private:
	static char msg[4096];
	static bool checked;
	static int rank;

};

} /* namespace common */
} /* namespace cci */
#endif /* DEBUG_H_ */
