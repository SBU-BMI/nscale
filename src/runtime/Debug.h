/*
 * Debug.h
 *
 *  Created on: Jun 19, 2012
 *      Author: tcpan
 */

#ifndef DEBUG_H_
#define DEBUG_H_

#include "mpi.h"
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

namespace cci {
namespace rt {

class Debug {
public:
	static void print(char* fmt, ...)
	{
		memset(msg, 0, 4096);

	    va_list args;
	    va_start(args,fmt);
	    vsprintf(msg, fmt,args);
	    va_end(args);

		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		printf("[DEBUG rank %d] %s", rank, msg);;
	}

private:
	static char msg[4096];

};


} /* namespace rt */
} /* namespace cci */
#endif /* DEBUG_H_ */
