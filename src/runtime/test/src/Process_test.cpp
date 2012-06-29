/*
 * Process_test.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "../../Process.h"



int main (int argc, char **argv){


	cci::rt::Process p(argc, argv, );

	p.setup();
	p.run();
	p.teardown();
	return 0;


}
