/*
 * Process_test.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "Process.h"

#include "SegConfigurator.h"

int main (int argc, char **argv){

	std::string iocode("MPI");
	cci::rt::ProcessConfigurator_I *conf = new cci::rt::adios::SegConfigurator(iocode);

	cci::rt::Process p(argc, argv, conf);

	p.setup();
	p.run();
	p.teardown();


	return 0;


}
