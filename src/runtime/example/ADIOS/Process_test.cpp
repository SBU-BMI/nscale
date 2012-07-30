/*
 * Process_test.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "Process.h"
#include <iostream>
#include "SCIOUtilsLogger.h"
#include "Debug.h"
#include "FileUtils.h"

#include "SegConfigurator.h"
#include "SegmentCmdParser.h"

int main (int argc, char **argv){

	long long t3, t4;
	t3= cciutils::event::timestampInUS();
	int threading_provided;
	int err  = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);
	MPI_Comm comm = MPI_COMM_WORLD;

	char hostname[256];
	gethostname(hostname, 255);  // from <iostream>

	int rank;
	MPI_Comm_rank(comm, &rank);

	// IMPORTANT: need to initialize random number generator right now.
	srand(rank);

	cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(rank, hostname, 0);\
	cciutils::SCIOLogSession *logsession = logger->getSession("setup");

	long long t1, t2;
	t1 = cciutils::event::timestampInUS();
	cci::rt::adios::SegmentCmdParser *parser = new cci::rt::adios::SegmentCmdParser(comm);
	if (rank == 0) {
		// create the directory
		FileUtils futils;
		futils.mkdirs(parser->getParam(cci::rt::adios::SegmentCmdParser::PARAM_OUTPUTDIR));
	}
	t2 = cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("parse cmd"), t1, t2, std::string(), ::cciutils::event::OTHER));

	if (!parser->parse(argc, argv)) {
		MPI_Finalize();
		return 0;
	}

	cci::rt::ProcessConfigurator_I *conf = new cci::rt::adios::SegConfigurator(parser->getParams(), logger);

	cci::rt::Process *p = new cci::rt::Process(comm, argc, argv, conf);

	t1 = cciutils::event::timestampInUS();
	p->setup();
	t2 = cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("proc setup"), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));

	p->run();

	t1 = cciutils::event::timestampInUS();
	p->teardown();
	t2 = cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("proc teardown"), t1, t2, std::string(), ::cciutils::event::NETWORK_WAIT));

	delete p;
	delete conf;

	t4= cciutils::event::timestampInUS();
	cci::rt::Debug::print("finished processing in %lu us.\n", long(t4-t3));
	t3 = cciutils::event::timestampInUS();


	std::string logfile = parser->getParam(cci::rt::adios::SegmentCmdParser::PARAM_OUTPUTDIR);

#if defined (WITH_MPI)
	logger->writeCollectively(logfile, rank, 0, comm);
#else
	logger->write(logfile);
#endif

	delete parser;
	delete logger;
	t4= cciutils::event::timestampInUS();
	cci::rt::Debug::print("finished writing log %s in %lu us.\n", logfile.c_str(), long(t4-t3));

	MPI_Finalize();


	return 0;


}
