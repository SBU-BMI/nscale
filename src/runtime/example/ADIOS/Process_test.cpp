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

#include <signal.h>

cciutils::SCIOLogger *logger = NULL;
cci::rt::adios::SegmentCmdParser *parser = NULL;


void writeLog() {
	if (parser == NULL) return;
	if (logger == NULL) return;

	long long t3, t4;

	t3 = cciutils::event::timestampInUS();


	std::string logfile = parser->getParam(cci::rt::adios::SegmentCmdParser::PARAM_OUTPUTDIR);

#if defined (WITH_MPI)
	MPI_Comm comm = MPI_COMM_WORLD;
	int rank;
	MPI_Comm_rank(comm, &rank);
	logger->writeCollectively(logfile, rank, 0, comm);
#else
	logger->write(logfile);
#endif

	t4= cciutils::event::timestampInUS();
	cci::rt::Debug::print("finished writing log in %lu us.\n", long(t4-t3));

}

void ctrlc_handler(int value) {
	cci::rt::Debug::print("%d terminating\n", value);

	writeLog();

}


int main (int argc, char **argv){

// DOES NOT WORK!
//	// setup signal trap to catch Ctrl-C
//	struct sigaction new_action, old_action;
//	new_action.sa_handler = ctrlc_handler;
//	sigemptyset(&new_action.sa_mask);
//	new_action.sa_flags = 0;
////    if( sigaction (SIGINT, NULL, &old_action) == -1)
////            perror("Failed to retrieve old handle");
////    if (old_action.sa_handler != SIG_IGN)
////            if( sigaction (SIGINT, &new_action, NULL) == -1)
////                    perror("Failed to set new Handle");
//    if( sigaction (SIGTERM, NULL, &old_action) == -1)
//            perror("Failed to retrieve old handle");
//    if (old_action.sa_handler != SIG_IGN)
//            if( sigaction (SIGTERM, &new_action, NULL) == -1)
//                    perror("Failed to set new Handle");


	// real work,
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

	logger = new cciutils::SCIOLogger(rank, hostname, 0);\
	cciutils::SCIOLogSession *logsession = logger->getSession("setup");

	long long t1, t2;
	t1 = cciutils::event::timestampInUS();
	parser = new cci::rt::adios::SegmentCmdParser(comm);
	if (rank == 0) {
		// create the directory
		FileUtils futils;
		futils.mkdirs(parser->getParam(cci::rt::adios::SegmentCmdParser::PARAM_OUTPUTDIR));
	}
	t2 = cciutils::event::timestampInUS();
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("parse cmd"), t1, t2, std::string(), ::cciutils::event::OTHER));

	if (!parser->parse(argc, argv)) {
		if (logger) delete logger;
		delete parser;

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

	if (p != NULL) delete p;
	if (conf != NULL) delete conf;

	t4= cciutils::event::timestampInUS();
	cci::rt::Debug::print("finished processing in %lu us.\n", long(t4-t3));

	// cleaning up.
	writeLog();

	if (parser != NULL) delete parser;
	if (logger != NULL) delete logger;

	MPI_Finalize();


	return 0;


}
