/*
 * SegmentNuclei.cpp
 *
 *  Created on: Jun 12, 2012
 *      Author: tcpan
 */

#include "Process.h"
#include <iostream>
#include "Logger.h"
#include "Debug.h"

#include "SegConfigurator.h"

#include <signal.h>
#include <unistd.h>

#include <string.h>
#include <vector>
#include <cstdlib>

cci::rt::ProcessConfigurator_I *conf = NULL;

void writeLog() {
	if (conf->getLogger() == NULL) return;

	long long t3, t4;

	t3 = cci::common::event::timestampInUS();

	int rank = 0;

#if defined (WITH_MPI)
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &rank);
	conf->getLogger()->writeCollectively(rank, 0, comm);
#else
	conf->getLogger()->write();
#endif

	t4= cci::common::event::timestampInUS();
	if (rank == 0) cci::common::Debug::print("finished writing log in %lu us.\n", long(t4-t3));

}

void exit_handler() {

	long long t1 = cci::common::event::timestampInUS();
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (conf != NULL) delete conf;

	long long t2 = cci::common::event::timestampInUS();
	if (rank ==0) cci::common::Debug::print("cleaned up conf in %lu us.\n", long(t2-t1));

	t1 = cci::common::event::timestampInUS();
	MPI_Finalize();
	t2 = cci::common::event::timestampInUS();
	if (rank ==0) cci::common::Debug::print("finalized MPI in %lu us.\n", long(t2-t1));

	time_t now = time(0);
	// Convert now to tm struct for local timezone
	tm* localtm = localtime(&now);
	if (rank == 0) {
		printf("The END local date and time is: %s\n", asctime(localtm));
		fflush(stdout);
	}


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

	atexit(exit_handler);

	// real work,
	long long t3, t4;
	t3= cci::common::event::timestampInUS();
	int threading_provided;
	int err  = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);
	MPI_Comm comm = MPI_COMM_WORLD;

	int rank=-1;
	MPI_Comm_rank(comm, &rank);

	time_t now = time(0);
	// Convert now to tm struct for local timezone
	tm* localtm = localtime(&now);
	if (rank == 0) cci::common::Debug::print("The START local date and time is: %s\n", asctime(localtm));

	if (rank == 0) cci::common::Debug::print("initialized MPI\n");
	// IMPORTANT: need to initialize random number generator right now. - for experiment, important to have the same set all the time...
	srand(rank*113 + 1);
	//srand(cci::common::event::timestampInUS());


	long long t1, t2;

	t1 = cci::common::event::timestampInUS();
	conf = new cci::rt::adios::SegConfigurator(argc, argv);
	cci::common::LogSession *logsession = NULL;
	if (conf->getLogger() != NULL) logsession = conf->getLogger()->getSession("setup");

	cci::rt::Process *p = new cci::rt::Process(comm, argc, argv, conf);
	p->setup();
	t2 = cci::common::event::timestampInUS();
	if (logsession != NULL) logsession->log(cci::common::event(0, std::string("proc setup"), t1, t2, std::string(), ::cci::common::event::NETWORK_WAIT));

	p->run();

	t1 = cci::common::event::timestampInUS();
	p->teardown();
	t2 = cci::common::event::timestampInUS();
	if (logsession != NULL) logsession->log(cci::common::event(0, std::string("proc teardown"), t1, t2, std::string(), ::cci::common::event::NETWORK_WAIT));

	if (p != NULL) delete p;
	MPI_Barrier(comm);

	t4= cci::common::event::timestampInUS();
	if (rank ==0) cci::common::Debug::print("finished processing in %lu us.\n", long(t4-t3));

	writeLog();


	exit(0);

	return 0;

}
