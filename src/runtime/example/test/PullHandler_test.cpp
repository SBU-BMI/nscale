/*
 * PullHandler_test.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: tcpan
 */




#include "PullCommHandler.h"
#include "Assign.h"
#include "Save.h"
#include "mpi.h"
#include <vector>
#include "Debug.h"
#include "RandomScheduler.h"

int main (int argc, char **argv){
	int threading_provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);
//	MPI_Init(&argc, &argv);

	MPI_Comm comm_world;
//	comm_world = MPI_COMM_NULL;
	comm_world = MPI_COMM_WORLD;

	int size, rank;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);

	if (size < 2) {
		cci::rt::Debug::print("ERROR:  this test program needs to be run with at least 2 processes.\n");
		MPI_Finalize();
		return -1;
	}

	std::vector<cci::rt::Communicator_I *> handlers;


	cci::rt::Scheduler_I *sch;
	if (rank  == 0) sch = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
	else sch = new cci::rt::RandomScheduler(false, true);

	cci::rt::CommHandler_I *handler = new cci::rt::PullCommHandler(&comm_world, 0, sch);
	handlers.push_back(handler);
	if (handler->isListener()) {
		cci::rt::Assign *assign = new cci::rt::Assign(&comm_world, -1);
		handlers.push_back(assign);
		assign->reference(&handlers);
		handler->setAction(assign);   // handler sets the refernce
	} else {
		cci::rt::Save *save = new cci::rt::Save(&comm_world, -1);
		handlers.push_back(save);
		save->reference(&handlers);
		handler->setAction(save);  // handler sets the reference;
	}

	int j = 0;

	int result;
	while (!handlers.empty() ) {
		++j;
		for (std::vector<cci::rt::Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			result = (*iter)->run();
			if (result == cci::rt::Communicator_I::DONE || result == cci::rt::Communicator_I::ERROR) {
				cci::rt::Debug::print("%s no output at iter j %d .  DONE or error state %d\n", (*iter)->getClassName(), j, result);
				if ((*iter)->dereference(&handlers) == 0) {
					delete (*iter);
				}
				iter = handlers.erase(iter);
			} else if (result == cci::rt::Communicator_I::READY ) {
				cci::rt::Debug::print("%s output generated at iteration j %d.  result = %d\n", (*iter)->getClassName(), j, result);
				++iter;
			} else {
				cci::rt::Debug::print("%s no output at iter j %d .  wait state %d\n", (*iter)->getClassName(), j, result);
				++iter;
			}
		}

	}


	MPI_Finalize();

}



