/*
 * Assign_test.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: tcpan
 */


#include "Assign.h"
#include "mpi.h"
#include <vector>


int main (int argc, char **argv){
//	int threading_provided;
//	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	MPI_Comm comm_world;
	comm_world = MPI_COMM_NULL;
//	comm_world = MPI_COMM_WORLD;

	std::vector<cci::rt::Communicator_I *> handlers;


	cci::rt::Action_I *assign = new cci::rt::Assign(&comm_world, -1);
	handlers.push_back(assign);
	assign->reference(&handlers);

	int j = 0;
	int count = sizeof(int);
	void *data = NULL;
	int *temp;

	int result, oresult;
	while (!handlers.empty() ) {
		for (std::vector<cci::rt::Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			printf("iterating\n");
			result = (*iter)->run();
			if (result == cci::rt::Communicator_I::DONE || result == cci::rt::Communicator_I::ERROR) {
				printf("no output at iter j %d .  DONE or error state %d\n", j, result);
				if ((*iter)->dereference(&handlers) == 0) {
					delete (*iter);
				}
				iter = handlers.erase(iter);
			} else if (result == cci::rt::Communicator_I::READY ) {
				oresult = ((cci::rt::Action_I*)(*iter))->getOutput(count, data);
				printf("output generated at iteration j %d: %d.  output result = %d\n", j, *((int*)data), oresult);
				free(data);
				data = NULL;
				++iter;
			} else {
				printf("no output at iter j %d .  wait state %d\n", j, result);
				++iter;
			}
		}
		++j;
	}


//	MPI_Finalize();

}


