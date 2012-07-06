/*
 * Assign_test.cpp
 *
 *  Created on: Jun 26, 2012
 *      Author: tcpan
 */


#include "Segment.h"
#include "mpi.h"
#include <vector>


int main (int argc, char **argv){
//	int threading_provided;
//	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	MPI_Comm comm_world;
	comm_world = MPI_COMM_NULL;
//	comm_world = MPI_COMM_WORLD;

	std::vector<cci::rt::Communicator_I *> handlers;


	cci::rt::Action_I *seg = new cci::rt::Segment(&comm_world, -1);
	handlers.push_back(seg);
	cci::rt::Communicator_I::reference(seg, &handlers);

	int j = 0;
	int count = sizeof(int);
	void *data;
	int *temp;

	int result, oresult;
	while (!handlers.empty() ) {
		printf("not empty\n");

		if (j >= 10 & j < 30) {
			data = malloc(sizeof(int));
			temp = (int*) data;
			temp[0] = j;
			seg->addInput(count, data);
			printf("input added at iteration j %d: %d\n", j, temp[0]);
			//free(data);

			data = malloc(sizeof(int));
			temp = (int*) data;
			temp[0] = j;
			seg->addInput(count, data);
			printf("input added at iteration j %d: %d\n", j, temp[0]);
			//free(data);
		} else if (j == 40)
			seg->markInputDone();

		// j < 10: ready and waiting
		// j >= 10, < 30:  ready and input coming, fast
		// j >= 30, j < 40:  ready, has input, but no new ones
		// j == 40:  mark input as done.
		// j > 40, <= 50:  done, has input still
		// j >50:  done, no more input.

		for (std::vector<cci::rt::Communicator_I *>::iterator iter = handlers.begin();
				iter != handlers.end(); ) {
			printf("iterating\n");
			result = (*iter)->run();
			if (result == cci::rt::Communicator_I::DONE || result == cci::rt::Communicator_I::ERROR) {
				printf("no output at iter j %d .  DONE or error state %d\n", j, result);
				cci::rt::Communicator_I::dereference(*iter, &handlers);

				iter = handlers.erase(iter);
			} else if (result == cci::rt::Communicator_I::READY ) {
				oresult = ((cci::rt::Action_I*)(*iter))->getOutput(count, data);
				printf("output generated at iteration j %d: %d.  output result = %d\n", j, *((int*)data), oresult);
				if (data != NULL) {
					free(data);
					data = NULL;
				}
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


