/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include <iostream>
#include <vector>
#include <string.h>
#include "mpi.h"
#include <cstdlib>
#include <string>
#include "pthread.h"

int main (int argc, char **argv) {

	// init MPI
	int ierr = MPI_Init(&argc, &argv);

	std::string hostname;
    char * temp = (char*)malloc(256);
    gethostname(temp, 255);
    hostname.assign(temp);
    free(temp);

	MPI_Comm comm_world = MPI_COMM_WORLD;
	int size, rank;
    MPI_Comm_size(comm_world, &size);
    MPI_Comm_rank(comm_world, &rank);

	printf("comm-world:  %s: %d of %d\n", hostname.c_str(), rank, size);

    // init sub communicator
	// create new group from old group
	int worker_size = size - 1;
	int managerid = 0;
	int *workers = (int*) malloc(worker_size * sizeof(int));
	for (int i = 0, id = 0; i < worker_size; ++i, ++id) {
		if (id == managerid) ++id;  // skip the manager id
		workers[i] = id;
	}
	// get old group
	MPI_Group world_group;
	MPI_Comm_group ( comm_world, &world_group );

	MPI_Group worker_group;
	MPI_Group_incl ( world_group, worker_size, workers, &worker_group );
	free(workers);

	MPI_Comm comm_worker;
	MPI_Comm_create(comm_world, worker_group, &comm_worker);
	
	int worker_rank;
	if (rank != managerid) {
		MPI_Comm_size(comm_worker, &worker_size);
	    MPI_Comm_rank(comm_worker, &worker_rank);
	} else {
		worker_rank = -1;
	}

	MPI_Barrier(comm_world);
	printf("comm-worker: world %d: worker %d of %d\n", rank, worker_rank, worker_size);


	// now do some tests
	// first try to communicate from manager to workers
	int t = 0;
	if (rank == managerid) t = 2;
	printf("rank %d value %d before Bcast\n", rank, t);

	MPI_Bcast(&t, 1, MPI_INT, managerid, comm_world);
	printf("rank %d value %d after Bcast\n", rank, t);

	// now do a test between workers
	if (worker_rank >= 0) {
		int s = 0;
		if (worker_rank == 0) s = 2;
		printf("rank %d worker rank %d value %d before Bcast\n", rank, worker_rank, s);

		MPI_Bcast(&s, 1, MPI_INT, 0, comm_worker);
		printf("rank %d worker rank %d value %d after Bcast\n", rank, worker_rank, s);
	}

	

	MPI_Finalize();
}