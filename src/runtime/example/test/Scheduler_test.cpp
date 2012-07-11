/*
 * Scheduler_test.cpp
 *
 *  Created on: Jul 11, 2012
 *      Author: tcpan
 */


#include "RoundRobinScheduler.h"
#include "RandomScheduler.h"
#include "mpi.h"
#include <vector>

#define IO_GROUP 0
#define COMPUTE_GROUP 1
#define UNUSED_GROUP 2
#define COMPUTE_TO_IO_GROUP 3

int main (int argc, char **argv){
	int threading_provided = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &threading_provided);

	MPI_Comm comm_world;
	comm_world = MPI_COMM_NULL;
	comm_world = MPI_COMM_WORLD;

	int size=0, rank=0;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);


	int compute_io_g = (rank % 4 == 0 ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
	cci::rt::Scheduler_I *sch = NULL;
	if (compute_io_g == COMPUTE_GROUP && rank == 1) sch = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
	else if (compute_io_g == IO_GROUP && rank == 0) sch = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
	else sch = new cci::rt::RandomScheduler(false, true);

	sch->configure(comm_world);

	for (int i= 0; i < 10; ++i) {
		printf("%d was randomly assigned to %d\n", rank, sch->getRootFromLeaf(rank));
	}

	// then the compute to IO communication group
	int compute_to_io_g = (compute_io_g == COMPUTE_GROUP && rank == 1 ? UNUSED_GROUP : COMPUTE_TO_IO_GROUP);

	cci::rt::Scheduler_I *sch2 = NULL;
	if (compute_io_g == IO_GROUP) sch2 = new cci::rt::RoundRobinScheduler(true, false);  // root at rank = 0
	else sch2 = new cci::rt::RoundRobinScheduler(false, true);

	sch2->configure(comm_world);

	for (int i= 0; i < 10; ++i) {
		printf("%d was round robin assigned to %d\n", rank, sch2->getRootFromLeaf(rank));
	}


	MPI_Finalize();

}
