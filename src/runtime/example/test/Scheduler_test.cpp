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
#include <iterator>
#include <iostream>
#include <math.h>

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

	srand(rank * 10);

	int compute_io_g = (rank % 4 == 0 ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
	cci::rt::Scheduler_I *sch = NULL;
	if (compute_io_g == COMPUTE_GROUP && rank == 1) sch = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
	else if (compute_io_g == IO_GROUP && rank == 0) sch = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
	else sch = new cci::rt::RandomScheduler(false, true);

	MPI_Comm comm;
	MPI_Comm_split(comm_world, compute_io_g, rank, &comm);

	sch->configure(comm);

	std::ostream_iterator<int> osi(std::cout, ", ");
	std::vector<int> roots;
	std::vector<int> leaves;

	roots = sch->getRoots();
	std::cout << rank << " (" << (compute_io_g == COMPUTE_GROUP ? "cp" : "io") << ") roots: ";
	std::copy(roots.begin(), roots.end(), osi);
	std::cout << std::endl;
	leaves = sch->getLeaves();
	std::cout << rank << " (" << (compute_io_g == COMPUTE_GROUP ? "cp" : "io") << ") leaves: ";
	std::copy(leaves.begin(), leaves.end(), osi);
	std::cout << std::endl;


	MPI_Barrier(comm_world);

	if (rank == 2) {
	for (int i= 0; i < 10; ++i) {
			printf("%d (%s) was randomly assigned to root %d\n", rank, (compute_io_g == COMPUTE_GROUP ? "cp" : "io"), sch->getRootFromLeaf(rank));
	}
	} else if (rank == 0 || rank == 1) {

		for (int i= 0; i < 10; ++i) {
			printf("%d (%s) was randomly assigned to leaf %d\n", rank, (compute_io_g == COMPUTE_GROUP ? "cp" : "io"), sch->getLeafFromRoot(rank));
		}
	}
	MPI_Barrier(comm_world);

	// then the compute to IO communication group
	int compute_to_io_g = (compute_io_g == COMPUTE_GROUP && rank == 1 ? UNUSED_GROUP : COMPUTE_TO_IO_GROUP);

	cci::rt::Scheduler_I *sch2 = NULL;
	if (compute_io_g == IO_GROUP) sch2 = new cci::rt::RoundRobinScheduler(true, false);  // root at rank = 0
	else sch2 = new cci::rt::RoundRobinScheduler(false, true);

	cci::rt::Scheduler_I *sch3 = NULL;
	if (compute_io_g == IO_GROUP) sch3 = new cci::rt::RandomScheduler(true, false);  // root at rank = 0
	else sch3 = new cci::rt::RandomScheduler(false, true);

	MPI_Comm comm2;
	MPI_Comm_split(comm_world, compute_to_io_g, rank, &comm2);

	sch2->configure(comm2);
	sch3->configure(comm2);

	if (rank == 2) {
	for (int i= 0; i < 10; ++i) {
			printf("%d (%s) was round robin assigned to %d\n", rank, (compute_to_io_g == COMPUTE_TO_IO_GROUP ? "c2io" : "unknown"), sch2->getRootFromLeaf(rank));
			printf("%d (%s) was random assigned to %d\n", rank, (compute_to_io_g == COMPUTE_TO_IO_GROUP ? "c2io" : "unknown"), sch3->getRootFromLeaf(rank));
	}
	}


	// testing randomness
	if (rank == 1) {
		int sum = 0;
		int i;
		for (i= 0; i <= 1000000; ++i) {
			if (i == 10 || i == 100 || i== 1000 || i == 10000 || i == 100000 || i == 1000000) {
				printf("%d (%s) random leaf assignment average value (MLE of mean) at iter %d is %f\n", rank, (compute_io_g == COMPUTE_GROUP ? "cp" : "io"), i, (double)sum / (double)i);
			}
			sum += sch->getLeafFromRoot(rank);
		}
		//printf("%d (%s) random leaf assignment average value (MLE of mean) at iter %d is %f\n", rank, (compute_io_g == COMPUTE_GROUP ? "cp" : "io"), i, (double)sum / (double)i);

	}


	MPI_Barrier(comm_world);


	int count = 0;
	if (rank == 0) {
		// print the list, add 1, get the list, remove 1, get the list
		roots = sch->getRoots();
		std::cout << "roots: ";
		std::copy(roots.begin(), roots.end(), osi);
		std::cout << std::endl;

		count = sch->addRoot(2);
		roots = sch->getRoots();
		std::cout << "roots: (" << count << ") ";
		std::copy(roots.begin(), roots.end(), osi);
		std::cout << std::endl;

		count = sch->removeRoot(0);
		roots = sch->getRoots();
		std::cout << "roots: (" << count << ") ";
		std::copy(roots.begin(), roots.end(), osi);
		std::cout << std::endl;

		count = sch->removeRoot(0);
		roots = sch->getRoots();
		std::cout << "roots: (" << count << ") ";
		std::copy(roots.begin(), roots.end(), osi);
		std::cout << std::endl;



		leaves = sch->getLeaves();
		std::cout << "leaves: ";
		std::copy(leaves.begin(), leaves.end(), osi);
		std::cout << std::endl;

		count = sch->addLeaf(0);
		leaves = sch->getLeaves();
		std::cout << "leaves: (" << count << ") ";
		std::copy(leaves.begin(), leaves.end(), osi);
		std::cout << std::endl;

		count = sch->removeLeaf(0);
		leaves = sch->getLeaves();
		std::cout << "leaves: (" << count << ") ";
		std::copy(leaves.begin(), leaves.end(), osi);
		std::cout << std::endl;

		count = sch->removeLeaf(0);
		leaves = sch->getLeaves();
		std::cout << "leaves: (" << count << ") ";
		std::copy(leaves.begin(), leaves.end(), osi);
		std::cout << std::endl;


	}

// now test the random number generator with different sizes
	srand(rank);
	long long sum2 = 0;
	int i, j;
	double expected;
	for (j = 8; j < 300000; j *= 2 ) {
		sum2 = 0;
		for (i= 0; i <= 10; ++i) {
			if (i == 10 || i == 100 || i== 1000 || i == 10000 || i == 100000 || i == 1000000) {
				expected = (double)sum2 / (double)i;
				printf("%d random number generator choosing between 0 and %d for iter %d has expected value of %f.\t%s\n", rank,  j, i, expected, (fabs(j/2.0 - expected) < 0.1 * j / 2.0? "==" : (j/2.0 > expected ? "<<" : ">")) );
			}
			sum2 += rand() % j;
		}
	}
	//printf("%d (%s) random leaf assignment average value (MLE of mean) at iter %d is %f\n", rank, (compute_io_g == COMPUTE_GROUP ? "cp" : "io"), i, (double)sum / (double)i);


	MPI_Finalize();

}
