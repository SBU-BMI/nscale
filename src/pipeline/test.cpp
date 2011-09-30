/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include <iostream>
#include <string>
#include <omp.h>
#include <stdio.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif



int main (int argc, char **argv){

#ifdef WITH_MPI
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
	printf("%d of %d\n", rank, size);
#else
    int size = 1;
    int rank = 0;
#endif


if (rank == 0) {

	printf("0 rank\n");
#ifdef WITH_MPI

} else {
	printf("other rank\n");
#endif
}// end if (rank == 0)

#ifdef WITH_MPI
#pragma omp parallel for shared(rank)
    for (int i = 0; i < 20; ++i) {
#else
#pragma omp parallel for shared(rank)
    for (int i = 0; i < 40; ++i) {
#endif

    	int tid = omp_get_thread_num();

printf("%d: %d::%d\n", i, rank, tid);

    }
#ifdef WITH_MPI
    MPI::COMM_WORLD.Barrier();
#endif
#ifdef WITH_MPI
#pragma omp parallel for shared(rank)
    for (int i = 0; i < 20; ++i) {
#else
#pragma omp parallel for shared(rank)
    for (int i = 0; i < 40; ++i) {
#endif

    	int tid = omp_get_thread_num();

printf("again %d: %d::%d\n", i, rank, tid);

    }

#ifdef WITH_MPI
    MPI::COMM_WORLD.Barrier();
#endif

    if (rank == 0) {

    	printf("back again to rank 0\n");

    }

#ifdef WITH_MPI
    MPI::Finalize();
#endif


//	waitKey();

	return 0;
}


