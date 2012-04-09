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
#include "adios.h"

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

	// now do an adios test with global comm.
//	int err = adios_init("adios_xml/test-adios2.xml");
//	if (err != 1) printf("ERROR: adios error code on init: %d \n", err);
//
//	const char * fn = "test-adios2.world.bp";
//	int fnlen = strlen(fn);
//	char * filename = (char*)malloc(fnlen + 1);
//	memset(filename, 0, fnlen + 1);
//	strncpy(filename, fn, fnlen);
//	int64_t adios_handle;
//	err = adios_open(&adios_handle, "source", filename, "w", &comm_world);
//	if (err != 0) printf("ERROR: adios error code on open: %d \n", err);
//	free(filename);
//
//	uint64_t total_size;
//	err = adios_group_size (adios_handle, 4, &total_size);
//	if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
//	printf("adios total size = %lu\n", total_size);
//
//	err = adios_write (adios_handle, "x", &rank);
//	if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
//
//	err = adios_close(adios_handle);
//	if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
//
//	MPI_Barrier(comm_world);
//	err = adios_finalize(rank);


	// now do another test with comm_worker only
	int err = adios_init("adios_xml/test-adios.xml");
	if (err != 1) printf("ERROR: adios error code on init: %d \n", err);

	if (worker_rank >= 0) {

		const char * fn2 = "test-adios2.worker.bp";
		int fnlen = strlen(fn2);
		char *filename = (char*)malloc(fnlen + 1);
		memset(filename, 0, fnlen + 1);
		strncpy(filename, fn2, fnlen);
		// test to see if ADIOS can open multiple groups in same file

		int64_t adios_handle;
		err = adios_open(&adios_handle, "source", filename, "w", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

		printf("adios handle %ld\n", adios_handle);




		// nodes can write different amount of data.
		uint64_t total_size;

		if (worker_rank > 0) {
			err = adios_group_size (adios_handle, 4, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			printf("adios total size = %lu\n", total_size);

			err = adios_write (adios_handle, "x", &worker_rank);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		} else {
			err = adios_group_size (adios_handle, 0, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			printf("adios total size = %lu\n", total_size);
		}

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);


		// append works fine.
		err = adios_open(&adios_handle, "source", filename, "a", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

		printf("adios handle %ld\n", adios_handle);


		// test to see if each node can handle different number of writes: NO

		if (worker_rank == 0) {
			err = adios_group_size (adios_handle, 4, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			printf("adios total size = %lu\n", total_size);

			err = adios_write (adios_handle, "x", &worker_rank);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		} else {
			err = adios_group_size (adios_handle, 0, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			printf("adios total size = %lu\n", total_size);
		}

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);

		// different amount of data in global array
		err = adios_open(&adios_handle, "source2", filename, "a", &comm_worker);
		if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

		printf("adios handle %ld\n", adios_handle);

		int size = worker_rank + 1;
		// get the size info from other workers
		int offset = 0, total=0;
		MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
		offset -= size;
		MPI_Allreduce(&size, &total, 1, MPI_INT, MPI_SUM, comm_worker);
		int *values = (int*)malloc(size * sizeof(int));
		for (int i = 0; i < size; ++i) {
			values[i] = rank;
		}
		printf("size %d, total %d, offset %d\n", size, total, offset);


		err = adios_group_size (adios_handle, 4 * 3 + size * 4, &total_size);
		if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
		printf("adios total size = %lu\n", total_size);

		err = adios_write (adios_handle, "size", &size);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "offset", &offset);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "total", &total);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
		err = adios_write (adios_handle, "values", values);
		if (err != 0) printf("ERROR: adios error code on write: %d \n", err);

		err = adios_close(adios_handle);
		if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
		MPI_Barrier(comm_worker);

		free(values);


		// different amount of data in global array at different timesteps
		for (int it = 0; it < 5; ++it) {
			err = adios_open(&adios_handle, "source3", filename, "a", &comm_worker);
			if (err != 0) printf("ERROR: adios error code on open: %d \n", err);

			printf("time index adios handle %ld\n", adios_handle);

			size = (worker_rank + 1)* (it + 1);
			offset = 0;
			total=0;
			MPI_Scan(&size, &offset, 1, MPI_INT, MPI_SUM, comm_worker);
			offset -= size;
			MPI_Allreduce(&size, &total, 1, MPI_INT, MPI_SUM, comm_worker);
			values = (int*)malloc(size * sizeof(int));
			for (int i = 0; i < size; ++i) {
				values[i] = rank;
			}
			printf("size %d, total %d, offset %d\n", size, total, offset);

			err = adios_group_size (adios_handle, 4 * 3 + size * 4, &total_size);
			if (err != 0) printf("ERROR: adios error code on groupsize: %d \n", err);
			printf("adios total size = %lu\n", total_size);

			err = adios_write (adios_handle, "size", &size);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
			err = adios_write (adios_handle, "offset", &offset);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
			err = adios_write (adios_handle, "total", &total);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);
			err = adios_write (adios_handle, "values", values);
			if (err != 0) printf("ERROR: adios error code on write: %d \n", err);

			err = adios_close(adios_handle);
			if (err != 0) printf("ERROR: adios error code on close: %d \n", err);
			MPI_Barrier(comm_worker);

			free(values);
		}

		free(filename);

	}
	err = adios_finalize(rank);

	MPI_Finalize();
}

