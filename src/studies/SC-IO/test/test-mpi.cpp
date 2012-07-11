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
#include "SCIOUtilsLogger.h"

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
	int managerid = 0;

	MPI_Comm comm_worker;
	MPI_Comm_split(comm_world, (rank == managerid ? 1 : 0), rank, &comm_worker);
	
	int worker_rank, worker_size;
	MPI_Comm_size(comm_worker, &worker_size);
	MPI_Comm_rank(comm_worker, &worker_rank);

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

	MPI_Comm_free(&comm_worker);

	int group_size, group_interleave;
	if (argc < 3) {
		group_size = 0;
		group_interleave = 1;
	} else {
		group_size = atoi(argv[1]);
		group_interleave = atoi(argv[2]);
	}
	int worker_group;
	// create new group from old group
	// first come up with the color  manager gets color 0.  everyone else: 1.
	if (group_size == 1) {
		// everyone is in his own group
		worker_group = rank;
	} else if (group_size < 1) {
		// everyone in one group
		if (rank == managerid) worker_group = 0;
		else worker_group = 1;
	} else {
		if (rank == managerid) worker_group = 0;
		else {
			if (group_interleave > 1) {
				// e.g. 0, 12, 24 go to group 1. 1,13,25 group 2,  144, 156, ... got to group 13.  for groupsize = 12 and interleave of 3.
				// block is a group of groups that are interleaved.
				int blockid = rank / (group_size * group_interleave);
				// each block has group_interleave number of groups
				// so the starting worker_group within a block is blockid*interleave
				worker_group = blockid * group_interleave + rank % group_interleave;

			} else {
				// interleave of 1 or less means adjacent proc ids in group.
				// e.g. 0 .. 11 go to group 1.
				worker_group = rank / group_size;
			}
			++worker_group;  // manager has group 0.
		}
	}
	// WORKER_GROUP should have value >= 0.

	MPI_Comm_split(comm_world, worker_group, rank, &comm_worker);


	MPI_Comm_size(comm_worker, &worker_size);
	MPI_Comm_rank(comm_worker, &worker_rank);


	MPI_Barrier(comm_world);
	printf("comm-worker: world %d: worker group %d, group rank %d\n", rank, worker_group, worker_rank);


	// now do some tests
	// first try to communicate from manager to workers
	t = 0;
	if (rank == managerid) t = 2;
	//printf("rank %d value %d before Bcast\n", rank, t);

	MPI_Bcast(&t, 1, MPI_INT, managerid, comm_world);
	//printf("rank %d value %d after Bcast\n", rank, t);

	// now do a test between workers
	if (worker_rank >= 0) {
		int s = 0;
		if (worker_rank == 0) s = rank;
		//printf("rank %d worker rank %d value %d before Bcast\n", rank, worker_rank, s);

		MPI_Bcast(&s, 1, MPI_INT, 0, comm_worker);
		//printf("rank %d worker rank %d value %d after Bcast\n", rank, worker_rank, s);
	}

	// now test delays
	MPI_Barrier(comm_world);
	long long t1 = cciutils::event::timestampInUS();

	sleep(worker_group);  // WORKER_GROUP should have value >= 0.


	MPI_Barrier(comm_worker);
	long long t2 = cciutils::event::timestampInUS();
	printf("rank %d worker rank %d group %d elapsed %lld\n", rank, worker_rank, worker_group, t2 - t1);

	MPI_Barrier(comm_world);
	if (worker_group == 1) {  // test if all communicators need to participate

		MPI_Barrier(comm_worker);
		printf("only group 1 has barrier\n");
	} else {
		printf("other groups don't have barrier\n");
	}
	
	MPI_Barrier(comm_worker);
	MPI_Comm_free(&comm_worker);

	MPI_Barrier(comm_world);


	// testing MPI with RMA
	printf("START RMA TEST rank %d\n", rank);
	int flag = 0;
	MPI_Win win;
	if (rank == 0) {
		// set up the receive window.
		MPI_Win_create(&flag, sizeof(int), sizeof(int), MPI_INFO_NULL, comm_world, &win);
	} else {

		// set up the send window
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, comm_world, &win);

		// now do selective.
		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
		MPI_Accumulate(&rank, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, win);
		MPI_Win_unlock(0, win);
	}
	MPI_Win_free(&win);
	printf("rank %d flag = %d\n", rank, flag);

	printf("START RMA TEST selective update rank %d\n", rank);
	int out = 0;
	flag = 0;
	if (rank == 0) {
		// set up the receive window.
		MPI_Win_create(&flag, sizeof(int), sizeof(int), MPI_INFO_NULL, comm_world, &win);
	} else {

		// set up the send window
		MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, comm_world, &win);

		for (int i = 0; i < 4; i++) {

		// now do selective.
		MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
		MPI_Get(&out, 1, MPI_INT, 0, 0, 1, MPI_INT, win);
		MPI_Win_unlock(0, win);

		sleep(rank);
		if (out < 2) {

			// now do selective.
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
			MPI_Accumulate(&rank, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, win);
			MPI_Win_unlock(0, win);
		}
		printf("rank %d flag = %d, out = %d\n", rank, flag, out);
		}
	}
	MPI_Win_free(&win);


	// testing issend, cancel, with iprobe.
	MPI_Status mstatus;
	bool done = false;
	bool wait = false;
	if (rank == 0) {

		while (!done) {
			int hasMessage;
			int worker_id;
			int worker_status;

			MPI_Iprobe(MPI_ANY_SOURCE, 1, comm_world, &hasMessage, &mstatus);
			if (hasMessage) {
				worker_id = mstatus.MPI_SOURCE;
//				printf("manager queue has %lu messages?\n", mstatus._ucount);

				MPI_Recv(&worker_status, 1, MPI_INT, worker_id, 1, comm_world, &mstatus);
				printf("manager received %d from %d with error %d canceled? %d\n", worker_status, worker_id, mstatus.MPI_ERROR, mstatus._cancelled);

				if (worker_status % 100== 1) {
					done = true;
				} else if (worker_status % 100== 2){
					wait = true;
				}
				MPI_Send(&worker_status, 1, MPI_INT, worker_id, 1, comm_world);

			}

			if (wait == true) {
				printf("wait a sec\n");
				sleep(1);
				wait = false;
			}
		}

	} else {

		// call manager with READY, DONE, or ERROR
		bool waitForManager = true;
		int completed = 0;
		int root = 0;
		int err;
		int runs = 0;
		int val = runs * 100 + 0;
		MPI_Request myRequest;
		int manager_status;

		while (!done) {
			if (runs % 10 == 1) {
				val = runs * 100 + 2;
			} else if (runs > 20) {
				val = runs * 100 + 1;
			} else {
				val = runs * 100;
			}
			runs++;

			waitForManager = true;

			while (waitForManager) {
				int iter = 0;
				completed = 0;
				err = MPI_Issend(&val, 1, MPI_INT, root, 1, comm_world, &myRequest);   // send the current status
				printf("%d issend %d status %d\n", rank, val, err);
				err = MPI_Test(&myRequest, &completed, &mstatus);
//				printf("%d checking root %d, completed = %d, error %d\n", rank, root, completed, err);
				while (completed == 0 && iter < 100) {
					usleep(1000);
					err = MPI_Test(&myRequest, &completed, &mstatus);
					++iter;
				}
				if (completed != 0) {
					printf("%d got root %d for %d complted = %d \n", rank, root, val, completed);
					waitForManager = false;  // myRequest is cleaned up
				} else if (completed == 0 && iter >= 100) {
	//				printf("%d did not get root %d.  compelted = %d\n", rank, root, completed);
					err = MPI_Cancel(&myRequest);
	//				printf("%d cancel status %d\n", rank, err);

					err = MPI_Request_free(&myRequest);   // clean up.
	//				printf("%d req free status %d\n", rank, err);

				}
			}

		//		root = scheduler->getRootFromLeaf(rank);
		//		MPI_Send(&buffer_status, 1, MPI_INT, root, CONTROL_TAG, comm);

			MPI_Recv(&manager_status, 1, MPI_INT, root, 1, comm_world, &mstatus);
			printf("manager status received = %d\n", manager_status);
			if (manager_status % 100 == 1) {
				done = true;
			}
		}

	}




	MPI_Finalize();
}
