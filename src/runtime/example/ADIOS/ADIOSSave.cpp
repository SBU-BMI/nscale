/*
 * ADIOSSave.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "ADIOSSave.h"
#include "Debug.h"
#include "mpi.h"


namespace cci {
namespace rt {
namespace adios {


ADIOSSave::ADIOSSave(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogger *logger, cciutils::ADIOSManager *_iomanager, std::string &iocode) :
		Action_I(_parent_comm, _gid), iomanager(_iomanager), local_iter(0), global_iter(0), buffer_max(4), local_count(0) {


	// determine if we are using AMR, if so, don't append time points
	bool appendInTime = true;
	if (strcmp(iocode.c_str(), "MPI_AMR") == 0 ||
		strcmp(iocode.c_str(), "gap-MPI_AMR") == 0) appendInTime = false;

	// always overwrite.
	bool overwrite = true;

	// specify output
	std::string outDir("/home/tcpan/PhD/path/Data/adios/async-yellowstone");

	// and the stages to capture.
	std::vector<int> stages;
	for (int i = 0; i < 200; i++) {
		stages.push_back(i);
	}

	// maximum number of image tiles (for testing only)
	int total = 100;


	writer = iomanager->allocateWriter(outDir, std::string("bp"), appendInTime, overwrite,
			stages, total, total * (long)256, total * (long)1024, total * (long)(4096 * 4096 * 4),
			buffer_max, 4096*4096*4,
			groupid, &comm);

	if (rank == 0) {
		// set up the receive window.
		MPI_Win_create(&global_iter, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &iter_win);
	} else {
		// set up the send window
		MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, comm, &iter_win);
	}

	local_total = 0;

}

ADIOSSave::~ADIOSSave() {
	Debug::print("%s destructor called.  total written out is %d\n", getClassName(), local_total);

	if (writer) writer->persistCountInfo();

	iomanager->freeWriter(writer);

	MPI_Win_free(&iter_win);
}

int ADIOSSave::compute(int const &input_size , void * const &input,
			int &output_size, void * &output) {
	return READY;
}


int ADIOSSave::run() {

	// if input is done, it enters into fence and wait until everyone else is done.

	call_count++;

	int max_iter;

	//// first check globally if we need to write because of others or my buffer

	// get the local status and local buffer size
	local_count = inputSizes.size();

	bool flushing;

	if (input_status == DONE || input_status == ERROR) {
		MPI_Win_fence(0, iter_win); // this makes the entire process wait.  but that's okay since the whole process's action is DONE...

//		MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, iter_win);
		MPI_Get(&max_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, iter_win);  // get the current max iter
//		MPI_Win_unlock(0, iter_win);

		MPI_Win_fence(0, iter_win);
		// get the global iteration id and number of remaining actions (processes)
		if (local_count != 0 && max_iter <= local_iter) {
			++local_iter;  // if there is some data to be written out, project that we have one extra
			MPI_Accumulate(&local_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, iter_win);
			--local_iter;
		} else {
			MPI_Accumulate(&local_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, iter_win);
		}
		MPI_Win_fence(0, iter_win); // this makes the entire process wait.  but that's okay since the whole process's action is DONE...

		Debug::print("DONE local rank %d status %d, local iter %d, max iter %d, local count %d\n", rank, input_status, local_iter, max_iter, local_count);
	}

	MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, iter_win);
	MPI_Get(&max_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, iter_win);
	MPI_Win_unlock(0, iter_win);

	// catch up.
	if (max_iter > local_iter) {
		while (max_iter - 1 > local_iter) {
			// write empty

			++local_iter;
		}
		// last iteration to catch up.
		// write
		write();

		++local_iter;
		Debug::print("CATCH UP ITERS local rank %d status %d, local iter %d, max iter %d, local count %d\n", rank, input_status, local_iter, max_iter, local_count);

	} else {
		if (input_status == WAIT || input_status == READY) {
			// this would be a new iteration, only if count is at max
			if (local_count >= buffer_max) {
				// write some new data
				write();

				++local_iter;
				// update remote
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, iter_win);
				MPI_Accumulate(&local_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, iter_win);
				MPI_Get(&max_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, iter_win);
				MPI_Win_unlock(0, iter_win);
				Debug::print("INIT NEW ITER local rank %d status %d, local iter %d, max iter %d, local count %d\n", rank, input_status, local_iter, max_iter, local_count);
			}  // else do nothing

		}

	}
	return input_status;
}

int ADIOSSave::write() {

	int input_size;  // allocate output vars because these are references
	void *input;
	int result = getInput(input_size, input);

	while (result == READY) {

		if (input != NULL) {
			free(input);
			input = NULL;
		}
		result = getInput(input_size, input);
		local_total++;
	}


}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
