/*
 * ADIOSSave.cpp
 *
 *  Created on: Jun 18, 2012
 *      Author: tcpan
 */

#include "ADIOSSave.h"
#include "Debug.h"
#include "mpi.h"

#include "CVImage.h"
#include "UtilsADIOS.h"

namespace cci {
namespace rt {
namespace adios {


ADIOSSave::ADIOSSave(MPI_Comm const * _parent_comm, int const _gid,
		std::string &outDir, std::string &iocode, int total, int _buffer_max,
		int tile_max, int imagename_max, int filename_max,
		ADIOSManager *_iomanager, cciutils::SCIOLogSession *_logsession) :
		Action_I(_parent_comm, _gid, _logsession), iomanager(_iomanager),
		local_iter(0), global_iter(0), local_total(0), global_done_count(0),
		done_marked(false), all_done(false), c(0),
		buffer_max(_buffer_max) {

	dones = (int*)calloc(size, sizeof(int));

	// determine if we are using AMR, if so, don't append time points
	bool appendInTime = true;
	if (strcmp(iocode.c_str(), "MPI_AMR") == 0 ||
		strcmp(iocode.c_str(), "gap-MPI_AMR") == 0) appendInTime = false;

	// always overwrite.
	bool overwrite = true;

	// and the stages to capture.
	std::vector<int> stages;
	for (int i = 0; i < 200; i++) {
		stages.push_back(i);
	}

	writer = iomanager->allocateWriter(outDir, std::string("bp"), overwrite,
			appendInTime, stages,
			total, buffer_max,
			tile_max, imagename_max, filename_max,
			comm, groupid);
	writer->setLogSession(this->logsession);

	long long t1, t2;
	t1 = ::cciutils::event::timestampInUS();
	if (rank == 0) {
		// set up the receive window.
		MPI_Win_create(&global_iter, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &iter_win);
		//MPI_Win_create(&global_done_count, sizeof(int), sizeof(int), MPI_INFO_NULL, comm, &done_win);
		MPI_Win_create(dones, sizeof(int) * size, sizeof(int), MPI_INFO_NULL, comm, &done_win);
	} else {
		// set up the send window
		MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, comm, &iter_win);
		MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, comm, &done_win);
	}
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO RMA init"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

}

ADIOSSave::~ADIOSSave() {
	Debug::print("%s destructor called.  total written out is %d\n", getClassName(), local_total);

	free(dones);

	if (writer) writer->persistCountInfo();

	iomanager->freeWriter(writer);

	MPI_Barrier(comm);
	MPI_Win_free(&iter_win);
	MPI_Win_free(&done_win);
}

int ADIOSSave::run() {



// first catch up.
	long long t1, t2;
	int max_iter = 0;

	// next see if we are already done.

	// if input is done, it CANNOT use fence and wait - need to participate in persist.

	int incr = 1;
	int status = input_status;
	int done_count;

//	if (test_input_status == DONE)
//		Debug::print("TEST start input status = %d\n", input_status);


	//TODO: change to MPI_AllReduce.

	// first check for done-ness.  if I am done, then increment the done count.
	// if done count == rank, mark atend true;
	MPI_Barrier(comm);   // must have barrier to use with Win_lock for this to work.
	t1 = ::cciutils::event::timestampInUS();
	if (input_status == DONE || input_status == ERROR) {
		c++;

		// can use accumulate or put.
		if (!done_marked) {
			if (rank != 0) {
				MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, done_win);
				Debug::print("Update done count locked\n");
				//MPI_Accumulate(&incr, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_SUM, done_win);
				MPI_Put(&incr, 1, MPI_INT, 0, rank, 1, MPI_INT, done_win);
				Debug::print("Update done count\n");
				MPI_Win_unlock(0, done_win);
				Debug::print("Update done count unlocked\n");
			} else {
				dones[0] = incr;
			}
			done_marked = true;
		}

//		if (test_input_status == DONE)
//			Debug::print("TEST 0.1 input status = %d\n", input_status);
	}

	MPI_Barrier(comm);
		if (rank != 0) {
			// check to see if everyone's done.  if yes, set status to DONE.
			MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, done_win);
	//		MPI_Get(&done_count, 1, MPI_INT, 0, 0, 1, MPI_INT, done_win);
			MPI_Get(dones, size, MPI_INT, 0, 0, size, MPI_INT, done_win);
			MPI_Win_unlock(0, done_win);
		}
		done_count = 0;
		for (int i = 0; i < size; ++i) {
			done_count += dones[i];
		}


//		if (test_input_status == DONE)
//			Debug::print("TEST 0.2 input status = %d\n", input_status);

		if (done_count >= size) {
			all_done = true;
			status = DONE;
		}
		else status = WAIT;

		Debug::print("%s, call_count = %ld, done_count = %d, all done = %d, status = %d, input_status = %d\n", getClassName(), c, done_count, (all_done ? 1 : 0), status, input_status);

		//if (status == WAIT) Debug::print("%s DONE local rank %d status %d, local iter %d, local count %d, done_count = %d\n", getClassName(), rank, input_status, local_iter, inputSizes.size(), done_count);
		t2 = ::cciutils::event::timestampInUS();
		//if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO final RMA"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));
	//}  // else status is same as input status:  READY or WAIT.


//	if (test_input_status == DONE)
//		Debug::print("TEST 1 input status = %d\n", input_status);



	t1 = ::cciutils::event::timestampInUS();

	// TODO: removed check for WAIT next to READY
	if ((inputSizes.size() >= buffer_max && (input_status == READY )) ||
			(inputSizes.size() > 0 && (input_status == DONE || input_status == ERROR))) {
		// not done and has full buffer, or done and has some data
		// increment self and accumulate
		++local_iter;

		MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, iter_win);  // okay to have shared lock for accumulate.
		MPI_Accumulate(&local_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, iter_win);
		MPI_Win_unlock(0, iter_win);

		--local_iter;
	}

	// now get the max_iter, and write.
	// get most current io iteration count
	MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, iter_win);
	MPI_Get(&max_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, iter_win);
	MPI_Win_unlock(0, iter_win);

	Debug::print("%s input_status = %d, max_iter = %d, global_iter = %d, local_iter = %d, buffer size = %ld, done marked = %d\n", getClassName(), input_status, max_iter, global_iter, local_iter, inputSizes.size(), (done_marked ? 1 : 0));

//	if (test_input_status == DONE)
//		Debug::print("TEST 2 input status = %d\n", input_status);

	t2 = ::cciutils::event::timestampInUS();
	//if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO RMA update iter"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));


	// local_count > 0 && max_iter <= local_iter;  //new data to write
	// local_count == 0 && max_iter > local_iter; //catch up with empty
	// local_count > 0 && max_iter > local_iter;  //catch up with some writes
	// local_count == 0 && max_iter <= local_iter;  // update remote, nothing to write locally.

	//	Debug::print("%s rank %d max iter = %d, local_iter = %d\n", getClassName(), rank, max_iter, local_iter);

	/**
	 *  catch up.
	 */
	// catch up.  so flush whatever's in buffer.
	if (max_iter > local_iter) Debug::print("%s rank %d catch up writing at iter %d, max_iter = %d\n", getClassName(), rank, local_iter, max_iter);
	while (max_iter > local_iter) {
		process();
	}

//	if (test_input_status == DONE)
//		Debug::print("TEST end input status = %d\n", input_status);

	//Debug::print("%s rank %d returning input status %d\n", getClassName(), rank, input_status);
	return status;
}

int ADIOSSave::process() {
	Debug::print("%s: IO group %d rank %d, write iter %d, tile count %d\n", getClassName(), groupid, rank, local_iter, inputSizes.size());

	// move data from action's buffer to adios' buffer

	int input_size;  // allocate output vars because these are references
	void *input;
	int result = getInput(input_size, input);

	while (result == READY) {

		if (input != NULL) {
			CVImage *adios_img = new CVImage(input_size, input);
			writer->saveIntermediate(adios_img, 1);

			delete adios_img;
			free(input);
			input = NULL;

			++local_total;
		}
		result = getInput(input_size, input);
	}

//	Debug::print("%s calling ADIOS to persist the files\n", getClassName());
	writer->persist(local_iter);
	// simulate persist. MPI_Barrier(comm);
	writer->clearBuffer();
	++local_iter;

//	Debug::print("%s persist completes at rank %d\n", getClassName(), rank);

	return 0;
}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
