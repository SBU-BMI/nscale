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
		local_iter(0), global_iter(0), local_total(0),
		buffer_max(_buffer_max) {


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
	} else {
		// set up the send window
		MPI_Win_create(NULL, 0, sizeof(int), MPI_INFO_NULL, comm, &iter_win);
	}
	t2 = ::cciutils::event::timestampInUS();
	if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO RMA init"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

}

ADIOSSave::~ADIOSSave() {
	Debug::print("%s total written out is %d\n", getClassName(), local_total);

	if (writer) writer->persistCountInfo();

	iomanager->freeWriter(writer);

	MPI_Win_free(&iter_win);
}

int ADIOSSave::run() {

	// if input is done, it enters into fence and wait until everyone else is done.
	long long t1, t2;

	int max_iter;

	//// first check globally if we need to write because of others or my buffer

	// get the local status and local buffer size
	int local_count = inputSizes.size();

	bool atend = false;


	// local_count > 0 && max_iter <= local_iter;  //new data to write
	// local_count == 0 && max_iter > local_iter; //catch up with empty
	// local_count > 0 && max_iter > local_iter;  //catch up with some writes
	// local_count == 0 && max_iter <= local_iter;  // update remote, nothing to write locally.

	/**
	 * if status is done, then wait for everyone else too and get the max iter across all processes
	 */
	if (input_status == DONE || input_status == ERROR) {
		t1 = ::cciutils::event::timestampInUS();

		atend = true;

		// this makes the entire application wait.  but that's okay since the whole process's action is DONE...
		MPI_Win_fence(MPI_MODE_NOPRECEDE, iter_win);

		// then everyone update the global iter so the max is accurate for all processed in communicator.
		//  ( in case this process has done more iterations than others)
		if (local_count > 0) ++local_iter;  // if there is some data to be written out, iteration count WILL increase by at least 1.
		MPI_Accumulate(&local_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, iter_win);
		if (local_count > 0) --local_iter;

		MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, iter_win);   // nostore on rank 0?:  no local modification to iter_win's content.  since last fence.


		Debug::print("%s DONE local rank %d status %d, local iter %d, local count %d\n", getClassName(), rank, input_status, local_iter, local_count);
		t2 = ::cciutils::event::timestampInUS();
		if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO final RMA"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));
	}

	// get most current io iteration count
	MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, iter_win);
	MPI_Get(&max_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, iter_win);
	MPI_Win_unlock(0, iter_win);

//	Debug::print("%s rank %d max iter = %d, local_iter = %d\n", getClassName(), rank, max_iter, local_iter);

	/**
	 *  catch up.
	 */
	if (max_iter > local_iter) {  // at DONE, max_iter accounts for the buffer content.
		while (max_iter - 1 > local_iter) {
			// write empty
			if (atend) Debug::print("%s rank %d catch up writing empty at iter %d\n", getClassName(), rank, local_iter);
			process(true);
		}
		// last iteration to catch up.
		// write what's in the buffer
		process(false);
		if (atend) Debug::print("%s CATCHUP ITER local rank %d status %d, local iter %d, local count %d\n", getClassName(), rank, input_status, local_iter, local_count);
	} else {  // new data to be written, and not a catch up.
		if (input_status == WAIT || input_status == READY) {
			// this would be a new iteration, only if count is at max
			if (local_count >= buffer_max) {
				// update remote - important to do this before calling process().  process() is blocking.
				// need for other processes to know about the latest iteration number,
				// so for the processes that do not have full buffer, they can enter into catchup mode
				// during the next iteration.
				t1 = ::cciutils::event::timestampInUS();

				++local_iter;  // pre-update.
				MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, iter_win);  // okay to have shared lock for accumulate.
				MPI_Accumulate(&local_iter, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_MAX, iter_win);
				MPI_Win_unlock(0, iter_win);
				--local_iter;
				t2 = ::cciutils::event::timestampInUS();
				if (this->logsession != NULL) this->logsession->log(cciutils::event(0, std::string("IO RMA put"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));


				// write some new data
				process(false);
				if (atend) Debug::print("%s INIT NEW ITER local rank %d status %d, local iter %d, local count %d\n", getClassName(), rank, input_status, local_iter, local_count);
			}  else {
				if (atend) Debug::print("%s local==max, no data. local rank %d status %d, local iter %d, local count %d\n", getClassName(), rank, input_status, local_iter, local_count);
			}

		} else {
			if (atend) Debug::print("%s local==max, input = done. local rank %d status %d, local iter %d, local count %d\n", getClassName(), rank, input_status, local_iter, local_count);
		}

	}



	//Debug::print("%s rank %d returning input status %d\n", getClassName(), rank, input_status);
	return input_status;
}

int ADIOSSave::process(bool emptyWrite) {
	Debug::print("%s: IO group %d rank %d, write iter %d, tile count %d\n", getClassName(), groupid, rank, local_iter, inputSizes.size());

	if (!emptyWrite) {
		// move data from action's buffer to adios' buffer

		int input_size;  // allocate output vars because these are references
		void *input;
		int result = getInput(input_size, input);

		while (result == READY) {

			if (input != NULL) {
				CVImage *adios_img = new CVImage(input_size, input);
				writer->saveIntermediate(*adios_img, 1);
//				int dummy;
//				Debug::print("%s memcpying tile %s from image %s\n", getClassName(), adios_img->getSourceFileName(dummy), adios_img->getImageName(dummy) );

				delete adios_img;
				free(input);
				input = NULL;

				++local_total;
			}
			result = getInput(input_size, input);
		}
	}

//	Debug::print("%s calling ADIOS to persist the files\n", getClassName());
	writer->persist(local_iter);
	++local_iter;

//	Debug::print("%s persist completes at rank %d\n", getClassName(), rank);

	return 0;
}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
