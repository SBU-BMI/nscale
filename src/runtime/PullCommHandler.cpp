/*
 * PullCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PullCommHandler.h"
#include "Debug.h"
#include <algorithm>

namespace cci {
namespace rt {

PullCommHandler::PullCommHandler(MPI_Comm const * _parent_comm, int const _gid, MPIDataBuffer *_buffer, Scheduler_I * _scheduler, cciutils::SCIOLogSession *_logsession)
: CommHandler_I(_parent_comm, _gid, _buffer, _scheduler, _logsession), send_count(0) {
}

PullCommHandler::~PullCommHandler() {
	MPI_Barrier(comm);
	if (isListener()) {
		Debug::print("%s destructor called.  total of %d data messages sent.\n", getClassName(), send_count);
	} else {
//		Debug::print("%s destructor called.\n", getClassName());
	}

}


/**
 * communicate between manager and workers.
 * workers can tell manager that it's in wait, ready, done, or error states
 * manager can tell all workers that it's in wait, ready, done, or error states
 *
 * data is sent only when both are in ready state
 *
 * data is pull from manager's action's queue, sent to worker in demand driven way,
 * and stored by worker into it's action's queue
 *
 * comm handler does not have "state" itself.  just validity.
 *
 */
int PullCommHandler::run() {

	// not need to check for action== NULL. NULL action sets status to ERROR
	long long t1, t2;
	t1 = cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);


	call_count++;

	//	if (call_count % 100 == 0) Debug::print("%s %s run called %d. \n", getClassName(), (isListener() ? "listener" : "requester"), call_count);

	int count = 0;
	void * data = NULL;
	int node_status = Communicator_I::READY;
	MPI_Status mstatus;
	MPI_Request myRequest;
	int node_id;
	int tag = 1;
	MPI_Datatype type;
	int lstatus;
	int numWorkToGet;
	int completed;

	if (isListener()) {
		// logic:
		// 1. clean up pending requests.
		// 2. if all workers done, then stop buffer, mark as done, receive and discard other messages
		// 3. if there are active workers,
		//		if there are no messages, wait
		//      if empty buffer and not stopped, wait
		// 4. get the message
		//		if done, remove worker from list
		//		if ready (request),
		//			if finished buffer, then send back 0
		//			(empty but not stopped is already checked)
		//			if buffer has content, send it back.

		// 1. sender:  clean up everything that's been sent.
		completed = buffer->checkRequests();
		//if (completed > 0) Debug::print("%s master completed %d requests\n", getClassName(), completed);

		if (!scheduler->hasLeaves()) {  // from previous runs - no more workers to consume
			buffer->stop();

			// consume and discard all remaining messages - can't service them anyways.
			while (waComm->iprobe(MPI_ANY_SOURCE, tag, &mstatus)) {
				node_id = mstatus.MPI_SOURCE;

				Debug::print("WARNING: %s there should not be any messages here\n", getClassName());

				// status update, "DONE".  receive it and terminate.
				MPI_Recv(&node_status, 1, MPI_INT, node_id, tag, comm, &mstatus);

			}

			return Communicator_I::DONE;
		}

		// else there is some worker requesting/listening
		if (!waComm->iprobe(MPI_ANY_SOURCE, tag, &mstatus)) {
			usleep(1000);
			return Communicator_I::WAIT;  // message may come later.  need to make sure all workers eventually are done.
		}

		// if buffer is empty but not stopped, we wait to receive any further messages
		if (!buffer->canTransmit() && !buffer->isStopped()) return Communicator_I::WAIT;

		// has a message.  now receive it.
		node_id = mstatus.MPI_SOURCE;
		MPI_Recv(&node_status, 1, MPI_INT, node_id, tag, comm, &mstatus);

		if (node_status == Communicator_I::DONE) {
			// worker sent message that it's done
			// remove it.
			scheduler->removeLeaf(node_id);

			return Communicator_I::WAIT;
		} else if (node_status == Communicator_I::READY) {
			// worker send ready, so this is a request.

			if (buffer->isFinished()) {
				// buffer finished, so send back 0 length data as response to indicate DONE
				data = NULL;
				MPI_Send(data, 0, MPI_CHAR, node_id, tag, comm);

				scheduler->removeLeaf(node_id);
				Debug::print("%s manager: worker %d finished\n", getClassName(), node_id);
				return Communicator_I::WAIT;
			} else if (buffer->canTransmit()){
				// has data

				buffer->transmit(node_id, tag, MPI_CHAR, comm, -1);
				++send_count;
				//Debug::print("%s manager sending %d data to %d\n", getClassName(), send_count, node_id);

				if (send_count % 100 == 0) Debug::print("%s manager sent %d data messages to workers.\n", getClassName(), send_count);

			} // else cannot transmit, and not stopped.  already handled before receiving

			return status;

		}

	} else {


		// 1. receiver:  clean up everything that's been sent.
		completed = buffer->checkRequests();
		if (completed > 0) Debug::print("%s worker completed %d requests\n", getClassName(), completed);

		// no more managers to send requests to.  this is done.
		if (!scheduler->hasRoots()) {
			buffer->stop();
			// no requests to send.  all responses should have been processed.
			return Communicator_I::DONE;
		}

		// NOTE:  enforce that worker has to send request and wait for the response.
		// if buffer can't accept any more (stopped), than we are done.  send messages to roots, and done.
		if (buffer->isStopped()) {
			// worker buffer is finished.  let all roots know to remove this worker from list.
			// notify all the roots
			std::vector<int> roots = scheduler->getRoots();
			std::random_shuffle(roots.begin(), roots.end());  // avoid synchronized termination.

			Debug::print("%s worker buffer DONE\n", getClassName());
			MPI_Request *reqs = new MPI_Request[roots.size()];
			int i = 0;

			status = Communicator_I::DONE;
			for (std::vector<int>::iterator iter=roots.begin();
					iter != roots.end(); ++iter) {

				MPI_Isend(&status, 1, MPI_INT, *iter, tag, comm, &(reqs[i]));
				++i;
//				MPI_Send(&status, 1, MPI_INT, *iter, Communicator_I::DONE, comm);
			}
			MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);
			Debug::print("%s worker notified ALL MANAGERS with DONE\n", getClassName());
			delete [] reqs;
			return status;

		} else if (!buffer->canTransmit()) return Communicator_I::WAIT;  // full buffer, or stopped buffer. but it's not stopped here.
		
		// else buffer is ready and not full.  send 1 request.
		// local buffer is READY.  send request and get data or status back.
		node_id = scheduler->getRootFromLeaf(rank);

		// double check to make sure that buffer is not full.
		//			Debug::print("%s worker sending request to %d with status %d\n", getClassName(), node_id, status);
		MPI_Send(&status, 1, MPI_INT, node_id, tag, comm);   // send the current status
//		Debug::print("%s worker sent request to %d with status %d\n", getClassName(), node_id, status);

		// handle all the received data (may be from the request just sent, or from an earlier request.  Doesn't matter.)

			// need to get data size.  use Probe.
//		Debug::print("%s worker getting data size from probe to %d\n", getClassName(), node_id);
		t1 = cciutils::event::timestampInUS();
		MPI_Probe(node_id, tag, comm, &mstatus);
		t2 = cciutils::event::timestampInUS();
		Debug::print("%s worker got data size from probe to %d in %ld us\n", getClassName(), node_id, (t2-t1));
		// receive some data.
		MPI_Get_count(&mstatus, MPI_CHAR, &count);

		if (count > 0) {  // receiving data!

			buffer->transmit(node_id, tag, MPI_CHAR, comm, count);
			++send_count;
			Debug::print("%s worker got data size %d from probe to %d, so far %d\n", getClassName(), count, node_id, send_count);
		} else {
			data = NULL;
			// manager is done
			MPI_Recv(data, 0, MPI_CHAR, node_id, tag, comm, MPI_STATUS_IGNORE);

			scheduler->removeRoot(node_id);
		}
		return status;

	}  // end worker.
}

} /* namespace rt */
} /* namespace cci */
