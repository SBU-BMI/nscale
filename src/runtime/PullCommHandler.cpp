/*
 * PullCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PullCommHandler.h"
#include "Debug.h"


namespace cci {
namespace rt {

PullCommHandler::PullCommHandler(MPI_Comm const * _parent_comm, int const _gid, MPIDataBuffer *_buffer, Scheduler_I * _scheduler, cciutils::SCIOLogSession *_logsession)
: CommHandler_I(_parent_comm, _gid, _buffer, _scheduler, _logsession), send_count(0) {
}

PullCommHandler::~PullCommHandler() {
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
	int hasMessage;
	int node_id;
	int tag;
	MPI_Datatype type;
	int lstatus;

	// status is only for checking error, done, and prescence of action.
	if (this->status == Communicator_I::DONE) return status;

	if (isListener()) {
		// using a different pattern than pushcommhanlder.  pull comm handler
		// relies on explicit request and response.
		// push comm handler:  presence of received data is same as request to send.


		// first update the scheduler with all workers that are done.
		MPI_Iprobe(MPI_ANY_SOURCE, Communicator_I::DONE, comm, &hasMessage, &mstatus);
		while (hasMessage) {
			node_id = mstatus.MPI_SOURCE;

			// status update, "DONE".  receive it and terminate.
			MPI_Recv(&node_status, 1, MPI_INT, node_id, Communicator_I::DONE, comm, &mstatus);

			if (scheduler->removeLeaf(node_id) == 0)  {
				status = Communicator_I::DONE;
				// if manager can't accept, then action can stop
				buffer->stop();
				Debug::print("%s all workers DONE.  buffer has %d entries\n", getClassName(), buffer->getBufferSize());
			}

			// check to see if there are any done messages from that node.
			MPI_Iprobe(MPI_ANY_SOURCE, Communicator_I::DONE, comm, &hasMessage, &mstatus);
		}
		if (status == Communicator_I::DONE) return status;  // no workers left to do work.
		// ELSE there is some worker.

		// if buffer is empty but not stopped, we wait to receive any further messages
		if (buffer->isEmpty() && !buffer->isStopped()) return Communicator_I::WAIT;
		// ELSE there is some worker and buffer is either stopped or not empty.
		//     if stopped, need to let worker know.  if not empty, need to send back data when requested.


		// READY.  now look for a request.
		MPI_Iprobe(MPI_ANY_SOURCE, Communicator_I::READY, comm, &hasMessage, &mstatus);
		if (hasMessage) {
			node_id = mstatus.MPI_SOURCE;

//			Debug::print("%s manager receiving request from %d\n", getClassName(), node_id);
			MPI_Recv(&node_status, 1, MPI_INT, node_id, Communicator_I::READY, comm, &mstatus);
//			Debug::print("%s manager received request from %d\n", getClassName(), node_id);

			// if worker is already done.  return after receiving the message
			if (!scheduler->isLeaf(node_id)) return status;
			// ELSE worker is active, there is either status or data to send back

			if (buffer->isFinished()) {
				// DONE.  notify each worker as they request work.

				lstatus = Communicator_I::DONE;
				MPI_Send(&lstatus, 1, MPI_INT, node_id, Communicator_I::DONE, comm);

				if (scheduler->removeLeaf(node_id) == 0) {
					Debug::print("%s manager marked all workers as DONE.\n", getClassName());
					status = Communicator_I::DONE;
					buffer->stop();
				}
//				Debug::print("%s manager notified worker %d it's DONE at time %lld.\n", getClassName(), node_id, cciutils::event::timestampInUS());

				t2 = cciutils::event::timestampInUS();
				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

			} else if (!buffer->isEmpty()) {  // has data to send back

				// worker ready.  send data please.
				DataBuffer::DataType dstruct;
				int stat = buffer->pop(dstruct);

				//				Debug::print("%s listener sending %d bytes at %x to %d\n", getClassName(), dstruct.first, dstruct.second, node_id);
				if (stat == DataBuffer::EMPTY) {
					dstruct.first = 0;
					dstruct.second = NULL;
				}

				// status is ready, send data.
				++send_count;
	//			Debug::print("%s manager sending data to %d\n", getClassName(), node_id);
				MPI_Send(dstruct.second, dstruct.first, MPI_CHAR, node_id, Communicator_I::READY, comm);
	//			Debug::print("%s manager sent data to %d\n", getClassName(), node_id);

				if (dstruct.first > 0 && dstruct.second != NULL) {
					free(dstruct.second);
				}

				t2 = cciutils::event::timestampInUS();
				sprintf(len, "%lu", (long)(count));
				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("pull data sent"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));

				if (send_count % 100 == 0) Debug::print("%s manager sent %d data messages to workers.\n", getClassName(), send_count);

			} else {  // empty but not stopped.  wait state.  should have been caught earlier
				// should not get here...
				return Communicator_I::WAIT;


			}

		} // ELSE no message.
		return status;

	} else {


		// worker
		if (buffer->isStopped()) {
			// worker buffer is finished.  let all roots know to remove this worker from list.
			// notify all the roots
			std::vector<int> roots = scheduler->getRoots();

//			Debug::print("%s worker buffer DONE\n", getClassName());
			status = Communicator_I::DONE;
			for (std::vector<int>::iterator iter=roots.begin();
					iter != roots.end(); ++iter) {

				//                              MPI_Isend(&buffer_status, 1, MPI_INT, *iter, CONTROL_TAG, comm, &myRequest);
				MPI_Send(&status, 1, MPI_INT, *iter, Communicator_I::DONE, comm);
			}
			Debug::print("%s worker notified ALL MANAGERS with DONE\n", getClassName());
			// TODO do waitall here.

			t2 = cciutils::event::timestampInUS();
			if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

			return status;
		} else if (buffer->isFull()) {
			return Communicator_I::WAIT;
		} else {

			// local buffer is READY.  send request and get data or status back.

			node_id = scheduler->getRootFromLeaf(rank);

			// double check to make sure that buffer is not full.
			int manager_status;
//			Debug::print("%s worker sending request to %d with status %d\n", getClassName(), node_id, status);
			MPI_Send(&status, 1, MPI_INT, node_id, Communicator_I::READY, comm);   // send the current status
//			Debug::print("%s worker sent request to %d with status %d\n", getClassName(), node_id, status);

			// need to get data size.  use Probe.
//			Debug::print("%s worker getting data size from probe to %d\n", getClassName(), node_id);
			MPI_Probe(node_id, MPI_ANY_TAG, comm, &mstatus);
			tag = mstatus.MPI_TAG;

			if (tag == Communicator_I::DONE) {
				// get the message
				MPI_Recv(&lstatus, 1, MPI_INT, node_id, Communicator_I::DONE, comm, &mstatus);

				// now mark the root as done
				if (scheduler->removeRoot(node_id) == 0) {
					Debug::print("%s worker marked all managers as DONE.\n", getClassName());
					status = Communicator_I::DONE;
					buffer->stop();
				}


			} else {  // tag == READY
				MPI_Get_count(&mstatus, MPI_CHAR, &count);
//				Debug::print("%s worker got data size from probe to %d\n", getClassName(), node_id);

				if (count > 0) {
					data = malloc(count);
				} else data = NULL;

//				Debug::print("%s worker receiving data from %d\n", getClassName(), node_id);
				MPI_Recv(data, count, MPI_CHAR, node_id, Communicator_I::READY, comm, &mstatus);
//				Debug::print("%s worker received data from %d\n", getClassName(), node_id);

				if (count > 0) {
					int stat = buffer->push(std::make_pair(count, data));

					if (stat != DataBuffer::READY) {
						Debug::print("%s ERROR: status has changed during a single invocation of run().  TODO: handle this better.\n", getClassName());
						free(data);
					}
				} else {
					Debug::print("%s RECVING nothing!\n", getClassName());
					// status remain the same.
				}
				t2 = cciutils::event::timestampInUS();
				sprintf(len, "%lu", (long)(count));
				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("pull data received"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));

			}

			return status;
		} // end buffer status if.

	}  // end worker.
}

} /* namespace rt */
} /* namespace cci */
