/*
 * PushCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PushCommHandler.h"
#include "Debug.h"
#include <unistd.h>

//int test_input_status;

namespace cci {
namespace rt {

PushCommHandler::PushCommHandler(MPI_Comm const * _parent_comm, int const _gid, MPIDataBuffer *_buffer, Scheduler_I * _scheduler, cciutils::SCIOLogSession *_logsession)
: CommHandler_I(_parent_comm, _gid, _buffer, _scheduler, _logsession), send_count(0) {
//	hascompletedworker = false;
//	test_input_status = ERROR;
}

PushCommHandler::~PushCommHandler() {
	if (isListener()) {
		Debug::print("%s destructor called.  total of %d data messages received.\n", getClassName(), send_count);
	} else {
//		Debug::print("%s destructor called.\n", getClassName());
	}
}

/**
 * communicate between manager and workers.
 * worker pushes data to manager  It only needs to send "DONE" signal, or send the payload.
 * manager can send worker DONE.  WAIT on manager is essentially handled by buffer on worker and MPI messaging behavior (blocking or non-blocking).
 *
 * data is sent, assuming both sides are in READY state 
 *
 * data is pull from worker's DataBuffer, sent to manager in first come first serve way,
 * and stored by manager into it's DataBuffer
 *
 */
int PushCommHandler::run() {

	// not need to check for action== NULL. NULL action sets status to ERROR
	long long t1, t2;
	t1 = cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);

	call_count++;
	//if (call_count % 100 == 0) Debug::print("PushCommHandler %s run called %d. \n", (isListener() ? "listener" : "requester"), call_count);

	int count;
	void * data = NULL;
	int node_status = Communicator_I::READY;
	MPI_Status mstatus;
	MPI_Request myRequest;
	int hasMessage;
	int node_id;
	int tag;
	MPI_Datatype type;

	// status is only for checking error, done, and prescence of action.
	if (this->status == Communicator_I::DONE) return status;

	if (isListener()) {
		if (buffer->isStopped()) {
			// DONE.  notify all workers via send or isend.
			// then we are done.  need to notify everyone.
			Debug::print("%s manager buffer done\n", getClassName());

			std::vector<int> leaves = scheduler->getLeaves();
			status = Communicator_I::DONE;
			for (std::vector<int>::iterator iter=leaves.begin();
					iter != leaves.end(); ++iter) {
//				MPI_Isend(&status, 1, MPI_INT, *iter, CONTROL_TAG, comm, &myRequest);
				MPI_Send(&status, 1, MPI_INT, *iter, Communicator_I::DONE, comm);
			}
			// and say done.

			t2 = cciutils::event::timestampInUS();
			if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

			return status;

		} else if (buffer->isFull()) {
			//	no need to wait.  the message will still be in buffer waiting.
			return Communicator_I::WAIT;
		} else {
			//	READY.  probe for messages, and receive them (data payloads directly or notice of worker stop.
			// probe for 1 message each call.

			MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &hasMessage, &mstatus);
			//		MPI_Probe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &mstatus);
			if (hasMessage) {
				node_id = mstatus.MPI_SOURCE;
				tag = mstatus.MPI_TAG;

				if (tag == Communicator_I::DONE) {

					// status update.  receive it
					MPI_Recv(&node_status, 1, MPI_INT, node_id, Communicator_I::DONE, comm, &mstatus);

					// already done
					if (!scheduler->isLeaf(node_id)) {
						return status;
					}

					if (scheduler->removeLeaf(node_id) == 0)  {
						status = Communicator_I::DONE;
						// if manager can't accept, then action can stop
						buffer->stop();
						Debug::print("%s all workers DONE.  buffer has %d entries\n", getClassName(), buffer->getBufferSize());
					}

					t2 = cciutils::event::timestampInUS();
	//				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

				} else {  // data
					MPI_Get_count(&mstatus, MPI_CHAR, &count);
					//data.  check to see if we can receive it (empty input, or has room.)
					if (count == 0 || buffer->canPush()) {
						data = malloc(count);
						memset(data, 0, count);

						// status update.  receive it
						MPI_Recv(data, count, MPI_CHAR, node_id, tag, comm, &mstatus);

						if (count > 0) {
							Debug::print("%s received some data.\n", getClassName());
							int stat = buffer->push(std::make_pair(count, data));
							++send_count;
							if (stat != DataBuffer::READY) {
								Debug::print("ERROR: %s push to buffer returned non-READY status %d.  TODO: HANDLE THIS\n", getClassName(), stat);
								free(data);
							}

						} else {
							Debug::print("%s RECEIVED nothing!\n", getClassName());
						}

						t2 = cciutils::event::timestampInUS();
						sprintf(len, "%lu", (long)(count));
						if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("push data received"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));

						if (send_count % 100 == 0) Debug::print("%s manager received %d data messages from workers.\n", getClassName(), send_count);

					} // else no room so leave the message in the MPI receive queue.

				}

			} // else no message so don't do anything.
			return status;

		}


	} else {
		// get all pending messages
		Debug::print("%s updating manager status\n", getClassName());
		// update with all received control_tag message received from roots.
		MPI_Iprobe(MPI_ANY_SOURCE, Communicator_I::DONE, comm, &hasMessage, &mstatus);
		while (hasMessage) {
			node_id = mstatus.MPI_SOURCE;

			MPI_Recv(&node_status, 1, MPI_INT, node_id, Communicator_I::DONE, comm, &mstatus);

			// status update, "DONE".  receive it and terminate.

			if (scheduler->removeRoot(node_id) == 0)  {
				status = Communicator_I::DONE;
				// if manager can't accept, then action can stop
				buffer->stop();
				Debug::print("%s all managers DONE.  buffer has %d entries\n", getClassName(), buffer->getBufferSize());
			}

			t2 = cciutils::event::timestampInUS();
//				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

			// check to see if there are any done messages from that node.
			MPI_Iprobe(MPI_ANY_SOURCE, Communicator_I::DONE, comm, &hasMessage, &mstatus);
		}
		// now check status.  if all managers are done, then worker is done.
		if (status == Communicator_I::DONE) return status;  // empty so nothing to be done.


		// else there is some communicator.  now check my buffer.
		if (buffer->isFinished()) {
			// worker buffer is finished.  let all roots know to remove this worker from list.
			// notify all the roots
			Debug::print("%s buffer is all done.\n", getClassName());

			std::vector<int> roots = scheduler->getRoots();
			status = Communicator_I::DONE;
			for (std::vector<int>::iterator iter=roots.begin();
					iter != roots.end(); ++iter) {

//				MPI_Isend(&buffer_status, 1, MPI_INT, *iter, CONTROL_TAG, comm, &myRequest);
				MPI_Send(&status, 1, MPI_INT, *iter, Communicator_I::DONE, comm);
			}
			// TODO do waitall here.

			t2 = cciutils::event::timestampInUS();
			if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

			return status;
		} else if (buffer->isEmpty()) {
			return Communicator_I::WAIT;
		} else {

			// local buffer is READY.
			node_id = scheduler->getRootFromLeaf(rank);

			// manager READY.  now set up the send.
			Debug::print("%s sending to %d\n", getClassName(), node_id);


			DataBuffer::DataType dstruct;
			int stat = buffer->pop(dstruct);

			if (stat != DataBuffer::EMPTY && dstruct.first > 0 && dstruct.second != NULL) {
				MPI_Send(dstruct.second, dstruct.first, MPI_CHAR, node_id, Communicator_I::READY, comm);
				free(dstruct.second);
			}

			t2 = cciutils::event::timestampInUS();
			sprintf(len, "%lu", (long)(dstruct.first));
			if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("push data sent"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));

			return status;
		}
	}
}

} /* namespace rt */
} /* namespace cci */
