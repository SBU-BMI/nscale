/*
 * PushCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PushCommHandler.h"
#include "Debug.h"
#include <unistd.h>
#include <algorithm>

//int test_input_status;

namespace cci {
namespace rt {

PushCommHandler::PushCommHandler(MPI_Comm const * _parent_comm, int const _gid, MPIDataBuffer *_buffer, Scheduler_I * _scheduler, cci::common::LogSession *_logsession)
: CommHandler_I(_parent_comm, _gid, _buffer, _scheduler, _logsession), send_count(0) {
}

PushCommHandler::~PushCommHandler() {

//	if (isListener()) {
//		cci::common::Debug::print("%s destructor:  %d msgs received.\n", getClassName(), send_count);
//	} else {
////		cci::common::Debug::print("%s destructor called.\n", getClassName());
//	}
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
	long long t1 = -1, t2 = -1;
	t1 = cci::common::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);

	call_count++;
	//if (call_count % 100 == 0) cci::common::Debug::print("PushCommHandler %s run called %d. \n", (isListener() ? "listener" : "requester"), call_count);

	int count = 0;
	void * data = NULL;
	int node_status = Communicator_I::READY;
	MPI_Status mstatus;
	int node_id = MPI_UNDEFINED;
	int tag = 2;
	int completed = 0;

	if (isListener()) {
                // logic:
                // 1. clean up pending requests.
                // 2. get all worker status updates
                //   if all workers done, then stop buffer, mark as done, receive and discard other messages
                //   if there are active workers,
                // 3. check local buffer status.
                //              if done, notify everyone
                //                      if there are any mpi messages, respond with 0 (worker is expecting to receive something)
                //      else if empty buffer wait
                //      else if empty mpi messages, wait
                //      else service requests.

		// receiver:  clean up everything that's been received.
		completed = buffer->checkRequests();
		//if (completed > 0) cci::common::Debug::print("%s master completed %d requests\n", getClassName(), completed);

		// if the buffer is stopped, then no more data.  notify workers and discard the data
        if (buffer->isStopped()) {
			// worker buffer is finished.  let all roots know to remove this worker from list.
        	//cci::common::Debug::print("%s master buffer is stopped\n", getClassName());

			// clear all pending done messages
            while (waComm->iprobe(MPI_ANY_SOURCE, tag, &mstatus)) {
				node_id = mstatus.MPI_SOURCE;
				MPI_Get_count(&mstatus, MPI_CHAR, &count);

				if (count > 0) {
					data = malloc(count);
				} else {
					// was a done message with 0 size anyways.  remove from leaves
					scheduler->removeLeaf(node_id);
					data = NULL;
				}
				//cci::common::Debug::print("%s master receiving from %d\n", getClassName(), node_id);
				MPI_Recv(data, count, MPI_CHAR, node_id, tag, comm, &mstatus);
				if (count > 0) free(data);
				//cci::common::Debug::print("%s master received from %d\n", getClassName(), node_id);

				// check to see if there are any done messages from that node.
            }
        	cci::common::Debug::print("%s master cleared MPI queue\n", getClassName());

            // then let all the remain nodes know
			std::vector<int> leaves = scheduler->getLeaves();
			std::random_shuffle(leaves.begin(), leaves.end());   // avoid synchronized notifications

			// notify all leaves;
			//cci::common::Debug::print("%s master buffer DONE\n", getClassName());
			MPI_Request *reqs = new MPI_Request[leaves.size()];
			int i = 0;

			//cci::common::Debug::print("%s master notifying all workers\n", getClassName());
			status = Communicator_I::DONE;
			for (std::vector<int>::iterator iter=leaves.begin();
							iter != leaves.end(); ++iter) {

					MPI_Isend(&status, 1, MPI_INT, *iter, tag, comm, &(reqs[i]));
					++i;
//                              MPI_Send(&status, 1, MPI_INT, *iter, Communicator_I::DONE, comm);
			}
			MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);
			cci::common::Debug::print("%s master notified ALL WORKERS DONE\n", getClassName());
			delete [] reqs;

			// all done.  all messages were already cleared.

			return Communicator_I::DONE;
        } else if (!buffer->canTransmit()) {
    		// if the buffer is full, then wait
        	return Communicator_I::WAIT;
        }

		// process the MPI buffer until the databuffer is full.
    	//cci::common::Debug::print("%s master receiving messages\n", getClassName());

    	int iter_send_count = 0;
        if (!waComm->iprobe(MPI_ANY_SOURCE, tag, &mstatus)) usleep(1000);

        while (waComm->iprobe(MPI_ANY_SOURCE, tag, &mstatus)) {
			node_id = mstatus.MPI_SOURCE;
			MPI_Get_count(&mstatus, MPI_CHAR, &count);

			if (count <= 0) {
				data = NULL;
				//cci::common::Debug::print("%s master receiving done from %d\n", getClassName(), node_id);
                MPI_Recv(data, 0, MPI_CHAR, node_id, tag, comm, MPI_STATUS_IGNORE);
				scheduler->removeLeaf(node_id);
//				cci::common::Debug::print("%s master received done from %d\n", getClassName(), node_id);

			} else {
				if (buffer->canTransmit()) {
					//cci::common::Debug::print("%s master receiving non-block from %d\n", getClassName(), node_id);
					buffer->transmit(node_id, tag, MPI_CHAR, comm, count);
					++send_count;
					// cci::common::Debug::print("%s worker got data size %d from probe to %d, so far %d\n", getClassName(), count, node_id, send_count);
					if (send_count % 100 == 0) cci::common::Debug::print("%s manager received %d data messages from workers.\n", getClassName(), send_count);
				} else {
//		        	cci::common::Debug::print("INFO: %s buffer is full\n", getClassName());

					return Communicator_I::WAIT;  // full buffer - don't check to see if leaves are all done - just wait.
				}
			}

        }
    	//cci::common::Debug::print("%s master done receiving %d messages\n", getClassName(), iter_send_count);

		// finally take care of whether there are workers left.
		if (!scheduler->hasLeaves()) {
        	// cci::common::Debug::print("%s master has no more workers. buffer stopped\n", getClassName());

			buffer->stop();
			// no requests to send.  all responses should have been processed.
			return Communicator_I::DONE;
		}

        return status;

	} else {
		// sender:  clean up everything that's been sent.
		completed = buffer->checkRequests();
		//if (completed > 0) cci::common::Debug::print("%s worker completed %d requests\n", getClassName(), completed);

		// first update the manager status
		// only messages to receive from manager are "done" messages
    	//cci::common::Debug::print("%s worker receiving done messages \n", getClassName());

        if (!waComm->iprobe(MPI_ANY_SOURCE,tag, &mstatus)) usleep(1000);

        while (waComm->iprobe(MPI_ANY_SOURCE, tag, &mstatus)) {
			node_id = mstatus.MPI_SOURCE;

			// status update, "DONE".  receive it and terminate.
			//cci::common::Debug::print("%s worker receiving from %d\n", getClassName(), node_id);
			MPI_Recv(&node_status, 1, MPI_INT, node_id, tag, comm, &mstatus);
			scheduler->removeRoot(node_id);
			//cci::common::Debug::print("%s worker received from %d\n", getClassName(), node_id);

        }

		// then check to see if we can send to ANYONE
        if (!scheduler->hasRoots()) {
			// if manager can't accept, then action can stop. remaining data is discarded
			buffer->stop();
			cci::common::Debug::print("%s all managers DONE.  buffer discards %d entries\n", getClassName(), buffer->debugBufferSize());

			return Communicator_I::DONE;
        }

		// then check to see if our buffer is done, empty, or ready
        if (buffer->isFinished()) {
			// let all workers know.
			t1 = cci::common::event::timestampInUS();
			// worker buffer is finished.  let all roots know to remove this worker from list.
			// notify all the roots
			std::vector<int> roots = scheduler->getRoots();
			std::random_shuffle(roots.begin(), roots.end());   // avoid synchronized termination

//                      cci::common::Debug::print("%s worker buffer DONE\n", getClassName());
			MPI_Request *reqs = new MPI_Request[roots.size()];
			int i = 0;

			status = Communicator_I::DONE;
			data = NULL;
			//cci::common::Debug::print("%s buffer is finished.  worker notifying ALL managers with DONE\n", getClassName());
			for (std::vector<int>::iterator iter=roots.begin();
							iter != roots.end(); ++iter) {
				MPI_Isend(data, 0, MPI_CHAR, *iter, tag, comm, &(reqs[i]));
				++i;
//              MPI_Send(&status, 1, MPI_INT, *iter, Communicator_I::DONE, comm);
			}
			MPI_Waitall(i, reqs, MPI_STATUSES_IGNORE);
			// cci::common::Debug::print("%s buffer finished.  worker managers DONE\n", getClassName());
			delete [] reqs;


			t2 = cci::common::event::timestampInUS();
			if (this->logsession != NULL) logsession->log(cci::common::event(0, std::string("worker done"), t1, t2, std::string(), ::cci::common::event::NETWORK_IO));

			return status;
        }

        if (buffer->canTransmit()) {
            // else - not stopped || buffer has entry || mpi_buffer has entry

			// local buffer has entry (stopped or not, MPI_buffer has entry or not.  send.
			node_id = scheduler->getRootFromLeaf(rank);

				// manager READY.  now set up the send.
			//cci::common::Debug::print("%s worker sending to %d\n", getClassName(), node_id);
			buffer->transmit(node_id, tag, MPI_CHAR, comm, -1);
			++send_count;

			//cci::common::Debug::print("%s worker sent %d items to %d\n", getClassName(), send_count, node_id);
			if (send_count % 100 == 0) cci::common::Debug::print("%s worker sent %d data messages to managers.\n", getClassName(), send_count);
			return status;
		} else {
			// else local buffer is empty.  buffer is not stopped or has mpi buffer entries.
			// either way, wait.
			return Communicator_I::WAIT;

		}
	}
}

} /* namespace rt */
} /* namespace cci */
