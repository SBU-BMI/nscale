/*
 * PushCommHandler.cpp
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#include "PushCommHandler.h"
#include "Debug.h"
#include <unistd.h>

namespace cci {
namespace rt {

PushCommHandler::PushCommHandler(MPI_Comm const * _parent_comm, int const _gid, Scheduler_I * _scheduler, cciutils::SCIOLogSession *_logger)
: CommHandler_I(_parent_comm, _gid, _scheduler, _logger) {
}

PushCommHandler::~PushCommHandler() {
//	Debug::print("%s destructor called.\n", getClassName());
}

/**
 * communicate between manager and workers.
 * workers can tell manager that it's in wait, ready, done, or error states
 * manager can tell all workers that it's in wait, ready, done, or error states
 *
 * data is sent only when both are in ready state
 *
 * data is pull from worker's action's queue, sent to manager in first come first serve way,
 * and stored by manager into it's action's queue
 *
 */
int PushCommHandler::run() {

	// not need to check for action== NULL. NULL action sets status to ERROR

	call_count++;
	//if (call_count % 100 == 0) Debug::print("PushCommHandler %s run called %d. \n", (isListener() ? "listener" : "requester"), call_count);

	int count;
	void * data = NULL;
	int buffer_status = READY;

	int worker_status = READY;
	int manager_status = READY;
	MPI_Status mstatus;
	MPI_Request myRequest;

	if (isListener()) {
		if (!this->isReady()) return status;

		if (action->canAddInput())
			buffer_status = READY;
		else
			buffer_status = action->getInputStatus();  // get the data, and the return status

//		Debug::print("%s buffer status = %d\n", getClassName(), buffer_status);


		int hasMessage;
		int worker_id;

		MPI_Iprobe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &hasMessage, &mstatus);
		if (hasMessage) {
//		MPI_Probe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &mstatus);
			worker_id = mstatus.MPI_SOURCE;

			MPI_Recv(&worker_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm, &mstatus);

			// track the worker status
			if (worker_status == DONE || worker_status == ERROR) {
				activeWorkers.erase(worker_id);

	
				// NOTE: workers are responsible for notifying all the masters it knows about.
				// if all workers are done, or in error state, then this is done too.
				if (activeWorkers.empty()) {
					status = DONE;  // nothing to send to workers. since they are all done.
					Debug::print("%s all workers DONE.\n", getClassName());
					// all the workers are done
					action->markInputDone();
				}
				return status;  // worker is in done or error state, don't send.

			} else {
				activeWorkers[worker_id] = worker_status;

				if (worker_status == WAIT) { // worker status is wait, so manager doesn't do anything.
					Debug::print("%s Worker waiting.  why did it send the message in the first place?\n", getClassName());
					return worker_status;  // return status is worker wait, but keep manager at ready.
				}
			}

			// READY to receive.



			MPI_Send(&buffer_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm);
			//Debug::print("buffer status sent \n", buffer_status);

			if (buffer_status == DONE || buffer_status == ERROR ) {  // if action is done or error,
				// then we are done.  need to notify everyone.
				// keep the commhandler status at READY.
				// and set message to all nodes

				activeWorkers.erase(worker_id);
				if (activeWorkers.empty()) {// all messages sent
					status = DONE;

					Debug::print("%s DONE\n", getClassName());
					// action is already done, so no need to change it.
				}
				return status;

			} else if (buffer_status == READY) {
				// status is ready, send data.
				MPI_Recv(&count, 1, MPI_INT, worker_id, DATA_TAG, comm, &mstatus);
				data = malloc(count);
				MPI_Recv(data, count, MPI_CHAR, worker_id, DATA_TAG, comm, &mstatus);

//				if (count > 0) {
//					int *i2 = (int *)data;
//					Debug::print("%s requester recv %d at %x from %d\n", getClassName(), *i2, data, worker_id);
//				} else
//					Debug::print("%s requester recv ?? at %x from %d\n", getClassName(), data, worker_id);


				if (count > 0 && data != NULL) {
					buffer_status = action->addInput(count, data);
				} else {
					Debug::print("%s READING nothing!\n", getClassName());
				}

				return buffer_status;
			} else {
//				Debug::print("%s waiting\n", getClassName());
			} // else wait.  so do nothing

			return status;
		} else {
			return WAIT;
		}

	} else {

		// status is only for checking error, done, and prescence of action.
		if (!this->isReady()) return status;

		buffer_status = action->getOutputStatus();
//		Debug::print("%s worker buffer_status is :%d\n", getClassName(), status);

		if (buffer_status == WAIT) {
			return buffer_status;  // nothing to output, so don't call manager
		} else if (buffer_status == DONE || buffer_status == ERROR) {
			// notify all the roots
			std::vector<int> roots = scheduler->getRoots();

			for (std::vector<int>::iterator iter=roots.begin();
					iter != roots.end(); ++iter) {
				MPI_Isend(&buffer_status, 1, MPI_INT, *iter, CONTROL_TAG, comm, &myRequest);
			}
			// and say done.
			status = DONE;

			return status;
		}  // else buffer status is READY.


		int root;
//		// call manager with READY, DONE, or ERROR
//		bool waitForManager = true;
//		int completed = 0;
//		int iter = 0;
//		int err;
//		int val = call_count * 100 + buffer_status;
//
//
//		while (waitForManager) {
//			iter = 0;
//			completed = 0;
//			root = scheduler->getRootFromLeaf(rank);
//			err = MPI_Issend(&val, 1, MPI_INT, root, CONTROL_TAG, comm, &myRequest);   // send the current status
//			printf("%d issend %d status %d\n", rank, val, err);
//			err = MPI_Test(&myRequest, &completed, &mstatus);
//			printf("%d checking root %d, completed = %d, error %d\n", rank, root, completed, err);
//			while (completed == 0 && iter < 100) {
//				usleep(1000);
//				err = MPI_Test(&myRequest, &completed, &mstatus);
//				++iter;
//			}
//			if (completed != 0) {
//				printf("%d got root %d complted = %d \n", rank, root, completed);
//				waitForManager = false;  // myRequest is cleaned up
//			} else if (iter >= 100) {
////				printf("%d did not get root %d.  compelted = %d\n", rank, root, completed);
//				err = MPI_Cancel(&myRequest);
////				printf("%d cancel status %d\n", rank, err);
//
//				err = MPI_Request_free(&myRequest);   // clean up.
////				printf("%d req free status %d\n", rank, err);
//
//			}
//		}

		root = scheduler->getRootFromLeaf(rank);
//		Debug::print("%s %d target root is %d\n", getClassName(), rank, root);
		MPI_Send(&buffer_status, 1, MPI_INT, root, CONTROL_TAG, comm);

		MPI_Recv(&manager_status, 1, MPI_INT, root, CONTROL_TAG, comm, &mstatus);

//		Debug::print("%s manager status is %d\n", getClassName(), manager_status);

		if (manager_status == READY) {
			status = action->getOutput(count, data);
			if (status != READY) Debug::print("%s ERROR: status has changed during a single invocation of run()\n", getClassName());

			MPI_Send(&count, 1, MPI_INT, root, DATA_TAG, comm);
			MPI_Send(data, count, MPI_CHAR, root, DATA_TAG, comm);
			if (data != NULL) free(data);
//			Debug::print("%s %d sent data to %d\n", getClassName(), rank, root);

		} else if (manager_status == DONE || manager_status == ERROR ) {
			// one manager is done.  remove it from the list.  if there is no roots left, done.
			if (scheduler->removeRoot(root) == 0)  {
				status = DONE;
				// if manager can't accept, then action can stop
				action->markInputDone();
				Debug::print("%s DONE\n", getClassName());
			}

		} else {
//			Debug::print("%s waiting\n", getClassName());
		}  // else we are in wait state.  nothing to be done.
		return status;

	}
}

} /* namespace rt */
} /* namespace cci */
