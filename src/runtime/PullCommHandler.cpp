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

PullCommHandler::PullCommHandler(MPI_Comm const * _parent_comm, int const _gid, Scheduler_I * _scheduler)
 : CommHandler_I(_parent_comm, _gid, _scheduler) {
}

PullCommHandler::~PullCommHandler() {
	//printf("PullCommHandler destructor called\n");

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

	call_count++;
	//if (call_count % 100 == 0) Debug::print("PullCommHandler %s run called %d. \n", (isListener() ? "listener" : "requester"), call_count);

	int count = 0;
	void * data = NULL;
	int buffer_status = READY;

	int worker_status = READY;
	int manager_status = READY;
	MPI_Status mstatus;

	if (isListener()) {

		// status is only for checking error, done, and prescence of action.
		// otherwise status is just a reflection of the buffer's status.
		if (!this->isReady()) return status;

		buffer_status = action->getOutputStatus();
		Debug::print("buffer status = %d\n", buffer_status);

		if (buffer_status == WAIT) return buffer_status;  // if buffer is not ready, then skip this call

		// else status is READY or WAIT (treat as READY), and buffer is READY/DONE/ERROR

		// find out the status of the worker.
		int hasMessage;
		int worker_id;
		// ready and listening
		MPI_Iprobe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &hasMessage, &mstatus);
		if (hasMessage) {
			worker_id = mstatus.MPI_SOURCE;

			MPI_Recv(&worker_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm, &mstatus);

			// track the worker status
			if (worker_status == DONE || worker_status == ERROR) {
				activeWorkers.erase(worker_id);  // remove from active worker list.

				// if all workers are done, or in error state, then this is done too.
				if (activeWorkers.empty()) {
					status = DONE;  // nothing to send to workers. since they are all done.
				}
				return status;  // worker is in done or error state, don't send.

			} else {
				activeWorkers[worker_id] = worker_status;

				if (worker_status == WAIT) { // worker status is wait, so manager doesn't do anything.
					Debug::print("Worker waiting.  why did it send the message in the first place?\n");
					return worker_status;  // worker waiting.  keep manager at ready.  should not be here...

				}
			}


			// worker is READY.  manager is ready/WAIT.  buffer is READY/DONE/ERROR

			// let worker know about the buffer status
			MPI_Send(&buffer_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm);

			if (buffer_status == DONE || buffer_status == ERROR ) {  // if worker is done or error,
				// then we are done.  need to notify everyone.
				// keep the commhandler status at READY.
				// and set message to all nodes

				activeWorkers.erase(worker_id);
				if (activeWorkers.empty()) {// all messages sent
					status = buffer_status;

					// if output is done, then no need to mark input as done.
					// action->markInputDone();
				}
				return status;
			} else if (buffer_status == READY) {
				data = NULL;
				buffer_status = action->getOutput(count, data);
				if (buffer_status != READY) Debug::print("ERROR:  status has changed during a single invocation of run()\n");

				Debug::print("%s listener sending %d bytes at %x to %d\n", getClassName(), count, data, worker_id);

				// status is ready, send data.
				MPI_Send(&count, 1, MPI_INT, worker_id, DATA_TAG, comm);
				MPI_Send(data, count, MPI_CHAR, worker_id, DATA_TAG, comm);
				if (data != NULL) {
					Debug::print("%s listener clearing %d byte at %x\n", getClassName(), count, data);
					free(data);
					data = NULL;
				}

				return buffer_status;
			}  else {
				Debug::print("%s buffer in wait state\n", getClassName());
				// else in wait state.  don't do anything.
			}

		}
		return status;

	} else {

		// status is only for checking error, done, and prescence of action.
		if (!this->isReady()) return status;

		status = action->getInputStatus();
		if (action->canAddInput())
			buffer_status = READY;  // assume wait is for action to wait, not for adding input
		else {
			buffer_status = status;
			// now continue on to let manager know about Done and Error.
		}

		int root = scheduler->getRootFromLeave(rank);
		printf("SENDING TO root %d\n", root);

		MPI_Send(&buffer_status, 1, MPI_INT, root, CONTROL_TAG, comm);   // send the current status

		if (buffer_status == READY) {  // only if status is ready then we would receive message back.
			MPI_Recv(&manager_status, 1, MPI_INT, root, CONTROL_TAG, comm, &mstatus);

			if (manager_status == READY) {
				MPI_Recv(&count, 1, MPI_INT, root, DATA_TAG, comm, &mstatus);
				data = malloc(count);
				Debug::print("%s initialized %d bytes at address %x\n", getClassName(), count, data);
				MPI_Recv(data, count, MPI_CHAR, root, DATA_TAG, comm, &mstatus);

//				Debug::print("HERE 5 count %d from source %d, error code %d!\n", count, mstatus.MPI_SOURCE, mstatus.MPI_ERROR);

				if (count > 0) {
					int *i2 = (int *)data;
					Debug::print("%s requester recv %d at %x from %d\n", getClassName(), *i2, data, root);
				} else
					Debug::print("%s requester recv ?? at %x from %d\n", getClassName(), data, root);

				if (count > 0 && data != NULL) {
					status = action->addInput(count, data);  // status takes on buffer status' value
//					free(data);
				} else {
					printf("READING nothing!\n");
				}
				// no need to mark worker as done since worker provided the status...

			} else { // control tag. that means either WAIT or DONE/ERROR
				// manager says to wait or be done/error
				if (manager_status == DONE || manager_status == ERROR) { // if DONE or ERROR,
					status = manager_status;

					//mark worker input as done as well.
					action->markInputDone();
				} // else manager status is WAIT, no need to change local status.

			}


		} // else we're in WAIT state.


		return status;

	}
}

} /* namespace rt */
} /* namespace cci */
