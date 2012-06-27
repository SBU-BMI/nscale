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

PullCommHandler::PullCommHandler(MPI_Comm const * _parent_comm, int const _gid, std::vector<int> _roots)
 : RootedCommHandler_I(_parent_comm, _gid, _roots) {
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

	int count;
	void * data = NULL;
	int buffer_status = READY;

	int worker_status = READY;
	int manager_status = READY;
	MPI_Status mstatus;

	if (isListener()) {

		// status is only for checking error, done, and prescence of action.
		// otherwise status is just a reflection of the buffer's status.
		if (!this->isReady()) return status;

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
					return status;
				}
			}


			// worker is ready for send.

//			// first see if there are any messages to be sent to this worker
//			if (msgToWorker.find(worker_id) != msgToWorker.end()) {  //there is something to send. so send it.
//				// send the message
//				MPI_Send(&msgToWorker[worker_id], 1, MPI_INT, worker_id, CONTROL_TAG, comm);
//				msgToWorker.erase(worker_id);
//				if (msgToWorker.empty()) {
//					status = DONE;  // message is either DONE or ERROR.  after all messages are sent, status is done
//				}
//
//			} else {  // no message to send

				buffer_status = action->getOutputStatus();

				Debug::print("buffer status = %d\n", buffer_status);

				MPI_Send(&buffer_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm);

				if (buffer_status == DONE || buffer_status == ERROR ) {  // if worker is done or error,
					// then we are done.  need to notify everyone.
					// keep the commhandler status at READY.
					// and set message to all nodes
//					for (std::tr1::unordered_map<int, int>::iterator iter = activeWorkers.begin();
//							iter != activeWorkers.end(); ++iter) {
//						msgToWorker[iter->first] = buffer_status;
//					}

//					msgToWorker.erase(worker_id);
					activeWorkers.erase(worker_id);
					if (activeWorkers.empty()) {// all messages sent
						status = DONE;
					}
				} else if (buffer_status == WAIT) {
					// tell this worker to wait.
					//status = buffer_status;
				} else {
					data = NULL;
					status = action->getOutput(count, data);
					if (status != READY) Debug::print("ERROR:  status has changed during a single invocation of run()\n");

					Debug::print("%s listener sending %d bytes at %x to %d\n", getClassName(), count, data, worker_id);

					// status is ready, send data.
					MPI_Send(&count, 1, MPI_INT, worker_id, DATA_TAG, comm);
					MPI_Send(&data, count, MPI_CHAR, worker_id, DATA_TAG, comm);
					if (data != NULL) {
						Debug::print("%s listener clearing %d byte at %x\n", getClassName(), count, data);
						free(data);
						data = NULL;
					}


				}
//			}

		}
		return status;

	} else {

		// status is only for checking error, done, and prescence of action.
		if (!this->isReady()) return status;
		if (action->canAddInput())
			status = READY;
		else
			status = action->getInputStatus();

		printf("SENDING TO root %d\n", roots[0]);

		MPI_Send(&status, 1, MPI_INT, roots[0], CONTROL_TAG, comm);   // send the current status

		if (status == READY) {  // only if status is ready then we would receive message back.
			MPI_Recv(&manager_status, 1, MPI_INT, roots[0], CONTROL_TAG, comm, &mstatus);

			if (manager_status == READY) {
				MPI_Recv(&count, 1, MPI_INT, roots[0], DATA_TAG, comm, &mstatus);
				data = malloc(count);
				Debug::print("%s initialized %d bytes at address %x\n", getClassName(), count, data);
				MPI_Recv(&data, count, MPI_CHAR, roots[0], DATA_TAG, comm, &mstatus);

				Debug::print("HERE 5 count %d from source %d, error code %d!\n", count, mstatus.MPI_SOURCE, mstatus.MPI_ERROR);

				if (count > 0) {
					int *i2 = (int *)data;
					Debug::print("%s requester recv %d at %x from %d\n", getClassName(), *i2, data, roots[0]);
				} else
					Debug::print("%s requester recv ?? at %x from %d\n", getClassName(), data, roots[0]);

				if (count > 0 && data != NULL) {
					status = action->addInput(count, data);
					free(data);
				} else {
					printf("READING nothing!\n");
				}
				// no need to mark worker as done since worker provided the status...

				Debug::print("HERE 6!\n");

			} else { // control tag. that means either wait or DONE/ERROR
				// manager says to wait or be done/error
				if (manager_status == DONE || manager_status == ERROR) { // if DONE or ERROR,
					status = manager_status;

					//mark worker input as done as well.
					action->markInputDone();
					Debug::print("HERE 7!\n");

				} // else manager status is WAIT, no need to change local status.
				Debug::print("HERE 8!\n");

			}


		} // else we're in DONE state.

		Debug::print("finally:  here 9!\n");
		return status;

	}
}

} /* namespace rt */
} /* namespace cci */
