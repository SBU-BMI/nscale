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
	long long t1, t2;
	t1 = cciutils::event::timestampInUS();
	char len[21];  // max length of uint64 is 20 digits
	memset(len, 0, 21);

	call_count++;
	//if (call_count % 100 == 0) Debug::print("PushCommHandler %s run called %d. \n", (isListener() ? "listener" : "requester"), call_count);

	int count;
	void * data = NULL;
	int buffer_status = Communicator_I::READY;
	int worker_status = Communicator_I::READY;
	int manager_status = Communicator_I::READY;
	MPI_Status mstatus;
	MPI_Request myRequest;

	// status is only for checking error, done, and prescence of action.
	if (this->status == Communicator_I::DONE) return status;


	if (isListener()) {


//		if (hascompletedworker && buffer_status != READY) Debug::print("%s manager %d status %d.\n", getClassName(), rank, buffer_status);

		if (buffer->isStopped()) buffer_status = Communicator_I::DONE;  // get the data, and the return status
		else if (buffer->isFull()) buffer_status = Communicator_I::WAIT;
		else buffer_status = Communicator_I::READY;
		//if (hascompletedworker) Debug::print("%s buffer status = %d\n", getClassName(), buffer_status);


		if (buffer_status == Communicator_I::WAIT) return buffer_status;  // if buffer is not ready, then skip this call


		int hasMessage;
		int worker_id;

		MPI_Iprobe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &hasMessage, &mstatus);
		if (hasMessage) {
//		MPI_Probe(MPI_ANY_SOURCE, CONTROL_TAG, comm, &mstatus);
			worker_id = mstatus.MPI_SOURCE;
			if (activeWorkers.find(worker_id) == activeWorkers.end()) return status;

			MPI_Recv(&worker_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm, &mstatus);

//			if (hascompletedworker) Debug::print("%s worker %d status = %d\n", getClassName(), worker_id, worker_status);

			// track the worker status
			if (worker_status == Communicator_I::DONE || worker_status == Communicator_I::ERROR) {
//				hascompletedworker = true;
//				Debug::print("%s worker %d status = %d\n", getClassName(), worker_id, worker_status);
//				action->debugOn();

				activeWorkers.erase(worker_id);
//				std::stringstream ss;
//				ss << "active workers: [";
//				for (std::tr1::unordered_map<int, int>::iterator iter = activeWorkers.begin();
//						iter != activeWorkers.end(); ++iter) {
//					ss << iter->first << ", ";
//				}
//				ss << "]";
//				Debug::print("%s %s\n", getClassName(), ss.str().c_str());

	
				// NOTE: workers are responsible for notifying all the masters it knows about.
				// if all workers are done, or in error state, then this is done too.
				if (activeWorkers.empty()) {
					status = Communicator_I::DONE;  // nothing to send to workers. since they are all done.
					Debug::print("%s all workers DONE.  buffer has %d entries\n", getClassName(), buffer->getBufferSize());
					// all the workers are done
					buffer->stop();
//					test_input_status = DONE;

				}
				t2 = cciutils::event::timestampInUS();
//				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

				return status;  // worker is in done or error state, don't send.

			} else {
				activeWorkers[worker_id] = worker_status;

				if (worker_status == Communicator_I::WAIT) { // worker status is wait, so manager doesn't do anything.
					Debug::print("%s Worker waiting.  why did it send the message in the first place?\n", getClassName());

					t2 = cciutils::event::timestampInUS();
//					if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker wait"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

					return worker_status;  // return status is worker wait, but keep manager at ready.
				}
			}

			// READY to receive.



			MPI_Send(&buffer_status, 1, MPI_INT, worker_id, CONTROL_TAG, comm);
			//Debug::print("buffer status sent \n", buffer_status);

			if (buffer_status == Communicator_I::DONE ) {  // if action is done or error,
				// then we are done.  need to notify everyone.
				// keep the commhandler status at READY.
				// and set message to all nodes
//				hascompletedworker = true;
//				action->debugOn();

				activeWorkers.erase(worker_id);
				if (activeWorkers.empty()) {// all messages sent
					status = Communicator_I::DONE;

					Debug::print("%s DONE\n", getClassName());
					// action is already done, so no need to change it.
				}
				t2 = cciutils::event::timestampInUS();
//				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("buffer done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

				return status;

			} else if (buffer_status == Communicator_I::READY) {
				// status is ready, receive data.

				++send_count;
				MPI_Recv(&count, 1, MPI_INT, worker_id, DATA_TAG, comm, &mstatus);
				data = malloc(count);
				MPI_Recv(data, count, MPI_CHAR, worker_id, DATA_TAG, comm, &mstatus);

//				if (count > 0) {
//					int *i2 = (int *)data;
//					Debug::print("%s requester recv %d at %x from %d\n", getClassName(), *i2, data, worker_id);
//				} else
//					Debug::print("%s requester recv ?? at %x from %d\n", getClassName(), data, worker_id);


				if (count > 0 && data != NULL) {
					int stat = buffer->push(std::make_pair(count, data));
					buffer_status = (stat == DataBuffer::STOP ? Communicator_I::DONE : (stat == DataBuffer::FULL ? Communicator_I::WAIT : Communicator_I::READY));
				} else {
					Debug::print("%s RECEIVED nothing!\n", getClassName());
				}

				t2 = cciutils::event::timestampInUS();
				sprintf(len, "%lu", (long)(count));
				if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("push data received"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));

				if (send_count % 100 == 0) Debug::print("%s manager received %d data messages from workers.\n", getClassName(), send_count);


				return buffer_status;
			} else {
//				Debug::print("%s waiting\n", getClassName());
			} // else wait.  so do nothing

			return status;
		} else {
			return Communicator_I::WAIT;
		}

	} else {


		if (buffer->isFinished()) buffer_status = Communicator_I::DONE;
		else if (buffer->isEmpty()) buffer_status = Communicator_I::WAIT;
		else buffer_status = Communicator_I::READY;
//		Debug::print("%s worker buffer_status is :%d\n", getClassName(), status);

		if (buffer_status == Communicator_I::WAIT) {
			return buffer_status;  // nothing to output, so don't call manager
		} else if (buffer_status == Communicator_I::DONE || buffer_status == Communicator_I::ERROR) {
			// notify all the roots
			std::vector<int> roots = scheduler->getRoots();

			for (std::vector<int>::iterator iter=roots.begin();
					iter != roots.end(); ++iter) {

//				Debug::print("%s root %d is notified as done.\n", getClassName(), *iter);

//				MPI_Isend(&buffer_status, 1, MPI_INT, *iter, CONTROL_TAG, comm, &myRequest);
				MPI_Send(&buffer_status, 1, MPI_INT, *iter, CONTROL_TAG, comm);
			}
			// TODO do waitall here.

			// and say done.
			status = Communicator_I::DONE;

			t2 = cciutils::event::timestampInUS();
			if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("worker done"), t1, t2, std::string(), ::cciutils::event::NETWORK_IO));

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

		DataBuffer::DataType dstruct;

		if (manager_status == Communicator_I::READY) {
			int stat = buffer->pop(dstruct);
			buffer_status = (stat == DataBuffer::EMPTY ?  Communicator_I::WAIT : Communicator_I::READY);
			if (buffer_status != Communicator_I::READY) Debug::print("%s ERROR: status has changed during a single invocation of run()\n", getClassName());

			MPI_Send(&(dstruct.first), 1, MPI_INT, root, DATA_TAG, comm);
			MPI_Send(dstruct.second, dstruct.first, MPI_CHAR, root, DATA_TAG, comm);
			if (dstruct.second != NULL) {
				free(dstruct.second);
				dstruct.second = NULL;

			}
//			Debug::print("%s %d sent data to %d\n", getClassName(), rank, root);


			t2 = cciutils::event::timestampInUS();
			sprintf(len, "%lu", (long)(count));
			if (this->logsession != NULL) logsession->log(cciutils::event(0, std::string("push data sent"), t1, t2, std::string(len), ::cciutils::event::NETWORK_IO));

		} else if (manager_status == Communicator_I::DONE || manager_status == Communicator_I::ERROR ) {
			// one manager is done.  remove it from the list.  if there is no roots left, done.
			if (scheduler->removeRoot(root) == 0)  {
				status = Communicator_I::DONE;
				// if manager can't accept, then action can stop
				buffer->stop();
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
