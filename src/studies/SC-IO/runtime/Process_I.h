/*
 * Process.h
 *
 *  Created on: Apr 9, 2012
 *      Author: tcpan
 */

#ifndef PROCESS_H_
#define PROCESS_H_

#include "mpi.h"
#include "Sender.h"
#include "Receiver.h"
#include "Worker.h"

namespace cci {

namespace runtime {

/**
 * represents a process.
 *
 * we assume this is a hierarchical organization, and this is using MPI
 *
 * purpose of this is to manage the MPI node setup, and also manage the control messages.
 *
 * this is the super class for any specialization.
 *
 */
class Process {
public:
	Process() {};
	virtual ~Process() {};

	virtual int setup() = 0;
	virtual int teardown() = 0;

	/**
	 * runs and messages between nodes.  called from main.
	 */
	virtual int run() = 0;

private:
	// different communicator.  note that this process should be part of all comms
	// world_comm is for everyone
	// parent_comm has the parent(s) and all their children
	// sibling is all the sibling.  this is same as parent comm minus the parents
	// chilren comm is self + all children.  this is the same as the chilren's parent comm.
	// node_comm is a communicator for the current node.  (need this?)
	MPI_Comm world_comm, parent_comm, sibling_comm, children_comm, node_comm;
	// these rank information that this node can't get from MPI_Comm_rank.
	int parent_rank;

	// process has the following objects.  has_a relationship allows easy reconfiguration.
	// they may be null.
	Sender *s;
	Receiver *r;
	Worker *w;

	// any queuing is done by subclasses.

};

}

}

#endif /* PROCESS_H_ */
