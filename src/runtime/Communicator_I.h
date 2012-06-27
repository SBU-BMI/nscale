/*
 * Communicator_I.h
 *
 *  Created on: Jun 13, 2012
 *      Author: tcpan
 */

#ifndef COMMUNICATOR_I_H_
#define COMMUNICATOR_I_H_

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <memory>

#include <tr1/unordered_set>

namespace cci {
namespace rt {

/**
 * handles the communication for 1 message exchange.
 * control messages are embedded and fixed for now
 * payload messages come out
 */
class Communicator_I {
public:
	Communicator_I(MPI_Comm const * _parent_comm, int const _gid);
	virtual ~Communicator_I();

	virtual char* getClassName() { return "Communicator_I"; };

	virtual int run() = 0;

	MPI_Comm * getComm() { return &comm; };

	static const int READY;
	static const int WAIT;
	static const int DONE;
	static const int ERROR;

	int reference(void *obj) {
		reference_sources.insert(obj);
		return reference_sources.size();
	};
	int dereference(void *obj) {
		reference_sources.erase(obj);
		return reference_sources.size();
	}

protected:
	MPI_Comm const *parent_comm;
	MPI_Comm comm;
	int const groupid;
	int rank;
	int size;
	int pcomm_rank;
	int pcomm_size;

	// some basic metadata tracking.
	long call_count;
	std::tr1::unordered_set<void *> reference_sources;
};



} /* namespace rt */
} /* namespace cci */
#endif /* COMMUNICATOR_I_H_ */
