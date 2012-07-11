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

#include "SCIOUtilsLogger.h"

namespace cci {
namespace rt {

/**
 * handles the communication for 1 message exchange.
 * control messages are embedded and fixed for now
 * payload messages come out
 */
class Communicator_I {
public:
	Communicator_I(MPI_Comm const * _parent_comm, int const _gid, cciutils::SCIOLogSession *_logger = NULL);

	virtual const char* getClassName() { return "Communicator_I"; };

	virtual int run() = 0;

	MPI_Comm * getComm() { return &comm; };

	static const int READY;
	static const int WAIT;
	static const int DONE;
	static const int ERROR;

	static int reference(Communicator_I* self, void *obj) {
		if (self == NULL) return -1;
		if (obj == NULL) return self->reference_sources.size();

		self->reference_sources.insert(obj);
		return self->reference_sources.size();
	};
	static int dereference(Communicator_I* self, void *obj) {
		if (self == NULL) return -1;
		if (obj == NULL) return self->reference_sources.size();

		self->reference_sources.erase(obj);
		int result = self->reference_sources.size();
		if (result == 0) {
			delete self;
			return result;
		}
		else return 0;
	};

protected:
	MPI_Comm const *parent_comm;
	MPI_Comm comm;
	int const groupid;
	int rank;
	int size;
	int pcomm_rank;
	int pcomm_size;
	char hostname[256];

	// some basic metadata tracking.
	long call_count;
	std::tr1::unordered_set<void *> reference_sources;

	virtual ~Communicator_I();

	cciutils::SCIOLogSession *logger;

};



} /* namespace rt */
} /* namespace cci */
#endif /* COMMUNICATOR_I_H_ */
