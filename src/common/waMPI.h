/*
 * waMPI.h
 *
 * work arounds for MPI issue.  Specifically address are ordered iprobe and probe.
 * wa stands for work around.
 *
 * assumes RNG seed has already been set.
 *  Created on: Oct 18, 2012
 *      Author: tcpan
 */

#ifndef WAMPI_H_
#define WAMPI_H_

#include "mpi.h"
#include <cstdlib>

namespace cci {
namespace rt {
namespace mpi {

class waMPI {
public:
	waMPI(MPI_Comm const _comm) : comm(_comm) {
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		probeIdx = rand() % size;
	}
	virtual ~waMPI() {};

	/* work around for MPI iProbe's rank-ordered probing.  same semantics as standard MPI call.
	 * source can be MPI_ANY_SOURCE, for which the fairness logic kicks in.  else the call is passed
	 * through to MPI.
	 */
	bool iprobe(int source, int tag, MPI_Status *status) {
		int hasMessage;
		if (source == MPI_ANY_SOURCE) {
			// go once through all the nodes
			for (int i = 0; i < size; ++i) {
				MPI_Iprobe(probeIdx, tag, comm, &hasMessage, status);
				probeIdx = (probeIdx + 1) % size;
				if (hasMessage) return true;
			}
			return false;
		} else {
			MPI_Iprobe(source, tag, comm, &hasMessage, status);
			return hasMessage > 0;
		}
	};
	void probe(int source, int tag, MPI_Status *status) {
		if (source == MPI_ANY_SOURCE) {
			int hasMessage = 0;
			while (hasMessage == 0) {
				MPI_Iprobe(probeIdx, tag, comm, &hasMessage, status);
				probeIdx = (probeIdx + 1) % size;
			}
		} else {
			MPI_Probe(source, tag, comm, status);
		}
	};

private:
	int probeIdx;
	int size;
	int rank;
	MPI_Comm comm;
};

} /* namespace mpi */
} /* namespace rt */
} /* namespace cci */
#endif /* WAMPI_H_ */
