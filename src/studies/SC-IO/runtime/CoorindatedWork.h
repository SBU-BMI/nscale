/*
 * CoorindatedWork.h
 *
 *  Created on: Apr 10, 2012
 *      Author: tcpan
 */

#ifndef COORINDATEDWORK_H_
#define COORINDATEDWORK_H_

#include "Work.h"

namespace cci {

namespace runtime {

class CoorindatedWork: public cci::runtime::Work {
public:
	CoorindatedWork(MPI_Comm *_comm, int _rank) :
		local_comm(_comm), local_rank(_rank) {};
	virtual ~CoorindatedWork() {};

private:
	MPI_Comm *local_comm;
	int local_rank;

};

}

}

#endif /* COORINDATEDWORK_H_ */
