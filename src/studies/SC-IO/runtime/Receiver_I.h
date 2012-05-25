/*
 * Receiver.h
 *
 *  Created on: May 25, 2012
 *      Author: tcpan
 */

#ifndef RECEIVER_H_
#define RECEIVER_H_

#include "mpi.h"

namespace cci {

namespace runtime {

/**
 * represents an activity to listen for a message.
 *
 */
class Receiver {
public:
	Receiver() {};
	virtual ~Receiver() {};

	virtual int receive() = 0;
};

}}
#endif /* RECEIVER_H_ */
