/*
 * Sender.h
 *
 *  Created on: May 25, 2012
 *      Author: tcpan
 */

#ifndef SENDER_H_
#define SENDER_H_

#include "mpi.h"

namespace cci {

namespace runtime {

/**
 * represents an activity to send a message.
 *
 */
class Sender {
public:
	Sender() {};
	virtual ~Sender() {};

	virtual int send() = 0;
};

}}

#endif /* SENDER_H_ */
