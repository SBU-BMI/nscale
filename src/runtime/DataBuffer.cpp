/*
 * DataBuffer.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#include "DataBuffer.h"
#include <cstdlib>
#include <cstdio>


namespace cci {
namespace rt {

const int DataBuffer::READY = 1;
const int DataBuffer::STOP = 0;
const int DataBuffer::EMPTY = 2;
const int DataBuffer::FULL = 3;
const int DataBuffer::BAD_DATA = -1;



DataBuffer::DataBuffer(int _capacity) : capacity(_capacity), status(DataBuffer::READY) {

}

DataBuffer::~DataBuffer() {
	if (!buffer.empty()) {
		printf("WARNING:  DataBuffer is not empty.  likely to have leaked memory.\n");
	}
}

void DataBuffer::dumpBuffer() {
	while (~buffer.empty()) {
		DataType d = buffer.front();
		buffer.pop();
		free(d.second);
	}
}


int DataBuffer::push(DataType const data) {
	if (isStopped()) return status;
	if (isFull()) return FULL;

	if (data.first == 0 || data.second == NULL) return BAD_DATA;

	if (this->canPush()) buffer.push(data);
	return status;  // should have value READY.
}

int DataBuffer::pop(DataType &data) {
	if (!canPop()) return EMPTY;

	data = buffer.front();
	buffer.pop();

	return status;
}


} /* namespace rt */
} /* namespace cci */
