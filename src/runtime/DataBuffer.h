/*
 * DataBuffer.h
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#ifndef DATABUFFER_H_
#define DATABUFFER_H_

#include <utility>
#include <queue>

namespace cci {
namespace rt {

class DataBuffer {
public:
	static const int READY;
	static const int STOP;
	static const int FULL;
	static const int EMPTY;
	static const int BAD_DATA;

	typedef std::pair<int, void *> DataType;

	DataBuffer(int _capacity);
	virtual ~DataBuffer();

	// for data addition
	virtual int getBufferSize() { return buffer.size(); };

	virtual bool isEmpty() { return getBufferSize() == 0; };
	virtual bool isFull() { return getBufferSize() >= capacity; };
	virtual bool isStopped() { return status == STOP; };
	// FINISHED - stopped and everything is consumed.
	virtual bool isFinished() { return isStopped() && isEmpty(); };

	// finish makes the buffer stop accept data permanently.  Buffer continues to flush
	virtual void stop() { status = STOP; };
	// forceStop makes the buffer stop accept data permanently, and discard everything that's in buffer and MPI buffer.
	virtual void kill() { status = STOP; dumpBuffer(); };

	virtual bool canPush() { return !isStopped() && !isFull(); };
	virtual bool canPop() { return !isEmpty(); };


	// add and remove data to buffer
	virtual int push(DataType const data);

	virtual int pop(DataType &data);

protected:
	std::queue<DataType> buffer;
	int status;
	int capacity;

	virtual void dumpBuffer();

};

} /* namespace rt */
} /* namespace cci */
#endif /* DATABUFFER_H_ */
