/*
 * DataBuffer.h
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#ifndef DATABUFFER_H_
#define DATABUFFER_H_

#include <utility>

namespace cci {
namespace rt {

class DataBuffer {
public:

	typedef std::pair<int, void *> DataType;

	DataBuffer() {};
	virtual ~DataBuffer() {};

	// for data addition
	int getBufferSize() = 0;

	bool isPaused() = 0;
	bool isFull() = 0;
	bool isFinished() = 0;
	bool isStopped() = 0;

	// finish makes the buffer stop accept data permanently.  Buffer continues to flush
	void finish() = 0;
	// forceStop makes the buffer stop accept data permanently, and discard everything that's in buffer and MPI buffer.
	void forceStop() = 0;
	// pause makes the buffer stop accept data temporarily
	void pause() = 0;
	void unpause() = 0;

	// add and remove data to buffer
	int addData(DataType const data) {
		int status = this->canAdd();
		if (status == READY && data.first > 0 && data.second != NULL) {
			buffer.push(data);
		}
		return buffer.size();
	};
	int getData(DataType data) {
		if (this->canGet()) {
			data = buffer.front();
			buffer.pop();
		} else {
			data = NULL;
		}
		return buffer.size();
	};
	bool canAdd() = 0;
	bool canGet() = 0;

protected:
	std::queue<DataType> buffer;
	int status;
	int capacity;

};

} /* namespace rt */
} /* namespace cci */
#endif /* DATABUFFER_H_ */
