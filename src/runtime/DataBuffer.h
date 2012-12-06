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
#include <tr1/unordered_set>
#include "Debug.h"
#include "Logger.h"

#include "boost/program_options.hpp"

namespace cci {
namespace rt {

class DataBuffer {
public:
	static const int READY;
	static const int STOP;
	static const int FULL;
	static const int EMPTY;
	static const int BAD_DATA;
	static const int UNSUPPORTED_OP;

	typedef std::pair<int, void *> DataType;

	static boost::program_options::options_description params;
	static const std::string PARAM_COMPRESSION;
	static const std::string PARAM_BUFFERSIZE;

	DataBuffer(int _capacity, bool _compression=false, cci::common::LogSession *_logsession = NULL) ;
	DataBuffer(boost::program_options::variables_map &_vm, cci::common::LogSession *_logsession = NULL);

	// for data addition
	virtual size_t debugBufferSize() { return buffer.size(); };

//	bool isEmpty() { return getBufferSize() == 0; };
	virtual bool isFull() { return buffer.size() >= capacity; };
	bool isStopped() { return status == STOP; };
	// FINISHED - stopped and everything is consumed.
	virtual bool isFinished() { return isStopped() && buffer.size() <= 0; };

	// finish makes the buffer stop accept data permanently.  Buffer continues to flush
	void stop() {
		status = STOP;
	};
//	// kill makes the buffer stop accept data permanently, and discard everything that's in buffer and MPI buffer.
//	void kill() {
//		status = STOP;
//		dumpBuffer();
//	};

	// can push only when the entire memory usage is under a threshold
	virtual bool canPush() { return !isStopped() && buffer.size() < capacity; };
	// can pop if there is something in the front of queue (i.e. in buffer, not counting mpi_buffer)
	virtual bool canPop() { return buffer.size() > 0; };


	// add and remove data to buffer
	virtual int push(DataType const data);

	virtual int pop(DataType &data);

	static int reference(DataBuffer* self, void *obj) {
			if (self == NULL) return -1;
			if (obj == NULL) return self->reference_sources.size();

			self->reference_sources.insert(obj);
			return self->reference_sources.size();
	};
	static int dereference(DataBuffer* self, void *obj) {
			if (self == NULL) return -1;
			if (obj == NULL) return self->reference_sources.size();

			self->reference_sources.erase(obj);
			int result = self->reference_sources.size();
			if (result == 0) {
					delete self;
			}
			return result;
	};

	virtual ~DataBuffer();

protected:
	std::queue<DataType> buffer;
	int status;
	int capacity;
	bool compression;

	cci::common::LogSession *logsession;

	std::tr1::unordered_set<void *> reference_sources;

private:
	static bool param_init;
	static bool initParams();
};

} /* namespace rt */
} /* namespace cci */
#endif /* DATABUFFER_H_ */
