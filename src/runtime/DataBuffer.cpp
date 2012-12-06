/*
 * DataBuffer.cpp
 *
 *  Created on: Aug 14, 2012
 *      Author: tcpan
 */

#include "DataBuffer.h"
#include <cstdlib>
#include <cstdio>
#include "CmdlineParser.h"

namespace cci {
namespace rt {

const int DataBuffer::READY = 11;
const int DataBuffer::STOP = 10;
const int DataBuffer::EMPTY = 12;
const int DataBuffer::FULL = 13;
const int DataBuffer::BAD_DATA = -11;
const int DataBuffer::UNSUPPORTED_OP = -12;

const std::string DataBuffer::PARAM_COMPRESSION = "compression";
const std::string DataBuffer::PARAM_BUFFERSIZE = "buffer_size";

bool DataBuffer::initParams() {
	DataBuffer::params.add_options()
		("buffer_size,b", boost::program_options::value<int>()->default_value(4), "buffer size in number of tiles.")
		("compression,c", boost::program_options::value<bool>()->default_value(false)->implicit_value(true), "MPI message compression on/off.")
		;
	return true;
}
boost::program_options::options_description DataBuffer::params("Buffer Options");
bool DataBuffer::param_init = DataBuffer::initParams();


DataBuffer::DataBuffer(boost::program_options::variables_map &_vm, cci::common::LogSession *_logsession) :
		status(DataBuffer::READY), logsession(_logsession) {
	while (!buffer.empty()) {
		DataType d = buffer.front();
		buffer.pop();
		free(d.second);
		d.second = NULL;
	}
	reference_sources.clear();

	capacity = cci::rt::CmdlineParser::getParamValueByName<int>(_vm, "buffer_size");
	compression = cci::rt::CmdlineParser::getParamValueByName<bool>(_vm, "compression");
}

DataBuffer::DataBuffer(int _capacity, bool _compression, cci::common::LogSession *_logsession) :
		capacity(_capacity), status(DataBuffer::READY), logsession(_logsession), compression(_compression) {
	while (!buffer.empty()) {
		DataType d = buffer.front();
		buffer.pop();
		free(d.second);
		d.second = NULL;
	}
	reference_sources.clear();
}


DataBuffer::~DataBuffer() {
	if (!buffer.empty()) {
		cci::common::Debug::print("WARNING: DataBuffer has %d entries left in buffer.  likely to have leaked memory.\n", buffer.size());
	}
	while (!buffer.empty()) {
		DataType d = buffer.front();
		buffer.pop();
		free(d.second);
		d.second = NULL;
	}
}


int DataBuffer::push(DataType const data) {

	if (isStopped()) return STOP;
	if (isFull()) return FULL;
	if (data.first == 0 || data.second == NULL) return BAD_DATA;

	if (this->canPush()) buffer.push(data);

	//cci::common::Debug::print("DataBuffer: push called.  %d load\n", buffer.size());

	return READY;  // should have value READY.
}

int DataBuffer::pop(DataType &data) {
	data.second = NULL;
	data.first = 0;
	if (isFinished()) return STOP;
	if (!canPop()) return EMPTY;

	data = buffer.front();
	buffer.pop();

	//cci::common::Debug::print("DataBuffer: pop called.  %d load\n", buffer.size());

	return READY;
}


} /* namespace rt */
} /* namespace cci */
