/*
 * MPIDataBuffer.cpp
 *
 *  Created on: Nov 28, 2012
 *      Author: tcpan
 */


#include "MPIDataBuffer.h"
#include "CmdlineParser.h"

namespace cci {
namespace rt {


const std::string MPIDataBuffer::PARAM_NONBLOCKING = "nonblocking";
bool MPIDataBuffer::initParams() {
	MPIDataBuffer::params.add_options()
		("nonblocking,l", boost::program_options::value<bool>()->default_value(true)->implicit_value(true), "MPI nonblocking transmission on/off.")
		;
	return true;
}
boost::program_options::options_description MPIDataBuffer::params("MPI Options");
bool MPIDataBuffer::param_init = MPIDataBuffer::initParams();

MPIDataBuffer::MPIDataBuffer(int _capacity, bool _compression, bool _non_blocking, cciutils::SCIOLogSession *_logsession)
	: DataBuffer(_capacity, _compression, _logsession), debug_complete_count(0), non_blocking(_non_blocking) {
	reqs = new MPI_Request[_capacity];
	reqptrs = new MPI_Request*[_capacity];
	completedreqs = new int[_capacity];

	if (!mpi_buffer.empty()) Debug::print("WARNING: constructing.  mpi_buffer is not empty.\n");
	mpi_buffer.clear();
	mpi_req_starttimes.clear();
}

MPIDataBuffer::MPIDataBuffer(boost::program_options::variables_map &_vm, cciutils::SCIOLogSession *_logsession)
	: DataBuffer(_vm, _logsession), debug_complete_count(0) {
	non_blocking = cci::rt::CmdlineParser::getParamValueByName<bool>(_vm, "nonblocking");

	reqs = new MPI_Request[capacity];
	reqptrs = new MPI_Request*[capacity];
	completedreqs = new int[capacity];

	if (!mpi_buffer.empty()) Debug::print("WARNING: constructing.  mpi_buffer is not empty.\n");
	mpi_buffer.clear();
	mpi_req_starttimes.clear();
}


}
}
