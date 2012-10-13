/*
 * SegmentCmdParser.h
 *
 *  Created on: Jul 12, 2012
 *      Author: tcpan
 */

#ifndef SEGMENTCMDPARSER_H_
#define SEGMENTCMDPARSER_H_

#include "CmdlineParser_I.h"
#include "mpi.h"

namespace cci {
namespace rt {
namespace adios {


class SegmentCmdParser : public cci::rt::CmdlineParser_I<14> {
public:
	SegmentCmdParser(MPI_Comm &_comm) : comm(_comm) {};
	virtual ~SegmentCmdParser() {};

	virtual bool parse(int argc, char** argv);
	virtual void printUsage(char *cmd);

	static const int PARAM_EXECUTABLEDIR;
	static const int PARAM_INPUT;
	static const int PARAM_OUTPUTDIR;
	static const int PARAM_INPUTCOUNT;
	static const int PARAM_TRANSPORT;
	static const int PARAM_IOSIZE;
	static const int PARAM_IOINTERLEAVE;
	static const int PARAM_SUBIOSIZE;
	static const int PARAM_SUBIOINTERLEAVE;
	static const int PARAM_BENCHMARK;
	static const int PARAM_PROCTYPE;
	static const int PARAM_GPUDEVICEID;
	static const int PARAM_IOBUFFERSIZE;
	static const int PARAM_COMPRESSION;


protected:
	MPI_Comm comm;
};

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
#endif /* SEGMENTCMDPARSER_H_ */
