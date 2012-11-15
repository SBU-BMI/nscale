/*
 * SegmentCmdParser.cpp
 *
 *  Created on: Jul 12, 2012
 *      Author: tcpan
 */

#include "SegmentCmdParser.h"
#include <iostream>
#include <sstream>

#include "Debug.h"
#include "FileUtils.h"

namespace cci {
namespace rt {



namespace adios {

const int SegmentCmdParser::PARAM_EXECUTABLEDIR = 0;
const int SegmentCmdParser::PARAM_INPUT = 1;
const int SegmentCmdParser::PARAM_OUTPUTDIR = 2;
const int SegmentCmdParser::PARAM_INPUTCOUNT = 3;
const int SegmentCmdParser::PARAM_PROCTYPE = 4;
const int SegmentCmdParser::PARAM_GPUDEVICEID = 5;
const int SegmentCmdParser::PARAM_TRANSPORT = 6;
const int SegmentCmdParser::PARAM_IOSIZE = 7;
const int SegmentCmdParser::PARAM_IOINTERLEAVE = 8;
const int SegmentCmdParser::PARAM_SUBIOSIZE = 9;
const int SegmentCmdParser::PARAM_SUBIOINTERLEAVE = 10;
const int SegmentCmdParser::PARAM_BENCHMARK = 11;
const int SegmentCmdParser::PARAM_IOBUFFERSIZE = 12;
const int SegmentCmdParser::PARAM_COMPRESSION = 13;
const int SegmentCmdParser::PARAM_NONBLOCKING = 14;




bool SegmentCmdParser::parse(int argc, char** argv) {

	if (argc < 3) {
		printUsage(argv[0]);
		return false;
	}

	std::string executable(argv[0]);
	std::string indir(argv[1]);
	std::string outdir(argv[2]);

	params[SegmentCmdParser::PARAM_EXECUTABLEDIR] = FileUtils::getDir(executable);
	params[SegmentCmdParser::PARAM_INPUT] = indir;
	params[SegmentCmdParser::PARAM_OUTPUTDIR] = outdir;

	// put in the defaults
	params[SegmentCmdParser::PARAM_INPUTCOUNT] = "1";
	params[SegmentCmdParser::PARAM_TRANSPORT] = "na-NULL";
	params[SegmentCmdParser::PARAM_IOSIZE] = "-1";
	params[SegmentCmdParser::PARAM_IOINTERLEAVE] = "4";
	params[SegmentCmdParser::PARAM_SUBIOSIZE] = "-1";
	params[SegmentCmdParser::PARAM_SUBIOINTERLEAVE] = "1";
	params[SegmentCmdParser::PARAM_BENCHMARK] = "0";
	params[SegmentCmdParser::PARAM_PROCTYPE] = "cpu";
	params[SegmentCmdParser::PARAM_GPUDEVICEID] = "0";
	params[SegmentCmdParser::PARAM_IOBUFFERSIZE] = "7";
	params[SegmentCmdParser::PARAM_COMPRESSION] = "off";
	params[SegmentCmdParser::PARAM_NONBLOCKING] = "off";

	int pos = 3;

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_INPUTCOUNT] = argv[pos];
		++pos;
	} else return true;

	if (argc > pos) {
		if (strcmp(argv[pos], "cpu") != 0 &&
			strcmp(argv[pos], "gpu") != 0) {
			printUsage(argv[0]);
			return false;
		}
 		params[SegmentCmdParser::PARAM_PROCTYPE] = argv[pos];
		if (strcmp(argv[pos], "gpu") == 0) {
			++pos;
			params[SegmentCmdParser::PARAM_GPUDEVICEID] = argv[pos];
		}

		++pos;
	} else return true;

	if (argc > pos) {
		if(	strcmp(argv[pos], "na-NULL") != 0 &&
			strcmp(argv[pos], "na-POSIX") != 0 &&
			strcmp(argv[pos], "NULL") != 0 &&
			strcmp(argv[pos], "POSIX") != 0 &&
			strcmp(argv[pos], "MPI") != 0 &&
			strcmp(argv[pos], "MPI_LUSTRE") != 0 &&
			strcmp(argv[pos], "MPI_AMR") != 0 &&
			strcmp(argv[pos], "gap-NULL") != 0 &&
			strcmp(argv[pos], "gap-POSIX") != 0 &&
			strcmp(argv[pos], "gap-MPI") != 0 &&
			strcmp(argv[pos], "gap-MPI_LUSTRE") != 0 &&
			strcmp(argv[pos], "gap-MPI_AMR") != 0) {
			printUsage(argv[0]);
			return false;
		}
		params[SegmentCmdParser::PARAM_TRANSPORT] = argv[pos];
		++pos;
	} else {
		return true;
	}

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_IOBUFFERSIZE] = argv[pos];
		++pos;
	} else return true;

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_IOSIZE] = argv[pos];
		++pos;
	} else return true;

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_IOINTERLEAVE] = argv[pos];
		++pos;
	} else return true;

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_SUBIOSIZE] = argv[pos];
		++pos;
	} else return true;

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_SUBIOINTERLEAVE] = argv[pos];
		++pos;
	} else return true;

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_COMPRESSION] = argv[pos];
		++pos;
	}

	if (argc > pos) {
		params[SegmentCmdParser::PARAM_NONBLOCKING] = argv[pos];
		++pos;
	}

	return true;
}

void SegmentCmdParser::printUsage(char *cmd) {
	int rank;
	MPI_Comm_rank(comm, &rank);

	if (rank == 0) {

		std::stringstream ss;
		ss << "Usage:  " << cmd << " <input_filename | input_dir> output_dir imagecount [cpu | gpu id] [tranport] [bufferSize] [IOSize] [IOInterleave] [subIOSize] [subIOInterleave] [compression] [nonblocking]" << std::endl;
		ss << "\t <input_filename | input_dir>: required. either an image filename or an image directory" << std::endl;
		ss << "\t output_dir: required. output directory" << std::endl;
		ss << "\t imagecount: required. number of images to process. no default." << std::endl;
		ss << "\t [cpu | gpu id]: optional. chooses CPU or GPU computation. CPU is single core." << std::endl;
		ss << "\t\t If GPU, specify which device to use." << std::endl;
		ss << "\t [transport]: optional. one of na-NULL | na-POSIX | NULL | POSIX | MPI | MPI_LUSTRE | MPI_AMR " << std::endl;
		ss << "\t\t | gap-NULL | gap-POSIX | gap-MPI | gap-MPI_LUSTRE | gap-MPI_AMR.  default is na-NULL.  na- prefix means non-adios.  gap- prefix means no compacting in the output space." << std::endl;
		ss << "\t [buffersize]: optional. number of tiles an IO node buffers. default = 7 " << std::endl;
		ss << "\t [IOsize], [IOInterleave]: clamp to [1, size]. optional. Determine the size of IO group and how they mix." << std::endl;
		ss << "\t\t IOInterleave <= 1: IO group size is specified by IOsize. IO procs are contiguous" << std::endl;
		ss << "\t\t IOInterleave > 1:  IO processes are identified by rank%IOInterleave, up to IOsize. " << std::endl;
		ss << "\t\t\t e.g. io c c c io c c c.  Default is IOInterleave=4, IOsize=size/4." << std::endl;
		ss << "\t [subIOsize] [subIOInterleave]: optional. how the IO subgroup sizes and how they mix together." << std::endl;
		ss << "\t\t subIOSize is clamped to [1, ioSize].  subIOInterleave is clamped to [1,...)." << std::endl;
		ss << "\t\t IO group is split into g groups of max size subIOSize.  subIOInterleave number of subgroups interleave together." << std::endl;
		ss << "\t\t\t e.g. 1 2 3 1 2 3. Default is subIOInterleave=1, subIOsize=IOsize (1 group)" << std::endl;
		ss << "\t [compression] = on|off: optional. turn on compression for MPI messages and IO. default off." << std::endl;
		ss << "\t [nonblocking]: optional. choosing to use blocking or non-blocking MPI.  default off " << std::endl;
	//	ss << "\tstages: the stages to capture.  syntax is a comma separated ranges.  Range could be a single value or a dash (-) separated range.  range is of form [...) " << std::endl;

		printf("%s\n", ss.str().c_str());
		ss.str(std::string());
	}
}

} /* namespace adios */
} /* namespace rt */
} /* namespace cci */
