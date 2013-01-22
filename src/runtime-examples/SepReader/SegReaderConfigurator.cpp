/*
 * SegReaderConfigurator.cpp
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#include "SegReaderConfigurator.h"

#include "PullCommHandler.h"
#include "PushCommHandler.h"

#include "ADIOSSave_Reduce.h"
#include "POSIXRawSave.h"

#include "ReadTiles.h"
#include "SegmentNoRead.h"
#include "SynthSegmentNoRead.h"
#include "RandomScheduler.h"
#include "RoundRobinScheduler.h"
#include "UtilsADIOS.h"
#include "Logger.h"
#include "FileUtils.h"
#include "NullSinkAction.h"
#include "MPISendDataBuffer.h"
#include "MPIRecvDataBuffer.h"

namespace cci {
namespace rt {
namespace adios {

const int SegReaderConfigurator::UNDEFINED_GROUP = MPI_UNDEFINED;
const int SegReaderConfigurator::COMPUTE_GROUP = 1;
const int SegReaderConfigurator::READ_GROUP = 2;
const int SegReaderConfigurator::WRITE_GROUP = 3;
const int SegReaderConfigurator::READ_TO_COMPUTE_GROUP = 4;
const int SegReaderConfigurator::COMPUTE_TO_WRITE_GROUP = 5;
const int SegReaderConfigurator::UNUSED_GROUP = 0;


SegReaderConfigurator::SegReaderConfigurator(int argc, char** argv) :
	ProcessConfigurator_I(), iomanager(NULL) {

	long long t1, t2;
	t1 = cci::common::event::timestampInUS();

	///////
		// Boost library program_options package:  for parsing
	CmdlineParser *parser = new CmdlineParser();

	parser->addParams(DataBuffer::params);
	parser->addParams(MPIDataBuffer::params);
	parser->addParams(cci::rt::adios::ReadTiles::params);
	parser->addParams(cci::rt::adios::SegmentNoRead::params);
	parser->addParams(cci::rt::adios::SynthSegmentNoRead::params);
	boost::program_options::options_description paramslocal;
	paramslocal.add_options()
		("synth_compute,S", boost::program_options::value<bool>()->default_value(false)->implicit_value(true), "Synthetic Computation on/off.")
				;
	parser->addParams(paramslocal);


	bool p_result = parser->parse(argc, argv);
	if (!p_result) {
		exit(-1);
	}
	params = parser->getParamValues();

	delete parser;


	// set up the logger
	char hostname[256];
	gethostname(hostname, 255);  // from <iostream>

		// end Boost Program_Options configuration.
	//////////
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (params.count(cci::rt::CmdlineParser::PARAM_LOG)) {

		logger = new cci::common::Logger(cci::rt::CmdlineParser::getParamValueByName<std::string>(params, cci::rt::CmdlineParser::PARAM_LOG), rank, hostname, 0);
	} else {
		logger = NULL;
	}

	if (rank == 0) {
		std::cout << "Recongized Options:" << std::endl;
		std::cout << "\t" << DataBuffer::PARAM_COMPRESSION << ":\t" << CmdlineParser::getParamValueByName<bool>(params, DataBuffer::PARAM_COMPRESSION) << std::endl;
		std::cout << "\t" << DataBuffer::PARAM_BUFFERSIZE <<  ":\t" << CmdlineParser::getParamValueByName<int>(params, DataBuffer::PARAM_BUFFERSIZE) << std::endl;
		std::cout << "\t" << MPIDataBuffer::PARAM_NONBLOCKING << ":\t" << CmdlineParser::getParamValueByName<bool>(params, "nonblocking") << std::endl;
		std::cout << "\t" << "input_directory" << ":\t" << CmdlineParser::getParamValueByName<std::string>(params, "input_directory") << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_INPUTCOUNT << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_INPUTCOUNT) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_OUTPUTDIR << ":\t" << CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_OUTPUTDIR) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOTRANSPORT << ":\t" << CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_IOTRANSPORT) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOSIZE) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOGROUPSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOGROUPSIZE) << std::endl;
		std::cout << "\t" << ReadTiles::PARAM_READSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, ReadTiles::PARAM_READSIZE) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_MAXIMGSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_MAXIMGSIZE) << std::endl;
		bool test = CmdlineParser::getParamValueByName<bool>(params, "synth_compute");
		std::cout << "\t" << "synth_compute" << ":\t" << (test ? "true" : "false") << std::endl;
		if (logger != NULL) std::cout << "\t" << CmdlineParser::PARAM_LOG << ":\t" << CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_LOG) << std::endl;
	}

	executable = argv[0];

	t2 = cci::common::event::timestampInUS();
	cci::common::LogSession *logsession = NULL;
	logsession = (logger == NULL ? NULL : logger->getSession("setup"));
	if (logsession != NULL) logsession->log(cci::common::event(0, std::string("parse cmd"), t1, t2, std::string(), ::cci::common::event::OTHER));
};


bool SegReaderConfigurator::init() {

	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM

	// need to initialize ADIOS by everyone because it has HARDCODED MPI_COMM_WORLD instead of taking a parameter for the comm.

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank = -1;
	MPI_Comm_rank(comm, &rank);

	cci::common::LogSession *session = (logger == NULL ? NULL : logger->getSession("setup"));

	// get the configuration file
	std::string iocode = CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_IOTRANSPORT);

	if (strncmp(iocode.c_str(), "na-", 3) != 0) {
		std::string adios_config(cci::common::FileUtils::getDir(executable));
		adios_config.append("/../adios_xml/image-tiles-globalarray-");
		adios_config.append(iocode);
		adios_config.append(".xml");

		//cci::common::Debug::print("iomanager created for %s using config %s\n", iocode.c_str(), adios_config.c_str());
		iomanager = new cci::rt::adios::ADIOSManager(adios_config.c_str(), params, rank, comm, session);
	}

	return true;
}

bool SegReaderConfigurator::finalize() {
	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM
	if (iomanager != NULL) {
		delete iomanager;
		iomanager = NULL;
	}

	return true;
}

bool SegReaderConfigurator::configure(MPI_Comm &comm, Process *proc) {

	long long t1, t2;
	t1 = cci::common::event::timestampInUS();

	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	if (size < 3) {
		printf("ERROR:  needs at least 3 mpi processes for segmentation: 1 read, 1 compute, 1 save\n");
		return false;
	}


	// now set up/

	// create the output directory
	if (rank == 0) {
		// create the directory
		cci::common::FileUtils::mkdirs(CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_OUTPUTDIR));
		printf("made directories for %s\n", CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_OUTPUTDIR).c_str());
	}

	///// first set up the comm handlers
//	cci::common::Debug::print("here1\n");

	CommHandler_I *r2c_handler, *c2w_handler;
	MPISendDataBuffer *sbuf = NULL;
	MPIRecvDataBuffer *rbuf = NULL;

	int buffersize = CmdlineParser::getParamValueByName<int>(params, DataBuffer::PARAM_BUFFERSIZE);
	int iosize = CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOSIZE);
	int readsize = CmdlineParser::getParamValueByName<int>(params, ReadTiles::PARAM_READSIZE);

	// first split into 3.
	int type_g=-1, io_sub_g=-1, compute_to_write_g = -1, read_to_compute_g = -1;
	type_g = (rank < readsize ? READ_GROUP : (rank < readsize+iosize ? WRITE_GROUP : COMPUTE_GROUP));
	MPI_Comm gcomm;
	MPI_Comm_split(comm, type_g, rank, &gcomm);

	// then create the communicating groups
	read_to_compute_g = (type_g == READ_GROUP || type_g == COMPUTE_GROUP ? READ_TO_COMPUTE_GROUP : UNUSED_GROUP);
	cci::common::LogSession *logsession = (logger == NULL ? NULL : logger->getSession("pull"));
	Scheduler_I *r2c_sch = new RandomScheduler(type_g == READ_GROUP, type_g == COMPUTE_GROUP);
	if (type_g == READ_GROUP) {
		sbuf = new MPISendDataBuffer(buffersize,
				CmdlineParser::getParamValueByName<bool>(params, MPIDataBuffer::PARAM_NONBLOCKING),
				logsession);
 		r2c_handler = new PullCommHandler(&comm, read_to_compute_g, sbuf, r2c_sch, logsession);
 	} else if (type_g == COMPUTE_GROUP) {
		rbuf = new MPIRecvDataBuffer(buffersize,
				CmdlineParser::getParamValueByName<bool>(params, MPIDataBuffer::PARAM_NONBLOCKING),
				logsession);
 		r2c_handler = new PullCommHandler(&comm, read_to_compute_g, rbuf, r2c_sch, logsession);
 	} else {
 		r2c_handler = new PullCommHandler(&comm, read_to_compute_g, NULL, r2c_sch, logsession);
 		delete r2c_handler;
 		r2c_handler = NULL;
 	}

	// now the compute to write group
	compute_to_write_g = (type_g == WRITE_GROUP || type_g == COMPUTE_GROUP ? COMPUTE_TO_WRITE_GROUP : UNUSED_GROUP);
	logsession = (logger == NULL ? NULL : logger->getSession("push"));
	Scheduler_I *c2w_sch = new RandomScheduler(type_g == WRITE_GROUP, type_g == COMPUTE_GROUP);
	if (type_g == COMPUTE_GROUP) {
		sbuf = new MPISendDataBuffer(params, logsession);
 		c2w_handler = new PushCommHandler(&comm, compute_to_write_g, sbuf, c2w_sch, logsession);
 	} else if (type_g == WRITE_GROUP) {
		rbuf = new MPIRecvDataBuffer(params, logsession);
 		c2w_handler = new PushCommHandler(&comm, compute_to_write_g, rbuf, c2w_sch, logsession);
 	} else {
 		c2w_handler = new PushCommHandler(&comm, compute_to_write_g, NULL, c2w_sch, logsession);
 		delete c2w_handler;
 		c2w_handler = NULL;
 	}


	t2 = cci::common::event::timestampInUS();
	if (this->logger != NULL) logger->getSession("setup")->log(cci::common::event(0, std::string("layout comms"), t1, t2, std::string(), ::cci::common::event::MEM_IO));

	// now set up the workers
	if (type_g == READ_GROUP) {
		Action_I *assign =
				new cci::rt::adios::ReadTiles(&gcomm, 1, NULL, sbuf,
						params,
						(logger == NULL ? NULL : logger->getSession("read")));
		proc->addHandler(assign);
		proc->addHandler(r2c_handler);

	} else if (type_g == COMPUTE_GROUP) {

		proc->addHandler(r2c_handler);
		bool synthcompute = CmdlineParser::getParamValueByName<bool>(params, "synth_compute");
		Action_I *seg;
		if (synthcompute) {
			seg =
					new cci::rt::adios::SynthSegmentNoRead(&comm, MPI_UNDEFINED, rbuf, sbuf,
							params,
							(logger == NULL ? NULL : logger->getSession("seg")));

		} else {
			seg =
				new cci::rt::adios::SegmentNoRead(&comm, MPI_UNDEFINED, rbuf, sbuf,
						params,
						(logger == NULL ? NULL : logger->getSession("seg")));
		}
		proc->addHandler(seg);
		proc->addHandler(c2w_handler);

	} else	{
//		cci::common::Debug::print("here5.3\n");

		t1 = cci::common::event::timestampInUS();
		int comm1_size;
		int comm1_rank;
		MPI_Comm_size(gcomm, &comm1_size);
		MPI_Comm_rank(gcomm, &comm1_rank);

		// then within IO group, split to subgroups, for adios.
		int subio_size = CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOGROUPSIZE);

		if (subio_size < 1) subio_size = comm1_size;
		else if (subio_size > comm1_size) subio_size = comm1_size;

//		cci::common::Debug::print("here5.3.1\n");

		if (subio_size == 1) {
			io_sub_g = comm1_rank;   // each in own group.  group id = rank in io group
		} else if (subio_size >= comm1_size) {
			io_sub_g = 0;  // everyone in same group.  group id = 0 for everyone.
		} else {
			io_sub_g = comm1_rank / subio_size;  // subgroups, with interleave of 1, so
		}
//		cci::common::Debug::print("here5.3.2\n");

		// io subgroups
		std::string iocode = CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_IOTRANSPORT);

		t2 = cci::common::event::timestampInUS();
		if (this->logger != NULL) logger->getSession("setup")->log(cci::common::event(0, std::string("layout adios"), t1, t2, std::string(), ::cci::common::event::MEM_IO));

//		cci::common::Debug::print("here5.3.3\n");

		Action_I *save;
		logsession = (logger == NULL ? NULL : logger->getSession("io"));
		if (strcmp(iocode.c_str(), "na-NULL") == 0) {
///			cci::common::Debug::print("here5.4a\n");

			save = new cci::rt::NullSinkAction(&gcomm, io_sub_g,
					rbuf, NULL,
					logsession);
		} else if (strcmp(iocode.c_str(), "na-POSIX") == 0) {
//			cci::common::Debug::print("here5.4b\n");

			save = new cci::rt::adios::POSIXRawSave(&gcomm, io_sub_g,
					rbuf, NULL,
				params,
				logsession);  // comm is group 1 IO comms, split into io_sub_g comms
		} else {
//			cci::common::Debug::print("here5.4c\n");
			save = new cci::rt::adios::ADIOSSave_Reduce(&gcomm, io_sub_g,
					rbuf, NULL,
				params,
				CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_MAXIMGSIZE) *
				CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_MAXIMGSIZE) * 4, 256, 1024,
				iomanager, logsession);  // comm is group 1 IO comms, split into io_sub_g comms
		}
		proc->addHandler(c2w_handler);
		proc->addHandler(save);
	}
//	MPI_Barrier(comm);

//	std::ostream_iterator<int> osi(std::cout, ", ");
//	std::vector<int> roots;
//	std::vector<int> leaves;
//
//	roots = sch->getRoots();
//	std::cout << "io or compute scheduler - " << rank << " (" << (compute_io_g == COMPUTE_GROUP ? "cp" : "io") << ") roots: ";
//	std::copy(roots.begin(), roots.end(), osi);
//	std::cout << std::endl;
//	leaves = sch->getLeaves();
//	std::cout << "io or compute scheduler - " << rank << " (" << (compute_io_g == COMPUTE_GROUP ? "cp" : "io") << ") leaves: ";
//	std::copy(leaves.begin(), leaves.end(), osi);
//	std::cout << std::endl;
//
//	roots = sch2->getRoots();
//	std::cout << "compute to IO scheduler - " << rank << " (" << (compute_to_io_g == COMPUTE_TO_IO_GROUP ? "c2io" : "unknown") << ") roots: ";
//	std::copy(roots.begin(), roots.end(), osi);
//	std::cout << std::endl;
//	leaves = sch2->getLeaves();
//	std::cout << "compute to IO scheduler - " << rank << " (" << (compute_to_io_g == COMPUTE_TO_IO_GROUP ? "c2io" : "unknown") << ") leaves: ";
//	std::copy(leaves.begin(), leaves.end(), osi);
//	std::cout << std::endl;

	return true;
}

}
} /* namespace rt */
} /* namespace cci */
