/*
 * SynDataConfiguratorPush.cpp
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#include "SynDataConfiguratorPush.h"

#include "PullCommHandler.h"
#include "PushCommHandler.h"

#include "ADIOSSave_Reduce.h"
#include "POSIXRawSave.h"

#include "GenerateOutputPush.h"
#include "RandomScheduler.h"
#include "RoundRobinScheduler.h"
#include "UtilsADIOS.h"
#include "SCIOUtilsLogger.h"
#include "FileUtils.h"
#include "NullSinkAction.h"
#include "MPISendDataBuffer.h"
#include "MPIRecvDataBuffer.h"

namespace cci {
namespace rt {
namespace syntest {

const int SynDataConfiguratorPush::UNDEFINED_GROUP = MPI_UNDEFINED;
const int SynDataConfiguratorPush::COMPUTE_GROUP = 1;
const int SynDataConfiguratorPush::IO_GROUP = 2;
const int SynDataConfiguratorPush::COMPUTE_TO_IO_GROUP = 3;
const int SynDataConfiguratorPush::UNUSED_GROUP = 0;

SynDataConfiguratorPush::SynDataConfiguratorPush(int argc, char** argv, cciutils::SCIOLogger *_logger) :
	ProcessConfigurator_I(_logger), iomanager(NULL) {

	long long t1, t2;
	t1 = cciutils::event::timestampInUS();

	///////
		// Boost library program_options package:  for parsing
	CmdlineParser *parser = new CmdlineParser();

	parser->addParams(DataBuffer::params);
	parser->addParams(MPIDataBuffer::params);

	bool p_result = parser->parse(argc, argv);
	if (!p_result) {
		exit(-1);
	}
	params = parser->getParamValues();

		// end Boost Program_Options configuration.
	//////////
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0) {
		std::cout << "Recongized Options:" << std::endl;
		std::cout << "\t" << DataBuffer::PARAM_COMPRESSION << ":\t" << CmdlineParser::getParamValueByName<bool>(params, DataBuffer::PARAM_COMPRESSION) << std::endl;
		std::cout << "\t" << DataBuffer::PARAM_BUFFERSIZE <<  ":\t" << CmdlineParser::getParamValueByName<int>(params, DataBuffer::PARAM_BUFFERSIZE) << std::endl;
		std::cout << "\t" << MPIDataBuffer::PARAM_NONBLOCKING << ":\t" << CmdlineParser::getParamValueByName<bool>(params, "nonblocking") << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_INPUTCOUNT << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_INPUTCOUNT) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_OUTPUTDIR << ":\t" << CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_OUTPUTDIR) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOTRANSPORT << ":\t" << CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_IOTRANSPORT) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOSIZE) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOINTERLEAVE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOINTERLEAVE) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOGROUPSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOGROUPSIZE) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_IOGROUPINTERLEAVE  << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOGROUPINTERLEAVE) << std::endl;
		std::cout << "\t" << CmdlineParser::PARAM_MAXIMGSIZE << ":\t" << CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_MAXIMGSIZE) << std::endl;
	}

	delete parser;

	executable = argv[0];

	t2 = cciutils::event::timestampInUS();
	cciutils::SCIOLogSession *logsession = logger->getSession("setup");
	if (logsession != NULL) logsession->log(cciutils::event(0, std::string("parse cmd"), t1, t2, std::string(), ::cciutils::event::OTHER));
};

bool SynDataConfiguratorPush::init() {

	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM

	// need to initialize ADIOS by everyone because it has HARDCODED MPI_COMM_WORLD instead of taking a parameter for the comm.

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank = -1;
	MPI_Comm_rank(comm, &rank);

	cciutils::SCIOLogSession *session = logger->getSession("setup");

	// get the configuration file
	std::string iocode = CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_IOTRANSPORT);

	if (strncmp(iocode.c_str(), "na-", 3) != 0) {
		std::string adios_config(FileUtils::getDir(executable));
		adios_config.append("/../adios_xml/image-tiles-globalarray-");
		adios_config.append(iocode);
		adios_config.append(".xml");

		//Debug::print("iomanager created for %s using config %s\n", iocode.c_str(), adios_config.c_str());
		iomanager = new cci::rt::adios::ADIOSManager(adios_config.c_str(), params, rank, comm, session);
	}

	return true;
}

bool SynDataConfiguratorPush::finalize() {
	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM
	if (iomanager != NULL) {
		delete iomanager;
		iomanager = NULL;
	}

	return true;
}

bool SynDataConfiguratorPush::configure(MPI_Comm &comm, Process *proc) {

	long long t1, t2;
	t1 = cciutils::event::timestampInUS();

	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	if (size < 3) {
		printf("ERROR:  needs at least 3 mpi processes for segmentation: 1 master, 1 compute, 1 save\n");
		return false;
	}


	// now set up/
	int compute_io_g=-1, io_sub_g=-1;

	// create the output directory
	if (rank == 0) {
		// create the directory
		FileUtils::mkdirs(CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_OUTPUTDIR));
		printf("made directories for %s\n", CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_OUTPUTDIR).c_str());
	}

	///// first set up the comm handlers


	CommHandler_I *handler, *handler2;
	MPISendDataBuffer *sbuf = NULL;
	MPIRecvDataBuffer *rbuf = NULL;

	int iointerleave = CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOINTERLEAVE);
	int iosize = CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOSIZE);
	if (iointerleave < 1) iointerleave = 1;
	else if (iointerleave > size) iointerleave = size;
	if (iosize < 1 || iosize > size) iosize = size/iointerleave;  // default


	// first split into 2.  focus on compute group.
	if (iointerleave == 1) {
		compute_io_g = (rank < iosize ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
	} else {
		compute_io_g = ((rank % iointerleave == 0) && (rank < iointerleave * iosize) ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
	}

	// compute and io groups
	Scheduler_I *sch = new RandomScheduler(false,false);
	handler = new PullCommHandler(&comm, compute_io_g, NULL, sch, logger->getSession("iogroup"));

	Scheduler_I *sch2 = NULL;
		if (compute_io_g == IO_GROUP) {
			rbuf = new MPIRecvDataBuffer(params, logger->getSession("push"));
			sch2 = new RandomScheduler(true, false);  // all io nodes are roots.
			handler2 = new PushCommHandler(&comm, 1, rbuf, sch2, logger->getSession("push"));
		} else {
			sbuf = new MPISendDataBuffer(params, logger->getSession("push"));
			sch2 = new RandomScheduler(false, true);
			handler2 = new PushCommHandler(&comm, 1, sbuf, sch2, logger->getSession("push"));
		}

	t2 = cciutils::event::timestampInUS();
	if (this->logger != NULL) logger->getSession("setup")->log(cciutils::event(0, std::string("layout comms"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

	// now set up the workers
	if (compute_io_g == COMPUTE_GROUP) {
			Action_I *seg =
					new cci::rt::syntest::GenerateOutputPush(&comm, MPI_UNDEFINED, NULL, sbuf,
							params,
							logger->getSession("seg"));
			proc->addHandler(seg);
			proc->addHandler(handler2);

	} else	{
		t1 = cciutils::event::timestampInUS();
		int comm1_size;
		int comm1_rank;
		MPI_Comm_size(*(handler->getComm()), &comm1_size);
		MPI_Comm_rank(*(handler->getComm()), &comm1_rank);

		// then within IO group, split to subgroups, for adios.
		int subio_size = CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOGROUPSIZE);
		int subio_interleave = CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_IOGROUPINTERLEAVE);

		if (subio_interleave < 1) subio_interleave = 1;
		else if (subio_interleave > comm1_size) subio_interleave = comm1_size;
		if (subio_size < 1) subio_size = comm1_size / subio_interleave;
		else if (subio_size > comm1_size) subio_size = comm1_size;

		if (subio_size == 1) {
			io_sub_g = comm1_rank;   // each in own group.  group id = rank in io group
		} else if (subio_size >= comm1_size) {
			io_sub_g = 0;  // everyone in same group.  group id = 0 for everyone.
		} else {
			if (subio_interleave > 1) {
				int blockid = comm1_rank / (subio_size * subio_interleave);
				io_sub_g = blockid * subio_interleave + comm1_rank % subio_interleave;
				// blocks of subio_interleave io subgroups, id within a block is the modulus.  id offset for a block is blockid * subio_interleave.
			} else {
				io_sub_g = comm1_rank / subio_size;  // subgroups, with interleave of 1, so
					// group id = block id of rank (bloc size = subio_size)
			}
		}
		// io subgroups
		std::string iocode = CmdlineParser::getParamValueByName<std::string>(params, CmdlineParser::PARAM_IOTRANSPORT);

		t2 = cciutils::event::timestampInUS();
		if (this->logger != NULL) logger->getSession("setup")->log(cciutils::event(0, std::string("layout adios"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

		Action_I *save;
		if (strcmp(iocode.c_str(), "na-NULL") == 0)
			save = new cci::rt::NullSinkAction(handler->getComm(), io_sub_g,
					rbuf, NULL,
					logger->getSession("io"));
		else if (strcmp(iocode.c_str(), "na-POSIX") == 0)
			save = new cci::rt::adios::POSIXRawSave(handler->getComm(), io_sub_g,
					rbuf, NULL,
				params,
				logger->getSession("io"));  // comm is group 1 IO comms, split into io_sub_g comms
		else
			save = new cci::rt::adios::ADIOSSave_Reduce(handler->getComm(), io_sub_g,
					rbuf, NULL,
				params,
				CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_MAXIMGSIZE) *
				CmdlineParser::getParamValueByName<int>(params, CmdlineParser::PARAM_MAXIMGSIZE) * 4, 256, 1024,
				iomanager, logger->getSession("io"));  // comm is group 1 IO comms, split into io_sub_g comms

		proc->addHandler(handler2);
		proc->addHandler(save);
	}
	delete handler;

	return true;
}

}
} /* namespace rt */
} /* namespace cci */
