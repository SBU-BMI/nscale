/*
 * SegConfigurator.cpp
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#include "SegConfigurator.h"

#include "PullCommHandler.h"
#include "PushCommHandler.h"

#include "ADIOSSave_Reduce.h"
#include "POSIXRawSave.h"

#include "AssignTiles.h"
#include "Segment.h"
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
namespace adios {

const int SegConfigurator::UNDEFINED_GROUP = MPI_UNDEFINED;
const int SegConfigurator::COMPUTE_GROUP = 1;
const int SegConfigurator::IO_GROUP = 2;
const int SegConfigurator::COMPUTE_TO_IO_GROUP = 3;
const int SegConfigurator::UNUSED_GROUP = 0;


bool SegConfigurator::init() {

	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM


	// need to initialize ADIOS by everyone because it has HARDCODED MPI_COMM_WORLD instead of taking a parameter for the comm.

	std::string iocode = params[SegmentCmdParser::PARAM_TRANSPORT];

	// get the configuration file


	// determine if we are looking at gapped output
	bool gapped = false;
	if (strncmp(iocode.c_str(), "gap-", 4) == 0) gapped = true;

	// are the adios processes groupped into subgroups
	bool grouped = false;
	int groupsize = atoi(params[SegmentCmdParser::PARAM_SUBIOSIZE].c_str());
	if (groupsize > 0 && groupsize < atoi(params[SegmentCmdParser::PARAM_IOSIZE].c_str())) grouped = true;

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank = -1;
	MPI_Comm_rank(comm, &rank);

	cciutils::SCIOLogSession *session = logger->getSession("setup");
	if (strncmp(iocode.c_str(), "na-", 3) != 0) {
		std::string adios_config(params[SegmentCmdParser::PARAM_EXECUTABLEDIR]);
		adios_config.append("/../adios_xml/image-tiles-globalarray-");
		adios_config.append(iocode);
		adios_config.append(".xml");

		//Debug::print("iomanager created for %s using config %s\n", iocode.c_str(), adios_config.c_str());
		iomanager = new ADIOSManager(adios_config.c_str(), params[SegmentCmdParser::PARAM_TRANSPORT], rank, comm, session, gapped, grouped);
	}

	return true;
}

bool SegConfigurator::finalize() {
	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM
	if (iomanager != NULL) {
		delete iomanager;
		iomanager = NULL;
	}

	return true;
}

bool SegConfigurator::configure(MPI_Comm &comm, Process *proc) {

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
	int compute_io_g=-1, io_sub_g=-1, compute_to_io_g = -1;

	// create the output directory
	if (rank == 0) {
		// create the directory
		FileUtils::mkdirs(params[SegmentCmdParser::PARAM_OUTPUTDIR]);
		printf("made directories for %s\n", params[SegmentCmdParser::PARAM_OUTPUTDIR].c_str());
	}

	///// first set up the comm handlers


	CommHandler_I *handler, *handler2;
	MPISendDataBuffer *sbuf = NULL;
	MPIRecvDataBuffer *rbuf = NULL;

	int iointerleave = atoi(params[SegmentCmdParser::PARAM_IOINTERLEAVE].c_str());
	int iosize = atoi(params[SegmentCmdParser::PARAM_IOSIZE].c_str());
	if (iointerleave < 1) iointerleave = 1;
	else if (iointerleave > size) iointerleave = size;
	if (iosize < 1 || iosize > size) iosize = size/iointerleave;  // default


	// first split into 2.  focus on compute group.
	Scheduler_I *sch = NULL;
	bool isroot = false;
	if (iointerleave == 1) {
		compute_io_g = (rank < iosize ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
		if (rank == iosize) isroot = true;  // compute root at rank = iosize
		else if (rank == 0) isroot = true;  // io root at rank = 0


	} else {
		compute_io_g = ((rank % iointerleave == 0) && (rank < iointerleave * iosize) ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
		if (rank == 1) isroot = true;  // compute root at rank = 1
		else if (rank == 0) isroot = true;  // io root at rank = 0
	}


	// set up the buffers
	if (compute_io_g == IO_GROUP) {  // io nodes
		sch = new RandomScheduler(false, false);
		// compute and io groups
		handler = new PullCommHandler(&comm, compute_io_g, NULL, sch, logger->getSession("pull"));
	} else {
		sch = new RandomScheduler(isroot, !isroot);
		if (isroot) {  // root of compute
			sbuf = new MPISendDataBuffer(100,
					(strcmp(params[SegmentCmdParser::PARAM_NONBLOCKING].c_str(), "on") == 0 ? true : false),
					logger->getSession("pull"));
			handler = new PullCommHandler(&comm, compute_io_g, sbuf, sch, logger->getSession("pull"));
		} else { // other compute
			rbuf = new MPIRecvDataBuffer(4,
					(strcmp(params[SegmentCmdParser::PARAM_NONBLOCKING].c_str(), "on") == 0 ? true : false),
					logger->getSession("pull"));
			handler = new PullCommHandler(&comm, compute_io_g, rbuf, sch, logger->getSession("pull"));
		}
	}

//	// then the compute to IO communication group
	// separate masters in the compute group
	compute_to_io_g = (compute_io_g == COMPUTE_GROUP && handler->isListener() ? UNUSED_GROUP : COMPUTE_TO_IO_GROUP);

	Scheduler_I *sch2 = NULL;
	if (compute_to_io_g == UNUSED_GROUP) {
		sch2 = new RandomScheduler(false, false);
		handler2 = new PushCommHandler(&comm, compute_to_io_g, NULL, sch2, logger->getSession("push"));
	} else {
		if (compute_io_g == IO_GROUP) {
			rbuf = new MPIRecvDataBuffer(atoi(params[SegmentCmdParser::PARAM_IOBUFFERSIZE].c_str()),
					(strcmp(params[SegmentCmdParser::PARAM_NONBLOCKING].c_str(), "on") == 0 ? true : false),
					logger->getSession("push"));
			sch2 = new RandomScheduler(true, false);  // all io nodes are roots.
			handler2 = new PushCommHandler(&comm, compute_to_io_g, rbuf, sch2, logger->getSession("push"));
		} else {
			sbuf = new MPISendDataBuffer(atoi(params[SegmentCmdParser::PARAM_IOBUFFERSIZE].c_str()),
					(strcmp(params[SegmentCmdParser::PARAM_NONBLOCKING].c_str(), "on") == 0 ? true : false),
					logger->getSession("push"));
			sch2 = new RandomScheduler(false, true);
			handler2 = new PushCommHandler(&comm, compute_to_io_g, sbuf, sch2, logger->getSession("push"));
		}
	}
//
	t2 = cciutils::event::timestampInUS();
	if (this->logger != NULL) logger->getSession("setup")->log(cciutils::event(0, std::string("layout comms"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

	// now set up the workers
	if (compute_io_g == COMPUTE_GROUP) {
		proc->addHandler(handler);
		//Debug::print("in compute setup\n");
		if (handler->isListener()) {  // master in the compute group
			Action_I *assign =
					new cci::rt::adios::AssignTiles(&comm, MPI_UNDEFINED, NULL, sbuf,
							params[SegmentCmdParser::PARAM_INPUT],
							atoi(params[SegmentCmdParser::PARAM_INPUTCOUNT].c_str()),
							logger->getSession("assign"));
			proc->addHandler(assign);
			delete handler2;
		} else {
			Action_I *seg =
					new cci::rt::adios::Segment(&comm, MPI_UNDEFINED, rbuf, sbuf,
							params[SegmentCmdParser::PARAM_PROCTYPE],
							atoi(params[SegmentCmdParser::PARAM_GPUDEVICEID].c_str()),
							(strcmp(params[SegmentCmdParser::PARAM_COMPRESSION].c_str(), "on") == 0 ? true : false),
							logger->getSession("seg"));
			proc->addHandler(seg);
			proc->addHandler(handler2);
		}

	} else	{
		t1 = cciutils::event::timestampInUS();
		int comm1_size;
		int comm1_rank;
		MPI_Comm_size(*(handler->getComm()), &comm1_size);
		MPI_Comm_rank(*(handler->getComm()), &comm1_rank);

		// then within IO group, split to subgroups, for adios.
		int subio_size = atoi(params[SegmentCmdParser::PARAM_SUBIOSIZE].c_str());
		int subio_interleave = atoi(params[SegmentCmdParser::PARAM_SUBIOINTERLEAVE].c_str());
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
		std::string iocode = params[SegmentCmdParser::PARAM_TRANSPORT];
		bool gapped = false;
		if (strncmp(iocode.c_str(), "gap-", 4) == 0) gapped = true;

		int total = atoi(params[SegmentCmdParser::PARAM_INPUTCOUNT].c_str());
		if (gapped) {
			total = total * comm1_size;  // worst case : all data went to 1.
		}
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
				params[SegmentCmdParser::PARAM_OUTPUTDIR],
				iocode,
				total,
				atoi(params[SegmentCmdParser::PARAM_IOBUFFERSIZE].c_str()),
				4096 * 4096 * 4, 256, 1024,
				logger->getSession("io"));  // comm is group 1 IO comms, split into io_sub_g comms
		else
			save = new cci::rt::adios::ADIOSSave_Reduce(handler->getComm(), io_sub_g,
					rbuf, NULL,
				params[SegmentCmdParser::PARAM_OUTPUTDIR],
				iocode,
				total,
				atoi(params[SegmentCmdParser::PARAM_IOBUFFERSIZE].c_str()),
				4096 * 4096 * 4, 256, 1024,
				iomanager, logger->getSession("io"));  // comm is group 1 IO comms, split into io_sub_g comms

		proc->addHandler(handler2);
		proc->addHandler(save);
		delete handler;
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
