/*
 * SegConfigurator.cpp
 *
 *  Created on: Jun 28, 2012
 *      Author: tcpan
 */

#include "SegConfigurator.h"

#include "PullCommHandler.h"
#include "PushCommHandler.h"
#include "ADIOSSave.h"
#include "ADIOSSave_Reduce.h"
#include "AssignTiles.h"
#include "Segment.h"
#include "RandomScheduler.h"
#include "RoundRobinScheduler.h"
#include "UtilsADIOS.h"
#include "SCIOUtilsLogger.h"

#include "NullSinkAction.h"

namespace cci {
namespace rt {
namespace adios {

const int SegConfigurator::UNDEFINED_GROUP = -1;
const int SegConfigurator::COMPUTE_GROUP = 1;
const int SegConfigurator::IO_GROUP = 2;
const int SegConfigurator::COMPUTE_TO_IO_GROUP = 3;
const int SegConfigurator::UNUSED_GROUP = 0;


bool SegConfigurator::init() {

	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM


	// need to initialize ADIOS by everyone because it has HARDCODED MPI_COMM_WORLD instead of taking a parameter for the comm.

	std::string iocode = params[SegmentCmdParser::PARAM_TRANSPORT];

	// get the configuration file
	std::string adios_config(params[SegmentCmdParser::PARAM_EXECUTABLEDIR]);
	adios_config.append("/../adios_xml/image-tiles-globalarray-");
	adios_config.append(iocode);
	adios_config.append(".xml");

	// determine if we are looking at gapped output
	bool gapped = false;
	if (strncmp(iocode.c_str(), "gap-", 4) == 0) gapped = true;

	// are the adios processes groupped into subgroups
	bool grouped = false;
	int groupsize = atoi(params[SegmentCmdParser::PARAM_SUBIOSIZE].c_str());
	if (groupsize > 1) grouped = true;

	MPI_Comm comm = MPI_COMM_WORLD;
	int rank = -1;
	MPI_Comm_rank(comm, &rank);

	cciutils::SCIOLogSession *session = logger->getSession("setup");
	iomanager = new ADIOSManager(adios_config.c_str(), rank, comm, session, gapped, grouped);

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


	///// first set up the comm handlers


	CommHandler_I *handler, *handler2;

	int iointerleave = atoi(params[SegmentCmdParser::PARAM_IOINTERLEAVE].c_str());
	int iosize = atoi(params[SegmentCmdParser::PARAM_IOSIZE].c_str());
	if (iointerleave < 1) iointerleave = 1;
	else if (iointerleave > size) iointerleave = size;
	if (iosize < 1 || iosize > size) iosize = size/iointerleave;  // default


	// first split into 2.  focus on compute group.
	Scheduler_I *sch = NULL;
	if (iointerleave == 1) {
		compute_io_g = (rank < iosize ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
		if (rank == iosize) sch = new RoundRobinScheduler(true, false);  // compute root at rank = iosize
		else if (rank == 0) sch = new RoundRobinScheduler(true, false);  // io root at rank = 0
		else sch = new RoundRobinScheduler(false, true);
	} else {
		compute_io_g = ((rank % iointerleave == 0) && (rank < iointerleave * iosize) ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
		if (rank == 1) sch = new RoundRobinScheduler(true, false);  // compute root at rank = 1
		else if (rank == 0) sch = new RoundRobinScheduler(true, false);  // io root at rank = 0
		else sch = new RoundRobinScheduler(false, true);
	}

	// compute and io groups
	handler = new PullCommHandler(&comm, compute_io_g, sch);

	// then the compute to IO communication group
	// separate masters in the compute group
	compute_to_io_g = (compute_io_g == COMPUTE_GROUP && handler->isListener() ? UNUSED_GROUP : COMPUTE_TO_IO_GROUP);

	Scheduler_I *sch2 = NULL;
	if (compute_to_io_g == UNUSED_GROUP) sch2 = new RoundRobinScheduler(false, false);
	else
		if (compute_io_g == IO_GROUP) sch2 = new RoundRobinScheduler(true, false);  // all io nodes are roots.
		else sch2 = new RoundRobinScheduler(false, true);

	handler2 = new PushCommHandler(&comm, compute_to_io_g, sch2);

	t2 = cciutils::event::timestampInUS();
	if (this->logger != NULL) logger->getSession("setup")->log(cciutils::event(0, std::string("layout comms"), t1, t2, std::string(), ::cciutils::event::MEM_IO));

	// now set up the workers
	if (compute_io_g == COMPUTE_GROUP) {
		proc->addHandler(handler);
		//Debug::print("in compute setup\n");
		if (handler->isListener()) {  // master in the compute group
			Action_I *assign =
					new cci::rt::adios::AssignTiles(&comm, -1,
							params[SegmentCmdParser::PARAM_INPUT],
							atoi(params[SegmentCmdParser::PARAM_INPUTCOUNT].c_str()),
							logger->getSession("assign"));
			proc->addHandler(assign);
			handler->setAction(assign);
			delete handler2;
		} else {
			Action_I *seg =
					new cci::rt::adios::Segment(&comm, -1,
							params[SegmentCmdParser::PARAM_PROCTYPE],
							atoi(params[SegmentCmdParser::PARAM_GPUDEVICEID].c_str()),
							logger->getSession("seg"));
			proc->addHandler(seg);
			proc->addHandler(handler2);
			handler->setAction(seg);
			handler2->setAction(seg);
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


		Action_I *save =
				new cci::rt::adios::ADIOSSave_Reduce(handler->getComm(), io_sub_g,
						params[SegmentCmdParser::PARAM_OUTPUTDIR],
						iocode,
						total,
						atoi(params[SegmentCmdParser::PARAM_IOBUFFERSIZE].c_str()),
						4096 * 4096 * 4, 256, 1024,
						iomanager, logger->getSession("io"));  // comm is group 1 IO comms, split into io_sub_g comms
//		Action_I *save = new cci::rt::NullSinkAction(handler->getComm(), io_sub_g, logger->getSession("io"));
		proc->addHandler(handler2);
		proc->addHandler(save);
		handler2->setAction(save);
		delete handler;
	}

	return true;
}

}
} /* namespace rt */
} /* namespace cci */
