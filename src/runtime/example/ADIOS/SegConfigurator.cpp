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
#include "AssignTiles.h"
#include "Segment.h"
#include "RandomScheduler.h"
#include "SCIOUtilsADIOS.h"
#include "SCIOUtilsLogger.h"

namespace cci {
namespace rt {
namespace adios {

const int SegConfigurator::UNDEFINED_GROUP = -1;
const int SegConfigurator::COMPUTE_GROUP = 1;
const int SegConfigurator::IO_GROUP = 2;
const int SegConfigurator::COMPUTE_TO_IO_GROUP = 3;
const int SegConfigurator::UNUSED_GROUP = 0;


bool SegConfigurator::init() {
	// need to initialize ADIOS by everyone because it has HARDCODED MPI_COMM_WORLD instead of taking a parameter for the comm.

	// get the configuration file
	std::string adios_config("/home/tcpan/PhD/path/src/nscale-debug/adios_xml/image-tiles-globalarray-");
	adios_config.append(iocode);
	adios_config.append(".xml");

	// determine if we are looking at gapped output
	bool gapped = false;
	if (strncmp(iocode.c_str(), "gap-", 4) == 0) gapped = true;

	// are the adios processes groupped into subgroups
	bool grouped = true;


	MPI_Comm comm = MPI_COMM_WORLD;
	int rank = -1;
	MPI_Comm_rank(comm, &rank);

	gethostname(hostname, 255);

	logger = new cciutils::SCIOLogger(rank, hostname, 0);
	cciutils::SCIOLogSession *session = logger->getSession("all");
	iomanager = new cciutils::ADIOSManager(adios_config.c_str(), rank, &comm, session, gapped, grouped);
	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM
}

bool SegConfigurator::finalize() {
	if (iomanager != NULL) {
		delete iomanager;
		iomanager = NULL;
	}
	if (logger != NULL) {
		delete logger;
		logger = NULL;
	}
	// ONLY HAS THIS CODE HERE BECAUSE ADIOS USES HARDCODED COMM
}

bool SegConfigurator::configure(MPI_Comm &comm, Process *proc) {

	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	if (size < 3) {
		printf("ERROR:  needs at least 3 mpi processes for segmentation: 1 master, 1 compute, 1 save\n");
		return false;
	}


	// now do a set up.
	int compute_io_g=-1, io_sub_g=-1, compute_to_io_g = -1;


	///// first set up the comm handlers

	// for fun, let's set up a compute group and an io group.
	// partitioning is arbitrary.  let the computesize be 3/4 of whole thing.
	// io be 1/4 of whole thing

	CommHandler_I *handler, *handler2;
	std::vector<int> roots;
//	std::ostream_iterator<int> out(std::cout, ",");


	// first split into 2.  focus on compute group.
	compute_io_g = (rank % 4 == 0 ? IO_GROUP : COMPUTE_GROUP);  // IO nodes have compute_io_g = 1; compute nodes compute_io_g = 0
	Scheduler_I *sch = NULL;
	if (compute_io_g == COMPUTE_GROUP && rank == 1) sch = new RandomScheduler(true, false);  // root at rank = 0
	else if (compute_io_g == IO_GROUP && rank == 0) sch = new RandomScheduler(true, false);  // root at rank = 0
	else sch = new RandomScheduler(false, true);

	// compute and io groups
	handler = new PullCommHandler(&comm, compute_io_g, sch);

	// then the compute to IO communication group
	compute_to_io_g = (compute_io_g == COMPUTE_GROUP && handler->isListener() ? UNUSED_GROUP : COMPUTE_TO_IO_GROUP);

	Scheduler_I *sch2 = NULL;
	if (compute_io_g == IO_GROUP) sch2 = new RandomScheduler(true, false);  // root at rank = 0
	else sch2 = new RandomScheduler(false, true);

	handler2 = new PushCommHandler(&comm, compute_to_io_g, sch2);
	//std::cout << "rank " << rank << ": ";
	//copy(roots.begin(), roots.end(), out);
	//std::cout << std::endl;

	// now set up the workers
	if (compute_io_g == COMPUTE_GROUP) {
		proc->addHandler(handler);
		//Debug::print("in compute setup\n");
		if (handler->isListener()) {  // master in the compute group
			Action_I *assign = new cci::rt::adios::AssignTiles(&comm, -1, std::string("/home/tcpan/PhD/path/Data/ValidationSet/20X_4096x4096_tiles/astroII.1"));
			proc->addHandler(assign);
			handler->setAction(assign);
			delete handler2;
		} else {
			Action_I *seg = new cci::rt::adios::Segment(&comm, -1);
			proc->addHandler(seg);
			proc->addHandler(handler2);
			handler->setAction(seg);
			handler2->setAction(seg);
		}

	} else	{


		// then within IO group, split to subgroups, for adios.
		int group_size = -1;
		int group_interleave = 1;
		int comm1_size;
		int comm1_rank;
		MPI_Comm_size(*(handler->getComm()), &comm1_size);
		MPI_Comm_rank(*(handler->getComm()), &comm1_rank);

		if (group_size == 1) {
			io_sub_g = comm1_rank;
		} else if (group_size < 1) {
			io_sub_g = 0;
		} else {
			if (group_interleave > 1) {
				int blockid = comm1_rank / (group_size * group_interleave);
				io_sub_g = blockid * group_interleave + comm1_rank % group_interleave;
			} else {
				io_sub_g = comm1_rank / group_size;
			}
			++io_sub_g;
		}
		// io subgroups
		Action_I *save = new cci::rt::adios::ADIOSSave(handler->getComm(), io_sub_g, logger, iomanager, iocode);  // comm is group 1 IO comms, split into io_sub_g comms
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
