/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
//#include "opencv2/opencv.hpp"
//#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <fstream>
//#include <stdio.h>
#include <vector>
#include <queue>
#include <string.h>
#include "TypeUtils.h"
#include "FileUtils.h"
#include <dirent.h>
#include "Logger.h"
#include "SCIOUtilsADIOS.h"
#include <mpi.h>
#include "waMPI.h"
#include "SCIOHistologicalEntities.h"

#include <unistd.h>

using namespace cv;


void printUsage(char ** argv);
int parseInput(int argc, char **argv, int &modecode, std::string &iocode, int &imageCount, int &maxbuf, int &groupSize, int &groupInterleave, std::string &workingDir, std::string &imageName, std::string &outDir, bool &benchmark, bool &compression);
void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output, const int &imageCount);
void compute(const char *input, const char *mask, const char *output, const int modecode, cci::common::LogSession *session, cciutils::SCIOADIOSWriter *writer);

void printUsage(char **argv) {
	std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> output_dir <transport> [imagecount] [buffersize] [cpu | gpu [id]] [groupsize] [groupInterleave] [benchmark] [compression]" << std::endl;
	std::cout << "transport is one of NULL | POSIX | MPI | MPI_LUSTRE | MPI_AMR | gap-NULL | gap-POSIX | gap-MPI | gap-MPI_LUSTRE | gap-MPI_AMR" << std::endl;
	std::cout << "imagecount: number of images to process.  -1 means all images." << std::endl;
	std::cout << "buffersize: number of images to buffer by a process before adios write.  default is 4." << std::endl;
	std::cout << "groupsize is the size of the adios IO subgroup (default -1 means all procs).  groupInterleave (integer) is how the groups mix together.  default is 1 for no interleaving: processes in a group have contiguous process ids." << std::endl;
	std::cout << "  groupInterleave value of less than 1 is treated as 1.  numbers greater than 1 interleaves that many groups. e.g. 1 2 3 1 2 3.  This is useful to match interleaves to node's core count." << std::endl;
	std::cout << "[compression] = on|off: optional. turn on compression for MPI messages and IO. default off." << std::endl;

}

int parseInput(int argc, char **argv, int &modecode, std::string &iocode, int &imageCount, int &maxbuf, int &groupSize, int &groupInterleave, std::string &workingDir, std::string &imageName, std::string &outDir, bool &benchmark, bool &compression) {
	if (argc < 4) {
		printUsage(argv);
		return -1;
	}

	std::string executable(argv[0]);
	workingDir.assign(cci::common::FileUtils::getDir(executable));

	imageName.assign(argv[1]);
	outDir.assign(argv[2]);

	if (argc > 3 &&
			strcmp(argv[3], "NULL") != 0 &&
			strcmp(argv[3], "POSIX") != 0 &&
			strcmp(argv[3], "MPI") != 0 &&
			strcmp(argv[3], "MPI_LUSTRE") != 0 &&
			strcmp(argv[3], "MPI_AMR") != 0 &&
			strcmp(argv[3], "gap-NULL") != 0 &&
			strcmp(argv[3], "gap-POSIX") != 0 &&
			strcmp(argv[3], "gap-MPI") != 0 &&
			strcmp(argv[3], "gap-MPI_LUSTRE") != 0 &&
			strcmp(argv[3], "gap-MPI_AMR") != 0) {
		printUsage(argv);
		return -1;

	} else {
		iocode.assign(argv[3]);
	}

	if (argc > 4) imageCount = atoi(argv[4]);
	if (argc > 5) maxbuf = atoi(argv[5]);


	const char* mode = argc > 6 ? argv[6] : "cpu";

	int i = 7;

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cci::common::type::DEVICE_CPU;
		// get core count

	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cci::common::type::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
		if (argc > i) {
			gpu::setDevice(atoi(argv[i]));
			++i;
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		printUsage(argv);
		return -1;
	}

	groupSize = -1;  // all available goes into same group.
	if (argc > i) groupSize = atoi(argv[i]);
	if (groupSize < 1) groupSize = -1;

	++i;
	groupInterleave = 1;
	if (argc > i) groupInterleave = atoi(argv[i]);
	if (groupInterleave < 1) groupInterleave = 1;

	++i;
	benchmark = false;
	if (argc > i && strcasecmp(argv[i], "benchmark") == 0) benchmark = true;

	++i;
	compression = (argc > i && strcmp(argv[i], "on") == 0 ? true : false);

	return 0;
}


void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output, const int &imageCount) {

	// check to see if it's a directory or a file
	std::vector<std::string> exts;
	exts.push_back(std::string(".tif"));
	exts.push_back(std::string(".tiff"));

	cci::common::FileUtils futils(exts);
	futils.traverseDirectory(imageName, filenames, cci::common::FileUtils::FILE, true);

	std::string dirname = imageName;
	if (filenames.size() == 1) {
		// if the maskname is actually a file, then the dirname is extracted from the maskname.
		if (strcmp(filenames[0].c_str(), imageName.c_str()) == 0) {
			dirname = imageName.substr(0, imageName.find_last_of("/\\"));
		}
	}

	srand(0);
	std::random_shuffle( filenames.begin(), filenames.end() );
	if (imageCount != -1 && imageCount < filenames.size()) {
		// randomize the file order.
		filenames.resize(imageCount);
	}

	std::string temp, tempdir;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
			// generate the output file name
		temp = futils.replaceExt(filenames[i], ".mask.pbm");
		temp = cci::common::FileUtils::replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		cci::common::FileUtils::mkdirs(tempdir);
		seg_output.push_back(temp);
		// generate the bounds output file name
		temp = futils.replaceExt(filenames[i], ".bounds.csv");
		temp = cci::common::FileUtils::replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		cci::common::FileUtils::mkdirs(tempdir);
		bounds_output.push_back(temp);
	}


}




void compute(const char *input, const char *mask, const char *output, const int modecode, cci::common::LogSession *session, cciutils::SCIOADIOSWriter *writer) {
	// compute

	int status;
	int *bbox = NULL;
	int compcount;

	if (writer == NULL) printf("why is writer null? \n");
	if (session == NULL) printf("why is log session null? \n");
	if (modecode == cci::common::type::DEVICE_GPU ) {
		nscale::gpu::SCIOHistologicalEntities *seg = new nscale::gpu::SCIOHistologicalEntities(std::string(input));
		status = seg->segmentNuclei(std::string(input), std::string(mask), compcount, bbox, NULL, session, writer);
		delete seg;

	} else {

		nscale::SCIOHistologicalEntities *seg = new nscale::SCIOHistologicalEntities(std::string(input));
		status = seg->segmentNuclei(std::string(input), std::string(mask), compcount, bbox, session, writer);
		delete seg;
	}


	free(bbox);



//	::cv::Mat image = cv::imread(input, -1);
	// for testing only.


//	if (writer != NULL) {
////		writer->open();
//		writer->saveIntermediate(image, 0, imagename, tilex, tiley);
////		writer->close();
//	}
//
//	free(imagename);
}


#if defined (WITH_MPI)
MPI_Comm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI_Comm init_workers(const MPI_Comm &comm_world, int managerid, int &worker_size, int &worker_rank, const int group_size, const int group_interleave, int &worker_group);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size,
		const std::string &hostname, std::vector<std::string> &filenames, std::vector<std::string > &seg_output,
		std::vector<std::string> &bounds_output, cci::common::Logger *logger, int maxWorkerLoad, const int worker_group);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank,
		const int modecode, const std::string &hostname, cciutils::SCIOADIOSWriter *writer, cci::common::Logger *logger, const int worker_group);


// initialize MPI
MPI_Comm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname) {
    int ierr = MPI_Init(&argc, &argv);

    char * temp = (char*)malloc(256);
    gethostname(temp, 255);
    hostname.assign(temp);
    free(temp);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return MPI_COMM_WORLD;
}

// not necessary to create a new comm object
// we split based on color, and also return the color information.

MPI_Comm init_workers(const MPI_Comm &comm_world, int managerid, int &worker_size, int &worker_rank, const int group_size, const int group_interleave, int &worker_group ) {

	int rank, size;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);
	
	// create new group from old group
	// first come up with the color  manager gets color 0.  everyone else: 1.
	if (group_size == 1) {
		// everyone is in his own group
		worker_group = rank;
	} else if (group_size < 1) {
		// everyone in one group
		if (rank == managerid) worker_group = 0;
		else worker_group = 1;
	} else {
		if (rank == managerid) worker_group = 0;
		else {
			if (group_interleave > 1) {
				// e.g. 0, 12, 24 go to group 1. 1,13,25 group 2,  144, 156, ... got to group 13.  for groupsize = 12 and interleave of 3.
				// block is a group of groups that are interleaved.
				int blockid = rank / (group_size * group_interleave);
				// each block has group_interleave number of groups
				// so the starting worker_group within a block is blockid*interleave
				worker_group = blockid * group_interleave + rank % group_interleave;

			} else {
				// interleave of 1 or less means adjacent proc ids in group.
				// e.g. 0 .. 11 go to group 1.
				worker_group = rank / group_size;
			}
			++worker_group;  // manager has group 0.
		}
	}
	
//	printf("rank %d work_group %d\n", rank, worker_group);

	MPI_Comm comm_worker;
	MPI_Comm_split(comm_world, worker_group, rank, &comm_worker);
	
	if (rank != managerid) {
		MPI_Comm_size(comm_worker, &worker_size);
		MPI_Comm_rank(comm_worker, &worker_rank);
	} else {
		worker_size = size-1;
		worker_rank = -1;
		worker_group = -1;
	}
	return comm_worker;
}

static const char MANAGER_READY = 10;
static const char MANAGER_REQUEST_IO = 11;
static const char MANAGER_WAIT = 12;
static const char MANAGER_FINISHED = 13;
static const char MANAGER_ERROR = -11;
static const char WORKER_READY = 20;
static const char WORKER_PROCESSING = 21;
static const char WORKER_ERROR = -21;
static const int TAG_CONTROL = 0;
static const int TAG_DATA = 1;
static const int TAG_METADATA = 2;


void manager_process(const MPI_Comm &comm_world, const int manager_rank, const int worker_size, const std::string &hostname,
	std::vector<std::string> &filenames, std::vector<std::string> &seg_output,
	std::vector<std::string> &bounds_output, cci::common::Logger *logger, int maxWorkerLoad, const int worker_group) {
	uint64_t t1, t0;

	int size;
	MPI_Comm_size(comm_world, &size);

	int *groupids = new int[size];
	memset((void *) groupids, 0, sizeof(int) * size);
	int g = worker_group;
//	printf("rank %d group %d\n", manager_rank, g);
	// send off the worker group info to the master
	MPI_Gather(&g, 1, MPI_INT, groupids, 1, MPI_INT, manager_rank, comm_world);

//	printf("GROUPS: ");
//	for (int i = 0; i < size; i++)
//		printf("%d, ", groupids[i]);
//	printf("\n");

	MPI_Barrier(comm_world);


	cci::common::LogSession *session = logger->getSession("m");

	// now start the loop to listen for messages
	int curr = 0;
	int total = filenames.size();
	// printf("total = %d\n", total);

	MPI_Status status;
	int worker_id;
	int worker_status[3];
	char *all, *input, *mask, *output;
	int inputlen, masklen, outputlen, alllen;
	int sizes[3];

	int hasMessage, hasOutgoing;

	std::tr1::unordered_map<int, std::vector<int> > groupToRank;
	std::tr1::unordered_map<int, int > groupIOIter;
	std::vector<std::deque<char> > messages;
	int gid;
	for (int i = 0; i < size; ++i) {
		gid = groupids[i];
		groupToRank[gid].push_back(i);
		groupIOIter[gid] = 0;

		messages.push_back(std::deque<char>());
		//printf("queue size is %ld\n", messages[i].size());
		//printf("set status of manager for worker %d to %d\n", i, (messages[i].empty() ? 10 : messages[i].front()));
	}
	int IOCount = 0;
	long long t2, t3;

	t2 = ::cci::common::event::timestampInUS();

	cci::common::mpi::waMPI *waComm = new cci::common::mpi::waMPI(comm_world);


	while (curr < total || IOCount > 0) {
		//usleep(1000);


		if (waComm->iprobe(MPI_ANY_SOURCE, TAG_CONTROL, &status)) {
/* where is it coming from */

// 			comment out to reduce amount of logging by master
//			t3 = ::cci::common::event::timestampInUS();
//			if (session != NULL) session->log(cci::common::event(90, std::string("manager found msg"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));
//
//			t2 = ::cci::common::event::timestampInUS();

			worker_id = status.MPI_SOURCE;
			MPI_Recv(&worker_status, 3, MPI_INT, worker_id, TAG_CONTROL, comm_world, &status);
			//printf("manager received request from worker %d\n",worker_id);
//			t3 = ::cci::common::event::timestampInUS();
			//if (session != NULL) session->log(cci::common::event(90, std::string("received msg"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));

			if (worker_id == manager_rank) continue;

			if (curr % 100 == 0) {
				printf("[ MANAGER STATUS at %lld ] %d tasks remaining.\n", ::cci::common::event::timestampInUS(), total - curr);
			}

			if(worker_status[0] == WORKER_READY) {
//				t2 = ::cci::common::event::timestampInUS();

				gid = groupids[worker_id];
				// first find out what the load is
				if (worker_status[1] >= maxWorkerLoad && worker_status[2] == groupIOIter[gid]) {
					// set everyone in the group to do IO.
					for (std::vector<int>::iterator iter = groupToRank[gid].begin(); iter != groupToRank[gid].end(); ++iter) {
//						printf("worker %d iterator has value %d\n", worker_id, *iter);
						messages[*iter].push_front(MANAGER_REQUEST_IO);
					}
					//printf("MANAGER %d send IO request to Group %d\n", manager_rank, gid);
					IOCount += groupToRank[gid].size();
					++groupIOIter[gid];
//					printf("current queue content = %d at front\n", messages[worker_id].front());
				}
//				t3 = ::cci::common::event::timestampInUS();
				//if (session != NULL) session->log(cci::common::event(90, std::string("queue IO"), t2, t3, std::string(), ::cci::common::event::MEM_IO));
//				t2 = ::cci::common::event::timestampInUS();

				char mstatus;
				if (messages[worker_id].empty()) {
					mstatus = MANAGER_READY;
				} else {
					mstatus = messages[worker_id].front();
				}
//				printf("manager status: %d \n", mstatus);

				if (mstatus == MANAGER_REQUEST_IO) {
					messages[worker_id].pop_front();

					// tell worker to do IO.
//					printf("manager sent IO request to worker %d for io iter.\n", worker_id);
					MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
//					if (curr >= total) messages[worker_id].push(MANAGER_WAIT);
//					else messages[worker_id].push(MANAGER_READY);
					--IOCount;

					t3 = ::cci::common::event::timestampInUS();
					if (session != NULL) session->log(cci::common::event(90, std::string("sent IO"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));

				} else if (mstatus == MANAGER_READY ){

					// tell worker that manager is ready
					//printf("manager sending work %d to %d.\n", mstatus, worker_id);
					MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
// conserve logging
//					t3 = ::cci::common::event::timestampInUS();
//					if (session != NULL) session->log(cci::common::event(90, std::string("manager sent ready"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));
//
//					//				printf("manager signal transfer\n");
//
//					/* send real data */
//					t2 = ::cci::common::event::timestampInUS();

					inputlen = filenames[curr].size();  // add one to create the zero-terminated string
					masklen = seg_output[curr].size();
					outputlen = bounds_output[curr].size();
					sizes[0] = inputlen;
					sizes[1] = masklen;
					sizes[2] = outputlen;
					alllen = inputlen + 1 + masklen + 1 + outputlen + 1;
					all = (char*)malloc(alllen);
					memset(all, 0, sizeof(char) * alllen);
					input = all;
					strncpy(input, filenames[curr].c_str(), inputlen);
					mask = input + inputlen + 1;
					strncpy(mask, seg_output[curr].c_str(), masklen);
					output = mask + masklen + 1;
					strncpy(output, bounds_output[curr].c_str(), outputlen);

					MPI_Send(&sizes, 3, MPI::INT, worker_id, TAG_METADATA, comm_world);

					// now send the actual string data
					MPI_Send(all, alllen, MPI::CHAR, worker_id, TAG_DATA, comm_world);

					free(all);

// conserve logging.
//					t3 = ::cci::common::event::timestampInUS();
//					if (session != NULL) session->log(cci::common::event(90, std::string("sent work"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));
//					t2 = ::cci::common::event::timestampInUS();

					++curr;
					if (curr >= total) {
						// at end.  tell everyone to wait for the remaining IO to complete

						for (int i = 0; i < size; ++i) {
							messages[i].push_back(MANAGER_WAIT);
						}
						//printf("current queue content = %d at back\n", messages[worker_id].back());
					} // else ready state.  don't change it.
// conserve logging
//					t3 = ::cci::common::event::timestampInUS();
//					if (session != NULL) session->log(cci::common::event(90, std::string("queue wait"), t2, t3, std::string(), ::cci::common::event::MEM_IO));
					t3 = ::cci::common::event::timestampInUS();
//					if (session != NULL) session->log(cci::common::event(90, std::string("sent work"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));

				} else {  // wait state.

					// tell worker to wait
					//printf("manager sending message %d to %d.\n", mstatus, worker_id);
					MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
//					t3 = ::cci::common::event::timestampInUS();
					//if (session != NULL) session->log(cci::common::event(90, std::string("sent wait"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));

				}
			}


			t2 = ::cci::common::event::timestampInUS();

		}
	}



/* tell everyone to quit */
	int active_workers = worker_size;
	//printf("active_worker count = %d\n", active_workers);
	t2 = ::cci::common::event::timestampInUS();
	while (active_workers > 0) {
		//usleep(1000);

		if (waComm->iprobe(MPI_ANY_SOURCE, TAG_CONTROL, &status)) {
		/* where is it coming from */

// conserve space...
//			t3 = ::cci::common::event::timestampInUS();
//			if (session != NULL) session->log(cci::common::event(90, std::string("manager found msg"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));
//
//			t2 = ::cci::common::event::timestampInUS();

			worker_id=status.MPI_SOURCE;
			MPI_Recv(&worker_status, 3, MPI::INT, worker_id, TAG_CONTROL, comm_world, &status);
//			printf("manager received request from worker %d\n",worker_id);
			//t3 = ::cci::common::event::timestampInUS();
			//if (session != NULL) session->log(cci::common::event(90, std::string("received msg"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));

			if (worker_id == manager_rank) continue;

			//t2 = ::cci::common::event::timestampInUS();

			if(worker_status[0] == WORKER_READY) {
				char mstatus = MANAGER_FINISHED;
				MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
				//printf("manager signal finished to %d\n", worker_id);
				--active_workers;
				t3 = ::cci::common::event::timestampInUS();
//				if (session != NULL) session->log(cci::common::event(90, std::string("sent END"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));

			}

			t2 = ::cci::common::event::timestampInUS();

		}
	}

	//printf("MANAGER waiting for MPI sync\n");
	// now all child processes will be doing the collective IO
	delete waComm;

	MPI_Barrier(comm_world);

	delete [] groupids;

}

void worker_process(const MPI_Comm &comm_world, const int manager_rank, const int rank,
		const MPI_Comm &comm_worker, const int modecode, const std::string &hostname,
		cciutils::SCIOADIOSWriter *writer, cci::common::Logger *logger, const int worker_group) {
	int flag = MANAGER_READY;
	int inputSize, outputSize, maskSize, sizeTotal;
	int sizes[3];
	char *input, *output, *mask, *all;
	MPI_Status status;

	int size;
	MPI_Comm_size(comm_world, &size);


	int iocount = 0;
	int g = worker_group;
//	printf("%d group %d\n", rank, g);
	// send off the worker group info to the master
	MPI_Gather(&g, 1, MPI_INT, NULL, 1, MPI_INT, manager_rank, comm_world);

	MPI_Barrier(comm_world);

	uint64_t t0, t1;
	//printf("worker %d ready\n", rank);
	//MPI_Barrier(comm_worker); // testing only

	std::string sessionName;
	int workerStatus[3];
	workerStatus[0] = WORKER_READY;
	workerStatus[1] = 0;
	workerStatus[2] = iocount;
	cci::common::LogSession *session;
	// per node
	if (logger) session = logger->getSession("w");
	if (writer) writer->setLogSession(session);
	bool first = true;

	long long t3, t2;

	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t0 = cci::common::event::timestampInUS();

		t2 = ::cci::common::event::timestampInUS();

		if (writer != NULL) workerStatus[1] = writer->currentLoad();
		workerStatus[2] = iocount;

		// tell the manager - ready
		MPI_Send(&workerStatus, 3, MPI_INT, manager_rank, TAG_CONTROL, comm_world);
		//printf("worker %d signal ready\n", rank);
		// get the manager status
		MPI_Recv(&flag, 1, MPI_CHAR, manager_rank, TAG_CONTROL, comm_world, &status);
		//printf("worker %d received manager status %d\n", rank, flag);
//		t3 = ::cci::common::event::timestampInUS();
//		if (session != NULL) session->log(cci::common::event(90, std::string("get status"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));
//
//		t2 = ::cci::common::event::timestampInUS();
		if (flag == MANAGER_READY) {

			// get data from manager

			MPI_Recv(&sizes, 3, MPI_INT, manager_rank, TAG_METADATA, comm_world, &status);
			inputSize = sizes[0];
			maskSize = sizes[1];
			outputSize = sizes[2];

			// allocate the buffers
			sizeTotal = inputSize + 1 + maskSize + 1 + outputSize + 1;
			all = (char *)malloc(sizeTotal);
			memset(all, 0, sizeTotal * sizeof(char));

			// get the file names
			MPI_Recv(all, sizeTotal, MPI_CHAR, manager_rank, TAG_DATA, comm_world, &status);
			input = all;
			mask = input + inputSize + 1;
			output = mask + maskSize + 1;

			t3 = ::cci::common::event::timestampInUS();
//			if (session != NULL) session->log(cci::common::event(90, std::string("get work"), t2, t3, std::string(), ::cci::common::event::NETWORK_IO));

			t0 = cci::common::event::timestampInUS();
//			printf("comm time for worker %d is %lu ms\n", rank, t1 -t0);

			// per tile
			// session = logger->getSession(std::string(input));
			// per node
			session = logger->getSession("w");
			if (writer) writer->setLogSession(session);
			compute(input, mask, output, modecode, session, writer);
			// now do some work

			t1 = cci::common::event::timestampInUS();
			//printf("worker %d processed \"%s\" in %lu ms\n", rank, input, t1 - t0);

			// clean up
			free(all);

		} else if (flag == MANAGER_REQUEST_IO) {
			// do some IO.
			//printf("iter %d manager-initiated IO for worker %d \n", iocount, rank);

			// per node
			session = logger->getSession("w");
			if (writer) writer->setLogSession(session);

			if (writer) writer->persist(iocount);
			++iocount;
		} else if (flag == MANAGER_WAIT) {
			//printf("manager told worker %d to wait\n", rank);
			t2 = ::cci::common::event::timestampInUS();

			usleep(100);
			t3 = ::cci::common::event::timestampInUS();
			//if (session != NULL) session->log(cci::common::event(90, std::string("wait"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));

		} else if (flag == MANAGER_FINISHED) {
			//printf("manager told worker %d finished\n", rank);
			t2 = ::cci::common::event::timestampInUS();
			t3 = ::cci::common::event::timestampInUS();
			//if (session != NULL) session->log(cci::common::event(90, std::string("manager finished"), t2, t3, std::string(), ::cci::common::event::NETWORK_WAIT));

		} else {
			printf("WANRING manager send unknown message %d to worker %d\n", flag, rank);
			t2 = ::cci::common::event::timestampInUS();
			usleep(100);
			t3 = ::cci::common::event::timestampInUS();
			//if (session != NULL) session->log(cci::common::event(90, std::string("unknown or error"), t2, t3, std::string(), ::cci::common::event::OTHER));
		}
	}

	// printf("WORKER %d waiting for MPI barrier\n", rank);

	// manager is now done.  now do IO again
	//printf("worker %d final IO \n", rank);
	// per node
	session = logger->getSession("w");
	if (writer) writer->setLogSession(session);
	if (writer) writer->persist(iocount);
	// printf("written out data %d \n", rank);
	// last tiles were just written.  now add teh count informaton
	if (writer) writer->persistCountInfo();
	//printf("written out data count %d \n", rank);

	// now do collective io.
	MPI_Barrier(comm_world);


}

int main (int argc, char **argv){


	// parse the input
	int modecode, groupSize, groupInterleave;
	std::string imageName, outDir, hostname, workingDir, iocode;
	bool benchmark, compression;
	int imageCount= -1, maxBuf = 4;
	int status = parseInput(argc, argv, modecode, iocode, imageCount, maxBuf, groupSize, groupInterleave, workingDir, imageName, outDir, benchmark, compression);
	if (status != 0) return status;


	// set up mpi
	int rank, size, worker_size, worker_rank, manager_rank, worker_group;
	MPI_Comm comm_world = init_mpi(argc, argv, size, rank, hostname);

	//printf("rank %d.  groupsize: %d, interleave %d\n", rank, groupSize, groupInterleave);
	
	manager_rank = size - 1;
	worker_group = 1;

	if (modecode == cci::common::type::DEVICE_GPU) {
		printf("WARNING:  GPU specified for an MPI run.   only CPU is supported.  please restart with CPU as the flag.\n");
		return -4;
	}
	std::vector<int> stages;
	for (int stage = 90; stage <= 100; ++stage) {
		stages.push_back(stage);
	}

	// get the input files and broadcast the count to all
	long total = 0;
	// first get the list of files to process
       	std::vector<std::string> filenames;
    	std::vector<std::string> seg_output;
    	std::vector<std::string> bounds_output;

	// first process gathers the filesnames
	uint64_t t1 = 0, t2 = 0;
	if (rank == manager_rank) {
    		t1 = cci::common::event::timestampInUS();
    		getFiles(imageName, outDir, filenames, seg_output, bounds_output, imageCount);

    		t2 = cci::common::event::timestampInUS();
    		total = filenames.size();
	    	printf("FILE LISTING took %lu ms for %ld files\n", t2 - t1, total);

	}
	// then if MPI, broadcast it
	if (size > 1) {
		MPI_Bcast(&total, 1, MPI_INT, manager_rank, comm_world);
	}



	/* now perform the computation
	*/
	std::string adios_config = workingDir;
	adios_config.append("/../adios_xml/image-tiles-globalarray-");
	adios_config.append(iocode);
	adios_config.append(".xml");

	// for testing
	bool gapped = false;
	if (strncmp(iocode.c_str(), "gap-", 4) == 0) gapped = true;

	bool appendInTime = true;
	if (strcmp(iocode.c_str(), "MPI_AMR") == 0 ||
		strcmp(iocode.c_str(), "gap-MPI_AMR") == 0) appendInTime = false;
	bool overwrite = true;


	cciutils::ADIOSManager *iomanager;

	cci::common::Logger *logger;
	cci::common::LogSession *session;
	if (size == 1) {
		std::string logfile(outDir);
		logfile.append("-");
		logfile.append(iocode);
		logger = new cci::common::Logger(logfile, rank, hostname, 0);
		session = logger->getSession("w");
		iomanager = new cciutils::ADIOSManager(adios_config.c_str(), rank, &comm_world, session, gapped, false, compression);

		int i = 0;

		// worker bees.  set to overwrite (third param set to true).

		cciutils::SCIOADIOSWriter *writer = iomanager->allocateWriter(outDir, std::string("bp"), appendInTime, overwrite,
				stages, total, total * (long)256, total * (long)1024, total * (long)(4096 * 4096 * 4),
				maxBuf, 4096*4096*4,
				worker_group, &comm_world);

		if (writer) writer->setLogSession(session);
	
		t1 = cci::common::event::timestampInUS();
		int iter = 0;
		while (i < total) {
			// per tile:
			// session = logger->getSession(filenames[i]);
			// per node

			compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode, session, writer);

//			printf("processed %s\n", filenames[i].c_str());
			++i;

			if (i % maxBuf == 0) {
				writer->persist(iter);
				++iter;
			}
		}
		t2 = cci::common::event::timestampInUS();
		printf("WORKER %d: FINISHED using CPU in %lu ms\n", rank, t2 - t1);

		if (writer) writer->persist(iter);
		if (writer) writer->persistCountInfo();
		iomanager->freeWriter(writer);


		logger->write();
	

	} else {
		// initialize the worker comm object

		// used by adios
		MPI_Comm comm_worker = init_workers(comm_world, manager_rank, worker_size, worker_rank, groupSize, groupInterleave, worker_group);

		std::string logfile(outDir);
		logfile.append("-");
		logfile.append(iocode);
		logger = new cci::common::Logger(logfile, rank, hostname, worker_group);
		session = logger->getSession(rank == manager_rank ? "m" : "w");
		iomanager = new cciutils::ADIOSManager(adios_config.c_str(), rank, &comm_world, session, gapped, true, compression);

		t1 = cci::common::event::timestampInUS();

		// decide based on rank of worker which way to process
		if (rank == manager_rank) {
			// manager thread
			manager_process(comm_world, manager_rank, worker_size, hostname, filenames, seg_output, bounds_output, logger, maxBuf, worker_group);
			t2 = cci::common::event::timestampInUS();
			printf("MANAGER %d : FINISHED in %lu ms\n", rank, t2 - t1);

		} else {



			// worker bees.  set to overwrite (third param set to true).
			cciutils::SCIOADIOSWriter *writer = iomanager->allocateWriter(outDir, std::string("bp"), appendInTime, overwrite,
					stages, total, total * (long)256, total * (long)1024, total * (long)(4096 * 4096 * 4),
					maxBuf, 4096*4096*4,
					worker_group, &comm_worker);
			writer->setLogSession(session);
			if (benchmark) writer->benchmark(0);

			worker_process(comm_world, manager_rank, rank, comm_worker, modecode, hostname, writer, logger, worker_group);
			t2 = cci::common::event::timestampInUS();
			//printf("WORKER %d: FINISHED using CPU in %lu ms\n", rank, t2 - t1);

			writer->setLogSession(session);
			if (benchmark) writer->benchmark(1);

			iomanager->freeWriter(writer);

		}
		MPI_Comm_free(&comm_worker);


		logger->writeCollectively(rank, manager_rank, comm_world);

	}
	delete iomanager;
	delete logger;
	

	MPI_Barrier(comm_world);


	MPI_Finalize();
	exit(0);

}
#else

    int main (int argc, char **argv){
    	printf("NOT compiled with MPI.  only works with MPI right now\n");
}
#endif

