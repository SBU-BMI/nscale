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
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>
#include "SCIOUtilsLogger.h"
#include "SCIOUtilsADIOS.h"
#include <mpi.h>
#include "SCIOHistologicalEntities.h"

using namespace cv;




int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir);
void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output);
void compute(const char *input, const char *mask, const char *output, const int modecode, cciutils::SCIOLogSession *session, cciutils::SCIOADIOSWriter *writer);

int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir) {
	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [numThreads] [id]]" << std::endl;
		return -1;
	}
	imageName.assign(argv[1]);
	outDir.assign(argv[2]);
	const char* mode = argc > 4 ? argv[4] : "cpu";

	int threadCount;
	if (argc > 5) threadCount = atoi(argv[5]);
	else threadCount = 1;

#if defined (WITH_MPI)
	threadCount = 1;
#endif

//	printf("number of threads: %d\n", threadCount);


	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		// get core count


	} else if (strcasecmp(mode, "mcore") == 0) {
//		modecode = cciutils::DEVICE_MCORE;
		// get core count
		modecode = cciutils::DEVICE_CPU;


	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
		if (argc > 6) {
			gpu::setDevice(atoi(argv[6]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> image_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [numThreads] [id]]" << std::endl;
		return -1;
	}

	return 0;
}


void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output) {

	// check to see if it's a directory or a file
	std::string suffix;
	suffix.assign(".tif");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(imageName, filenames);

	std::string dirname = imageName;
	if (filenames.size() == 1) {
		// if the maskname is actually a file, then the dirname is extracted from the maskname.
		if (strcmp(filenames[0].c_str(), imageName.c_str()) == 0) {
			dirname = imageName.substr(0, imageName.find_last_of("/\\"));
		}
	}


	std::string temp, tempdir;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
			// generate the output file name
		temp = futils.replaceExt(filenames[i], ".tif", ".mask.pbm");
		temp = futils.replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		futils.mkdirs(tempdir);
		seg_output.push_back(temp);
		// generate the bounds output file name
		temp = futils.replaceExt(filenames[i], ".tif", ".bounds.csv");
		temp = futils.replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		futils.mkdirs(tempdir);
		bounds_output.push_back(temp);
	}

}




void compute(const char *input, const char *mask, const char *output, const int modecode, cciutils::SCIOLogSession *session, cciutils::SCIOADIOSWriter *writer) {
	// compute

	int status;
	int *bbox = NULL;
	int compcount;

	if (writer == NULL) printf("why is writer null? \n");
	if (session == NULL) printf("why is log session null? \n");
	if (modecode == cciutils::DEVICE_GPU ) {
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
MPI_Comm init_workers(const MPI_Comm &comm_world, int managerid, int &worker_size, int &worker_rank);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size,
		const std::string &hostname, std::vector<std::string> &filenames, std::vector<std::string > &seg_output,
		std::vector<std::string> &bounds_output, cciutils::SCIOLogger *logger);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank,
		const int modecode, const std::string &hostname, cciutils::SCIOADIOSWriter *writer, cciutils::SCIOLogger *logger);


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
MPI_Comm init_workers(const MPI_Comm &comm_world, int managerid, int &worker_size, int &worker_rank ) {

	int rank, size;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);
	
	// create new group from old group
	MPI_Comm comm_worker;
	MPI_Comm_split(comm_world, (rank == managerid ? 1 : 0), rank, &comm_worker);
	
	if (rank != managerid) {
		MPI_Comm_size(comm_worker, &worker_size);
		MPI_Comm_rank(comm_worker, &worker_rank);
	} else {
		worker_size = size-1;
		worker_rank = -1;
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
	std::vector<std::string> &bounds_output, cciutils::SCIOLogger *logger) {
	uint64_t t1, t0;

	MPI_Barrier(comm_world);
	cciutils::SCIOLogSession *session = logger->getSession(hostname);

	// now start the loop to listen for messages
	int curr = 0;
	int total = filenames.size();
	// printf("total = %d\n", total);

	MPI_Status status;
	int worker_id;
	int worker_status[3];
	char *input;
	char *mask;
	char *output;
	int inputlen;
	int masklen;
	int outputlen;
	int hasMessage;
	int size;
	int ioiter = 0;

	MPI_Comm_size(comm_world, &size);
	int workerLoad[size];
	std::vector<std::deque<char> > messages;
	for (int i = 0; i < size; ++i) {
		messages.push_back(std::deque<char>());
		//printf("queue size is %ld\n", messages[i].size());
		workerLoad[i] = 0;
		//printf("set status of manager for worker %d to %d\n", i, (messages[i].empty() ? 10 : messages[i].front()));
	}
	int maxWorkerLoad = 2;
	int IOCount = 0;
	long long t2, t3;

	t2 = ::cciutils::event::timestampInUS();

	while (curr < total || IOCount > 0) {
		//usleep(1000);


		MPI_Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, comm_world, &hasMessage, &status);


		if (hasMessage != 0) {
/* where is it coming from */

			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("manager found msg"), t2, t3, std::string(), ::cciutils::event::NETWORK_WAIT));

			t2 = ::cciutils::event::timestampInUS();

			worker_id = status.MPI_SOURCE;
			MPI_Recv(&worker_status, 3, MPI_INT, worker_id, TAG_CONTROL, comm_world, &status);
//			printf("manager received request from worker %d\n",worker_id);
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("manager received msg"), t2, t3, std::string(), ::cciutils::event::NETWORK_IO));

			if (worker_id == manager_rank) continue;

			if (curr % 100 == 0) {
				printf("[ MANAGER STATUS ] %d tasks remaining.\n", total - curr);
			}

			if(worker_status[0] == WORKER_READY) {
				t2 = ::cciutils::event::timestampInUS();

				// first find out what the load is
				if (worker_status[1] >= maxWorkerLoad && worker_status[2] == ioiter) {
					// set everyone to do IO.
					for (int i = 0; i < size; ++i) {
						messages[i].push_front(MANAGER_REQUEST_IO);
					}
					IOCount += worker_size;
					++ioiter;
					//printf("current queue content = %d at front\n", messages[worker_id].front());
				}
				t3 = ::cciutils::event::timestampInUS();
				if (session != NULL) session->log(cciutils::event(90, std::string("manager queue IO"), t2, t3, std::string(), ::cciutils::event::MEM_IO));
				t2 = ::cciutils::event::timestampInUS();

				char mstatus;
				if (messages[worker_id].empty()) {
					mstatus = MANAGER_READY;
				} else {
					mstatus = messages[worker_id].front();
				}
				//printf("manager status: %d \n", mstatus);

				if (mstatus == MANAGER_REQUEST_IO) {
					messages[worker_id].pop_front();

					// tell worker to do IO.
//					printf("manager sent IO request to worker %d for io iter.\n", worker_id);
					MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
//					if (curr >= total) messages[worker_id].push(MANAGER_WAIT);
//					else messages[worker_id].push(MANAGER_READY);
					--IOCount;

					t3 = ::cciutils::event::timestampInUS();
					if (session != NULL) session->log(cciutils::event(90, std::string("manager sent IO"), t2, t3, std::string(), ::cciutils::event::NETWORK_MSG));

				} else if (mstatus == MANAGER_READY ){

					// tell worker that manager is ready
					//printf("manager sending work %d to %d.\n", mstatus, worker_id);
					MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
					t3 = ::cciutils::event::timestampInUS();
					if (session != NULL) session->log(cciutils::event(90, std::string("manager sent ready"), t2, t3, std::string(), ::cciutils::event::NETWORK_MSG));

					//				printf("manager signal transfer\n");

					/* send real data */
					t2 = ::cciutils::event::timestampInUS();

					inputlen = filenames[curr].size() + 1;  // add one to create the zero-terminated string
					masklen = seg_output[curr].size() + 1;
					outputlen = bounds_output[curr].size() + 1;
					input = new char[inputlen];
					memset(input, 0, sizeof(char) * inputlen);
					strncpy(input, filenames[curr].c_str(), inputlen);
					mask = new char[masklen];
					memset(mask, 0, sizeof(char) * masklen);
					strncpy(mask, seg_output[curr].c_str(), masklen);
					output = new char[outputlen];
					memset(output, 0, sizeof(char) * outputlen);
					strncpy(output, bounds_output[curr].c_str(), outputlen);

					MPI_Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA, comm_world);
					MPI_Send(&masklen, 1, MPI::INT, worker_id, TAG_METADATA, comm_world);
					MPI_Send(&outputlen, 1, MPI::INT, worker_id, TAG_METADATA, comm_world);

					// now send the actual string data
					MPI_Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA, comm_world);
					MPI_Send(mask, masklen, MPI::CHAR, worker_id, TAG_DATA, comm_world);
					MPI_Send(output, outputlen, MPI::CHAR, worker_id, TAG_DATA, comm_world);

					delete [] input;
					delete [] mask;
					delete [] output;

					t3 = ::cciutils::event::timestampInUS();
					if (session != NULL) session->log(cciutils::event(90, std::string("manager sent work"), t2, t3, std::string(), ::cciutils::event::NETWORK_IO));
					t2 = ::cciutils::event::timestampInUS();

					++curr;
					if (curr >= total) {
						// at end.  tell everyone to wait for the remaining IO to complete

						for (int i = 0; i < size; ++i) {
							messages[i].push_back(MANAGER_WAIT);
						}
						//printf("current queue content = %d at back\n", messages[worker_id].back());
					} // else ready state.  don't change it.
					t3 = ::cciutils::event::timestampInUS();
					if (session != NULL) session->log(cciutils::event(90, std::string("manager queue wait"), t2, t3, std::string(), ::cciutils::event::MEM_IO));

				} else {  // wait state.

					// tell worker to wait
					//printf("manager sending message %d to %d.\n", mstatus, worker_id);
					MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
					t3 = ::cciutils::event::timestampInUS();
					if (session != NULL) session->log(cciutils::event(90, std::string("manager sent wait"), t2, t3, std::string(), ::cciutils::event::NETWORK_MSG));

				}
			}


			t2 = ::cciutils::event::timestampInUS();

		}
	}



/* tell everyone to quit */
	int active_workers = worker_size;
	//printf("active_worker count = %d\n", active_workers);
	t2 = ::cciutils::event::timestampInUS();
	while (active_workers > 0) {
		//usleep(1000);

		MPI_Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, comm_world, &hasMessage, &status);

		if (hasMessage != 0) {
		/* where is it coming from */
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("manager found msg"), t2, t3, std::string(), ::cciutils::event::NETWORK_WAIT));

			t2 = ::cciutils::event::timestampInUS();

			worker_id=status.MPI_SOURCE;
			MPI_Recv(&worker_status, 3, MPI::INT, worker_id, TAG_CONTROL, comm_world, &status);
//			printf("manager received request from worker %d\n",worker_id);
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("manager received msg"), t2, t3, std::string(), ::cciutils::event::NETWORK_IO));

			if (worker_id == manager_rank) continue;

			t2 = ::cciutils::event::timestampInUS();

			if(worker_status[0] == WORKER_READY) {
				char mstatus = MANAGER_FINISHED;
				MPI_Send(&mstatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
				//printf("manager signal finished to %d\n", worker_id);
				--active_workers;
				t3 = ::cciutils::event::timestampInUS();
				if (session != NULL) session->log(cciutils::event(90, std::string("manager sent END"), t2, t3, std::string(), ::cciutils::event::NETWORK_MSG));

			}

			t2 = ::cciutils::event::timestampInUS();

		}
	}

	//printf("MANAGER waiting for MPI sync\n");
	// now all child processes will be doing the collective IO

	MPI_Barrier(comm_world);


	// now do a collective io for the log
	std::vector<std::string> timings = logger->toOneLineStrings();
	std::stringstream ss;
	for (int i = 0; i < timings.size(); i++) {
		ss << timings[i] << std::endl;
	}
	std::string logstr = ss.str();
	int logsize = logstr.size();
	// write out to file
	std::ofstream ofs("manager.csv");
	ofs << logstr << std::endl;
	ofs.close();

	logsize = 0;
	int *recbuf = (int *) malloc(size * sizeof(int));

	// now send the thing to manager
	// 	first gather sizes
	MPI_Gather(&logsize, 1, MPI_INT, recbuf, 1, MPI_INT, manager_rank, comm_world);

	
	// 	then gatherv the messages.
	char *sendlog = (char *)malloc(sizeof(char) * logstr.size() + 1);
	memset(sendlog, 0, sizeof(char) * logstr.size() + 1);
	strncpy(sendlog, logstr.c_str(), logstr.size());
	ss.str(std::string());

	// then perform exclusive prefix sum to get the displacement and the total length
	int *displbuf = (int *) malloc(size * sizeof(int));
	displbuf[0] = 0;
	for (int i = 1; i < size; i++) {
		displbuf[i] = displbuf[i-1] + recbuf[i-1];	
	}
	int logtotalsize = displbuf[size - 1] + recbuf[size - 1];

	
	char *logdata = (char*) malloc(logtotalsize * sizeof(char) + 1);
	memset(logdata, 0, logtotalsize * sizeof(char) + 1);
	
	MPI_Gatherv(sendlog, logsize, MPI_CHAR, logdata, recbuf, displbuf, MPI_CHAR, manager_rank, comm_world);


	free(sendlog);
	free(displbuf);
	free(recbuf);

	std::ofstream ofs2("workers.csv");
	ofs2 << logdata << std::endl;
	ofs2.close();

	//printf("%s\n", logdata);


	free(logdata);

}

void worker_process(const MPI_Comm &comm_world, const int manager_rank, const int rank,
		const MPI_Comm &comm_worker, const int modecode, const std::string &hostname,
		cciutils::SCIOADIOSWriter *writer, cciutils::SCIOLogger *logger) {
	int flag = MANAGER_READY;
	int inputSize;
	int outputSize;
	int maskSize;
	char *input;
	char *output;
	char *mask;
	MPI_Status status;

	int iocount = 0;

	MPI_Barrier(comm_world);
	uint64_t t0, t1;
	//printf("worker %d ready\n", rank);
	MPI_Barrier(comm_worker); // testing only

	std::string sessionName;
	int workerStatus[3];
	workerStatus[0] = WORKER_READY;
	workerStatus[1] = 0;
	workerStatus[2] = iocount;
	cciutils::SCIOLogSession *session;
	// per node
	if (logger) session = logger->getSession(hostname);
	if (writer) writer->setLogSession(session);
	bool first = true;

	long long t3, t2;

	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t0 = cciutils::ClockGetTime();

		t2 = ::cciutils::event::timestampInUS();

		if (writer != NULL) workerStatus[1] = writer->currentLoad();
		workerStatus[2] = iocount;

		// tell the manager - ready
		MPI_Send(&workerStatus, 3, MPI_INT, manager_rank, TAG_CONTROL, comm_world);
		//printf("worker %d signal ready\n", rank);
		// get the manager status
		MPI_Recv(&flag, 1, MPI_CHAR, manager_rank, TAG_CONTROL, comm_world, &status);
		//printf("worker %d received manager status %d\n", rank, flag);
		t3 = ::cciutils::event::timestampInUS();
		if (session != NULL) session->log(cciutils::event(90, std::string("worker message"), t2, t3, std::string(), ::cciutils::event::NETWORK_MSG));


		t2 = ::cciutils::event::timestampInUS();
		if (flag == MANAGER_READY) {

			// get data from manager
			MPI_Recv(&inputSize, 1, MPI::INT, manager_rank, TAG_METADATA, comm_world, &status);
			MPI_Recv(&maskSize, 1, MPI::INT, manager_rank, TAG_METADATA, comm_world, &status);
			MPI_Recv(&outputSize, 1, MPI::INT, manager_rank, TAG_METADATA, comm_world, &status);

			// allocate the buffers
			input = (char *)malloc(inputSize);
			mask = (char *)malloc(maskSize);
			output = (char *)malloc(outputSize);
			memset(input, 0, inputSize * sizeof(char));
			memset(mask, 0, maskSize * sizeof(char));
			memset(output, 0, outputSize * sizeof(char));

			// get the file names
			MPI_Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA, comm_world, &status);
			MPI_Recv(mask, maskSize, MPI::CHAR, manager_rank, TAG_DATA, comm_world, &status);
			MPI_Recv(output, outputSize, MPI::CHAR, manager_rank, TAG_DATA, comm_world, &status);
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("worker get work"), t2, t3, std::string(), ::cciutils::event::NETWORK_IO));

			t0 = cciutils::ClockGetTime();
//			printf("comm time for worker %d is %lu us\n", rank, t1 -t0);

			// per tile
			// session = logger->getSession(std::string(input));
			// per node
			session = logger->getSession(hostname);
			if (writer) writer->setLogSession(session);
			compute(input, mask, output, modecode, session, writer);
			// now do some work

			t1 = cciutils::ClockGetTime();
//			printf("worker %d processed \"%s\" + \"%s\" -> \"%s\" in %lu us\n", rank, input, mask, output, t1 - t0);
			printf("worker %d processed \"%s\" in %lu us\n", rank, input, t1 - t0);

			// clean up
			free(input);
			free(mask);
			free(output);

		} else if (flag == MANAGER_REQUEST_IO) {
			// do some IO.
			printf("iter %d manager-initiated IO for worker %d \n", iocount, rank);

			// per node
			session = logger->getSession(hostname);
			if (writer) writer->setLogSession(session);

			if (writer) writer->persist();
			++iocount;
		} else if (flag == MANAGER_WAIT) {
			//printf("manager told worker %d to wait\n", rank);
			t2 = ::cciutils::event::timestampInUS();

			usleep(100);
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("worker wait"), t2, t3, std::string(), ::cciutils::event::NETWORK_WAIT));

		} else if (flag == MANAGER_FINISHED) {
			//printf("manager told worker %d finished\n", rank);
			t2 = ::cciutils::event::timestampInUS();
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("manager finished"), t2, t3, std::string(), ::cciutils::event::NETWORK_WAIT));

		} else {
			printf("manager send unknown message %d to worker %d\n", flag, rank);
			t2 = ::cciutils::event::timestampInUS();
			usleep(100);
			t3 = ::cciutils::event::timestampInUS();
			if (session != NULL) session->log(cciutils::event(90, std::string("unknown or error"), t2, t3, std::string(), ::cciutils::event::OTHER));
		}
	}

	// printf("WORKER %d waiting for MPI barrier\n", rank);

	// manager is now done.  now do IO again
	//printf("worker %d final IO \n", rank);
	// per node
	session = logger->getSession(hostname);
	if (writer) writer->setLogSession(session);
	if (writer) writer->persist();
	// printf("written out data %d \n", rank);
	// last tiles were just written.  now add teh count informaton
	if (writer) writer->persistCountInfo();
	//printf("written out data count %d \n", rank);

	// now do collective io.
	MPI_Barrier(comm_world);


	// and do the logging.
	std::vector<std::string> timings = logger->toOneLineStrings();
	std::stringstream ss;
	for (int i = 0; i < timings.size(); i++) {
		ss << timings[i] << std::endl;	
	}
	std::string logstr = ss.str();
	int logsize = logstr.size();
	
	int *recbuf = NULL;

	// now send the thing to manager
	// 	first gather sizes
	MPI_Gather(&logsize, 1, MPI_INT, recbuf, 1, MPI_INT, manager_rank, comm_world);

	// 	then gatherv the messages.
	char *sendlog = (char *)malloc(sizeof(char) * logstr.size() + 1);
	memset(sendlog, 0, sizeof(char) * logstr.size() + 1);
	strncpy(sendlog, logstr.c_str(), logstr.size());

	char *logdata = NULL;
	int * displbuf = NULL;
	MPI_Gatherv(sendlog, logsize, MPI_CHAR, logdata, recbuf, displbuf, MPI_CHAR, manager_rank, comm_world);

	ss.str(std::string());
	
	free(sendlog);


}

int main (int argc, char **argv){

//printf("press a character followed by enter to continue\n");
//	char dummy[256];
//	std::cin >> dummy;

	// parse the input
	int modecode;
	std::string imageName, outDir, hostname;
	int status = parseInput(argc, argv, modecode, imageName, outDir);
	if (status != 0) return status;

	// set up mpi
	int rank, size, worker_size, worker_rank, manager_rank;
	MPI_Comm comm_world = init_mpi(argc, argv, size, rank, hostname);

	manager_rank = size - 1;

//	if (rank == manager_rank) {
//		printf("Using MPI.  if GPU is specified, will be changed to use CPU\n");
//	}

	if (modecode == cciutils::DEVICE_GPU) {
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
    		t1 = cciutils::ClockGetTime();
	    	getFiles(imageName, outDir, filenames, seg_output, bounds_output);

    		t2 = cciutils::ClockGetTime();
	    	printf("file read took %lu us\n", t2 - t1);

		total = filenames.size();
		printf("TOTAL FILES = %ld\n", total);
	}
	// then if MPI, broadcast it
	if (size > 1) {
		MPI_Bcast(&total, 1, MPI_INT, manager_rank, comm_world);
	}



	/* now perform the computation
	*/
	cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(rank, hostname);
	cciutils::SCIOLogSession *session = logger->getSession(hostname);
	cciutils::ADIOSManager *iomanager = new cciutils::ADIOSManager("adios_xml/image-tiles-globalarray.xml", rank, &comm_world, session);
	if (size == 1) {

	    int i = 0;


		// worker bees.  set to overwrite (third param set to true).

		cciutils::SCIOADIOSWriter *writer = iomanager->allocateWriter(outDir, std::string("bp"), true,
				stages, total, total * (long)256, total * (long)1024, total * (long)(4096 * 4096 * 4),
				rank, &comm_world);

		t1 = cciutils::ClockGetTime();

		while (i < total) {
			// per tile:
			// session = logger->getSession(filenames[i]);
			// per node
			session = logger->getSession(hostname);
			if (writer) writer->setLogSession(session);

			compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode, session, writer);

			printf("processed %s\n", filenames[i].c_str());
			++i;
		}
		t2 = cciutils::ClockGetTime();
		printf("WORKER %d: FINISHED using CPU in %lu us\n", rank, t2 - t1);

		if (writer) writer->persist();
		if (writer) writer->persistCountInfo();

		iomanager->freeWriter(writer);


		std::vector<std::string> timings = logger->toOneLineStrings();

		for (int i = 0; i < timings.size(); i++) {
			printf("%s\n", timings[i].c_str());
		}


	} else {

		// initialize the worker comm object

		// used by adios
		MPI_Comm comm_worker = init_workers(comm_world, manager_rank, worker_size, worker_rank);

		t1 = cciutils::ClockGetTime();

		// decide based on rank of worker which way to process
		if (rank == manager_rank) {
			// manager thread
			manager_process(comm_world, manager_rank, worker_size, hostname, filenames, seg_output, bounds_output, logger);
			t2 = cciutils::ClockGetTime();
			printf("MANAGER %d : FINISHED in %lu us\n", rank, t2 - t1);

		} else {
			// worker bees.  set to overwrite (third param set to true).
			cciutils::SCIOADIOSWriter *writer = iomanager->allocateWriter(outDir, std::string("bp"), true,
				stages, total, total * (long)256, total * (long)1024, total * (long)(4096 * 4096 * 4),
					worker_rank, &comm_worker);

			worker_process(comm_world, manager_rank, rank, comm_worker, modecode, hostname, writer, logger);
			t2 = cciutils::ClockGetTime();
			printf("WORKER %d: FINISHED using CPU in %lu us\n", rank, t2 - t1);

			iomanager->freeWriter(writer);

		}
		MPI_Comm_free(&comm_worker);

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

