/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
//#include "opencv2/opencv.hpp"
//#include "opencv2/gpu/gpu.hpp"
#include <iostream>
//#include <stdio.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include <string.h>
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>
#include "SCIOUtilsLogger.h"
#include "SCIOUtilsCVImageIO.h"
#include "SCIOHistologicalEntities.h"

using namespace cv;


// COMMENT OUT WHEN COMPILE for editing purpose only.
//#define WITH_MPI


#if defined (WITH_MPI)
#include <mpi.h>
#endif

#if defined (_OPENMP)
#include <omp.h>
#endif



void printUsage(char **argv);
void parseStages(const char *stagestr, std::vector<int> &stages);
int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir, int &imageCount, std::vector<int> &stages);
void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output, const int &imageCount);
void compute(const char *input, const char *mask, const char *output, const int modecode, cciutils::SCIOLogSession *session, const std::vector<int> &stages);

void printUsage(char **argv) {
	std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir [imagecount] [stages,...] [cpu [numThreads] | gpu [numThreads] [id]] " << std::endl;
	std::cout << "\t<image_filename | image_dir>: either an image filename or an image directory" << std::endl;
	std::cout << "\tmask_dir: output directory" << std::endl;
	std::cout << "\timagecount: number of images to process.  -1 means all images." << std::endl;
	std::cout << "\tstages: the stages to capture.  syntax is a comma separated ranges.  Range could be a single value or a dash (-) separated range.  range is of form [...) " << std::endl;
	std::cout << "\tcpu [numThreads]: use CPU computation.  number of threads relevant if compiled with OpenMP" << std::endl;
	std::cout << "\tgpu [numThreads] [id]]: use GPU for computation.  number of threads relevant if compile with OpenMP.  id is the device ID." << std::endl;
}

void parseStages(const char* stagestr, std::vector<int> &stages) {
	const char *newpos = strchr(stagestr, ',');
	// parse  (newpos is pointing to ',', so don't count that.)
	int length = (newpos == NULL ? strlen(stagestr) : newpos - stagestr);
	char *token = new char[length + 1];
	memset(token, 0, length + 1);
	strncpy(token, stagestr, length);

	char *rangeSep = strchr(token, '-');
	if (rangeSep == NULL) stages.push_back(atoi(token));
	else {
		*rangeSep = 0;  // create a separator
		int start = atoi(token);
		int end = atoi(rangeSep + 1);

		for (int s = start; s < end; ++s) {
			stages.push_back(s);
		}
	}

	delete [] token;

	if (newpos != NULL) {
		// recurse
		parseStages(newpos + 1, stages);
	}
}

int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir, int &imageCount, std::vector<int> &stages) {
	if (argc < 4) {
		printUsage(argv);
		return -1;
	}
	imageName.assign(argv[1]);
	outDir.assign(argv[2]);
	if (argc > 3) imageCount = atoi(argv[3]);
	if (argc > 4) {
		parseStages(argv[4], stages);
	} else {
		for (int stage = 0; stage <= 200; ++stage) {
			stages.push_back(stage);
		}
	}
	const char* mode = argc > 5 ? argv[5] : "cpu";

	int threadCount;
	if (argc > 6) threadCount = atoi(argv[6]);
	else threadCount = 1;

#if defined (WITH_MPI)
	threadCount = 1;
#endif

	printf("number of threads: %d\n", threadCount);

#if defined (_OPENMP)
	omp_set_num_threads(threadCount);
#endif

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		// get core count

	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
#if defined (_OPENMP)
	omp_set_num_threads(1);
#endif
		if (argc > 7) {
			gpu::setDevice(atoi(argv[7]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		printUsage(argv);
		return -1;
	}

	std::ostream_iterator< int > output( std::cout, " " );
	std::cout << "Selected stages: ";
	std::copy( stages.begin(), stages.end(), output );
	std::cout << std::endl;

	return 0;
}


void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output, const int &imageCount) {

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

	srand(0);
	if (imageCount != -1) {
		// randomize the file order.
		std::random_shuffle( filenames.begin(), filenames.end() );
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




void compute(const char *input, const char *mask, const char *output, const int modecode, cciutils::SCIOLogSession *session, const std::vector<int> &stages) {
	// compute

	int status;
	int *bbox = NULL;
	int compcount;

	FileUtils fu(".mask.pbm");
	std::string fmask(mask);

	std::string prefix = fu.replaceExt(fmask, ".mask.pbm", "");
	std::string suffix;
	suffix.assign(".mask.pbm");
	::cciutils::cv::IntermediateResultHandler *iwrite = new ::cciutils::cv::IntermediateResultWriter(prefix, suffix, stages);


	if (modecode == cciutils::DEVICE_GPU ) {
		nscale::gpu::SCIOHistologicalEntities *seg2 = new nscale::gpu::SCIOHistologicalEntities(std::string(input));
		status = seg2->segmentNuclei(std::string(input), std::string(mask), compcount, bbox, NULL, session, iwrite);
		delete seg2;

	} else {
	
		nscale::SCIOHistologicalEntities *seg = new nscale::SCIOHistologicalEntities(std::string(input));
		status = seg->segmentNuclei(std::string(input), std::string(mask), compcount, bbox, session, iwrite);
		delete seg;
	}
	

	delete iwrite;

	free(bbox);


}



#if defined (WITH_MPI)
MPI_Comm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI_Comm init_workers(const MPI_Comm &comm_world, int managerid, int &worker_size, int &worker_rank);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &imageName, std::string &outDir, const int imageCount);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, const int modecode, const std::string &hostname, const std::vector<int> &stages);


// initialize MPI
MPI_Comm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname) {
    int ierr = MPI_Init(&argc, &argv);

    char * temp = new char[256];
    gethostname(temp, 255);
    hostname.assign(temp);
    delete [] temp;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return MPI_COMM_WORLD;
}

// not necessary to create a new comm object
MPI_Comm init_workers(const MPI_Comm &comm_world, int managerid, int &worker_size, int &worker_rank) {

	int rank, size;
	MPI_Comm_size(comm_world, &size);
	MPI_Comm_rank(comm_world, &rank);
	
	// create new group from old group
	worker_size = size - 1;
	int *workers = (int*) malloc(worker_size * sizeof(int));
	for (int i = 0, id = 0; i < worker_size; ++i, ++id) {
		if (id == managerid) ++id;  // skip the manager id
		workers[i] = id;
	}
	// get old group
	MPI_Group world_group;
	MPI_Comm_group ( comm_world, &world_group );

	MPI_Group worker_group;
	MPI_Group_incl ( world_group, worker_size, workers, &worker_group );
	free(workers);
	
	MPI_Comm comm_worker;
	MPI_Comm_create(comm_world, worker_group, &comm_worker);
	
	if (rank != managerid) {
		MPI_Comm_size(comm_worker, &worker_size);
	    	MPI_Comm_rank(comm_worker, &worker_rank);
	} else {
		worker_rank = -1;
	}
	return comm_worker;
}

static const char MANAGER_READY = 10;
static const char MANAGER_FINISHED = 12;
static const char MANAGER_ERROR = -11;
static const char WORKER_READY = 20;
static const char WORKER_PROCESSING = 21;
static const char WORKER_ERROR = -21;
static const int TAG_CONTROL = 0;
static const int TAG_DATA = 1;
static const int TAG_METADATA = 2;
void manager_process(const MPI_Comm &comm_world, const int manager_rank, const int worker_size, std::string &maskName, std::string &outDir, const int imageCount) {
	// first get the list of files to process
   	std::vector<std::string> filenames;
	std::vector<std::string> seg_output;
	std::vector<std::string> bounds_output;
	uint64_t t1, t0;

	t0 = cciutils::ClockGetTime();
	getFiles(maskName, outDir, filenames, seg_output, bounds_output, imageCount);

	t1 = cciutils::ClockGetTime();
	printf("Manager ready at %d, file read took %lu us\n", manager_rank, t1 - t0);
	MPI_Barrier(comm_world);

	// now start the loop to listen for messages
	int curr = 0;
	int total = (imageCount == -1) ? filenames.size() : (imageCount > filenames.size() ? filenames.size() : imageCount);
	MPI_Status status;
	int worker_id;
	char ready;
	char *input;
	char *mask;
	char *output;
	int inputlen;
	int masklen;
	int outputlen;
	int hasMessage;
	char managerStatus;
	while (curr < total) {
		usleep(1000);

		managerStatus = MANAGER_READY;

		MPI_Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, comm_world, &hasMessage, &status);
		if (hasMessage != 0) {
/* where is it coming from */
			worker_id = status.MPI_SOURCE;
			MPI_Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world, &status);
//			printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				// tell worker that manager is ready
				MPI_Send(&managerStatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
//				printf("manager signal transfer\n");
/* send real data */
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
				curr++;

				delete [] input;
				delete [] mask;
				delete [] output;

			}

			if (curr % 100 == 1) {
				printf("[ MANAGER STATUS ] %d tasks remaining.\n", total - curr);
			}

		}
	}


	managerStatus = MANAGER_FINISHED;
/* tell everyone to quit */
	int active_workers = worker_size;
	while (active_workers > 0) {
		usleep(1000);

		MPI_Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, comm_world, &hasMessage, &status);
		if (hasMessage != 0) {
		/* where is it coming from */
			worker_id=status.MPI_SOURCE;
			MPI_Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world, &status);
//			printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				MPI_Send(&managerStatus, 1, MPI::CHAR, worker_id, TAG_CONTROL, comm_world);
//				printf("manager signal finished\n");
				--active_workers;
			}
		}
	}

	MPI_Barrier(comm_world);


	// now do a collective io for the log
	int logsize = 0;
	int world_size;
	MPI_Comm_size(comm_world, &world_size);

	int *recbuf = (int *) malloc(world_size * sizeof(int));

	// now send the thing to manager
	// 	first gather sizes
	MPI_Gather(&logsize, 1, MPI_INT, recbuf, 1, MPI_INT, manager_rank, comm_world);
	
	// then perform exclusive prefix sum to get the displacement and the total length
	int *displbuf = (int *) malloc(world_size * sizeof(int));
	displbuf[0] = 0;
	for (int i = 1; i < world_size; i++) {
		displbuf[i] = displbuf[i-1] + recbuf[i-1];	
	}
	int logtotalsize = displbuf[world_size - 1] + recbuf[world_size - 1];
	
	char *logdata = (char*) malloc(logtotalsize * sizeof(char) + 1);
	memset(logdata, 0, logtotalsize * sizeof(char) + 1);
	
	char *sendlog = (char*) malloc(sizeof(char))	;
	sendlog[0] = 0;

	MPI_Gatherv(sendlog, logsize, MPI_CHAR, logdata, recbuf, displbuf, MPI_CHAR, manager_rank, comm_world);

	printf("%s\n", logdata);


	free(logdata);

	free(displbuf);
	free(recbuf);
}

void worker_process(const MPI_Comm &comm_world, const int manager_rank, const int rank, const int modecode, const std::string &hostname, const std::vector<int> &stages) {
	char flag = MANAGER_READY;
	int inputSize;
	int outputSize;
	int maskSize;
	char *input;
	char *output;
	char *mask;
	MPI_Status status;

	MPI_Barrier(comm_world);
	uint64_t t0, t1, t2, t3;
	printf("worker %d ready\n", rank);

	cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(rank, hostname);
	char workerStatus = WORKER_READY;
	cciutils::SCIOLogSession *session;

	int count = 0;

	t2 = cciutils::ClockGetTime();
	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t0 = cciutils::ClockGetTime();

		// tell the manager - ready
		MPI_Send(&workerStatus, 1, MPI::CHAR, manager_rank, TAG_CONTROL, comm_world);
//		printf("worker %d signal ready\n", rank);
		// get the manager status
		MPI_Recv(&flag, 1, MPI::CHAR, manager_rank, TAG_CONTROL, comm_world, &status);
//		printf("worker %d received manager status %d\n", rank, flag);

		if (flag == MANAGER_READY) {
			// get data from manager
			MPI_Recv(&inputSize, 1, MPI::INT, manager_rank, TAG_METADATA, comm_world, &status);
			MPI_Recv(&maskSize, 1, MPI::INT, manager_rank, TAG_METADATA, comm_world, &status);
			MPI_Recv(&outputSize, 1, MPI::INT, manager_rank, TAG_METADATA, comm_world, &status);

			// allocate the buffers
			input = new char[inputSize];
			mask = new char[maskSize];
			output = new char[outputSize];
			memset(input, 0, inputSize * sizeof(char));
			memset(mask, 0, maskSize * sizeof(char));
			memset(output, 0, outputSize * sizeof(char));

			// get the file names
			MPI_Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA, comm_world, &status);
			MPI_Recv(mask, maskSize, MPI::CHAR, manager_rank, TAG_DATA, comm_world, &status);
			MPI_Recv(output, outputSize, MPI::CHAR, manager_rank, TAG_DATA, comm_world, &status);

			t0 = cciutils::ClockGetTime();
//			printf("comm time for worker %d is %lu us\n", rank, t1 -t0);

			session = logger->getSession(std::string(input));
			compute(input, mask, output, modecode, session, stages);
			++count;
			// now do some work

			t1 = cciutils::ClockGetTime();
//			printf("worker %d processed \"%s\" + \"%s\" -> \"%s\" in %lu us\n", rank, input, mask, output, t1 - t0);
			printf("worker %d processed \"%s\" in %lu us\n", rank, input, t1 - t0);

			// clean up
			delete [] input;
			delete [] mask;
			delete [] output;

		}
	}
	t3 = cciutils::ClockGetTime();

	// now do collective io.
	MPI_Barrier(comm_world);
	printf("worker %d processed %d jobs in %lu us\n", rank, count, t3 - t2);

	std::vector<std::string> timings = logger->toStrings();
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

	delete logger;
}

int main (int argc, char **argv){

	printf("Using MPI.  if GPU is specified, will be changed to use CPU\n");

   	std::vector<int> stages;

	// parse the input
	int modecode, imageCount;
	std::string imageName, outDir, hostname;
	int status = parseInput(argc, argv, modecode, imageName, outDir, imageCount, stages);
	if (status != 0) return status;

	// set up mpi
	int rank, size, worker_size, worker_rank, manager_rank;
	MPI_Comm comm_world = init_mpi(argc, argv, size, rank, hostname);


	if (size == 1) {
		printf("ERROR:  this program can only be run with 2 or more MPI nodes.  The head node does not process data\n");
		return -4;
	}

	if (modecode == cciutils::DEVICE_GPU) {
		printf("WARNING:  GPU specified for an MPI run.   only CPU is supported.  please restart with CPU as the flag.\n");
		return -4;
	}

	// initialize the worker comm object
	worker_size = size - 1;
	manager_rank = size - 1;

	// NOT NEEDED
	MPI_Comm comm_worker = init_workers(comm_world, manager_rank, worker_size, worker_rank);

	uint64_t t1 = 0, t2 = 0;
	t1 = cciutils::ClockGetTime();

	// decide based on rank of worker which way to process
	if (rank == manager_rank) {
		// manager thread
		manager_process(comm_world, manager_rank, worker_size, imageName, outDir, imageCount);
		t2 = cciutils::ClockGetTime();
		printf("MANAGER %d : FINISHED in %lu us\n", rank, t2 - t1);

	} else {
		// worker bees
		worker_process(comm_world, manager_rank, rank, modecode, hostname, stages);
		t2 = cciutils::ClockGetTime();
		printf("WORKER %d: FINISHED using CPU in %lu us\n", rank, t2 - t1);

	}
	MPI_Barrier(comm_world);
	MPI_Finalize();
	exit(0);

}


#else

    int main (int argc, char **argv){
    	printf("NOT compiled with MPI.  Using OPENMP if CPU, or GPU (multiple streams)\n");


       	std::vector<int> stages;

       	// parse the input
    	int modecode, imageCount;
    	std::string imageName, outDir, hostname;
    	int status = parseInput(argc, argv, modecode, imageName, outDir, imageCount, stages);
    	if (status != 0) return status;

    	uint64_t t0 = 0, t1 = 0, t2 = 0;
    	t1 = cciutils::ClockGetTime();

    	// first get the list of files to process
       	std::vector<std::string> filenames;
    	std::vector<std::string> seg_output;
    	std::vector<std::string> bounds_output;

    	t0 = cciutils::ClockGetTime();
    	getFiles(imageName, outDir, filenames, seg_output, bounds_output, imageCount);

    	t1 = cciutils::ClockGetTime();
    	printf("file read took %lu us\n", t1 - t0);

    	int total = (imageCount == -1) ? filenames.size() : (imageCount > filenames.size() ? filenames.size() : imageCount);
    	int i = 0;

    	// openmp bag of task
//#define _OPENMP
#if defined (_OPENMP)

    	if (omp_get_max_threads() == 1) {
        	printf("1 omp thread\n");
    		cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(0, std::string("localhost"));
		cciutils::SCIOLogSession *session;
	
        	while (i < total) {
			session = logger->getSession(filenames[i]);
        		compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode, session, stages);
        		printf("processed %s\n", filenames[i].c_str());
        		++i;
        	}
		std::vector<std::string> timings = logger->toStrings();
		for (int i = 0; i < timings.size(); i++) {
			printf("%s\n", timings[i].c_str());	
		}
        	delete logger;

    	} else {
        	printf("omp %d\n", omp_get_max_threads());
		cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(omp_get_thread_num(), std::string("localhost"));
#pragma omp parallel
    	{
#pragma omp single private(i)
    		{
    			

    			while (i < total) {
    				int ti = i;
    				// has to use firstprivate - private does not work.
				cciutils::SCIOLogSession *session = logger->getSession(filenames[ti]);
#pragma omp task firstprivate(ti, session) shared(filenames, seg_output, bounds_output, modecode)
    				{
//        				printf("t i: %d, %d \n", i, ti);
    					compute(filenames[ti].c_str(), seg_output[ti].c_str(), bounds_output[ti].c_str(), modecode, session, stages);
    	        		printf("processed %s\n", filenames[ti].c_str());
    				}
    				i++;
    			}
    		}
#pragma omp taskwait
    	}
		std::vector<std::string> timings = logger->toStrings();
		for (int i = 0; i < timings.size(); i++) {
			printf("%s\n", timings[i].c_str());	
		}
		delete logger;
    	}
#else
	cciutils::SCIOLogger *logger = new cciutils::SCIOLogger(0, std::string("localhost"));
	cciutils::SCIOLogSession *session;
    	printf("not omp\n");
    	while (i < total) {
		session = logger->getSession(filenames[i]);
    		compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode, session, stages);
    		printf("processed %s\n", filenames[i].c_str());
    		++i;
    	}
	std::vector<std::string> timings = logger->toStrings();
	for (int i = 0; i < timings.size(); i++) {
		printf("%s\n", timings[i].c_str());	
	}
    	delete logger;
#endif
		t2 = cciutils::ClockGetTime();
		printf("FINISHED in %lu us\n", t2 - t1);

    	return 0;
    }
#endif
