/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */


#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include "HistologicalEntities.h"
#include "MorphologicOperations.h"
#include "Logger.h"
#include "FileUtils.h"
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif

#ifndef _MSC_VER
#include <unistd.h>
#endif
using namespace cv;

#if defined(_WIN32) || defined(_WIN64)
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif


// COMMENT OUT WHEN COMPILE for editing purpose only.
//#define WITH_MPI


#if defined (WITH_MPI)
#include <mpi.h>
#endif

#if defined (_OPENMP)
#include <omp.h>
#endif




int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir);
void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output);
void compute(const char *input, const char *mask, const char *output, const int modecode);

int parseInput(int argc, char **argv, int &modecode, std::string &imageName, std::string &outDir) {
	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
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

	printf("number of threads: %d\n", threadCount);

#if defined (_OPENMP)
	omp_set_num_threads(threadCount);
#endif

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cci::common::type::DEVICE_CPU;
		// get core count


	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cci::common::type::DEVICE_MCORE;
		// get core count


	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cci::common::type::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
#if defined (_OPENMP)
	omp_set_num_threads(1);
#endif
		if (argc > 6) {
			gpu::setDevice(atoi(argv[6]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> image_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}


void getFiles(const std::string &imageName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &bounds_output) {

	// check to see if it's a directory or a file
	std::vector<std::string> exts;
	exts.push_back(std::string(".tif"));
	exts.push_back(std::string(".tiff"));
	exts.push_back(std::string(".png"));

	cci::common::FileUtils futils(exts);
	futils.traverseDirectory(imageName, filenames, cci::common::FileUtils::getFILE(), true);

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
		temp = futils.replaceExt(filenames[i], ".mask.png");
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




void compute(const char *input, const char *mask, const char *output, const int modecode) {
	// compute

	int status;
	int *bbox = NULL;
	int compcount;

	switch (modecode) {
	case cci::common::type::DEVICE_CPU :
	case cci::common::type::DEVICE_MCORE :
		status = nscale::HistologicalEntities::segmentNuclei(std::string(input), std::string(mask), compcount, bbox);

		break;
	case cci::common::type::DEVICE_GPU :
		status = nscale::gpu::HistologicalEntities::segmentNuclei(std::string(input), std::string(mask), compcount, bbox);
		break;
	default :
		break;
	}

	free(bbox);

#ifdef PRINT_CONTOUR_TEXT
	Mat out = imread(mask, 0);
	if (out.data > 0) {
		// for Lee and Jun to test the contour correctness.
		Mat temp = Mat::zeros(out.size() + Size(2,2), out.type());
		copyMakeBorder(out, temp, 1, 1, 1, 1, BORDER_CONSTANT, 0);

		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

		// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
		findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		//TODO: TEMP std::cout << "num contours = " << contours.size() << std::endl;

		std::ofstream fid(output);
		int counter = 0;
		if (contours.size() > 0) {
			// iterate over all top level contours (all siblings, draw with own label color
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
				fid << idx << ": ";
				for (unsigned int ptc = 0; ptc < contours[idx].size(); ++ptc) {
					fid << contours[idx][ptc].x << "," << contours[idx][ptc].y << "; ";
				}
				fid << std::endl;
			}
			++counter;
		}
		fid.close();
	}
#endif

}



#if defined (WITH_MPI)
MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI::Intracomm init_workers(const MPI::Intracomm &comm_world, int managerid);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &imageName, std::string &outDir);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, const int modecode);


// initialize MPI
MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname) {
    MPI::Init(argc, argv);

    char * temp = new char[256];
    gethostname(temp, 255);
    hostname.assign(temp);
    delete [] temp;

    size = MPI::COMM_WORLD.Get_size();
    rank = MPI::COMM_WORLD.Get_rank();

    return MPI::COMM_WORLD;
}

// not necessary to create a new comm object
MPI::Intracomm init_workers(const MPI::Intracomm &comm_world, int managerid) {
	// get old group
	MPI::Group world_group = comm_world.Get_group();
	// create new group from old group
	int worker_size = comm_world.Get_size() - 1;
	int *workers = new int[worker_size];
	for (int i = 0, id = 0; i < worker_size; ++i, ++id) {
		if (id == managerid) ++id;  // skip the manager id
		workers[i] = id;
	}
	MPI::Group worker_group = world_group.Incl(worker_size, workers);
	delete [] workers;
	return comm_world.Create(worker_group);
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
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName, std::string &outDir) {
	// first get the list of files to process
   	std::vector<std::string> filenames;
	std::vector<std::string> seg_output;
	std::vector<std::string> bounds_output;
	uint64_t t1, t0;

	t0 = cci::common::event::timestampInUS();
	getFiles(maskName, outDir, filenames, seg_output, bounds_output);

	t1 = cci::common::event::timestampInUS();
	printf("Manager ready at %d, file read took %lu us\n", manager_rank, t1 - t0);
	comm_world.Barrier();

	// now start the loop to listen for messages
	int curr = 0;
	int total = filenames.size();
	MPI::Status status;
	int worker_id;
	char ready;
	char *input;
	char *mask;
	char *output;
	int inputlen;
	int masklen;
	int outputlen;
	while (curr < total) {
		usleep(1000);

		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
//			printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				// tell worker that manager is ready
				comm_world.Send(&MANAGER_READY, 1, MPI::CHAR, worker_id, TAG_CONTROL);
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

				comm_world.Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA);
				comm_world.Send(&masklen, 1, MPI::INT, worker_id, TAG_METADATA);
				comm_world.Send(&outputlen, 1, MPI::INT, worker_id, TAG_METADATA);

				// now send the actual string data
				comm_world.Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA);
				comm_world.Send(mask, masklen, MPI::CHAR, worker_id, TAG_DATA);
				comm_world.Send(output, outputlen, MPI::CHAR, worker_id, TAG_DATA);
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
/* tell everyone to quit */
	int active_workers = worker_size;
	while (active_workers > 0) {
		usleep(1000);

		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
		/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
//			printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				comm_world.Send(&MANAGER_FINISHED, 1, MPI::CHAR, worker_id, TAG_CONTROL);
//				printf("manager signal finished\n");
				--active_workers;
			}
		}
	}

}

void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, const int modecode) {
	char flag = MANAGER_READY;
	int inputSize;
	int outputSize;
	int maskSize;
	char *input;
	char *output;
	char *mask;

	comm_world.Barrier();
	uint64_t t0, t1;
	printf("worker %d ready\n", rank);

	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t0 = cci::common::event::timestampInUS();

		// tell the manager - ready
		comm_world.Send(&WORKER_READY, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
//		printf("worker %d signal ready\n", rank);
		// get the manager status
		comm_world.Recv(&flag, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
//		printf("worker %d received manager status %d\n", rank, flag);

		if (flag == MANAGER_READY) {
			// get data from manager
			comm_world.Recv(&inputSize, 1, MPI::INT, manager_rank, TAG_METADATA);
			comm_world.Recv(&maskSize, 1, MPI::INT, manager_rank, TAG_METADATA);
			comm_world.Recv(&outputSize, 1, MPI::INT, manager_rank, TAG_METADATA);

			// allocate the buffers
			input = new char[inputSize];
			mask = new char[maskSize];
			output = new char[outputSize];
			memset(input, 0, inputSize * sizeof(char));
			memset(mask, 0, maskSize * sizeof(char));
			memset(output, 0, outputSize * sizeof(char));

			// get the file names
			comm_world.Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA);
			comm_world.Recv(mask, maskSize, MPI::CHAR, manager_rank, TAG_DATA);
			comm_world.Recv(output, outputSize, MPI::CHAR, manager_rank, TAG_DATA);

			t0 = cci::common::event::timestampInUS();
//			printf("comm time for worker %d is %lu us\n", rank, t1 -t0);


			compute(input, mask, output, modecode);
			// now do some work

			t1 = cci::common::event::timestampInUS();
//			printf("worker %d processed \"%s\" + \"%s\" -> \"%s\" in %lu us\n", rank, input, mask, output, t1 - t0);
			printf("worker %d processed \"%s\" in %lu us\n", rank, input, t1 - t0);

			// clean up
			delete [] input;
			delete [] mask;
			delete [] output;

		}
	}
}

int main (int argc, char **argv){

	printf("Using MPI.  if GPU is specified, will be changed to use CPU\n");

	// parse the input
	int modecode;
	std::string imageName, outDir, hostname;
	int status = parseInput(argc, argv, modecode, imageName, outDir);
	if (status != 0) return status;

	// set up mpi
	int rank, size, worker_size, manager_rank;
	MPI::Intracomm comm_world = init_mpi(argc, argv, size, rank, hostname);

	if (size == 1) {
		printf("ERROR:  this program can only be run with 2 or more MPI nodes.  The head node does not process data\n");
		return -4;
	}

	if (modecode == cci::common::type::DEVICE_GPU) {
		printf("WARNING:  GPU specified for an MPI run.   only CPU is supported.  please restart with CPU as the flag.\n");
		return -4;
	}

	// initialize the worker comm object
	worker_size = size - 1;
	manager_rank = size - 1;

	// NOT NEEDED
	//MPI::Intracomm comm_worker = init_workers(MPI::COMM_WORLD, manager_rank);
	//int worker_rank = comm_worker.Get_rank();


	uint64_t t1 = 0, t2 = 0;
	t1 = cci::common::event::timestampInUS();

	// decide based on rank of worker which way to process
	if (rank == manager_rank) {
		// manager thread
		manager_process(comm_world, manager_rank, worker_size, imageName, outDir);
		t2 = cci::common::event::timestampInUS();
		printf("MANAGER %d : FINISHED in %lu us\n", rank, t2 - t1);

	} else {
		// worker bees
		worker_process(comm_world, manager_rank, rank, modecode);
		t2 = cci::common::event::timestampInUS();
		printf("WORKER %d: FINISHED using CPU in %lu us\n", rank, t2 - t1);

	}
	comm_world.Barrier();
	MPI::Finalize();
	exit(0);

}


#else

    int main (int argc, char **argv){
    	printf("NOT compiled with MPI.  Using OPENMP if CPU, or GPU (multiple streams)\n");

    	// parse the input
    	int modecode;
    	std::string imageName, outDir, hostname;
    	int status = parseInput(argc, argv, modecode, imageName, outDir);
    	if (status != 0) return status;

    	uint64_t t0 = 0, t1 = 0, t2 = 0;
    	t1 = cci::common::event::timestampInUS();

    	// first get the list of files to process
       	std::vector<std::string> filenames;
    	std::vector<std::string> seg_output;
    	std::vector<std::string> bounds_output;

    	t0 = cci::common::event::timestampInUS();
    	getFiles(imageName, outDir, filenames, seg_output, bounds_output);

    	t1 = cci::common::event::timestampInUS();
    	printf("file read took %lu us\n", t1 - t0);

    	int total = filenames.size();
    	int i = 0;

    	// openmp bag of task
//#define _OPENMP
#if defined (_OPENMP)

    	if (omp_get_max_threads() == 1) {
        	printf("1 omp thread\n");
        	while (i < total) {
        		compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode);
        		printf("processed %s\n", filenames[i].c_str());
        		++i;
        	}

    	} else {
        	printf("omp %d\n", omp_get_max_threads());

#pragma omp parallel
    		{
#pragma omp single firstprivate(i)
    			{
    			while (i < total) {
    				int ti = i;
    				// has to use firstprivate - private does not work.
#ifndef _MSC_VER
					//MSVC does not support tasks. They were incorporated in OMP 3.0 and MSVC only supports up to 2.0 http://stackoverflow.com/questions/23545930/openmp-tasks-in-visual-studio
#pragma omp task firstprivate(ti) shared(filenames, seg_output, bounds_output, modecode)
#endif
    				{
//        				printf("t i: %d, %d \n", i, ti);
    					compute(filenames[ti].c_str(), seg_output[ti].c_str(), bounds_output[ti].c_str(), modecode);
    		        		printf("processed %s\n", filenames[ti].c_str());
    				}
    				i++;
    			}
    			}	
#ifndef _MSC_VER
#pragma omp taskwait
#endif
    		}
    	}
#else
    	printf("not omp\n");
    	while (i < total) {
    		compute(filenames[i].c_str(), seg_output[i].c_str(), bounds_output[i].c_str(), modecode);
    		printf("processed %s\n", filenames[i].c_str());
    		++i;
    	}
#endif
		t2 = cci::common::event::timestampInUS();
		printf("FINISHED in %lu us\n", t2 - t1);

    	return 0;
    }
#endif
