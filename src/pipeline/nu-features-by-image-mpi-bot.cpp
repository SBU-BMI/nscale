/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>
#include <sstream>
#include "opencv2/gpu/gpu.hpp"
#include "hdf5.h"
#include "hdf5_hl.h"

using namespace cv;

// COMMENT OUT WHEN COMPILE for editing purpose only.
//#define WITH_MPI

#ifdef WITH_MPI
#include <mpi.h>


MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI::Intracomm init_workers(const MPI::Intracomm &comm_world, int managerid);
int parseInput(int argc, char **argv, int &modecode, std::string &inputName);
void getFiles(const std::string &filename, std::vector<std::string> &filenames);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &inputName);
void worker_process(const MPI::Intracomm &comm_world, const MPI::Intracomm &comm_worker, const int manager_rank, const int rank);
void computeStage1(const char *input, double *node_sums, double *node_sum_squares, long &node_rows, const unsigned int n_cols);
void computeStage3(const char *input, double *global_sums, double *global_sum_squares, long &global_rows, const unsigned int n_cols);



int parseInput(int argc, char **argv, int &modecode, std::string &inputName) {
	if (argc < 3) {
		std::cout << "Usage:  " << argv[0] << " feature_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	inputName.assign(argv[1]);
	const char* mode = argc > 3 ? argv[3] : "cpu";

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		// get core count

	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
		if (argc > 5) {
			gpu::setDevice(atoi(argv[5]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <feature_filename | feature_dir> " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}


void getFiles(const std::string &filename, std::vector<std::string> &filenames) {

	std::string suffix;
	suffix.assign(".features.h5");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(filename, filenames);
	std::string dirname;
	if (filenames.size() == 1) {
		dirname = filename.substr(0, filename.find_last_of("/\\"));
	} else {
		dirname = filename;
	}

}



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

int main (int argc, char **argv){
	// parse the input
	int modecode;
	std::string inputName;
	int status = parseInput(argc, argv, modecode, inputName);
	if (status != 0) return status;

	// set up mpi
	int rank, size, worker_size, manager_rank;
	std::string hostname;
	MPI::Intracomm comm_world = init_mpi(argc, argv, size, rank, hostname);

	if (size == 1) {
		printf("ERROR:  this program can only be run with 2 or more MPI nodes.  The head node does not process data\n");
		return -4;
	}

	// initialize the worker comm object
	worker_size = size - 1;
	manager_rank = size - 1;

	MPI::Intracomm comm_worker = init_workers(MPI::COMM_WORLD, manager_rank);
	//int worker_rank = comm_worker.Get_rank();


	uint64_t t1 = 0, t2 = 0;
	t1 = cciutils::ClockGetTime();


	// decide based on rank of worker which way to process
	if (rank == manager_rank) {
		// manager thread
		manager_process(comm_world, manager_rank, worker_size, inputName);
		t2 = cciutils::ClockGetTime();
		printf("MANAGER %d : FINISHED in %lu us\n", rank, t2 - t1);

	} else {
		// worker bees
		worker_process(comm_world, comm_worker, manager_rank, rank);
		t2 = cciutils::ClockGetTime();
		printf("WORKER %d: FINISHED in %lu us\n", rank, t2 - t1);

	}
	comm_world.Barrier();
	MPI::Finalize();
	exit(0);

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
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &inputName) {
	// first get the list of files to process
   	std::vector<std::string> filenames;
	uint64_t t1, t0;

	t0 = cciutils::ClockGetTime();

	getFiles(inputName, filenames);

	t1 = cciutils::ClockGetTime();
	printf("Manager ready at %d, file read took %lu us\n", manager_rank, t1 - t0);

	// first part - get the global size of the num of columns, and broadcast
	unsigned int n_cols;

	hsize_t dims[2];
	// open the file
	hid_t file_id = H5Fopen ( filenames[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
	herr_t hstatus = H5LTget_dataset_info ( file_id, "/data", dims, NULL, NULL );
	n_cols = dims[1];
	H5Fclose ( file_id );

	// simulate a broadcast
	int size = comm_world.Get_size();
	for (int i = 0; i < size; ++i) {
		if (i == manager_rank) continue;
		comm_world.Send(&n_cols, 1, MPI::INT, i, TAG_METADATA);
	}


	comm_world.Barrier();

	// now start the loop to listen for messages
	int curr = 0;
	int total = filenames.size();
	MPI::Status status;
	int worker_id;
	char ready;
	char *input;
	int inputlen;
	while (curr < total) {
		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
			printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				// tell worker that manager is ready
				comm_world.Send(&MANAGER_READY, 1, MPI::CHAR, worker_id, TAG_CONTROL);
				printf("manager signal transfer\n");
/* send real data */
				inputlen = filenames[curr].size() + 1;  // add one to create the zero-terminated string
				input = new char[inputlen];
				memset(input, 0, sizeof(char) * inputlen);
				strncpy(input, filenames[curr].c_str(), inputlen);

				comm_world.Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA);

				// now send the actual string data
				comm_world.Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA);
				curr++;

				delete [] input;

			}
		}

		if (curr % 100 == 1) {
			printf("[ MANAGER STATUS ] %d tasks remaining in pass 1.\n", total - curr);
		}

	}
/* tell everyone to quit */
	int active_workers = worker_size;
	while (active_workers > 0) {
		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
		/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
			printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				comm_world.Send(&MANAGER_FINISHED, 1, MPI::CHAR, worker_id, TAG_CONTROL);
				printf("manager signal finished\n");
				--active_workers;
			}
		}
	}

	comm_world.Barrier();

	// worker is doing some worker stuff right now...

	// now listen for worker again.
	curr = 0;
	while (curr < total) {
		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
			printf("manager received fle update request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				// tell worker that manager is ready
				comm_world.Send(&MANAGER_READY, 1, MPI::CHAR, worker_id, TAG_CONTROL);
				printf("manager signal transfer for file update\n");
/* send real data */
				inputlen = filenames[curr].size() + 1;  // add one to create the zero-terminated string
				input = new char[inputlen];
				memset(input, 0, sizeof(char) * inputlen);
				strncpy(input, filenames[curr].c_str(), inputlen);

				comm_world.Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA);

				// now send the actual string data
				comm_world.Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA);
				curr++;

				delete [] input;

			}
		}
		if (curr % 100 == 1) {
			printf("[ MANAGER STATUS ] %d tasks remaining in pass 2.\n", total - curr);
		}

	}
/* tell everyone to quit */
	active_workers = worker_size;
	while (active_workers > 0) {
		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
		/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
			printf("manager received file update request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				comm_world.Send(&MANAGER_FINISHED, 1, MPI::CHAR, worker_id, TAG_CONTROL);
				printf("manager signal finished file update\n");
				--active_workers;
			}
		}
	}


}

void worker_process(const MPI::Intracomm &comm_world, const MPI::Intracomm &comm_worker, const int manager_rank, const int rank) {
	char flag = MANAGER_READY;
	int inputSize;
	char *input;

	// receive the broadcasted number of columns
	unsigned int n_cols;
	comm_world.Recv(&n_cols, 1, MPI::INT, manager_rank, TAG_METADATA);

	double *node_sums = new double[n_cols];
	double *node_sum_squares = new double[n_cols];
	long node_rows = 0;
	for (unsigned int i = 0; i < n_cols; i++) {
		node_sums[i] = 0.;
		node_sum_squares[i] = 0.;
	}


	comm_world.Barrier();
	uint64_t t2, t1;

	// now get the individual files to update
	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t1 = cciutils::ClockGetTime();

		// tell the manager - ready
		comm_world.Send(&WORKER_READY, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
		printf("worker %d signal ready\n", rank);
		// get the manager status
		comm_world.Recv(&flag, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
		printf("worker %d received manager status %d\n", rank, flag);

		if (flag == MANAGER_READY) {
			// get data from manager
			comm_world.Recv(&inputSize, 1, MPI::INT, manager_rank, TAG_METADATA);

			// allocate the buffers
			input = new char[inputSize];
			memset(input, 0, inputSize * sizeof(char));

			// get the file names
			comm_world.Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA);

			t2 = cciutils::ClockGetTime();
			printf("comm time for worker %d is %lu us\n", rank, t2 -t1);

			// now do some work
			computeStage1(input, node_sums, node_sum_squares, node_rows, n_cols);

			t2 = cciutils::ClockGetTime();
			printf("worker %d processed \"%s\" in %lu us\n", rank, input, t2 - t1);

			// clean up
			delete [] input;

		}
	}

	comm_world.Barrier();

	double *global_sums = new double[n_cols];
	double *global_sum_squares = new double[n_cols];
	long global_rows = 0;
	for (unsigned int i = 0; i < n_cols; i++) {
		global_sums[i] = 0.;
		global_sum_squares[i] = 0.;
	}

	// COMPUTE STAGE 2
	t1 = cciutils::ClockGetTime();

	// now that each node has its own data,  reduce within the worker communicator
	comm_worker.Allreduce(&node_rows, &global_rows, 1, MPI::INT, MPI::SUM);

	printf("worker %d total rows: %lu, node rows: %lu\n", rank, global_rows, node_rows);
	// compute the local contribution to the mean
	std::stringstream ss;

	for (unsigned int j = 0; j < n_cols; ++j) {
		node_sums[j] /= (double)global_rows;
		ss<< node_sums[j] << ",";
	}
	printf("node means: %s \n", ss.str().c_str());
	ss.str(std::string());
	ss.clear();

	// global reduction to get global mean
	comm_worker.Allreduce(node_sums, global_sums, n_cols, MPI::DOUBLE, MPI::SUM);

	for (unsigned int j = 0; j < n_cols; ++j) {
		ss<< global_sums[j] << ",";
	}
	printf("worker %d global means: %s \n", rank, ss.str().c_str());
	ss.str(std::string());
	ss.clear();

	for (unsigned int j = 0; j < n_cols; ++j) {
		node_sum_squares[j] /= (double)global_rows;
		ss<< node_sum_squares[j] << ",";
	}
	printf("node vars: %s \n", ss.str().c_str());
	ss.str(std::string());
	ss.clear();

	comm_worker.Allreduce(node_sum_squares, global_sum_squares, n_cols, MPI::DOUBLE, MPI::SUM);


	for (unsigned int j = 0; j < n_cols; ++j) {
		global_sum_squares[j] -= global_sums[j] * global_sums[j];
		global_sum_squares[j] = sqrt(global_sum_squares[j]);
		ss<<  global_sum_squares[j] << ",";
	}
	if (rank == 0) printf("global stdev: %s \n", ss.str().c_str());
	ss.str(std::string());
	ss.clear();

	t2 = cciutils::ClockGetTime();
	printf("worker %d global summarize took %lu us\n", rank, t2-t1);

	delete [] node_sums;
	delete [] node_sum_squares;

	comm_worker.Barrier();

	flag = MANAGER_READY;
	// update the individual files again
	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t1 = cciutils::ClockGetTime();

		// tell the manager - ready
		comm_world.Send(&WORKER_READY, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
		printf("worker %d signal ready for file update\n", rank);
		// get the manager status
		comm_world.Recv(&flag, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
		printf("worker %d during file update received manager status %d\n", rank, flag);

		if (flag == MANAGER_READY) {
			// get data from manager
			comm_world.Recv(&inputSize, 1, MPI::INT, manager_rank, TAG_METADATA);

			// allocate the buffers
			input = new char[inputSize];
			memset(input, 0, inputSize * sizeof(char));

			// get the file names
			comm_world.Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA);

			t2 = cciutils::ClockGetTime();
			printf("comm time for worker %d is %lu us\n", rank, t2 -t1);

			// now do some work
			computeStage3(input, global_sums, global_sum_squares, global_rows, n_cols);

			t2 = cciutils::ClockGetTime();
			printf("worker %d updated file \"%s\" in %lu us\n", rank, input, t2 - t1);

			// clean up
			delete [] input;

		}
	}

	delete [] global_sums;
	delete [] global_sum_squares;
}





void computeStage1(const char *input, double *node_sums, double *node_sum_squares, long &node_rows, const unsigned int n_cols) {
	if (strcmp(input, "") == 0) return;

	double t;
	//double sum, sum_square;
	double *file_sums = new double[n_cols];
	double *file_sum_squares = new double[n_cols];

	// open the file
	hsize_t ldims[2];
	hid_t file_id = H5Fopen ( input, H5F_ACC_RDONLY, H5P_DEFAULT );
	herr_t hstatus = H5LTget_dataset_info ( file_id, "/data", ldims, NULL, NULL );
	unsigned int n_rows = ldims[0];
	float *data = new float[n_rows * n_cols];
	H5LTread_dataset (file_id, "/data", H5T_NATIVE_FLOAT, data);
	H5Fclose ( file_id );

	// compute the per file variables
	for (unsigned int j = 0; j < n_cols; j++) {
		file_sums[j] = 0.;
		file_sum_squares[j] = 0.;
	}

	float *currdata = data;
	for (unsigned int k = 0; k < n_rows; k++) {

		currdata = data + k*n_cols;
		for (unsigned int j = 0; j < n_cols; j++) {
			t = currdata[j];
			file_sums[j] += t;
			file_sum_squares[j] += (t * t);
		}
	}

	// open same file again and update the attributes
	// write out to file
	file_id = H5Fopen ( input, H5F_ACC_RDWR, H5P_DEFAULT );
	hstatus = H5LTset_attribute_double(file_id, "/data", "sums", file_sums, n_cols);
	hstatus = H5LTset_attribute_double(file_id, "/data", "square_sums", file_sum_squares, n_cols);
	H5Fclose ( file_id );

	// update the per node summary
	node_rows += n_rows;
	for (unsigned int j = 0; j < n_cols; j++) {
		node_sums[j] += file_sums[j];
		node_sum_squares[j] += file_sum_squares[j];
	}

	delete [] file_sums;
	delete [] file_sum_squares;
	delete [] data;
}

void computeStage3(const char *input, double *global_sums, double *global_sum_squares, long &global_rows, const unsigned int n_cols) {
	if (strcmp(input, "") == 0) return;


	// open same file again and update the attributes
//	hsize_t ldims[2];
	hid_t file_id = H5Fopen ( input, H5F_ACC_RDWR, H5P_DEFAULT );
	herr_t hstatus = H5LTset_attribute_long(file_id, "/data", "global_count", &global_rows, 1);
	hstatus = H5LTset_attribute_double(file_id, "/data", "global_means", global_sums, n_cols);
	hstatus = H5LTset_attribute_double(file_id, "/data", "global_stdevs", global_sum_squares, n_cols);
	H5Fclose ( file_id );

}


#else
int main (int argc, char **argv){
	printf("THIS PROGRAM REQUIRES MPI.  PLEASE RECOMPILE WITH MPI ENABLED.  EXITING\n");
	return -1;
}
#endif




	/*		// now normalize the data
			double t;

	//#pragma omp parallel for private(j, k, t, sum, sum_square) shared(file_sums, file_sum_squares, n_cols, data)
			for (j = 0; j < n_cols; j++) {

	//#pragma omp parallel for private(k, t) shared(n_cols, data) reduction(+: sum, sum_square)
				for (k = 0; k < n_rows; k++) {
					t = (double)(data[k * n_cols + j]);
					t -= global_sums[j];
					t /= global_sum_squares[j];
					data[k * n_cols + j] = (float)t;
				}

			}

	#pragma omp critical
			{
			// write out to file
			file_id = H5Fopen ( fin.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );

			hstatus = H5LTmake_dataset ( file_id, "/normal_data",
								2, // rank
								ldims, // dims
									 H5T_NATIVE_FLOAT, data );
			H5Fclose ( file_id );

			}
	*/


