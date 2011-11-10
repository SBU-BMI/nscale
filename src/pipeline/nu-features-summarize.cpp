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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WITH_MPI
#include <mpi.h>
#endif

using namespace cv;





int main (int argc, char **argv){

#ifdef WITH_MPI
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
    printf( " MPI enabled: rank %d \n", rank);
#else
    int size = 1;
    int rank = 0;
    printf( " MPI disabled\n");
#endif

    // relevant to head node only
    std::vector<std::string> filenames;
	char *inputBufAll;
	inputBufAll=NULL;
	int dataCount;

	// relevant to all nodes
	int modecode = 0;
	uint64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	std::string fin;
	unsigned int perNodeCount=0, maxLenInput=0;
	char *inputBuf;
	inputBuf=NULL;

	if (argc < 3) {
		std::cout << "Usage:  " << argv[0] << " <feature_filename | feature_dir> " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	std::string filename(argv[1]);
	std::string runid(argv[2]);
	const char* mode = argc > 3 ? argv[3] : "cpu";

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		// get core count

#ifdef _OPENMP
		if (argc > 4) {
	//		omp_set_num_threads(atoi(argv[4]) > omp_get_max_threads() ? omp_get_max_threads() : atoi(argv[4]));
			omp_set_num_threads(atoi(argv[4]));	
		printf("number of threads used = %d\n", omp_get_num_threads());
		}
#endif
	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
#ifdef _OPENMP
		if (argc > 4) {
//			omp_set_num_threads(atoi(argv[4]) > omp_get_max_threads() ? omp_get_max_threads() : atoi(argv[4]));
			omp_set_num_threads(atoi(argv[4]));
			printf("number of threads used = %d\n", omp_get_num_threads());
		}
#endif
	} else if (strcasecmp(mode, "gpu") == 0) {
		modecode = cciutils::DEVICE_GPU;
		// get device count
		int numGPU = gpu::getCudaEnabledDeviceCount();
		if (numGPU < 1) {
			printf("gpu requested, but no gpu available.  please use cpu or mcore option.\n");
			return -2;
		}
		if (argc > 4) {
			gpu::setDevice(atoi(argv[4]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <feature_filename | feature_dir> " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	std::string aggregate;
	if (rank == 0) {
		// check to see if it's a directory or a file
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


		std::string temp, tempdir;
		for (unsigned int i = 0; i < filenames.size(); ++i) {
				// generate the input file name
			maxLenInput = maxLenInput > filenames[i].length() ? maxLenInput : filenames[i].length();
		}
		dataCount= filenames.size();
		
		aggregate.assign(filename);
		aggregate.append("/");
		aggregate.append(runid);
		aggregate.append(".features.h5");
	}

	hid_t file_id;
	herr_t hstatus;
	hsize_t dims[2];
	unsigned int n_rows, n_cols;


	if (rank == 0) {
		// open the file
		file_id = H5Fopen ( filenames[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
		hstatus = H5LTget_dataset_info ( file_id, "/data", dims, NULL, NULL );
		n_cols = dims[1];
		H5Fclose ( file_id );


		// and create the file
		

	}

#ifdef WITH_MPI
	if (rank == 0) {
		printf("headnode: total count is %d, size is %d\n", filenames.size(), size);

		perNodeCount = filenames.size() / size + (filenames.size() % size == 0 ? 0 : 1);

		printf("headnode: rank is %d here.  perNodeCount is %d, inputLen %d \n", rank, perNodeCount, maxLenInput);

		// allocate the sendbuffer
		inputBufAll= (char*)malloc(perNodeCount * size * maxLenInput * sizeof(char));
		memset(inputBufAll, 0, perNodeCount * size * maxLenInput);

		// copy data into the buffers
		for (unsigned int i = 0; i < filenames.size(); ++i) {
			strncpy(inputBufAll + i * maxLenInput, filenames[i].c_str(), maxLenInput);
		}
	}
	//	printf("rank: %d\n ", rank);
	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Bcast(&n_cols, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&perNodeCount, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&maxLenInput, 1, MPI::INT, 0);


	printf("rank is %d here.  perNodeCount is %d, inputLen %d \n", rank, perNodeCount, maxLenInput);

	// allocate the receive buffer
	inputBuf = (char*)malloc(perNodeCount * maxLenInput * sizeof(char));

	// scatter
	MPI::COMM_WORLD.Scatter(inputBufAll, perNodeCount * maxLenInput, MPI::CHAR,
		inputBuf, perNodeCount * maxLenInput, MPI::CHAR,
		0);

	MPI::COMM_WORLD.Barrier();

#endif
	if (rank == 0)	t3 = cciutils::ClockGetTime();

	double *node_sums = new double[n_cols];
	double *node_sum_squares = new double[n_cols];
	double *global_sums = new double[n_cols];
	double *global_sum_squares = new double[n_cols];
	unsigned int i, j, k;
	for (i = 0; i < n_cols; i++) {
		node_sums[i] = 0.;
		node_sum_squares[i] = 0.;
		global_sums[i] = 0.;
		global_sum_squares[i] = 0.;
	}

	long global_rows = 0;
	long node_rows = 0;


// this program is mostly doing file io.  don't use openmp because we are not using parallel hdf5
#ifdef WITH_MPI
#pragma omp parallel for shared(perNodeCount, inputBuf, maxLenInput, rank, n_cols, node_sums, node_sum_squares) private(file_id, hstatus, n_rows, i, j, k, fin, t1, t2) reduction(+: node_rows)
    for (i = 0; i < perNodeCount; ++i) {
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
		printf("in MPI feature summary loop with rank %d, loop %d.  \"%s\"\n", rank, i, fin.c_str());

#else
#pragma omp parallel for shared(filenames, rank, n_cols, node_sums, node_sum_squares) private(file_id, hstatus, n_rows, i, j, k, fin, t1, t2) reduction(+: node_rows)
    for (i = 0; i < dataCount; ++i) {
		fin = filenames[i];
#endif

		t1 = cciutils::ClockGetTime();

#ifdef _OPENMP
    	int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif

		if (strcmp(fin.c_str(), "") == 0) continue;
		float *data;
		// open the file
		hsize_t ldims[2];
#pragma omp critical
		{
			file_id = H5Fopen ( fin.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
			hstatus = H5LTget_dataset_info ( file_id, "/data", ldims, NULL, NULL );
			n_rows = ldims[0];
			data = new float[n_rows * n_cols];

			H5LTread_dataset (file_id, "/data", H5T_NATIVE_FLOAT, data);
			H5Fclose ( file_id );
		}
//		t2 = cciutils::ClockGetTime();
		//printf("file read took %lu us for %s\n", t2-t1, fin.c_str());
//		t1 = cciutils::ClockGetTime();

		double t;
		double sum, sum_square;
		double *file_sums = new double[n_cols];
		double *file_sum_squares = new double[n_cols];

		// reinitialize
//#pragma omp parallel for private(j, k, t, sum, sum_square) shared(file_sums, file_sum_squares, n_cols, data)
		for (j = 0; j < n_cols; j++) {
			file_sums[j] = 0.;
			file_sum_squares[j] = 0.;
			sum = 0;
			sum_square = 0;

//#pragma omp parallel for private(k, t) shared(n_cols, data) reduction(+: sum, sum_square)
			for (k = 0; k < n_rows; k++) {
				t = data[k * n_cols + j];
				sum += t;
				sum_square += t * t;
			}

			file_sums[j] = sum;
			file_sum_squares[j] = sum_square;
		}
//		t2 = cciutils::ClockGetTime();
//		printf("file summarize %d entries took %lu us for %s\n", n_rows, t2-t1, fin.c_str());
//		t1 = cciutils::ClockGetTime();

#pragma omp critical
		{

			// write out to file
			file_id = H5Fopen ( fin.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
			hstatus = H5LTset_attribute_double(file_id, "/data", "sums", file_sums, n_cols);
			hstatus = H5LTset_attribute_double(file_id, "/data", "square_sums", file_sum_squares, n_cols);
			H5Fclose ( file_id );
		}

		t2 = cciutils::ClockGetTime();
		printf("file read, summarize, write %d x %d entries took %lu us for %s\n", n_rows, n_cols, t2-t1, fin.c_str());
//		t1 = cciutils::ClockGetTime();

		// gather some summary info for this node
		node_rows += n_rows;

#pragma omp critical
		{
			for (j = 0; j < n_cols; j++) {
				node_sums[j] += file_sums[j];
				node_sum_squares[j] += file_sum_squares[j];
			}
		}
//		t2 = cciutils::ClockGetTime();
//		printf("node summarize took %lu us\n", t2-t1);

		delete [] data;
	    delete [] file_sums;
	    delete [] file_sum_squares;
	//	std::cout << rank << "::" << tid << ":" << fin << std::endl;
    }


#ifdef WITH_MPI
	MPI::COMM_WORLD.Barrier();
#endif

	t1 = cciutils::ClockGetTime();


#ifdef WITH_MPI
	// global reduction to get total count
	MPI::COMM_WORLD.Allreduce(&node_rows, &global_rows, 1, MPI::INT, MPI::SUM);
#else
	global_rows = node_rows;
#endif


	printf("total rows: %lu, node rows: %lu\n", global_rows, node_rows);
	// compute the local contribution to the mean
	std::stringstream ss;

	for (j = 0; j < n_cols; ++j) {
		node_sums[j] /= (double)global_rows;
//		ss<< node_sums[i] << ",";
	}
//	printf("node means: %s \n", ss.str().c_str());
//	ss.str(std::string());
//	ss.clear();

#ifdef WITH_MPI
	// global reduction to get global mean
	MPI::COMM_WORLD.Allreduce(node_sums, global_sums, n_cols, MPI::DOUBLE, MPI::SUM);
#else
	global_sums = node_sums;  // actually the mean.
#endif

	for (j = 0; j < n_cols; ++j) {
		ss<< global_sums[j] << ",";
	}
	if (rank == 0) printf("global means: %s \n", ss.str().c_str());
	ss.str(std::string());
	ss.clear();

	for (j = 0; j < n_cols; ++j) {
		node_sum_squares[j] /= (double)global_rows;
//		ss<< node_sum_squares[i] << ",";
	}
//	printf("node vars: %s \n", ss.str().c_str());
//	ss.str(std::string());
//	ss.clear();

#ifdef WITH_MPI
	MPI::COMM_WORLD.Allreduce(node_sum_squares, global_sum_squares, n_cols, MPI::DOUBLE, MPI::SUM);

#else
	global_sum_squares = node_sum_squares; // actually variance
#endif

	for (j = 0; j < n_cols; ++j) {
		global_sum_squares[j] -= global_sums[j] * global_sums[j];
		global_sum_squares[j] = sqrt(global_sum_squares[j]);
		ss<<  global_sum_squares[j] << ",";
	}
	if (rank == 0) printf("global stdev: %s \n", ss.str().c_str());
	ss.str(std::string());
	ss.clear();

	t2 = cciutils::ClockGetTime();
	printf("global summarize took %lu us\n", t2-t1);


	// this program is mostly doing file io.  don't use openmp because we are not using parallel hdf5
#ifdef WITH_MPI
#pragma omp parallel for shared(perNodeCount, inputBuf, maxLenInput, rank) private(fin, t1, t2)
	for (i = 0; i < perNodeCount; ++i) {
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
		printf("in MPI summary update loop with rank %d, loop %d. \"%s\"\n", rank, i, fin.c_str());

#else
#pragma omp parallel for shared(filenames, rank) private(fin, t1, t2)
	for (i = 0; i < dataCount; ++i) {
		fin = filenames[i];
#endif

		t1 = cciutils::ClockGetTime();

//#ifdef _OPENMP
//    	int tid = omp_get_thread_num();
//#else
		int tid = 0;
//#endif

		if (strcmp(fin.c_str(), "") == 0) continue;
		float *data;
		hsize_t ldims[2];
#pragma omp critical
		{
		// write out to file
		file_id = H5Fopen ( fin.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
		hstatus = H5LTset_attribute_long(file_id, "/data", "global_count", &global_rows, 1);
		hstatus = H5LTset_attribute_double(file_id, "/data", "global_means", global_sums, n_cols);
		hstatus = H5LTset_attribute_double(file_id, "/data", "global_stdevs", global_sum_squares, n_cols);

//		hstatus = H5LTget_dataset_info ( file_id, "/data", ldims, NULL, NULL );
//		n_rows = ldims[0];
//		data = new float[n_rows * n_cols];

//		H5LTread_dataset (file_id, "/data", H5T_NATIVE_FLOAT, data);
		H5Fclose ( file_id );

		}

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

		t2 = cciutils::ClockGetTime();
		printf("file update took %lu us %s\n", t2-t1, fin.c_str());


	//	std::cout << rank << "::" << tid << ":" << fin << std::endl;
	}


	delete [] node_sums;
	delete [] node_sum_squares;
	delete [] global_sums;
	delete [] global_sum_squares;



    if (rank == 0) {

    	t4 = cciutils::ClockGetTime();
		printf("**** Feature Summary took %lu us \n", t4-t3);
	//	std::cout << "**** Feature Extraction took " << t4-t3 << " us" << std::endl;

    }



#ifdef WITH_MPI
    if (rank == 0) {
		free(inputBufAll);
    }

	free(inputBuf);

	MPI::Finalize();
#endif


//	waitKey();

	return 0;
}


