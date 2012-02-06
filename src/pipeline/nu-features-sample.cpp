/*
 * aggreate features.
 *
 * MPI only, using bag of tasks paradigm
 *
 * pattern adopted from http://inside.mines.edu/mio/tutorial/tricks/workerbee.c
 *
 * this function is used to down sample the files.
 *
 * also calculates local mean and stdev.
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <math.h>
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "datatypes.h"
#include "h5utils.h"

using namespace cv;

// COMMENT OUT WHEN COMPILE for editing purpose only.
//#define WITH_MPI

#ifdef WITH_MPI
#include <mpi.h>

MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI::Intracomm init_workers(const MPI::Intracomm &comm_world, int managerid);
int parseInput(int argc, char **argv, int &modecode, std::string &maskName, std::string &outdir, float &ratio);
void getFiles(const std::string &maskName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &output);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName, std::string &outdir);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, const float ratio);
void compute(const char *input, const char *output, const float ratio);



int parseInput(int argc, char **argv, int &modecode, std::string &maskName, std::string &outdir, float &ratio) {
	if (argc < 5) {
		std::cout << "Usage:  " << argv[0] << " indir outdir ratio run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	maskName.assign(argv[1]);
	outdir.assign(argv[2]);
	ratio = atof(argv[3]);
	FileUtils futils;
	futils.mkdirs(outdir);

	const char* mode = argc > 5 ? argv[5] : "cpu";

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
		if (argc > 6) {
			gpu::setDevice(atoi(argv[6]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " indir outdir ratio run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}
void getFiles(const std::string &maskName, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &output) {

	// check to see if it's a directory or a file
	std::string suffix;
	suffix.assign(".features.h5");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(maskName, filenames);
	std::string dirname;
	if (filenames.size() == 1) {
		dirname = maskName.substr(0, maskName.find_last_of("/\\"));
	} else {
		dirname = maskName;
	}

	std::string temp, tempdir;
	for (unsigned int i = 0; i < filenames.size(); ++i) {
			// generate the input file name
		temp = futils.replaceExt(filenames[i], ".features.h5", ".sampled.features.h5");
		temp = futils.replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		futils.mkdirs(tempdir);
		output.push_back(temp);
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
	std::string maskName, outdir;
	float ratio;
	int status = parseInput(argc, argv, modecode, maskName, outdir, ratio);
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

	// NOT NEEDED
	//MPI::Intracomm comm_worker = init_workers(MPI::COMM_WORLD, manager_rank);
	//int worker_rank = comm_worker.Get_rank();


	uint64_t t1 = 0, t2 = 0;
	t1 = cciutils::ClockGetTime();

	// decide based on rank of worker which way to process
	if (rank == manager_rank) {
		// manager thread
		manager_process(comm_world, manager_rank, worker_size, maskName, outdir);
		t2 = cciutils::ClockGetTime();
		printf("MANAGER %d : FINISHED in %lu us\n", rank, t2 - t1);

	} else {
		// worker bees
		worker_process(comm_world, manager_rank, rank, ratio);
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
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName, std::string &outdir) {
	// first get the list of files to process
   	std::vector<std::string> filenames, outputs;
	uint64_t t1, t0;

	t0 = cciutils::ClockGetTime();

	getFiles(maskName, outdir, filenames, outputs);

	t1 = cciutils::ClockGetTime();
	printf("Manager ready at %d, file read took %lu us\n", manager_rank, t1 - t0);

	comm_world.Barrier();

	// now start the loop to listen for messages
	int curr = 0;
	int total = filenames.size();
	MPI::Status status;
	int worker_id;
	char ready;
	char *input;
	int inputlen;
	char *output;
	int outputlen;
	while (curr < total) {
		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
			//printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				// tell worker that manager is ready
				comm_world.Send(&MANAGER_READY, 1, MPI::CHAR, worker_id, TAG_CONTROL);
		//		printf("manager signal transfer\n");
/* send real data */
				inputlen = filenames[curr].size() + 1;  // add one to create the zero-terminated string
				input = new char[inputlen];
				memset(input, 0, sizeof(char) * inputlen);
				strncpy(input, filenames[curr].c_str(), inputlen);

				outputlen = outputs[curr].size() + 1;  // add one to create the zero-terminated string
				output = new char[outputlen];
				memset(output, 0, sizeof(char) * outputlen);
				strncpy(output, outputs[curr].c_str(), outputlen);

				comm_world.Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA);
				comm_world.Send(&outputlen, 1, MPI::INT, worker_id, TAG_METADATA);

				// now send the actual string data
				comm_world.Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA);
				comm_world.Send(output, outputlen, MPI::CHAR, worker_id, TAG_DATA);
				curr++;

				delete [] input;
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
		if (comm_world.Iprobe(MPI_ANY_SOURCE, TAG_CONTROL, status)) {
		/* where is it coming from */
			worker_id=status.Get_source();
			comm_world.Recv(&ready, 1, MPI::CHAR, worker_id, TAG_CONTROL);
			//printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				comm_world.Send(&MANAGER_FINISHED, 1, MPI::CHAR, worker_id, TAG_CONTROL);
				printf("manager signal finished\n");
				--active_workers;
			}
		}
	}
}

void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, float ratio) {
	char flag = MANAGER_READY;
	int inputSize;
	char *input;
	int outputSize;
	char *output;

	comm_world.Barrier();
	uint64_t t0, t1;


	while (flag != MANAGER_FINISHED && flag != MANAGER_ERROR) {
		t0 = cciutils::ClockGetTime();

		// tell the manager - ready
		comm_world.Send(&WORKER_READY, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
		//printf("worker %d signal ready\n", rank);
		// get the manager status
		comm_world.Recv(&flag, 1, MPI::CHAR, manager_rank, TAG_CONTROL);
	//	printf("worker %d received manager status %d\n", rank, flag);

		if (flag == MANAGER_READY) {
			// get data from manager
			comm_world.Recv(&inputSize, 1, MPI::INT, manager_rank, TAG_METADATA);
			comm_world.Recv(&outputSize, 1, MPI::INT, manager_rank, TAG_METADATA);

			// allocate the buffers
			input = new char[inputSize];
			memset(input, 0, inputSize * sizeof(char));
			output = new char[outputSize];
			memset(output, 0, outputSize * sizeof(char));

			// get the file names
			comm_world.Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA);
			comm_world.Recv(output, outputSize, MPI::CHAR, manager_rank, TAG_DATA);

			t1 = cciutils::ClockGetTime();
			//printf("comm time for worker %d is %lu us\n", rank, t1 -t0);

			// now do some work
			compute(input, output, ratio);

			t1 = cciutils::ClockGetTime();
		//	printf("worker %d processed \"%s\" in %lu us\n", rank, input, t1 - t0);

			// clean up
			delete [] input;
			delete [] output;

		}
	}
}





void compute(const char *input, const char *output, float ratio) {

	// first get the list of filenames
	herr_t hstatus;
	
	// open the files
	hsize_t ldims[2], newdims[2], maxdims[2], chunk[2];
	hid_t filetype, memtype;
	hid_t dset, space;
	hid_t file_id = H5Fopen(input, H5F_ACC_RDONLY, H5P_DEFAULT );
	hid_t out_file_id = H5Fcreate(output, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	int n_cols, n_rows;
	//
	// first copy the features
	//
	if (H5Lexists(file_id, NS_FEATURE_SET, H5P_DEFAULT)  <=0) {
		printf("FEATURE DOES NOT EXIST. SKIPPING %s\n", input);
		hstatus = H5Fclose(out_file_id);
		hstatus = H5Fclose(file_id);
		return;
	}

	hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET, ldims, NULL, NULL );
	n_rows = ldims[0];
	n_cols = ldims[1];


	// sampling
	int num_selected =(int)ceil((float)(n_rows) * ratio);
	int selected[num_selected];
	memset(selected, 0, sizeof(int) * num_selected);
	float temp = 0.;
	for (int i = 0, id=0; i < n_rows && id < num_selected; ++i) {
		temp += ratio;
		if (temp >= 1.0) {
			temp -= 1.0;
			selected[id] = i;
			++id;
		}
	}
//	for (int i = 0; i < num_selected; ++i) {
//		printf("%d, ", selected[i]);
//	}
//	printf("\n");


	// now read
	float *data = new float[n_rows * n_cols];
	H5LTread_dataset (file_id, NS_FEATURE_SET, H5T_NATIVE_FLOAT, data);

	// create feature dataset as extensible.
	maxdims[0] = H5S_UNLIMITED;
	maxdims[1] = n_cols;
	chunk[0] = 100;
	chunk[1] = n_cols;
	createExtensibleDataset(out_file_id, 2, maxdims, chunk, H5T_IEEE_F32LE, NS_FEATURE_SET);

	// do the selection
	float *data2 = new float[num_selected * n_cols];
	for (int i = 0; i < num_selected; ++i) {
		memcpy(data2 + i * n_cols, data + selected[i] * n_cols, sizeof(float) * n_cols);
	}
	newdims[0] = num_selected;
	newdims[1] = n_cols;

	// write to new file
	extendAndWrite(out_file_id, NS_FEATURE_SET, 2, newdims, H5T_NATIVE_FLOAT, data2, true);
	delete [] data;

	//
	// also compute the tile's partial sums
	//
	nu_sum_t imagesums;
	imagesums.nu_count = num_selected;
	for (unsigned int j = 0; j < n_cols; j++) {
		// initialize
		imagesums.nu_sum[j] = 0.;
		imagesums.nu_sum_square[j] = 0.;
		imagesums.bad_values[j] = 0;
	}
	float *currdata = data2;
	double t;
	for (unsigned int k = 0; k < num_selected; k++) {
		currdata = data2 + k*n_cols;
		for (unsigned int j = 0; j < n_cols; j++) {
			t = (double)currdata[j];
			if (isnan(t)) {
				//printf("NaN in %s, row %d, col %d\n", in.c_str(), k, j);
				++imagesums.bad_values[j];
				continue;
			}
			if (isinf(t)) {
				//printf("inf in %s, row %d, col %d\n", in.c_str(), k, j);
				++imagesums.bad_values[j];
				continue;
			}
			imagesums.nu_sum[j] += t;
			imagesums.nu_sum_square[j] += (t * t);
		}
	}
	// calculate the sums and mean/stdev.
	// write the partial sums to the file, as attribute on the sum table.
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_SUM_ATTR, imagesums.nu_sum, n_cols);
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_SUM_SQUARE_ATTR, imagesums.nu_sum_square, n_cols);
	hstatus = H5LTset_attribute_uint(out_file_id, NS_FEATURE_SET, NS_NUM_BAD_VALUES_ATTR, imagesums.bad_values, n_cols);

	// compute the mean, std, etc for this file.
	for (int i = 0; i < n_cols; i++) {
		if (isnan(imagesums.nu_sum[i]) ) printf("sum is not a number %d\n", i);
		if (isnan(imagesums.nu_sum_square[i]) ) printf("sum square is not a number %d\n", i);
		if (isinf(imagesums.nu_sum[i]) ) printf("sum is infinite %d\n", i);
		if (isinf(imagesums.nu_sum_square[i]) ) printf("sum square is infinite %d\n", i);
		imagesums.nu_sum[i] /= (double)(imagesums.nu_count - imagesums.bad_values[i]);  // mean
		imagesums.nu_sum_square[i] /= (double)(imagesums.nu_count - imagesums.bad_values[i]);  // average square sum
		imagesums.nu_sum_square[i] -= imagesums.nu_sum[i] * imagesums.nu_sum[i]; // - square average
		imagesums.nu_sum_square[i] = sqrt(imagesums.nu_sum_square[i]);
	}
	// write the mean and stdev to file as attributes
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_MEAN_ATTR, imagesums.nu_sum, n_cols);
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_STDEV_ATTR, imagesums.nu_sum_square, n_cols);


	delete [] data2;



	// check for dataset presence.
	if (H5Lexists(file_id, NS_TILE_INFO_SET, H5P_DEFAULT)  <=0) {
		// if tile_info is present, then we are looking at image features.

		// now copy the feature's attributes;
		// just the original mean and stdev...  unfortunately, we don't have these for the tile images.
		double mean[n_cols];
		if (H5Aexists_by_name(file_id, NS_FEATURE_SET, NS_MEAN_ATTR, H5P_DEFAULT)) {
			// read the version attribute.
			hstatus = H5LTget_attribute_double(file_id, NS_FEATURE_SET, NS_MEAN_ATTR, mean);
			hstatus = H5LTset_attribute_double( out_file_id, NS_FEATURE_SET, NS_FULL_MEAN_ATTR, mean, n_cols);
		}
		double stdev[n_cols];
		if (H5Aexists_by_name(file_id, NS_FEATURE_SET, NS_STDEV_ATTR, H5P_DEFAULT)) {
			// read the version attribute.
			hstatus = H5LTget_attribute_double( file_id, NS_FEATURE_SET, NS_STDEV_ATTR, stdev);
			hstatus = H5LTset_attribute_double( out_file_id, NS_FEATURE_SET, NS_FULL_STDEV_ATTR, stdev, n_cols);
		}


		//
		// next sample the nu-info
		//


		// create the tile info types
		filetype = createNuInfoFiletype();
		maxdims[0] = H5S_UNLIMITED;
		chunk[0] = 100;
		createExtensibleDataset(out_file_id, 1, maxdims, chunk, filetype, NS_NU_INFO_SET);
		hstatus = H5Tclose(filetype);

		// load the tile info into memory
		dset = H5Dopen(file_id, NS_NU_INFO_SET, H5P_DEFAULT);
		space = H5Dget_space(dset);
		memtype = createNuInfoMemtype();
		int ndims = H5Sget_simple_extent_dims(space, ldims, NULL);
		nu_info_t *nuinfo = (nu_info_t *)malloc(ldims[0] * sizeof(nu_info_t));
		hstatus = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, nuinfo);

		nu_info_t *nuinfo2 = (nu_info_t *)malloc(num_selected * sizeof(nu_info_t));
		for (int i = 0; i < num_selected; ++i) {
			memcpy(nuinfo2 + i, nuinfo + selected[i], sizeof(nu_info_t));
		}
		newdims[0] = num_selected;

		// now write out to other dataset
		extendAndWrite(out_file_id, NS_NU_INFO_SET, 1, newdims, memtype, nuinfo, true);
		free(nuinfo2);
		free(nuinfo);

		hstatus = H5Tclose(memtype);
		hstatus = H5Dclose(dset);
		hstatus = H5Sclose(space);

		//
		// copy tile info
		//

		// create the tile info types
		filetype = createTileInfoFiletype();
		maxdims[0] = H5S_UNLIMITED;
		chunk[0] = 4;
		createExtensibleDataset(out_file_id, 1, maxdims, chunk, filetype, NS_TILE_INFO_SET);
		hstatus = H5Tclose(filetype);


		// load the tile info into memory
		dset = H5Dopen(file_id, NS_TILE_INFO_SET, H5P_DEFAULT);
		space = H5Dget_space(dset);
		memtype = createTileInfoMemtype();
		ndims = H5Sget_simple_extent_dims(space, ldims, NULL);
		tile_info_t *tileinfo = (tile_info_t *)malloc(ldims[0] * sizeof(tile_info_t));
		hstatus = H5Dread(dset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, tileinfo);

		// now write out to other dataset
		extendAndWrite(out_file_id, NS_TILE_INFO_SET, 1, ldims, memtype, tileinfo, true);



		hstatus = H5Dvlen_reclaim(memtype, space, H5P_DEFAULT, tileinfo);
		free(tileinfo);

		hstatus = H5Tclose(memtype);
		hstatus = H5Dclose(dset);
		hstatus = H5Sclose(space);


	} else {
		// else we are looking at tile image.

		hstatus = H5LTget_dataset_info ( file_id, NS_NU_INFO_SET, ldims, NULL, NULL );
		n_rows = ldims[0];
		n_cols = ldims[1];

		// now read
		data = new float[n_rows * n_cols];
		H5LTread_dataset (file_id, NS_NU_INFO_SET, H5T_NATIVE_FLOAT, data);

		// create feature dataset as extensible.
		maxdims[0] = H5S_UNLIMITED;
		maxdims[1] = n_cols;
		chunk[0] = 100;
		chunk[1] = n_cols;
		createExtensibleDataset(out_file_id, 2, maxdims, chunk, H5T_IEEE_F32LE, NS_NU_INFO_SET);

		// do the selection
		data2 = new float[num_selected * n_cols];
		for (int i = 0; i < num_selected; ++i) {
			memcpy(data2 + i * n_cols, data + selected[i] * n_cols, sizeof(float) * n_cols);
		}
		newdims[0] = num_selected;
		newdims[1] = n_cols;

		// write to new file
		extendAndWrite(out_file_id, NS_NU_INFO_SET, 2, newdims, H5T_NATIVE_FLOAT, data2, true);
		delete [] data;
		delete [] data2;
	}

	// copy the remaining attributes
	char imagename[256];
	if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, H5P_DEFAULT)) {
		// read the version attribute.
		hstatus = H5LTget_attribute_string( file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename );
		hstatus = H5LTset_attribute_string( out_file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename);
	}
	hstatus = H5LTset_attribute_string ( out_file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, "0.2" );
	hstatus = H5LTset_attribute_string ( out_file_id, NS_NU_INFO_SET, NS_FILE_CONTENT_TYPE, "sampled features" );
	hstatus = H5LTset_attribute_float ( out_file_id, NS_NU_INFO_SET, NS_SAMPLE_RATE_ATTR, &ratio, 1 );



	hstatus = H5Fclose(file_id);
	hstatus = H5Fclose(out_file_id);

	// done.


	return;

}



#else
int main (int argc, char **argv){
	printf("THIS PROGRAM REQUIRES MPI.  PLEASE RECOMPILE WITH MPI ENABLED.  EXITING\n");
	return -1;
}
#endif



