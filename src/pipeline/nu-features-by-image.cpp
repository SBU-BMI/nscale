/*
 * aggreate features.
 *
 * MPI only, using bag of tasks paradigm
 *
 * pattern adopted from http://inside.mines.edu/mio/tutorial/tricks/workerbee.c
 *
 * this function is used to aggregate all features for a particular image across all tiles.
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <math.h>
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "datatypes.h"
#include "h5utils.h"

#include <unistd.h>

using namespace cv;

// COMMENT OUT WHEN COMPILE for editing purpose only.
//#define WITH_MPI

#ifdef WITH_MPI
#include <mpi.h>

MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI::Intracomm init_workers(const MPI::Intracomm &comm_world, int managerid);
int parseInput(int argc, char **argv, int &modecode, std::string &maskName, std::string &outdir);
void getDirs(const std::string &dirname, std::vector<std::string> &dirnames);
void getFiles(const std::string &dirName, std::vector<std::string> &filenames);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, std::string &outdir);
void compute(const char *dirname, std::string &outdir);



int parseInput(int argc, char **argv, int &modecode, std::string &maskName, std::string &outdir) {
	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " indir outdir run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	maskName.assign(argv[1]);
	outdir.assign(argv[2]);
	FileUtils::mkdirs(outdir);

	const char* mode = argc > 4 ? argv[4] : "cpu";

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
		std::cout << "Usage:  " << argv[0] << " indir outdir run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}
void getDirs(const std::string &dirName, std::vector<std::string> &dirnames) {

	// check to see if it's a directory or a file

	FileUtils futils;
	futils.traverseDirectory(dirName, dirnames, FileUtils::DIRECTORY, false);

	if (dirnames.empty()) {  // no subdir, so must be operating on the current directory.
		dirnames.push_back(dirName);
	}

}


void getFiles(const std::string &dirName, std::vector<std::string> &filenames) {

	// check to see if it's a directory or a file

	FileUtils futils(std::string(".features.h5"));
	futils.traverseDirectory(dirName, filenames, FileUtils::FILE, false);
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
	int status = parseInput(argc, argv, modecode, maskName, outdir);
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
		manager_process(comm_world, manager_rank, worker_size, maskName);
		t2 = cciutils::ClockGetTime();
		printf("MANAGER %d : FINISHED in %lu us\n", rank, t2 - t1);

	} else {
		// worker bees
		worker_process(comm_world, manager_rank, rank, outdir);
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
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName) {
	// first get the list of files to process
   	std::vector<std::string> dirnames;
	uint64_t t1, t0;

	t0 = cciutils::ClockGetTime();

	getDirs(maskName, dirnames);
	printf("dirname: %s\n", maskName.c_str());
	for (unsigned int i = 0; i < dirnames.size(); ++i) {
		printf("   dir: %s\n", dirnames[i].c_str());
	}

	t1 = cciutils::ClockGetTime();
	printf("Manager ready at %d, file read took %lu us\n", manager_rank, t1 - t0);

	comm_world.Barrier();

	// now start the loop to listen for messages
	int curr = 0;
	int total = dirnames.size();
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
			//printf("manager received request from worker %d\n",worker_id);
			if (worker_id == manager_rank) continue;

			if(ready == WORKER_READY) {
				// tell worker that manager is ready
				comm_world.Send(&MANAGER_READY, 1, MPI::CHAR, worker_id, TAG_CONTROL);
		//		printf("manager signal transfer\n");
/* send real data */
				inputlen = dirnames[curr].size() + 1;  // add one to create the zero-terminated string
				input = new char[inputlen];
				memset(input, 0, sizeof(char) * inputlen);
				strncpy(input, dirnames[curr].c_str(), inputlen);

				comm_world.Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA);

				// now send the actual string data
				comm_world.Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA);
				curr++;

				delete [] input;

			}

			if (curr % 10 == 1) {
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

void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank, std::string &outdir) {
	char flag = MANAGER_READY;
	int inputSize;
	char *input;

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

			// allocate the buffers
			input = new char[inputSize];
			memset(input, 0, inputSize * sizeof(char));

			// get the file names
			comm_world.Recv(input, inputSize, MPI::CHAR, manager_rank, TAG_DATA);

			t1 = cciutils::ClockGetTime();
//			printf("comm time for worker %d is %lu us\n", rank, t1 -t0);

			// now do some work
			compute(input, outdir);

			t1 = cciutils::ClockGetTime();
		//	printf("worker %d processed \"%s\" in %lu us\n", rank, input, t1 - t0);

			// clean up
			delete [] input;

		}
	}
}





void compute(const char *dirname, std::string &outdir) {

	// first get the list of filenames
	std::vector<std::string> filenames;
	getFiles(dirname, filenames);
	herr_t hstatus;
	
//	printf("here:  %s, with %d files\n", dirname, filenames.size());

	// open the first file to get some information
	if (filenames.empty()) return;
	hsize_t ldims[2];
	unsigned int n_feat_cols;
	unsigned int n_meta_cols;
	char imagename[256];
	hid_t file_id = H5Fopen(filenames[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
	if (H5Lexists(file_id, NS_FEATURE_SET, H5P_DEFAULT)) {
		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET, ldims, NULL, NULL );
		n_feat_cols = ldims[1];
	}
	if (H5Lexists(file_id, NS_NU_INFO_SET, H5P_DEFAULT)) {
		hstatus = H5LTget_dataset_info ( file_id, NS_NU_INFO_SET, ldims, NULL, NULL );
		n_meta_cols = ldims[1];
		
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, H5P_DEFAULT)) {
			// read the version attribute.
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename);
		}
	}
	H5Fclose(file_id);
	
	
	// next set up the output h5 file.  this is just the dirname + ".features.all.h5"
	stringstream ss;
	ss << outdir << "/" << imagename << ".image.features.h5";
	string ofn = ss.str();
	hid_t out_file_id = H5Fcreate(ofn.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	
	// create feature dataset as extensible.
	hsize_t maxdims[2] = {H5S_UNLIMITED, n_feat_cols};
	hsize_t chunk[2] = {1000, n_feat_cols};

	createExtensibleDataset(out_file_id, 2, maxdims, chunk, H5T_IEEE_F32LE, NS_FEATURE_SET);

	// create nu-info dataset as extensible
	// create compound data type for dataset creation.
	hid_t file_sum_type = createTileSumFiletype();
	maxdims[0] = H5S_UNLIMITED;
	chunk[0] = 4;
	createExtensibleDataset(out_file_id, 1, maxdims, chunk, file_sum_type, NS_TILE_SUM_SET);
	hstatus = H5Tclose(file_sum_type);



	// create nu-info dataset as extensible
	// create compound data type for dataset creation.
	hid_t file_nu_type = createNuInfoFiletype();
	maxdims[0] = H5S_UNLIMITED;
	chunk[0] = 1000;
	createExtensibleDataset(out_file_id, 1, maxdims, chunk, file_nu_type, NS_NU_INFO_SET);
	hstatus = H5Tclose(file_nu_type);


	// create the tile info types
	hid_t file_tile_type = createTileInfoFiletype();
	chunk[0] = 4;
	createExtensibleDataset(out_file_id, 1, maxdims, chunk, file_tile_type, NS_TILE_INFO_SET);

	hstatus = H5Tclose(file_tile_type);


//	hstatus = H5Fclose(file_id);


	unsigned int n_rows, n_cols;
	float *data;
	char imgtilename[256];
	char masktilename[256];
	char featuretilename[256];
	nu_sum_t imagesums;
	imagesums.nu_count = 0;
	for (unsigned int j = 0; j < n_feat_cols; j++) {
		imagesums.nu_sum[j] = 0.;
		imagesums.nu_sum_square[j] = 0.;
		imagesums.bad_values[j] = 0;
	}
	double t;


//	out_file_id = H5Fopen(ofn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
	// create nuinfo for IO.
	hid_t mem_nu_type = createNuInfoMemtype();
	hid_t mem_tile_type = createTileInfoMemtype();
	hid_t mem_sum_type = createTileSumMemtype();

	// loop through the filenames and aggregate.
	for (unsigned int i = 0; i < filenames.size(); ++i) {
		std::string in = filenames[i];

		// read from the file
		// first read the data
		// open the file
		file_id = H5Fopen(in.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
	

		if (H5Lexists(file_id, NS_FEATURE_SET, H5P_DEFAULT)  <=0 ||
			H5Lexists(file_id, NS_NU_INFO_SET, H5P_DEFAULT)  <=0) {
			hstatus = H5Fclose(file_id);
			printf("ERROR: input features.h5 file %s is missing either features, or metadata, or both.\n", in.c_str());
			continue;
		}

		// copy the features
		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET, ldims, NULL, NULL );
		n_rows = ldims[0];
		n_cols = ldims[1];

		data = new float[n_rows * n_cols];
		H5LTread_dataset (file_id, NS_FEATURE_SET, H5T_NATIVE_FLOAT, data);

		// write to new file
		extendAndWrite(out_file_id, NS_FEATURE_SET, 2, ldims, H5T_NATIVE_FLOAT, data, (i==0));

		//
		// also compute the tile's partial sums
		//
		nu_sum_t sinfo;
		sinfo.nu_count = (long)n_rows;
		for (unsigned int j = 0; j < n_feat_cols; j++) {
			// initialize
			sinfo.nu_sum[j] = 0.;
			sinfo.nu_sum_square[j] = 0.;
			sinfo.bad_values[j] = 0;
		}
		float *currdata = data;
		for (unsigned int k = 0; k < n_rows; k++) {
			currdata = data + k*n_cols;
			for (unsigned int j = 0; j < n_cols; j++) {
				t = (double)currdata[j];
				if (isnan(t)) {
					//printf("NaN in %s, row %d, col %d\n", in.c_str(), k, j);
					++sinfo.bad_values[j];
					continue;
				}
				if (isinf(t)) {
					//printf("inf in %s, row %d, col %d\n", in.c_str(), k, j);
					++sinfo.bad_values[j];
					continue;
				}
				sinfo.nu_sum[j] += t;
				sinfo.nu_sum_square[j] += (t * t);
			}
		}

		// and update image file's sum and sumsquare
		imagesums.nu_count += sinfo.nu_count;
		for (unsigned int j = 0; j < n_cols; j++) {
			imagesums.nu_sum[j] += sinfo.nu_sum[j];
			imagesums.nu_sum_square[j] += sinfo.nu_sum_square[j];
			imagesums.bad_values[j] += sinfo.bad_values[j];
		}
		//
		// and save it.
		// open same file again and update the attributes
		// write out to file
		ldims[0] = 1;
		extendAndWrite(out_file_id, NS_TILE_SUM_SET, 1, ldims, mem_sum_type, &sinfo, (i==0));

		// done with getting partial results for tile

		delete [] data;


		// copy the metadata
		hstatus = H5LTget_dataset_info ( file_id, NS_NU_INFO_SET, ldims, NULL, NULL );
		n_rows = ldims[0];
		n_cols = ldims[1];

		data = new float[n_rows * n_cols];
		H5LTread_dataset (file_id, NS_NU_INFO_SET, H5T_NATIVE_FLOAT, data);

		// now format the data
		nu_info_t *info = new nu_info_t[n_rows];
		float *currpos = data;
		for (int j = 0; j < n_rows; ++j) {
			info[j].tile_id = i;
			currpos = data + j * n_cols;
			info[j].bbox_x = currpos[0];
			info[j].bbox_y = currpos[1];
			info[j].bbox_w = currpos[2];
			info[j].bbox_h = currpos[3];
			info[j].centroid_x = currpos[4];
			info[j].centroid_y = currpos[5];
		}

		// write to new file
		extendAndWrite(out_file_id, NS_NU_INFO_SET, 1, ldims, mem_nu_type, info, (i==0));

		delete [] data;
		delete [] info;


		// copy the attributes into a table.
		memset(imgtilename, 0, sizeof(char) * 256);
		memset(masktilename, 0, sizeof(char) * 256);
		memset(featuretilename, 0, sizeof(char) * 256);
		int tile_x;
		int tile_y;
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_IMG_TILE_ATTR, H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET, NS_IMG_TILE_ATTR, imgtilename);
		}
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_MASK_TILE_ATTR, H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET, NS_MASK_TILE_ATTR, masktilename);
		}
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_TILE_X_ATTR, H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_X_ATTR, &tile_x);
		}
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_TILE_Y_ATTR, H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_Y_ATTR, &tile_y);
		}
		strcpy(featuretilename, in.c_str());

		tile_info_t tinfo;
		tinfo.img_tile_name = imgtilename;
		tinfo.mask_tile_name = masktilename;
		tinfo.feature_tile_name = featuretilename;
		tinfo.tile_x = tile_x;
		tinfo.tile_y = tile_y;
		ldims[0] = 1;
		extendAndWrite(out_file_id, NS_TILE_INFO_SET, 1, ldims, mem_tile_type, &tinfo, (i==0));


		// now add the attributes
		hstatus = H5LTset_attribute_string( out_file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename);
		hstatus = H5LTset_attribute_string ( out_file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, "0.2" );
		hstatus = H5LTset_attribute_string ( out_file_id, NS_NU_INFO_SET, NS_FILE_CONTENT_TYPE, "raw image features" );


		hstatus = H5Fclose(file_id);

		// need to release the vlen data?  only if read and created vlen data automatically.
		//H5Dvlen_reclaim(mem_tile_type, space, H5P_DEFAULT, rdata);
		// delete [] rdata;
	}
	hstatus = H5Tclose(mem_nu_type);
	hstatus = H5Tclose(mem_tile_type);
	hstatus = H5Tclose(mem_sum_type);
//	printf("total: %lu\n ", imagesums.nu_count );

	// NOW SAVE THE MEAN AND STDEV, and PARTIAL SUMS FOR THESE.

	// write the partial sums to the file, as attribute on the sum table.
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_SUM_ATTR, imagesums.nu_sum, n_feat_cols);
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_SUM_SQUARE_ATTR, imagesums.nu_sum_square, n_feat_cols);
	hstatus = H5LTset_attribute_uint(out_file_id, NS_FEATURE_SET, NS_NUM_BAD_VALUES_ATTR, imagesums.bad_values, n_feat_cols);

	// compute the mean, std, etc for this file.
	for (int i = 0; i < n_feat_cols; i++) {
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
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_MEAN_ATTR, imagesums.nu_sum, n_feat_cols);
	hstatus = H5LTset_attribute_double(out_file_id, NS_FEATURE_SET, NS_STDEV_ATTR, imagesums.nu_sum_square, n_feat_cols);


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



