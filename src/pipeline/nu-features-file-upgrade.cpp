/*
 * compute features.
 *
 * MPI only, using bag of tasks paradigm
 *
 * pattern adopted from http://inside.mines.edu/mio/tutorial/tricks/workerbee.c
 *
 * this function is used to upgrade the data organization of h5 files to latest version.
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
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
int parseInput(int argc, char **argv, int &modecode, std::string &maskName);
void getFiles(const std::string &maskName, std::vector<std::string> &filenames);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank);
void compute(const char *input);



int parseInput(int argc, char **argv, int &modecode, std::string &maskName) {
	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	maskName.assign(argv[1]);
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
		if (argc > 4) {
			gpu::setDevice(atoi(argv[4]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}


void getFiles(const std::string &maskName, std::vector<std::string> &filenames) {

	// check to see if it's a directory or a file
	std::string suffix;
	suffix.assign(".features.h5");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(maskName, filenames);
	std::string dirname = maskName;
	if (filenames.size() == 1) {
		// if the maskname is actually a file, then the dirname is extracted from the maskname.
		if (strcmp(filenames[0].c_str(), maskName.c_str()) == 0) {
			dirname = maskName.substr(0, maskName.find_last_of("/\\"));
		}
	}


//	for (int i = 0; i < filenames.size(); i++) {
//		std::cout << "filename: " << filenames[i] << std::endl;
//	}
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
	std::string maskName;
	int status = parseInput(argc, argv, modecode, maskName);
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
		worker_process(comm_world, manager_rank, rank);
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
   	std::vector<std::string> filenames;
	uint64_t t1, t0;

	t0 = cciutils::ClockGetTime();

	getFiles(maskName, filenames);

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

				comm_world.Send(&inputlen, 1, MPI::INT, worker_id, TAG_METADATA);

				// now send the actual string data
				comm_world.Send(input, inputlen, MPI::CHAR, worker_id, TAG_DATA);
				curr++;

				delete [] input;

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

void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank) {
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
			compute(input);

			t1 = cciutils::ClockGetTime();
		//	printf("worker %d processed \"%s\" in %lu us\n", rank, input, t1 - t0);

			// clean up
			delete [] input;

		}
	}
}





void compute(const char *input) {

	// first read the data
	// open the file
	hsize_t ldims[2];
	hid_t file_id = H5Fopen(input, H5F_ACC_RDONLY, H5P_DEFAULT );
	herr_t hstatus;
	char *version = new char[10];  memset(version, 0, sizeof(char) * 10);
	if (H5Lexists(file_id, NS_NU_INFO_SET, H5P_DEFAULT) &&
			H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, H5P_DEFAULT)) {
		// read the version attribute.
		hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, version);
	} else if (H5Lexists(file_id, NS_NU_INFO_SET_01, H5P_DEFAULT)) {
		strcpy(version, "0.1");
	} else
		strcpy(version, "0.0");  // this version only has a /data

	//printf("version: %s\n", version);


	if (strcmp(version, "0.0")== 0) {  // version 0.0

		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET_01, ldims, NULL, NULL );
		unsigned int n_rows = ldims[0];
		unsigned int n_cols = ldims[1];

		float *data = new float[n_rows * n_cols];
		H5LTread_dataset (file_id, NS_FEATURE_SET_01, H5T_NATIVE_FLOAT, data);

		char *imgfilename = new char[1024];  memset(imgfilename, 0, sizeof(char) * 1024);
		char *mskfilename = new char[1024];	memset(mskfilename, 0, sizeof(char) * 1024);

		if (H5Aexists_by_name(file_id, NS_FEATURE_SET_01, "image_file", H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_string(file_id, NS_FEATURE_SET_01, "image_file", imgfilename);
		}
		if (H5Aexists_by_name(file_id, NS_FEATURE_SET_01, "mask_file", H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_string(file_id, NS_FEATURE_SET_01, "mask_file", mskfilename);
		}
		H5Fclose ( file_id );

		//printf("image file name [%s].\n", imgfilename);
		//printf("mask file name [%s].\n", mskfilename);

		// partition the data
		unsigned int featureSize = n_cols - 6;
		float *metadata = new float[n_rows * 6];
		float *newdata = new float[n_rows * featureSize];
		for (unsigned int i = 0; i < n_rows; ++i) {
			memcpy(metadata + i * 6, data + i * n_cols, sizeof(float) * 6);
			memcpy(newdata + i * featureSize, data + i * n_cols + 6, sizeof(float) * featureSize);
		}

		// overwrite the original file
		//printf("writing out %s\n", input);

		hsize_t dims[2];
		file_id = H5Fcreate ( input, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

		dims[0] = n_rows; dims[1] = featureSize;
		hstatus = H5LTmake_dataset ( file_id, NS_FEATURE_SET,
				2, // rank
				dims, // dims
				H5T_NATIVE_FLOAT, newdata );

		dims[0] = n_rows; dims[1] = 6;
		hstatus = H5LTmake_dataset ( file_id, NS_NU_INFO_SET,
				2, // rank
				dims, // dims
				H5T_NATIVE_FLOAT, metadata );
		// attach the attributes
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_IMG_TILE_ATTR, imgfilename );
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_MASK_TILE_ATTR, mskfilename );


		// clear the data
		delete [] data;
		delete [] newdata;
		delete [] metadata;
		delete [] mskfilename;


		// unable to get rename working.
//		file_id = H5Fopen(input, H5F_ACC_RDONLY, H5P_DEFAULT );
//		hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT);
//		hstatus = H5Lmove(group_id, "data", group_id, "features", H5P_DEFAULT, H5P_DEFAULT);
//		hstatus = H5Lmove(group_id, "metadata",group_id,  "nu-info", H5P_DEFAULT, H5P_DEFAULT);
//		H5Gclose(group_id);

		// parse the input string
		string suffix;
		suffix.assign(".tif");
		FileUtils futils(suffix);
		string infile;
		infile.assign(imgfilename);
		string filename = futils.getFile(infile);
		// get the image name
		size_t pos = filename.rfind('.');
		if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", imgfilename);
		string prefix = filename.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", imgfilename);
		string ystr = prefix.substr(pos + 1);
		prefix = prefix.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", imgfilename);
		string xstr = prefix.substr(pos + 1);
		string imagename = prefix.substr(0, pos);
		int tilex = atoi(xstr.c_str());
		int tiley = atoi(ystr.c_str());


		hstatus = H5LTset_attribute_string( file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename.c_str());
		hstatus = H5LTset_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_X_ATTR, &tilex, 1);
		hstatus = H5LTset_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_Y_ATTR, &tiley, 1);
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, "0.2" );
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_FILE_CONTENT_TYPE, "raw tile features upgraded from v0.0 to v0.2" );


		delete [] imgfilename;

		strcpy(version, "0.2");
	} // brought to version 0.2 compliance.
	else if (strcmp(version, "0.1")== 0) {  // version 0.1

		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET_01, ldims, NULL, NULL );
		unsigned int n_rows = ldims[0];
		unsigned int n_cols = ldims[1];

		float *data = new float[n_rows * n_cols];
		H5LTread_dataset (file_id, NS_FEATURE_SET_01, H5T_NATIVE_FLOAT, data);

		hstatus = H5LTget_dataset_info ( file_id, NS_NU_INFO_SET_01, ldims, NULL, NULL );
		unsigned int n_rows_meta = ldims[0];
		unsigned int n_cols_meta = ldims[1];

		float *metadata = new float[n_rows_meta * n_cols_meta];
		H5LTread_dataset (file_id, NS_NU_INFO_SET_01, H5T_NATIVE_FLOAT, metadata);

		char *imgfilename = new char[1024];  memset(imgfilename, 0, sizeof(char) * 1024);
		char *mskfilename = new char[1024];	memset(mskfilename, 0, sizeof(char) * 1024);

		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET_01, NS_IMG_TILE_ATTR, H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET_01, NS_IMG_TILE_ATTR, imgfilename);
		}
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET_01, NS_MASK_TILE_ATTR, H5P_DEFAULT)) {
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET_01, NS_MASK_TILE_ATTR, mskfilename);
		}

		H5Fclose ( file_id );

		//printf("image file name [%s].\n", imgfilename);
		//printf("mask file name [%s].\n", mskfilename);

		// overwrite the original file
		//printf("writing out %s\n", input);

		hsize_t dims[2];
		file_id = H5Fcreate ( input, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

		dims[0] = n_rows; dims[1] = n_cols;
		hstatus = H5LTmake_dataset ( file_id, NS_FEATURE_SET,
				2, // rank
				dims, // dims
				H5T_NATIVE_FLOAT, data );

		dims[0] = n_rows_meta; dims[1] = n_cols_meta;
		hstatus = H5LTmake_dataset ( file_id, NS_NU_INFO_SET,
				2, // rank
				dims, // dims
				H5T_NATIVE_FLOAT, metadata );
		// attach the attributes
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_IMG_TILE_ATTR, imgfilename );
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_MASK_TILE_ATTR, mskfilename );


		// clear the data
		delete [] data;
		delete [] metadata;
		delete [] mskfilename;


		// unable to get rename working.
//		file_id = H5Fopen(input, H5F_ACC_RDONLY, H5P_DEFAULT );
//		hid_t group_id = H5Gopen(file_id, "/", H5P_DEFAULT);
//		hstatus = H5Lmove(group_id, "data", group_id, "features", H5P_DEFAULT, H5P_DEFAULT);
//		hstatus = H5Lmove(group_id, "metadata",group_id,  "nu-info", H5P_DEFAULT, H5P_DEFAULT);
//		H5Gclose(group_id);

		// parse the input string
		string suffix;
		suffix.assign(".tif");
		FileUtils futils(suffix);
		string infile;
		infile.assign(imgfilename);
		string filename = futils.getFile(infile);
		// get the image name
		size_t pos = filename.rfind('.');
		if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", imgfilename);
		string prefix = filename.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", imgfilename);
		string ystr = prefix.substr(pos + 1);
		prefix = prefix.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", imgfilename);
		string xstr = prefix.substr(pos + 1);
		string imagename = prefix.substr(0, pos);
		int tilex = atoi(xstr.c_str());
		int tiley = atoi(ystr.c_str());


		hstatus = H5LTset_attribute_string( file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename.c_str());
		hstatus = H5LTset_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_X_ATTR, &tilex, 1);
		hstatus = H5LTset_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_Y_ATTR, &tiley, 1);
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, "0.2" );
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_FILE_CONTENT_TYPE, "raw tile features upgraded from v0.1 to v0.2" );


		delete [] imgfilename;

		strcpy(version, "0.2");
	} // brought to version 0.2 compliance.

	// nothing else to do at this point

	H5Fclose(file_id);

	delete [] version;
	return;

}



#else
int main (int argc, char **argv){
	printf("THIS PROGRAM REQUIRES MPI.  PLEASE RECOMPILE WITH MPI ENABLED.  EXITING\n");
	return -1;
}
#endif



