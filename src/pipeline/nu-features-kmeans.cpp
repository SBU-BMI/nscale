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
#include <algorithm>
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>

#include "hdf5.h"
#include "hdf5_hl.h"

#include "datatypes.h"
#include "h5utils.h"
#include "nwkmeans/kmeans.h"

using namespace cv;

// COMMENT OUT WHEN COMPILE for editing purpose only.
#define WITH_MPI

#ifdef WITH_MPI
#include <mpi.h>


int
mpi_kmeans(float    **objects,     /* in: [numObjs][numCoords] */
	   unsigned long        numCoords,   /* no. coordinates */
	   unsigned long        numObjs,     /* no. objects */
	   int        numClusters, /* no. clusters */
	   float      thresh,   /* % objects change membership */
	   int       *membership,  /* out: [numObjs] */
	   float    **clusters,    /* out: [numClusters][numCoords] */
	   float	*distance,	   /* out: [numObjs] */
	   MPI_Comm   comm);        /* MPI communicator */


bool compareNuCount(feature_info_t fi1, feature_info_t fi2) {
	return fi1.nu_count > fi2.nu_count;
}

static const int TAG_CONTROL = 0;
static const int TAG_DATA = 1;
static const int TAG_METADATA = 2;


MPI_Comm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);

int parseInput(int argc, char **argv, int &modecode, std::string &runid, std::string &infile, std::string &outDir, int &k);
void getFiles(const std::string &dirName, std::vector<feature_info_t> &featurefiles, unsigned long &totalcount, double *mean, double *stdev, int &n_cols);
void normalize();
unsigned long run_kmeans(float* data, unsigned long numObjs, int numCoords, int numClusters, int rank, int managerRank, float thresh,
		int *membership, float *distance, float *centers,
		MPI_Comm   comm);
void writeClusters(int nu_count, int thresh, int k, float *initial_centers, int numClusters, float *centers, int *membership, float *distance);

int parseInput(int argc, char **argv, int &modecode, std::string &runid, std::string &infile, std::string &outDir, int &k, float &thresh) {
	if (argc < 6) {
		std::cout << "Usage:  " << argv[0] << " input outDir k thresh run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	infile.assign(argv[1]);
	outDir.assign(argv[2]);
	k = atoi(argv[3]);
	thresh = atof(argv[4]);
	runid.assign(argv[5]);
	printf("outfile directory = %s\n", argv[2]);
	FileUtils futils;
	futils.mkdirs(outDir);

	const char* mode = argc > 6 ? argv[6] : "cpu";

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
		if (argc > 7) {
			gpu::setDevice(atoi(argv[7]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " infile outDir k thresh run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	return 0;
}



void getFiles(const std::string &infile, std::vector<feature_info_t> &featurefiles, unsigned long &totalcount, double **mean, double **stdev, int &n_cols) {

	// open the HDF5 summary file
	hid_t file_id = H5Fopen(infile.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
	hsize_t ldims[2];
	herr_t hstatus;
	int n_rows = 0;
	if (H5Lexists(file_id, NS_IMAGE_INFO_SET, H5P_DEFAULT)) {
		hstatus = H5LTget_dataset_info ( file_id, NS_IMAGE_INFO_SET, ldims, NULL, NULL );
		n_rows = ldims[0];
	}
	// init memory
	feature_info_t *features = (feature_info_t *)malloc(ldims[0] * sizeof(feature_info_t));
	
	// get all the feature file references
	hid_t dset = H5Dopen(file_id, NS_IMAGE_SUM_SET, H5P_DEFAULT);
	hid_t space = H5Dget_space(dset);
	int ndims = H5Sget_simple_extent_dims(space, ldims, NULL);
	hid_t count_mem_type = createImageInfoNuCountMemtype();
	hstatus = H5Dread(dset, count_mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, features);
	H5Sclose(space);
	H5Dclose(dset);
	H5Tclose(count_mem_type);


	hid_t dset2 = H5Dopen(file_id, NS_IMAGE_INFO_SET, H5P_DEFAULT);
	space = H5Dget_space(dset2);
	ndims = H5Sget_simple_extent_dims(space, ldims, NULL);
	hid_t feature_mem_type = createImageInfoFeatureMemtype();
	hstatus = H5Dread(dset2, feature_mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, features);
		
	// put into priority queue
	for (int i = 0; i < n_rows; i++) {

		feature_info_t fi;
		fi.nu_count = features[i].nu_count;
		fi.feature_name = new char[strlen(features[i].feature_name) + 1];
		memset(fi.feature_name, 0, strlen(features[i].feature_name) + 1);
		strcpy(fi.feature_name, features[i].feature_name);
//		printf("inside: %s: %lu\n", fi.feature_name, fi.nu_count);
		featurefiles.push_back(fi);
	}

	// clean up.
	hstatus = H5Dvlen_reclaim (feature_mem_type, space, H5P_DEFAULT, features);
    free (features);
	H5Tclose(feature_mem_type);
	H5Sclose(space);
	H5Dclose(dset2);

	hstatus = H5LTget_attribute_ulong(file_id, NS_IMAGE_SUM_SET, NS_COUNT_ATTR, &totalcount);

	dset = H5Dopen(file_id, NS_IMAGE_SUM_SET, H5P_DEFAULT);
	hid_t attr = H5Aopen(dset, NS_MEAN_ATTR, H5P_DEFAULT);
	space = H5Aget_space(attr);
	ndims = H5Sget_simple_extent_dims(space, ldims, NULL);
	n_cols = ldims[0];
//	printf("n cols = %d\n", n_cols);
	hstatus = H5Sclose(space);
	hstatus = H5Aclose(attr);
	hstatus = H5Dclose(dset);

	*mean = new double[n_cols];
	*stdev = new double[n_cols];
	hstatus = H5LTget_attribute_double(file_id, NS_IMAGE_SUM_SET, NS_MEAN_ATTR, *mean);
	hstatus = H5LTget_attribute_double(file_id, NS_IMAGE_SUM_SET, NS_STDEV_ATTR, *stdev);


	hstatus = H5Fclose(file_id);
}



// initialize MPI
MPI_Comm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname) {
    MPI_Init(&argc, &argv);

    char * temp = new char[256];
    gethostname(temp, 255);
    hostname.assign(temp);
    delete [] temp;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return MPI_COMM_WORLD;
}

int main (int argc, char **argv){
	// parse the input
	int modecode;
	std::string infile, outDir, runid;
	int k;
	float thresh;
	int status = parseInput(argc, argv, modecode, runid, infile, outDir, k, thresh);
	if (status != 0) return status;

	// set up mpi
	int rank, size, worker_size, manager_rank;
	std::string hostname;
	MPI_Comm comm_world = init_mpi(argc, argv, size, rank, hostname);

	if (size == 1) {
		printf("ERROR:  this program can only be run with 2 or more MPI nodes.  The head node does not process data\n");
		return -4;
	}

	// initialize the worker comm object
	worker_size = size;   // notice worker pool is the whole set here.
	manager_rank = size - 1;

//	MPI::Intracomm comm_worker = init_workers(MPI::COMM_WORLD, manager_rank);
//	int worker_rank = comm_worker.Get_rank();


	uint64_t t1 = 0, t2 = 0;
	t1 = cciutils::ClockGetTime();

	// decide based on rank of worker which way to process
	// broadcast to everyone.
	std::vector<feature_info_t> work;
	int sizeofwork;
	int inputlen;
	double *mean;
	double *stdev;
	int n_cols;

	if (rank == manager_rank) {
		// manager thread
	   	std::vector<feature_info_t> featurefiles;
		uint64_t t1, t0;
		unsigned long total;

		t0 = cciutils::ClockGetTime();

		getFiles(infile, featurefiles, total, &mean, &stdev, n_cols);

		printf("dirname: %s\n", infile.c_str());

		// sort the vector
		std::sort(featurefiles.begin(), featurefiles.end(), compareNuCount);


		// allocate the files
		std::vector<std::vector<feature_info_t> > assignments(worker_size);
		unsigned long totals[worker_size];
		memset(totals, 0, worker_size * sizeof(unsigned long));
		int id = 0;
		unsigned long min;
//		int direction = 1;
		for (unsigned int i = 0; i < featurefiles.size(); ++i) {
			// sorted and put with the next min - works better for the smaller feature sets.
			min = std::numeric_limits<unsigned long>::max();
			for (int j = 0; j < worker_size; j++) {
				if (min > totals[j]) {
					min = totals[j];
					id = j;
				}
			}

			feature_info_t fi = featurefiles[i];
			totals[id] += fi.nu_count;

			assignments[id].push_back(fi);

//			printf("%lu, %s\n", fi.nu_count, fi.feature_name);

			// determine the next place to assign
			// round robin with the sorted version - works okay...
			//++id;
			//if (id == worker_size) id = 0;
		}

//		printf("totals: \n");
		for (int i = 0; i < worker_size; ++i) {
//			printf("  %d: %lu, %d\n", i, totals[i], assignments[i].size());
			for (unsigned int j = 0; j < assignments[i].size(); ++j) {
//				printf("       %d, %d: %lu, %s\n", i, j, assignments[i][j].nu_count, assignments[i][j].feature_name);
			}
		}

		t1 = cciutils::ClockGetTime();
		printf("Manager ready at %d, file read took %lu us\n", manager_rank, t1 - t0);


		// now send
		for (int i = 0; i < worker_size; ++i) {
			if (i == manager_rank) {
				work = assignments[rank];
				sizeofwork = work.size();
				continue;
			}

			std::vector<feature_info_t> worktosend = assignments[i];
			int numtosend = worktosend.size();
			MPI_Send(&numtosend, 1, MPI_INT, i, TAG_METADATA, comm_world);

			for (int j = 0; j < numtosend; ++j) {
				feature_info_t fi = worktosend[j];

				// send the count
				MPI_Send(&(fi.nu_count), 1, MPI_UNSIGNED_LONG, i, TAG_DATA, comm_world);

				// send the filename size
				inputlen = strlen(fi.feature_name) + 1;  // add one to create the zero-terminated string
				MPI_Send(&inputlen, 1, MPI_INT, i, TAG_METADATA, comm_world);

				// send the filename
				MPI_Send(fi.feature_name, inputlen, MPI_CHAR, i, TAG_DATA, comm_world);

				// delete after sending it...
				delete [] fi.feature_name;
			}
		}


	} else {
		MPI_Status stat;

		MPI_Recv(&sizeofwork, 1, MPI_INT, manager_rank, TAG_METADATA, comm_world, &stat);

//		printf("%d number to receive %d\n", rank, sizeofwork);

		int count = 0;
		for (int j = 0; j < sizeofwork; ++j) {
			feature_info_t fi;

			// get the count
			MPI_Recv(&(fi.nu_count), 1, MPI_UNSIGNED_LONG, manager_rank, TAG_DATA, comm_world, &stat);

			// get the filename size
			MPI_Recv(&inputlen, 1, MPI_INT, manager_rank, TAG_METADATA, comm_world, &stat);

			fi.feature_name = new char[inputlen];
			memset(fi.feature_name, 0, inputlen * sizeof(char));
			// get the filename
			MPI_Recv(fi.feature_name, inputlen, MPI_CHAR, manager_rank, TAG_DATA, comm_world, &stat);
			work.push_back(fi);
			count ++;
		}
		printf("worker %d received %d \n", rank, count);
	}

	// manager broadcast mean and stdev.
	MPI_Bcast(&n_cols, 1, MPI_INT, manager_rank, comm_world);
	if (rank != manager_rank) {
		mean = new double[n_cols];
		stdev = new double[n_cols];
	}
	MPI_Bcast(mean, n_cols, MPI_DOUBLE, manager_rank, comm_world);
	MPI_Bcast(stdev, n_cols, MPI_DOUBLE, manager_rank, comm_world);

	printf("%d cols %d, mean %f %f %f, stdev %f %f %f\n", rank, n_cols, mean[0], mean[1], mean[n_cols - 1], stdev[0], stdev[1], stdev[n_cols - 1]);


	MPI_Barrier(comm_world);

	// get the total count
	unsigned long nu_count = 0;
	for (int i = 0; i < work.size(); ++i) {
		nu_count += work[i].nu_count;
	}
	printf("%d assigned work \n", rank);

	// worker bees
	float *data = new float[nu_count * n_cols];
	memset(data, 0, nu_count * n_cols * sizeof(float));

	float *currpos = data;
	hsize_t ldims[2];
	hid_t file_id;
	herr_t hstatus;
	unsigned int n_rows, curr_count = 0;
	for (int i = 0; i < work.size(); ++i) {

		// load data
		file_id = H5Fopen(work[i].feature_name, H5F_ACC_RDONLY, H5P_DEFAULT );

		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET, ldims, NULL, NULL );
		n_rows = ldims[0];

		hstatus = H5LTread_dataset(file_id, NS_FEATURE_SET, H5T_NATIVE_FLOAT, currpos);
		currpos += n_rows * n_cols;
		curr_count += n_rows;

		hstatus = H5Fclose(file_id);

//		if (rank == 0)
//		{
//			printf("%d rank reading file %s\n", rank, work[i].feature_name);
////		for (int k = 0; k < n_rows; k++) {
////			printf("%f ", data[k * n_cols]);
////		}
////		printf("\n");
//		}

	}

	printf("%d finished reading data.  curr_count %lu, nucount %lu\n", rank, curr_count, nu_count);
	// normalize
	currpos = data;
	for (unsigned int i = 0; i < nu_count; i++) {
		for (int j = 0; j < n_cols; j++) {
			currpos[j] = (float)((currpos[j] - mean[j]) / stdev[j]);
		}
		currpos += n_cols;
	}
	printf("%d finished normalizing data \n", rank);

	if (rank == manager_rank) {
		  stringstream ss;
		  ss << rank << " rank normalized: ";
		for (int j = 0; j < nu_count; j+= 10000) {
			ss << data[j * n_cols + 3] << ", ";
		}
		ss << std::endl;
		printf("%s", ss.str().c_str());
		ss.clear();
	}

	///////////////////////////
	// compute kmeans

	int *membership = new int[nu_count];
	memset(membership, 0, nu_count * sizeof(int));
	float *distance = new float[nu_count];
	memset(distance, 0, nu_count * sizeof(float));
	float *centers = new float[k * n_cols];
	memset(centers, 0, k * n_cols * sizeof(float));
	float *initial_centers = new float[k * n_cols];
	memset(centers, 0, k * n_cols * sizeof(float));


	// initialize with a random seed
	if (rank == manager_rank) {
		for (int i = 0; i < k; i++) {
			/*
			 *  randomly pick a cluster to copy to initial object set
			 */
			int pick;

			//      pick = (int) ((numObjs - 1) * (rand() / (RAND_MAX + 1.0)));
			pick = (int)(rand() % nu_count);
			//printf ("pick %d\n", pick);

			memcpy(initial_centers + i * n_cols, data + pick * n_cols, n_cols * sizeof(float));
		}
	}
	memcpy(centers, initial_centers, n_cols * k * sizeof(float));


	// run the thing

	unsigned long numClusters = run_kmeans(data, nu_count, n_cols, k, rank, manager_rank, thresh, membership, distance, centers, comm_world);

	// write out results.

	//writeClusters(nu_count, thresh, k, initial_centers, numClusters, centers, membership, distance);

	MPI_Barrier(comm_world);;




	// clean up
	for (int i =0; i < sizeofwork; ++i) {
		delete [] work[i].feature_name;
	}
	delete [] mean;
	delete [] stdev;

	delete [] data;
	delete [] membership;
	delete [] distance;
	delete [] centers;


	printf("%d rank.  nu_count %d, n_cols %d, k %d, thresh %f\n", rank, nu_count, n_cols, k, thresh);

	MPI_Finalize();




	exit(0);

}

// Based on Michael's code
unsigned long run_kmeans(float* data, unsigned long numObjs, int numCoords, int numClusters, int rank, int managerRank, float thresh,
		int *membership, float *distance, float *centers,
		MPI_Comm comm) {

	printf("%d numObjs: %d, numclus: %d, manager: %d, thresh: %f\n", rank, numObjs, numClusters, managerRank, thresh);


	// transform the input
	  /* allocate object array */
	  float** objects = (float**) calloc ( numObjs, sizeof(float*) );
	  assert ( objects );
	  objects[0] = data;
	  for (unsigned long i = 1; i < numObjs; i++)
	    objects[i] = objects[i-1] + numCoords;

	  if (rank == managerRank) {
	  	  stringstream ss;
			  ss << rank << " Data: ";
	  			for (int j = 0; j < numObjs; j+=10000) {
	  				ss << objects[j][3] << ", ";
	  			}
	  			ss << std::endl;
	  			printf("%s", ss.str().c_str());
	  }


	/* allocate a 2D space for clusters[] (coordinates of cluster centers)
	     this array should be the same across all processes                  */

	  float** clusters = (float**) malloc(numClusters * sizeof(float*));
	  assert(clusters != NULL);
	  clusters[0] = centers;
	  for (int i=1; i<numClusters; i++)
	    clusters[i] = clusters[i-1] + numCoords;

//		for (int i = 0; i < numClusters; ++i) {
//			printf("%d center %d: ", rank, i);
//			for (int j = 0; j < numCoords; j++) {
//				printf("%f, ", clusters[i][j]);
//			}
//			printf("\n");
//		}

//	for (int i = 0; i < numClusters; ++i) {
//		printf("%d center %d: ", rank, i);
//		for (int j = 0; j < numCoords; j++) {
//			printf("%f, ", clusters[i][j]);
//		}
//		printf("\n");
//	}


	  MPI_Bcast(clusters[0], numClusters*numCoords, MPI_FLOAT, managerRank, comm);

//if (rank == managerRank) {
//	  stringstream ss;
//		  ss.clear();
//		for (int i = 0; i < numClusters; ++i) {
//			  ss << rank << " center " << i << ": ";
//			for (int j = 0; j < numCoords; j++) {
//				ss << clusters[i][j] << ", ";
//			}
//			ss << std::endl;
//			printf("%s", ss.str().c_str());
//			ss.str("");
//			ss.clear();
//		}
//}

	  /* membership: the cluster id for each data object */
//	  int *membership = (int*) malloc(numObjs * sizeof(int));
	  assert(membership != NULL);



	   /* distance is the square of the distance of each data
			object from the centroid of the cluster it belongs to.
		*/
//	  float *distance = (float *) malloc(numObjs * sizeof(float));
	  assert(distance != NULL);

	  MPI_Barrier(comm);

	  mpi_kmeans ( objects, numCoords, numObjs,
		       numClusters, thresh,
		       membership, clusters,
		       distance, comm );

	  MPI_Barrier(comm);


if (rank == managerRank) {
	  stringstream ss;
			  ss << rank << " membership: ";
			for (int j = 0; j < numObjs; j+=10000) {
				ss << membership[j] << ", ";
			}
			ss << std::endl;
			printf("%s", ss.str().c_str());
			ss.clear();
}


	  // numClusters may be adjusted (fewer.  allocate new mem to hold the final.
	  // should only use the first few

	  delete [] clusters;
	  delete [] objects;

	  return numClusters;
}



// write out a single file? - need to be able to trace back to original files
// or to multiple files?

void writeClusters(int nu_count, int thresh, int k, float *initial_centers, int numClusters, float *centers, int *membership, float *distance)
{
	// first get the list of filenames
	std::vector<std::string> filenames;
	getFiles(dirname, filenames);
	herr_t hstatus;
	
	printf("here:  %s, with %d files\n", dirname, filenames.size());

	// open the first file to get some information
	if (filenames.empty()) return;
	hsize_t ldims[2], maxdims[2], chunk[2];
	unsigned int n_cols;
	unsigned int n_rows = filenames.size();
	char imagename[256], featurename[256];

	hid_t file_id = H5Fopen(filenames[0].c_str(), H5F_ACC_RDONLY, H5P_DEFAULT );
	if (H5Lexists(file_id, NS_FEATURE_SET, H5P_DEFAULT)) {
		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET, ldims, NULL, NULL );
		n_cols = ldims[1];
	}
	hstatus = H5Fclose(file_id);
	
	// next set up the output h5 file.  this is just the dirname + ".features.all.h5"
	stringstream ss;
	ss << outDir << "/" << runid << ".features-summary.h5";
	string ofn = ss.str();
	printf("output goes to %s\n", ofn.c_str());

	hid_t out_file_id = H5Fcreate(ofn.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	
	// create image sum dataset as extensible
	// create compound data type for dataset creation.
	hid_t file_sum_type = createImageSumFiletype();
	maxdims[0] = n_rows;
	chunk[0] = 4;
	createExtensibleDataset(out_file_id, 1, maxdims, chunk, file_sum_type, NS_IMAGE_SUM_SET);
	hstatus = H5Tclose(file_sum_type);

	// create the image info types
	hid_t file_image_type = createImageInfoFiletype();
	createExtensibleDataset(out_file_id, 1, maxdims, chunk, file_image_type, NS_IMAGE_INFO_SET);
	hstatus = H5Tclose(file_image_type);


//	hstatus = H5Fclose(file_id);

	image_sum_t imagesums;
	imagesums.nu_count = 0;
	for (unsigned int j = 0; j < n_cols; j++) {
		imagesums.nu_sum[j] = 0.;
		imagesums.nu_sum_square[j] = 0.;
		imagesums.nu_mean[j] = 0.;
		imagesums.nu_stdev[j] = 0.;
		imagesums.bad_values[j] = 0;
	}
	double t;


//	out_file_id = H5Fopen(ofn.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
	// create nuinfo for IO.
	hid_t mem_image_type = createImageInfoMemtype();
	hid_t mem_sum_type = createImageSumMemtype();
	int image_rows;

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



		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, H5P_DEFAULT)) {
			// read the version attribute.
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename);
		}

		hstatus = H5LTget_dataset_info ( file_id, NS_FEATURE_SET, ldims, NULL, NULL );
		image_rows = ldims[0];


		//
		// also compute the tile's partial sums
		//
		image_sum_t sinfo;
		sinfo.nu_count = (long)image_rows;
		for (unsigned int j = 0; j < n_cols; j++) {
			// initialize
			sinfo.nu_sum[j] = 0.;
			sinfo.nu_sum_square[j] = 0.;
			sinfo.nu_mean[j] = 0.;
			sinfo.nu_stdev[j] = 0.;
			sinfo.bad_values[j] = 0;
		}
		// read from the input file.
		hstatus = H5LTget_attribute_double(file_id, NS_FEATURE_SET, NS_SUM_ATTR, sinfo.nu_sum);
		hstatus = H5LTget_attribute_double(file_id, NS_FEATURE_SET, NS_SUM_SQUARE_ATTR, sinfo.nu_sum_square);
		hstatus = H5LTget_attribute_double(file_id, NS_FEATURE_SET, NS_MEAN_ATTR, sinfo.nu_mean);
		hstatus = H5LTget_attribute_double(file_id, NS_FEATURE_SET, NS_STDEV_ATTR, sinfo.nu_stdev);
		hstatus = H5LTget_attribute_uint(file_id, NS_FEATURE_SET, NS_NUM_BAD_VALUES_ATTR, sinfo.bad_values);

		// and save it.
		// open same file again and update the attributes
		// write out to file
		ldims[0] = 1;
		extendAndWrite(out_file_id, NS_IMAGE_SUM_SET, 1, ldims, mem_sum_type, &sinfo, (i==0));




		// copy the attributes into a table.
		memset(imagename, 0, sizeof(char) * 256);
		memset(featurename, 0, sizeof(char) * 256);
		if (H5Aexists_by_name(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, H5P_DEFAULT)) {
			// read the version attribute.
			hstatus = H5LTget_attribute_string(file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename);
		}
		strcpy(featurename, in.c_str());

		image_info_t tinfo;
		tinfo.img_name = imagename;
		tinfo.feature_name = featurename;
		ldims[0] = 1;
		extendAndWrite(out_file_id, NS_IMAGE_INFO_SET, 1, ldims, mem_image_type, &tinfo, (i==0));






		// and update image file's sum and sumsquare
		imagesums.nu_count += sinfo.nu_count;
		for (unsigned int j = 0; j < n_cols; j++) {
			imagesums.nu_sum[j] += sinfo.nu_sum[j];
			imagesums.nu_sum_square[j] += sinfo.nu_sum_square[j];
			imagesums.bad_values[j] += sinfo.bad_values[j];
		}
		//


		// done with getting partial results for tile

		hstatus = H5Fclose(file_id);

		// need to release the vlen data?  only if read and created vlen data automatically.
		//H5Dvlen_reclaim(mem_tile_type, space, H5P_DEFAULT, rdata);
		// delete [] rdata;
	}
	hstatus = H5Tclose(mem_image_type);
	hstatus = H5Tclose(mem_sum_type);
//	printf("total: %lu\n ", imagesums.nu_count );

	// NOW SAVE THE MEAN AND STDEV, and PARTIAL SUMS FOR THESE.

	// write the partial sums to the file, as attribute on the sum table.
	hstatus = H5LTset_attribute_double(out_file_id, NS_IMAGE_SUM_SET, NS_SUM_ATTR, imagesums.nu_sum, n_cols);
	hstatus = H5LTset_attribute_double(out_file_id, NS_IMAGE_SUM_SET, NS_SUM_SQUARE_ATTR, imagesums.nu_sum_square, n_cols);
	hstatus = H5LTset_attribute_uint(out_file_id, NS_IMAGE_SUM_SET, NS_NUM_BAD_VALUES_ATTR, imagesums.bad_values, n_cols);

	// compute the mean, std, etc for this file.
	for (unsigned int i = 0; i < n_cols; i++) {
		if (isnan(imagesums.nu_sum[i]) ) printf("sum is NaN %d\n", i);
		if (isnan(imagesums.nu_sum_square[i]) ) printf("sum square is NaN %d\n", i);
		if (isinf(imagesums.nu_sum[i]) ) printf("sum is inf %d\n", i);
		if (isinf(imagesums.nu_sum_square[i]) ) printf("sum square is inf %d\n", i);
		imagesums.nu_mean[i] = imagesums.nu_sum[i] / (double)(imagesums.nu_count - imagesums.bad_values[i]);  // mean
		imagesums.nu_stdev[i] = imagesums.nu_sum_square[i] / (double)(imagesums.nu_count - imagesums.bad_values[i]);  // average square sum
		imagesums.nu_stdev[i] -= imagesums.nu_mean[i] * imagesums.nu_mean[i]; // - square average
		imagesums.nu_stdev[i] = sqrt(imagesums.nu_stdev[i]);
	}
	// write the mean and stdev to file as attributes
	hstatus = H5LTset_attribute_double(out_file_id, NS_IMAGE_SUM_SET, NS_MEAN_ATTR, imagesums.nu_mean, n_cols);
	hstatus = H5LTset_attribute_double(out_file_id, NS_IMAGE_SUM_SET, NS_STDEV_ATTR, imagesums.nu_stdev, n_cols);

	// now add the attributes
	hstatus = H5LTset_attribute_string ( out_file_id, NS_IMAGE_INFO_SET, NS_H5_VER_ATTR, "0.2" );
	hstatus = H5LTset_attribute_string ( out_file_id, NS_IMAGE_INFO_SET, NS_FILE_CONTENT_TYPE, "feature summary" );




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



