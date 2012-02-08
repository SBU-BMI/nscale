/*
 * compute features.
 *
 * MPI only, using bag of tasks paradigm
 *
 * pattern adopted from http://inside.mines.edu/mio/tutorial/tricks/workerbee.c
 *
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>
#include "RegionalMorphologyAnalysis.h"
#include "PixelOperations.h"

#include "hdf5.h"
#include "hdf5_hl.h"

#include "datatypes.h"
#include "h5utils.h"

using namespace cv;

// COMMENT OUT WHEN COMPILE for editing purpose only.
//#define WITH_MPI

#if defined (WITH_MPI)
#include <mpi.h>
#endif

#if defined (_OPENMP)
#include <omp.h>
#endif


int parseInput(int argc, char **argv, int &modecode, std::string &maskName, std::string &imageDir, std::string &outdir, bool &overwrite);
void getFiles(const std::string &maskName, const std::string &imgDir, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &features_output, bool overwrite);
void compute(const char *input, const char *mask, const char *output);
void saveData(vector<vector<float> >& nucleiFeatures, vector<vector<float> >& cytoplasmFeatures_G, vector<vector<float> >& cytoplasmFeatures_H, vector<vector<float> >& cytoplasmFeatures_E,
		const char* input, const char* mask, const char* output);


int parseInput(int argc, char **argv, int &modecode, std::string &maskName, std::string &imageDir, std::string &outdir, bool &overwrite) {
	if (argc < 5) {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> image_dir feature_dir " << "<overwrite(0|1)> [cpu [numThreads] | mcore [numThreads] | gpu [numThreads] [id]]" << std::endl;
		return -1;
	}
	maskName.assign(argv[1]);
	imageDir.assign(argv[2]);
	outdir.assign(argv[3]);
	if (strcasecmp(argv[4], "1") == 0 || strcasecmp(argv[4], "T") == 0 || strcasecmp(argv[4], "TRUE") == 0) {
		overwrite = true;
	} else {
		overwrite = false;
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
#if defined (_OPENMP)
	omp_set_num_threads(1);
#endif

		if (argc > 7) {
			gpu::setDevice(atoi(argv[7]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> image_dir feature_dir " << "<overwrite(0|1)> [cpu [numThreads] | mcore [numThreads] | gpu [numThreads] [id]]" << std::endl;
		return -1;
	}

	return 0;
}


void getFiles(const std::string &maskName, const std::string &imgDir, const std::string &outDir, std::vector<std::string> &filenames,
		std::vector<std::string> &seg_output, std::vector<std::string> &features_output, bool overwrite) {

	// check to see if it's a directory or a file
	std::string suffix;
	suffix.assign(".mask.pbm");

	FileUtils futils(suffix);
	futils.traverseDirectoryRecursive(maskName, seg_output);
	std::string dirname = maskName;
	if (seg_output.size() == 1) {
		// if the maskname is actually a file, then the dirname is extracted from the maskname.
		if (strcmp(seg_output[0].c_str(), maskName.c_str()) == 0) {
			dirname = maskName.substr(0, maskName.find_last_of("/\\"));
		}
	}

//	printf("seg_output size for dir %s = %d, entry 1 = %s \n", maskName.c_str(), seg_output.size(), seg_output[0].c_str());

	std::string temp, temp2, tempdir;
	FILE *file;
	for (unsigned int i = 0; i < seg_output.size(); ++i) {

		// generate the output file name
		temp = futils.replaceExt(seg_output[i], ".mask.pbm", ".features.h5");
		temp = futils.replaceDir(temp, dirname, outDir);
		tempdir = temp.substr(0, temp.find_last_of("/\\"));
		futils.mkdirs(tempdir);
		if (!overwrite && (file = fopen(temp.c_str(), "r"))) {
			fclose(file);
		} else {
			features_output.push_back(temp);
			continue;
		}

		// generate the input file name
		temp = futils.replaceExt(seg_output[i], ".mask.pbm", ".tif");
		temp = futils.replaceDir(temp, dirname, imgDir);
		temp2 = futils.replaceExt(seg_output[i], ".mask.pbm", ".tiff");
		temp2 = futils.replaceDir(temp2, dirname, imgDir);

//		printf("image file names: %s %s\n", temp.c_str(), temp2.c_str());

		if ((file = fopen(temp.c_str(), "r"))) {
			fclose(file);
			filenames.push_back(temp);
		} else if ((file = fopen(temp2.c_str(), "r"))) {
			fclose(file);
			filenames.push_back(temp2);
		} else {
			printf("unable to find corresponing image file for %s in dir %s.  skipping\n", seg_output[i].c_str(), imgDir.c_str());
			continue;
		}

//		printf("feature filename: %s\n", temp.c_str());
	}

}




void compute(const char *input, const char *mask, const char *output) {
	// Load input images
	::cv::Mat maskMat = imread(mask, -1);
	if (! maskMat.data) {
		printf("can't read original image mask\n");
		return;
	}

	::cv::Mat image = imread(input, -1);
	if (! image.data) {
		printf("can't read original image\n");
		return;
	}

	IplImage originalImageMask(maskMat);

	//bool isNuclei = true;

	// Convert color image to grayscale
	::cv::Mat grayMat = ::nscale::PixelOperations::bgr2gray(image);
	//	cvSaveImage("newGrayScale.png", grayscale);
	IplImage grayscale(grayMat);

	// This is another option for inialize the features computation, where the path to the images are given as parameter
	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(&originalImageMask, &grayscale, true);

	// Create H and E images
	//initialize H and E channels
	Mat H = Mat::zeros(image.size(), CV_8UC1);
	Mat E = Mat::zeros(image.size(), CV_8UC1);
	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);

	::nscale::PixelOperations::ColorDeconv(image, M, b, H, E);

	IplImage ipl_image_H(H);
	IplImage ipl_image_E(E);


	// This is another option for inialize the features computation, where the path to the images are given as parameter
	//	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);

	vector<vector<float> > nucleiFeatures;

	/////////////// Compute nuclei based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)BoundingBox (BB) X; 1) BB.y; 2) BB.width; 3) BB.height; 4) Centroid.x; 5) Centroid.y) 7)Area; 8)Perimeter; 9)Eccentricity;
	//	10)Circularity/Compacteness; 11)MajorAxis; 12)MinorAxis; 13)ExtentRatio; 14)MeanIntensity 15)MaxIntensity; 16)MinIntensity;
	//	17)StdIntensity; 18)EntropyIntensity; 19)EnergyIntensity; 20)SkewnessIntensity;	21)KurtosisIntensity; 22)MeanGrad; 23)StdGrad;
	//	24)EntropyGrad; 25)EnergyGrad; 26)SkewnessGrad; 27)KurtosisGrad; 28)CannyArea; 29)MeanCanny
	regional->doNucleiPipelineFeatures(nucleiFeatures, &grayscale);

	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_G;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_G, &grayscale);


	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_H;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_H, &ipl_image_H);

	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_E;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_E, &ipl_image_E);

	delete regional;

	image.release();
	maskMat.release();
	grayMat.release();

	H.release();
	E.release();
	M.release();
	b.release();



	saveData(nucleiFeatures, cytoplasmFeatures_G, cytoplasmFeatures_H, cytoplasmFeatures_E,
			input, mask, output);
	nucleiFeatures.clear();
	cytoplasmFeatures_G.clear();
	cytoplasmFeatures_H.clear();
	cytoplasmFeatures_E.clear();

}

void saveData(vector<vector<float> >& nucleiFeatures, vector<vector<float> >& cytoplasmFeatures_G, vector<vector<float> >& cytoplasmFeatures_H, vector<vector<float> >& cytoplasmFeatures_E,
		const char* input, const char* mask, const char* output) {
	// create a single data field
	if (nucleiFeatures.size() > 0) {

		// first deal with the metadata
		unsigned int metadataSize = 6;
		float *metadata = new float[nucleiFeatures.size() * metadataSize];
		float *currData;
		for (unsigned int i = 0; i < nucleiFeatures.size(); i++) {
			currData = metadata + i * metadataSize;
			for (unsigned int j = 0; j < metadataSize; j++) {
				currData[j] = nucleiFeatures[i][j];
#ifdef	PRINT_FEATURES
					printf("%f, ", currData[j]);
#endif
			}
		}

		unsigned int nuFeatureSize = nucleiFeatures[0].size() - 6;
		unsigned int recordSize = nuFeatureSize + cytoplasmFeatures_G[0].size() + cytoplasmFeatures_H[0].size() + cytoplasmFeatures_E[0].size();
		unsigned int featureSize;
		float *data = new float[nucleiFeatures.size() * recordSize];
		for(unsigned int i = 0; i < nucleiFeatures.size(); i++) {

			currData = data + i * recordSize;
			featureSize = nuFeatureSize;
			for(unsigned int j = 0; j < featureSize; j++) {
				if (j < nucleiFeatures[i].size() - 6) {
					currData[j] = nucleiFeatures[i][j+6];

#ifdef	PRINT_FEATURES
					printf("%f, ", currData[j]);
#endif
				}
			}

			currData += featureSize;
			featureSize = cytoplasmFeatures_G[0].size();
			for(unsigned int j = 0; j < featureSize; j++) {
				if (j < cytoplasmFeatures_G[i].size()) {
					currData[j] = cytoplasmFeatures_G[i][j];
#ifdef	PRINT_FEATURES
					printf("%f, ", currData[j]);
#endif
				}
			}

			currData += featureSize;
			featureSize = cytoplasmFeatures_H[0].size();
			for(unsigned int j = 0; j < featureSize; j++) {
				if (j < cytoplasmFeatures_H[i].size()) {
					currData[j] = cytoplasmFeatures_H[i][j];
#ifdef	PRINT_FEATURES
					printf("%f, ", currData[j]);
#endif
				}
			}

			currData += featureSize;
			featureSize = cytoplasmFeatures_E[0].size();
			for(unsigned int j = 0; j < featureSize; j++) {
				if (j < cytoplasmFeatures_E[i].size()) {
					currData[j] = cytoplasmFeatures_E[i][j];
#ifdef	PRINT_FEATURES
					printf("%f, ", currData[j]);
#endif
				}
			}

		} // end looping through all nuclei


		hid_t file_id;
		herr_t hstatus;

//		printf("writing out %s\n", output);

#if defined (_OPENMP)
#pragma omp critical
		{
#endif

		hsize_t dims[2];
		file_id = H5Fcreate ( output, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

		dims[0] = nucleiFeatures.size(); dims[1] = recordSize;
		hstatus = H5LTmake_dataset ( file_id, NS_FEATURE_SET,
				2, // rank
				dims, // dims
				H5T_NATIVE_FLOAT, data );

		dims[0] = nucleiFeatures.size(); dims[1] = metadataSize;
		hstatus = H5LTmake_dataset ( file_id, NS_NU_INFO_SET,
				2, // rank
				dims, // dims
				H5T_NATIVE_FLOAT, metadata );
		// attach the attributes
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_IMG_TILE_ATTR, input );
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_MASK_TILE_ATTR, mask );


		// version 0.2 stuff below.

		// parse the input string
		string suffix;
		suffix.assign(".tif");
		FileUtils futils(suffix);
		string infile;
		infile.assign(input);
		string filename = futils.getFile(infile);
		// get the image name
		unsigned int pos = filename.rfind('.');
		if (pos == std::string::npos) printf("ERROR:  file %s does not have extension\n", input);
		string prefix = filename.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", input);
		string ystr = prefix.substr(pos + 1);
		prefix = prefix.substr(0, pos);
		pos = prefix.rfind("-");
		if (pos == std::string::npos) printf("ERROR:  file %s does not have a properly formed x, y coords\n", input);
		string xstr = prefix.substr(pos + 1);
		string imagename = prefix.substr(0, pos);
		int tilex = atoi(xstr.c_str());
		int tiley = atoi(ystr.c_str());

		hstatus = H5LTset_attribute_string( file_id, NS_NU_INFO_SET, NS_IMG_NAME_ATTR, imagename.c_str());
		hstatus = H5LTset_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_X_ATTR, &tilex, 1);
		hstatus = H5LTset_attribute_int(file_id, NS_NU_INFO_SET, NS_TILE_Y_ATTR, &tiley, 1);
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_H5_VER_ATTR, "0.2" );
		hstatus = H5LTset_attribute_string ( file_id, NS_NU_INFO_SET, NS_FILE_CONTENT_TYPE, "raw tile features" );

		H5Fclose ( file_id );
#if defined (_OPENMP)
		}
#endif

		// clear the data
		delete [] data;
		delete [] metadata;
	}

}



#if defined (WITH_MPI)
MPI::Intracomm init_mpi(int argc, char **argv, int &size, int &rank, std::string &hostname);
MPI::Intracomm init_workers(const MPI::Intracomm &comm_world, int managerid);
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName, std::string &imgDir, std::string &outDir, bool overwrite);
void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank);


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
void manager_process(const MPI::Intracomm &comm_world, const int manager_rank, const int worker_size, std::string &maskName, std::string &imgDir, std::string &outDir, bool overwrite) {
	// first get the list of files to process
   	std::vector<std::string> filenames;
	std::vector<std::string> seg_output;
	std::vector<std::string> features_output;
	uint64_t t1, t0;

	t0 = cciutils::ClockGetTime();
	getFiles(maskName, imgDir, outDir, filenames, seg_output, features_output, overwrite);

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
				outputlen = features_output[curr].size() + 1;
				input = new char[inputlen];
				memset(input, 0, sizeof(char) * inputlen);
				strncpy(input, filenames[curr].c_str(), inputlen);
				mask = new char[masklen];
				memset(mask, 0, sizeof(char) * masklen);
				strncpy(mask, seg_output[curr].c_str(), masklen);
				output = new char[outputlen];
				memset(output, 0, sizeof(char) * outputlen);
				strncpy(output, features_output[curr].c_str(), outputlen);

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

void worker_process(const MPI::Intracomm &comm_world, const int manager_rank, const int rank) {
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
		t0 = cciutils::ClockGetTime();

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

			t0 = cciutils::ClockGetTime();
//			printf("comm time for worker %d is %lu us\n", rank, t1 -t0);
			printf("worker %d processed \"%s\"\n", rank, mask);


			// now do some work
			compute(input, mask, output);

			t1 = cciutils::ClockGetTime();
//			printf("worker %d processed \"%s\" + \"%s\" -> \"%s\" in %lu us\n", rank, input, mask, output, t1 - t0);
			printf("worker %d processed \"%s\" in %lu us\n", rank, mask, t1 - t0);

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
	std::string maskName, imgDir,  outDir;
	bool overwrite;
	int status = parseInput(argc, argv, modecode, maskName, imgDir, outDir, overwrite);
	if (status != 0) return status;

	// set up mpi
	int rank, size, worker_size, manager_rank;
	std::string hostname;
	MPI::Intracomm comm_world = init_mpi(argc, argv, size, rank, hostname);

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
	//MPI::Intracomm comm_worker = init_workers(MPI::COMM_WORLD, manager_rank);
	//int worker_rank = comm_worker.Get_rank();


	uint64_t t1 = 0, t2 = 0;
	t1 = cciutils::ClockGetTime();

	// decide based on rank of worker which way to process
	if (rank == manager_rank) {
		// manager thread
		manager_process(comm_world, manager_rank, worker_size, maskName, imgDir, outDir, overwrite);
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


#else
int main (int argc, char **argv){
	printf("NOT compiled with MPI.  Using OPENMP if CPU, or GPU (multiple streams)\n");

	// parse the input
	int modecode;
	std::string maskName, imgDir,  outDir;
	bool overwrite;
	int status = parseInput(argc, argv, modecode, maskName, imgDir, outDir, overwrite);
	if (status != 0) return status;

	uint64_t t0 = 0, t1 = 0, t2 = 0;
	t1 = cciutils::ClockGetTime();

	// first get the list of files to process
   	std::vector<std::string> filenames;
	std::vector<std::string> seg_output;
	std::vector<std::string> features_output;

	t0 = cciutils::ClockGetTime();
	getFiles(maskName, imgDir, outDir, filenames, seg_output, features_output, overwrite);

	printf("file read took %lu us\n", t1 - t0);

	int total = filenames.size();
	printf("num files = %d\n", total);
	int i = 0;

	// openmp bag of task
//#define _OPENMP
#if defined (_OPENMP)

	if (omp_get_max_threads() == 1) {
    	printf("1 omp thread\n");
    	while (i < total) {
			// now do some work
			compute(filenames[i].c_str(), seg_output[i].c_str(), features_output[i].c_str());
    		printf("processed %s %s\n", filenames[i].c_str(), seg_output[i].c_str());
    		++i;
    	}

	} else {
    	printf("omp %d\n", omp_get_max_threads());

#pragma omp parallel
	{
#pragma omp single private(i)
		{
			while (i < total) {
				int ti = i;
				// has to use firstprivate - private does not work.
#pragma omp task firstprivate(ti) shared(filenames, seg_output, features_output)
				{
//        				printf("t i: %d, %d \n", i, ti);
					compute(filenames[ti].c_str(), seg_output[ti].c_str(), features_output[ti].c_str());
	        		printf("processed %s %s \n", filenames[ti].c_str(), seg_output[ti].c_str());
				}
				i++;
			}
		}
#pragma omp taskwait
	}
	}
#else
	printf("not omp\n");
	while (i < total) {
		compute(filenames[i].c_str(), seg_output[i].c_str(), features_output[i].c_str());
		printf("processed %s %s \n", filenames[i].c_str(), seg_output[i].c_str());
		++i;
	}
#endif
	t2 = cciutils::ClockGetTime();
	printf("FINISHED in %lu us\n", t2 - t1);

	return 0;
}

#endif



