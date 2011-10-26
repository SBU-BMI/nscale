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
#include <string>
#include "HistologicalEntities.h"
#include "utils.h"
#include "FileUtils.h"
#include <dirent.h>
#include "RegionalMorphologyAnalysis.h"

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
	std::vector<std::string> seg_output;
	std::vector<std::string> features_output;
	char *inputBufAll, *maskBufAll, *featuresBufAll;
	inputBufAll=NULL;
	maskBufAll=NULL;
	featuresBufAll=NULL;
	int dataCount;

	// relevant to all nodes
	int modecode = 0;
	uint64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	std::string fin, fmask, ffeatures;
	unsigned int perNodeCount=0, maxLenInput=0, maxLenMask=0, maxLenFeatures=0;
	char *inputBuf, *maskBuf, *featuresBuf;
	inputBuf=NULL;
	maskBuf=NULL;
	featuresBuf=NULL;

	if (argc < 4) {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> image_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	std::string maskname(argv[1]);
	std::string outDir(argv[2]);
	const char* mode = argc > 4 ? argv[4] : "cpu";

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		// get core count

#ifdef _OPENMP
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]));
		}
#endif
	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
#ifdef _OPENMP
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]));
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
		if (argc > 5) {
			gpu::setDevice(atoi(argv[5]));
		}
		printf(" number of cuda enabled devices = %d\n", gpu::getCudaEnabledDeviceCount());
	} else {
		std::cout << "Usage:  " << argv[0] << " <mask_filename | mask_dir> image_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	if (rank == 0) {
		// check to see if it's a directory or a file
		std::string suffix;
		suffix.assign(".mask.pbm");

		FileUtils futils(suffix);
		futils.traverseDirectoryRecursive(maskname, seg_output);
		std::string dirname;
		if (filenames.size() == 1) {
			dirname = maskname.substr(0, maskname.find_last_of("/\\"));
		} else {
			dirname = maskname;
		}


		std::string temp, tempdir;
		for (unsigned int i = 0; i < seg_output.size(); ++i) {
			maxLenMask = maxLenMask > seg_output[i].length() ? maxLenMask : seg_output[i].length();
				// generate the input file name
			temp = futils.replaceExt(seg_output[i], ".mask.pbm", ".tif");
			temp = futils.replaceDir(temp, dirname, outDir);
			filenames.push_back(temp);
			maxLenInput = maxLenInput > temp.length() ? maxLenInput : temp.length();

			// generate the output file name
			temp = futils.replaceExt(seg_output[i], ".mask.pbm", ".features.h5");
			features_output.push_back(temp);
			maxLenFeatures = maxLenFeatures > temp.length() ? maxLenFeatures : temp.length();
		}
		dataCount= seg_output.size();
	}

#ifdef WITH_MPI
	if (rank == 0) {
		printf("headnode: total count is %d, size is %d\n", seg_output.size(), size);

		perNodeCount = seg_output.size() / size + (seg_output.size() % size == 0 ? 0 : 1);

		printf("headnode: rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d %d \n", rank, perNodeCount, maxLenFeatures, maxLenInput, maxLenMask);

		// allocate the sendbuffer
		inputBufAll= (char*)malloc(perNodeCount * size * maxLenInput * sizeof(char));
		maskBufAll= (char*)malloc(perNodeCount * size * maxLenMask * sizeof(char));
		featuresBufAll= (char*)malloc(perNodeCount * size * maxLenFeatures * sizeof(char));
		memset(inputBufAll, 0, perNodeCount * size * maxLenInput);
		memset(maskBufAll, 0, perNodeCount * size * maxLenMask);
		memset(featuresBufAll, 0, perNodeCount * size * maxLenFeatures);

		// copy data into the buffers
		for (unsigned int i = 0; i < filenames.size(); ++i) {
			strncpy(inputBufAll + i * maxLenInput, filenames[i].c_str(), maxLenInput);
			strncpy(maskBufAll + i * maxLenMask, seg_output[i].c_str(), maxLenMask);
			strncpy(featuresBufAll + i * maxLenFeatures, features_output[i].c_str(), maxLenFeatures);

		}
	}
	//	printf("rank: %d\n ", rank);
	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Bcast(&perNodeCount, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&maxLenInput, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&maxLenMask, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&maxLenFeatures, 1, MPI::INT, 0);


//	printf("rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d \n", rank, perNodeCount, maxLenMask, maxLenInput);

	// allocate the receive buffer
	inputBuf = (char*)malloc(perNodeCount * maxLenInput * sizeof(char));
	maskBuf = (char*)malloc(perNodeCount * maxLenMask * sizeof(char));
	featuresBuf = (char*)malloc(perNodeCount * maxLenFeatures * sizeof(char));


	// scatter
	MPI::COMM_WORLD.Scatter(inputBufAll, perNodeCount * maxLenInput, MPI::CHAR,
		inputBuf, perNodeCount * maxLenInput, MPI::CHAR,
		0);

	MPI::COMM_WORLD.Scatter(maskBufAll, perNodeCount * maxLenMask, MPI::CHAR,
		maskBuf, perNodeCount * maxLenMask, MPI::CHAR,
		0);

	MPI::COMM_WORLD.Scatter(featuresBufAll, perNodeCount * maxLenFeatures, MPI::CHAR,
		featuresBuf, perNodeCount * maxLenFeatures, MPI::CHAR,
		0);

	MPI::COMM_WORLD.Barrier();

#endif
	if (rank == 0)	t3 = cciutils::ClockGetTime();

#ifdef WITH_MPI
#pragma omp parallel for shared(perNodeCount, inputBuf, maskBuf, featuresBuf, maxLenInput, maxLenMask, maxLenFeatures, rank) private(fin, fmask, ffeatures, t1, t2)
    for (unsigned int i = 0; i < perNodeCount; ++i) {
		fmask = std::string(maskBuf + i * maxLenMask, maxLenMask);
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
		ffeatures = std::string(featuresBuf + i * maxLenFeatures, maxLenFeatures);
		printf("in MPI feature loop with rank %d, loop %d.  %s, %s, %s\n", rank, i, fin.c_str(), fmask.c_str(), ffeatures.c_str());

#else
#pragma omp parallel for shared(filenames, seg_output, features_output, rank) private(fin, fmask, ffeatures, t1, t2)
    for (unsigned int i = 0; i < dataCount; ++i) {
		fmask = seg_output[i];
		fin = filenames[i];
		ffeatures = features_output[i];
#endif

		t1 = cciutils::ClockGetTime();

		Mat test = imread(fmask);
		if (!test.data) continue;
		test = imread(fin);
		if (!test.data) continue;

#ifdef _OPENMP
    	int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif
    	// This is another option for inialize the features computation, where the path to the images are given as parameter
    	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(fmask, fin);

    	/////////////// Computes Morphometry based features ////////////////////////
    	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
    	//	Area; MajorAxisLength; MinorAxisLength; Eccentricity; Orientation; ConvexArea; FilledArea; EulerNumber;
    	// 	EquivalentDiameter; Solidity; Extent; Perimeter; ConvexDeficiency; Compacteness; Porosity; AspectRatio;
    	//	BendingEnergy; ReflectionSymmetry; CannyArea; SobelArea;
    	vector<vector<float> > morphoFeatures;
    	regional->doMorphometryFeatures(morphoFeatures);

		/////////////// Computes Pixel Intensity based features ////////////////////////
		// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
		// 	MeanIntensity; MedianIntensity; MinIntensity; MaxIntensity; FirstQuartileIntensity; ThirdQuartileIntensity;
		vector<vector<float> > intensityFeatures;
		regional->doIntensityBlob(intensityFeatures);

		/////////////// Computes Gradient based features ////////////////////////
		// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
		// MeanGradMagnitude; MedianGradMagnitude; MinGradMagnitude; MaxGradMagnitude; FirstQuartileGradMagnitude; ThirdQuartileGradMagnitude;
		vector<vector<float> > gradientFeatures;
		regional->doGradientBlob(gradientFeatures);

		/////////////// Computes Haralick based features ////////////////////////
		// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
		// 	Inertia; Energy; Entropy; Homogeneity; MaximumProbability; ClusterShade; ClusterProminence
		vector<vector<float> > haralickFeatures;
		regional->doCoocPropsBlob(haralickFeatures);


		t2 = cciutils::ClockGetTime();
//		printf("%d::%d: %d features %lu us, in %s, out %s\n", rank, tid, morphoFeatures.size(), t2-t1, fin.c_str(), ffeatures.c_str());

		delete regional;

		t1 = cciutils::ClockGetTime();

		// also calculate the mean and stdev

		// create a single data field
		if (morphoFeatures.size() > 0) {


			unsigned int recordSize = morphoFeatures[0].size() + intensityFeatures[0].size() + gradientFeatures[0].size() + haralickFeatures[0].size();
			unsigned int featureSize;
			float *data = new float[morphoFeatures.size() * recordSize];
			double *sums = new double[recordSize];
			double *squareSums = new double[recordSize];
			for (unsigned int i = 0; i < recordSize; i++) {
				sums[i] = 0.;
				squareSums[i] = 0.;
			}
			float *currData;
			int offset;
			for(unsigned int i = 0; i < morphoFeatures.size(); i++) {

//				printf("[%d] m ", i);
				offset = 0;
				currData = data + i * recordSize;
				featureSize = morphoFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < morphoFeatures[i].size()) {
						currData[j] = morphoFeatures[i][j];

						sums[j + offset] += currData[j];
						squareSums[j + offset] += currData[j] * currData[j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", currData[j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] i ", i);

				offset += featureSize;
				currData += featureSize;
				featureSize = intensityFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < intensityFeatures[i].size()) {
						currData[j] = intensityFeatures[i][j];
						sums[j + offset] += currData[j];
						squareSums[j + offset] += currData[j] * currData[j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", currData[j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] g ", i);

				offset += featureSize;
				currData += featureSize;
				featureSize = gradientFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < gradientFeatures[i].size()) {
						currData[j] = gradientFeatures[i][j];
						sums[j + offset] += currData[j];
						squareSums[j + offset] += currData[j] * currData[j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", currData[j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] h ", i);

				offset += featureSize;
				currData += featureSize;
				featureSize = haralickFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < haralickFeatures[i].size()) {
						currData[j] = haralickFeatures[i][j];
						sums[j + offset] += currData[j];
						squareSums[j + offset] += currData[j] * currData[j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", currData[j]);
	#endif
					}
				}
//				printf("\n");

			}

			// compute the average within the tile

#ifdef	PRINT_FEATURES
			for (unsigned int i = 0; i < recordSize; i++) {
				printf("%f, %f; ", sums[i], squareSums[i]);
			}
			printf("\n");
#endif

#pragma omp critical
			{
			  hid_t file_id;
			  herr_t hstatus;

			hsize_t dims[2];
			file_id = H5Fcreate ( ffeatures.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

			dims[0] = morphoFeatures.size(); dims[1] = recordSize;
			hstatus = H5LTmake_dataset ( file_id, "/data",
					2, // rank
					dims, // dims
						 H5T_NATIVE_FLOAT, data );
			unsigned long ul = morphoFeatures.size();
			hstatus = H5LTset_attribute_ulong ( file_id, "/data", "num_objs", &ul, 1 );
			ul = recordSize;
			hstatus = H5LTset_attribute_ulong ( file_id, "/data", "num_coords", &ul, 1 );

			hstatus = H5LTset_attribute_string ( file_id, "/data", "image_file", fin.c_str() );

			dims[0] = 1;
			hstatus = H5LTmake_dataset ( file_id, "/meta-sum",
					2, // rank
					dims, // dims
						 H5T_NATIVE_DOUBLE, sums );
			ul = morphoFeatures.size();
			hstatus = H5LTset_attribute_ulong ( file_id, "/meta-sum", "num_objs", &ul, 1 );
			ul = recordSize;
			hstatus = H5LTset_attribute_ulong ( file_id, "/meta-sum", "num_coords", &ul, 1 );

			hstatus = H5LTmake_dataset ( file_id, "/meta-square-sum",
					2, // rank
					dims, // dims
						 H5T_NATIVE_DOUBLE, squareSums );

			// attach the attributes
			ul = morphoFeatures.size();
			hstatus = H5LTset_attribute_ulong ( file_id, "/meta-square-sum", "num_objs", &ul, 1 );
			ul = recordSize;
			hstatus = H5LTset_attribute_ulong ( file_id, "/meta-square-sum", "num_coords", &ul, 1 );

			H5Fclose ( file_id );
			}

			delete [] data;
			delete [] sums;
			delete [] squareSums;

		}
		t2 = cciutils::ClockGetTime();
//		printf("%d::%d: hdf5 %lu us, in %s, out %s\n", rank, tid, t2-t1, fin.c_str(), ffeatures.c_str());



	//	std::cout << rank << "::" << tid << ":" << fin << std::endl;
    }
#ifdef WITH_MPI
	MPI::COMM_WORLD.Barrier();
#endif

    if (rank == 0) {

    	t4 = cciutils::ClockGetTime();
		printf("**** Feature Extraction took %lu us \n", t4-t3);
	//	std::cout << "**** Feature Extraction took " << t4-t3 << " us" << std::endl;

    }



#ifdef WITH_MPI
    if (rank == 0) {
		free(inputBufAll);
		free(maskBufAll);
		free(featuresBufAll);
    }

	free(inputBuf);
	free(maskBuf);
	free(featuresBuf);


	MPI::Finalize();
#endif


//	waitKey();

	return 0;
}


