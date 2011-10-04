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
#include <omp.h>
#include "RegionalMorphologyAnalysis.h"

#include "hdf5.h"
#include "hdf5_hl.h"


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
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}
	std::string imagename(argv[1]);
	std::string outDir(argv[2]);
	const char* mode = argc > 4 ? argv[4] : "cpu";

	if (strcasecmp(mode, "cpu") == 0) {
		modecode = cciutils::DEVICE_CPU;
		// get core count
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]));
		}
	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]));
		}

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
		std::cout << "Usage:  " << argv[0] << " <image_filename | image_dir> mask_dir " << "run-id [cpu [numThreads] | mcore [numThreads] | gpu [id]]" << std::endl;
		return -1;
	}

	if (rank == 0) {
		// check to see if it's a directory or a file
		std::string suffix;
		suffix.assign(".tif");

		FileUtils futils(suffix);
		futils.traverseDirectoryRecursive(imagename, filenames);
		std::string dirname;
		if (filenames.size() == 1) {
			dirname = imagename.substr(0, imagename.find_last_of("/\\"));
		} else {
			dirname = imagename;
		}


		std::string temp;
		for (unsigned int i = 0; i < filenames.size(); ++i) {
			maxLenInput = maxLenInput > filenames[i].length() ? maxLenInput : filenames[i].length();
				// generate the output file name
			temp = futils.replaceExt(filenames[i], ".tif", ".mask.pbm");
			temp = futils.replaceDir(temp, dirname, outDir);
			seg_output.push_back(temp);
			maxLenMask = maxLenMask > temp.length() ? maxLenMask : temp.length();
			// generate the output file name
			temp = futils.replaceExt(filenames[i], ".tif", ".features.h5");
			temp = futils.replaceDir(temp, dirname, outDir);
			features_output.push_back(temp);
			maxLenFeatures = maxLenFeatures > temp.length() ? maxLenFeatures : temp.length();
		}
		dataCount= filenames.size();
	}

#ifdef WITH_MPI
	if (rank == 0) {
		perNodeCount = filenames.size() / size + (filenames.size() % size == 0 ? 0 : 1);

		printf("headnode: rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d \n", rank, perNodeCount, maxLenMask, maxLenInput);

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

	// scatter
	MPI::COMM_WORLD.Scatter(inputBufAll, perNodeCount * maxLenInput, MPI::CHAR,
		inputBuf, perNodeCount * maxLenInput, MPI::CHAR,
		0);

	MPI::COMM_WORLD.Scatter(maskBufAll, perNodeCount * maxLenMask, MPI::CHAR,
		maskBuf, perNodeCount * maxLenMask, MPI::CHAR,
		0);
#endif

	if (rank == 0) {
		t3 = cciutils::ClockGetTime();
	} // end if (rank == 0)


#ifdef WITH_MPI
#pragma omp parallel for shared(perNodeCount, inputBuf, maskBuf, maxLenInput, maxLenMask, modecode, rank) private(fin, fmask, t1, t2)
    for (unsigned int i = 0; i < perNodeCount; ++i) {
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
		fmask = std::string(maskBuf + i * maxLenMask, maxLenMask);
#else
#pragma omp parallel for shared(filenames, seg_output, modecode, rank) private(fin, fmask, t1, t2)
    for (unsigned int i = 0; i < dataCount; ++i) {
		fin = filenames[i];
		fmask = seg_output[i];
//		printf("in seq seg loop with rank %d, loop %d\n", rank, i);
#endif

    	int tid = omp_get_thread_num();


//  	std::cout << outfile << std::endl;

		t1 = cciutils::ClockGetTime();

		int status;

		switch (modecode) {
		case cciutils::DEVICE_CPU :
		case cciutils::DEVICE_MCORE :
			status = nscale::HistologicalEntities::segmentNuclei(fin, fmask);
			break;
		case cciutils::DEVICE_GPU :
			status = nscale::gpu::HistologicalEntities::segmentNuclei(fin, fmask);
			break;
		default :
			break;
		}

		if (status != nscale::HistologicalEntities::SUCCESS) {
#ifdef WITH_MPI
			memset(maskBuf + i * maxLenMask, 0, maxLenMask);
#else
			seg_output[i] = std::string("");
#endif
		}
		t2 = cciutils::ClockGetTime();
		printf("%d::%d: segment %d us, in %s\n", rank, tid, (int)(t2-t1), fin.c_str());
//		std::cout << rank <<"::" << tid << ":" << t2-t1 << " us, in " << fin << ", out " << fmask << std::endl;

    }

#ifdef WITH_MPI
    MPI::COMM_WORLD.Barrier();
#endif
    if (rank == 0)  {
    	t4 = cciutils::ClockGetTime();
		printf("**** Segment took %d us\n", (int)(t4-t3));
	//	std::cout << "**** Segment took " << t4-t3 << " us" << std::endl;

		t3 = cciutils::ClockGetTime();
    }

#ifdef WITH_MPI
    MPI::COMM_WORLD.Barrier();

    // gather the results
	MPI::COMM_WORLD.Gather(maskBuf, perNodeCount * maxLenMask, MPI::CHAR,
		maskBufAll, perNodeCount * maxLenMask, MPI::CHAR,
		0);

	memset(maskBuf, 0, perNodeCount * maxLenMask * sizeof(char));
	memset(inputBuf, 0, perNodeCount * maxLenInput * sizeof(char));
	featuresBuf = (char*)malloc(perNodeCount * maxLenFeatures * sizeof(char));

	char* currPos;
	unsigned int newId;
	if (rank == 0) {
	    // and clean it up
		newId = 0;
		for (unsigned int i = 0; i < perNodeCount * size; ++i) {
			currPos = maskBufAll + i * maxLenMask;

			if (*currPos > 0 && newId != i) {  // not an empty string
				memset(maskBufAll + newId * maxLenMask, 0, maxLenMask);
				strncpy(maskBufAll + newId * maxLenMask, currPos, maxLenMask);
				memset(inputBufAll + newId * maxLenInput, 0, maxLenInput);
				strncpy(inputBufAll + newId * maxLenInput, inputBufAll + i * maxLenInput, maxLenInput);
				memset(featuresBufAll + newId * maxLenFeatures, 0, maxLenFeatures);
				strncpy(featuresBufAll + newId * maxLenFeatures, featuresBufAll + i * maxLenFeatures, maxLenFeatures);

				fmask = std::string(maskBufAll + newId * maxLenMask, maxLenMask);
				fin = std::string(inputBufAll + newId * maxLenInput, maxLenInput);
				ffeatures = std::string(featuresBufAll + newId * maxLenFeatures, maxLenFeatures);
				printf("in %s, mask %s, feature %s\n", fin.c_str(), fmask.c_str(), ffeatures.c_str());

				++newId;
			}
		}
		memset(maskBufAll + newId*maxLenMask, 0, (perNodeCount*size - newId) * maxLenMask);
		memset(inputBufAll + newId*maxLenInput, 0, (perNodeCount*size - newId) * maxLenInput);
		memset(featuresBufAll + newId*maxLenFeatures, 0, (perNodeCount*size - newId) * maxLenFeatures);

		dataCount = newId;
		perNodeCount = dataCount / size + (dataCount % size == 0 ? 0 : 1);

		printf("headnode: rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d \n", rank, perNodeCount, maxLenMask, maxLenInput);
	}
//	printf("rank: %d\n ", rank);
	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Bcast(&perNodeCount, 1, MPI::INT, 0);

//	printf("rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d \n", rank, perNodeCount, maxLenMask, maxLenInput);

	// allocate the receive buffer

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
#else
	std::vector<std::string> tempInput;
	std::vector<std::string> tempMask;
	std::vector<std::string> tempFeatures;

	dataCount = 0;
	for (unsigned int i = 0; i < seg_output.size(); i++) {
		if (seg_output[i].compare(std::string("")) != 0) {
			tempMask.push_back(seg_output[i]);
			tempInput.push_back(filenames[i]);
			tempFeatures.push_back(features_output[i]);
			++dataCount;
		}
	}
	filenames = tempInput;
	features_output = tempFeatures;
	seg_output = tempMask;
#endif


/*  - DONT NEED THIS ANY MORE
#pragma omp parallel for
#ifdef WITH_MPI
    for (int i = 0; i < perNodeCout; ++i) {
	fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
	fmask = std::string(grayBuf + i * maxLenGray, maxLenGray);
#else
    for (int i = 0; i < filenames.size(); ++i) {
	fin = filenames[i];
	fmask = gray_img[i];
#endif

    	cv::imwrite(fmask, cv::imread(fin, 0));
    	std::cout << " grayscale conversion from " << fin << " to " << fmask << std::endl;
    }

#ifdef WITH_MPI
    MPI::COMM_WORLD.barrier();
#endif

    if (rank ==0 ) {
	t4 = cciutils::ClockGetTime();
	std::cout << "**** BGR2GRAY took " << t4-t3 << " us" << std::endl;

	t3 = cciutils::ClockGetTime();
    }
*/
#ifdef WITH_MPI
#pragma omp parallel for shared(perNodeCount, inputBuf, maskBuf, featuresBuf, maxLenInput, maxLenMask, maxLenFeatures, rank) private(fin, fmask, ffeatures)
    for (unsigned int i = 0; i < perNodeCount; ++i) {
		fmask = std::string(maskBuf + i * maxLenMask, maxLenMask);
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
		ffeatures = std::string(featuresBuf + i * maxLenFeatures, maxLenFeatures);
		printf("in MPI feature loop with rank %d, loop %d.  %s, %s, %s\n", rank, i, fin.c_str(), fmask.c_str(), ffeatures.c_str());

#else
#pragma omp parallel for shared(filenames, seg_output, features_output, rank) private(fin, fmask, ffeatures)
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


    	int tid = omp_get_thread_num();

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
		printf("%d::%d: features %d us, in %s, out %s\n", rank, tid, (int)(t2-t1), fin.c_str(), fmask.c_str());

		delete regional;

		printf("%d::%d: features: %d, %d, %d, %d\n", rank, tid, morphoFeatures.size(), intensityFeatures.size(), gradientFeatures.size(), haralickFeatures.size());
		t1 = cciutils::ClockGetTime();

		// create a single data field
		if (morphoFeatures.size() > 0) {
			unsigned int recordSize = morphoFeatures[0].size() + intensityFeatures[0].size() + gradientFeatures[0].size() + haralickFeatures[0].size();
			unsigned int featureSize;
			float *data = new float[morphoFeatures.size() * recordSize];
			float *currData;
			for(unsigned int i = 0; i < morphoFeatures.size(); i++) {
//				printf("[%d] m ", i);
				currData = data + i * recordSize;
				featureSize = morphoFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < morphoFeatures[i].size()) {
						currData[j] = morphoFeatures[i][j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", morphoFeatures[i][j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] i ", i);

				currData += featureSize;
				featureSize = intensityFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < intensityFeatures[i].size()) {
						currData[j] = intensityFeatures[i][j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", intensityFeatures[i][j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] g ", i);

				currData += featureSize;
				featureSize = gradientFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < gradientFeatures[i].size()) {
						currData[j] = gradientFeatures[i][j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", gradientFeatures[i][j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] h ", i);

				currData += featureSize;
				featureSize = haralickFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < haralickFeatures[i].size()) {
						currData[j] = haralickFeatures[i][j];
	#ifdef	PRINT_FEATURES
						printf("%f, ", haralickFeatures[i][j]);
	#endif
					}
				}
//				printf("\n");

			}
			  hid_t file_id;
			  herr_t hstatus;

			hsize_t dims[2];
			file_id = H5Fcreate ( ffeatures.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );

			dims[0] = morphoFeatures.size(); dims[1] = recordSize;
			hstatus = H5LTmake_dataset ( file_id, "/features",
					2, // rank
					dims, // dims
						 H5T_NATIVE_FLOAT, data );

			// attach the attributes
			unsigned long ul = morphoFeatures.size();
			hstatus = H5LTset_attribute_ulong ( file_id, "/features", "num_objs", &ul, 1 );
			ul = recordSize;
			hstatus = H5LTset_attribute_ulong ( file_id, "/features", "num_coords", &ul, 1 );
			hstatus = H5LTset_attribute_string ( file_id, "/features", "image_file", fin.c_str() );
			H5Fclose ( file_id );

			delete [] data;

		}
		t2 = cciutils::ClockGetTime();
		printf("%d::%d: hdf5 %d us, in %s, out %s\n", rank, tid, (int)(t2-t1), fin.c_str(), ffeatures.c_str());



	//	std::cout << rank << "::" << tid << ":" << fin << std::endl;
    }
#ifdef WITH_MPI
	MPI::COMM_WORLD.Barrier();
#endif

    if (rank == 0) {

    	t4 = cciutils::ClockGetTime();
		printf("**** Feature Extraction took %d us \n", (int)(t4-t3));
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


