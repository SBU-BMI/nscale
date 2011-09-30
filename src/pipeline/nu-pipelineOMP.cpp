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
	std::vector<std::string> gray_img;
	char *inputBufAll, *outputBufAll;
	inputBufAll=NULL;
	outputBufAll=NULL;

	// relevant to all nodes
	int modecode = 0;
	uint64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;
	std::string fin, fout;
	unsigned int perNodeCount=0, maxLenInput=0, maxLenOutput=0;
	char *inputBuf, *outputBuf;
	inputBuf=NULL;
	outputBuf=NULL;

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
			maxLenOutput = maxLenOutput > temp.length() ? maxLenOutput : temp.length();
		}
	}

#ifdef WITH_MPI
	if (rank == 0) {
		perNodeCount = filenames.size() / size + (filenames.size() % size == 0 ? 0 : 1);

		printf("headnode: rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d \n", rank, perNodeCount, maxLenOutput, maxLenInput);

		// allocate the sendbuffer
		inputBufAll= (char*)malloc(perNodeCount * size * maxLenInput * sizeof(char));
		outputBufAll= (char*)malloc(perNodeCount * size * maxLenOutput * sizeof(char));
		memset(inputBufAll, 0, perNodeCount * size * maxLenInput);
		memset(outputBufAll, 0, perNodeCount * size * maxLenOutput);

		// copy data into the buffers
		for (unsigned int i = 0; i < filenames.size(); ++i) {
			strncpy(inputBufAll + i * maxLenInput, filenames[i].c_str(), maxLenInput);
			strncpy(outputBufAll + i * maxLenOutput, seg_output[i].c_str(), maxLenOutput);
		}
	}
//	printf("rank: %d\n ", rank);
	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Bcast(&perNodeCount, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&maxLenInput, 1, MPI::INT, 0);
	MPI::COMM_WORLD.Bcast(&maxLenOutput, 1, MPI::INT, 0);


//	printf("rank is %d here.  perNodeCount is %d, outputLen %d, inputLen %d \n", rank, perNodeCount, maxLenOutput, maxLenInput);

	// allocate the receive buffer
	inputBuf = (char*)malloc(perNodeCount * maxLenInput * sizeof(char));
	outputBuf = (char*)malloc(perNodeCount * maxLenOutput * sizeof(char));

	// scatter
	MPI::COMM_WORLD.Scatter(inputBufAll, perNodeCount * maxLenInput, MPI::CHAR,
		inputBuf, perNodeCount * maxLenInput, MPI::CHAR,
		0);

	MPI::COMM_WORLD.Scatter(outputBufAll, perNodeCount * maxLenOutput, MPI::CHAR,
		outputBuf, perNodeCount * maxLenOutput, MPI::CHAR,
		0);
#endif

	if (rank == 0) {
		t3 = cciutils::ClockGetTime();
	} // end if (rank == 0)


#ifdef WITH_MPI
#pragma omp parallel for shared(perNodeCount, inputBuf, outputBuf, maxLenInput, maxLenOutput, modecode, rank) private(fin, fout, t1, t2)
    for (unsigned int i = 0; i < perNodeCount; ++i) {
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
		fout = std::string(outputBuf + i * maxLenOutput, maxLenOutput);
//		printf("in MPI seg loop with rank %d, loop %d\n", rank, i);
#else
#pragma omp parallel for shared(filenames, seg_output, modecode, rank) private(fin, fout, t1, t2)
    for (unsigned int i = 0; i < filenames.size(); ++i) {
		fin = filenames[i];
		fout = seg_output[i];
//		printf("in seq seg loop with rank %d, loop %d\n", rank, i);
#endif

    	int tid = omp_get_thread_num();


//  	std::cout << outfile << std::endl;

		t1 = cciutils::ClockGetTime();

		int status;

		switch (modecode) {
		case cciutils::DEVICE_CPU :
		case cciutils::DEVICE_MCORE :
			status = nscale::HistologicalEntities::segmentNuclei(fin, fout);
			break;
		case cciutils::DEVICE_GPU :
			status = nscale::gpu::HistologicalEntities::segmentNuclei(fin, fout);
			break;
		default :
			break;
		}

		t2 = cciutils::ClockGetTime();
		printf("%d::%d: %d us, in %s, out %s\n", rank, tid, (int)(t2-t1), fin.c_str(), fout.c_str());
//		std::cout << rank <<"::" << tid << ":" << t2-t1 << " us, in " << fin << ", out " << fout << std::endl;

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
/*  - DONT NEED THIS ANY MORE
#pragma omp parallel for
#ifdef WITH_MPI
    for (int i = 0; i < perNodeCout; ++i) {
	fin = std::string(inputBuf + i * maxLenInput, maxLenInput);
	fout = std::string(grayBuf + i * maxLenGray, maxLenGray);
#else
    for (int i = 0; i < filenames.size(); ++i) {
	fin = filenames[i];
	fout = gray_img[i];
#endif

    	cv::imwrite(fout, cv::imread(fin, 0));
    	std::cout << " grayscale conversion from " << fin << " to " << fout << std::endl;
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
#pragma omp parallel for shared(perNodeCount, inputBuf, outputBuf, maxLenInput, maxLenOutput, rank) private(fin, fout)
    for (unsigned int i = 0; i < perNodeCount; ++i) {
		fout = std::string(outputBuf + i * maxLenOutput, maxLenOutput);
		fin = std::string(inputBuf + i * maxLenInput, maxLenInput);

#else
#pragma omp parallel for shared(filenames, seg_output, rank) private(fin, fout)
    for (unsigned int i = 0; i < filenames.size(); ++i) {
		fout = seg_output[i];
		fin = filenames[i];
#endif

    	int tid = omp_get_thread_num();

    	// This is another option for inialize the features computation, where the path to the images are given as parameter
    	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(fout, fin);

    	/////////////// Computes Morphometry based features ////////////////////////
    	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
    	//	Area; MajorAxisLength; MinorAxisLength; Eccentricity; Orientation; ConvexArea; FilledArea; EulerNumber;
    	// 	EquivalentDiameter; Solidity; Extent; Perimeter; ConvexDeficiency; Compacteness; Porosity; AspectRatio;
    	//	BendingEnergy; ReflectionSymmetry; CannyArea; SobelArea;
    	vector<vector<float> > morphoFeatures;
    	regional->doMorphometryFeatures(morphoFeatures);

#ifdef	PRINT_FEATURES
	for(unsigned int i = 0; i < morphoFeatures.size(); i++){
		printf("Id = %d ", i);
		for(unsigned int j = 0; j < morphoFeatures[i].size(); j++){
			printf("MorphFeature %d = %f ", j, morphoFeatures[i][j]);
		}
		printf("\n");
	}
#endif

		/////////////// Computes Pixel Intensity based features ////////////////////////
		// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
		// 	MeanIntensity; MedianIntensity; MinIntensity; MaxIntensity; FirstQuartileIntensity; ThirdQuartileIntensity;
		vector<vector<float> > intensityFeatures;
		regional->doIntensityBlob(intensityFeatures);

#ifdef	PRINT_FEATURES
	for(unsigned int i = 0; i < intensityFeatures.size(); i++){
		printf("Id = %d ", i);
		for(unsigned int j = 0; j < intensityFeatures[i].size(); j++){
			printf("IntensityFeature %d = %f ", j, intensityFeatures[i][j]);
		}
		printf("\n");
	}
#endif
		/////////////// Computes Gradient based features ////////////////////////
		// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
		// MeanGradMagnitude; MedianGradMagnitude; MinGradMagnitude; MaxGradMagnitude; FirstQuartileGradMagnitude; ThirdQuartileGradMagnitude;
		vector<vector<float> > gradientFeatures;
		regional->doGradientBlob(gradientFeatures);

#ifdef	PRINT_FEATURES
	for(unsigned int i = 0; i < gradientFeatures.size(); i++){
		printf("Id = %d ", i);
		for(unsigned int j = 0; j < gradientFeatures[i].size(); j++){
			printf("GradientFeature %d = %f ", j, gradientFeatures[i][j]);
		}
		printf("\n");
	}
#endif
		/////////////// Computes Haralick based features ////////////////////////
		// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
		// 	Inertia; Energy; Entropy; Homogeneity; MaximumProbability; ClusterShade; ClusterProminence
		vector<vector<float> > haralickFeatures;
		regional->doCoocPropsBlob(haralickFeatures);

#ifdef	PRINT_FEATURES
	for(unsigned int i = 0; i < haralickFeatures.size(); i++){
		printf("Id = %d ", i);
		for(unsigned int j = 0; j < haralickFeatures[i].size(); j++){
			printf("HaralickFeature %d = %f ", j, haralickFeatures[i][j]);
		}
		printf("\n");
	}
#endif
		delete regional;

		printf("%d::%d: %s\n", rank, tid, fin.c_str());
	//	std::cout << rank << "::" << tid << ":" << fin << std::endl;
    }
#ifdef WITH_MPI
	MPI::COMM_WORLD.Barrier();
#endif

    if (rank == 0) {

    	t4 = cciutils::ClockGetTime();
		printf("**** Feature Extraction took %d us \n", (int)(t4-t3));
	//	std::cout << "**** Feature Extraction took " << t4-t3 << " us" << std::endl;

		free(inputBufAll);
		free(outputBufAll);
    }


	free(inputBuf);
	free(outputBuf);

#ifdef WITH_MPI
    MPI::Finalize();
#endif


//	waitKey();

	return 0;
}


