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
#include "BGR2GRAY.h"
#include "ColorDeconv_final.h"

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
			omp_set_num_threads(atoi(argv[5]) > omp_get_max_threads() ? omp_get_max_threads() : atoi(argv[5]));
			printf("number of threads used = %d\n", omp_get_num_threads());
		}
#endif
	} else if (strcasecmp(mode, "mcore") == 0) {
		modecode = cciutils::DEVICE_MCORE;
		// get core count
#ifdef _OPENMP
		if (argc > 5) {
			omp_set_num_threads(atoi(argv[5]) > omp_get_max_threads() ? omp_get_max_threads() : atoi(argv[5]));
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
		if (seg_output.size() == 1) {
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
		for (unsigned int i = 0; i < seg_output.size(); ++i) {
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

#ifdef _OPENMP
    	int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif
		// Load input images
		IplImage *originalImageMask = cvLoadImage(fmask.c_str(), -1);
		IplImage *originalImage = cvLoadImage(fin.c_str(), -1);

		if (! originalImageMask) continue;
		if (! originalImage) continue;

		bool isNuclei = true;

		// Convert color image to grayscale
		IplImage *grayscale = bgr2gray(originalImage);
	//	cvSaveImage("newGrayScale.png", grayscale);

		// This is another option for inialize the features computation, where the path to the images are given as parameter
    	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(originalImageMask, grayscale, true);

    	// Create H and E images
    	Mat image(originalImage);

    	//initialize H and E channels
    	Mat H = Mat::zeros(image.size(), CV_8UC1);
    	Mat E = Mat::zeros(image.size(), CV_8UC1);
    	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
    	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);

    	ColorDeconv(image, M, b, H, E);

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
    	regional->doNucleiPipelineFeatures(nucleiFeatures, grayscale);

    	/////////////// Compute cytoplasm based features ////////////////////////
    	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
    	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
    	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
    	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
    	vector<vector<float> > cytoplasmFeatures_G;
    	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_G, grayscale);


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


		t2 = cciutils::ClockGetTime();
//		printf("%d::%d: %d features %lu us, in %s, out %s\n", rank, tid, nucleiFeatures.size(), t2-t1, fin.c_str(), ffeatures.c_str());

		delete regional;

		cvReleaseImage(&originalImage);
		cvReleaseImage(&originalImageMask);
		cvReleaseImage(&grayscale);

		H.release();
		E.release();
		M.release();
		b.release();

		t1 = cciutils::ClockGetTime();

		// also calculate the mean and stdev

		// create a single data field
		if (nucleiFeatures.size() > 0) {


			unsigned int recordSize = nucleiFeatures[0].size() + cytoplasmFeatures_G[0].size() + cytoplasmFeatures_H[0].size() + cytoplasmFeatures_E[0].size();
			unsigned int featureSize;
			float *data = new float[nucleiFeatures.size() * recordSize];
			float *currData;
			for(unsigned int i = 0; i < nucleiFeatures.size(); i++) {

//				printf("[%d] m ", i);
				currData = data + i * recordSize;
				featureSize = nucleiFeatures[0].size();
				for(unsigned int j = 0; j < featureSize; j++) {
					if (j < nucleiFeatures[i].size()) {
						currData[j] = nucleiFeatures[i][j];

	#ifdef	PRINT_FEATURES
						printf("%f, ", currData[j]);
	#endif
					}
				}
//				printf("\n");
//				printf("[%d] i ", i);

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
//				printf("\n");
//				printf("[%d] g ", i);

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
//				printf("\n");
//				printf("[%d] h ", i);

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

			dims[0] = nucleiFeatures.size(); dims[1] = recordSize;
			hstatus = H5LTmake_dataset ( file_id, "/data",
					2, // rank
					dims, // dims
						 H5T_NATIVE_FLOAT, data );
			hstatus = H5LTset_attribute_string ( file_id, "/data", "image_file", fin.c_str() );

			// attach the attributes
			H5Fclose ( file_id );
			}

			delete [] data;
			nucleiFeatures.clear();
			cytoplasmFeatures_G.clear();
			cytoplasmFeatures_H.clear();
			cytoplasmFeatures_E.clear();
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


