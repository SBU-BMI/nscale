/*
 * TestRegional.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */
#ifdef _MSC_VER
#include "time_win.h"
#else
#include <sys/time.h>
#endif
#include "RegionalMorphologyAnalysis.h"
#include "Blob.h"
#include "Contour.h"

// ProcessTime example
struct timeval startTime;
struct timeval endTime;

void beginTimer(){
	gettimeofday(&startTime, NULL);
}

void printElapsedTime(){
	gettimeofday(&endTime, NULL);
	// calculate time in microseconds
	double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
	double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
	printf(" %lf\n", tE - tS);
}

int main (int argc, char **argv){

	// read image in mask image that is expected to be binary
	IplImage *originalImageMask = cvLoadImage(argv[1], -1 );
	if(originalImageMask == NULL){
		cout << "Could not load image: "<< argv[1] <<endl;
		exit(1);
	}else{
		if(originalImageMask->nChannels != 1){
			cout << "Error: Mask image should have only one channel"<<endl;
			exit(1);
		}
	}

	// read actual image
	IplImage *originalImage = cvLoadImage(argv[2], -1 );

	if(originalImage == NULL){
		cout << "Cound not open input image:"<< argv[2] <<endl;
		exit(1);
	}else{
		if(originalImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel"<<endl;
			cvReleaseImage(&originalImage);
			exit(1);
		}
	}

	beginTimer();

	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(originalImageMask, originalImage);
//	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);
	printf("IntRegional: ");
	printElapsedTime();

	// Warm up GPU
	regional->uploadImageToGPU();
	regional->releaseGPUImage();
//	regional->printStats();

	// get the current time
	// - NULL because we don't care about time zone
	int procType=Constant::GPU;
	bool includeCopyDataCost=true;
	//	regional->doRegionProps();

	beginTimer();
	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}else{
		regional->uploadImageToGPU();
		regional->uploadImageMaskToGPU();
		regional->uploadImageMaskNucleusToGPU();
	}


	regional->doCoocPropsBlob(Constant::ANGLE_0, procType);
	printf("CoocPropsBlob_0: ");

	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}


	regional->doCoocPropsBlob(Constant::ANGLE_45, procType);
	printf("CoocPropsBlob_45: ");

	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}

	regional->doCoocPropsBlob(Constant::ANGLE_90, procType);
	printf("CoocPropsBlob_90: ");

	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}

	regional->doCoocPropsBlob(Constant::ANGLE_135, procType);
	printf("CoocPropsBlob_135: ");
	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}
	regional->doCoocPropsBlob(Constant::ANGLE_0, procType);
	printf("CoocPropsBlob_0: ");

	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}


	regional->doCoocPropsBlob(Constant::ANGLE_45, procType);
	printf("CoocPropsBlob_45: ");

	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}

	regional->doCoocPropsBlob(Constant::ANGLE_90, procType);
	printf("CoocPropsBlob_90: ");

	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}

	regional->doCoocPropsBlob(Constant::ANGLE_135, procType);
	printf("CoocPropsBlob_135: ");
	if(includeCopyDataCost){
		regional->releaseGPUImage();
		regional->releaseGPUMask();
		regional->releaseImageMaskNucleusToGPU();
	}

	delete regional;
	printElapsedTime();
//
//	beginTimer();
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_0, procType, false);
//	printf("PropsImg_0: ");
//	printElapsedTime();
//
//	beginTimer();
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_45, procType, false);
//	printf("PropsImg_45: ");
//	printElapsedTime();
//
//	beginTimer();
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_90, procType, false);
//	printf("PropsImg_90: ");
//	printElapsedTime();
//
//	beginTimer();
//	regional->inertiaFromCoocMatrix(Constant::ANGLE_135, procType, false);
//	printf("PropsImg_135: ");
//	printElapsedTime();
//
//
//	beginTimer();
//	regional->calcMaxIntensity(false, procType);
//	printf("MaxIntensity_image_not_masked:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}
//
//	beginTimer();
//	regional->calcMaxIntensity(true, procType, false);
//	printf("MaxIntensity_image_masked:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}
//
//
//	beginTimer();
//	regional->calcMaxGradientMagnitude(true, procType, false);
//	printf("MaxGrad_image_masked:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}
//
//	beginTimer();
//	regional->calcMaxGradientMagnitude(false, procType, false);
//	printf("MaxGrad_image_no_masked:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}
//
//	beginTimer();
//	regional->doIntensityBlob(procType);
//	printf("doIntensity_blob:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}
//
//	beginTimer();
//	regional->doGradientBlob(procType);
//	printf("doGradient_blob:");
//	printElapsedTime();
//
//	beginTimer();
//	regional->calcSobelArea(procType, 2, 2, 7, false, procType);
//	printf("Sobel_not_masked:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}
//
//	beginTimer();
//	regional->calcSobelArea(procType, 2, 2, 7, true);
//	printf("Sobel_masked:");
//	printElapsedTime();
//
//	if(includeCopyDataCost){
//		regional->releaseGPUImage();
//		regional->releaseGPUMask();
//		regional->releaseImageMaskNucleusToGPU();
//	}




/*	regional->doIntensityBlob(procType);
	regional->doGradientBlob(procType);*/
//	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(Constant::ANGLE_0, procType)<<endl;
/*	regional->inertiaFromCoocMatrix(Constant::ANGLE_45, procType);
	regional->inertiaFromCoocMatrix(Constant::ANGLE_90, procType);
	regional->inertiaFromCoocMatrix(Constant::ANGLE_135, procType);*/

//	regional->calcMaxIntensity(false, procType);
/*	regional->calcMaxGradientMagnitude(false, procType);
	regional->calcCannyArea(procType, 0, 130, 7, 0);
	regional->calcSobelArea(procType, 1, 1, 7, false);*/


/*	regional->calcMaxIntensity(true, procType, false);
	regional->calcMaxGradientMagnitude(true, procType, false);
	regional->calcCannyArea(procType, 0, 130, 7, 0);
	regional->calcSobelArea(procType, 1, 1, 7, true);*/




/*	bool reuseResults = true;

	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_0, CPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_0, CPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_0, CPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_0, CPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_0, CPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_0, CPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_0, CPU, reuseResults)<<endl;
	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_45, CPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_45, CPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_45, CPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_45, CPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_45, CPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_45, CPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_45, CPU, reuseResults)<<endl;
	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_90, CPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_90, CPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_90, CPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_90, CPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_90, CPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_90, CPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_90, CPU, reuseResults)<<endl;
	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_135, CPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_135, CPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_135, CPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_135, CPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_135, CPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_135, CPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_135, CPU, reuseResults)<<endl;
	cout << " Image mean intensity = "<< regional->calcMeanIntensity(true, CPU, reuseResults) <<endl;
	cout << " Image mean intensity = "<< regional->calcMeanIntensity(false, CPU, reuseResults) <<endl;
	cout << " Std intensity = "<< regional->calcStdIntensity(true, CPU, reuseResults) <<endl;
	cout << " Std intensity = "<< regional->calcStdIntensity(false, CPU, reuseResults) <<endl;
	cout << " Image median intensity = " << regional->calcMedianIntensity(false, CPU, reuseResults)<<endl;
	cout << " Image median intensity = " << regional->calcMedianIntensity(true, CPU, reuseResults)<<endl;
	cout << " Image min intensity = " << regional->calcMinIntensity(false, CPU, reuseResults)<<endl;
	cout << " Image min intensity = " << regional->calcMinIntensity(true, CPU, reuseResults)<<endl;
	cout << " Image max intensity = " << regional->calcMaxIntensity(false, CPU, reuseResults)<<endl;
	cout << " Image max intensity = " << regional->calcMaxIntensity(true, CPU, reuseResults)<<endl;
	cout << " Image first quartile intensity = " << regional->calcFirstQuartileIntensity(false, CPU, reuseResults)<<endl;
	cout << " Image first quartile intensity = " << regional->calcFirstQuartileIntensity(true, CPU, reuseResults)<<endl;
	cout << " Image second quartile intensity = " << regional->calcSecondQuartileIntensity(false, CPU, reuseResults) <<endl;
	cout << " Image second quartile intensity = " << regional->calcSecondQuartileIntensity(true, CPU, reuseResults) <<endl;
	cout << " Image third quartile intensity = " << regional->calcThirdQuartileIntensity(false, CPU, reuseResults)<<endl;
	cout << " Image third quartile intensity = " << regional->calcThirdQuartileIntensity(true, CPU, reuseResults)<<endl;
	cout << " Sobel area CPU: "<< regional->calcSobelArea(CPU, 1, 1, 7, reuseResults) <<endl;
	cout << " Image median grad mag. = "<< regional->calcMedianGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image median grad mag. = "<< regional->calcMedianGradientMagnitude(true, CPU, reuseResults)<<endl;
	cout << " Image mean grad mag. = "<< regional->calcMeanGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image mean grad mag. = "<< regional->calcMeanGradientMagnitude(true, CPU, reuseResults)<<endl;
	cout << " Image min grad mag. = "<< regional->calcMinGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image min grad mag. = "<< regional->calcMinGradientMagnitude(true, CPU, reuseResults)<<endl;
	cout << " Image max grad mag. = "<< regional->calcMaxGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image max grad mag. = "<< regional->calcMaxGradientMagnitude(true, CPU, reuseResults)<<endl;
	cout << " Image first grad mag. = "<< regional->calcFirstQuartileGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image first grad mag. = "<< regional->calcFirstQuartileGradientMagnitude(true, CPU, reuseResults)<<endl;
	cout << " Image second grad mag. = "<< regional->calcSecondQuartileGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image second grad mag. = "<< regional->calcSecondQuartileGradientMagnitude(true, CPU, reuseResults)<<endl;
	cout << " Image third grad mag. = "<< regional->calcThirdQuartileGradientMagnitude(false, CPU, reuseResults)<<endl;
	cout << " Image third grad mag. = "<< regional->calcThirdQuartileGradientMagnitude(true, CPU, reuseResults)<<endl;*/



/*	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_0, GPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_0, GPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_0, GPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_0, GPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_0, GPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_0, GPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_0, GPU, reuseResults)<<endl;
	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_45, GPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_45, GPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_45, GPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_45, GPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_45, GPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_45, GPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_45, GPU, reuseResults)<<endl;
	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_90, GPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_90, GPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_90, GPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_90, GPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_90, GPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_90, GPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_90, GPU, reuseResults)<<endl;
	cout << "Inertia = "<< regional->inertiaFromCoocMatrix(ANGLE_135, GPU, reuseResults) <<endl;
	cout << "Energy = "<< regional->energyFromCoocMatrix(ANGLE_135, GPU, reuseResults) <<endl;
	cout << "Homogeneity = "<<regional->homogeneityFromCoocMatrix(ANGLE_135, GPU, reuseResults)<<endl;
	cout << "Entropy = "<< regional->entropyFromCoocMatrix(ANGLE_135, GPU, reuseResults) <<endl;
	cout << "MaxProb = "<< regional->maximumProbabilityFromCoocMatrix(ANGLE_135, GPU, reuseResults) <<endl;
	cout << "Cluster shade = " <<regional->clusterShadeFromCoocMatrix(ANGLE_135, GPU, reuseResults)<<endl;
	cout << "Cluster prominence = " <<regional->clusterProminenceFromCoocMatrix(ANGLE_135, GPU, reuseResults)<<endl;
	cout << " Image mean intensity = "<< regional->calcMeanIntensity(true, GPU, reuseResults) <<endl;
	cout << " Image mean intensity = "<< regional->calcMeanIntensity(false, GPU, reuseResults) <<endl;
	cout << " Std intensity = "<< regional->calcStdIntensity(true, GPU, reuseResults) <<endl;
	cout << " Std intensity = "<< regional->calcStdIntensity(false, GPU, reuseResults) <<endl;
	cout << " Image median intensity = " << regional->calcMedianIntensity(false, GPU, reuseResults)<<endl;
	cout << " Image median intensity = " << regional->calcMedianIntensity(true, GPU, reuseResults)<<endl;
	cout << " Image min intensity = " << regional->calcMinIntensity(false, GPU, reuseResults)<<endl;
	cout << " Image min intensity = " << regional->calcMinIntensity(true, GPU, reuseResults)<<endl;
	cout << " Image max intensity = " << regional->calcMaxIntensity(false, GPU, reuseResults)<<endl;
	cout << " Image max intensity = " << regional->calcMaxIntensity(true, GPU, reuseResults)<<endl;
	cout << " Image first quartile intensity = " << regional->calcFirstQuartileIntensity(false, GPU, reuseResults)<<endl;
	cout << " Image first quartile intensity = " << regional->calcFirstQuartileIntensity(true, GPU, reuseResults)<<endl;
	cout << " Image second quartile intensity = " << regional->calcSecondQuartileIntensity(false, GPU, reuseResults) <<endl;
	cout << " Image second quartile intensity = " << regional->calcSecondQuartileIntensity(true, GPU, reuseResults) <<endl;
	cout << " Image third quartile intensity = " << regional->calcThirdQuartileIntensity(false, GPU, reuseResults)<<endl;
	cout << " Image third quartile intensity = " << regional->calcThirdQuartileIntensity(true, GPU, reuseResults)<<endl;
	cout << " Sobel area GPU: "<< regional->calcSobelArea(GPU, 1, 1, 7, true) <<endl;
	cout << " Image median grad mag. = "<< regional->calcMedianGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image median grad mag. = "<< regional->calcMedianGradientMagnitude(true, GPU, reuseResults)<<endl;
	cout << " Image mean grad mag. = "<< regional->calcMeanGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image mean grad mag. = "<< regional->calcMeanGradientMagnitude(true, GPU, reuseResults)<<endl;
	cout << " Image min grad mag. = "<< regional->calcMinGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image min grad mag. = "<< regional->calcMinGradientMagnitude(true, GPU, reuseResults)<<endl;
	cout << " Image max grad mag. = "<< regional->calcMaxGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image max grad mag. = "<< regional->calcMaxGradientMagnitude(true, GPU, reuseResults)<<endl;
	cout << " Image first grad mag. = "<< regional->calcFirstQuartileGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image first grad mag. = "<< regional->calcFirstQuartileGradientMagnitude(true, GPU, reuseResults)<<endl;
	cout << " Image second grad mag. = "<< regional->calcSecondQuartileGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image second grad mag. = "<< regional->calcSecondQuartileGradientMagnitude(true, GPU, reuseResults)<<endl;
	cout << " Image third grad mag. = "<< regional->calcThirdQuartileGradientMagnitude(false, GPU, reuseResults)<<endl;
	cout << " Image third grad mag. = "<< regional->calcThirdQuartileGradientMagnitude(true, GPU, reuseResults)<<endl;*/

	gettimeofday(&endTime, NULL);
	// calculate time in microseconds
	double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
	double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
	printf("Total Time Taken: %lf\n", tE - tS);

	cvReleaseImage(&originalImage);
	cvReleaseImage(&originalImageMask);
}
