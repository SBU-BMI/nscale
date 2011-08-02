/*
 * TestRegional.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */
#include <sys/time.h>
#include "RegionalMorphologyAnalysis.h"
#include "Blob.h"
#include "Contour.h"

int main (int argc, char **argv){

	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);
	// ProcessTime example
	struct timeval startTime;
	struct timeval endTime;

	// Warm up GPU
	regional->uploadImageToGPU();
	regional->releaseGPUImage();

	// get the current time
	// - NULL because we don't care about time zone
	gettimeofday(&startTime, NULL);

	//	IplImage* inputImage = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
//	regional->doRegionProps();
//	regional->doAll();


	bool reuseResults = true;

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
	cout << " Image third grad mag. = "<< regional->calcThirdQuartileGradientMagnitude(true, CPU, reuseResults)<<endl;



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
	delete regional;
}
