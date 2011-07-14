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
	// get the current time
	// - NULL because we don't care about time zone
	gettimeofday(&startTime, NULL);

	//	IplImage* inputImage = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
	regional->doRegionProps();
	//regional->doAll();
	//	regional->doIntensity(inputImage);
	gettimeofday(&endTime, NULL);
	// calculate time in microseconds
	double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
	double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
	printf("Total Time Taken: %lf\n", tE - tS);

}
