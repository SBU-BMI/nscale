/*
 * Test.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#include "RegionalMorphologyAnalysis.h"
#include "Blob.h"
#include "Contour.h"

int main (int argc, char **argv){

	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);
//	IplImage* inputImage = cvLoadImage( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
//	regional->doRegionProps(inputImage);
	regional->doAll();
//	regional->doIntensity(inputImage);

}
