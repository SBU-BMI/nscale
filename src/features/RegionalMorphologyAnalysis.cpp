/*
 * RegionalMorphologyAnalysis.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: george
 */

#include "RegionalMorphologyAnalysis.h"


RegionalMorphologyAnalysis::RegionalMorphologyAnalysis(string maskInputFileName)
{
	initializeContours(maskInputFileName.c_str());
}

RegionalMorphologyAnalysis::~RegionalMorphologyAnalysis() {
	internal_blobs.clear();
}


void RegionalMorphologyAnalysis::initializeContours(string maskInputFileName)
{
	// read image in grayscale
	IplImage* img_8uc1 = cvLoadImage( maskInputFileName.c_str(), CV_LOAD_IMAGE_GRAYSCALE );

	// create temporary image used to store threshold results
	IplImage* img_edge = cvCreateImage( cvGetSize(img_8uc1), 8, 1 );
	cvThreshold(img_8uc1, img_edge, 128, 255, CV_THRESH_BINARY );

	// create storage to be used by the findContours
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;

	int Nc = cvFindContours(
		img_edge,
		storage,
		&first_contour,
		sizeof(CvContour),
		CV_RETR_TREE,
		CV_CHAIN_APPROX_SIMPLE
		);

	// for all components found in the same first level
	for(CvSeq* c= first_contour; c!= NULL; c=c->h_next){
		// create a blob with the current component and store it in the region
		Blob* curBlob = new Blob(c);
		this->internal_blobs.push_back(curBlob);
	}

	cvClearMemStorage(storage);
	cvReleaseImage(&img_8uc1);
	cvReleaseImage(&img_edge);
}

void RegionalMorphologyAnalysis::doAll()
{
	for(int i = 0; i < internal_blobs.size(); i++){
		Blob *curBlob = internal_blobs[i];
		printf("Blob #%d - perimeter=%lf - area=%lf ED=%lf ",  i, curBlob->getPerimeter(), curBlob->getArea(), curBlob->getEquivalentDiameter());

		printf(" MajorAxisLength=%lf MinorAxisLength=%lf Orientation=%lf", curBlob->getMajorAxisLength(), curBlob->getMinorAxisLength(), curBlob->getOrientation());
		printf(" Circularity = %lf Extent = %lf Eccentricity = %lf", curBlob->getCircularity(), curBlob->getExtent(), curBlob->getEccentricity());
		printf(" ConvexArea = %lf Solidity = %lf Deficiency = %lf", curBlob->getConvexArea(), curBlob->getSolidity(), curBlob->getConvexDeficiency());
		printf(" Compactness = %lf FilledArea = %lf Euler# = %d Porosity = %lf", curBlob->getCompacteness(), curBlob->getFilledArea(), curBlob->getEulerNumber(), curBlob->getPorosity());
		printf(" AspectRatio = %lf\n", curBlob->getAspectRatio());
	}
}


