/*
 * Test.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#include "Contour.h"
#include "Blob.h"

int main (int argc, char **argv){

	cvNamedWindow(argv[0], 1);
	IplImage* img_8uc1 = cvLoadImage( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
	IplImage* img_edge = cvCreateImage( cvGetSize(img_8uc1), 8, 1 );
	IplImage* img_8uc3 = cvCreateImage( cvGetSize(img_8uc1), 8, 3 );

	cvThreshold(img_8uc1, img_edge, 128, 255, CV_THRESH_BINARY );

	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;

	int Nc = cvFindContours(
		img_edge,
		storage,
		&first_contour,
		sizeof(CvContour),
		CV_RETR_TREE,
		CV_CHAIN_APPROX_SIMPLE,
		cvPoint(1,1)
	);

	int n=0;
	printf("Total Contours Detected: %d\n", Nc);

	for(CvSeq* c= first_contour; c!= NULL; c=c->h_next){
		cvCvtColor(img_8uc1, img_8uc3, CV_GRAY2BGR);
		cvDrawContours(
			img_8uc3,
			c,
			CV_RGB(0xff,0x00,0x00),
//			CVX_RED,
			CV_RGB(0x00,0x00,0xff),
//			CVX_BLUE,
			0,
			2,
			8
		);
		Blob* curBlob = new Blob(c);
		printf("Blob #%d - perimeter=%lf - area=%lf ED=%lf ",  n, curBlob->getPerimeter(), curBlob->getArea(), curBlob->getEquivalentDiameter());

		if(c->total >= 6){
			printf(" MajorAxisLength=%lf MinorAxisLength=%lf Orientation=%lf", curBlob->getMajorAxisLength(), curBlob->getMinorAxisLength(), curBlob->getOrientation());
			printf(" Circularity = %lf Extent = %lf Eccentricity = %lf", curBlob->getCircularity(), curBlob->getExtent(), curBlob->getEccentricity());
			printf(" ConvexArea = %lf Solidity = %lf Deficiency = %lf", curBlob->getConvexArea(), curBlob->getSolidity(), curBlob->getConvexDeficiency());
			printf(" Compactness = %lf FilledArea = %lf Euler# = %d Porosity = %lf\n", curBlob->getCompacteness(), curBlob->getFilledArea(), curBlob->getEulerNumber(), curBlob->getPorosity());
		}else{
			printf("\n");
		}
		cvShowImage(argv[0], img_8uc3);
		cvWaitKey(0);
		n++;
		delete curBlob;
	}

	printf("Finished all contours.\n");
	cvCvtColor(img_8uc1, img_8uc3, CV_GRAY2BGR);
	cvShowImage(argv[0], img_8uc3);
	cvWaitKey(0);

	cvReleaseImage(&img_8uc1);
	cvReleaseImage(&img_8uc3);
	cvReleaseImage(&img_edge);
}
