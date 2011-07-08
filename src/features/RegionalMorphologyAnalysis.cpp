/*
 * RegionalMorphologyAnalysis.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: george
 */

#include "RegionalMorphologyAnalysis.h"


RegionalMorphologyAnalysis::RegionalMorphologyAnalysis(string maskInputFileName, string grayInputFileName)
{
	initializeContours(maskInputFileName.c_str());
	originalImage = cvLoadImage( grayInputFileName.c_str(), -1 );
	if(originalImage == NULL){
		cout << "Cound not open input image:"<<grayInputFileName <<endl;
		exit(1);
	}else{
		if(originalImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			cvReleaseImage(&originalImage);
			exit(1);
		}
	}
}

RegionalMorphologyAnalysis::~RegionalMorphologyAnalysis() {
	internal_blobs.clear();
	if(originalImage){
		cvReleaseImage(&originalImage);
	}
}


void RegionalMorphologyAnalysis::initializeContours(string maskInputFileName)
{
	// read image in mask image that is expected to be binary
	IplImage* img_edge = cvLoadImage( maskInputFileName.c_str(), -1 );
	if(img_edge == NULL){
		cout << "Could not load image: "<< maskInputFileName <<endl;
		exit(1);
	}else{
		if(img_edge->nChannels != 1){
			cout << "Error: Mask image should have only one channel"<<endl;
			exit(1);
		}
	}

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
		Blob* curBlob = new Blob(c, cvGetSize(img_edge));
		this->internal_blobs.push_back(curBlob);
	}

	cvClearMemStorage(storage);
	cvReleaseImage(&img_edge);
}

void RegionalMorphologyAnalysis::doRegionProps()
{
	if(originalImage == NULL){
		cout << "DoRegionProps: input image is NULL." <<endl;
		exit(1);
	}else{
		if(originalImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			exit(1);
		}
	}

#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

#pragma omp parallel for
	for(int i = 0; i < internal_blobs.size(); i++){
		Blob *curBlob = internal_blobs[i];
#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(255, 0, 0), CV_RGB(0,0,0));
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif
		printf("Blob #%d - Area=%lf MajorAxisLength=%lf MinorAxisLength=%lf Eccentricity = %lf ", i, curBlob->getArea(),curBlob->getMajorAxisLength(), curBlob->getMinorAxisLength(), curBlob->getEccentricity());
		printf("Orientation=%lf ConvexArea=%lf FilledArea=%lf EulerNumber=%d EquivDiameter=%lf ", curBlob->getOrientation(), curBlob->getConvexArea(), curBlob->getFilledArea(), curBlob->getEulerNumber(), curBlob->getEquivalentDiameter());
		printf("Solidity=%lf Extent=%lf Perimeter=%lf MeanIntensity=%lf MinIntensity=%d MaxIntensity=%d\n", curBlob->getSolidity(), curBlob->getExtent(), curBlob->getPerimeter(), curBlob->getMeanIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage));



		printf(" ConvexArea = %lf Solidity = %lf Deficiency = %lf", curBlob->getConvexArea(), curBlob->getSolidity(), curBlob->getConvexDeficiency());
		printf(" Compactness = %lf FilledArea = %lf Euler# = %d Porosity = %lf", curBlob->getCompacteness(), curBlob->getFilledArea(), curBlob->getEulerNumber(), curBlob->getPorosity());
		printf("Orientation=%lf ", curBlob->getOrientation());
		printf(" MeanPixelIntensity=%lf MedianPixelIntensity=%d MinPixelIntensity=%d MaxPixelIntensity=%d \n", curBlob->getMeanIntensity(originalImage), curBlob->getMedianIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage));

#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(0, 0, 255), CV_RGB(0,0,0));
#endif


	}

#ifdef VISUAL_DEBUG
	cvDestroyWindow("Input Image");
	cvReleaseImage(&visualizationImage);
#endif

}


void RegionalMorphologyAnalysis::doAll()
{
	if(originalImage == NULL){
		cout << "DoRegionProps: input image is NULL." <<endl;
		exit(1);
	}else{
		if(originalImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			exit(1);
		}
	}

#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

	for(int i = 0; i < internal_blobs.size(); i++){
		Blob *curBlob = internal_blobs[i];
#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(255, 0, 0), CV_RGB(0,0,0));
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif
		printf("Blob #%d - area=%lf perimeter=%lf Eccentricity = %lf ED=%lf ",  i, curBlob->getArea(), curBlob->getPerimeter(), curBlob->getEccentricity(), curBlob->getEquivalentDiameter());
		printf(" MajorAxisLength=%lf MinorAxisLength=%lf ", curBlob->getMajorAxisLength(), curBlob->getMinorAxisLength());
		printf("  Extent = %lf ",  curBlob->getExtent());
		printf(" ConvexArea = %lf Solidity = %lf Deficiency = %lf", curBlob->getConvexArea(), curBlob->getSolidity(), curBlob->getConvexDeficiency());
		printf(" Compactness = %lf FilledArea = %lf Euler# = %d Porosity = %lf", curBlob->getCompacteness(), curBlob->getFilledArea(), curBlob->getEulerNumber(), curBlob->getPorosity());
		printf(" AspectRatio = %lf BendingEnergy = %lf Orientation=%lf ", curBlob->getAspectRatio(), curBlob->getBendingEnery(), curBlob->getOrientation());
		fflush(stdout);
		printf(" MeanPixelIntensity=%lf MedianPixelIntensity=%d MinPixelIntensity=%d MaxPixelIntensity=%d FirstQuartilePixelIntensity=%d ThirdQuartilePixelIntensity=%d ", curBlob->getMeanIntensity(originalImage), curBlob->getMedianIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage), curBlob->getFirstQuartileIntensity(originalImage), curBlob->getThirdQuartileIntensity(originalImage));
		fflush(stdout);
		printf(" MeanGradMagnitude=%lf MedianGradMagnitude=%d MinGradMagnitude=%d MaxGradMagnitude=%d FirstQuartileGradMagnitude=%d ThirdQuartileGradMagnitude=%d ", curBlob->getMeanGradMagnitude(originalImage), curBlob->getMedianGradMagnitude(originalImage), curBlob->getMinGradMagnitude(originalImage), curBlob->getMaxGradMagnitude(originalImage), curBlob->getFirstQuartileGradMagnitude(originalImage), curBlob->getThirdQuartileGradMagnitude(originalImage));
		fflush(stdout);
		printf(" ReflectionSymmetry = %lf ", curBlob->getReflectionSymmetry());
		fflush(stdout);
		printf(" CannyArea = %d", curBlob->getCannyArea(originalImage, 70.0, 90.0));
		fflush(stdout);
		printf(" SobelArea = %d\n", curBlob->getSobelArea(originalImage, 1, 1, 5 ));
		fflush(stdout);

#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(0, 0, 255), CV_RGB(0,0,0));
#endif
//		delete curBlob;
	}

#ifdef VISUAL_DEBUG
	cvDestroyWindow("Input Image");
	cvReleaseImage(&visualizationImage);
#endif

}

void RegionalMorphologyAnalysis::doIntensity(){
	if(originalImage == NULL){
		cout << "DoRegionProps: input image is NULL." <<endl;
		exit(1);
	}else{
		if(originalImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			exit(1);
		}
	}

#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

	for(int i = 0; i < internal_blobs.size(); i++){
		Blob *curBlob = internal_blobs[i];

#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(255, 0, 0), CV_RGB(0,0,0));
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

		printf("Blob #%d - MeanIntensity=%lf MedianIntensity=%d MinIntensity=%d MaxIntensity=%d FirstQuartile=%d ThirdQuartile=%d ", i, curBlob->getMeanIntensity(originalImage), curBlob->getMedianIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage), curBlob->getFirstQuartileIntensity(originalImage), curBlob->getThirdQuartileIntensity(originalImage));
		printf(" MeanGrad=%lf MedianGrad=%d MinGrad=%d MaxGrad=%d FirstQuartile=%d ThirdQuartile=%d ", curBlob->getMeanGradMagnitude(originalImage), curBlob->getMedianGradMagnitude(originalImage), curBlob->getMinGradMagnitude(originalImage), curBlob->getMaxGradMagnitude(originalImage), curBlob->getFirstQuartileGradMagnitude(originalImage), curBlob->getThirdQuartileGradMagnitude(originalImage));
		printf("ReflectionSymmetry = %lf ", curBlob->getReflectionSymmetry());
		printf(" CannyArea = %d", curBlob->getCannyArea(originalImage, 70.0, 90.0));
		printf(" SobelArea = %d\n", curBlob->getSobelArea(originalImage, 1, 1, 5 ));


#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(0, 0, 255), CV_RGB(0,0,0));
#endif


	}

#ifdef VISUAL_DEBUG
	cvDestroyWindow("Input Image");
	cvReleaseImage(&visualizationImage);

#endif

}

