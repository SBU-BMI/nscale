/*
 * RegionalMorphologyAnalysis.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: george
 */

#include "RegionalMorphologyAnalysis.h"

void coocMatrixGPU(char *h_inputImage, int width, int height, unsigned int* coocMatrix, int coocSize,  int angle, int device);

RegionalMorphologyAnalysis::RegionalMorphologyAnalysis(string maskInputFileName, string grayInputFileName)
{
	// read image in mask image that is expected to be binary
	originalImageMask = cvLoadImage( maskInputFileName.c_str(), -1 );
	if(originalImageMask == NULL){
		cout << "Could not load image: "<< maskInputFileName <<endl;
		exit(1);
	}else{
		if(originalImageMask->nChannels != 1){
			cout << "Error: Mask image should have only one channel"<<endl;
			exit(1);
		}
	}

	cvThreshold(originalImageMask, originalImageMask, 1, 1, CV_THRESH_TRUNC);


	originalImage = cvLoadImage( grayInputFileName.c_str(), -1 );
	if(originalImage == NULL){
		cout << "Cound not open input image:"<<grayInputFileName <<endl;
		exit(1);
	}else{
		if(originalImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel"<<endl;
			cvReleaseImage(&originalImage);
			exit(1);
		}
	}
	initializeContours();

	// allocate pointer to the coocurrence matrix for the 4 possible angles
	coocMatrix = (unsigned int **)malloc(sizeof(unsigned int *) * NUM_ANGLES);
	coocMatrixCount = (unsigned int *)malloc(sizeof(unsigned int) * NUM_ANGLES);

	for(int i = 0; i < NUM_ANGLES; i++){
		coocMatrix[i] = NULL;
		coocMatrixCount[i] = 0;
	}
	coocSize = 8;
	intensity_hist = NULL;
	gradient_hist = NULL;
	originalImageGPU = NULL;
	originalImageMaskGPU = NULL;
}

RegionalMorphologyAnalysis::~RegionalMorphologyAnalysis() {

	for(int i = 0; i < internal_blobs.size(); i++){
		delete internal_blobs[i];
	}
	internal_blobs.clear();

	if(originalImage){
		cvReleaseImage(&originalImage);
	}
	if(originalImageMask){
		cvReleaseImage(&originalImageMask);
	}
	if(intensity_hist != NULL){
		free(intensity_hist);
	}
	if(gradient_hist != NULL){
		free(gradient_hist);
	}

	for(int i = 0; i < NUM_ANGLES; i++){
		if(coocMatrix[i] != NULL){
			free(coocMatrix[i]);
		}
	}
	free(coocMatrix);
	free(coocMatrixCount);

	if(originalImageGPU != NULL){
		delete originalImageGPU;
	}
	if(originalImageMaskGPU != NULL){
		delete originalImageMaskGPU;
	}

}


void RegionalMorphologyAnalysis::initializeContours()
{
	// create storage to be used by the findContours
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;

	IplImage *tempMask = cvCreateImage(cvGetSize(originalImage), IPL_DEPTH_8U, 1);
	cvCopy(originalImageMask, tempMask);

	int Nc = cvFindContours(
		tempMask,
		storage,
		&first_contour,
		sizeof(CvContour),
		CV_RETR_TREE,
		CV_CHAIN_APPROX_SIMPLE
		);

	// for all components found in the same first level
	for(CvSeq* c= first_contour; c!= NULL; c=c->h_next){
		// create a blob with the current component and store it in the region
		Blob* curBlob = new Blob(c, cvGetSize(originalImageMask));
		this->internal_blobs.push_back(curBlob);
	}

	cvReleaseImage(&tempMask);
	cvReleaseMemStorage(&storage);
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
		printf("Solidity=%lf Extent=%lf Perimeter=%lf MeanIntensity=%lf MinIntensity=%d MaxIntensity=%d", curBlob->getSolidity(), curBlob->getExtent(), curBlob->getPerimeter(), curBlob->getMeanIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage));
		printf(" ConvexArea = %lf Solidity = %lf Deficiency = %lf", curBlob->getConvexArea(), curBlob->getSolidity(), curBlob->getConvexDeficiency());
		printf(" Compactness = %lf FilledArea = %lf Euler# = %d Porosity = %lf", curBlob->getCompacteness(), curBlob->getFilledArea(), curBlob->getEulerNumber(), curBlob->getPorosity());
		printf(" Orientation=%lf", curBlob->getOrientation());
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

#pragma omp parallel for
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
		printf(" MeanPixelIntensity=%lf MedianPixelIntensity=%d MinPixelIntensity=%d MaxPixelIntensity=%d FirstQuartilePixelIntensity=%d ThirdQuartilePixelIntensity=%d", curBlob->getMeanIntensity(originalImage), curBlob->getMedianIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage), curBlob->getFirstQuartileIntensity(originalImage), curBlob->getThirdQuartileIntensity(originalImage));
		printf(" MeanGradMagnitude=%lf MedianGradMagnitude=%d MinGradMagnitude=%d MaxGradMagnitude=%d FirstQuartileGradMagnitude=%d ThirdQuartileGradMagnitude=%d", curBlob->getMeanGradMagnitude(originalImage), curBlob->getMedianGradMagnitude(originalImage), curBlob->getMinGradMagnitude(originalImage), curBlob->getMaxGradMagnitude(originalImage), curBlob->getFirstQuartileGradMagnitude(originalImage), curBlob->getThirdQuartileGradMagnitude(originalImage));
		printf(" ReflectionSymmetry = %lf ", curBlob->getReflectionSymmetry());
		printf(" CannyArea = %d", curBlob->getCannyArea(originalImage, 70.0, 90.0));
		printf(" SobelArea = %d\n", curBlob->getSobelArea(originalImage, 2, 2, 7 ));


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

void RegionalMorphologyAnalysis::doCoocMatrix(unsigned int angle)
{

	if(coocMatrix[angle] == NULL){
		coocMatrix[angle] = (unsigned int *)calloc(coocSize * coocSize,  sizeof(unsigned int));
		// TODO: check memory allocation return
	}else{
		// It has been calculated before, so clean it up.
		memset(coocMatrix[angle], 0, coocSize * coocSize * sizeof(unsigned int));
		coocMatrixCount[angle] = 0;
	}

	// allocate memory for the normalized image
	float *normImg = (float*)malloc(sizeof(float)*originalImage->height*originalImage->width);
	if(normImg == NULL){
		cout << "ComputeCoocMatrix: Could not allocate temporary normalized image" <<endl;
		exit(1);
	}

	//compute normalized image
	float slope = ((float)coocSize-1.0) / 255.0;
	float intercept = 1.0 ;
	for(int i=0; i<originalImage->height; i++){
		for(int j =0; j < originalImage->width; j++){
			CvScalar elementIJ = cvGet2D(originalImage, i, j);
			normImg[i*originalImage->width + j] = round((slope*(float)elementIJ.val[0] + intercept));
		}
	}

	switch(angle){

		case ANGLE_0:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height; i++){
				int offSet = i*originalImage->width;
				for(int j=0; j<originalImage->width-1; j++){
					if(((normImg[offSet+j])-1) < coocSize && ((normImg[offSet+j+1])-1) < coocSize){
						unsigned int coocAddress = (unsigned int )((normImg[offSet+j])-1) * coocSize;
						coocAddress += normImg[offSet+j+1]-1;
						coocMatrix[angle][coocAddress]++;
					}
				}
			}
			break;

		case ANGLE_45:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height-1; i++){
				int offSetI = i*originalImage->width;
				int offSetI2 = (i+1)*originalImage->width;
				for(int j=0; j<originalImage->width-1; j++){
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j])-1) * coocSize;
					coocAddress += normImg[offSetI +j +1 ] -1;
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;
		case ANGLE_90:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height-1; i++){
				int offSetI = i*originalImage->width;
				int offSetI2 = (i+1)*originalImage->width;
				for(int j=0; j<originalImage->width; j++){
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j])-1) * coocSize;
					coocAddress += normImg[offSetI + j ] -1;
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;

		case ANGLE_135:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height-1; i++){
				int offSetI = i*originalImage->width;
				int offSetI2 = (i+1)*originalImage->width;
				for(int j=0; j<originalImage->width-1; j++){
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j+1])-1) * coocSize;
					coocAddress += normImg[offSetI + j ] -1;
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;
		default:
			cout<< "Unknown angle:"<< angle <<endl;
	}
	free(normImg);

	for(int i = 0; i < coocSize; i++){
		for(int j = 0; j < coocSize; j++){
			coocMatrixCount[angle] += coocMatrix[angle][i*coocSize + j];
		}
	}
}





double RegionalMorphologyAnalysis::inertiaFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double inertia = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	inertia = Operators::inertiaFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return inertia;
}

double RegionalMorphologyAnalysis::energyFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double energy = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	energy = Operators::energyFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return energy;
}

double RegionalMorphologyAnalysis::entropyFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double entropy = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	entropy = Operators::entropyFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return entropy;
}


double RegionalMorphologyAnalysis::homogeneityFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double homogeneity = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	homogeneity = Operators::homogeneityFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return homogeneity;
}

double RegionalMorphologyAnalysis::maximumProbabilityFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double maximumProbability = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	maximumProbability = Operators::maximumProbabilityFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return maximumProbability;
}

double RegionalMorphologyAnalysis::clusterShadeFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double clusterShade = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	clusterShade = Operators::clusterShadeFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return clusterShade;
}

double RegionalMorphologyAnalysis::clusterProminenceFromCoocMatrix(unsigned int angle, unsigned int procType, bool reuseItermediaryResults, unsigned int gpuId)
{
	double clusterProminence = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		if(procType == CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	clusterProminence = Operators::clusterProminenceFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return clusterProminence;
}


void RegionalMorphologyAnalysis::doCoocMatrixGPU(unsigned int angle){
	if(coocMatrix[angle] == NULL){
		coocMatrix[angle] = (unsigned int *)calloc(coocSize * coocSize,  sizeof(unsigned int));
	}else{
		// It has been calculated before, so clean it up.
		memset(coocMatrix[angle], 0, coocSize * coocSize* sizeof(unsigned int));
		coocMatrixCount[angle] = 0;
	}

	unsigned int width = originalImage->width;
	unsigned int height = originalImage->height;
	unsigned int *test;

	coocMatrixGPU(originalImage->imageData , originalImage->width, originalImage->height, coocMatrix[angle],  coocSize, angle,  0 );


	for(int i = 0; i < coocSize; i++){
		for(int j = 0; j < coocSize; j++){
			coocMatrixCount[angle] += coocMatrix[angle][i*coocSize + j];
		}
	}
}



void RegionalMorphologyAnalysis::printCoocMatrix(unsigned int angle)
{
	if(coocMatrix[angle] != NULL){
		const int printWidth = 12;
		for(int i = 0; i < coocSize; i++){
			int offSet = i * coocSize;
			for(int j = 0; j < coocSize; j++){
				cout << setw(printWidth) << coocMatrix[angle][offSet + j]<< " ";
			}
			cout <<endl;
		}
		cout <<endl;
	}else{
		cout << "Could not print coocMatrix. It has not been calculated."<<endl;
	}
}


double RegionalMorphologyAnalysis::calcMeanIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	double meanIntensity = 0.0;

	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}

				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}
		}
	}
	meanIntensity = Operators::calcMeanFromHistogram(intensity_hist, 256);

	return meanIntensity;
}



double RegionalMorphologyAnalysis::calcStdIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	double stdIntensity = 0.0;


	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	stdIntensity = Operators::calcStdFromHistogram(intensity_hist, 256);

	return stdIntensity;
}



int RegionalMorphologyAnalysis::calcMedianIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int medianIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	medianIntensity = Operators::calcMedianFromHistogram(intensity_hist, 256);

	return medianIntensity;
}



int RegionalMorphologyAnalysis::calcMinIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int minIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	minIntensity = Operators::calcMinFromHistogram(intensity_hist, 256);

	return minIntensity;
}



int RegionalMorphologyAnalysis::calcMaxIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int maxIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	maxIntensity = Operators::calcMaxFromHistogram(intensity_hist, 256);

	return maxIntensity;
}



int RegionalMorphologyAnalysis::calcFirstQuartileIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int firstQuartileIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	firstQuartileIntensity = Operators::calcFirstQuartileFromHistogram(intensity_hist, 256);

	return firstQuartileIntensity;
}



int RegionalMorphologyAnalysis::calcSecondQuartileIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int secondQuartileIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	secondQuartileIntensity = Operators::calcSecondQuartileFromHistogram(intensity_hist, 256);

	return secondQuartileIntensity;
}



int RegionalMorphologyAnalysis::calcThirdQuartileIntensity(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int thirdQuartileIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		if(procType == CPU){
			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}
			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU, originalImageMaskGPU);
			}else{
				intensity_hist = Operators::buildHistogram256GPU(originalImageGPU);
			}

		}
	}
	thirdQuartileIntensity = Operators::calcThirdQuartileFromHistogram(intensity_hist, 256);

	return thirdQuartileIntensity;
}



bool RegionalMorphologyAnalysis::uploadImageToGPU()
{
	originalImageGPU = new cv::gpu::GpuMat(originalImage);
}


void RegionalMorphologyAnalysis::releaseGPUImage()
{
	if(originalImageGPU != NULL){
		delete originalImageGPU;
		originalImageGPU = NULL;
	}
}

int RegionalMorphologyAnalysis::calcCannyArea(int procType, double lowThresh, double highThresh, int apertureSize, int gpuId)
{
	int cannyPixels = 0;

	if(procType == CPU){
		IplImage *cannyRes = cvCreateImage( cvSize(originalImage->width, originalImage->height), IPL_DEPTH_8U, 1);

		cvCopy(originalImage, cannyRes, NULL);

		cvAnd(cannyRes, this->getMask(), cannyRes);

		cvCanny(cannyRes, cannyRes, lowThresh, highThresh, apertureSize);

		// Calculate the #white pixels and divide by blob area
		cannyPixels = cvCountNonZero(cannyRes);

		cvReleaseImage(&cannyRes);

	}else{
		if(originalImageGPU == NULL){
			this->uploadImageMaskToGPU();
		}
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}

	}

	return cannyPixels;
}



IplImage *RegionalMorphologyAnalysis::getMask()
{
	return this->originalImageMask;
}

bool RegionalMorphologyAnalysis::uploadImageMaskToGPU()
{
	originalImageMaskGPU = new cv::gpu::GpuMat(originalImageMask);
}

void RegionalMorphologyAnalysis::releaseGPUMask()
{
	if(originalImageMaskGPU != NULL){
		delete originalImageMaskGPU;
		originalImageMaskGPU = NULL;
	}

}

unsigned int *RegionalMorphologyAnalysis::calcGradientHistogram(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	unsigned int *grad_mag_hist_ret = NULL;
	if(procType == CPU){

		// Create a Mat header pointing to the C image loaded
		Mat originalImgHeader(originalImage);
		Mat dest;

		Mat element;

		morphologyEx(originalImgHeader, dest, MORPH_GRADIENT, element,  Point(-1,-1), 1);

		IplImage magImg = dest;
		// This is the data used to store the gradient results
		//IplImage* magImg = cvCreateImage( cvGetSize(originalImage), IPL_DEPTH_8U, 1);

		// This is a temporary structure required by the MorphologyEx operation we'll perform
		//IplImage* tempImg = cvCreateImage( cvGetSize(originalImage), IPL_DEPTH_8U, 1);

	//	cvMorphologyEx(originalImage, magImg, tempImg, NULL, CV_MOP_GRADIENT, 10);

		if(useMask){
			grad_mag_hist_ret = Operators::buildHistogram256CPU(&magImg, originalImageMask);
		}else{
			grad_mag_hist_ret = Operators::buildHistogram256CPU(&magImg);
		}
//		cvReleaseImage(&magImg);
	//	cvReleaseImage(&tempImg);

	}else{
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}
		// This is the data used to store the gradient results
		cv::gpu::GpuMat *magImg = new cv::gpu::GpuMat(originalImageGPU->size(), CV_8UC1);

		// This is a temporary structure required by the MorphologyEx operation we'll perform
		cv::gpu::GpuMat *tempImg = new cv::gpu::GpuMat(originalImageGPU->size(), CV_8UC1);

		cv::gpu::GpuMat kernel;
		cv::gpu::morphologyEx(*originalImageGPU, *magImg, MORPH_GRADIENT, kernel, Point(-1,-1), 1);
		delete tempImg;

		if(useMask){
			if(originalImageMaskGPU == NULL){
				this->uploadImageMaskToGPU();
			}
			grad_mag_hist_ret = Operators::buildHistogram256GPU(magImg, originalImageMaskGPU);


		}else{
			grad_mag_hist_ret = Operators::buildHistogram256GPU(magImg);
		}
		delete magImg;
	}
	return grad_mag_hist_ret;
}

double RegionalMorphologyAnalysis::calcMeanGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	double meanGradientMagnitude = 0.0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	meanGradientMagnitude = Operators::calcMeanFromHistogram(gradient_hist, 256);
	return meanGradientMagnitude;
}

double RegionalMorphologyAnalysis::calcStdGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	double stdGradientMagnitude = 0.0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	stdGradientMagnitude = Operators::calcStdFromHistogram(gradient_hist, 256);
	return stdGradientMagnitude;
}


int RegionalMorphologyAnalysis::calcMedianGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int medianGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	medianGradientMagnitude = Operators::calcMedianFromHistogram(gradient_hist, 256);
	return medianGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcMinGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int minGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	minGradientMagnitude = Operators::calcMinFromHistogram(gradient_hist, 256);
	return minGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcMaxGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int maxGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	maxGradientMagnitude = Operators::calcMaxFromHistogram(gradient_hist, 256);
	return maxGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcFirstQuartileGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int firstQuartileGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	firstQuartileGradientMagnitude = Operators::calcFirstQuartileFromHistogram(gradient_hist, 256);
	return firstQuartileGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcSecondQuartileGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int secondQuartileGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	secondQuartileGradientMagnitude = Operators::calcSecondQuartileFromHistogram(gradient_hist, 256);
	return secondQuartileGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcThirdQuartileGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, int gpuId)
{
	int thirdQuartileGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, gpuId);
	}
	thirdQuartileGradientMagnitude = Operators::calcThirdQuartileFromHistogram(gradient_hist, 256);
	return thirdQuartileGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcSobelArea(int procType, int xorder, int yorder, int apertureSize, bool useMask, int gpuId)
{
	int sobelPixels = 0;

	if(procType == CPU){
		// Create a Mat header pointing to the C image loaded
		Mat originalImgHeader(originalImage);
		Mat originalImgMaskHeader(originalImageMask);

		// Allocate space to store results
		Mat destTransf;

		if(useMask){
			destTransf = originalImgHeader.mul(originalImgMaskHeader);
			Sobel(destTransf, destTransf, CV_8U, 1, 1, 7);
		}else{
			destTransf.create(originalImgHeader.size(), originalImgHeader.type());
			Sobel(originalImgHeader, destTransf, CV_8U, 1, 1, 7);

		}

		// Calculate the #white pixels and divide by blob area
		sobelPixels = countNonZero(destTransf);

		// Make sure that data is released
		destTransf.release();

	}else{
		if(originalImageMaskGPU == NULL && useMask){
			this->uploadImageMaskToGPU();
		}
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}

		cv::gpu::GpuMat *sobelResGPU = new cv::gpu::GpuMat(cvGetSize(originalImage), CV_8U);
		cv::gpu::GpuMat *sobelResGPU2;

		if(useMask){
			sobelResGPU2 = new cv::gpu::GpuMat(cvGetSize(originalImage), CV_8U);

			cv::gpu::multiply(*originalImageGPU, *originalImageMaskGPU, *sobelResGPU);
			cv::gpu::Sobel(*sobelResGPU, *sobelResGPU2, CV_8U, xorder, yorder, apertureSize);

			sobelPixels = gpu::countNonZero(*sobelResGPU2);
			delete sobelResGPU2;

		}else{

			cv::gpu::Sobel(*originalImageGPU, *sobelResGPU, CV_8U, xorder, yorder, apertureSize);
			sobelPixels = gpu::countNonZero(*sobelResGPU);
		}

		delete sobelResGPU;

	}

	return sobelPixels;
}
