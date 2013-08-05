/*
 * RegionalMorphologyAnalysis.cpp
 *
 *  Created on: Jun 22, 2011
 *      Author: george
 */

#include "RegionalMorphologyAnalysis.h"

void coocMatrixGPU(char *h_inputImage, int width, int height, unsigned int* coocMatrix, int coocSize,  int angle, bool copyData, int device){};
int *coocMatrixBlobGPU(char *h_inputImage, int width, int height, int nBlobs, char *maksData, int maskSize, int coocSize, int angle, bool copyData, bool downloadRes, bool mallocTempData, char* tempData, int device){};
float *calcHaralickGPUBlob(int *d_coocMatrix, int coocSize, int nBlobs, int device, float *tempData){};
float *intensityFeaturesBlobGPU(char *h_inputImage, int width, int height, int nBlobs, char *maskData, int maskSize, bool copyData, int device){};
void *cudaMallocWrapper(int size){};
void cudaFreeWrapper(void *data_ptr){};

RegionalMorphologyAnalysis::RegionalMorphologyAnalysis(string maskInputFileName, string grayInputFileName, bool initCytoplasm)
{
	struct timeval startTime;
	struct timeval endTime;
	isImage = true;


//	gettimeofday(&startTime, NULL);


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

	// read actual image: forcing to be grayscale
	originalImage = cvLoadImage( grayInputFileName.c_str(), 0 );

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

	// allocate pointer to the coocurrence matrix for the 4 possible angles
	coocMatrix = (unsigned int **)malloc(sizeof(unsigned int *) * Constant::NUM_ANGLES);
	coocMatrixCount = (unsigned int *)malloc(sizeof(unsigned int) * Constant::NUM_ANGLES);

	for(int i = 0; i < Constant::NUM_ANGLES; i++){
		coocMatrix[i] = NULL;
		coocMatrixCount[i] = 0;
	}
	coocSize = 8;
	intensity_hist = NULL;
	gradient_hist = NULL;

#ifdef USE_GPU
	originalImageGPU = NULL;
	originalImageMaskGPU = NULL;
	originalImageMaskNucleusBoxesGPU = NULL;
#endif
	// Warning. Do not move this function call before the initialization of the
	// variable blobMaskAllocatedMemory. It will modify its content.
	// ProcessTime example

	initializeContours(initCytoplasm);

}


RegionalMorphologyAnalysis::RegionalMorphologyAnalysis(IplImage *originalImageMaskParam, IplImage *originalImageParam, bool initCytoplasm)
{
	struct timeval startTime;
	struct timeval endTime;
	isImage = false;

	originalImageMask = cvCreateImageHeader( cvGetSize(originalImageMaskParam), originalImageMaskParam->depth, originalImageMaskParam->nChannels );
	originalImageMask->origin = originalImageMaskParam->origin;
	originalImageMask->widthStep = originalImageMaskParam->widthStep;
	originalImageMask->imageData = originalImageMaskParam->imageData;

	originalImage = cvCreateImageHeader( cvGetSize(originalImageParam), originalImageParam->depth, originalImageParam->nChannels );
	originalImage->origin = originalImageParam->origin;
	originalImage->widthStep = originalImageParam->widthStep;
	originalImage->imageData = originalImageParam->imageData;


	// allocate pointer to the coocurrence matrix for the 4 possible angles
	coocMatrix = (unsigned int **)malloc(sizeof(unsigned int *) * Constant::NUM_ANGLES);
	coocMatrixCount = (unsigned int *)malloc(sizeof(unsigned int) * Constant::NUM_ANGLES);

	for(int i = 0; i < Constant::NUM_ANGLES; i++){
		coocMatrix[i] = NULL;
		coocMatrixCount[i] = 0;
	}
	coocSize = 8;
	intensity_hist = NULL;
	gradient_hist = NULL;

#ifdef	USE_GPU
	originalImageGPU = NULL;
	originalImageMaskGPU = NULL;
	originalImageMaskNucleusBoxesGPU = NULL;
#endif

	// Warning. Do not move this function call before the initialization of the
	// variable blobMaskAllocatedMemory. It will modify its content.
	// ProcessTime example
	initializeContours(initCytoplasm);

}







RegionalMorphologyAnalysis::~RegionalMorphologyAnalysis() {

	// Cleanning up blobs structures
	for(int i = 0; i < internal_blobs.size(); i++){
		delete internal_blobs[i];
	}
	internal_blobs.clear();

	// Cleanning up blobs structures
	for(int i = 0; i < cytoplasm_blobs.size(); i++){
		delete cytoplasm_blobs[i];
	}
	cytoplasm_blobs.clear();

	if(isImage){
		if(originalImage){
			cvReleaseImage(&originalImage);
		}
		if(originalImageMask){
			cvReleaseImage(&originalImageMask);
		}
	}else{
	// Ops. This object contains pointers to image headers
		if(originalImage){
			cvReleaseImageHeader(&originalImage);
		}
		if(originalImageMask){
			cvReleaseImageHeader(&originalImageMask);
		}

	}
	if(originalImageMaskNucleusBoxes){
		delete originalImageMaskNucleusBoxes;
	}
	
	if(intensity_hist != NULL){
		free(intensity_hist);
	}
	if(gradient_hist != NULL){
		free(gradient_hist);
	}

	for(int i = 0; i < Constant::NUM_ANGLES; i++){
		if(coocMatrix[i] != NULL){
			free(coocMatrix[i]);
		}
	}
	free(coocMatrix);
	free(coocMatrixCount);

#ifdef	USE_GPU
	if(originalImageGPU != NULL){
		delete originalImageGPU;
	}
	if(originalImageMaskGPU != NULL){
		delete originalImageMaskGPU;
	}

	if(originalImageMaskNucleusBoxesGPU != NULL){
		delete originalImageMaskNucleusBoxesGPU;
	}
#endif


}


void RegionalMorphologyAnalysis::initializeContours(bool initCytoplasm)
{
	// create storage to be used by the findContours
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* first_contour = NULL;

//	IplImage *tempMask = cvCreateImage(cvGetSize(originalImage), IPL_DEPTH_8U, 1);
//	cvCopy(originalImageMask, tempMask);

	IplImage *tempMask = cvCreateImage(cvSize(originalImage->width+2, originalImage->height+2), IPL_DEPTH_8U, 1);
	cvCopyMakeBorder(originalImageMask, tempMask, cvPoint(1,1), IPL_BORDER_CONSTANT, cvScalarAll(0) );
//	cvSaveImage("mask-border.tif", tempMask);

	int Nc = cvFindContours(
		tempMask,
		storage,
		&first_contour,
		sizeof(CvContour),
		CV_RETR_TREE,
//		CV_CHAIN_APPROX_SIMPLE
		CV_CHAIN_APPROX_NONE
		);
	cvReleaseImage(&tempMask);

	int maskSizeElements = 0;
	// for all components found in the same first level
	for(CvSeq* c= first_contour; c!= NULL; c=c->h_next){

		CvPoint offsetInImage;
		offsetInImage.x = -1;
		offsetInImage.y = -1;


		// create a blob with the current component and store it in the region
		Blob* curBlob = new Blob(c, cvGetSize(originalImageMask), offsetInImage);
		CvRect bounding_box = curBlob->getNonInclinedBoundingBox();
		this->internal_blobs.push_back(curBlob);

		maskSizeElements += bounding_box.height * bounding_box.width;
	}


	blobsMaskAllocatedMemorySize = sizeof(char) * maskSizeElements + 4*sizeof(int) *internal_blobs.size() + sizeof(int) * internal_blobs.size();


	originalImageMaskNucleusBoxes = new Mat(1, blobsMaskAllocatedMemorySize, CV_8UC1);

	char *blobsMaskAllocatedMemory = (char*)originalImageMaskNucleusBoxes->data;

//	blobsMaskAllocatedMemory = (char *) malloc( blobsMaskAllocatedMemorySize );
	int offset = internal_blobs.size() * sizeof(int) * 5;
	int *offset_ptr = (int *) blobsMaskAllocatedMemory;


	for(int i = 0; i < internal_blobs.size(); i++){
		CvRect bounding_box = internal_blobs[i]->getNonInclinedBoundingBox();
		offset_ptr[(i*5)] = offset;
		offset_ptr[(i*5) + 1] = bounding_box.x;
		offset_ptr[(i*5) + 2] = bounding_box.y;
		offset_ptr[(i*5) + 3] = bounding_box.width;
		offset_ptr[(i*5) + 4] = bounding_box.height;

		internal_blobs[i]->setMaskInUserDataRegion(blobsMaskAllocatedMemory +offset);
		offset += bounding_box.width * bounding_box.height;

	}

	// Calculate cytoplasm mask, and create a blob describing each cytoplasm
	if(initCytoplasm){
		for(int i = 0; i < internal_blobs.size(); i++){
//			if(i == 0){
//				IplImage *blobMask = internal_blobs[i]->getMask();
//				cvSaveImage("blob1.tif", blobMask);
//			}
			CvPoint offsetCyto;
			IplImage *cytoplasmMask = internal_blobs[i]->getCytoplasmMask(cvGetSize(originalImageMask), 8, offsetCyto);  // a new image is created,  but blob has a refernece and will delete it.
			// Should be fine til here
//			if(i == 0){
//				cvSaveImage("cyto1-0.tif", cytoplasmMask);
//			}

			// Create padding around cytoplasm mask to avoid find contours to be truncated by the image bounds
			IplImage *tempMaskCytoplasm = cvCreateImage(cvSize(cytoplasmMask->width+2, cytoplasmMask->height+2), IPL_DEPTH_8U, 1);

			// set the ROI of the temp Mask to be of the same size as the cytoplasm, but internally 
			// to the image with 1 pixel padding in each direction. This craziness is need to avoid 
			// find contours from truncating the contours for pixels touching the image border
			cvSetImageROI(tempMaskCytoplasm, cvRect(1, 1, cytoplasmMask->width, cytoplasmMask->height));

			// copy mask to image with padding
			cvCopy(cytoplasmMask, tempMaskCytoplasm);

			// reset ROI of the padded image
			cvResetImageROI(tempMaskCytoplasm);
			
			// change offset of the contours to be found in tempMaskCytoplams, because the mask is one pixel shift in X and Y
			// in this image, when compare to its original mask (cytoplasmMask). Thus, any contours found will also have X,Y values
			// increased by 1,1 when compared to the input image.
//			offsetCyto.x-=1;
//			offsetCyto.y-=1;

			// Run find contours in the temp padded image.
			int Nc = cvFindContours(
					tempMaskCytoplasm,
					storage,
					&first_contour,
					sizeof(CvContour),
					CV_RETR_TREE,
					CV_CHAIN_APPROX_NONE
					);
//
//			if(first_contour->h_next != NULL){
//				cout << "Warnning: Cytoplasm is being defined through more than one blob!" <<endl;
//				cvSaveImage("cyto-warnning.tif", cytoplasmMask);
//				IplImage *blobMask = internal_blobs[i]->getMask();
//				cvSaveImage("blob-warnning.tif", blobMask);
//
//			}

			cvReleaseImage(&tempMaskCytoplasm);

			CvRect *bb = new CvRect();
			bb->x = 0;
			bb->y = 0;
			bb->width = cytoplasmMask->width;
			bb->height = cytoplasmMask->height;
			
			// create blob representing the cytoplams we just calculated
			Blob *cytoBlob = new Blob(first_contour, cvGetSize(originalImageMask), offsetCyto, cytoplasmMask, bb);
			this->cytoplasm_blobs.push_back(cytoBlob);

//			if(first_contour->h_next != NULL){
//			//	cout << "Warnning: Cytoplasm is being defined through more than one blob!" <<endl;
//			//	cvSaveImage("cyto-warnning.tif", cytoplasmMask);
//			//	IplImage *blobMask = internal_blobs[i]->getMask();
//			//	cvSaveImage("blob-warnning.tif", blobMask);
//				cout << "bb->x="<< bb->x << " bb->y="<<bb->y<< " bb->width="<<bb->width<<" bb->height="<<bb->height<<endl;
//				IplImage* cytoMask = cytoBlob->getMask();
//
//				cvSaveImage("cyto-II-warnning.tif", cytoMask);
//
//			}

			delete bb;
			// Release images used to calc. cytoplasm
	//		cvReleaseImage(&cytoplasmMask);
//			if(i == 0){
//				IplImage* cytoMask = cytoBlob->getMask();
//
//				cvSaveImage("cyto1.tif", cytoMask);
//			}

			//		IplImage* cytoMask = cytoBlob->getMask();
			//		cvSaveImage("cytoplasmMask.tif", cytoplasmMask);
			//		cvSaveImage("cytoplasmMaskBlobComp.tif", cytoMask);
			//
			//		cvNamedWindow("CytoMask - Press any key to continue!");
			//		cvShowImage("CytoMask - Press any key to continue!", cytoMask);
			//		cvNamedWindow("CytoMaskComp - Press any key to continue!");
			//		cvShowImage("CytoMaskComp - Press any key to continue!", cytoplasmMask);
			//
			//		cvWaitKey(0);
			//		cvDestroyWindow("CytoMask - Press any key to continue!");
			//		cvDestroyWindow("CytoMaskComp - Press any key to continue!");
		}	
	}
	cvReleaseMemStorage(&storage);
}

// Assuming n > 0
int rndint(float n)//round float to the nearest integer
{	
	int ret = (int)floor(n);
	float t;
	t=n-floor(n);
	if (t>=0.5)    
	{
		ret = (int)floor(n) + 1;
	}
	return ret;
}


IplImage* RegionalMorphologyAnalysis::gradient(IplImage *inputImage){

	IplImage* drv32f = cvCreateImage(cvGetSize(inputImage), IPL_DEPTH_32F, 1);
	IplImage* magDrv = cvCreateImage(cvGetSize(inputImage), IPL_DEPTH_32F, 1);
	IplImage* magDrvUint = cvCreateImage(cvGetSize(inputImage), IPL_DEPTH_8U, 1);
	cvSetZero( magDrv);

	// Create convolution kernel for x
	float kernel_array_x[] = {-0.5, 0, 0.5 }; 
	CvMat kernel_x;
	cvInitMatHeader(&kernel_x, 1 ,3, CV_32F, kernel_array_x);

	// Calculate derivative in x
	cvFilter2D(inputImage, drv32f, &kernel_x);
	
	// fix values in the first and last colums
	for(int y = 0; y<drv32f->height; y++){
		float *derive_ptr = (float*)(drv32f->imageData + y * drv32f->widthStep);
		derive_ptr[0] *= 2;
		derive_ptr[drv32f->width-1] *= 2;
	}

	// Accumulate square of dx
	cvSquareAcc( drv32f, magDrv);

	// Create convolution kernel for y
	CvMat kernel_y;
	cvInitMatHeader(&kernel_y, 3 ,1, CV_32F, kernel_array_x);

	// Calculate derivative in x
	cvFilter2D(inputImage, drv32f, &kernel_y);


	// fix values in the first and last lines
	float *derive_ptr_first = (float*)(drv32f->imageData);
	float *derive_ptr_last = (float*)(drv32f->imageData + (drv32f->height-1) * drv32f->widthStep);
	for(int y = 0; y < drv32f->width; y++){
		derive_ptr_first[y] *= 2;	
		derive_ptr_last[y] *= 2;	
	}
	
	// Accumulate square of dx
	cvSquareAcc( drv32f, magDrv);

	Mat magTemp(magDrv);
	cv::sqrt( magTemp, magTemp );

	// This is uint8 from MATLAB
	for(int y = 0; y<drv32f->height; y++){
		float *mag_dvr_ptr = (float*)(magDrv->imageData + y * magDrv->widthStep);
		unsigned char *magDrvUint_ptr = (unsigned char*)(magDrvUint->imageData + y * magDrvUint->widthStep);
		for(int x=0; x < drv32f->width; x++){
			if(mag_dvr_ptr[x] < 0.0){
				magDrvUint_ptr[x] = 0;
			}else{
				if(mag_dvr_ptr[x] > 255.0){
					magDrvUint_ptr[x] = 255;
				}else{
					magDrvUint_ptr[x] = (unsigned char)rndint((mag_dvr_ptr[x]));
				}
			}
		}
	}
	cvReleaseImage(&drv32f);
	cvReleaseImage(&magDrv);
	return magDrvUint;
}

void RegionalMorphologyAnalysis::doNucleiPipelineFeatures(vector<vector<float> > &nucleiFeatures, IplImage *inputImage)
{
	vector<Blob *>::iterator blobsIt = internal_blobs.begin();
	vector<Blob *>::iterator blobsEndIt = internal_blobs.end();

	// this is calculation per cytoplasm

	if(originalImage == NULL){
		cout << "doNucleiPipelineFeatures: input image is NULL." <<endl;
		exit(1);
	}
	inputImage = originalImage;

	if(inputImage->nChannels != 1){
		cout << "Error: input image should be grayscale with one channel only"<<endl;
		exit(1);
	}

	IplImage *magDrvUint = gradient(inputImage);

	for(int i = 0; blobsIt < blobsEndIt; blobsIt++, i++){
		Blob *curBlob = *blobsIt;

		vector<float> blobFeatures;
		// Calculate bounding box information
		CvRect bb = curBlob->getNonInclinedBoundingBox();
		blobFeatures.push_back(bb.x);
		blobFeatures.push_back(bb.y);
		blobFeatures.push_back(bb.width);
		blobFeatures.push_back(bb.height);
		blobFeatures.push_back((float)bb.x+((float)(bb.width-1)/2.0) );
		blobFeatures.push_back((float)bb.y+((float)(bb.height-1)/2.0));

		//0)
		blobFeatures.push_back(curBlob->getArea());
		//1)
		blobFeatures.push_back(curBlob->getPerimeter());
		//2)
		blobFeatures.push_back(curBlob->getEccentricity());
		// Compactness -> Circularity
		//3)
		blobFeatures.push_back(curBlob->getCompacteness());
		//4)
		blobFeatures.push_back(curBlob->getMajorAxisLength());
		//5)
		blobFeatures.push_back(curBlob->getMinorAxisLength());
		//6)
		blobFeatures.push_back(curBlob->getExtent());
		//7)
		blobFeatures.push_back(curBlob->getMeanIntensity(inputImage));
		//8)
		blobFeatures.push_back(curBlob->getMaxIntensity(inputImage));
		//9)
		blobFeatures.push_back(curBlob->getMinIntensity(inputImage));
		//10)
		blobFeatures.push_back(curBlob->getStdIntensity(inputImage));
		//11)
		blobFeatures.push_back(curBlob->getEntropyIntensity(inputImage));
		//12)
		blobFeatures.push_back(curBlob->getEnergyIntensity(inputImage));
		//13)
		blobFeatures.push_back(curBlob->getSkewnessIntensity(inputImage));
		//14)
		blobFeatures.push_back(curBlob->getKurtosisIntensity(inputImage));
		// Comment. The Mean and Std grad are calculated on top of the uint8 version of 
		// the gradient resulting image, instead of the float image. Jun aggreed it should
		// not make any difference
		//15)
		blobFeatures.push_back(curBlob->getMeanGradMagnitude(magDrvUint));
		//16)
		blobFeatures.push_back(curBlob->getStdGradMagnitude(magDrvUint));
		//17)
		blobFeatures.push_back(curBlob->getEntropyGradMagnitude(magDrvUint));
		//18)
		blobFeatures.push_back(curBlob->getEnergyGradMagnitude(magDrvUint));
		//19)
		blobFeatures.push_back(curBlob->getSkewnessGradMagnitude(magDrvUint));
		//20)
		blobFeatures.push_back(curBlob->getKurtosisGradMagnitude(magDrvUint));
		//21)
		blobFeatures.push_back(curBlob->getCannyArea(inputImage, 75.0, 100.0));
		//22)
		blobFeatures.push_back(curBlob->getMeanCanny(inputImage, 75.0, 100.0));
		
		nucleiFeatures.push_back(blobFeatures);
	}
	cvReleaseImage(&magDrvUint);

}

void RegionalMorphologyAnalysis::doCytoplasmPipelineFeatures(vector<vector<float> > &cytoplamsFeatures, IplImage *inputImage)
{
	assert(this->internal_blobs.size() == this->cytoplasm_blobs.size());
	assert(inputImage != NULL);
	vector<Blob *>::iterator blobsIt = cytoplasm_blobs.begin();
	vector<Blob *>::iterator blobsEndIt = cytoplasm_blobs.end();
	
	if(inputImage == NULL){
		cout << "doCytoplasmPipelineFeatures: input image is NULL." <<endl;
		exit(1);
	}

	if(inputImage->nChannels != 1){
		cout << "Error: input image should be grayscale with one channel only"<<endl;
		exit(1);
	}

	IplImage *magImg = gradient(inputImage);


	for(int i = 0; blobsIt < blobsEndIt; blobsIt++, i++){
		Blob *curBlob = *blobsIt;
		vector<float> blobFeatures;
		// Clear blobs internal intermediary data, what will force 
		// it to recompute intermediary results for the new image.
		curBlob->resetInternalData();
			
		blobFeatures.push_back(curBlob->getMeanIntensity(inputImage));
//		cout << "INTENSITY histogram"<<endl;
//		curBlob->printIntensityHistogram(inputImage);

		blobFeatures.push_back(curBlob->getMeanIntensity(inputImage) - curBlob->getMedianIntensity(inputImage) );
//		cout << "MEDIAN intensity = "<< curBlob->getMedianIntensity(inputImage)<<endl;
		blobFeatures.push_back(curBlob->getMaxIntensity(inputImage));
		blobFeatures.push_back(curBlob->getMinIntensity(inputImage));
		blobFeatures.push_back(curBlob->getStdIntensity(inputImage));
		blobFeatures.push_back(curBlob->getEntropyIntensity(inputImage));
		blobFeatures.push_back(curBlob->getEnergyIntensity(inputImage));
		blobFeatures.push_back(curBlob->getSkewnessIntensity(inputImage));
		blobFeatures.push_back(curBlob->getKurtosisIntensity(inputImage));
		blobFeatures.push_back(curBlob->getMeanGradMagnitude(magImg));
		blobFeatures.push_back(curBlob->getStdGradMagnitude(magImg));
		blobFeatures.push_back(curBlob->getEntropyGradMagnitude(magImg));
		blobFeatures.push_back(curBlob->getEnergyGradMagnitude(magImg));
		blobFeatures.push_back(curBlob->getSkewnessGradMagnitude(magImg));
		blobFeatures.push_back(curBlob->getKurtosisGradMagnitude(magImg));
		blobFeatures.push_back(curBlob->getCannyArea(inputImage, 75.0, 100.0));
		blobFeatures.push_back(curBlob->getMeanCanny(inputImage, 75.0, 100.0));
		
		cytoplamsFeatures.push_back(blobFeatures);
	}
	cvReleaseImage(&magImg);
}

// Computes morphometry features for each nuclei in the input image. And returns a array of array of features.
// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
// 	Area; MajorAxisLength;MinorAxisLength; Eccentricity; Orientation; ConvexArea; FilledArea; EulerNumber; 
// 	EquivalentDiameter; Solidity; Extent; Perimeter; ConvexDeficiency; Compacteness; Porosity; AspectRation; BendingEnergy; ReflectionSymmetry; CannyArea; MeanCanny, SobelArea;
void RegionalMorphologyAnalysis::doMorphometryFeatures(vector<vector<float> > &morphoFeatures, bool nuclei, IplImage *inputImage)
{
	vector<Blob *>::iterator blobsIt = internal_blobs.begin();
	vector<Blob *>::iterator blobsEndIt = internal_blobs.end();

#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

	// this is calculation per cytoplasm
	if(nuclei == false){
		assert(this->internal_blobs.size() == this->cytoplasm_blobs.size());
		assert(inputImage != NULL);
		blobsIt = cytoplasm_blobs.begin();
		blobsEndIt = cytoplasm_blobs.end();
		cout << "computing features per cytoplasm"<<endl;
	}else{

		if(originalImage == NULL){
			cout << "doMorphometryFeatures: input image is NULL." <<endl;
			exit(1);
		}
		inputImage = originalImage;
	}

	if(inputImage->nChannels != 1){
		cout << "Error: input image should be grayscale with one channel only"<<endl;
		exit(1);
	}


	for(; blobsIt < blobsEndIt; blobsIt++){
		Blob *curBlob = *blobsIt;//internal_blobs[i];
#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(255, 0, 0), CV_RGB(0,0,0));
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif
		vector<float> blobFeatures;
		blobFeatures.push_back(curBlob->getArea());
		blobFeatures.push_back(curBlob->getMajorAxisLength());
		blobFeatures.push_back(curBlob->getMinorAxisLength());
		blobFeatures.push_back(curBlob->getEccentricity());
		blobFeatures.push_back(curBlob->getOrientation());
		blobFeatures.push_back(curBlob->getConvexArea());
		blobFeatures.push_back(curBlob->getFilledArea());
		blobFeatures.push_back(curBlob->getEulerNumber());
		blobFeatures.push_back(curBlob->getEquivalentDiameter());
		blobFeatures.push_back(curBlob->getSolidity());
		blobFeatures.push_back(curBlob->getExtent());
		blobFeatures.push_back(curBlob->getPerimeter());
		blobFeatures.push_back(curBlob->getConvexDeficiency());
		blobFeatures.push_back(curBlob->getCompacteness());
		blobFeatures.push_back(curBlob->getPorosity());
		blobFeatures.push_back(curBlob->getAspectRatio());
		blobFeatures.push_back(curBlob->getBendingEnery());
		blobFeatures.push_back(curBlob->getReflectionSymmetry());
		blobFeatures.push_back(curBlob->getCannyArea(inputImage, 75, 100, 3));
		blobFeatures.push_back(curBlob->getMeanCanny(inputImage, 75, 100, 3));
		blobFeatures.push_back(curBlob->getSobelArea(inputImage, 5, 5, 7));

		
		morphoFeatures.push_back(blobFeatures);

#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(0, 0, 255), CV_RGB(0,0,0));
#endif


	}

#ifdef VISUAL_DEBUG
	cvDestroyWindow("Input Image");
	cvReleaseImage(&visualizationImage);
#endif
}


// Computes Gradient features for each nuclei in the input image. And returns a array of array of features.
// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
// 	0)Mean;  1)Std; 2)Energy; 3)Entropy; 4)Kurtosis; 5)Skewness;
//	6)Median; 7)Min; 8)Max; 9)FirstQuartile; 10)ThirdQuartile;
void RegionalMorphologyAnalysis::doGradientBlob(vector<vector<float> > &gradientFeatures, bool nuclei, IplImage *inputImage, unsigned int procType){
	if(procType == Constant::CPU){
		vector<Blob *>::iterator blobsIt = internal_blobs.begin();
		vector<Blob *>::iterator blobsEndIt = internal_blobs.end();

#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

		// this is calculation per cytoplasm
		if(nuclei == false){
			assert(this->internal_blobs.size() == this->cytoplasm_blobs.size());
			assert(inputImage != NULL);
			blobsIt = cytoplasm_blobs.begin();
			blobsEndIt = cytoplasm_blobs.end();
			cout << "computing features per cytoplasm"<<endl;
		}else{

			if(originalImage == NULL){
				cout << "doGradientBlob: input image is NULL." <<endl;
				exit(1);
			}
			inputImage = originalImage;
		}

		if(inputImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			exit(1);
		}

		Mat imgMat(inputImage);

		// This is a temporary structure required by the MorphologyEx operation we'll perform
		IplImage* magImg = cvCreateImage( cvSize(inputImage->width, inputImage->height), IPL_DEPTH_8U, 1);
		Mat dest(magImg);

		Mat kernelCPU;
		morphologyEx(imgMat, dest, MORPH_GRADIENT, kernelCPU, Point(-1,-1), 1);

		for(; blobsIt < blobsEndIt; blobsIt++){
			Blob *curBlob = *blobsIt;
			// Cleare intermediary data. Thus, gradient is recalculated for given input
			curBlob->clearGradientData();
			vector<float> blobGradFeatures;
			blobGradFeatures.push_back(curBlob->getMeanGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getStdGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getEnergyGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getEntropyGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getKurtosisGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getSkewnessGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getMedianGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getMinGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getMaxGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getFirstQuartileGradMagnitude(magImg));
			blobGradFeatures.push_back(curBlob->getThirdQuartileGradMagnitude(magImg));
			gradientFeatures.push_back(blobGradFeatures);

#ifdef PRINT_FEATURES
			printf("Blob #%d MeanGrad=%lf MedianGrad=%d MinGrad=%d MaxGrad=%d FirstQuartile=%d ThirdQuartile=%d\n", i,curBlob->getMeanGradMagnitude(magImg), curBlob->getMedianGradMagnitude(magImg), curBlob->getMinGradMagnitude(magImg), curBlob->getMaxGradMagnitude(magImg), curBlob->getFirstQuartileGradMagnitude(magImg), curBlob->getThirdQuartileGradMagnitude(magImg));
#endif
		}

		cvReleaseImage(&magImg);

	}else{

#ifdef	USE_GPU
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}

		if(originalImageMaskNucleusBoxesGPU == NULL){
			this->uploadImageMaskNucleusToGPU();
		}

		// This is the data used to store the gradient results
		cv::gpu::GpuMat *magImgGPU = new cv::gpu::GpuMat(originalImageGPU->size(), CV_8UC1);
		cv::gpu::GpuMat kernel;
		cv::gpu::morphologyEx(*originalImageGPU, *magImgGPU, MORPH_GRADIENT, kernel, Point(-1,-1), 1);


		float *intensityFeatures = intensityFeaturesBlobGPU((char*)magImgGPU->data , originalImage->width, originalImage->height, internal_blobs.size(), (char*)originalImageMaskNucleusBoxesGPU->data, blobsMaskAllocatedMemorySize, false, 0);

		for(int blobId = 0; blobId < internal_blobs.size(); blobId++){
#ifdef PRINT_FEATURES
			printf("Blob #%d - MeanIntensity=%lf MedianIntensity=%d MinIntensity=%d MaxIntensity=%d FirstQuartile=%d ThirdQuartile=%d\n", blobId, intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+1], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+2], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+3], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+4], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+5]);
#endif
		}

		free(intensityFeatures);
		delete magImgGPU;
#endif

	}
}

// Computes Pixel Intensity features for each nuclei in the input image. And returns a array of array of features.
// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
// 	MeanIntensity; StdIntensity; EnergyIntensity, EntropyIntensity, KurtosisIntensity, SkewnessIntensity, 
//	MedianIntensity; MinIntensity; MaxIntensity; FirstQuartileIntensity; ThirdQuartileIntensity;
void RegionalMorphologyAnalysis::doIntensityBlob(vector<vector<float> > &intensityFeatures, bool nuclei, IplImage *inputImage, unsigned int procType){

	//	vector<vector<float> > intensityFeatures;
	if(procType == Constant::CPU){
		vector<Blob *>::iterator blobsIt = internal_blobs.begin();
		vector<Blob *>::iterator blobsEndIt = internal_blobs.end();

#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

		// this is calculation per cytoplasm
		if(nuclei == false){
			assert(this->internal_blobs.size() == this->cytoplasm_blobs.size());
			assert(inputImage != NULL);
			blobsIt = cytoplasm_blobs.begin();
			blobsEndIt = cytoplasm_blobs.end();
			cout << "computing features per cytoplasm"<<endl;
		}else{

			if(originalImage == NULL){
				cout << "doMorphometryFeatures: input image is NULL." <<endl;
				exit(1);
			}
			inputImage = originalImage;
		}

		if(inputImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			exit(1);
		}


		for(; blobsIt < blobsEndIt; blobsIt++){
			Blob *curBlob = *blobsIt;
			// Make sure to reset blob internal structure to 
			// ensure that intensity features a recalculated
			curBlob->clearIntensityData();
#ifdef PRINT_FEATURES
			printf("Blob #%d - MeanIntensity=%lf MedianIntensity=%d MinIntensity=%d MaxIntensity=%d FirstQuartile=%d ThirdQuartile=%d\n", i, curBlob->getMeanIntensity(originalImage), curBlob->getMedianIntensity(originalImage), curBlob->getMinIntensity(originalImage), curBlob->getMaxIntensity(originalImage), curBlob->getFirstQuartileIntensity(originalImage), curBlob->getThirdQuartileIntensity(originalImage));
#else
			vector<float> blobFeatures;
			blobFeatures.push_back(curBlob->getMeanIntensity(inputImage));
			blobFeatures.push_back(curBlob->getStdIntensity(inputImage));
			blobFeatures.push_back(curBlob->getEnergyIntensity(inputImage));
			blobFeatures.push_back(curBlob->getEntropyIntensity(inputImage));
			blobFeatures.push_back(curBlob->getKurtosisIntensity(inputImage));
			blobFeatures.push_back(curBlob->getSkewnessIntensity(inputImage));
			blobFeatures.push_back(curBlob->getMedianIntensity(inputImage));
			blobFeatures.push_back(curBlob->getMinIntensity(inputImage));
			blobFeatures.push_back(curBlob->getMaxIntensity(inputImage));
			blobFeatures.push_back(curBlob->getFirstQuartileIntensity(inputImage));
			blobFeatures.push_back(curBlob->getThirdQuartileIntensity(inputImage));
			intensityFeatures.push_back(blobFeatures);
#endif
		}
	}else{
#ifdef	USE_GPU
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}
		if(originalImageMaskNucleusBoxesGPU == NULL){
			this->uploadImageMaskNucleusToGPU();
		}


		float *intensityFeatures = intensityFeaturesBlobGPU((char*)originalImageGPU->data , originalImage->width, originalImage->height, internal_blobs.size(), (char*)originalImageMaskNucleusBoxesGPU->data, blobsMaskAllocatedMemorySize, false, 0);

		for(int blobId = 0; blobId < internal_blobs.size(); blobId++){
#ifdef PRINT_FEATURES
			printf("Blob #%d - MeanIntensity=%lf MedianIntensity=%d MinIntensity=%d MaxIntensity=%d FirstQuartile=%d ThirdQuartile=%d\n", blobId, intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+1], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+2], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+3], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+4], (int)intensityFeatures[blobId*Constant::N_INTENSITY_FEATURES+5]);
#endif
		}

		free(intensityFeatures);
#endif
	}
//	return intensityFeatures;
}

// Computes Haralick features for each nuclei in the input image. And returns a array of array of features.
// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
// 	Inertia; Energy; Entropy; Homogeneity; MaximumProbability; ClusterShade; ClusterProminence
void RegionalMorphologyAnalysis::doCoocPropsBlob(vector<vector<float> > &haralickFeatures, bool nuclei, IplImage *inputImage, unsigned int angle, unsigned int procType,  bool reuseItermediaryResults,  char*gpuTempData)
{

	bool reuseRes = true;
	bool useMask = true;
	if(procType == Constant::CPU){
		vector<Blob *>::iterator blobsIt = internal_blobs.begin();
		vector<Blob *>::iterator blobsEndIt = internal_blobs.end();

		// this is calculation per cytoplasm
		if(nuclei == false){
			assert(this->internal_blobs.size() == this->cytoplasm_blobs.size());
			assert(inputImage != NULL);
			blobsIt = cytoplasm_blobs.begin();
			blobsEndIt = cytoplasm_blobs.end();
			cout << "computing features per cytoplasm"<<endl;
		}else{

			if(originalImage == NULL){
				cout << "doCoocPropsBlob: input image is NULL." <<endl;
				exit(1);
			}
			inputImage = originalImage;
		}

		if(inputImage->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel only"<<endl;
			exit(1);
		}


		for(; blobsIt < blobsEndIt; blobsIt++){
			Blob *curBlob = *blobsIt;//internal_blobs[i];
			
			// Clear intermediary data and ensure that coocpropos is recalculated for whatwhever is the input image
			curBlob->clearCoocPropsData();

			vector<float> blobHaralickFeatures;
			blobHaralickFeatures.push_back(curBlob->inertiaFromCoocMatrix(angle, inputImage, useMask, reuseRes));
			blobHaralickFeatures.push_back(curBlob->energyFromCoocMatrix(angle, inputImage, useMask, reuseRes));
			blobHaralickFeatures.push_back(curBlob->entropyFromCoocMatrix(angle, inputImage, useMask, reuseRes));
			blobHaralickFeatures.push_back(curBlob->homogeneityFromCoocMatrix(angle, inputImage, useMask, reuseRes));
			blobHaralickFeatures.push_back(curBlob->maximumProbabilityFromCoocMatrix(angle, inputImage, useMask, reuseRes));
			blobHaralickFeatures.push_back(curBlob->clusterShadeFromCoocMatrix(angle, inputImage, useMask, reuseRes));
			blobHaralickFeatures.push_back(curBlob->clusterProminenceFromCoocMatrix(angle, inputImage, useMask, reuseRes));

			haralickFeatures.push_back(blobHaralickFeatures);

#ifdef PRINT_FEATURES
			cout << fixed << setprecision( 4 ) << "BlobId = "<< i << " inertia="<<curBlob->getInertia()<<" energy="<<curBlob->getEnergy()<<" entropy="<<curBlob->getEntropy()<<" homogeneity="<<curBlob->getHomogeneity()<< " max="<<curBlob->getMaximumProb()<<" shade="<<curBlob->getClusterShade()<<" prominence="<<curBlob->getClusterProminence()<<endl;
#endif

		}
	}else{
#ifdef	USE_GPU
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}
		if(originalImageMaskNucleusBoxesGPU == NULL){
			this->uploadImageMaskNucleusToGPU();
		}

		bool downloadRes = false;
		struct timeval startTime;
		struct timeval endTime;
		
		int *coocMatrixOut;

//		gettimeofday(&startTime, NULL);
		if(gpuTempData != NULL){
			coocMatrixOut = coocMatrixBlobGPU((char *)originalImageGPU->data , originalImage->width, originalImage->height, internal_blobs.size(), (char*)originalImageMaskNucleusBoxesGPU->data, blobsMaskAllocatedMemorySize, 8, angle, false, downloadRes, false, gpuTempData, 0);
		}else{
			coocMatrixOut = coocMatrixBlobGPU((char *)originalImageGPU->data , originalImage->width, originalImage->height, internal_blobs.size(), (char*)originalImageMaskNucleusBoxesGPU->data, blobsMaskAllocatedMemorySize, 8, angle, false, downloadRes, true, gpuTempData, 0);
		}
		gettimeofday(&endTime, NULL);
      
      		// calculate time in microseconds
//		double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
//		double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);

//		printf("CoocMatrixTime = %lf\n", (tE - tS)/1000000);



/*		gettimeofday(&startTime, NULL);
		cudaFreeWrapper(aux_normTemp);

		gettimeofday(&endTime, NULL);
		 tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
		 tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
		mallocTime += (tE - tS)/1000000;
		printf("Malloc/Free time = %lf\n", mallocTime);*/
	

		if(downloadRes){
			for(int blobId = 0; blobId < internal_blobs.size(); blobId++){
				//		unsigned int *auxPtrCooc = &coocMatrixOut[(coocSize * coocSize * blobId)];
				const int printWidth = 12;
				for(int i = 0; i < coocSize; i++){
					int offSet = i * coocSize;
					for(int j = 0; j < coocSize; j++){
						cout << setw(printWidth) << coocMatrixOut[(coocSize * coocSize * blobId)+ offSet + j]<< " ";
					}
					cout <<endl;
				}
				cout <<endl;
			}
			free(coocMatrixOut);
		}else{

//			gettimeofday(&startTime, NULL);
			float *haraLickFeatures = calcHaralickGPUBlob(coocMatrixOut, 8, (int)internal_blobs.size(), 0, (float*)gpuTempData);
//			gettimeofday(&endTime, NULL);
      
      			// calculate time in microseconds
//			tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
//			tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
//
//			printf("HaralickFTime = %lf\n", (tE - tS)/1000000);

			for(int i = 0; i < internal_blobs.size(); i++){
#ifdef PRINT_FEATURES
				cout << fixed << setprecision( 4 ) << "BlobId = "<< i << " inertia="<<haraLickFeatures[i * 7]<<" energy="<<haraLickFeatures[i * 7 + 1]<<" entropy="<<haraLickFeatures[i * 7 + 2]<<" homogeneity="<<haraLickFeatures[i * 7 + 3]<< " max="<<haraLickFeatures[i * 7 + 4]<<" shade="<<haraLickFeatures[i * 7 + 5]<<" prominence="<<haraLickFeatures[i * 7 + 6]<<endl;
#endif
			}
			free(haraLickFeatures);
		}
#endif
	}
//	return haralickFeatures;
}


void RegionalMorphologyAnalysis::doAll(bool nuclei, IplImage *inputImage)
{
	vector<Blob *>::iterator blobsIt = internal_blobs.begin();
	vector<Blob *>::iterator blobsEndIt = internal_blobs.end();

	// this is calculation per cytoplasm
	if(nuclei == false){
		assert(this->internal_blobs.size() == this->cytoplasm_blobs.size());
		assert(inputImage != NULL);
		blobsIt = cytoplasm_blobs.begin();
		blobsEndIt = cytoplasm_blobs.end();
		cout << "computing features per cytoplasm"<<endl;
	}else{

		if(originalImage == NULL){
			cout << "doAll: input image is NULL." <<endl;
			exit(1);
		}
		inputImage = originalImage;
	}

	if(inputImage->nChannels != 1){
		cout << "Error: input image should be grayscale with one channel only"<<endl;
		exit(1);
	}


#ifdef VISUAL_DEBUG
		IplImage* visualizationImage = cvCreateImage( cvGetSize(originalImage), 8, 3);
		cvCvtColor(originalImage, visualizationImage, CV_GRAY2BGR);

		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvWaitKey(0);
#endif

//#pragma omp parallel for
	
	for(int i = 0; blobsIt < blobsEndIt; blobsIt++, i++){
		Blob *curBlob = *blobsIt;
#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(255, 0, 0), CV_RGB(255,0,0));
		cvNamedWindow("Input Image");
		cvShowImage("Input Image", visualizationImage);
		cvNamedWindow("Mask");
		cvShowImage("Mask", curBlob->getMask());
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
		printf(" CannyArea = %d", curBlob->getCannyArea(originalImage, 35, 100));
		printf(" SobelArea = %d\n", curBlob->getSobelArea(originalImage, 5, 5, 7 ));


#ifdef VISUAL_DEBUG
		DrawAuxiliar::DrawBlob(visualizationImage, curBlob, CV_RGB(0, 0, 255), CV_RGB(0,0,255));

		cvDestroyWindow("Mask");
#endif
//		delete curBlob;
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
		unsigned char *ptr = (unsigned char *) originalImage->imageData + i * originalImage->widthStep;
		for(int j =0; j < originalImage->width; j++){
			unsigned char elementIJ = ptr[j];

			normImg[i*originalImage->width + j] = round((slope*(float)elementIJ + intercept));
	//		CvScalar elementIJ = cvGet2D(originalImage, i, j);
	//		normImg[i*originalImage->width + j] = round((slope*(float)elementIJ.val[0] + intercept));
		}
	}

	switch(angle){

		case Constant::ANGLE_0:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height; i++){
				int offSet = i*originalImage->width;
				for(int j=0; j<originalImage->width-1; j++){
					if(((normImg[offSet+j])-1) < coocSize && ((normImg[offSet+j+1])-1) < coocSize){
						unsigned int coocAddress = (unsigned int )((normImg[offSet+j])-1) * coocSize;
						coocAddress += (int)(normImg[offSet+j+1]-1);
						coocMatrix[angle][coocAddress]++;
					}
				}
			}
			break;

		case Constant::ANGLE_45:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height-1; i++){
				int offSetI = i*originalImage->width;
				int offSetI2 = (i+1)*originalImage->width;
				for(int j=0; j<originalImage->width-1; j++){
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j])-1) * coocSize;
					coocAddress += (int)(normImg[offSetI +j +1 ] -1);
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;
		case Constant::ANGLE_90:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height-1; i++){
				int offSetI = i*originalImage->width;
				int offSetI2 = (i+1)*originalImage->width;
				for(int j=0; j<originalImage->width; j++){
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j])-1) * coocSize;
					coocAddress += (int)(normImg[offSetI + j ] -1);
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;

		case Constant::ANGLE_135:
			//build co-occurrence matrix
			for (int i=0; i<originalImage->height-1; i++){
				int offSetI = i*originalImage->width;
				int offSetI2 = (i+1)*originalImage->width;
				for(int j=0; j<originalImage->width-1; j++){
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j+1])-1) * coocSize;
					coocAddress += (int)(normImg[offSetI + j ] -1);
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
		if(procType == Constant::CPU){
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
		if(procType == Constant::CPU){
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
		if(procType == Constant::CPU){
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
		if(procType == Constant::CPU){
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
		if(procType == Constant::CPU){
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
		if(procType == Constant::CPU){
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
		if(procType == Constant::CPU){
			doCoocMatrix(angle);
		}else{
			doCoocMatrixGPU(angle);
		}
	}
	clusterProminence = Operators::clusterProminenceFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return clusterProminence;
}


void RegionalMorphologyAnalysis::doCoocMatrixGPU(unsigned int angle){
#ifdef	USE_GPU
	if(coocMatrix[angle] == NULL){
		coocMatrix[angle] = (unsigned int *)calloc(coocSize * coocSize,  sizeof(unsigned int));
	}else{
		// It has been calculated before, so clean it up.
		memset(coocMatrix[angle], 0, coocSize * coocSize* sizeof(unsigned int));
		coocMatrixCount[angle] = 0;
	}

	if(originalImageGPU == NULL){
		this->uploadImageToGPU();
	}

	unsigned int width = originalImage->width;
	unsigned int height = originalImage->height;
	unsigned int *test;

	coocMatrixGPU((char*)originalImageGPU->data , originalImage->width, originalImage->height, coocMatrix[angle],  coocSize, angle, false, 0 );


	for(int i = 0; i < coocSize; i++){
		for(int j = 0; j < coocSize; j++){
			coocMatrixCount[angle] += coocMatrix[angle][i*coocSize + j];
		}
	}
#endif
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


unsigned int *RegionalMorphologyAnalysis::calcIntensityHistogram(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId){
	unsigned int *intensity_hist_ret = NULL;

	if(procType == Constant::CPU){
		if(roi){
			cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
			cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
		}

		if(useMask){

			intensity_hist_ret = Operators::buildHistogram256CPU(originalImage, originalImageMask);
		}else{
			intensity_hist_ret = Operators::buildHistogram256CPU(originalImage);
		}

		if(roi){
			cvResetImageROI(originalImage);
			cvResetImageROI(originalImageMask);
		}


	}else{
#ifdef	USE_GPU
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}

		cv::gpu::GpuMat *gpuImageRegionInterest = this->originalImageGPU;
		if(roi != NULL){
			gpuImageRegionInterest = new cv::gpu::GpuMat(*originalImageGPU, Rect( roi->x, roi->y, roi->width, roi->height ));
		}	

		if(useMask){
			if(originalImageMaskGPU == NULL){
				this->uploadImageMaskToGPU();
			}

			cv::gpu::GpuMat *gpuImageMaskRegionInterest = this->originalImageMaskGPU;
			if(roi != NULL){
				gpuImageMaskRegionInterest = new cv::gpu::GpuMat(*originalImageMaskGPU, Rect( roi->x, roi->y, roi->width, roi->height ));
			}

			intensity_hist_ret = Operators::buildHistogram256GPU(gpuImageRegionInterest, gpuImageMaskRegionInterest);

			if(roi != NULL){
				delete gpuImageMaskRegionInterest;
			}
		}else{
			intensity_hist_ret = Operators::buildHistogram256GPU(gpuImageRegionInterest);
		}

		if(roi != NULL){
			delete gpuImageRegionInterest;
		}
#endif
	}
	return intensity_hist_ret;
}

double RegionalMorphologyAnalysis::calcMeanIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	double meanIntensity = 0.0;

	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);
/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}

			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
			}

		}else{
			if(originalImageGPU == NULL){
				this->uploadImageToGPU();
			}

			cv::gpu::GpuMat *gpuImageRegionInterest = this->originalImageGPU;
			if(roi != NULL){
				gpuImageRegionInterest = new cv::gpu::GpuMat(*originalImageGPU, Rect( roi->x, roi->y, roi->width, roi->height ));
			}	

			if(useMask){
				if(originalImageMaskGPU == NULL){
					this->uploadImageMaskToGPU();
				}

				cv::gpu::GpuMat *gpuImageMaskRegionInterest = this->originalImageMaskGPU;
				if(roi != NULL){
					gpuImageMaskRegionInterest = new cv::gpu::GpuMat(*originalImageMaskGPU, Rect( roi->x, roi->y, roi->width, roi->height ));
				}

				intensity_hist = Operators::buildHistogram256GPU(gpuImageRegionInterest, gpuImageMaskRegionInterest);

				if(roi != NULL){
					delete gpuImageMaskRegionInterest;
				}
			}else{
				intensity_hist = Operators::buildHistogram256GPU(gpuImageRegionInterest);
			}

			if(roi != NULL){
				delete gpuImageRegionInterest;
			}	

		}*/
	}
	meanIntensity = Operators::calcMeanFromHistogram(intensity_hist, 256);

	return meanIntensity;
}



double RegionalMorphologyAnalysis::calcStdIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	double stdIntensity = 0.0;


	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);

/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}


			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}

			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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

		}*/
	}
	stdIntensity = Operators::calcStdFromHistogram(intensity_hist, 256);

	return stdIntensity;
}



int RegionalMorphologyAnalysis::calcMedianIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int medianIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);

		/*if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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

		}*/
	}
	medianIntensity = Operators::calcMedianFromHistogram(intensity_hist, 256);

	return medianIntensity;
}



int RegionalMorphologyAnalysis::calcMinIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int minIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);

/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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

		}*/
	}
	minIntensity = Operators::calcMinFromHistogram(intensity_hist, 256);

	return minIntensity;
}



int RegionalMorphologyAnalysis::calcMaxIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int maxIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);

/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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

		}*/
	}
	maxIntensity = Operators::calcMaxFromHistogram(intensity_hist, 256);

	return maxIntensity;
}



int RegionalMorphologyAnalysis::calcFirstQuartileIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int firstQuartileIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);

/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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

		}*/
	}
	firstQuartileIntensity = Operators::calcFirstQuartileFromHistogram(intensity_hist, 256);

	return firstQuartileIntensity;
}



int RegionalMorphologyAnalysis::calcSecondQuartileIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int secondQuartileIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}
		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);

/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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
		}*/
	}
	secondQuartileIntensity = Operators::calcSecondQuartileFromHistogram(intensity_hist, 256);

	return secondQuartileIntensity;
}



int RegionalMorphologyAnalysis::calcThirdQuartileIntensity(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int thirdQuartileIntensity = 0;
	if(reuseItermediaryResults != true || intensity_hist == NULL){
		if(intensity_hist != NULL){
			free(intensity_hist);
		}

		intensity_hist = calcIntensityHistogram(useMask, procType, reuseItermediaryResults, roi);
/*		if(procType == Constant::CPU){
			if(roi){
				cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
				cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			}

			if(useMask){
				intensity_hist = Operators::buildHistogram256CPU(originalImage, originalImageMask);
			}else{
				intensity_hist = Operators::buildHistogram256CPU(originalImage);
			}
			if(roi){
				cvResetImageROI(originalImage);
				cvResetImageROI(originalImageMask);
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

		}*/
	}
	thirdQuartileIntensity = Operators::calcThirdQuartileFromHistogram(intensity_hist, 256);

	return thirdQuartileIntensity;
}



bool RegionalMorphologyAnalysis::uploadImageToGPU()
{
#ifdef	USE_GPU
	originalImageGPU = new cv::gpu::GpuMat(originalImage);
#endif
}

bool RegionalMorphologyAnalysis::uploadImageMaskToGPU()
{
#ifdef	USE_GPU
	originalImageMaskGPU = new cv::gpu::GpuMat(originalImageMask);
#endif
}

bool RegionalMorphologyAnalysis::uploadImageMaskNucleusToGPU(){
#ifdef	USE_GPU
	originalImageMaskNucleusBoxesGPU = new cv::gpu::GpuMat(*originalImageMaskNucleusBoxes);
#endif
}

void RegionalMorphologyAnalysis::releaseGPUImage()
{
#ifdef	USE_GPU
	if(originalImageGPU != NULL){
		delete originalImageGPU;
		originalImageGPU = NULL;
	}
#endif
}

void RegionalMorphologyAnalysis::releaseGPUMask()
{
#ifdef	USE_GPU
	if(originalImageMaskGPU != NULL){
		delete originalImageMaskGPU;
		originalImageMaskGPU = NULL;
	}
#endif
}

void RegionalMorphologyAnalysis::releaseImageMaskNucleusToGPU(){
#ifdef	USE_GPU
	if(originalImageMaskNucleusBoxesGPU != NULL){
		delete originalImageMaskNucleusBoxesGPU;
		originalImageMaskNucleusBoxesGPU = NULL;
	}
#endif
}

int RegionalMorphologyAnalysis::calcCannyArea(int procType, double lowThresh, double highThresh, int apertureSize, ROI *roi, int gpuId)
{
	int cannyPixels = 0;

	if(procType == Constant::CPU){
		if(roi){
			cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
			cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
		}

		IplImage *cannyRes = cvCreateImage( cvSize(originalImage->width, originalImage->height), IPL_DEPTH_8U, 1);

		cvCopy(originalImage, cannyRes, NULL);

		cvAnd(cannyRes, this->getMask(), cannyRes);

		cvCanny(cannyRes, cannyRes, lowThresh, highThresh, apertureSize);

		// Calculate the #white pixels and divide by blob area
		cannyPixels = cvCountNonZero(cannyRes);

		cvReleaseImage(&cannyRes);

		if(roi){
			cvResetImageROI(originalImage);
			cvResetImageROI(originalImageMask);
		}

	}else{
#ifdef	USE_GPU
		if(originalImageGPU == NULL){
			this->uploadImageMaskToGPU();
		}
		printf("Warnning: CannyArea is not implemented for GPU!!\n");
#endif
	}

	return cannyPixels;
}



IplImage *RegionalMorphologyAnalysis::getMask()
{
	return this->originalImageMask;
}





unsigned int *RegionalMorphologyAnalysis::calcGradientHistogram(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	cout << "calcGradHist"<<endl;
	unsigned int *grad_mag_hist_ret = NULL;
	if(procType == Constant::CPU){
		if(roi){
			cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
			cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
		}
		// Create a Mat header pointing to the C image loaded
		Mat originalImgHeader(originalImage);
		Mat dest;

		Mat element;

		morphologyEx(originalImgHeader, dest, MORPH_GRADIENT, element,  Point(-1,-1), 1);

		IplImage magImg = dest;

		if(useMask){
			grad_mag_hist_ret = Operators::buildHistogram256CPU(&magImg, originalImageMask);
		}else{
			grad_mag_hist_ret = Operators::buildHistogram256CPU(&magImg);
		}

		if(roi){
			cvResetImageROI(originalImage);
			cvResetImageROI(originalImageMask);
		}

	}else{
#ifdef	USE_GPU
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}

		cv::gpu::GpuMat *gpuImageRegionInterest = this->originalImageGPU;
		if(roi != NULL){
			gpuImageRegionInterest = new cv::gpu::GpuMat(*originalImageGPU, Rect( roi->x, roi->y, roi->width, roi->height ));

		}
		// This is the data used to store the gradient results
		cv::gpu::GpuMat *magImg = new cv::gpu::GpuMat(gpuImageRegionInterest->size(), CV_8UC1);
		cout << "magImg->width = "<< gpuImageRegionInterest->size().width <<endl;

		cv::gpu::GpuMat kernel;
		cv::gpu::morphologyEx(*gpuImageRegionInterest, *magImg, MORPH_GRADIENT, kernel, Point(-1,-1), 1);

		if(useMask){
			if(originalImageMaskGPU == NULL){
				this->uploadImageMaskToGPU();
			}

			cv::gpu::GpuMat *gpuImageMaskRegionInterest = this->originalImageMaskGPU;
			if(roi != NULL){
				gpuImageMaskRegionInterest = new cv::gpu::GpuMat(*originalImageMaskGPU, Rect( roi->x, roi->y, roi->width, roi->height ));
			}

			grad_mag_hist_ret = Operators::buildHistogram256GPU(magImg, gpuImageMaskRegionInterest);
			if(roi != NULL){
				delete gpuImageMaskRegionInterest;
			}

		}else{
			grad_mag_hist_ret = Operators::buildHistogram256GPU(magImg);
		}
		if(roi != NULL){
			delete gpuImageRegionInterest;
		}
		delete magImg;
#endif
	}
	return grad_mag_hist_ret;
}

double RegionalMorphologyAnalysis::calcMeanGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	double meanGradientMagnitude = 0.0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}


		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	meanGradientMagnitude = Operators::calcMeanFromHistogram(gradient_hist, 256);
	return meanGradientMagnitude;
}

double RegionalMorphologyAnalysis::calcStdGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	double stdGradientMagnitude = 0.0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}

		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	stdGradientMagnitude = Operators::calcStdFromHistogram(gradient_hist, 256);
	return stdGradientMagnitude;
}


int RegionalMorphologyAnalysis::calcMedianGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int medianGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}

		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	medianGradientMagnitude = Operators::calcMedianFromHistogram(gradient_hist, 256);
	return medianGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcMinGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int minGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}

		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	minGradientMagnitude = Operators::calcMinFromHistogram(gradient_hist, 256);
	return minGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcMaxGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int maxGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}

		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	maxGradientMagnitude = Operators::calcMaxFromHistogram(gradient_hist, 256);
	return maxGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcFirstQuartileGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int firstQuartileGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}
		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	firstQuartileGradientMagnitude = Operators::calcFirstQuartileFromHistogram(gradient_hist, 256);
	return firstQuartileGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcSecondQuartileGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int secondQuartileGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}

		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	secondQuartileGradientMagnitude = Operators::calcSecondQuartileFromHistogram(gradient_hist, 256);
	return secondQuartileGradientMagnitude;
}

int RegionalMorphologyAnalysis::calcThirdQuartileGradientMagnitude(bool useMask, int procType, bool reuseItermediaryResults, ROI *roi, int gpuId)
{
	int thirdQuartileGradientMagnitude = 0;
	if(reuseItermediaryResults != true || gradient_hist == NULL){
		if(gradient_hist != NULL){
			free(gradient_hist);
		}

		gradient_hist = calcGradientHistogram(useMask, procType, reuseItermediaryResults, roi);

	}
	thirdQuartileGradientMagnitude = Operators::calcThirdQuartileFromHistogram(gradient_hist, 256);
	return thirdQuartileGradientMagnitude;
}

void RegionalMorphologyAnalysis::printBlobsAllocatedInfo()
{
	if(originalImageMaskNucleusBoxes->data != NULL){
		// jump area where offsets for each mask are stored
//		int offset = internal_blobs.size() * sizeof(int);

		// create pointer to 'walk' on the offsets
		int *offset_ptr = (int *) originalImageMaskNucleusBoxes->data;

		// print information related to each blob
		for(int i = 0; i < internal_blobs.size(); i++){
			int offset = offset_ptr[i];
			cout << "bounding_box.x=" << ((int*)(originalImageMaskNucleusBoxes->data+offset))[0];
			cout << " bounding_box.y=" << ((int*)(originalImageMaskNucleusBoxes->data+offset))[1];
			cout << " bounding_box.width=" << ((int*)(originalImageMaskNucleusBoxes->data+offset))[2];
			cout << " bounding_box.height=" << ((int*)(originalImageMaskNucleusBoxes->data+offset))[3];
			cout << " accOffset = "<< offset<<endl;

		}
		cout << " blobMaskAllocatedMemorySize = "<< blobsMaskAllocatedMemorySize<<endl;
	}
}


void RegionalMorphologyAnalysis::printStats()
{
	cout << "NumBlobs, "<<internal_blobs.size()<<endl;
	for(int i = 0; i < internal_blobs.size(); i++){
		Blob *curBlob = internal_blobs[i];
		float area = cvCountNonZero((curBlob->getMask()));
		cout << i <<" "<<area<<endl;
	}
}

int RegionalMorphologyAnalysis::calcSobelArea(int procType, int xorder, int yorder, int apertureSize, bool useMask, ROI *roi, int gpuId)
{
	int sobelPixels = 0;

	if(procType == Constant::CPU){
		if(roi){
			cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
			cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
		}

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

		if(roi){
			cvResetImageROI(originalImage);
			cvResetImageROI(originalImageMask);
		}

	}else{
#ifdef	USE_GPU
		if(originalImageMaskGPU == NULL && useMask){
			this->uploadImageMaskToGPU();
		}
		if(originalImageGPU == NULL){
			this->uploadImageToGPU();
		}

		cv::gpu::GpuMat *gpuImageRegionInterest = this->originalImageGPU;
		cv::gpu::GpuMat *gpuImageRegionInterestMask = this->originalImageGPU;
		if(roi != NULL){
			cvSetImageROI(originalImage, cvRect(roi->x, roi->y, roi->width, roi->height));
			cvSetImageROI(originalImageMask, cvRect(roi->x, roi->y, roi->width, roi->height));
			gpuImageRegionInterest = new cv::gpu::GpuMat(*originalImageGPU, Rect( roi->x, roi->y, roi->width, roi->height ));

			gpuImageRegionInterestMask = new cv::gpu::GpuMat(*originalImageMaskGPU, Rect( roi->x, roi->y, roi->width, roi->height ));
		}	

		cv::gpu::GpuMat *sobelResGPU = new cv::gpu::GpuMat(cvGetSize(originalImage), CV_8U);
		cv::gpu::GpuMat *sobelResGPU2;

		if(useMask){
			sobelResGPU2 = new cv::gpu::GpuMat(cvGetSize(originalImage), CV_8U);

			cv::gpu::bitwise_and(*gpuImageRegionInterest, *gpuImageRegionInterestMask, *sobelResGPU);

//			cv::gpu::multiply(*originalImageGPU, *originalImageMaskGPU, *sobelResGPU);
			cv::gpu::Sobel(*sobelResGPU, *sobelResGPU2, CV_8U, xorder, yorder, apertureSize);

			sobelPixels = gpu::countNonZero(*sobelResGPU2);


			delete sobelResGPU2;

		}else{

			cv::gpu::Sobel(*gpuImageRegionInterest, *gpuImageRegionInterestMask, CV_8U, xorder, yorder, apertureSize);
//			cv::gpu::Sobel(*originalImageGPU, *sobelResGPU, CV_8U, xorder, yorder, apertureSize);
			sobelPixels = gpu::countNonZero(*sobelResGPU);
		}
		if(roi != NULL){
			cvResetImageROI(originalImage);
			cvResetImageROI(originalImageMask);
			delete gpuImageRegionInterest;
			delete gpuImageRegionInterestMask;
		}	

		delete sobelResGPU;
#endif
	}

	return sobelPixels;
}
