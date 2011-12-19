/*
 * Blob.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#include "Blob.h"


Blob::Blob(CvSeq* first_contour, CvSize originalImageSize, CvPoint offsetOriginalImage, IplImage *inputMask, CvRect *bb ) {

	self_storage = cvCreateMemStorage();

	// this is the offset of the point represented in the contours inside the original image
	offsetInImage.x = offsetOriginalImage.x;
	offsetInImage.y = offsetOriginalImage.y;

	if(first_contour != NULL){
		external_contour = new Contour(first_contour, bb);
	}

	this->originalImageSize = originalImageSize;
	CvSeq *aux = first_contour;

	if(aux->v_next){
		aux = aux->v_next;
		Contour *contour = new Contour(aux);
		internal_contours.push_back(contour);


		while(aux->h_next){
			aux = aux->h_next;
			Contour *contour = new Contour(aux);
			internal_contours.push_back(contour);
		}
	}
	mask = inputMask;
	ROISubImage = NULL;
	intensity_hist = NULL;
	grad_hist = NULL;
	min_fitting_ellipse.size.width = -1.0;
	majorAxisLength = -1.0;
	minorAxisLength = -1.0;
	isImage = true;

	// allocate pointer to the coocurrence matrix for the 4 possible angles
	coocMatrix = (unsigned int **) malloc(sizeof(unsigned int *) * Constant::NUM_ANGLES);
	coocMatrixCount = (unsigned int *)malloc(sizeof(unsigned int) * Constant::NUM_ANGLES);

	for(int i = 0; i < Constant::NUM_ANGLES; i++){
		coocMatrix[i] = NULL;
		coocMatrixCount[i] = 0;
	}
	coocSize = 8;

	setClusterProminence(0.0);
	setClusterShade(0.0);
	setEnergy(0.0);
	setEntropy(0.0);
	setHomogeneity(0.0);
	setInertia(0.0);
	setMaximumProb(0.0);
}

Blob::~Blob() {
	if(external_contour != NULL){
		delete external_contour;
	}

	for(int i = 0; i < internal_contours.size(); i++){
		delete internal_contours[i];
	}

	internal_contours.clear();

	cvReleaseMemStorage(&self_storage);

	if(mask != NULL && isImage){
		cvReleaseImage(&mask);
	}
	if(mask != NULL && !isImage){
		cvReleaseImageHeader(&mask);
	}
	if(intensity_hist != NULL){
		cvReleaseHist(&intensity_hist);
	}
	if(grad_hist != NULL){
		cvReleaseHist(&grad_hist);
	}
	if(ROISubImage != NULL){
		cvReleaseImageHeader(&ROISubImage);
	}
	for(int i = 0; i < Constant::NUM_ANGLES; i++){
		if(coocMatrix[i] != NULL){
			free(coocMatrix[i]);
		}
	}
	free(coocMatrix);
	free(coocMatrixCount);

}

void Blob::clearIntensityData(){
	if(intensity_hist != NULL){
		cvReleaseHist(&intensity_hist);
		intensity_hist = NULL;
		intensity_hist_points = 0;
	}

}
void Blob::clearGradientData(){
	if(grad_hist != NULL){
		cvReleaseHist(&grad_hist);
		grad_hist = NULL;
		grad_hist_points = 0;
	}
}
void Blob::clearCoocPropsData(){
	for(int i = 0; i < Constant::NUM_ANGLES; i++){
		if(coocMatrix[i] != NULL){
			free(coocMatrix[i]);
			coocMatrix[i] = NULL;
			coocMatrixCount[i] = 0;
		}
	}

}

float Blob::getArea()
{
	float areaContour = external_contour->getArea();

	IplImage *maskContour = getMask();

	for(int i = 0; i < internal_contours.size(); i++){
		areaContour -= internal_contours[i]->getArea();
	}

//	cout << "AreaContour = "<< areaContour<<endl;
	float area = cvCountNonZero(this->getMask());
	return area;
//	return area;
}

double Blob::getMoment(int p, int q)
{
	double moment = external_contour->getMoment(p, q);

	for(int i = 0; i < internal_contours.size(); i++){
		moment -= internal_contours[i]->getMoment(p, q);
	}
	return moment;
}

CvBox2D Blob::getEllipse(){

	// Do we have to fit the ellipse?
	if(min_fitting_ellipse.size.width == -1.0){
//		try{
//			min_fitting_ellipse = cvFitEllipse2(external_contour->getCl());
//	
//		}
//		catch( cv:: Exception& e){
//			const char* err_msg = e.what();
//			min_fitting_ellipse.size.width = 0;
//			min_fitting_ellipse.size.height = 0;
//			min_fitting_ellipse.angle = 0;
////			printf("Warning: failed to fitEllipse: %s", err_msg);
//		}
		double u00,u11,u01,u10,u20,u02, delta, num, den, temp;

		// central moments calculation
		u00 = this->getMoment(0, 0);

		// empty blob?
		if ( u00 <= 0 ){

			min_fitting_ellipse.size.width = 0;
			min_fitting_ellipse.size.height = 0;
			min_fitting_ellipse.center.x = 0;
			min_fitting_ellipse.center.y = 0;
			min_fitting_ellipse.angle = 0;

		}else{
			u10 = this->getMoment(1,0) / u00;
			u01 = this->getMoment(0,1) / u00;

			u11 = -(this->getMoment(1,1) - this->getMoment(1,0) * this->getMoment(0,1) / u00 ) / u00;
			u20 = (this->getMoment(2,0) - this->getMoment(1,0) * this->getMoment(1,0) / u00 ) / u00;
			u02 = (this->getMoment(0,2) - this->getMoment(0,1) * this->getMoment(0,1) / u00 ) / u00;


			// elipse calculation
			delta = sqrt( 4*u11*u11 + (u20-u02)*(u20-u02) );
			min_fitting_ellipse.center.x = u10;
			min_fitting_ellipse.center.y = u01;

			temp = u20 + u02 + delta;
			if( temp > 0 )
			{
				min_fitting_ellipse.size.width = sqrt( 2*(u20 + u02 + delta ));
			}
			else
			{
				min_fitting_ellipse.size.width = 0;
				return min_fitting_ellipse;
			}

			temp = u20 + u02 - delta;
			if( temp > 0 )
			{
				min_fitting_ellipse.size.height = sqrt( 2*(u20 + u02 - delta ) );
			}
			else
			{
				min_fitting_ellipse.size.height = 0;
				return min_fitting_ellipse;
			}

			// elipse orientation
			if (u20 > u02)
			{
				num = u02 - u20 + sqrt((u02 - u20)*(u02 - u20) + 4*u11*u11);
				den = 2*u11;
			}
			else
			{
				num = 2*u11;
				den = u20 - u02 + sqrt((u20 - u02)*(u20 - u02) + 4*u11*u11);
			}
			if( num != 0 && den  != 00 )
			{
//				min_fitting_ellipse.angle = 180.0 + (180.0 / CV_PI) * atan( num / den );

				min_fitting_ellipse.angle = (180.0 / CV_PI) * atan( num / den );
			}
			else
			{
				min_fitting_ellipse.angle = 0;
			}
		}
	}
	return min_fitting_ellipse;
}

float Blob::getConvexArea()
{
	return external_contour->getConvexArea();
}

float Blob::getExtent()
{
//	CvRect boundingBox = this->getNonInclinedBoundingBox();
	float extent = this->getArea()/ this->getNonInclinedBoundingBoxArea();
	return extent;
}

float Blob::getAspectRatio()
{
	float aspectRatio = 0.0;
	if(external_contour->getBoundingBoxHeight() != -1.0 && external_contour->getBoundingBoxWidth() != -1.0){
		aspectRatio = external_contour->getBoundingBoxWidth()/external_contour->getBoundingBoxHeight();
	}
	return aspectRatio;
}

float Blob::getPerimeter()
{
	float perimeter = external_contour->getPerimeter();

//	for(int i = 0; i < internal_contours.size(); i++){
//		perimeter += internal_contours[i]->getPerimeter();
//	}
	return perimeter;
}

float Blob::getEquivalentDiameter()
{
	float equivalentDiameter = 2 * sqrtf(getArea()) / sqrtf(CV_PI) ;
	return equivalentDiameter;
}

float Blob::getFilledArea()
{
	return external_contour->getArea();
}


float Blob::getMinorAxisLength()
{
	if(minorAxisLength == -1.0){
		// Do we have to fit the elipse?
		if(min_fitting_ellipse.size.width == -1.0){
			this->getEllipse();
		//	min_fitting_ellipse = cvFitEllipse2(c_l);
		}
		if(min_fitting_ellipse.size.width > min_fitting_ellipse.size.height){
			minorAxisLength = 2 * min_fitting_ellipse.size.height;
		}else{
			minorAxisLength = 2 * min_fitting_ellipse.size.width;
		}
	}
    return minorAxisLength;

	//return external_contour->getMinorAxisLength();
}

float Blob::getSolidity()
{
	return (getArea()/getConvexArea());
}

float Blob::getOrientation()
{
	return min_fitting_ellipse.angle;
}


float Blob::getMajorAxisLength()
{
	if(majorAxisLength == -1.0){
		// Do we have to fit the ellipse?
		if(min_fitting_ellipse.size.width == -1.0){
			this->getEllipse();
			//min_fitting_ellipse = cvFitEllipse2(c_l);
		}
		if(min_fitting_ellipse.size.width > min_fitting_ellipse.size.height){
			majorAxisLength = 2 * min_fitting_ellipse.size.width;
		}else{
			majorAxisLength = 2 * min_fitting_ellipse.size.height;
		}
	}
    return majorAxisLength;

	//return external_contour->getMajorAxisLength();
}

float Blob::getCompacteness()
{
	return (4*CV_PI*getArea()/pow(getPerimeter(),2));
}


float Blob::getEccentricity()
{
	float foci = pow((getMajorAxisLength()),2) - pow((getMinorAxisLength()), 2);

	if(foci > 0){
		foci =  sqrtf(foci);
	}else{
		foci = 0.0;
	}

	float eccentricity = foci / getMajorAxisLength();
#ifdef DEBUG
		cout << "Eccentricity = "<< eccentricity<<endl<< " majorAxis ="<< getMajorAxisLength() << " minorAxis = "<< getMinorAxisLength() << " foci = "<<foci<<endl;
#endif

    return eccentricity;
}

float Blob::getConvexDeficiency()
{
	 return (getConvexArea()-getArea())/getArea();
}


int Blob::getEulerNumber()
{
	return (1-internal_contours.size());
}

float Blob::getEllipticity()
{
	//return external_contour->getEllipticity();
	return (getMajorAxisLength()-getMinorAxisLength())/getMajorAxisLength();
}

CvRect Blob::getNonInclinedBoundingBox()
{
	CvRect bb = this->external_contour->getNonInclinedBoundingBox();
	bb.x += offsetInImage.x;
	bb.y += offsetInImage.y;
	return bb;
}

void Blob::calcIntensityHistogram(IplImage *img)
{
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ) && intensity_hist == NULL){

		// This is the first time we're calculating the histogram, so
		// we have to create its structure.
		int numBins = 256;
		float range[] = {0, 256};
		float *ranges[] = { range };
		intensity_hist = cvCreateHist(1, &numBins, CV_HIST_ARRAY, ranges, 1);

		// Region of interest is the same as the bounding box of the blob
		IplImage *ROISubImage = getROISubImage(img);


		// Do not be silly and use the cvSetImageROI as bellow. It will store the ROI in the
		// Image itself, and as the library may be multithreaded it is likely to incurr in errors.
//		cvSetImageROI(img, blob_bounding_box);

		// Calculates the histogram in the input image for the pixels in the input mask
		cvCalcHist(&ROISubImage, intensity_hist, 0, this->getMask());

		intensity_hist_points = 0;

//		cout<< "Starting hist"<<endl;
		for(int i=0;i<256;i++){
				float histValue = cvQueryHistValue_1D(intensity_hist, i);
				intensity_hist_points += (unsigned int)histValue;
//				cout<< histValue<<endl;
		}
//		cvAnd(ROISubImage, this->getMask(), ROISubImage);
//		cout << "NonZero = " << cvCountNonZero(ROISubImage) <<" histPoints = " << intensity_hist_points <<endl;

//		cout<< "End hist"<<endl;
#ifdef VISUAL_DEBUG
		cvNamedWindow("Img");
		cvShowImage("Img", img);
		cvNamedWindow("Img-roi");
		cvShowImage("Img-roi", ROISubImage);
		cvNamedWindow("Img-mask");
		cvShowImage("Img-mask", this->getMask());

		cvSaveImage("cytoMask.tif", this->getMask());
		cvSaveImage("cytoROI.tif", ROISubImage);
	
		IplImage* histDraw = DrawAuxiliar::DrawHistogram(intensity_hist);
		cvNamedWindow("Resulting Intensity Histogram");
		cvShowImage("Resulting Intensity Histogram", histDraw);
		cvReleaseImage(&histDraw);
		cvWaitKey(0);
		cvDestroyWindow("Resulting Intensity Histogram");
		cvDestroyWindow("Img-roi");
		cvDestroyWindow("Img-mask");
		cvDestroyWindow("Img");
#endif

	}
}

float Blob::getPorosity()
{
	float porosity = 0.0;
	if(internal_contours.size() > 0){
		float areaOfInternalContours = 0.0;
		for(int i = 0; i < internal_contours.size(); i++){
			areaOfInternalContours += internal_contours[i]->getArea();
		}
		porosity = areaOfInternalContours / getArea();
	}
	return porosity;
}


CvRect getCytoplasmBounds(CvSize tileSize, CvRect bounding_box,  int delta){
	CvRect cytoplasmBounds;
//	cout << "Cytobounding box = x -> "<<bounding_box.x << " y= " << bounding_box.y<< " width = "<< bounding_box.width << " height = "<<bounding_box.height<<endl;

	// limit box from moving outside x limits
	cytoplasmBounds.x = max(0, (bounding_box.x - delta));

	// calc actually change in X
	int leftMoveX = bounding_box.x - cytoplasmBounds.x;

	// limit box from move outside y limits
	cytoplasmBounds.y = max(0, (bounding_box.y - delta));

	// calc actually change in Y
	int upMoveY = bounding_box.y - cytoplasmBounds.y;


	if(bounding_box.x + bounding_box.width + delta > tileSize.width){
		cytoplasmBounds.width = tileSize.width - bounding_box.x + leftMoveX;
	}else{
		cytoplasmBounds.width = bounding_box.width + delta + leftMoveX;
	}

	if(bounding_box.y + bounding_box.height + delta > tileSize.height){
		cytoplasmBounds.height = tileSize.height - bounding_box.y + upMoveY;
	}else{
		cytoplasmBounds.height = bounding_box.height + delta + upMoveY;
	}

//	cout << "After- delta - Cytobounding box = x -> "<<cytoplasmBounds.x << " y= " << cytoplasmBounds.y<< " width = "<< cytoplasmBounds.width << " height = "<<cytoplasmBounds.height<<endl;
	return cytoplasmBounds;
}

IplImage *Blob::getCytoplasmMask(CvSize tileSize, int delta, CvPoint &offset)
{

	// This is the bounding_box of the nucleus
	CvRect bounding_box = this->getNonInclinedBoundingBox();
	IplImage *cytoplasmMaskRet = NULL;
	// Make sure that the bounding box is okay
	if(bounding_box.height != 0 && bounding_box.width != 0 ){
		// Calculate limit of the dilation for this nucleus --- this is another bounding box
		CvRect cytoplasmRect = getCytoplasmBounds(tileSize, bounding_box, delta);

		// Create mask within the same size as the bounding box
		IplImage *cytoplasmMask = cvCreateImage( cvSize(cytoplasmRect.width, cytoplasmRect.height), IPL_DEPTH_8U, 1);

		// Fill the image with background
		cvSetZero(cytoplasmMask);

		// The offset of the location of these contours in the original image to the location in
		// the mask that has the same dimensions as the bounding box. Thus, the contours can now be
		// draw from position (0,0) in a new image
//		offset.x = -bounding_box.x;
//		offset.y = -bounding_box.y;

		// However, the bounding box increases before the dilation, and the contours "nucleus" should be draw 
		// in the new image before the dilation. The new position of the contours takes into account how many
		// pixels the original nucleus bounding box increases in X and Y.

		// Calculate changes in X for the new bounding box: cytoplasmLeftMove 
		int cytoplasmLeftMove = bounding_box.x - cytoplasmRect.x + offsetInImage.x;
	
		// Calculate changes in Y for the new bounding box: cytoplasmLeftMove 
		int cytoplasmUpMove = bounding_box.y - cytoplasmRect.y + offsetInImage.y;

		// Update the offset with values calculated before
		CvPoint auxOffsetLocalCytoplasmMask;
		auxOffsetLocalCytoplasmMask.x = -bounding_box.x;
		auxOffsetLocalCytoplasmMask.x += cytoplasmLeftMove;
		auxOffsetLocalCytoplasmMask.y = -bounding_box.y;
		auxOffsetLocalCytoplasmMask.y += cytoplasmUpMove;


		// First draw the external contour
		cvDrawContours( cytoplasmMask, this->external_contour->getCl(), CV_RGB(255,255,255), CV_RGB(255,255,255),0, CV_FILLED, 8, auxOffsetLocalCytoplasmMask );

		// Fill each hole in the mask
		for(int i = 0; i < internal_contours.size(); i++){
			cvDrawContours( cytoplasmMask, internal_contours[i]->getCl(), CV_RGB(0,0,0), CV_RGB(0,0,0),0, CV_FILLED, 8, auxOffsetLocalCytoplasmMask );
		}

		static int disk_r8_17_17[289] = {
			0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
			0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,
			0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
			0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
			0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
			0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
			0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
			0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
			1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
			0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
			0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
			0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
			0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
			0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
			0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,
			0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,
			0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0
		};
	
		IplImage* cytoplasmDilated = cvCloneImage(cytoplasmMask);
	
		IplConvKernel *kernel_disk8 = cvCreateStructuringElementEx(17,17,8,8,CV_SHAPE_CUSTOM,disk_r8_17_17);
		cvDilate(cytoplasmMask, cytoplasmDilated, kernel_disk8);

//		cvSaveImage("cytoMask.tif", cytoplasmMask);
//		cvSaveImage("cytoOpencvDilated.tif", cytoplasmDilated);
//		cvSaveImage("cytoNucleusMaskCalc.tif", cytoplasmMask );

		IplImage* cytoplasmDilatedXor = cvCreateImage( cvSize(cytoplasmRect.width, cytoplasmRect.height), IPL_DEPTH_8U, 1);
		cvSetZero(cytoplasmDilatedXor);
	//	cvSetImageROI(cytoplasmDilatedXor, cvRect(2,2,cytoplasmRect.width,cytoplasmRect.height));
		//IplImage* cytoplasmDilatedXor = cvCloneImage(cytoplasmDilated);
		cvXor(cytoplasmDilated, cytoplasmMask, cytoplasmDilatedXor);
		cvResetImageROI(cytoplasmDilatedXor);
		
//		cvSaveImage("cytoOpencvDilateXor.tif", cytoplasmDilatedXor);
#ifdef VISUAL_DEBUG
		IplImage *curNucleusMask = DrawAuxiliar::DrawBlob(this,  CV_RGB(255,255,255),  CV_RGB(0,0,0));
		cvSaveImage("curNucleusMask.tif", curNucleusMask);

		cvNamedWindow("Nucleus - Press any key to continue!");
		cvShowImage("Nucleus - Press any key to continue!", cytoplasmMask);
		cvNamedWindow("Cytoplams - Press any key to continue!");
		cvShowImage("Cytoplams - Press any key to continue!", cytoplasmDilated);
		cvNamedWindow("CytoplamsXor - Press any key to continue!");
		cvShowImage("CytoplamsXor - Press any key to continue!", cytoplasmDilatedXor);


		cvWaitKey(0);
		cvDestroyWindow("Nucleus - Press any key to continue!");
		cvDestroyWindow("Cytoplams - Press any key to continue!");
		cvDestroyWindow("CytoplamsXor - Press any key to continue!");
#endif

		cvReleaseStructuringElement( &kernel_disk8 );
		cvReleaseImage(&cytoplasmDilated);
		cvReleaseImage(&cytoplasmMask);


		offset.x = cytoplasmRect.x;
		offset.y = cytoplasmRect.y;

		cytoplasmMaskRet = cytoplasmDilatedXor;
//		offset.x *= -1;
//		offset.y *= -1;
	}
	return cytoplasmMaskRet;
}


IplImage *Blob::getMask()
{
	// Bounding box has the position of the contour in the input image. It is already corrected due to shifts caused
	// by padding the input image.
	CvRect bounding_box = this->getNonInclinedBoundingBox();


	// Make sure that the bounding box is okay
	if(bounding_box.height != 0 && bounding_box.width != 0 && mask == NULL){

		// Create mask within the same size as the bounding box
		mask = cvCreateImage( cvSize(bounding_box.width, bounding_box.height), IPL_DEPTH_8U, 1);

//		cout << "Mask - width = "<< bounding_box.width << " height =  "<< bounding_box.height <<endl;
		// Fill the image with background
		cvSetZero(mask);

		// The offset of the location of these contours in the original image to the location in
		// the mask that has the same dimensions as the bounding box
		// Thus, from the bounding box location, I should subtract the bounding box location from the contours shift to 
		// get draw the contour in the current mask (0,0) axis.
		CvPoint offset;
		offset.x = -bounding_box.x + offsetInImage.x;
		offset.y = -bounding_box.y + offsetInImage.y;
	
		// First draw the external contour
		cvDrawContours( mask, this->external_contour->getCl(), CV_RGB(255,255,255), CV_RGB(255,255,255),0, CV_FILLED, 8, offset );

		// Fill each hole in the mask
		for(int i = 0; i < internal_contours.size(); i++){
			cvDrawContours( mask, internal_contours[i]->getCl(), CV_RGB(0,0,0), CV_RGB(0,0,0),0, CV_FILLED, 8, offset );

			// the draw contours includes the external line, and we remover it here
			cvDrawContours( mask, internal_contours[i]->getCl(), CV_RGB(255,255,255), CV_RGB(255,255,255),0, 1, 8, offset );
		}

//		cout <<endl<< "bounding_box.width = "<< bounding_box.width <<" mask->stepSize = "<< mask->widthStep <<endl;

#ifdef VISUAL_DEBUG
//		cvNamedWindow("Mask - Press any key to continue!");
//		cvShowImage("Mask - Press any key to continue!", mask);
//		cvWaitKey(0);
//		cvDestroyWindow("Mask - Press any key to continue!");
#endif

	}
	return mask;
}

// These are the set of pixel Intensity features


float Blob::getMeanIntensity(IplImage *img)
{

	float meanIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);

		float histBinValue = 0.0;
		// Iterates on the histogram of the input image and calculates the
		// summation of the image pixels intensity
		for(int i = 0; i < 256; i++){
			histBinValue = cvQueryHistValue_1D(intensity_hist, i);
			meanIntensity += i * histBinValue;
		}

		// Average the pixels intensity according to the number of pixels in the image
		meanIntensity = meanIntensity / this->getIntensityHistPoints();
	}
	return meanIntensity;
}

float Blob::getStdIntensity(IplImage *img)
{

	float stdIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);
		stdIntensity = Operators::calcStdFromHistogram(intensity_hist, 256);
	}
	return stdIntensity;
}

float Blob::getEnergyIntensity(IplImage *img)
{
	float energyIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);
		energyIntensity = Operators::calcEneryFromHistogram(intensity_hist, 256);
	}
	return energyIntensity;
}

float Blob::getEntropyIntensity(IplImage *img)
{
	float entropyIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);
		entropyIntensity = Operators::calcEntropyFromHistogram(intensity_hist, 256);
	}
	return entropyIntensity;
}

float Blob::getKurtosisIntensity(IplImage *img)
{
	float kurtosisIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);
		kurtosisIntensity = Operators::calcKurtosisFromHistogram(intensity_hist, 256);
	}
	return kurtosisIntensity;
}

float Blob::getSkewnessIntensity(IplImage *img)
{
	float skewnessIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);
		skewnessIntensity = Operators::calcSkewnessFromHistogram(intensity_hist, 256);
	}
	return skewnessIntensity;
}

unsigned int Blob::getMedianIntensity(IplImage *img)
{

	unsigned int median = 1000;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);

		median = Operators::calcMedianFromHistogram(intensity_hist, 256);

	}
	return median;
}

unsigned int Blob::getMinIntensity(IplImage *img)
{
	unsigned int min = 1000;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);

		// Iterates on the histogram of the input image until find the
		// first non-zero bin, which corresponds to the minimum pixel intensity
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(intensity_hist, i);
			if(histValue != 0){
				min = i;
				break;
			}
		}
	}
	return min;
}


unsigned int Blob::getMaxIntensity(IplImage *img)
{
	unsigned int max = 0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);

		// Iterates on the histogram of the input image from the highest bin to the smallest.
		// The first non-zero bin corresponds to the maximum pixel intensity
		for(int i=255;i>-1;i--){
			float histValue = cvQueryHistValue_1D(intensity_hist, i);
			if(histValue != 0){
				max = i;
				break;
			}
		}
	}
	return max;
}

unsigned int Blob::getFirstQuartileIntensity(IplImage *img)
{
	unsigned int firstQuartile = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);

		// Get size of the input image
		CvSize imgDims = cvGetSize(img);

		// used to count the accumulated number of pixels up to a given intensity
		unsigned int acc = 0;

		unsigned int firstQuartilePixelsValue = this->getIntensityHistPoints() / 4;

		// Iterates on bins until reach the minimum intensity (ith bin) that contains 25% of the pixels.
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(intensity_hist, i);
			acc += (int)histValue;

			if( acc >= firstQuartilePixelsValue ){
				firstQuartile = i;
				break;
			}
		}

	}
	return firstQuartile;
}


unsigned int Blob::getThirdQuartileIntensity(IplImage *img)
{
	unsigned int thirdQuartile = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcIntensityHistogram(img);

		// Get size of the input image
		CvSize imgDims = cvGetSize(img);

		// used to count the accumulated number of pixels up to a given intensity
		unsigned int acc = 0;

		unsigned int thirdQuartilePixelsValue = (unsigned int)((this->getIntensityHistPoints() / 4.0) * 3.0);


		// Iterates on bins from highest to lowest intensity until find the third quartile
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(intensity_hist, i);
			acc += (int)histValue;

			if( acc >= thirdQuartilePixelsValue ){
				thirdQuartile = i;
				break;
			}
		}
	}
	return thirdQuartile;
}

// These are the set of pixel Intensity features
void Blob::calcGradientHistogram(IplImage *img)
{
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ) && grad_hist == NULL){


		if(grad_hist != NULL){
			cvClearHist(grad_hist);
		}else{
			// This is the first time we're calculating the histogram, so
			// we have to create its structure.
			int numBins = 256;
			float range[] = {0, 256};
			float *ranges[] = { range };
			grad_hist = cvCreateHist(1, &numBins, CV_HIST_ARRAY, ranges, 1);
		}
//		// Region of interest is the same as the bounding box of the blob
//		IplImage *ROISubImage = getROISubImage(img);
//		Mat ROIMat(ROISubImage);
//
//		// copy data
//		Mat Res(ROISubImage, true);
//
//		// This is the data used to store the gradient results
//		IplImage* magImg = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_8U, 1);
//
//		// copy data
//		Mat magImageMat(magImg);
//
//
//		// This is a temporary structure required by the MorphologyEx operation we'll perform
//	//	IplImage* tempImg = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_8U, 1);
//
////		cvMorphologyEx(ROISubImage, magImg, tempImg, NULL, CV_MOP_GRADIENT);
//		Mat kernelCPU;
//		morphologyEx(ROIMat, magImageMat, MORPH_GRADIENT, kernelCPU, Point(-1,-1), 1);


//		CvRect blob_bounding_box = this->getNonInclinedBoundingBox();
//		blob_bounding_box.x += offsetInImage.x;
//		blob_bounding_box.y += offsetInImage.y;

		IplImage *ROISubImage = cvCreateImageHeader(cvSize(blob_bounding_box.width, blob_bounding_box.height), img->depth, img->nChannels);
		ROISubImage->origin = img->origin;
		ROISubImage->widthStep = img->widthStep;
		ROISubImage->imageData = img->imageData + blob_bounding_box.y * img->widthStep + blob_bounding_box.x * img->nChannels;

		cvCalcHist(&ROISubImage, grad_hist, 0, this->getMask());

		grad_hist_points = 0;
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(grad_hist, i);
			grad_hist_points += (unsigned int)histValue;
		}
		// Calculates the histogram in the input image for the pixels in the input mask
	//	cvCalcHist(&magImg, grad_hist, 0, this->getMask());


		cvReleaseImageHeader(&ROISubImage);

//		cout<< "Starting hist"<<endl;
//		for(int i=0;i<256;i++){
//				float histValue = cvQueryHistValue_1D(grad_hist, i);
//				intensity_hist_points += (unsigned int)histValue;
//				cout<< histValue<<endl;
//		}
////		cvAnd(ROISubImage, this->getMask(), ROISubImage);
////		cout << "NonZero = " << cvCountNonZero(ROISubImage) <<" histPoints = " << intensity_hist_points <<endl;
//		cout<< "End hist"<<endl;


		// This operation is available in opencv.. So it can be used as a ground truth
 		// Region of interest is the same as the bounding box of the blob
//		CvScalar avgScalar = cvAvg(magImg, this->getMask());
//		cout << "OpenCV Mean Magnitude = "<< avgScalar.val[0]<<endl;

//		cvReleaseImage(&tempImg);
//		cvReleaseImage(&magImg);


/*		// Images used to store temporary results with gradient values
		IplImage* drv = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_16S, 1);
		IplImage* drv32f = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_32F, 1);

		// Magnitude result image
		IplImage* mag = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_32F, 1);

		// 1)
		// Calculate Gradient in X dimension
		cvSobel(img, drv, 1, 0);

		// Convert to float point
		cvConvertScale(drv, drv32f);

		// Calculate square value for each element and store in drv32f
		cvSquareAcc( drv32f, mag);

		// 2)
		// Calculate Gradient in Y dimension
		cvSobel(img, drv, 0, 1);

		// Convert to float point
		cvConvertScale( drv, drv32f);

		// Calculate square value for each element and store in drv32f
		// mag = Dx^2 + Dy^2
		cvSquareAcc(drv32f, mag);

		// mag = ( Dx^2 + Dy^2 )^1/2
		cvbSqrt( (float*)(mag->imageData), (float*)(mag->imageData), mag->imageSize/sizeof(float));

		// Calculates the histogram in the input image for the pixels in the input mask
		cvCalcHist(&mag, grad_hist, 0, this->getMask());


		CvScalar avgScalar = cvAvg(mag, this->getMask());
		cout << "OpenCV Mean Magnitude = "<< avgScalar.val[0]<<endl;

		// Restore the Region of interest to the entire original image
		cvResetImageROI( img );

		// Release images used during this transformation
		cvReleaseImage( &drv );
		cvReleaseImage( &drv32f );
		cvReleaseImage( &mag );*/

#ifdef VISUAL_DEBUG
		IplImage* histDraw = DrawAuxiliar::DrawHistogram(grad_hist);
		cvNamedWindow("Gradient Hist");
		cvShowImage("Gradient Hist", histDraw);
		cvReleaseImage(&histDraw);
		cvWaitKey(0);
		cvDestroyWindow("Gradient Hist");
#endif

	}
}



double Blob::getMeanGradMagnitude(IplImage *img)
{
	double meanGradMagnitude = 0.0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculate the histogram
		this->calcGradientHistogram(img);

		float histBinValue = 0.0;
		unsigned int acc = 0;

		// Iterates on the histogram of the input image and calculates the
		// summation of the image pixels intensity
		for(int i = 0; i < 256; i++){
			histBinValue = cvQueryHistValue_1D(grad_hist, i);
			meanGradMagnitude += i * histBinValue;
			acc+=(int)histBinValue;
		}

		// Average the pixels intensity according to the number of pixels in the image
		meanGradMagnitude = meanGradMagnitude / this->getGradHistPoints();
	}
	return meanGradMagnitude;
}

float Blob::getStdGradMagnitude(IplImage *img)
{
	float std = 0.0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculate the histogram
		this->calcGradientHistogram(img);
		std = Operators::calcStdFromHistogram(grad_hist, 256);
	}
	return std;
}

float Blob::getEnergyGradMagnitude(IplImage *img){
	float energy = 0.0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculate the histogram
		this->calcGradientHistogram(img);
		energy = Operators::calcEneryFromHistogram(grad_hist, 256);
	}
	return energy;
}


float Blob::getEntropyGradMagnitude(IplImage *img){
	float entropy = 0.0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculate the histogram
		this->calcGradientHistogram(img);
		entropy = Operators::calcEntropyFromHistogram(grad_hist, 256);
	}
	return entropy;
}

float Blob::getKurtosisGradMagnitude(IplImage *img){
	float kurtosis = 0.0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculate the histogram
		this->calcGradientHistogram(img);
		kurtosis = Operators::calcKurtosisFromHistogram(grad_hist, 256);
	}
	return kurtosis;
}

float Blob::getSkewnessGradMagnitude(IplImage *img){
	float skewness = 0.0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculate the histogram
		this->calcGradientHistogram(img);
		skewness = Operators::calcSkewnessFromHistogram(grad_hist, 256);
	}
	return skewness;
}

unsigned int Blob::getMedianGradMagnitude(IplImage *img)
{
	unsigned int median = 1000;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcGradientHistogram(img);

		median = Operators::calcMedianFromHistogram(grad_hist, 256);

	}
	return median;
}



unsigned int Blob::getMinGradMagnitude(IplImage *img)
{
	unsigned int min = 1000;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcGradientHistogram(img);

		// Iterates on the histogram of the input image until find the
		// first non-zero bin, which corresponds to the minimum pixel intensity
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(grad_hist, i);
			if(histValue != 0){
				min = i;
				break;
			}
		}
	}
	return min;

}


unsigned int Blob::getMaxGradMagnitude(IplImage *img)
{
	unsigned int max = 0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcGradientHistogram(img);

		// Iterates on the histogram of the input image from the highest bin to the smallest.
		// The first non-zero bin corresponds to the maximum pixel intensity
		for(int i=255;i>-1;i--){
			float histValue = cvQueryHistValue_1D(grad_hist, i);
			if(histValue != 0){
				max = i;
				break;
			}
		}
	}
	return max;
}

unsigned int Blob::getFirstQuartileGradMagnitude(IplImage *img)
{
	unsigned int firstQuartile = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcGradientHistogram(img);

		// Get size of the input image
		CvSize imgDims = cvGetSize(img);

		// used to count the accumulated number of pixels up to a given intensity
		unsigned int acc = 0;

		unsigned int firstQuartilePixelsValue = this->getGradHistPoints() / 4;

		// Iterates on bins until reach the minimum intensity (ith bin) that contains 25% of the pixels.
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(grad_hist, i);
			acc +=(int)histValue;

			if( acc >= firstQuartilePixelsValue ){
				firstQuartile = i;
				break;
			}
		}

	}
	return firstQuartile;

}




unsigned int Blob::getThirdQuartileGradMagnitude(IplImage *img)
{
	unsigned int thirdQuartile = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{
		// Calculates the histogram
		this->calcGradientHistogram(img);

		// Get size of the input image
		CvSize imgDims = cvGetSize(img);

		// used to count the accumulated number of pixels up to a given intensity
		unsigned int acc = 0;

		unsigned int thirdQuartilePixelsValue = (unsigned int)((this->getGradHistPoints() / 4.0) * 3.0);

		// Iterates on bins from highest to lowest intensity until find the third quartile
		for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(grad_hist, i);
			acc +=(int)histValue;

			if( acc >= thirdQuartilePixelsValue ){
				thirdQuartile = i;
				break;
			}
		}
	}
	return thirdQuartile;
}


unsigned int Blob::getCannyArea(IplImage *img, double lowThresh, double highThresh, int apertureSize)
{
	unsigned int cannyArea = 0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img )){
		// Create edge image to store the result of applying
		// the Canny with the same size as the bounding box
		IplImage *edges = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_8U, 1);

		IplImage *roiImg = getROISubImage(img);

		// Make a copy only of the ROI in this image
		cvCopy(roiImg, edges, NULL);

		cvCanny(edges, edges, lowThresh, highThresh, apertureSize);

		cvAnd(edges, this->getMask(), edges);

		// Calculate the #white pixels
		cannyArea = cvCountNonZero(edges);


#ifdef VISUAL_DEBUG

//		cvNamedWindow("Canny - Input image");
//		cvShowImage("Canny - Input image", img);
		cvNamedWindow("Canny - Resulting image");
		cvShowImage("Canny - Resulting image", edges);
		cvWaitKey(0);
//		cvDestroyWindow("Canny - Input image");
		cvDestroyWindow("Canny - Resulting image");
#endif


		// release temporary image used to calculate Canny
		cvReleaseImage(&edges);
	}
	return cannyArea;
}
float Blob::getMeanCanny(IplImage *img, double lowThresh, double highThresh, int apertureSize)
{
	float meanCanny = 0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img )){
		// Create edge image to store the result of applying
		// the Canny with the same size as the bounding box
		IplImage *edges = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_8U, 1);

		IplImage *roiImg = getROISubImage(img);

		// Make a copy only of the ROI in this image
		cvCopy(roiImg, edges, NULL);

		cvAnd(edges, this->getMask(), edges);

		cvCanny(edges, edges, lowThresh, highThresh, apertureSize);

		// Calculate the #white pixels
		int cannyArea = cvCountNonZero(edges);

		meanCanny = cannyArea / getArea();

#ifdef VISUAL_DEBUG

		cvNamedWindow("Canny - Input image");
		cvShowImage("Canny - Input image", img);
		cvNamedWindow("Canny - Resulting image");
		cvShowImage("Canny - Resulting image", edges);
		cvWaitKey(0);
		cvDestroyWindow("Canny - Input image");
		cvDestroyWindow("Canny - Resulting image");
#endif


		// release temporary image used to calculate Canny
		cvReleaseImage(&edges);
	}

	return meanCanny;
}

unsigned int Blob::getSobelArea(IplImage *img, int xorder, int yorder, int apertureSize)
{
	unsigned int sobelArea = 0;
	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img )){
		// Create edge image to store the result of applying
		// the Sobel with the same size as the bounding box
		IplImage *edges = cvCreateImage( cvSize(blob_bounding_box.width, blob_bounding_box.height), IPL_DEPTH_16S, 1);

		// Region of interest is the same as the bounding box of the blob
		IplImage *roiImg = getROISubImage(img);
//		cvSetImageROI(img, blob_bounding_box);

		cvSobel(roiImg, edges, xorder, yorder, apertureSize);

		// Calculate the #white pixels
		sobelArea = cvCountNonZero(edges);

#ifdef VISUAL_DEBUG

//		cvNamedWindow("Sobel - Input image");
//		cvShowImage("Sobel - Input image", img);
//		cvNamedWindow("Sobel - Resulting image");
//		cvShowImage("Sobel - Resulting image", edges);
//		cvWaitKey(0);
//		cvDestroyWindow("Sobel - Input image");
//		cvDestroyWindow("Sobel - Resulting image");
#endif

		// release temporary image used to calculate Sobel
		cvReleaseImage(&edges);

		// Set the ROI to the entire original image
//		cvResetImageROI(img);

	}
	return sobelArea;
}


float Blob::getReflectionSymmetry()
{
	float reflectionSymmetry  = 0.0;
	IplImage * mask = this->getMask();
	if(mask){
		IplImage* flipMask = cvCreateImage( cvGetSize(mask), 8, 1);
		cvFlip(mask, flipMask, 1);

#ifdef VISUAL_DEBUG
		cvNamedWindow("ReflectionSymmetry - Symmetric of mask");
		cvShowImage("ReflectionSymmetry - Symmetric of mask", flipMask);
#endif

		cvAnd(mask, flipMask, flipMask);

#ifdef VISUAL_DEBUG
		cvNamedWindow("CommonArea");
		cvShowImage("CommonArea", flipMask);
		cvWaitKey(0);
		cvDestroyWindow("ReflectionSymmetry - Symmetric of mask");
		cvDestroyWindow("CommonArea");
#endif
		unsigned int pixelsInMaskAndSymmetric = cvCountNonZero(flipMask);
		unsigned int pixelsMask = cvCountNonZero(mask);

		reflectionSymmetry = (float)pixelsInMaskAndSymmetric / (float)pixelsMask;


		cvReleaseImage(&flipMask);
	}
	return reflectionSymmetry;
}

float Blob::getBendingEnery()
{
	return external_contour->getBendingEnergy();
}


IplImage *Blob::getROISubImage(IplImage *img)
{
	if(ROISubImage == NULL){
		CvRect blob_bounding_box = this->getNonInclinedBoundingBox();
//		blob_bounding_box.x += offsetInImage.x;
//		blob_bounding_box.y += offsetInImage.y;

		ROISubImage = cvCreateImageHeader(cvSize(blob_bounding_box.width, blob_bounding_box.height), img->depth, img->nChannels);
		ROISubImage->origin = img->origin;
		ROISubImage->widthStep = img->widthStep;
		ROISubImage->imageData = img->imageData + blob_bounding_box.y * img->widthStep + blob_bounding_box.x * img->nChannels;
	}
	return ROISubImage;
}

void Blob::releaseROISubImage(){
	if(ROISubImage != NULL){
		cvReleaseImageHeader(&ROISubImage);
		ROISubImage = NULL;
	}
}

void Blob::resetInternalData(){
	this->clearIntensityData();
	this->clearGradientData();
	this->clearCoocPropsData();
	this->releaseROISubImage();
}

void Blob::setMaskInUserDataRegion(char *data)
{
	CvRect bounding_box = this->getNonInclinedBoundingBox();

	isImage = false;

	// Init blobs mask;
	mask = cvCreateImageHeader(cvSize(bounding_box.width, bounding_box.height), IPL_DEPTH_8U, 1);
	cvSetData(mask, data, bounding_box.width);


	// Fill the image with background
	cvSetZero(mask);

	// The offset of the location of these contours in the original image to the location in
	// the mask that has the same dimensions as the bounding box
	CvPoint offset;
	offset.x = -bounding_box.x + offsetInImage.x;
	offset.y = -bounding_box.y + offsetInImage.y;

	// First draw the external contour
	cvDrawContours( mask, external_contour->getCl(), CV_RGB(255,255,255), CV_RGB(1,1,1),0, CV_FILLED, 8, offset );

	// Fill each hole in the mask
	for(int j = 0; j < internal_contours.size(); j++){
		cvDrawContours( mask, internal_contours[j]->getCl(), CV_RGB(0,0,0), CV_RGB(0,0,0),0, CV_FILLED, 8, offset );
	}
//	cvSaveImage("oneNucleusOpencvMaskDraw.tif", mask);
//
//	cout << "Count non zero "<< cvCountNonZero(mask) << endl;
/*	char test=255;
	cout << "Count non zero "<< cvCountNonZero(mask)<<" test= "<< (int)test<<endl;
	cout <<endl<< "bounding_box.width = "<< bounding_box.width <<" mask->stepSize = "<< mask->widthStep <<endl;
	cvNamedWindow("Mask - Press any key to continue!");
	cvShowImage("Mask - Press any key to continue!", mask);
	cvWaitKey(0);
	cvDestroyWindow("Mask - Press any key to continue!");*/
}

float Blob::getClusterProminence() const
{
    return clusterProminence;
}

float Blob::getClusterShade() const
{
    return clusterShade;
}

float Blob::getEnergy() const
{
    return energy;
}

float Blob::getEntropy() const
{
    return entropy;
}

float Blob::getHomogeneity() const
{
    return homogeneity;
}

float Blob::getInertia() const
{
    return inertia;
}

float Blob::getMaximumProb() const
{
    return maximumProb;
}

void Blob::setClusterProminence(float clusterProminence)
{
    this->clusterProminence = clusterProminence;
}

void Blob::setClusterShade(float clusterShade)
{
    this->clusterShade = clusterShade;
}

void Blob::setEnergy(float energy)
{
    this->energy = energy;
}

void Blob::setEntropy(float entropy)
{
    this->entropy = entropy;
}

void Blob::setHomogeneity(float homogeneity)
{
    this->homogeneity = homogeneity;
}

void Blob::setInertia(float inertia)
{
    this->inertia = inertia;
}

void Blob::printIntensityHistogram(IplImage *img)
{
	double meanIntensity = 0;

	// Get blob bounding box that is used to set the
	// region of interest in the input image
	CvRect blob_bounding_box = this->getNonInclinedBoundingBox();

	// if parameters are okay, so calculate the mean intensity
	if (blob_bounding_box.height != 0 && blob_bounding_box.width != 0 && CV_IS_IMAGE( img ))
	{

		// Calculates the histogram
		this->calcIntensityHistogram(img);

		int histBinValue = 0;
		for(int i = 0; i < 256; i++){
			histBinValue = (int)cvQueryHistValue_1D(intensity_hist, i);
			cout << i<<":"<<histBinValue<< " ";
		}
		for(int i = 0; i < 256; i++){
			histBinValue = (int)cvQueryHistValue_1D(intensity_hist, i);
			cout <<histBinValue<<endl;
		}
		cout<<endl;

	}
}

void Blob::setMaximumProb(float maximumProb)
{
    this->maximumProb = maximumProb;
}

float Blob::getNonInclinedBoundingBoxArea()
{
	CvRect bounding = external_contour->getNonInclinedBoundingBox();
	return (bounding.width * bounding.height);
}

void Blob::doCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask)
{

	if(coocMatrix[angle] == NULL){
		coocMatrix[angle] = (unsigned int *)calloc(coocSize * coocSize,  sizeof(unsigned int));
		// TODO: check memory allocation return
	}else{
		// It has been calculated before, so clean it up.
		memset(coocMatrix[angle], 0, coocSize * coocSize * sizeof(unsigned int));
		coocMatrixCount[angle] = 0;
	}

	IplImage *imgRoi = getROISubImage(inImage);
	// allocate memory for the normalized image
	float *normImg = (float*)malloc(sizeof(float)*imgRoi->height*imgRoi->width);

	if(normImg == NULL){
		cout << "ComputeCoocMatrix: Could not allocate temporary normalized image" <<endl;
		exit(1);
	}

	//compute normalized image
	float slope = ((float)coocSize-1.0) / 255.0;
	float intercept = 1.0 ;
	for(int i=0; i<imgRoi->height; i++){
		for(int j =0; j < imgRoi->width; j++){
			CvScalar elementIJ = cvGet2D(imgRoi, i, j);
			normImg[i*imgRoi->width + j] = round((slope*(float)elementIJ.val[0] + intercept));
		}
	}

	switch(angle){

		case Constant::ANGLE_0:
//			cout << "CalcCooc ANGLE_0: height="<< imgRoi->height<< " width="<< imgRoi->width <<endl;
			//build co-occurrence matrix
			for (int i=0; i<imgRoi->height; i++){
				int offSet = i*imgRoi->width;
				for(int j=0; j<imgRoi->width-1; j++){
					if(((normImg[offSet+j])-1) < coocSize && ((normImg[offSet+j+1])-1) < coocSize){
						if(useMask){
							int jMaskValue = (int)(cvGet2D(mask, i, j)).val[0];
							int j1MaskValue = (int)(cvGet2D(mask, i, j+1)).val[0];
							if(jMaskValue == 0 || j1MaskValue == 0) continue;
						}
						unsigned int coocAddress = (unsigned int )((normImg[offSet+j])-1) * coocSize;
						coocAddress += (int)(normImg[offSet+j+1]-1);
						coocMatrix[angle][coocAddress]++;
					}
				}
			}
			break;

		case Constant::ANGLE_45:
			//build co-occurrence matrix
			for (int i=0; i<imgRoi->height-1; i++){
				int offSetI = i*imgRoi->width;
				int offSetI2 = (i+1)*imgRoi->width;
				for(int j=0; j<imgRoi->width-1; j++){
				   if(useMask){
						int jMaskValue = (int)(cvGet2D(mask, i, j+1)).val[0];
						int j1MaskValue = (int)(cvGet2D(mask, i+1, j)).val[0];
						if(jMaskValue == 0 || j1MaskValue == 0) continue;
					}
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j])-1) * coocSize;
					coocAddress += (int)(normImg[offSetI +j +1 ] -1);
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;
		case Constant::ANGLE_90:
			//build co-occurrence matrix
			for (int i=0; i<imgRoi->height-1; i++){
				int offSetI = i*imgRoi->width;
				int offSetI2 = (i+1)*imgRoi->width;
				for(int j=0; j<imgRoi->width; j++){
					if(useMask){
						int jMaskValue = (int)(cvGet2D(mask, i, j)).val[0];
						int j1MaskValue = (int)(cvGet2D(mask, i+1, j)).val[0];
						if(jMaskValue == 0 || j1MaskValue == 0) continue;
					}
					unsigned int coocAddress = (unsigned int )((normImg[offSetI2+j])-1) * coocSize;
					coocAddress += (int)(normImg[offSetI + j ] -1);
					coocMatrix[angle][coocAddress]++;
				}
			}
			break;

		case Constant::ANGLE_135:
			//build co-occurrence matrix
			for (int i=0; i<imgRoi->height-1; i++){
				int offSetI = i*imgRoi->width;
				int offSetI2 = (i+1)*imgRoi->width;
				for(int j=0; j<imgRoi->width-1; j++){
					if(useMask){
						int jMaskValue =(int) (cvGet2D(mask, i, j)).val[0];
						int j1MaskValue = (int)(cvGet2D(mask, i+1, j+1)).val[0];
						if(jMaskValue == 0 || j1MaskValue == 0) continue;
					}
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




void Blob::printCoocMatrix(unsigned int angle)
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

float Blob::inertiaFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float inertia = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	inertia = Operators::inertiaFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return inertia;
}

float Blob::energyFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float energy = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	energy = Operators::energyFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return energy;
}


float Blob::entropyFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float entropy = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	entropy = Operators::entropyFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return entropy;
}


float Blob::homogeneityFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float homogeneity = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	homogeneity = Operators::homogeneityFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return homogeneity;
}

float Blob::maximumProbabilityFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float maximumProbability = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	maximumProbability = Operators::maximumProbabilityFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return maximumProbability;
}

float Blob::clusterShadeFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float clusterShade = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	clusterShade = Operators::clusterShadeFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return clusterShade;
}

float Blob::clusterProminenceFromCoocMatrix(unsigned int angle, IplImage *inImage, bool useMask, bool reuseItermediaryResults)
{
	float clusterProminence = 0.0;
	// If the coocMatrix has not been computed yet for the given angle, or it is not allowed to reused
	// intermediary results, then compute coocmatrix again for the given angle
	if(reuseItermediaryResults != true || coocMatrix[angle] == NULL){
		doCoocMatrix(angle, inImage, useMask);
	}
	clusterProminence = Operators::clusterProminenceFromCoocMatrix(coocMatrix[angle], coocSize, coocMatrixCount[angle]);
	return clusterProminence;
}




