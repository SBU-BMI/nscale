/*
 * DrawAuxiliar.cpp
 *
 *  Created on: Jun 29, 2011
 *      Author: george
 */

#include "DrawAuxiliar.h"

DrawAuxiliar::DrawAuxiliar() {
	// TODO Auto-generated constructor stub

}

DrawAuxiliar::~DrawAuxiliar() {
	// TODO Auto-generated destructor stub
}
IplImage *DrawAuxiliar::DrawHistogram(unsigned int *hist, float scaleX, float scaleY)
{
	float histMax = 0;
	for(int i = 0; i < 255;i++){
		if(hist[i] > histMax) histMax = hist[i];
	}

	IplImage* imgHist = cvCreateImage(cvSize((int)(256*scaleX), (int)(64*scaleY)), 8 ,1);
	cvZero(imgHist);
	for(int i=0;i<255;i++)
	{
		float histValue = hist[i];
		float nextValue = hist[i+1];

		CvPoint pt1 = cvPoint((int)(i*scaleX), (int)(64*scaleY));
		CvPoint pt2 = cvPoint((int)(i*scaleX+scaleX), (int)(64*scaleY));
		CvPoint pt3 = cvPoint((int)(i*scaleX+scaleX), (int)((64-nextValue*64/histMax)*scaleY));
		CvPoint pt4 = cvPoint((int)(i*scaleX), (int)((64-histValue*64/histMax)*scaleY));

		int numPts = 5;
		CvPoint pts[] = {pt1, pt2, pt3, pt4, pt1};

		cvFillConvexPoly(imgHist, pts, numPts, cvScalar(255));
	}

	return imgHist;
}

IplImage *DrawAuxiliar::DrawHistogram(CvHistogram *hist, float scaleX, float scaleY)
{
	float histMax = 0;
	cvGetMinMaxHistValue(hist, 0, &histMax, 0, 0);
	IplImage* imgHist = cvCreateImage(cvSize((int)(256*scaleX), (int)(64*scaleY) ), 8 ,1);
	cvZero(imgHist);
	for(int i=0;i<255;i++)
	{
		float histValue = cvQueryHistValue_1D(hist, i);
		float nextValue = cvQueryHistValue_1D(hist, i+1);

		CvPoint pt1 = cvPoint((int)(i*scaleX), (int)(64*scaleY));
		CvPoint pt2 = cvPoint((int)(i*scaleX+scaleX), (int)(64*scaleY));
		CvPoint pt3 = cvPoint((int)(i*scaleX+scaleX), (int)((64-nextValue*64/histMax)*scaleY) );
		CvPoint pt4 = cvPoint((int)(i*scaleX), (int)((64-histValue*64/histMax)*scaleY) );

		int numPts = 5;
		CvPoint pts[] = {pt1, pt2, pt3, pt4, pt1};

		cvFillConvexPoly(imgHist, pts, numPts, cvScalar(255));
	}

	return imgHist;
}



IplImage *DrawAuxiliar::DrawBlob(Blob *blobToPrint, CvScalar external, CvScalar holes)
{
	CvRect bounding_box = blobToPrint->getNonInclinedBoundingBox();
	IplImage* printedBlob = NULL;

	// Make sure that the bounding box is okay
	if(bounding_box.height != 0 && bounding_box.width != 0){
		// Create mask within the same size as the bounding box
		printedBlob = cvCreateImage( cvSize(bounding_box.width, bounding_box.height), IPL_DEPTH_8U, 3);

		// Fill the image with background
		cvSetZero(printedBlob);

		// The offset of the location of these contours in the original image to the location in
		// the mask that has the same dimensions as the bounding box
		CvPoint offset;
		offset.x = -bounding_box.x;
		offset.y = -bounding_box.y;

		// First draw the external contour
		cvDrawContours( printedBlob, blobToPrint->external_contour->getCl(), external, external,0, CV_FILLED, 8, offset );

		// Fill each hole in the mask
		for(int i = 0; i < blobToPrint->internal_contours.size(); i++){

			cvDrawContours( printedBlob, blobToPrint->internal_contours[i]->getCl(), holes, holes,0, CV_FILLED, 8, offset );

		}

	}
	return printedBlob;
}

void DrawAuxiliar::DrawBlob(IplImage* printBlob, Blob *blobToPrint, CvScalar external, CvScalar holes)
{

	// First draw the external contour
//	cvDrawContours( printBlob, blobToPrint->external_contour->getCl(), external, external,0, CV_FILLED);

	CvPoint offset = blobToPrint->getOffsetInImage();
	cvDrawContours( printBlob, blobToPrint->external_contour->getCl(), external, external,0, 1, 8, offset);
	// Fill each hole in the mask
	for(int i = 0; i < blobToPrint->internal_contours.size(); i++){
		cvDrawContours( printBlob, blobToPrint->internal_contours[i]->getCl(), holes, holes,0, 1, 8, offset);

	}

}

