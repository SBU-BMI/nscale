#include "highgui.h"
#include "cv.h"
#include "cxcore.h"
#include <stdio.h>
#include <math.h>
#include <iostream>


using namespace std;

///// List 1 ///////
float getContourArea(CvSeq *c_l){
	float area = -1.0;
	area = fabs( cvContourArea(c_l) );
	return area;
}

float getContourPerimeter(CvSeq *c){
	return cvArcLength(c, CV_WHOLE_SEQ, 1);
}
// Equivalent diameter = (4*area/Pi)^1/2
float getContourEquivalentDiameter(CvSeq *c, float area=-1.0){
	float ED = -1.0;
	if(area == -1){
		area = getContourArea(c);
	}
	ED = sqrtf((4*area)/CV_PI);
	return ED;
}

float getContourSphericity(CvSeq *c){
	return -1.0;
}

float getContourFractalDimension(CvSeq *c){
	return -1.0;
}

float getContourPerimenterCurvature(CvSeq *c){
	return -1.0;
}

float getContourInertiaShape(CvSeq *c){
	return -1.0;
}


///// List 2 ///////

// Blobs library implements its own fitting. Check what's the difference
float getContourMajorAxisLength(CvSeq *c){
		float majorAxis = -1.0;
		CvBox2D elipse = cvFitEllipse2(c);
		if(elipse.size.width > elipse.size.height){
			majorAxis = elipse.size.width;
		}else{
			majorAxis = elipse.size.height;
		}
		return majorAxis;
}

float getContourMinorAxisLength(CvSeq *c){
		float minorAxis = -1.0;
		CvBox2D elipse = cvFitEllipse2(c);
		if(elipse.size.width > elipse.size.height){
			minorAxis = elipse.size.height;
		}else{
			minorAxis = elipse.size.width;
		}
		return minorAxis;
}

float getContourEllipseAngle(CvSeq *c){
		CvBox2D elipse = cvFitEllipse2(c);
		return elipse.angle;
}

// extent - (area of the blob) / (area of the minimum bounding box)
float getContourExtent(CvSeq *c){
	float extent = -1.0;
	float contourArea = getContourArea(c);
	CvBox2D minRect = cvMinAreaRect2(c);
	float minBoxArea = minRect.size.width * minRect.size.height;
	extent = contourArea / minBoxArea;
#ifdef DEBUG
	cout << "Contour area - "<< contourArea << " minRectArea = " << minBoxArea << " extent = "<< extent<<endl;
	
#endif
	return extent;

}

// Eccentricity - foci/major axis - foci = ((Major Axis/2)^2 - (Minor Axis/2)^2)^1/2
float getContourEccentricity(CvSeq *c){
	float eccentricity = -10.0;
	float majorAxis = getContourMajorAxisLength(c);
	float minorAxis = getContourMinorAxisLength(c);
	
	float foci = pow((majorAxis/2),2) - pow((minorAxis/2), 2);

	if(foci > 0)
		foci =  sqrtf(foci);

	if(foci < 0) foci = 0.0;

	eccentricity = foci / getContourMajorAxisLength(c);
//#ifdef DEBUG
	cout << "Eccentricity = "<< eccentricity<<endl<< " majorAxis ="<< majorAxis << " minorAxis = "<< minorAxis << " foci = "<<foci<<endl;
//#endif
	return eccentricity;
}

// Circularity = (perimeter * perimeter)/area - 4.*PI
float getContourCircularity(CvSeq *c){
	float perimeter = getContourPerimeter(c);
	float area = getContourArea(c);
#ifdef DEBUG
	cout <<endl<< "Circularity calculation"<<endl;
	cout << "perimeter - " << perimeter << " area - "<< area<<endl;
#endif

	float circularity = (perimeter * perimeter)/area;
	circularity -= 4.*CV_PI;
	return circularity;
}

float getContourConvexArea(CvSeq *c){
	float hullArea = -1.0;
	CvSeq *convexHullContour = cvConvexHull2(c);
	if(convexHullContour){
		// convert null into a seq_conutor
		CvMemStorage* g_storage = cvCreateMemStorage();
		CvPoint pt;

		int hullcount = convexHullContour->total;

		CvSeq* ptseq  = cvCreateSeq( CV_SEQ_CONTOUR|CV_32SC2, sizeof(CvContour), sizeof(CvPoint),  g_storage );

		for(int  i = 0; i < hullcount; i++){
			CvPoint pt = **CV_GET_SEQ_ELEM( CvPoint*, convexHullContour, i );
			cvSeqPush( ptseq , &pt );
		}

		// After calculate the convexe hull area with cvContourArea( ptseq  ) so:
		hullArea =  cvContourArea( ptseq) ;

		cvClearMemStorage(g_storage);
	}
	return hullArea;	
}
// Solidity - (Area of blob)/(ConvexArea)[1,5]
float getContourSolidity(CvSeq *c){
	float solidity = -1.0;
	float area = getContourArea(c);
	float convexArea = getContourConvexArea(c);
	solidity = area / convexArea;
	return solidity;
}

// Deficiency - (convexArea-Area)/Area [5] 
float getContourDeficiency(CvSeq *c){
	float deficiency = -1.0;
	float area = getContourArea(c);
	float convexArea = getContourConvexArea(c);
	deficiency = (convexArea-area)/area;
	return deficiency;
}

// Compactness - 4PI*Area/Perimeter^2 [5]
float getContourCompactness(CvSeq *c){
	float compactness = -1.0;
	float area = getContourArea(c);
	float perimeter = getContourPerimeter(c);
	compactness = 4*CV_PI*area/pow(perimeter,2);
	return compactness;
}

float getContourFilledArea(CvSeq *c){
	return fabs( cvContourArea(c) );
}

//// E ( E = C - H, where C is the # of connected components and H is the # of holes
//float getCountorEulerNumber(){
//	CvSeq *firstContour = NULL;
//	CvMemStorage* storage = cvCreateMemStorage(0);
//	int holes = 0;
//
//	cvFindContours(img, storage, &firstContour, sizeof(CvChain),
//			CV_RETR_CCOMP, CV_CHAIN_CODE);
//
//	if(firstContour != NULL)
//	{
//		CvSeq *aux = firstContour;
//		if(aux->v_next)
//		{
//			holes++;
//			aux = aux->v_next;
//		}
//
//		while(aux->h_next)
//		{
//			aux = aux->h_next;
//			holes++;
//		}
//	}
//
//}
// perimeter curvature[5]
float getCountorPerimeterCurvature(CvSeq *c){


}

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
		CV_RETR_LIST,
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
		printf("Contour #%d - perimeter=%lf - area=%lf ED=%lf ",  n, getContourPerimeter(c), getContourArea(c), getContourEquivalentDiameter(c));
		if(c->total >= 6){
			printf("MajorAxisLength=%lf MinorAxisLength=%lf Orientation=%lf", getContourMajorAxisLength(c), getContourMinorAxisLength(c), getContourEllipseAngle(c));
			printf(" Circularity = %lf Extent = %lf Eccentricity = %lf", getContourCircularity(c), getContourExtent(c), getContourEccentricity(c));
			printf(" ConvexArea = %lf Solidity = %lf Deficiency = %lf", getContourConvexArea(c), getContourSolidity(c), getContourDeficiency(c));
			printf(" Compactness = %lf FilledArea = %lf\n", getContourCompactness(c), getContourFilledArea(c));
		}else{
			printf("\n");
		}
		cvShowImage(argv[0], img_8uc3);
		cvWaitKey(0);
		n++;
	}

	printf("Finished all contours.\n");
	cvCvtColor(img_8uc1, img_8uc3, CV_GRAY2BGR);
	cvShowImage(argv[0], img_8uc3);
	cvWaitKey(0);
}
