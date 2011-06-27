/*
 * Contour.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#include "Contour.h"

// This method is private and not used;
Contour::Contour() {
}

Contour::Contour(CvSeq *c_l)
{
	// Initialize memory block used to store Contour structs
	 this->self_storage = cvCreateMemStorage();

	 // Copy the given list sequence of objects defining the Contour
	 this->c_l = cvCloneSeq(c_l, this->self_storage);

	 // Initialize contour features, which is used to
	 // check whether it was calculated before or not
	 this->area = -1.0;
	 this->circularity = -1.0;
	 this->compacteness = -1.0;
	 this->convexArea = -1.0;
	 this->convexDeficiency = -1.0;
	 this->eccentricity = -1.0;
	 this->ellipseAngle = -1.0;
	 this->equivalentDiameter = -1.0;
	 this->extent = -1.0;
	 this->majorAxisLength = -1.0;
	 this->minorAxisLength = -1.0;
	 this->perimeter = -1.0;
	 this->perimeterCurvature = -1.0;
	 this->solidity = -1.0;
	 this->sphericity = -1.0;
	 this->min_fitting_ellipse.size.width = -1.0;
	 this->min_bounding_box.size.width = -1.0;
	 this->convexHull = NULL;
}

Contour::~Contour() {
	cvClearMemStorage(this->self_storage);
}

float Contour::getArea()
{
	if(area == -1.0){
		area = fabs( cvContourArea(c_l) );
	}
    return area;
}

float Contour::getPerimeter()
{
	if(perimeter == -1.0){
		perimeter =  cvArcLength(c_l, CV_WHOLE_SEQ, 1);
	}
    return perimeter;
}
float Contour::getEquivalentDiameter()
{
	if(equivalentDiameter == -1.0){
		// Call getArea instead of using the ``area'' directly
		// to assert that the area is correctly initialized
		equivalentDiameter = 2* sqrtf(CV_PI/getArea());
	}
	return equivalentDiameter;
}

float Contour::getCompacteness()
{
	if(compacteness == -1.0){
		compacteness = 4*CV_PI*getArea()/pow(getPerimeter(),2);
	}
    return compacteness;
}

float Contour::getMajorAxisLength()
{
	if(majorAxisLength == -1.0){
		// Do we have to fit the ellipse?
		if(min_fitting_ellipse.size.width == -1.0){
			min_fitting_ellipse = cvFitEllipse2(c_l);
		}
		if(min_fitting_ellipse.size.width > min_fitting_ellipse.size.height){
			majorAxisLength = min_fitting_ellipse.size.width;
		}else{
			majorAxisLength = min_fitting_ellipse.size.height;
		}
	}
    return majorAxisLength;
}

float Contour::getMinorAxisLength()
{
	if(minorAxisLength == -1.0){
		// Do we have to fit the elipse?
		if(min_fitting_ellipse.size.width == -1.0){
			min_fitting_ellipse = cvFitEllipse2(c_l);
		}
		if(min_fitting_ellipse.size.width > min_fitting_ellipse.size.height){
			minorAxisLength = min_fitting_ellipse.size.height;
		}else{
			minorAxisLength = min_fitting_ellipse.size.width;
		}
	}
    return minorAxisLength;
}

float Contour::getOrientation()
{
	if(ellipseAngle == -1.0){
		// Do we have to fit the elipse?
		if(min_fitting_ellipse.size.width == -1.0){
			min_fitting_ellipse = cvFitEllipse2(c_l);
		}
		ellipseAngle = min_fitting_ellipse.angle;
		if(ellipseAngle > 90){
			ellipseAngle -= 180;
		}
	}
    return ellipseAngle;
}

float Contour::getEccentricity()
{
	if(eccentricity == -1.0){
		float foci = pow((getMajorAxisLength()/2),2) - pow((getMinorAxisLength()/2), 2);

		if(foci > 0){
			foci =  sqrtf(foci);
		}else{
			foci = 0.0;
		}

		eccentricity = foci / getMajorAxisLength();
#ifdef DEBUG
		cout << "Eccentricity = "<< eccentricity<<endl<< " majorAxis ="<< getMajorAxisLength() << " minorAxis = "<< getMinorAxisLength() << " foci = "<<foci<<endl;
#endif
	}
    return eccentricity;
}

float Contour::getCircularity()
{
	if(circularity == -1.0){
		circularity = (pow(getPerimeter(),2)/getArea()) - (4.*CV_PI);

#ifdef DEBUG
		cout <<endl<< "Circularity calculation"<<endl;
		cout << "Perimeter - " << getPerimeter() << " Area - "<< getArea() <<endl;
#endif

	}
    return circularity;
}

float Contour::getConvexArea()
{
	if(convexArea == -1.0){
		// Check whether the convex hull is already calculated
		if(convexHull == NULL){
			convexHull = cvConvexHull2(c_l);
		}

		// Local storage to create a Sequence of points with the given convex hull
		CvMemStorage* local_storage = cvCreateMemStorage();
		CvPoint pt;

		// Convert convex hull into a sequence of points to calculate area
		int hullcount = convexHull->total;
		CvSeq* ptseq  = cvCreateSeq( CV_SEQ_CONTOUR|CV_32SC2, sizeof(CvContour), sizeof(CvPoint),  local_storage );

		for(int  i = 0; i < hullcount; i++){
			CvPoint pt = **CV_GET_SEQ_ELEM( CvPoint*, convexHull, i );
			cvSeqPush( ptseq , &pt );
		}

		// After calculating the convex hull area with cvContourArea( ptseq  ) so:
		convexArea =  cvContourArea(ptseq) ;

		// Release memory used to allocate points describing the convex hull
		cvClearMemStorage(local_storage);
	}
    return convexArea;
}

float Contour::getSolidity()
{
	if(solidity == -1.0){
		solidity = getArea() / getConvexArea();
	}
	return solidity;
}

float Contour::getConvexDeficiency()
{
	if(convexDeficiency = -1.0){
		convexDeficiency = (getConvexArea()-getArea())/getArea();
	}
    return convexDeficiency;
}


float Contour::getExtent()
{

	if(extent == -1.0){
		// Verify if minimum bounding box containing the contour
		// is calculated, and calculate it if necessary
		if(min_bounding_box.size.width == -1.0){
			min_bounding_box = cvMinAreaRect2(c_l);
		}

		// Calculate area of the given bounding box
		float minBoxArea = min_bounding_box.size.width * min_bounding_box.size.height;
		extent = getArea() / minBoxArea;

#ifdef DEBUG
		cout << "Contour area - "<< getArea() << " minRectArea = " << minBoxArea << " extent = "<< extent<<endl;
#endif
	}
	return extent;
}

float Contour::getMinBoundingBoxArea()
{
	// Verify if minimum bounding box containing the contour
	// is calculated, and calculate it if necessary
	if(min_bounding_box.size.width == -1.0){
		min_bounding_box = cvMinAreaRect2(c_l);
	}

	// Calculate area of the given bounding box
	return (min_bounding_box.size.width * min_bounding_box.size.height);
}


float Contour::getBoundingBoxWidth()
{
	// Verify if minimum bounding box containing the contour
	// is calculated, and calculate it if necessary
	if(min_bounding_box.size.width == -1.0){
		min_bounding_box = cvMinAreaRect2(c_l);
	}
	return min_bounding_box.size.width;
}

float Contour::getBoundingBoxHeight()
{
	// Verify if minimum bounding box containing the contour
	// is calculated, and calculate it if necessary
	if(min_bounding_box.size.width == -1.0){
		min_bounding_box = cvMinAreaRect2(c_l);
	}
	return min_bounding_box.size.height;
}

float Contour::getEllipticity()
{
	return ((getMajorAxisLength()-getMinorAxisLength())/getMajorAxisLength());
}

float Contour::getSphericity()
{
	// TODO: implement
    return sphericity;
}



float Contour::getPerimeterCurvature()
{
	// TODO: implement
    return perimeterCurvature;
}



