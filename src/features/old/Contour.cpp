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

Contour::Contour(CvSeq *c_l_param)
{
	// Initialize memory block used to store Contour structs
	 self_storage = cvCreateMemStorage();

	 // Copy the given list sequence of objects defining the Contour
	 this->c_l = cvCloneSeq(c_l_param, self_storage);

	 // Initialize contour features, which is used to
	 // check whether it was calculated before or not
	 this->area = -1.0;
	 this->circularity = -1.0;
	 this->compacteness = -1.0;
	 this->convexArea = -1.0;
	 this->convexDeficiency = -1.0;
	 this->eccentricity = -1.0;
	 this->equivalentDiameter = -1.0;
	 this->extent = -1.0;
	 this->perimeter = -1.0;
	 this->bendingEnergy = -1.0;
	 this->solidity = -1.0;
	 this->min_bounding_box.size.width = -1.0;
	 this->min_bounding_box.size.height = -1.0;
	 this->m_bounding_box.width = -1;
	 this->convexHull = NULL;
	 this->m_moments.m00 = -1.0;
}

Contour::~Contour() {
	cvReleaseMemStorage(&self_storage);
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
		cvReleaseMemStorage(&local_storage);
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

CvRect Contour::getNonInclinedBoundingBox(CvSize originalImageSize )
{
	// it is calculated?
	if( m_bounding_box.width != -1 )
	{
		return m_bounding_box;
	}


	CvSeqReader reader;
	CvPoint actualPoint;


	// it is an empty blob?
	if( !c_l )
	{
		m_bounding_box.x = 0;
		m_bounding_box.y = 0;
		m_bounding_box.width = 0;
		m_bounding_box.height = 0;

		return m_bounding_box;
	}

	cvStartReadSeq( c_l, &reader);

	m_bounding_box.x = originalImageSize.width;
	m_bounding_box.y = originalImageSize.height;
	m_bounding_box.width = 0;
	m_bounding_box.height = 0;

	for( int i=0; i< c_l->total; i++)
	{
		CV_READ_SEQ_ELEM( actualPoint, reader);

		m_bounding_box.x = MIN( actualPoint.x, m_bounding_box.x );
		m_bounding_box.y = MIN( actualPoint.y, m_bounding_box.y );

		m_bounding_box.width = MAX( actualPoint.x, m_bounding_box.width );
		m_bounding_box.height = MAX( actualPoint.y, m_bounding_box.height );
	}

	//m_boundingBox.x = max( m_boundingBox.x , 0 );
	//m_boundingBox.y = max( m_boundingBox.y , 0 );

	m_bounding_box.width -= m_bounding_box.x;
	m_bounding_box.height -= m_bounding_box.y;

	return m_bounding_box;
}



double Contour::getMoment(int p, int q)
{
	// is a valid moment?
	if ( p < 0 || q < 0 || p > 3 || q > 3 )
	{
		return -1;
	}

	// Has been calculated?
	if( m_moments.m00 == -1)
	{
		cvMoments( this->getCl(), &m_moments );
	}

	return cvGetSpatialMoment( &m_moments, p, q );

}



CvSeq *Contour::getCl()
{
	return this->c_l;
}

float Contour::getBendingEnergy()
{

	if(bendingEnergy == -1.0){
		CvSeqReader reader;
		CvPoint actualPoint;
		CvPoint nextPoint;

		cvStartReadSeq( c_l, &reader);

		bendingEnergy = 0.0;

		// Iterate on contour points calculating Bending Energy as summation of
		// perimeter curvature for each pair of points on the blob Contour
		if(c_l->total <= 1){
			bendingEnergy = 0.0;
		}else{
			CV_READ_SEQ_ELEM( actualPoint, reader);

			for( int i=0; i< c_l->total-1; i++){
				CV_READ_SEQ_ELEM( nextPoint, reader);

				bendingEnergy += atan( (float)(nextPoint.y-actualPoint.y)/(float)(nextPoint.x-actualPoint.x) );

				actualPoint = nextPoint;
			}
			// calculate perimeter curvature for the last and first points of the Contour,
			// which is different since the n+1 point is the first point of the Contour
			cvStartReadSeq( c_l, &reader);
			CV_READ_SEQ_ELEM( nextPoint, reader);
			bendingEnergy += atan( (float)(nextPoint.y-actualPoint.y)/(float)(nextPoint.x-actualPoint.x) );
		}
	}
	return bendingEnergy;
}



