/*
 * Blob.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#include "Blob.h"

Blob::Blob(CvSeq* first_contour) {

	self_storage = cvCreateMemStorage();

	if(first_contour != NULL){
		external_contour = new Contour(first_contour);
	}

	CvSeq *aux = first_contour;

	if(aux->v_next){
		aux = aux->v_next;
		cout<<"Blob:Found nested contour"<<endl;
		Contour *contour = new Contour(aux);
		internal_contours.push_back(contour);


		while(aux->h_next){
			cout<<"Blob:Found nested contour"<<endl;
			Contour *contour = new Contour(aux);
			internal_contours.push_back(contour);
			aux = aux->h_next;
		}
	}
}

Blob::~Blob() {
	if(external_contour != NULL){
		delete external_contour;
	}
	internal_contours.clear();
	cvClearMemStorage(self_storage);
}

float Blob::getArea()
{
	float area = external_contour->getArea();
	for(list<Contour*>::const_iterator it = internal_contours.begin(); it != internal_contours.end(); it++){
		area -= (*it)->getArea();
	}
	return area;
}

float Blob::getConvexArea()
{
	return external_contour->getConvexArea();
}

float Blob::getExtent()
{
	float extent = external_contour->getArea()/ external_contour->getMinBoundingBoxArea();
	return extent;
}

float Blob::getAspectRatio()
{
	return (external_contour->getBoundingBoxHeight()/external_contour->getBoundingBoxWidth());
}

float Blob::getPerimeter()
{
	float perimeter = external_contour->getPerimeter();
	for(list<Contour*>::const_iterator it = internal_contours.begin(); it != internal_contours.end(); it++){
		perimeter += (*it)->getPerimeter();
	}
	return perimeter;
}

float Blob::getEquivalentDiameter()
{
	float equivalentDiameter = 2 * sqrtf((CV_PI/getArea()));
	return equivalentDiameter;
}

float Blob::getFilledArea()
{
	return external_contour->getArea();
}


float Blob::getMinorAxisLength()
{
	return external_contour->getMinorAxisLength();
}

float Blob::getSolidity()
{
	return (getArea()/getConvexArea());
}

float Blob::getOrientation()
{
	return external_contour->getOrientation();
}


float Blob::getCircularity()
{
	return (pow(getPerimeter(),2)/getArea()) - (4.*CV_PI);
}

float Blob::getMajorAxisLength()
{
	return external_contour->getMajorAxisLength();
}

float Blob::getCompacteness()
{
	return (4*CV_PI*getArea()/pow(getPerimeter(),2));
}


float Blob::getEccentricity()
{
	float foci = pow((getMajorAxisLength()/2),2) - pow((getMinorAxisLength()/2), 2);

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
	return external_contour->getEllipticity();
}

float Blob::getPorosity()
{
	float porosity = 0.0;
	if(internal_contours.size() > 0){
		float areaOfInternalContours = 0.0;
		for(list<Contour*>::const_iterator it = internal_contours.begin(); it != internal_contours.end(); it++){
			areaOfInternalContours += (*it)->getArea();
		}
		porosity = areaOfInternalContours / getArea();
	}
	return porosity;
}



// TODO: Implement these functions

float Blob::getPerimeterCurvature()
{

}

float Blob::getSphericity()
{

}



