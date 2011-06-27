/*
 * Blob.h
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#ifndef BLOB_H_
#define BLOB_H_

#include "Contour.h"
#include <list>
using namespace std;

class Blob {
private:
	Contour *external_contour;
	list<Contour*> internal_contours;

	CvMemStorage* self_storage;
	Blob();

public:
	Blob(CvSeq* c);

	virtual ~Blob();

	float getArea();
	float getCircularity();

	float getCompacteness();
	float getConvexArea();
	float getConvexDeficiency();
	float getEccentricity();
	float getOrientation();
	float getEquivalentDiameter();
	float getExtent();
	float getEllipticity();
	float getFilledArea();
	float getMajorAxisLength();
	float getMinorAxisLength();
	float getAspectRatio();
	float getPerimeter();
	float getPerimeterCurvature();
	float getSolidity();
	float getSphericity();
	int getEulerNumber();
	float getPorosity();
};

#endif /* BLOB_H_ */
