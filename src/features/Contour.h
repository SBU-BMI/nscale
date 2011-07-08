/*
 * Contour.h
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#ifndef CONTOUR_H_
#define CONTOUR_H_

#include "highgui.h"
#include "cv.h"
#include "cxcore.h"
#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace std;

/*!
	This is the basic class used in our analysis. It implements methods to return basic
	properties that are further utilized by other classes to calculate observations in
	complex objects. It is important to note that a Contour is an object that does not
	contain any nested object, or if it contains the Contour will ignore it. The properties
	that require an analysis of nested objects is handled at Blob level - See Blob Class
	for details.
 */

// TODO: Make sure we have 6 or more points in a Contour before fitting an Ellipse
class Contour {
private:


	/*!
	 * Aux. variables used to store values of properties measured for each Contours.
	 * These variables are initiated with (-1.0) to indicate that the property was
	 * not calculated yet, and they receive the adequate value as the corresponent
	 * get${VarName} function is called.
	 */
	float area;
	float circularity;
	float compacteness;
	float convexArea;
	float convexDeficiency;
	float eccentricity;

	float equivalentDiameter;
	float extent;
	float perimeter;
	float bendingEnergy;
	float solidity;

	//! List of points defining this contour.
	CvSeq *c_l;

	//! Ellipse fitting the contour
	//CvBox2D min_fitting_ellipse;

	//! Moments of the contour
	CvMoments m_moments;

	//! Minimum bounding box
	CvBox2D min_bounding_box;

	//! Bounding box without inclination
	CvRect m_bounding_box;

	//! Convex Hull of the given Contour
	CvSeq *convexHull;

	//! Storage used by opencv to store memory allocated to various structures
	CvMemStorage* self_storage;


	/*!
		This constructor is private to avoid its
	 	 use. The appropriate constructor is public.
	 */
	Contour();

public:
	/*!
		Constructor that receives a list of points defining the Contour,
		and correctly initialize its properties/variables.
	*/
	Contour(CvSeq *c_l);

	//! A simple class destructor that takes care of deallocate used data structures
	virtual ~Contour();


	CvSeq *getCl();
	double getMoment(int p, int q);

	/*!
	 * Calculates the area inside the contour
	 * \return Contour's area
	 */
    float getArea();

    /*!
     * Calculates Contour's circularity, which is defined as follows:
     *  Circularity = Perimeter^2/Area - (4 * PI)
     */
    float getCircularity();

    /*!
     * Calculates Contour's Compacteness, which is defined as follows:
     *
     *  Compacteness = (4 * PI * Area) / Perimeter^2
     */
    float getCompacteness();

    /*!
     * Calculates Contour's Convex Area(ConvexArea). It first calculates the
     * Convex Hull, if was not initialized, transforms the points defining the
     * Convex Hull into a Contour and calculates its area.
     */
    float getConvexArea();

    /*!
     * Calculates Contour's Convex Deficiency. The Convex Deficiency is defined
     * as follows:
     *
     *  ConvexDeficiency = (ConvexArea - Area) / Area
     */
    float getConvexDeficiency();

    /*!
     * Calculates a Contour's Equivalent Diameter, which is the diameter of the circle
     * with the same area as the Contour. Follows the definition:
     *
     * 	EquivDiameter = 2 * (PI/Area)^1/2
     */
    float getEquivalentDiameter();

    /*!
     * Calculates a Contour's extent, which is the fraction of pixels within the bounding
     * box that are also within the Contour:
     *
     *	Extent = Area / (bounding.box.width*bounding.box.height)
     */
    float getExtent();

    /*!
     * It calculates Contour's Perimeter. The function calculates it as sum of lengths of
     *  segments between subsequent points that defines the Contour.
     */
    float getPerimeter();

    /*!
     * Calculates the Contour's Solidity. It is defined as the fraction of pixels within the
     * Convex Hull that are also within the Contour.
     *
     * 	Solidity = Area/ConvexArea
     */
    float getSolidity();

    /*!
     * Calculates the Contour's bounding box are. The bounding box refers to the minimum
     * rectangle that encompasses all points of the contour. The Area is further calculated
     * as the width*height.
     */
    float getMinBoundingBoxArea();
    float getBoundingBoxWidth();
    float getBoundingBoxHeight();

    /*!
     * Calculates the Contour's bounding box without inclination
     */
    CvRect getNonInclinedBoundingBox(CvSize originalImageSize );


    float getBendingEnergy();
};

#endif /* CONTOUR_H_ */
