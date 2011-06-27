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
	float ellipseAngle;
	float equivalentDiameter;
	float extent;
	float majorAxisLength;
	float minorAxisLength;
	float perimeter;
	float perimeterCurvature;
	float solidity;
	float sphericity;

	//! List of points defining this contour.
	CvSeq *c_l;

	//! Ellipse fitting the contour
	CvBox2D min_fitting_ellipse;

	//! Minimum bounding box
	CvBox2D min_bounding_box;

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
     * Calculates Contour's Eccentricity. The Eccentricity calculation depends on
     * characteristics of the Ellipse fitting the contours: Its definition is:
     *
     * 	Eccentricity = ( 2* ((MajorAxis/2)^2 - (MinorAxis/2)^2)^1/2 )/MajorAxis, where
     * 	Major/Minor Axis refers to the fitting ellipse
     */
    float getEccentricity();

    /*!
     * Calculates a Contour's Orientation. The Orientations is defined by the angle
     * among the fitting Ellipse Major Axis and the x-axis. Thus, it obviously depends
     * on fitting an Ellipse.
     */
    float getOrientation();

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
     * Calculates a Contour's Major Axis that is the longest diameter of an ellipse, which
     * passes through the center and foci. It requires fitting an ellipse to the contour, and
     * then checking the highest value among its width and height.
     */
    float getMajorAxisLength();

    /*!
     * Calculates a Contour's Minor Axis. It is a line through the center of an ellipse that
     * is also perpendicular to the Major Axis. As in the Major Axis, it requires fitting an
     * ellipse.
     */
    float getMinorAxisLength();

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
     * Calculate the Contour's Ellipticity, which is actually based on the Contour fitting
     * Ellipse. It is a measure of the ``squashing'' of the spheroi's pole, towards its
     * equator.
     *
     * 	Ellipticity = (MajorAxis-MinorAxis)/MajorAxis
     */
    float getEllipticity();

    /*!
     *  This function is not implemented. It requires fitting a circle external to the contour, which i
     */
    float getSphericity();

    // This function is not implemented. I still in doubts about how to calculate it, but I think we can do it.
    float getPerimeterCurvature();
};

#endif /* CONTOUR_H_ */
