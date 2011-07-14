/*
 * Blob.h
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */

#ifndef BLOB_H_
#define BLOB_H_

#include "Contour.h"
#include "DrawAuxiliar.h"
#include <list>

using namespace std;
using namespace cv;

class Blob {
private:
	Contour *external_contour;
	vector<Contour*> internal_contours;

	//! Dimensions of the original image
	CvSize originalImageSize;

	//! Ellipse fitting the blob
	CvBox2D min_fitting_ellipse;

	//! Mask image provided for this Blob
	IplImage *mask;

	//! Image header set to the ROI in the original image
	IplImage *ROISubImage;

	CvHistogram* intensity_hist;
	unsigned int intensity_hist_points;

	CvHistogram* grad_hist;
	unsigned int grad_hist_points;

	CvMemStorage* self_storage;

	// Computes the pixels histogram of the gray input image
	// and stores it in instensity_hist
	void calcIntensityHistogram(IplImage* img);

	// Computes the Morphology gradient histogram of the
	// grayscale input image and stores it in instensity_hist
	void calcGradientHistogram(IplImage *img);

	unsigned int getIntensityHistPoints(){
		return intensity_hist_points;
	}
	unsigned int getGradHistPoints(){
		return grad_hist_points;
	}

	// Retrieves the pointes to the ROI of this
	// blob in the original input image
	IplImage *getROISubImage(IplImage *img);

	double getMoment(int p, int q);
	CvBox2D getEllipse();

	float majorAxisLength;
	float minorAxisLength;

	Blob();
	friend class DrawAuxiliar;
public:
	Blob(CvSeq* c, CvSize originalImageSize );

	virtual ~Blob();

	/*!
	 * Calculates Blob's area, which is defined as the area of the
	 * external contour minus area of the holes.
	 */
	float getArea();

    /*!
     * Calculates Blob's Compacteness, which is defined as follows:
     *
     *  Compacteness = (4 * PI * Area) / Perimeter^2
     */
	float getCompacteness();

	/*!
     * Calculates Blob's Convex Area(ConvexArea). It first calculates the
     * Convex Hull, if was not initialized, transforms the points defining the
     * Convex Hull into a Contour and calculates its area.
     */
	float getConvexArea();

	/*!
	 * Calculates Convex Deficiency. The Convex Deficiency is defined
	 * as follows:
	 *
	 *  ConvexDeficiency = (ConvexArea - Area) / Area
	 */
	float getConvexDeficiency();

    /*!
     * The Eccentricity calculation depends on
     * characteristics of the Ellipse fitting the contours: Its definition is:
     *
     * 	Eccentricity = ( 2* ((MajorAxis/2)^2 - (MinorAxis/2)^2)^1/2 )/MajorAxis, where
     * 	Major/Minor Axis refers to the fitting ellipse
     */
	float getEccentricity();

	/*!
    * Calculates a Blob's Orientation. The Orientations is defined by the angle
    * among the fitting Ellipse Major Axis and the x-axis. Thus, it obviously depends
    * on fitting an Ellipse.
    */
	float getOrientation();

    /*!
     * Calculates a Blob's Equivalent Diameter, which is the diameter of the circle
     * with the same area as the Blob. Follows the definition:
     *
     * 	EquivDiameter = 2 * (PI/Area)^1/2
     */
	float getEquivalentDiameter();

    /*!
     * Calculates a Blob's extent, which is the fraction of pixels within the bounding
     * box that are also within the Contour:
     *
     *	Extent = Area / (bounding.box.width*bounding.box.height)
     */
	float getExtent();

    /*!
     * Calculate the Blob's Ellipticity, which is actually based on fitting
     * an Ellipse. It is a measure of the ``squashing'' of the spheroi's pole, towards its
     * equator.
     *
     * 	Ellipticity = (MajorAxis-MinorAxis)/MajorAxis
     */
	float getEllipticity();

	/*!
	 * It calculates the area of the region contained in the Blob's external Contour. In other
	 * words: Area of the Blob + Area of the internal holes
	 */
	float getFilledArea();

	/*!
	 * Calculates a Blobs's Major Axis that is the longest diameter of an ellipse, which
	 * passes through the center and foci. It requires fitting an ellipse to the blob, and
	 * then checking the highest value among its width and height.
	 */
	float getMajorAxisLength();

	/*!
	 * Calculates a Blob's Minor Axis. It is a line through the center of an ellipse that
	 * is also perpendicular to the Major Axis. As in the Major Axis, it requires fitting an
	 * ellipse.
	 */
	float getMinorAxisLength();

	/*!
	 * Calculates the Blob's AspectRatio, defined as:
	 * (Width of the bounding box / height of the bounding box)
	 */
	float getAspectRatio();

    /*!
     * It calculates Blob's perimeter. The function calculates it as sum of lengths of
     *  segments between subsequent points that defines the blob boundaries. It also includes
     *  the boundaries separating the external contour and the holes inside the blob.
     */
	float getPerimeter();

	/*!
	 * It calculates the Blob's Bending Energy based on the external Contours.
	 * The PerimeterCurvature(Θ) is used to calculate Bending Energy, and is defined in
	 * terms of each par of points in the blob curvature. So, in this case the ending
	 * result feature is the Bending Energy. The definitions:
	 * 	Θ_n = arctan( ( y(n+1) - y( n) ) / ( x(n+1) - x( n) ) ),
	 * 	n = 1, ..., N and x(N+1) = x(1) and y(N+1) = y(1).
	 * 	BendingEnergy = SUM_n=1...N ( Θ(n+1) - Θ( n) )
	 */
	float getBendingEnery();

	/*!
	 * Solidity is calculated as: Solidity = Area/ConvexArea
	 */
	float getSolidity();

	/*!
	 *  Euler Number is defined as: 1 - #ofHole in the Blob
	 */
	int getEulerNumber();

	/*!
	 * The Blob's porosity is: Porosity = Area of holes / Area of the Blob
	 */
	float getPorosity();

	/*!
	 * Calculates the Blob's reflection symmetry, which is defined as:
	 * #pixels that are in the blob AND its reflection/#pixels in the blob
	 */
	float getReflectionSymmetry();

	//! Creates an mask for this blob with the same size as
	// its bounding box without inclination
	IplImage *getMask();

	//! Get Blob bounding box without inclination
    CvRect getNonInclinedBoundingBox();

    /*!
     * Calculates area of the bounding box without inclination
     */
    float getNonInclinedBoundingBoxArea( );

    //! Calculate pixels intensity
    double getMeanIntensity(IplImage* img);
    unsigned int getMedianIntensity(IplImage* img);
    unsigned int getMinIntensity(IplImage* img);
    unsigned int getMaxIntensity(IplImage* img);
    unsigned int getFirstQuartileIntensity(IplImage* img);
    unsigned int getThirdQuartileIntensity(IplImage* img);

    //! Calculate pixels Gradient Magnitude
    double getMeanGradMagnitude(IplImage* img);
    unsigned int getMedianGradMagnitude(IplImage* img);
    unsigned int getMinGradMagnitude(IplImage* img);
    unsigned int getMaxGradMagnitude(IplImage* img);
    unsigned int getFirstQuartileGradMagnitude(IplImage* img);
    unsigned int getThirdQuartileGradMagnitude(IplImage* img);
    unsigned int getCannyArea(IplImage* img, double lowThresh, double highThresh, int apertureSize = 3);
    unsigned int getSobelArea( IplImage *img, int xorder, int yorder, int apertureSize=3 );

};

#endif /* BLOB_H_ */
