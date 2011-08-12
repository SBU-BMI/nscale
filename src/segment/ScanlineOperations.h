/*
 * ScanlineOperations.h
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#ifndef SCANLINEOPERATIONS_H_
#define SCANLINEOPERATIONS_H_

#include <vector>

#include "cv.h"


namespace nscale {

class ScanlineOperations {

public:

	// require that the contour be generated using "CV_CHAIN_APPROX_NONE" - can't skip boundary pixels.
	static uint64_t getContourArea(const std::vector<cv::Point>& contour);

	// require that the contour be generated using "CV_CHAIN_APPROX_NONE" - can't skip boundary pixels.
	// if color >= 0, then all contours are filled with that color.  else the color is changed per contour.
//	static cv::Mat fillContours(const std::vector<std::vector<cv::Point> >& contours, cv::Scalar& color);

protected:
	static std::vector<cv::Point> cleanContour(const std::vector<cv::Point>& contour);
	static std::vector<cv::Point> duplicateVertices(const std::vector<cv::Point>& contour);
	static inline void duplicateVertices1(
			const int& dx0, const int& dy0, const int& dx1, const int& dy1,
			const cv::Point& v1, std::vector<cv::Point>& newContour);

private:
	static inline bool compareX(cv::Point i, cv::Point j) {return (i.x < j.x);};
	static inline bool compareY(cv::Point i, cv::Point j) {return (i.y < j.y);};
};

}

#endif /* SCANLINEOPERATIONS_H_ */
