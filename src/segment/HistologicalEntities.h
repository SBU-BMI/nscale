/*
 * HistologicalEntities.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef HistologicalEntities_H_
#define HistologicalEntities_H_

#include "cv.h"

namespace nscale {

class HistologicalEntities {

public:
	static cv::Mat getRBC(const cv::Mat& img);
	static cv::Mat getRBC(const std::vector<cv::Mat>& rgb);

	static int segmentNuclei(const cv::Mat& img, cv::Mat& output);

};

}
#endif /* HistologicalEntities_H_ */
