/*
 * RedBloodCell.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef REDBLOODCELL_H_
#define REDBLOODCELL_H_

#include "cv.h"

namespace nscale {

class RedBloodCell {
public:
	RedBloodCell();
	virtual ~RedBloodCell();

	cv::Mat rbcMask(cv::Mat img);
	cv::Mat rbcMask(std::vector<cv::Mat> rgb);
};

}
#endif /* REDBLOODCELL_H_ */
