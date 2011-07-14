/*
 * RedBloodCell.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef REDBLOODCELL_H_
#define REDBLOODCELL_H_

#include "cv.h"

using namespace cv;

namespace nscale {

class RedBloodCell {
protected:
	RedBloodCell() {};
	virtual ~RedBloodCell() {};

public:
	static Mat rbcMask(Mat img);
	static Mat rbcMask(std::vector<Mat> rgb);
};

}
#endif /* REDBLOODCELL_H_ */
