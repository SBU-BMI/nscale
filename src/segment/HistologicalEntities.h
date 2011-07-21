/*
 * HistologicalEntities.h
 *
 *  Created on: Jul 1, 2011
 *      Author: tcpan
 */

#ifndef HistologicalEntities_H_
#define HistologicalEntities_H_

#include "cv.h"

using namespace cv;

namespace nscale {

class HistologicalEntities {
protected:
	HistologicalEntities() {};
	virtual ~HistologicalEntities() {};

public:
	static Mat getRBC(Mat img);
	static Mat getRBC(std::vector<Mat> rgb);
};

}
#endif /* HistologicalEntities_H_ */
