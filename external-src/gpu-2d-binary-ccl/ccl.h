#ifndef _CCL_H_
#define _CCL_H_

#include "cclBase.h"

//the main class that solves the CCL problem.

class CCL : public CCLBase
{
public:
	CCL();
	virtual ~CCL();
	//find connected regions (components) on a segmented data of dimensions imgSizeX and imgSizeY
	virtual bool FindRegions(int imgSizeX, int imgSizeY, CudaBuffer* segmentedData, int connectivity);	
};

#endif
