/*
 * ConnComponents.h
 *
 *  Created on: Feb 20, 2012
 *      Author: tcpan
 */

#ifndef CONNCOMPONENTS_H_
#define CONNCOMPONENTS_H_

#include <stdio.h>
#include <stdlib.h>

namespace nscale {

class ConnComponents {
public:
	ConnComponents();
	virtual ~ConnComponents();

	void label(unsigned char *img, int w, int h, int *label, int bgval, int connectivity);
	int areaThreshold(unsigned char *img, int w, int h, int *label, int bgval, int lower, int upper, int connectivity);

	int relabel(int w, int h, int *label, int bgval);

protected:
	int find(int *label, int x);
	void merge(int *label, int x, int y);
	void mergeWithArea(int *label, int x, int y, int *areas);
	int flatten(int *label, int x, int bgval);
};

}

#endif /* CONNCOMPONENTS_H_ */
