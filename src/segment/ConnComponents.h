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
#if( _MSC_VER == 1800 )
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
namespace nscale {


struct box {
	int minx;
	int maxx;
	int miny;
	int maxy;
};


#ifdef _MSC_VER
class	__declspec(dllexport) ConnComponents {
#else
class ConnComponents {
#endif
public:
	ConnComponents();
	virtual ~ConnComponents();

	void label(unsigned char *img, int w, int h, int *label, int bgval, int connectivity);
	int areaThreshold(unsigned char *img, int w, int h, int *label, int bgval, int lower, int upper, int connectivity);

	int relabel(int w, int h, int *label, int bgval);
	int areaThresholdLabeled(const int *label, const int w, const int h, int *n_label, const int bgval, const int lower, const int upper);
	int* boundingBox(const int w, const int h, const int* label, int bgval, int &compcount);

protected:
	int find(int *label, int x);
	void merge(int *label, int x, int y);
	void mergeWithArea(int *label, int x, int y, int *areas);
	int flatten(int *label, int x, int bgval);
};

}

#endif /* CONNCOMPONENTS_H_ */
