/*
 * RegionalMorphologyAnalysis.h
 *
 *  Created on: Jun 22, 2011
 *      Author: george
 */

#ifndef REGIONALMORPHOLOGYANALYSIS_H_
#define REGIONALMORPHOLOGYANALYSIS_H_

#include "Blob.h"
#include "Contour.h"

class RegionalMorphologyAnalysis {
private:
	vector<Blob *> internal_blobs;
	RegionalMorphologyAnalysis();
	IplImage *originalImage;

	void initializeContours(string maskInputFileName);
public:
	RegionalMorphologyAnalysis(string maskInputFileName, string grayInputFileName);
	virtual ~RegionalMorphologyAnalysis();

	void doAll();
	void doRegionProps();
	void doIntensity();
};

#endif /* REGIONALMORPHOLOGYANALYSIS_H_ */
