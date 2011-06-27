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

	void initializeContours(string maskInputFileName);
public:
	RegionalMorphologyAnalysis(string maskInputFileName);
	virtual ~RegionalMorphologyAnalysis();

	void doAll();
};

#endif /* REGIONALMORPHOLOGYANALYSIS_H_ */
