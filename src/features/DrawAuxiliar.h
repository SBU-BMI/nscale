/*
 * DrawAuxiliar.h
 *
 *  Created on: Jun 29, 2011
 *      Author: george
 */

#ifndef DRAWAUXILIAR_H_
#define DRAWAUXILIAR_H_

#include "Blob.h"
class Blob;

class DrawAuxiliar {
public:
	DrawAuxiliar();
	virtual ~DrawAuxiliar();
	static IplImage* DrawHistogram(CvHistogram *hist, float scaleX=1, float scaleY=1);
	static IplImage *DrawHistogram(unsigned int *hist, float scaleX=1, float scaleY=1);
	static IplImage* DrawBlob(Blob *blobToPrint, CvScalar external, CvScalar holes );
	static void DrawBlob(IplImage* printBlob, Blob *blobToPrint, CvScalar external, CvScalar holes);
};

#endif /* DRAWAUXILIAR_H_ */
