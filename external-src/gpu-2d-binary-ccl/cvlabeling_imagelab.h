// cvLabelingImageLab: an impressively fast labeling routine for OpenCV
// Copyright (C) 2009 - Costantino Grana and Daniele Borghesani
//
// This library is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free 
// Software Foundation; either version 3 of the License, or (at your option) 
// any later version.
//
// This library is distributed in the hope that it will be useful, but WITHOUT 
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more 
// details.
//
// You should have received a copy of the GNU Lesser General Public License along
// with this library; if not, see <http://www.gnu.org/licenses/>.
//
// For further information contact us at
// Costantino Grana/Daniele Borghesani - University of Modena and Reggio Emilia - 
// Via Vignolese 905/b - 41100 Modena, Italy - e-mail: {name.surname}@unimore.it

#ifndef _CVLABELING_IMAGELAB_H_
#define _CVLABELING_IMAGELAB_H_
#include "cv.h"

// fast block based labeling with decision tree optimization
//
// src: single channel binary image of type IPL_DEPTH_8U 
// dst: single channel label image of type IPL_DEPTH_32S
CVAPI(void) cvLabelingImageLab (IplImage* srcImage, IplImage* dstImage, 
								unsigned char byForeground, int *numLabels);

#endif/*_CVLABELING_IMAGELAB_H_*/
