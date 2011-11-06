#ifndef	__BGR2GRAY
#define	__BGR2GRAY

#include <cv.h>
#include <highgui.h>
using namespace cv;

#include <iostream>
#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>

using namespace std;


IplImage *bgr2gray(IplImage* colorImage);

#endif
