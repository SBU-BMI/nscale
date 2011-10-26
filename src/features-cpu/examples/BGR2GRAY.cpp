#include "BGR2GRAY.h"

int rndint(float n)//round float to the nearest integer
{	
	int ret = floor(n);
	float t;
	t=n-floor(n);
	if (t>=0.5)    
	{
		ret = floor(n) + 1;
	}
	return ret;
}

IplImage *bgr2gray(IplImage* colorImage){
	if(colorImage == NULL){
		cout << __FILE__<<":"<<__LINE__<<" Input color image is NULL!"<<endl;
		exit(1);
		
	}
	if(colorImage->nChannels != 3){
		cout << __FILE__<<":"<<__LINE__<<" Number of channels of input color image is must be 3!"<<endl;
		exit(1);
		
	}
	double r_const = 0.298936021293776;
	double g_const = 0.587043074451121;
	double b_const = 0.114020904255103;

	IplImage *grayImage = cvCreateImage(cvGetSize(colorImage),IPL_DEPTH_8U, 1);


	for(int y = 0; y < colorImage->height; y++){
		uchar *color_ptr = (uchar*)( colorImage->imageData + y * colorImage->widthStep );
		uchar *gray_ptr = (uchar*)( grayImage->imageData + y * grayImage->widthStep );

		for(int x = 0; x < colorImage->width; x++){
			uchar b = color_ptr[3*x];
			uchar g = color_ptr[3*x+1];
			uchar r = color_ptr[3*x+2];
			double grayPixelValue = r_const * (double)r + g_const * (double)g + b_const * (double)b;
			int grayPixelValueInt = rndint(grayPixelValue);
			gray_ptr[x] = (uchar)grayPixelValueInt;
		}
	}

	return grayImage;
}


