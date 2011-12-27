#include <cv.h>
#include <highgui.h>
using namespace cv;

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ColorDeconv_final.h"
#include "utils.h"

using namespace std;

int main(int argc, char** argv) {
	//initialize stain deconvolution matrix and channel selection matrix
	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);

	//read image
	Mat image;
	image = imread( argv[1], 1 );  //For each pixel, BGR
	if( argc != 2 || !image.data )
	{
		printf( "No image data \n" );
		return -1;
	}

	//specify if color channels should be re-ordered
	bool BGR2RGB = true;
    //BGRtoRGB(Mat& image);

	//initialize H and E channels
	Mat H = Mat::zeros(image.size(), CV_8UC1);
	Mat E = Mat::zeros(image.size(), CV_8UC1);

	long t1 = cciutils::ClockGetTime();
	//color deconvolution
	ColorDeconv( image, M, b, H, E, BGR2RGB);

	long t2 = cciutils::ClockGetTime();
	cout << "Conv original = "<< t2-t1<<endl;

	//show result
//	namedWindow( "Color Image", CV_WINDOW_AUTOSIZE );
//	imshow( "Color Image", image );
//	imshow( "H Image", H );
//	imshow( "E Image", E );
  // 	waitKey(5000);

	//initialize H and E channels
	Mat H_opt = Mat::zeros(image.size(), CV_8UC1);
	Mat E_opt = Mat::zeros(image.size(), CV_8UC1);

	t1 = cciutils::ClockGetTime();

	//color deconvolution
	ColorDeconvOptimized( image, M, b, H_opt, E_opt, BGR2RGB);

	t2 = cciutils::ClockGetTime();
	cout << "Conv optimized = "<< t2-t1<<endl;

	if(countNonZero(H != H_opt) || countNonZero(E != E_opt)){
		printf("Error: E or H images are not the same!\n");
		exit(1);
	}

	//initialize H and E channels
	Mat H_opt_float = Mat::zeros(image.size(), CV_8UC1);
	Mat E_opt_float = Mat::zeros(image.size(), CV_8UC1);

	t1 = cciutils::ClockGetTime();

	//color deconvolution
	ColorDeconvOptimizedFloat( image, M, b, H_opt_float, E_opt_float, BGR2RGB);

	t2 = cciutils::ClockGetTime();
	cout << "Conv optimized float = "<< t2-t1<<endl;
	if(countNonZero(H != H_opt_float) || countNonZero(E != E_opt_float)){
		printf("Error: E or H images are not the same! H = %d and E = %d\n", countNonZero(H != H_opt_float), countNonZero(E != E_opt_float));
		exit(1);
	}


	//release memory
	H.release();
	E.release();
	H_opt.release();
	E_opt.release();
	M.release();
	b.release();
	image.release();

    return 0;
}
