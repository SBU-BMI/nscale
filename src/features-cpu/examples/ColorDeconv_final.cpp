//============================================================================
// Name        : ColorDeconv.cpp
// Author      : Jun Kong
// Version     :
// Copyright   : Your copyright notice
// Description : color deconvolution method
//============================================================================

#include <cv.h>
#include <highgui.h>
using namespace cv;

#include <iostream>
#include <math.h>
//#include <stdio.h>
//#include <stdlib.h>

using namespace std;

//convert double to unsigned char
unsigned char double2uchar(double d){
    double truncate = std::min( std::max(d,(double)0.0), (double)255.0);
	double pt;
	double c = modf(truncate, &pt)>=.5?ceil(truncate):floor(truncate);
	return (unsigned char)c;
}

//show a Image of type Mat
void showImage(const Mat& M, const char* S, const int c){
	printf("%s, channel=%d\n",S, c);
	for (int i=0; i<M.rows; i++){
		for (int j=0; j<M.cols; j++){
			if(M.channels() == 1)
				printf("%d \t", M.at<unsigned char>(i,j));
			else if(M.channels() == 3)
				printf("%d \t", M.at<Vec3b>(i,j)[c]);

		}
		printf("\n");
	}
}

//show a matrix of type Mat
void showMat(const Mat& M, const char* S, const int c = 0){
	printf("%s\n",S);
	for (int i=0; i<M.rows; i++){
		for (int j=0; j<M.cols; j++){
			if(M.channels() == 1)
				printf("%7.4lf \t", M.at<double>(i,j) );
			else if (M.channels() == 2)
				printf("%7.4lf \t", M.at<Vec2d>(i,j)[c]);
			else if(M.channels() == 3)
				printf("%7.4lf \t", M.at<Vec3d>(i,j)[c]);
		}
		printf("\n");
	}
}

void ColorDeconv( const Mat& image, const Mat& M, const Mat& b, Mat& H, Mat& E, bool BGR2RGB){
	//initialize normalized stain deconvolution matrix
	Mat normal_M;
	M.copyTo(normal_M);

	//stain normalization
	double col_Norm;
	for(int i=0; i< M.cols; i++){
		col_Norm = norm(M.col(i));
		if(  col_Norm > (double) 0 ){
			normal_M.col(i) = M.col(i) / col_Norm;
		}
	}
	//showMat(normal_M, "normal_M--stain normalization: ");

    //find last column of the normalized stain deconvolution matrix
	int last_col_index = M.cols-1;
	Mat last_col = normal_M.col(last_col_index); //or//Mat last_col = normal_M(Range::all(), Range(2,3));
	//showMat( last_col, "last column " );

	//normalize the stain deconvolution matrix again
	if(norm(last_col) == (double) 0){
		for(int i=0; i< normal_M.rows; i++){

			if( norm(normal_M.row(i)) > 1 ){
				normal_M.at<double>(i,last_col_index) = 0;
			}
			else{
				normal_M.at<double>(i,last_col_index) =  sqrt( 1 - pow(norm(normal_M.row(i)), 2) );
			}
		}
		normal_M.col(last_col_index) = normal_M.col(last_col_index) / norm(normal_M.col(last_col_index));
	}
	//showMat(normal_M, "normal_M");

	//take the inverse of the normalized stain deconvolution matrix
	Mat Q = normal_M.inv();
	if (b.size().height != 1){
		printf("b has to be a row vector \n");
		exit(-1);
	}

	//select rows in Q with a true marker in b
	Mat T(1,Q.cols,Q.type());
	for (int i=0; i<b.size().width; i++){
		if( b.at<char>(0,i) == 1 )
			T.push_back(Q.row(i));
	}
	Q = T.rowRange(Range(1,T.rows));
	//showMat(Q, "Q\n");

    //normalized image
	int nr = image.rows, nc = image.cols;
	Mat dn = Mat::zeros(nr, nc, CV_64FC3);
    if (image.channels() == 1){
		for(int i=0; i<nr; i++){
			const uchar* data_in = image.ptr<uchar>(i);
			double* data_out = dn.ptr<double>(i);
			for(int j=0; j<nc; j++){
				data_out[j] = -(255.0*log(((double)(data_in[j])+1.0)/255.0))/log(255.0);
			}
		}
	}else if(image.channels() == 3){
		for(int i=0; i<nr; i++){
			for(int j=0; j<nc; j++){
				for(int k=0; k<image.channels(); k++)
					dn.at<Vec3d>(i,j)[k] = -(255.0*log(((double)image.at<Vec3b>(i,j)[k] +1.0)/255.0))/log(255.0);
			}
		}
	}
    //showMat(dn, "dn: ");

    //channel deconvolution
	Mat cn = Mat::zeros(nr, nc, CV_64FC2);
	for(int i=0; i<nr; i++){
		for(int j=0; j<nc; j++){
			if (dn.channels() == 3){
				for(int k=0; k<dn.channels(); k++){
					for(int Q_i=0; Q_i<Q.rows; Q_i++)
						if( BGR2RGB )
							cn.at<Vec2d>(i,j)[Q_i] += Q.at<double>(Q_i,k) * dn.at<Vec3d>(i,j)[dn.channels()-1-k];
						else
							cn.at<Vec2d>(i,j)[Q_i] += Q.at<double>(Q_i,k) * dn.at<Vec3d>(i,j)[k];
				}
			}
			else{
				printf("Image must have 3 channels for color deconvolution \n");
				exit(-1);
			}
		}
	}
    //showMat(cn, "cn: ", 0);
	//showMat(cn, "cn: ", 1);

	//denormalized H and E channels
	double temp;
	for(int i=0; i<nr; i++){
		for(int j=0; j<nc; j++){

			temp = exp(-(cn.at<Vec2d>(i,j)[0]-255.0)*log(255.0)/255.0);
			H.at<uchar>(i,j) =  double2uchar(temp);

			temp = exp(-(cn.at<Vec2d>(i,j)[1]-255.0)*log(255.0)/255.0);
			E.at<uchar>(i,j) =  double2uchar(temp);

		}
	}
	//showImage(H, "check H:", 0);
	//showImage(E, "check E:", 0);

}

//
////change the order of {BGR} to {RGB}
//void BGRtoRGB(Mat& image){
//	unsigned char temp;
//	for(int i=0; i<image.rows; i++){
//		for(int j=0; j<image.cols; j++){
//			temp = image.at<Vec3b>(i,j)[0];
//			image.at<Vec3b>(i,j)[0] = image.at<Vec3b>(i,j)[2];
//			image.at<Vec3b>(i,j)[2] = temp;
//		}
//	}
//
//}


//int main(int argc, char** argv) {
//	//initialize stain deconvolution matrix and channel selection matrix
//	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
//	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);
//
//	//read image
//	Mat image;
//	image = imread( argv[1], 1 );  //For each pixel, BGR
//	if( argc != 2 || !image.data )
//	{
//		printf( "No image data \n" );
//		return -1;
//	}
//
//	//specify if color channels should be re-ordered
//	bool BGR2RGB = true;
//    //BGRtoRGB(Mat& image);
//
//	//initialize H and E channels
//	Mat H = Mat::zeros(image.size(), CV_8UC1);
//	Mat E = Mat::zeros(image.size(), CV_8UC1);
//
//	//color deconvolution
//	ColorDeconv( image, M, b, H, E, BGR2RGB);
//
//	//show result
//	namedWindow( "Color Image", CV_WINDOW_AUTOSIZE );
//	imshow( "Color Image", image );
//    imshow( "H Image", H );
//	imshow( "E Image", E );
//    waitKey(5000);
//
//    //write H and E channels to disk
//    imwrite("H.bmp", H);
//    imwrite("E.bmp", E);
//
//    //release memory
//    H.release();
//    E.release();
//    M.release();
//    b.release();
//    image.release();
//
//    return 0;
//}
