/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include <iostream>
#ifdef _MSC_VER
#include "direntWin.h"
#else
#include <dirent.h>
#endif
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "Logger.h"


using namespace cv;


int main (int argc, char **argv){

	Mat img1 = imread(argv[1], -1);
	Mat img2 = imread(argv[2], -1);
	std::cout << "img1.nChannels="<< img1.channels() << std::endl;
	std::cout << "imt2.nChannels="<< img2.channels() << std::endl;

	Mat img1mask = img1 > 0;	
	Mat img2mask = img2 > 0;

	
	Mat common = (img1mask+img2mask) > 1;

	Mat firstNotSecond = (img1mask-img2mask) > 0;
	std::cout << "Non-zero = "<< countNonZero(firstNotSecond) << std::endl;

	// In second, and not first
	Mat secondNotFirst = (img2mask-img1mask) > 0;

	Mat colorDiff (img1.size(), CV_8UC3);


	colorDiff.setTo(Scalar(0,255,0), common);
	colorDiff.setTo(Scalar(255, 0, 0), firstNotSecond);
	colorDiff.setTo(Scalar(0, 0, 255), secondNotFirst);


//	namedWindow("diff", CV_WINDOW_AUTOSIZE);
//	imshow("diff", colorDiff);
	imwrite("diff.pbm", colorDiff);

	Mat image1 = img1;
	Mat image2 = img2;
	int square_side = 128;
	if(image1.rows == image2.rows && image1.cols == image2.cols && image1.rows > 0 && image1.cols > 0){
		std::vector<float> diffPercentage;
		for(int i = 0; i < image1.cols; i+=square_side){
			for(int j = 0; j < image1.rows; j+=square_side){
				Mat img1ROI(image1, Rect(i, j, square_side, square_side));
				Mat img2ROI(image2, Rect(i, j, square_side, square_side));
				int diffPixels = countNonZero(img1ROI != img2ROI);
				int nonZero1 = countNonZero(img1ROI);
				int nonZero2 = countNonZero(img2ROI);
				if(nonZero1 > 100){
					std::cout << "Diff: " << 100*diffPixels/((nonZero1+nonZero2)/2) << " diff: "<< diffPixels<<std::endl;
				//	std::cout << "Diff: " << "i: "<< i << " j:"<<j<< " diff:"<< diffPixels << " image1: "<< nonZero1 << " image2: "<< nonZero2<< std::endl;
					diffPercentage.push_back(100*diffPixels/((nonZero1+nonZero2)/2));
				}
			}
		}
		int avg = 0;
		double std = 0;
		for(int i =0; i < diffPercentage.size(); i++){
			avg += diffPercentage[i];
			std += diffPercentage[i]*diffPercentage[i];
		}
		avg /= diffPercentage.size();
		std /= diffPercentage.size();
		std -= (avg * avg);// variance
		std::cout << "variance: "<< std << std::endl;
		std = sqrt(std); // std
		std::cout << "Diff: avg: "<< avg << " std: " << std << std::endl;

		cv::Mat testWrite(1, diffPercentage.size(), CV_32FC1, &(diffPercentage[0]));
		float * testPtr =  testWrite.ptr<float>(0);
		for(int i = 0; i < 10; i++){
			std::cout << "test["<< i <<"]:"<< testPtr[i] << std::endl;
		}
		cv::FileStorage fs("mat.xml", cv::FileStorage::WRITE);
		fs << "mat" << testWrite;
		fs.release();

		cv::FileStorage fs2("mat.xml", cv::FileStorage::READ);
		cv::Mat testRead;
		fs2["mat"] >> testRead;
		fs2.release();

		if(countNonZero(testRead != testWrite)==0){
			std::cout << "Mats are equal!!"<< std::endl;
		}else{
			std::cout << "Mats are different!!"<< std::endl;

		}





	}


//	imwrite("test/out-gray.tif", gray);
//	imwrite("test/out-gray.ppm", gray);

	waitKey();

	return 0;
}

