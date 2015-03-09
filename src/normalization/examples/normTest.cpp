#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <dirent.h>

#include "Logger.h"
#include "Normalization.h"

using namespace cv;
using namespace cv::gpu;


int main (int argc, char **argv){

	if(argc !=2){
		std::cout << "Usage: ./normTest <input image>"<< std::endl;
		exit(1);
	}
	cv::Mat inputImg = cv::imread(argv[1], -1);


	//float meanT[3] = {-0.576, -0.0233, 0.0443};
	float meanT[3] = {-0.176, -0.0033, 0.0143};
	float stdT[3] = {0.2317, 0.0491, 0.0156};

	uint64_t t1, t0;// timing variables
	t0 = cci::common::event::timestampInUS();

	cv::Mat normalized = nscale::Normalization::normalization(inputImg, meanT, stdT);

	t1 = cci::common::event::timestampInUS();
	std::cout << "Normalization Time = "<< t1-t0 <<std::endl;

	cv::imwrite("normalized.tiff", normalized);

	nscale::Normalization::targetParameters(inputImg, meanT, stdT);
	for(int i=0; i < 3; i++){
		std::cout << std::setprecision(12) << "Mean["<<i<<"]="<< meanT[i] << "std["<<i<<"]="<< stdT[i] <<std::endl;
	}
	inputImg.release();
	normalized.release();
	return 0;

}


