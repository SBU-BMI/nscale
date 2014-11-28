#include "opencv2/opencv.hpp"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "NeighborOperations.h"

#include "Logger.h"
#include <stdio.h>


#if defined (WITH_CUDA)
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/stream_accessor.hpp"
#endif

using namespace cv;
using namespace cv::gpu;


int main (int argc, char **argv){

	if(argc != 3){
		printf("Usage: ./imreconTest <inputImage> <connectivity=4|8>");
	}
	Mat input = imread(argv[1], -1);
	int connectivity = atoi(argv[2]);
	std::cout << "Cols: " << input.cols << " Rows: "<< input.rows<< std::endl; 
	if(input.channels() == 3){
		input = nscale::PixelOperations::bgr2gray(input);

		imwrite("in-cpu-watershed-gray.png", input);
		Mat auxInput;
		input.convertTo(auxInput, CV_16U);
		input = auxInput;
		auxInput.release();
		input.convertTo(auxInput, CV_8U);

		imwrite("in-cpu-watershed-gray2.png", auxInput);
	}
	uint64_t t1, t2;

	std::cout << "Cols: " << input.cols << " Rows: "<< input.rows<< std::endl; 
	t1 = cci::common::event::timestampInUS();

	imwrite("in-cpu-watershed.png", input);
	Mat waterResult;
	waterResult = nscale::watershed(input, connectivity);

	t2 = cci::common::event::timestampInUS();
	std::cout << "cpu watershed loop took " << (t2-t1)/1000 << "ms" << std::endl;
	//imwrite("out-cpu-watershed.", waterResult);
	Mat intermediate = waterResult;
	std::stringstream ss;
	ss << "32SC";
		ss << intermediate.channels();


		std::string filename = "waterresult" ;//getFileName(stage, std::string(RAW), ss.str());
		FILE* fid = fopen(filename.c_str(), "wb");
		if (!fid) printf("ERROR: can't open %s to write\n", filename.c_str());

		const unsigned char* imgPtr;
		int elSize = intermediate.elemSize();
		for (int j = 0; j < intermediate.rows; ++j) {
			imgPtr = intermediate.ptr(j);

			fwrite(imgPtr, elSize, intermediate.cols, fid);
		}
		fclose(fid);

	//cciutils::cv::imwriteRaw("out-watershed", waterResult);

	return 0;
}

