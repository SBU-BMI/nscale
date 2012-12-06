
#include "opencv2/opencv.hpp"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "Logger.h"

#include <iostream>
#include <iomanip>


using namespace cv;


int main (int argc, char **argv){

	if(argc != 2){
		std::cout << "./distTransform <maskImage>" << std::endl;
		exit(1);
	}
	Mat input = imread(argv[1], -1);
	if(input.data == NULL){
		printf("Failed reading");
		exit(1);
	}
//	std::cout << "input - " << (int) input.ptr(10)[20] << std::endl;

       int zoomFactor = 2; 

        if(zoomFactor > 1){
                Mat tempMask = Mat::zeros((input.cols*zoomFactor) ,(input.rows*zoomFactor), input.type());
                for(int x = 0; x < zoomFactor; x++){
                        for(int y = 0; y <zoomFactor; y++){
                                Mat roiMask(tempMask, cv::Rect((input.cols*x), input.rows*y, input.cols, input.rows ));
                                input.copyTo(roiMask);
                        }
                }
                input = tempMask;
        }



	
	gpu::setDevice(2);
	Mat point(10,10, CV_8UC1);
	point.ones(10,10, CV_8UC1);

	for(int x = 0; x < point.rows; x++){
		uchar* ptr = point.ptr(x);
		for(int y = 0; y < point.cols; y++){
			ptr[y] = 1;
			if(x==1 && y==3){
				ptr[y] = 0;
			}
//			if(x==9 && y==9){
//				ptr[y] = 0;
//			}
//			if(x==6 && y==1){
//				ptr[y] = 0;
//			}


//			std::cout << (int) ptr[y] <<" ";
		}
//		std::cout<<std::endl;
	}
//	uchar *ptr = point.ptr(1);
//	ptr[3] = 0;

	Mat dist(point.size(), CV_32FC1);


	uint64_t t1 = cci::common::event::timestampInUS();
	distanceTransform(input, dist, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "distTransf CPU  took " << t2-t1 <<" ms"<<std::endl;
	dist.release();

	t1 = cci::common::event::timestampInUS();
	Mat queueBasedDist = nscale::distanceTransform(input);
	t2 = cci::common::event::timestampInUS();
	std::cout << "distTranf CPU queue took "<< t2-t1 << " ms" << std::endl;
	queueBasedDist.release();


#if defined (WITH_CUDA)
	GpuMat g_warm(input);
	g_warm.release();
#endif

	t1 = cci::common::event::timestampInUS();
	Mat queueBasedTiled = nscale::distanceTransformParallelTile(input,4096, 8);
	t2 = cci::common::event::timestampInUS();
	std::cout << "distTranf CPU queue tiled took "<< t2-t1 << " ms" << std::endl;
	queueBasedTiled.release();
//	for(int x = 0; x < queueBasedDist.rows; x++){
//		float* ptr = queueBasedTiled.ptr<float>(x);
//		for(int y = 0; y < queueBasedTiled.cols; y++){
//			std::cout << std::setprecision(2) << ptr[y] <<"\t ";
//		}
//		std::cout<<std::endl;
//	}
//
#if defined (WITH_CUDA)
	t1 = cci::common::event::timestampInUS();
	GpuMat g_mask(input);
	t2 = cci::common::event::timestampInUS();
	std::cout << "upload:"<< t2-t1 << std::endl;
	Stream stream;

	t1 = cci::common::event::timestampInUS();
	GpuMat g_distance = nscale::gpu::distanceTransform(g_mask, stream);

	stream.waitForCompletion();
	t2 = cci::common::event::timestampInUS();
	std::cout << "distTransf GPU  took " << t2-t1 <<" ms"<<std::endl;

	t1 = cci::common::event::timestampInUS();
	Mat h_distance(g_distance);
	t2 = cci::common::event::timestampInUS();

	std::cout << "download:"<< t2-t1 << std::endl;
#endif
//	for(int x = 0; x < h_distance.rows; x++){
//		float* ptr = (float*)h_distance.ptr(x);
//		for(int y = 0; y < h_distance.cols; y++){
//			std::cout << std::setprecision(10) << ptr[y] <<"\t ";
//		}
//		std::cout<<std::endl;
//	}

//	Mat diff = (h_distance - dist) > 0.01;
//	std::cout << "NonZero=" << countNonZero(diff) << std::endl;
//	Mat diff = (queueBasedDist - dist) > 0.1 ;
//	std::cout << "NonZeroCPU=" << countNonZero(diff) << std::endl;
//	diff = (queueBasedTiled - dist) > 0.1;
//	std::cout << "NonZeroCPUTiled=" << countNonZero(diff) << std::endl;

//	imwrite("diff.jpg", diff);

//	for(int x = 0; x < queueBasedDist.rows; x++){
//		bool* ptr = diff.ptr<bool>(x);
//		for(int y = 0; y < queueBasedTiled.cols; y++){
//			std::cout << std::setprecision(2) << ptr[y] <<"\t ";
//		}
//		std::cout<<std::endl;
//	}

//	h_distance.release();
//	g_distance.release();
//	g_mask.release();


	return 0;
}

