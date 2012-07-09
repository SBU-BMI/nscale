/*
 * test.cpp
 *
 *  Created on: Jun 28, 2011
 *      Author: tcpan
 */
#include "opencv2/opencv.hpp"
#include <iostream>
#include <dirent.h>
#include <vector>
#include <errno.h>
#include <time.h>
#include "MorphologicOperations.h"
#include "utils.h"
#include <stdio.h>
#include <omp.h>

#include "opencv2/gpu/gpu.hpp"


pthread_attr_t gomp_thread_attr;
using namespace cv;
using namespace cv::gpu;
using namespace std;

int main (int argc, char **argv){
	if(argc != 5){
		std::cout << "Usage: ./imreconMulticore <marker-img> <mask-img> <#Threads> <tileSize>" <<std::endl;
		exit(1);
	}

	Mat marker = imread(argv[1], -1);
	Mat mask = imread(argv[2], -1);
	int nThreads = atoi(argv[3]);
	int tileSize = atoi(argv[4]);
	omp_set_num_threads(nThreads);

	int zoomFactor = 16;
        if(zoomFactor > 1){
                Mat tempMarker = Mat::zeros((marker.cols*zoomFactor)+2,(marker.rows*zoomFactor)+2, marker.type());
                Mat tempMask = Mat::zeros((mask.cols*zoomFactor)+2 ,(mask.rows*zoomFactor)+2, mask.type());
                for(int x = 0; x < zoomFactor; x++){
                        for(int y = 0; y <zoomFactor; y++){
                                Mat roi(tempMarker, cv::Rect((marker.cols*x)+1, marker.rows*y+1, marker.cols, marker.rows));
                                marker.copyTo(roi);
                                Mat roiMask(tempMask, cv::Rect((mask.cols*x)+1, mask.rows*y+1, mask.cols, mask.rows ));
                                mask.copyTo(roiMask);
                        }
                }
                marker = tempMarker;
                mask = tempMask;
        }
	uint64_t t1 = cciutils::ClockGetTime();
//	Mat recon1 = nscale::imreconstruct<unsigned char>(marker, mask, 8);
	uint64_t t2 = cciutils::ClockGetTime();
	std::cout << "SequentialTime="<< t2-t1 << std::endl;

	Mat marker_border(marker.size() + Size(2,2), marker.type());
	copyMakeBorder(marker, marker_border, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat mask_border(mask.size() + Size(2,2), mask.type());
	copyMakeBorder(mask, mask_border, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	mask.release();marker.release();
	Mat marker_copy(marker_border, Rect(1,1,marker_border.cols-2,marker_border.rows-2));
	Mat mask_copy(mask_border, Rect(1,1,mask_border.cols-2,mask_border.rows-2));

	t1 = cciutils::ClockGetTime();
	Mat reconQueue = nscale::imreconstructParallelQueue<unsigned char>(marker_border,mask_border,8,true, nThreads);
	t2 = cciutils::ClockGetTime();
	std::cout << "QueueTime = "<< t2-t1 << std::endl;

//	t1 = cciutils::ClockGetTime();
//	Mat reconTile = nscale::imreconstructParallelTile<unsigned char>(marker,mask,8,tileSize, nThreads);
//	t2 = cciutils::ClockGetTime();
//	std::cout << "TiledTime = "<< t2-t1 << std::endl;

//	std::cout << "comp reconQueue= "<<countNonZero(recon1!=reconQueue) << std::endl;
//	std::cout << "comp reconTile= "<<countNonZero(recon1!=reconTile) << std::endl;
	return 0;
}

