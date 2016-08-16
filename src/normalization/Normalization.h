/*
 * Normalization.h
 *
 *  Created on: Jun 11, 2014
 *      Author: gteodor
 */

#ifndef NORMALIZATION_H_
#define NORMALIZATION_H_

// Includes to use opencv2/GPU
#include "opencv2/opencv.hpp"

#ifdef WITH_CUDA
#include "opencv2/gpu/gpu.hpp"
#endif 

#include <sys/time.h>

#include "PixelOperations.h"



namespace nscale{

class Normalization {
private:
	static cv::Mat segFG(cv::Mat I, cv::Mat M);
	static void PixelClass(cv::Mat I, cv::Mat o_fg, cv::Mat o_bg, cv::Mat& o_fg_lab, cv::Mat& o_bg_lab);
	static cv::Mat TransferI(cv::Mat fg_lab, cv::Mat fg_mask, float meanT[3], float stdT[3]);
	static cv::Mat bgr2Lab(cv::Mat I);
	static cv::Mat lab2BGR(cv::Mat LAB);
	static int rndint(float n);

public:
	// normalization operations that mimics our matlab code. It uses as an input the BGR image and
	// mean/std of the lab channels computed from the target image using the function targetParameters bellow.
	static cv::Mat normalization(const cv::Mat& originalI, float targetMean[3], float targetStd[3]);
	static void targetParameters(const cv::Mat& originalI, float (&targetMean)[3], float (&targetStd)[3]);



};


}// end nscale
#endif /* NORMALIZATION_H_ */
