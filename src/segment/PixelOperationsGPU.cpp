/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "PixelOperations.h"
#include <limits>

#include "precomp.hpp"

#if defined (HAVE_CUDA)
#include "cuda/pixel-ops.cuh"
#endif

using namespace cv;
using namespace cv::gpu;

namespace nscale {


namespace gpu {



#if !defined (HAVE_CUDA)
template <typename T>
GpuMat PixelOperations::invert(const GpuMat& img, Stream& stream) { throw_nogpu(); }
template <typename T>
GpuMat PixelOperations::threshold(const GpuMat& img, T lower, T upper, Stream& stream) { throw_nogpu(); }
void PixelOperations::convertIntToChar(GpuMat& input, GpuMat&result, Stream& stream){ throw_nogpu();};
void PixelOperations::convertIntToCharAndRemoveBorder(GpuMat& input, GpuMat&result, int top, int bottom, int left, int right, Stream& stream){ throw_nogpu();};
void PixelOperations::ColorDeconv( GpuMat& image, const Mat& M, const Mat& b, GpuMat& H, GpuMat& E, Stream& stream, bool BGR2RGB){ throw_nogpu();};
GpuMat PixelOperations::bgr2gray(const GpuMat& img, Stream& stream){ throw_nogpu();};
#else

void PixelOperations::convertIntToChar(GpuMat& input, GpuMat&result, Stream& stream){
	// TODO: check if types/size are okay	
	::nscale::gpu::convertIntToChar(input.rows, input.cols, (int*)input.data, (unsigned char*)result.data,  StreamAccessor::getStream(stream));

}

void PixelOperations::convertIntToCharAndRemoveBorder(GpuMat& input, GpuMat&result, int top, int bottom, int left, int right, Stream& stream){ 

	::nscale::gpu::convertIntToCharAndRemoveBorder(input.rows, input.cols, top, bottom, left, right, (int*)input.data, (unsigned char*)result.data,  StreamAccessor::getStream(stream));

};

void PixelOperations::ColorDeconv( GpuMat& g_image, const Mat& M, const Mat& b, GpuMat& g_H, GpuMat& g_E, Stream& stream, bool BGR2RGB){
	long t1 = cciutils::ClockGetTime();
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

	long t2 = cciutils::ClockGetTime();

	cout << "	Before normalized = "<< t2-t1 <<endl;
	assert(g_image.channels() == 3);
    	//normalized image
	int nr = g_image.rows, nc = g_image.cols;
	GpuMat g_dn = GpuMat(nr, nc, CV_64FC3);
	
	convLoop1(nr, nc, g_image.channels(), g_image, g_dn, StreamAccessor::getStream(stream));

	long t1loop = cciutils::ClockGetTime();
	cout << "	After first loop = "<< t1loop - t2 <<endl;

	GpuMat g_cn = GpuMat(nr, nc, CV_64FC2);
	int dn_channels = g_dn.channels();
	int cn_channels = g_cn.channels();

	GpuMat g_Q;
	stream.enqueueUpload( Q, g_Q);
	stream.waitForCompletion();

	convLoop2(nr, nc, g_cn.channels(), g_cn, g_dn.channels(), g_dn, g_Q, Q.rows, BGR2RGB, StreamAccessor::getStream(stream));

	stream.waitForCompletion();

	long t2loop = cciutils::ClockGetTime();
	cout << "	After 2 loop = "<< t2loop - t1loop <<endl;


	convLoop3(nr, nc, g_cn.channels(), g_cn, g_E, g_H, StreamAccessor::getStream(stream));

	stream.waitForCompletion();
	long t3loop = cciutils::ClockGetTime();
	cout << "	After 3 loop = "<< t3loop - t2loop <<endl;

	g_dn.release();
	g_cn.release();
	g_Q.release();
}

GpuMat PixelOperations::bgr2gray(const GpuMat& img, Stream& stream){ 
	int imageChannels = img.channels();
	assert(imageChannels == 3);
	assert(img.type() == CV_8UC3);

	GpuMat gray = GpuMat(img.size(), CV_8UC1);

	bgr2grayCaller(img.rows, img.cols, img, gray , StreamAccessor::getStream(stream));
	return gray;
};



template <typename T>
GpuMat PixelOperations::invert(const GpuMat& img, Stream& stream) {
	// write the raw image

    const Size size = img.size();
    const int depth = img.depth();
    const int cn = img.channels();

    GpuMat result(size, CV_MAKE_TYPE(depth, cn));

	if (std::numeric_limits<T>::is_integer) {

		if (std::numeric_limits<T>::is_signed) {
			invertIntCaller<T>(size.height, size.width, cn, img, result, StreamAccessor::getStream(stream));
		} else {
			// unsigned int
			invertUIntCaller<T>(size.height, size.width, cn, img, result, StreamAccessor::getStream(stream));
		}

	} else {
		// floating point type
		invertFloatCaller<T>(size.height, size.width, cn, img, result, StreamAccessor::getStream(stream));
	}

    return result;
}



template <typename T>
GpuMat PixelOperations::threshold(const GpuMat& img, T lower, T upper, Stream& stream) {
	// write the raw image
	CV_Assert(img.channels() == 1);

    const Size size = img.size();
    const int depth = img.depth();

    GpuMat result(size, CV_8UC1);

    thresholdCaller<T>(size.height, size.width, img, result, lower, upper, StreamAccessor::getStream(stream));

    return result;
}

#endif

template GpuMat PixelOperations::invert<unsigned char>(const GpuMat&, Stream&);
template GpuMat PixelOperations::threshold<unsigned char>(const GpuMat&, unsigned char, unsigned char, Stream&);
template GpuMat PixelOperations::threshold<float>(const GpuMat&, float, float, Stream&);

}

}


