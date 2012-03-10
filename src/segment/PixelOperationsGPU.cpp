/*
 * ScanlineOperations.cpp
 *
 *  Created on: Aug 2, 2011
 *      Author: tcpan
 */

#include "PixelOperations.h"
#include <limits>
#include "utils.h"
#include "gpu_utils.h"


//#define HAVE_CUDA


#if defined (HAVE_CUDA)
#include "opencv2/gpu/stream_accessor.hpp"
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
GpuMat PixelOperations::threshold(const GpuMat& img, T lower, bool lower_inclusive, T upper, bool up_inclusive, Stream& stream) { throw_nogpu(); }
template <typename T>
GpuMat PixelOperations::replace(const GpuMat& img, T oldval, T newval, Stream& stream) { throw_nogpu(); }
template <typename T>
GpuMat PixelOperations::divide(const GpuMat& num, const GpuMat& den, Stream& stream) { throw_nogpu(); }
template <typename T>
GpuMat PixelOperations::mod(GpuMat& img, T mod, Stream& stream) { throw_nogpu(); }
template <typename T>
GpuMat PixelOperations::mask(const GpuMat& input, const GpuMat& mask, T background, Stream& stream) { throw_nogpu(); }

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
//	int dn_channels = g_dn.channels();
//	int cn_channels = g_cn.channels();

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
GpuMat PixelOperations::threshold(const GpuMat& img, T lower, bool lower_inclusive, T upper, bool up_inclusive,  Stream& stream) {
	// write the raw image
	CV_Assert(img.channels() == 1);

    const Size size = img.size();
//    const int depth = img.depth();

    GpuMat result(size, CV_8UC1);

    thresholdCaller<T>(size.height, size.width, img, result, lower, lower_inclusive, upper, up_inclusive, StreamAccessor::getStream(stream));

    return result;
}


template <typename T>
GpuMat PixelOperations::replace(const GpuMat& img, T oldval, T newval, Stream& stream) {
	// write the raw image
	CV_Assert(img.channels() == 1);

    const Size size = img.size();
//    const int depth = img.depth();

    GpuMat result(size, img.type());

    replaceCaller<T>(size.height, size.width, img, result, oldval, newval, StreamAccessor::getStream(stream));

    return result;
}


template <typename T>
GpuMat PixelOperations::divide(const GpuMat& num, const GpuMat& den, Stream& stream) {
	CV_Assert(num.cols == den.cols && num.rows == den.rows);
	CV_Assert(num.channels() == den.channels());

//    const int channels = num.channels();

    GpuMat result(num.size(), num.type());

    divideCaller<T>(num.rows, num.cols, num, den, result, StreamAccessor::getStream(stream));

    return result;
}

template <typename T>
GpuMat PixelOperations::mask(const GpuMat& input, const GpuMat& mask, T background, Stream& stream) {
	CV_Assert(input.cols == mask.cols && input.rows == mask.rows);
	CV_Assert(mask.channels() == 1);
	

//    const int channels = input.channels();

    GpuMat result(input.size(), input.type());

    maskCaller<T>(input.rows, input.cols, input, mask, result, background, StreamAccessor::getStream(stream));

    return result;
}

template <typename T>
GpuMat PixelOperations::mod(GpuMat& img, T mod, Stream& stream) {
	// write the raw image
	CV_Assert(img.channels() == 1);
	CV_Assert(std::numeric_limits<T>::is_integer);

    GpuMat result(img.size(), img.type());

    modCaller<T>(img.rows, img.cols, img, result, mod, StreamAccessor::getStream(stream));

    return result;
}

#endif

template GpuMat PixelOperations::invert<unsigned char>(const GpuMat&, Stream&);
template GpuMat PixelOperations::invert<int>(const GpuMat&, Stream&);  // for imfillholes
template GpuMat PixelOperations::invert<float>(const GpuMat&, Stream&);
template GpuMat PixelOperations::threshold<float>(const GpuMat&, float, bool, float, bool, Stream&);
template GpuMat PixelOperations::threshold<double>(const GpuMat&, double, bool, double, bool, Stream&);
template GpuMat PixelOperations::threshold<unsigned char>(const GpuMat&, unsigned char, bool, unsigned char, bool, Stream&);
template GpuMat PixelOperations::threshold<int>(const GpuMat&, int, bool, int, bool, Stream&);
template GpuMat PixelOperations::replace<unsigned char>(const GpuMat&, unsigned char, unsigned char, Stream&);
template GpuMat PixelOperations::replace<int>(const GpuMat&, int, int, Stream&);
template GpuMat PixelOperations::divide<double>(const GpuMat&, const GpuMat&,  Stream&);
template GpuMat PixelOperations::mask<unsigned char>(const GpuMat&, const GpuMat&, unsigned char background, Stream&);
template GpuMat PixelOperations::mask<int>(const GpuMat&, const GpuMat&, int background, Stream&);


template GpuMat PixelOperations::mod<unsigned char>(GpuMat&, unsigned char, Stream&);
template GpuMat PixelOperations::mod<int>(GpuMat&, int, Stream&);


}

}


