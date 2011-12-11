/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *      Author: tcpan
 */

#include <algorithm>
#include <queue>
#include <iostream>
#include <list>
#include <limits>
#include "highgui.h"

#include "utils.h"
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "precomp.hpp"


#if defined (HAVE_CUDA)
#include "cuda/imreconstruct_int_kernel.cuh"
#include "cuda/imreconstruct_float_kernel.cuh"
#include "cuda/imreconstruct_binary_kernel.cuh"
#include "cuda/imrecon_queue_int_kernel.cuh"

#endif

using namespace std;

extern "C" int listComputation(void *d_Data, int dataElements, unsigned char *seeds, unsigned char *image, int ncols, int nrows);
extern "C" int morphRecon(int *d_input_list, int dataElements, int *d_seeds, unsigned char *d_image, int ncols, int nrows);
extern "C" int morphReconVector(int nImages, int **h_InputListPtr, int* h_ListSize, int **h_Seeds, unsigned char **h_images, int* ncols, int* nrows, int connectivity);

namespace nscale {

namespace gpu {

using namespace cv;
using namespace cv::gpu;


template <typename T>
inline void propagate(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, T* iPtr, T* oPtr, const T& pval) {
	T qval = oPtr[x];
	T ival = iPtr[x];
	if ((qval < pval) && (ival != qval)) {
		oPtr[x] = min(pval, ival);
		xQ.push(x);
		yQ.push(y);
	}
}




#if !defined (HAVE_CUDA)

template <typename T>
GpuMat imreconstruct(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) { throw_nogpu();}
template <typename T>
GpuMat imreconstructQueue(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream) { throw_nogpu();}
template <typename T>
GpuMat imreconstructQ(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) { throw_nogpu();}
// Operates on BINARY IMAGES ONLY
template <typename T>
GpuMat bwselect(const GpuMat& binaryImage, const GpuMat& seeds, int connectivity, Stream& stream) { throw_nogpu();}
template <typename T>
GpuMat imreconstructBinary(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {throw_nogpu();}
template <typename T>
GpuMat imfillHoles(const GpuMat& image, bool binary, int connectivity, Stream& stream) { throw_nogpu();}
template <typename T>
#else

/** slightly optimized serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 this is slightly optimized by avoiding conditional where possible.
 */
/**
 * based on implementation from Pavlo
 */
template <typename T>
GpuMat imreconstruct(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_32FC1 || seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_32FC1 || image.type() == CV_8UC1);

    // allocate results
	GpuMat marker = createContinuous(seeds.size(), seeds.type());
	GpuMat markerFirstPass = createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();
	if (std::numeric_limits<T>::is_integer) {
	    iter = imreconstructIntCaller<T>(marker.data, mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream), markerFirstPass.data);
	} else {
		iter = imreconstructFloatCaller<T>(marker.data, mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
	}
	stream.waitForCompletion();
	mask.release();
	// get the result out
	return marker;
}


void gold_imreconstructIntCallerBuildQueue(GpuMat& marker, GpuMat mask, int *d_queuePixels, int queueSize){
// CPU gold initQueue
	// structure that will hold pixels candidate to propagation calculated by the GPU 
	list<int> gpu_stl_list;

	// this list will hold, hopefully, the same set of pixels as the gpu_list, which are calculated
	// bellow in this function for validation purposes 
	list<int> cpu_stl_list;
	
	// Create memory space to store list of propagation candidate pixels 	
	int *queueCPU = (int *) malloc(sizeof(int) * queueSize);

	// Copy pixels from the GPU to the host memory
	cudaMemcpy(queueCPU, d_queuePixels, sizeof(int) * queueSize, cudaMemcpyDeviceToHost);

	// Initializes the gpu list of pixels with data copied from GPU
	for(int i = 0; i < queueSize; i++){
		gpu_stl_list.push_back(queueCPU[i]);
	}

	// Sort elements to guarantee the same ordering of the CPU list
	gpu_stl_list.sort();

	// release array used to copy data from GPU
	free(queueCPU);

	// Download intermediary results calculated by the GPU (marker) in order to 
	// perform calculate pixels that are candidate to the propagation phase
	Mat markerCPUAfter(marker);
	Mat maskCPUAfter(mask);

	// Gold CPU code that finds propagation candidate pixels, and build a list with them to compare to the GPU results.	
	for (int y = 0; y < marker.rows-1; y++) {
		unsigned char* yrowMarkerPtr = markerCPUAfter.ptr<unsigned char>(y);
		unsigned char* yplusrowMarkerPtr = markerCPUAfter.ptr<unsigned char>(y+1);

		unsigned char* yrowMaskPtr = maskCPUAfter.ptr<unsigned char>(y);
		unsigned char* yplusrowMaskPtr = maskCPUAfter.ptr<unsigned char>(y+1);

		for (int x = 0; x < marker.cols - 1; x++) {
			unsigned char pval = yrowMarkerPtr[x];
			
			// right neighbor
			unsigned char rMarker = yrowMarkerPtr[x+1];
			unsigned char rMask = yrowMaskPtr[x+1];

			if( (rMarker < min(pval, rMask)) ){
				cpu_stl_list.push_back((y*marker.cols + x));
				continue;
			}
			// down neighbor
			unsigned char dMarker = yplusrowMarkerPtr[x];
			unsigned char dMask = yplusrowMaskPtr[x];

			if( (dMarker < min(pval, dMask)) ){
				cpu_stl_list.push_back((y*marker.cols + x));
			}
		}
	}
	// End identification of pixels.

	// Sort list to guarantee the same order as GPU code. I guees
	cpu_stl_list.sort();

	cout << "	Queue size = "<< cpu_stl_list.size()<<endl;
	
	// Compare CPU and GPU lists 
	if(cpu_stl_list.size() == gpu_stl_list.size()){
		// Are equal?
		if(!std::equal(cpu_stl_list.begin(), cpu_stl_list.end(), gpu_stl_list.begin())){
			cout << "	Error: content of CPU and GPU lists are different!" <<endl;
			exit(1);
		}else{
			cout << "	CPU list equals to list built by GPU. Well done." <<endl;
		}
	}else{
		cout << "	Sizes of lists are different! CPU list size = "<< cpu_stl_list.size() << " GPU list size = "<< gpu_stl_list.size()<<endl;
		exit(1);
	}
	// write intermediary results to disk for further visual inspection purposes only.
	imwrite("test/out-first-pass-gpu.pbm", markerCPUAfter);
}

template <typename T>
vector<GpuMat> imreconstructQueueThroughput(vector<GpuMat> & seeds, vector<GpuMat> & image, int connectivity, int nItFirstPass, Stream& stream) {
	cout << "Throughput 2"<<endl;
	uint64_t t11 = cciutils::ClockGetTime();
	assert(seeds.size() == image.size());

	vector<GpuMat> maskVector(seeds.size());

	for(int i = 0; i < seeds.size(); i++){
		CV_Assert(image[i].channels() == 1);
		CV_Assert(seeds[i].channels() == 1);
		CV_Assert(seeds[i].type() == CV_8UC1);
		CV_Assert(image[i].type() == CV_8UC1);

		maskVector[i] = createContinuous(image[i].size(), image[i].type());
		stream.enqueueCopy(image[i], maskVector[i]);
	}

	vector<GpuMat> markerVector(seeds.size());
	for(int i = 0; i < seeds.size(); i++){
		// allocate results data. Which is a copy of seeds and voids the user data from being modified
		markerVector[i] = createContinuous(seeds[i].size(), seeds[i].type());

		// Copy seeds to marker
		stream.enqueueCopy(seeds[i], markerVector[i]);
	}

	// Yep. Wait til copies are complete
	stream.waitForCompletion();

	uint64_t endUpload = cciutils::ClockGetTime();
//	cout << "	Init+upload = "<< endUpload-t11 <<endl;

	int *queuePixelsGPUSizeVector = (int*)malloc(sizeof(int) * seeds.size());
	int **queuePixelsGPUVector = (int **)malloc(sizeof(int*) * seeds.size());

	for(int i = 0; i < seeds.size();i++){

		int queuePixelsGPUSize;
		int *g_queuePixelsGPU = ::nscale::gpu::imreconstructIntCallerBuildQueue<T>(markerVector[i].data, maskVector[i].data, markerVector[i].cols, markerVector[i].rows, connectivity, queuePixelsGPUSize, nItFirstPass, StreamAccessor::getStream(stream));

		queuePixelsGPUSizeVector[i] = queuePixelsGPUSize;
		queuePixelsGPUVector[i] = g_queuePixelsGPU;
//		printf("	Queue[%d]Ptr = %p size = %d\n", i, queuePixelsGPUVector[i], queuePixelsGPUSizeVector[i]);
	}
	uint64_t imreconBuildEnd = cciutils::ClockGetTime(); 
	cout << "	FirstPass+buildqueue = "<< imreconBuildEnd-endUpload <<endl;
	// Gold function implemented on CPU to validate calculate of pixels candidate to propagation in next step
/////	gold_imreconstructIntCallerBuildQueue(marker, mask1, g_queuePixelsGPU, queuePixelsGPUSize);


	stream.waitForCompletion();
	vector<GpuMat> markerIntVector(seeds.size());
	for(int i = 0; i < seeds.size();i++){
		// Create an int version of the input marker
		markerIntVector[i] = createContinuous(seeds[i].size(), CV_32S);
	
		// Perform appropriate conversion from uchar to int
		markerVector[i].convertTo(markerIntVector[i], CV_32S);

	}
	uint64_t t31 = cciutils::ClockGetTime();

	int **markerIntPtr = (int **)malloc(sizeof(int*) * seeds.size());
	unsigned char **maskUcharPtr = (unsigned char **)malloc(sizeof(unsigned char*) * seeds.size());
	int *cols = (int*)malloc(sizeof(int) * seeds.size());
	int *rows = (int*)malloc(sizeof(int) * seeds.size());

	// prepare arrays with information that are used inside the queue propagation kernel
	for(int i = 0; i < seeds.size();i++){
		markerIntPtr[i] = (int*)markerIntVector[i].data;
		maskUcharPtr[i] = maskVector[i].data;
		cols[i] = maskVector[i].cols;
		rows[i] = maskVector[i].rows;
	}

	// apply morphological reconstruction using the Queue based algorithm
	morphReconVector(seeds.size(), queuePixelsGPUVector, queuePixelsGPUSizeVector, markerIntPtr, maskUcharPtr, cols, rows, connectivity);
	uint64_t t41 = cciutils::ClockGetTime();
	cout << "	queue time = "<< t41-t31<<endl;

	for(int i = 0; i < seeds.size(); i++){
		::nscale::gpu::PixelOperations::convertIntToChar(markerIntVector[i], markerVector[i], stream);
	}

	for(int i = 0; i < seeds.size(); i++){
		maskVector[i].release();
		markerIntVector[i].release();
	}

	free(queuePixelsGPUSizeVector);
	free(queuePixelsGPUVector);
	free(markerIntPtr);
	free(maskUcharPtr);
	free(cols);
	free(rows);

	return markerVector;
}


template <typename T>
GpuMat imreconstructQueue(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_8UC1);

	uint64_t t11 = cciutils::ClockGetTime();

	GpuMat mask1;

//	if(!image.isContinuous()){
		mask1 = createContinuous(image.size(), image.type());
		stream.enqueueCopy(image, mask1);
//	}else{
//		mask1 = image;
//	}

	// allocate results data. Which is a copy of seeds and voids the user data from being modified
	GpuMat marker = createContinuous(seeds.size(), seeds.type());

	// Copy seeds to marker
	stream.enqueueCopy(seeds, marker);
///	std::cout << " is seeds continuous? " << (seeds.isContinuous() ? "YES" : "NO") << std::endl;
///
///	// TODO: this is unecessary, unless input data is not continuous
///	GpuMat mask1 = createContinuous(image.size(), image.type());
///	stream.enqueueCopy(image, mask1);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	// Yep. Wait til copies are complete
	stream.waitForCompletion();

	uint64_t endUpload = cciutils::ClockGetTime();
//	cout << "	Init+upload = "<< endUpload-t11 <<endl;

	// Will be used to store size of pixels candidate to propagation
	int queuePixelsGPUSize;
	int numIterationsFirstPass = 10;
	// Perform first pass (parallell raster and anti-raster) as Pavlo's code does
	int *g_queuePixelsGPU = ::nscale::gpu::imreconstructIntCallerBuildQueue<T>(marker.data, mask1.data, marker.cols, marker.rows, connectivity, queuePixelsGPUSize, numIterationsFirstPass, StreamAccessor::getStream(stream));
	uint64_t imreconBuildEnd = cciutils::ClockGetTime(); 
	cout << "	FirstPass+buildqueue = "<< imreconBuildEnd-endUpload <<endl;
	// Gold function implemente on CPU to validate calculate of pixels candidate to propagation in next step
//	gold_imreconstructIntCallerBuildQueue(marker, mask1, g_queuePixelsGPU, queuePixelsGPUSize);

	// Create an int version of the input marker
	GpuMat g_markerInt_1 = createContinuous(seeds.size(), CV_32S);

	// Perform appropriate convertion from uchar to int
	marker.convertTo(g_markerInt_1, CV_32S);
	
	uint64_t t31 = cciutils::ClockGetTime();
	cout << "	ConvertToInt = "<< t31-imreconBuildEnd<<endl;
	// apply morphological reconstruction using the Queue based algorithm
	morphRecon(g_queuePixelsGPU, queuePixelsGPUSize, (int*)g_markerInt_1.data, mask1.data, mask1.cols, mask1.rows);


	uint64_t t41 = cciutils::ClockGetTime();
//	cout << "	queue time = "<< t41-t31<<endl;
//	cout << "End morphRecon. time = " << t41-t11 <<endl;
	// This is char matrix is used to save the uchar version of the result. 
	// It is computed from the int version of the result (g_makerInt).
//	GpuMat g_markerChar_1 = createContinuous(seeds.size(), CV_8UC1);

	::nscale::gpu::PixelOperations::convertIntToChar(g_markerInt_1, marker, stream);

//	if(!image.isContinuous()){
		mask1.release();
//	}
	uint64_t t21 = cciutils::ClockGetTime();

	g_markerInt_1.release();
	std::cout << "    total time = " << t21-t11 << "ms for. ConvertToChar = "<< t21-t41 << std::endl;


	return marker;
}

/**
 * based on implementation from Pavlo
 */
template <typename T>
GpuMat imreconstructQ(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_32FC1 || seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_32FC1 || image.type() == CV_8UC1);

//	Mat c_seeds;
//	seeds.download(c_seeds);
//	Mat c_image;
//	image.download(c_image);
//	Mat c_output = ::nscale::imreconstruct<T>(c_seeds, c_image, connectivity);
//
//	GpuMat output(c_output);
//	return output;
	T mn = cciutils::min<T>();

    // allocate results
	GpuMat temp1;
	copyMakeBorder(seeds, temp1, 2, 2, 2, 2, Scalar(mn), stream);
	GpuMat marker = createContinuous(temp1.size(), temp1.type());
	stream.enqueueCopy(temp1, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat temp2;
	copyMakeBorder(image, temp2, 2, 2, 2, 2, Scalar(mn), stream);
	GpuMat mask = createContinuous(temp2.size(), temp2.type());
	stream.enqueueCopy(temp2, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();
	temp1.release();
	temp2.release();
	iter = imreconQueueIntCaller<T>(marker.data, mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
	StreamAccessor::getStream(stream);

	stream.waitForCompletion();
	mask.release();

	Rect roi = Rect(2, 2, image.cols, image.rows);
	GpuMat output(marker, roi);
	marker.release();

    return output;
}



/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 connectivity is either 4 or 8, default 4.  background is assume to be 0, foreground is assumed to be NOT 0.

 */
template <typename T>
GpuMat imreconstructBinary(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {

	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_8UC1);

    // allocate results
	GpuMat marker = createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();

  	iter = imreconstructBinaryCaller<T>(marker.data, mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
	stream.waitForCompletion();
	mask.release();

	return marker;

//	Mat c_seeds;
//	seeds.download(c_seeds);
//	Mat c_image;
//	image.download(c_image);
//	Mat c_output = ::nscale::imreconstructBinary<T>(c_seeds, c_image, connectivity);
//
//	GpuMat output(c_output);
//
//	return output;

}


//
//template <typename T>
//Mat imfill(const Mat& image, const Mat& seeds, bool binary, int connectivity) {
//	CV_Assert(image.channels() == 1);
//	CV_Assert(seeds.channels() == 1);
//
//	/* MatLAB imfill code:
//	 *     mask = imcomplement(I);
//    marker = mask;
//    marker(:) = 0;
//    marker(locations) = mask(locations);
//    marker = imreconstruct(marker, mask, conn);
//    I2 = I | marker;
//	 */
//
//	Mat mask = nscale::gpu::PixelOperations::invert<T>(image, stream);  // validated
//
//	Mat marker = Mat::zeros(mask.size(), mask.type());
//
//	mask.copyTo(marker, seeds);
//
//	if (binary) marker = imreconstructBinary<T>(marker, mask, connectivity);
//	else marker = imreconstruct<T>(marker, mask, connectivity);
//
//	return image | marker;
//}
//
template <typename T>
GpuMat imfillHoles(const GpuMat& image, bool binary, int connectivity, Stream& stream) {
	CV_Assert(image.channels() == 1);

	/* MatLAB imfill hole code:
    if islogical(I)
        mask = uint8(I);
    else
        mask = I;
    end
    mask = padarray(mask, ones(1,ndims(mask)), -Inf, 'both');

    marker = mask;
    idx = cell(1,ndims(I));
    for k = 1:ndims(I)
        idx{k} = 2:(size(marker,k) - 1);
    end
    marker(idx{:}) = Inf;

    mask = imcomplement(mask);
    marker = imcomplement(marker);
    I2 = imreconstruct(marker, mask, conn);
    I2 = imcomplement(I2);
    I2 = I2(idx{:});

    if islogical(I)
        I2 = I2 ~= 0;
    end
	 */

	T mn = cciutils::min<T>();
	T mx = std::numeric_limits<T>::max();
	Rect roi = Rect(1, 1, image.cols, image.rows);

	// copy the input and pad with -inf.
	GpuMat mask2;
	copyMakeBorder(image, mask2, 1, 1, 1, 1, Scalar(mn), stream);
	// create marker with inf inside and -inf at border, and take its complement
	GpuMat marker;
	GpuMat marker2(image.size(), image.type());
	stream.enqueueMemSet(marker2, Scalar(mn));

	// them make the border - OpenCV does not replicate the values when one Mat is a region of another.
	copyMakeBorder(marker2, marker, 1, 1, 1, 1, Scalar(mx), stream);

	// now do the work...
	GpuMat mask = nscale::gpu::PixelOperations::invert<T>(mask2, stream);
	stream.waitForCompletion();
	marker2.release();
	mask2.release();

	uint64_t t1 = cciutils::ClockGetTime();
	GpuMat output2;
	if (binary) output2 = imreconstructBinary<T>(marker, mask, connectivity, stream);
	else output2 = imreconstruct<T>(marker, mask, connectivity, stream);
//	output2 = imreconstruct2<T>(marker, mask, connectivity, stream);
	stream.waitForCompletion();
	uint64_t t2 = cciutils::ClockGetTime();
//	std::cout << "    imfill hole imrecon took " << t2-t1 << "ms" << std::endl;
	stream.waitForCompletion();
	marker.release();
	mask.release();

	GpuMat output3 = nscale::gpu::PixelOperations::invert<T>(output2, stream);
	stream.waitForCompletion();
	output2.release();
	GpuMat output(output3, roi);
	output3.release();

	return output;
}

// Operates on BINARY IMAGES ONLY
template <typename T>
GpuMat bwselect(const GpuMat& binaryImage, const GpuMat& seeds, int connectivity, Stream& stream) {
	CV_Assert(binaryImage.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	// only works for binary images.  ~I and max-I are the same....

	/** adopted from bwselect and imfill
	 * bwselet:
	 * seed_indices = sub2ind(size(BW), r(:), c(:));
		BW2 = imfill(~BW, seed_indices, n);
		BW2 = BW2 & BW;
	 *
	 * imfill:
	 * see imfill function.
	 */

	// general simplified bwselect

	// since binary, seeds already have the same values as binary images
	// at the selected places.  If not, the marker will be forced to 0 by imrecon.

//	GpuMat marker = imreconstruct2<T>(seeds, binaryImage, connectivity, stream);
	return imreconstructBinary<T>(seeds, binaryImage, connectivity, stream);

	// no need to and between marker and binaryImage - since marker is always <= binary image
}
//
//// Operates on BINARY IMAGES ONLY
//// ideally, output should be 64 bit unsigned.
////	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
//// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
////  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
//// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
////  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
//Mat_<int> bwlabel(const Mat& binaryImage, bool contourOnly, int connectivity) {
//	CV_Assert(binaryImage.channels() == 1);
//	// only works for binary images.
//
//	int lineThickness = CV_FILLED;
//	if (contourOnly) lineThickness = 1;
//
//	// based on example from
//	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
//	// only outputs
//
////	Mat_<int> output = Mat_<int>::zeros(binaryImage.size());
////	Mat input = binaryImage.clone();
//	Mat_<int> output = Mat_<int>::zeros(binaryImage.size() + Size(2,2));
//	Mat input(output.size(), binaryImage.type());
//	copyMakeBorder(binaryImage, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);
//
//	std::vector<std::vector<Point> > contours;
//	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.
//
//	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
//	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//	std::cout << "num contours = " << contours.size() << std::endl;
//
//	int color = 1;
//	uint64_t t1 = cciutils::ClockGetTime();
//	// iterate over all top level contours (all siblings, draw with own label color
//	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
//		// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//		drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
//	}
//	uint64_t t2 = cciutils::ClockGetTime();
//	std::cout << "    bwlabel drawing took " << t2-t1 << "ms" << std::endl;
//
//	return output(Rect(1,1,binaryImage.cols, binaryImage.rows));
//}
//
//// Operates on BINARY IMAGES ONLY
//// ideally, output should be 64 bit unsigned.
////	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
//// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
////  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
//// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
////  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
//template <typename T>
//Mat bwlabelFiltered(const Mat& binaryImage, bool binaryOutput,
//		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
//		bool contourOnly, int connectivity) {
//	// only works for binary images.
//	if (contourFilter == NULL) {
//		return bwlabel(binaryImage, contourOnly, connectivity);
//	}
//	CV_Assert(binaryImage.channels() == 1);
//
//	int lineThickness = CV_FILLED;
//	if (contourOnly) lineThickness = 1;
//
//	// based on example from
//	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
//	// only outputs
//
//	Mat output = Mat::zeros(binaryImage.size(), (binaryOutput ? binaryImage.type() : CV_32S));
//	Mat input = binaryImage.clone();
//
//	std::vector<std::vector<Point> > contours;
//	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.
//
//	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
//	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//
//	if (binaryOutput) {
//		Scalar color(std::numeric_limits<T>::max());
//		// iterate over all top level contours (all siblings, draw with own label color
//		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
//			if (contourFilter(contours, hierarchy, idx)) {
//				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//				drawContours( output, contours, idx, color, lineThickness, connectivity, hierarchy );
//			}
//		}
//
//	} else {
//		int color = 1;
//		// iterate over all top level contours (all siblings, draw with own label color
//		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
//			if (contourFilter(contours, hierarchy, idx)) {
//				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//				drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
//			}
//		}
//	}
//	return output;
//}
//
//// inclusive min, exclusive max
//bool contourAreaFilter(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {
//
//	int area = contourArea(contours[idx]);
//	int circum = contours[idx].size() / 2 + 1;
//
//	area += circum;
//
//	if (area < minArea) return false;
//
//	int i = hierarchy[idx][2];
//	for ( ; i >= 0; i = hierarchy[i][0]) {
//		area -= (contourArea(contours[i]) + contours[i].size() / 2 + 1);
//		if (area < minArea) return false;
//	}
//
//	if (area >= maxArea) return false;
////	std::cout << idx << " total area = " << area << std::endl;
//
//	return true;
//}
//
//// inclusive min, exclusive max
//template <typename T>
//Mat bwareaopen(const Mat& binaryImage, int minSize, int maxSize, int connectivity) {
//	// only works for binary images.
//	CV_Assert(binaryImage.channels() == 1);
//	CV_Assert(minSize > 0);
//	CV_Assert(maxSize > 0);
//
//	// based on example from
//	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
//	// only outputs
//
//	Mat_<T> output = Mat_<T>::zeros(binaryImage.size());
//	Mat input = binaryImage.clone();
//
//	std::vector<std::vector<Point> > contours;
//	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.
//
//	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
//	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
//	std::cout << "num contours = " << contours.size() << std::endl;
//
//	Scalar color(std::numeric_limits<T>::max());
//	// iterate over all top level contours (all siblings, draw with own label color
//	for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
//		if (contourAreaFilter(contours, hierarchy, idx, minSize, maxSize)) {
//			// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
//			drawContours(output, contours, idx, color, CV_FILLED, connectivity, hierarchy );
//		}
//	}
//	return output;
//}
//
//template <typename T>
//Mat imhmin(const Mat& image, T h, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	//	IMHMIN(I,H) suppresses all minima in I whose depth is less than h
//	// MatLAB implementation:
//	/**
//	 *
//		I = imcomplement(I);
//		I2 = imreconstruct(imsubtract(I,h), I, conn);
//		I2 = imcomplement(I2);
//	 *
//	 */
//	Mat mask = nscale::gpu::PixelOperations::invert<T>(image, stream);
//	Mat marker = mask - h;
//	Mat output = imreconstruct<T>(marker, mask, connectivity);
//	return nscale::gpu::PixelOperations::invert<T>(output,stream);
//}
//
//// input should have foreground > 0, and 0 for background
//Mat_<int> watershed2(const Mat& origImage, const Mat_<float>& image, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	/*
//	 * MatLAB implementation:
//		cc = bwconncomp(imregionalmin(A, conn), conn);
//		L = watershed_meyer(A,conn,cc);
//
//	 */
//
//	Mat minima = localMinima<float>(image, connectivity);
//	Mat_<int> labels = bwlabel(minima, true, connectivity);
//
//	// convert to grayscale
//	// now scale, shift, clear background.
//	double mmin, mmax;
//	minMaxLoc(image, &mmin, &mmax);
//	std::cout << " image: min=" << mmin << " max=" << mmax<< std::endl;
//	double range = (mmax - mmin);
//	double scaling = std::numeric_limits<uchar>::max() / range;
//	Mat shifted(image.size(), CV_8U);
//	image.convertTo(shifted, CV_8U, scaling, 0.0);
//
//
//	Mat image3(shifted.size(), CV_8UC3);
//	cvtColor(shifted, image3, CV_GRAY2BGR);
//
//	watershed(origImage, labels);
//
//	mmin, mmax;
//	minMaxLoc(labels, &mmin, &mmax);
//	std::cout << " watershed: min=" << mmin << " max=" << mmax<< std::endl;
//
//
//	return labels;
//}
//
//// only works with integer images
//template <typename T>
//Mat_<uchar> localMaxima(const Mat& image, int connectivity) {
//	CV_Assert(image.channels() == 1);
//
//	// use morphologic reconstruction.
//	Mat marker = image - 1;
//	Mat_<uchar> candidates =
//			marker < imreconstruct<T>(marker, image, connectivity);
////	candidates marked as 0 because floodfill with mask will fill only 0's
////	return (image - imreconstruct(marker, image, 8)) >= (1 - std::numeric_limits<T>::epsilon());
//	//return candidates;
//
//	// now check the candidates
//	// first pad the border
//	T mn = cciutils::min<T>();
//	T mx = std::numeric_limits<uchar>::max();
//	Mat_<uchar> output(candidates.size() + Size(2,2));
//	copyMakeBorder(candidates, output, 1, 1, 1, 1, BORDER_CONSTANT, mx);
//	Mat input(image.size() + Size(2,2), image.type());
//	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);
//
//	int maxy = input.rows-1;
//	int maxx = input.cols-1;
//	int xminus, xplus;
//	T val;
//	T *iPtr, *iPtrMinus, *iPtrPlus;
//	uchar *oPtr;
//	Rect reg(1, 1, image.cols, image.rows);
//	Scalar zero(0);
//	Scalar smx(mx);
//	Range xrange(1, maxx);
//	Range yrange(1, maxy);
//	Mat inputBlock = input(yrange, xrange);
//
//	// next iterate over image, and set candidates that are non-max to 0 (via floodfill)
//	for (int y = 1; y < maxy; ++y) {
//
//		iPtr = input.ptr<T>(y);
//		iPtrMinus = input.ptr<T>(y-1);
//		iPtrPlus = input.ptr<T>(y+1);
//		oPtr = output.ptr<uchar>(y);
//
//		for (int x = 1; x < maxx; ++x) {
//
//			// not a candidate, continue.
//			if (oPtr[x] > 0) continue;
//
//			xminus = x-1;
//			xplus = x+1;
//
//			val = iPtr[x];
//			// compare values
//
//			// 4 connected
//			if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) || (val < iPtr[xminus]) || (val < iPtr[xplus])) {
//				// flood with type minimum value (only time when the whole image may have mn is if it's flat)
//				floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
//				continue;
//			}
//
//			// 8 connected
//			if (connectivity == 8) {
//				if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {
//					// flood with type minimum value (only time when the whole image may have mn is if it's flat)
//					floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
//					continue;
//				}
//			}
//
//		}
//	}
//	return output(yrange, xrange) == 0;  // similar to bitwise not.
//
//}
//
//template <typename T>
//Mat_<uchar> localMinima(const Mat& image, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	Mat cimage = nscale::gpu::PixelOperations::invert<T>(image, stream);
//	return localMaxima<T>(cimage, connectivity);
//}




//template Mat imfill<uchar>(const Mat& image, const Mat& seeds, bool binary, int connectivity);
//template Mat bwselect<uchar>(const Mat& binaryImage, const Mat& seeds, int connectivity);
//template Mat bwlabelFiltered<uchar>(const Mat& binaryImage, bool binaryOutput,
//		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
//		bool contourOnly, int connectivity);
//template Mat bwareaopen<uchar>(const Mat& binaryImage, int minSize, int maxSize, int connectivity);
//template Mat imhmin(const Mat& image, uchar h, int connectivity);
//template Mat imhmin(const Mat& image, float h, int connectivity);
//template Mat_<uchar> localMaxima<float>(const Mat& image, int connectivity);
//template Mat_<uchar> localMinima<float>(const Mat& image, int connectivity);
//template Mat_<uchar> localMaxima<uchar>(const Mat& image, int connectivity);
//template Mat_<uchar> localMinima<uchar>(const Mat& image, int connectivity);

#endif

//template GpuMat imreconstruct<float>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstruct<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
//template GpuMat imreconstruct<float>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstruct<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstructQueue<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template vector<GpuMat> imreconstructQueueThroughput<unsigned char>(vector<GpuMat> & seeds, vector<GpuMat> & image, int connectivity, int nItFirstPass, Stream& stream);
template GpuMat bwselect<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstructBinary<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstructBinary<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imfillHoles<unsigned char>(const GpuMat&, bool, int, Stream&);

template GpuMat imreconstructQ<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstructQ<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);


}

}

