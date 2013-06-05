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

#include "Logger.h"
#include "TypeUtils.h"

#include "gpu_utils.h"

#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "NeighborOperations.h"

//#define WITH_CUDA

#if defined (WITH_CUDA)
#include "opencv2/gpu/stream_accessor.hpp"
#include "cuda/imreconstruct_int_kernel.cuh"
#include "cuda/imreconstruct_float_kernel.cuh"
#include "cuda/imreconstruct_binary_kernel.cuh"
#include "cuda/imrecon_queue_int_kernel.cuh"
#include "cuda/watershed-ca-korbes.cuh"
#include "cuda/watershed-dw-korbes.cuh"
#include "cuda/ccl_uf.cuh"
#include "cuda/pixel-ops.cuh"
#include "cuda/global_queue_dist.cuh"

extern "C" int listComputation(void *d_Data, int dataElements, unsigned char *seeds, unsigned char *image, int ncols, int nrows);
extern "C" int morphRecon(int *d_input_list, int dataElements, int *d_seeds, unsigned char *d_image, int ncols, int nrows);
extern "C" int morphReconVector(int nImages, int **h_InputListPtr, int* h_ListSize, int **h_Seeds, unsigned char **h_images, int* ncols, int* nrows, int connectivity);

extern "C" int morphReconSpeedup( int *g_InputListPtr, int h_ListSize, int *g_Seed, unsigned char *g_Image, int h_ncols, int h_nrows, int connectivity, int nBlocks, float queue_increase_factor);
extern "C" int morphReconSpeedupFloat( int *g_InputListPtr, int h_ListSize, int *g_Seed, int *g_Image, int h_ncols, int h_nrows, int connectivity, int nBlocks, float queue_increase_factor);
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;

namespace nscale {

namespace gpu {


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




#if !defined (WITH_CUDA)

template <typename T>
GpuMat imreconstruct(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) { throw_nogpu();}
template <typename T>
GpuMat imreconstructQueue(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream) { throw_nogpu();}
template <typename T>
vector<GpuMat> imreconstructQueueThroughput(vector<GpuMat> & seeds, vector<GpuMat> & image, int connectivity, int nItFirstPass, Stream& stream) {throw_nogpu();};
template <typename T>
GpuMat imreconstructQueueSpeedup(GpuMat &seeds, GpuMat &image, int connectivity, int nItFirstPass, Stream& stream, int nBlocks, bool binary) {throw_nogpu();};
GpuMat imreconstructQueueSpeedupFloat(GpuMat &seeds, GpuMat &image, int connectivity, int nItFirstPass, Stream& stream, int nBlocks) {throw_nogpu();};

template <typename T>
GpuMat imreconstructQ(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) { throw_nogpu();}
//// Operates on BINARY IMAGES ONLY
template <typename T>
GpuMat bwselect(const GpuMat& binaryImage, const GpuMat& seeds, int connectivity, Stream& stream) { throw_nogpu();}
template <typename T>
GpuMat imreconstructBinary(const GpuMat& seeds, const GpuMat& image, int connectivity, Stream& stream, unsigned int& iter) {throw_nogpu();}
template <typename T>
GpuMat imfillHoles(const GpuMat& image, bool binary, int connectivity, Stream& stream) { throw_nogpu();}


GpuMat bwlabel(const GpuMat& binaryImage, int connectivity, bool relab, Stream& stream) { throw_nogpu(); }

//// input should have foreground > 0, and 0 for background
//GpuMat watershedCA(const GpuMat& origImage, const GpuMat& image, int connectivity, Stream& stream) { throw_nogpu(); }
// input should have foreground > 0, and 0 for background
GpuMat watershedDW(const GpuMat& origImage, const GpuMat& image, int connectivity, Stream& stream) { throw_nogpu(); }
// input should have foreground > 0, and 0 for background
template <typename T>
GpuMat imhmin(const GpuMat& image, T h, int connectivity, Stream& stream) { throw_nogpu(); }
template <typename T>
GpuMat morphOpen(const GpuMat& image, const Mat& kernel, Stream& stream) {throw_nogpu(); }
template <typename T>
GpuMat morphErode(const GpuMat& image, const Mat& kernel, Stream& stream) {throw_nogpu(); }
template <typename T>
GpuMat morphDilate(const GpuMat& image, const Mat& kernel, Stream& stream) {throw_nogpu(); }

GpuMat distanceTransform(const GpuMat& mask, Stream& stream) {throw_nogpu(); }

#else

GpuMat distanceTransform(const GpuMat& mask, Stream& stream, bool calcDist, int tIdX, int tIdY, int tileSize, int imgCols) {
	CV_Assert(mask.channels() == 1);
	CV_Assert(mask.type() ==  CV_8UC1);
	
	// create nearest neighbors map
	GpuMat g_nearestNeighbors(mask.size(), CV_32S);

	int g_queue_size;
	int queue_propagation_increase =2;
	int retCode=1;
	
	// try the computation untill it succeeds without "exploding the queue size".
	do{
//		std::cout << "Call build queue rows= "<< mask.rows<< " cols="<< mask.cols << std::endl;

		uint64_t t1 = cci::common::event::timestampInUS();
		// build queue with propagation frontier pixels
		int *g_queue = nscale::gpu::distQueueBuildCaller(mask.rows, mask.cols, mask, g_nearestNeighbors, g_queue_size, StreamAccessor::getStream(stream));

		uint64_t t2 = cci::common::event::timestampInUS();
		std::cout << "After Call build queue - queue size = "<< g_queue_size << " elapsedTime:"<<t2-t1 <<std::endl;

		stream.waitForCompletion();

		// Calculate the propagation phase, where the nearest neighbors to the fontier elements are propated.
		retCode = nscale::gpu::distTransformPropagation( g_queue, g_queue_size, mask , g_nearestNeighbors, mask.cols, mask.rows, queue_propagation_increase);
//		std::cout << "retCode = "<< retCode<< std::endl;
		queue_propagation_increase *=4;
	}while(retCode);

	if(calcDist){
		uint64_t t1 = cci::common::event::timestampInUS();
		GpuMat g_distanceMap(mask.size(), CV_32FC1);
		nscale::gpu::distMapCalcCaller(g_nearestNeighbors.rows, g_nearestNeighbors.cols, g_nearestNeighbors, g_distanceMap, StreamAccessor::getStream(stream));
		stream.waitForCompletion();
		uint64_t t2 = cci::common::event::timestampInUS();
		std::cout << "DistMapCalc Time:" << t2-t1 << std::endl;

		g_nearestNeighbors.release();
		return g_distanceMap;
	}else{
		std::cout << "neighborCalcCaller"<<std::endl;
		// calc global value of data calculated for this tile
		::nscale::gpu::neighborCalcCaller(g_nearestNeighbors.rows, g_nearestNeighbors.cols, g_nearestNeighbors, tIdX, tIdY, tileSize, imgCols, StreamAccessor::getStream(stream));
		return g_nearestNeighbors;
	}
}

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
	CV_Assert(seeds.type() == CV_32FC1 || seeds.type() == CV_8UC1 || seeds.type() == CV_32SC1);
	CV_Assert(image.type() == CV_32FC1 || image.type() == CV_8UC1 || image.type() == CV_32SC1);

    // allocate results
	GpuMat marker = createContinuous(seeds.size(), seeds.type());
//	GpuMat markerFirstPass = createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();
	if (std::numeric_limits<T>::is_integer) {
	    iter = imreconstructIntCaller<T>((T*)marker.data, (T*)mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
	    //, markerFirstPass.data);
	} else {
		iter = imreconstructFloatCaller<T>((T*)marker.data, (T*)mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
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
	//imwrite("test/out-first-pass-gpu.pbm", markerCPUAfter);
}

template <typename T>
GpuMat imreconstructQueueSpeedup(GpuMat &seeds, GpuMat &image, int connectivity, int nItFirstPass, Stream& stream, int nBlocks, bool binary) {
//	cout << "Throughput 2"<<endl;
	uint64_t t11 = cci::common::event::timestampInUS();
	CV_Assert(seeds.size() == image.size());
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_8UC1);
	CV_Assert(image.type() == CV_8UC1);

	// Copy mask to GPU
	GpuMat g_mask = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, g_mask);

	// Copy marker to GPU
	GpuMat g_marker =  createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, g_marker);

	// Wait til copies are complete
	stream.waitForCompletion();

	GpuMat g_markerInt;

	uint64_t endUpload = cci::common::event::timestampInUS();
//	cout << "	Init+upload = "<< endUpload-t11 <<endl;
	float queue_increase_factor =2;
	int number_raster_passes = nItFirstPass;

	int morphRetCode = 0;

	do{
		int queuePixelsGPUSize;
		// Perform Raster and Anti-Raster passes and build queue used in the queue-based computation phase
		int *g_queuePixelsGPU;
		if(binary){
			std::cout << "Binary queue"<<std::endl;
			g_queuePixelsGPU = ::nscale::gpu::imreconstructBinaryCallerBuildQueue<T>(g_marker.data, g_mask.data, g_mask.cols, g_mask.rows, connectivity, queuePixelsGPUSize, number_raster_passes, StreamAccessor::getStream(stream));
		}else{
			g_queuePixelsGPU = ::nscale::gpu::imreconstructIntCallerBuildQueue<T>(g_marker.data, g_mask.data, g_mask.cols, g_mask.rows, connectivity, queuePixelsGPUSize, number_raster_passes, StreamAccessor::getStream(stream));
		}
		uint64_t imreconBuildEnd = cci::common::event::timestampInUS(); 
//		cout << "	FirstPass+buildqueue = "<< imreconBuildEnd-endUpload <<endl;

		// Gold function implemented on CPU to validate calculate of pixels candidate to propagation in next step
		/////	gold_imreconstructIntCallerBuildQueue(marker, mask1, g_queuePixelsGPU, queuePixelsGPUSize);

		stream.waitForCompletion();
	
		if(g_markerInt.rows == 0)
			g_markerInt = createContinuous(seeds.size(), CV_32S);

		g_marker.convertTo(g_markerInt, CV_32S);

		uint64_t t31 = cci::common::event::timestampInUS();


		// apply morphological reconstruction using the Queue based algorithm
		morphRetCode = morphReconSpeedup(g_queuePixelsGPU, queuePixelsGPUSize, (int*)g_markerInt.data, g_mask.data, g_mask.cols, g_mask.rows, connectivity, nBlocks, queue_increase_factor);
		uint64_t t41 = cci::common::event::timestampInUS();
//		cout << "	queue time = "<< t41-t31<<" nBlocks="<< nBlocks<<" morphRetCode="<<morphRetCode<<endl;

		::nscale::gpu::PixelOperations::convertIntToChar(g_markerInt, g_marker, stream);
		number_raster_passes = 0;

	 	queue_increase_factor =4;
	// If the queue size has been exceed, run the reconstruction again.
	}while(morphRetCode == 1);



	g_mask.release();
	g_markerInt.release();

	return g_marker;
}


GpuMat imreconstructQueueSpeedupFloat(GpuMat &seeds, GpuMat &image, int connectivity, int nItFirstPass, Stream& stream, int nBlocks) {
//	cout << "Throughput 2"<<endl;
	uint64_t t11 = cci::common::event::timestampInUS();
	CV_Assert(seeds.size() == image.size());
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);

	// This is the float guy.
	CV_Assert(seeds.type() == CV_32FC1);
	CV_Assert(image.type() == CV_32FC1);

	// Copy mask to GPU
	GpuMat g_mask_f = createContinuous(image.size(), image.type());
	stream.enqueueCopy(image, g_mask_f);

	// Copy marker to GPU
	GpuMat g_marker_f =  createContinuous(seeds.size(), seeds.type());
	stream.enqueueCopy(seeds, g_marker_f);


	// Will hold int to float convertion of mask and marker.
	GpuMat g_mask_i = createContinuous(image.size(), CV_32SC1);
	// Integer version of marker
	GpuMat g_marker_i =  createContinuous(seeds.size(), CV_32SC1);

	// Wait til copies are complete
//	stream.waitForCompletion();

	::nscale::gpu::convFloatToIntOrderedCaller(g_mask_f.rows, g_mask_f.cols, g_mask_f, g_mask_i, StreamAccessor::getStream(stream));

	::nscale::gpu::convFloatToIntOrderedCaller(g_marker_f.rows, g_marker_f.cols, g_marker_f, g_marker_i, StreamAccessor::getStream(stream));
	stream.waitForCompletion();

	g_mask_f.release();

	GpuMat g_markerInt;

	uint64_t endUpload = cci::common::event::timestampInUS();
	cout << "	Init+upload = "<< endUpload-t11 <<endl;
	float queue_increase_factor =2;
	int number_raster_passes = nItFirstPass;

	int morphRetCode = 0;

	do{
		int queuePixelsGPUSize;
		// Perform Raster and Anti-Raster passes and build queue used in the queue-based computation phase
		int *g_queuePixelsGPU;

		g_queuePixelsGPU = ::nscale::gpu::imreconstructIntCallerBuildQueue<int>((int*)g_marker_i.data, (const int*)g_mask_i.data, g_mask_i.cols, g_mask_i.rows, connectivity, queuePixelsGPUSize, number_raster_passes, StreamAccessor::getStream(stream));
		std::cout << "QueueSize = "<< queuePixelsGPUSize << " numRasters="<< number_raster_passes <<std::endl;

		stream.waitForCompletion();
		uint64_t imreconBuildEnd = cci::common::event::timestampInUS();
		cout << "	FirstPass+buildqueue = "<< imreconBuildEnd-endUpload <<endl;
	
//		// apply morphological reconstruction using the Queue based algorithm
		morphRetCode = morphReconSpeedupFloat(g_queuePixelsGPU, queuePixelsGPUSize, (int*)g_marker_i.data, (int*)g_mask_i.data, g_mask_i.cols, g_mask_i.rows, connectivity, nBlocks, queue_increase_factor);
		uint64_t t41 = cci::common::event::timestampInUS();
		cout << "	queue time = "<< t41-imreconBuildEnd<<" nBlocks="<< nBlocks<<" morphRetCode="<<morphRetCode<<endl;
		number_raster_passes = 0;

	 	queue_increase_factor*=2;
		// If the queue size has been exceed, run the reconstruction again.
	}while(morphRetCode == 1);
//
//
//
//	g_mask.release();
//	g_markerInt.release();

	::nscale::gpu::convIntToFloatOrderedCaller(g_marker_i.rows, g_marker_i.cols, g_marker_i, g_marker_f, StreamAccessor::getStream(stream));
	stream.waitForCompletion();

	g_mask_i.release();
	g_marker_i.release();

	return g_marker_f;
}





template <typename T>
vector<GpuMat> imreconstructQueueThroughput(vector<GpuMat> & seeds, vector<GpuMat> & image, int connectivity, int nItFirstPass, Stream& stream) {
	cout << "Throughput 2"<<endl;
//	uint64_t t11 = cci::common::event::timestampInUS();
	assert(seeds.size() == image.size());

	vector<GpuMat> maskVector(seeds.size());

	for(unsigned int i = 0; i < seeds.size(); i++){
		CV_Assert(image[i].channels() == 1);
		CV_Assert(seeds[i].channels() == 1);
		CV_Assert(seeds[i].type() == CV_8UC1);
		CV_Assert(image[i].type() == CV_8UC1);

		maskVector[i] = createContinuous(image[i].size(), image[i].type());
		stream.enqueueCopy(image[i], maskVector[i]);
	}

	vector<GpuMat> markerVector(seeds.size());
	for(unsigned int i = 0; i < seeds.size(); i++){
		// allocate results data. Which is a copy of seeds and voids the user data from being modified
		markerVector[i] = createContinuous(seeds[i].size(), seeds[i].type());

		// Copy seeds to marker
		stream.enqueueCopy(seeds[i], markerVector[i]);
	}

	// Yep. Wait til copies are complete
	stream.waitForCompletion();

	uint64_t endUpload = cci::common::event::timestampInUS();
//	cout << "	Init+upload = "<< endUpload-t11 <<endl;

	int *queuePixelsGPUSizeVector = (int*)malloc(sizeof(int) * seeds.size());
	int **queuePixelsGPUVector = (int **)malloc(sizeof(int*) * seeds.size());

	for(unsigned int i = 0; i < seeds.size();i++){

		int queuePixelsGPUSize;
		int *g_queuePixelsGPU = ::nscale::gpu::imreconstructIntCallerBuildQueue<T>(markerVector[i].data, maskVector[i].data, markerVector[i].cols, markerVector[i].rows, connectivity, queuePixelsGPUSize, nItFirstPass, StreamAccessor::getStream(stream));

		queuePixelsGPUSizeVector[i] = queuePixelsGPUSize;
		queuePixelsGPUVector[i] = g_queuePixelsGPU;
//		printf("	Queue[%d]Ptr = %p size = %d\n", i, queuePixelsGPUVector[i], queuePixelsGPUSizeVector[i]);
	}
	uint64_t imreconBuildEnd = cci::common::event::timestampInUS(); 
	cout << "	FirstPass+buildqueue = "<< imreconBuildEnd-endUpload <<endl;
	// Gold function implemented on CPU to validate calculate of pixels candidate to propagation in next step
/////	gold_imreconstructIntCallerBuildQueue(marker, mask1, g_queuePixelsGPU, queuePixelsGPUSize);


	stream.waitForCompletion();
	vector<GpuMat> markerIntVector(seeds.size());
	for(unsigned int i = 0; i < seeds.size();i++){
		// Create an int version of the input marker
		markerIntVector[i] = createContinuous(seeds[i].size(), CV_32S);
	
		// Perform appropriate conversion from unsigned char to int
		markerVector[i].convertTo(markerIntVector[i], CV_32S);

	}
	uint64_t t31 = cci::common::event::timestampInUS();

	int **markerIntPtr = (int **)malloc(sizeof(int*) * seeds.size());
	unsigned char **maskUcharPtr = (unsigned char **)malloc(sizeof(unsigned char*) * seeds.size());
	int *cols = (int*)malloc(sizeof(int) * seeds.size());
	int *rows = (int*)malloc(sizeof(int) * seeds.size());

	// prepare arrays with information that are used inside the queue propagation kernel
	for(unsigned int i = 0; i < seeds.size();i++){
		markerIntPtr[i] = (int*)markerIntVector[i].data;
		maskUcharPtr[i] = maskVector[i].data;
		cols[i] = maskVector[i].cols;
		rows[i] = maskVector[i].rows;
	}

	// apply morphological reconstruction using the Queue based algorithm
	morphReconVector(seeds.size(), queuePixelsGPUVector, queuePixelsGPUSizeVector, markerIntPtr, maskUcharPtr, cols, rows, connectivity);
	uint64_t t41 = cci::common::event::timestampInUS();
	cout << "	queue time = "<< t41-t31<<endl;

	for(unsigned int i = 0; i < seeds.size(); i++){
		::nscale::gpu::PixelOperations::convertIntToChar(markerIntVector[i], markerVector[i], stream);
	}

	for(unsigned int i = 0; i < seeds.size(); i++){
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

	uint64_t t11 = cci::common::event::timestampInUS();

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

	uint64_t endUpload = cci::common::event::timestampInUS();
//	cout << "	Init+upload = "<< endUpload-t11 <<endl;

	// Will be used to store size of pixels candidate to propagation
	int queuePixelsGPUSize;
	int numIterationsFirstPass = 10;
	// Perform first pass (parallell raster and anti-raster) as Pavlo's code does
	int *g_queuePixelsGPU = ::nscale::gpu::imreconstructIntCallerBuildQueue<T>(marker.data, mask1.data, marker.cols, marker.rows, connectivity, queuePixelsGPUSize, numIterationsFirstPass, StreamAccessor::getStream(stream));
	uint64_t imreconBuildEnd = cci::common::event::timestampInUS(); 
	cout << "	FirstPass+buildqueue = "<< imreconBuildEnd-endUpload <<endl;
	// Gold function implemente on CPU to validate calculate of pixels candidate to propagation in next step
//	gold_imreconstructIntCallerBuildQueue(marker, mask1, g_queuePixelsGPU, queuePixelsGPUSize);

	// Create an int version of the input marker
	GpuMat g_markerInt_1 = createContinuous(seeds.size(), CV_32S);

	// Perform appropriate convertion from uchar to int
	marker.convertTo(g_markerInt_1, CV_32S);
	
	uint64_t t31 = cci::common::event::timestampInUS();
	cout << "	ConvertToInt = "<< t31-imreconBuildEnd<<endl;
	// apply morphological reconstruction using the Queue based algorithm
	morphRecon(g_queuePixelsGPU, queuePixelsGPUSize, (int*)g_markerInt_1.data, mask1.data, mask1.cols, mask1.rows);


	uint64_t t41 = cci::common::event::timestampInUS();
//	cout << "	queue time = "<< t41-t31<<endl;
//	cout << "End morphRecon. time = " << t41-t11 <<endl;
	// This is char matrix is used to save the uchar version of the result. 
	// It is computed from the int version of the result (g_makerInt).
//	GpuMat g_markerChar_1 = createContinuous(seeds.size(), CV_8UC1);

	::nscale::gpu::PixelOperations::convertIntToChar(g_markerInt_1, marker, stream);

//	if(!image.isContinuous()){
		mask1.release();
//	}
	uint64_t t21 = cci::common::event::timestampInUS();

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
	CV_Assert(seeds.type() == CV_32FC1 || seeds.type() == CV_8UC1 || seeds.type() == CV_32SC1);
	CV_Assert(image.type() == CV_32FC1 || image.type() == CV_8UC1 || image.type() == CV_32SC1);

//	Mat c_seeds;
//	seeds.download(c_seeds);
//	Mat c_image;
//	image.download(c_image);
//	Mat c_output = ::nscale::imreconstruct<T>(c_seeds, c_image, connectivity);
//
//	GpuMat output(c_output);
//	return output;
	T mn = cci::common::type::min<T>();

    // allocate results
	GpuMat temp1;
	copyMakeBorder(seeds, temp1, 2, 2, 2, 2, BORDER_CONSTANT ,Scalar(mn), stream);
	GpuMat marker = createContinuous(temp1.size(), temp1.type());
	stream.enqueueCopy(temp1, marker);
//	std::cout << " is marker continuous? " << (marker.isContinuous() ? "YES" : "NO") << std::endl;

	GpuMat temp2;
	copyMakeBorder(image, temp2, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(mn), stream);
	GpuMat mask = createContinuous(temp2.size(), temp2.type());
	stream.enqueueCopy(temp2, mask);
//	std::cout << " is mask continuous? " << (mask.isContinuous() ? "YES" : "NO") << std::endl;

	stream.waitForCompletion();
	temp1.release();
	temp2.release();
	iter = imreconQueueIntCaller<T>((T*)marker.data, (T*)mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
	StreamAccessor::getStream(stream);

	stream.waitForCompletion();
	mask.release();

	Rect roi = Rect(2, 2, image.cols, image.rows);
	GpuMat output(image.size(), marker.type());
	stream.enqueueCopy(marker(roi), output);
	stream.waitForCompletion();
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

    iter = imreconstructBinaryCaller<T>((T*)marker.data, (T*)mask.data, seeds.cols, seeds.rows, connectivity, StreamAccessor::getStream(stream));
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
//	if (binary == true) marker = imreconstructBinary<T>(marker, mask, connectivity);
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

	T mn = cci::common::type::min<T>();
	T mx = std::numeric_limits<T>::max();


	printf("fillHoles: input.rows:%d\n", image.rows);
	// copy the input and pad with -inf.
	GpuMat mask2;
	copyMakeBorder(image, mask2, 1, 1, 1, 1, BORDER_CONSTANT, Scalar_<T>(mn), stream);
	// create marker with inf inside and -inf at border, and take its complement
	GpuMat marker2(image.size(), image.type());
	stream.enqueueMemSet(marker2, Scalar_<T>(mn));
	stream.waitForCompletion();

	// them make the border - OpenCV does not replicate the values when one Mat is a region of another.
	GpuMat marker;
	copyMakeBorder(marker2, marker, 1, 1, 1, 1, BORDER_CONSTANT, Scalar_<T>(mx), stream);
	stream.waitForCompletion();

	// now do the work...
	GpuMat mask = nscale::gpu::PixelOperations::invert<T>(mask2, stream);
	stream.waitForCompletion();
	marker2.release();
	mask2.release();
	GpuMat output2;
	if (binary == true) {
//		output2 = imreconstructBinary<T>(marker, mask, connectivity, stream);
		std::cout << "Call imrecont binary"<<std::endl;
		if(connectivity == 4){
			output2 = imreconstructQueueSpeedup<unsigned char>(marker, mask, connectivity, 2, stream, 12, binary);
		}else{
			output2 = imreconstructQueueSpeedup<unsigned char>(marker, mask, connectivity, 1, stream, 12, binary);
		}
	}
	else if (sizeof(T) == 1 && !(std::numeric_limits<T>::is_signed)) {
		output2 = imreconstructQueueSpeedup<unsigned char>(marker, mask, connectivity, 1, stream);

	} else {
		output2 = imreconstruct<T>(marker, mask, connectivity, stream);
	}
	stream.waitForCompletion();
	marker.release();
	mask.release();

	GpuMat output3 = nscale::gpu::PixelOperations::invert<T>(output2, stream);
	stream.waitForCompletion();
	output2.release();

	Rect roi = Rect(1, 1, image.cols, image.rows);
	GpuMat output(image.size(), output3.type());
	stream.enqueueCopy(output3(roi), output);
	stream.waitForCompletion();
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
//	GpuMat mask(binaryImage.size(), binaryImage.type());
//	bitwise_and(binaryImage, binaryImage, mask, seeds, stream);
	unsigned char bg = 0;
	GpuMat mask1 = ::nscale::gpu::PixelOperations::mask(binaryImage, seeds, bg, stream);

//	GpuMat marker = imreconstruct2<T>(seeds, binaryImage, connectivity, stream);
	return imreconstructBinary<T>(mask1, binaryImage, connectivity, stream);

	// no need to and between marker and binaryImage - since marker is always <= binary image
}


// Operates on BINARY IMAGES ONLY
// ideally, output should be 64 bit unsigned.
//	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
//  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
//  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
GpuMat bwlabel(const GpuMat& binaryImage, int connectivity, bool relab, Stream& stream) {
	CV_Assert(binaryImage.channels() == 1);
	CV_Assert(binaryImage.type() == CV_8U);
	// only works for binary images.

	GpuMat input = createContinuous(binaryImage.size(), binaryImage.type());
	stream.enqueueCopy(binaryImage, input);

	GpuMat output = createContinuous(binaryImage.size(), CV_32SC1);

	::nscale::gpu::CCL((unsigned char*)input.data, input.cols, input.rows, (int*)output.data, -1, connectivity, StreamAccessor::getStream(stream));
	stream.waitForCompletion();
	if (relab == true) {
		int j = ::nscale::gpu::relabel(output.cols, output.rows, (int*)output.data, -1, StreamAccessor::getStream(stream));
		printf("gpu bwlabel num components = %d\n", j);
	}

	input.release();

	return output;

}

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
//	if (contourOnly == true) lineThickness = 1;
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
//	if (binaryOutput == true) {
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

// inclusive min, exclusive max
GpuMat bwareaopen(const GpuMat& binaryImage, bool labeled, bool flatten, int minSize, int maxSize, int connectivity, int& count, Stream& stream) {

	CV_Assert(binaryImage.channels() == 1);
	if (labeled == false)
		CV_Assert(binaryImage.type() == CV_8U);
	else
		CV_Assert(binaryImage.type() == CV_32S);

	// only works for binary images.

	GpuMat input = createContinuous(binaryImage.size(), binaryImage.type());
	stream.enqueueCopy(binaryImage, input);
	GpuMat output;

	if (labeled == false) {
		GpuMat temp = createContinuous(binaryImage.size(), CV_32SC1);
		::nscale::gpu::CCL((unsigned char*)input.data, input.cols, input.rows, (int*)temp.data, -1, connectivity, StreamAccessor::getStream(stream));
		count = ::nscale::gpu::areaThreshold(temp.cols, temp.rows, (int*)temp.data, -1, minSize, maxSize, StreamAccessor::getStream(stream));
		printf("inside bwareaopen: count unlabeled = %d\n", count);
		if (flatten == true) output = ::nscale::gpu::PixelOperations::threshold(temp, 0, true, std::numeric_limits<int>::max(), true, stream);
		else output = temp;
		stream.waitForCompletion();
		temp.release();
	} else {
		count = ::nscale::gpu::areaThreshold(input.cols, input.rows, (int*)input.data, -1, minSize, maxSize, StreamAccessor::getStream(stream));
		printf("inside bwareaopen: count labeled = %d\n", count);
			if (flatten == true) output = ::nscale::gpu::PixelOperations::threshold(input, 0, true, std::numeric_limits<int>::max(), true, stream);
		else output = input;
		stream.waitForCompletion();
	}

	input.release();

	return output;
}

GpuMat bwareaopen2(const GpuMat& binaryImage, bool labeled, bool flatten, int minSize, int maxSize, int connectivity, int& count, Stream& stream) {

	CV_Assert(binaryImage.channels() == 1);
	if (labeled == false)
		CV_Assert(binaryImage.type() == CV_8U);
	else
		CV_Assert(binaryImage.type() == CV_32S);
	// only works for binary images.

	GpuMat input = createContinuous(binaryImage.size(), binaryImage.type());
	stream.enqueueCopy(binaryImage, input);
	GpuMat output;

	if (labeled == false) {
		GpuMat temp = createContinuous(binaryImage.size(), CV_32SC1);
		::nscale::gpu::CCL((unsigned char*)input.data, input.cols, input.rows, (int*)temp.data, -1, connectivity, StreamAccessor::getStream(stream));
		count = ::nscale::gpu::areaThreshold2(temp.cols, temp.rows, (int*)temp.data, -1, minSize, maxSize, StreamAccessor::getStream(stream));
		printf("inside bwareaopen2: count unlabeled = %d\n", count);
		if (flatten == true) output = ::nscale::gpu::PixelOperations::threshold(temp, 0, true, std::numeric_limits<int>::max(), true, stream);
		else output = temp;
		stream.waitForCompletion();
	} else {
		count = ::nscale::gpu::areaThreshold2(input.cols, input.rows, (int*)input.data, -1, minSize, maxSize, StreamAccessor::getStream(stream));
		printf("inside bwareaopen2: count labeled = %d\n", count);
		if (flatten == true) output = ::nscale::gpu::PixelOperations::threshold(input, 0, true, std::numeric_limits<int>::max(), true, stream);
		else output = input;
		stream.waitForCompletion();
	}

	input.release();


	return output;
}

template <typename T>
GpuMat imhmin(const GpuMat& image, T h, int connectivity, Stream& stream) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	//	IMHMIN(I,H) suppresses all minima in I whose depth is less than h
	// MatLAB implementation:
	/**
	 *
		I = imcomplement(I);
		I2 = imreconstruct(imsubtract(I,h), I, conn);
		I2 = imcomplement(I2);
	 *
	 */
	GpuMat mask = nscale::gpu::PixelOperations::invert<T>(image, stream);
	stream.waitForCompletion();
	GpuMat marker(mask.size(), mask.type());
	subtract(mask, Scalar(h), marker);
	GpuMat recon;
	GpuMat recon2;
	if (sizeof(T) == 1 && !(std::numeric_limits<T>::is_signed)){
		recon = nscale::gpu::imreconstructQueueSpeedup<unsigned char>(marker, mask, connectivity, 1, stream);
	}else{
		recon = imreconstructQueueSpeedupFloat(marker, mask, connectivity, 1, stream);

//		recon = nscale::gpu::imreconstruct<T>(marker, mask, connectivity, stream);
//		GpuMat res(recon.size(), recon.type());
//		subtract(recon, recon2, res);
//		GpuMat res = recon - recon2;
//		std::cout << "NonZero = " << cv::gpu::countNonZero(res) << std::endl;
		
	}

	GpuMat output = nscale::gpu::PixelOperations::invert<T>(recon,stream);
	stream.waitForCompletion();
	recon.release();
	marker.release();
	mask.release();
	return output;
}


//// input should have foreground > 0, and 0 for background
//GpuMat watershedCA(const GpuMat& origImage, const GpuMat& image, int connectivity, Stream& stream) {
//
//	CV_Assert(image.channels() == 1);
//	CV_Assert(image.type() == CV_32FC1);
////	CV_Assert(image.type() == CV_8UC1);
//
//
//	/*
//	 * MatLAB implementation:
//		cc = bwconncomp(imregionalmin(A, conn), conn);
//		L = watershed_meyer(A,conn,cc);
//	 */
//
//	// this implementation requires seed image.
//	Mat h_img(image.size(), image.type());
//	stream.enqueueDownload(image, h_img);
//	Mat minima = localMinima<float>(h_img, connectivity);
////imwrite("test-minima.pbm", minima);
//	Mat_<int> h_labels = bwlabel(minima, false, connectivity);
////imwrite("test-bwlabel.png", labels);
//	GpuMat d_labels(h_labels.size(), h_labels.type());
//	stream.enqueueUpload(h_labels, d_labels);
//	GpuMat seeds(d_labels.rows + 2, d_labels.cols + 2, d_labels.type());
//	copyMakeBorder(d_labels, seeds, 1, 1, 1, 1, Scalar(0), stream);
//	stream.waitForCompletion();
//	h_img.release();
//	minima.release();
//	h_labels.release();
//	d_labels.release();
//
//
//
//	GpuMat input = createContinuous(image.size().height + 2, image.size().width + 2, image.type());
//	copyMakeBorder(image, input, 1, 1, 1, 1, Scalar(0), stream);
//
//	// allocate results
//	GpuMat labels = createContinuous(image.size().height + 2, image.size().width + 2, CV_32SC1);
//	stream.enqueueMemSet(labels, Scalar(0));
//	stream.waitForCompletion();
//
//	// here call the cuda function.
//	ca::ws_kauffmann((int*)(labels.data), (float*)(input.data), (int*)seeds.data, input.cols, input.rows, connectivity);
////	::ws_kauffmann((int*)labels.data, (unsigned char*)input.data, input.cols, input.rows, connectivity);
//
//    stream.waitForCompletion();
//    input.release();
//    seeds.release();
//
//    GpuMat output(image.size(), labels.type());
//    stream.enqueueCopy(labels(Rect(1,1, image.cols, image.rows)), output);
//    stream.waitForCompletion();
//    labels.release();
//    return output;
//}

// input should have foreground > 0, and 0 for background
GpuMat watershedDW(const GpuMat& maskImage, const GpuMat& image, int background, int connectivity, Stream& stream) {

	CV_Assert(image.channels() == 1);
	CV_Assert(image.type() == CV_32FC1);
//	CV_Assert(image.type() == CV_8UC1);


	/*
	 * MatLAB implementation:
		cc = bwconncomp(imregionalmin(A, conn), conn);
		L = watershed_meyer(A,conn,cc);
	 */

	// this implementation does not require seed image, nor the original image (at all).

	GpuMat input = createContinuous(image.size().height + 2, image.size().width + 2, image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0.0), stream);

	// allocate results
	GpuMat labels = createContinuous(image.size().height + 2, image.size().width + 2, CV_32SC1);
	stream.enqueueMemSet(labels, Scalar(0));
	stream.waitForCompletion();

	// here call the cuda function.
	dw::giwatershed((int*)(labels.data), (float*)(input.data), input.cols, input.rows, connectivity, StreamAccessor::getStream(stream));
//	::giwatershed((int*)labels.data, (unsigned*)input.data, input.cols, input.rows, connectivity);
	// this code does generate borders but not between touching blobs.  for the borders, they are 4 connected.
	stream.waitForCompletion();

	GpuMat mask = createContinuous(maskImage.size().height + 2, maskImage.size().width + 2, maskImage.type());
	copyMakeBorder(maskImage, mask, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0), stream);
	// allocate results
	GpuMat labels2 = createContinuous(labels.size(), labels.type());
	stream.enqueueMemSet(labels2, Scalar(background));
	stream.waitForCompletion();


	// now clean up the borders.
	dw::giwatershed_cleanup(mask, labels, labels2, input.cols, input.rows, background, connectivity, StreamAccessor::getStream(stream));
	stream.waitForCompletion();

	// and generate the missing borders
	GpuMat bordered = NeighborOperations::border(labels2, background, connectivity, stream);

    stream.waitForCompletion();
    labels.release();
    mask.release();
    input.release();
    labels2.release();

    GpuMat output(image.size(), bordered.type());
    stream.enqueueCopy(bordered(Rect(1,1, image.cols, image.rows)), output);
    stream.waitForCompletion();
    bordered.release();
    return output;
}


//
//// only works with integer images
//template <typename T>
//Mat_<unsigned char> localMaxima(const Mat& image, int connectivity) {
//	CV_Assert(image.channels() == 1);
//
//	// use morphologic reconstruction.
//	Mat marker = image - 1;
//	Mat_<unsigned char> candidates =
//			marker < imreconstruct<T>(marker, image, connectivity);
////	candidates marked as 0 because floodfill with mask will fill only 0's
////	return (image - imreconstruct(marker, image, 8)) >= (1 - std::numeric_limits<T>::epsilon());
//	//return candidates;
//
//	// now check the candidates
//	// first pad the border
//	T mn = cci::common::type::min<T>();
//	T mx = std::numeric_limits<unsigned char>::max();
//	Mat_<unsigned char> output(candidates.size() + Size(2,2));
//	copyMakeBorder(candidates, output, 1, 1, 1, 1, BORDER_CONSTANT, mx);
//	Mat input(image.size() + Size(2,2), image.type());
//	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);
//
//	int maxy = input.rows-1;
//	int maxx = input.cols-1;
//	int xminus, xplus;
//	T val;
//	T *iPtr, *iPtrMinus, *iPtrPlus;
//	unsigned char *oPtr;
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
//		oPtr = output.ptr<unsigned char>(y);
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
//Mat_<unsigned char> localMinima(const Mat& image, int connectivity) {
//	// only works for intensity images.
//	CV_Assert(image.channels() == 1);
//
//	Mat cimage = nscale::gpu::PixelOperations::invert<T>(image, stream);
//	return localMaxima<T>(cimage, connectivity);
//}


// this is different for CPU and GPU.  CPU uses the whole range.  GPU uses only the inside.
// also GPU seems to leave a 1 pixel wide column and row on the right and bottom side that is not processed.
// hence the bw+1
// also, morphologyEx does not work correctly for open.  it still generates a border.
template <typename T>
GpuMat morphOpen(const GpuMat& image, const Mat& kernel, Stream& stream) {
//	CV_Assert(kernel.cols == kernel.rows );
//	CV_Assert(kernel.cols > 1);
//	CV_Assert((kernel.cols % 2) == 1 );
//
//	int bw = (kernel.cols - 1) / 2;
//
//
//	GpuMat t_img;
//	copyMakeBorder(image, t_img, bw, bw+1, bw, bw+1, Scalar(std::numeric_limits<T>::max()), stream);
//	// this is same for CPU and GPU
//
////	if (bw > 1) {
////		::cv::Mat output(t_img.size(), t_img.type());
////		t_img.download(output);
////		imwrite("test-input-gpu.ppm", output);
////	}
//	GpuMat t_erode(t_img.size(), t_img.type());
//	erode(t_img, t_erode, kernel, Point(-1,-1), 1, stream);
////	if (bw > 1) {
////		::cv::Mat output(t_erode.size(), t_erode.type());
////		t_erode.download(output);
////		imwrite("test-erode-gpu.ppm", output);
////	}
//	Rect roi = Rect(bw, bw, image.cols, image.rows);
//    GpuMat g_erode(image.size(), t_erode.type());
//    stream.enqueueCopy(t_erode(roi), g_erode);
//    stream.waitForCompletion();
//    t_erode.release();
//
//	GpuMat t_erode2;
//	copyMakeBorder(g_erode, t_erode2, bw, bw+1, bw, bw+1, Scalar(std::numeric_limits<T>::min()), stream);
////	if (bw > 1) {
////		::cv::Mat output(t_erode2.size(), t_erode2.type());
////		t_erode2.download(output);
////		imwrite("test-input2-gpu.ppm", output);
////	}
//	GpuMat t_open(t_erode2.size(), t_erode2.type());
//	dilate(t_erode2, t_open, kernel, Point(-1,-1), 1, stream);
////	if (bw > 1) {
////		::cv::Mat output(t_open.size(), t_open.type());
////		t_open.download(output);
////		imwrite("test-open-gpu.ppm", output);
////	}
//	GpuMat g_open(image.size(), t_open.type());
//    stream.enqueueCopy(t_open(roi), g_open);
//    stream.waitForCompletion();
//    t_open.release();
//
//	t_erode2.release();
//	g_erode.release();
//	t_img.release();
//
//	return g_open;

	GpuMat erode = ::nscale::gpu::morphErode<T>(image, kernel, stream);

//	stream.waitForCompletion();
	GpuMat open = ::nscale::gpu::morphDilate<T>(erode, kernel, stream);
	
	stream.waitForCompletion();
	erode.release();
	return open;
}


template <typename T>
GpuMat morphErode(const GpuMat& image, const Mat& kernel, Stream& stream) {
	CV_Assert(kernel.cols == kernel.rows );
	CV_Assert(kernel.cols > 1);
	CV_Assert((kernel.cols % 2) == 1 );

	int bw = (kernel.cols - 1) / 2;
	std::cout << "erodeCopyBorder: Image.cols = "<< image.cols << " bw="<< bw<<" image.data=" << (image.data==NULL) << std::endl;
	GpuMat t_img;
	copyMakeBorder(image, t_img, bw, bw+1, bw, bw+1, BORDER_CONSTANT, Scalar(std::numeric_limits<T>::max()), stream);
	stream.waitForCompletion();
	// this is same for CPU and GPU

//	if (bw > 1) {
//		::cv::Mat output(t_img.size(), t_img.type());
//		t_img.download(output);
//		imwrite("test-input-gpu.ppm", output);
//	}
	GpuMat t_erode(t_img.size(), t_img.type());
	GpuMat buf; //THIS IS AN EMPTY BUFFER USED TO CONFORM TO OPENCV-2.4.1 WHICH HAS AN EXTRA "buf" ARGUMENT
	erode(t_img, t_erode, kernel, buf,Point(-1,-1), 1, stream);
//		::cv::Mat output(t_erode.size(), t_erode.type());
//		t_erode.download(output);
//		imwrite("test-erode-gpu.ppm", output);
//	}
	Rect roi = Rect(bw, bw, image.cols, image.rows);
    GpuMat g_erode(image.size(), t_erode.type());

	stream.waitForCompletion();
    stream.enqueueCopy(t_erode(roi), g_erode);
    stream.waitForCompletion();
    t_erode.release();

	t_img.release();

	return g_erode;

}
template <typename T>
GpuMat morphDilate(const GpuMat& image, const Mat& kernel, Stream& stream) {
	CV_Assert(kernel.cols == kernel.rows );
	CV_Assert(kernel.cols > 1);
	CV_Assert((kernel.cols % 2) == 1 );


	int bw = (kernel.cols - 1) / 2;


	GpuMat t_img;
	copyMakeBorder(image, t_img, bw, bw+1, bw, bw+1, BORDER_CONSTANT, Scalar(std::numeric_limits<T>::min()), stream);
	// this is same for CPU and GPU

	stream.waitForCompletion();
//	if (bw > 1) {
//		::cv::Mat output(t_img.size(), t_img.type());
//		t_img.download(output);
//		imwrite("test-input2-gpu.ppm", output);
//	}
	GpuMat t_dilate(t_img.size(), t_img.type());
	GpuMat buf; //THIS IS AN EMPTY BUFFER USED TO CONFORM TO OPENCV-2.4.1 WHICH HAS AN EXTRA "buf" ARGUMENt
	dilate(t_img, t_dilate, kernel, buf, Point(-1,-1), 1, stream);

	stream.waitForCompletion();
//	if (bw > 1) {
//		::cv::Mat output(t_open.size(), t_open.type());
//		t_open.download(output);
//		imwrite("test-open-gpu.ppm", output);
//	}
	Rect roi = Rect(bw, bw, image.cols, image.rows);
	GpuMat g_dilate(image.size(), t_dilate.type());
    stream.enqueueCopy(t_dilate(roi), g_dilate);
    stream.waitForCompletion();
    t_dilate.release();

	t_img.release();


	return g_dilate;
}




//template Mat imfill<unsigned char>(const Mat& image, const Mat& seeds, bool binary, int connectivity);
//template Mat bwselect<unsigned char>(const Mat& binaryImage, const Mat& seeds, int connectivity);
//template Mat bwlabelFiltered<unsigned char>(const Mat& binaryImage, bool binaryOutput,
//		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
//		bool contourOnly, int connectivity);
//template Mat bwareaopen<unsigned char>(const Mat& binaryImage, int minSize, int maxSize, int connectivity);
//template Mat imhmin(const Mat& image, unsigned char h, int connectivity);
//template Mat imhmin(const Mat& image, float h, int connectivity);
//template Mat_<unsigned char> localMaxima<float>(const Mat& image, int connectivity);
//template Mat_<unsigned char> localMinima<float>(const Mat& image, int connectivity);
//template Mat_<unsigned char> localMaxima<unsigned char>(const Mat& image, int connectivity);
//template Mat_<unsigned char> localMinima<unsigned char>(const Mat& image, int connectivity);

#endif

template GpuMat imreconstruct<float>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstruct<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstruct<float>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstruct<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstructQueue<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template vector<GpuMat> imreconstructQueueThroughput<unsigned char>(vector<GpuMat> & seeds, vector<GpuMat> & image, int connectivity, int nItFirstPass, Stream& stream);
template GpuMat imreconstructQueueSpeedup<unsigned char>(GpuMat &seeds, GpuMat &image, int connectivity, int nItFirstPass, Stream& stream, int nBlocks=14, bool binary=false);
//template GpuMat imreconstructQueueSpeedupFloat>(GpuMat &seeds, GpuMat &image, int connectivity, int nItFirstPass, Stream& stream, int nBlocks=14, bool binary=false);

template GpuMat bwselect<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstructBinary<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstructBinary<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imfillHoles<unsigned char>(const GpuMat&, bool, int, Stream&);

template GpuMat imfillHoles<int>(const GpuMat&, bool, int, Stream&);
template GpuMat imreconstruct<int>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstruct<int>(const GpuMat&, const GpuMat&, int, Stream&);
template GpuMat imreconstructBinary<int>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstructBinary<int>(const GpuMat&, const GpuMat&, int, Stream&);

template GpuMat imreconstructQ<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&, unsigned int&);
template GpuMat imreconstructQ<unsigned char>(const GpuMat&, const GpuMat&, int, Stream&);

template GpuMat imhmin(const GpuMat& image, unsigned char h, int connectivity, Stream&);
template GpuMat imhmin(const GpuMat& image, float h, int connectivity, Stream&);
template GpuMat morphOpen<unsigned char>(const GpuMat& image, const Mat& kernel, Stream&);
template GpuMat morphErode<unsigned char>(const GpuMat& image, const Mat& kernel, Stream&);
template GpuMat morphDilate<unsigned char>(const GpuMat& image, const Mat& kernel, Stream&);

}

}

