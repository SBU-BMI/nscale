/*
 * MorphologicOperation.cpp
 *
 *  Created on: Jul 7, 2011
 *    
 */

#include <algorithm>
#include <queue>
#include <iostream>
#include <limits>
#include <omp.h>
#include "highgui.h"

#include "Logger.h"
#include "TypeUtils.h"
#include "MorphologicOperations.h"
#include "PixelOperations.h"
#include "NeighborOperations.h"
#include "ConnComponents.h"


using namespace cv;

using namespace cv::gpu;
using namespace std;


namespace nscale {

// verify if this 0 has a 1 neighbor
bool checkDistNeighbors8(int x, int y, const Mat &mask){
        bool isCandidate = false;

	int rows = mask.rows;
	int cols = mask.cols;
        // upper line
        if(y>0){
                // uppper left corner
                if(x > 0){
                        if(mask.ptr<uchar>(y-1)[x-1] != 0 ){
                                isCandidate = true;
                        }
                }
                // upper right corner
                if(x < (cols-1)){
                        if(mask.ptr<uchar>(y-1)[x+1] != 0 ){
                                isCandidate = true;
                        }
                }
                // upper center
                if(mask.ptr<uchar>(y-1)[x] != 0 ){
                        isCandidate = true;
                }
        }

        // lower line
        if(y < (rows-1)){
                // lower left corner
                if(x > 0){
                        if(mask.ptr<uchar>(y+1)[x-1] != 0 ){
                                isCandidate = true;
                        }
                }
                // lower right corner
                if(x < (cols-1)){
                        if(mask.ptr<uchar>(y+1)[x+1] != 0 ){
                                isCandidate = true;
                        }
                }
                // lower center
                if(mask.ptr<uchar>(y+1)[x] != 0 ){
                        isCandidate = true;
                }
        }
        // left item
        if(x>0){
                if(mask.ptr<uchar>(y)[x-1] != 0){
                        isCandidate = true;
                }
        }
        // right item
        if(x < (cols-1)){
                if(mask.ptr<uchar>(y)[x+1] != 0){
                        isCandidate = true;
                }
        }
        return isCandidate;
}

bool propagateDist(int x, int y, Mat &nearestNeighbor, int tenNNx, int tenNNy){
	bool isTentativeShorter = false;

	int rows = nearestNeighbor.rows;
	int cols = nearestNeighbor.cols;

	// Current nearest background pixel of (x,y)
	int curNN = nearestNeighbor.ptr<int>(y)[x];
	int curNNx = curNN % cols;
	int curNNy = curNN / cols;

	// Current nearest background pixel of (tenNNx,tenNNy)
	int tenCurNeighbor = nearestNeighbor.ptr<int>(tenNNy)[tenNNx];
	int tenCurNNx = tenCurNeighbor % cols;
	int tenCurNNy = tenCurNeighbor / cols;



	float curDist = sqrt(pow((double)(tenNNx-tenCurNNx),2) +pow((double)(tenNNy-tenCurNNy),2));
	float distThroughX = sqrt(pow((double)(curNNx-tenNNx),2) + pow((double)(curNNy-tenNNy),2));

	if(distThroughX < curDist){
		isTentativeShorter = true;
		int *nnPtr = nearestNeighbor.ptr<int>(tenNNy);
		nnPtr[tenNNx] = curNN;
	}

	return isTentativeShorter;
}

// try to propagate distance to a neighbor pixel
void propagateDist8(int x, int y, Mat &nearestNeighbor, std::queue<int> &xQ, std::queue<int> &yQ){
	int rows = nearestNeighbor.rows;
	int cols = nearestNeighbor.cols;
        // upper line
        if(y>0){
                // uppper left corner
                if(x > 0){
			bool propagate = propagateDist(x, y, nearestNeighbor, x-1, y-1);
			if(propagate){
				xQ.push(x-1);
				yQ.push(y-1);
			}
                }
                // upper right corner
                if(x < (cols-1)){
			bool propagate = propagateDist(x, y, nearestNeighbor, x+1, y-1);
			if(propagate){
				xQ.push(x+1);
				yQ.push(y-1);
			}
                }
                // upper center
		bool propagate = propagateDist(x, y, nearestNeighbor, x, y-1);
		if(propagate){
			xQ.push(x);
			yQ.push(y-1);
		}
        }

        // lower line
        if(y < (rows-1)){
                // lower left corner
                if(x > 0){
			bool propagate = propagateDist(x, y, nearestNeighbor, x-1, y+1);
			if(propagate){
				xQ.push(x-1);
				yQ.push(y+1);
			}
                }
                // lower right corner
                if(x < (cols-1)){
			bool propagate = propagateDist(x, y, nearestNeighbor, x+1, y+1);
			if(propagate){
				xQ.push(x+1);
				yQ.push(y+1);
			}
                }
                // lower center
		bool propagate = propagateDist(x, y, nearestNeighbor, x, y+1);
		if(propagate){
			xQ.push(x);
			yQ.push(y+1);
		}
        }
        // left item
        if(x>0){
		bool propagate = propagateDist(x, y, nearestNeighbor, x-1, y);
		if(propagate){
			xQ.push(x-1);
			yQ.push(y);
		}
        }
        // right item
        if(x < (cols-1)){
		bool propagate = propagateDist(x, y, nearestNeighbor, x+1, y);
		if(propagate){
			xQ.push(x+1);
			yQ.push(y);
		}
        }
}



Mat distanceTransform(const Mat& mask, bool calcDist) {
	CV_Assert(mask.channels() == 1);
	CV_Assert(mask.type() ==  CV_8UC1);
	
	// create nearest neighbors map
	Mat nearestNeighbor(mask.size(), CV_32S);

	// save x and y dimension of pixel to be propagated
	std::queue<int> xQ;
	std::queue<int> yQ;

	// Initialization phase: find initial wavefront pixels and init. nearestNeighbor matrix.
	for(int y = 0; y < nearestNeighbor.rows; y++){
		// get point to current line of matrices
		int *nnPtr = nearestNeighbor.ptr<int>(y);
		const uchar *maskPtr = mask.ptr<uchar>(y);

		// iterate over column to init neartes background pixel, and detect wavefront pixels
		for(int x = 0; x < nearestNeighbor.cols; x++){
			nnPtr[x] = nearestNeighbor.rows * nearestNeighbor.cols * 3;

			// if this is a background pixel
			if(maskPtr[x] == 0){
				nnPtr[x] = y*nearestNeighbor.cols+x;

				bool isWavefrontPixel = nscale::checkDistNeighbors8(x, y, mask);
				if(isWavefrontPixel){
					xQ.push(x);
					yQ.push(y);
				}
			}
		}
	}

//	std::cout << "#pixel in initial wavefront: "<< xQ.size() << " x="<<xQ.front() << " y="<< yQ.front()<<std::endl;
	int count = 0;
	while(!xQ.empty()){
		count++;
		int y = yQ.front();
		int x = xQ.front();
		xQ.pop();
		yQ.pop();

		// try to propagate nearest backgroung of (x,y) pixel to its neighbors
		propagateDist8(x, y, nearestNeighbor, xQ, yQ);

	}

	
	// Print nearest neighbors matrix
//        for(int x = 0; x < nearestNeighbor.rows; x++){
//                int* ptr = nearestNeighbor.ptr<int>(x);
//                for(int y = 0; y < nearestNeighbor.cols; y++){
//                        std::cout << std::setprecision(2) << ptr[y] <<"\t ";
//                }
//                std::cout<<std::endl;
//        }
//
	if(calcDist){
		Mat distanceMap(mask.size(), CV_32FC1);

		// Calculate the distance map, based on the nearest neighbors map.
		for(int y = 0; y < mask.rows; y++){
			int* nnPtr = nearestNeighbor.ptr<int>(y);
			float* nnDist = distanceMap.ptr<float>(y);
			for(int x=0; x < mask.cols; x++){
				int curNN = nnPtr[x];
				int x_neighbor = curNN % mask.cols;
				int y_neighbor = curNN / mask.cols;

				nnDist[x] = sqrt( (x-x_neighbor)*(x-x_neighbor)+ (y-y_neighbor)*(y-y_neighbor));
			}
		}

		return distanceMap;
	}else{
		return nearestNeighbor;
	}
}



Mat distTransformFixTilingEffects(Mat& nearestNeighbor, int tileSize, bool calcDist) {
	CV_Assert(nearestNeighbor.channels() == 1);

	int nTiles = nearestNeighbor.cols/tileSize;
	
	// save x and y dimension of pixel to be propagated
	std::queue<int> xQ;
	std::queue<int> yQ;

	uint64_t t1 = cci::common::event::timestampInUS();

	std::cout << "nTiles="<< nTiles*nTiles << " tileSize="<<tileSize <<std::endl;
	int count = 0;

	// pass over entire image image
	for (int y = 0; y < nearestNeighbor.rows; ++y) {
		int xIncrement=1;
		for (int x = 0; x < nearestNeighbor.cols; x+=xIncrement) {
//			std::cout << "("<<y<<","<<x<<"):"<< std::endl;

			propagateDist8(x, y, nearestNeighbor, xQ, yQ);

			if( (x%(tileSize) == (tileSize-1)) || (y%(tileSize)==(tileSize-1) || y%(tileSize)==0) ){
				xIncrement=1;
			}else{
				xIncrement=tileSize-1;
			}

		}
	}
	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "    scan time = " << t2-t1 << "ms for " << count << " queue entries="<< xQ.size()<< std::endl;

	count = 0;
	while(!xQ.empty()){
		count++;
		int y = yQ.front();
		int x = xQ.front();
		xQ.pop();
		yQ.pop();

		// try to propagate nearest backgroung of (x,y) pixel to its neighbors
		propagateDist8(x, y, nearestNeighbor, xQ, yQ);

	}

	uint64_t t3 = cci::common::event::timestampInUS();
	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queue entries "<< std::endl;
	if(calcDist){

		Mat distanceMap(nearestNeighbor.size(), CV_32FC1);

		//        for(int x = 0; x < nearestNeighbor.rows; x++){
		//                int* ptr = nearestNeighbor.ptr<int>(x);
		//                for(int y = 0; y < nearestNeighbor.cols; y++){
		//                        std::cout << std::setprecision(2) << ptr[y] <<"\t ";
		//                }
		//                std::cout<<std::endl;
		//        }

		// Calculate the distance map, based on the nearest neighbors map.
#pragma omp parallel for
		for(int y = 0; y < nearestNeighbor.rows; y++){
			int* nnPtr = nearestNeighbor.ptr<int>(y);
			float* nnDist = distanceMap.ptr<float>(y);
			for(int x=0; x < nearestNeighbor.cols; x++){
				int curNN = nnPtr[x];
				int x_neighbor = curNN % nearestNeighbor.cols;
				int y_neighbor = curNN / nearestNeighbor.cols;

				nnDist[x] = sqrt( (x-x_neighbor)*(x-x_neighbor)+ (y-y_neighbor)*(y-y_neighbor));
			}
		}

		return distanceMap;
	}else{
		return nearestNeighbor;
	}

}


cv::Mat distanceTransformParallelTile(const cv::Mat& mask, int tileSize, int nThreads, bool calcDist){

#if defined (_OPENMP)
	if(nThreads >0)
		omp_set_num_threads(nThreads);
#endif

	int tileWidth=tileSize;
	int tileHeight=tileSize;
	int nTilesX=mask.cols/tileWidth;
	int nTilesY=mask.rows/tileHeight;
	uint64_t t1, t2; 
	
	uint64_t t1_tiled = cci::common::event::timestampInUS();
	Mat nearestNeighbor(mask.size(), CV_32S);

#pragma omp parallel for schedule(dynamic,1)
	for(int tileY=0; tileY < nTilesY; tileY++){
#pragma omp parallel for  schedule(dynamic,1)
		for(int tileX=0; tileX < nTilesX; tileX++){
			Mat roiMask(mask, Rect(tileX*tileWidth, tileY*tileHeight , tileWidth, tileHeight));
			Mat roiNeighborMap(nearestNeighbor, Rect(tileX*tileWidth, tileY*tileHeight , tileWidth, tileHeight));	
			t1 = cci::common::event::timestampInUS();
        
/*			Stream stream;
			GpuMat g_mask(roiMask);
			GpuMat g_distance = nscale::gpu::distanceTransform(g_mask, stream, false, tileX, tileY, tileSize, nearestNeighbor.cols);
			stream.waitForCompletion();
			g_distance.download(roiNeighborMap);
	//		Mat neighborMapTile(g_distance);
			g_mask.release();
			g_distance.release();
*/
			Mat neighborMapTile = nscale::distanceTransform(roiMask, false);

			for(int y = 0; y < neighborMapTile.rows; y++){
				int* NMTPtr = neighborMapTile.ptr<int>(y);
				for(int x = 0; x < neighborMapTile.cols; x++){
					int colId = NMTPtr[x] % neighborMapTile.cols + tileX * tileSize;
					int rowId = NMTPtr[x] / neighborMapTile.cols + tileY * tileSize;
					NMTPtr[x] = rowId*nearestNeighbor.cols + colId;

				} 
			}
//			uint64_t t1_copy = cci::common::event::timestampInUS();
			neighborMapTile.copyTo(roiNeighborMap);
//			uint64_t t2_copy = cci::common::event::timestampInUS();
//			std::cout << "copyDataInCPUMemory" << t2_copy-t1_copy << "ms" << std::endl;			

			uint64_t t2 = cci::common::event::timestampInUS();

			std::cout << " Tile took " << t2-t1 << "ms" << std::endl;
		}
	}
	uint64_t t2_tiled = cci::common::event::timestampInUS();
	std::cout << " Tile total took " << t2_tiled-t1_tiled << "ms" << std::endl;

	t1 = cci::common::event::timestampInUS();
	Mat distanceMap = nscale::distTransformFixTilingEffects(nearestNeighbor, tileSize, calcDist);
	t2 = cci::common::event::timestampInUS();
	std::cout << "fix tiling recon8 took " << t2-t1 << "ms" << std::endl;

	return distanceMap;
}


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

template <typename T>
inline void propagateAtomic(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, T* iPtr, T* oPtr, const T* pval) {
	T ival = iPtr[x];
	T* oPtrP = &(oPtr[x]);
	while(true){
		// Current value in marker image
		T qval = oPtr[x];
		if ((qval < pval[0]) && (ival != qval)) {
			//		#pragma omp atomic

			T min_val = min(pval[0], ival);
			bool success = false;
			if (std::numeric_limits<T>::is_integer) {

#if defined(_MSC_VER)
				//from http://stackoverflow.com/questions/1158374/portable-compare-and-swap-atomic-operations-c-c-library
				success = InterlockedCompareExchange16( (short*)oPtrP, (short)min_val, (short)qval );
#else
					success = __sync_val_compare_and_swap((unsigned char*)oPtrP, (unsigned char)qval, (unsigned char)min_val);
#endif
			}else{
#if defined(_MSC_VER)
				//from http://stackoverflow.com/questions/1158374/portable-compare-and-swap-atomic-operations-c-c-library
				success = InterlockedCompareExchange((long*)oPtrP, (long)min_val, (long)qval);
#else
				success = __sync_val_compare_and_swap((int*)oPtrP, (int)qval, (int)min_val);
#endif
			}
			// if update did not work, read data again and check the propagation condition
			if(!success) continue;
//			oPtrP[0] = min(pval[0], ival);


			xQ.push(x);
			yQ.push(y);
			break;
		}else{
			break;
		}
	}


//	T qval = oPtr[x];
//	T old_pval = pval[0];
//	if ((qval < old_pval) && (ival != qval)) {
////		#pragma omp atomic
//		oPtrP[0] = min(pval[0], ival);
//		xQ.push(x);
//		yQ.push(y);
//	}else{
//	//	break;
//	}
}

template
void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, unsigned char* iPtr, unsigned char* oPtr, const unsigned char&);
template
void propagate(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, float* iPtr, float* oPtr, const float&);

template <typename T>
inline void propagate3D(std::queue<int>& xQ, std::queue<int>& yQ, std::queue<int>& zQ,
		int x, int y, int z, T* iPtr, T* oPtr, const T& pval) {

	T qval = oPtr[x];
	T ival = iPtr[x];
	if ((qval < pval) && (ival != qval)) {
		oPtr[x] = min(pval, ival);
		xQ.push(x);
		yQ.push(y);
		zQ.push(z);
	}
}

template void propagate3D(std::queue<int>& xQ, std::queue<int>& yQ, std::queue<int>& zQ,
		int x, int y, int z, unsigned short int* iPtr, unsigned short int* oPtr, const unsigned short int& pval);

template void propagate3D(std::queue<int>& xQ, std::queue<int>& yQ, std::queue<int>& zQ,
		int x, int y, int z, float* iPtr, float* oPtr, const float& pval);


template <typename T>
std::vector<cv::Mat> imreconstruct3D6(const std::vector<cv::Mat>& seeds, const std::vector<cv::Mat>& image){
	// TODOD: assert all layers have same dimenson/type;
	// create a copy of seeds/mask w/ border around data
	std::vector<Mat> output, mask;

	// The border is meant to avoid conditional branchs in the code, which
	// significantly reduce performance for a number of reasons.
	// add a top border layer
	output.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));
	mask.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));

	for(int i = 0; i < seeds.size(); i++){
		// check type and size of images
		CV_Assert(seeds[i].size() == image[i].size());
		CV_Assert(seeds[i].type() == image[i].type());
		CV_Assert(image[i].channels() == 1);
		CV_Assert(seeds[i].channels() == 1);

		// copy layers and add a border
		Mat outputAux(seeds[i].size() + Size(2,2), seeds[i].type());
		copyMakeBorder(seeds[i], outputAux, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		output.push_back(outputAux);

		Mat inputAux(image[i].size() + Size(2,2), image[i].type());
		copyMakeBorder(image[i], inputAux, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		mask.push_back(inputAux);
	}
	// add bottom border layer
	output.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));
	mask.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));

	T pval, preval;
	int xminus, xplus, yminus, yplus;
	int maxx = output[0].cols - 1;
	int maxy = output[0].rows - 1;
	int maxz = output.size() - 1;
	//std::cout << "cols: "<< output[0].cols << " layers: "<< mask.size()<<" "<< output.size() <<std::endl;
	//std::cout << "maxx: "<< maxx << " maxy: "<< maxy << " maxz: "<< maxz<< std::endl;

	std::queue<int> xQ;
	std::queue<int> yQ;
	std::queue<int> zQ;

	// pointers to pixels in output image
	T* oPtr; //actual row
	T* oPtrMinus; // row = actual -1
	T* oPtrPlus; // row = actual + 1
	T* oPtrZMinus; // upper pixel
	T* oPtrZPlus; // down pixel

	// pointers to pixels in input mask image
	T* iPtr;
	T* iPtrMinus;
	T* iPtrPlus;
	T* iPtrZMinus;
	T* iPtrZPlus;
	int count = 0;
//	uint64_t t1 = cci::common::event::timestampInUS();


	// raster scan
	for(int z = 1; z < maxz; ++z){
		for (int y = 1; y < maxy; ++y) {

			oPtr = output[z].ptr<T>(y);
			oPtrMinus = output[z].ptr<T>(y-1);
			oPtrZMinus = output[z-1].ptr<T>(y);

			iPtr = mask[z].ptr<T>(y);

			preval = oPtr[0];
			for (int x = 1; x < maxx; ++x) {
				xminus = x-1;
				xplus = x+1;
				pval = oPtr[x];

				// walk through the neighbor pixels, left and up (N+(p)), top layer
				pval = max(pval, max(preval, max(oPtrMinus[x], oPtrZMinus[x])));

				preval = min(pval, iPtr[x]);
				oPtr[x] = preval;

			}
		}
	}

	// anti-raster scan
	for (int z = maxz-1; z > 0; --z){
		for (int y = maxy-1; y > 0; --y) {
			oPtr = output[z].ptr<T>(y);
			oPtrPlus = output[z].ptr<T>(y+1);
			oPtrZPlus = output[z+1].ptr<T>(y);
			oPtrMinus = output[z].ptr<T>(y-1);

			iPtr = mask[z].ptr<T>(y);
			iPtrPlus = mask[z].ptr<T>(y+1);
			iPtrZPlus = mask[z+1].ptr<T>(y);

			preval = oPtr[maxx];
			for (int x = maxx-1; x > 0; --x) {
				xminus = x-1;
				xplus = x+1;

				pval = oPtr[x];

				// walk through the neighbor pixels, right and down (N-(p)), and layer below
				pval = max(pval, max(preval, max(oPtrPlus[x], oPtrZPlus[x])));

				preval = min(pval, iPtr[x]);

				oPtr[x] = preval;

				// capture the seeds
				// walk through the neighbor pixels, right and down (N-(p)) , and layer below
				pval = oPtr[x];


				if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||
						(oPtrPlus[x] < min(pval, iPtrPlus[x])) || (oPtrZPlus[x] < min(pval, iPtrZPlus[x]))) {

					xQ.push(x);
					yQ.push(y);
					zQ.push(z);
					++count;

				//	continue;
				}
			}
		}
	}


//	uint64_t t2 = cci::common::event::timestampInUS();
//	std::cout << "    scan time = " << (t2-t1)/1000 << "ms for " << count << " queue entries."<< std::endl;


	// now process the queue.
	//	T qval, ival;
	int x, y, z;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		z = zQ.front();
		xQ.pop();
		yQ.pop();
		zQ.pop();

		xminus = x-1;
		xplus = x+1;
		yminus = y-1;
		yplus = y+1;

		oPtr = output[z].ptr<T>(y);
		oPtrPlus = output[z].ptr<T>(yplus);
		oPtrMinus = output[z].ptr<T>(yminus);
		oPtrZMinus = output[z-1].ptr<T>(y);
		oPtrZPlus = output[z+1].ptr<T>(y);

		iPtr = mask[z].ptr<T>(y);
		iPtrPlus = mask[z].ptr<T>(yplus);
		iPtrMinus = mask[z].ptr<T>(yminus);
		iPtrZMinus = mask[z-1].ptr<T>(y);
		iPtrZPlus = mask[z+1].ptr<T>(y);

		pval = oPtr[x];

		// look at the 6 connected components
		propagate3D<T>(xQ, yQ, zQ, x, yminus, z, iPtrMinus, oPtrMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yplus, z, iPtrPlus, oPtrPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, xminus, y, z, iPtr, oPtr, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, y, z, iPtr, oPtr, pval);
		propagate3D<T>(xQ, yQ, zQ, x, y, z+1, iPtrZPlus, oPtrZPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, y, z-1, iPtrZMinus, oPtrZMinus, pval);

	}


//	uint64_t t3 = cci::common::event::timestampInUS();
//	std::cout << "    queue time = " << (t3-t2)/1000 << "ms for " << count << " queue entries "<< std::endl;


	std::vector<Mat> outputRet;

	// [1,size-2] range removes top and botton border layers
	for(int i = 1; i < output.size()-1; i++){
		// crop inner part of the layer to remove border
		Mat aux = output[i](Range(1, maxy), Range(1,maxx));
		outputRet.push_back(aux);

		output[i].release();
		mask[i].release();
	}
	output[0].release();
	output[output.size()-1].release();
	mask[0].release();
	mask[output.size()-1].release();

	return outputRet;
}


template <typename T>
std::vector<cv::Mat> imreconstruct3D26(const std::vector<cv::Mat>& seeds, const std::vector<cv::Mat>& image){
	// TODOD: assert all layers have same dimenson/type;
	// create a copy of seeds/mask w/ border around data
	std::vector<Mat> output, mask;

	// The border is meant to avoid conditional branchs in the code, which
	// significantly reduce performance for a number of reasons.
	// add a top border layer
	output.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));
	mask.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));

	for(int i = 0; i < seeds.size(); i++){
		// check type and size of images
		CV_Assert(seeds[i].size() == image[i].size());
		CV_Assert(seeds[i].type() == image[i].type());
		CV_Assert(image[i].channels() == 1);
		CV_Assert(seeds[i].channels() == 1);

		// copy layers and add a border
		Mat outputAux(seeds[i].size() + Size(2,2), seeds[i].type());
		copyMakeBorder(seeds[i], outputAux, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		output.push_back(outputAux);

		Mat inputAux(image[i].size() + Size(2,2), image[i].type());
		copyMakeBorder(image[i], inputAux, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		mask.push_back(inputAux);
	}
	// add bottom border layer
	output.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));
	mask.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));

	T pval, preval;
	int xminus, xplus, yminus, yplus, zminus, zplus;
	int maxx = output[0].cols - 1;
	int maxy = output[0].rows - 1;
	int maxz = output.size() - 1;
	//std::cout << "cols: "<< output[0].cols << " layers: "<< mask.size()<<" "<< output.size() <<std::endl;
	//std::cout << "maxx: "<< maxx << " maxy: "<< maxy << " maxz: "<< maxz<< std::endl;

	std::queue<int> xQ;
	std::queue<int> yQ;
	std::queue<int> zQ;

	// pointers to pixels in output image
	T* oPtr; //actual row
	T* oPtrMinus; // row = actual -1
	T* oPtrPlus; // row = actual + 1
	T* oPtrZMinus; // upper layer, current row
	T* oPtrZMinusMinus; // upper layer, current row - 1
	T* oPtrZMinusPlus; // uppper layer, current row + 1
	T* oPtrZPlus; // down layer, current row
	T* oPtrZPlusMinus; // down layer, current row - 1
	T* oPtrZPlusPlus; // down layer, current row + 1

	// pointers to pixels in input mask image
	T* iPtr;
	T* iPtrMinus;
	T* iPtrPlus;
	T* iPtrZMinus;
	T* iPtrZMinusMinus;
	T* iPtrZMinusPlus;
	T* iPtrZPlus;
	T* iPtrZPlusMinus;
	T* iPtrZPlusPlus;

	int count = 0;
	uint64_t t1 = cci::common::event::timestampInUS();


	// raster scan
	for(int z = 1; z < maxz; ++z){
		for (int y = 1; y < maxy; ++y) {

			oPtr = output[z].ptr<T>(y);
			oPtrMinus = output[z].ptr<T>(y-1);
			oPtrZMinus = output[z-1].ptr<T>(y);
			oPtrZMinusMinus = output[z-1].ptr<T>(y-1);
			oPtrZMinusPlus = output[z-1].ptr<T>(y+1);

			iPtr = mask[z].ptr<T>(y);

			preval = oPtr[0];
			for (int x = 1; x < maxx; ++x) {
				xminus = x-1;
				xplus = x+1;
				pval = oPtr[x];

				// check top layer, line y -1
				preval = max(preval, max(oPtrZMinusMinus[xminus], max(oPtrZMinusMinus[x], oPtrZMinusMinus[xplus])));
				// check top layer, line y
				preval = max(preval, max(oPtrZMinus[xminus], max(oPtrZMinus[x], oPtrZMinus[xplus])));
				// check top layer, line y + 1
				preval = max(preval, max(oPtrZMinusPlus[xminus], max(oPtrZMinusPlus[x], oPtrZMinusPlus[xplus])));


				// walk through the neighbor pixels in current layer: left, up, top-left, top-right
				pval = max(pval, max(preval, max(oPtrMinus[x], max(oPtrMinus[xminus], oPtrMinus[xplus]))));

				preval = min(pval, iPtr[x]);
				oPtr[x] = preval;

			}
		}
	}

	// anti-raster scan
	for (int z = maxz-1; z > 0; --z){
		for (int y = maxy-1; y > 0; --y) {
			oPtr = output[z].ptr<T>(y);
			oPtrPlus = output[z].ptr<T>(y+1);
			//oPtrMinus = output[z].ptr<T>(y-1);
			oPtrZPlus = output[z+1].ptr<T>(y);

			oPtrZPlusPlus = output[z+1].ptr<T>(y+1);
			oPtrZPlusMinus = output[z+1].ptr<T>(y-1);

			iPtr = mask[z].ptr<T>(y);
			iPtrPlus = mask[z].ptr<T>(y+1);
			iPtrZPlus = mask[z+1].ptr<T>(y);
			iPtrZPlusPlus = mask[z+1].ptr<T>(y+1);
			iPtrZPlusMinus = mask[z+1].ptr<T>(y-1);

			preval = oPtr[maxx];
			for (int x = maxx-1; x > 0; --x) {
				xminus = x-1;
				xplus = x+1;

				pval = oPtr[x];

				// check down layer, line y -1
				preval = max(preval, max(oPtrZPlusMinus[xminus], max(oPtrZPlusMinus[x], oPtrZPlusMinus[xplus])));
				// check down layer, line y
				preval = max(preval, max(oPtrZPlus[xminus], max(oPtrZPlus[x], oPtrZPlus[xplus])));
				// check down layer, line y + 1
				preval = max(preval, max(oPtrZPlusPlus[xminus], max(oPtrZPlusPlus[x], oPtrZPlusPlus[xplus])));

				// walk through the neighbor pixels in current layer, right (preval), down, down-left, down-right
				pval = max(pval, max(preval, max(oPtrPlus[x], max(oPtrPlus[xminus], oPtrPlus[xplus]))));

				preval = min(pval, iPtr[x]);

				oPtr[x] = preval;

				// capture the seeds
				// walk through the neighbor pixels, right and down (N-(p)) , and layer below
				pval = oPtr[x];

				// check anti-raster propagation condition. does it propagate to any of the neighbors?
				if (	(oPtr[xplus] < min(pval, iPtr[xplus])) || // right
						(oPtrPlus[x] < min(pval, iPtrPlus[x])) ||  // down
						(oPtrPlus[xminus] < min(pval, iPtrPlus[xminus])) ||  // left-diagonal
						(oPtrPlus[xplus] < min(pval, iPtrPlus[xplus])) ||  // right-diagonal
						(oPtrZPlusMinus[xminus] < min(pval, iPtrZPlusMinus[xminus])) ||  // down layer
						(oPtrZPlusMinus[x] < min(pval, iPtrZPlusMinus[x])) ||  // down layer
						(oPtrZPlusMinus[xplus] < min(pval, iPtrZPlusMinus[xplus])) ||  // down layer
						(oPtrZPlus[xminus] < min(pval, iPtrZPlus[xminus])) ||  // down layer
						(oPtrZPlus[x] < min(pval, iPtrZPlus[x])) ||  // down layer
						(oPtrZPlus[xplus] < min(pval, iPtrZPlus[xplus])) ||  // down layer
						(oPtrZPlusPlus[xminus] < min(pval, iPtrZPlusPlus[xminus])) ||  // down layer
						(oPtrZPlusPlus[x] < min(pval, iPtrZPlusPlus[x])) ||  // down layer
						(oPtrZPlusPlus[xplus] < min(pval, iPtrZPlusPlus[xplus])) ) {

					xQ.push(x);
					yQ.push(y);
					zQ.push(z);
					++count;

				//	continue;
				}
			}
		}
	}


	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "    scan time = " << (t2-t1)/1000 << "ms for " << count << " queue entries."<< std::endl;


	// now process the queue.
	//	T qval, ival;
	int x, y, z;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		z = zQ.front();
		xQ.pop();
		yQ.pop();
		zQ.pop();

		xminus = x-1;
		xplus = x+1;
		yminus = y-1;
		yplus = y+1;
		zminus = z-1;
		zplus = z+1;

		oPtr = output[z].ptr<T>(y);
		oPtrPlus = output[z].ptr<T>(yplus);
		oPtrMinus = output[z].ptr<T>(yminus);
		oPtrZMinus = output[z-1].ptr<T>(y);
		oPtrZMinusMinus = output[z-1].ptr<T>(y-1);
		oPtrZMinusPlus = output[z-1].ptr<T>(y+1);
		oPtrZPlus = output[z+1].ptr<T>(y);
		oPtrZPlusPlus = output[z+1].ptr<T>(y+1);
		oPtrZPlusMinus = output[z+1].ptr<T>(y-1);

		iPtr = mask[z].ptr<T>(y);
		iPtrPlus = mask[z].ptr<T>(yplus);
		iPtrMinus = mask[z].ptr<T>(yminus);
		iPtrZMinus = mask[z-1].ptr<T>(y);
		iPtrZMinusMinus = mask[z-1].ptr<T>(y-1);
		iPtrZMinusPlus = mask[z-1].ptr<T>(y+1);
		iPtrZPlus = mask[z+1].ptr<T>(y);
		iPtrZPlusPlus = mask[z+1].ptr<T>(y+1);
		iPtrZPlusMinus = mask[z+1].ptr<T>(y-1);

		pval = oPtr[x];

		// look at the 8 elements in same layer
		propagate3D<T>(xQ, yQ, zQ, xminus, yminus, z, iPtrMinus, oPtrMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yminus, z, iPtrMinus, oPtrMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, yminus, z, iPtrMinus, oPtrMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, xminus, y, z, iPtr, oPtr, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, y, z, iPtr, oPtr, pval);
		propagate3D<T>(xQ, yQ, zQ, xminus, yplus, z, iPtrPlus, oPtrPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yplus, z, iPtrPlus, oPtrPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, yplus, z, iPtrPlus, oPtrPlus, pval);

		// look at the 9 elements in the upper layer (z-1)
		propagate3D<T>(xQ, yQ, zQ, xminus, yminus, zminus, iPtrZMinusMinus, oPtrZMinusMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yminus, zminus, iPtrZMinusMinus, oPtrZMinusMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, yminus, zminus, iPtrZMinusMinus, oPtrZMinusMinus, pval);

		propagate3D<T>(xQ, yQ, zQ, xminus, y, zminus, iPtrZMinus, oPtrZMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, y, zminus, iPtrZMinus, oPtrZMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, y, zminus, iPtrZMinus, oPtrZMinus, pval);

		propagate3D<T>(xQ, yQ, zQ, xminus, yplus, zminus, iPtrZMinusPlus, oPtrZMinusPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yplus, zminus, iPtrZMinusPlus, oPtrZMinusPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, yplus, zminus, iPtrZMinusPlus, oPtrZMinusPlus, pval);

		// look at the 9 elements in the down layer (z+1)
		propagate3D<T>(xQ, yQ, zQ, xminus, yminus, zplus, iPtrZPlusMinus, oPtrZPlusMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yminus, zplus, iPtrZPlusMinus, oPtrZPlusMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, yminus, zplus, iPtrZPlusMinus, oPtrZPlusMinus, pval);

		propagate3D<T>(xQ, yQ, zQ, xminus, y, zplus, iPtrZPlus, oPtrZPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, y, zplus, iPtrZPlus, oPtrZPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, y, zplus, iPtrZPlus, oPtrZPlus, pval);

		propagate3D<T>(xQ, yQ, zQ, xminus, yplus, zplus, iPtrZPlusPlus, oPtrZPlusPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yplus, zplus, iPtrZPlusPlus, oPtrZPlusPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, yplus, zplus, iPtrZPlusPlus, oPtrZPlusPlus, pval);


	}


	uint64_t t3 = cci::common::event::timestampInUS();
	std::cout << "    queue time = " << (t3-t2)/1000 << "ms for " << count << " queue entries "<< std::endl;


	std::vector<Mat> outputRet;

	// [1,size-2] range removes top and botton border layers
	for(int i = 1; i < output.size()-1; i++){
		// crop inner part of the layer to remove border
		Mat aux = output[i](Range(1, maxy), Range(1,maxx));
		outputRet.push_back(aux);

		output[i].release();
		mask[i].release();
	}
	output[0].release();
	output[output.size()-1].release();
	mask[0].release();
	mask[output.size()-1].release();

	return outputRet;
}

template <typename T>
std::vector<cv::Mat> imreconstruct3D(const std::vector<cv::Mat>& seeds, const std::vector<cv::Mat>& image, int connectivity){

	CV_Assert(connectivity == 6 || connectivity == 26);// the only one implemented so far
	CV_Assert(seeds.size() == image.size());// data have same number of layers
	CV_Assert(seeds.size() > 0);// data have at least one layer

	std::vector<Mat> output;

	switch(connectivity){
		case 6:
			output = imreconstruct3D6<T>(seeds, image);
			break;
		case 26:
			output = imreconstruct3D26<T>(seeds, image);
			break;
		default:
			std::cout << __FILE__ << ":"<< __LINE__ <<" Unknown connectivity: "<< connectivity << ". Options are: 6, 26, and 74." << std::endl;
			exit(1);
	}

	return output;
	/*// TODOD: assert all layers have same dimenson/type;
	// create a copy of seeds/mask w/ border around data
	std::vector<Mat> output, mask;

	// The border is meant to avoid conditional branchs in the code, which
	// significantly reduce performance for a number of reasons.
	// add a top border layer
	output.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));
	mask.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));

	for(int i = 0; i < seeds.size(); i++){
		// check type and size of images
		CV_Assert(seeds[i].size() == image[i].size());
		CV_Assert(seeds[i].type() == image[i].type());
		CV_Assert(image[i].channels() == 1);
		CV_Assert(seeds[i].channels() == 1);

		// copy layers and add a border
		Mat outputAux(seeds[i].size() + Size(2,2), seeds[i].type());
		copyMakeBorder(seeds[i], outputAux, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		output.push_back(outputAux);

		Mat inputAux(image[i].size() + Size(2,2), image[i].type());
		copyMakeBorder(image[i], inputAux, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		mask.push_back(inputAux);
	}
	// add bottom border layer
	output.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));
	mask.push_back(Mat::zeros(seeds[0].size() + Size(2,2), seeds[0].type()));

	T pval, preval;
	int xminus, xplus, yminus, yplus;
	int maxx = output[0].cols - 1;
	int maxy = output[0].rows - 1;
	int maxz = output.size() - 1;
	//std::cout << "cols: "<< output[0].cols << " layers: "<< mask.size()<<" "<< output.size() <<std::endl;
	//std::cout << "maxx: "<< maxx << " maxy: "<< maxy << " maxz: "<< maxz<< std::endl;

	std::queue<int> xQ;
	std::queue<int> yQ;
	std::queue<int> zQ;

	// pointers to pixels in output image
	T* oPtr; //actual row
	T* oPtrMinus; // row = actual -1
	T* oPtrPlus; // row = actual + 1
	T* oPtrZMinus; // upper pixel
	T* oPtrZPlus; // down pixel

	// pointers to pixels in input mask image
	T* iPtr;
	T* iPtrMinus;
	T* iPtrPlus;
	T* iPtrZMinus;
	T* iPtrZPlus;
	int count = 0;
//	uint64_t t1 = cci::common::event::timestampInUS();


	// raster scan
	for(int z = 1; z < maxz; ++z){
		for (int y = 1; y < maxy; ++y) {

			oPtr = output[z].ptr<T>(y);
			oPtrMinus = output[z].ptr<T>(y-1);
			oPtrZMinus = output[z-1].ptr<T>(y);

			iPtr = mask[z].ptr<T>(y);

			preval = oPtr[0];
			for (int x = 1; x < maxx; ++x) {
				xminus = x-1;
				xplus = x+1;
				pval = oPtr[x];

				// walk through the neighbor pixels, left and up (N+(p)), top layer
				pval = max(pval, max(preval, max(oPtrMinus[x], oPtrZMinus[x])));

				preval = min(pval, iPtr[x]);
				oPtr[x] = preval;

			}
		}
	}

	// anti-raster scan
	for (int z = maxz-1; z > 0; --z){
		for (int y = maxy-1; y > 0; --y) {
			oPtr = output[z].ptr<T>(y);
			oPtrPlus = output[z].ptr<T>(y+1);
			oPtrZPlus = output[z+1].ptr<T>(y);
			oPtrMinus = output[z].ptr<T>(y-1);

			iPtr = mask[z].ptr<T>(y);
			iPtrPlus = mask[z].ptr<T>(y+1);
			iPtrZPlus = mask[z+1].ptr<T>(y);

			preval = oPtr[maxx];
			for (int x = maxx-1; x > 0; --x) {
				xminus = x-1;
				xplus = x+1;

				pval = oPtr[x];

				// walk through the neighbor pixels, right and down (N-(p)), and layer below
				pval = max(pval, max(preval, max(oPtrPlus[x], oPtrZPlus[x])));

				preval = min(pval, iPtr[x]);

				oPtr[x] = preval;

				// capture the seeds
				// walk through the neighbor pixels, right and down (N-(p)) , and layer below
				pval = oPtr[x];


				if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||
						(oPtrPlus[x] < min(pval, iPtrPlus[x])) || (oPtrZPlus[x] < min(pval, iPtrZPlus[x]))) {

					xQ.push(x);
					yQ.push(y);
					zQ.push(z);
					++count;

				//	continue;
				}
			}
		}
	}


//	uint64_t t2 = cci::common::event::timestampInUS();
//	std::cout << "    scan time = " << (t2-t1)/1000 << "ms for " << count << " queue entries."<< std::endl;


	// now process the queue.
	//	T qval, ival;
	int x, y, z;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		z = zQ.front();
		xQ.pop();
		yQ.pop();
		zQ.pop();

		xminus = x-1;
		xplus = x+1;
		yminus = y-1;
		yplus = y+1;

		oPtr = output[z].ptr<T>(y);
		oPtrPlus = output[z].ptr<T>(yplus);
		oPtrMinus = output[z].ptr<T>(yminus);
		oPtrZMinus = output[z-1].ptr<T>(y);
		oPtrZPlus = output[z+1].ptr<T>(y);

		iPtr = mask[z].ptr<T>(y);
		iPtrPlus = mask[z].ptr<T>(yplus);
		iPtrMinus = mask[z].ptr<T>(yminus);
		iPtrZMinus = mask[z-1].ptr<T>(y);
		iPtrZPlus = mask[z+1].ptr<T>(y);

		pval = oPtr[x];

		// look at the 6 connected components
		propagate3D<T>(xQ, yQ, zQ, x, yminus, z, iPtrMinus, oPtrMinus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, yplus, z, iPtrPlus, oPtrPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, xminus, y, z, iPtr, oPtr, pval);
		propagate3D<T>(xQ, yQ, zQ, xplus, y, z, iPtr, oPtr, pval);
		propagate3D<T>(xQ, yQ, zQ, x, y, z+1, iPtrZPlus, oPtrZPlus, pval);
		propagate3D<T>(xQ, yQ, zQ, x, y, z-1, iPtrZMinus, oPtrZMinus, pval);

	}


//	uint64_t t3 = cci::common::event::timestampInUS();
//	std::cout << "    queue time = " << (t3-t2)/1000 << "ms for " << count << " queue entries "<< std::endl;


	std::vector<Mat> outputRet;

	// [1,size-2] range removes top and botton border layers
	for(int i = 1; i < output.size()-1; i++){
		// crop inner part of the layer to remove border
		Mat aux = output[i](Range(1, maxy), Range(1,maxx));
		outputRet.push_back(aux);

		output[i].release();
		mask[i].release();
	}
	output[0].release();
	output[output.size()-1].release();
	mask[0].release();
	mask[output.size()-1].release();

	return outputRet;*/
}


template <typename T>
std::vector<Mat> imhmax3D(const std::vector<Mat>& image, T h, int connectivity) {
	// imhmax from Matlab
	// I2 = imreconstruct(imsubtract(I,H), I, conn);

	CV_Assert(image.size() >0);
	CV_Assert(connectivity == 6);

	std::vector<Mat> marker;
	// only works for intensity images.
	for(int i = 0; i < image.size(); i++){
		CV_Assert(image[i].channels() == 1);
		marker.push_back(image[i].clone()-h);
	}

	std::vector<Mat>  recon = imreconstruct3D<T>(marker, image, connectivity);

	// release Marker
	for(int i = 0; i < marker.size(); i++){
		marker[i].release();
	}

	return recon;
}

/** slightly optimized serial implementation,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 this is the fast hybrid grayscale reconstruction

 connectivity is either 4 or 8, default 4.

 this is slightly optimized by avoiding conditional where possible.
 */
template <typename T>
Mat imreconstruct(const Mat& seeds, const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);


	Mat output(seeds.size() + Size(2,2), seeds.type());
	copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	T pval, preval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	T* oPtr;
	T* oPtrMinus;
	T* oPtrPlus;
	T* iPtr;
	T* iPtrPlus;
	T* iPtrMinus;

//	uint64_t t1 = cci::common::event::timestampInUS();

	// raster scan
	for (int y = 1; y < maxy; ++y) {

		oPtr = output.ptr<T>(y);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		preval = oPtr[0];
		for (int x = 1; x < maxx; ++x) {
			xminus = x-1;
			xplus = x+1;
			pval = oPtr[x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			pval = max(pval, max(preval, oPtrMinus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrMinus[xplus], oPtrMinus[xminus]));
			}
			preval = min(pval, iPtr[x]);
			oPtr[x] = preval;
		}
	}

	// anti-raster scan
	int count = 0;
	for (int y = maxy-1; y > 0; --y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);

		preval = oPtr[maxx];
		for (int x = maxx-1; x > 0; --x) {
			xminus = x-1;
			xplus = x+1;

			pval = oPtr[x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = max(pval, max(preval, oPtrPlus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrPlus[xplus], oPtrPlus[xminus]));
			}

			preval = min(pval, iPtr[x]);
			oPtr[x] = preval;

			// capture the seeds
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];

			if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||
					(oPtrPlus[x] < min(pval, iPtrPlus[x]))) {
				xQ.push(x);
				yQ.push(y);
				++count;
				continue;
			}

			if (connectivity == 8) {
				if ((oPtrPlus[xplus] < min(pval, iPtrPlus[xplus])) ||
						(oPtrPlus[xminus] < min(pval, iPtrPlus[xminus]))) {
					xQ.push(x);
					yQ.push(y);
					++count;
					continue;
				}
			}
		}
	}

//	uint64_t t2 = cci::common::event::timestampInUS();
//	std::cout << "    scan time = " << t2-t1 << "ms for " << count << " queue entries."<< std::endl;

	// now process the queue.
//	T qval, ival;
	int x, y;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		xplus = x+1;
		yminus = y-1;
		yplus = y+1;

		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(yplus);
		oPtrMinus = output.ptr<T>(yminus);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(yplus);
		iPtrMinus = input.ptr<T>(yminus);

		pval = oPtr[x];

		// look at the 4 connected components
		if (y > 0) {
			propagate<T>(input, output, xQ, yQ, x, yminus, iPtrMinus, oPtrMinus, pval);
		}
		if (y < maxy) {
			propagate<T>(input, output, xQ, yQ, x, yplus, iPtrPlus, oPtrPlus,pval);
		}
		if (x > 0) {
			propagate<T>(input, output, xQ, yQ, xminus, y, iPtr, oPtr,pval);
		}
		if (x < maxx) {
			propagate<T>(input, output, xQ, yQ, xplus, y, iPtr, oPtr,pval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yminus, iPtrMinus, oPtrMinus, pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yminus, iPtrMinus, oPtrMinus, pval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yplus, iPtrPlus, oPtrPlus,pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yplus, iPtrPlus, oPtrPlus,pval);
				}

			}
		}
	}


//	uint64_t t3 = cci::common::event::timestampInUS();
//	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queue entries "<< std::endl;

//	std::cout <<  count << " queue entries "<< std::endl;

	return output(Range(1, maxy), Range(1, maxx));

}


template <typename T>
Mat imreconstructFixTilingEffects(const Mat& seeds, const Mat& image, int connectivity, int tileIdX, int tileIdY, int tileSize, bool withBorder) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);


	uint64_t t1 = cci::common::event::timestampInUS();
	Mat input, output;

	int nTiles = seeds.cols/tileSize;
	
	if(withBorder){
		output = seeds;
		input = image;

		nTiles = seeds.cols-2/tileSize;
	}else{
		output.create(seeds.size() + Size(2,2), seeds.type());
		copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		input.create(image.size() + Size(2,2), image.type());
		copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		nTiles = seeds.cols/tileSize;
	}

	std::cout << "Copy time="<< cci::common::event::timestampInUS()-t1<<std::endl;

	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	T* oPtr;
	T* oPtrMinus;
	T* oPtrPlus;
	T* iPtr;
	T* iPtrPlus;
	T* iPtrMinus;

	t1 = cci::common::event::timestampInUS();

	std::cout << "nTiles="<< nTiles*nTiles << " tileSize="<<tileSize <<std::endl;
	int count = 0;

	t1 = cci::common::event::timestampInUS();
	// pass over entire image image
	for (int y = 1; y <= maxy-1; ++y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);
		iPtrMinus = input.ptr<T>(y-1);
		int xIncrement=1;
		for (int x = 1; x<= maxx-1; x+=xIncrement) {
			xminus = x-1;
			xplus = x+1;

			bool candidateFound=false;
			// capture the seeds
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];

			if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||		  // right
					(oPtrPlus[x] < min(pval, iPtrPlus[x]))||  // down
					(oPtr[xminus] < min(pval, iPtr[xminus]))|| // left
					(oPtrMinus[x] < min(pval, iPtrMinus[x]))   // up
					) {
				

				xQ.push(x);
				yQ.push(y);
				candidateFound=true;
				++count;
			}

			if (connectivity == 8 && !candidateFound) {
				if ((oPtrPlus[xplus] < min(pval, iPtrPlus[xplus])) || // right/down corner
						(oPtrPlus[xminus] < min(pval, iPtrPlus[xminus])) || // left/down corner
						(oPtrMinus[xplus] < min(pval, iPtrMinus[xplus])) || // right/up 
						(oPtrMinus[xminus] < min(pval, iPtrMinus[xminus])) // left/up
				) {
					xQ.push(x);
					yQ.push(y);
					++count;
				}
			}

			if( (x%(tileSize) == 0) || (y%(tileSize)==0 || y%(tileSize)==1) ){
				xIncrement=1;
			}else{
//				xIncrement=1;
				xIncrement=tileSize-1;
			}

		}
	}
	uint64_t t2 = cci::common::event::timestampInUS();
	std::cout << "    scan time = " << t2-t1 << "ms for " << count << " queue entries="<< xQ.size()<< std::endl;

	// now process the queue.
//	T qval, ival;
	int x, y;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		xplus = x+1;
		yminus = y-1;
		yplus = y+1;

		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(yplus);
		oPtrMinus = output.ptr<T>(yminus);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(yplus);
		iPtrMinus = input.ptr<T>(yminus);

		pval = oPtr[x];

		// look at the 4 connected components
		if (y > 0) {
			propagate<T>(input, output, xQ, yQ, x, yminus, iPtrMinus, oPtrMinus, pval);
		}
		if (y < maxy) {
			propagate<T>(input, output, xQ, yQ, x, yplus, iPtrPlus, oPtrPlus,pval);
		}
		if (x > 0) {
			propagate<T>(input, output, xQ, yQ, xminus, y, iPtr, oPtr,pval);
		}
		if (x < maxx) {
			propagate<T>(input, output, xQ, yQ, xplus, y, iPtr, oPtr,pval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yminus, iPtrMinus, oPtrMinus, pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yminus, iPtrMinus, oPtrMinus, pval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagate<T>(input, output, xQ, yQ, xminus, yplus, iPtrPlus, oPtrPlus,pval);
				}
				if (x < maxx) {
					propagate<T>(input, output, xQ, yQ, xplus, yplus, iPtrPlus, oPtrPlus,pval);
				}

			}
		}
	}


	uint64_t t3 = cci::common::event::timestampInUS();
	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queue entries "<< std::endl;


	return output(Range(1, maxy), Range(1, maxx));

}

template <typename T>
Mat imreconstructFixTilingEffectsParallel(const Mat& seeds, const Mat& image, int connectivity, int tileSize, bool withBorder) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);


	uint64_t t1 = cci::common::event::timestampInUS();
	Mat input, output;

	int nTiles;
	
	if(withBorder){
		output = seeds;
		input = image;

		nTiles = (seeds.cols-2)/tileSize;
	}else{
		output.create(seeds.size() + Size(2,2), seeds.type());
		copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		input.create(image.size() + Size(2,2), image.type());
		copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		nTiles = seeds.cols/tileSize;
	}

	std::cout << "Copy time="<< cci::common::event::timestampInUS()-t1<<std::endl;
//	omp_set_num_threads(1);
	int nThreads;
	#pragma omp parallel
	{
#if defined (_OPENMP)
		nThreads = omp_get_num_threads();
#else
		nThreads = 1;
#endif
	}

	std::cout << "nThreads = "<< nThreads << std::endl;
	T pval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	vector<std::queue<int> > xQ(nThreads);
	vector<std::queue<int> > yQ(nThreads);

	std::cout << "Queue.size = "<< xQ.size() <<" Queue[0].size()="<< xQ[0].size()<<std::endl;
	T* oPtr;
	T* oPtrMinus;
	T* oPtrPlus;
	T* iPtr;
	T* iPtrPlus;
	T* iPtrMinus;

	t1 = cci::common::event::timestampInUS();

	std::cout << "nTiles="<< nTiles*nTiles << " tileSize="<<tileSize <<std::endl;
	int count = 0;

	t1 = cci::common::event::timestampInUS();
	// pass over entire image image
	int tid = 0;
//	int tid = omp_get_thread_num();
//#pragma omp parallel for private(oPtr,oPtrPlus,oPtrMinus,iPtr,iPtrPlus,iPtrMinus,xminus,xplus,pval) //schedule(static) 
	for (int y = 1; y <= maxy-1; ++y) {
//		int tid = omp_get_thread_num();
			oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);
		iPtrMinus = input.ptr<T>(y-1);
		int xIncrement=1;
		for (int x = 1; x<= maxx-1; x+=xIncrement) {
			xminus = x-1;
			xplus = x+1;

			bool candidateFound=false;
			// capture the seeds
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];

			if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||		  // right
					(oPtrPlus[x] < min(pval, iPtrPlus[x]))||  // down
					(oPtr[xminus] < min(pval, iPtr[xminus]))|| // left
					(oPtrMinus[x] < min(pval, iPtrMinus[x]))   // up
					) {
				

				xQ[tid].push(x);
				yQ[tid].push(y);
				candidateFound=true;
				++count;
	if(xQ[tid].size() % 1000 == 0){
				tid++;
				tid %= nThreads;
			}

			}

			if (connectivity == 8 && !candidateFound) {
				if ((oPtrPlus[xplus] < min(pval, iPtrPlus[xplus])) || // right/down corner
						(oPtrPlus[xminus] < min(pval, iPtrPlus[xminus])) || // left/down corner
						(oPtrMinus[xplus] < min(pval, iPtrMinus[xplus])) || // right/up 
						(oPtrMinus[xminus] < min(pval, iPtrMinus[xminus])) // left/up
				) {
					xQ[tid].push(x);
					yQ[tid].push(y);
					++count;
					if(xQ[tid].size() % 1000 == 0){
						tid++;
						tid %= nThreads;
					}

				}
			}

			if( (x%(tileSize) == 0) || (y%(tileSize)==0 || y%(tileSize)==1) ){
				xIncrement=1;
			}else{
				xIncrement=tileSize-1;
			}
		
		}
	}

	uint64_t t2 = cci::common::event::timestampInUS();
	count = 0;
	for(int i = 0; i < xQ.size(); i++){
		std::cout << "Queue["<<i<<"]="<< xQ[i].size() << std::endl;
		count+=xQ[i].size();
	}
	std::cout << "    scan time = " << t2-t1 << "ms for queue entries="<< count<< std::endl;

	T*ppval;
	// now process the queue.
//	T qval, ival;
//	int x, y;
//	count = 0;
	#pragma omp parallel private(xminus,xplus,yminus,yplus,oPtr,oPtrPlus,oPtrMinus,iPtr,iPtrPlus,iPtrMinus,ppval)
	{
#if defined (_OPENMP)
		int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif
		while (!(xQ[tid].empty())) {
			++count;
			int x = xQ[tid].front();
			int y = yQ[tid].front();
			xQ[tid].pop();
			yQ[tid].pop();
			xminus = x-1;
			xplus = x+1;
			yminus = y-1;
			yplus = y+1;

			oPtr = output.ptr<T>(y);
			oPtrPlus = output.ptr<T>(yplus);
			oPtrMinus = output.ptr<T>(yminus);
			iPtr = input.ptr<T>(y);
			iPtrPlus = input.ptr<T>(yplus);
			iPtrMinus = input.ptr<T>(yminus);

			ppval = &(oPtr[x]);

			pval = oPtr[x];

			// look at the 4 connected components
			if (y > 0) {
				//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], x, yminus, iPtrMinus, oPtrMinus, ppval);
				propagate<T>(input, output, xQ[tid], yQ[tid], x, yminus, iPtrMinus, oPtrMinus, pval);
			}
			if (y < maxy) {
				//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], x, yplus, iPtrPlus, oPtrPlus,ppval);
				propagate<T>(input, output, xQ[tid], yQ[tid], x, yplus, iPtrPlus, oPtrPlus,pval);
			}
			if (x > 0) {
				//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xminus, y, iPtr, oPtr,ppval);
				propagate<T>(input, output, xQ[tid], yQ[tid], xminus, y, iPtr, oPtr,pval);
			}
			if (x < maxx) {
				//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xplus, y, iPtr, oPtr,ppval);
				propagate<T>(input, output, xQ[tid], yQ[tid], xplus, y, iPtr, oPtr,pval);
			}
			
					// now 8 connected
			if (connectivity == 8) {
			
				if (y > 0) {
					if (x > 0) {
						//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xminus, yminus, iPtrMinus, oPtrMinus, ppval);
						propagate<T>(input, output, xQ[tid], yQ[tid], xminus, yminus, iPtrMinus, oPtrMinus, pval);
					}
					if (x < maxx) {
						//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xplus, yminus, iPtrMinus, oPtrMinus, ppval);
						propagate<T>(input, output, xQ[tid], yQ[tid], xplus, yminus, iPtrMinus, oPtrMinus, pval);
					}
			
				}
				if (y < maxy) {
					if (x > 0) {
						//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xminus, yplus, iPtrPlus, oPtrPlus,ppval);
						propagate<T>(input, output, xQ[tid], yQ[tid], xminus, yplus, iPtrPlus, oPtrPlus,pval);
					}
					if (x < maxx) {

						//propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xplus, yplus, iPtrPlus, oPtrPlus,ppval);
						propagate<T>(input, output, xQ[tid], yQ[tid], xplus, yplus, iPtrPlus, oPtrPlus,pval);
					}
			
				}
			}
		}
	}


	uint64_t t3 = cci::common::event::timestampInUS();
	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queue entries "<< std::endl;


	return output(Range(1, maxy), Range(1, maxx));

}


template <typename T>
Mat imreconstructParallelQueue(const Mat& seeds, const Mat& image, int connectivity, bool withBorder, int nThreads) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);

#if defined (_OPENMP)
	if(nThreads >0)
		omp_set_num_threads(nThreads);
#endif

	uint64_t t1 = cci::common::event::timestampInUS();
	Mat input, output;

	if(withBorder){
		output = seeds;
		input = image;

	}else{
		output.create(seeds.size() + Size(2,2), seeds.type());
		copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
		input.create(image.size() + Size(2,2), image.type());
		copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	}

	std::cout << "Copy time="<< cci::common::event::timestampInUS()-t1<<std::endl;
//	omp_set_num_threads(2);
//	int nThreads;
	#pragma omp parallel
	{
#if defined (_OPENMP)
		nThreads = omp_get_num_threads();
#else
		nThreads = 1;
#endif
	}

	std::cout << "nThreads = "<< nThreads << std::endl;
	T pval, preval;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	vector<std::queue<int> > xQ(nThreads);
	vector<std::queue<int> > yQ(nThreads);

	std::cout << "Queue.size = "<< xQ.size() <<std::endl;
	T* oPtr;
	T* oPtrMinus;
	T* oPtrPlus;
	T* iPtr;
	T* iPtrPlus;
	T* iPtrMinus;

	t1 = cci::common::event::timestampInUS();

	int count = 0;

	t1 = cci::common::event::timestampInUS();


	// raster scan
	#pragma omp parallel for private(oPtr,oPtrMinus,iPtr,xminus,xplus,pval,preval)
	for (int y = 1; y < maxy; ++y) {

		oPtr = output.ptr<T>(y);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		preval = oPtr[0];
		for (int x = 1; x < maxx; ++x) {
			xminus = x-1;
			xplus = x+1;
			pval = oPtr[x];

			// walk through the neighbor pixels, left and up (N+(p)) only
			pval = max(pval, max(preval, oPtrMinus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrMinus[xplus], oPtrMinus[xminus]));
			}
			preval = min(pval, iPtr[x]);
			oPtr[x] = preval;
		}
	}
	// anti-raster
	#pragma omp parallel for private(oPtr,oPtrPlus,oPtrMinus,iPtr,iPtrPlus,preval,xminus,xplus,pval)
	for (int y = maxy-1; y > 0; --y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);

		preval = oPtr[maxx];
		for (int x = maxx-1; x > 0; --x) {
			xminus = x-1;
			xplus = x+1;

			pval = oPtr[x];

			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = max(pval, max(preval, oPtrPlus[x]));

			if (connectivity == 8) {
				pval = max(pval, max(oPtrPlus[xplus], oPtrPlus[xminus]));
			}

			preval = min(pval, iPtr[x]);
			oPtr[x] = preval;

		}
	}


	// pass over entire image image

#pragma omp parallel for private(oPtr,oPtrPlus,oPtrMinus,iPtr,iPtrPlus,iPtrMinus,xminus,xplus,pval) //schedule(static) 
	for (int y = 1; y <= maxy-1; ++y) {
#if defined (_OPENMP)
		int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif

		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);
		iPtrPlus = input.ptr<T>(y+1);
		iPtrMinus = input.ptr<T>(y-1);
		int xIncrement=1;
		for (int x = 1; x<= maxx-1; x+=1) {
			xminus = x-1;
			xplus = x+1;

			bool candidateFound=false;
			// capture the seeds
			// walk through the neighbor pixels, right and down (N-(p)) only
			pval = oPtr[x];

			if ((oPtr[xplus] < min(pval, iPtr[xplus])) ||		  // right
					(oPtrPlus[x] < min(pval, iPtrPlus[x]))||  // down
					(oPtr[xminus] < min(pval, iPtr[xminus]))|| // left
					(oPtrMinus[x] < min(pval, iPtrMinus[x]))   // up
					) {
				

				xQ[tid].push(x);
				yQ[tid].push(y);
				candidateFound=true;
				++count;
			}

			if (connectivity == 8 && !candidateFound) {
				if ((oPtrPlus[xplus] < min(pval, iPtrPlus[xplus])) || // right/down corner
						(oPtrPlus[xminus] < min(pval, iPtrPlus[xminus])) || // left/down corner
						(oPtrMinus[xplus] < min(pval, iPtrMinus[xplus])) || // right/up 
						(oPtrMinus[xminus] < min(pval, iPtrMinus[xminus])) // left/up
				) {
					xQ[tid].push(x);
					yQ[tid].push(y);
					++count;
				}
			}
		}
	}
	uint64_t t2 = cci::common::event::timestampInUS();
	count = 0;
	for(int i = 0; i < xQ.size(); i++){
		count+=xQ[i].size();
	}
	std::cout << "    scan time = " << t2-t1 << "ms for queue entries="<< count<< std::endl;

	T*ppval;
	// now process the queue.
	#pragma omp parallel private(xminus,xplus,yminus,yplus,oPtr,oPtrPlus,oPtrMinus,iPtr,iPtrPlus,iPtrMinus,ppval)
	{
#if defined (_OPENMP)
		int tid = omp_get_thread_num();
#else
		int tid = 0;
#endif
		while (!(xQ[tid].empty())) {
			++count;
			int x = xQ[tid].front();
			int y = yQ[tid].front();
			xQ[tid].pop();
			yQ[tid].pop();
			xminus = x-1;
			xplus = x+1;
			yminus = y-1;
			yplus = y+1;

			oPtr = output.ptr<T>(y);
			oPtrPlus = output.ptr<T>(yplus);
			oPtrMinus = output.ptr<T>(yminus);
			iPtr = input.ptr<T>(y);
			iPtrPlus = input.ptr<T>(yplus);
			iPtrMinus = input.ptr<T>(yminus);

			ppval = &(oPtr[x]);

			// look at the 4 connected components
			if (y > 0) {
				propagateAtomic<T>(input, output, xQ[tid], yQ[tid], x, yminus, iPtrMinus, oPtrMinus, ppval);
			}
			if (y < maxy) {
				propagateAtomic<T>(input, output, xQ[tid], yQ[tid], x, yplus, iPtrPlus, oPtrPlus,ppval);
			}
			if (x > 0) {
				propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xminus, y, iPtr, oPtr,ppval);
			}
			if (x < maxx) {
				propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xplus, y, iPtr, oPtr,ppval);
			}
			
					// now 8 connected
			if (connectivity == 8) {
			
				if (y > 0) {
					if (x > 0) {
						propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xminus, yminus, iPtrMinus, oPtrMinus, ppval);
					}
					if (x < maxx) {
						propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xplus, yminus, iPtrMinus, oPtrMinus, ppval);
					}
			
				}
				if (y < maxy) {
					if (x > 0) {
						propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xminus, yplus, iPtrPlus, oPtrPlus,ppval);
					}
					if (x < maxx) {
						propagateAtomic<T>(input, output, xQ[tid], yQ[tid], xplus, yplus, iPtrPlus, oPtrPlus,ppval);
					}
				}
			}
		}
	}


	uint64_t t3 = cci::common::event::timestampInUS();
	std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queue entries "<< std::endl;

	return output(Range(1, maxy), Range(1, maxx));

}


template <typename T>
cv::Mat imreconstructParallelTile(const cv::Mat& seeds, const cv::Mat& image, int connectivity, int tileSize, int nThreads){

#if defined (_OPENMP)
	if(nThreads >0)
		omp_set_num_threads(nThreads);
#endif

	int tileWidth=tileSize;
	int tileHeight=tileSize;
	int nTilesX=seeds.cols/tileWidth;
	int nTilesY=seeds.rows/tileHeight;
	uint64_t t1, t2; 
	uint64_t t1_tiled = cci::common::event::timestampInUS();

	Mat marker_copy(seeds);

#pragma omp parallel for schedule(dynamic,1)
	for(int tileY=0; tileY < nTilesY; tileY++){
#pragma omp parallel for  schedule(dynamic,1)
		for(int tileX=0; tileX < nTilesX; tileX++){
			Mat roiMarker(marker_copy, Rect(tileX*tileWidth, tileY*tileHeight , tileWidth, tileHeight ));
			Mat roiMask(image, Rect(tileX*tileWidth, tileY*tileHeight , tileWidth, tileHeight));
		
			t1 = cci::common::event::timestampInUS();

			Mat reconTile = nscale::imreconstruct<T>(roiMarker, roiMask, 8);
			reconTile.copyTo(roiMarker);
			uint64_t t2 = cci::common::event::timestampInUS();

			std::cout << " Tile took " << t2-t1 << "ms" << std::endl;
		}
	}
	uint64_t t2_tiled = cci::common::event::timestampInUS();
	std::cout << " Tile total took " << t2_tiled-t1_tiled << "ms" << std::endl;

	t1 = cci::common::event::timestampInUS();

	Mat reconCopy = nscale::imreconstructFixTilingEffects<T>(marker_copy, image, 8, 0, 0, tileSize);
//	Mat reconCopy = nscale::imreconstructFixTilingEffectsParallel<T>(marker_copy, image, 8, tileSize);
	t2 = cci::common::event::timestampInUS();
	std::cout << "fix tiling recon8 took " << t2-t1 << "ms" << std::endl;

	return reconCopy;
}



inline void propagateUchar(int *irev, int *ifwd,
		int& x, int offset, unsigned char* iPtr, unsigned char* oPtr, unsigned char& pval) {
    unsigned char val1 = oPtr[x];
    unsigned char ival = iPtr[x];
    int val2 = pval < ival ? pval : ival;
    if (val1 < val2) {
      if (val1 != 0) {  // if the neighbor's value is going to be replaced, remove the neighbor from list
        ifwd[irev[offset]] = ifwd[offset];
        if (ifwd[offset] >= 0)
          irev[ifwd[offset]] = irev[offset];
      }
      oPtr[x] = val2;  // replace the value
      irev[offset] = -val2;  // and insert into the list...
      ifwd[offset] = irev[-val2];
      irev[-val2] = offset;
      if (ifwd[offset] >= 0)
        irev[ifwd[offset]] = offset;
    }
}

Mat imreconstructUChar(const Mat& seeds, const Mat& image, int connectivity) {

	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);
	CV_Assert(seeds.type() == CV_8U);

	Mat output(seeds.size() + Size(2,2), seeds.type());
	copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	int width = input.cols;
	int height = input.rows;

	  //int ix,iy,ox,oy;
	  int offset, currentP;
	  int currentQ;  // current value in downhill
	  int pixPerImg=width*height;
	  unsigned char val, maxVal = 0;
	  int val1;
	  int *istart,*irev,*ifwd;

	  double mmin, mmax;
	  minMaxLoc(seeds, &mmin, &mmax);
	  maxVal = (int)mmax;

	  // create the downhill list
	  istart = (int*)malloc((maxVal+pixPerImg*2)*sizeof(int));
	  irev = istart+maxVal;
	  ifwd = irev+pixPerImg;
	  // initialize the heads of the lists
	  for (offset = -maxVal; offset < 0; offset++)
	    irev[offset] = offset;

	  // populate the lists with pixel locations - essentially sorting the image by pixel values
	  // backward traversal here will result in forward traversal in the next step.
	  MatIterator_<unsigned char> mend = output.end<unsigned char>();
	  for (offset = pixPerImg-1, --mend; offset >= 0; --mend, --offset) {
		  val = *mend;
		  if (val > 0) {
			  val1 = -val;
			  irev[offset] = val1;  // set the end of the list
			  ifwd[offset] = irev[val1];  // move the head of the list into ifwd
			  irev[val1] = offset;   // insert the new head
			  if (ifwd[offset] >= 0)  // if the list was not previously empty
				  irev[ifwd[offset]] = offset;  // then also set the irev for previous head to the current head
		  }
	  }

	  // now do the processing
	  int xminus, xplus, yminus, yplus;
	  int maxx = width - 1;
	  int maxy = height - 1;
	  unsigned char pval;
	  int x, y;
	  unsigned char *oPtr, *oPtrPlus, *oPtrMinus, *iPtr, *iPtrPlus, *iPtrMinus;
	  for (currentQ = -maxVal; currentQ < 0; ++currentQ) {
	    currentP = irev[currentQ];   // get the head of the list for the curr value
	    while (currentP >= 0) {  // non empty list
	      irev[currentQ] = ifwd[currentP];  // pop the "stack"
	      irev[currentP] = currentQ;   // remove the end.
	      x = currentP%width;  // get the current position
	      y = currentP/width;
	      //std::cout << "x, y = " << x << ", " << y << std::endl;

			xminus = x-1;
			xplus = x+1;
			yminus = y-1;
			yplus = y+1;

			oPtr = output.ptr<unsigned char>(y);
			oPtrPlus = output.ptr<unsigned char>(yplus);
			oPtrMinus = output.ptr<unsigned char>(yminus);
			iPtr = input.ptr<unsigned char>(y);
			iPtrPlus = input.ptr<unsigned char>(yplus);
			iPtrMinus = input.ptr<unsigned char>(yminus);

			pval = oPtr[x];

			// look at the 4 connected components
			if (y > 0) {
				propagateUchar(irev, ifwd, x, x+yminus*width, iPtrMinus, oPtrMinus, pval);
			}
			if (y < maxy) {
				propagateUchar(irev, ifwd, x, x+yplus*width, iPtrPlus, oPtrPlus, pval);
			}
			if (x > 0) {
				propagateUchar(irev, ifwd, xminus, xminus+y*width, iPtr, oPtr, pval);
			}
			if (x < maxx) {
				propagateUchar(irev, ifwd, xplus, xplus+y*width, iPtr, oPtr, pval);
			}

			// now 8 connected
			if (connectivity == 8) {

				if (y > 0) {
					if (x > 0) {
						propagateUchar(irev, ifwd, xminus, xminus+yminus*width, iPtrMinus, oPtrMinus, pval);
					}
					if (x < maxx) {
						propagateUchar(irev, ifwd, xplus, xplus+yminus*width, iPtrMinus, oPtrMinus, pval);
					}

				}
				if (y < maxy) {
					if (x > 0) {
						propagateUchar(irev, ifwd, xminus, xminus+yplus*width, iPtrPlus, oPtrPlus, pval);
					}
					if (x < maxx) {
						propagateUchar(irev, ifwd, xplus, xplus+yplus*width, iPtrPlus, oPtrPlus, pval);
					}

				}
			}


	      currentP = irev[currentQ];
	    }
	  }
	  free(istart);

	return output(Range(1, maxy), Range(1, maxx));

}



template <typename T>
inline void propagateBinary(const Mat& image, Mat& output, std::queue<int>& xQ, std::queue<int>& yQ,
		int x, int y, T* iPtr, T* oPtr, const T& foreground) {
	if ((oPtr[x] == 0) && (iPtr[x] != 0)) {
		oPtr[x] = foreground;
		xQ.push(x);
		yQ.push(y);
	}
}

template
void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, unsigned char* iPtr, unsigned char* oPtr, const unsigned char&);
template
void propagateBinary(const Mat&, Mat&, std::queue<int>&, std::queue<int>&,
		int, int, float* iPtr, float* oPtr, const float&);

/** optimized serial implementation for binary,
 from Vincent paper on "Morphological Grayscale Reconstruction in Image Analysis: Applicaitons and Efficient Algorithms"

 connectivity is either 4 or 8, default 4.  background is assume to be 0, foreground is assumed to be NOT 0.

 */
template <typename T>
Mat imreconstructBinary(const Mat& seeds, const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);

	Mat output(seeds.size() + Size(2,2), seeds.type());
	copyMakeBorder(seeds, output, 1, 1, 1, 1, BORDER_CONSTANT, 0);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	T pval, ival;
	int xminus, xplus, yminus, yplus;
	int maxx = output.cols - 1;
	int maxy = output.rows - 1;
	std::queue<int> xQ;
	std::queue<int> yQ;
	T* oPtr;
	T* oPtrPlus;
	T* oPtrMinus;
	T* iPtr;
	T* iPtrPlus;
	T* iPtrMinus;

//	uint64_t t1 = cci::common::event::timestampInUS();

	int count = 0;
	// contour pixel determination.  if any neighbor of a 1 pixel is 0, and the image is 1, then boundary
	for (int y = 1; y < maxy; ++y) {
		oPtr = output.ptr<T>(y);
		oPtrPlus = output.ptr<T>(y+1);
		oPtrMinus = output.ptr<T>(y-1);
		iPtr = input.ptr<T>(y);

		for (int x = 1; x < maxx; ++x) {

			pval = oPtr[x];
			ival = iPtr[x];

			if (pval != 0 && ival != 0) {
				xminus = x - 1;
				xplus = x + 1;

				// 4 connected
				if ((oPtrMinus[x] == 0) ||
						(oPtrPlus[x] == 0) ||
						(oPtr[xplus] == 0) ||
						(oPtr[xminus] == 0)) {
					xQ.push(x);
					yQ.push(y);
					++count;
					continue;
				}

				// 8 connected

				if (connectivity == 8) {
					if ((oPtrMinus[xminus] == 0) ||
						(oPtrMinus[xplus] == 0) ||
						(oPtrPlus[xminus] == 0) ||
						(oPtrPlus[xplus] == 0)) {
								xQ.push(x);
								yQ.push(y);
								++count;
								continue;
					}
				}
			}
		}
	}

//	uint64_t t2 = cci::common::event::timestampInUS();
	//std::cout << "    scan time = " << t2-t1 << "ms for " << count << " queued "<< std::endl;


	// now process the queue.
//	T qval;
	T outval = std::numeric_limits<T>::max();
	int x, y;
	count = 0;
	while (!(xQ.empty())) {
		++count;
		x = xQ.front();
		y = yQ.front();
		xQ.pop();
		yQ.pop();
		xminus = x-1;
		yminus = y-1;
		yplus = y+1;
		xplus = x+1;

		oPtr = output.ptr<T>(y);
		oPtrMinus = output.ptr<T>(y-1);
		oPtrPlus = output.ptr<T>(y+1);
		iPtr = input.ptr<T>(y);
		iPtrMinus = input.ptr<T>(y-1);
		iPtrPlus = input.ptr<T>(y+1);

		// look at the 4 connected components
		if (y > 0) {
			propagateBinary<T>(input, output, xQ, yQ, x, yminus, iPtrMinus, oPtrMinus, outval);
		}
		if (y < maxy) {
			propagateBinary<T>(input, output, xQ, yQ, x, yplus, iPtrPlus, oPtrPlus, outval);
		}
		if (x > 0) {
			propagateBinary<T>(input, output, xQ, yQ, xminus, y, iPtr, oPtr, outval);
		}
		if (x < maxx) {
			propagateBinary<T>(input, output, xQ, yQ, xplus, y, iPtr, oPtr, outval);
		}

		// now 8 connected
		if (connectivity == 8) {

			if (y > 0) {
				if (x > 0) {
					propagateBinary<T>(input, output, xQ, yQ, xminus, yminus, iPtrMinus, oPtrMinus, outval);
				}
				if (x < maxx) {
					propagateBinary<T>(input, output, xQ, yQ, xplus, yminus, iPtrMinus, oPtrMinus, outval);
				}

			}
			if (y < maxy) {
				if (x > 0) {
					propagateBinary<T>(input, output, xQ, yQ, xminus, yplus, iPtrPlus, oPtrPlus,outval);
				}
				if (x < maxx) {
					propagateBinary<T>(input, output, xQ, yQ, xplus, yplus, iPtrPlus, oPtrPlus,outval);
				}

			}
		}

	}

//	uint64_t t3 = cci::common::event::timestampInUS();
	//std::cout << "    queue time = " << t3-t2 << "ms for " << count << " queued" << std::endl;

	return output(Range(1, maxy), Range(1, maxx));

}



template <typename T>
Mat imfill(const Mat& image, const Mat& seeds, bool binary, int connectivity) {
	CV_Assert(image.channels() == 1);
	CV_Assert(seeds.channels() == 1);

	/* MatLAB imfill code:
	 *     mask = imcomplement(I);
    marker = mask;
    marker(:) = 0;
    marker(locations) = mask(locations);
    marker = imreconstruct(marker, mask, conn);
    I2 = I | marker;
	 */

	Mat mask = nscale::PixelOperations::invert<T>(image);  // validated

	Mat marker = Mat::zeros(mask.size(), mask.type());

	mask.copyTo(marker, seeds);

	if (binary == true) marker = imreconstructBinary<T>(marker, mask, connectivity);
	else marker = imreconstruct<T>(marker, mask, connectivity);

	return image | marker;
}

template <typename T>
Mat imfillHoles(const Mat& image, bool binary, int connectivity) {
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
	Rect roi = Rect(1, 1, image.cols, image.rows);

	// copy the input and pad with -inf.
	Mat mask(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, mask, 1, 1, 1, 1, BORDER_CONSTANT, mn);
	// create marker with inf inside and -inf at border, and take its complement
	Mat marker;
	Mat marker2(image.size(), image.type(), Scalar(mn));
	// them make the border - OpenCV does not replicate the values when one Mat is a region of another.
	copyMakeBorder(marker2, marker, 1, 1, 1, 1, BORDER_CONSTANT, mx);

	// now do the work...
	mask = nscale::PixelOperations::invert<T>(mask);

//	uint64_t t1 = cci::common::event::timestampInUS();
	Mat output;
	if (binary == true) {
//		imwrite("in-imrecon-binary-marker.pgm", marker);
//		imwrite("in-imrecon-binary-mask.pgm", mask);


//		imwrite("test/in-fillholes-bin-marker.pgm", marker);
//		imwrite("test/in-fillholes-bin-mask.pgm", mask);
		output = imreconstructBinary<T>(marker, mask, connectivity);
	} else {
//		imwrite("test/in-fillholes-gray-marker.pgm", marker);
//		imwrite("test/in-fillholes-gray-mask.pgm", mask);
		output = imreconstruct<T>(marker, mask, connectivity);
	}
//	uint64_t t2 = cci::common::event::timestampInUS();
	//TODO: TEMP std::cout << "    imfill hole imrecon took " << t2-t1 << "ms" << std::endl;

	output = nscale::PixelOperations::invert<T>(output);

	return output(roi);
}

// Operates on BINARY IMAGES ONLY
template <typename T>
Mat bwselect(const Mat& binaryImage, const Mat& seeds, int connectivity) {
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

	Mat marker = Mat::zeros(seeds.size(), seeds.type());
	binaryImage.copyTo(marker, seeds);

	marker = imreconstructBinary<T>(marker, binaryImage, connectivity);

	return marker & binaryImage;
}

// Operates on BINARY IMAGES ONLY
// ideally, output should be 64 bit unsigned.
//	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
//  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
//  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
Mat_<int> bwlabel(const Mat& binaryImage, bool contourOnly, int connectivity, bool bbox, std::vector<Vec4i> &boundingBoxes) {
	CV_Assert(binaryImage.channels() == 1);
	// only works for binary images.

	int lineThickness = CV_FILLED;
	if (contourOnly == true) lineThickness = 1;

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

//	Mat_<int> output = Mat_<int>::zeros(binaryImage.size());
//	Mat input = binaryImage.clone();
	Mat_<int> output = Mat_<int>::zeros(binaryImage.size() + Size(2,2));
	Mat input;
	copyMakeBorder(binaryImage, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	int j = 0;
	if (contours.size() > 0) {
		int color = 1;
//		uint64_t t1 = cci::common::event::timestampInUS();
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
			// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
			drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );

			if (bbox == true) {
				CvRect boundbox  = boundingRect(contours[j]);
				Vec4i b;
				b[0] = boundbox.x - 1;  // border padded
				b[1] = b[0] + boundbox.width - 1;
				b[2] = boundbox.y - 1;  // border padded
				b[3] = b[2] + boundbox.height - 1;
				boundingBoxes.push_back(b);
			}
			j++;
		}
//		uint64_t t2 = cci::common::event::timestampInUS();
		//TODO: TEMP std::cout << "    bwlabel drawing took " << t2-t1 << "ms" << std::endl;
	}
//	std::cout << "num contours = " << contours.size() << " vs outer contours " << j << std::endl;
	return output(Rect(1,1,binaryImage.cols, binaryImage.rows));
}

// Operates on BINARY IMAGES ONLY
// perform bwlabel using union find.
Mat_<int> bwlabel2(const Mat& binaryImage, int connectivity, bool relab) {
	CV_Assert(binaryImage.channels() == 1);
	// only works for binary images.
	CV_Assert(binaryImage.type() == CV_8U);

	//copy, to make data continuous.
	Mat input = Mat::zeros(binaryImage.size(), binaryImage.type());
	binaryImage.copyTo(input);

	ConnComponents cc;
	Mat_<int> output = Mat_<int>::zeros(input.size());
	cc.label((unsigned char*) input.data, input.cols, input.rows, (int *)output.data, -1, connectivity);

	// relabel if requested
	int j = 0;
	if (relab == true) {
		j = cc.relabel(output.cols, output.rows, (int *)output.data, -1);
//		printf("%d number of components\n", j);
	}

	input.release();

	return output;
}


// Operates on BINARY IMAGES ONLY
// ideally, output should be 64 bit unsigned.
//	maximum number of labels in a single image is 18 exa-labels, in an image with minimum size of 2^32 x 2^32 pixels.
// rationale for going to this size is that if we go with 32 bit, even if unsigned, we can support 4 giga labels,
//  in an image with minimum size of 2^16 x 2^16 pixels, which is only 65k x 65k.
// however, since the contour finding algorithm uses Vec4i, the maximum index can only be an int.  similarly, the image size is
//  bound by Point's internal representation, so only int.  The Mat will therefore be of type CV_32S, or int.
template <typename T>
Mat bwlabelFiltered(const Mat& binaryImage, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
		bool contourOnly, int connectivity) {
	// only works for binary images.
	if (contourFilter == NULL) {
		std::vector<Vec4i> dummy;
		return bwlabel(binaryImage, contourOnly, connectivity, false, dummy);
	}
	CV_Assert(binaryImage.channels() == 1);

	int lineThickness = CV_FILLED;
	if (contourOnly == true) lineThickness = 1;

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

//	Mat output = Mat::zeros(binaryImage.size(), (binaryOutput ? binaryImage.type() : CV_32S));
//	Mat input = binaryImage.clone();
	Mat output = Mat::zeros(binaryImage.size() + Size(2,2),(binaryOutput ? binaryImage.type() : CV_32S));
	Mat input;
	copyMakeBorder(binaryImage, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);

	if (contours.size() > 0) {
		if (binaryOutput == true) {
			Scalar color(std::numeric_limits<T>::max());
			// iterate over all top level contours (all siblings, draw with own label color
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
				if (contourFilter(contours, hierarchy, idx)) {
					// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
					drawContours( output, contours, idx, color, lineThickness, connectivity, hierarchy );
				}
			}

		} else {
			int color = 1;
			// iterate over all top level contours (all siblings, draw with own label color
			for (int idx = 0; idx >= 0; idx = hierarchy[idx][0], ++color) {
				if (contourFilter(contours, hierarchy, idx)) {
					// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
					drawContours( output, contours, idx, Scalar(color), lineThickness, connectivity, hierarchy );
				}
			}
		}
	}
	return output(Rect(1,1, binaryImage.cols, binaryImage.rows));
}

// inclusive min, exclusive max
bool contourAreaFilter(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {
	CV_Assert(contours.size() > 0);
	CV_Assert(contours.size() > idx);
	CV_Assert(idx > 0);

	int area = (int)rint(contourArea(contours[idx]));
	int circum = contours[idx].size() / 2 + 1;

	area += circum;

	if (area < minArea) return false;

	int i = hierarchy[idx][2];
	for ( ; i >= 0; i = hierarchy[i][0]) {
		area -= ((int)rint(contourArea(contours[i])) + contours[i].size() / 2 + 1);
		if (area < minArea) return false;
	}

	if (area >= maxArea) return false;
	//std::cout << idx << " total area = " << area << std::endl;

	return true;
}

// get area of contour
int getContourArea(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx) {
	CV_Assert(contours.size() > 0);
	CV_Assert(contours.size() > idx);
	CV_Assert(idx >= 0);

	std::vector<Point> contour = contours[idx];
	if (contour.size() == 0) return 0;	

	Rect box = boundingRect(Mat(contour));
	Mat canvas = Mat::zeros(box.height, box.width, CV_8U);
	Point offset(-box.x, -box.y);
	drawContours(canvas, contours, idx, Scalar(255), CV_FILLED, 8, hierarchy, INT_MAX, offset);
	int area= countNonZero(canvas);
	canvas.release();
	return area;
}




// inclusive min, exclusive max
bool contourAreaFilter2(const std::vector<std::vector<Point> >& contours, const std::vector<Vec4i>& hierarchy, int idx, int minArea, int maxArea) {

	// using scanline operation's getContourArea does not work correctly.  There are a lot of special cases that cause problems.
	//uint64_t area = ScanlineOperations::getContourArea(contours, hierarchy, idx);
	int area = getContourArea(contours, hierarchy, idx);	

	//std::cout << idx << " total area = " << area << std::endl;

	if (area < minArea || area >= maxArea) return false;
	else return true;
}



// inclusive min, exclusive max
template <typename T>
Mat bwareaopen(const Mat& binaryImage, int minSize, int maxSize, int connectivity, int& count) {
	// only works for binary images.
	CV_Assert(binaryImage.channels() == 1);
	CV_Assert(minSize > 0);
	CV_Assert(maxSize > 0);

	// based on example from
	// http://opencv.willowgarage.com/documentation/cpp/imgproc_structural_analysis_and_shape_descriptors.html#cv-drawcontours
	// only outputs

	Mat_<T> output = Mat_<T>::zeros(binaryImage.size() + Size(2,2));
	Mat input;
	copyMakeBorder(binaryImage, input, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	std::vector<std::vector<Point> > contours;
	std::vector<Vec4i> hierarchy;  // 3rd entry in the vec is the child - holes.  1st entry in the vec is the next.

	// using CV_RETR_CCOMP - 2 level hierarchy - external and hole.  if contour inside hole, it's put on top level.
	findContours(input, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	//TODO: TEMP std::cout << "num contours = " << contours.size() << std::endl;
	count = 0;
	if (contours.size() > 0) {
		Scalar color(std::numeric_limits<T>::max());
		// iterate over all top level contours (all siblings, draw with own label color
		for (int idx = 0; idx >= 0; idx = hierarchy[idx][0]) {
			if (contourAreaFilter2(contours, hierarchy, idx, minSize, maxSize)) {
				// draw the outer bound.  holes are taken cared of by the function when hierarchy is used.
				drawContours(output, contours, idx, color, CV_FILLED, connectivity, hierarchy );
				++count;
			}
		}
	}
	return output(Rect(1,1, binaryImage.cols, binaryImage.rows));
}

// inclusive min, exclusive max
Mat bwareaopen2(const Mat& image, bool labeled, bool flatten, int minSize, int maxSize, int connectivity, int& count) {
	// only works for binary images.
	CV_Assert(image.channels() == 1);
	// only works for binary images.
	if (labeled == false)
		CV_Assert(image.type() == CV_8U);
	else
		CV_Assert(image.type() == CV_32S);

	//copy, to make data continuous.
	Mat input = Mat::zeros(image.size(), image.type());
	image.copyTo(input);
	Mat_<int> output = Mat_<int>::zeros(input.size());

	ConnComponents cc;
	if (labeled == false) {
		Mat_<int> temp = Mat_<int>::zeros(input.size());
		cc.label((unsigned char*)input.data, input.cols, input.rows, (int *)temp.data, -1, connectivity);
		count = cc.areaThresholdLabeled((int *)temp.data, temp.cols, temp.rows, (int *)output.data, -1, minSize, maxSize);
		temp.release();
	} else {
		count = cc.areaThresholdLabeled((int *)input.data, input.cols, input.rows, (int *)output.data, -1, minSize, maxSize);
	}

	input.release();
	if (flatten == true) {
		Mat O2 = Mat::zeros(output.size(), CV_8U);
		O2 = output > -1;
		output.release();
		return O2;
	} else
		return output;

}
// inclusive min, exclusive max
Mat bwareaopen3(const Mat& binaryImage, bool flatten, int minSize, int maxSize, int connectivity, int& count) {
	// only works for binary images.
	CV_Assert(binaryImage.channels() == 1);
	// only works for binary images.
	CV_Assert(binaryImage.type() == CV_8U);

	//copy, to make data continuous.
	Mat input = Mat::zeros(binaryImage.size(), binaryImage.type());
	binaryImage.copyTo(input);

	ConnComponents cc;
	Mat_<int> temp = Mat_<int>::zeros(input.size());
	count = cc.areaThreshold((unsigned char*)input.data, input.cols, input.rows, (int *)temp.data, -1, minSize, maxSize, connectivity);

	input.release();
	if (flatten == true) {
		Mat output = Mat::zeros(temp.size(), CV_8U);
		output = temp > -1;
		temp.release();
		return output;
	} else
		return temp;

}


template <typename T>
Mat imhmin(const Mat& image, T h, int connectivity) {
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
	Mat mask = nscale::PixelOperations::invert<T>(image);
	Mat marker = mask - h;

//	imwrite("in-imrecon-float-marker.exr", marker);
//	imwrite("in-imrecon-float-mask.exr", mask);

	Mat output = imreconstruct<T>(marker, mask, connectivity);
	return nscale::PixelOperations::invert<T>(output);
}

#define UNVISITED 0x808080
#define BORDER    0x818181
#define MASK      0x80000000
#define RIDGE     0

inline int *neighvector(int w, int conn)
{
	static int neigh[8];
	int i = 0;
	switch(conn) {
		case 4:
			neigh[i++] = - w;
			neigh[i++] = - 1;
			neigh[i++] =   1;
			neigh[i++] =   w;
			break;
		case 8:
			neigh[i++] = - w - 1;
			neigh[i++] = - w;
			neigh[i++] = - w + 1;
			neigh[i++] = - 1;
			neigh[i++] =   1;
			neigh[i++] =   w - 1;
			neigh[i++] =   w;
			neigh[i++] =   w + 1;
			break;
		default:
			exit(1);
	}
	return neigh;
}

// input should have foreground > 0, and 0 for background
Mat_<int> watershed(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);
	CV_Assert(image.type() ==  CV_16U);
	CV_Assert(connectivity == 4 || connectivity == 8);


	// create copy of the input image w/ border
	Mat imageB;
	copyMakeBorder( image, imageB, 1, 1, 1, 1, BORDER_CONSTANT, std::numeric_limits<unsigned short int>::max());

	// create image to be labeled: same size as input image w/ border
	Mat W(image.size(), CV_32S);
	W = UNVISITED;
	copyMakeBorder(W, W, 1, 1, 1, 1, BORDER_CONSTANT, BORDER);

	CV_Assert(W.size() ==  imageB.size());

	// build neighbors vector
	int *neigh = neighvector(W.cols, connectivity);

	// TODO: reshape if not continuous
	CV_Assert(W.isContinuous() && imageB.isContinuous());

	// queue used for flooding
	std::list<int> Q;
	unsigned short int hmin, h;
	int qmin, q, p, i;
	int *WPtr = (int*)W.data;
	unsigned short int *imageBPtr = (unsigned short int*)imageB.data;

	// start in second row and stop before last row
	for(p = W.cols+1; p < W.cols*(W.rows-1); p++) {
		if(WPtr[p] == UNVISITED) {
			hmin = imageBPtr[p];
			qmin = p;
			for(i = 0; i < connectivity; i++) {
				q = p + neigh[i];
				h = imageBPtr[q];
				if(h < hmin) {
					qmin = q;
					hmin = h;
				}
			}
			if(imageBPtr[p] > hmin) {
				WPtr[p] = -qmin;
				Q.push_back(p);
			}
		}
	}
	//imwrite("before.png", W);
	while(!Q.empty()){
		p = Q.front();
		Q.pop_front();
		h = imageBPtr[p];
		for(i = 0; i < connectivity; i++) {
			q = p + neigh[i];
			// if it is not a peak or plateu
			if(WPtr[q] != UNVISITED || imageBPtr[q] != h) continue;
			WPtr[q] = -p;
			Q.push_back(q);	
		}
	}
	int label, LABEL = 1, p1;

	// start in second row and stop before last row
	for(p = W.cols+1; p < W.cols*(W.rows-1); p++) {
		if(WPtr[p] == UNVISITED) {
			WPtr[p] = label = LABEL++;
			Q.clear();
			Q.push_back(p);
			h = imageBPtr[p];
			while(!Q.empty()) {
				p1 = Q.front();
				Q.pop_front();
				for(i = 0; i < connectivity; i++) {
					q = p1 + neigh[i];
					if(WPtr[q] == UNVISITED) {
						WPtr[q] = label;
						Q.push_back(q);
					}
				}
			}

		}
	}
	// start in second row and stop before last row
	for(p = W.cols+1; p < W.cols*(W.rows-1); p++) {
		label = WPtr[p];
		if(label == BORDER || label > 0) continue;
		Q.clear();
		for(;;) {
			q = -label;
			label = WPtr[q];
			if(label > 0) break;
			Q.push_back(q);
		}
		WPtr[p] = label;
		while(!Q.empty()) {
			p1 = Q.front();
			Q.pop_front();
			WPtr[p1] = label;
		}

	}
	// remove border
	return W(Rect(1,1, image.cols, image.rows));
}


#define MASK1 	-2
#define PLATEAU -1
int find(int p, int*adrPtr){
	int q = p;
	// find root
	while(adrPtr[q] != q){
		q = adrPtr[q];
	}
	int u = p;
	int v;
	while(adrPtr[u] != q){
		v = adrPtr[u];
		adrPtr[u] = q;
		u = v;
	}
	return q;
}


// "An efficient watershed algorithm based on connected components" Pattern Recognition, 2000.
Mat_<int> watershedCC(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);
	CV_Assert(image.type() ==  CV_16U);
	CV_Assert(connectivity == 4 || connectivity == 8);


	// create copy of the input image w/ border
	Mat imageB;
	copyMakeBorder( image, imageB, 1, 1, 1, 1, BORDER_CONSTANT, std::numeric_limits<unsigned short int>::max());

	// create image to be labeled: same size as input image w/ border
	Mat lab(image.size(), CV_32S);
	copyMakeBorder(lab, lab, 1, 1, 1, 1, BORDER_CONSTANT, BORDER);
	lab = MASK;

	Mat adr = Mat::zeros(lab.size(), lab.type());

	CV_Assert(lab.size() ==  imageB.size());

	// build neighbors vector
	int *neigh = neighvector(lab.cols, connectivity);
	//std::cout<< "lab.cols: " << lab.cols << std::endl;
	// TODO: reshape if not continuous
	CV_Assert(lab.isContinuous() && imageB.isContinuous());

	// queue used for flooding
	std::list<int> Q;

	unsigned short int hmin, h;
	int qmin, q, u, v, p, i, r;
	//int *WPtr = (int*)W.data;
	int *adrPtr = (int *)adr.data;
	int *labPtr = (int *)lab.data;
	unsigned short int *imageBPtr = (unsigned short int*)imageB.data;

	// start in second row and stop before last row
	for(p = lab.cols+1; p < lab.cols*(lab.rows-1); p++) {
		q = p;

		// for each neighbor of p, get neighbor w/ smaller value
		for(i = 0; i < connectivity; i++) {
				u = p + neigh[i];
				// che
				if(imageBPtr[u] < imageBPtr[q]){
					q = u;
				}
		}
		if(q != p){
			adrPtr[p] = q;
		}else{
			adrPtr[p] = PLATEAU;
		}

	}


	//std::cout << "adrPtr[18]: "<< adrPtr[18] <<std::endl;
	//imwrite("out1.ppm", adr);
	// #step 2
	for(p = lab.cols+1; p < lab.cols*(lab.rows-1); p++) {
		if( adrPtr[p] != PLATEAU){
			continue;
		}
		// for each neighbor of p, get neighbor w/ smaller value
		for(i = 0; i < connectivity; i++) {
			q = p + neigh[i];
			if(adrPtr[q] ==  PLATEAU || imageBPtr[q] != imageBPtr[p]){
				continue;
			}
			Q.push_back(q);
		}

	}
	while(!Q.empty()){
		p = Q.front();
		Q.pop_front();
		h = imageBPtr[p];
		for(i = 0; i < connectivity; i++) {
			q = p + neigh[i];
			// if it is not a peak or plateu
			if(adrPtr[q] != PLATEAU || imageBPtr[q] != imageBPtr[p]) {
				continue;
			}

			adrPtr[q] = p;
			Q.push_back(q);
		}
	}

	// step 3
	// start in second row and stop before last row
	for(p = lab.cols+1; p < lab.cols*(lab.rows-1); p++) {
		if(adrPtr[p] != PLATEAU) {
			continue;
		}
		adrPtr[p] = p;

		for(i = 0; i < connectivity; i++) {
			q = p + neigh[i];
			if(q > p || imageBPtr[q] != imageBPtr[p]) {
				continue;
			}
			u = find(p, adrPtr);
			v = find(q, adrPtr);

			adrPtr[u] = adrPtr[v] = std::min(u,v);
		}
	}

	int basins =1;
	// start in second row and stop before last row
	for(p = lab.cols+1; p < lab.cols*(lab.rows-1); p++) {
		r = find(p, adrPtr);
		adrPtr[p] = r;
		if(labPtr[r] == MASK){
			labPtr[r] = basins;
			basins++;
		}
		labPtr[p] = labPtr[r];
	}

/*	for(i = 0; i < 11; i++){
		for(int j = 0; j < 10; j++ ){
				std::cout << adrPtr[(i+1)*12+j+1] << " ";

		}
		std::cout <<endl;
	}*/
/*	for(i = 0; i < 11; i++){
		for(int j = 0; j < 10; j++ ){
				std::cout << labPtr[(i+1)*12+j+1] << " ";

		}
		std::cout <<endl;
	}*/
	// remove border
	return lab(Rect(1,1, image.cols, image.rows));
}

// input should have foreground > 0, and 0 for background
Mat_<int> watershed(const Mat& origImage, const Mat_<float>& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);
	CV_Assert(origImage.channels() == 3);

	/*
	 * MatLAB implementation:
		cc = bwconncomp(imregionalmin(A, conn), conn);
		L = watershed_meyer(A,conn,cc);

	 */

//	t1 = ::cci::common::event::timestampInUS();
	Mat minima = localMinima<float>(image, connectivity);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    cpu localMinima = %lld\n", t2-t1);

//	t1 = ::cci::common::event::timestampInUS();
	std::vector<Vec4i> dummy;
	Mat_<int> labels = bwlabel(minima, false, connectivity, false, dummy);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    cpu opencv bwlabel = %lld\n", t2-t1);

// need borders, else get edges at edge.
	Mat input, temp, output;
	copyMakeBorder(labels, temp, 1, 1, 1, 1, BORDER_CONSTANT, Scalar_<int>(0));
	copyMakeBorder(origImage, input, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

//	t1 = ::cci::common::event::timestampInUS();
	watershed(input, temp);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    cpu watershed = %lld\n", t2-t1);

//	t1 = ::cci::common::event::timestampInUS();
	output = nscale::NeighborOperations::border<int>(temp, (int)-1);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    CPU watershed border fix = %lld\n", t2-t1);

	return output(Rect(1,1, image.cols, image.rows));
}

// input should have foreground > 0, and 0 for background
Mat_<int> watershed2(const Mat& origImage, const Mat_<float>& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);
	CV_Assert(origImage.channels() == 3);

	/*
	 * MatLAB implementation:
		cc = bwconncomp(imregionalmin(A, conn), conn);
		L = watershed_meyer(A,conn,cc);

	 */
//	long long int t1, t2;
//	t1 = ::cci::common::event::timestampInUS();
	Mat minima = localMinima<float>(image, connectivity);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    cpu localMinima = %lld\n", t2-t1);

//	t1 = ::cci::common::event::timestampInUS();
	// watershed is sensitive to label values.  need to relabel.
	Mat_<int> labels = bwlabel2(minima, connectivity, true);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    cpu UF bwlabel2 = %lld\n", t2-t1);


// need borders, else get edges at edge.
	Mat input, temp, output;
	copyMakeBorder(labels, temp, 1, 1, 1, 1, BORDER_CONSTANT, Scalar_<int>(0));
	copyMakeBorder(origImage, input, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

//	t1 = ::cci::common::event::timestampInUS();

		// input: seeds are labeled from 1 to n, with 0 as background or unknown regions
	// output has -1 as borders.
	watershed(input, temp);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    CPU watershed = %lld\n", t2-t1);

//	t1 = ::cci::common::event::timestampInUS();
	output = nscale::NeighborOperations::border<int>(temp, (int)-1);
//	t2 = ::cci::common::event::timestampInUS();
//	printf("    CPU watershed border fix = %lld\n", t2-t1);

	return output(Rect(1,1, image.cols, image.rows));
}

// only works with integer images
template <typename T>
Mat_<unsigned char> localMaxima(const Mat& image, int connectivity) {
	CV_Assert(image.channels() == 1);

	// use morphologic reconstruction.
	Mat marker = image - 1;
	Mat_<unsigned char> candidates =
			marker < imreconstruct<T>(marker, image, connectivity);
//	candidates marked as 0 because floodfill with mask will fill only 0's
//	return (image - imreconstruct(marker, image, 8)) >= (1 - std::numeric_limits<T>::epsilon());
	//return candidates;

	// now check the candidates
	// first pad the border
	T mn = cci::common::type::min<T>();
	T mx = std::numeric_limits<unsigned char>::max();
	Mat_<unsigned char> output(candidates.size() + Size(2,2));
	copyMakeBorder(candidates, output, 1, 1, 1, 1, BORDER_CONSTANT, mx);
	Mat input(image.size() + Size(2,2), image.type());
	copyMakeBorder(image, input, 1, 1, 1, 1, BORDER_CONSTANT, mn);

	int maxy = input.rows-1;
	int maxx = input.cols-1;
	int xminus, xplus;
	T val;
	T *iPtr, *iPtrMinus, *iPtrPlus;
	unsigned char *oPtr;
	Rect reg(1, 1, image.cols, image.rows);
	Scalar zero(0);
	Scalar smx(mx);
//	Range xrange(1, maxx);
//	Range yrange(1, maxy);
	Mat inputBlock = input(reg);

	// next iterate over image, and set candidates that are non-max to 0 (via floodfill)
	for (int y = 1; y < maxy; ++y) {

		iPtr = input.ptr<T>(y);
		iPtrMinus = input.ptr<T>(y-1);
		iPtrPlus = input.ptr<T>(y+1);
		oPtr = output.ptr<unsigned char>(y);

		for (int x = 1; x < maxx; ++x) {

			// not a candidate, continue.
			if (oPtr[x] > 0) continue;

			xminus = x-1;
			xplus = x+1;

			val = iPtr[x];
			// compare values

			// 4 connected
			if ((val < iPtrMinus[x]) || (val < iPtrPlus[x]) || (val < iPtr[xminus]) || (val < iPtr[xplus])) {
				// flood with type minimum value (only time when the whole image may have mn is if it's flat)
				floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
				continue;
			}

			// 8 connected
			if (connectivity == 8) {
				if ((val < iPtrMinus[xminus]) || (val < iPtrMinus[xplus]) || (val < iPtrPlus[xminus]) || (val < iPtrPlus[xplus])) {
					// flood with type minimum value (only time when the whole image may have mn is if it's flat)
					floodFill(inputBlock, output, Point(xminus, y-1), smx, &reg, zero, zero, FLOODFILL_FIXED_RANGE | FLOODFILL_MASK_ONLY | connectivity);
					continue;
				}
			}

		}
	}
	return output(reg) == 0;  // similar to bitwise not.

}

template <typename T>
Mat_<unsigned char> localMinima(const Mat& image, int connectivity) {
	// only works for intensity images.
	CV_Assert(image.channels() == 1);

	Mat cimage = nscale::PixelOperations::invert<T>(image);
	return localMaxima<T>(cimage, connectivity);
}


template <typename T>
Mat morphOpen(const Mat& image, const Mat& kernel) {
	CV_Assert(kernel.rows == kernel.cols);
	CV_Assert(kernel.rows > 1);
	CV_Assert((kernel.rows % 2) == 1);

	int bw = (kernel.rows - 1) / 2;

	// can't use morphologyEx.  the erode phase is not creating a border even though the method signature makes it appear that way.
	// because of this, and the fact that erode and dilate need different border values, have to do the erode and dilate myself.
	//	morphologyEx(image, seg_open, CV_MOP_OPEN, disk3, Point(1,1)); //, Point(-1, -1), 1, BORDER_REFLECT);
	Mat t_image;

	copyMakeBorder(image, t_image, bw, bw, bw, bw, BORDER_CONSTANT, std::numeric_limits<unsigned char>::max());
//	if (bw > 1)	imwrite("test-input-cpu.ppm", t_image);
	Mat t_erode = Mat::zeros(t_image.size(), t_image.type());
	erode(t_image, t_erode, kernel);
//	if (bw > 1) imwrite("test-erode-cpu.ppm", t_erode);

	Mat erode_roi = t_erode(Rect(bw, bw, image.cols, image.rows));
	Mat t_erode2;
	copyMakeBorder(erode_roi,t_erode2, bw, bw, bw, bw, BORDER_CONSTANT, std::numeric_limits<unsigned char>::min());
//	if (bw > 1)	imwrite("test-input2-cpu.ppm", t_erode2);
	Mat t_open = Mat::zeros(t_erode2.size(), t_erode2.type());
	dilate(t_erode2, t_open, kernel);
//	if (bw > 1) imwrite("test-open-cpu.ppm", t_open);
	Mat open = t_open(Rect(bw, bw,image.cols, image.rows));

	t_open.release();
	t_erode2.release();
	erode_roi.release();
	t_erode.release();

	return open;
}

// Tiled based version
template DllExport Mat imreconstructParallelTile<unsigned char>(const cv::Mat& seeds, const cv::Mat& image, int connectivity, int tileSize, int nThreads);
template DllExport Mat imreconstructParallelTile<float>(const cv::Mat& seeds, const cv::Mat& image, int connectivity, int tileSize, int nThreads);


// Parallel queue
template DllExport Mat imreconstructParallelQueue<unsigned char>(const Mat& seeds, const Mat& image, int connectivity, bool withBorder, int nThreads);
template DllExport Mat imreconstructParallelQueue<float>(const Mat& seeds, const Mat& image, int connectivity, bool withBorder, int nThreads);


template DllExport Mat imreconstructFixTilingEffects<unsigned char>(const Mat& seeds, const Mat& image, int connectivity, int tileIdX, int tileIdY, int tileSize, bool withBorder);
template DllExport Mat imreconstructFixTilingEffects<float>(const Mat& seeds, const Mat& image, int connectivity, int tileIdX, int tileIdY, int tileSize, bool withBorder);

template DllExport Mat imreconstructFixTilingEffectsParallel<unsigned char>(const Mat& seeds, const Mat& image, int connectivity, int tileSize, bool withBorder);
template DllExport Mat imreconstructFixTilingEffectsParallel<float>(const Mat& seeds, const Mat& image, int connectivity, int tileSize, bool withBorder);

//template Mat imreconstructGeorge<unsigned char>(const Mat& seeds, const Mat& image, int connectivity);
template DllExport Mat imreconstruct<unsigned char>(const Mat& seeds, const Mat& image, int connectivity);
template DllExport Mat imreconstruct<unsigned short int>(const Mat& seeds, const Mat& image, int connectivity);
template DllExport Mat imreconstruct<float>(const Mat& seeds, const Mat& image, int connectivity);

template DllExport std::vector<cv::Mat> imreconstruct3D<unsigned short int>(const std::vector<cv::Mat>& seeds, const std::vector<cv::Mat>& image, int connectivity);
template DllExport std::vector<cv::Mat> imreconstruct3D<float>(const std::vector<cv::Mat>& seeds, const std::vector<cv::Mat>& image, int connectivity);

template DllExport std::vector<cv::Mat> imhmax3D(const std::vector<cv::Mat>& input, unsigned short int h, int connectivity);
template DllExport std::vector<cv::Mat> imhmax3D(const std::vector<cv::Mat>& input, float h, int connectivity);

template DllExport Mat imreconstructBinary<unsigned char>(const Mat& seeds, const Mat& binaryImage, int connectivity);
template DllExport Mat imfill<unsigned char>(const Mat& image, const Mat& seeds, bool binary, int connectivity);
template DllExport Mat imfillHoles<unsigned char>(const Mat& image, bool binary, int connectivity);
template DllExport Mat imfillHoles<int>(const Mat& image, bool binary, int connectivity);
template DllExport Mat imreconstruct<int>(const Mat& seeds, const Mat& image, int connectivity);
template DllExport Mat imreconstructBinary<int>(const Mat& seeds, const Mat& binaryImage, int connectivity);

template DllExport Mat bwselect<unsigned char>(const Mat& image, const Mat& seeds, int connectivity);
template DllExport Mat bwlabelFiltered<unsigned char>(const Mat& image, bool binaryOutput,
		bool (*contourFilter)(const std::vector<std::vector<Point> >&, const std::vector<Vec4i>&, int),
		bool contourOnly, int connectivity);
template DllExport Mat bwareaopen<unsigned char>(const Mat& image, int minSize, int maxSize, int connectivity, int& count);
template DllExport Mat imhmin(const Mat& image, unsigned char h, int connectivity);
template DllExport Mat imhmin(const Mat& image, float h, int connectivity);
template DllExport Mat_<unsigned char> localMaxima<float>(const Mat& image, int connectivity);
template DllExport Mat_<unsigned char> localMinima<float>(const Mat& image, int connectivity);
template DllExport Mat_<unsigned char> localMaxima<unsigned char>(const Mat& image, int connectivity);
template DllExport Mat_<unsigned char> localMinima<unsigned char>(const Mat& image, int connectivity);
template DllExport Mat morphOpen<unsigned char>(const Mat& image, const Mat& kernel);

}

