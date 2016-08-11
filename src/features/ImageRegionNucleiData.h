#ifndef IMAGE_REGION_NUCLEI_DATA_H
#define IMAGE_REGION_NUCLEI_DATA_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <dirent.h>
#include <fstream>
#include <stdlib.h> 

#include "opencv2/opencv.hpp"

#include "ObjFeatures.h"
#include "CytoplasmCalc.h"
#include "HistologicalEntities.h"
#include "Logger.h"

using namespace cv;

// ImageRegionNucleiData class -- data about nuclei segmented in a region
#define NUCLEUS_THRESHOLD 10
#define NUM_CHANNELS    3
#define RED_CHANNEL 	0
#define GREEN_CHANNEL   1
#define BLUE_CHANNEL    2

typedef std::vector<std::vector<cv::Point> > PolygonList;

typedef struct _TextureFeatureList {
		float  *h_intensityFeatures;
		float  *h_gradientFeatures;
		float  *h_cannyFeatures;
		float  *h_cytoIntensityFeatures;
		float  *h_cytoGradientFeatures;
		float  *h_cytoCannyFeatures;
} TextureFeatureList;

typedef struct _ShapeFeatureList {
		double  *cpuPerimeter;
		int     *cpuArea;
		double  *cpuMajorAxis;
		double  *cpuMinorAxis;
		double  *cpuEccentricity;
		double  *cpuExtentRatio;
		double  *cpuCircularity; 
} ShapeFeatureList;

class ImageRegionNucleiData {
	private: 
		// Region bounding box
		unsigned int rgnMinx, rgnMiny, rgnMaxx, rgnMaxy;
		
		int compCount;	// number of nuclei 
		int *bbox;      // bounding boxes of nuclei with label data
						// int label = bbox[i]; label in segmentation mask of nucleus i
						// int minx = bbox[objCount+i];
						// int maxx = bbox[objCount*2+i];
						// int miny = bbox[objCount*3+i];
						// int maxy = bbox[objCount*4+i];

		int cytoplasmDataSize; // Total size of the cytoplasmBB array
		int *cytoplasmBB; 	// bounding boxes of cytoplasmic regions around nuclei 
						  	// int dataOffset = cytoplasmBB[i*5];
							// int minx       = cytoplasmBB[i*5+1];
							// int miny       = cytoplasmBB[i*5+2];
							// int width      = cytoplasmBB[i*5+3];
							// int height     = cytoplasmBB[i*5+4];
							// 	
							// dataOffset points to address where the corresponding cytoplasm 
							// mask is stored: root address + dataOffset
							//     char *dataAddress = ((char*)(cytoplasmBB))+dataOffset;
							// dataAddress is a byte array, can be used to create a mask, e.g.:  
							//     cv::Mat objMask(height, width, CV_8UC1, dataAddress );
	
		PolygonList *contours; // contours representing nucleus boundaries
		
		// object shape features -- arrays of objCount elements 
		ShapeFeatureList shapeList;

		// object texture features -- arrays of objCount elements
		TextureFeatureList textureList[NUM_CHANNELS]; // RGB channels

	public:
		ImageRegionNucleiData() {
			rgnMinx = rgnMiny = rgnMaxx = rgnMaxy = 0;
		
			compCount = 0; 
			bbox = NULL;

			cytoplasmDataSize = 0; 
			cytoplasmBB = NULL;  
	
			contours = NULL; 
		
			shapeList.cpuPerimeter = NULL;
			shapeList.cpuArea = NULL;
			shapeList.cpuMajorAxis = NULL;
			shapeList.cpuMinorAxis = NULL;
			shapeList.cpuEccentricity = NULL;
			shapeList.cpuExtentRatio = NULL;
			shapeList.cpuCircularity = NULL;

			for (int i=0;i<NUM_CHANNELS;i++) {
				textureList[i].h_intensityFeatures = NULL;
				textureList[i].h_gradientFeatures = NULL;
				textureList[i].h_cannyFeatures = NULL;
				textureList[i].h_cytoIntensityFeatures = NULL;
				textureList[i].h_cytoGradientFeatures = NULL;
				textureList[i].h_cytoCannyFeatures = NULL;
			}
		}	

		ImageRegionNucleiData(int rgnMinx, int rgnMiny, int rgnMaxx, int rgnMaxy) {
		 	this->rgnMinx = rgnMinx;
			this->rgnMiny = rgnMiny;
			this->rgnMaxx = rgnMaxx;
			this->rgnMaxy = rgnMaxy;
		
			compCount = 0; 
			bbox = NULL;

			cytoplasmDataSize = 0; 
			cytoplasmBB = NULL;  
	
			contours = NULL; 
		
			shapeList.cpuPerimeter = NULL;
			shapeList.cpuArea = NULL;
			shapeList.cpuMajorAxis = NULL;
			shapeList.cpuMinorAxis = NULL;
			shapeList.cpuEccentricity = NULL;
			shapeList.cpuExtentRatio = NULL;
			shapeList.cpuCircularity = NULL;

			for (int i=0;i<NUM_CHANNELS;i++) {
				textureList[i].h_intensityFeatures = NULL;
				textureList[i].h_gradientFeatures = NULL;
				textureList[i].h_cannyFeatures = NULL;
				textureList[i].h_cytoIntensityFeatures = NULL;
				textureList[i].h_cytoGradientFeatures = NULL;
				textureList[i].h_cytoCannyFeatures = NULL;
			}
		}	
	
		~ImageRegionNucleiData() {
			if (bbox!=NULL) free(bbox);
			if (cytoplasmBB!=NULL) free(cytoplasmBB);  
			if (contours!=NULL) delete[] contours; 
		
			if (shapeList.cpuPerimeter!=NULL) free(shapeList.cpuPerimeter);
			if (shapeList.cpuArea!=NULL) free(shapeList.cpuArea);
			if (shapeList.cpuMajorAxis!=NULL) free(shapeList.cpuMajorAxis);
			if (shapeList.cpuMinorAxis!=NULL) free(shapeList.cpuMinorAxis);
			if (shapeList.cpuEccentricity!=NULL) free(shapeList.cpuEccentricity);
			if (shapeList.cpuExtentRatio!=NULL) free(shapeList.cpuExtentRatio);
			if (shapeList.cpuCircularity!=NULL) free(shapeList.cpuCircularity);

			for (int i=0;i<NUM_CHANNELS;i++) {
				if (textureList[i].h_intensityFeatures!=NULL) free(textureList[i].h_intensityFeatures);
				if (textureList[i].h_gradientFeatures!=NULL) free(textureList[i].h_gradientFeatures);
				if (textureList[i].h_cannyFeatures!=NULL) free(textureList[i].h_cannyFeatures);
				if (textureList[i].h_cytoIntensityFeatures!=NULL) free(textureList[i].h_cytoIntensityFeatures);
				if (textureList[i].h_cytoGradientFeatures!=NULL) free(textureList[i].h_cytoGradientFeatures);
				if (textureList[i].h_cytoCannyFeatures!=NULL) free(textureList[i].h_cytoCannyFeatures);
			}
		}

		int getImageRegionMinx() { return rgnMinx; }
		int getImageRegionMiny() { return rgnMiny; }
		int getImageRegionMaxx() { return rgnMaxx; }
		int getImageRegionMaxy() { return rgnMaxy; }

		int extractBoundingBoxesFromLabeledMask(cv::Mat& labeledMask) {
			::nscale::ConnComponents cc;
			bbox = cc.boundingBox(labeledMask.cols, labeledMask.rows, (int *)labeledMask.data, 0, compCount);
			return 0;
		} 
		int getNumberOfNuclei() { return compCount; }
		int *getBoundingBoxes() { return bbox; } 

		int extractCytoplasmRegions(cv::Mat& labeledMask) {
			cytoplasmBB = nscale::CytoplasmCalc::calcCytoplasm(cytoplasmDataSize, bbox, compCount, labeledMask);
			return 0;
		}
		int *getCytoplasmRegions() { return cytoplasmBB; }
		int getCytoplasmDataSize() { return cytoplasmDataSize; }

		int extractPolygonsFromLabeledMask(cv::Mat& labeledMask); 
		PolygonList *getPolygons() { return contours; }

		int computeShapeFeatures(cv::Mat& labeledMask);
		ShapeFeatureList& getShapeList() { return shapeList; } 

		int computeRedBlueChannelTextureFeatures(cv::Mat& imgTile, cv::Mat& labeledMask);
		TextureFeatureList *getTextureList() { return textureList; }
};

#if defined (WITH_CUDA)

#include "opencv2/gpu/gpu.hpp"
#include "opencv2/gpu/stream_accessor.hpp"

using namespace cv::gpu;

// GPU Enabled 
#define ASSERT_ERROR() \
	{ \
		cudaError_t error; \
 		error = cudaGetLastError(); \
    	if (error != cudaSuccess) { \
       		printf("CUDA error : %s\n", cudaGetErrorString(error)); \
    	} \
	}

class ImageRegionNucleiDataGPU {
	private:
		int rgnMinx, rgnMiny, rgnMaxx, rgnMaxy;

		cv::Mat inputMask;
		cv::Mat imgTile;
		int tileSize;

		cv::gpu::GpuMat g_labeledMask;

		int *g_bbox;
		int *bbox;
		int compCount;

		int *g_cyto_bbox;
		int *cytoplasmBB;
 		int cytoplasmDataSize;

		cv::gpu::Stream stream; 

		PolygonList *contours; // contours representing nucleus boundaries
		
		// object shape features -- arrays of objCount elements 
		ShapeFeatureList shapeList;

		// object texture features -- arrays of objCount elements
		TextureFeatureList textureList[NUM_CHANNELS]; // RGB channels

	public:
		ImageRegionNucleiDataGPU(cv::gpu::Stream& stream) {
			rgnMinx = rgnMiny = rgnMaxx = rgnMaxy = 0;

			this->stream = stream;
		
			compCount = 0; 
			g_bbox = NULL;
			bbox = NULL;

			cytoplasmDataSize = 0; 
			cytoplasmBB = NULL;  
			g_cyto_bbox = NULL;
	
			contours = NULL; 

			shapeList.cpuPerimeter = NULL;
			shapeList.cpuArea = NULL;
			shapeList.cpuMajorAxis = NULL;
			shapeList.cpuMinorAxis = NULL;
			shapeList.cpuEccentricity = NULL;
			shapeList.cpuExtentRatio = NULL;
			shapeList.cpuCircularity = NULL;

			for (int i=0;i<NUM_CHANNELS;i++) {
				textureList[i].h_intensityFeatures = NULL;
				textureList[i].h_gradientFeatures = NULL;
				textureList[i].h_cannyFeatures = NULL;
				textureList[i].h_cytoIntensityFeatures = NULL;
				textureList[i].h_cytoGradientFeatures = NULL;
				textureList[i].h_cytoCannyFeatures = NULL;
			}
		}	

		ImageRegionNucleiDataGPU(int rgnMinx, int rgnMiny, int rgnMaxx, int rgnMaxy, cv::gpu::Stream& stream) {
		 	this->rgnMinx = rgnMinx;
			this->rgnMiny = rgnMiny;
			this->rgnMaxx = rgnMaxx;
			this->rgnMaxy = rgnMaxy;

			this->stream  = stream;

			compCount = 0; 
			g_bbox = NULL;
			bbox = NULL;

			cytoplasmDataSize = 0; 
			cytoplasmBB = NULL;  
			g_cyto_bbox = NULL;

			contours = NULL; 

			shapeList.cpuPerimeter = NULL;
			shapeList.cpuArea = NULL;
			shapeList.cpuMajorAxis = NULL;
			shapeList.cpuMinorAxis = NULL;
			shapeList.cpuEccentricity = NULL;
			shapeList.cpuExtentRatio = NULL;
			shapeList.cpuCircularity = NULL;

			for (int i=0;i<NUM_CHANNELS;i++) {
				textureList[i].h_intensityFeatures = NULL;
				textureList[i].h_gradientFeatures = NULL;
				textureList[i].h_cannyFeatures = NULL;
				textureList[i].h_cytoIntensityFeatures = NULL;
				textureList[i].h_cytoGradientFeatures = NULL;
				textureList[i].h_cytoCannyFeatures = NULL;
			}
		}	
	
		~ImageRegionNucleiDataGPU() {
			if (g_bbox!=NULL) nscale::gpu::cudaFreeCaller(g_bbox);
			if (bbox!=NULL) free(bbox);

			if (cytoplasmBB!=NULL) free(cytoplasmBB);  
			if (g_cyto_bbox!=NULL) nscale::gpu::cudaFreeCaller(g_cyto_bbox); 

			if (contours!=NULL) delete[] contours; 
		
			if (shapeList.cpuPerimeter!=NULL) free(shapeList.cpuPerimeter);
			if (shapeList.cpuArea!=NULL) free(shapeList.cpuArea);
			if (shapeList.cpuMajorAxis!=NULL) free(shapeList.cpuMajorAxis);
			if (shapeList.cpuMinorAxis!=NULL) free(shapeList.cpuMinorAxis);
			if (shapeList.cpuEccentricity!=NULL) free(shapeList.cpuEccentricity);
			if (shapeList.cpuExtentRatio!=NULL) free(shapeList.cpuExtentRatio);
			if (shapeList.cpuCircularity!=NULL) free(shapeList.cpuCircularity);

			for (int i=0;i<NUM_CHANNELS;i++) {
				if (textureList[i].h_intensityFeatures!=NULL) free(textureList[i].h_intensityFeatures);
				if (textureList[i].h_gradientFeatures!=NULL) free(textureList[i].h_gradientFeatures);
				if (textureList[i].h_cannyFeatures!=NULL) free(textureList[i].h_cannyFeatures);
				if (textureList[i].h_cytoIntensityFeatures!=NULL) free(textureList[i].h_cytoIntensityFeatures);
				if (textureList[i].h_cytoGradientFeatures!=NULL) free(textureList[i].h_cytoGradientFeatures);
				if (textureList[i].h_cytoCannyFeatures!=NULL) free(textureList[i].h_cytoCannyFeatures);
			}
		}

		int setInputMaskImageTile(cv::Mat& inpMask, cv::Mat& inpTile, int rgnMinx, int rgnMiny, int tileSize) {
			this->tileSize = tileSize;
			this->rgnMinx  = rgnMinx;
			this->rgnMiny  = rgnMiny;
			this->rgnMaxx  = inpMask.cols+rgnMinx-1;
			this->rgnMaxy  = inpMask.rows+rgnMiny-1;

			// Check if input mask and tile dimensions are multiples of 256.
			// GPU texture memory requires they be multiples of 256. 
			unsigned int colsDiv = (int) (inpMask.cols/256); 
			unsigned int rowsDiv = (int) (inpMask.rows/256);
			if (inpMask.cols!=(colsDiv*256) || inpMask.rows!=(rowsDiv*256)) {
				unsigned int newCols = inpMask.cols;
				unsigned int newRows = inpMask.rows;
				if (inpMask.cols!=(colsDiv*256)) newCols = (colsDiv+1)*256;
				if (inpMask.rows!=(rowsDiv*256)) newRows = (rowsDiv+1)*256;

				if (newRows>newCols)  // make it square
					newCols=newRows;
				else
					newRows=newCols;

				inputMask = Mat::zeros(newRows,newCols,inpMask.type());
				imgTile   = Mat::zeros(newRows,newCols,inpTile.type());
				inpMask.copyTo(inputMask(Rect(0,0,inpMask.cols,inpMask.rows)));
				inpTile.copyTo(imgTile(Rect(0,0,inpTile.cols,inpTile.rows)));
			} else {
				inputMask = inpMask;
				imgTile   = inpTile;
			}
			return 0;
		}

		int computeLabeledMask() {
			cv::gpu::GpuMat g_maskImage;

			stream.enqueueUpload(inputMask, g_maskImage);
			stream.waitForCompletion(); ASSERT_ERROR();

			g_labeledMask = nscale::gpu::bwlabel(g_maskImage, 8, true, stream);
			stream.waitForCompletion(); ASSERT_ERROR();

			g_maskImage.release();

			return 0;
		}

		int getImageRegionMinx() { return rgnMinx; }
		int getImageRegionMiny() { return rgnMiny; }
		int getImageRegionMaxx() { return rgnMaxx; }
		int getImageRegionMaxy() { return rgnMaxy; }

		int extractBoundingBoxesFromLabeledMask(); 
		int getNumberOfNuclei() { return compCount; }
		int *getBoundingBoxes() { return bbox; } 

		int extractCytoplasmRegions(); 
		int *getCytoplasmRegions() { return cytoplasmBB; }
		int getCytoplasmDataSize() { return cytoplasmDataSize; }

		int extractPolygonsFromLabeledMask(); 
		PolygonList *getPolygons() { return contours; }

		int computeShapeFeatures();
		int computeShapeFeaturesGPU();
		ShapeFeatureList& getShapeList() { return shapeList; } 

		int computeRedBlueChannelTextureFeatures();
		TextureFeatureList *getTextureList() { return textureList; }
};

#endif // WITH_CUDA

#endif // IMAGE_REGION_NUCLEI_DATA
