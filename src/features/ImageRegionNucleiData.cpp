#include "ImageRegionNucleiData.h"

int ImageRegionNucleiData::extractPolygonsFromLabeledMask(cv::Mat& labeledMask)
{
	if (compCount<=0) return 1;

	contours = new PolygonList[compCount];
	if (contours==NULL) {
		std::cerr << "Cannot allocate space for polygons." << std::endl;
		return 1;
	}

	for(int i = 0; i < compCount; i++){
		int label = bbox[i];
		int minx  = bbox[compCount+i];
		int maxx  = bbox[compCount*2+i];
		int miny  = bbox[compCount*3+i];
		int maxy  = bbox[compCount*4+i];

		cv::Mat boxImg((maxy-miny+1)+2,(maxx-minx+1)+2,CV_8UC1,Scalar(0));
		int pixCount = 0;
		for (int x=minx;x<=maxx;x++) {
			for (int y=miny;y<=maxy;y++) {
				if (labeledMask.at<int>(y,x)==label) {
					boxImg.at<char>(y-miny+1,x-minx+1) = 255;
					pixCount++;
				}
			}
		}

		std::vector<Vec4i> hierarchy;  
		cv::findContours(boxImg, contours[i], hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE); 
		if (contours[i].size()>1) 
			std::cout << "Number of contours[" << i << "]: " << contours[i].size() << std::endl;
		for (int idx=0;idx<contours[i].size();idx++) {
			for (unsigned int ptc=0; ptc<contours[i][idx].size(); ptc++) {
				contours[i][idx][ptc].x += minx-1;
				contours[i][idx][ptc].y += miny-1;
			}
		}	
	}
	return 0;
}

int ImageRegionNucleiData::computeShapeFeatures(cv::Mat& labeledMask) 
{
	if (compCount<=0) return 1;
	
	shapeList.cpuPerimeter = nscale::ObjFeatures::perimeter((const int*)bbox,compCount,labeledMask);
	shapeList.cpuArea = nscale::ObjFeatures::area((const int*)bbox , compCount , labeledMask);
	nscale::ObjFeatures::ellipse((const int*)bbox,(const int*)shapeList.cpuArea, compCount , labeledMask, 
								shapeList.cpuMajorAxis, shapeList.cpuMinorAxis, shapeList.cpuEccentricity);
	shapeList.cpuExtentRatio = nscale::ObjFeatures::extent_ratio((const int*)bbox , (const int)compCount , (const int *)shapeList.cpuArea);
	shapeList.cpuCircularity = nscale::ObjFeatures::circularity(compCount , shapeList.cpuArea , shapeList.cpuPerimeter);
	return 0;
}

int ImageRegionNucleiData::computeRedBlueChannelTextureFeatures(cv::Mat& imgTile, cv::Mat& labeledMask)
{
	if (compCount<=0) return 1;

	// Texture features
	vector<cv::Mat> bgr;
	split(imgTile, bgr);

	// Red channel
	textureList[RED_CHANNEL].h_intensityFeatures = nscale::ObjFeatures::intensityFeatures(bbox, compCount, labeledMask, bgr[2]);
	textureList[RED_CHANNEL].h_gradientFeatures  = nscale::ObjFeatures::gradientFeatures(bbox, compCount, labeledMask, bgr[2]);
	textureList[RED_CHANNEL].h_cannyFeatures     = nscale::ObjFeatures::cannyFeatures(bbox, compCount, labeledMask, bgr[2]);

	textureList[RED_CHANNEL].h_cytoIntensityFeatures = nscale::ObjFeatures::cytoIntensityFeatures(cytoplasmBB, compCount, bgr[2]);
	textureList[RED_CHANNEL].h_cytoGradientFeatures = nscale::ObjFeatures::cytoGradientFeatures(cytoplasmBB, compCount, bgr[2]);
	textureList[RED_CHANNEL].h_cytoCannyFeatures = nscale::ObjFeatures::cytoCannyFeatures(cytoplasmBB, compCount, bgr[2]);

	// Blue channel
	textureList[BLUE_CHANNEL].h_intensityFeatures = nscale::ObjFeatures::intensityFeatures(bbox, compCount, labeledMask, bgr[0]);
	textureList[BLUE_CHANNEL].h_gradientFeatures  = nscale::ObjFeatures::gradientFeatures(bbox, compCount, labeledMask, bgr[0]);
	textureList[BLUE_CHANNEL].h_cannyFeatures     = nscale::ObjFeatures::cannyFeatures(bbox, compCount, labeledMask, bgr[0]);

	textureList[BLUE_CHANNEL].h_cytoIntensityFeatures = nscale::ObjFeatures::cytoIntensityFeatures(cytoplasmBB, compCount, bgr[0]);
	textureList[BLUE_CHANNEL].h_cytoGradientFeatures = nscale::ObjFeatures::cytoGradientFeatures(cytoplasmBB, compCount, bgr[0]);
	textureList[BLUE_CHANNEL].h_cytoCannyFeatures = nscale::ObjFeatures::cytoCannyFeatures(cytoplasmBB, compCount, bgr[0]);
	return 0;
}

