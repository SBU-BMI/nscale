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

void ImageRegionNucleiData::setFeatureNamesVector() 
{
	featureNames.push_back("AreaInPixels");
	// Computed in Yi's code -- featureNames.push_back("Perimeter");
	featureNames.push_back("MajorAxis");
	featureNames.push_back("MinorAxis");
	featureNames.push_back("Eccentricity");
	featureNames.push_back("ExtentRatio");
	featureNames.push_back("Circularity");

	featureNames.push_back("r_IntensityMean");
	featureNames.push_back("r_IntensityMax");
	featureNames.push_back("r_IntensityMin");
	featureNames.push_back("r_IntensityStd");
	featureNames.push_back("r_IntensityEntropy");
	featureNames.push_back("r_IntensityEnergy");
	featureNames.push_back("r_IntensitySkewness");
	featureNames.push_back("r_IntensityKurtosis");
	featureNames.push_back("r_GradientMean");
	featureNames.push_back("r_GradientStd");
	featureNames.push_back("r_GradientEntropy");
	featureNames.push_back("r_GradientEnergy");
	featureNames.push_back("r_GradientSkewness");
	featureNames.push_back("r_GradientKurtosis");
	featureNames.push_back("r_CannyNonZero");
	featureNames.push_back("r_CannyMean");
	featureNames.push_back("r_cytoIntensityMean");
	featureNames.push_back("r_cytoIntensityMax");
	featureNames.push_back("r_cytoIntensityMin");
	featureNames.push_back("r_cytoIntensityStd");
	featureNames.push_back("r_cytoIntensityEntropy");
	featureNames.push_back("r_cytoIntensityEnergy");
	featureNames.push_back("r_cytoIntensitySkewness");
	featureNames.push_back("r_cytoIntensityKurtosis");
	featureNames.push_back("r_cytoGradientMean");
	featureNames.push_back("r_cytoGradientStd");
	featureNames.push_back("r_cytoGradientEntropy");
	featureNames.push_back("r_cytoGradientEnergy");
	featureNames.push_back("r_cytoGradientSkewness");
	featureNames.push_back("r_cytoGradientKurtosis");
	featureNames.push_back("r_cytoCannyNonZero");
	featureNames.push_back("r_cytoCannyMean");

	featureNames.push_back("b_IntensityMean");
	featureNames.push_back("b_IntensityMax");
	featureNames.push_back("b_IntensityMin");
	featureNames.push_back("b_IntensityStd");
	featureNames.push_back("b_IntensityEntropy");
	featureNames.push_back("b_IntensityEnergy");
	featureNames.push_back("b_IntensitySkewness");
	featureNames.push_back("b_IntensityKurtosis");
	featureNames.push_back("b_GradientMean");
	featureNames.push_back("b_GradientStd");
	featureNames.push_back("b_GradientEntropy");
	featureNames.push_back("b_GradientEnergy");
	featureNames.push_back("b_GradientSkewness");
	featureNames.push_back("b_GradientKurtosis");
	featureNames.push_back("b_CannyNonZero");
	featureNames.push_back("b_CannyMean");
	featureNames.push_back("b_cytoIntensityMean");
	featureNames.push_back("b_cytoIntensityMax");
	featureNames.push_back("b_cytoIntensityMin");
	featureNames.push_back("b_cytoIntensityStd");
	featureNames.push_back("b_cytoIntensityEntropy");
	featureNames.push_back("b_cytoIntensityEnergy");
	featureNames.push_back("b_cytoIntensitySkewness");
	featureNames.push_back("b_cytoIntensityKurtosis");
	featureNames.push_back("b_cytoGradientMean");
	featureNames.push_back("b_cytoGradientStd");
	featureNames.push_back("b_cytoGradientEntropy");
	featureNames.push_back("b_cytoGradientEnergy");
	featureNames.push_back("b_cytoGradientSkewness");
	featureNames.push_back("b_cytoGradientKurtosis");
	featureNames.push_back("b_cytoCannyNonZero");
	featureNames.push_back("b_cytoCannyMean");

	featureNames.push_back("objID");

	numFeatureNames = featureNames.size();
}

std::vector<std::vector<double> >& ImageRegionNucleiData::getFeatureValuesVector() {
	
	clearFeatureValuesVector();

	int compCount = getNumberOfNuclei();
	if (compCount<=0) return featureValuesVector;

	int maxLabel = 0;
	for (int i=0;i<compCount;i++) {
		int *bbox = getBoundingBoxes();
		if (bbox[i]>maxLabel) maxLabel = bbox[i];
	}
	featureValuesVector.resize(maxLabel);

	for (int i=0;i<compCount;i++) {
		int *bbox = getBoundingBoxes();
		int label = bbox[i];

		std::vector<double> featureValues;

		featureValues.push_back(getShapeList().cpuArea[i]); 

		// Shape features
		// Computed in Yi's code -- featureValues.push_back(getShapeList().cpuPerimeter[i]); 
		featureValues.push_back(getShapeList().cpuMajorAxis[i]); 
		featureValues.push_back(getShapeList().cpuMinorAxis[i]); 
		featureValues.push_back(getShapeList().cpuEccentricity[i]); 
		featureValues.push_back(getShapeList().cpuExtentRatio[i]); 
		featureValues.push_back(getShapeList().cpuCircularity[i]);

		// Red channel texture features
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i]);
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i]);
		featureValues.push_back(getTextureList()[RED_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i]);

		// Blue channel texture features
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i]);
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]); 
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i]);
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i]);
		featureValues.push_back(getTextureList()[BLUE_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i]);

		featureValues.push_back((double)label);

		int vecSize = featureValues.size();
		for (int j=0;j<vecSize;j++) 
			featureValuesVector[label-1].push_back(featureValues[j]);
	} 

	return featureValuesVector;
}
			

