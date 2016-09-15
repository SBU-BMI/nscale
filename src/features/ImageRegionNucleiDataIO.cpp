#include "ImageRegionNucleiDataIO.h"

int savePolygonMask(char *outFile, PolygonList *contours, int compCount, int rows, int cols) 
{
	cv::Mat drawing(rows,cols, CV_8UC3, Scalar(0,0,0));

    cv::Scalar color = cv::Scalar(255, 255, 255);
	for (int i=0;i<compCount;i++) {
		if (contours[i].size()==1 && cv::contourArea(contours[i][0])>50)
			for (int j=0;j<contours[i].size();j++) {
           		cv::drawContours(drawing, contours[i], j, color, 1, 8, noArray(), 0, Point());
			}
	}

	cv::imwrite(outFile,drawing);
	return 0;
}

int writeCSVFile(char *outFile, ShapeFeatureList& shapeList, TextureFeatureList* textureList, int* bbox, int compCount, int locX, int locY) 
{  
	std::ofstream outfile;
	outfile.open(outFile);

	outfile << "minx,miny,maxx,maxy,"
			<< "Area,Perimeter,MajorAxis,MinorAxis,Eccentricity,ExtentRatio,Circularity,"
			<< "r_IntensityMean,r_IntensityMax,r_IntensityMin,r_IntensityStd,"
			<< "r_IntensityEntropy,r_IntensityEnergy,r_IntensitySkewness,r_IntensityKurtosis,"
			<< "r_GradientMean,r_GradientStd,r_GradientEntropy,r_GradientEnergy,"
			<< "r_GradientSkewness,r_GradientKurtosis,"
			<< "r_CannyNonZero,r_CannyMean,"
			<< "r_cytoIntensityMean,r_cytoIntensityMax,r_cytoIntensityMin,r_cytoIntensityStd,"
			<< "r_cytoIntensityEntropy,r_cytoIntensityEnergy,r_cytoIntensitySkewness,r_cytoIntensityKurtosis,"
			<< "r_cytoGradientMean,r_cytoGradientStd,r_cytoGradientEntropy,r_cytoGradientEnergy,"
			<< "r_cytoGradientSkewness,r_cytoGradientKurtosis,"
			<< "r_cytoCannyNonZero,r_cytoCannyMean,"
			<< "b_IntensityMean,b_IntensityMax,b_IntensityMin,b_IntensityStd,"
			<< "b_IntensityEntropy,b_IntensityEnergy,b_IntensitySkewness,b_IntensityKurtosis,"
			<< "b_GradientMean,b_GradientStd,b_GradientEntropy,b_GradientEnergy,"
			<< "b_GradientSkewness,b_GradientKurtosis,"
			<< "b_CannyNonZero,b_CannyMean,"
			<< "b_cytoIntensityMean,b_cytoIntensityMax,b_cytoIntensityMin,b_cytoIntensityStd,"
			<< "b_cytoIntensityEntropy,b_cytoIntensityEnergy,b_cytoIntensitySkewness,b_cytoIntensityKurtosis,"
			<< "b_cytoGradientMean,b_cytoGradientStd,b_cytoGradientEntropy,b_cytoGradientEnergy,"
			<< "b_cytoGradientSkewness,b_cytoGradientKurtosis,"
			<< "b_cytoCannyNonZero,b_cytoCannyMean\n";

	for (int i=0;i<compCount;i++) {
		int label = bbox[i];
		int minx  = bbox[compCount+i];
		int maxx  = bbox[compCount*2+i];
		int miny  = bbox[compCount*3+i];
		int maxy  = bbox[compCount*4+i];

		// bounding box
		outfile 	<< minx+locX << "," << miny+locY << "," << maxx+locX << "," << maxy+locY << ",";

		// Shape features
		outfile  	<< shapeList.cpuArea[i] << "," 
					<< shapeList.cpuPerimeter[i] << "," 
					<< shapeList.cpuMajorAxis[i] << "," 
					<< shapeList.cpuMinorAxis[i] << "," 
					<< shapeList.cpuEccentricity[i] << "," 
					<< shapeList.cpuExtentRatio[i] << "," 
					<< shapeList.cpuCircularity[i] << ",";

		// Red channel texture features
		outfile 	<< textureList[RED_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[RED_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[RED_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< textureList[RED_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","; 
		outfile 	<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[RED_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[RED_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[RED_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< textureList[RED_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","; 

		// Blue channel texture features
		outfile 	<< textureList[BLUE_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[BLUE_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[BLUE_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< textureList[BLUE_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","; 
		outfile 	<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[BLUE_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< textureList[BLUE_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< textureList[BLUE_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< textureList[BLUE_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] 
					<< "\n";
	} 

	outfile.close();
	return 0;
}


int writeU24CSVFile(char *outFile, ImageRegionNucleiData& nucleiData)
{  
	std::ofstream outfile;
	outfile.open(outFile);
	int compCount = nucleiData.getNumberOfNuclei();

	outfile << "AreaInPixels,"  
			<< "Perimeter,MajorAxis,MinorAxis,Eccentricity,ExtentRatio,Circularity,"
			<< "r_IntensityMean,r_IntensityMax,r_IntensityMin,r_IntensityStd,"
			<< "r_IntensityEntropy,r_IntensityEnergy,r_IntensitySkewness,r_IntensityKurtosis,"
			<< "r_GradientMean,r_GradientStd,r_GradientEntropy,r_GradientEnergy,"
			<< "r_GradientSkewness,r_GradientKurtosis,"
			<< "r_CannyNonZero,r_CannyMean,"
			<< "r_cytoIntensityMean,r_cytoIntensityMax,r_cytoIntensityMin,r_cytoIntensityStd,"
			<< "r_cytoIntensityEntropy,r_cytoIntensityEnergy,r_cytoIntensitySkewness,r_cytoIntensityKurtosis,"
			<< "r_cytoGradientMean,r_cytoGradientStd,r_cytoGradientEntropy,r_cytoGradientEnergy,"
			<< "r_cytoGradientSkewness,r_cytoGradientKurtosis,"
			<< "r_cytoCannyNonZero,r_cytoCannyMean,"
			<< "b_IntensityMean,b_IntensityMax,b_IntensityMin,b_IntensityStd,"
			<< "b_IntensityEntropy,b_IntensityEnergy,b_IntensitySkewness,b_IntensityKurtosis,"
			<< "b_GradientMean,b_GradientStd,b_GradientEntropy,b_GradientEnergy,"
			<< "b_GradientSkewness,b_GradientKurtosis,"
			<< "b_CannyNonZero,b_CannyMean,"
			<< "b_cytoIntensityMean,b_cytoIntensityMax,b_cytoIntensityMin,b_cytoIntensityStd,"
			<< "b_cytoIntensityEntropy,b_cytoIntensityEnergy,b_cytoIntensitySkewness,b_cytoIntensityKurtosis,"
			<< "b_cytoGradientMean,b_cytoGradientStd,b_cytoGradientEntropy,b_cytoGradientEnergy,"
			<< "b_cytoGradientSkewness,b_cytoGradientKurtosis,"
			<< "b_cytoCannyNonZero,b_cytoCannyMean,"
			<< "Polygon\n";

	PolygonList *contours = nucleiData.getPolygons(); // contours representing nucleus boundaries
	int locX = nucleiData.getImageRegionMinx();
	int locY = nucleiData.getImageRegionMiny();

	for (int i=0;i<compCount;i++) {

		int *bbox = nucleiData.getBoundingBoxes();

		outfile  	<< nucleiData.getShapeList().cpuArea[i] << ","; 

		// Shape features
		outfile  	<< nucleiData.getShapeList().cpuPerimeter[i] << "," 
					<< nucleiData.getShapeList().cpuMajorAxis[i] << "," 
					<< nucleiData.getShapeList().cpuMinorAxis[i] << "," 
					<< nucleiData.getShapeList().cpuEccentricity[i] << "," 
					<< nucleiData.getShapeList().cpuExtentRatio[i] << "," 
					<< nucleiData.getShapeList().cpuCircularity[i] << ",";

		// Red channel texture features
		outfile 	<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","; 
		outfile 	<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ",";

		// Blue channel texture features
		outfile 	<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","; 
		outfile 	<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << ","
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ","
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << ",";

		std::cout << "Polygons: " << contours[i].size() << std::endl;
		outfile << "[";
		unsigned int ptc;
		for (ptc = 0; ptc < contours[i][0].size()-1; ++ptc) {
			outfile << (contours[i][0][ptc].x + locX) << ":";
			outfile << (contours[i][0][ptc].y + locY) << ":";
		}
		outfile << (contours[i][0][ptc].x + locX) << ":";
		outfile << (contours[i][0][ptc].y + locY) << "]";

		outfile	<< "\n";
	} 

	outfile.close();
	return 0;
}



int writeTSVFile(char *outFile, ImageRegionNucleiData& nucleiData)
{  
	std::ofstream outfile;
	outfile.open(outFile);
	int compCount = nucleiData.getNumberOfNuclei();

	outfile << "ObjectID\tX\tY\tArea\t"  
			<< "Perimeter\tMajorAxis\tMinorAxis\tEccentricity\tExtentRatio\tCircularity\t"
			<< "r_IntensityMean\tr_IntensityMax\tr_IntensityMin\tr_IntensityStd\t"
			<< "r_IntensityEntropy\tr_IntensityEnergy\tr_IntensitySkewness\tr_IntensityKurtosis\t"
			<< "r_GradientMean\tr_GradientStd\tr_GradientEntropy\tr_GradientEnergy\t"
			<< "r_GradientSkewness\tr_GradientKurtosis\t"
			<< "r_CannyNonZero\tr_CannyMean\t"
			<< "r_cytoIntensityMean\tr_cytoIntensityMax\tr_cytoIntensityMin\tr_cytoIntensityStd\t"
			<< "r_cytoIntensityEntropy\tr_cytoIntensityEnergy\tr_cytoIntensitySkewness\tr_cytoIntensityKurtosis\t"
			<< "r_cytoGradientMean\tr_cytoGradientStd\tr_cytoGradientEntropy\tr_cytoGradientEnergy\t"
			<< "r_cytoGradientSkewness\tr_cytoGradientKurtosis\t"
			<< "r_cytoCannyNonZero\tr_cytoCannyMean\t"
			<< "b_IntensityMean\tb_IntensityMax\tb_IntensityMin\tb_IntensityStd\t"
			<< "b_IntensityEntropy\tb_IntensityEnergy\tb_IntensitySkewness\tb_IntensityKurtosis\t"
			<< "b_GradientMean\tb_GradientStd\tb_GradientEntropy\tb_GradientEnergy\t"
			<< "b_GradientSkewness\tb_GradientKurtosis\t"
			<< "b_CannyNonZero\tb_CannyMean\t"
			<< "b_cytoIntensityMean\tb_cytoIntensityMax\tb_cytoIntensityMin\tb_cytoIntensityStd\t"
			<< "b_cytoIntensityEntropy\tb_cytoIntensityEnergy\tb_cytoIntensitySkewness\tb_cytoIntensityKurtosis\t"
			<< "b_cytoGradientMean\tb_cytoGradientStd\tb_cytoGradientEntropy\tb_cytoGradientEnergy\t"
			<< "b_cytoGradientSkewness\tb_cytoGradientKurtosis\t"
			<< "b_cytoCannyNonZero\tb_cytoCannyMean\t"
			<< "Boundaries\n";

	PolygonList *contours = nucleiData.getPolygons(); // contours representing nucleus boundaries
	int locX = nucleiData.getImageRegionMinx();
	int locY = nucleiData.getImageRegionMiny();

	for (int i=0;i<compCount;i++) {

		int *bbox = nucleiData.getBoundingBoxes();

		outfile 	<< i << "\t";
		outfile 	<< bbox[compCount+i] << "\t";    // minx
		outfile 	<< bbox[compCount*3+i] << "\t";  // miny
		outfile  	<< nucleiData.getShapeList().cpuArea[i] << "\t"; 

		// Shape features
		outfile  	<< nucleiData.getShapeList().cpuPerimeter[i] << "\t" 
					<< nucleiData.getShapeList().cpuMajorAxis[i] << "\t" 
					<< nucleiData.getShapeList().cpuMinorAxis[i] << "\t" 
					<< nucleiData.getShapeList().cpuEccentricity[i] << "\t" 
					<< nucleiData.getShapeList().cpuExtentRatio[i] << "\t" 
					<< nucleiData.getShapeList().cpuCircularity[i] << "\t";

		// Red channel texture features
		outfile 	<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t"; 
		outfile 	<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[RED_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t";

		// Blue channel texture features
		outfile 	<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_gradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t"; 
		outfile 	<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[1 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[2 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "," 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[3 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[4 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[5 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[6 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoIntensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[0 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[1 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[2 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[3 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[4 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t" 
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoGradientFeatures[5 + nscale::ObjFeatures::N_GRADIENT_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoCannyFeatures[0 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t"
					<< nucleiData.getTextureList()[BLUE_CHANNEL].h_cytoCannyFeatures[1 + nscale::ObjFeatures::N_CANNY_FEATURES * i] << "\t";

		std::cout << "Polygons: " << contours[i].size() << std::endl;

		for (int idx = 0; idx < contours[i].size(); idx++) {
			if (contours[i][idx].size()>2) {
				for (unsigned int ptc = 0; ptc < contours[i][idx].size(); ++ptc) {
					outfile << (contours[i][idx][ptc].x + locX) << ",";
					outfile << (contours[i][idx][ptc].y + locY) << ";";
				}
			}
		}

		outfile		<< "\n";
	} 

	outfile.close();
	return 0;
}

int updateFCSFileHeaderTextSegment(FILE *outfile, int dataOffset, int64_t dataLen, 
									int compCount, int numFeatures, char *featureNames[]);

int initFCSFileHeaderTextSegment(FILE *outfile, int numFeatures, char *featureNames[])
{
	 return updateFCSFileHeaderTextSegment(outfile,1,2,2,numFeatures,featureNames); 
}

int updateFCSFileHeaderTextSegment(FILE *outfile, int dataOffset, int64_t dataLen, 
									int compCount, int numFeatures, char *featureNames[]) 
{
	fseek(outfile,0,SEEK_SET);

	char byte_offset[128];
	int  str_length = 0;

	// [58,dataOffset-1] is where the TEXT segment is
	sprintf(byte_offset,"FCS3.1    %8d%8d%8d%8d%8d%8d",58,dataOffset-1,0,0,0,0);
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	// TEXT Segment
	sprintf(byte_offset,"/");
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	sprintf(byte_offset,"$BEGINDATA/%010d/$ENDDATA/%010d/",dataOffset,dataOffset+dataLen-1);
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	sprintf(byte_offset,"$BEGINANALYSIS/%010d/$ENDANALYSIS/%010d/",0,0);
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);
	
	sprintf(byte_offset,"$BEGINSTEXT/%010d/$ENDSTEXT/%010d/",0,0);
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	sprintf(byte_offset,"$BYTEORD/1,2,3,4/");
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);
	
	sprintf(byte_offset,"$DATATYPE/F/");
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	sprintf(byte_offset,"$MODE/L/");
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	sprintf(byte_offset,"$NEXTDATA/0/");
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);

	sprintf(byte_offset,"$PAR/%04d/",numFeatures);
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);
	for (int i=0;i<numFeatures;i++) {
		sprintf(byte_offset,"$P%dB/32/",i+1);
		fprintf(outfile,"%s",byte_offset);
		str_length += strlen(byte_offset);
		sprintf(byte_offset,"$P%dE/0,0/",i+1);
		fprintf(outfile,"%s",byte_offset);
		str_length += strlen(byte_offset);
		sprintf(byte_offset,"$P%dN/%s/",i+1,featureNames[i]);
		fprintf(outfile,"%s",byte_offset);
		str_length += strlen(byte_offset);
		sprintf(byte_offset,"$P%dR/1000000/",i+1);
		fprintf(outfile,"%s",byte_offset);	
		str_length += strlen(byte_offset);
	}

	sprintf(byte_offset,"$TOT/%010d/",compCount);
	fprintf(outfile,"%s",byte_offset);
	str_length += strlen(byte_offset);
	fflush(outfile);

	std::cout << "Length: " << str_length << std::endl;
	return str_length;
}

int64_t writeFCSDataSegment(FILE *outfile, ShapeFeatureList& shapeList, TextureFeatureList* textureList, 
							int compCount, int *featureIdx, int numFeatures)
{
	int64_t data_len = 0;
	float   data_val = 0.0;
	for (int i=0;i<compCount;i++) {
		for (int j=0;j<numFeatures;j++) {
			int idx = featureIdx[j];
			switch (idx) {
				case 0: 
					data_val = (float)shapeList.cpuArea[i];
					break;
				case 1:
					data_val = (float)shapeList.cpuCircularity[i];
					break;
				case 2:
					data_val = (float)textureList[RED_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i];
					break;
				case 3:
					data_val = (float)textureList[RED_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i];
					break;
				case 4: 
					data_val = (float)shapeList.cpuMajorAxis[i];
					break;
				case 5: 
					data_val = (float)shapeList.cpuMinorAxis[i];
					break;
				case 6:
					data_val = (float)textureList[BLUE_CHANNEL].h_intensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i];
					break;
				case 7:
					data_val = (float)textureList[BLUE_CHANNEL].h_cytoIntensityFeatures[0 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i];
					break;
				case 8:
					data_val = (float)textureList[BLUE_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i];
					break;
				case 9:
					data_val = (float)textureList[RED_CHANNEL].h_intensityFeatures[7 + nscale::ObjFeatures::N_INTENSITY_FEATURES * i];
					break;
				default: 
					std::cout << "ERROR in feature selection." << std::endl;
					break;
			}
			if (data_val<=0.0) {
				std::cout << "ERROR: Value is negative: " << idx << " = " << data_val << std::endl;
			}
			fwrite(&data_val,sizeof(float),1,outfile);
			data_len += sizeof(float);
		}
	}

	return data_len;
}

int writeU24CSVFileFromVector(char *outFile, ImageRegionNucleiData& nucleiData)
{  
	std::ofstream outfile;
	outfile.open(outFile);
	int compCount = nucleiData.getNumberOfNuclei();

	std::vector<std::string> featureNames = nucleiData.getFeatureNamesVector();
	for (int i=0;i<featureNames.size()-1;i++) 
		outfile << featureNames[i] << ",";
	outfile << featureNames[featureNames.size()-1] << std::endl;

	std::vector<std::vector<double> > featureValueVector = nucleiData.getFeaturesVector();
	int featSize = featureNames.size();
	for (int i=0;i<featureValueVector.size();i++) {
		for (int j=0;j<featSize-1;j++) {
			outfile << featureValueVector[i][j] << ",";
		}
		outfile << featureValueVector[i][featSize-1] << std::endl;
	}
	outfile.close();
	return 0;
}


