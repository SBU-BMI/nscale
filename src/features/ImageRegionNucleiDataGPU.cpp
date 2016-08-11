#include "ImageRegionNucleiData.h"

int ImageRegionNucleiDataGPU::extractBoundingBoxesFromLabeledMask() 
{
	g_bbox = ::nscale::gpu::boundingBox2(g_labeledMask, compCount, &stream);
	stream.waitForCompletion();
	ASSERT_ERROR();

	if (compCount<=0) return 1;

	bbox = (int *)malloc(sizeof(int) * compCount * 5);
	cudaMemcpy(bbox, g_bbox, sizeof(int) * compCount * 5, cudaMemcpyDeviceToHost);

	return 0;
}

int ImageRegionNucleiDataGPU::extractCytoplasmRegions() 
{
	if (compCount<=0) return 1;

	cv::Mat labeledMask = cv::Mat::zeros(g_labeledMask.size(), g_labeledMask.type());
	stream.enqueueDownload(g_labeledMask, labeledMask);
	stream.waitForCompletion();
	ASSERT_ERROR();

	cytoplasmBB = nscale::CytoplasmCalc::calcCytoplasm(cytoplasmDataSize, bbox, compCount, labeledMask);

	g_cyto_bbox = (int*)nscale::gpu::cudaMallocCaller(cytoplasmDataSize);
	nscale::gpu::cudaUploadCaller(g_cyto_bbox, cytoplasmBB, cytoplasmDataSize);

	return 0;
}

int ImageRegionNucleiDataGPU::extractPolygonsFromLabeledMask()
{
	if (compCount<=0) return 1;

	cv::Mat labeledMask = cv::Mat::zeros(g_labeledMask.size(), g_labeledMask.type());
	stream.enqueueDownload(g_labeledMask, labeledMask);
	stream.waitForCompletion();
	ASSERT_ERROR();

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

int ImageRegionNucleiDataGPU::computeShapeFeatures()
{
	if (compCount<=0) return 1;

	cv::Mat labeledMask = cv::Mat::zeros(g_labeledMask.size(), g_labeledMask.type());
    stream.enqueueDownload(g_labeledMask, labeledMask);
    stream.waitForCompletion();
    ASSERT_ERROR();

    shapeList.cpuPerimeter = nscale::ObjFeatures::perimeter((const int*)bbox,compCount,labeledMask);
    shapeList.cpuArea = nscale::ObjFeatures::area((const int*)bbox , compCount , labeledMask);
    nscale::ObjFeatures::ellipse((const int*)bbox,(const int*)shapeList.cpuArea, compCount , labeledMask,
                                shapeList.cpuMajorAxis, shapeList.cpuMinorAxis, shapeList.cpuEccentricity);
    shapeList.cpuExtentRatio = nscale::ObjFeatures::extent_ratio((const int*)bbox , (const int)compCount , (const int *)shapeList.cpuArea);
    shapeList.cpuCircularity = nscale::ObjFeatures::circularity(compCount , shapeList.cpuArea , shapeList.cpuPerimeter);
    return 0;
}

int ImageRegionNucleiDataGPU::computeShapeFeaturesGPU()
{
	if (compCount<=0) return 1;
	
	// std::cout << "Computing Shape Features using GPU" << std::endl;

	int   *gpuArea = (int*)nscale::gpu::cudaMallocCaller(sizeof(int) * compCount);
	float *gpuPerimeter = (float*)nscale::gpu::cudaMallocCaller(sizeof(float) * compCount);
	float *gpuMajorAxis;
	float *gpuMinorAxis;
	float *gpuEccentricity;
	float *gpuExtentRatio;
	float *gpuCircularity;

	int *area = nscale::gpu::ObjFeatures::calculateArea(g_bbox , compCount , g_labeledMask , stream);
	nscale::gpu::cudaUploadCaller(gpuArea, area, sizeof(int) * compCount);

	float *perimeter = nscale::gpu::ObjFeatures::calculatePerimeter(g_bbox , compCount , g_labeledMask , stream);
	nscale::gpu::cudaUploadCaller(gpuPerimeter, perimeter, sizeof(int) * compCount);

	gpuExtentRatio = nscale::gpu::ObjFeatures::calculateExtentRatio (g_bbox , compCount , gpuArea , stream);
	gpuCircularity = nscale::gpu::ObjFeatures::calculateCircularity(compCount , gpuArea , gpuPerimeter , stream);

	nscale::gpu::ObjFeatures::calculateEllipse(g_bbox , compCount , g_labeledMask , gpuArea , gpuMajorAxis , gpuMinorAxis , gpuEccentricity , stream);

	shapeList.cpuArea = (int*) malloc(sizeof(int)*compCount);
	shapeList.cpuPerimeter = (double*)malloc(sizeof(double)*compCount);
	shapeList.cpuMajorAxis = (double*)malloc(sizeof(double)*compCount);
	shapeList.cpuMinorAxis = (double*)malloc(sizeof(double)*compCount);
	shapeList.cpuEccentricity = (double*)malloc(sizeof(double)*compCount);
	shapeList.cpuExtentRatio = (double*)malloc(sizeof(double)*compCount);
	shapeList.cpuCircularity = (double*)malloc(sizeof(double)*compCount);
	if (shapeList.cpuArea==NULL || shapeList.cpuPerimeter==NULL ||
		shapeList.cpuMajorAxis==NULL || shapeList.cpuMinorAxis==NULL ||
		shapeList.cpuEccentricity==NULL || shapeList.cpuExtentRatio==NULL ||
		shapeList.cpuCircularity==NULL) 
		return 1;

	for (int i=0;i<compCount;i++) {
		shapeList.cpuArea[i]         = (int) area[i];
		shapeList.cpuPerimeter[i]    = (double) perimeter[i];
		shapeList.cpuMajorAxis[i]    = (double) gpuMajorAxis[i]; 
		shapeList.cpuMinorAxis[i]    = (double) gpuMinorAxis[i]; 
		shapeList.cpuEccentricity[i] = (double) gpuEccentricity[i]; 
		shapeList.cpuExtentRatio[i]  = (double) gpuExtentRatio[i]; 
		shapeList.cpuCircularity[i]  = (double) gpuCircularity[i]; 
	}

	free(area);
	free(perimeter);
	free(gpuCircularity);
	free(gpuExtentRatio);
	nscale::gpu::cudaFreeCaller(gpuArea);
	nscale::gpu::cudaFreeCaller(gpuPerimeter);
	free(gpuMajorAxis);
	free(gpuMinorAxis);
	free(gpuEccentricity);

	return 0;
}

int ImageRegionNucleiDataGPU::computeRedBlueChannelTextureFeatures()
{
	if (compCount<=0) return 1;

	cv::gpu::GpuMat g_gray;

	std::cout << "Rows and Cols: " << imgTile.rows << " " << imgTile.cols << std::endl;

	// Create gray image by splitting color image into B,G,R arrays
	vector<cv::Mat> bgr;
    cv::split(imgTile, bgr);

	// RED Channel
	stream.enqueueUpload(bgr[2], g_gray);
    stream.waitForCompletion();

	std::cout << "NEW Rows and Cols: " << g_gray.rows << " " << g_gray.cols << std::endl;
	std::cout << "Mask Rows and Cols: " << g_labeledMask.rows << " " << g_labeledMask.cols << std::endl;

	textureList[RED_CHANNEL].h_intensityFeatures = nscale::gpu::ObjFeatures::intensityFeatures(g_bbox,  compCount, g_labeledMask , g_gray, stream);
    // stream.waitForCompletion();
	textureList[RED_CHANNEL].h_gradientFeatures = nscale::gpu::ObjFeatures::gradientFeatures(g_bbox, compCount, g_labeledMask, g_gray, stream);
	// stream.waitForCompletion();
	textureList[RED_CHANNEL].h_cannyFeatures = nscale::gpu::ObjFeatures::cannyFeatures(g_bbox, compCount, g_labeledMask, g_gray, stream);
	// stream.waitForCompletion();

	textureList[RED_CHANNEL].h_cytoIntensityFeatures = nscale::gpu::ObjFeatures::cytoIntensityFeatures(g_cyto_bbox,  compCount, g_gray, stream);
	// stream.waitForCompletion();
	textureList[RED_CHANNEL].h_cytoGradientFeatures = nscale::gpu::ObjFeatures::cytoGradientFeatures(g_cyto_bbox, compCount, g_gray, stream);
	// stream.waitForCompletion();
	textureList[RED_CHANNEL].h_cytoCannyFeatures = nscale::gpu::ObjFeatures::cytoCannyFeatures(g_cyto_bbox, compCount, g_gray, stream);
	stream.waitForCompletion();

	std::cout << "NEW Rows and Cols: " << imgTile.rows << " " << imgTile.cols << std::endl;
   
	// BLUE Channel
	stream.enqueueUpload(bgr[0], g_gray);
    stream.waitForCompletion();

	textureList[BLUE_CHANNEL].h_intensityFeatures = nscale::gpu::ObjFeatures::intensityFeatures(g_bbox,  compCount, g_labeledMask , g_gray, stream);
    // stream.waitForCompletion();
	textureList[BLUE_CHANNEL].h_gradientFeatures = nscale::gpu::ObjFeatures::gradientFeatures(g_bbox, compCount, g_labeledMask, g_gray, stream);
	// stream.waitForCompletion();
	textureList[BLUE_CHANNEL].h_cannyFeatures = nscale::gpu::ObjFeatures::cannyFeatures(g_bbox, compCount, g_labeledMask, g_gray, stream);
	// stream.waitForCompletion();

	textureList[BLUE_CHANNEL].h_cytoIntensityFeatures = nscale::gpu::ObjFeatures::cytoIntensityFeatures(g_cyto_bbox,  compCount, g_gray, stream);
	// stream.waitForCompletion();
	textureList[BLUE_CHANNEL].h_cytoGradientFeatures = nscale::gpu::ObjFeatures::cytoGradientFeatures(g_cyto_bbox, compCount, g_gray, stream);
	// stream.waitForCompletion();
	textureList[BLUE_CHANNEL].h_cytoCannyFeatures = nscale::gpu::ObjFeatures::cytoCannyFeatures(g_cyto_bbox, compCount, g_gray, stream);
	stream.waitForCompletion();

	g_gray.release();

	return 0;
}
