/*
 * TestRegional.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */
#include <sys/time.h>
#include "RegionalMorphologyAnalysis.h"
#include "ColorDeconv_final.h"
#include "BGR2GRAY.h"

IplImage *readImage(string imageFileName){
	IplImage *readImage = cvLoadImage(imageFileName.c_str(), 0 );
	if(readImage == NULL){
		cout << "Could not load image: "<< imageFileName <<endl;
		exit(1);
	}else{
		if(readImage->nChannels != 1){
			cout << "Error: Image should have only one channel"<<endl;
			exit(1);
		}
	}
	return readImage;
}

int main (int argc, char **argv){
	//initialize stain deconvolution matrix and channel selection matrix
	Mat b = (Mat_<char>(1,3) << 1, 1, 0);
	Mat M = (Mat_<double>(3,3) << 0.650, 0.072, 0, 0.704, 0.990, 0, 0.286, 0.105, 0);


	if(argc != 3){
      		cout << "Usage: ./compFeatures <image-mask> <image>" <<endl;
      		exit(1);
	}
	// Load input images
	IplImage *originalImageMask = readImage(argv[1]);
//	IplImage *originalImage = readImage(argv[2]);

	IplImage *originalImage = cvLoadImage(argv[2], 1);


	bool isNuclei = true;

	/* create new image for the grayscale version */
//	IplImage *grayscale = cvCreateImage( cvGetSize(originalImage), IPL_DEPTH_8U, 1 );
 
	/* CV_RGB2GRAY: convert BGR image to grayscale */
//	cvCvtColor( originalImage, grayscale, CV_BGR2GRAY );

	IplImage *grayscale = bgr2gray(originalImage);

	cvSaveImage("newGrayScale.png", grayscale);

	// Find nuclei in image and create an internal representation for each of them.
	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(originalImageMask, grayscale, true);

	// Create H and E images
	Mat image(originalImage);

	//initialize H and E channels
	Mat H = Mat::zeros(image.size(), CV_8UC1);
	Mat E = Mat::zeros(image.size(), CV_8UC1);

	ColorDeconv(image, M, b, H, E);

	IplImage ipl_image_H(H);
	IplImage ipl_image_E(E);


	// This is another option for inialize the features computation, where the path to the images are given as parameter
//	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);

	vector<vector<float> > nucleiFeatures;

	/////////////// Compute nuclei based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)BoundingBox (BB) X; 1) BB.y; 2) BB.width; 3) BB.height; 4) Centroid.x; 5) Centroid.y) 7)Area; 8)Perimeter; 9)Eccentricity; 
	//	10)Circularity/Compacteness; 11)MajorAxis; 12)MinorAxis; 13)ExtentRatio; 14)MeanIntensity 15)MaxIntensity; 16)MinIntensity; 
	//	17)StdIntensity; 18)EntropyIntensity; 19)EnergyIntensity; 20)SkewnessIntensity;	21)KurtosisIntensity; 22)MeanGrad; 23)StdGrad; 
	//	24)EntropyGrad; 25)EnergyGrad; 26)SkewnessGrad; 27)KurtosisGrad; 28)CannyArea; 29)MeanCanny
	regional->doNucleiPipelineFeatures(nucleiFeatures, grayscale);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < nucleiFeatures.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < nucleiFeatures[i].size(); j++){
			printf("Nuclei %d = %f ", j, nucleiFeatures[i][j]);
		}
		printf("\n");
	}
#endif

	nucleiFeatures.clear();
	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_G;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_G, grayscale);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < cytoplasmFeatures_G.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < cytoplasmFeatures_G[i].size(); j++){
			printf("Cytoplams_G %d = %f ", j, cytoplasmFeatures_G[i][j]);
		}
		printf("\n");
	}
#endif
	cytoplasmFeatures_G.clear();

	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_H;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_H, &ipl_image_H);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < cytoplasmFeatures_H.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < cytoplasmFeatures_H[i].size(); j++){
			printf("Cytoplams_H %d = %f ", j, cytoplasmFeatures_H[i][j]);
		}
		printf("\n");
	}
#endif


	cvSaveImage("ipl_image_h.png", &ipl_image_H);

	/////////////// Compute cytoplasm based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0)MeanIntensity; 1) MedianIntensity-MeanIntensity; 2)MaxIntensity; 3)MinIntensity; 4)StdIntensity; 5)EntropyIntensity;
	//	6)EnergyIntensity; 7)SkewnessIntensity; 8)KurtosisIntensity; 9)MeanGrad; 10)StdGrad; 11)EntropyGrad; 12)EnergyGrad;
	//	13)SkewnessGrad; 14)KurtosisGrad; 15)CannyArea; 16)MeanCanny;
	vector<vector<float> > cytoplasmFeatures_E;
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_E, &ipl_image_E);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < cytoplasmFeatures_E.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < cytoplasmFeatures_E[i].size(); j++){
			printf("Cytoplams_E %d = %f ", j, cytoplasmFeatures_E[i][j]);
		}
		printf("\n");
	}
#endif

	cvSaveImage("ipl_image_e.png", &ipl_image_E);
	delete regional;

//	namedWindow( "Color Image", CV_WINDOW_AUTOSIZE );
//	imshow( "Color Image", image );
//	imshow( "H Image", H );	
//	imshow( "E Image", E );
//	waitKey();


	cvReleaseImage(&originalImage);
	cvReleaseImage(&originalImageMask);
	cvReleaseImage(&grayscale);
	
	H.release();
	E.release();
	M.release();
	b.release();

}
