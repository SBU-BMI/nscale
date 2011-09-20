/*
 * TestRegional.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */
#include <sys/time.h>
#include "RegionalMorphologyAnalysis.h"

IplImage *readImage(string imageFileName){
	IplImage *readImage = cvLoadImage(imageFileName.c_str(), -1 );
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

	if(argc != 3){
		cout << "Usage: ./compFeatures <image-mask> <image>" <<endl;
		exit(1);
	}
	// Load input images
	IplImage *originalImageMask = readImage(argv[1]);
	IplImage *originalImage = readImage(argv[2]);

	// Find nuclei in image and create an internal representation for each of them.
	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(originalImageMask, originalImage);

	// This is another option for inialize the features computation, where the path to the images are given as parameter
//	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);

	/////////////// Computes Morphometry based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	//	Area; MajorAxisLength; MinorAxisLength; Eccentricity; Orientation; ConvexArea; FilledArea; EulerNumber; 
	// 	EquivalentDiameter; Solidity; Extent; Perimeter; ConvexDeficiency; Compacteness; Porosity; AspectRatio; 
	//	BendingEnergy; ReflectionSymmetry; CannyArea; SobelArea;
	vector<vector<float> > morphoFeatures;
	regional->doMorphometryFeatures(morphoFeatures);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < morphoFeatures.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < morphoFeatures[i].size(); j++){
			printf("MorphFeature %d = %f ", j, morphoFeatures[i][j]);
		}
		printf("\n");
	}
#endif

	/////////////// Computes Pixel Intensity based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	MeanIntensity; MedianIntensity; MinIntensity; MaxIntensity; FirstQuartileIntensity; ThirdQuartileIntensity;
	vector<vector<float> > intensityFeatures;
	regional->doIntensityBlob(intensityFeatures);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < intensityFeatures.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < intensityFeatures[i].size(); j++){
			printf("IntensityFeature %d = %f ", j, intensityFeatures[i][j]);
		}
		printf("\n");
	}
#endif
	/////////////// Computes Gradient based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// MeanGradMagnitude; MedianGradMagnitude; MinGradMagnitude; MaxGradMagnitude; FirstQuartileGradMagnitude; ThirdQuartileGradMagnitude;
	vector<vector<float> > gradientFeatures;
	regional->doGradientBlob(gradientFeatures);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < gradientFeatures.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < gradientFeatures[i].size(); j++){
			printf("GradientFeature %d = %f ", j, gradientFeatures[i][j]);
		}
		printf("\n");
	}
#endif
	/////////////// Computes Haralick based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	Inertia; Energy; Entropy; Homogeneity; MaximumProbability; ClusterShade; ClusterProminence
	vector<vector<float> > haralickFeatures;
	regional->doCoocPropsBlob(haralickFeatures);

#ifdef	PRINT_FEATURES
	for(int i = 0; i < haralickFeatures.size(); i++){
		printf("Id = %d ", i);
		for(int j = 0; j < haralickFeatures[i].size(); j++){
			printf("HaralickFeature %d = %f ", j, haralickFeatures[i][j]);
		}
		printf("\n");
	}
#endif
	delete regional;

	cvReleaseImage(&originalImage);
	cvReleaseImage(&originalImageMask);
}
