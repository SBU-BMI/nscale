/*
 * TestRegional.cpp
 *
 *  Created on: Jun 21, 2011
 *      Author: george
 */
#include <sys/time.h>
#include "RegionalMorphologyAnalysis.h"

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
	if(argc != 3){
		cout << "Usage: ./compFeatures <image-mask> <image>" <<endl;
		exit(1);
	}
	// Load input images
	IplImage *originalImageMask = readImage(argv[1]);
	IplImage *originalImage = readImage(argv[2]);

	// Find nuclei in image and create an internal representation for each of them.
	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(originalImageMask, originalImage, true);

	bool isNuclei = true;

	// This is another option for inialize the features computation, where the path to the images are given as parameter
//	RegionalMorphologyAnalysis *regional = new RegionalMorphologyAnalysis(argv[1], argv[2]);

	/////////////// Computes Morphometry based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	//	0) Area; 1) MajorAxisLength; 2) MinorAxisLength; 3) Eccentricity; 4) Orientation; 5) ConvexArea; 6) FilledArea; 7) EulerNumber; 
	// 	8) EquivalentDiameter; 9) Solidity; 10) ExtentRatio; 11) Perimeter; 12) ConvexDeficiency; 13) Compacteness/Circularity; 14) Porosity; 15) AspectRatio; 
	//	16) BendingEnergy; 17) ReflectionSymmetry; 18) CannyArea; 19) MeanCanny; 20) SobelArea;
	vector<vector<float> > morphoFeatures;

	regional->doMorphometryFeatures(morphoFeatures, isNuclei, originalImage);

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
	// 	0)MeanIntensity; 1)StdIntensity; 2)EnergyIntensity; 3)EntropyIntensity; 4)KurtosisIntensity; 5)SkewnessIntensity;
	//	6)MedianIntensity; 7)MinIntensity; 8)MaxIntensity; 9)FirstQuartileIntensity; 10)ThirdQuartileIntensity;
	vector<vector<float> > intensityFeatures;
	regional->doIntensityBlob(intensityFeatures, isNuclei, originalImage);

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
	// 0)Mean; 1)Std; 2)Energy; 3)Entropy; 4)Kurtosis; 5)Skewness;
	// 6)Median; 7)Min; 8)Max; 9)FirstQuartile; 10)ThirdQuartile;
	vector<vector<float> > gradientFeatures;
	regional->doGradientBlob(gradientFeatures, isNuclei, originalImage);

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
	// 0)Inertia; 1)Energy; 2)Entropy; 3)Homogeneity; 4)MaximumProbability; 5)ClusterShade; 6)ClusterProminence
	vector<vector<float> > haralickFeatures;
	regional->doCoocPropsBlob(haralickFeatures, isNuclei, originalImage);

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
