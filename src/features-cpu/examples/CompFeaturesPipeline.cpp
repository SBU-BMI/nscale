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

	vector<vector<float> > nucleiFeatures;

	/////////////// Compute nuclei based features ////////////////////////
	// Each line vector of features returned corresponds to a given nucleus, and contains the following features (one per column):
	// 	0) bb.x; 1) bb.y; 2) bb.width; 3) bb.height; 4) centroid.x; 5) centroid.y; 6)Area; 7)Perimeter; 8)Eccentricity; 
	//	9) Circularity/Compacteness; 10)MajorAxis; 11)MinorAxis; 12)ExtentRatio; 13)MeanIntensity; 14)MaxIntensity; 15)MinIntensity; 
	//	16) StdIntensity; 17)EntropyIntensity; 18)EnergyIntensity; 19)SkewnessIntensity; 20)KurtosisIntensity; 21)MeanGrad; 
	//	22)StdGrad; 23)EntropyGrad; 24)EnergyGrad; 25)SkewnessGrad; 26)KurtosisGrad; 27)CannyArea; 28)MeanCanny
	regional->doNucleiPipelineFeatures(nucleiFeatures, originalImage);

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
	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_G, originalImage);

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
//	vector<vector<float> > cytoplasmFeatures_M;
//	regional->doCytoplasmPipelineFeatures(cytoplasmFeatures_M, originalImage2);
//
//#ifdef	PRINT_FEATURES
//	for(int i = 0; i < cytoplasmFeatures_M.size(); i++){
//		printf("Id = %d ", i);
//		for(int j = 0; j < cytoplasmFeatures_M[i].size(); j++){
//			printf("Cytoplams %d = %f ", j, cytoplasmFeatures_M[i][j]);
//		}
//		printf("\n");
//	}
//#endif

	delete regional;

	cvReleaseImage(&originalImage);
	cvReleaseImage(&originalImageMask);
}
