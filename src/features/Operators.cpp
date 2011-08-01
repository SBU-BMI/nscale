/*
 * Operators.cpp
 *
 *  Created on: Jul 21, 2011
 *      Author: george
 */

#include "Operators.h"
// Matlab calls it contrast
double Operators::inertiaFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double inertia = 0.0;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j = 0; j < coocMatrixSize; j++){
			float ij = i-j;
			inertia += pow(ij, 2) * ((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount);
		}
	}
	return inertia;
}



double Operators::energyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double energy = 0.0;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j = 0; j < coocMatrixSize; j++){
			energy += pow(((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount),2);
		}
	}
	return energy;
}



double Operators::entropyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double entropy = 0.0;
	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j = 0; j < coocMatrixSize; j++){
			double auxDivision = (double)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount;
			if(auxDivision == 0)continue;
			entropy += (auxDivision) * log2(auxDivision);
		}
	}
	return entropy;
}



double Operators::homogeneityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double homogeneity = 0.0;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j = 0; j < coocMatrixSize; j++){
			homogeneity += (1.0/(1.0 + pow((float)(i-j),2)))*((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount);
		}
	}
	return homogeneity;
}



double Operators::maximumProbabilityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double maximumProbability = 0.0;
	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j = 0; j < coocMatrixSize; j++){
			if(maximumProbability < coocMatrix[i*coocMatrixSize + j]){
				maximumProbability = coocMatrix[i*coocMatrixSize + j];
			}
		}
	}
	return maximumProbability/coocMatrixCount;
}



double Operators::clusterShadeFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k)
{
	double clusterShade = 0.0;
	double mx = calcMxFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);
	double my = calcMyFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);

	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			clusterShade += pow((k-mx + j-my), 3) * ((float)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount);
		}
	}
	return clusterShade;
}



double Operators::clusterProminenceFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k)
{
	double clusterProminence = 0.0;
	double mx = calcMxFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);
	double my = calcMyFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);

	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			clusterProminence += pow((k-mx + j-my), 4) * ((float)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount);
		}
	}
	return clusterProminence;
}



double Operators::calcMxFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double mx = 0.0;
	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			mx += i * ((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount);
		}
	}
	return mx;

}

double Operators::calcMyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	double my = 0.0;
	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			my += j * ((float)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount);
		}
	}
	return my;
}

// inputImage		-> The grayscale image from where the histogram should be calculated
// inputImageMaks 	-> The mask describing the point from the input image that should be
//					   considered when calculating the histogram
unsigned int *Operators::buildHistogram256CPU(IplImage *inputImage, IplImage * inputImageMask)
{
	if(CV_IS_IMAGE(inputImage) && inputImage->depth != IPL_DEPTH_8U){
		cout << "BuildHisogram256: input mat should be CV_8UC1" <<endl;
		exit(1);
	}

	unsigned int *hist = (unsigned int *)calloc(256, sizeof(unsigned int));
	if(hist == NULL){
		cout << "buildHistogram256: failed to allocate histogram memory" <<endl;
		exit(1);
	}

	int intensity_hist_points = 0;
	//build histogram from the input image
	for (int i=0; i<inputImage->height; i++){
//		unsigned char *ptr = (unsigned char*)(inputImage->imageData + i * inputImage->widthStep);
		for(int j=0; j<inputImage->width; j++){
//			CvScalar s=cvGet2D(inputImage, i, j);
//			hist[(unsigned int)s.val[0]]++;
			// verify if histogram is masked
			if(inputImageMask != NULL){
				// check if mask is not 0 for the given entry in the input image
				if((int)(((unsigned char*)(inputImageMask->imageData + i * inputImageMask->widthStep + j))[0]) == 0){
					continue;
				}else{
					if((int)(((unsigned char*)(inputImageMask->imageData + i * inputImageMask->widthStep + j))[0]) != 1){
						cout <<" Mask not 0 or 1.  = "<< (int)(((unsigned char*)(inputImageMask->imageData + i * inputImageMask->widthStep + j))[0]) <<endl;
						exit(1);
					}
				}
			}
			// get value of the input image pixel (i,j)
			int histAddr = (int)(((unsigned char*)(inputImage->imageData + i * inputImage->widthStep + j))[0]);

			// increment histogram entry according to value of pixel (i,j)
			hist[histAddr]++;
		}
	}

/*	int numBins = 256;
	float range[] = {0, 256};
	float *ranges[] = { range };
	CvHistogram* intensity_hist_temp = cvCreateHist(1, &numBins, CV_HIST_ARRAY, ranges, 1);

	IplImage * tempImg = cvCreateImage(cvSize(inputImage->width, inputImage->height), IPL_DEPTH_8U, 1);

	cvMul(inputImage, inputImageMask, tempImg);

	// Calculates the histogram in the input image for the pixels in the input mask
	cvCalcHist(&tempImg, intensity_hist_temp, 0);

	intensity_hist_points = 0;

	for(int i=0;i<256;i++){
			float histValue = cvQueryHistValue_1D(intensity_hist_temp, i);
			intensity_hist_points += (unsigned int)histValue;
			cout<< "mat["<<i<<"]="<<histValue<<endl;
	}*/


#ifdef	DEBUG
	for(int i=0;i<256;i++){
		intensity_hist_points += hist[i];
	}
	cout << "Total hist points = "<< intensity_hist_points<<endl;
#endif
	return hist;
}

// inputImage		-> The grayscale image from where the histogram should be calculated
// inputImageMaks 	-> The mask describing the point from the input image that should be
//					   considered when calculating the histogram
unsigned int *Operators::buildHistogram256GPU(cv::gpu::GpuMat *inputImage, cv::gpu::GpuMat *inputImageMask)
{
	// array with the histogram that is returned
	unsigned int *hist = (unsigned int *)calloc(256, sizeof(unsigned int));
	if(hist == NULL){
		cout << "buildHistogram256: failed to allocate histogram memory" <<endl;
		exit(1);
	}

	int zerosMask = 0;

	// Create memory used to store histogram results
	cv::gpu::GpuMat *histGPU = new cv::gpu::GpuMat(1, 256,  CV_32S);

	if(inputImageMask != NULL){
		// copy input image to a temporary, which can be modified
		cv::gpu::GpuMat *tempInput = new cv::gpu::GpuMat(inputImage->size(), CV_8UC1);

		cv::gpu::multiply(*inputImage, *inputImageMask, *tempInput);

		// calc histogram
		cv::gpu::histEven(*tempInput, *histGPU, 256, 0, 256);

		// calculate number of zeros in the mask, to subtract from the histogram results.
		int nonZeroMask = cv::gpu::countNonZero(*inputImageMask);

		zerosMask = (inputImage->rows * inputImage->cols) - nonZeroMask;

		delete tempInput;
	}else{
		cv::gpu::histEven(*inputImage, *histGPU, 257, 0, 256);
	}

	// download histogram to the CPU memory
	cv::Mat histCPU= (cv::Mat)*histGPU;

	// get pointer to the beginning of the histogram data
	int *M0 = histCPU.ptr<int>(0);

	// Copy hist data to the adequate data buffer.
	memcpy(hist, M0, sizeof(int) * 256);
	hist[0] -= zerosMask;

	delete histGPU;

	return hist;
}

double Operators::calcMeanFromHistogram(unsigned int *hist, unsigned int numBins)
{
	double histMean = 0.0;
	int numElements = 0;
	for(int i = 0; i < numBins; i++){
		histMean += hist[i] * i;
		numElements += hist[i];
	}
	if(numElements == 0) return 0.0;

	return histMean /= (numElements);
}



double Operators::calcStdFromHistogram(unsigned int *hist, unsigned int numBins)
{
	  if(numBins == 0)
	        return 0.0;

	  double sum = 0.0;
	  double numElements = 0;
	  for(int i = 0; i < numBins; i++){
		  sum += i * hist[i];
		  numElements += hist[i];
	  }

	  double mean = sum/numElements;

	  double sq_diff_sum = 0.0;
	  for(int i = 0; i < numBins; i++){
		  double diff = i - mean;
		  sq_diff_sum += diff * diff * hist[i];
	  }
	  double variance = sq_diff_sum/numElements;
	  return sqrt(variance);
}




int Operators::calcMinFromHistogram(unsigned int *hist, unsigned int numBins)
{
	unsigned int minIntensity = numBins;

	for(int i = 0; i < numBins; i++){
		if(hist[i] > 0){
			minIntensity = i;
			break;
		}
	}
	return minIntensity;
}



int Operators::calcMaxFromHistogram(unsigned int *hist, unsigned int numBins)
{
	unsigned int maxIntensity = 0;

	for(int i = numBins -1; i >= 0; i--){
		if( hist[i] > 0){
			maxIntensity = i;
			break;
		}
	}
	return maxIntensity;
}


int Operators::calcMedianFromHistogram(unsigned int *hist, unsigned int numBins)
{
	int numElements = Operators::calcNumElementsFromHistogram(hist, numBins);
	int accElements = 0;
	int median = 0;
	for(int i = 0; i < numBins; i++){
		accElements += hist[i];
		if(accElements >= numElements/2){
			median = i;
			break;
		}
	}
	return median;
}




int Operators::calcFirstQuartileFromHistogram(unsigned int *hist, unsigned int numBins)
{
	int numElements = Operators::calcNumElementsFromHistogram(hist, numBins);
	int accElements = 0;
	int firstQuartile = 0;
	for(int i = 0; i < numBins; i++){
		accElements += hist[i];
		if(accElements >= numElements/4){
			firstQuartile = i;
			break;
		}
	}
	return firstQuartile;
}



int Operators::calcSecondQuartileFromHistogram(unsigned int *hist, unsigned int numBins)
{
	return Operators::calcMedianFromHistogram(hist, numBins);
}



int Operators::calcThirdQuartileFromHistogram(unsigned int *hist, unsigned int numBins)
{
	int numElements = Operators::calcNumElementsFromHistogram(hist, numBins);
	int accElements = 0;
	int thirdQuarile = 0;
	for(int i = 0; i < numBins; i++){
		accElements += hist[i];
		if(accElements >= (3*numElements)/4){
			thirdQuarile = i;
			break;
		}
	}
	return thirdQuarile;
}

int Operators::calcNumElementsFromHistogram(unsigned int *hist, unsigned int numBins)
{
	int numElements  = 0;
	for(int i = 0; i < numBins; i++){
		numElements+=hist[i];
	}
	return numElements;
}










/*double Operators::correlationFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k)
{
	double correlation = 0.0;
	double mx = calcMxFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);
	double my = calcMyFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);

	cout << "mx = "<< mx<<endl;
	cout << "my = "<< my<<endl;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j =0; j <coocMatrixSize; j++){
			correlation += pow((k-mx + j-my), 4) * ((float)coocMatrix[i*coocMatrixSize + j]/ coocMatrixCount);
		}
	}
	return correlation;
}
*/


