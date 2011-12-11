/*
 * Operators.cpp
 *
 *  Created on: Jul 21, 2011
 *      Author: george
 */

#include "Operators.h"

// Assuming n > 0
int Operators::rndint(float n)//round float to the nearest integer
{	
	int ret = (int)floor(n);
	float t;
	t=n-floor(n);
	if (t>=0.5)    
	{
		ret = (int)floor(n) + 1;
	}
	return ret;
}

// Matlab calls it contrast
float Operators::inertiaFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	float inertia = 0.0;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j = 0; j < coocMatrixSize; j++){
			float ij = i-j;
			inertia += pow(ij, 2) * ((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount);
		}
	}
	return inertia;
}



float Operators::energyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	float energy = 0.0;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j = 0; j < coocMatrixSize; j++){
			energy += pow(((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount),2);
		}
	}
	return energy;
}



float Operators::entropyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	float entropy = 0.0;
	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j = 0; j < coocMatrixSize; j++){
			float auxDivision = (float)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount;
			if(auxDivision == 0)continue;
			entropy += (auxDivision) * log2(auxDivision);
		}
	}
	return (-1.0*entropy);
}



float Operators::homogeneityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	float homogeneity = 0.0;
	for(int i = 0; i < coocMatrixSize; i++){
		for(int j = 0; j < coocMatrixSize; j++){
			homogeneity += (1.0/(1.0 + pow((float)(i-j),2)))*((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount);
		}
	}
	return homogeneity;
}



float Operators::maximumProbabilityFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
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



float Operators::clusterShadeFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k)
{
	float clusterShade = 0.0;
	float mx = calcMxFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);
	float my = calcMyFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);

	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			clusterShade += pow((k-mx + j-my), 3) * ((float)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount);
		}
	}
	return clusterShade;
}



float Operators::clusterProminenceFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount, unsigned int k)
{
	float clusterProminence = 0.0;
	float mx = calcMxFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);
	float my = calcMyFromCoocMatrix(coocMatrix, coocMatrixSize, coocMatrixCount);

	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			clusterProminence += pow((k-mx + j-my), 4) * ((float)coocMatrix[i*coocMatrixSize + j]/coocMatrixCount);
		}
	}
	return clusterProminence;
}



float Operators::calcMxFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	float mx = 0.0;
	for(unsigned int i = 0; i < coocMatrixSize; i++){
		for(unsigned int j =0; j <coocMatrixSize; j++){
			mx += i * ((float)coocMatrix[i*coocMatrixSize + j]/(float)coocMatrixCount);
		}
	}
	return mx;

}

float Operators::calcMyFromCoocMatrix(unsigned int *coocMatrix, unsigned int coocMatrixSize, unsigned int coocMatrixCount)
{
	float my = 0.0;
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

	int height = inputImage->height;
	int width = inputImage->width;
	char *imageData = inputImage->imageData;
	char *imageDataMask = NULL;
	if(inputImageMask != NULL){
		imageDataMask = inputImageMask->imageData;
	}
	if(inputImage->roi != NULL){
		height = inputImage->roi->height;
		width = inputImage->roi->width;

		imageData += inputImage->roi->yOffset * inputImage->widthStep + inputImage->roi->xOffset * inputImage->nChannels;

		if(inputImageMask != NULL){
			imageDataMask += inputImage->roi->yOffset * inputImageMask->widthStep + inputImage->roi->xOffset * inputImageMask->nChannels;
		}
	}

	int intensity_hist_points = 0;
	//build histogram from the input image
	for (int i=0; i < height; i++){
		for(int j=0; j < width; j++){
			// verify if histogram is masked
			if(inputImageMask != NULL){
				// check if mask is not 0 for the given entry in the input image
//				if((int)(((unsigned char*)(inputImageMask->imageData + i * inputImageMask->widthStep + j))[0]) == 0){
				if((int)(((unsigned char*)(imageDataMask + i * inputImageMask->widthStep + j))[0]) == 0){
					continue;
				}
			}
			// get value of the input image pixel (i,j)
//			int histAddr = (int)(((unsigned char*)(inputImage->imageData + i * inputImage->widthStep + j))[0]);
			int histAddr = (int)(((unsigned char*)(imageData + i * inputImage->widthStep + j))[0]);
	
			// increment histogram entry according to value of pixel (i,j)
			hist[histAddr]++;
		}
	}


#ifdef	DEBUG
	for(int i=0;i<256;i++){
		intensity_hist_points += hist[i];
	}
	cout << "Total hist points = "<< intensity_hist_points<<endl;
#endif
	return hist;
}

#ifdef	USE_GPU
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
		cv::gpu::bitwise_and(*inputImage, *inputImageMask, *tempInput);

		// calc histogram
		cv::gpu::histEven(*tempInput, *histGPU, 257, 0, 256);

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
#endif

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

int Operators::calcNumElementsFromHistogram(CvHistogram *hist, int numBins)
{
	int numElements  = 0;
	for(int i = 0; i < numBins; i++){
		numElements += (float)cvQueryHistValue_1D(hist, i);
	}
	return numElements;
}

int Operators::calcMedianFromHistogram(CvHistogram *hist, int numBins)
{
	int median = 0;
	int numElements = Operators::calcNumElementsFromHistogram(hist, numBins);
	bool odd = numElements%2;

	// there is a central pixel that represents the median
	if(odd){
		int accElements = 0;
		for(int i = 0; i < numBins; i++){
			accElements += (int)cvQueryHistValue_1D(hist, i);;
			if(accElements >= (numElements/2 +1)){
				median = i;
			break;
			}
		}

	}else{
		// There isn't a central pixel, and the median is calculated as mean of two midle pixels values
		int accElements = 0;
		for(int i = 0; i < numBins; i++){
			accElements += (int)cvQueryHistValue_1D(hist, i);
			if(accElements >= (numElements/2)){
				// If two midle pixels are in the same bin
				if(accElements > (numElements/2)){
					median = i;
				}else{
					// accElements is equal to numElements/2, thus we have to find the 
					// next non NULL bin to calculate mean of midle pixels
					for(int j=i+1; j <numBins;j++){
						int j_element = (int)cvQueryHistValue_1D(hist, j);
						if(j_element != 0){
							median = Operators::rndint((float(i+j)/2.0));

					//		median = (i+j)/2;
							break;					
						}
					}
				}
				break;
			}
		}

	}
	return median;
}

float Operators::calcStdFromHistogram(CvHistogram *hist, int numBins)
{
	  if(numBins == 0)
	        return 0.0;

	  float sum = 0.0;
	  float numElements = 0;
	  for(int i = 0; i < numBins; i++){
		  sum += i * (float)cvQueryHistValue_1D(hist, i);
		  numElements += (float)cvQueryHistValue_1D(hist, i);
	  }

	  float mean = sum/numElements;

	  float sq_diff_sum = 0.0;
	  for(int i = 0; i < numBins; i++){
		  float diff = i - mean;
		  sq_diff_sum += diff * diff * (float)cvQueryHistValue_1D(hist, i);
	  }
	  float variance = sq_diff_sum/(numElements-1);
	  return sqrt(variance);
}


float Operators::calcEneryFromHistogram(CvHistogram *hist, int numBins)
{
	  float energy = 0.0;
	  float numElements = 0;
	  for(int i = 0; i < numBins; i++){
		  numElements += (float)cvQueryHistValue_1D(hist, i);
	  }

	  for(int i = 0; i < numBins; i++){
		  energy += pow((float)cvQueryHistValue_1D(hist, i)/numElements, 2);
	  }

	  return energy;
}

void Operators::printHistogram(CvHistogram *hist, int numBins){
	for(int i = 0; i < numBins; i++){
		cout << "i = "<< i<< " hist[i] = " << cvQueryHistValue_1D(hist, i) <<endl;
	}
}

float Operators::calcEntropyFromHistogram(CvHistogram *hist, int numBins)
{
//	cout << "Entropy "<<endl;
	float entropy = 0.0;
	float numElements = 0;
	for(int i = 0; i < numBins; i++){
		numElements += (float)cvQueryHistValue_1D(hist, i);
	}
	for(int i = 0; i < numBins; i++){
		float p_i = (float)cvQueryHistValue_1D(hist, i)/numElements;
		// ignore 0 entries	
		if((int)cvQueryHistValue_1D(hist, i) == 0)continue;
		entropy += ((p_i) * log2(p_i));
//		cout << "i="<< i<< " hist[i] = "<< cvQueryHistValue_1D(hist, i) <<"  p_i="<< p_i << " log2(p_i) = "<< log2(p_i)<<endl;
	}
//	cout << "End entropy" <<endl;
	return (-1.0*entropy);
}

// Using MATLAB default definition:
// k_1 = (1/n SUM_{i to n} ( x_i - x_avg)^4) / (1/n SUM_{i to n} ( x_i - x_avg)^2 )^2
float Operators::calcKurtosisFromHistogram(CvHistogram *hist, int numBins)
{
	float kurtosis = 0.0;
	float avg = 0.0;
	float numElements = 0;
	for(int i = 0; i < numBins; i++){
		numElements += (float)cvQueryHistValue_1D(hist, i);
		avg += i * (float)cvQueryHistValue_1D(hist, i);
	}
	//  AVG calculation
	avg /= numElements;


	float dividend = 0.0;
	float divisor = 0.0;
	
	for(int i = 0; i < numBins; i++){
		float e_i = (float)cvQueryHistValue_1D(hist, i);
		dividend += e_i * pow(i - avg, 4);
		divisor += e_i * pow(i - avg, 2);

	}

	if(dividend != 0 && divisor != 0 && numElements != 0){
		dividend /= numElements;
		divisor /= numElements;

		divisor = pow( divisor, 2);

		kurtosis = dividend / divisor;
	}
	return kurtosis;
}

// Using MATLAB default definition:
// k_1 = (1/n SUM_{i to n} ( x_i - x_avg)^3) / (  (1/n SUM_{i to n} ( x_i - x_avg)^2)^1/2 )^3
float Operators::calcSkewnessFromHistogram(CvHistogram *hist, int numBins)
{
	float skewness = 0.0;
	float avg = 0.0;
	float numElements = 0;
	for(int i = 0; i < numBins; i++){
		numElements += (float)cvQueryHistValue_1D(hist, i);
		avg += i * (float)cvQueryHistValue_1D(hist, i);
	}
	// finilizing AVG calculation
	avg /= numElements;

	float dividend = 0.0;
	float divisor = 0.0;
	
	for(int i = 0; i < numBins; i++){
		float e_i = (float)cvQueryHistValue_1D(hist, i);
		dividend += e_i * pow(i - avg, 3);
		divisor += e_i * pow(i - avg, 2);


	}

	if(dividend != 0 && divisor != 0 && numElements != 0){
		dividend /= numElements;
		divisor /= numElements;
		divisor = sqrt(divisor);
		divisor = pow( divisor, 3);
		skewness = dividend / divisor;
	}
	return skewness;
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
	  double variance = sq_diff_sum/(numElements-1);
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
	int median = 0;
	int numElements = Operators::calcNumElementsFromHistogram(hist, numBins);
	bool odd = numElements%2;

	// there is a central pixel that represents the median
	if(odd){
		int accElements = 0;
		for(int i = 0; i < numBins; i++){
			accElements += (int)hist[i];
			if(accElements >= (numElements/2 +1)){
				median = i;
			break;
			}
		}

	}else{
		// There isn't a central pixel, and the median is calculated as mean of two midle pixels values
		int accElements = 0;
		for(int i = 0; i < numBins; i++){
			accElements += (int)hist[i];
			if(accElements >= (numElements/2)){
				// If two midle pixels are in the same bin
				if(accElements > (numElements/2)){
					median = i;
				}else{
					// accElements is equal to numElements/2, thus we have to find the 
					// next non NULL bin to calculate mean of midle pixels
					for(int j=i+1; j <numBins;j++){
						int j_element = (int)hist[i];
						if(j_element != 0){
							median = i+j/2;
							break;					
						}
					}
				}
				break;
			}
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


