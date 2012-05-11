/*
 * Operators.cpp
 *
 *  Created on: Jul 21, 2011
 *      Author: george
 */

#include "Operators.h"
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
	return entropy;
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

//		cv::Mat tempCpu = (cv::Mat)*tempInput;
//
//		cv::Mat inputCpu = (cv::Mat)*inputImage;
//
//		cv::Mat maskCpu = (cv::Mat)*inputImageMask;
//
//		for(int j =0; j < tempCpu.rows; j++){
//			cout << "bla"<<endl;
//
//			unsigned char *tempD = (unsigned char*) tempCpu.ptr<char*>(j);
//			for(int i = 0; i < tempCpu.cols; i++){
//
//				if((int)tempD[i] != 0){
//					cout << "temp["<< j <<"]["<< i <<"] = "<< (int)tempD[i] <<endl;
//
//					unsigned char *temp = (unsigned char*) inputCpu.ptr<char*>(j);
//					cout << "input["<< j <<"]["<< i <<"] = "<< (int)temp[i] <<endl;
//					temp = (unsigned char*) maskCpu.ptr<char*>(j);
//					cout << "mask["<< j <<"]["<< i <<"] = "<< (int)temp[i] <<endl;
//					
//
//					fflush(stdout);
//					exit(1);
//				}
//			}
//		}

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
//	for(int i = 0; i < 256; i++){
//		cout << "hist["<< i <<"]="<<hist[i]<<endl;
//	}
	delete histGPU;

	return hist;
}

double Operators::calcMeanFromHistogram( int *hist,  int numBins)
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



double Operators::calcStdFromHistogram( int *hist,  int numBins)
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




int Operators::calcMinFromHistogram( int *hist,  int numBins)
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



int Operators::calcMaxFromHistogram( int *hist,  int numBins)
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


int Operators::calcMedianFromHistogram( int *hist,  int numBins)
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




int Operators::calcFirstQuartileFromHistogram( int *hist,  int numBins)
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



int Operators::calcSecondQuartileFromHistogram( int *hist,  int numBins)
{
	return Operators::calcMedianFromHistogram(hist, numBins);
}



int Operators::calcThirdQuartileFromHistogram( int *hist,  int numBins)
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

int Operators::calcNumElementsFromHistogram( int *hist,  int numBins)
{
	int numElements  = 0;
	for(int i = 0; i < numBins; i++){
		numElements+=hist[i];
	}
	return numElements;
}

float Operators::calcEntropyFromHistogram(int* hist, int numBins) {

		float entropy = 0.0;
		float numElements = Operators::calcNumElementsFromHistogram(hist, numBins);

		for(int i = 0; i < numBins; i++){
			float p_i = (float)hist[i]/numElements;
			// ignore 0 entries
			if(hist[i] == 0)continue;
			entropy += ((p_i) * log2(p_i));
		}
		return (-1.0*entropy);
}

float Operators::calcEnergyFromHistogram(int* hist, int numBins) {
	  float energy = 0.0;
	  float numElements = Operators::calcNumElementsFromHistogram(hist, numBins);

	  for(int i = 0; i < numBins; i++){
		  energy += pow((float)hist[i]/numElements, 2);
	  }

	  return energy;
}

float Operators::calcSkewnessFromHistogram(int* hist, int numBins) {
	float skewness = 0.0;
	float avg = 0.0;
	float numElements = 0;
	for(int i = 0; i < numBins; i++){
		numElements += (float)hist[i];
		avg += i * (float)hist[i];
	}
	// finalizing AVG calculation
	avg /= numElements;

	float dividend = 0.0;
	float divisor = 0.0;

	for(int i = 0; i < numBins; i++){
		float e_i = (float)hist[i];
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

float Operators::calcKurtosisFromHistogram(int* hist, int numBins) {
	float kurtosis = 0.0;
	float avg = 0.0;
	float numElements = 0;

	for(int i = 0; i < numBins; i++){
		numElements += (float)hist[i];
		avg += i * (float)hist[i];
	}
	//  AVG calculation
	avg /= numElements;


	float dividend = 0.0;
	float divisor = 0.0;

	for(int i = 0; i < numBins; i++){
		float e_i = (float)hist[i];
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

int Operators::calcNonZeroFromHistogram(int* hist, int numBins) {
	int nonZeroElements = 0;
	for(int i = 1; i < numBins; i++){
		nonZeroElements += hist[i];
	}
	return nonZeroElements;
}

int* Operators::buildHistogram256CPU(const cv::Mat& labeledMask, const cv::Mat& grayImage,  int minx,  int maxx,  int miny,  int maxy,  int label) {
	int *hist = (int *)calloc(256, sizeof(int));;
	const int *labeledImgPtr;
	const unsigned char* grayImagePtr;

	for(int y = miny; y <= maxy; y++){
		labeledImgPtr =  labeledMask.ptr<int>(y);
		grayImagePtr = grayImage.ptr<unsigned char>(y);

		for(int x = minx; x <= maxx; x++){
			if(labeledImgPtr[x] == label){
				hist[grayImagePtr[x]]++;
			}
		}
	}
	return hist;
}

int* Operators::buildHistogram256CPUObjMask(const cv::Mat& objMask, const cv::Mat& grayImage,  int minx,  int maxx,  int miny,  int maxy,  int label) {

	int *hist = (int *)calloc(256, sizeof(int));;
	const int *objImgPtr;
	const unsigned char* grayImagePtr;

	for(int y = miny; y <= maxy; y++){
		objImgPtr =  objMask.ptr<int>(y-miny);
		grayImagePtr = grayImage.ptr<unsigned char>(y);

		for(int x = minx; x <= maxx; x++){
			if(objImgPtr[x-minx] == label){
				hist[grayImagePtr[x]]++;
			}
		}
	}
	return hist;
}

// Assuming n > 0
int rndint(float n)//round float to the nearest integer
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



void Operators::gradient(cv::Mat& inputImageMat, cv::Mat& gradientMat){

	IplImage inputImage = inputImageMat;

	IplImage* drv32f = cvCreateImage(cvGetSize(&inputImage), IPL_DEPTH_32F, 1);
	IplImage* magDrv = cvCreateImage(cvGetSize(&inputImage), IPL_DEPTH_32F, 1);
	IplImage* magDrvUint = cvCreateImage(cvGetSize(&inputImage), IPL_DEPTH_8U, 1);
	cvSetZero( magDrv);

	// Create convolution kernel for x
	float kernel_array_x[] = {-0.5, 0, 0.5 };
	CvMat kernel_x;
	cvInitMatHeader(&kernel_x, 1 ,3, CV_32F, kernel_array_x);

	// Calculate derivative in x
	cvFilter2D(&inputImage, drv32f, &kernel_x);

	// fix values in the first and last colums
	for(int y = 0; y<drv32f->height; y++){
		float *derive_ptr = (float*)(drv32f->imageData + y * drv32f->widthStep);
		derive_ptr[0] *= 2;
		derive_ptr[drv32f->width-1] *= 2;
	}

	// Accumulate square of dx
	cvSquareAcc( drv32f, magDrv);

	// Create convolution kernel for y
	CvMat kernel_y;
	cvInitMatHeader(&kernel_y, 3 ,1, CV_32F, kernel_array_x);

	// Calculate derivative in x
	cvFilter2D(&inputImage, drv32f, &kernel_y);


	// fix values in the first and last lines
	float *derive_ptr_first = (float*)(drv32f->imageData);
	float *derive_ptr_last = (float*)(drv32f->imageData + (drv32f->height-1) * drv32f->widthStep);
	for(int y = 0; y < drv32f->width; y++){
		derive_ptr_first[y] *= 2;
		derive_ptr_last[y] *= 2;
	}

	// Accumulate square of dx
	cvSquareAcc( drv32f, magDrv);

	cv::Mat magTemp(magDrv);
	cv::sqrt( magTemp, magTemp );

	// This is uint8 from MATLAB
	for(int y = 0; y<drv32f->height; y++){
		float *mag_dvr_ptr = (float*)(magDrv->imageData + y * magDrv->widthStep);
		unsigned char *magDrvUint_ptr = (unsigned char*)(magDrvUint->imageData + y * magDrvUint->widthStep);
		for(int x=0; x < drv32f->width; x++){
			if(mag_dvr_ptr[x] < 0.0){
				magDrvUint_ptr[x] = 0;
			}else{
				if(mag_dvr_ptr[x] > 255.0){
					magDrvUint_ptr[x] = 255;
				}else{
					magDrvUint_ptr[x] = (unsigned char)rndint((mag_dvr_ptr[x]));
				}
			}
		}
	}

	cv::Mat magMatTemp(magDrvUint);
	magMatTemp.copyTo(gradientMat);

	cvReleaseImage(&drv32f);
	cvReleaseImage(&magDrv);
	cvReleaseImage(&magDrvUint);
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


