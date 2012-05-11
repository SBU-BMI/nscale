#include "hist-ops.cuh"
#include <limits>


#define HIST_BINS			256
#define THREAD_N_BLOCK_INTENSITY	32
#define N_INTENSITY_FEATURES		8
#define N_GRADIENT_FEATURES		6
#define N_CANNY_FEATURES		2	

namespace nscale {

namespace gpu {

using namespace cv::gpu;



void *cudaMallocCaller(int size){
	void *aux_ptr;
	cudaMalloc((void**)&aux_ptr, size);
	return aux_ptr;
}

void cudaFreeCaller(void *data_ptr){
	if(data_ptr != NULL){
		cudaFree(data_ptr);
	}
}

void cudaUploadCaller(void *dest, void *source, int size){
	cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice);
}
void cudaDownloadCaller(void *dest, void *source, int size){
	cudaMemcpy(dest, source, size, cudaMemcpyDeviceToHost);
}

__global__ void histCalcKernel(const PtrStep_<int> labeledImg, PtrStep_<unsigned char> grayImage, int *bbInfo, int numComponents, int* d_components_hist)
{
	int blobId = blockIdx.x;
	int minx = bbInfo[blobId + numComponents];
	int maxx = bbInfo[blobId + 2*numComponents];
	int miny = bbInfo[blobId + 3*numComponents];
	int maxy = bbInfo[blobId + 4*numComponents]; 
	int label = bbInfo[blobId];

	int hist_offset = blobId * HIST_BINS;

	for(int pos=threadIdx.x; pos < (HIST_BINS); pos+=blockDim.x){
		d_components_hist[hist_offset+pos] = 0;
	}
	__syncthreads();

	// Each thread takes care of one line of the blob
	for(int i = threadIdx.x; i < (maxy-miny+1); i+=blockDim.x){
		for(int j = minx; j <= maxx; j++){
			if( labeledImg.ptr(i+miny)[j] == label){
				int data = grayImage.ptr(i+miny)[j];
				atomicAdd(&d_components_hist[hist_offset + data],1);
			}
		}
	}
}




// build a histogram for each component in the image described using bbInfo
int* calcHist256Caller(const cv::gpu::PtrStep_<int> labeledImg, cv::gpu::PtrStep_<unsigned char> grayImage, int *bbInfo, int numComponents, cudaStream_t stream)
{
	int *d_components_hist;
	dim3 grid(numComponents);
	dim3 threads(THREAD_N_BLOCK_INTENSITY);

	cudaMalloc( (void**)&d_components_hist, HIST_BINS * sizeof( int ) * numComponents );

	histCalcKernel<<<grid, threads, 0, stream>>>(labeledImg, grayImage, bbInfo, numComponents, d_components_hist);

 	cudaGetLastError() ;
	if (stream == 0)
        	cudaDeviceSynchronize();

	return d_components_hist;
}

__global__ void histCalcCytoKernel(PtrStep_<unsigned char> grayImage, int *bbInfo, int numComponents, int* d_components_hist)
{
	int blobId = blockIdx.x;
	int dataOffset = bbInfo[blobId*5];
	int minx = bbInfo[blobId*5 + 1];
	int miny = bbInfo[blobId*5 + 2];
	int width = bbInfo[blobId*5 + 3];
	int height = bbInfo[blobId*5 + 4];
	int label = 255;
	char *dataAddress = ((char*)(bbInfo))+dataOffset;

	int hist_offset = blobId * HIST_BINS;

	for(int pos=threadIdx.x; pos < (HIST_BINS); pos+=blockDim.x){
		d_components_hist[hist_offset+pos] = 0;
	}
	__syncthreads();

	// Each thread takes care of one line of the blob
	for(int i = threadIdx.x; i < height; i+=blockDim.x){
		for(int j = 0; j < width; j++){
			if( dataAddress[i*width+j] == label){
				int data = grayImage.ptr(i+miny)[j+minx];
				atomicAdd(&d_components_hist[hist_offset + data],1);
			}
		}
	}
}




// build a histogram for each component in the image described using bbInfo
int* calcHist256CytoBBCaller(cv::gpu::PtrStep_<unsigned char> grayImage, int *bbInfo, int numComponents, cudaStream_t stream)
{
	int *d_components_hist;
	dim3 grid(numComponents);
	dim3 threads(THREAD_N_BLOCK_INTENSITY);

	cudaMalloc( (void**)&d_components_hist, HIST_BINS * sizeof( int ) * numComponents );

	histCalcCytoKernel<<<grid, threads, 0, stream>>>(grayImage, bbInfo, numComponents, d_components_hist);

 	cudaGetLastError() ;
	if (stream == 0)
        	cudaDeviceSynchronize();

	return d_components_hist;
}


__global__ void calcFeaturesHistKernel(int *hist, int numHists, float *output){
	// calcuate the id of the blob of which this thread calcualtes the features
	int histId=blockIdx.x * THREAD_N_BLOCK_INTENSITY + threadIdx.x;
	float minValue = 0.0;
	float maxValue = 0.0;
	float sum = 0.0;
	float mean = 0.0;
	int nPixels = 0;

	if(histId >= numHists){
		histId = numHists-1;
	}
	const int memoryBaseIndex=histId * HIST_BINS;
	int *hist_ptr = &hist[memoryBaseIndex];


	for(int i = 0; i < HIST_BINS; i++){
		sum += hist_ptr[i] * i;
		nPixels += hist_ptr[i];
		
		if(hist_ptr[i] != 0){
			maxValue = i;
		}
		if(minValue == 0 && hist_ptr[i] != 0){
			minValue = i;
		}

	}

	mean  = sum/(float)nPixels;
	
	// Calculate STD/Entropy/Energy
	float sq_diff_sum = 0.0;
	float entropy = 0.0;
	float energy = 0.0;
	for(int i = 0; i < HIST_BINS; i++){
		//STD
		float diff = i - mean;
		sq_diff_sum += diff * diff * hist_ptr[i];
		
		//ENTROPY
		float p_i = (float)hist_ptr[i]/nPixels;
		// ignore 0 entries
		if(hist_ptr[i] != 0){
			entropy += ((p_i) * log2(p_i));
		}
		
		energy += pow(p_i, 2);
	}
	float variance = sq_diff_sum/nPixels;
	entropy *= -1.0;
	

	// SKEWNESS
	float skewness = 0.0;
	float dividend = 0.0;
	float divisor = 0.0;
	for(int i = 0; i < HIST_BINS; i++){
		float e_i = (float)hist_ptr[i];
		dividend += e_i * pow(i - mean, 3);
		divisor += e_i * pow(i - mean, 2);
	}

	if(dividend != 0 && divisor != 0 && nPixels != 0){
		dividend /= nPixels;
		divisor /= nPixels;
		divisor = sqrt(divisor);
		divisor = pow( divisor, 3);
		skewness = dividend / divisor;
	}


	float dividend_kurtosis = 0.0;
	float divisor_kurtosis = 0.0;
	float kurtosis = 0.0;
	for(int i = 0; i < HIST_BINS; i++){
		float e_i = (float)hist_ptr[i];
		dividend_kurtosis += e_i * pow(i - mean, 4);
		divisor_kurtosis += e_i * pow(i - mean, 2);

	}

	if(dividend_kurtosis != 0 && divisor_kurtosis != 0 && nPixels != 0){
		dividend_kurtosis /= nPixels;
		divisor_kurtosis /= nPixels;

		divisor_kurtosis = pow( divisor_kurtosis, 2);

		kurtosis = dividend_kurtosis / divisor_kurtosis;
	}

	output[histId  * N_INTENSITY_FEATURES] = mean; // mean
	output[histId  * N_INTENSITY_FEATURES + 1] = maxValue;
	output[histId  * N_INTENSITY_FEATURES + 2] = minValue;
	output[histId  * N_INTENSITY_FEATURES + 3] = sqrt(variance);
	output[histId  * N_INTENSITY_FEATURES + 4] = entropy;
	output[histId  * N_INTENSITY_FEATURES + 5] = energy;
	output[histId  * N_INTENSITY_FEATURES + 6] = skewness;
	output[histId  * N_INTENSITY_FEATURES + 7] = kurtosis;
	
}

void calcFeaturesFromHist256Caller(int *hist, int numHists, float *output, cudaStream_t stream){
	int n_blocks = (numHists%THREAD_N_BLOCK_INTENSITY != 0) ? (numHists/THREAD_N_BLOCK_INTENSITY +1) : (numHists)/THREAD_N_BLOCK_INTENSITY;
	
	dim3 grid(n_blocks);
	dim3 threads(THREAD_N_BLOCK_INTENSITY);
	
	calcFeaturesHistKernel<<<grid, threads, 0, stream>>>(hist, numHists, output);

}


__global__ void calcGradFeaturesHistKernel(int *hist, int numHists, float *output){
	// calcuate the id of the blob of which this thread calcualtes the features
	int histId=blockIdx.x * THREAD_N_BLOCK_INTENSITY + threadIdx.x;
	float sum = 0.0;
	float mean = 0.0;
	int nPixels = 0;

	if(histId >= numHists){
		histId = numHists-1;
	}
	const int memoryBaseIndex=histId * HIST_BINS;
	int *hist_ptr = &hist[memoryBaseIndex];


	for(int i = 0; i < HIST_BINS; i++){
		sum += hist_ptr[i] * i;
		nPixels += hist_ptr[i];
	}

	mean  = sum/(float)nPixels;
	
	// Calculate STD/Entropy/Energy
	float sq_diff_sum = 0.0;
	float entropy = 0.0;
	float energy = 0.0;
	for(int i = 0; i < HIST_BINS; i++){
		//STD
		float diff = i - mean;
		sq_diff_sum += diff * diff * hist_ptr[i];
		
		//ENTROPY
		float p_i = (float)hist_ptr[i]/nPixels;
		// ignore 0 entries
		if(hist_ptr[i] != 0){
			entropy += ((p_i) * log2(p_i));
		}
		
		energy += pow(p_i, 2);
	}
	float variance = sq_diff_sum/nPixels;
	entropy *= -1.0;
	

	// SKEWNESS
	float skewness = 0.0;
	float dividend = 0.0;
	float divisor = 0.0;
	for(int i = 0; i < HIST_BINS; i++){
		float e_i = (float)hist_ptr[i];
		dividend += e_i * pow(i - mean, 3);
		divisor += e_i * pow(i - mean, 2);
	}

	if(dividend != 0 && divisor != 0 && nPixels != 0){
		dividend /= nPixels;
		divisor /= nPixels;
		divisor = sqrt(divisor);
		divisor = pow( divisor, 3);
		skewness = dividend / divisor;
	}


	float dividend_kurtosis = 0.0;
	float divisor_kurtosis = 0.0;
	float kurtosis = 0.0;
	for(int i = 0; i < HIST_BINS; i++){
		float e_i = (float)hist_ptr[i];
		dividend_kurtosis += e_i * pow(i - mean, 4);
		divisor_kurtosis += e_i * pow(i - mean, 2);

	}

	if(dividend_kurtosis != 0 && divisor_kurtosis != 0 && nPixels != 0){
		dividend_kurtosis /= nPixels;
		divisor_kurtosis /= nPixels;

		divisor_kurtosis = pow( divisor_kurtosis, 2);

		kurtosis = dividend_kurtosis / divisor_kurtosis;
	}

	output[histId  * N_GRADIENT_FEATURES] = mean; // mean
	output[histId  * N_GRADIENT_FEATURES + 1] = sqrt(variance);
	output[histId  * N_GRADIENT_FEATURES + 2] = entropy;
	output[histId  * N_GRADIENT_FEATURES + 3] = energy;
	output[histId  * N_GRADIENT_FEATURES + 4] = skewness;
	output[histId  * N_GRADIENT_FEATURES + 5] = kurtosis;
	
}

void calcGradFeaturesFromHist256Caller(int *hist, int numHists, float *output, cudaStream_t stream){
	int n_blocks = (numHists%THREAD_N_BLOCK_INTENSITY != 0) ? (numHists/THREAD_N_BLOCK_INTENSITY +1) : (numHists)/THREAD_N_BLOCK_INTENSITY;
	dim3 grid(n_blocks);
	dim3 threads(THREAD_N_BLOCK_INTENSITY);
	
	calcGradFeaturesHistKernel<<<grid,threads, 0, stream>>>(hist, numHists, output);

}



__global__ void calcCannyFeaturesHistKernel(int *hist, int numHists, float *output){
	// calcuate the id of the blob of which this thread calcualtes the features
	int histId=blockIdx.x * THREAD_N_BLOCK_INTENSITY + threadIdx.x;
	float sum = 0.0;
	float mean = 0.0;
	int nPixels = 0;

	if(histId >= numHists){
		histId = numHists-1;
	}
	
	const int memoryBaseIndex=histId * HIST_BINS;
	int *hist_ptr = &hist[memoryBaseIndex];


	for(int i = 0; i < HIST_BINS; i++){
		sum += hist_ptr[i] * i;
		nPixels += hist_ptr[i];
	}

	mean  = sum/(float)nPixels;
	int nonZero = nPixels - hist_ptr[0];
	

	output[histId  * N_CANNY_FEATURES] = nonZero; // area
	output[histId  * N_CANNY_FEATURES + 1] = mean;

	
}



void calcCannyFeaturesFromHist256Caller(int *hist, int numHists, float *output, cudaStream_t stream){
	int n_blocks = (numHists%THREAD_N_BLOCK_INTENSITY != 0) ? (numHists/THREAD_N_BLOCK_INTENSITY +1) : (numHists)/THREAD_N_BLOCK_INTENSITY;
	dim3 grid(n_blocks);
	dim3 threads(THREAD_N_BLOCK_INTENSITY);
	
	calcCannyFeaturesHistKernel<<<grid,threads, 0, stream>>>(hist, numHists, output);

}



}}




//}}
