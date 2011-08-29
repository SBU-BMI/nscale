#include "cutil.h"
#include <string.h>
#include <stdio.h>

#define	N_HARALICK_FEATURES	7
#define N_INTENSITY_FEATURES	6

void coocGPU( unsigned char*, int, int, unsigned int*, int, int);
void NormMatGPU(unsigned char*,unsigned char*, int , int , int);
//void coocGPUBlob( unsigned char *, int, int, int *, int, int, char *, int);
void haralickGPUBlob( int *, int, float*);
float *intensityGPUBlob( char *, int, int, int, char*);
int *coocGPUBlob( unsigned char *, int , int , int , int , char *, int );
int *intensityHistGPUBlob( char * d_image, int width, int height, int nBlobs, char* d_mask);
int *intensityHistGPUBlobBAK( char * d_image, int width, int height, int nBlobs, char* d_mask);

void initDevice(int argc, char **argv) {
	CUT_DEVICE_INIT();
}

void coocMatrixGPU(char *h_inputImage, int width, int height, unsigned int* coocMatrix, int coocSize, int angle, bool copyData, int device ){
// TODO: remove this function call
//	initDevice(0, NULL);

	void *d_inputImage;
	void *d_normImage;
	unsigned int inputImageSize = width * height * sizeof(char);
	unsigned int normImageSize = width * height * sizeof(char);
	// choose the appropriate device
//	cudaSetDevice(device);

	if(copyData){
		// alloc memory to store input image
		cudaMalloc( (void**)&d_inputImage, inputImageSize );

		// copy input matrix to GPU memory
		cudaMemcpy(d_inputImage, h_inputImage , inputImageSize, cudaMemcpyHostToDevice );
	}else{
		d_inputImage = h_inputImage;
	}

	// alloc memory to store normalized version of the input matrix
	cudaMalloc( (void**)&d_normImage, normImageSize );

	// Normalize input image from gray scale [0 255] to [1 8]. 
	// Same as Matlab default number of gray levels
	NormMatGPU((unsigned char *)d_inputImage, (unsigned char *)d_normImage, width, height, coocSize);	

	// compute coocurrence matrix on top of the normalized image
	coocGPU((unsigned char *)d_normImage, width, height, coocMatrix, coocSize, angle);

	if(copyData){
		cudaFree(d_inputImage);
	}
	cudaFree(d_normImage);
}


// If downloadRes == true, we alloc coocurrence matrix in the CPU memory and copy it from the GPU memory.
// whether it is false, the pointer to the GPU main memory is retrieved
int *coocMatrixBlobGPU(char *h_inputImage, int width, int height, int nBlobs, char *maskData, int maskSize,  int coocSize, int angle, bool copyData, bool downloadRes, int device){
// TODO: remove this function call
//	initDevice(device, NULL);

	void *d_inputImage;
	void *d_normImage;
	void *d_mask;
	
	unsigned int inputImageSize = width * height * sizeof(char);
	unsigned int normImageSize = width * height * sizeof(char);

	// choose the appropriate device
//	cudaSetDevice(device);

	if(copyData){
		// alloc memory to store input image
		cudaMalloc( (void**)&d_inputImage, inputImageSize );

		// copy input matrix to GPU memory
		cudaMemcpy(d_inputImage, h_inputImage , inputImageSize, cudaMemcpyHostToDevice );

		// alloc memory to store input mask
		CUDA_SAFE_CALL(cudaMalloc( (void**)&d_mask, maskSize*sizeof(char) ));

		// copy blob masks to GPU memory
		CUDA_SAFE_CALL(cudaMemcpy(d_mask, maskData, maskSize*sizeof(char), cudaMemcpyHostToDevice ));

	}else{
		d_inputImage = h_inputImage;
		d_mask = maskData;

	}

	// alloc memory to store normalized version of the input matrix
	cudaMalloc( (void**)&d_normImage, normImageSize );
	

	// Normalize input image from gray scale [0 255] to [1 8]. 
	// Same as Matlab default number of gray levels
	NormMatGPU((unsigned char *)d_inputImage, (unsigned char *)d_normImage, width, height, coocSize);	

	
	int *d_cooc_matrix = coocGPUBlob( (unsigned char *)d_normImage, width, height, coocSize, nBlobs, (char*)d_mask,  angle);

	int *outCoocMatrix = d_cooc_matrix;

	if(downloadRes == true){
		outCoocMatrix = ( int *)malloc(sizeof( int) * coocSize * coocSize * nBlobs);
		cudaMemcpy( (void*)outCoocMatrix, (void*)d_cooc_matrix, nBlobs*coocSize*coocSize*sizeof(int), cudaMemcpyDeviceToHost );
		cudaFree(d_cooc_matrix);
	}

	if(copyData){
		cudaFree(d_inputImage);
		cudaFree(d_mask);
	}
	cudaFree(d_normImage);
	
	return outCoocMatrix;
}

// function assumes that coocurrence matrix is stored at the device memory.
float *calcHaralickGPUBlob(int *d_coocMatrix, int coocSize, int nBlobs, int device){
	float *haralickFeatures = (float *) malloc(sizeof(float) * nBlobs * N_HARALICK_FEATURES);		

	haralickGPUBlob( d_coocMatrix, nBlobs, haralickFeatures);

	return haralickFeatures;
}

float *intensityFeaturesBlobGPU(char *h_inputImage, int width, int height, int nBlobs, char *maskData, int maskSize, bool copyData, int device){
//	initDevice(device, NULL);

	void *d_inputImage;
	void *d_mask;
	
	unsigned int inputImageSize = width * height * sizeof(char);

	// choose the appropriate device
//	cudaSetDevice(device);
	
	if(copyData){
		// alloc memory to store input image
		cudaMalloc( (void**)&d_inputImage, inputImageSize );

		// copy input matrix to GPU memory
		cudaMemcpy(d_inputImage, h_inputImage , inputImageSize, cudaMemcpyHostToDevice );

		// alloc memory to store input mask
		CUDA_SAFE_CALL(cudaMalloc( (void**)&d_mask, maskSize*sizeof(char) ));
	
		// copy blob masks to GPU memory
		CUDA_SAFE_CALL(cudaMemcpy(d_mask, maskData, maskSize*sizeof(char), cudaMemcpyHostToDevice ));

	}else{
		d_inputImage = h_inputImage;
		d_mask = maskData;
	}

	// Actual GPU code that calculate features
	float *d_intensity_res = intensityGPUBlob( (char*)d_inputImage, width, height, nBlobs, (char*)d_mask);

	// allocate space to store outputed features
	float *outIntensityFeaturesMatrix = ( float *)malloc(sizeof( float) * N_INTENSITY_FEATURES *nBlobs);
	
	cudaMemcpy( (void*)outIntensityFeaturesMatrix, (void*)d_intensity_res, nBlobs*sizeof(float)*N_INTENSITY_FEATURES, cudaMemcpyDeviceToHost );

	cudaFree(d_intensity_res);

	if(copyData){
		cudaFree(d_inputImage);

		cudaFree(d_mask);
	}

	
	return outIntensityFeaturesMatrix;

}




//// If downloadRes == true, we alloc coocurrence matrix in the CPU memory and copy it from the GPU memory.
//// whether it is false, the pointer to the GPU main memory is retrieved
//float *coocMatrixBlobGPUANDFEATURES(char *h_inputImage, int width, int height, int nBlobs, char *maskData, int maskSize,  int coocSize, int angle, bool downloadRes, int device){
//// TODO: remove this function call
//	initDevice(device, NULL);
//
//	void *d_inputImage;
//	void *d_normImage;
//	void *d_mask;
//	
//	unsigned int inputImageSize = width * height * sizeof(char);
//	unsigned int normImageSize = width * height * sizeof(char);
//
////#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
////	printf("Using shared memory atomic operations!\n");
////#else
////	printf("Using global memory atomic operations!\n");
////#endif
//	
//	// choose the appropriate device
//	cudaSetDevice(device);
//
//	// alloc memory to store input image
//	cudaMalloc( (void**)&d_inputImage, inputImageSize );
//
//	// copy input matrix to GPU memory
//	cudaMemcpy(d_inputImage, h_inputImage , inputImageSize, cudaMemcpyHostToDevice );
//
//
//	// alloc memory to store normalized version of the input matrix
//	cudaMalloc( (void**)&d_normImage, normImageSize );
//	
//	// alloc memory to store input mask
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_mask, maskSize*sizeof(char) ));
//	
//	// copy blob masks to GPU memory
//	CUDA_SAFE_CALL(cudaMemcpy(d_mask, maskData, maskSize*sizeof(char), cudaMemcpyHostToDevice ));
//
//	// Normalize input image from gray scale [0 255] to [1 8]. 
//	// Same as Matlab default number of gray levels
//	NormMatGPU((unsigned char *)d_inputImage, (unsigned char *)d_normImage, width, height, coocSize);	
//
//	
//	float *d_features = coocGPUBlobANDFEATURES( (unsigned char *)d_normImage, width, height, coocSize, nBlobs, (char*)d_mask,  angle);
//
//
//	float *haralickFeatures = (float *) malloc(sizeof(float) * nBlobs * N_FEATURES);		
//	
//
//	cudaMemcpy( (void*)haralickFeatures, (void*)d_features, nBlobs*sizeof(float) * 7, cudaMemcpyDeviceToHost );
//	
//
//
//	cudaFree(d_inputImage);
//	cudaFree(d_normImage);
//	cudaFree(d_mask);
//	
//	return haralickFeatures;
//}
