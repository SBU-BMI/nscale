#include "cutil.h"
#include <string.h>
#include <stdio.h>

void coocGPU( unsigned char*, int, int, unsigned int*, int, int);
void NormMatGPU(unsigned char*,unsigned char*, int , int , int);

void initDevice(int argc, char **argv) {
	CUT_DEVICE_INIT();
}

void coocMatrixGPU(char *h_inputImage, int width, int height, unsigned int* coocMatrix, int coocSize, int angle, int device ){
// TODO: remove this function call
	initDevice(0, NULL);

	void *d_inputImage;
	void *d_normImage;
	unsigned int inputImageSize = width * height * sizeof(char);
	unsigned int normImageSize = width * height * sizeof(char);
	// choose the appropriate device
	cudaSetDevice(device);

	// alloc memory to store input image
	cudaMalloc( (void**)&d_inputImage, inputImageSize );

	// copy input matrix to GPU memory
	cudaMemcpy(d_inputImage, h_inputImage , inputImageSize, cudaMemcpyHostToDevice );


	// alloc memory to store normalized version of the input matrix
	cudaMalloc( (void**)&d_normImage, normImageSize );

	// Normalize input image from gray scale [0 255] to [1 8]. 
	// Same as Matlab default number of gray levels
	NormMatGPU((unsigned char *)d_inputImage, (unsigned char *)d_normImage, width, height, coocSize);	

	// compute coocurrence matrix on top of the normalized image
	coocGPU((unsigned char *)d_normImage, width, height, coocMatrix, coocSize, angle);

	cudaFree(d_inputImage);
	cudaFree(d_normImage);
}

