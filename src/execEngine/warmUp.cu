#include "cutil.h"
#include <string.h>
#include <stdio.h>

void warmUp(int device){
	void *d_data;
	int *h_data = (int *) malloc(sizeof(int));
	h_data[0] = 10;

	// choose the appropriate device
	cudaSetDevice(device);

	cudaMalloc( (void**)&d_data, sizeof(int) );

	cudaMemcpy(d_data, h_data, sizeof(int), cudaMemcpyHostToDevice );

	cudaFree(d_data);
	free(h_data);
}
