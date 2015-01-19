/*
 * queueTest.cpp
 *
 *  Created on: Nov 30, 2011
 *      Author: tcpan
 */
#include "stdio.h"

#ifdef _MSC_VER
#include "time_win.h"
#else
	#include <sys/time.h>
#endif

#if defined (WITH_CUDA)

#include "cuda/queue.cuh"





long ClockGetTime3()
{
	struct timeval ts;
	gettimeofday(&ts, NULL);
 //   timespec ts;
//    clock_gettime(CLOCK_REALTIME, &ts);
	return (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL;
//    return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
}


int main (int argc, char **argv){

	unsigned int size = 4000;
	unsigned int interval = 4000;
	int *data = (int*)malloc(size * sizeof(int));
	int threshold = 127;
	srand(0);

	unsigned int count = 0;
	int val;
	for (unsigned int s = 0; s < size; ++s) {
		val = rand() % (2*threshold + 1);
		if (val > threshold) {
			data[s] = 1;
			++count;
		} else {
			data[s] = 0;
		}
	}

	int * output = (int*)malloc(size * sizeof(int));
	unsigned int total = 0;

	cudaSetDevice(0);

	cudaStream_t stream;
	cudaError_t error;
	error = cudaStreamCreate(&stream);


	// now do the tests
	// CPU
	long t1, t2;
	t1 = ClockGetTime3();
	total = nscale::gpu::SelectCPUTesting(data, size, output );
	t2 = ClockGetTime3();
	printf("cpu: %d total, %lu ms\n", total, t2-t1);

	// thrust
	for (unsigned int s = 0; s < size; s++) {
		if ((s % (size / interval)) == 0) {
			printf("%d, ", data[s]);
		}
	}
	printf("\n");
	t1 = ClockGetTime3();
	total = nscale::gpu::SelectThrustScanTesting(data, size, output, stream);
	error = cudaStreamSynchronize(stream);
	t2 = ClockGetTime3();
	printf("thrust scan: %d total, %lu ms\n", total, t2-t1);
	for (unsigned int s = 0; s < size; s++) {
		if ((s % (size / interval)) == 0) {
			printf("%d, ", output[s]);
		}
	}
	printf("\n");
	cudaDeviceSynchronize();

	// warp scan unordered
//	for (unsigned int s = 0; s < size; s++) {
//		if ((s % 10000) == 0) {
//			printf("%d, ", data[s]);
//		}
//	}
	t1 = ClockGetTime3();
	total = nscale::gpu::SelectWarpScanUnorderedTesting(data, size, output, stream);
//	error = cudaStreamSynchronize(stream);
//	cudaDeviceSynchronize();
	t2 = ClockGetTime3();
	printf("warp scan unordered: %d total, %lu ms\n", total, t2-t1);
	for (unsigned int s = 0; s < size; s++) {
		if ((s % (size / interval)) == 0) {
			printf("%d, ", output[s]);
		}
	}
	printf("\n");
//	cudaDeviceSynchronize();
	int count2;
	// warp scan ordered
	t1 = ClockGetTime3();
	total = nscale::gpu::SelectWarpScanOrderedTesting(data, size, output, stream);
//	cudaDeviceSynchronize();
	t2 = ClockGetTime3();
	printf("warp scan ordered: %d total, %lu ms\n", total, t2-t1);
	for (unsigned int s = 0; s < size; s++) {
		if ((s % (size / interval)) == 0) {
			printf("%d, ", output[s]);
		}
	}
	printf("\n");

	error = cudaStreamDestroy(stream);

	free(data);
	free(output);

	return 0;
}

#else
int main (int argc, char **argv){
	printf("NEED CUDA!\n");
	return -1;
}

#endif
