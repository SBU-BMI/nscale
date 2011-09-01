
#include "cutil.h"

#define COOC_SIZE 8
#define BIN_COUNT (COOC_SIZE*COOC_SIZE)
#define BLOCK_DIM 16
#define THREAD_N 128
#define BLOCK_MEMORY (THREAD_N*BIN_COUNT)
#define BLOCK_DATA (THREAD_N*255)
#define round(num) (num+0.5f)
#define THREAD_N_BLOB	32
#define THREAD_N_BLOB_HARALICK	64
#define N_HARALICK_FEATURES 	7
#define N_INTENSITY_FEATURES	6
#define PIXEL_INTENSITY_BINS	256

#define IMUL(a,b)  __mul24(a,b) 

__global__ void reduceCooc64GPU(unsigned int *d_Result, int blockN){
	const int globalTid= IMUL(blockIdx.x, blockDim.x) + threadIdx.x;

	const int numThreads= IMUL(gridDim.x, blockDim.x);
	const int dataSize= IMUL(blockN, BIN_COUNT);

	int sum=0;
	for (int pos= globalTid; pos<dataSize;pos+=numThreads)
		sum+= d_Result[pos];

	d_Result[globalTid]=sum;
}

__device__ void addPairPixels(unsigned char *s_coocmat, int threadPos, unsigned int data1, unsigned int data2, unsigned int cooc_size){
	unsigned int data= data2 + IMUL(cooc_size, data1);
	s_coocmat[threadPos + IMUL(THREAD_N, data)]++;
}

__global__ void coocurrenceGPU(unsigned char* image, int width, int height, unsigned int * matrix, int cooc_size ){

	int dataN = width * height;
	//Global base index in input data for current block
	const int baseIndex= IMUL(BLOCK_DATA, blockIdx.x);
	
	//Current block size, clamp by array border
	const int dataSize= min(dataN - baseIndex, BLOCK_DATA);

	const int threadPos = threadIdx.x;
	
	//Per-thread coocurrence matrix 
	__shared__ unsigned char s_coocmat[BLOCK_MEMORY];
	
	for (int pos = threadIdx.x; pos < (BLOCK_MEMORY / 4); pos+= blockDim.x)
		((unsigned int *)s_coocmat)[pos]=0;

	__syncthreads();

	for (int pos= threadIdx.x;pos<dataSize;pos+= blockDim.x){
           if ((baseIndex+pos) < dataN-1 && ((baseIndex+pos+1)%width)!= 0 ){
				unsigned char data1= image[baseIndex+pos];
				unsigned char data2= image[baseIndex+pos+1];
				addPairPixels(s_coocmat, threadPos, (data1) -1,(data2) -1, cooc_size);
           }
	}

	__syncthreads();

	if (threadIdx.x < BIN_COUNT){
		unsigned int sum=0;
		const int value = threadIdx.x;
		
		const int valueBase = IMUL(value, THREAD_N);
		const int startPos= IMUL(threadIdx.x & 15, 4);
		
		for (int i=0, accumPos = startPos; i< THREAD_N; i++){
			sum+= s_coocmat[valueBase + accumPos];
			if (++accumPos == THREAD_N) accumPos=0;
		}

		matrix[IMUL(BIN_COUNT, blockIdx.x)+ value]= sum;
	}
}

__global__ void coocurrenceGPU45(unsigned char* image, int width, int height, unsigned int * matrix, int cooc_size ){

	int dataN = width * height;
	//Global base index in input data for current block
	const int baseIndex= IMUL(BLOCK_DATA, blockIdx.x);
	
	//Current block size, clamp by array border
	const int dataSize= min(dataN - baseIndex, BLOCK_DATA);

	const int threadPos = threadIdx.x;
	
	//Per-thread coocurrence matrix 
	__shared__ unsigned char s_coocmat[BLOCK_MEMORY];
	
	for (int pos = threadIdx.x; pos < (BLOCK_MEMORY / 4); pos+= blockDim.x)
		((unsigned int *)s_coocmat)[pos]=0;

	__syncthreads();

	for (int pos= threadIdx.x;pos<dataSize;pos+= blockDim.x){
           if ((baseIndex+pos) >= width &&  (baseIndex+pos) < dataN-1 && ((baseIndex+pos+1)%width)!= 0 ){
     			unsigned char data1= image[baseIndex+pos];
				unsigned char data2= image[baseIndex+pos+1-width];
				addPairPixels(s_coocmat, threadPos, (data1) -1,(data2) -1, cooc_size);
           }
	}

	__syncthreads();

	if (threadIdx.x < BIN_COUNT){
		unsigned int sum=0;
		const int value = threadIdx.x;
		
		const int valueBase = IMUL(value, THREAD_N);
		const int startPos= IMUL(threadIdx.x & 15, 4);
		
		for (int i=0, accumPos = startPos; i< THREAD_N; i++){
			sum+= s_coocmat[valueBase + accumPos];
			if (++accumPos == THREAD_N) accumPos=0;
		}

		matrix[IMUL(BIN_COUNT, blockIdx.x)+ value]= sum;
	}
}

__global__ void coocurrenceGPU90(unsigned char* image, int width, int height, unsigned int * matrix, int cooc_size ){

	int dataN = width * height;
	//Global base index in input data for current block
	const int baseIndex= IMUL(BLOCK_DATA, blockIdx.x);
	
	//Current block size, clamp by array border
	const int dataSize= min(dataN - baseIndex, BLOCK_DATA);

	const int threadPos = threadIdx.x;
	
	//Per-thread coocurrence matrix 
	__shared__ unsigned char s_coocmat[BLOCK_MEMORY];
	
	for (int pos = threadIdx.x; pos < (BLOCK_MEMORY / 4); pos+= blockDim.x)
		((unsigned int *)s_coocmat)[pos]=0;

	__syncthreads();

	for (int pos= threadIdx.x;pos<dataSize;pos+= blockDim.x){
           if ((baseIndex+pos) >= width &&  (baseIndex+pos) < dataN ){
     			unsigned char data1= image[baseIndex+pos];
				unsigned char data2= image[baseIndex+pos-width];
				addPairPixels(s_coocmat, threadPos, (data1) -1,(data2) -1, cooc_size);
           }
	}

	__syncthreads();

	if (threadIdx.x < BIN_COUNT){
		unsigned int sum=0;
		const int value = threadIdx.x;
		
		const int valueBase = IMUL(value, THREAD_N);
		const int startPos= IMUL(threadIdx.x & 15, 4);
		
		for (int i=0, accumPos = startPos; i< THREAD_N; i++){
			sum+= s_coocmat[valueBase + accumPos];
			if (++accumPos == THREAD_N) accumPos=0;
		}

		matrix[IMUL(BIN_COUNT, blockIdx.x)+ value]= sum;
	}
}


__global__ void coocurrenceGPU135(unsigned char* image, int width, int height, unsigned int * matrix, int cooc_size ){

	int dataN = width * height;
	//Global base index in input data for current block
	const int baseIndex= IMUL(BLOCK_DATA, blockIdx.x);
	
	//Current block size, clamp by array border
	const int dataSize= min(dataN - baseIndex, BLOCK_DATA);

	const int threadPos = threadIdx.x;
	
	//Per-thread coocurrence matrix 
	__shared__ unsigned char s_coocmat[BLOCK_MEMORY];
	
	for (int pos = threadIdx.x; pos < (BLOCK_MEMORY / 4); pos+= blockDim.x)
		((unsigned int *)s_coocmat)[pos]=0;

	__syncthreads();

	for (int pos= threadIdx.x;pos<dataSize;pos+= blockDim.x){
           if ((baseIndex+pos) > width &&  (baseIndex+pos) < dataN && ((baseIndex+pos)%width)!= 0  ){
     			unsigned char data1= image[baseIndex+pos];
				unsigned char data2= image[baseIndex+pos-width-1];
				addPairPixels(s_coocmat, threadPos, (data1) -1,(data2) -1, cooc_size);
           }
	}

	__syncthreads();

	if (threadIdx.x < BIN_COUNT){
		unsigned int sum=0;
		const int value = threadIdx.x;
		
		const int valueBase = IMUL(value, THREAD_N);
		const int startPos= IMUL(threadIdx.x & 15, 4);
		
		for (int i=0, accumPos = startPos; i< THREAD_N; i++){
			sum+= s_coocmat[valueBase + accumPos];
			if (++accumPos == THREAD_N) accumPos=0;
		}

		matrix[IMUL(BIN_COUNT, blockIdx.x)+ value]= sum;
	}
}


void coocGPU( unsigned char * inImage, int width, int height, unsigned int *cooc, int cooc_size, int angle){
	const int BLOCK_N=((width*height)%(THREAD_N*255)!=0) ? ((width*height)/(THREAD_N*255) +1) : (width*height)/(THREAD_N*255);

	unsigned int *h_result= (unsigned int *) malloc(BLOCK_N*cooc_size*cooc_size*sizeof(unsigned int));

	unsigned int * d_Result;

	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, cooc_size*cooc_size*BLOCK_N*sizeof( unsigned int ) ));

	switch(angle){
		case 0:
			coocurrenceGPU<<<BLOCK_N,THREAD_N>>>(inImage, width, height, d_Result, cooc_size );
			break;
		case 1:
			coocurrenceGPU45<<<BLOCK_N,THREAD_N>>>(inImage, width, height, d_Result, cooc_size );
			break;
		case 2:
			coocurrenceGPU90<<<BLOCK_N,THREAD_N>>>(inImage, width, height, d_Result, cooc_size );
			break;
			
		case 3:
			coocurrenceGPU135<<<BLOCK_N,THREAD_N>>>(inImage, width, height, d_Result, cooc_size );		
			break;
			
		default:
			coocurrenceGPU<<<BLOCK_N,THREAD_N>>>(inImage, width, height, d_Result, cooc_size );
	}
	
	cudaThreadSynchronize();
	// reduce histograms from each block thread into a single block
	reduceCooc64GPU<<<min(BLOCK_N,32), BIN_COUNT>>>(d_Result, BLOCK_N);
	cudaThreadSynchronize();

	// 3. Accumulation: 

	cudaMemcpy( (void*)h_result, (void*)d_Result, min(BLOCK_N,32)*cooc_size*cooc_size*sizeof(unsigned int), cudaMemcpyDeviceToHost );

	for (int i=0;i<BIN_COUNT;i++)
		for (int blk=1;blk<min(BLOCK_N,32);blk++)
			h_result[i]+= h_result[i + blk*BIN_COUNT];

	for (int i=0;i<BIN_COUNT;i++)
		cooc[i]=h_result[i];


	// 4. Free memory
	cudaFree(d_Result);
	free(h_result);

}


// Input image linear normalization 
__global__ void normalization( unsigned char * image, unsigned char * imageNorm, int width, int height, int cooc_size ){

	unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

	if (xIndex >= width)
		xIndex=width-1;

	if (yIndex >= height)
		yIndex=height-1;

	unsigned int index_in = xIndex + width * yIndex;
	unsigned int index_block = threadIdx.y * BLOCK_DIM + threadIdx.x;

	__shared__ unsigned char s_data[BLOCK_DIM*BLOCK_DIM];

	s_data[index_block] = image[index_in];

	__syncthreads();

	float slope = ((float)cooc_size-1.0f) / 255.0f;
	float intercept = 1.0f;
	unsigned char element;

	element=s_data[index_block];
	element = (int)round( ( slope * (float)element + intercept ) );

	s_data[index_block]=element;
	__syncthreads();

	imageNorm[index_in] = s_data[index_block];

}



void NormMatGPU(unsigned char *inImage, unsigned char *outImage, int width, int height, int cooc_size){
	dim3 dimBlockNorm( 16, 16 );
	dim3 dimGridNorm( (int)width/16+1, (int)height/16+1 );

	normalization<<<dimGridNorm, dimBlockNorm>>>( inImage, outImage, width, height, cooc_size );
}





__global__ void coocurrenceGPUPerBlob(unsigned char* image, int width, int height, int *matrix, int cooc_size,  char* mask ){
	const int blobId=blockIdx.x;

// If does not support atomic operations using shared memory, 
// go with the atomic at the global memory which is much slower.
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	// Per-blob/block coocurrence matrix 
	__shared__ int s_coocmat[COOC_SIZE*COOC_SIZE];
#else
	int *s_coocmat = &matrix[blobId * (COOC_SIZE * COOC_SIZE)];
#endif

	

	char *blobData_ptr = mask + sizeof(int) * 5 * blobId;

	int offset = ((int *)blobData_ptr)[0];
	int inImageX = ((int *)blobData_ptr)[1];
	int inImageY = ((int *)blobData_ptr)[2];
	int maskWidth = ((int *)blobData_ptr)[3];
	int maskHeight = ((int *)blobData_ptr)[4];
	

	// cleanup cooc matrix
	for(int pos = threadIdx.x; pos < (COOC_SIZE * COOC_SIZE); pos+= blockDim.x)
		s_coocmat[pos]=0;

	__syncthreads();


	// Each thread processes a line from the input image
	for (int i=threadIdx.x; i < maskHeight; i+=blockDim.x){
		// create a pointer to the begining of line "i" in the input mask
		char *blobDataLine = mask + offset + i * maskWidth;
		char *normImageDataLine = (char *)image + (i+inImageY) * width + inImageX;

		for(int j = 0; j < maskWidth-1; j++){
			// is the mask one for both pixels?
			if(((int)blobDataLine[j]) != 0  && ((int)blobDataLine[j+1]) != 0 ){
				int data=(cooc_size * (normImageDataLine[j]-1)); 
				data +=  normImageDataLine[j+1] -1;
				atomicAdd(&s_coocmat[data], 1);
			}

		}
	}

	// this phase of writing results back to global memory is only need for version 
	// that uses atomic shared memory operations
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	__syncthreads();
	for (int i= threadIdx.x; i < (COOC_SIZE * COOC_SIZE); i+=blockDim.x){
		matrix[blobId * (COOC_SIZE * COOC_SIZE) + i] = s_coocmat[i];
	}
#endif
	__syncthreads();	
}


__global__ void coocurrenceGPUPerBlob45(unsigned char* image, int width, int height, int *matrix, int cooc_size,  char* mask ){
	const int blobId=blockIdx.x;

// If does not support atomic operations using shared memory, 
// go with the atomic at the global memory which is much slower.
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	// Per-blob/block coocurrence matrix 
	__shared__ int s_coocmat[COOC_SIZE*COOC_SIZE];
#else
	int *s_coocmat = &matrix[blobId * (COOC_SIZE * COOC_SIZE)];
#endif

	

	char *blobData_ptr = mask + sizeof(int) * 5 * blobId;

	int offset = ((int *)blobData_ptr)[0];
	int inImageX = ((int *)blobData_ptr)[1];
	int inImageY = ((int *)blobData_ptr)[2];
	int maskWidth = ((int *)blobData_ptr)[3];
	int maskHeight = ((int *)blobData_ptr)[4];
	

	// cleanup cooc matrix
	for(int pos = threadIdx.x; pos < (COOC_SIZE * COOC_SIZE); pos+= blockDim.x)
		s_coocmat[pos]=0;

	__syncthreads();


	// Each thread processes a line from the input image
	for (int i=threadIdx.x+1; i < maskHeight; i+=blockDim.x){
		// create a pointer to the begining of line "i" in the input mask
		
		char *blobDataLine1 = mask + offset + (i-1) * maskWidth;
		char *blobDataLine2 = mask + offset + i * maskWidth;

		char *normImageDataLine1 = (char *)image + (i-1+inImageY) * width + inImageX;
		char *normImageDataLine2 = (char *)image + (i+inImageY) * width + inImageX;

		for(int j = 0; j < maskWidth-1; j++){
			// is the mask one for both pixels?
			if(((int)blobDataLine2[j]) != 0  && ((int)blobDataLine1[j+1]) != 0 ){
				int data=(cooc_size * (normImageDataLine2[j]-1)); 
				data +=  normImageDataLine1[j+1] -1;
				atomicAdd(&s_coocmat[data], 1);
			}

		}
	}

	// this phase of writing results back to global memory is only need for version 
	// that uses atomic shared memory operations
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	__syncthreads();
	for (int i= threadIdx.x; i < (COOC_SIZE * COOC_SIZE); i+=blockDim.x){
		matrix[blobId * (COOC_SIZE * COOC_SIZE) + i] = s_coocmat[i];
	}
#endif
	__syncthreads();	
}

__global__ void coocurrenceGPUPerBlob90(unsigned char* image, int width, int height, int *matrix, int cooc_size,  char* mask ){
	const int blobId=blockIdx.x;

// If does not support atomic operations using shared memory, 
// go with the atomic at the global memory which is much slower.
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	// Per-blob/block coocurrence matrix 
	__shared__ int s_coocmat[COOC_SIZE*COOC_SIZE];
#else
	int *s_coocmat = &matrix[blobId * (COOC_SIZE * COOC_SIZE)];
#endif

	

	char *blobData_ptr = mask + sizeof(int) * 5 * blobId;

	int offset = ((int *)blobData_ptr)[0];
	int inImageX = ((int *)blobData_ptr)[1];
	int inImageY = ((int *)blobData_ptr)[2];
	int maskWidth = ((int *)blobData_ptr)[3];
	int maskHeight = ((int *)blobData_ptr)[4];
	

	// cleanup cooc matrix
	for(int pos = threadIdx.x; pos < (COOC_SIZE * COOC_SIZE); pos+= blockDim.x)
		s_coocmat[pos]=0;

	__syncthreads();


	// Each thread processes a line from the input image
	for (int i=threadIdx.x+1; i < maskHeight; i+=blockDim.x){
		// create a pointer to the begining of line "i" in the input mask
		char *blobDataLine1 = mask + offset + (i-1) * maskWidth;
		char *blobDataLine2 = mask + offset + i * maskWidth;

		char *normImageDataLine1 = (char *)image + (i-1+inImageY) * width + inImageX;
		char *normImageDataLine2 = (char *)image + (i+inImageY) * width + inImageX;


		for(int j = 0; j < maskWidth; j++){
			// is the mask one for both pixels?
			if(((int)blobDataLine1[j]) != 0  && ((int)blobDataLine2[j]) != 0 ){
				int data=(cooc_size * (normImageDataLine2[j]-1)); 
				data +=  normImageDataLine1[j] -1;
				atomicAdd(&s_coocmat[data], 1);
			}

		}
	}

	// this phase of writing results back to global memory is only need for version 
	// that uses atomic shared memory operations
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	__syncthreads();
	for (int i= threadIdx.x; i < (COOC_SIZE * COOC_SIZE); i+=blockDim.x){
		matrix[blobId * (COOC_SIZE * COOC_SIZE) + i] = s_coocmat[i];
	}
#endif
	__syncthreads();	
}


__global__ void coocurrenceGPUPerBlob135(unsigned char* image, int width, int height, int *matrix, int cooc_size,  char* mask ){
	const int blobId=blockIdx.x;
	int *cooc_ptr;

// If does not support atomic operations using shared memory, 
// go with the atomic at the global memory which is much slower.
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	// Per-blob/block coocurrence matrix 
	__shared__ int s_coocmat[COOC_SIZE*COOC_SIZE];
	cooc_ptr = s_coocmat;
#else
	cooc_ptr = &matrix[blobId * (COOC_SIZE * COOC_SIZE)];
#endif

	

	char *blobData_ptr = mask + sizeof(int) * 5 * blobId;

	int offset = ((int *)blobData_ptr)[0];
	int inImageX = ((int *)blobData_ptr)[1];
	int inImageY = ((int *)blobData_ptr)[2];
	int maskWidth = ((int *)blobData_ptr)[3];
	int maskHeight = ((int *)blobData_ptr)[4];
	

	// cleanup cooc matrix
	for(int pos = threadIdx.x; pos < (COOC_SIZE * COOC_SIZE); pos+= blockDim.x)
		cooc_ptr[pos]=0;

	__syncthreads();


	// Each thread processes a line from the input image
	for (int i=threadIdx.x+1; i < maskHeight; i+=blockDim.x){
		// create a pointer to the begining of line "i" in the input mask
		char *blobDataLine1 = mask + offset + (i-1) * maskWidth;
		char *blobDataLine2 = mask + offset + i * maskWidth;

		char *normImageDataLine1 = (char *)image + (i-1+inImageY) * width + inImageX;
		char *normImageDataLine2 = (char *)image + (i+inImageY) * width + inImageX;

		for(int j = 1; j < maskWidth; j++){
			// is the mask one for both pixels?
			if(((int)blobDataLine2[j]) != 0  && ((int)blobDataLine1[j-1]) != 0 ){
				int data=(cooc_size * (normImageDataLine2[j]-1)); 
				data +=  normImageDataLine1[j-1] -1;
				atomicAdd(&cooc_ptr[data], 1);
			}

		}
	}

	// this phase of writing results back to global memory is only need for version 
	// that uses atomic shared memory operations
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	__syncthreads();
	for (int i= threadIdx.x; i < (COOC_SIZE * COOC_SIZE); i+=blockDim.x){
		matrix[blobId * (COOC_SIZE * COOC_SIZE) + i] = cooc_ptr[i];
	}
#endif
	__syncthreads();	
}



int *coocGPUBlob( unsigned char * inImage, int width, int height, int cooc_size, int nBlobs, char *d_maskData, int angle){

	const int BLOCK_N=nBlobs;
	
	// Allocate one coocurrence matrix per blob at the GPU memory	
	int *d_Result;

	// Allocate space at GPU memory to store cooc. matrices
	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, cooc_size*cooc_size* BLOCK_N* sizeof( int ) ));

	switch(angle){
		case 0:
			coocurrenceGPUPerBlob<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData );
			break;
		case 1:
			coocurrenceGPUPerBlob45<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData );
			break;
		case 2:
			coocurrenceGPUPerBlob90<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData );
			break;
			
		case 3:
			coocurrenceGPUPerBlob135<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData );
			break;
			
		default:
			coocurrenceGPUPerBlob<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData );
	}
	// Execut cooc. calculation
	
	cudaThreadSynchronize();	
	
	return d_Result;
}



__global__ void haralickFeaturesPerBlob_kernel(int *coocMatrix, int nBlobs, float *haralickFeatures ){
	// calcuate the id of the blob of which this thread calcualtes the features
	int blobId=blockIdx.x * THREAD_N_BLOB_HARALICK + threadIdx.x;

	if(blobId >= nBlobs) blobId = nBlobs-1;

	const int memoryBaseIndex=blobId * BIN_COUNT;

	int sumCoocMatrix = 0;

	// each thread init. its shared memory cooc. matrix; and calculate the sum of itens in the cooc.
	for(int i = 0; i < BIN_COUNT; i++){
		sumCoocMatrix +=  coocMatrix[memoryBaseIndex + i];
	}

	__syncthreads();	

	// Calculate Mx and My, which are used by the clusterShade and clusterProminence
	// Mx calculation
	float mx = 0.0;
	float my = 0.0;
	for(int i = 0; i < COOC_SIZE; i++){
		for(int j = 0; j < COOC_SIZE; j++){
			float entryIJProbability = (float)coocMatrix[memoryBaseIndex +i * COOC_SIZE + j]/(float)sumCoocMatrix;
			mx += i * entryIJProbability;
			my += j * entryIJProbability;
		}
	}

	// compute inertia
	float inertia=0.0;
	float energy=0.0;
	float entropy=0.0;
	float homogeneity=0.0;
	float maximumProbability=0.0;
	float clusterShade=0.0;
	float clusterProminence=0.0;

	const int k=1; // distance from pixels when calculating cooc. matrix. We're only doing it for distance=1;

#pragma unroll 8
	for(int i = 0; i < COOC_SIZE; i++){
#pragma unroll 8
		for(int j = 0; j < COOC_SIZE; j++){
			float ij = i - j;
			float entryIJProbability = (float)coocMatrix[memoryBaseIndex + i * COOC_SIZE + j]/(float)sumCoocMatrix;

			inertia += powf(ij,2) * entryIJProbability; 

			energy += powf( entryIJProbability, 2) ;

			if(entryIJProbability != 0.0){
				entropy += entryIJProbability * log2f(entryIJProbability) ;
			}
			homogeneity += (1.0/(1.0 + powf(ij, 2)) * entryIJProbability);
			if(entryIJProbability > maximumProbability){
				maximumProbability = entryIJProbability;
			}

			clusterShade += powf( (k-mx + j-my), 3) * entryIJProbability;
			clusterProminence += powf( (k-mx + j-my), 4) * entryIJProbability;
		}
	}

	// calculate base address into shared memory to store features
	const int haralickFeaturesBaseIndex = blobId * N_HARALICK_FEATURES;

	// write results back to GPU main memory
	haralickFeatures[haralickFeaturesBaseIndex] = inertia;
	haralickFeatures[haralickFeaturesBaseIndex+1] = energy;
	haralickFeatures[haralickFeaturesBaseIndex+2] = entropy;
	haralickFeatures[haralickFeaturesBaseIndex+3] = homogeneity;
	haralickFeatures[haralickFeaturesBaseIndex+4] = maximumProbability;
	haralickFeatures[haralickFeaturesBaseIndex+5] = clusterShade;
	haralickFeatures[haralickFeaturesBaseIndex+6] = clusterProminence;

}
//
//__global__ void haralickFeaturesPerBlob_kernel_shared(int *coocMatrix, int nBlobs, float *haralickFeatures ){
//	// calcuate the id of the blob of which this thread calcualtes the features
//	const int blobId=blockIdx.x * THREAD_N_BLOB_HARALICK + threadIdx.x;
//
//	const int memoryBaseIndex=blobId * BIN_COUNT;
//	// calculate the offset in the shared memory cooc. matrix used by this thread
//	const int baseIndex=threadIdx.x * BIN_COUNT;	
//
//	//Per-thread coocurrence matrix
//	__shared__ unsigned int s_coocmat[BIN_COUNT * THREAD_N_BLOB];
//	
//	int sumCoocMatrix = 0;
//
//	// each thread init. its shared memory cooc. matrix; and calculate the sum of itens in the cooc.
//	for(int i = 0; i < BIN_COUNT; i++){
//		s_coocmat[baseIndex + i] = coocMatrix[memoryBaseIndex + i];
//		sumCoocMatrix += s_coocmat[baseIndex + i];
//	}
//
//	__syncthreads();	
//
//	// Calculate Mx and My, which are used by the clusterShade and clusterProminence
//	// Mx calculation
//	float mx = 0.0;
//	float my = 0.0;
//	for(int i = 0; i < COOC_SIZE; i++){
//		for(int j = 0; j < COOC_SIZE; j++){
//			float entryIJProbability = (float)s_coocmat[baseIndex +i * COOC_SIZE + j]/(float)sumCoocMatrix;
//			mx += i * entryIJProbability;
//			my += j * entryIJProbability;
//		}
//	}
//
//	// compute inertia
//	float inertia=0.0;
//	float energy=0.0;
//	float entropy=0.0;
//	float homogeneity=0.0;
//	float maximumProbability=0.0;
//	float clusterShade=0.0;
//	float clusterProminence=0.0;
//
//	const int k=1; // distance from pixels when calculating cooc. matrix. We're only doing it for distance=1;
//
//	for(int i = 0; i < COOC_SIZE; i++){
//#pragma unroll 8
//		for(int j = 0; j < COOC_SIZE; j++){
//			float ij = i - j;
//			float entryIJProbability = (float)s_coocmat[baseIndex + i * COOC_SIZE + j]/(float)sumCoocMatrix;
//
//			inertia += powf(ij,2) * entryIJProbability; 
//
//			energy += powf( entryIJProbability, 2) ;
//
//			if(entryIJProbability != 0.0){
//				entropy += entryIJProbability * log2f(entryIJProbability) ;
//			}
//			homogeneity += (1.0/(1.0 + powf(ij, 2)) * entryIJProbability);
//			if(entryIJProbability > maximumProbability){
//				maximumProbability = entryIJProbability;
//			}
//
//			clusterShade += powf( (k-mx + j-my), 3) * entryIJProbability;
//			clusterProminence += powf( (k-mx + j-my), 4) * entryIJProbability;
//		}
//	}
//
//	// calculate base address into shared memory to store features
//	const int haralickFeaturesBaseIndex = blobId * N_HARALICK_FEATURES;
//
//	// write results back to GPU main memory
//	haralickFeatures[haralickFeaturesBaseIndex] = inertia;
//	haralickFeatures[haralickFeaturesBaseIndex+1] = energy;
//	haralickFeatures[haralickFeaturesBaseIndex+2] = entropy;
//	haralickFeatures[haralickFeaturesBaseIndex+3] = homogeneity;
//	haralickFeatures[haralickFeaturesBaseIndex+4] = maximumProbability;
//	haralickFeatures[haralickFeaturesBaseIndex+5] = clusterShade;
//	haralickFeatures[haralickFeaturesBaseIndex+6] = clusterProminence;
//
//}

void haralickGPUBlob( int  *d_coocMatrix, int nBlobs, float * haralickFeatures){
	// calcute number of blocks such that each thread will calculate haralick features for a single blob
	const int BLOCK_N=(nBlobs%THREAD_N_BLOB_HARALICK != 0) ? (nBlobs/THREAD_N_BLOB_HARALICK +1) : (nBlobs)/THREAD_N_BLOB_HARALICK;

	// alloc space used to store Haralick feature in GPU memory
	float *d_HaralickFeatures;
	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_HaralickFeatures, N_HARALICK_FEATURES * nBlobs * sizeof( float ) ));

	// calcute features
	haralickFeaturesPerBlob_kernel<<<BLOCK_N,THREAD_N_BLOB_HARALICK>>>(d_coocMatrix, nBlobs, d_HaralickFeatures );

	// copy results back to CPU memory
	cudaMemcpy( (void*)haralickFeatures, (void*)d_HaralickFeatures, N_HARALICK_FEATURES * nBlobs * sizeof(float), cudaMemcpyDeviceToHost );	
	
	// free GPU memory used to calculate features
	cudaFree(d_HaralickFeatures);

}

__global__ void intensityHistGPUPerBlob_kernel(unsigned char* image, int width, int height, float *d_result, char* mask, int *blobs_hists){
	const int blobId=blockIdx.x;

	int *pixel_hists_ptr;


// If does not support atomic operations using shared memory, 
// go with the atomic at the global memory which is much slower.
#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	// Per-blob/block coocurrence matrix 
	__shared__ int s_coocmat[PIXEL_INTENSITY_BINS];
	pixel_hists_ptr = s_coocmat;
#else
	pixel_hists_ptr = &blobs_hists[blobId * (PIXEL_INTENSITY_BINS)];
#endif

	char *blobData_ptr = mask + sizeof(int) * 5 * blobId;
	
	int offset = ((int *)blobData_ptr)[0];
	int inImageX = ((int *)blobData_ptr)[1];
	int inImageY = ((int *)blobData_ptr)[2];
	int maskWidth = ((int *)blobData_ptr)[3];
	int maskHeight = ((int *)blobData_ptr)[4];
	

	// cleanup cooc matrix
	for(int pos = threadIdx.x; pos < (PIXEL_INTENSITY_BINS); pos+= blockDim.x)
		pixel_hists_ptr[pos]=0;

	__syncthreads();

	// Each thread process a line from the input image
	for (int i=threadIdx.x; i < maskHeight; i+=blockDim.x){
		// create a pointer to the begining of line "i" in the input mask
		char *blobMaskLine = mask + offset + i * maskWidth;
		char *inImageLine = (char *)image + (i+inImageY) * width + inImageX;

		for(int j = 0; j < maskWidth; j++){
			// is the mask one for both pixels?
			if( (blobMaskLine[j]) != 0 ){
				int data=(unsigned char)inImageLine[j];

				atomicAdd(&pixel_hists_ptr[data], 1);
			}

		}
	}

	__syncthreads();

#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
	// this phase of writing results back to global memory is only necessary when 
	// atomic operations at shared memory are available

	for (int i= threadIdx.x; i < PIXEL_INTENSITY_BINS; i+=blockDim.x){
		blobs_hists[blobId * (PIXEL_INTENSITY_BINS) + i] = pixel_hists_ptr[i];
	}
	__syncthreads();
#endif

	
/*	if(threadIdx.x==0){
		float meanIntensity = 0.0;
		int nPixels = 0;
		float minIntensity = 0;
		float maxIntensity = 0;
		for(int i = 0; i < PIXEL_INTENSITY_BINS; i++){
			meanIntensity += pixel_hists_ptr[i] * i;
			nPixels += pixel_hists_ptr[i];
			if(minIntensity == 0 && pixel_hists_ptr[i] != 0){
				minIntensity = i;
			}
			if(pixel_hists_ptr[i] != 0){
				maxIntensity = i;
			}
			
		}
		float firstQuartile = 0;
		float thirdQuartile = 0;
		float median = 0;
		int accPixels = 0;
		for(int i = 0; i < PIXEL_INTENSITY_BINS; i++){
			accPixels += pixel_hists_ptr[i];
			if(firstQuartile == 0 && accPixels >= nPixels/4){
				firstQuartile = i;
			}
			if(median == 0 && accPixels >= nPixels/2){
				median = i;
			}
			if(thirdQuartile == 0 && accPixels >= (3*nPixels/4)){
				thirdQuartile = i;
			}

		}
		meanIntensity /=(float)nPixels;
		d_result[blobId * N_INTENSITY_FEATURES] = meanIntensity;
		d_result[blobId * N_INTENSITY_FEATURES + 1] = median;
		d_result[blobId * N_INTENSITY_FEATURES + 2] = minIntensity;
		d_result[blobId * N_INTENSITY_FEATURES + 3] = maxIntensity;
		d_result[blobId * N_INTENSITY_FEATURES + 4] = firstQuartile;
		d_result[blobId * N_INTENSITY_FEATURES + 5] = thirdQuartile;
	}*/	
}



__global__ void intensityFeaturesPerBlob_kernel(int *blobs_hists, int nBlobs, float *d_Result ){
	// calcuate the id of the blob of which this thread calcualtes the features
	int blobId=blockIdx.x * THREAD_N_BLOB_HARALICK + threadIdx.x;

	if(blobId >= nBlobs){
		blobId = nBlobs-1;
	}
	
	const int memoryBaseIndex=blobId * PIXEL_INTENSITY_BINS;
	int *pixel_hists_ptr = &blobs_hists[memoryBaseIndex];


	float meanIntensity = 0.0;
	float minIntensity = 0;
	float maxIntensity = 0;
	int nPixels = 0;
	for(int i = 0; i < PIXEL_INTENSITY_BINS; i++){
		meanIntensity += pixel_hists_ptr[i] * i;
		nPixels += pixel_hists_ptr[i];
		if(minIntensity == 0 && pixel_hists_ptr[i] != 0){
			minIntensity = i;
		}
		if(pixel_hists_ptr[i] != 0){
			maxIntensity = i;
		}

	}
	float firstQuartile = 0;
	float thirdQuartile = 0;
	float median = 0;
	int accPixels = 0;
	for(int i = 0; i < PIXEL_INTENSITY_BINS; i++){
		accPixels += pixel_hists_ptr[i];
		if(firstQuartile == 0 && accPixels >= nPixels/4){
			firstQuartile = i;
		}
		if(median == 0 && accPixels >= nPixels/2){
			median = i;
		}
		if(thirdQuartile == 0 && accPixels >= (3*nPixels)/4){
			thirdQuartile = i;
		}

	}
	meanIntensity /=(float)nPixels;
	d_Result[blobId * N_INTENSITY_FEATURES] = meanIntensity;
	d_Result[blobId * N_INTENSITY_FEATURES + 1] = median;
	d_Result[blobId * N_INTENSITY_FEATURES + 2] = minIntensity;
	d_Result[blobId * N_INTENSITY_FEATURES + 3] = maxIntensity;
	d_Result[blobId * N_INTENSITY_FEATURES + 4] = firstQuartile;
	d_Result[blobId * N_INTENSITY_FEATURES + 5] = thirdQuartile;

}



float *intensityGPUBlob( char * d_image, int width, int height, int nBlobs, char* d_mask){
	int BLOCK_N=nBlobs;
	float *d_Result;
	
	// Allocate space at GPU memory to store cooc. matrices
	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, N_INTENSITY_FEATURES * sizeof( float ) * nBlobs ));

	int *d_blobs_hists;
	// Allocate space at GPU memory to store cooc. matrices
	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_blobs_hists, PIXEL_INTENSITY_BINS * sizeof( int ) * nBlobs ));

	// Calculate grayscale histogram for each blob
	intensityHistGPUPerBlob_kernel<<<BLOCK_N,THREAD_N_BLOB>>>((unsigned char *)d_image, width, height, d_Result, d_mask, d_blobs_hists );


	// calcute number of blocks such that each thread will calculate intensity features for a single blob
	BLOCK_N=(nBlobs%THREAD_N_BLOB_HARALICK != 0) ? (nBlobs/THREAD_N_BLOB_HARALICK +1) : (nBlobs)/THREAD_N_BLOB_HARALICK;

	intensityFeaturesPerBlob_kernel<<<BLOCK_N,THREAD_N_BLOB_HARALICK>>>( d_blobs_hists, nBlobs, d_Result );
	// Calculate actual intensity features from the histogram

	cudaThreadSynchronize();

	CUDA_SAFE_CALL(cudaFree(d_blobs_hists));
	return d_Result;
}


///////*************** The code bellow is used for testing only ********************/////////////////

//int *intensityHistGPUBlob( char * d_image, int width, int height, int nBlobs, char* d_mask){
//	const int BLOCK_N=nBlobs;
//	float *d_Result;
//	
//	// Allocate space at GPU memory to store cooc. matrices
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, N_INTENSITY_FEATURES * sizeof( float ) * nBlobs ));
//
//	int *d_blobs_hists;
//	// Allocate space at GPU memory to store cooc. matrices
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_blobs_hists, PIXEL_INTENSITY_BINS * sizeof( int ) * nBlobs ));
//
//	intensityGPUPerBlob_kernel<<<BLOCK_N,THREAD_N_BLOB>>>((unsigned char*)d_image, width, height, d_Result, d_mask, d_blobs_hists );
//
//	cudaThreadSynchronize();
//
////	CUDA_SAFE_CALL(cudaFree(d_blobs_hists));
//	return d_blobs_hists;
//}
//


//void haralickGPUBlob2( int  *d_coocMatrix, int nBlobs, float * haralickFeatures){
//	// calcute number of blocks such that each thread will calculate haralick features for a single blob
//	const int BLOCK_N=(nBlobs%THREAD_N_BLOB != 0) ? (nBlobs/THREAD_N_BLOB +1) : (nBlobs)/THREAD_N_BLOB;
//
//	// alloc space used to store Haralick feature in GPU memory
//	float *d_HaralickFeatures;
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_HaralickFeatures, N_HARALICK_FEATURES * nBlobs * sizeof( float ) ));
//
//	// calcute features
//	haralickFeaturesPerBlob_kernel_shared<<<BLOCK_N,THREAD_N_BLOB>>>(d_coocMatrix, nBlobs, d_HaralickFeatures );
//
//	// copy results back to CPU memory
//	cudaMemcpy( (void*)haralickFeatures, (void*)d_HaralickFeatures, N_HARALICK_FEATURES * nBlobs * sizeof(float), cudaMemcpyDeviceToHost );	
//	
//	// free GPU memory used to calculate features
//	cudaFree(d_HaralickFeatures);
//
//}


//__global__ void coocurrenceGPUPerBlobAndFeatures(unsigned char* image, int width, int height, int *matrix, int cooc_size,  char* mask, float* haralickFeatures ){
//	const int blobId=blockIdx.x;
//
//// If does not support atomic operations using shared memory, 
//// go with the atomic at the global memory which is much slower.
//#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
//	// Per-blob/block coocurrence matrix 
//	__shared__ int s_coocmat[COOC_SIZE*COOC_SIZE];
//#else
//	int *s_coocmat = &matrix[blobId * (COOC_SIZE * COOC_SIZE)];
//#endif
//
//	
//
//	char *blobData_ptr = mask + sizeof(int) * 5 * blobId;
//
//	int offset = ((int *)blobData_ptr)[0];
//	int inImageX = ((int *)blobData_ptr)[1];
//	int inImageY = ((int *)blobData_ptr)[2];
//	int maskWidth = ((int *)blobData_ptr)[3];
//	int maskHeight = ((int *)blobData_ptr)[4];
//	
//
//	// cleanup cooc matrix
//	for(int pos = threadIdx.x; pos < (COOC_SIZE * COOC_SIZE); pos+= blockDim.x)
//		s_coocmat[pos]=0;
//
//	__syncthreads();
//
//
//	// Each thread processes a line from the input image
//	for (int i=threadIdx.x; i < maskHeight; i+=blockDim.x){
//		// create a pointer to the begining of line "i" in the input mask
//		char *blobDataLine = mask + offset + i * maskWidth;
//		char *normImageDataLine = (char *)image + (i+inImageY) * width + inImageX;
//
//		for(int j = 0; j < maskWidth-1; j++){
//			// is the mask one for both pixels?
//			if(((int)blobDataLine[j]) != 0  && ((int)blobDataLine[j+1]) != 0 ){
//				int data=(cooc_size * (normImageDataLine[j]-1)); 
//				data +=  normImageDataLine[j+1] -1;
//				atomicAdd(&s_coocmat[data], 1);
//			}
//
//		}
//	}
//
////	// this phase of writing results back to global memory is only need for version 
////	// that uses atomic shared memory operations
////#ifndef CUDA_NO_SM_12_ATOMIC_INTRINSICS
////	__syncthreads();
////	for (int i= threadIdx.x; i < (COOC_SIZE * COOC_SIZE); i+=blockDim.x){
////		matrix[blobId * (COOC_SIZE * COOC_SIZE) + i] = s_coocmat[i];
////	}
////#endif
//	__syncthreads();
//
//
//	if(threadIdx.x == 0){
//		int sumCoocMatrix = 0;
//		for(int i = 0; i < COOC_SIZE; i++){
//			for(int j = 0; j < COOC_SIZE; j++){
//				sumCoocMatrix += s_coocmat[i * COOC_SIZE + j];
//			}
//		}
//
//		// Calculate Mx and My, which are used by the clusterShade and clusterProminence
//		// Mx calculation
//		float mx = 0.0;
//		float my = 0.0;
//		for(int i = 0; i < COOC_SIZE; i++){
//			for(int j = 0; j < COOC_SIZE; j++){
//				float entryIJProbability = (float)s_coocmat[i * COOC_SIZE + j]/(float)sumCoocMatrix;
//				mx += i * entryIJProbability;
//				my += j * entryIJProbability;
//			}
//		}
//
//		// compute inertia
//		float inertia=0.0;
//		float energy=0.0;
//		float entropy=0.0;
//		float homogeneity=0.0;
//		float maximumProbability=0.0;
//		float clusterShade=0.0;
//		float clusterProminence=0.0;
//
//		const int k=1; // distance from pixels when calculating cooc. matrix. We're only doing it for distance=1;
//
//		for(int i = 0; i < COOC_SIZE; i++){
//			for(int j = 0; j < COOC_SIZE; j++){
//				float ij = i - j;
//				float entryIJProbability = (float)s_coocmat[ i * COOC_SIZE + j]/(float)sumCoocMatrix;
//
//				inertia += powf(ij,2) * entryIJProbability; 
//
//				energy += powf( entryIJProbability, 2) ;
//
//				if(entryIJProbability != 0.0){
//					entropy += entryIJProbability * log2f(entryIJProbability) ;
//				}
//				homogeneity += (1.0/(1.0 + powf(ij, 2)) * entryIJProbability);
//				if(entryIJProbability > maximumProbability){
//					maximumProbability = entryIJProbability;
//				}
//
//				clusterShade += powf( (k-mx + j-my), 3) * entryIJProbability;
//				clusterProminence += powf( (k-mx + j-my), 4) * entryIJProbability;
//			}
//		}
//
//		// calculate base address into shared memory to store features
//		const int haralickFeaturesBaseIndex = blobId * N_HARALICK_FEATURES;
//
//		// write results back to GPU main memory
//		haralickFeatures[haralickFeaturesBaseIndex] = inertia;
//		haralickFeatures[haralickFeaturesBaseIndex+1] = energy;
//		haralickFeatures[haralickFeaturesBaseIndex+2] = entropy;
//		haralickFeatures[haralickFeaturesBaseIndex+3] = homogeneity;
//		haralickFeatures[haralickFeaturesBaseIndex+4] = maximumProbability;
//		haralickFeatures[haralickFeaturesBaseIndex+5] = clusterShade;
//		haralickFeatures[haralickFeaturesBaseIndex+6] = clusterProminence;
//
//		__syncthreads();	
//	}
//}
//
//
//float *coocGPUBlobANDFEATURES( unsigned char * inImage, int width, int height, int cooc_size, int nBlobs, char *d_maskData, int angle){
//
//	const int BLOCK_N=nBlobs;
//	
//	
//	// Allocate one coocurrence matrix per blob at the GPU memory	
//	int *d_Result;
//
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, cooc_size*cooc_size* BLOCK_N* sizeof( int ) ));
//	// alloc space used to store Haralick feature in GPU memory
//	float *d_HaralickFeatures;
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_HaralickFeatures, N_HARALICK_FEATURES * nBlobs * sizeof( float ) ));
//
//
//	coocurrenceGPUPerBlobAndFeatures<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData, d_HaralickFeatures );
//	
//	cudaThreadSynchronize();	
//
//	CUDA_SAFE_CALL(cudaFree(d_Result));
//	
//
//	return d_HaralickFeatures;
//}
//
//
//int *intensityHistGPUBlobBAK( char * d_image, int width, int height, int nBlobs, char* d_mask){
//	const int BLOCK_N=2;
//	float *d_Result;
//	
//	// Allocate space at GPU memory to store cooc. matrices
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, N_INTENSITY_FEATURES * sizeof( float ) * 3 ));
//
//	int *d_blobs_hists;
//	// Allocate space at GPU memory to store cooc. matrices
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_blobs_hists, PIXEL_INTENSITY_BINS * sizeof( int ) * 3 ));
//
//	char *inImage = d_image;
//	char *d_maskData;
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_maskData, 10*sizeof(int) +18*sizeof(char )));
//
//
//	int *h_temp = (int *) malloc(10*sizeof(int) +18*sizeof(char ));// + 2 * sizeof(char));
//	h_temp[0] = 10*sizeof(int);
//	h_temp[1] = 0;
//	h_temp[2] = 0;
//	h_temp[3] = 3;
//	h_temp[4] = 3;
//	h_temp[5] = 10*sizeof(int)+h_temp[3]*h_temp[4];
//	h_temp[6] = 3;
//	h_temp[7] = 0;
//	h_temp[8] = 3;
//	h_temp[9] = 3;
//	char *auxTemp = (char*)h_temp + 10*sizeof(int);
//	auxTemp[0] = 1;
//	auxTemp[1] = 0;
//	auxTemp[2] = 1;
//	auxTemp[3] = 1;
//	auxTemp[4] = 1;
//	auxTemp[5] = 1;
//	auxTemp[6] = 1;
//	auxTemp[7] = 1;
//	auxTemp[8] = 1;
//	auxTemp[9] = 1;
//	auxTemp[10] = 0;
//	auxTemp[11] = 1;
//	auxTemp[12] = 1;
//	auxTemp[13] = 1;
//	auxTemp[14] = 1;
//	auxTemp[15] = 1;
//	auxTemp[16] = 1;
//	auxTemp[17] = 1;
//	// copy blob masks to GPU memory
//	CUDA_SAFE_CALL(cudaMemcpy(d_maskData, h_temp, 10*sizeof(int) + 18*sizeof(char), cudaMemcpyHostToDevice ));
//
////	cudaFree(inImage);
//	char *h_inImage = (char*) malloc(sizeof(char) * 18);
//	h_inImage[0] = 1;
//	h_inImage[1] = 1;
//	h_inImage[2] = 2;
//	h_inImage[3] = 1;
//	h_inImage[4] = 2;
//	h_inImage[5] = 2;
//	h_inImage[6] = 2;
//	h_inImage[7] = 3;
//	h_inImage[8] = 3;
//	h_inImage[9] = 1;
//	h_inImage[10] = 1;
//	h_inImage[11] = 2;
//	h_inImage[12] = 1;
//	h_inImage[13] = 2;
//	h_inImage[14] = 2;
//	h_inImage[15] = 2;
//	h_inImage[16] = 3;
//	h_inImage[17] = 4;
//
//	// copy blob masks to GPU memory
//	CUDA_SAFE_CALL(cudaMemcpy(inImage, h_inImage, 18*sizeof(char), cudaMemcpyHostToDevice ));
//
//
//	width=6;
//	height=3;
//
//
//	intensityGPUPerBlob_kernel<<<BLOCK_N,THREAD_N_BLOB>>>((unsigned char*)inImage, width, height, d_Result, d_maskData, d_blobs_hists );
//
//	cudaThreadSynchronize();
//
////	CUDA_SAFE_CALL(cudaFree(d_blobs_hists));
//	return d_blobs_hists;
//}
//
//
//
//void coocGPUBlobBak( unsigned char * inImage, int width, int height, int *cooc, int cooc_size, int nBlobs, char *d_maskData, int angle){
//
//	const int BLOCK_N=2;
//	
//	
//	// Allocate one coocurrence matrix per blob at the GPU memory	
//	int *d_Result;
//
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_Result, cooc_size*cooc_size* BLOCK_N* sizeof( int ) ));
//	cudaFree(d_maskData);
//
//	CUDA_SAFE_CALL(cudaMalloc( (void**)&d_maskData, 10*sizeof(int) +18*sizeof(char )));
//
//	int *h_temp = (int *) malloc(10*sizeof(int) +18*sizeof(char ));// + 2 * sizeof(char));
//	h_temp[0] = 10*sizeof(int);
//	h_temp[1] = 0;
//	h_temp[2] = 0;
//	h_temp[3] = 3;
//	h_temp[4] = 3;
//	h_temp[5] = 10*sizeof(int)+h_temp[3]*h_temp[4];
//	h_temp[6] = 3;
//	h_temp[7] = 0;
//	h_temp[8] = 3;
//	h_temp[9] = 3;
//	char *auxTemp = (char*)h_temp + 10*sizeof(int);
//	auxTemp[0] = 1;
//	auxTemp[1] = 0;
//	auxTemp[2] = 1;
//	auxTemp[3] = 1;
//	auxTemp[4] = 1;
//	auxTemp[5] = 1;
//	auxTemp[6] = 1;
//	auxTemp[7] = 1;
//	auxTemp[8] = 1;
//	auxTemp[9] = 1;
//	auxTemp[10] = 0;
//	auxTemp[11] = 1;
//	auxTemp[12] = 1;
//	auxTemp[13] = 1;
//	auxTemp[14] = 1;
//	auxTemp[15] = 1;
//	auxTemp[16] = 1;
//	auxTemp[17] = 1;
//	// copy blob masks to GPU memory
//	CUDA_SAFE_CALL(cudaMemcpy(d_maskData, h_temp, 10*sizeof(int) + 18*sizeof(char), cudaMemcpyHostToDevice ));
//
////	cudaFree(inImage);
//	char *h_inImage = (char*) malloc(sizeof(char) * 18);
//	h_inImage[0] = 1;
//	h_inImage[1] = 1;
//	h_inImage[2] = 2;
//	h_inImage[3] = 1;
//	h_inImage[4] = 2;
//	h_inImage[5] = 2;
//	h_inImage[6] = 2;
//	h_inImage[7] = 3;
//	h_inImage[8] = 3;
//	h_inImage[9] = 1;
//	h_inImage[10] = 1;
//	h_inImage[11] = 2;
//	h_inImage[12] = 1;
//	h_inImage[13] = 2;
//	h_inImage[14] = 2;
//	h_inImage[15] = 2;
//	h_inImage[16] = 3;
//	h_inImage[17] = 4;
//
//	// copy blob masks to GPU memory
//	CUDA_SAFE_CALL(cudaMemcpy(inImage, h_inImage, 18*sizeof(char), cudaMemcpyHostToDevice ));
//
//
//	width=6;
//	height=3;
//
//	
//	coocurrenceGPUPerBlob<<<BLOCK_N,THREAD_N_BLOB>>>(inImage, width, height, d_Result, cooc_size, d_maskData );
//	
//	cudaThreadSynchronize();	
//	
//	cudaMemcpy( (void*)cooc, (void*)d_Result, BLOCK_N*cooc_size*cooc_size*sizeof(int), cudaMemcpyDeviceToHost );	
//
//	
//	
//	// TODO: cudaFree
//	cudaFree(d_Result);
//}
//
