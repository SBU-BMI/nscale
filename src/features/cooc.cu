
#include "cutil.h"

#define COOC_SIZE 8
#define BIN_COUNT (COOC_SIZE*COOC_SIZE)
#define BLOCK_DIM 16
#define THREAD_N 128
#define BLOCK_MEMORY (THREAD_N*BIN_COUNT)
#define BLOCK_DATA (THREAD_N*255)
#define round(num) (num+0.5f)


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
	
	//Per-thread coocurrence matrix storage
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
	
	//Per-thread coocurrence matrix storage
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
	
	//Per-thread coocurrence matrix storage
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
	
	//Per-thread coocurrence matrix storage
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
