#include "cudaScanFunctions.h"
#include "cudaScan_kernels.cu"
#include "cudaUtil.h"

template<class T>
bool cudaScanInclusiveSingleBlockInternal(int bytesPerElement, void* dDataIn, void* dDataOut, int elements, int threads)
{
	scanInclusiveSingleBlock<T><<<1, threads, 2*bytesPerElement*threads>>>((T*)dDataIn, (T*)dDataOut, elements);
	return true;
}

extern "C" bool cudaScanInclusiveSingleBlock(long dataType, int bytesPerElement, void* dDataIn, void* dDataOut, int elements, int threads)
{
	switch(dataType) {
		case eScan_CUDA_Type_Int:
			return cudaScanInclusiveSingleBlockInternal<int>(bytesPerElement, dDataIn, dDataOut, elements, threads);
		default: 
			return false;
	}
	return true;
}

template<class T>
bool cudaScanInclusiveMultiBlockInternal(int bytesPerElement, void* dDataIn, void* dDataOut, void* dBlockOut, int elements, int blocks, int threads)
{
	scanInclusiveMultiBlock<T><<<blocks, threads, 2*bytesPerElement*threads>>>((T*)dDataIn, (T*)dDataOut, (T*)dBlockOut, elements);
	return true;
}

extern "C" bool cudaScanInclusiveMultiBlock(long dataType, int bytesPerElement, void* dDataIn, void* dDataOut, void* dBlockOut, int elements, int blocks, int threads)
{	
	switch(dataType) {
		case eScan_CUDA_Type_Int:
			return cudaScanInclusiveMultiBlockInternal<int>(bytesPerElement, dDataIn, dDataOut, dBlockOut, elements, blocks, threads);
		default:
			return false;
	}
	return true;
}

template<class T>
bool cudaScanUniformUpdateInternal(int bytesPerElement, void* dDataInOut, void* dBlockIn, int elements, int blocks, int threads)
{
	uniformUpdate<T><<<blocks, threads>>>((T*)dDataInOut, (T*)dBlockIn);
	return true;
}

extern "C" bool cudaScanUniformUpdate(long dataType, int bytesPerElement, void* dDataInOut, void* dBlockIn, int elements, int blocks, int threads)
{
	switch(dataType) {
		case eScan_CUDA_Type_Int:
			return cudaScanUniformUpdateInternal<int>(bytesPerElement, dDataInOut, dBlockIn, elements, blocks, threads);
		default:
			return false;
	}
	return true;
}