#ifndef _CUDA_SCAN_FUNCTIONS_H_
#define _CUDA_SCAN_FUNCTIONS_H_

enum eScanCUDADataTypes
{
	eScan_CUDA_Type_Int = 0,	//scan integer elements
	eScan_CUDA_Type_Float,
	eScan_CUDA_Type_Float2,
	eScan_CUDA_Type_Float4,
	eScan_CUDA_Types_Count	
};

extern "C" bool cudaScanInclusiveSingleBlock(long dataType, int bytesPerElement, void* dDataIn, void* dDataOut, int elements, int threads);
extern "C" bool cudaScanInclusiveMultiBlock(long dataType, int bytesPerElement, void* dDataIn, void* dDataOut, void* dBlockOut, int elements, int blocks, int threads);
extern "C" bool cudaScanUniformUpdate(long dataType, int bytesPerElement, void* dDataInOut, void* dBlockIn, int elements, int blocks, int threads);

#endif