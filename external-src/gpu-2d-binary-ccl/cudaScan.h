#ifndef _CUDA_SCAN_H_
#define _CUDA_SCAN_H_


#include "cudaBuffer.h"
#include "cudaScanFunctions.h"
#include <vector>

class CudaScan
{
public:
	CudaScan();
	~CudaScan();

	//expectedNoElements == expected number of elements that will be processed (can be 0)
	bool Init(long dataType, int expectedNoElements);
	void Clear();
	//note dataIn can be the same as dataOut
	bool ScanInclusive(long dataType, CudaBuffer* dataIn, CudaBuffer* dataOut, int noOfElements);	
private:
	bool PrepareBlockBuffers(int numElements, int bytesPerElement);
	bool AllocateBufferForLevel(int level, int size);

	bool ScanInclusiveRecursive(long dataType, int level, CudaBuffer* dataIn, CudaBuffer* dataOut, int noOfElements, int bytesPerElement);
private:
	

	//buffer containing scan data for multi-block scan for different levels
	std::vector<CudaBuffer*> m_blockBuf;

	int m_threads;
};

#endif