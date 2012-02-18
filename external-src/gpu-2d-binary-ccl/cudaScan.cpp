#include "cudaScan.h"
#include "cudaBasicUtils.h"
#include "cudaScanFunctions.h"
#include <string>

int ScanCUDADataTypeBytesPerElement[] =
{
	4,
	4,
	8,
	16
};

CudaScan::CudaScan()
{
	m_threads = 512;		//DO NOT CHANGE!!!!!!!!!!!!!!
}

CudaScan::~CudaScan()
{
	Clear();
}

void CudaScan::Clear()
{
	for(unsigned int i=0;i<m_blockBuf.size();++i) delete m_blockBuf[i];
	m_blockBuf.clear();
}

bool CudaScan::Init(long dataType, int expectedElements)
{	
	if(expectedElements > 0)
	{
		if(!PrepareBlockBuffers(expectedElements, ScanCUDADataTypeBytesPerElement[dataType]))
			return false;
	}
	return true;
}

bool CudaScan::PrepareBlockBuffers(int numElements, int bytesPerElement)
{
	if(numElements <= m_threads) return true;
	//calc how many blocks will be needed for each level
	int actLevel = 0;
	int levelElements = numElements;
	int blocksNeeded = 0;
	do {
		blocksNeeded = CudaBasicUtils::RoundUp(m_threads, levelElements) / m_threads;
		blocksNeeded -= 1;		//we don't need to care about the last block
		int size = blocksNeeded * bytesPerElement;
		if(!AllocateBufferForLevel(actLevel, size))
			return false;
		actLevel++;
		levelElements = blocksNeeded;
	} while(blocksNeeded > m_threads);
	return true;
}

bool CudaScan::AllocateBufferForLevel(int level, int size)
{
	if(m_blockBuf.size() <= level) m_blockBuf.resize(level+1, 0);
	if(m_blockBuf[level] == 0) m_blockBuf[level] = new CudaBuffer();
	if(m_blockBuf[level]->GetSize() < size) {
		if(!m_blockBuf[level]->Delete())
			return false;
		if(!m_blockBuf[level]->Create(CudaBasicUtils::RoundUp(256, size)))
			return false;
	}
	return true;
}

bool CudaScan::ScanInclusive(long dataType, CudaBuffer* dataIn, CudaBuffer* dataOut, int numElements)
{
	int bytesPerElement = ScanCUDADataTypeBytesPerElement[dataType];
	if(!PrepareBlockBuffers(numElements, bytesPerElement)) 
		return false;
	if(numElements <= m_threads) {
		//run the single-block scan version
		if(!cudaScanInclusiveSingleBlock(dataType, bytesPerElement, dataIn->GetData(), dataOut->GetData(), numElements, m_threads))
			return false;
	} else {
		//run the multi-block version
		if(!ScanInclusiveRecursive(dataType, 0, dataIn, dataOut, numElements, bytesPerElement))
			return false;
	}
	return true;
}

bool CudaScan::ScanInclusiveRecursive(long dataType, int level, CudaBuffer* dataIn, CudaBuffer* dataOut, int numElements, int bytesPerElement)
{
	if(numElements <= m_threads) {
		//last level
		if(!cudaScanInclusiveSingleBlock(dataType, bytesPerElement, dataIn->GetData(), dataOut->GetData(), numElements, m_threads))
			return false;		
		return true;
	}
	//else run a multiblock prescan first
	int workSize = CudaBasicUtils::RoundUp(m_threads, numElements);
	int blocks = workSize / m_threads;

	if(!cudaScanInclusiveMultiBlock(dataType, bytesPerElement, dataIn->GetData(), dataOut->GetData(), m_blockBuf[level]->GetData(), numElements, blocks, m_threads))
		return false;
	//no calculate no of elements that is needed for the next level
	int nextLevelElements = workSize / m_threads;
	nextLevelElements -= 1;
	if(!ScanInclusiveRecursive(dataType, level+1,  m_blockBuf[level],  m_blockBuf[level], nextLevelElements, bytesPerElement))
		return false;
	//do the uniform update
	if(!cudaScanUniformUpdate(dataType, bytesPerElement, dataOut->GetData(), m_blockBuf[level]->GetData(), numElements, blocks-1, m_threads))
		return false;

	return true;
}
