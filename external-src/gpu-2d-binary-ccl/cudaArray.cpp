#include "cudaArray.h"
#include "cudaBuffer.h"
#include <cuda.h>
#include <channel_descriptor.h>

CudaArray::CudaArray()
{
	m_array = 0;
	m_w = 0;
	m_h = 0;
	m_d = 0;
	m_is3D = false;
	m_createdExternally = false;
}

CudaArray::~CudaArray()
{
	Clear();
}

bool CudaArray::Clear()
{
	if(m_array && !m_createdExternally) {
		cudaError_t err = cudaFreeArray(m_array);
		if(err != cudaSuccess) 
			return false;
	}
	m_array = 0;
	m_w = 0;
	m_h = 0;
	m_d = 0;
	return true;
}

bool CudaArray::Create2D(int w, int h, eCudaArrayChannelFormatDesc desc)
{
	Clear();
	cudaChannelFormatDesc cD;
	switch(desc) {
		case eCUDA_CHANNEL_INT:
			cD = cudaCreateChannelDesc<int>(); break;
		case eCUDA_CHANNEL_INT4:
			cD = cudaCreateChannelDesc<int4>(); break;
		case eCUDA_CHANNEL_UCHAR:
			cD = cudaCreateChannelDesc<uchar1>(); break;
		case eCUDA_CHANNEL_UCHAR4:
			cD = cudaCreateChannelDesc<uchar4>(); break;
		case eCUDA_CHANNEL_FLOAT:		
			cD = cudaCreateChannelDesc<float>(); break;
		case eCUDA_CHANNEL_FLOAT4: 
			cD = cudaCreateChannelDesc<float4>(); break;
		default:
			return false;
	}
	cudaError_t err = cudaMallocArray(&m_array, &cD, w, h);
	if(err != cudaSuccess) 
		return false;
	m_w = w;
	m_h = h;
	m_d = 1;
	m_is3D = false;
	m_createdExternally = false;
	return true;
}

bool CudaArray::CreateFromCUDAData(cudaArray* array, int w, int h, int d, bool is3D)
{
	m_array = array;
	m_w = w;
	m_h = h;
	m_d = d;
	m_is3D = is3D;
	m_createdExternally = true;
	return true;
}

bool CudaArray::CopyFrom(CudaBuffer* buf)
{
	//lets handle only 2D copy for now
	cudaError_t err = cudaMemcpy2DToArray(m_array, 0, 0, buf->GetData(), buf->GetPitch(), buf->GetSize(0), buf->GetSize(1), (cudaMemcpyKind)GetMemcpyKind(buf));
	if(err != cudaSuccess) 
		return false;
	return true;
}

bool CudaArray::CopyFrom(CudaBuffer* buf, int dstOffset, int size)
{
	//1D copy
	cudaError_t err = cudaMemcpyToArray(m_array, dstOffset, 0, buf->GetData(), size, (cudaMemcpyKind)GetMemcpyKind(buf));
	if(err != cudaSuccess) 
		return false;
	return true;
}

long CudaArray::GetMemcpyKind(CudaBuffer* srcBuffer)
{
	if(srcBuffer->IsOnDevice())  return cudaMemcpyDeviceToDevice;
	return cudaMemcpyHostToDevice;	
}