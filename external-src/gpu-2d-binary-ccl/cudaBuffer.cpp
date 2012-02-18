#include "cudaBuffer.h"
#include "cudaArray.h"
#include <cuda_runtime_api.h>
#include <driver_functions.h>
#include <string.h>
#include <stdlib.h>

CudaBuffer::CudaBuffer()
{
	m_data = 0;	
	m_deviceMem = true;
	memset(m_size, 0 , sizeof(m_size));
	m_pitch = 0;
}

CudaBuffer::~CudaBuffer()
{
	Delete();
}

bool CudaBuffer::Delete()
{
	if(m_data) {
		cudaError_t err;
		if(m_deviceMem) {
			err = cudaFree(m_data);
		} else {
			err = cudaFreeHost(m_data);
		}
		if(err != cudaSuccess) 
			return false;
	}
	m_data = 0;
	memset(m_size, 0 , sizeof(m_size));
	m_pitch = 0;
	return true;
}

bool CudaBuffer::Create(int size, bool onDevice, bool pinned)
{
	Delete();
	cudaError_t err = cudaSuccess;
	if(onDevice) {
		err = cudaMalloc(&m_data, size);		
	} else {	
		if(pinned) err = cudaMallocHost(&m_data, size);
		else m_data = malloc(size);
	}	
	if(err != cudaSuccess)
		return false;
	m_size[0] = size;
	m_size[1] = 1;
	m_size[2] = 1;
	m_pitch = size;
	m_deviceMem = onDevice;
	return true;
}

bool CudaBuffer::Create2D(int sizeX/*in bytes*/, int height/*in rows*/, bool onDevice)
{
	Delete();
	cudaError_t err;
	size_t pitch = 0;
	if(onDevice) {
		err = cudaMallocPitch(&m_data, &pitch, sizeX, height);
	} else {
		err = cudaMallocHost(&m_data, sizeX*height);
	}	
	if(err != cudaSuccess)
		return false;
	m_size[0] = sizeX;
	m_size[1] = height;
	m_size[2] = 1;
	m_pitch = pitch;
	m_deviceMem = onDevice;
	return true;
}

bool CudaBuffer::CopyFrom(CudaBuffer* srcBuffer)
{
	int off[3] = { 0,0,0};

	return CopyFrom(srcBuffer, off, off, m_size);
}

bool CudaBuffer::CopyFrom(CudaBuffer* srcBuffer, int srcOffset, int dstOffset, int size)
{
	int soff[3] = { srcOffset,0,0};
	int doff[3] = { dstOffset,0,0};
	int sizeA[3] = { size, 1, 1};

	return CopyFrom(srcBuffer, soff, doff, sizeA);
}

bool CudaBuffer::CopyFrom(CudaBuffer* srcBuffer, int srcOffset[3], int dstOffset[3], int size[3])
{
	cudaError_t err;
	if(m_size[2] > 1 || srcBuffer->m_size[2] > 1) {
		//copy 3D - TODO
		return false;
	} 
	if(m_size[1] > 1 || srcBuffer->m_size[1] > 1) {
		//copy 2D
		int srcPitch = srcBuffer->m_pitch;
		if(srcPitch == 0) srcPitch = srcBuffer->m_size[0];
		int dstPitch = m_pitch;
		if(dstPitch == 0) dstPitch = m_size[0];
		int sOffset = srcOffset[0] + srcPitch*srcOffset[1];
		int dOffset = dstOffset[0] + dstPitch*dstOffset[1];		
		err = cudaMemcpy2D((unsigned char*)m_data+dOffset, dstPitch, (unsigned char*)srcBuffer->m_data+sOffset, srcPitch, size[0], size[1], (cudaMemcpyKind)GetMemcpyKind(srcBuffer));
		if(err != cudaSuccess) {
			return false;
		}
		return true;
	}
	//else 1d memcpy	
	err = cudaMemcpy((unsigned char*)m_data+dstOffset[0], (unsigned char*)srcBuffer->m_data+srcOffset[0], size[0], (cudaMemcpyKind)GetMemcpyKind(srcBuffer));
	if(err != cudaSuccess) {
		return false;
	}
	return true;
}

bool CudaBuffer::CopyFrom(void* data, int srcPitch, int w /*in bytes*/, int h)
{
	cudaError_t err;
	cudaMemcpyKind kind = (m_deviceMem?cudaMemcpyHostToDevice:cudaMemcpyHostToHost);
	err = cudaMemcpy2D(m_data, GetPitch(), data, srcPitch, w, h, kind);
	if(err != cudaSuccess)
		return false;
	return true;
}

bool CudaBuffer::CopyToHost(void* dstData, int dstPitch, int w, int h)
{
	if(m_size[1] > 1) {
		cudaError_t err;
		cudaMemcpyKind kind = (m_deviceMem?cudaMemcpyDeviceToHost:cudaMemcpyHostToHost);
		err = cudaMemcpy2D(dstData, dstPitch, m_data, GetPitch(), w, h, kind);
		if(err != cudaSuccess)
			return false;
		return true;
	}
	//other device to host copying not implemented yet
	return false;
}

bool CudaBuffer::SetZeroData()
{
	cudaError_t err;
	if(m_size[2] > 1) {
		err = cudaMemset3D(make_cudaPitchedPtr(m_data, m_pitch, m_size[0], m_size[1]*m_size[2]),
					 0, make_cudaExtent(m_size[0], m_size[1], m_size[2]));
	} else if(m_size[1] > 1) {
		err = cudaMemset2D(m_data, m_pitch, 0, m_size[0], m_size[1]);
	} else {
		err = cudaMemset(m_data, 0, m_size[0]);
	}
	if(err != cudaSuccess)
		return false;
	return true;
}

bool CudaBuffer::SetUcharData(unsigned char val)
{
	cudaError_t err;
	if(m_size[2] > 1) {
		err = cudaMemset3D(make_cudaPitchedPtr(m_data, m_pitch, m_size[0], m_size[1]*m_size[2]),
					 val, make_cudaExtent(m_size[0], m_size[1], m_size[2]));
	} else if(m_size[1] > 1) {
		err = cudaMemset2D(m_data, m_pitch, val, m_size[0], m_size[1]);
	} else {
		err = cudaMemset(m_data, val, m_size[0]);
	}
	if(err != cudaSuccess)
		return false;
	return true;
}

long CudaBuffer::GetMemcpyKind(CudaArray* srcArray)
{
	if(m_deviceMem) return cudaMemcpyDeviceToDevice;
	return cudaMemcpyDeviceToHost;
}

long CudaBuffer::GetMemcpyKind(CudaBuffer* srcBuffer)
{
	if(m_deviceMem) {
		if(srcBuffer->m_deviceMem) return cudaMemcpyDeviceToDevice;
		return cudaMemcpyHostToDevice;
	}
	if(srcBuffer->m_deviceMem) return cudaMemcpyDeviceToHost;
	return cudaMemcpyHostToHost;
}