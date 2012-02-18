#ifndef _CUDA_BUFFER_H_
#define _CUDA_BUFFER_H_

class CudaArray;
//a linear buffer (can be allocated both on the host or on the device

class CudaBuffer
{
public:
	CudaBuffer();
	~CudaBuffer();
	bool Delete();
	bool Create(int size, bool onDevice=true, bool pinned=true);
	bool Create2D(int sizeX/*in bytes*/, int height/*in rows*/, bool onDevice=true);

	bool CopyFrom(CudaBuffer* srcBuffer);
	bool CopyFrom(CudaBuffer* srcBuffer, int srcOffset, int dstOffset, int size);
	//copy two buffers ( size[0] .. bytes per row, size[1] .. number of rows, size[2] depth)
	bool CopyFrom(CudaBuffer* srcBuffer, int srcOffset[3], int dstOffset[3], int size[3]);	
	bool CopyFrom(void* data, int srcPitch, int w /*in bytes*/, int h);
	bool CopyToHost(void* dstData, int dstPitch, int w, int h);
	bool SetZeroData();
	bool SetUcharData(unsigned char val);
	void* GetData() { return m_data; }
	int GetSize(int dim=0) const { return m_size[dim]; }
	int GetPitch() const { return m_pitch; }
	bool IsOnDevice() const { return m_deviceMem; }
protected:
	long GetMemcpyKind(CudaArray* srcArray);
	long GetMemcpyKind(CudaBuffer* srcBuffer);
private:
	void* m_data;
	int m_size[3];
	int m_pitch;	//used when working with 2D and 3D buffers
	//is the buffer allocated on a device?
	bool m_deviceMem;
};

#endif