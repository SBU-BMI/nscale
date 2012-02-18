#ifndef _CUDA_ARRAY_H_
#define _CUDA_ARRAY_H_

struct cudaArray;
class CudaBuffer;

enum eCudaArrayChannelFormatDesc
{
	eCUDA_CHANNEL_INT,
	eCUDA_CHANNEL_INT4,
	eCUDA_CHANNEL_UCHAR,
	eCUDA_CHANNEL_UCHAR4,
	eCUDA_CHANNEL_FLOAT,
	eCUDA_CHANNEL_FLOAT4,
};

class CudaArray
{
public:
	CudaArray();
	~CudaArray();
	bool Clear();
	bool Create2D(int w, int h, eCudaArrayChannelFormatDesc desc);
	bool CreateFromCUDAData(cudaArray* array, int w, int h, int d, bool is3D);
	bool CopyFrom(CudaBuffer* buf);
	bool CopyFrom(CudaBuffer* buf, int dstOffset, int size);
	bool Is3D() const { return m_is3D; }
	cudaArray* GetArray() { return m_array; }
private:
	long GetMemcpyKind(CudaBuffer* srcBuffer);
private:
	cudaArray* m_array;

	int m_w;
	int m_h;
	int m_d;		
	bool m_is3D;	//is it a 3D array?
	bool m_createdExternally;
};

#endif