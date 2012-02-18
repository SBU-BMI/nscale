#ifndef _CUDA_FUNCTIONS_BASE_H_
#define _CUDA_FUNCTIONS_BASE_H_

struct cudaArray;

typedef unsigned char uchar;
typedef unsigned int uint;

struct sCudaBuffer2D
{
	sCudaBuffer2D(void* data, int pitch) { m_data = data; m_pitch = pitch; }
	void* m_data;	
	int m_pitch;
};

struct sUchar4 {
	sUchar4(unsigned char v0, unsigned char v1, unsigned char v2, unsigned char v3) { m_val[0] = v0; m_val[1] = v1; m_val[2] = v2; m_val[3] = v3; }
	unsigned char m_val[4];
};

#endif