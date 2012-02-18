#ifndef _CUDA_BASIC_UTILS_H_
#define _CUDA_BASIC_UTILS_H_

//note: in order to use some of the utils it is necessary to initialise OpenCLBasicKernels (the class is not initialized automatically right now)


class CudaBasicUtils 
{
public:
	static int RoundUp(int base, int num);
	//works only for 2^ numbers
	static int Log2Base(int num);
	//get the first x^2 that can accomodate a given number
	static int FirstPow2(int num);
};

#endif