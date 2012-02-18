#include "cudaBasicUtils.h"

int CudaBasicUtils::RoundUp(int base, int num)
{
	int r = num % base;
	if(r == 0) return num;
	return num+base-r;
}

int CudaBasicUtils::Log2Base(int num)
{
	if(num == 0) return 0;
	int ret = 0;
	while(num > 0) {
		num >>= 1;
		ret++;
	}
	return ret-1;
}

int CudaBasicUtils::FirstPow2(int num)
{
	if(num == 0) return 0;
	int l2Base = Log2Base(num);	
	int ret = (1 << l2Base);
	if(ret == num) return ret;	
	ret = (ret<<1);
	return ret;
}