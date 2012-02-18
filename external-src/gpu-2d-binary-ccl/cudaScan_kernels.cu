#include "CUDAFunctionsBase.h"
#include "cudaUtil.h"

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//do not change this unless you modify the code below !!!!!!!!!!
#define NUM_THREADS 512

template<class T,  int scanSize>
inline __device__ T warpScanInclusive(volatile T iData, volatile T *l_Data){
        uint pos = 2 * threadIdx.x - (threadIdx.x & (scanSize-1));
        l_Data[pos] = iData;       		
		l_Data[pos + scanSize] = FromInt<T>::op(0);
		
		if(1<scanSize) l_Data[pos] += l_Data[pos +  1];
        if(2<scanSize) l_Data[pos] += l_Data[pos +  2];
        if(4<scanSize) l_Data[pos] += l_Data[pos +  4];
        if(8<scanSize) l_Data[pos] += l_Data[pos +  8];
        if(16<scanSize) l_Data[pos] += l_Data[pos + 16];
       
        return l_Data[pos];
    }
 
template<class T, int scanSize>
inline __device__ T warpScanExclusive(T idata, volatile T *l_Data){       
        return warpScanInclusive<T,  scanSize>(idata, l_Data) - idata;
    }
    
template<class T>   
inline __device__ T scanInclusive(T idata, T *l_Data){	
        //Bottom-level inclusive warp scan
        T warpResult = warpScanInclusive<T,32>(idata, l_Data);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because l_Data is being overwritten)
        __syncthreads();
        if( (threadIdx.x & (WARP_SIZE - 1)) == 0 )
            l_Data[threadIdx.x >> LOG2_WARP_SIZE] = warpResult;

        //wait for warp scans to complete
         __syncthreads();
        //right now works only for NUM_THREADS = 512
        if( threadIdx.x < (16) ){
			//only the first warp goes here
            //grab top warp elements
            T val = l_Data[threadIdx.x];           
            //calculate exclsive scan and write back to shared memory
            l_Data[threadIdx.x] = warpScanExclusive<T, 16>(val, l_Data);
        }

        //return updated warp scans with exclusive scan results
         __syncthreads();
       // return  l_Data[get_local_id(0) >> LOG2_WARP_SIZE]; 
        return warpResult + l_Data[threadIdx.x >> LOG2_WARP_SIZE];       
    }

template<class T>
__global__ void scanInclusiveSingleBlock( T* d_in,  T* d_out, int elements)
{
	T val = FromInt<T>::op(0);
	extern __shared__ T l_Data[];
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < elements) val = d_in[id];
	val = scanInclusive<T>(val, l_Data);
	if(id < elements) d_out[id] = val;
}

template<class T>
__global__ void scanInclusiveMultiBlock( T* d_in,
										 T* d_out,
										 T* d_blockScan,										
										 int elements)
{
	T val = FromInt<T>::op(0);
	extern __shared__ T l_Data[];
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < elements) val = d_in[id];
	val = scanInclusive<T>(val, l_Data);
	if(id < elements) d_out[id] = val;
	//write results from the first thread of each block
	if(threadIdx.x == 0 && id > 0) {
		//results are shifted  - exclusive scan
		d_blockScan[blockIdx.x-1] = val;
	}
}

template<class T>
__global__ void uniformUpdate(T* d_inOut,
							  T* d_blockScan)
{
	//index is always in range (the last block is not updated
	__shared__ T buf[1];
	int id = threadIdx.x + blockIdx.x*blockDim.x;
    T data = d_inOut[id];

    if(threadIdx.x == 0)
        buf[0] = d_blockScan[blockIdx.x];

    __syncthreads();
    data += buf[0];
    d_inOut[id] = data;
}