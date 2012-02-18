#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_

template<class T>
class FromInt
{
public:
	inline static __device__ T op(int x) { return x; }
};

template<>
class FromInt<float2>
{
public:
	inline static __device__ float2 op(int x) { return make_float2(x, x); }
};

template<class T>
class ToVolatile
{
public:
	inline static __device__ void op(volatile T& vOut, T& vIn) { vOut = vIn; }
};

template<>
class ToVolatile<float2> {
public: inline static __device__ void op(volatile float2& vOut, float2& vIn) { 			
			vOut.x = vIn.x;
			vOut.y = vIn.y;			
		}
};

template<class T>
class FromVolatile
{
public:
	inline static __device__ void op(T& vOut, volatile T& vIn) { vOut = vIn; }
};

template<>
class FromVolatile<float2> {
public: inline static __device__ void op(float2& vOut, volatile float2& vIn) { 			
			vOut.x = vIn.x;
			vOut.y = vIn.y;			
		}
};

inline __device__ void operator+=(volatile float2& v0, volatile float2& v1)
{
	v0.x += v1.x;
	v0.y += v1.y;
}


#endif