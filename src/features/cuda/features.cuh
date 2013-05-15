/*** Written by Salil Deosthale 11/30/2012 ***/

#include "cutil.h"
#include <cuda_runtime.h>
#include "opencv2/gpu/devmem2d.hpp"

namespace nscale
{
	namespace gpu {
		using namespace cv::gpu;
		void AreaCaller(const int* boundingBoxInfo , const int compCount ,const cv::gpu::PtrStep_<int> labeledMask, int *areaRes, cudaStream_t stream);
		void PerimeterCaller(const int* boundingBoxInfo , const int compCount , const cv::gpu::PtrStep_ <int> labeledMask , float *perimeterRes, cudaStream_t stream);
		void EllipseCaller(const int* boundingBoxInfo , const int compCount , const cv::gpu::PtrStep_ <int> labeledMask , int *areaRes , float *majorAxis , float *minorAxis , float *ecc, cudaStream_t stream);
		void ExtentRatioCaller(const int* boundingBoxInfo , const int compCount , const int *areaRes , float* extent_ratio , cudaStream_t stream);
		void CircularityCaller(const int compCount , const int *areaRes , const float *perimeterRes , float *circ, cudaStream_t stream);
		void BigFeaturesCaller(const int* boundingBoxInfo , const int compCount , const cv::gpu::PtrStep_<int> labeledMask , int* areaRes , float* perimeterRes , float* majorAxis , float* minorAxis , float* ecc, cudaStream_t stream);
		void SmallFeaturesCaller(const int *boundingBoxInfo , const int compCount , const int *areaRes , const float *perimeterRes , float *extent_ratio , float* circ , cudaStream_t stream);
	}
}
