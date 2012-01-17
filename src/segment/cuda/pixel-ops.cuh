/*
 * kernel for utility functions
  */
#include "internal_shared.hpp"


namespace nscale { 
namespace gpu {

template <typename T>
void invertUIntCaller(int rows, int cols, int cn, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);
template <typename T>
void invertIntCaller(int rows, int cols, int cn, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);
template <typename T>
void invertFloatCaller(int rows, int cols, int cn, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);

template <typename T>
void thresholdCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<unsigned char> result, T lower, bool lower_inclusive, T upper, bool up_inclusive, cudaStream_t stream);

template <typename T>
void divideCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1, const cv::gpu::PtrStep_<T> img2,
 cv::gpu::PtrStep_<T> result, cudaStream_t stream);

template <typename T>
void maskCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1, const cv::gpu::PtrStep_<T> img2,
 cv::gpu::PtrStep_<T> result, T background, cudaStream_t stream);
 
template <typename T>
void modCaller(int rows, int cols, const cv::gpu::PtrStep_<T> img1,
 cv::gpu::PtrStep_<T> result, T mod, cudaStream_t stream);

void bgr2grayCaller(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> img1, 
	cv::gpu::PtrStep_<unsigned char> result, cudaStream_t stream);

void convLoop1(int rows, int cols, int cn, const cv::gpu::PtrStep_<unsigned char> img1, 
	cv::gpu::PtrStep_<double> result, cudaStream_t stream);
void convLoop2(int rows, int cols, int cn_channels, const cv::gpu::PtrStep_<double> g_cn,
 	int dn_channels, cv::gpu::PtrStep_<double> g_dn, cv::gpu::PtrStep_<double> g_Q, int Q_rows, bool BGR2RGB, cudaStream_t stream);

void convLoop3(int rows, int cols, int cn_channels, const cv::gpu::PtrStep_<double> g_cn, cv::gpu::PtrStep_<unsigned char> g_E, cv::gpu::PtrStep_<unsigned char> g_H, cudaStream_t stream);

void convertIntToChar(int rows, int cols, int* input, unsigned char *result, cudaStream_t stream);
void convertIntToCharAndRemoveBorder(int rows, int cols, int top, int bottom, int left, int right, int* input, unsigned char *result, cudaStream_t stream);

}}
