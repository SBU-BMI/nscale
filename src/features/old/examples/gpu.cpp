#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include <sys/time.h>

using namespace std;
using namespace cv;

int main (int argc, char* argv[])
{
	try
	{
		if(cv::gpu::CudaMem::canMapHostMemory()){
			cout<< "canMapHostMem"<<endl;
		}


		IplImage *inputImage = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

		Mat imgMat(inputImage);
		Mat dest(inputImage);
		// ProcessTime example
		struct timeval startTime;
		struct timeval endTime;
		// get the current time
		// - NULL because we don't care about time zone
		cv::gpu::GpuMat dst, src = cv::gpu::GpuMat(inputImage);
		cv::gpu::GpuMat kernel;

		gettimeofday(&startTime, NULL);




		cv::gpu::morphologyEx(src, dst, MORPH_GRADIENT, kernel, Point(-1,-1), 1);

		int nonZeroGPU = cv::gpu::countNonZero(dst);
		cout << "NonZeroGradGPU = "<< nonZeroGPU <<endl;


		Mat kernelCPU;
		morphologyEx(imgMat, dest, MORPH_GRADIENT, kernelCPU, Point(-1,-1), 1);
		int nonZeroCPU = countNonZero(dest);
		cout << "NonZeroGradCPU = "<< nonZeroCPU<<endl;

		// This is a temporary structure required by the MorphologyEx operation we'll perform
		IplImage* tempImg = cvCreateImage( cvSize(inputImage->width, inputImage->height), IPL_DEPTH_8U, 1);

		// This is a temporary structure required by the MorphologyEx operation we'll perform
		IplImage* magImg = cvCreateImage( cvSize(inputImage->width, inputImage->height), IPL_DEPTH_8U, 1);

		cvMorphologyEx(inputImage, magImg, tempImg, NULL, CV_MOP_GRADIENT);
		int nonZeroCPU2 = cvCountNonZero(magImg);
		cout << "NonZeroGradCPU2 = "<< nonZeroCPU2<<endl;

/*		Sobel(imgMat, dest, CV_8U, 1, 1, 7);

		int pixelsSobel = countNonZero(dest);
		cout << " Sobel area = "<< pixelsSobel <<endl;*/

		gettimeofday(&endTime, NULL);

			// calculate time in microseconds
			double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
			double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
			printf("Total Time Taken: %lf\n", tE - tS);
/*		// ProcessTime example
		struct timeval startTime;
		struct timeval endTime;
		// get the current time
		// - NULL because we don't care about time zone
		gettimeofday(&startTime, NULL);

		IplImage *sobelResCPU = cvCreateImage( cvSize(inputImage->width, inputImage->height), IPL_DEPTH_16S, 1);
		IplImage *cannyResCPU = cvCreateImage( cvSize(inputImage->width, inputImage->height), IPL_DEPTH_8U, 1);

//		cvCopy(inputImage, sobelResCPU, NULL);
		cvSobel(inputImage, sobelResCPU, 1, 1, 7);

//		cvCopy(inputImage, cannyResCPU, NULL);
//		cvCanny(cannyResCPU, cannyResCPU, 10, 230, 7);

//		int cannyArea = cvCountNonZero(cannyResCPU);
		int sobelArea = cvCountNonZero(sobelResCPU);

		cout<< " sobelArea = "<< sobelArea <<endl;
	//	cout<< " cannyArea = "<< cannyArea <<endl;

		gettimeofday(&endTime, NULL);

		// calculate time in microseconds
		double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
		double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
		printf("Total Time Taken: %lf\n", tE - tS);

//		cv::gpu::GpuMat dst, src = cv::gpu::GpuMat(inputImage);
	//	cv::gpu::Sobel(src, dst, CV_16S, 1, 1);
//		cv::gpu::threshold(src, dst, 200.0, 255.0, CV_THRESH_BINARY);

//		cv::gpu::GpuMat *histGPU = new cv::gpu::GpuMat(1, 256,  CV_32S);

	//	cv::gpu::histEven(src, *histGPU, 257, 0, 256);

//		cv::Mat histCPU= (cv::Mat)*histGPU;

//		cout<<"M.rows="<< histCPU.rows << " M.cols=" << histCPU.cols<<endl;


//		int *M0 = histCPU.ptr<int>(0);
//		for(int i =0; i < 256; i++){
//			cout<< "M[0]["<< i<<"] = "<< M0[i] <<endl;
//		}

	//	cout << "Sum = "<< gpu::countNonZero(*ROISubImage)<<endl;

//		cv::Mat ResGPUInCPU = (cv::Mat)dst;
//		cv::imshow("SobelGPU", (cv::Mat)dst);
//		cvShowImage("SobelCPU", sobelResCPU);
//		cvShowImage("SobelCPU", cannyResCPU);
//		cv::waitKey();

//		if(sobelResCPU==ResGPUInCPU)
//			cout << "equal"<<endl;*/
	}
	catch(const cv::Exception& ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	return 0;
}

