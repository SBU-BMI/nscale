#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;

int main (int argc, char* argv[])
{
    try
    {
	if(cv::gpu::CudaMem::canMapHostMemory()){
		cout<< "canMapHostMem"<<endl;
	}
        cv::gpu::GpuMat dst, src = cv::gpu::GpuMat(cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE));

        cv::gpu::threshold(src, dst, 200.0, 255.0, CV_THRESH_BINARY);

        cv::imshow("Result", (cv::Mat)dst);
	cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}

