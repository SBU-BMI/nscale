
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;

int main (int argc, char* argv[]){
	try{
		cv::gpu::GpuMat dst, src = cv::gpu::GpuMat(cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE));

		cv::gpu::threshold(src, dst, 200.0, 255.0, CV_THRESH_BINARY);
		cv::gpu::GpuMat *histGPU = new cv::gpu::GpuMat(1, 256, CV_32S);

		cv::gpu::histEven(src, *histGPU, 256, 0, 255);

		cv::Mat histCPU= (cv::Mat)*histGPU;

		cout<<"M.rows="<< histCPU.rows << " M.cols=" << histCPU.cols<<endl;


		float *M0 = histCPU.ptr<float>(0);
		for(int i =0; i < 256; i++){
			cout<< "M[0]["<< i<<"] = "<< M0[i] <<endl;
		}
		cv::imshow("Result", (cv::Mat)dst);
		cv::waitKey();
	}
	catch(const cv::Exception& ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}
	return 0;
}
