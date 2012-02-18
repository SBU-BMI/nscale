// Host code
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "highgui.h"
#include "cv.h"
#include <math.h>
#include "label.h"
#include "CUDAFunctionsBase.h"
#include "ccl.h"
#include "cvlabeling_imagelab.h"
//#include "sbla.h"
#include <sys/time.h>

#define cutilSafeCall(x) x


#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

class CPerformanceCounter 
{
	long long m_i64Frequency,m_i64Start,m_i64Stop;
	bool m_bRunning;

	double GetElapsed (double dMultiplier) 
	{
		double dElapsed;
		long long i64Stop(m_i64Stop);
		if (m_bRunning) {
	struct timeval ts;
	gettimeofday(&ts, NULL);
	i64Stop = (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL; 

//			QueryPerformanceCounter ((long long *)(&i64Stop));
		}
		dElapsed = (double)i64Stop;
		dElapsed -= m_i64Start;
		dElapsed *= dMultiplier;
		dElapsed /= m_i64Frequency;
		return dElapsed;
	}

public:
	CPerformanceCounter () 
	{
		m_i64Frequency = 1000;
		//QueryPerformanceFrequency ((long long *)(&m_i64Frequency));
		Start();
	}

	void Start () 
	{

	struct timeval ts;
	gettimeofday(&ts, NULL);
	m_i64Start = (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL; 
//		QueryPerformanceCounter ((long long *)(&m_i64Start));
		m_i64Stop = m_i64Start;
		m_bRunning = true;
	}

	void Stop () 
	{
	struct timeval ts;
	gettimeofday(&ts, NULL);
	m_i64Stop = (ts.tv_sec*1000000 + (ts.tv_usec))/1000LL; 

//		QueryPerformanceCounter ((long long *)(&m_i64Stop));
		m_bRunning = false;
	}

	double GetElapsedSeconds () 
	{
		return GetElapsed(1);
	}
	double GetElapsedMilliSeconds () 
	{
		return GetElapsed(1000);
	}	
};

float gpu2time;

void boundCheck(IplImage *img){
	char *data=img->imageData;
	for (int i=0;i<img->height;i++){
		for (int j=0;j<img->width;j++){
			if ( (i==img->height-1)||(j==img->width-1)||(i==0)||(j==0)){//保证边界为0
				//data[i*width+j]=255;
				data[i*img->widthStep+j]=0;
			}
		}
	}
}

bool checkLabels(ushort* ref0, ushort* ref1, ushort width, uint height, ushort tlabels)
{
	ushort *tt;
	bool flag = 0;
	uint cnt0 = 0, cnt1 = 0;
	tt = (ushort*)malloc(tlabels * sizeof(ushort));
	for(int i = 0; i < tlabels; i++)
		tt[i] = LBMAX;
	
	for(int j = 0; j < height; j++)
		for(int i = 0; i < width; i++){
			ushort r0 = ref0[j*width+i];
			ushort r1 = ref1[j*width+i];
			if(r0 != LBMAX){
				if(r0 >= tlabels){
					printf("@(%d, %d), ref0 %d too large.\n", i, j, r0);
					flag = 1;
					cnt1++;
					continue;
				}
				ushort r2 = tt[r0];
				if(r1 == LBMAX){
					printf("@(%d, %d), should not be invalid.\n", i, j);
					flag = 1;
					cnt1++;
					continue;
				}
				else if(r2 == LBMAX){
					tt[r0] = r1;
					r2 = r1;
				}
				if(r1 != r2){
					printf("@(%d, %d), should be %d, found %d\n", i, j, r2, r1);
					flag = 1;
					cnt1++;
				}
				else
					cnt0++;
			}
		}

		printf("found correct blocks: %d, incorrect blocks: %d\n", cnt0, cnt1);
	free(tt);
	return flag;
}

int main(int argc, char** argv)
{
	bool srcbin = 0;
	bool invbk = 0;
	if(argc < 3){
		printf("Not enough args!\narg1: target image\narg2: source image\n");
//		getchar();
		return 1;
	}

	IplImage* srcimg= 0, *alg2dst = 0, *srcimgb= 0;
	srcimg= cvLoadImage(argv[2], -1);
	if (!srcimg)
	{
		printf("src img %s load failed!\n", argv[2]);
//		getchar();
		return 1;
	}
	
	int bn = 8;
	int nwidth = 512;
	if(srcimg->width > 512){
		nwidth = 1024;
		bn = 6;
	}
	if(srcimg->width > 1024){
		nwidth = 2048;
		bn = 3;
	}
	if(srcimg->width > 2048){
		nwidth = 4096;
		bn = 1;
	}
	if(srcimg->width > 4096){
		printf("warning, image too wide, max support 2048. image is truncated.\n");
//		return 1;
	}

	int devCount;
	int smCnt = 0;
    cudaGetDeviceCount(&devCount);
 
    // Iterate through devices
	int devChosen = 0;
    for (int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
		if(devProp.major >= 2){//only one device supported
			smCnt = max(smCnt, devProp.multiProcessorCount);
			if(devProp.multiProcessorCount == smCnt)
				devChosen = i;
		}
    }
	
	if(smCnt == 0){
		printf("Error, no device with cap 2.x found. Only cpu alg will be run.\n");
//		getchar();
//		return 1;
	}
	
	if(smCnt != 0){
		cudaSetDevice(devChosen);
		bn = bn * smCnt;
	}

	int nheight = (cvGetSize(srcimg).height-2) / (2*bn);
	if((nheight*2*bn+2) < cvGetSize(srcimg).height)
		nheight++;
	nheight = nheight*2*bn+2;

	if(smCnt != 0)
		printf("gpu ccl for image width 512, 1024, 2048.\nchoosing device %d, width %d, height %d, blocks %d\n", devChosen, nwidth, nheight, bn);

	srcimgb= cvCreateImage(cvSize(nwidth, cvGetSize(srcimg).height),IPL_DEPTH_8U,1);
	cvSetImageROI(srcimg, cvRect(0, 0, min(cvGetSize(srcimg).width, nwidth), cvGetSize(srcimg).height));
	cvSetImageROI(srcimgb, cvRect(0, 0, min(cvGetSize(srcimg).width, nwidth), cvGetSize(srcimg).height));
	cvThreshold(srcimg, srcimgb, 0.0, 1.0, CV_THRESH_BINARY);
	cvResetImageROI(srcimgb);
	boundCheck(srcimgb);
	
	float elapsedMilliSeconds1 = 0.0f;
	float elapsedMilliSeconds3 = 0.0f;

	
	IplImage *src2(0),*dst2(0);
	int iNumLabels;
	float elapsedMilliSeconds2;
	{
		CPerformanceCounter perf;
		src2 = cvCreateImage( cvGetSize(srcimgb), IPL_DEPTH_8U, 1 );
		cvCopyImage(srcimgb,src2);
		dst2 = cvCreateImage( cvGetSize(srcimgb), IPL_DEPTH_32S, 1 );
		perf.Start();
		cvLabelingImageLab(src2, dst2, 1, &iNumLabels);
		elapsedMilliSeconds2 = (float)perf.GetElapsedMilliSeconds();
		printf("cpu BBDT used %f ms, total labels %u\n", elapsedMilliSeconds2, iNumLabels);
		cvSaveImage("bbdt.bmp", dst2);
//		cvReleaseImage(&src2);
//		cvReleaseImage(&dst2);
	}

	if(smCnt != 0){

		CPerformanceCounter perf;

	perf.Start();
	CudaBuffer srcBuf;
	srcBuf.Create2D(nwidth, cvGetSize(srcimgb).height);
	elapsedMilliSeconds3 = (float)perf.GetElapsedMilliSeconds();
	printf("gpu gen allocate img used %f ms\n", elapsedMilliSeconds3);
	perf.Start();
	srcBuf.SetZeroData();
	elapsedMilliSeconds3 = (float)perf.GetElapsedMilliSeconds();
	printf("gpu gen initialize img used %f ms\n", elapsedMilliSeconds3);
	perf.Start();
	srcBuf.CopyFrom(srcimgb->imageData, srcimgb->widthStep, nwidth, cvGetSize(srcimgb).height);
	elapsedMilliSeconds3 = (float)perf.GetElapsedMilliSeconds();
	printf("gpu gen upload img used %f ms\n", elapsedMilliSeconds3);


	float elapsedTimeInMs = 0.0f;
    
	//-------------------gpu part----------------------------
	alg2dst= cvCreateImage(cvSize(nwidth*4, cvGetSize(srcimgb).height),IPL_DEPTH_8U,1);

	CCLBase* m_ccl;
	m_ccl = new CCL();	
	{
		perf.Start();
	m_ccl->FindRegions(nwidth, cvGetSize(srcimgb).height, &srcBuf, 4);
		elapsedMilliSeconds1 = (float)perf.GetElapsedMilliSeconds();
		perf.Start();
	m_ccl->GetConnectedRegionsBuffer()->CopyToHost(alg2dst->imageData, alg2dst->widthStep, nwidth*4, cvGetSize(srcimgb).height);
		elapsedMilliSeconds3 = (float)perf.GetElapsedMilliSeconds();

		printf("gpu gem used %f ms\n", elapsedMilliSeconds1);
		printf("gpu gen download used %f ms\n", elapsedMilliSeconds3);
		
	}
	delete m_ccl;
	cvSaveImage("alg2.bmp", alg2dst);

	cvReleaseImage(&alg2dst);
//	}
	//cvWaitKey(0);
	
//	if(smCnt != 0){
/*		ushort *gpures, *cpures;
		uint sz = nwidth * (cvGetSize(srcimgb).height/2);
		gpures = (ushort*)malloc(sz);
		cpures = (ushort*)malloc(sz);
		dstBuf.CopyToHost(gpures, nwidth, nwidth, (cvGetSize(srcimgb).height/2));

		for(int j = 0; j < (cvGetSize(srcimgb).height/2); j++)
			for(int i = 0; i < (nwidth/2); i++){
				uint* cpup;
				ushort res = LBMAX;
				uint y = j*2, x = i*2;
				cpup = (uint*)(dst2->imageData + y*dst2->widthStep);
//				if(y < cvGetSize(srcimgb).height){
					if(cpup[x] != 0)
						res = cpup[x]-1;
					if(cpup[x+1] != 0)
						res = cpup[x+1]-1;
//				}
				y++;
				cpup = (uint*)(dst2->imageData + y*dst2->widthStep);
//				if(y < cvGetSize(srcimgb).height){
					if(cpup[x] != 0)
						res = cpup[x]-1;
					if(cpup[x+1] != 0)
						res = cpup[x+1]-1;
//				}
				cpures[i + j*(nwidth/2)] = res;
			}
		
		if(iNumLabels > LBMAX)
			printf("too much cc, compare abort.\n");
		else{
			//create a error
			//cpures[5] = 12;
			//cpures[15] = 18;
			printf("Checking correctness of gpu alg1\nChecking gpu ref by cpu.\n");
			checkLabels(cpures, gpures, nwidth/2, cvGetSize(srcimgb).height/2, iNumLabels);

			printf("Checking cpu ref by gpu.\n");
			checkLabels(gpures, cpures, nwidth/2, cvGetSize(srcimgb).height/2, tlabel);
		}

		free(gpures);
		free(cpures);
		printf("speedup is %f, %f, %f\n", gpu2time/elapsedTimeInMs, elapsedMilliSeconds1/elapsedTimeInMs, elapsedMilliSeconds2/elapsedTimeInMs);
*/	}

	cvReleaseImage(&srcimgb);
//	cvReleaseImage(&srcimgb2);
	cvReleaseImage(&dst2);
	cvReleaseImage(&src2);

    cutilSafeCall( cudaThreadExit() );
	return 0;

}

