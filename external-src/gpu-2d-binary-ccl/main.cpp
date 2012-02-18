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
		printf("Not enough args!\narg1: target image\narg2: source image\narg3: do source image adaptive threshold or not\narg4: invert back ground or not\n");
//		getchar();
		return 1;
	}
	if(argc >= 4){
		if(!strcmp(argv[3], "1"))
			srcbin = 1;
	}
	if(argc >= 5){
		if(!strcmp(argv[4], "1"))
			invbk = 1;
	}

	IplImage* srcimg= 0, *srcimgb= 0, *srcimgb2 = 0, *bimg = 0, *b2img = 0,*bugimg = 0, *alg2dst = 0;
	srcimg= cvLoadImage(argv[2], 1);
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
	srcimgb2= cvCreateImage(cvSize(nwidth, cvGetSize(srcimg).height),IPL_DEPTH_8U,1);
	cvSetImageROI(srcimg, cvRect(0, 0, min(cvGetSize(srcimg).width, nwidth), cvGetSize(srcimg).height));
	cvSetImageROI(srcimgb2, cvRect(0, 0, min(cvGetSize(srcimg).width, nwidth), cvGetSize(srcimg).height));
	cvSet(srcimgb2, cvScalar(0,0,0));
	cvCvtColor(srcimg, srcimgb2, CV_BGRA2GRAY);
	cvResetImageROI(srcimgb2);
	cvReleaseImage(&srcimg);
	if(srcbin)
		cvAdaptiveThreshold(srcimgb2, srcimgb, 1.0, CV_ADAPTIVE_THRESH_MEAN_C, invbk ? CV_THRESH_BINARY_INV :  CV_THRESH_BINARY);
	else
		cvThreshold(srcimgb2, srcimgb, 0.0, 1.0, invbk ? CV_THRESH_BINARY_INV :  CV_THRESH_BINARY);
	boundCheck(srcimgb);

	cvScale(srcimgb, srcimgb2, 255);
	cvSaveImage("bsrc.bmp", srcimgb2);
	cvSet(srcimgb2, cvScalar(0,0,0));
	
	float elapsedMilliSeconds1 = 0.0f;
//	{
//		LABELDATATYPE *data=(LABELDATATYPE *)malloc(srcimgb->width * srcimgb->height * sizeof(LABELDATATYPE));
//	
//		for(int j = 0; j<srcimgb->height; j++)
//			for(int i = 0; i<srcimgb->width; i++)
//				data[i + j*srcimgb->width] = (srcimgb->imageData[i + j*srcimgb->widthStep]) ? 1 : 0;
//
//		int iNumLabels;
//		CPerformanceCounter perf;
//		perf.Start();
//    	iNumLabels = LabelSBLA(data, srcimgb->width, srcimgb->height);
//		elapsedMilliSeconds1 = (float)perf.GetElapsedMilliSeconds();
//		printf("cpu SBLA used %f ms, total labels %u\n", elapsedMilliSeconds1, iNumLabels);
//		free(data);
//	}
	
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

	bugimg = cvCreateImage(cvSize(nwidth, 9*bn),IPL_DEPTH_8U,1);
	bimg = cvCreateImage(cvSize(nwidth, 2*bn),IPL_DEPTH_8U,1);
	b2img = cvCreateImage(cvSize(nwidth, 2*bn),IPL_DEPTH_8U,1);

//    cvNamedWindow("src",CV_WINDOW_AUTOSIZE);
//	cvShowImage("src",srcimg);


	CudaBuffer srcBuf, dstBuf, dstBuf2, bBuf, b2Buf, errBuf, glabel;
	srcBuf.Create2D(nwidth, nheight);
	dstBuf.Create2D(nwidth, (nheight-2)/2);//(nheight-2)/2
	dstBuf2.Create2D(nwidth,(nheight-2)/2);//(nheight-2)/2
	glabel.Create2D(4, 1);
	errBuf.Create2D(nwidth, 9*bn);
	bBuf.Create2D(nwidth, 2 * bn);
	b2Buf.Create2D(nwidth, 2 * bn);

	srcBuf.SetZeroData();
	srcBuf.CopyFrom(srcimgb->imageData, srcimgb->widthStep, nwidth, cvGetSize(srcimgb).height);

	float elapsedTimeInMs = 0.0f;
    
	//-------------------gpu part----------------------------
    cudaEvent_t start, stop;
    cutilSafeCall  ( cudaEventCreate( &start ) );
    cutilSafeCall  ( cudaEventCreate( &stop ) );
    cutilSafeCall( cudaEventRecord( start, 0 ) );  
	
	if(nwidth == 512)
		label_512(&dstBuf, &dstBuf2, &srcBuf, &bBuf, &b2Buf, &glabel, nheight, bn, &errBuf);
	else if(nwidth == 1024)
		label_1024(&dstBuf, &dstBuf2, &srcBuf, &bBuf, &b2Buf, &glabel, nheight, bn, &errBuf);
	else if(nwidth == 2048)
		label_2048(&dstBuf, &dstBuf2, &srcBuf, &bBuf, &b2Buf, &glabel, nheight, bn, &errBuf);

    cutilSafeCall( cudaEventRecord( stop, 0 ) );
//	cutilCheckMsg("kernel launch failure");
	cudaEventSynchronize(stop);
	cutilSafeCall( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
	uint tlabel = 0;

	cudaMemcpy(&tlabel, glabel.GetData(), 4, cudaMemcpyDeviceToHost);
	printf("gpu alg 1 used %f ms, total labels %u\n", elapsedTimeInMs, tlabel);

	dstBuf.CopyToHost(srcimgb->imageData, srcimgb->widthStep, nwidth, (nheight-2)/2);
	dstBuf2.CopyToHost(srcimgb2->imageData, srcimgb->widthStep, nwidth, (nheight-2)/2);
	errBuf.CopyToHost(bugimg->imageData, bugimg->widthStep, nwidth, 9*bn);
	bBuf.CopyToHost(bimg->imageData, bimg->widthStep, nwidth, 2*bn);
	b2Buf.CopyToHost(b2img->imageData, bimg->widthStep, nwidth, 2*bn);

//	cvNamedWindow("gpu",CV_WINDOW_AUTOSIZE);
//	cvShowImage("gpu",srcimgb);
	cvSaveImage(argv[1], srcimgb);
	cvSaveImage("gpu2.bmp", srcimgb2);
	cvSaveImage("bug.bmp", bugimg);
	cvSaveImage("b.bmp", bimg);
	cvSaveImage("b2.bmp", b2img);

	alg2dst= cvCreateImage(cvSize(nwidth*4, cvGetSize(srcimgb).height),IPL_DEPTH_8U,1);
	CCLBase* m_ccl;
	m_ccl = new CCL();	

	m_ccl->FindRegions(nwidth, cvGetSize(srcimgb).height, &srcBuf, 8);
	m_ccl->GetConnectedRegionsBuffer()->CopyToHost(alg2dst->imageData, alg2dst->widthStep, nwidth*4, cvGetSize(srcimgb).height);
	delete m_ccl;
	cvSaveImage("alg2.bmp", alg2dst);

	cvReleaseImage(&bugimg);
	cvReleaseImage(&bimg);
	cvReleaseImage(&b2img);
	cvReleaseImage(&alg2dst);
//	}
	//cvWaitKey(0);
	
//	if(smCnt != 0){
		ushort *gpures, *cpures;
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
	}

	cvReleaseImage(&srcimgb);
	cvReleaseImage(&srcimgb2);
	cvReleaseImage(&dst2);
	cvReleaseImage(&src2);

    cutilSafeCall( cudaThreadExit() );
	return 0;

}

