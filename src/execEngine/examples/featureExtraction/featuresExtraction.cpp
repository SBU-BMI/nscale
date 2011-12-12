

#include <stdio.h>
#include <sys/time.h>
#include "TaskFeature.h"
#include "ExecutionEngine.h"

void * hostMemAlloc(int dataSize);
void freeMem(void *data_ptr);
void warmUp1(int device);

// ProcessTime example
struct timeval startTime;
struct timeval endTime;

void beginTimer(){
	gettimeofday(&startTime, NULL);
}

void printElapsedTime(){
	gettimeofday(&endTime, NULL);
	// calculate time in microseconds
	double tS = startTime.tv_sec*1000000 + (startTime.tv_usec);
	double tE = endTime.tv_sec*1000000  + (endTime.tv_usec);
	printf(" %lf\n", (tE - tS)/1000000.0);
}


int main(int argc, char **argv){
	vector<TaskFeature *> auxTasks;
	if(argc != 9){
		printf("Usage: <imgMask1> <imgGray1> <imgMask2> <imgGray2> <PercentFirstImg(0-100)> <cpuThreads> <gpuThreads> <schedType=1(FCFS),2(PRIORITY)>");
		exit(1);
	}


	int percentFirstIMG = atoi(argv[5]);
	int numCPUThreads = atoi(argv[6]);
	int numGPUThreads = atoi(argv[7]);
	int schedType = atoi(argv[8]);

	int totalImg = 40;
	int nFirstImg = (totalImg * percentFirstIMG )/100;
	int nSecondImg = totalImg - nFirstImg;


/////// BEGIN LOAD IMAGES /////////////////

	// read image in mask image that is expected to be binary
	IplImage *originalImageMask1 = cvLoadImage(argv[1], -1 );
	if(originalImageMask1 == NULL){
		cout << "Could not load image: "<< argv[1] <<endl;
		exit(1);
	}else{
		if(originalImageMask1->nChannels != 1){
			cout << "Error: Mask image should have only one channel"<<endl;
			exit(1);
		}
	}

	// read actual image
	IplImage *originalImage1 = cvLoadImage(argv[2], -1 );

	if(originalImage1 == NULL){
		cout << "Cound not open input image:"<< argv[2] <<endl;
		exit(1);
	}else{
		if(originalImage1->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel"<<endl;
			cvReleaseImage(&originalImage1);
			exit(1);
		}
	}

	IplImage *originalImageMask2 = cvLoadImage(argv[3], -1 );
	if(originalImageMask2 == NULL){
		cout << "Could not load image: "<< argv[3] <<endl;
		exit(1);
	}else{
		if(originalImageMask2->nChannels != 1){
			cout << "Error: Mask image should have only one channel"<<endl;
			exit(1);
		}
	}

	// read actual image
	IplImage *originalImage2 = cvLoadImage(argv[4], -1 );

	if(originalImage2 == NULL){
		cout << "Cound not open input image:"<< argv[4] <<endl;
		exit(1);
	}else{
		if(originalImage2->nChannels != 1){
			cout << "Error: input image should be grayscale with one channel"<<endl;
			cvReleaseImage(&originalImage2);
			exit(1);
		}
	}
	/////////END LOAD IMAGES /////////////


	ExecutionEngine execEngine(numCPUThreads, numGPUThreads,schedType);

	beginTimer();
	for(int i =0; i < nFirstImg; i++){
		TaskFeature *tf = new TaskFeature(originalImageMask1, originalImage1);
		tf->setSpeedup(Constant::GPU, 2.0);
		auxTasks.push_back(tf);
	}

	for(int i =0; i < nSecondImg; i++){
		TaskFeature *tf = new TaskFeature(originalImageMask2, originalImage2);
		tf->setSpeedup(Constant::GPU, 30.0);
		auxTasks.push_back(tf);
	}

	random_shuffle(auxTasks.begin(), auxTasks.end());
	random_shuffle(auxTasks.begin(), auxTasks.end());

	for(int i = 0; i < auxTasks.size(); i++){
		execEngine.insertTask(auxTasks[i]);
	}

	sleep(2);
	printf("Init-time:");
	printElapsedTime();
	

	beginTimer();
	execEngine.startupExecution();
	execEngine.endExecution();
	printf("FeatureTime: ");
	printElapsedTime();

	
	return 0;
}


