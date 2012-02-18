#include <stdio.h>
#include <cuda.h>

#include "pgm.h"

#include "ccl_gold.cpp"
#include "ccl_uf.cu"
#include "ccl_lequiv.cu"
#include "ccl_naive_prop.cu"


int number_cc(int* label, int w, int h) {
    bool* mask = (bool*)malloc(w*h*sizeof(bool));
    for (int i=0; i<w*h; i++) {
        mask[i] = (label[i] == i);
    }
    int count = 0;
    for (int i=0; i<w*h; i++) {
        if (mask[i]) {count++;}
    }
    free(mask);
    return count;
}


#define TRIES 1
float et_v[TRIES];

#define START_TIME cudaEventRecord(start,0)
#define STOP_TIME  cudaEventRecord(stop,0 ); \
                   cudaEventSynchronize(stop); \
                   cudaEventElapsedTime( &et, start, stop )

int w,h;
unsigned char* img;
int *label, *label_gold;

void VERIFY() {
    for (int k=0;k<w*h;k++) { 
        if (label[k] != label_gold[k]) { 
            printf("WRONG!\n"); 
            break; 
        } 
    } 
    for (int k=0;k<w*h;k++) { 
        label[k] = -1; 
    } 
}

float MIN_ET() {
    float et = et_v[0]; 
    for (int t=0; t<TRIES; t++) { 
        et = (et_v[t] < et)? et_v[t] : et; 
    }
    return et;
}

int main(int argc, char* argv[]) {
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    float et;

      {
//        img = load_ppm(argv[1], &w, &h);
//	FILE* fid = fopen("img.raw", "wb");
//	fwrite(img, sizeof(unsigned char), w * h, fid);
//	fclose(fid);

	w = atoi(argv[2]);
	h = atoi(argv[3]);
      	img = (unsigned char*)malloc(w*h*sizeof(unsigned char));
  	FILE* fid = fopen(argv[1], "r");
	fread(img, sizeof(unsigned char), w * h, fid);
	fclose(fid);

        label_gold = (int*)malloc(w*h*sizeof(int));
        label = (int*)malloc(w*h*sizeof(int));

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            gold::CCL(img, w, h, label_gold, -1, 8);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();
        printf("cc: %d\n", number_cc(label_gold, w, h));
        printf("gold: %.3f\n", et);        

	fid = fopen("cpuccl.raw", "wb");
	fwrite(label_gold, sizeof(int), w * h, fid);
	fclose(fid);

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            uf::CCL(img, w, h, label, -1, 8, false);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();
        
        printf("uf: %.3f\n", et);        

        VERIFY();

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            uf::CCL(img, w, h, label, -1, 4, true);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();
        
        printf("uf_hybrid: %.3f\n", et);        

        VERIFY();

        for (int t=0;t<TRIES;t++) {
            START_TIME;
            lequiv::CCL(img, w, h, label, -1, 4);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();

        printf("lequiv: %.3f\n", et);        

        VERIFY();
/*
        for (int t=0;t<TRIES;t++) {
            START_TIME;
            naive_prop::CCL(img, w, h, label);
            STOP_TIME;
            et_v[t] = et;
        }
        et = MIN_ET();

        printf("naive prop: %.3f\n", et);        

        VERIFY();
*/
	fid = fopen("test.raw", "wb");
	fwrite(label, sizeof(int), w * h, fid);
	fclose(fid);


        free(img);
        free(label);
      }


    return 0;
}
