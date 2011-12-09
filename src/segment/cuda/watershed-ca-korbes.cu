/*
 * watershed-CA-korbes.cu
 *
 *  Created on: Dec 3, 2011
 *      from http://parati.dca.fee.unicamp.br/adesso/wiki/watershed/ismm2011_ca/view/
 *
 *      cellular automata approach.  not as fast as DW.
 */

#include <cuda_runtime.h>

#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

namespace nscale { namespace gpu { namespace ca {


#define IMUL(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE_2D       16
#define REAL_BLOCK          14

//------------------------------------------------------------------------------
// CUDA error check and print
//------------------------------------------------------------------------------
void checkCUDAError(const char *msg);

//------------------------------------------------------------------------------
// Texture image
//------------------------------------------------------------------------------
texture<float, 2, cudaReadModeElementType> f_tex;
texture<int, 2, cudaReadModeElementType> mins_tex;

//------------------------------------------------------------------------------
// Constant memory for neigh
//------------------------------------------------------------------------------
__constant__ int c_neigh[16];

//------------------------------------------------------------------------------
// Shared memory for image
//------------------------------------------------------------------------------
__shared__ unsigned short s_img[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
__shared__ unsigned short s_lambda[BLOCK_SIZE_2D][BLOCK_SIZE_2D];
__shared__ unsigned short s_label[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

//------------------------------------------------------------------------------
// Round up the division a/b
//------------------------------------------------------------------------------
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//------------------------------------------------------------------------------
// Generate neighborhood offsets
//------------------------------------------------------------------------------
void neighvector(int conn)
{
    int neigh[16];
    int i = 0;
    switch(conn) {
        case 4:
            // for Y
            neigh[i++] = -1;
            neigh[i++] =  0;
            neigh[i++] =  0;
            neigh[i++] =  1;
            // for X
            neigh[i++] =  0;
            neigh[i++] = -1;
            neigh[i++] =  1;
            neigh[i++] =  0;
            break;
        case 8:
            // for Y
            neigh[i++] = -1;
            neigh[i++] = -1;
            neigh[i++] = -1;
            neigh[i++] =  0;
            neigh[i++] =  0;
            neigh[i++] =  1;
            neigh[i++] =  1;
            neigh[i++] =  1;
            // for X
            neigh[i++] = -1;
            neigh[i++] =  0;
            neigh[i++] =  1;
            neigh[i++] = -1;
            neigh[i++] =  1;
            neigh[i++] = -1;
            neigh[i++] =  0;
            neigh[i++] =  1;
            break;
    }

    cudaMemcpyToSymbol(c_neigh, neigh, 16*sizeof(int), 0, cudaMemcpyHostToDevice);
}


__device__ float cost(int fp, int fq, int px, int py, int qx, int qy)
{

    if (fp > fq)
    {
        int minv = tex2D(mins_tex, px, py);
        return fp - minv;
    }
    else if (fq > fp)
    {
        int minv = tex2D(mins_tex, qx, qy);
        return fq - minv;
    }
    else
    {
        int minv = tex2D(mins_tex, px, py);
        int minv2 = tex2D(mins_tex, qx, qy);
        return ((fp - minv)+(fq - minv2))/2.0;
    }

}

//------------------------------------------------------------------------------
// Perform iterations of Ford-Bellmann inside block
//------------------------------------------------------------------------------
__global__ void cuFordBellmann(int *label, int *label_next, int *lambda, int *lambda_next, int *flag, int w, int h, int conn)
{
    const int px = REAL_BLOCK*blockIdx.x + threadIdx.x - 1;
    const int py = REAL_BLOCK*blockIdx.y + threadIdx.y - 1;
    const int idp = py*w + px;
    int count = 0;
    __shared__ int bChanged;

    if (px < 0 || py < 0 || px >= w || py >= h)
    {
        s_img[threadIdx.y][threadIdx.x] = 0xFFFF;
        s_lambda[threadIdx.y][threadIdx.x] = 0;
        s_label[threadIdx.y][threadIdx.x] = 0;
    }
    else
    {
        s_img[threadIdx.y][threadIdx.x] = tex2D(f_tex,(float)px,(float)py);
        s_lambda[threadIdx.y][threadIdx.x] = lambda[idp];
        s_label[threadIdx.y][threadIdx.x] = label[idp];
    }

    bChanged = 1;

    __syncthreads();

    if (px < 0 || py < 0 || px >= w || py >= h ||
        threadIdx.x == 0 || threadIdx.x >= BLOCK_SIZE_2D - 1 ||
        threadIdx.y == 0 || threadIdx.y >= BLOCK_SIZE_2D - 1)
        return;


    while (bChanged && count < 28)
    {
        bChanged = 0;

        count++;

        int fp, fq;
        int u = 0x00FFFFFF;
        int ux, uy;
        int lambdap = s_lambda[threadIdx.y][threadIdx.x];
        int idu = -1;

        fp = s_img[threadIdx.y][threadIdx.x];

        for(int pos = 0; pos < conn ; pos++)
        {
            int qx = px + c_neigh[pos+conn];
            int qy = py + c_neigh[pos];
            int sqx = threadIdx.x + c_neigh[pos+conn];
            int sqy = threadIdx.y + c_neigh[pos];

            if (qx >= 0 && qy >= 0 && qx < w && qy < h)
            {
                int lambdaq = s_lambda[sqy][sqx];
                fq = s_img[sqy][sqx];

                float c = cost(fp,
                               fq,
                               px,
                               py,
                               qx,
                               qy);

                if (lambdaq + c < u)
                {
                    u = lambdaq + c;
                    ux = sqx;
                    uy = sqy;
                    idu = 1;
                }
            }
        }

        int ulabel = 0;
        if (idu >= 0)
            ulabel = s_label[uy][ux];

        __syncthreads();

        if (idu >= 0 && u < lambdap)
        {
            s_lambda[threadIdx.y][threadIdx.x] = u;
            s_label[threadIdx.y][threadIdx.x] = ulabel;
            *flag += 1;
            bChanged = 1;
        }

        __syncthreads();

    }

    lambda_next[idp] = s_lambda[threadIdx.y][threadIdx.x];
    label_next[idp] = s_label[threadIdx.y][threadIdx.x];
}

//------------------------------------------------------------------------------
// Initialize the lambda memory on the seeds and find neighborhood minima
//------------------------------------------------------------------------------
__global__ void cuInit(int *lambda, int *seeds, int *mins, int w, int h, int conn)
{
    const int px = REAL_BLOCK*blockIdx.x + threadIdx.x - 1;
    const int py = REAL_BLOCK*blockIdx.y + threadIdx.y - 1;
    const int idp = py*w + px;

    if (px < 0 || py < 0 || px >= w || py >= h)
    {
        s_img[threadIdx.y][threadIdx.x] = 0xFFFF;
    }
    else
    {
        s_img[threadIdx.y][threadIdx.x] = tex2D(f_tex,(float)px,(float)py);
    }

    if (px < 0 || py < 0 || px >= w || py >= h ||
        threadIdx.x == 0 || threadIdx.x >= BLOCK_SIZE_2D - 1 ||
        threadIdx.y == 0 || threadIdx.y >= BLOCK_SIZE_2D - 1)
        return;

    lambda[idp] = (seeds[idp] == 0) * 0x00FFFFFF;

    int minv = 0x7FFFFFFF;
    int fv;
    for(int pos = 0; pos < conn ; pos++)
    {
        int vx = threadIdx.x + c_neigh[pos+conn];
        int vy = threadIdx.y + c_neigh[pos];
        fv = s_img[vy][vx];
        if (fv < minv)
            minv = fv;
    }

    mins[idp] = minv;

}

//------------------------------------------------------------------------------
// Watershed by Kauffmann & Piche
//------------------------------------------------------------------------------
__host__ float ws_kauffmann(int *label, // output
                           float *f, // input
                           int *seeds, // seeds (regional minima)
                           int w,  // width
                           int h, //height
                           int conn) // connectivity (4 or 8)

{

    int *label_d,
        *label_next_d,
        *lambda_d,
        *lambda_next_d,
        *flag_d,
        *mins_d;

    unsigned int timer;
    float measuredTime;

    timeval tim;


    cudaArray *f_d;
    cudaArray *mins_a_d;

    int flag;

    int sizei = w * h * sizeof(int);
    int sizec = w * h * sizeof(float);

    // Setup the grid hierarchy
    dim3 dimBlock(BLOCK_SIZE_2D,BLOCK_SIZE_2D);
    dim3 dimGrid(iDivUp(w,REAL_BLOCK),iDivUp(h,REAL_BLOCK));

    neighvector(conn);

    cudaChannelFormatDesc desc8 = cudaCreateChannelDesc<float>();
    checkCUDAError("cudaCreateChannelDesc8");

    cudaChannelFormatDesc desc32 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    checkCUDAError("cudaCreateChannelDesc32");


    // Allocate memory
    cudaMallocArray(&f_d, &desc8, w, h);
    checkCUDAError("cudaMallocArray f_d");

    cudaMemcpyToArray(f_d, 0, 0, f, sizec, cudaMemcpyDeviceToDevice);
    checkCUDAError("cudaMemcpyToArray f_d");

    cudaBindTextureToArray(f_tex, f_d);
    checkCUDAError("cudaBindTextureToArray f_tex");


    cudaMalloc((void**)&label_d, sizei);
    cudaMemset(label_d, 0, sizei);

    cudaMalloc((void**)&label_next_d, sizei);
    cudaMemcpy(label_next_d, seeds, sizei, cudaMemcpyDeviceToDevice);

    cudaMalloc((void**)&lambda_d, sizei);
    cudaMemset(lambda_d, 0, sizei);

    cudaMalloc((void**)&lambda_next_d, sizei);
    cudaMemset(lambda_next_d, 0, sizei);

    cudaMalloc((void**)&mins_d, sizei);
    cudaMemset(mins_d, 0, sizei);

    cudaMalloc((void**)&flag_d, sizeof(int));

    gettimeofday(&tim, NULL);
    double t1=tim.tv_sec+(tim.tv_usec/1000000.0);

    // Initialize the lambda image with zeros on minima
    cuInit<<<dimGrid, dimBlock>>>(lambda_next_d, label_next_d, mins_d, w, h, conn);
    cudaThreadSynchronize();

    cudaMallocArray(&mins_a_d, &desc32, w, h);
    checkCUDAError("cudaMallocArray mins_a_d");

    cudaMemcpyToArray(mins_a_d, 0, 0, mins_d, sizei, cudaMemcpyDeviceToDevice);
    checkCUDAError("cudaMemcpyToArray mins_a_d");

    cudaBindTextureToArray(mins_tex, mins_a_d);
    checkCUDAError("cudaBindTextureToArray mins_a_d");

    // Iterate until stabilization
    int iter = 0;
    do{
        iter++;
        //pyprintf("iter\n");
        cudaMemset(flag_d, 0, sizeof(int));
        cudaMemcpy(lambda_d, lambda_next_d, sizei, cudaMemcpyDeviceToDevice);
        cudaMemcpy(label_d, label_next_d, sizei, cudaMemcpyDeviceToDevice);

        cuFordBellmann<<<dimGrid, dimBlock>>>(label_d, label_next_d, lambda_d, lambda_next_d, flag_d, w, h, conn);

        cudaThreadSynchronize();

        cudaMemcpy(&flag, flag_d, sizeof(int), cudaMemcpyDeviceToHost);

    }while(flag > 0 && iter < 2000);

    //cutStopTimer(timer);

    gettimeofday(&tim, NULL);
    double t2=tim.tv_sec+(tim.tv_usec/1000000.0);

    //pyprintf("iter: %d\n", iter);

    // Copy the labels
    cudaMemcpy(label, label_d, sizei, cudaMemcpyDeviceToDevice);

    // Free and Unbind memory
    cudaFree(mins_d);
    cudaFreeArray(f_d);
    cudaFreeArray(mins_a_d);
    cudaFree(lambda_d);
    cudaFree(lambda_next_d);
    cudaFree(label_d);
    cudaFree(label_next_d);
    cudaFree(flag_d);
    cudaUnbindTexture(f_tex);
    cudaUnbindTexture(mins_tex);

    return t2-t1;

}

//------------------------------------------------------------------------------
// Error Check
//------------------------------------------------------------------------------
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        printf("Cuda error: %s: %s. \n", msg, cudaGetErrorString(err));
        exit(-1);
    }

}

}}}
