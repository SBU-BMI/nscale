/*
 * watershed-cilamce2009.cu
 *
 *  Created on: Dec 3, 2011
 *
 *
 *  from  http://parati.dca.fee.unicamp.br/adesso/wiki/watershed/cilamce2009/view/
 */

#include <cuda_runtime.h>


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "simple_arrays.h"


#define IMUL(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE_2D 16  //5  //16
#define BLOCK_SIZE_1D 256 //8 //256
#define UNVISITED 0xFFFFFFE
#define BORDER 0xFFFFFFF
#define KERNEL_RADIUS 1
#define RANGE_W 14 //3  //14  //16
#define RANGE_H 14 //3  //14  //10
#define KERNEL_RADIUS_ALIGNED 1

void deleteTexture(cudaArray *cu_array);
void checkCUDAError(const char *msg);

//------------------------------------------------------------------------------
// Declaração de textura referencia
texture<unsigned char, 2, cudaReadModeElementType> tex_img;
texture<int, 2, cudaReadModeElementType> tex_label;
texture<int, 1, cudaReadModeElementType> tex_resp;

//Arredonda a / b para o valor do vizinho superior
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

int flag(unsigned char*, int);

//------------------------------------------------------------------------------
// Função que retorna as posições dos vizinhos de um pixel.
//------------------------------------------------------------------------------
__device__ void neighvector(int* neigh,int conn)
{

    int i = 0;
    switch(conn) {
        case 4:
            // em função de Y
        neigh[i++] = -1;
            neigh[i++] =  0;
            neigh[i++] =  0;
        neigh[i++] =  1;
            // em função de X
        neigh[i++] =  0;
        neigh[i++] = -1;
        neigh[i++] =  1;
        neigh[i++] =  0;
            break;
        case 8:
          // em função de Y
            neigh[i++] = -1;
            neigh[i++] = -1;
            neigh[i++] = -1;
            neigh[i++] =  0;
            neigh[i++] =  0;
            neigh[i++] =  1;
            neigh[i++] =  1;
            neigh[i++] =  1;
        // em função de X
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

}
//------------------------------------------------------------------------------
// Indexa a descida
//------------------------------------------------------------------------------
__global__ void cuIndexDescida( int* d_result,
                      int width,
                      int height,
                      int conn      )
{
    unsigned int x = IMUL(blockIdx.x,blockDim.x) + threadIdx.x,
             y = IMUL(blockIdx.y,blockDim.y) + threadIdx.y,
             xinner = threadIdx.x,
             yinner = threadIdx.y;

    __shared__ int p,q,aux,out[BLOCK_SIZE_2D][BLOCK_SIZE_2D];

    int neigh[16];

    neighvector(neigh,conn);
    p = tex2D(tex_img,(float)x,(float)y);
    out[xinner][yinner]=UNVISITED;

    if(x < width && y < height){

        // Indexa as descidas
        for(int pos=0; pos < conn ; pos++){
            int idx = x + neigh[pos+conn];
            int idy = y + neigh[pos];

            q = tex2D(tex_img,(float)idx,(float)idy);

            // Add os apontadores OUT
            if(q < p && idx >=0 && idx < width && idy >=0 && idy < height){
                p = q;
                out[xinner][yinner] = -(IMUL(idy,width) + idx);
            }
        }

        aux =(out[xinner][yinner] == UNVISITED)? (IMUL(y,width) + x) : out[xinner][yinner];
        d_result[IMUL(y,width) + x] = aux;
    }
    __syncthreads();
}

//------------------------------------------------------------------------------
// Propaga a descida para "plateur"
//------------------------------------------------------------------------------
__global__ void cuPropagaDescida(int* d_result,
                       //int* d_data,
                       int width,
                       int height,
                       int conn,
                       unsigned char *control  )
{

    __shared__ int neigh[16];

    //Memória compartilhada para dados com borda.
    __shared__ int data[KERNEL_RADIUS + RANGE_H + KERNEL_RADIUS][KERNEL_RADIUS + RANGE_W + KERNEL_RADIUS];
    __shared__ int posicao[KERNEL_RADIUS + RANGE_H + KERNEL_RADIUS][KERNEL_RADIUS + RANGE_W + KERNEL_RADIUS];
    __shared__ int  pt[KERNEL_RADIUS + RANGE_H + KERNEL_RADIUS][KERNEL_RADIUS + RANGE_W + KERNEL_RADIUS];
    __shared__ int aux[KERNEL_RADIUS + RANGE_H + KERNEL_RADIUS][KERNEL_RADIUS + RANGE_W + KERNEL_RADIUS];
    __shared__ int count[KERNEL_RADIUS + RANGE_H + KERNEL_RADIUS][KERNEL_RADIUS + RANGE_W + KERNEL_RADIUS];

        //Index relativos a faixa e borda de dados para o bloco
    const int         rangeStartx = IMUL(blockIdx.y, RANGE_H);
    const int         rangeStarty = IMUL(blockIdx.x, RANGE_W);
    const int           rangeEndx = rangeStartx + RANGE_H - 1;
    const int           rangeEndy = rangeStarty + RANGE_W - 1;

    const int borderStartClampedx = max((rangeStartx-KERNEL_RADIUS), 0);
    const int borderStartClampedy = max((rangeStarty-KERNEL_RADIUS), 0);

    const int   borderEndClampedx = min((rangeEndx+KERNEL_RADIUS), height - 1);
    const int   borderEndClampedy = min((rangeEndy+KERNEL_RADIUS), width - 1);

    const int loadPosx = (rangeStartx - KERNEL_RADIUS) + threadIdx.y;
    const int loadPosy = (rangeStarty - KERNEL_RADIUS_ALIGNED) + threadIdx.x;

    const int writePosx = rangeStartx + threadIdx.y;
    const int writePosy = rangeStarty + threadIdx.x;

    neighvector(neigh,conn);

    // carrega os dados da memGlobal para memShared com as bordas...
    if(loadPosx >= (rangeStartx-KERNEL_RADIUS) && loadPosy >= (rangeStarty-KERNEL_RADIUS) ){

        const int smemPosx = loadPosx - (rangeStartx-KERNEL_RADIUS);// --> borderStartx
        const int smemPosy = loadPosy - (rangeStarty-KERNEL_RADIUS);// --> borderStarty

            data[smemPosx][smemPosy] =
                ((loadPosy >= borderStartClampedy) && (loadPosy <= borderEndClampedy) &&
             (loadPosx >= borderStartClampedx) && (loadPosx <= borderEndClampedx)) ?
                    tex1Dfetch(tex_resp,IMUL(loadPosx,width) + loadPosy) : BORDER;

        posicao[smemPosx][smemPosy] = -(IMUL(loadPosx,width) + loadPosy);
        pt[smemPosx][smemPosy] = tex2D(tex_img,(float)loadPosy,(float)loadPosx);

    }

    __syncthreads();

    //Executa o processamento dentro da janela especifica
    if(writePosx <= min(rangeEndx, height - 1) && writePosy <= min(rangeEndy, width - 1)){
        const int WsmemPosx = writePosx - (rangeStartx-KERNEL_RADIUS);
        const int WsmemPosy = writePosy - (rangeStarty-KERNEL_RADIUS);

        count[WsmemPosx][WsmemPosy]=0;

        do{
            count[WsmemPosx][WsmemPosy]++;
            int p = pt[WsmemPosx][WsmemPosy];
            int out =(int) data[WsmemPosx][WsmemPosy];
            int mask = 1;
            aux[WsmemPosx][WsmemPosy]=0;

            __syncthreads();

            for(int pos=0; pos < conn ; pos++){
                int idx = WsmemPosx + neigh[pos];
                int idy = WsmemPosy + neigh[pos+conn];
                int mask_aux = 0;

                int q_out = data[idx][idy];
                __syncthreads();

                if(out >= 0 && q_out < 0) mask_aux = 1;

                int q = pt[idx][idy];
                __syncthreads();

                if(mask_aux == 1 && q == p && mask == 1){

                    data[WsmemPosx][WsmemPosy] = posicao[idx][idy];
                    aux[WsmemPosx][WsmemPosy] = 1;
                    mask = 0;
                }
                __syncthreads();

            }

        }while(any(aux[WsmemPosx][WsmemPosy]));

        d_result[IMUL(writePosx,width) + writePosy] = data[WsmemPosx][WsmemPosy];
        control[IMUL(writePosx,width) + writePosy] = (unsigned char)count[WsmemPosx][WsmemPosy]>1;
        //control[IMUL(writePosx,width) + writePosy] = (unsigned char)count[WsmemPosx][WsmemPosy];
        __syncthreads();
    }
}
//----------------------------------------------------------------------------------------------
// Função que faz o scan verificando a rotulação dos seus vizinhos e atribuindo o de menor valor
//----------------------------------------------------------------------------------------------
__global__ void cuAgrupaPixel( int *L,
                     int *R,
                     unsigned char *m,
                     int conn,
                     int width,
                     int height )
{
    // inicializando o indice dos threads com os blocos
    const int x = IMUL(blockIdx.x,blockDim.x) + threadIdx.x ;
    const int y = IMUL(blockIdx.y,blockDim.y) + threadIdx.y ;
    const int id = IMUL(y,width) + x;
    unsigned char md=0;

    int label1 = tex2D(tex_label,(float)x,(float)y);
    int label2 = BORDER;

    int neigh[16];
    neighvector(neigh,conn);

    L[id] = label1;
    R[id] = label1;
    m[id] = 0;

    if(x < width && y < height)
    {
        for(int pos=0; pos < conn ; pos++)
        {
            int idx = x + neigh[pos+conn];
            int idy = y + neigh[pos];

            int label_v = tex2D(tex_label,(float)idx,(float)idy);

            if( label_v >= 0 && label1 >= 0)
            {
                label2 = min(label2,label_v);
            }
        }
        __syncthreads();


        if (label2 < label1)
        {
            //atomicMin(&R[id],label2);
            atomicExch(&R[label1],label2); // OBS.: PULO DO GATO
            md=1;
        }
        m[id]= md;
    }
    __syncthreads();
}

//----------------------------------------------------------------------------------------------
// Função que analisa a equivalencia dos pixels Agrupados
//----------------------------------------------------------------------------------------------
__global__ void cuPropagaIndex(int *L,
                     int *R  )
{
    // inicializando o indice dos threads com os blocos
    int id = IMUL(blockIdx.x,blockDim.x) + threadIdx.x ;
    int label = L[id];

    //R[id] = tex1Dfetch(tex_resp,id);

    if (label == id){
        int ref;
        ref = label;
        label = tex1Dfetch(tex_resp,ref);

        do{
            ref = label;
            label = tex1Dfetch(tex_resp,ref);

        }while(label != ref);
        __syncthreads();

        R[id] = label;
    }

    __syncthreads();
}
//----------------------------------------------------------------------------------------------
// Função que Atribui os rotulos
//----------------------------------------------------------------------------------------------
__global__ void cuLabel(int *L)
{
    // inicializando o indice dos threads com os blocos
    const int id    = IMUL(blockIdx.x,blockDim.x) + threadIdx.x ;
    const int label = L[id];
    const int ref   = tex1Dfetch(tex_resp,id);

    L[id] = (label>=0)?tex1Dfetch(tex_resp,ref):label;


    __syncthreads();
}

//----------------------------------------------------------------------------------------------
// Função que Ajusta os vetores para o passo 4.
//----------------------------------------------------------------------------------------------
__global__ void cuAjustaVetor(int *L,
                    int *R  )
{
    // inicializando o indice dos threads com os blocos
    const int id    = IMUL(blockIdx.x,blockDim.x) + threadIdx.x ;
    const int label = L[id];

    L[id] = id;
    R[id] = (label>=0)?label:-label;

    __syncthreads();
}


//------------------------------------------------------------------------------
// Função Watershed
//------------------------------------------------------------------------------
__host__ void giwatershed( int *o_dados, // saida
                          unsigned char *i_dados, // entrada
                          unsigned char *control_h, // imagem de trabalho
                          int w,  // largura
                          int h, //altura
                          int conn, // conectividade (4 ou 8)
                          int verbose   ) // imprime mensagens (nao utilizado)
{

    //unsigned int Timer1,Timer2, Timer3, Timer4, TimerCopy, TimerGeral ;

    //cutilCheckError( cutCreateTimer(&Timer1) );cutilCheckError( cutCreateTimer(&Timer2) );cutilCheckError( cutCreateTimer(&Timer3) );cutilCheckError( cutCreateTimer(&Timer4) );
        //cutilCheckError( cutCreateTimer(&TimerCopy) );cutilCheckError( cutCreateTimer(&TimerGeral) );

    //cutilCheckError( cutStartTimer(TimerGeral)   );

    int *L_d,
        *R_d,
        f;

    unsigned char *control_d;

    int sizei = w * h * sizeof(int);
    int sizec = w * h * sizeof(unsigned char);

    // Dimensionando os blocos para o 1 kernel
    dim3 dimBlock1(BLOCK_SIZE_2D,BLOCK_SIZE_2D);
      dim3 dimGrid1( iDivUp(w,BLOCK_SIZE_2D),iDivUp(h,BLOCK_SIZE_2D));

    // Dimensionando os blocos para o 2 kernel
    dim3 dimBlock2(KERNEL_RADIUS_ALIGNED + RANGE_W + KERNEL_RADIUS, KERNEL_RADIUS + RANGE_H + KERNEL_RADIUS);
      dim3 dimGrid2 (iDivUp(w,RANGE_W),iDivUp(h,RANGE_H));

    // Dimensionando os blocos para o kernel analisys
    dim3  dimBlock3(BLOCK_SIZE_1D);
        dim3  dimGrid3(iDivUp(w*h,BLOCK_SIZE_1D));


    // >> PASSO 1 <<

    // Coloca os dados na memoria de textura

    cudaArray *tdados_img;
    cudaChannelFormatDesc desc8 = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned);
        checkCUDAError("cudaCreateChannelDesc 1 passo");

        (cudaMallocArray(&tdados_img, &desc8, w, h));
    checkCUDAError("cudacudaMallocArray");

        //cutilCheckError( cutStartTimer(TimerCopy)   );
    (cudaMemcpyToArray(tdados_img, 0, 0, i_dados,sizec, cudaMemcpyHostToDevice));
        checkCUDAError("cudaMemcpyToArray");
        //cutilCheckError( cutStopTimer(TimerCopy)   );

        ( cudaBindTextureToArray( tex_img,tdados_img));
    checkCUDAError("cudaBindTextureToArray 2D HD");

    ( cudaMalloc((void**)&L_d, sizei));


    //cutilCheckError( cutStartTimer(Timer1)   );
        cuIndexDescida<<<dimGrid1, dimBlock1>>>(L_d, w, h,conn);
        checkCUDAError("Kernel execution");
        (cudaThreadSynchronize());
    //cutilCheckError( cutStopTimer(Timer1)   );

    // >> PASSO 2 <<

    ( cudaMalloc((void**)&R_d, sizei));
    ( cudaMalloc((void**)&control_d,sizec));

    //cutilCheckError( cutStartTimer(Timer2)   );
    do{
        // Coloca os dados na memoria de textura
        ( cudaBindTexture(0,tex_resp,L_d,sizei));
        checkCUDAError("cudaBindTextureToArray 1D culabel");

        cuPropagaDescida<<<dimGrid2, dimBlock2>>>(L_d, w, h,conn,control_d);
        (cudaThreadSynchronize());

        ( cudaMemcpy(control_h,control_d,sizec,cudaMemcpyDeviceToHost));

    }while(flag(control_h,(w*h)));

    //cutilCheckError( cutStopTimer(Timer2)   );

    // >> PASSO 3 <<

    cudaArray *tdados_lab;
    cudaChannelFormatDesc desc32 = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    checkCUDAError("cudaCreateChannelDesc");
    (cudaMallocArray(&tdados_lab, &desc32, w, h));
    checkCUDAError("cudacudaMallocArray");

    //cutilCheckError( cutStartTimer(Timer3)   );
    do
    {
        // Coloca os dados na memoria de textura
        (cudaMemcpyToArray(tdados_lab, 0, 0, L_d,sizei, cudaMemcpyDeviceToDevice));
        checkCUDAError("cudaMemcpyToArray");
        ( cudaBindTextureToArray( tex_label,tdados_lab));
        checkCUDAError("cudaBindTextureToArray 2D DD");

        cuAgrupaPixel<<<dimGrid1, dimBlock1>>>(L_d,R_d,control_d,conn,w,h);
        (cudaThreadSynchronize());

        ( cudaMemcpy(control_h,control_d,sizec,cudaMemcpyDeviceToHost));
        f = flag(control_h,(w*h));

        if(f)
        {
            // Coloca os dados na memoria de textura
            ( cudaBindTexture(0,tex_resp,R_d,sizei));
            checkCUDAError("cudaBindTextureToArray 1D propagaIndex");

            cuPropagaIndex<<<dimGrid3, dimBlock3>>>(L_d,R_d);
            (cudaThreadSynchronize());

            // Coloca os dados na memoria de textura
            ( cudaBindTexture(0,tex_resp,R_d,sizei));
            checkCUDAError("cudaBindTextureToArray 1D culabel");

            cuLabel<<<dimGrid3, dimBlock3>>>(L_d);
            (cudaThreadSynchronize());
        }
    }while(f);

    //cutilCheckError( cutStopTimer(Timer3)   );

    // >> PASSO 4 <<

    //cutilCheckError( cutStartTimer(Timer4)   );

    cuAjustaVetor<<<dimGrid3, dimBlock3>>>(L_d,R_d);

    ( cudaBindTexture(0,tex_resp,R_d,sizei));
    cuPropagaIndex<<<dimGrid3, dimBlock3>>>(L_d,R_d);
    (cudaThreadSynchronize());

    //cutilCheckError( cutStopTimer(Timer4)   );


    //cutilCheckError( cutStartTimer(TimerCopy)   );
    (cudaMemcpy(o_dados,R_d,sizei,cudaMemcpyDeviceToHost));
    //cutilCheckError( cutStopTimer(TimerCopy)   );

    //cutilCheckError( cutStopTimer(TimerGeral)   );

    /*
    if(verbose){
        printf( "\n\n :: Processing time ::\n");
        printf( " Passo1: %f (ms)\n", cutGetAverageTimerValue(Timer1));
        printf( " Passo2: %f (ms)\n", cutGetAverageTimerValue(Timer2));
        printf( " Passo3: %f (ms)\n", cutGetAverageTimerValue(Timer3));
        printf( " Passo4: %f (ms)\n", cutGetAverageTimerValue(Timer4));
        printf( " Copy data: %f (ms)\n", cutGetAverageTimerValue(TimerCopy));
        printf( " Geral: %f (ms)\n", cutGetAverageTimerValue(TimerGeral));
    }
    */
    // Libera as alocações de memória do GPU
    deleteTexture(tdados_img);
    deleteTexture(tdados_lab);

    (cudaFree(control_d));
    (cudaFree(L_d));
    (cudaFree(R_d));

    (cudaUnbindTexture(tex_img));
    (cudaUnbindTexture(tex_label));
    (cudaUnbindTexture(tex_resp));

      //cutilCheckError( cutDeleteTimer(Timer1) );cutilCheckError( cutDeleteTimer(Timer2) );cutilCheckError( cutDeleteTimer(Timer3) );cutilCheckError( cutDeleteTimer(Timer4) );
      //cutilCheckError( cutDeleteTimer(TimerCopy) );cutilCheckError( cutDeleteTimer(TimerGeral) );

}
//------------------------------------------------------------------------------
void deleteTexture(cudaArray *array)
{
    (cudaFreeArray(array));
}
//------------------------------------------------------------------------------
// Função que verifica se ocorreu um erro...
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s. \n", msg, cudaGetErrorString(err));
        exit(-1);
    }

}
