/*
 * watershed-dw-korbes.cu
 *
 *  Created on: Dec 3, 2011
 *      from: http://parati.dca.fee.unicamp.br/adesso/wiki/watershed/ismm2011_dw/view/
 *
 *      fastest according to korbes.
 *
 *	adapted to support float
 */

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "watershed-dw-korbes.cuh"


namespace nscale { namespace gpu { namespace dw {

#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE

//#include "simple_arrays.h"

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define BORDER 0xFFFFFFF
#define UNVISITED 0xFFFFFFE

#define BLOCK_TILE 16  //7 //16
#define REAL_TILE  14  //5 //14
#define BLOCK_SIZE_1D 512 //8 //256 //512

//------------------------------------------------------------------------------
// Declaração de textura referencia
//------------------------------------------------------------------------------
texture<float, 2, cudaReadModeElementType> timg;
texture<int, 2, cudaReadModeElementType> tarrow;
texture<int, 1, cudaReadModeElementType> taux;

//------------------------------------------------------------------------------
// Constant memory for neigh
//------------------------------------------------------------------------------
__constant__ int c_neigh[16];


//------------------------------------------------------------------------------
//Arredonda a / b para o valor do vizinho superior
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
      switch(conn)
      {
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

//------------------------------------------------------------------------------
// Indexa a descida
//------------------------------------------------------------------------------
__global__ void cuIndexDescida(   int *gArrow,
                                  int   *gDist,
                                  int  width,
                                  int  height,
                                  int  conn      )
{

    int x = REAL_TILE * blockIdx.x + threadIdx.x - 1;
    int y = REAL_TILE * blockIdx.y + threadIdx.y - 1;
    int idg = y * width + x;

    int p,
        count = 0,
        arrow = UNVISITED,
        dist  = UNVISITED;

    __shared__ int  smImg[BLOCK_TILE][BLOCK_TILE],
                   smCtrl[BLOCK_TILE][BLOCK_TILE],
                   smDist[BLOCK_TILE][BLOCK_TILE];

    smImg[threadIdx.y][threadIdx.x] = (x < width && x >= 0 && y >= 0 && y < height)? tex2D(timg,(float)x,(float)y) : BORDER;
    smDist[threadIdx.y][threadIdx.x] = dist;
    smCtrl[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    if (    x < 0 || y < 0 ||     x >= width || y >= height ||
        threadIdx.x == 0 || threadIdx.x == BLOCK_TILE - 1 ||
        threadIdx.y == 0 || threadIdx.y == BLOCK_TILE - 1    )
            return;

    p = smImg[threadIdx.y][threadIdx.x];

    for(int pos=0; pos < conn ; pos++)
    {
        int dx = c_neigh[pos+conn];
        int dy = c_neigh[pos];
        int q  = smImg[threadIdx.y + dy][threadIdx.x + dx];

        if(q < p )
        {
            p = q;
            arrow = -( (y+dy) * width + (x+dx) );
            dist = 0;
        }
    }

    smDist[threadIdx.y][threadIdx.x] = dist;

    __syncthreads();

    p = smImg[threadIdx.y][threadIdx.x];

    if(smDist[threadIdx.y][threadIdx.x] == UNVISITED )
    {
        do
        {
            smCtrl[1][1] = 0;
            count = 0;
            __syncthreads();

            for(int pos=0; pos < conn ; pos++)
            {
                int dx = c_neigh[pos+conn];
                int dy = c_neigh[pos];

                int q = smImg[threadIdx.y + dy][threadIdx.x + dx];
                int d = smDist[threadIdx.y + dy][threadIdx.x + dx];

                if(q == p && (dist-1) > d )
                {
                    dist = d + 1;
                    arrow = -( (y+dy) * width + (x+dx) );
                    count = 1;
                }
            }
            smDist[threadIdx.y][threadIdx.x] = dist;
            if(count == 1) smCtrl[1][1] = 1;
            __syncthreads();

        }while(smCtrl[1][1]);
    }

    if(arrow == UNVISITED) arrow = idg;

    gArrow[idg] = arrow;
     gDist[idg] = dist;

}



//------------------------------------------------------------------------------
// Propaga e resolve a distancia do plateau
//------------------------------------------------------------------------------
__global__ void cuPropagaDescida(  int      *gArrow,
                                   int      *gDist,
                                   int       width,
                                   int       height,
                                   int       conn,
                                   int      *flag  )
{

    const int x = REAL_TILE * blockIdx.x + threadIdx.x - 1;
    const int y = REAL_TILE * blockIdx.y + threadIdx.y - 1;
    const int idg = y * width + x;

    int p,
        arrow=0,
        dist =0,
        count=0;

    //Memória compartilhada para dados com borda.
    __shared__ int   smImg[BLOCK_TILE][BLOCK_TILE],
                    smCtrl[BLOCK_TILE][BLOCK_TILE],
                    smDist[BLOCK_TILE][BLOCK_TILE];


    smImg[threadIdx.y][threadIdx.x]   = (x < width && x >= 0 && y >= 0 && y < height)? tex2D(timg,(float)x,(float)y) : BORDER;
    smDist[threadIdx.y][threadIdx.x]  = (x < width && x >= 0 && y >= 0 && y < height)? tex1Dfetch(taux,idg) : BORDER;
    smCtrl[threadIdx.y][threadIdx.x]  = 0;

    __syncthreads();

    if (    x < 0 || y < 0 ||     x >= width || y >= height ||
        threadIdx.x == 0 || threadIdx.x == BLOCK_TILE - 1 ||
        threadIdx.y == 0 || threadIdx.y == BLOCK_TILE - 1    )
            return;


    p = smImg[threadIdx.y][threadIdx.x];
    dist = smDist[threadIdx.y][threadIdx.x];

    if(smDist[threadIdx.y][threadIdx.x] > 0 )
    {
        do
        {
            smCtrl[1][1] = 0;
            count = 0;
            __syncthreads();

            for(int pos=0; pos < conn ; pos++)
            {
                int dx = c_neigh[pos+conn];
                int dy = c_neigh[pos];

                int q = smImg[threadIdx.y + dy][threadIdx.x + dx];
                int d = smDist[threadIdx.y + dy][threadIdx.x + dx];

                if(q == p && (dist-1) > d)
                {
                    dist = d + 1;
                    arrow = -( (y+dy) * width + (x+dx) );
                    count = 1;
                }
            }
            smDist[threadIdx.y][threadIdx.x] = dist;
            if(count == 1) smCtrl[1][1] = 1;
            __syncthreads();

        }while(smCtrl[1][1]);
    }

    if(arrow < 0)
    {
        gArrow[idg] = arrow;
        *flag += 1;
    }

    gDist[idg] = dist;

}
//----------------------------------------------------------------------------------------------
// Função que faz o scan verificando a rotulação dos seus vizinhos e atribuindo o de menor valor
//----------------------------------------------------------------------------------------------

__global__ void cuAgrupaPixel(   int *R,
                                 int *flag,
                                 int conn,
                                 int width,
                                 int height )

{

    int x = REAL_TILE * blockIdx.x + threadIdx.x - 1;
    int y = REAL_TILE * blockIdx.y + threadIdx.y - 1;
    int idg = y * width + x;

    __shared__ int  smArrow[BLOCK_TILE][BLOCK_TILE];

    smArrow[threadIdx.y][threadIdx.x] = tex2D(tarrow,(float)x,(float)y);

    __syncthreads();

    if (    x < 0 || y < 0 ||     x >= width || y >= height ||
        threadIdx.x == 0 || threadIdx.x == BLOCK_TILE - 1 ||
        threadIdx.y == 0 || threadIdx.y == BLOCK_TILE - 1    )
            return;

    int label = smArrow[threadIdx.y][threadIdx.x],
        label2= BORDER;

    R[idg] = label;

    for(int pos=0; pos < conn ; pos++)
    {
        int dx = c_neigh[pos+conn];
        int dy = c_neigh[pos];

        int label_v = smArrow[threadIdx.y + dy][threadIdx.x + dx];

        if( label_v >= 0 && label >= 0)
            label2 = min(label2,label_v);

    }
    __syncthreads();

    if (label2 < label)
    {
        atomicExch(&R[label],label2);
        *flag += 1;
    }
}
//----------------------------------------------------------------------------------------------
// Função que analisa a equivalencia dos pixels Agrupados
//----------------------------------------------------------------------------------------------
__global__ void cuPropagaIndex(  int *L,
                                 int *R,
                                 int dim )

{

    // inicializando o indice dos threads com os blocos

    int id = blockIdx.x * blockDim.x + threadIdx.x ;


    if(id >= dim) return;

    int label = L[id];

    if (label == id)
    {
        int ref = label;
        label = tex1Dfetch(taux,ref);

        do
        {
            ref = label;
            label = tex1Dfetch(taux,ref);

        }while(label != ref);

        __syncthreads();

        R[id] = label;
    }


}
//----------------------------------------------------------------------------------------------
// Função que Atribui os rotulos
//----------------------------------------------------------------------------------------------
__global__ void cuLabel(int *L,
                int dim )

{

    // inicializando o indice dos threads com os blocos

    const int id    = blockIdx.x * blockDim.x + threadIdx.x ;
    const int ref   = tex1Dfetch(taux,id);

    if(id >= dim) return;
    if(ref>=0) L[id] = tex1Dfetch(taux,ref);


}
//----------------------------------------------------------------------------------------------
// Função que Ajusta os vetores para o passo 4.
//----------------------------------------------------------------------------------------------
__global__ void cuAjustaVetor(int *L,
                    int *R,
                    int dim  )

{

    // inicializando o indice dos threads com os blocos
    const int id    = blockIdx.x * blockDim.x + threadIdx.x ;

    if(id >= dim) return;

    const int label = tex1Dfetch(taux,id);;
    __syncthreads();

    L[id] = id;
    R[id] = (label>=0)?label:-label;

}

//------------------------------------------------------------------------------
// Função Watershed
//------------------------------------------------------------------------------
__host__ void giwatershed( int *hdataOut,
                                      float *hdataIn,
                                      int w,
                                      int h,
                                      int conn,
                                      cudaStream_t stream)
{
    int *gArrow,
        *gDist,
        *gCtrl,
         hCtrl;

    neighvector(conn);

    // Dimensionando os kernels
    dim3 dimBlock(BLOCK_TILE,BLOCK_TILE);
    dim3 dimGrid( iDivUp(w,REAL_TILE),iDivUp(h,REAL_TILE));

    dim3  dimBlock1(BLOCK_SIZE_1D);
    dim3  dimGrid1(iDivUp(w*h,BLOCK_SIZE_1D));

    //Alocando memórias...
    int sizei = w * h * sizeof(int);
    cudaArray *cuarrayImg;
    cudaChannelFormatDesc descimg = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuarrayImg, &descimg, w, h);

    cudaArray *cuarrayArrow;
    cudaChannelFormatDesc desc32 = cudaCreateChannelDesc(sizeof(int) * 8, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaMallocArray(&cuarrayArrow, &desc32, w, h);

    cudaMalloc((void**)&gArrow, sizei);
    cudaMalloc((void**)&gDist , sizei);
    cudaMalloc((void**)&gCtrl, sizeof(int));

    // >> PASSO 1 <<
    cudaMemcpyToArray(cuarrayImg, 0, 0, hdataIn, w * h * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaBindTextureToArray(timg,cuarrayImg);
    cuIndexDescida<<<dimGrid, dimBlock, 0, stream>>>(gArrow, gDist, w, h,conn);

    // >> PASSO 2 <<

    do{

        cudaMemset(gCtrl, 0, sizeof(int));
        cudaBindTexture(0,taux,gDist,sizei);
        cuPropagaDescida<<<dimGrid, dimBlock, 0, stream>>>(gArrow, gDist, w, h, conn, gCtrl);
        cudaMemcpy(&hCtrl,gCtrl, sizeof(int),cudaMemcpyDeviceToHost);

    }while(hCtrl);

    // >> PASSO 3 <<

    do{
        cudaMemset(gCtrl, 0, sizeof(int));
        cudaMemcpyToArray(cuarrayArrow, 0, 0, gArrow, sizei, cudaMemcpyDeviceToDevice);
        cudaBindTextureToArray( tarrow,cuarrayArrow);
        cuAgrupaPixel<<<dimGrid, dimBlock, 0, stream>>>(gDist, gCtrl,conn,w,h);
        cudaMemcpy(&hCtrl,gCtrl, sizeof(int),cudaMemcpyDeviceToHost);

        if(hCtrl)
        {
            cudaBindTexture(0,taux,gDist,sizei);
            cuPropagaIndex<<<dimGrid1, dimBlock1, 0, stream>>>(gArrow,gDist,w*h);
            cudaUnbindTexture(taux);

            cudaBindTexture(0,taux,gDist,sizei);
            cuLabel<<<dimGrid1, dimBlock1, 0, stream>>>(gArrow, w*h);
            cudaUnbindTexture(taux);
            cudaThreadSynchronize();
        }
    }while(hCtrl);

    // >> PASSO 4 <<

    cudaBindTexture(0,taux,gArrow,sizei);
    cuAjustaVetor<<<dimGrid1, dimBlock1, 0, stream>>>(gArrow,gDist, w*h);
    cudaBindTexture(0,taux,gDist,sizei);
    cuPropagaIndex<<<dimGrid1, dimBlock1, 0, stream>>>(gArrow,gDist, w*h);
    cudaThreadSynchronize();

    // Copiando resultado para o host...
    cudaMemcpy(hdataOut,gDist, sizei,cudaMemcpyDeviceToDevice);

    //Liberando as memórias...
    cudaFreeArray(cuarrayImg);
    cudaFreeArray(cuarrayArrow);

    cudaFree(gArrow);
    cudaFree(gDist);
    cudaFree(gCtrl);

    cudaUnbindTexture(timg);
    cudaUnbindTexture(tarrow);
    cudaUnbindTexture(taux);
}



__global__ void cleanBorderKernel(int rows, int cols, const cv::gpu::PtrStep_<unsigned char> mask, const cv::gpu::PtrStep_<int> label, cv::gpu::PtrStep_<int> result, int background, int connectivity)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // treat this as a specialized erosion
    if (y > 0 && y < (rows-1) && x > 0 && x < (cols-1))
    {
    	int p = label.ptr(y)[x];
    	int output = p;
    	// if p is already background, this will not change it so run it through
		unsigned char q, q1, q2, q3, q4;

		q = mask.ptr(y)[x];
		q1 = mask.ptr(y-1)[x];
		q2 = mask.ptr(y)[x-1];
		q3 = mask.ptr(y)[x+1];
		q4 = mask.ptr(y+1)[x];
		// if any of them is background in the mask
		if (q == 0 ||
				q1 == 0 ||
				q2 == 0 ||
				q3 == 0 ||
				q4 == 0) {
			q = 0;
			output = background;
		}

		if (connectivity == 8) {

			q1 = mask.ptr(y-1)[x-1];
			q2 = mask.ptr(y-1)[x+1];
			q3 = mask.ptr(y+1)[x-1];
			q4 = mask.ptr(y+1)[x+1];
			if (q == 0 ||
					q1 == 0 ||
					q2 == 0 ||
					q3 == 0 ||
					q4 == 0) {
				output = background;
			}
		}

    	result.ptr(y)[x] = output;
    }
}

__host__ void giwatershed_cleanup( const cv::gpu::PtrStep_<unsigned char> mask,
		const cv::gpu::PtrStep_<int> label,
		cv::gpu::PtrStep_<int> result,
                                      int w,
                                      int h,
                                      int background,
                                      int conn,
                                      cudaStream_t stream)
{
	// check to see if a pixel's neighbor is background.  if so, call the current one background as well.

	   dim3 threads(16, 16);
	    dim3 grid((w + threads.x -1) / threads.x, (h + threads.y - 1) / threads.y);

	    cleanBorderKernel<<<grid, threads, 0, stream>>>(h, w, mask, label, result, background, conn);
	    cudaGetLastError();

	    if (stream == 0)
	        cudaDeviceSynchronize();


}

}}}
