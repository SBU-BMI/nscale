/*** Written by Salil Deosthale 11/30/2012 ***/

#include "features.cuh"
#include <limits>
#include <iostream>



namespace nscale {
	namespace gpu {
		
		using namespace cv::gpu;
		
		__global__ void area(const int *boundingBoxInfo, int compCount, const cv::gpu::PtrStep_<int> labeledMask, int *areaRes)
		{
			//Declare a shared array "thread_area[NumThreads]". This will hold the value of the area each thread walks through
			__shared__ int thread_area[32];
			//Zero out the thread_area array
			thread_area[threadIdx.x] = 0;
			//Pointer to a row of the image 
			const int *labeledImgPtr;
			//Label of this current component
			int label = boundingBoxInfo[blockIdx.x];
			int maxX = boundingBoxInfo[2 * compCount + blockIdx.x];
			int maxY = boundingBoxInfo[4 * compCount + blockIdx.x];

			for(int x = boundingBoxInfo[compCount + blockIdx.x] +threadIdx.x; x <= maxX ; x+=blockDim.x)
			{
				for(int y = boundingBoxInfo[3 * compCount + blockIdx.x] ; y <= maxY ; y++)
				{
					labeledImgPtr = labeledMask.ptr(y);
					thread_area[threadIdx.x] += (labeledImgPtr[x] == label ? 1 : 0);
				}
			}

			__syncthreads();

			//Now, we do a parallel reduction using sequential addressing
			unsigned int s;
			for(s = blockDim.x /2 ; s > 0 ; s >>= 1)
			{
				if(threadIdx.x < s)
				{
					thread_area[threadIdx.x] += thread_area[threadIdx.x + s];
				}
				__syncthreads();
			}

			if (threadIdx.x == 0) areaRes[blockIdx.x] = thread_area[0];
				
		}
	
		__global__ void perimeter(const int *boundingBoxInfo, int compCount, const cv::gpu::PtrStep_<int> labeledMask, float *perimeterRes)
		{
			//Declare a shared array called 'lookup'. This will hold the lookup table. Each block will have this lookup table in its shared memory
			__shared__ float lookup[16];
			lookup[8] = 0.70710678118;
			lookup[4] = 0.70710678118;
			lookup[2] = 0.70710678118;
			lookup[1] = 0.70710678118;
			lookup[3] = 1.0;
			lookup[6] = 1.0;
			lookup[9] = 1.0;
			lookup[12] = 1.0;
			lookup[7] = 0.70710678118;
			lookup[11] = 0.70710678118;
			lookup[13] = 0.70710678118;
			lookup[14] = 0.70710678118;
			lookup[10] = 1.41421356237;
			lookup[5] = 1.41421356237;
			lookup[0] = 0.0;
			lookup[15] = 0.0;
			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

			//Declare shared array for the perimeter that each thread encounters. Initialize it by zeroing it out.
			__shared__ float thread_perimeter[32];
			thread_perimeter[threadIdx.x] = 0.0;
			//Declare a shared mask array for each block. 
			__shared__ int mask[32];

			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

			//Label of current component, maxX and maxY of current bounding box		
			int label = boundingBoxInfo[blockIdx.x];
			int minX = boundingBoxInfo[compCount + blockIdx.x];
			int minY = boundingBoxInfo[3 * compCount + blockIdx.x];
			int maxX = boundingBoxInfo[2 * compCount + blockIdx.x];
			int maxY = boundingBoxInfo[4 * compCount + blockIdx.x];

			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
			
			//Walk through the centre of the bounding box. From xmin to xmax-1
			for(int x = minX + threadIdx.x ; x < maxX ; x+=blockDim.x)
			{
				for(int y = minY ; y < maxY ; y++)
				{
					mask[threadIdx.x] = 0;
					mask[threadIdx.x] = (labeledMask.ptr(y)[x] == label); //(0,0)
					mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y)[x+1] == label );//(1,0)
					mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[x+1] == label );//(1,1)
					mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[x] == label );//(0,1)
	
					thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
				}
			}

	
			//Walk through the top and bottom edges of the bounding box.
			for(int x = minX + threadIdx.x ; x < maxX ; x+=blockDim.x)
			{
				//Top row : Read->leftshift->read->leftshiftby2
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(maxY)[x] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(maxY)[x+1] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
	
				//Bottom row : leftshiftby2->read->leftshift->read
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(minY)[x+1] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(minY)[x] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
	

			
			//Walk through the left and right edges of the bounding box
			for(int y = minY + threadIdx.x ; y < maxY ; y+=blockDim.x)
			{
				//Left edge : leftshift->read->leftshift->read->leftshift
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(y)[minX] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[minX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
				
				//Right edge : read->leftshiftby3->read
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(y)[maxX] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[maxX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			


			//Corners
			if(threadIdx.x == 0) //Bottom left corner (0,0)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(minY)[minX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
				
			}
			if(threadIdx.x == 8) //Bottom right corner (1,0)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(minY)[maxX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			if(threadIdx.x == 16) // Top right corner (1,1)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(maxY)[maxX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			if(threadIdx.x == 24) // Top left  corner (0,1)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(maxY)[minX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			__syncthreads();


	
			//Now, we do a parallel reduction using sequential addressing
			unsigned int s;
			for(s = blockDim.x /2 ; s > 0 ; s >>= 1)
			{
				if(threadIdx.x < s)
				{
					thread_perimeter[threadIdx.x] += thread_perimeter[threadIdx.x + s];
				}
				__syncthreads();
			}
			if (threadIdx.x == 0) perimeterRes[blockIdx.x] = thread_perimeter[0];

			
		}
		
		__global__ void ellipse(const int* boundingBoxInfo , const int compCount , const cv::gpu::PtrStep_ <int> labeledMask , int *areaRes , float *majorAxis , float* minorAxis , float* ecc)
		{
			//Create shared arrays for sx , sy , sxy , ssqx , ssqy
			__shared__ float sx[32]; sx[threadIdx.x] = 0;
			__shared__ float sy[32]; sy[threadIdx.x] = 0;
			__shared__ float sxy[32]; sxy[threadIdx.x] = 0;
			__shared__ float ssqx[32]; ssqx[threadIdx.x] = 0;
			__shared__ float ssqy[32]; ssqy[threadIdx.x] = 0;
			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
			//fix the parameters according to the blockId
			int label = boundingBoxInfo[blockIdx.x];
			int minX = boundingBoxInfo[compCount + blockIdx.x];
			int maxX = boundingBoxInfo[2 * compCount + blockIdx.x];
			float midX = (float)(minX+maxX)/2.0;
			int minY = boundingBoxInfo[3 * compCount + blockIdx.x];
			int maxY = boundingBoxInfo[4 * compCount + blockIdx.x];
			float midY = (float)(minY+maxY)/2.0;
			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
			//Walk through the labeled Image
			for( int x = minX + threadIdx.x ; x <= maxX ; x+=blockDim.x)
			{
				float cx = (float)x - midX;
				for( int y = minY ; y <=maxY ; y++)
				{
					float cy = (float)y - midY;
					bool temp = (labeledMask.ptr(y))[x] == label;
					sx[threadIdx.x] += ( temp ? cx : 0);
					sy[threadIdx.x] += ( temp ? cy : 0);
					sxy[threadIdx.x] += ( temp ? cx*cy : 0);
					ssqx[threadIdx.x] += ( temp ? cx*cx : 0);
					ssqy[threadIdx.x] += ( temp ? cy*cy : 0);
				}
			}
			__syncthreads();
			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
			
			//Now do a parallel scan to complete the summation
			unsigned int s;
			for(s = blockDim.x /2 ; s > 0 ; s >>= 1)
			{
				if(threadIdx.x < s)
				{
					sx[threadIdx.x] += sx[threadIdx.x + s];
					sy[threadIdx.x] += sy[threadIdx.x + s];
					sxy[threadIdx.x] += sxy[threadIdx.x + s];
					ssqx[threadIdx.x] += ssqx[threadIdx.x + s];
					ssqy[threadIdx.x] += ssqy[threadIdx.x + s];
				}
				__syncthreads();
			}
			//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
			
			//Now do the remaining calculations
			if (threadIdx.x == 0)
			{
				float frac = 1.0 / 12.0;
				float root = sqrtf(8.0);
				float area = (float)areaRes[blockIdx.x];
				float xbar = (float)sx[0] / area;
				float ybar = (float)sy[0] / area;
				float mxx = (float)ssqx[0]/area - xbar*xbar+ frac;
				float myy = (float)ssqy[0]/area - ybar*ybar+ frac;
				float mxy = (float)sxy[0]/area - xbar*ybar;
	
				float delta = sqrtf((mxx-myy)*(mxx-myy) + 4.0 * mxy * mxy); //discriminant = sqrt(b*b-4*a*c)
				majorAxis[blockIdx.x] = root*sqrtf(mxx+myy+delta);
				minorAxis[blockIdx.x] = root*sqrtf(mxx+myy-delta);
				ecc[blockIdx.x] = sqrtf(fabsf(majorAxis[blockIdx.x] * majorAxis[blockIdx.x] - minorAxis[blockIdx.x] * minorAxis[blockIdx.x]))/majorAxis[blockIdx.x];
				
			}
	
				
			
		}
	
		
	
		__global__ void big_features(const int *boundingBoxInfo, int compCount, const cv::gpu::PtrStep_<int> labeledMask, int *areaRes, float* perimeterRes , float* majorAxis , float* minorAxis , float* ecc)
		{
			/*****************************************FIRST DECLARE THE SHARED ARRAYS FOR ALL THE FEATURES****************************************************/
			__shared__ int thread_area[32]; //shared array for thread area
			thread_area[threadIdx.x] = 0;

			__shared__ float lookup[16]; //shared array lookup table for thread perimeter
			lookup[8] = 0.70710678118;
			lookup[4] = 0.70710678118;
			lookup[2] = 0.70710678118;
			lookup[1] = 0.70710678118;
			lookup[3] = 1.0;
			lookup[6] = 1.0;
			lookup[9] = 1.0;
			lookup[12] = 1.0;
			lookup[7] = 0.70710678118;
			lookup[11] = 0.70710678118;
			lookup[13] = 0.70710678118;
			lookup[14] = 0.70710678118;
			lookup[10] = 1.41421356237;
			lookup[5] = 1.41421356237;
			lookup[0] = 0.0;
			lookup[15] = 0.0;

			__shared__ float thread_perimeter[32]; //shared array for thread perimeter
			thread_perimeter[threadIdx.x] = 0.0;
			
			__shared__ int mask[32]; //shared array mask for perimeter
	
			__shared__ float sx[32]; sx[threadIdx.x] = 0; //shared arrays for ellipse calculations
			__shared__ float sy[32]; sy[threadIdx.x] = 0;
			__shared__ float sxy[32]; sxy[threadIdx.x] = 0;
			__shared__ float ssqx[32]; ssqx[threadIdx.x] = 0;
			__shared__ float ssqy[32]; ssqy[threadIdx.x] = 0;
			
			/******************************************NOW DECLARE ALL THE DETAILS RELATED TO BOUNDING BOX**********************************************/
			int label = boundingBoxInfo[blockIdx.x];

			int minX = boundingBoxInfo[compCount + blockIdx.x];
			int maxX = boundingBoxInfo[2 * compCount + blockIdx.x];
			float midX = (float)(minX+maxX)/2.0;

			int minY = boundingBoxInfo[3 * compCount + blockIdx.x];
			int maxY = boundingBoxInfo[4 * compCount + blockIdx.x];
			float midY = (float)(minY+maxY)/2.0;
	
			/******************************************NOW WALK THROUGH THE IMAGE FOR ONE IMAGE AT A TIME**************************************************/

			///////////////////////////////////////////////////////////////////////////AREA//////////////////////////////////////////////////////////////////
			for(int x = minX +threadIdx.x; x <= maxX ; x+=blockDim.x)
			{
				for(int y = minY ; y <= maxY ; y++)
				{
					thread_area[threadIdx.x] += ((labeledMask.ptr(y))[x] == label ? 1 : 0);
				}
			}

			__syncthreads();
	
			/////////////////////////////////////////////////////////////////////////PERIMETER//////////////////////////////////////////////////////////////
			//Walk through the centre of the bounding box. From xmin to xmax-1
			for(int x = minX + threadIdx.x ; x < maxX ; x+=blockDim.x)
			{
				for(int y = minY ; y < maxY ; y++)
				{
					mask[threadIdx.x] = 0;
					mask[threadIdx.x] = (labeledMask.ptr(y)[x] == label); //(0,0)
					mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y)[x+1] == label );//(1,0)
					mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[x+1] == label );//(1,1)
					mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[x] == label );//(0,1)
	
					thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
				}
			}

	
			//Walk through the top and bottom edges of the bounding box.
			for(int x = minX + threadIdx.x ; x < maxX ; x+=blockDim.x)
			{
				//Top row : Read->leftshift->read->leftshiftby2
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(maxY)[x] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(maxY)[x+1] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
	
				//Bottom row : leftshiftby2->read->leftshift->read
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(minY)[x+1] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(minY)[x] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
	

			
			//Walk through the left and right edges of the bounding box
			for(int y = minY + threadIdx.x ; y < maxY ; y+=blockDim.x)
			{
				//Left edge : leftshift->read->leftshift->read->leftshift
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(y)[minX] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[minX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
				
				//Right edge : read->leftshiftby3->read
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(y)[maxX] == label);
				mask[threadIdx.x] = (mask[threadIdx.x] << 1) | (labeledMask.ptr(y+1)[maxX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			


			//Corners
			if(threadIdx.x == 0) //Bottom left corner (0,0)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(minY)[minX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
				
			}
			if(threadIdx.x == 8) //Bottom right corner (1,0)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(minY)[maxX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			if(threadIdx.x == 16) // Top right corner (1,1)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(maxY)[maxX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			if(threadIdx.x == 24) // Top left  corner (0,1)
			{
				mask[threadIdx.x] = 0;
				mask[threadIdx.x] = (labeledMask.ptr(maxY)[minX] == label);
				thread_perimeter[threadIdx.x] += lookup[mask[threadIdx.x]];
			}
			__syncthreads();
	
			/////////////////////////////////////////////////////////ELLIPSE/////////////////////////////////////////////////////////////////////////
			//Walk through the labeled Image
			for( int x = minX + threadIdx.x ; x <= maxX ; x+=blockDim.x)
			{
				float cx = (float)x - midX;
				for( int y = minY ; y <=maxY ; y++)
				{
					float cy = (float)y - midY;
					bool temp = (labeledMask.ptr(y))[x] == label;
					sx[threadIdx.x] += ( temp ? cx : 0);
					sy[threadIdx.x] += ( temp ? cy : 0);
					sxy[threadIdx.x] += ( temp ? cx*cy : 0);
					ssqx[threadIdx.x] += ( temp ? cx*cx : 0);
					ssqy[threadIdx.x] += ( temp ? cy*cy : 0);
				}
			}
			__syncthreads();
	
			/************************************************PARALLEL SCAN OPERATION FOR ALL THE SHARED ARRAYS********************************************************/
			unsigned int s;
			for(s = blockDim.x /2 ; s > 0 ; s >>= 1)
			{
				if(threadIdx.x < s)
				{
					//Area
					thread_area[threadIdx.x] += thread_area[threadIdx.x + s];	
				
					//Perimeter
					thread_perimeter[threadIdx.x] += thread_perimeter[threadIdx.x + s];

					//Ellipse
					sx[threadIdx.x] += sx[threadIdx.x + s];
					sy[threadIdx.x] += sy[threadIdx.x + s];
					sxy[threadIdx.x] += sxy[threadIdx.x + s];
					ssqx[threadIdx.x] += ssqx[threadIdx.x + s];
					ssqy[threadIdx.x] += ssqy[threadIdx.x + s];
				}
				__syncthreads();
			}
	
			/*********************************************************************CONSOLIDATE!!!!*******************************************************************/
			if (threadIdx.x == 0)
			{
				
				//Area
				areaRes[blockIdx.x] = thread_area[0];
				
				//Perimeter
				perimeterRes[blockIdx.x] = thread_perimeter[0];

				//Ellipse
				float frac = 1.0 / 12.0;
				float root = sqrtf(8.0);
				float area = (float)areaRes[blockIdx.x];
				float xbar = (float)sx[0] / area;
				float ybar = (float)sy[0] / area;
				float mxx = (float)ssqx[0]/area - xbar*xbar+ frac;
				float myy = (float)ssqy[0]/area - ybar*ybar+ frac;
				float mxy = (float)sxy[0]/area - xbar*ybar;
				float delta = sqrtf((mxx-myy)*(mxx-myy) + 4.0 * mxy * mxy); //discriminant = sqrt(b*b-4*a*c)

				majorAxis[blockIdx.x] = root*sqrtf(mxx+myy+delta);
				minorAxis[blockIdx.x] = root*sqrtf(mxx+myy-delta);
				ecc[blockIdx.x] = sqrtf(majorAxis[blockIdx.x] * majorAxis[blockIdx.x] - minorAxis[blockIdx.x] * minorAxis[blockIdx.x])/majorAxis[blockIdx.x];
				
			}
	
			
		}

		__global__ void extentratio(const int *boundingBoxInfo , const int compCount , const int *areaRes , float *extent_ratio)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if(tid < compCount)
			{
				int xmin = boundingBoxInfo[compCount + tid];
				int xmax = boundingBoxInfo[2 * compCount + tid];
				int ymin = boundingBoxInfo[3 * compCount + tid];
				int ymax = boundingBoxInfo[4 * compCount + tid];
				
				extent_ratio[tid] = (float)areaRes[tid] / (float)((xmax-xmin+1) * (ymax-ymin+1));
			}
		}
	
		__global__ void circularity(const int compCount , const int *areaRes , const float *perimeterRes , float *circ)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if(tid < compCount)
			{
				circ[tid] = (4.0 * 3.14159265359 * (float)areaRes[tid]) / (perimeterRes[tid] * perimeterRes[tid]) ;
			}
		}


		__global__ void small_features(const int *boundingBoxInfo , const int compCount , const int *areaRes , const float *perimeterRes , float *extent_ratio , float *circ)
		{
			int tid = blockDim.x * blockIdx.x + threadIdx.x;
			if( tid < compCount)
			{
				int xmin = boundingBoxInfo[compCount + tid];
				int xmax = boundingBoxInfo[2 * compCount + tid];
				int ymin = boundingBoxInfo[3 * compCount + tid];
				int ymax = boundingBoxInfo[4 * compCount + tid];
				
				extent_ratio[tid] = (float)areaRes[tid] / (float)((xmax-xmin+1) * (ymax-ymin+1));
				circ[tid] = (4.0 * 3.14159265359 * (float)areaRes[tid]) / (perimeterRes[tid] * perimeterRes[tid]);
			}
		}




////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
		void AreaCaller(const int* boundingBoxInfo , const int compCount ,const cv::gpu::PtrStep_<int> labeledMask, int *areaRes, cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid(compCount, 1);
			area<<<grid,threads,0,stream>>>(boundingBoxInfo, compCount, labeledMask,areaRes);

			cudaGetLastError();
			
			if(stream == 0)
				cudaDeviceSynchronize();

		}

		void PerimeterCaller(const int* boundingBoxInfo , const int compCount ,const cv::gpu::PtrStep_<int> labeledMask , float *perimeterRes, cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid(compCount,1);
			perimeter <<< grid, threads, 0 ,stream >>>(boundingBoxInfo, compCount , labeledMask, perimeterRes);
			
			cudaGetLastError();

			if(stream == 0)
				cudaDeviceSynchronize();
		}
	
		void EllipseCaller(const int* boundingBoxInfo , const int compCount , const cv::gpu::PtrStep_ <int> labeledMask , int *areaRes , float *majorAxis , float *minorAxis , float *ecc, cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid(compCount , 1);
			ellipse <<<grid , threads , 0 , stream >>>(boundingBoxInfo , compCount , labeledMask , areaRes , majorAxis , minorAxis , ecc);
			
			cudaGetLastError();
	
			if(stream == 0)
				cudaDeviceSynchronize();
		}
	
		void ExtentRatioCaller(const int *boundingBoxInfo , const int compCount , const int *areaRes , float *extent_ratio , cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid((compCount + 32 -1)/32 , 1);
			extentratio <<< grid , threads , 0 , stream >>>(boundingBoxInfo , compCount , areaRes , extent_ratio);
	
			cudaGetLastError();
	
			if(stream == 0)
				cudaDeviceSynchronize(); 
		}
	
		void CircularityCaller(const int compCount , const int *areaRes , const float *perimeterRes , float *circ, cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid((compCount + 32 - 1)/32 , 1);
			circularity <<< grid , threads , 0 , stream >>>(compCount , areaRes , perimeterRes , circ);
		
			cudaGetLastError();
			
			if(stream == 0)
				cudaDeviceSynchronize();
		}
	
		void BigFeaturesCaller(const int* boundingBoxInfo , const int compCount , const cv::gpu::PtrStep_<int> labeledMask , int* areaRes , float* perimeterRes , float* majorAxis , float* minorAxis , float* ecc, cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid(compCount , 1);
			big_features <<< grid , threads , 0 , stream >>> (boundingBoxInfo , compCount , labeledMask , areaRes , perimeterRes , majorAxis , minorAxis , ecc );
	
			cudaGetLastError();
			if(stream == 0)
				cudaDeviceSynchronize();
		}
	
	
		void SmallFeaturesCaller(const int *boundingBoxInfo , const int compCount , const int *areaRes , const float *perimeterRes , float *extent_ratio , float* circ , cudaStream_t stream)
		{
			dim3 threads(32,1);
			dim3 grid((compCount + 32 -1)/32 , 1);
			small_features <<<grid , threads , 0 , stream >>> (boundingBoxInfo , compCount , areaRes , perimeterRes , extent_ratio , circ);
	
			cudaGetLastError();
			if(stream == 0)
				cudaDeviceSynchronize();
		}
		
	}
}
