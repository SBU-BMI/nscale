#include "ccl.h"
#include "cudaBasicUtils.h"
#include "cclFunctions.h"
#include <cuda_runtime_api.h>
#include <stdio.h>

extern float gpu2time;

CCL::CCL()
{
	
}

CCL::~CCL()
{
}

bool CCL::FindRegions(int imgSizeX, int imgSizeY, CudaBuffer* segmentedData, int connectivity)
{
	if(!Init(imgSizeX, imgSizeY))
		return false;	
	
	//number of detected connected components (regions)
	m_numRegions = 0;
	m_segmentedData = segmentedData;

	//used for timing
	OnStartFindLabels();
	float t0 = 0.0f;
	float t1 = 0.0f;
	float t2 = 0.0f;
	float t3 = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );

	cudaEventRecord( start, 0 );
	//call KERNEL 1 - compute local CCL solution
	if(!cclSolveCCLShared(sCudaBuffer2D(m_label2DData[0].GetData(), m_label2DData[0].GetPitch()),							
						  sCudaBuffer2D(m_segmentedData->GetData(), m_segmentedData->GetPitch()), 
						  m_threadsX, m_threadsY,
						  m_imgWidth, m_imgHeight, m_dataWidth, m_log2DataWidth, connectivity))
		   return false;	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &t0, start, stop );
	
	cudaEventRecord( start, 0 );
	//apply the merging scheme
	if(!cclMergeEquivalenceTrees
							   (sCudaBuffer2D(m_label2DData[0].GetData(), m_label2DData[0].GetPitch()),                                 
								sCudaBuffer2D(m_segmentedData->GetData(), m_segmentedData->GetPitch()), 
								(int*)m_int1Buf.GetData(),
								m_threadsX, m_threadsY,
								m_imgWidth, m_imgHeight, m_dataWidth, m_log2DataWidth, m_threadsX, connectivity))
		return false;	
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &t1, start, stop );

	cudaEventRecord( start, 0 );
	//final update - flattening of the equivalence trees
	//Kernel 4
	if(!cclFlattenEquivalenceTrees(sCudaBuffer2D(m_label2DData[0].GetData(), m_label2DData[0].GetPitch()),								 
								 sCudaBuffer2D(m_label2DData[0].GetData(), m_label2DData[0].GetPitch()),
								 m_threadsX, m_threadsY,
								 m_imgWidth, m_imgHeight, m_dataWidth, m_log2DataWidth))
		 return false;			
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &t2, start, stop );
	float tt = t0 + t1 + t2;
	gpu2time = tt;
	OnEndFindLabels();

	if(!m_label2DData[1].CopyFrom(&m_label2DData[0]))
			return false;
	//the labels are now merged and all continous areas are detected but the labels are not correctly indexed so we need to reindex them
	cudaEventRecord( start, 0 );
	bool res = ReindexLabels();
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize(stop);
	cudaEventElapsedTime( &t3, start, stop );

	printf("gpu alg 2 used %f ms, [%f, %f, %f, %f]\n", tt, t0, t1, t2, t3); 
	return res;
	
}
