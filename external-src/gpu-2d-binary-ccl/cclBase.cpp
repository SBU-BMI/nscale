#include "ccl.h"
#include "cudaBasicUtils.h"
#include <cuda_runtime_api.h>
#include "cclFunctions.h"
#include "profiler.h"


CCLBase::CCLBase()
{
	m_threadsX = 16;
	m_threadsY = 16;
	m_imgWidth = 0;
	m_imgHeight = 0;
	m_dataWidth = 0;
	m_dataHeight = 0;
	m_log2DataWidth = 0;

	m_numRegions = 0;
}

CCLBase::~CCLBase()
{
}

bool CCLBase::Init(int imgSizeX, int imgSizeY)
{
	m_imgWidth = CudaBasicUtils::RoundUp(m_threadsX*m_threadsX, imgSizeX);
	m_imgHeight = CudaBasicUtils::RoundUp(m_threadsY*m_threadsY, imgSizeY);
	int dataWidth = CudaBasicUtils::FirstPow2(m_imgWidth);
	int dataHeight = CudaBasicUtils::FirstPow2(m_imgHeight);
	if(dataWidth <= m_dataWidth && dataHeight <= m_dataHeight) 
		return true;
	m_dataWidth = dataWidth;
	m_dataHeight = dataHeight;
	m_log2DataWidth = CudaBasicUtils::Log2Base(m_dataWidth);
	for(int i=0;i<3;++i) {
		if(!m_label2DData[i].Create2D(dataWidth*sizeof(int), dataHeight))
			return false;
	}
	if(!m_int1Buf.Create(sizeof(int)))
		return false;
	if(!m_int1HostBuf.Create(sizeof(int), false))
		return false;
	if(!m_labelLinBuffer.Create(sizeof(int)*dataWidth*dataHeight))
		return false;
	return true;
}

bool CCLBase::ReindexLabels()
{
	//make all region ids (labels) in a range [0..numOfRegions] .. all elements that do not belong to any region will have an id == -1	
	//first prepare the regions for a scan in order to determine the number of unique connected regions
	if(!m_labelLinBuffer.SetZeroData())
		return false;
	if(!cclPrepareLabelsForScan(m_labelLinBuffer.GetData(), sCudaBuffer2D(m_label2DData[0].GetData(), m_label2DData[0].GetPitch()), m_threadsX, m_threadsY, m_imgWidth, m_imgHeight, m_dataWidth, m_log2DataWidth))
		return false;
	//apply the scan	
	if(!m_scan.ScanInclusive(eScan_CUDA_Type_Int, &m_labelLinBuffer, &m_labelLinBuffer, m_dataWidth*m_dataHeight))
		return false;	
	//get the first element from the buffer to determine the number of regions
	if(!m_int1HostBuf.CopyFrom(&m_labelLinBuffer, 0, 0, sizeof(int)))
		return false;
	m_numRegions = *(int*)m_int1HostBuf.GetData();	

	printf("number of regions = %d\n", m_numRegions);

	if(m_numRegions == 0) return true;
	//reindex the regions based on the computed data
	if(!cclReindexLabels(sCudaBuffer2D(m_label2DData[1].GetData(), m_label2DData[1].GetPitch()), sCudaBuffer2D(m_label2DData[0].GetData(), m_label2DData[0].GetPitch()), m_labelLinBuffer.GetData(), m_numRegions, m_threadsX, m_threadsY, m_imgWidth, m_imgHeight, m_dataWidth, m_log2DataWidth))
		return false;		
	return true;
}

int CCLBase::GetRegionIndexOnPos(int x, int y)
{
	//load the region index on a given position
	if(x < 0 || y < 0) return -1;
	if(x >= m_imgWidth || y >= m_imgHeight) return -1;
	int srcOffset[3] = { sizeof(int)*x, y, 0 };
	int dstOffset[3] = {0, 0, 0};
	int size[3] = { sizeof(int), 1, 1};
	if(!m_int1HostBuf.CopyFrom(GetConnectedRegionsBuffer(), srcOffset, dstOffset, size))
		return -1;
	return ((int*)m_int1HostBuf.GetData())[0];
}

void CCLBase::OnStartFindLabels()
{
	if(g_profiler.IsEnabled()) {
		cudaThreadSynchronize();		
		g_profiler.StartTime(0, false);
	}
}

void CCLBase::OnEndFindLabels()
{
	if(g_profiler.IsEnabled()) {
		static int count = 0;
		cudaThreadSynchronize();		
		count++;
		if((count % 10) == 0) {
			g_profiler.EndTime(0, "time");
			g_profiler.ClearTime(0);
		}
		else g_profiler.EndTime(0);
	}
}
