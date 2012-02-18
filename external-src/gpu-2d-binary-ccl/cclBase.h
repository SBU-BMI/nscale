#ifndef _FIND_CONNECTED_REGIONS_BASE_H_
#define _FIND_CONNECTED_REGIONS_BASE_H_

#include "cudaBuffer.h"
#include "cudaScan.h"

//the main class that solves the CCL problem.

class CCLBase
{
public:
	CCLBase();
	virtual ~CCLBase();
	//find connected regions (components) on a segmented data of dimensions imgSizeX and imgSizeY
	virtual bool FindRegions(int imgSizeX, int imgSizeY, CudaBuffer* segmentedData, int connectivity) = 0;
	//returns the number of detected connected regions 
	int GetNumConnectedRegions() const { return m_numRegions; }
	//get the buffer that contains a regionId for each data element
	virtual CudaBuffer* GetConnectedRegionsBuffer() { return &m_label2DData[1]; }
	int GetImgWidth() const { return m_imgWidth; }
	int GetImgHeight() const { return m_imgHeight; }
	int GetDataWidth() const { return m_dataWidth; }
	int GetDataHeight() const { return m_dataHeight; }

	virtual int GetRegionIndexOnPos(int x, int y);
protected:	
	virtual bool ReindexLabels();
	virtual bool Init(int imgSizeX, int imgSizeY);

	void OnStartFindLabels();
	void OnEndFindLabels();
protected:
	CudaBuffer* m_segmentedData;
	CudaBuffer m_label2DData[3];
	CudaBuffer m_int1Buf;
	CudaBuffer m_int1HostBuf;
	CudaBuffer m_labelLinBuffer;
	CudaScan m_scan;

	int m_threadsX;
	int m_threadsY;

	int m_imgWidth;
	int m_imgHeight;
	int m_dataWidth;
	int m_dataHeight;
	int m_log2DataWidth;

	//number of detected connected regions
	int m_numRegions;
};

#endif
