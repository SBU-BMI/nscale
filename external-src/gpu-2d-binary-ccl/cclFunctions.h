#ifndef _CCL_FUNCTIONS_H_
#define _CCL_FUNCTIONS_H_

#include "CUDAFunctionsBase.h"

extern "C" bool cclSolveCCLShared(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inSegBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth, int connectivity);
extern "C" bool cclMergeEquivalenceTrees(sCudaBuffer2D inOutLabelsBuf, sCudaBuffer2D inSegBuf, int* dChanged, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth, int subBlockSize, int connectivity);
extern "C" bool cclFlattenEquivalenceTrees(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inLabelsBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth);
extern "C" bool cclPrepareLabelsForScan(void* outBuf, sCudaBuffer2D inLabelsBuf, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth);
extern "C" bool cclReindexLabels(sCudaBuffer2D outLabelsBuf, sCudaBuffer2D inLabelsBuf, void* inScanBuf, int numComponents, int threadsX, int threadsY, int imgWidth, int imgHeight, int dataWidth, int log2DataWidth);

#endif
