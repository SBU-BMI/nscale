#ifndef _LABEL_H_
#define _LABEL_H_

#include "cudaBuffer.h"
#define WP 32


#define BM (WP-1)
#define dt unsigned short
#define dts 2

#define TGTG  1
#define TGNTG 0

#define LBMAX 0xffff
#define LB2MAX 0xffffffff

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

extern void label_512(CudaBuffer* pt, CudaBuffer* pt2, CudaBuffer* ps, CudaBuffer* b, CudaBuffer* b2, CudaBuffer* glabel, uint h, uint bn, CudaBuffer* eb);
extern void label_1024(CudaBuffer* pt, CudaBuffer* pt2, CudaBuffer* ps, CudaBuffer* b, CudaBuffer* b2, CudaBuffer* glabel, uint h, uint bn, CudaBuffer* eb);
extern void label_2048(CudaBuffer* pt, CudaBuffer* pt2, CudaBuffer* ps, CudaBuffer* b, CudaBuffer* b2, CudaBuffer* glabel, uint h, uint bn, CudaBuffer* eb);

extern void label_zhl(CudaBuffer* pt, CudaBuffer* pt2, CudaBuffer* ps, CudaBuffer* b, CudaBuffer* b2, CudaBuffer* glabel, uint h, uint bn, CudaBuffer* eb);

#endif
