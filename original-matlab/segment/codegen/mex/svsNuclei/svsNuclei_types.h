/*
 * svsNuclei_types.h
 *
 * Code generation for function 'svsNuclei'
 *
 * C source code generated on: Tue Jun 21 16:24:30 2011
 *
 */

#ifndef __SVSNUCLEI_TYPES_H__
#define __SVSNUCLEI_TYPES_H__

/* Type Definitions */
#ifndef struct_emxArray__common
#define struct_emxArray__common
typedef struct emxArray__common
{
    void *data;
    int32_T *size;
    int32_T allocatedSize;
    int32_T numDimensions;
    boolean_T canFreeData;
} emxArray__common;
#endif
#ifndef struct_emxArray_char_T
#define struct_emxArray_char_T
typedef struct emxArray_char_T
{
    char_T *data;
    int32_T *size;
    int32_T allocatedSize;
    int32_T numDimensions;
    boolean_T canFreeData;
} emxArray_char_T;
#endif
#ifndef struct_emxArray_int32_T
#define struct_emxArray_int32_T
typedef struct emxArray_int32_T
{
    int32_T *data;
    int32_T *size;
    int32_T allocatedSize;
    int32_T numDimensions;
    boolean_T canFreeData;
} emxArray_int32_T;
#endif
#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T
typedef struct emxArray_real_T
{
    real_T *data;
    int32_T *size;
    int32_T allocatedSize;
    int32_T numDimensions;
    boolean_T canFreeData;
} emxArray_real_T;
#endif
typedef struct
{
    struct
    {
        real_T imR2G[16777216];
        uint32_T water[16777216];
        boolean_T b_imR2G[16777216];
        boolean_T rbc[16777216];
        boolean_T bw[16777216];
        uint8_T rc[16777216];
        uint8_T rc_recon[16777216];
    } f0;
    struct
    {
        real_T L[16777216];
        uint8_T I[50331648];
        boolean_T BW[16777216];
        uint8_T R[16777216];
        uint8_T G[16777216];
        uint8_T B[16777216];
    } f1;
} svsNucleiStackData;

#endif
/* End of code generation (svsNuclei_types.h) */
