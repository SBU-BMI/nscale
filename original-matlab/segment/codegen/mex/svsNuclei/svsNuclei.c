/*
 * svsNuclei.c
 *
 * Code generation for function 'svsNuclei'
 *
 * C source code generated on: Tue Jun 21 16:24:30 2011
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "svsNuclei.h"
#include "svsNuclei_emxutil.h"
#include "ind2sub.h"
#include "find.h"
#include "svsNuclei_mexutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = { 55, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo b_emlrtRSI = { 58, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo c_emlrtRSI = { 59, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo d_emlrtRSI = { 63, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo e_emlrtRSI = { 69, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo f_emlrtRSI = { 106, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo g_emlrtRSI = { 107, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo h_emlrtRSI = { 108, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo i_emlrtRSI = { 109, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo j_emlrtRSI = { 110, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo k_emlrtRSI = { 111, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo l_emlrtRSI = { 113, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo n_emlrtRSI = { 140, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo o_emlrtRSI = { 146, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo p_emlrtRSI = { 147, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo q_emlrtRSI = { 179, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo r_emlrtRSI = { 183, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo s_emlrtRSI = { 189, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo t_emlrtRSI = { 192, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo u_emlrtRSI = { 197, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo v_emlrtRSI = { 202, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo w_emlrtRSI = { 204, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo x_emlrtRSI = { 210, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo y_emlrtRSI = { 213, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo ab_emlrtRSI = { 220, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo bb_emlrtRSI = { 222, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo cb_emlrtRSI = { 227, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo db_emlrtRSI = { 229, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo eb_emlrtRSI = { 234, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo fb_emlrtRSI = { 237, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo gb_emlrtRSI = { 243, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo hb_emlrtRSI = { 252, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo ib_emlrtRSI = { 254, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo jb_emlrtRSI = { 262, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo kb_emlrtRSI = { 263, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRSInfo pb_emlrtRSI = { 11, "eml_li_find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_li_find.m" };
static emlrtRSInfo qb_emlrtRSI = { 14, "eml_li_find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_li_find.m" };
static emlrtMCInfo emlrtMCI = { 55, 7, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo b_emlrtMCI = { 58, 13, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo c_emlrtMCI = { 63, 21, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo d_emlrtMCI = { 63, 13, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo e_emlrtMCI = { 106, 8, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo f_emlrtMCI = { 111, 5, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo g_emlrtMCI = { 113, 5, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo i_emlrtMCI = { 147, 14, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo j_emlrtMCI = { 179, 15, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo k_emlrtMCI = { 183, 16, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo l_emlrtMCI = { 189, 11, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo m_emlrtMCI = { 192, 11, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo n_emlrtMCI = { 202, 11, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo o_emlrtMCI = { 213, 10, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo p_emlrtMCI = { 220, 18, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo q_emlrtMCI = { 222, 16, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo r_emlrtMCI = { 227, 12, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo s_emlrtMCI = { 229, 15, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo t_emlrtMCI = { 234, 11, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo u_emlrtMCI = { 243, 17, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo v_emlrtMCI = { 252, 13, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo w_emlrtMCI = { 262, 11, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo x_emlrtMCI = { 197, 13, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo y_emlrtMCI = { 263, 13, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtMCInfo cb_emlrtMCI = { 14, 5, "eml_li_find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_li_find.m" };
static emlrtRTEInfo emlrtRTEI = { 2, 10, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRTEInfo d_emlrtRTEI = { 121, 18, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRTEInfo e_emlrtRTEI = { 140, 5, "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m" };
static emlrtRTEInfo i_emlrtRTEI = { 20, 9, "eml_li_find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_li_find.m" };
static emlrtBCInfo emlrtBCI = { 1, 16777216, 107, 17, "R", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo b_emlrtBCI = { 1, 16777216, 108, 17, "G", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo c_emlrtBCI = { 1, 16777216, 109, 17, "B", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo d_emlrtBCI = { 1, 50331648, 110, 7, "", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo e_emlrtBCI = { 1, 50331648, 110, 7, "", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo f_emlrtBCI = { 1, 50331648, 110, 7, "", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo g_emlrtBCI = { 1, 16777216, 237, 5, "distance", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtBCInfo h_emlrtBCI = { 1, 16777216, 254, 5, "seg_big", "svsNuclei", "/home/tcpan/PhD/path/src/nscale/src/matlab/segment/svsNuclei.m", 0 };
static emlrtDCInfo emlrtDCI = { 20, 34, "eml_li_find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_li_find.m", 4 };
static emlrtBCInfo k_emlrtBCI = { -1, -1, 28, 13, "", "eml_li_find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_li_find.m", 0 };

/* Function Declarations */
static void b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, uint8_T y[50331648]);
static const mxArray *b_emlrt_marshallOut(const boolean_T u[16777216]);
static const mxArray *bwareaopen(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *bwdist(const mxArray *b, emlrtMCInfo *location);
static const mxArray *bwlabel(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *bwperim(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *bwselect(const mxArray *b, const mxArray *c, const mxArray *d, const mxArray *e, emlrtMCInfo *location);
static real_T c_emlrt_marshallIn(const mxArray *b_mrdivide, const char_T *identifier);
static const mxArray *c_emlrt_marshallOut(const char_T u[5]);
static real_T d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId);
static const mxArray *d_emlrt_marshallOut(emxArray_real_T *u);
static void e_emlrt_marshallIn(const mxArray *b_bwperim, const char_T *identifier, boolean_T y[16777216]);
static const mxArray *e_emlrt_marshallOut(const uint8_T u[16777216]);
static void eml_li_find(const boolean_T x[16777216], emxArray_int32_T *y);
static void emlrt_marshallIn(const mxArray *b_imread, const char_T *identifier, uint8_T y[50331648]);
static const mxArray *emlrt_marshallOut(real_T u);
static void f_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, boolean_T y[16777216]);
static const mxArray *f_emlrt_marshallOut(const real_T u[361]);
static void g_emlrt_marshallIn(const mxArray *b_imreconstruct, const char_T *identifier, uint8_T y[16777216]);
static const mxArray *g_emlrt_marshallOut(const real_T u[16777216]);
static void h_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, uint8_T y[16777216]);
static void i_emlrt_marshallIn(const mxArray *b_bwlabel, const char_T *identifier, real_T y[16777216]);
static const mxArray *imdilate(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *imfill(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *imhmin(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *imopen(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *imread(const mxArray *b, emlrtMCInfo *location);
static const mxArray *imreconstruct(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static void imwrite(const mxArray *b, const mxArray *c, const mxArray *d, const mxArray *e, emlrtMCInfo *location);
static const mxArray *ismember(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static void j_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, real_T y[16777216]);
static void k_emlrt_marshallIn(const mxArray *b_regionprops, const char_T *identifier);
static void l_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId);
static void m_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId);
static const mxArray *mrdivide(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static void n_emlrt_marshallIn(const mxArray *b_watershed, const char_T *identifier, uint32_T y[16777216]);
static const mxArray *numel(const mxArray *b, emlrtMCInfo *location);
static void o_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, uint32_T y[16777216]);
static void r_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, uint8_T ret[50331648]);
static const mxArray *regionprops(const mxArray *b, const mxArray *c, emlrtMCInfo *location);
static const mxArray *rgb2gray(const mxArray *b, emlrtMCInfo *location);
static real_T s_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId);
static void save(const mxArray *b, const mxArray *c, const mxArray *d, const mxArray *e, const mxArray *f, emlrtMCInfo *location);
static void segNucleiMorphMeanshift(svsNucleiStackData *SD, const uint8_T color_img[50331648], real_T L[16777216]);
static void t_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, boolean_T ret[16777216]);
static void u_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, uint8_T ret[16777216]);
static void v_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, real_T ret[16777216]);
static void w_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId);
static const mxArray *watershed(const mxArray *b, emlrtMCInfo *location);
static void x_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, uint32_T ret[16777216]);

/* Function Definitions */

static void b_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, uint8_T y[50331648])
{
    r_emlrt_marshallIn(emlrtAlias(u), parentId, y);
    emlrtDestroyArray(&u);
}

static const mxArray *b_emlrt_marshallOut(const boolean_T u[16777216])
{
    const mxArray *y;
    static const int32_T iv25[2] = { 4096, 4096 };
    const mxArray *m5;
    y = NULL;
    m5 = mxCreateLogicalArray(2, iv25);
    emlrtInitLogicalArray(16777216, m5, u);
    emlrtAssign(&y, m5);
    return y;
}

static const mxArray *bwareaopen(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m22;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m22, 2, pArrays, "bwareaopen", TRUE, location);
}

static const mxArray *bwdist(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    const mxArray *m24;
    pArray = b;
    return emlrtCallMATLAB(1, &m24, 1, &pArray, "bwdist", TRUE, location);
}

static const mxArray *bwlabel(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m20;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m20, 2, pArrays, "bwlabel", TRUE, location);
}

static const mxArray *bwperim(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m15;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m15, 2, pArrays, "bwperim", TRUE, location);
}

static const mxArray *bwselect(const mxArray *b, const mxArray *c, const mxArray *d, const mxArray *e, emlrtMCInfo *location)
{
    const mxArray *pArrays[4];
    const mxArray *m16;
    pArrays[0] = b;
    pArrays[1] = c;
    pArrays[2] = d;
    pArrays[3] = e;
    return emlrtCallMATLAB(1, &m16, 4, pArrays, "bwselect", TRUE, location);
}

static real_T c_emlrt_marshallIn(const mxArray *b_mrdivide, const char_T *identifier)
{
    real_T y;
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    y = d_emlrt_marshallIn(emlrtAlias(b_mrdivide), &thisId);
    emlrtDestroyArray(&b_mrdivide);
    return y;
}

static const mxArray *c_emlrt_marshallOut(const char_T u[5])
{
    const mxArray *y;
    static const int32_T iv26[2] = { 1, 5 };
    const mxArray *m6;
    y = NULL;
    m6 = mxCreateCharArray(2, iv26);
    emlrtInitCharArray(5, m6, u);
    emlrtAssign(&y, m6);
    return y;
}

static real_T d_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId)
{
    real_T y;
    y = s_emlrt_marshallIn(emlrtAlias(u), parentId);
    emlrtDestroyArray(&u);
    return y;
}

static const mxArray *d_emlrt_marshallOut(emxArray_real_T *u)
{
    const mxArray *y;
    const mxArray *m7;
    real_T (*pData)[];
    int32_T i4;
    int32_T i;
    y = NULL;
    m7 = mxCreateNumericArray(1, u->size, mxDOUBLE_CLASS, mxREAL);
    pData = (real_T (*)[])mxGetPr(m7);
    i4 = 0;
    for (i = 0; i < u->size[0]; i++) {
        (*pData)[i4] = u->data[i];
        i4++;
    }
    emlrtAssign(&y, m7);
    return y;
}

static void e_emlrt_marshallIn(const mxArray *b_bwperim, const char_T *identifier, boolean_T y[16777216])
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    f_emlrt_marshallIn(emlrtAlias(b_bwperim), &thisId, y);
    emlrtDestroyArray(&b_bwperim);
}

static const mxArray *e_emlrt_marshallOut(const uint8_T u[16777216])
{
    const mxArray *y;
    static const int32_T iv27[2] = { 4096, 4096 };
    const mxArray *m8;
    uint8_T (*pData)[];
    int32_T i;
    y = NULL;
    m8 = mxCreateNumericArray(2, (int32_T *)&iv27, mxUINT8_CLASS, mxREAL);
    pData = (uint8_T (*)[])mxGetData(m8);
    for (i = 0; i < 16777216; i++) {
        (*pData)[i] = u[i];
    }
    emlrtAssign(&y, m8);
    return y;
}

/*
 * 
 */
static void eml_li_find(const boolean_T x[16777216], emxArray_int32_T *y)
{
    int32_T k;
    int32_T i;
    const mxArray *b_y;
    const mxArray *m3;
    int32_T j;
    EMLRTPUSHRTSTACK(&pb_emlrtRSI);
    k = 0;
    for (i = 0; i < 16777216; i++) {
        if (x[i]) {
            k++;
        }
    }
    EMLRTPOPRTSTACK(&pb_emlrtRSI);
    if (k <= 16777216) {
    } else {
        EMLRTPUSHRTSTACK(&qb_emlrtRSI);
        b_y = NULL;
        m3 = mxCreateString("Assertion failed.");
        emlrtAssign(&b_y, m3);
        error(b_y, &cb_emlrtMCI);
        EMLRTPOPRTSTACK(&qb_emlrtRSI);
    }
    emlrtNonNegativeCheckR2011a((real_T)k, &emlrtDCI, &emlrtContextGlobal);
    j = y->size[0];
    y->size[0] = k;
    emxEnsureCapacity((emxArray__common *)y, j, (int32_T)sizeof(int32_T), &i_emlrtRTEI);
    j = 1;
    for (i = 0; i < 16777216; i++) {
        if (x[i]) {
            y->data[emlrtDynamicBoundsCheckR2011a(j, 1, y->size[0], &k_emlrtBCI, &emlrtContextGlobal) - 1] = i + 1;
            j++;
        }
    }
}

static void emlrt_marshallIn(const mxArray *b_imread, const char_T *identifier, uint8_T y[50331648])
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    b_emlrt_marshallIn(emlrtAlias(b_imread), &thisId, y);
    emlrtDestroyArray(&b_imread);
}

static const mxArray *emlrt_marshallOut(real_T u)
{
    const mxArray *y;
    const mxArray *m4;
    y = NULL;
    m4 = mxCreateDoubleScalar(u);
    emlrtAssign(&y, m4);
    return y;
}

static void f_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, boolean_T y[16777216])
{
    t_emlrt_marshallIn(emlrtAlias(u), parentId, y);
    emlrtDestroyArray(&u);
}

static const mxArray *f_emlrt_marshallOut(const real_T u[361])
{
    const mxArray *y;
    static const int32_T iv28[2] = { 19, 19 };
    const mxArray *m9;
    real_T (*pData)[];
    int32_T i;
    y = NULL;
    m9 = mxCreateNumericArray(2, (int32_T *)&iv28, mxDOUBLE_CLASS, mxREAL);
    pData = (real_T (*)[])mxGetPr(m9);
    for (i = 0; i < 361; i++) {
        (*pData)[i] = u[i];
    }
    emlrtAssign(&y, m9);
    return y;
}

static void g_emlrt_marshallIn(const mxArray *b_imreconstruct, const char_T *identifier, uint8_T y[16777216])
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    h_emlrt_marshallIn(emlrtAlias(b_imreconstruct), &thisId, y);
    emlrtDestroyArray(&b_imreconstruct);
}

static const mxArray *g_emlrt_marshallOut(const real_T u[16777216])
{
    const mxArray *y;
    static const int32_T iv29[2] = { 4096, 4096 };
    const mxArray *m10;
    real_T (*pData)[];
    int32_T i;
    y = NULL;
    m10 = mxCreateNumericArray(2, (int32_T *)&iv29, mxDOUBLE_CLASS, mxREAL);
    pData = (real_T (*)[])mxGetPr(m10);
    for (i = 0; i < 16777216; i++) {
        (*pData)[i] = u[i];
    }
    emlrtAssign(&y, m10);
    return y;
}

static void h_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, uint8_T y[16777216])
{
    u_emlrt_marshallIn(emlrtAlias(u), parentId, y);
    emlrtDestroyArray(&u);
}

static void i_emlrt_marshallIn(const mxArray *b_bwlabel, const char_T *identifier, real_T y[16777216])
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    j_emlrt_marshallIn(emlrtAlias(b_bwlabel), &thisId, y);
    emlrtDestroyArray(&b_bwlabel);
}

static const mxArray *imdilate(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m23;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m23, 2, pArrays, "imdilate", TRUE, location);
}

static const mxArray *imfill(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m19;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m19, 2, pArrays, "imfill", TRUE, location);
}

static const mxArray *imhmin(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m25;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m25, 2, pArrays, "imhmin", TRUE, location);
}

static const mxArray *imopen(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m17;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m17, 2, pArrays, "imopen", TRUE, location);
}

static const mxArray *imread(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    const mxArray *m11;
    pArray = b;
    return emlrtCallMATLAB(1, &m11, 1, &pArray, "imread", TRUE, location);
}

static const mxArray *imreconstruct(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m18;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m18, 2, pArrays, "imreconstruct", TRUE, location);
}

static void imwrite(const mxArray *b, const mxArray *c, const mxArray *d, const mxArray *e, emlrtMCInfo *location)
{
    const mxArray *pArrays[4];
    pArrays[0] = b;
    pArrays[1] = c;
    pArrays[2] = d;
    pArrays[3] = e;
    emlrtCallMATLAB(0, NULL, 4, pArrays, "imwrite", TRUE, location);
}

static const mxArray *ismember(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m21;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m21, 2, pArrays, "ismember", TRUE, location);
}

static void j_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, real_T y[16777216])
{
    v_emlrt_marshallIn(emlrtAlias(u), parentId, y);
    emlrtDestroyArray(&u);
}

static void k_emlrt_marshallIn(const mxArray *b_regionprops, const char_T *identifier)
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    l_emlrt_marshallIn(emlrtAlias(b_regionprops), &thisId);
    emlrtDestroyArray(&b_regionprops);
}

static void l_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId)
{
    emlrtMsgIdentifier thisId;
    static const char * fieldNames[1] = { "Area" };
    thisId.fParent = parentId;
    emlrtCheckStructR2011a(parentId, u, 1, fieldNames, 0U, 0);
    thisId.fIdentifier = "Area";
    m_emlrt_marshallIn(emlrtAlias(emlrtGetField(u, 0, "Area")), &thisId);
    emlrtDestroyArray(&u);
}

static void m_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId)
{
    w_emlrt_marshallIn(emlrtAlias(u), parentId);
    emlrtDestroyArray(&u);
}

static const mxArray *mrdivide(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m14;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m14, 2, pArrays, "mrdivide", TRUE, location);
}

static void n_emlrt_marshallIn(const mxArray *b_watershed, const char_T *identifier, uint32_T y[16777216])
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    o_emlrt_marshallIn(emlrtAlias(b_watershed), &thisId, y);
    emlrtDestroyArray(&b_watershed);
}

static const mxArray *numel(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    const mxArray *m13;
    pArray = b;
    return emlrtCallMATLAB(1, &m13, 1, &pArray, "numel", TRUE, location);
}

static void o_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, uint32_T y[16777216])
{
    x_emlrt_marshallIn(emlrtAlias(u), parentId, y);
    emlrtDestroyArray(&u);
}

static void r_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, uint8_T ret[50331648])
{
    int32_T i;
    static const int16_T iv30[3] = { 4096, 4096, 3 };
    int32_T iv31[3];
    int32_T i5;
    int32_T i6;
    for (i = 0; i < 3; i++) {
        iv31[i] = iv30[i];
    }
    emlrtCheckBuiltInR2011a(msgId, src, "uint8", FALSE, 3U, iv31);
    for (i = 0; i < 3; i++) {
        for (i5 = 0; i5 < 4096; i5++) {
            for (i6 = 0; i6 < 4096; i6++) {
                ret[(i6 + (i5 << 12)) + (i << 24)] = (*(uint8_T (*)[50331648])mxGetData(src))[(i6 + (i5 << 12)) + (i << 24)];
            }
        }
    }
    emlrtDestroyArray(&src);
}

static const mxArray *regionprops(const mxArray *b, const mxArray *c, emlrtMCInfo *location)
{
    const mxArray *pArrays[2];
    const mxArray *m27;
    pArrays[0] = b;
    pArrays[1] = c;
    return emlrtCallMATLAB(1, &m27, 2, pArrays, "regionprops", TRUE, location);
}

static const mxArray *rgb2gray(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    const mxArray *m12;
    pArray = b;
    return emlrtCallMATLAB(1, &m12, 1, &pArray, "rgb2gray", TRUE, location);
}

static real_T s_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId)
{
    real_T ret;
    emlrtCheckBuiltInR2011a(msgId, src, "double", FALSE, 0U, 0);
    ret = *(real_T *)mxGetData(src);
    emlrtDestroyArray(&src);
    return ret;
}

static void save(const mxArray *b, const mxArray *c, const mxArray *d, const mxArray *e, const mxArray *f, emlrtMCInfo *location)
{
    const mxArray *pArrays[5];
    pArrays[0] = b;
    pArrays[1] = c;
    pArrays[2] = d;
    pArrays[3] = e;
    pArrays[4] = f;
    emlrtCallMATLAB(0, NULL, 5, pArrays, "save", TRUE, location);
}

/*
 * function [f,L] = segNucleiMorphMeanshift(color_img)
 */
static void segNucleiMorphMeanshift(svsNucleiStackData *SD, const uint8_T color_img[50331648], real_T L[16777216])
{
    int32_T i1;
    int32_T loop_ub;
    emxArray_real_T *ind;
    emxArray_real_T *cols;
    emxArray_real_T *b_ind;
    static const real_T se10[361] = { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0 };
    const mxArray *rc_open = NULL;
    uint32_T qY;
    static const char_T u[5] = { 'h', 'o', 'l', 'e', 's' };
    const mxArray *y;
    static const int32_T iv5[2] = { 4096, 4096 };
    const mxArray *m1;
    const mxArray *b_y;
    const mxArray *c_y;
    static const int32_T iv6[2] = { 1, 4 };
    static const char_T cv4[4] = { 'A', 'r', 'e', 'a' };
    const mxArray *d_y;
    static const int32_T iv7[2] = { 4096, 4096 };
    real_T (*pData)[];
    const mxArray *e_y;
    static const int32_T iv8[2] = { 0, 0 };
    emxArray_real_T *c_ind;
    const mxArray *f_y;
    static const int32_T iv9[2] = { 4096, 4096 };
    emxArray_real_T *b_u;
    const mxArray *g_y;
    emxArray_real_T *c_u;
    const mxArray *h_y;
    const mxArray *i_y;
    const mxArray *j_y;
    static const int32_T iv10[2] = { 4096, 4096 };
    const mxArray *k_y;
    static const int32_T iv11[2] = { 1, 5 };
    static const char_T cv5[5] = { 'h', 'o', 'l', 'e', 's' };
    const mxArray *l_y;
    static const int32_T iv12[2] = { 4096, 4096 };
    const mxArray *m_y;
    static const int32_T iv13[2] = { 3, 3 };
    static const int8_T iv14[9] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
    const mxArray *n_y;
    static const int32_T iv15[2] = { 4096, 4096 };
    const mxArray *o_y;
    const mxArray *p_y;
    static const int32_T iv16[2] = { 4096, 4096 };
    const mxArray *q_y;
    static const int32_T iv17[2] = { 3, 3 };
    const mxArray *r_y;
    static const int32_T iv18[2] = { 4096, 4096 };
    emxArray_int32_T *r3;
    const mxArray *s_y;
    static const int32_T iv19[2] = { 4096, 4096 };
    const mxArray *t_y;
    const mxArray *u_y;
    static const int32_T iv20[2] = { 4096, 4096 };
    const mxArray *v_y;
    static const int32_T iv21[2] = { 4096, 4096 };
    const mxArray *w_y;
    const mxArray *x_y;
    static const int32_T iv22[2] = { 4096, 4096 };
    const mxArray *y_y;
    static const int32_T iv23[2] = { 1, 4 };
    emlrtHeapReferenceStackEnterFcn();
    /* 'svsNuclei:122' f = []; */
    /* TONY */
    /* 'svsNuclei:124' coder.extrinsic('bwselect', 'imopen', 'imreconstruct', 'imfill', 'bwlabel',... */
    /* 'svsNuclei:125'         'regionprops', 'bwareaopen', 'imdilate', 'bwdist', 'imhmin', 'watershed',... */
    /* 'svsNuclei:126'         'ismember'); */
    /* 'svsNuclei:127' L = zeros(size(color_img, 1), size(color_img,2), 'double'); */
    /* END TONY */
    /* 'svsNuclei:130' r = color_img( :, :, 1); */
    /* 'svsNuclei:131' g = color_img( :, :, 2); */
    /* 'svsNuclei:132' b = color_img( :, :, 3); */
    /* T1=2.5; T2=2; */
    /* 'svsNuclei:135' T1=5; */
    /* 'svsNuclei:135' T2=4; */
    /* 'svsNuclei:137' imR2G = double(r)./(double(g)+eps); */
    for (i1 = 0; i1 < 4096; i1++) {
        for (loop_ub = 0; loop_ub < 4096; loop_ub++) {
            SD->f0.imR2G[loop_ub + (i1 << 12)] = (real_T)color_img[loop_ub + (i1 << 12)] / ((real_T)color_img[16777216 + (loop_ub + (i1 << 12))] + 2.2204460492503131E-16);
        }
    }
    /* 'svsNuclei:138' bw1 = imR2G > T1; */
    /* 'svsNuclei:139' bw2 = imR2G > T2; */
    /* 'svsNuclei:140' ind = find(bw1); */
    EMLRTPUSHRTSTACK(&n_emlrtRSI);
    for (i1 = 0; i1 < 16777216; i1++) {
        SD->f0.b_imR2G[i1] = (SD->f0.imR2G[i1] > 5.0);
    }
    emxInit_real_T(&ind, 1, &e_emlrtRTEI, TRUE);
    find(SD->f0.b_imR2G, ind);
    EMLRTPOPRTSTACK(&n_emlrtRSI);
    /*  TONY */
    /* 'svsNuclei:143' bw = false(size(color_img,1), size(color_img,2)); */
    /* 'svsNuclei:144' rbc = false(size(imR2G)); */
    for (i1 = 0; i1 < 16777216; i1++) {
        SD->f0.rbc[i1] = FALSE;
    }
    /* 'svsNuclei:145' if ~isempty(ind) */
    emxInit_real_T(&cols, 1, &d_emlrtRTEI, TRUE);
    if (!(ind->size[0] == 0)) {
        emxInit_real_T(&b_ind, 1, &d_emlrtRTEI, TRUE);
        /* 'svsNuclei:146' [rows, cols]=ind2sub(size(imR2G),ind); */
        EMLRTPUSHRTSTACK(&o_emlrtRSI);
        i1 = b_ind->size[0];
        b_ind->size[0] = ind->size[0];
        emxEnsureCapacity((emxArray__common *)b_ind, i1, (int32_T)sizeof(real_T), &d_emlrtRTEI);
        loop_ub = ind->size[0] - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
            b_ind->data[i1] = ind->data[i1];
        }
        ind2sub(b_ind, ind, cols);
        EMLRTPOPRTSTACK(&o_emlrtRSI);
        /* 'svsNuclei:147' bw = bwselect(bw2, cols, rows, 8); */
        EMLRTPUSHRTSTACK(&p_emlrtRSI);
        emxFree_real_T(&b_ind);
        for (i1 = 0; i1 < 16777216; i1++) {
            SD->f0.b_imR2G[i1] = (SD->f0.imR2G[i1] > 4.0);
        }
        e_emlrt_marshallIn(bwselect(b_emlrt_marshallOut(SD->f0.b_imR2G), d_emlrt_marshallOut(cols), d_emlrt_marshallOut(ind), emlrt_marshallOut(8.0), &i_emlrtMCI), "bwselect", SD->f0.bw);
        EMLRTPOPRTSTACK(&p_emlrtRSI);
        /* 'svsNuclei:148' rbc = bw & ((double(r)./(double(b)+eps)) > 1); */
        for (i1 = 0; i1 < 4096; i1++) {
            for (loop_ub = 0; loop_ub < 4096; loop_ub++) {
                SD->f0.rbc[loop_ub + (i1 << 12)] = (SD->f0.bw[loop_ub + (i1 << 12)] && ((real_T)color_img[loop_ub + (i1 << 12)] / ((real_T)color_img[33554432 + (loop_ub + (i1 << 12))] + 2.2204460492503131E-16) > 1.0));
            }
        }
    }
    /*     if ~isempty(ind) */
    /*         [rows, cols]=ind2sub(size(imR2G),ind); */
    /*         rbc = bwselect(bw2,cols,rows,8) & (double(r)./(double(b)+eps)>1); */
    /*     else */
    /*         rbc = zeros(size(imR2G)); */
    /*     end */
    /* END TONY */
    /* 'svsNuclei:158' rc = 255 - r; */
    for (i1 = 0; i1 < 4096; i1++) {
        for (loop_ub = 0; loop_ub < 4096; loop_ub++) {
            SD->f0.rc[loop_ub + (i1 << 12)] = (uint8_T)(255U - (uint32_T)color_img[loop_ub + (i1 << 12)]);
        }
    }
    /*  TONY */
    /* 'svsNuclei:160' se10 = [     0     0     0     0     1     1     1     1     1     1     1     1     1     1     1     0     0     0     0;... */
    /* 'svsNuclei:161'      0     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0     0;... */
    /* 'svsNuclei:162'      0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0;... */
    /* 'svsNuclei:163'      0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0;... */
    /* 'svsNuclei:164'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:165'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:166'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:167'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:168'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:169'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:170'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:171'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:172'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:173'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:174'      1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1;... */
    /* 'svsNuclei:175'      0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0;... */
    /* 'svsNuclei:176'      0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0;... */
    /* 'svsNuclei:177'      0     0     0     1     1     1     1     1     1     1     1     1     1     1     1     1     0     0     0;... */
    /* 'svsNuclei:178'      0     0     0     0     1     1     1     1     1     1     1     1     1     1     1     0     0     0     0]; */
    /* 'svsNuclei:179' rc_open = imopen(rc, se10); */
    EMLRTPUSHRTSTACK(&q_emlrtRSI);
    emlrtAssign(&rc_open, imopen(e_emlrt_marshallOut(SD->f0.rc), f_emlrt_marshallOut(se10), &j_emlrtMCI));
    EMLRTPOPRTSTACK(&q_emlrtRSI);
    /*     rc_open = imopen(rc, strel('disk',10)); */
    /* 'svsNuclei:181' rc_recon = zeros(size(rc), 'uint8'); */
    /*  END TONY */
    /* 'svsNuclei:183' rc_recon = imreconstruct(rc_open,rc); */
    EMLRTPUSHRTSTACK(&r_emlrtRSI);
    g_emlrt_marshallIn(imreconstruct(emlrtAlias(rc_open), e_emlrt_marshallOut(SD->f0.rc), &k_emlrtMCI), "imreconstruct", SD->f0.rc_recon);
    EMLRTPOPRTSTACK(&r_emlrtRSI);
    /* 'svsNuclei:184' diffIm = rc-rc_recon; */
    for (i1 = 0; i1 < 16777216; i1++) {
        loop_ub = (int32_T)SD->f0.rc[i1];
        qY = (uint32_T)loop_ub - (uint32_T)SD->f0.rc_recon[i1];
        if (qY > (uint32_T)loop_ub) {
            qY = 0U;
        }
        loop_ub = (int32_T)qY;
        SD->f0.rc[i1] = (uint8_T)loop_ub;
    }
    /* 'svsNuclei:186' G1=80; */
    /* 'svsNuclei:186' G2=45; */
    /*  default settings */
    /* G1=80; G2=30;  % 2nd run */
    /* 'svsNuclei:189' bw1 = imfill(diffIm>G1,'holes'); */
    EMLRTPUSHRTSTACK(&s_emlrtRSI);
    for (i1 = 0; i1 < 16777216; i1++) {
        SD->f0.b_imR2G[i1] = (SD->f0.rc[i1] > 80);
    }
    e_emlrt_marshallIn(imfill(b_emlrt_marshallOut(SD->f0.b_imR2G), c_emlrt_marshallOut(u), &l_emlrtMCI), "imfill", SD->f0.bw);
    EMLRTPOPRTSTACK(&s_emlrtRSI);
    /* CHANGE */
    /* 'svsNuclei:192' [L] = bwlabel(bw1, 8); */
    EMLRTPUSHRTSTACK(&t_emlrtRSI);
    y = NULL;
    m1 = mxCreateLogicalArray(2, iv5);
    emlrtInitLogicalArray(16777216, m1, SD->f0.bw);
    emlrtAssign(&y, m1);
    b_y = NULL;
    m1 = mxCreateDoubleScalar(8.0);
    emlrtAssign(&b_y, m1);
    i_emlrt_marshallIn(bwlabel(y, b_y, &m_emlrtMCI), "bwlabel", L);
    EMLRTPOPRTSTACK(&t_emlrtRSI);
    /* TONY */
    /* 'svsNuclei:194' stats = struct('Area', []); */
    /* 'svsNuclei:195' assert(isa(stats(1).Area, 'double')); */
    /* END TONY */
    /* 'svsNuclei:197' stats = regionprops(L, 'Area'); */
    EMLRTPUSHRTSTACK(&u_emlrtRSI);
    c_y = NULL;
    m1 = mxCreateCharArray(2, iv6);
    emlrtInitCharArray(4, m1, cv4);
    emlrtAssign(&c_y, m1);
    k_emlrt_marshallIn(regionprops(g_emlrt_marshallOut(L), c_y, &x_emlrtMCI), "regionprops");
    EMLRTPOPRTSTACK(&u_emlrtRSI);
    /* 'svsNuclei:198' areas = [stats.Area]; */
    /* CHANGE */
    /* 'svsNuclei:201' ind = find(areas>10 & areas<1000); */
    /* 'svsNuclei:202' bw1 = ismember(L,ind); */
    EMLRTPUSHRTSTACK(&v_emlrtRSI);
    d_y = NULL;
    m1 = mxCreateNumericArray(2, (int32_T *)&iv7, mxDOUBLE_CLASS, mxREAL);
    pData = (real_T (*)[])mxGetPr(m1);
    for (loop_ub = 0; loop_ub < 16777216; loop_ub++) {
        (*pData)[loop_ub] = L[loop_ub];
    }
    emlrtAssign(&d_y, m1);
    e_y = NULL;
    m1 = mxCreateNumericArray(2, (int32_T *)&iv8, mxDOUBLE_CLASS, mxREAL);
    emlrtAssign(&e_y, m1);
    e_emlrt_marshallIn(ismember(d_y, e_y, &n_emlrtMCI), "ismember", SD->f0.bw);
    EMLRTPOPRTSTACK(&v_emlrtRSI);
    /* 'svsNuclei:203' bw2 = diffIm>G2; */
    /* 'svsNuclei:204' ind = find(bw1); */
    EMLRTPUSHRTSTACK(&w_emlrtRSI);
    find(SD->f0.bw, ind);
    EMLRTPOPRTSTACK(&w_emlrtRSI);
    /* 'svsNuclei:206' if isempty(ind) */
    if (ind->size[0] == 0) {
    } else {
        emxInit_real_T(&c_ind, 1, &d_emlrtRTEI, TRUE);
        /* 'svsNuclei:210' [rows,cols] = ind2sub(size(diffIm),ind); */
        EMLRTPUSHRTSTACK(&x_emlrtRSI);
        i1 = c_ind->size[0];
        c_ind->size[0] = ind->size[0];
        emxEnsureCapacity((emxArray__common *)c_ind, i1, (int32_T)sizeof(real_T), &d_emlrtRTEI);
        loop_ub = ind->size[0] - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
            c_ind->data[i1] = ind->data[i1];
        }
        ind2sub(c_ind, ind, cols);
        EMLRTPOPRTSTACK(&x_emlrtRSI);
        /*  TONY */
        /* 'svsNuclei:212' bw = false(size(bw2)); */
        /* 'svsNuclei:213' bw = bwselect(bw2,cols,rows,8); */
        EMLRTPUSHRTSTACK(&y_emlrtRSI);
        f_y = NULL;
        m1 = mxCreateLogicalArray(2, iv9);
        emxFree_real_T(&c_ind);
        for (i1 = 0; i1 < 16777216; i1++) {
            SD->f0.b_imR2G[i1] = (SD->f0.rc[i1] > 45);
        }
        emxInit_real_T(&b_u, 1, &d_emlrtRTEI, TRUE);
        emlrtInitLogicalArray(16777216, m1, SD->f0.b_imR2G);
        emlrtAssign(&f_y, m1);
        i1 = b_u->size[0];
        b_u->size[0] = cols->size[0];
        emxEnsureCapacity((emxArray__common *)b_u, i1, (int32_T)sizeof(real_T), &d_emlrtRTEI);
        loop_ub = cols->size[0] - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
            b_u->data[i1] = cols->data[i1];
        }
        g_y = NULL;
        m1 = mxCreateNumericArray(1, b_u->size, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        i1 = 0;
        for (loop_ub = 0; loop_ub < b_u->size[0]; loop_ub++) {
            (*pData)[i1] = b_u->data[loop_ub];
            i1++;
        }
        emxFree_real_T(&b_u);
        emxInit_real_T(&c_u, 1, &d_emlrtRTEI, TRUE);
        emlrtAssign(&g_y, m1);
        i1 = c_u->size[0];
        c_u->size[0] = ind->size[0];
        emxEnsureCapacity((emxArray__common *)c_u, i1, (int32_T)sizeof(real_T), &d_emlrtRTEI);
        loop_ub = ind->size[0] - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
            c_u->data[i1] = ind->data[i1];
        }
        h_y = NULL;
        m1 = mxCreateNumericArray(1, c_u->size, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        i1 = 0;
        for (loop_ub = 0; loop_ub < c_u->size[0]; loop_ub++) {
            (*pData)[i1] = c_u->data[loop_ub];
            i1++;
        }
        emxFree_real_T(&c_u);
        emlrtAssign(&h_y, m1);
        i_y = NULL;
        m1 = mxCreateDoubleScalar(8.0);
        emlrtAssign(&i_y, m1);
        e_emlrt_marshallIn(bwselect(f_y, g_y, h_y, i_y, &o_emlrtMCI), "bwselect", SD->f0.bw);
        EMLRTPOPRTSTACK(&y_emlrtRSI);
        /* seg_norbc = bwselect(bw2,cols,rows,8) & ~rbc; */
        /* 'svsNuclei:215' seg_norbc = bw & ~rbc; */
        /* 'svsNuclei:216' seg_nohole = false(size(bw)); */
        /* 'svsNuclei:217' seg_open = false(size(bw)); */
        /* 'svsNuclei:218' se = [0 1 0; 1 1 1; 0 1 0]; */
        /* END TONY */
        /* 'svsNuclei:220' seg_nohole = imfill(seg_norbc,'holes'); */
        EMLRTPUSHRTSTACK(&ab_emlrtRSI);
        j_y = NULL;
        m1 = mxCreateLogicalArray(2, iv10);
        for (i1 = 0; i1 < 16777216; i1++) {
            SD->f0.b_imR2G[i1] = (SD->f0.bw[i1] && (!SD->f0.rbc[i1]));
        }
        emlrtInitLogicalArray(16777216, m1, SD->f0.b_imR2G);
        emlrtAssign(&j_y, m1);
        k_y = NULL;
        m1 = mxCreateCharArray(2, iv11);
        emlrtInitCharArray(5, m1, cv5);
        emlrtAssign(&k_y, m1);
        e_emlrt_marshallIn(imfill(j_y, k_y, &p_emlrtMCI), "imfill", SD->f0.bw);
        EMLRTPOPRTSTACK(&ab_emlrtRSI);
        /*     seg_open = imopen(seg_nohole,strel('disk',1)); */
        /* 'svsNuclei:222' seg_open = imopen(seg_nohole,se); */
        EMLRTPUSHRTSTACK(&bb_emlrtRSI);
        l_y = NULL;
        m1 = mxCreateLogicalArray(2, iv12);
        emlrtInitLogicalArray(16777216, m1, SD->f0.bw);
        emlrtAssign(&l_y, m1);
        m_y = NULL;
        m1 = mxCreateNumericArray(2, (int32_T *)&iv13, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        for (loop_ub = 0; loop_ub < 9; loop_ub++) {
            (*pData)[loop_ub] = (real_T)iv14[loop_ub];
        }
        emlrtAssign(&m_y, m1);
        e_emlrt_marshallIn(imopen(l_y, m_y, &q_emlrtMCI), "imopen", SD->f0.bw);
        EMLRTPOPRTSTACK(&bb_emlrtRSI);
        /* CHANGE */
        /* TONY */
        /* 'svsNuclei:226' bwao = false(size(seg_open)); */
        /* 'svsNuclei:227' bwao = bwareaopen(seg_open,30); */
        EMLRTPUSHRTSTACK(&cb_emlrtRSI);
        n_y = NULL;
        m1 = mxCreateLogicalArray(2, iv15);
        emlrtInitLogicalArray(16777216, m1, SD->f0.bw);
        emlrtAssign(&n_y, m1);
        o_y = NULL;
        m1 = mxCreateDoubleScalar(30.0);
        emlrtAssign(&o_y, m1);
        e_emlrt_marshallIn(bwareaopen(n_y, o_y, &r_emlrtMCI), "bwareaopen", SD->f0.bw);
        EMLRTPOPRTSTACK(&cb_emlrtRSI);
        /* 'svsNuclei:228' seg_big = false(size(bwao)); */
        /* 'svsNuclei:229' seg_big = imdilate(bwao, se); */
        EMLRTPUSHRTSTACK(&db_emlrtRSI);
        p_y = NULL;
        m1 = mxCreateLogicalArray(2, iv16);
        emlrtInitLogicalArray(16777216, m1, SD->f0.bw);
        emlrtAssign(&p_y, m1);
        q_y = NULL;
        m1 = mxCreateNumericArray(2, (int32_T *)&iv17, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        for (loop_ub = 0; loop_ub < 9; loop_ub++) {
            (*pData)[loop_ub] = (real_T)iv14[loop_ub];
        }
        emlrtAssign(&q_y, m1);
        e_emlrt_marshallIn(imdilate(p_y, q_y, &s_emlrtMCI), "imdilate", SD->f0.bw);
        EMLRTPOPRTSTACK(&db_emlrtRSI);
        /*     seg_big = imdilate(bwareaopen(seg_open,30),strel('disk',1)); */
        /* 'svsNuclei:231' not_seg_big = false(size(seg_big)); */
        /* 'svsNuclei:232' not_seg_big = ~seg_big; */
        for (i1 = 0; i1 < 16777216; i1++) {
            SD->f0.rbc[i1] = !SD->f0.bw[i1];
        }
        /* 'svsNuclei:233' bwd = zeros(size(seg_big), 'double'); */
        /* 'svsNuclei:234' bwd = bwdist(not_seg_big); */
        EMLRTPUSHRTSTACK(&eb_emlrtRSI);
        r_y = NULL;
        m1 = mxCreateLogicalArray(2, iv18);
        emlrtInitLogicalArray(16777216, m1, SD->f0.rbc);
        emlrtAssign(&r_y, m1);
        i_emlrt_marshallIn(bwdist(r_y, &t_emlrtMCI), "bwdist", SD->f0.imR2G);
        EMLRTPOPRTSTACK(&eb_emlrtRSI);
        /* 'svsNuclei:235' distance = zeros(size(seg_big), 'double'); */
        /* 'svsNuclei:236' distance = -bwd; */
        for (i1 = 0; i1 < 16777216; i1++) {
            SD->f0.imR2G[i1] = -SD->f0.imR2G[i1];
        }
        emxInit_int32_T(&r3, 1, &d_emlrtRTEI, TRUE);
        /* 'svsNuclei:237' distance(not_seg_big) = -Inf; */
        EMLRTPUSHRTSTACK(&fb_emlrtRSI);
        eml_li_find(SD->f0.rbc, r3);
        EMLRTPOPRTSTACK(&fb_emlrtRSI);
        loop_ub = r3->size[0] - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
            SD->f0.imR2G[emlrtBoundsCheckR2011a(r3->data[i1], &g_emlrtBCI, &emlrtContextGlobal) - 1] = rtMinusInf;
        }
        /*     distance = -bwdist(~seg_big); */
        /* distance(~seg_big) = -Inf; */
        /* 'svsNuclei:240' distance2 = zeros(size(distance), 'double'); */
        /* END TONY */
        /* 'svsNuclei:243' distance2 = imhmin(distance, 1); */
        EMLRTPUSHRTSTACK(&gb_emlrtRSI);
        s_y = NULL;
        m1 = mxCreateNumericArray(2, (int32_T *)&iv19, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        for (loop_ub = 0; loop_ub < 16777216; loop_ub++) {
            (*pData)[loop_ub] = SD->f0.imR2G[loop_ub];
        }
        emlrtAssign(&s_y, m1);
        t_y = NULL;
        m1 = mxCreateDoubleScalar(1.0);
        emlrtAssign(&t_y, m1);
        i_emlrt_marshallIn(imhmin(s_y, t_y, &u_emlrtMCI), "imhmin", SD->f0.imR2G);
        EMLRTPOPRTSTACK(&gb_emlrtRSI);
        /* lines=ones(size(distance)); */
        /* lines(watershed(distance2)==0)=0; */
        /* TONY clear distance %END TONY */
        /* TONY */
        /* 'svsNuclei:251' water = zeros(size(distance2), 'uint32'); */
        /* 'svsNuclei:252' water = watershed(distance2); */
        EMLRTPUSHRTSTACK(&hb_emlrtRSI);
        u_y = NULL;
        m1 = mxCreateNumericArray(2, (int32_T *)&iv20, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        for (loop_ub = 0; loop_ub < 16777216; loop_ub++) {
            (*pData)[loop_ub] = SD->f0.imR2G[loop_ub];
        }
        emlrtAssign(&u_y, m1);
        n_emlrt_marshallIn(watershed(u_y, &v_emlrtMCI), "watershed", SD->f0.water);
        EMLRTPOPRTSTACK(&hb_emlrtRSI);
        /*     seg_big(watershed(distance2)==0) = 0; */
        /* 'svsNuclei:254' seg_big(water==0) = 0; */
        EMLRTPUSHRTSTACK(&ib_emlrtRSI);
        for (i1 = 0; i1 < 16777216; i1++) {
            SD->f0.b_imR2G[i1] = (SD->f0.water[i1] == 0U);
        }
        eml_li_find(SD->f0.b_imR2G, r3);
        EMLRTPOPRTSTACK(&ib_emlrtRSI);
        loop_ub = r3->size[0] - 1;
        for (i1 = 0; i1 <= loop_ub; i1++) {
            SD->f0.bw[emlrtBoundsCheckR2011a(r3->data[i1], &h_emlrtBCI, &emlrtContextGlobal) - 1] = FALSE;
        }
        emxFree_int32_T(&r3);
        /* END TONY */
        /* 'svsNuclei:256' seg_nonoverlap = seg_big; */
        /* seg_nonoverlap = lines & seg_big; */
        /* TONY clear lines distance2 %END TONY */
        /* CHANGE */
        /* 'svsNuclei:262' [L] = bwlabel(seg_nonoverlap, 4); */
        EMLRTPUSHRTSTACK(&jb_emlrtRSI);
        v_y = NULL;
        m1 = mxCreateLogicalArray(2, iv21);
        emlrtInitLogicalArray(16777216, m1, SD->f0.bw);
        emlrtAssign(&v_y, m1);
        w_y = NULL;
        m1 = mxCreateDoubleScalar(4.0);
        emlrtAssign(&w_y, m1);
        i_emlrt_marshallIn(bwlabel(v_y, w_y, &w_emlrtMCI), "bwlabel", L);
        EMLRTPOPRTSTACK(&jb_emlrtRSI);
        /* 'svsNuclei:263' stats = regionprops(L, 'Area'); */
        EMLRTPUSHRTSTACK(&kb_emlrtRSI);
        x_y = NULL;
        m1 = mxCreateNumericArray(2, (int32_T *)&iv22, mxDOUBLE_CLASS, mxREAL);
        pData = (real_T (*)[])mxGetPr(m1);
        for (loop_ub = 0; loop_ub < 16777216; loop_ub++) {
            (*pData)[loop_ub] = L[loop_ub];
        }
        emlrtAssign(&x_y, m1);
        y_y = NULL;
        m1 = mxCreateCharArray(2, iv23);
        emlrtInitCharArray(4, m1, cv4);
        emlrtAssign(&y_y, m1);
        k_emlrt_marshallIn(regionprops(x_y, y_y, &y_emlrtMCI), "regionprops");
        EMLRTPOPRTSTACK(&kb_emlrtRSI);
        /* 'svsNuclei:264' areas = [stats.Area]; */
        /* CHANGE */
        /* 'svsNuclei:267' ind = find(areas>20 & areas<1000); */
        /* 'svsNuclei:269' if isempty(ind) */
    }
    emxFree_real_T(&cols);
    emxFree_real_T(&ind);
    emlrtDestroyArray(&rc_open);
    emlrtHeapReferenceStackLeaveFcn();
}

static void t_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, boolean_T ret[16777216])
{
    int32_T i;
    int32_T iv32[2];
    int32_T i7;
    for (i = 0; i < 2; i++) {
        iv32[i] = 4096;
    }
    emlrtCheckBuiltInR2011a(msgId, src, "logical", FALSE, 2U, iv32);
    for (i = 0; i < 4096; i++) {
        for (i7 = 0; i7 < 4096; i7++) {
            ret[i7 + (i << 12)] = (*(boolean_T (*)[16777216])mxGetLogicals(src))[i7 + (i << 12)];
        }
    }
    emlrtDestroyArray(&src);
}

static void u_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, uint8_T ret[16777216])
{
    int32_T i;
    int32_T iv33[2];
    int32_T i8;
    for (i = 0; i < 2; i++) {
        iv33[i] = 4096;
    }
    emlrtCheckBuiltInR2011a(msgId, src, "uint8", FALSE, 2U, iv33);
    for (i = 0; i < 4096; i++) {
        for (i8 = 0; i8 < 4096; i8++) {
            ret[i8 + (i << 12)] = (*(uint8_T (*)[16777216])mxGetData(src))[i8 + (i << 12)];
        }
    }
    emlrtDestroyArray(&src);
}

static void v_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, real_T ret[16777216])
{
    int32_T i;
    int32_T iv34[2];
    int32_T i9;
    for (i = 0; i < 2; i++) {
        iv34[i] = 4096;
    }
    emlrtCheckBuiltInR2011a(msgId, src, "double", FALSE, 2U, iv34);
    for (i = 0; i < 4096; i++) {
        for (i9 = 0; i9 < 4096; i9++) {
            ret[i9 + (i << 12)] = (*(real_T (*)[16777216])mxGetData(src))[i9 + (i << 12)];
        }
    }
    emlrtDestroyArray(&src);
}

static void w_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId)
{
    int32_T i;
    int32_T iv35[2];
    for (i = 0; i < 2; i++) {
        iv35[i] = 0;
    }
    emlrtCheckBuiltInR2011a(msgId, src, "double", FALSE, 2U, iv35);
    emlrtDestroyArray(&src);
}

static const mxArray *watershed(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    const mxArray *m26;
    pArray = b;
    return emlrtCallMATLAB(1, &m26, 1, &pArray, "watershed", TRUE, location);
}

static void x_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, uint32_T ret[16777216])
{
    int32_T i;
    int32_T iv36[2];
    int32_T i10;
    for (i = 0; i < 2; i++) {
        iv36[i] = 4096;
    }
    emlrtCheckBuiltInR2011a(msgId, src, "uint32", FALSE, 2U, iv36);
    for (i = 0; i < 4096; i++) {
        for (i10 = 0; i10 < 4096; i10++) {
            ret[i10 + (i << 12)] = (*(uint32_T (*)[16777216])mxGetData(src))[i10 + (i << 12)];
        }
    }
    emlrtDestroyArray(&src);
}

/*
 * function svsNuclei(impath,filename,resultpath, folder,tile)
 */
void svsNuclei(svsNucleiStackData *SD, const emxArray_char_T *impath, const emxArray_char_T *filename, const emxArray_char_T *resultpath)
{
    emxArray_char_T *u;
    int32_T j;
    int32_T area_bg;
    const mxArray *y;
    const mxArray *m0;
    const mxArray *b_y;
    static const int32_T iv0[3] = { 4096, 4096, 3 };
    uint8_T (*pData)[];
    const mxArray *grayI = NULL;
    emxArray_real_T *x;
    const mxArray *c_y;
    real_T ratio;
    const mxArray *d_y;
    static const int32_T iv1[2] = { 4096, 4096 };
    const mxArray *e_y;
    emxArray_int32_T *r0;
    const mxArray *f_y;
    static const int32_T iv2[3] = { 4096, 4096, 3 };
    emxArray_char_T *b_u;
    static const char_T cv0[10] = { '.', 'g', 'r', 'i', 'd', '4', '.', 'j', 'p', 'g' };
    emxArray_char_T *c_u;
    const mxArray *g_y;
    const mxArray *h_y;
    static const int32_T iv3[2] = { 1, 7 };
    static const char_T cv1[7] = { 'Q', 'u', 'a', 'l', 'i', 't', 'y' };
    const mxArray *i_y;
    static const char_T cv2[10] = { '.', 'g', 'r', 'i', 'd', '4', '.', 'm', 'a', 't' };
    const mxArray *j_y;
    const mxArray *k_y;
    const mxArray *l_y;
    const mxArray *m_y;
    const mxArray *n_y;
    static const int32_T iv4[2] = { 1, 5 };
    static const char_T cv3[5] = { '-', 'v', '7', '.', '3' };
    emlrtHeapReferenceStackEnterFcn();
    emxInit_char_T(&u, 2, &emlrtRTEI, TRUE);
    /*  removed return var of "nuclei" */
    /* 'svsNuclei:5' if nargin==0 */
    /* ============START================ */
    /* 'svsNuclei:28' p.AN='NS-MORPH'; */
    /* TCGA dataset */
    /* 'svsNuclei:29' p.SN='1'; */
    /* 'svsNuclei:30' p.PA.THR=0.9; */
    /* 'svsNuclei:30' p.Par.T1=5; */
    /* 'svsNuclei:30' p.Par.T2=4; */
    /* 'svsNuclei:30' p.Par.G1=80; */
    /* 'svsNuclei:30' p.Par.G2=45; */
    /* p.SN='2'; */
    /* p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=30; */
    /* 'svsNuclei:33' p.SR='20x'; */
    /* 'svsNuclei:34' p.AR='20x'; */
    /* 'svsNuclei:35' p.CR='1'; */
    /*  p.AN='NS-MORPH';    %VALIDATION dataset */
    /*  p.SN='1'; */
    /*  p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=45; */
    /*  %p.SN='2'; */
    /*  %p.PA.THR=0.9;p.Par.T1=5;p.Par.T2=4;p.Par.G1=80;p.Par.G2=30; */
    /*  p.SR='40x';      %Scanning Resolution   %40X--18slides; %20x--necrosis */
    /*  p.AR='20x';      %Analysis Resolution */
    /*  p.CR='0.5'; */
    /* TONY - imread is not supported by matlab coder */
    /* 'svsNuclei:48' coder.extrinsic('imread', 'rgb2gray', 'tic', 'bwperim', 'imwrite', 'save'); */
    /* END TONY */
    /* TONY */
    /* 'svsNuclei:52' I=zeros(4096,4096,3, 'uint8'); */
    /* 'svsNuclei:53' assert(isa(I,'uint8')); */
    /* END TONY */
    /* 'svsNuclei:55' I=imread([impath,filename]); */
    EMLRTPUSHRTSTACK(&emlrtRSI);
    j = u->size[0] * u->size[1];
    u->size[0] = 1;
    u->size[1] = impath->size[1] + filename->size[1];
    emxEnsureCapacity((emxArray__common *)u, j, (int32_T)sizeof(char_T), &emlrtRTEI);
    area_bg = impath->size[1] - 1;
    for (j = 0; j <= area_bg; j++) {
        u->data[u->size[0] * j] = impath->data[impath->size[0] * j];
    }
    area_bg = filename->size[1] - 1;
    for (j = 0; j <= area_bg; j++) {
        u->data[u->size[0] * (j + impath->size[1])] = filename->data[filename->size[0] * j];
    }
    y = NULL;
    m0 = mxCreateCharArray(2, u->size);
    emlrtInitCharArray(u->size[1], m0, u->data);
    emlrtAssign(&y, m0);
    emlrt_marshallIn(imread(y, &emlrtMCI), "imread", SD->f1.I);
    EMLRTPOPRTSTACK(&emlrtRSI);
    /* 'svsNuclei:57' THR = 0.9; */
    /* 'svsNuclei:58' grayI = rgb2gray(I); */
    EMLRTPUSHRTSTACK(&b_emlrtRSI);
    b_y = NULL;
    m0 = mxCreateNumericArray(3, (int32_T *)&iv0, mxUINT8_CLASS, mxREAL);
    pData = (uint8_T (*)[])mxGetData(m0);
    emxFree_char_T(&u);
    for (area_bg = 0; area_bg < 50331648; area_bg++) {
        (*pData)[area_bg] = SD->f1.I[area_bg];
    }
    emlrtAssign(&b_y, m0);
    emlrtAssign(&grayI, rgb2gray(b_y, &b_emlrtMCI));
    EMLRTPOPRTSTACK(&b_emlrtRSI);
    /* 'svsNuclei:59' area_bg = length(find(I(:,:,1)>=220&I(:,:,2)>=220&I(:,:,3)>=220)); */
    EMLRTPUSHRTSTACK(&c_emlrtRSI);
    for (j = 0; j < 4096; j++) {
        for (area_bg = 0; area_bg < 4096; area_bg++) {
            SD->f1.BW[area_bg + (j << 12)] = ((SD->f1.I[area_bg + (j << 12)] >= 220) && (SD->f1.I[16777216 + (area_bg + (j << 12))] >= 220) && (SD->f1.I[33554432 + (area_bg + (j << 12))] >= 220));
        }
    }
    emxInit_real_T(&x, 1, &emlrtRTEI, TRUE);
    find(SD->f1.BW, x);
    if (x->size[0] == 0) {
        area_bg = 0;
    } else if (x->size[0] > 1) {
        area_bg = x->size[0];
    } else {
        area_bg = 1;
    }
    emxFree_real_T(&x);
    EMLRTPOPRTSTACK(&c_emlrtRSI);
    /* TONY */
    /* 'svsNuclei:61' ratio = zeros(1, 1, 'double'); */
    /* END TONY */
    /* 'svsNuclei:63' ratio = area_bg/numel(grayI); */
    EMLRTPUSHRTSTACK(&d_emlrtRSI);
    c_y = NULL;
    m0 = mxCreateDoubleScalar((real_T)area_bg);
    emlrtAssign(&c_y, m0);
    ratio = c_emlrt_marshallIn(mrdivide(c_y, numel(emlrtAlias(grayI), &c_emlrtMCI), &d_emlrtMCI), "mrdivide");
    EMLRTPOPRTSTACK(&d_emlrtRSI);
    /* 'svsNuclei:64' if ratio >= THR */
    if (ratio >= 0.9) {
    } else {
        /* TONY */
        /* 'svsNuclei:69' [f,L] = segNucleiMorphMeanshift(I); */
        EMLRTPUSHRTSTACK(&e_emlrtRSI);
        segNucleiMorphMeanshift(SD, SD->f1.I, SD->f1.L);
        EMLRTPOPRTSTACK(&e_emlrtRSI);
        /*     tic; */
        /*     [f,L] = segNucleiMorphMeanshift(I); */
        /*     t=toc; */
        /* END TONY */
        /*  BW = L>0; */
        /*  LL = zeros(size(BW)); */
        /*  boundaries = bwboundaries(BW); */
        /*  num = length(boundaries); */
        /*  */
        /*  for i = 1:num */
        /*      b = boundaries{i}; */
        /*  */
        /*      if size(b,1) > 15 */
        /*          b(:,2) = lowB(b(:,2)); */
        /*          b(:,1) = lowB(b(:,1)); */
        /*          if any(b(:)<0 | b(:)>4096) */
        /*            continue; */
        /*          end */
        /*          boundaries{i} = b; */
        /*      end */
        /*  */
        /*      tempBW = roipoly(BW,b(:,2),b(:,1)); */
        /*      if sum(tempBW(:)) == 0 */
        /*          continue; */
        /*      end */
        /*  */
        /*      LL(tempBW) = count; */
        /*      count = count + 1; */
        /*  end */
        /* TONY */
        /* 'svsNuclei:103' assert(isa(L, 'double')); */
        /* 'svsNuclei:104' BW = false(size(L)); */
        /* END TONY */
        /* 'svsNuclei:106' BW=bwperim(L>0,4); */
        EMLRTPUSHRTSTACK(&f_emlrtRSI);
        d_y = NULL;
        m0 = mxCreateLogicalArray(2, iv1);
        for (j = 0; j < 16777216; j++) {
            SD->f1.BW[j] = (SD->f1.L[j] > 0.0);
        }
        emlrtInitLogicalArray(16777216, m0, SD->f1.BW);
        emlrtAssign(&d_y, m0);
        e_y = NULL;
        m0 = mxCreateDoubleScalar(4.0);
        emlrtAssign(&e_y, m0);
        e_emlrt_marshallIn(bwperim(d_y, e_y, &e_emlrtMCI), "bwperim", SD->f1.BW);
        EMLRTPOPRTSTACK(&f_emlrtRSI);
        /* 'svsNuclei:107' R=I(:,:,1); */
        for (j = 0; j < 4096; j++) {
            memcpy((void *)&SD->f1.R[j << 12], (void *)&SD->f1.I[j << 12], sizeof(uint8_T) << 12);
        }
        emxInit_int32_T(&r0, 1, &emlrtRTEI, TRUE);
        /* 'svsNuclei:107' R(BW)=0; */
        EMLRTPUSHRTSTACK(&g_emlrtRSI);
        eml_li_find(SD->f1.BW, r0);
        EMLRTPOPRTSTACK(&g_emlrtRSI);
        area_bg = r0->size[0] - 1;
        for (j = 0; j <= area_bg; j++) {
            SD->f1.R[emlrtBoundsCheckR2011a(r0->data[j], &emlrtBCI, &emlrtContextGlobal) - 1] = 0;
        }
        /* 'svsNuclei:108' G=I(:,:,2); */
        for (j = 0; j < 4096; j++) {
            memcpy((void *)&SD->f1.G[j << 12], (void *)&SD->f1.I[16777216 + (j << 12)], sizeof(uint8_T) << 12);
        }
        /* 'svsNuclei:108' G(BW)=255; */
        EMLRTPUSHRTSTACK(&h_emlrtRSI);
        eml_li_find(SD->f1.BW, r0);
        EMLRTPOPRTSTACK(&h_emlrtRSI);
        area_bg = r0->size[0] - 1;
        for (j = 0; j <= area_bg; j++) {
            SD->f1.G[emlrtBoundsCheckR2011a(r0->data[j], &b_emlrtBCI, &emlrtContextGlobal) - 1] = MAX_uint8_T;
        }
        /* 'svsNuclei:109' B=I(:,:,3); */
        for (j = 0; j < 4096; j++) {
            memcpy((void *)&SD->f1.B[j << 12], (void *)&SD->f1.I[33554432 + (j << 12)], sizeof(uint8_T) << 12);
        }
        /* 'svsNuclei:109' B(BW)=0; */
        EMLRTPUSHRTSTACK(&i_emlrtRSI);
        eml_li_find(SD->f1.BW, r0);
        EMLRTPOPRTSTACK(&i_emlrtRSI);
        area_bg = r0->size[0] - 1;
        for (j = 0; j <= area_bg; j++) {
            SD->f1.B[emlrtBoundsCheckR2011a(r0->data[j], &c_emlrtBCI, &emlrtContextGlobal) - 1] = 0;
        }
        emxFree_int32_T(&r0);
        /* 'svsNuclei:110' I=cat(3,R,G,B); */
        EMLRTPUSHRTSTACK(&j_emlrtRSI);
        area_bg = 0;
        for (j = 0; j < 16777216; j++) {
            area_bg++;
            SD->f1.I[emlrtBoundsCheckR2011a(area_bg, &f_emlrtBCI, &emlrtContextGlobal) - 1] = SD->f1.R[j];
        }
        for (j = 0; j < 16777216; j++) {
            area_bg++;
            SD->f1.I[emlrtBoundsCheckR2011a(area_bg, &e_emlrtBCI, &emlrtContextGlobal) - 1] = SD->f1.G[j];
        }
        for (j = 0; j < 16777216; j++) {
            area_bg++;
            SD->f1.I[emlrtBoundsCheckR2011a(area_bg, &d_emlrtBCI, &emlrtContextGlobal) - 1] = SD->f1.B[j];
        }
        EMLRTPOPRTSTACK(&j_emlrtRSI);
        /* 'svsNuclei:111' imwrite(I, [resultpath, filename,'.grid4.jpg'], 'Quality',80); */
        EMLRTPUSHRTSTACK(&k_emlrtRSI);
        f_y = NULL;
        m0 = mxCreateNumericArray(3, (int32_T *)&iv2, mxUINT8_CLASS, mxREAL);
        pData = (uint8_T (*)[])mxGetData(m0);
        for (area_bg = 0; area_bg < 50331648; area_bg++) {
            (*pData)[area_bg] = SD->f1.I[area_bg];
        }
        emxInit_char_T(&b_u, 2, &emlrtRTEI, TRUE);
        emlrtAssign(&f_y, m0);
        j = b_u->size[0] * b_u->size[1];
        b_u->size[0] = 1;
        b_u->size[1] = (resultpath->size[1] + filename->size[1]) + 10;
        emxEnsureCapacity((emxArray__common *)b_u, j, (int32_T)sizeof(char_T), &emlrtRTEI);
        area_bg = resultpath->size[1] - 1;
        for (j = 0; j <= area_bg; j++) {
            b_u->data[b_u->size[0] * j] = resultpath->data[resultpath->size[0] * j];
        }
        area_bg = filename->size[1] - 1;
        for (j = 0; j <= area_bg; j++) {
            b_u->data[b_u->size[0] * (j + resultpath->size[1])] = filename->data[filename->size[0] * j];
        }
        for (j = 0; j < 10; j++) {
            b_u->data[b_u->size[0] * ((j + resultpath->size[1]) + filename->size[1])] = cv0[j];
        }
        emxInit_char_T(&c_u, 2, &emlrtRTEI, TRUE);
        g_y = NULL;
        m0 = mxCreateCharArray(2, b_u->size);
        emlrtInitCharArray(b_u->size[1], m0, b_u->data);
        emlrtAssign(&g_y, m0);
        h_y = NULL;
        m0 = mxCreateCharArray(2, iv3);
        emlrtInitCharArray(7, m0, cv1);
        emlrtAssign(&h_y, m0);
        i_y = NULL;
        m0 = mxCreateDoubleScalar(80.0);
        emlrtAssign(&i_y, m0);
        imwrite(f_y, g_y, h_y, i_y, &f_emlrtMCI);
        EMLRTPOPRTSTACK(&k_emlrtRSI);
        /* 'svsNuclei:113' save([resultpath, filename,'.grid4.mat'],'f','L','t','-v7.3'); */
        EMLRTPUSHRTSTACK(&l_emlrtRSI);
        j = c_u->size[0] * c_u->size[1];
        c_u->size[0] = 1;
        c_u->size[1] = (resultpath->size[1] + filename->size[1]) + 10;
        emxEnsureCapacity((emxArray__common *)c_u, j, (int32_T)sizeof(char_T), &emlrtRTEI);
        emxFree_char_T(&b_u);
        area_bg = resultpath->size[1] - 1;
        for (j = 0; j <= area_bg; j++) {
            c_u->data[c_u->size[0] * j] = resultpath->data[resultpath->size[0] * j];
        }
        area_bg = filename->size[1] - 1;
        for (j = 0; j <= area_bg; j++) {
            c_u->data[c_u->size[0] * (j + resultpath->size[1])] = filename->data[filename->size[0] * j];
        }
        for (j = 0; j < 10; j++) {
            c_u->data[c_u->size[0] * ((j + resultpath->size[1]) + filename->size[1])] = cv2[j];
        }
        j_y = NULL;
        m0 = mxCreateCharArray(2, c_u->size);
        emlrtInitCharArray(c_u->size[1], m0, c_u->data);
        emlrtAssign(&j_y, m0);
        k_y = NULL;
        m0 = emlrtCreateString1('f');
        emlrtAssign(&k_y, m0);
        l_y = NULL;
        m0 = emlrtCreateString1('L');
        emlrtAssign(&l_y, m0);
        m_y = NULL;
        m0 = emlrtCreateString1('t');
        emlrtAssign(&m_y, m0);
        n_y = NULL;
        m0 = mxCreateCharArray(2, iv4);
        emlrtInitCharArray(5, m0, cv3);
        emlrtAssign(&n_y, m0);
        save(j_y, k_y, l_y, m_y, n_y, &g_emlrtMCI);
        EMLRTPOPRTSTACK(&l_emlrtRSI);
        /* pais(resultpath,[filename '.grid4.mat'],resultpath,folder,tile,p); */
        emxFree_char_T(&c_u);
    }
    emlrtDestroyArray(&grayI);
    emlrtHeapReferenceStackLeaveFcn();
}
/* End of code generation (svsNuclei.c) */
