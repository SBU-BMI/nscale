/*
 * svsNuclei_api.c
 *
 * Code generation for function 'svsNuclei_api'
 *
 * C source code generated on: Tue Jun 21 16:24:31 2011
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "svsNuclei.h"
#include "svsNuclei_api.h"
#include "svsNuclei_emxutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */
static emlrtRTEInfo j_emlrtRTEI = { 1, 1, "svsNuclei_api", "" };

/* Function Declarations */
static void p_emlrt_marshallIn(const mxArray *impath, const char_T *identifier, emxArray_char_T *y);
static void q_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, emxArray_char_T *y);
static void y_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, emxArray_char_T *ret);

/* Function Definitions */

static void p_emlrt_marshallIn(const mxArray *impath, const char_T *identifier, emxArray_char_T *y)
{
    emlrtMsgIdentifier thisId;
    thisId.fIdentifier = identifier;
    thisId.fParent = NULL;
    q_emlrt_marshallIn(emlrtAlias(impath), &thisId, y);
    emlrtDestroyArray(&impath);
}

static void q_emlrt_marshallIn(const mxArray *u, const emlrtMsgIdentifier *parentId, emxArray_char_T *y)
{
    y_emlrt_marshallIn(emlrtAlias(u), parentId, y);
    emlrtDestroyArray(&u);
}

static void y_emlrt_marshallIn(const mxArray *src, const emlrtMsgIdentifier *msgId, emxArray_char_T *ret)
{
    int32_T i11;
    int32_T iv37[2];
    static const boolean_T bv0[2] = { FALSE, TRUE };
    boolean_T bv1[2];
    for (i11 = 0; i11 < 2; i11++) {
        iv37[i11] = 1 + -2 * i11;
        bv1[i11] = bv0[i11];
    }
    emlrtCheckVsBuiltInR2011a(msgId, src, "char", FALSE, 2U, iv37, bv1, ret->size);
    i11 = ret->size[0] * ret->size[1];
    ret->size[0] = 1;
    emxEnsureCapacity((emxArray__common *)ret, i11, (int32_T)sizeof(char_T), (emlrtRTEInfo *)NULL);
    emlrtImportArrayR2008b(src, ret->data, 1);
    emlrtDestroyArray(&src);
}

void svsNuclei_api(svsNucleiStackData *SD, const mxArray * const prhs[3])
{
    emxArray_char_T *impath;
    emxArray_char_T *filename;
    emxArray_char_T *resultpath;
    emlrtHeapReferenceStackEnterFcn();
    emxInit_char_T(&impath, 2, &j_emlrtRTEI, TRUE);
    emxInit_char_T(&filename, 2, &j_emlrtRTEI, TRUE);
    emxInit_char_T(&resultpath, 2, &j_emlrtRTEI, TRUE);
    /* Marshall function inputs */
    p_emlrt_marshallIn(emlrtAliasP(prhs[0]), "impath", impath);
    p_emlrt_marshallIn(emlrtAliasP(prhs[1]), "filename", filename);
    p_emlrt_marshallIn(emlrtAliasP(prhs[2]), "resultpath", resultpath);
    /* Invoke the target function */
    svsNuclei(SD, impath, filename, resultpath);
    emxFree_char_T(&resultpath);
    emxFree_char_T(&filename);
    emxFree_char_T(&impath);
    emlrtHeapReferenceStackLeaveFcn();
}
/* End of code generation (svsNuclei_api.c) */
