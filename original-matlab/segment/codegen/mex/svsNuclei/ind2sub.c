/*
 * ind2sub.c
 *
 * Code generation for function 'ind2sub'
 *
 * C source code generated on: Tue Jun 21 16:24:31 2011
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "svsNuclei.h"
#include "ind2sub.h"
#include "svsNuclei_emxutil.h"
#include "svsNuclei_mexutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */
static emlrtRSInfo lb_emlrtRSI = { 32, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtRSInfo mb_emlrtRSI = { 32, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtRSInfo nb_emlrtRSI = { 38, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtMCInfo ab_emlrtMCI = { 33, 5, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtMCInfo bb_emlrtMCI = { 32, 15, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtRTEInfo f_emlrtRTEI = { 1, 22, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtRTEInfo g_emlrtRTEI = { 34, 1, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtRTEInfo h_emlrtRTEI = { 36, 5, "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m" };
static emlrtECInfo b_emlrtECI = { 1, 11, 5, "eml_index_minus", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/eml/eml_index_minus.m" };
static emlrtBCInfo i_emlrtBCI = { -1, -1, 61, 10, "", "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m", 0 };
static emlrtBCInfo j_emlrtBCI = { -1, -1, 61, 24, "", "ind2sub", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/ind2sub.m", 0 };

/* Function Declarations */
static const mxArray *message(const mxArray *b, emlrtMCInfo *location);

/* Function Definitions */

static const mxArray *message(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    const mxArray *m28;
    pArray = b;
    return emlrtCallMATLAB(1, &m28, 1, &pArray, "message", TRUE, location);
}

/*
 * 
 */
void ind2sub(const emxArray_real_T *ndx, emxArray_real_T *varargout_1, emxArray_real_T *varargout_2)
{
    uint32_T k;
    int32_T exitg1;
    boolean_T b0;
    const mxArray *y;
    static const int32_T iv24[2] = { 1, 36 };
    const mxArray *m2;
    static const char_T cv6[36] = { 'C', 'o', 'd', 'e', 'r', ':', 'M', 'A', 'T', 'L', 'A', 'B', ':', 'i', 'n', 'd', '2', 's', 'u', 'b', '_', 'I', 'n', 'd', 'e', 'x', 'O', 'u', 't', 'O', 'f', 'R', 'a', 'n', 'g', 'e' };
    emxArray_int32_T *v1;
    int32_T i2;
    int32_T loop_ub;
    real_T d0;
    int32_T i3;
    emxArray_int32_T *vk;
    emlrtHeapReferenceStackEnterFcn();
    EMLRTPUSHRTSTACK(&lb_emlrtRSI);
    k = 1U;
    do {
        exitg1 = 0U;
        if (k <= (uint32_T)ndx->size[0]) {
            if ((ndx->data[emlrtDynamicBoundsCheckR2011a((int32_T)k, 1, ndx->size[0], &i_emlrtBCI, &emlrtContextGlobal) - 1] >= 1.0) && (ndx->data[emlrtDynamicBoundsCheckR2011a((int32_T)k, 1, ndx->size[0], &j_emlrtBCI, &emlrtContextGlobal) - 1] <= 1.6777216E+7)) {
                b0 = TRUE;
            } else {
                b0 = FALSE;
            }
            if (!b0) {
                b0 = FALSE;
                exitg1 = 1U;
            } else {
                k++;
            }
        } else {
            b0 = TRUE;
            exitg1 = 1U;
        }
    } while (exitg1 == 0U);
    EMLRTPOPRTSTACK(&lb_emlrtRSI);
    if (b0) {
    } else {
        EMLRTPUSHRTSTACK(&mb_emlrtRSI);
        y = NULL;
        m2 = mxCreateCharArray(2, iv24);
        emlrtInitCharArray(36, m2, cv6);
        emlrtAssign(&y, m2);
        error(message(y, &ab_emlrtMCI), &bb_emlrtMCI);
        EMLRTPOPRTSTACK(&mb_emlrtRSI);
    }
    emxInit_int32_T(&v1, 1, &g_emlrtRTEI, TRUE);
    i2 = v1->size[0];
    v1->size[0] = ndx->size[0];
    emxEnsureCapacity((emxArray__common *)v1, i2, (int32_T)sizeof(int32_T), &f_emlrtRTEI);
    loop_ub = ndx->size[0] - 1;
    for (i2 = 0; i2 <= loop_ub; i2++) {
        d0 = ndx->data[i2];
        d0 = d0 < 0.0 ? muDoubleScalarCeil(d0 - 0.5) : muDoubleScalarFloor(d0 + 0.5);
        if (d0 < 2.147483648E+9) {
            if (d0 >= -2.147483648E+9) {
                i3 = (int32_T)d0;
            } else {
                i3 = MIN_int32_T;
            }
        } else if (d0 >= 2.147483648E+9) {
            i3 = MAX_int32_T;
        } else {
            i3 = 0;
        }
        v1->data[i2] = i3 - 1;
    }
    emxInit_int32_T(&vk, 1, &h_emlrtRTEI, TRUE);
    i2 = vk->size[0];
    vk->size[0] = v1->size[0];
    emxEnsureCapacity((emxArray__common *)vk, i2, (int32_T)sizeof(int32_T), &f_emlrtRTEI);
    loop_ub = v1->size[0] - 1;
    for (i2 = 0; i2 <= loop_ub; i2++) {
        i3 = v1->data[i2];
        i3 += i3 < 0 ? 4095 : 0;
        vk->data[i2] = i3 >= 0 ? (int32_T)((uint32_T)i3 >> 12) : ~(int32_T)((uint32_T)~i3 >> 12);
    }
    i2 = varargout_2->size[0];
    varargout_2->size[0] = vk->size[0];
    emxEnsureCapacity((emxArray__common *)varargout_2, i2, (int32_T)sizeof(real_T), &f_emlrtRTEI);
    loop_ub = vk->size[0] - 1;
    for (i2 = 0; i2 <= loop_ub; i2++) {
        varargout_2->data[i2] = (real_T)(vk->data[i2] + 1);
    }
    EMLRTPUSHRTSTACK(&nb_emlrtRSI);
    i2 = vk->size[0];
    emxEnsureCapacity((emxArray__common *)vk, i2, (int32_T)sizeof(int32_T), &f_emlrtRTEI);
    loop_ub = vk->size[0] - 1;
    for (i2 = 0; i2 <= loop_ub; i2++) {
        vk->data[i2] <<= 12;
    }
    emlrtSizeEqCheckNDR2011a(*(int32_T (*)[1])v1->size, *(int32_T (*)[1])vk->size, &b_emlrtECI, &emlrtContextGlobal);
    i2 = v1->size[0];
    emxEnsureCapacity((emxArray__common *)v1, i2, (int32_T)sizeof(int32_T), &f_emlrtRTEI);
    loop_ub = v1->size[0] - 1;
    for (i2 = 0; i2 <= loop_ub; i2++) {
        v1->data[i2] -= vk->data[i2];
    }
    emxFree_int32_T(&vk);
    EMLRTPOPRTSTACK(&nb_emlrtRSI);
    i2 = varargout_1->size[0];
    varargout_1->size[0] = v1->size[0];
    emxEnsureCapacity((emxArray__common *)varargout_1, i2, (int32_T)sizeof(real_T), &f_emlrtRTEI);
    loop_ub = v1->size[0] - 1;
    for (i2 = 0; i2 <= loop_ub; i2++) {
        varargout_1->data[i2] = (real_T)(v1->data[i2] + 1);
    }
    emxFree_int32_T(&v1);
    emlrtHeapReferenceStackLeaveFcn();
}
/* End of code generation (ind2sub.c) */
