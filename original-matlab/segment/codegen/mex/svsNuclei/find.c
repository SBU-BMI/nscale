/*
 * find.c
 *
 * Code generation for function 'find'
 *
 * C source code generated on: Tue Jun 21 16:24:31 2011
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "svsNuclei.h"
#include "find.h"
#include "svsNuclei_emxutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */
static emlrtRTEInfo b_emlrtRTEI = { 101, 5, "find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/find.m" };
static emlrtRTEInfo c_emlrtRTEI = { 1, 20, "find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/find.m" };
static emlrtECInfo emlrtECI = { -1, 221, 17, "find", "/usr/local/MATLAB/R2011a/toolbox/eml/lib/matlab/elmat/find.m" };

/* Function Declarations */

/* Function Definitions */

/*
 * 
 */
void find(const boolean_T x[16777216], emxArray_real_T *i)
{
    int32_T idx;
    int32_T i0;
    int32_T ii;
    boolean_T exitg1;
    boolean_T guard1 = FALSE;
    emxArray_int32_T *r1;
    emxArray_int32_T *r2;
    emxArray_real_T *b_i;
    emlrtHeapReferenceStackEnterFcn();
    idx = 0;
    i0 = i->size[0];
    i->size[0] = 16777216;
    emxEnsureCapacity((emxArray__common *)i, i0, (int32_T)sizeof(real_T), &b_emlrtRTEI);
    ii = 1;
    exitg1 = 0U;
    while ((exitg1 == 0U) && (ii <= 16777216)) {
        guard1 = FALSE;
        if (x[ii - 1]) {
            idx++;
            i->data[idx - 1] = (real_T)ii;
            if (idx >= 16777216) {
                exitg1 = 1U;
            } else {
                guard1 = TRUE;
            }
        } else {
            guard1 = TRUE;
        }
        if (guard1 == TRUE) {
            ii++;
        }
    }
    if (1 > idx) {
        idx = 0;
    }
    emxInit_int32_T(&r1, 1, &c_emlrtRTEI, TRUE);
    i0 = r1->size[0];
    r1->size[0] = idx;
    emxEnsureCapacity((emxArray__common *)r1, i0, (int32_T)sizeof(int32_T), &c_emlrtRTEI);
    ii = idx - 1;
    for (i0 = 0; i0 <= ii; i0++) {
        r1->data[i0] = 1 + i0;
    }
    b_emxInit_int32_T(&r2, 2, &c_emlrtRTEI, TRUE);
    i0 = r2->size[0] * r2->size[1];
    r2->size[0] = 1;
    emxEnsureCapacity((emxArray__common *)r2, i0, (int32_T)sizeof(int32_T), &c_emlrtRTEI);
    ii = r1->size[0];
    i0 = r2->size[0] * r2->size[1];
    r2->size[1] = ii;
    emxEnsureCapacity((emxArray__common *)r2, i0, (int32_T)sizeof(int32_T), &c_emlrtRTEI);
    ii = r1->size[0] - 1;
    for (i0 = 0; i0 <= ii; i0++) {
        r2->data[i0] = r1->data[i0];
    }
    emxFree_int32_T(&r1);
    emxInit_real_T(&b_i, 1, &c_emlrtRTEI, TRUE);
    emlrtVectorVectorIndexCheckR2011a(16777216, 1, 1, r2->size[1], &emlrtECI, &emlrtContextGlobal);
    i0 = b_i->size[0];
    b_i->size[0] = r2->size[1];
    emxEnsureCapacity((emxArray__common *)b_i, i0, (int32_T)sizeof(real_T), &c_emlrtRTEI);
    ii = r2->size[1] - 1;
    for (i0 = 0; i0 <= ii; i0++) {
        b_i->data[i0] = i->data[r2->data[i0] - 1];
    }
    emxFree_int32_T(&r2);
    i0 = i->size[0];
    i->size[0] = b_i->size[0];
    emxEnsureCapacity((emxArray__common *)i, i0, (int32_T)sizeof(real_T), &c_emlrtRTEI);
    ii = b_i->size[0] - 1;
    for (i0 = 0; i0 <= ii; i0++) {
        i->data[i0] = b_i->data[i0];
    }
    emxFree_real_T(&b_i);
    emlrtHeapReferenceStackLeaveFcn();
}
/* End of code generation (find.c) */
