/*
 * svsNuclei_mexutil.c
 *
 * Code generation for function 'svsNuclei_mexutil'
 *
 * C source code generated on: Tue Jun 21 16:24:30 2011
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "svsNuclei.h"
#include "svsNuclei_mexutil.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */

void error(const mxArray *b, emlrtMCInfo *location)
{
    const mxArray *pArray;
    pArray = b;
    emlrtCallMATLAB(0, NULL, 1, &pArray, "error", TRUE, location);
}
/* End of code generation (svsNuclei_mexutil.c) */
