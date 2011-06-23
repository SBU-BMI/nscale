/*
 * svsNuclei_terminate.c
 *
 * Code generation for function 'svsNuclei_terminate'
 *
 * C source code generated on: Tue Jun 21 16:24:31 2011
 *
 */

/* Include files */
#include "rt_nonfinite.h"
#include "svsNuclei.h"
#include "svsNuclei_terminate.h"

/* Type Definitions */

/* Named Constants */

/* Variable Declarations */

/* Variable Definitions */

/* Function Declarations */

/* Function Definitions */

void svsNuclei_atexit(void)
{
    emlrtExitTimeCleanup(&emlrtContextGlobal);
}

void svsNuclei_terminate(void)
{
    emlrtLeaveRtStack(&emlrtContextGlobal);
}
/* End of code generation (svsNuclei_terminate.c) */
