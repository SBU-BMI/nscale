/*
 * svsNuclei_mex.c
 *
 * Code generation for function 'svsNuclei'
 *
 * C source code generated on: Tue Jun 21 16:24:31 2011
 *
 */

/* Include files */
#include "mex.h"
#include "svsNuclei_api.h"
#include "svsNuclei_initialize.h"
#include "svsNuclei_terminate.h"

/* Type Definitions */

/* Function Declarations */
static void svsNuclei_mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

/* Variable Definitions */
emlrtContext emlrtContextGlobal = { true, false, EMLRT_VERSION_INFO, NULL, "svsNuclei", NULL, false, NULL };

/* Function Definitions */
static void svsNuclei_mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  svsNucleiStackData* svsNucleiStackDataLocal = (svsNucleiStackData*)mxCalloc(1,sizeof(svsNucleiStackData));
  /* Check for proper number of arguments. */
  if(nrhs != 3) {
    mexErrMsgIdAndTxt("emlcoder:emlmex:WrongNumberOfInputs","3 inputs required for entry-point 'svsNuclei'.");
  } else if(nlhs > 0) {
    mexErrMsgIdAndTxt("emlcoder:emlmex:TooManyOutputArguments","Too many output arguments for entry-point 'svsNuclei'.");
  }
  /* Module initialization. */
  svsNuclei_initialize(&emlrtContextGlobal);
  /* Call the function. */
  svsNuclei_api(svsNucleiStackDataLocal, prhs);
  /* Module finalization. */
  svsNuclei_terminate();
  mxFree(svsNucleiStackDataLocal);
}
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Initialize the memory manager. */
  mexAtExit(svsNuclei_atexit);
  emlrtClearAllocCount(&emlrtContextGlobal, 0, 0, NULL);
  /* Dispatch the entry-point. */
  svsNuclei_mexFunction(nlhs, plhs, nrhs, prhs);
}
/* End of code generation (svsNuclei_mex.c) */
