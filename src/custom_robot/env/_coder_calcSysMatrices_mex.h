//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: _coder_calcSysMatrices_mex.h
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 23-Nov-2023 17:42:08
//

#ifndef _CODER_CALCSYSMATRICES_MEX_H
#define _CODER_CALCSYSMATRICES_MEX_H

// Include Files
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"

// Function Declarations
MEXFUNCTION_LINKAGE void mexFunction(int32_T nlhs, mxArray *plhs[],
                                     int32_T nrhs, const mxArray *prhs[]);

emlrtCTX mexFunctionCreateRootTLS();

void unsafe_calcSysMatrices_mexFunction(int32_T nlhs, mxArray *plhs[2],
                                        int32_T nrhs, const mxArray *prhs[6]);

#endif
//
// File trailer for _coder_calcSysMatrices_mex.h
//
// [EOF]
//
