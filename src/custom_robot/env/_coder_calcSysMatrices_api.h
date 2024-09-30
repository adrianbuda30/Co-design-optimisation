//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: _coder_calcSysMatrices_api.h
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 23-Nov-2023 17:42:08
//

#ifndef _CODER_CALCSYSMATRICES_API_H
#define _CODER_CALCSYSMATRICES_API_H

// Include Files
#include "emlrt.h"
#include "tmwtypes.h"
#include <algorithm>
#include <cstring>

// Variable Declarations
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

// Function Declarations
void calcSysMatrices(real_T q[3], real_T dq[3], real_T rho, real_T radius,
                     real_T arm_length[3], real_T torque[3],
                     real_T joint_acc[3], real_T pos_tcp[3]);

void calcSysMatrices_api(const mxArray *const prhs[6], int32_T nlhs,
                         const mxArray *plhs[2]);

void calcSysMatrices_atexit();

void calcSysMatrices_initialize();

void calcSysMatrices_terminate();

void calcSysMatrices_xil_shutdown();

void calcSysMatrices_xil_terminate();

#endif
//
// File trailer for _coder_calcSysMatrices_api.h
//
// [EOF]
//
