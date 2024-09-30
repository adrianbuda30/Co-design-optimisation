//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: calcSysMatrices.h
//
// MATLAB Coder version            : 5.6
// C/C++ source code generated on  : 23-Nov-2023 17:42:08
//

#ifndef CALCSYSMATRICES_H
#define CALCSYSMATRICES_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern void calcSysMatrices(const double q[3], const double dq[3], double rho,
                            double radius, const double arm_length[3],
                            const double torque[3], double joint_acc[3],
                            double pos_tcp[3]);

#endif
//
// File trailer for calcSysMatrices.h
//
// [EOF]
//
