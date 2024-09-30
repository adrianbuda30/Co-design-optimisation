/*
 * True0.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "True0".
 *
 * Model version              : 14.1
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Sun Jul 23 21:03:36 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: 32-bit Generic
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#include "True0.h"
#include "rtwtypes.h"
#include <cmath>
#include <cstring>
#include "True0_private.h"
#include "rt_defines.h"

extern "C"
{

#include "rt_nonfinite.h"

}

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
void True0::rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  /* Solver Matrices */
  static const real_T rt_ODE3_A[3]{
    1.0/2.0, 3.0/4.0, 1.0
  };

  static const real_T rt_ODE3_B[3][3]{
    { 1.0/2.0, 0.0, 0.0 },

    { 0.0, 3.0/4.0, 0.0 },

    { 2.0/9.0, 1.0/3.0, 4.0/9.0 }
  };

  time_T t { rtsiGetT(si) };

  time_T tnew { rtsiGetSolverStopTime(si) };

  time_T h { rtsiGetStepSize(si) };

  real_T *x { rtsiGetContStates(si) };

  ODE3_IntgData *id { static_cast<ODE3_IntgData *>(rtsiGetSolverData(si)) };

  real_T *y { id->y };

  real_T *f0 { id->f[0] };

  real_T *f1 { id->f[1] };

  real_T *f2 { id->f[2] };

  real_T hB[3];
  int_T i;
  int_T nXc { 17 };

  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  True0_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  this->step0();
  True0_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  this->step0();
  True0_derivatives();

  /* tnew = t + hA(3);
     ynew = y + f*hB(:,3); */
  for (i = 0; i <= 2; i++) {
    hB[i] = h * rt_ODE3_B[2][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2]);
  }

  rtsiSetT(si, tnew);
  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

void rt_mldivide_U1d3x3_U2d_JBYZyA3A(const real_T u0[9], const real_T u1[3],
  real_T y[3])
{
  real_T A[9];
  real_T B[3];
  real_T a21;
  real_T maxval;
  real_T x;
  int32_T THREE;
  int32_T r1;
  int32_T r2;
  std::memcpy(&A[0], &u0[0], 9U * sizeof(real_T));
  B[0] = u1[0];
  B[1] = u1[1];
  B[2] = u1[2];
  THREE = 2;
  r1 = 0;
  r2 = 1;
  x = A[0];
  x = std::abs(x);
  maxval = x;
  x = A[1];
  x = std::abs(x);
  a21 = x;
  if (a21 > maxval) {
    maxval = a21;
    r1 = 1;
    r2 = 0;
  }

  x = A[2];
  x = std::abs(x);
  a21 = x;
  if (a21 > maxval) {
    r1 = 2;
    r2 = 1;
    THREE = 0;
  }

  A[r2] /= A[r1];
  A[THREE] /= A[r1];
  A[r2 + 3] -= A[r1 + 3] * A[r2];
  A[THREE + 3] -= A[r1 + 3] * A[THREE];
  A[r2 + 6] -= A[r1 + 6] * A[r2];
  A[THREE + 6] -= A[r1 + 6] * A[THREE];
  x = A[THREE + 3];
  x = std::abs(x);
  a21 = x;
  x = A[r2 + 3];
  x = std::abs(x);
  maxval = x;
  if (a21 > maxval) {
    int32_T ONE;
    ONE = r2 + 1;
    r2 = THREE;
    THREE = ONE - 1;
  }

  A[THREE + 3] /= A[r2 + 3];
  A[THREE + 6] -= A[THREE + 3] * A[r2 + 6];
  y[0] = B[r1];
  y[1] = B[r2] - y[0] * A[r2];
  y[2] = (B[THREE] - y[0] * A[THREE]) - A[THREE + 3] * y[1];
  y[2] /= A[THREE + 6];
  y[0] -= A[r1 + 6] * y[2];
  y[1] -= A[r2 + 6] * y[2];
  y[1] /= A[r2 + 3];
  y[0] -= A[r1 + 3] * y[1];
  y[0] /= A[r1];
}

real_T rt_atan2d_snf(real_T u0, real_T u1)
{
  real_T y;
  if (std::isnan(u0) || std::isnan(u1)) {
    y = (rtNaN);
  } else if (std::isinf(u0) && std::isinf(u1)) {
    int32_T tmp;
    int32_T tmp_0;
    if (u1 > 0.0) {
      tmp = 1;
    } else {
      tmp = -1;
    }

    if (u0 > 0.0) {
      tmp_0 = 1;
    } else {
      tmp_0 = -1;
    }

    y = std::atan2(static_cast<real_T>(tmp_0), static_cast<real_T>(tmp));
  } else if (u1 == 0.0) {
    if (u0 > 0.0) {
      y = RT_PI / 2.0;
    } else if (u0 < 0.0) {
      y = -(RT_PI / 2.0);
    } else {
      y = 0.0;
    }
  } else {
    y = std::atan2(u0, u1);
  }

  return y;
}

/* Model step function for TID0 */
void True0::step0()                    /* Sample time: [0.0s, 0.0s] */
{
  /* local scratch DWork variables */
  int32_T ForEach_itr;
  real_T rtb_ImpSel_InsertedFor_MotorMat[17];
  real_T tmp[9];
  real_T Product1_a_tmp;
  real_T Product2_i4_tmp;
  real_T Product_pk_tmp;
  real_T TrigonometricFunction5_tmp;
  real_T TrigonometricFunction_tmp;
  real_T cphi;
  real_T cpsi;
  real_T ctheta;
  real_T phi;
  real_T psi;
  real_T psi_tmp;
  real_T psi_tmp_0;
  real_T rtb_ImpSel_InsertedFor_RPM_comm;
  real_T spsi;
  real_T theta;
  int32_T i;
  int32_T tmp_0;
  int8_T rtAction;
  if (rtmIsMajorTimeStep((&True0_M))) {
    /* set solver stop time */
    if (!((&True0_M)->Timing.clockTick0+1)) {
      rtsiSetSolverStopTime(&(&True0_M)->solverInfo, (((&True0_M)
        ->Timing.clockTickH0 + 1) * (&True0_M)->Timing.stepSize0 * 4294967296.0));
    } else {
      rtsiSetSolverStopTime(&(&True0_M)->solverInfo, (((&True0_M)
        ->Timing.clockTick0 + 1) * (&True0_M)->Timing.stepSize0 + (&True0_M)
        ->Timing.clockTickH0 * (&True0_M)->Timing.stepSize0 * 4294967296.0));
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.002s, 0.0s]  */
    ((&True0_M)->Timing.RateInteraction.TID1_2)++;
    if (((&True0_M)->Timing.RateInteraction.TID1_2) > 1) {
      (&True0_M)->Timing.RateInteraction.TID1_2 = 0;
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep((&True0_M))) {
    (&True0_M)->Timing.t[0] = rtsiGetT(&(&True0_M)->solverInfo);
  }

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  if (rtmIsMajorTimeStep((&True0_M))) {
    /* MATLAB Function: '<S7>/MATLAB Function' incorporates:
     *  Constant: '<S2>/Constant'
     */
    /* :  roll  = quat_input(1); */
    /* :  pitch = quat_input(2); */
    /* :  yaw   = quat_input(3); */
    /* :  phi   = roll / 2; */
    phi = True0_P.Att_init[0] / 2.0;

    /* :  theta = pitch / 2; */
    theta = True0_P.Att_init[1] / 2.0;

    /* :  psi   = yaw / 2; */
    psi = True0_P.Att_init[2] / 2.0;

    /* :  cphi   = cos(phi); */
    cphi = std::cos(phi);

    /* :  sphi   = sin(phi); */
    phi = std::sin(phi);

    /* :  ctheta = cos(theta); */
    ctheta = std::cos(theta);

    /* :  stheta = sin(theta); */
    theta = std::sin(theta);

    /* :  cpsi   = cos(psi); */
    cpsi = std::cos(psi);

    /* :  spsi   = sin(psi); */
    spsi = std::sin(psi);

    /* :  q0 = cphi * ctheta * cpsi + sphi * stheta * spsi; */
    psi_tmp = cphi * ctheta;
    psi_tmp_0 = phi * theta;
    psi = psi_tmp * cpsi + psi_tmp_0 * spsi;

    /* :  q1 = sphi * ctheta * cpsi - cphi * stheta * spsi; */
    cphi *= theta;
    ctheta *= phi;
    phi = ctheta * cpsi - cphi * spsi;

    /* :  q2 = cphi * stheta * cpsi + sphi * ctheta * spsi; */
    ctheta = cphi * cpsi + ctheta * spsi;

    /* :  q3 = cphi * ctheta * spsi - sphi * stheta * cpsi; */
    cphi = psi_tmp * spsi - psi_tmp_0 * cpsi;

    /* :  if (q0 < 0.0) */
    if (psi < 0.0) {
      /* :  q0 = -q0; */
      psi = -psi;

      /* :  q1 = -q1; */
      phi = -phi;

      /* :  q2 = -q2; */
      ctheta = -ctheta;

      /* :  q3 = -q3; */
      cphi = -cphi;
    }

    /* :  quat_output = [0 0 0 0]'; */
    /* :  quat_output(1,1) = q0; */
    True0_B.quat_output[0] = psi;

    /* :  quat_output(2,1) = q1; */
    True0_B.quat_output[1] = phi;

    /* :  quat_output(3,1) = q2; */
    True0_B.quat_output[2] = ctheta;

    /* :  quat_output(4,1) = q3; */
    True0_B.quat_output[3] = cphi;

    /* End of MATLAB Function: '<S7>/MATLAB Function' */
  }

  /* Integrator: '<S7>/Q-Integrator' */
  if (True0_DW.QIntegrator_IWORK != 0) {
    True0_X.QIntegrator_CSTATE[0] = True0_B.quat_output[0];
    True0_X.QIntegrator_CSTATE[1] = True0_B.quat_output[1];
    True0_X.QIntegrator_CSTATE[2] = True0_B.quat_output[2];
    True0_X.QIntegrator_CSTATE[3] = True0_B.quat_output[3];
  }

  /* Integrator: '<S7>/Q-Integrator' */
  psi = True0_X.QIntegrator_CSTATE[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.QIntegrator[0] = psi;

  /* Math: '<S22>/transpose' */
  True0_B.transpose[0] = psi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S22>/Product' */
  Product_pk_tmp = psi * psi;

  /* Integrator: '<S7>/Q-Integrator' */
  psi = True0_X.QIntegrator_CSTATE[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.QIntegrator[1] = psi;

  /* Math: '<S22>/transpose' */
  True0_B.transpose[1] = psi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S22>/Product' */
  Product_pk_tmp += psi * psi;

  /* Integrator: '<S7>/Q-Integrator' */
  psi = True0_X.QIntegrator_CSTATE[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.QIntegrator[2] = psi;

  /* Math: '<S22>/transpose' */
  True0_B.transpose[2] = psi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S22>/Product' */
  Product_pk_tmp += psi * psi;

  /* Integrator: '<S7>/Q-Integrator' */
  psi = True0_X.QIntegrator_CSTATE[3];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.QIntegrator[3] = psi;

  /* Math: '<S22>/transpose' */
  True0_B.transpose[3] = psi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S22>/Product' */
  Product_pk_tmp += psi * psi;

  /* Product: '<S22>/Product' */
  True0_B.Product = Product_pk_tmp;

  /* Sqrt: '<S21>/Sqrt' */
  True0_B.Sqrt = std::sqrt(True0_B.Product);

  /* ComplexToRealImag: '<S21>/Complex to Real-Imag' */
  True0_B.ComplextoRealImag = True0_B.Sqrt;

  /* Product: '<S18>/Divide' incorporates:
   *  Integrator: '<S7>/Q-Integrator'
   */
  True0_B.Divide[0] = True0_B.QIntegrator[0] / True0_B.ComplextoRealImag;
  True0_B.Divide[1] = True0_B.QIntegrator[1] / True0_B.ComplextoRealImag;
  True0_B.Divide[2] = True0_B.QIntegrator[2] / True0_B.ComplextoRealImag;
  True0_B.Divide[3] = True0_B.QIntegrator[3] / True0_B.ComplextoRealImag;

  /* Product: '<S23>/Product' incorporates:
   *  Product: '<S24>/Product'
   *  Product: '<S25>/Product'
   */
  cphi = True0_B.Divide[0] * True0_B.Divide[0];

  /* Product: '<S23>/Product' */
  True0_B.Product_o = cphi;

  /* Product: '<S23>/Product2' incorporates:
   *  Product: '<S24>/Product2'
   *  Product: '<S25>/Product2'
   */
  phi = True0_B.Divide[1] * True0_B.Divide[1];

  /* Product: '<S23>/Product2' */
  True0_B.Product2 = phi;

  /* Product: '<S23>/Product3' incorporates:
   *  Product: '<S24>/Product3'
   *  Product: '<S25>/Product3'
   */
  ctheta = True0_B.Divide[2] * True0_B.Divide[2];

  /* Product: '<S23>/Product3' */
  True0_B.Product3 = ctheta;

  /* Product: '<S23>/Product4' incorporates:
   *  Product: '<S24>/Product4'
   *  Product: '<S25>/Product4'
   */
  theta = True0_B.Divide[3] * True0_B.Divide[3];

  /* Product: '<S23>/Product4' */
  True0_B.Product4 = theta;

  /* Sum: '<S23>/Add' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   *  Fcn: '<S8>/Fcn4'
   */
  psi = ((True0_B.Product_o + True0_B.Product2) - True0_B.Product3) -
    True0_B.Product4;
  True0_B.VectorConcatenate[0] = psi;

  /* Product: '<S28>/Product' incorporates:
   *  Product: '<S26>/Product'
   */
  cpsi = True0_B.Divide[1] * True0_B.Divide[2];

  /* Product: '<S28>/Product' */
  True0_B.Product_p = cpsi;

  /* Product: '<S28>/Product2' incorporates:
   *  Product: '<S26>/Product2'
   */
  spsi = True0_B.Divide[0] * True0_B.Divide[3];

  /* Product: '<S28>/Product2' */
  True0_B.Product2_n = spsi;

  /* Sum: '<S28>/Add' */
  True0_B.Add = True0_B.Product_p - True0_B.Product2_n;

  /* Gain: '<S28>/Gain' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  True0_B.VectorConcatenate[1] = True0_P.Gain_Gain * True0_B.Add;

  /* Product: '<S30>/Product' incorporates:
   *  Product: '<S27>/Product'
   */
  psi_tmp = True0_B.Divide[1] * True0_B.Divide[3];

  /* Product: '<S30>/Product' */
  True0_B.Product_l = psi_tmp;

  /* Product: '<S30>/Product2' incorporates:
   *  Product: '<S27>/Product2'
   */
  psi_tmp_0 = True0_B.Divide[0] * True0_B.Divide[2];

  /* Product: '<S30>/Product2' */
  True0_B.Product2_nt = psi_tmp_0;

  /* Sum: '<S30>/Add' */
  True0_B.Add_c = True0_B.Product_l + True0_B.Product2_nt;

  /* Gain: '<S30>/Gain' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  True0_B.VectorConcatenate[2] = True0_P.Gain_Gain_e * True0_B.Add_c;

  /* Product: '<S26>/Product' */
  True0_B.Product_c = cpsi;

  /* Product: '<S26>/Product2' */
  True0_B.Product2_i = spsi;

  /* Sum: '<S26>/Add' incorporates:
   *  Fcn: '<S8>/Fcn2'
   */
  cpsi = True0_B.Product_c + True0_B.Product2_i;

  /* Sum: '<S26>/Add' */
  True0_B.Add_cz = cpsi;

  /* Gain: '<S26>/Gain' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  True0_B.VectorConcatenate[3] = True0_P.Gain_Gain_ex * True0_B.Add_cz;

  /* Product: '<S24>/Product' */
  True0_B.Product_lj = cphi;

  /* Product: '<S24>/Product2' */
  True0_B.Product2_ib = phi;

  /* Product: '<S24>/Product3' */
  True0_B.Product3_i = ctheta;

  /* Product: '<S24>/Product4' */
  True0_B.Product4_p = theta;

  /* Sum: '<S24>/Add' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   *  Sum: '<S25>/Add'
   */
  spsi = True0_B.Product_lj - True0_B.Product2_ib;
  True0_B.VectorConcatenate[4] = (spsi + True0_B.Product3_i) -
    True0_B.Product4_p;

  /* Product: '<S31>/Product' incorporates:
   *  Product: '<S29>/Product'
   */
  Product_pk_tmp = True0_B.Divide[2] * True0_B.Divide[3];

  /* Product: '<S31>/Product' */
  True0_B.Product_pk = Product_pk_tmp;

  /* Product: '<S31>/Product2' incorporates:
   *  Product: '<S29>/Product2'
   */
  Product2_i4_tmp = True0_B.Divide[0] * True0_B.Divide[1];

  /* Product: '<S31>/Product2' */
  True0_B.Product2_i4 = Product2_i4_tmp;

  /* Sum: '<S31>/Add' */
  True0_B.Add_h = True0_B.Product_pk - True0_B.Product2_i4;

  /* Gain: '<S31>/Gain' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  True0_B.VectorConcatenate[5] = True0_P.Gain_Gain_f * True0_B.Add_h;

  /* Product: '<S27>/Product' */
  True0_B.Product_lb = psi_tmp;

  /* Product: '<S27>/Product2' */
  True0_B.Product2_o = psi_tmp_0;

  /* Sum: '<S27>/Add' incorporates:
   *  Fcn: '<S8>/Fcn1'
   */
  psi_tmp = True0_B.Product_lb - True0_B.Product2_o;

  /* Sum: '<S27>/Add' */
  True0_B.Add_i = psi_tmp;

  /* Gain: '<S27>/Gain' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  True0_B.VectorConcatenate[6] = True0_P.Gain_Gain_c * True0_B.Add_i;

  /* Product: '<S29>/Product' */
  True0_B.Product_f = Product_pk_tmp;

  /* Product: '<S29>/Product2' */
  True0_B.Product2_b = Product2_i4_tmp;

  /* Sum: '<S29>/Add' incorporates:
   *  Fcn: '<S8>/Fcn'
   */
  psi_tmp_0 = True0_B.Product_f + True0_B.Product2_b;

  /* Sum: '<S29>/Add' */
  True0_B.Add_k = psi_tmp_0;

  /* Gain: '<S29>/Gain' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  True0_B.VectorConcatenate[7] = True0_P.Gain_Gain_k * True0_B.Add_k;

  /* Product: '<S25>/Product' */
  True0_B.Product_cs = cphi;

  /* Product: '<S25>/Product2' */
  True0_B.Product2_k = phi;

  /* Product: '<S25>/Product3' */
  True0_B.Product3_k = ctheta;

  /* Product: '<S25>/Product4' */
  True0_B.Product4_b = theta;

  /* Sum: '<S25>/Add' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   *  Fcn: '<S8>/Fcn3'
   */
  spsi = (spsi - True0_B.Product3_k) + True0_B.Product4_b;
  True0_B.VectorConcatenate[8] = spsi;
  for (i = 0; i < 3; i++) {
    /* Math: '<S4>/Math Function2' incorporates:
     *  Concatenate: '<S32>/Vector Concatenate'
     */
    True0_B.DCM_bi[3 * i] = True0_B.VectorConcatenate[i];
    True0_B.DCM_bi[3 * i + 1] = True0_B.VectorConcatenate[i + 3];
    True0_B.DCM_bi[3 * i + 2] = True0_B.VectorConcatenate[i + 6];

    /* Integrator: '<S2>/V_b' */
    True0_B.V_b[i] = True0_X.V_b_CSTATE[i];
  }

  /* Product: '<S4>/Product' incorporates:
   *  Integrator: '<S2>/V_b'
   *  Math: '<S4>/Math Function2'
   */
  std::memcpy(&tmp[0], &True0_B.DCM_bi[0], 9U * sizeof(real_T));
  phi = True0_B.V_b[0];
  ctheta = True0_B.V_b[1];
  theta = True0_B.V_b[2];
  for (i = 0; i < 3; i++) {
    /* Product: '<S4>/Product' */
    cphi = tmp[i] * phi;

    /* Math: '<S5>/Math Function2' incorporates:
     *  Concatenate: '<S32>/Vector Concatenate'
     */
    True0_B.DCM_bi_c[3 * i] = True0_B.VectorConcatenate[i];

    /* Product: '<S4>/Product' */
    cphi += tmp[i + 3] * ctheta;

    /* Math: '<S5>/Math Function2' incorporates:
     *  Concatenate: '<S32>/Vector Concatenate'
     */
    True0_B.DCM_bi_c[3 * i + 1] = True0_B.VectorConcatenate[i + 3];

    /* Product: '<S4>/Product' */
    cphi += tmp[i + 6] * theta;

    /* Math: '<S5>/Math Function2' incorporates:
     *  Concatenate: '<S32>/Vector Concatenate'
     */
    True0_B.DCM_bi_c[3 * i + 2] = True0_B.VectorConcatenate[i + 6];

    /* Product: '<S4>/Product' */
    True0_B.Product_b[i] = cphi;
  }

  /* RateTransition: '<S1>/Rate Transition1' */
  if (rtmIsMajorTimeStep((&True0_M)) && ((&True0_M)
       ->Timing.RateInteraction.TID1_2 == 1)) {
    /* RateTransition: '<S1>/Rate Transition1' */
    True0_B.RateTransition1[0] = True0_DW.RateTransition1_Buffer0[0];
    True0_B.RateTransition1[1] = True0_DW.RateTransition1_Buffer0[1];
    True0_B.RateTransition1[2] = True0_DW.RateTransition1_Buffer0[2];
    True0_B.RateTransition1[3] = True0_DW.RateTransition1_Buffer0[3];
  }

  /* End of RateTransition: '<S1>/Rate Transition1' */

  /* Integrator: '<S2>/omega' */
  True0_B.omega[0] = True0_X.omega_CSTATE[0];
  True0_B.omega[1] = True0_X.omega_CSTATE[1];
  True0_B.omega[2] = True0_X.omega_CSTATE[2];

  /* Product: '<S111>/Product' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   *  Inport: '<Root>/Wind_i'
   */
  std::memcpy(&tmp[0], &True0_B.VectorConcatenate[0], 9U * sizeof(real_T));
  phi = True0_U.Wind_i[0];
  ctheta = True0_U.Wind_i[1];
  theta = True0_U.Wind_i[2];
  for (i = 0; i < 3; i++) {
    /* Product: '<S111>/Product' */
    cphi = tmp[i] * phi;
    cphi += tmp[i + 3] * ctheta;
    cphi += tmp[i + 6] * theta;
    True0_B.Product_oc[i] = cphi;

    /* Sum: '<S56>/Sum1' incorporates:
     *  Integrator: '<S2>/V_b'
     *  Product: '<S111>/Product'
     */
    True0_B.TrueairspeedBodyaxes[i] = True0_B.V_b[i] - cphi;
  }

  /* Outputs for Iterator SubSystem: '<S37>/For Each Subsystem' incorporates:
   *  ForEach: '<S55>/For Each'
   */
  for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
    /* ForEachSliceSelector generated from: '<S55>/MotorMatrix_real' incorporates:
     *  Inport: '<Root>/MotorMatrix_real'
     */
    for (i = 0; i < 17; i++) {
      rtb_ImpSel_InsertedFor_MotorMat[i] = True0_U.MotorMatrix_real[(i << 2) +
        ForEach_itr];
    }

    /* End of ForEachSliceSelector generated from: '<S55>/MotorMatrix_real' */

    /* ForEachSliceSelector generated from: '<S55>/RPM_commands' incorporates:
     *  RateTransition: '<S1>/Rate Transition1'
     */
    rtb_ImpSel_InsertedFor_RPM_comm = True0_B.RateTransition1[ForEach_itr];

    /* Integrator: '<S61>/Integrator' */
    True0_B.CoreSubsys[ForEach_itr].Integrator = True0_X.CoreSubsys[ForEach_itr]
      .Integrator_CSTATE;

    /* RelationalOperator: '<S64>/LowerRelop1' */
    True0_B.CoreSubsys[ForEach_itr].LowerRelop1 =
      (True0_B.CoreSubsys[ForEach_itr].Integrator >
       rtb_ImpSel_InsertedFor_MotorMat[11]);

    /* Switch: '<S64>/Switch2' */
    if (True0_B.CoreSubsys[ForEach_itr].LowerRelop1) {
      /* Switch: '<S64>/Switch2' */
      True0_B.CoreSubsys[ForEach_itr].Switch2 = rtb_ImpSel_InsertedFor_MotorMat
        [11];
    } else {
      /* RelationalOperator: '<S64>/UpperRelop' */
      True0_B.CoreSubsys[ForEach_itr].UpperRelop =
        (True0_B.CoreSubsys[ForEach_itr].Integrator <
         rtb_ImpSel_InsertedFor_MotorMat[10]);

      /* Switch: '<S64>/Switch' */
      if (True0_B.CoreSubsys[ForEach_itr].UpperRelop) {
        /* Switch: '<S64>/Switch' */
        True0_B.CoreSubsys[ForEach_itr].Switch_k =
          rtb_ImpSel_InsertedFor_MotorMat[10];
      } else {
        /* Switch: '<S64>/Switch' */
        True0_B.CoreSubsys[ForEach_itr].Switch_k =
          True0_B.CoreSubsys[ForEach_itr].Integrator;
      }

      /* End of Switch: '<S64>/Switch' */

      /* Switch: '<S64>/Switch2' */
      True0_B.CoreSubsys[ForEach_itr].Switch2 = True0_B.CoreSubsys[ForEach_itr].
        Switch_k;
    }

    /* End of Switch: '<S64>/Switch2' */
    if (rtmIsMajorTimeStep((&True0_M))) {
      /* Product: '<S57>/Product' */
      True0_B.CoreSubsys[ForEach_itr].Product = rtb_ImpSel_InsertedFor_RPM_comm *
        rtb_ImpSel_InsertedFor_MotorMat[4];
    }

    /* Sum: '<S57>/Sum1' */
    True0_B.CoreSubsys[ForEach_itr].Sum1 = True0_B.CoreSubsys[ForEach_itr].
      Product - True0_B.CoreSubsys[ForEach_itr].Switch2;

    /* Product: '<S57>/Divide' */
    True0_B.CoreSubsys[ForEach_itr].Divide = True0_B.CoreSubsys[ForEach_itr].
      Sum1 / rtb_ImpSel_InsertedFor_MotorMat[5];

    /* RelationalOperator: '<S62>/Compare' incorporates:
     *  Constant: '<S62>/Constant'
     */
    True0_B.CoreSubsys[ForEach_itr].Compare = (True0_B.CoreSubsys[ForEach_itr].
      Divide < True0_P.CoreSubsys.Constant_Value_e);

    /* RelationalOperator: '<S63>/Compare' incorporates:
     *  Constant: '<S63>/Constant'
     */
    True0_B.CoreSubsys[ForEach_itr].Compare_j = (True0_B.CoreSubsys[ForEach_itr]
      .Divide > True0_P.CoreSubsys.Constant_Value_c);

    /* RelationalOperator: '<S61>/Relational Operator' */
    True0_B.CoreSubsys[ForEach_itr].RelationalOperator =
      (True0_B.CoreSubsys[ForEach_itr].Integrator <=
       rtb_ImpSel_InsertedFor_MotorMat[11]);

    /* Logic: '<S61>/Logical Operator' */
    True0_B.CoreSubsys[ForEach_itr].LogicalOperator =
      (True0_B.CoreSubsys[ForEach_itr].RelationalOperator ||
       (True0_B.CoreSubsys[ForEach_itr].Compare != 0));

    /* RelationalOperator: '<S61>/Relational Operator1' */
    True0_B.CoreSubsys[ForEach_itr].RelationalOperator1 =
      (True0_B.CoreSubsys[ForEach_itr].Integrator >=
       rtb_ImpSel_InsertedFor_MotorMat[10]);

    /* Logic: '<S61>/Logical Operator1' */
    True0_B.CoreSubsys[ForEach_itr].LogicalOperator1 =
      ((True0_B.CoreSubsys[ForEach_itr].Compare_j != 0) ||
       True0_B.CoreSubsys[ForEach_itr].RelationalOperator1);

    /* Logic: '<S61>/Logical Operator2' */
    True0_B.CoreSubsys[ForEach_itr].LogicalOperator2 =
      (True0_B.CoreSubsys[ForEach_itr].LogicalOperator &&
       True0_B.CoreSubsys[ForEach_itr].LogicalOperator1);

    /* Switch: '<S61>/Switch' */
    if (True0_B.CoreSubsys[ForEach_itr].LogicalOperator2) {
      /* Switch: '<S61>/Switch' */
      True0_B.CoreSubsys[ForEach_itr].Switch = True0_B.CoreSubsys[ForEach_itr].
        Divide;
    } else {
      /* Switch: '<S61>/Switch' incorporates:
       *  Constant: '<S61>/Constant'
       */
      True0_B.CoreSubsys[ForEach_itr].Switch =
        True0_P.CoreSubsys.Constant_Value_g;
    }

    /* End of Switch: '<S61>/Switch' */
    if (rtmIsMajorTimeStep((&True0_M))) {
      /* Gain: '<S59>/Conversion deg to rad' */
      True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad = True0_P.d2r *
        rtb_ImpSel_InsertedFor_MotorMat[0];

      /* Trigonometry: '<S59>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].Motorarmxcomponent = std::cos
        (True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad);

      /* Trigonometry: '<S59>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].Motorarmycomponent = std::sin
        (True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad);

      /* Abs: '<S59>/Abs' */
      True0_B.CoreSubsys[ForEach_itr].Abs = std::abs
        (rtb_ImpSel_InsertedFor_MotorMat[1]);

      /* Product: '<S59>/Product4' */
      True0_B.CoreSubsys[ForEach_itr].Motorlocationxyvector[0] =
        True0_B.CoreSubsys[ForEach_itr].Motorarmxcomponent *
        True0_B.CoreSubsys[ForEach_itr].Abs;
      True0_B.CoreSubsys[ForEach_itr].Motorlocationxyvector[1] =
        True0_B.CoreSubsys[ForEach_itr].Motorarmycomponent *
        True0_B.CoreSubsys[ForEach_itr].Abs;

      /* Reshape: '<S59>/Reshape' */
      True0_B.CoreSubsys[ForEach_itr].Vectorfromgeometriccentertoprop[0] =
        True0_B.CoreSubsys[ForEach_itr].Motorlocationxyvector[0];
      True0_B.CoreSubsys[ForEach_itr].Vectorfromgeometriccentertoprop[1] =
        True0_B.CoreSubsys[ForEach_itr].Motorlocationxyvector[1];
      True0_B.CoreSubsys[ForEach_itr].Vectorfromgeometriccentertoprop[2] =
        rtb_ImpSel_InsertedFor_MotorMat[2];

      /* Sum: '<S59>/Subtract' incorporates:
       *  Inport: '<Root>/CoG_real'
       *  Reshape: '<S59>/Reshape'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[0] =
        True0_B.CoreSubsys[ForEach_itr].Vectorfromgeometriccentertoprop[0] -
        True0_U.CoG_real[0];
      True0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1] =
        True0_B.CoreSubsys[ForEach_itr].Vectorfromgeometriccentertoprop[1] -
        True0_U.CoG_real[1];
      True0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[2] =
        True0_B.CoreSubsys[ForEach_itr].Vectorfromgeometriccentertoprop[2] -
        True0_U.CoG_real[2];
    }

    /* Product: '<S65>/Product' */
    True0_B.CoreSubsys[ForEach_itr].u2v3 = True0_B.omega[1] *
      True0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[2];

    /* Product: '<S65>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].u3v1 = True0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[0] * True0_B.omega[2];

    /* Product: '<S65>/Product2' */
    True0_B.CoreSubsys[ForEach_itr].u1v2 = True0_B.omega[0] *
      True0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1];

    /* Product: '<S66>/Product' */
    True0_B.CoreSubsys[ForEach_itr].u3v2 = True0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[1] * True0_B.omega[2];

    /* Product: '<S66>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].u1v3 = True0_B.omega[0] *
      True0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[2];

    /* Product: '<S66>/Product2' */
    True0_B.CoreSubsys[ForEach_itr].u2v1 = True0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[0] * True0_B.omega[1];

    /* Sum: '<S58>/Sum' */
    True0_B.CoreSubsys[ForEach_itr].Sum[0] = True0_B.CoreSubsys[ForEach_itr].
      u2v3 - True0_B.CoreSubsys[ForEach_itr].u3v2;
    True0_B.CoreSubsys[ForEach_itr].Sum[1] = True0_B.CoreSubsys[ForEach_itr].
      u3v1 - True0_B.CoreSubsys[ForEach_itr].u1v3;
    True0_B.CoreSubsys[ForEach_itr].Sum[2] = True0_B.CoreSubsys[ForEach_itr].
      u1v2 - True0_B.CoreSubsys[ForEach_itr].u2v1;

    /* Product: '<S74>/Product4' */
    True0_B.CoreSubsys[ForEach_itr].Product4 = True0_B.CoreSubsys[ForEach_itr].
      Switch2 * rtb_ImpSel_InsertedFor_MotorMat[6];

    /* Product: '<S74>/Product5' incorporates:
     *  Product: '<S73>/Product1'
     */
    cphi = True0_B.CoreSubsys[ForEach_itr].Switch2 *
      True0_B.CoreSubsys[ForEach_itr].Switch2;

    /* Product: '<S74>/Product5' */
    True0_B.CoreSubsys[ForEach_itr].Product5 = cphi;

    /* Product: '<S74>/Product6' */
    True0_B.CoreSubsys[ForEach_itr].Product6 = True0_B.CoreSubsys[ForEach_itr].
      Product5 * rtb_ImpSel_InsertedFor_MotorMat[7];

    /* Sum: '<S74>/Sum1' */
    True0_B.CoreSubsys[ForEach_itr].Hoverthrustmagnitude =
      True0_B.CoreSubsys[ForEach_itr].Product4 + True0_B.CoreSubsys[ForEach_itr]
      .Product6;
    if (rtmIsMajorTimeStep((&True0_M))) {
      /* Gain: '<S68>/Conversion deg to rad' */
      True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad_n[0] = True0_P.d2r *
        rtb_ImpSel_InsertedFor_MotorMat[12];
      True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad_n[1] = True0_P.d2r *
        rtb_ImpSel_InsertedFor_MotorMat[13];
      True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad_n[2] = True0_P.d2r *
        rtb_ImpSel_InsertedFor_MotorMat[14];

      /* Trigonometry: '<S99>/Trigonometric Function1' incorporates:
       *  Trigonometry: '<S100>/Trigonometric Function1'
       *  Trigonometry: '<S104>/Trigonometric Function1'
       *  Trigonometry: '<S107>/Trigonometric Function1'
       */
      phi = std::cos(True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad_n[1]);

      /* Trigonometry: '<S99>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1 = phi;

      /* Trigonometry: '<S99>/Trigonometric Function3' incorporates:
       *  Trigonometry: '<S102>/Trigonometric Function3'
       *  Trigonometry: '<S103>/Trigonometric Function'
       *  Trigonometry: '<S105>/Trigonometric Function4'
       *  Trigonometry: '<S106>/Trigonometric Function'
       */
      ctheta = std::cos(True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad_n[2]);

      /* Trigonometry: '<S99>/Trigonometric Function3' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3 = ctheta;

      /* Product: '<S99>/Product' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[0] =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1 *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3;

      /* Trigonometry: '<S102>/Trigonometric Function12' incorporates:
       *  Trigonometry: '<S103>/Trigonometric Function12'
       *  Trigonometry: '<S104>/Trigonometric Function3'
       *  Trigonometry: '<S105>/Trigonometric Function5'
       *  Trigonometry: '<S106>/Trigonometric Function5'
       */
      theta = std::sin(True0_B.CoreSubsys[ForEach_itr].Conversiondegtorad_n[0]);

      /* Trigonometry: '<S102>/Trigonometric Function12' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction12 = theta;

      /* Trigonometry: '<S102>/Trigonometric Function1' incorporates:
       *  Trigonometry: '<S101>/Trigonometric Function1'
       *  Trigonometry: '<S103>/Trigonometric Function2'
       *  Trigonometry: '<S105>/Trigonometric Function2'
       *  Trigonometry: '<S106>/Trigonometric Function1'
       */
      Product_pk_tmp = std::sin(True0_B.CoreSubsys[ForEach_itr].
        Conversiondegtorad_n[1]);

      /* Trigonometry: '<S102>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_h = Product_pk_tmp;

      /* Trigonometry: '<S102>/Trigonometric Function3' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_m = ctheta;

      /* Product: '<S102>/Product' incorporates:
       *  Product: '<S103>/Product1'
       */
      Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction12 *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_h;

      /* Product: '<S102>/Product' */
      True0_B.CoreSubsys[ForEach_itr].Product_d = Product2_i4_tmp *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_m;

      /* Trigonometry: '<S102>/Trigonometric Function5' incorporates:
       *  Trigonometry: '<S103>/Trigonometric Function5'
       *  Trigonometry: '<S105>/Trigonometric Function12'
       *  Trigonometry: '<S106>/Trigonometric Function12'
       *  Trigonometry: '<S107>/Trigonometric Function3'
       */
      TrigonometricFunction5_tmp = std::cos(True0_B.CoreSubsys[ForEach_itr].
        Conversiondegtorad_n[0]);

      /* Trigonometry: '<S102>/Trigonometric Function5' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction5 =
        TrigonometricFunction5_tmp;

      /* Trigonometry: '<S102>/Trigonometric Function' incorporates:
       *  Trigonometry: '<S100>/Trigonometric Function3'
       *  Trigonometry: '<S103>/Trigonometric Function4'
       *  Trigonometry: '<S105>/Trigonometric Function'
       *  Trigonometry: '<S106>/Trigonometric Function3'
       */
      TrigonometricFunction_tmp = std::sin(True0_B.CoreSubsys[ForEach_itr].
        Conversiondegtorad_n[2]);

      /* Trigonometry: '<S102>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction =
        TrigonometricFunction_tmp;

      /* Product: '<S102>/Product1' */
      True0_B.CoreSubsys[ForEach_itr].Product1 = True0_B.CoreSubsys[ForEach_itr]
        .TrigonometricFunction5 * True0_B.CoreSubsys[ForEach_itr].
        TrigonometricFunction;

      /* Sum: '<S102>/Sum' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[1] =
        True0_B.CoreSubsys[ForEach_itr].Product_d -
        True0_B.CoreSubsys[ForEach_itr].Product1;

      /* Trigonometry: '<S105>/Trigonometric Function12' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction12_c =
        TrigonometricFunction5_tmp;

      /* Trigonometry: '<S105>/Trigonometric Function2' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction2 = Product_pk_tmp;

      /* Trigonometry: '<S105>/Trigonometric Function4' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction4 = ctheta;

      /* Product: '<S105>/Product1' incorporates:
       *  Product: '<S106>/Product'
       */
      Product1_a_tmp = True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction12_c
        * True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction2;

      /* Product: '<S105>/Product1' */
      True0_B.CoreSubsys[ForEach_itr].Product1_a = Product1_a_tmp *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction4;

      /* Trigonometry: '<S105>/Trigonometric Function5' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction5_f = theta;

      /* Trigonometry: '<S105>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_o =
        TrigonometricFunction_tmp;

      /* Product: '<S105>/Product2' */
      True0_B.CoreSubsys[ForEach_itr].Product2 = True0_B.CoreSubsys[ForEach_itr]
        .TrigonometricFunction5_f * True0_B.CoreSubsys[ForEach_itr].
        TrigonometricFunction_o;

      /* Sum: '<S105>/Sum' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[2] =
        True0_B.CoreSubsys[ForEach_itr].Product1_a +
        True0_B.CoreSubsys[ForEach_itr].Product2;

      /* Trigonometry: '<S100>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_c = phi;

      /* Trigonometry: '<S100>/Trigonometric Function3' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_e =
        TrigonometricFunction_tmp;

      /* Product: '<S100>/Product' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[3] =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_c *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_e;

      /* Trigonometry: '<S103>/Trigonometric Function12' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction12_n = theta;

      /* Trigonometry: '<S103>/Trigonometric Function2' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction2_a = Product_pk_tmp;

      /* Trigonometry: '<S103>/Trigonometric Function4' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction4_h =
        TrigonometricFunction_tmp;

      /* Product: '<S103>/Product1' */
      True0_B.CoreSubsys[ForEach_itr].Product1_f = Product2_i4_tmp *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction4_h;

      /* Trigonometry: '<S103>/Trigonometric Function5' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction5_a =
        TrigonometricFunction5_tmp;

      /* Trigonometry: '<S103>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_f = ctheta;

      /* Product: '<S103>/Product2' */
      True0_B.CoreSubsys[ForEach_itr].Product2_c =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction5_a *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_f;

      /* Sum: '<S103>/Sum' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[4] =
        True0_B.CoreSubsys[ForEach_itr].Product1_f +
        True0_B.CoreSubsys[ForEach_itr].Product2_c;

      /* Trigonometry: '<S106>/Trigonometric Function12' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction12_o =
        TrigonometricFunction5_tmp;

      /* Trigonometry: '<S106>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_l = Product_pk_tmp;

      /* Trigonometry: '<S106>/Trigonometric Function3' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_n =
        TrigonometricFunction_tmp;

      /* Product: '<S106>/Product' */
      True0_B.CoreSubsys[ForEach_itr].Product_e = Product1_a_tmp *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_n;

      /* Trigonometry: '<S106>/Trigonometric Function5' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction5_k = theta;

      /* Trigonometry: '<S106>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_f0 = ctheta;

      /* Product: '<S106>/Product1' */
      True0_B.CoreSubsys[ForEach_itr].Product1_j =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction5_k *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_f0;

      /* Sum: '<S106>/Sum' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[5] =
        True0_B.CoreSubsys[ForEach_itr].Product_e -
        True0_B.CoreSubsys[ForEach_itr].Product1_j;

      /* Trigonometry: '<S101>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_m = Product_pk_tmp;

      /* Gain: '<S101>/Gain' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[6] =
        True0_P.CoreSubsys.Gain_Gain_hr * True0_B.CoreSubsys[ForEach_itr].
        TrigonometricFunction1_m;

      /* Trigonometry: '<S104>/Trigonometric Function3' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_mh = theta;

      /* Trigonometry: '<S104>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_a = phi;

      /* Product: '<S104>/Product' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[7] =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_mh *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_a;

      /* Trigonometry: '<S107>/Trigonometric Function3' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_j =
        TrigonometricFunction5_tmp;

      /* Trigonometry: '<S107>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_m3 = phi;

      /* Product: '<S107>/Product' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       */
      True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[8] =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction3_j *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_m3;
    }

    /* Sum: '<S55>/Sum1' incorporates:
     *  Sum: '<S56>/Sum1'
     *  Sum: '<S58>/Sum'
     */
    True0_B.CoreSubsys[ForEach_itr].TotallinearvelocityatpropBodyax[0] =
      True0_B.CoreSubsys[ForEach_itr].Sum[0] + True0_B.TrueairspeedBodyaxes[0];
    True0_B.CoreSubsys[ForEach_itr].TotallinearvelocityatpropBodyax[1] =
      True0_B.CoreSubsys[ForEach_itr].Sum[1] + True0_B.TrueairspeedBodyaxes[1];
    True0_B.CoreSubsys[ForEach_itr].TotallinearvelocityatpropBodyax[2] =
      True0_B.CoreSubsys[ForEach_itr].Sum[2] + True0_B.TrueairspeedBodyaxes[2];

    /* Product: '<S68>/Product' incorporates:
     *  Concatenate: '<S108>/Vector Concatenate'
     *  Sum: '<S55>/Sum1'
     */
    std::memcpy(&tmp[0], &True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[0],
                9U * sizeof(real_T));
    phi = True0_B.CoreSubsys[ForEach_itr].TotallinearvelocityatpropBodyax[0];
    ctheta = True0_B.CoreSubsys[ForEach_itr].TotallinearvelocityatpropBodyax[1];
    theta = True0_B.CoreSubsys[ForEach_itr].TotallinearvelocityatpropBodyax[2];
    for (i = 0; i < 3; i++) {
      /* Product: '<S68>/Product' */
      Product2_i4_tmp = tmp[i] * phi;
      Product2_i4_tmp += tmp[i + 3] * ctheta;
      Product2_i4_tmp += tmp[i + 6] * theta;
      True0_B.CoreSubsys[ForEach_itr].TrueairspeedatpropMotoraxes[i] =
        Product2_i4_tmp;
    }

    /* Gain: '<S83>/Gain' */
    True0_B.CoreSubsys[ForEach_itr].Climbspeedv_c =
      True0_P.CoreSubsys.Gain_Gain_k * True0_B.CoreSubsys[ForEach_itr].
      TrueairspeedatpropMotoraxes[2];
    if (rtmIsMajorTimeStep((&True0_M))) {
      /* Outputs for IfAction SubSystem: '<S84>/Vortex ring state -2 <= vc//vh < 0 ' incorporates:
       *  ActionPort: '<S92>/Action Port'
       */
      /* If: '<S84>/If' incorporates:
       *  Constant: '<S71>/Induced velocity at hover'
       *  Product: '<S84>/Divide'
       *  Product: '<S92>/Divide'
       */
      phi = True0_B.CoreSubsys[ForEach_itr].Climbspeedv_c / True0_P.v_h;

      /* End of Outputs for SubSystem: '<S84>/Vortex ring state -2 <= vc//vh < 0 ' */

      /* Product: '<S84>/Divide' */
      True0_B.CoreSubsys[ForEach_itr].v_cv_h = phi;

      /* If: '<S84>/If' */
      if (rtsiIsModeUpdateTimeStep(&(&True0_M)->solverInfo)) {
        if (True0_B.CoreSubsys[ForEach_itr].v_cv_h >= 0.0) {
          rtAction = 0;
        } else if (True0_B.CoreSubsys[ForEach_itr].v_cv_h >= -2.0) {
          rtAction = 1;
        } else {
          rtAction = 2;
        }

        True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem = rtAction;
      } else {
        rtAction = True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem;
      }

      switch (rtAction) {
       case 0:
        /* Outputs for IfAction SubSystem: '<S84>/Normal working state vc//vh >= 0' incorporates:
         *  ActionPort: '<S91>/Action Port'
         */
        /* Gain: '<S91>/Gain' */
        True0_B.CoreSubsys[ForEach_itr].Gain_a = True0_P.CoreSubsys.Gain_Gain *
          True0_B.CoreSubsys[ForEach_itr].Climbspeedv_c;

        /* Product: '<S91>/Product' */
        True0_B.CoreSubsys[ForEach_itr].Product_o =
          True0_B.CoreSubsys[ForEach_itr].Gain_a *
          True0_B.CoreSubsys[ForEach_itr].Gain_a;

        /* Product: '<S91>/Product1' incorporates:
         *  Constant: '<S91>/Induced velocity at hover'
         */
        True0_B.CoreSubsys[ForEach_itr].Product1_h = True0_P.v_h * True0_P.v_h;

        /* Sum: '<S91>/Sum1' */
        True0_B.CoreSubsys[ForEach_itr].Sum1_i = True0_B.CoreSubsys[ForEach_itr]
          .Product_o + True0_B.CoreSubsys[ForEach_itr].Product1_h;

        /* Sqrt: '<S91>/Sqrt' */
        True0_B.CoreSubsys[ForEach_itr].Sqrt_e = std::sqrt
          (True0_B.CoreSubsys[ForEach_itr].Sum1_i);

        /* Merge: '<S84>/Merge' incorporates:
         *  Sum: '<S91>/Sum'
         */
        True0_B.CoreSubsys[ForEach_itr].Merge = True0_B.CoreSubsys[ForEach_itr].
          Sqrt_e - True0_B.CoreSubsys[ForEach_itr].Gain_a;

        /* End of Outputs for SubSystem: '<S84>/Normal working state vc//vh >= 0' */
        break;

       case 1:
        /* Outputs for IfAction SubSystem: '<S84>/Vortex ring state -2 <= vc//vh < 0 ' incorporates:
         *  ActionPort: '<S92>/Action Port'
         */
        /* Product: '<S92>/Divide' */
        True0_B.CoreSubsys[ForEach_itr].Divide_n = phi;

        /* Gain: '<S92>/Gain' */
        True0_B.CoreSubsys[ForEach_itr].Gain_e = True0_P.k1 *
          True0_B.CoreSubsys[ForEach_itr].Divide_n;

        /* Product: '<S92>/Product' */
        True0_B.CoreSubsys[ForEach_itr].Product_ge =
          True0_B.CoreSubsys[ForEach_itr].Divide_n *
          True0_B.CoreSubsys[ForEach_itr].Divide_n;

        /* Gain: '<S92>/Gain1' */
        True0_B.CoreSubsys[ForEach_itr].Gain1_e = True0_P.k2 *
          True0_B.CoreSubsys[ForEach_itr].Product_ge;

        /* Product: '<S92>/Product1' */
        True0_B.CoreSubsys[ForEach_itr].Product1_o =
          True0_B.CoreSubsys[ForEach_itr].Product_ge *
          True0_B.CoreSubsys[ForEach_itr].Divide_n;

        /* Gain: '<S92>/Gain2' */
        True0_B.CoreSubsys[ForEach_itr].Gain2_c = True0_P.k3 *
          True0_B.CoreSubsys[ForEach_itr].Product1_o;

        /* Product: '<S92>/Product2' */
        True0_B.CoreSubsys[ForEach_itr].Product2_es =
          True0_B.CoreSubsys[ForEach_itr].Product1_o *
          True0_B.CoreSubsys[ForEach_itr].Divide_n;

        /* Gain: '<S92>/Gain3' */
        True0_B.CoreSubsys[ForEach_itr].Gain3 = True0_P.k4 *
          True0_B.CoreSubsys[ForEach_itr].Product2_es;

        /* Sum: '<S92>/Add' incorporates:
         *  Constant: '<S92>/Constant'
         */
        True0_B.CoreSubsys[ForEach_itr].Add_c = (((True0_P.kappa +
          True0_B.CoreSubsys[ForEach_itr].Gain_e) +
          True0_B.CoreSubsys[ForEach_itr].Gain1_e) +
          True0_B.CoreSubsys[ForEach_itr].Gain2_c) +
          True0_B.CoreSubsys[ForEach_itr].Gain3;

        /* Merge: '<S84>/Merge' incorporates:
         *  Constant: '<S92>/Induced velocity at hover'
         *  Product: '<S92>/Product3'
         */
        True0_B.CoreSubsys[ForEach_itr].Merge = True0_B.CoreSubsys[ForEach_itr].
          Add_c * True0_P.v_h;

        /* End of Outputs for SubSystem: '<S84>/Vortex ring state -2 <= vc//vh < 0 ' */
        break;

       default:
        /* Outputs for IfAction SubSystem: '<S84>/Windmill braking state vc//vh < -2' incorporates:
         *  ActionPort: '<S93>/Action Port'
         */
        /* Gain: '<S93>/Gain' */
        True0_B.CoreSubsys[ForEach_itr].Gain_l = True0_P.CoreSubsys.Gain_Gain_b *
          True0_B.CoreSubsys[ForEach_itr].Climbspeedv_c;

        /* Product: '<S93>/Product' */
        True0_B.CoreSubsys[ForEach_itr].Product_j =
          True0_B.CoreSubsys[ForEach_itr].Gain_l *
          True0_B.CoreSubsys[ForEach_itr].Gain_l;

        /* Product: '<S93>/Product1' incorporates:
         *  Constant: '<S93>/Induced velocity at hover'
         */
        True0_B.CoreSubsys[ForEach_itr].Product1_c = True0_P.v_h * True0_P.v_h;

        /* Sum: '<S93>/Sum1' */
        True0_B.CoreSubsys[ForEach_itr].Sum1_c = True0_B.CoreSubsys[ForEach_itr]
          .Product_j - True0_B.CoreSubsys[ForEach_itr].Product1_c;

        /* Sqrt: '<S93>/Sqrt' */
        True0_B.CoreSubsys[ForEach_itr].Sqrt_d = std::sqrt
          (True0_B.CoreSubsys[ForEach_itr].Sum1_c);

        /* Merge: '<S84>/Merge' incorporates:
         *  Sum: '<S93>/Sum'
         */
        True0_B.CoreSubsys[ForEach_itr].Merge = (0.0 -
          True0_B.CoreSubsys[ForEach_itr].Gain_l) -
          True0_B.CoreSubsys[ForEach_itr].Sqrt_d;

        /* End of Outputs for SubSystem: '<S84>/Windmill braking state vc//vh < -2' */
        break;
      }
    }

    /* Math: '<S90>/transpose' incorporates:
     *  Product: '<S68>/Product'
     */
    Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
      TrueairspeedatpropMotoraxes[0];

    /* Math: '<S90>/transpose' */
    True0_B.CoreSubsys[ForEach_itr].transpose[0] = Product2_i4_tmp;

    /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
     *  ActionPort: '<S76>/Action Port'
     */
    /* Outputs for IfAction SubSystem: '<S82>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S85>/Action Port'
     */
    /* If: '<S70>/If' incorporates:
     *  If: '<S82>/If'
     *  Product: '<S80>/Product'
     *  Product: '<S81>/Product'
     *  Product: '<S89>/Product'
     *  Product: '<S90>/Product'
     */
    Product_pk_tmp = Product2_i4_tmp * Product2_i4_tmp;

    /* End of Outputs for SubSystem: '<S82>/Nonzero airspeed' */
    /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */

    /* Math: '<S90>/transpose' incorporates:
     *  Product: '<S68>/Product'
     */
    Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
      TrueairspeedatpropMotoraxes[1];

    /* Math: '<S90>/transpose' */
    True0_B.CoreSubsys[ForEach_itr].transpose[1] = Product2_i4_tmp;

    /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
     *  ActionPort: '<S76>/Action Port'
     */
    /* Outputs for IfAction SubSystem: '<S82>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S85>/Action Port'
     */
    /* If: '<S70>/If' incorporates:
     *  If: '<S82>/If'
     *  Product: '<S80>/Product'
     *  Product: '<S81>/Product'
     *  Product: '<S89>/Product'
     *  Product: '<S90>/Product'
     */
    phi = Product2_i4_tmp * Product2_i4_tmp + Product_pk_tmp;

    /* End of Outputs for SubSystem: '<S82>/Nonzero airspeed' */
    /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */

    /* Product: '<S90>/Product' */
    Product_pk_tmp = phi;

    /* Math: '<S90>/transpose' incorporates:
     *  Product: '<S68>/Product'
     */
    Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
      TrueairspeedatpropMotoraxes[2];

    /* Math: '<S90>/transpose' */
    True0_B.CoreSubsys[ForEach_itr].transpose[2] = Product2_i4_tmp;

    /* Product: '<S90>/Product' */
    Product_pk_tmp += Product2_i4_tmp * Product2_i4_tmp;

    /* Product: '<S90>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_g = Product_pk_tmp;

    /* Sqrt: '<S87>/Sqrt' */
    True0_B.CoreSubsys[ForEach_itr].Sqrt = std::sqrt
      (True0_B.CoreSubsys[ForEach_itr].Product_g);

    /* ComplexToRealImag: '<S87>/Complex to Real-Imag' */
    True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag =
      True0_B.CoreSubsys[ForEach_itr].Sqrt;

    /* If: '<S82>/If' */
    if (rtsiIsModeUpdateTimeStep(&(&True0_M)->solverInfo)) {
      rtAction = static_cast<int8_T>(!(True0_B.CoreSubsys[ForEach_itr].
        ComplextoRealImag == 0.0));
      True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_l = rtAction;
    } else {
      rtAction = True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_l;
    }

    if (rtAction == 0) {
      /* Outputs for IfAction SubSystem: '<S82>/Zero airspeed' incorporates:
       *  ActionPort: '<S86>/Action Port'
       */
      if (rtmIsMajorTimeStep((&True0_M))) {
        /* Merge: '<S82>/Merge' incorporates:
         *  Constant: '<S86>/Constant'
         */
        True0_B.CoreSubsys[ForEach_itr].Angleofattackrad =
          True0_P.CoreSubsys.Constant_Value;
      }

      /* End of Outputs for SubSystem: '<S82>/Zero airspeed' */
    } else {
      /* Outputs for IfAction SubSystem: '<S82>/Nonzero airspeed' incorporates:
       *  ActionPort: '<S85>/Action Port'
       */
      /* Math: '<S89>/transpose' */
      Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
        TrueairspeedatpropMotoraxes[0];

      /* End of Outputs for SubSystem: '<S82>/Nonzero airspeed' */

      /* Math: '<S89>/transpose' */
      True0_B.CoreSubsys[ForEach_itr].transpose_i[0] = Product2_i4_tmp;

      /* Outputs for IfAction SubSystem: '<S82>/Nonzero airspeed' incorporates:
       *  ActionPort: '<S85>/Action Port'
       */
      /* Math: '<S89>/transpose' */
      Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
        TrueairspeedatpropMotoraxes[1];

      /* End of Outputs for SubSystem: '<S82>/Nonzero airspeed' */

      /* Math: '<S89>/transpose' */
      True0_B.CoreSubsys[ForEach_itr].transpose_i[1] = Product2_i4_tmp;

      /* Outputs for IfAction SubSystem: '<S82>/Nonzero airspeed' incorporates:
       *  ActionPort: '<S85>/Action Port'
       */
      /* Product: '<S89>/Product' */
      True0_B.CoreSubsys[ForEach_itr].Product_l = phi;

      /* Sqrt: '<S88>/Sqrt' */
      True0_B.CoreSubsys[ForEach_itr].Sqrt_a = std::sqrt
        (True0_B.CoreSubsys[ForEach_itr].Product_l);

      /* ComplexToRealImag: '<S88>/Complex to Real-Imag' */
      True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_n =
        True0_B.CoreSubsys[ForEach_itr].Sqrt_a;

      /* Product: '<S85>/Divide1' */
      True0_B.CoreSubsys[ForEach_itr].Divide1 = 1.0 /
        True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_n *
        True0_B.CoreSubsys[ForEach_itr].TrueairspeedatpropMotoraxes[2];

      /* Merge: '<S82>/Merge' incorporates:
       *  Trigonometry: '<S85>/Trigonometric Function'
       */
      True0_B.CoreSubsys[ForEach_itr].Angleofattackrad = std::atan
        (True0_B.CoreSubsys[ForEach_itr].Divide1);

      /* End of Outputs for SubSystem: '<S82>/Nonzero airspeed' */
    }

    /* Switch: '<S71>/Switch' incorporates:
     *  Constant: '<S71>/Constant1'
     */
    if (True0_P.Dyn_thrust > True0_P.CoreSubsys.Switch_Threshold_n) {
      /* Trigonometry: '<S71>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_j = std::sin
        (True0_B.CoreSubsys[ForEach_itr].Angleofattackrad);

      /* Product: '<S71>/Product2' */
      True0_B.CoreSubsys[ForEach_itr].Product2_nr =
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_j *
        True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag;

      /* Sum: '<S71>/Sum2' */
      True0_B.CoreSubsys[ForEach_itr].Sum2 = True0_B.CoreSubsys[ForEach_itr].
        Merge - True0_B.CoreSubsys[ForEach_itr].Product2_nr;

      /* Product: '<S71>/Divide' incorporates:
       *  Constant: '<S71>/Induced velocity at hover'
       */
      True0_B.CoreSubsys[ForEach_itr].Divide_p = True0_P.v_h /
        True0_B.CoreSubsys[ForEach_itr].Sum2;

      /* Switch: '<S71>/Switch' */
      True0_B.CoreSubsys[ForEach_itr].ThrustratioTT_h =
        True0_B.CoreSubsys[ForEach_itr].Divide_p;
    } else {
      /* Switch: '<S71>/Switch' incorporates:
       *  Constant: '<S71>/Constant'
       */
      True0_B.CoreSubsys[ForEach_itr].ThrustratioTT_h =
        True0_P.CoreSubsys.Constant_Value_im;
    }

    /* End of Switch: '<S71>/Switch' */

    /* Product: '<S67>/Product7' */
    True0_B.CoreSubsys[ForEach_itr].Dynamicthrustmagnitude =
      True0_B.CoreSubsys[ForEach_itr].Hoverthrustmagnitude *
      True0_B.CoreSubsys[ForEach_itr].ThrustratioTT_h;

    /* Math: '<S81>/transpose' */
    Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
      TrueairspeedatpropMotoraxes[0];

    /* Math: '<S81>/transpose' */
    True0_B.CoreSubsys[ForEach_itr].transpose_e[0] = Product2_i4_tmp;

    /* Math: '<S81>/transpose' */
    Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
      TrueairspeedatpropMotoraxes[1];

    /* Math: '<S81>/transpose' */
    True0_B.CoreSubsys[ForEach_itr].transpose_e[1] = Product2_i4_tmp;

    /* Product: '<S81>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_eh = phi;

    /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
     *  ActionPort: '<S76>/Action Port'
     */
    /* If: '<S70>/If' incorporates:
     *  Sqrt: '<S78>/Sqrt'
     *  Sqrt: '<S79>/Sqrt'
     */
    ctheta = std::sqrt(True0_B.CoreSubsys[ForEach_itr].Product_eh);

    /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */

    /* Sqrt: '<S78>/Sqrt' */
    True0_B.CoreSubsys[ForEach_itr].Sqrt_n = ctheta;

    /* ComplexToRealImag: '<S78>/Complex to Real-Imag' */
    True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_p =
      True0_B.CoreSubsys[ForEach_itr].Sqrt_n;

    /* If: '<S70>/If' */
    if (rtsiIsModeUpdateTimeStep(&(&True0_M)->solverInfo)) {
      rtAction = static_cast<int8_T>(!(True0_B.CoreSubsys[ForEach_itr].
        ComplextoRealImag_p == 0.0));
      True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_e = rtAction;
    } else {
      rtAction = True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_e;
    }

    if (rtAction == 0) {
      /* Outputs for IfAction SubSystem: '<S70>/Zero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S77>/Action Port'
       */
      if (rtmIsMajorTimeStep((&True0_M))) {
        /* Merge: '<S70>/Merge' incorporates:
         *  Constant: '<S77>/Constant'
         */
        True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0] =
          True0_P.CoreSubsys.Constant_Value_l[0];

        /* Merge: '<S70>/Merge1' incorporates:
         *  Constant: '<S77>/Constant1'
         */
        True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[0] =
          True0_P.CoreSubsys.Constant1_Value[0];

        /* Merge: '<S70>/Merge' incorporates:
         *  Constant: '<S77>/Constant'
         */
        True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1] =
          True0_P.CoreSubsys.Constant_Value_l[1];

        /* Merge: '<S70>/Merge1' incorporates:
         *  Constant: '<S77>/Constant1'
         */
        True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[1] =
          True0_P.CoreSubsys.Constant1_Value[1];

        /* Merge: '<S70>/Merge' incorporates:
         *  Constant: '<S77>/Constant'
         */
        True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2] =
          True0_P.CoreSubsys.Constant_Value_l[2];

        /* Merge: '<S70>/Merge1' incorporates:
         *  Constant: '<S77>/Constant1'
         */
        True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[2] =
          True0_P.CoreSubsys.Constant1_Value[2];
      }

      /* End of Outputs for SubSystem: '<S70>/Zero airspeed in rotor plane' */
    } else {
      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Math: '<S80>/transpose' */
      Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
        TrueairspeedatpropMotoraxes[0];

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */

      /* Math: '<S80>/transpose' */
      True0_B.CoreSubsys[ForEach_itr].transpose_h[0] = Product2_i4_tmp;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Math: '<S80>/transpose' */
      Product2_i4_tmp = True0_B.CoreSubsys[ForEach_itr].
        TrueairspeedatpropMotoraxes[1];

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */

      /* Math: '<S80>/transpose' */
      True0_B.CoreSubsys[ForEach_itr].transpose_h[1] = Product2_i4_tmp;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Product: '<S80>/Product' */
      True0_B.CoreSubsys[ForEach_itr].Product_dg = phi;

      /* Sqrt: '<S79>/Sqrt' */
      True0_B.CoreSubsys[ForEach_itr].Sqrt_f = ctheta;

      /* ComplexToRealImag: '<S79>/Complex to Real-Imag' */
      True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_m =
        True0_B.CoreSubsys[ForEach_itr].Sqrt_f;

      /* Switch: '<S76>/Switch' incorporates:
       *  Constant: '<S76>/Constant2'
       */
      if (True0_P.Blade_flapping > True0_P.CoreSubsys.Switch_Threshold) {
        /* Switch: '<S76>/Switch' incorporates:
         *  Constant: '<S76>/Blade flapping gain [deg//(m//s)]'
         */
        True0_B.CoreSubsys[ForEach_itr].Switch_e = True0_P.k_a1s;
      } else {
        /* Switch: '<S76>/Switch' incorporates:
         *  Constant: '<S76>/Blade flapping disengaged'
         */
        True0_B.CoreSubsys[ForEach_itr].Switch_e =
          True0_P.CoreSubsys.Bladeflappingdisengaged_Value;
      }

      /* End of Switch: '<S76>/Switch' */

      /* Product: '<S76>/Product4' */
      True0_B.CoreSubsys[ForEach_itr].Bladeflappinganglea_1sdeg =
        True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_m *
        True0_B.CoreSubsys[ForEach_itr].Switch_e;

      /* Gain: '<S76>/Conversion deg to rad' */
      True0_B.CoreSubsys[ForEach_itr].Flappinganglerad = True0_P.d2r *
        True0_B.CoreSubsys[ForEach_itr].Bladeflappinganglea_1sdeg;

      /* Trigonometry: '<S76>/Trigonometric Function1' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_o = std::cos
        (True0_B.CoreSubsys[ForEach_itr].Flappinganglerad);

      /* Gain: '<S76>/Gain1' */
      True0_B.CoreSubsys[ForEach_itr].Gain1_o = True0_P.CoreSubsys.Gain1_Gain *
        True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction1_o;

      /* Trigonometry: '<S76>/Trigonometric Function' */
      True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_p = std::sin
        (True0_B.CoreSubsys[ForEach_itr].Flappinganglerad);

      /* Product: '<S76>/Product1' incorporates:
       *  Constant: '<S76>/Constant'
       *  Constant: '<S76>/Constant1'
       */
      True0_B.CoreSubsys[ForEach_itr].MotorhubmomentMotoraxes[2] =
        True0_P.CoreSubsys.Constant_Value_i * True0_P.k_beta *
        True0_B.CoreSubsys[ForEach_itr].Flappinganglerad;

      /* Product: '<S76>/Divide' */
      phi = True0_B.CoreSubsys[ForEach_itr].TrueairspeedatpropMotoraxes[0] /
        True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_m;

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
      True0_B.CoreSubsys[ForEach_itr].Airspeeddirectionintherotorplan[0] = phi;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Gain: '<S76>/Gain' incorporates:
       *  Product: '<S76>/Divide'
       */
      phi *= True0_P.CoreSubsys.Gain_Gain_h;

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
      True0_B.CoreSubsys[ForEach_itr].Gain_m[0] = phi;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Product: '<S76>/Product' incorporates:
       *  Gain: '<S76>/Gain'
       */
      phi *= True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_p;

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
      True0_B.CoreSubsys[ForEach_itr].Product_d5[0] = phi;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Reshape: '<S76>/Reshape1' */
      True0_B.CoreSubsys[ForEach_itr].Reshape1[0] = phi;

      /* Product: '<S76>/Divide' */
      phi = True0_B.CoreSubsys[ForEach_itr].TrueairspeedatpropMotoraxes[1] /
        True0_B.CoreSubsys[ForEach_itr].ComplextoRealImag_m;

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
      True0_B.CoreSubsys[ForEach_itr].Airspeeddirectionintherotorplan[1] = phi;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Gain: '<S76>/Gain' incorporates:
       *  Product: '<S76>/Divide'
       */
      phi *= True0_P.CoreSubsys.Gain_Gain_h;

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
      True0_B.CoreSubsys[ForEach_itr].Gain_m[1] = phi;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Product: '<S76>/Product' incorporates:
       *  Gain: '<S76>/Gain'
       */
      phi *= True0_B.CoreSubsys[ForEach_itr].TrigonometricFunction_p;

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
      True0_B.CoreSubsys[ForEach_itr].Product_d5[1] = phi;

      /* Outputs for IfAction SubSystem: '<S70>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S76>/Action Port'
       */
      /* Reshape: '<S76>/Reshape1' */
      True0_B.CoreSubsys[ForEach_itr].Reshape1[1] = phi;

      /* Gain: '<S76>/Gain2' */
      True0_B.CoreSubsys[ForEach_itr].Gain2 = True0_P.CoreSubsys.Gain2_Gain *
        True0_B.CoreSubsys[ForEach_itr].Airspeeddirectionintherotorplan[1];

      /* Product: '<S76>/Product1' incorporates:
       *  Constant: '<S76>/Constant1'
       */
      True0_B.CoreSubsys[ForEach_itr].MotorhubmomentMotoraxes[0] =
        True0_B.CoreSubsys[ForEach_itr].Gain2 * True0_P.k_beta *
        True0_B.CoreSubsys[ForEach_itr].Flappinganglerad;
      True0_B.CoreSubsys[ForEach_itr].MotorhubmomentMotoraxes[1] =
        True0_B.CoreSubsys[ForEach_itr].Airspeeddirectionintherotorplan[0] *
        True0_P.k_beta * True0_B.CoreSubsys[ForEach_itr].Flappinganglerad;

      /* Reshape: '<S76>/Reshape1' */
      True0_B.CoreSubsys[ForEach_itr].Reshape1[2] =
        True0_B.CoreSubsys[ForEach_itr].Gain1_o;

      /* Product: '<S76>/Product2' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       *  Reshape: '<S76>/Reshape1'
       */
      phi = True0_B.CoreSubsys[ForEach_itr].Reshape1[0];
      ctheta = True0_B.CoreSubsys[ForEach_itr].Reshape1[1];
      theta = True0_B.CoreSubsys[ForEach_itr].Reshape1[2];
      std::memcpy(&tmp[0], &True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[0],
                  9U * sizeof(real_T));
      for (i = 0; i < 3; i++) {
        /* Product: '<S76>/Product2' */
        Product_pk_tmp = tmp[3 * i] * phi;
        Product_pk_tmp += tmp[3 * i + 1] * ctheta;
        Product_pk_tmp += tmp[3 * i + 2] * theta;
        True0_B.CoreSubsys[ForEach_itr].Product2_j[i] = Product_pk_tmp;
      }

      /* Product: '<S76>/Product3' incorporates:
       *  Concatenate: '<S108>/Vector Concatenate'
       *  Product: '<S76>/Product1'
       */
      phi = True0_B.CoreSubsys[ForEach_itr].MotorhubmomentMotoraxes[0];
      ctheta = True0_B.CoreSubsys[ForEach_itr].MotorhubmomentMotoraxes[1];
      theta = True0_B.CoreSubsys[ForEach_itr].MotorhubmomentMotoraxes[2];
      std::memcpy(&tmp[0], &True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[0],
                  9U * sizeof(real_T));
      for (i = 0; i < 3; i++) {
        /* Product: '<S76>/Product3' */
        Product_pk_tmp = tmp[3 * i] * phi;
        Product_pk_tmp += tmp[3 * i + 1] * ctheta;
        Product_pk_tmp += tmp[3 * i + 2] * theta;
        True0_B.CoreSubsys[ForEach_itr].Product3[i] = Product_pk_tmp;

        /* Merge: '<S70>/Merge' incorporates:
         *  Product: '<S76>/Product2'
         *  Reshape: '<S76>/Reshape2'
         */
        True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[i] =
          True0_B.CoreSubsys[ForEach_itr].Product2_j[i];

        /* Merge: '<S70>/Merge1' incorporates:
         *  Product: '<S76>/Product3'
         *  Reshape: '<S76>/Reshape4'
         */
        True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[i] =
          Product_pk_tmp;
      }

      /* End of Outputs for SubSystem: '<S70>/Nonzero airspeed in rotor plane' */
    }

    /* Product: '<S67>/Product9' incorporates:
     *  Merge: '<S70>/Merge'
     */
    True0_B.CoreSubsys[ForEach_itr].Product9[0] = True0_B.CoreSubsys[ForEach_itr]
      .Dynamicthrustmagnitude * True0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[0];
    True0_B.CoreSubsys[ForEach_itr].Product9[1] = True0_B.CoreSubsys[ForEach_itr]
      .Dynamicthrustmagnitude * True0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[1];
    True0_B.CoreSubsys[ForEach_itr].Product9[2] = True0_B.CoreSubsys[ForEach_itr]
      .Dynamicthrustmagnitude * True0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[2];

    /* Product: '<S73>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_i = True0_B.CoreSubsys[ForEach_itr].
      Switch2 * rtb_ImpSel_InsertedFor_MotorMat[8];

    /* Product: '<S73>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].Product1_ar = cphi *
      rtb_ImpSel_InsertedFor_MotorMat[9];

    /* Sum: '<S73>/Sum' */
    True0_B.CoreSubsys[ForEach_itr].Motortorquemagnitude =
      True0_B.CoreSubsys[ForEach_itr].Product_i + True0_B.CoreSubsys[ForEach_itr]
      .Product1_ar;
    if (rtmIsMajorTimeStep((&True0_M))) {
      for (i = 0; i < 3; i++) {
        /* Math: '<S75>/Math Function' incorporates:
         *  Concatenate: '<S108>/Vector Concatenate'
         */
        True0_B.CoreSubsys[ForEach_itr].MathFunction[3 * i] =
          True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[i];
        True0_B.CoreSubsys[ForEach_itr].MathFunction[3 * i + 1] =
          True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[i + 3];
        True0_B.CoreSubsys[ForEach_itr].MathFunction[3 * i + 2] =
          True0_B.CoreSubsys[ForEach_itr].VectorConcatenate[i + 6];
      }

      /* Product: '<S75>/Product9' incorporates:
       *  Constant: '<S75>/Constant'
       *  Math: '<S75>/Math Function'
       */
      std::memcpy(&tmp[0], &True0_B.CoreSubsys[ForEach_itr].MathFunction[0], 9U *
                  sizeof(real_T));
      phi = True0_P.CoreSubsys.Constant_Value_b[0];
      ctheta = True0_P.CoreSubsys.Constant_Value_b[1];
      theta = True0_P.CoreSubsys.Constant_Value_b[2];
      for (i = 0; i < 3; i++) {
        Product_pk_tmp = tmp[i] * phi;
        Product_pk_tmp += tmp[i + 3] * ctheta;
        Product_pk_tmp += tmp[i + 6] * theta;

        /* Product: '<S75>/Product9' */
        True0_B.CoreSubsys[ForEach_itr].Product9_p[i] = Product_pk_tmp;
      }

      /* End of Product: '<S75>/Product9' */
    }

    /* Product: '<S67>/Product3' */
    cphi = True0_B.CoreSubsys[ForEach_itr].Motortorquemagnitude *
      rtb_ImpSel_InsertedFor_MotorMat[3];

    /* Product: '<S67>/Product3' incorporates:
     *  Product: '<S75>/Product9'
     */
    phi = cphi * True0_B.CoreSubsys[ForEach_itr].Product9_p[0];
    True0_B.CoreSubsys[ForEach_itr].Momentinducedbyaerodynamicdragp[0] = phi;

    /* Product: '<S67>/Product8' incorporates:
     *  Product: '<S67>/Product3'
     */
    True0_B.CoreSubsys[ForEach_itr].Product8[0] = phi *
      True0_B.CoreSubsys[ForEach_itr].ThrustratioTT_h;

    /* Product: '<S67>/Product3' incorporates:
     *  Product: '<S75>/Product9'
     */
    phi = cphi * True0_B.CoreSubsys[ForEach_itr].Product9_p[1];
    True0_B.CoreSubsys[ForEach_itr].Momentinducedbyaerodynamicdragp[1] = phi;

    /* Product: '<S67>/Product8' incorporates:
     *  Product: '<S67>/Product3'
     */
    True0_B.CoreSubsys[ForEach_itr].Product8[1] = phi *
      True0_B.CoreSubsys[ForEach_itr].ThrustratioTT_h;

    /* Product: '<S67>/Product3' incorporates:
     *  Product: '<S75>/Product9'
     */
    phi = cphi * True0_B.CoreSubsys[ForEach_itr].Product9_p[2];
    True0_B.CoreSubsys[ForEach_itr].Momentinducedbyaerodynamicdragp[2] = phi;

    /* Product: '<S67>/Product8' incorporates:
     *  Product: '<S67>/Product3'
     */
    True0_B.CoreSubsys[ForEach_itr].Product8[2] = phi *
      True0_B.CoreSubsys[ForEach_itr].ThrustratioTT_h;
    if (rtmIsMajorTimeStep((&True0_M))) {
      /* Gain: '<S95>/Gain' */
      True0_B.CoreSubsys[ForEach_itr].Gain = True0_P.CoreSubsys.Gain_Gain_n *
        rtb_ImpSel_InsertedFor_MotorMat[15];

      /* Product: '<S95>/Product7' */
      True0_B.CoreSubsys[ForEach_itr].Product7 = True0_B.CoreSubsys[ForEach_itr]
        .Gain * True0_B.CoreSubsys[ForEach_itr].Gain *
        rtb_ImpSel_InsertedFor_MotorMat[16];

      /* Gain: '<S95>/Gain1' */
      True0_B.CoreSubsys[ForEach_itr].Gain1 = True0_P.CoreSubsys.Gain1_Gain_c *
        True0_B.CoreSubsys[ForEach_itr].Product7;
    }

    /* Product: '<S96>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_h = True0_B.omega[1] *
      True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2];

    /* Product: '<S96>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].Product1_m = True0_B.CoreSubsys[ForEach_itr]
      .NewtiltedthrustdirectionBodyaxe[0] * True0_B.omega[2];

    /* Product: '<S96>/Product2' */
    True0_B.CoreSubsys[ForEach_itr].Product2_e = True0_B.omega[0] *
      True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1];

    /* Product: '<S97>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_i4 = True0_B.CoreSubsys[ForEach_itr]
      .NewtiltedthrustdirectionBodyaxe[1] * True0_B.omega[2];

    /* Product: '<S97>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].Product1_ar5 = True0_B.omega[0] *
      True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2];

    /* Product: '<S97>/Product2' */
    True0_B.CoreSubsys[ForEach_itr].Product2_h = True0_B.CoreSubsys[ForEach_itr]
      .NewtiltedthrustdirectionBodyaxe[0] * True0_B.omega[1];

    /* Sum: '<S94>/Sum' */
    True0_B.CoreSubsys[ForEach_itr].Sum_j[0] = True0_B.CoreSubsys[ForEach_itr].
      Product_h - True0_B.CoreSubsys[ForEach_itr].Product_i4;
    True0_B.CoreSubsys[ForEach_itr].Sum_j[1] = True0_B.CoreSubsys[ForEach_itr].
      Product1_m - True0_B.CoreSubsys[ForEach_itr].Product1_ar5;
    True0_B.CoreSubsys[ForEach_itr].Sum_j[2] = True0_B.CoreSubsys[ForEach_itr].
      Product2_e - True0_B.CoreSubsys[ForEach_itr].Product2_h;

    /* Gain: '<S72>/Conversion rpm to rad//s' */
    True0_B.CoreSubsys[ForEach_itr].Conversionrpmtorads = True0_P.rpm2radpersec *
      True0_B.CoreSubsys[ForEach_itr].Switch2;

    /* Product: '<S109>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_ez = True0_B.CoreSubsys[ForEach_itr]
      .VectorfromrealCoGtopropellerBod[1] * True0_B.CoreSubsys[ForEach_itr].
      Product9[2];

    /* Product: '<S109>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].Product1_jt = True0_B.CoreSubsys[ForEach_itr]
      .Product9[0] * True0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[2];

    /* Product: '<S109>/Product2' */
    True0_B.CoreSubsys[ForEach_itr].Product2_p = True0_B.CoreSubsys[ForEach_itr]
      .VectorfromrealCoGtopropellerBod[0] * True0_B.CoreSubsys[ForEach_itr].
      Product9[1];

    /* Product: '<S110>/Product' */
    True0_B.CoreSubsys[ForEach_itr].Product_m = True0_B.CoreSubsys[ForEach_itr].
      Product9[1] * True0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[2];

    /* Product: '<S110>/Product1' */
    True0_B.CoreSubsys[ForEach_itr].Product1_k = True0_B.CoreSubsys[ForEach_itr]
      .VectorfromrealCoGtopropellerBod[0] * True0_B.CoreSubsys[ForEach_itr].
      Product9[2];

    /* Product: '<S110>/Product2' */
    True0_B.CoreSubsys[ForEach_itr].Product2_n = True0_B.CoreSubsys[ForEach_itr]
      .Product9[0] * True0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[1];

    /* Sum: '<S69>/Sum' */
    True0_B.CoreSubsys[ForEach_itr].Sum_h[0] = True0_B.CoreSubsys[ForEach_itr].
      Product_ez - True0_B.CoreSubsys[ForEach_itr].Product_m;
    True0_B.CoreSubsys[ForEach_itr].Sum_h[1] = True0_B.CoreSubsys[ForEach_itr].
      Product1_jt - True0_B.CoreSubsys[ForEach_itr].Product1_k;
    True0_B.CoreSubsys[ForEach_itr].Sum_h[2] = True0_B.CoreSubsys[ForEach_itr].
      Product2_p - True0_B.CoreSubsys[ForEach_itr].Product2_n;

    /* Product: '<S72>/Product5' */
    cphi = True0_B.CoreSubsys[ForEach_itr].Gain1 *
      rtb_ImpSel_InsertedFor_MotorMat[3];

    /* Product: '<S72>/Product5' */
    phi = cphi * True0_B.CoreSubsys[ForEach_itr].Sum_j[0] *
      True0_B.CoreSubsys[ForEach_itr].Conversionrpmtorads;
    True0_B.CoreSubsys[ForEach_itr].Product5_k[0] = phi;

    /* Sum: '<S60>/Add' incorporates:
     *  Merge: '<S70>/Merge1'
     *  Product: '<S67>/Product8'
     *  Product: '<S72>/Product5'
     *  Sum: '<S69>/Sum'
     */
    phi = ((True0_B.CoreSubsys[ForEach_itr].Product8[0] +
            True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[0])
           + phi) + True0_B.CoreSubsys[ForEach_itr].Sum_h[0];
    True0_B.CoreSubsys[ForEach_itr].Add[0] = phi;

    /* ForEachSliceAssignment generated from: '<S55>/Motor_moment' incorporates:
     *  Sum: '<S60>/Add'
     */
    True0_B.ImpAsg_InsertedFor_Motor_moment[3 * ForEach_itr] = phi;

    /* ForEachSliceAssignment generated from: '<S55>/Motor_force' incorporates:
     *  Product: '<S67>/Product9'
     */
    True0_B.ImpAsg_InsertedFor_Motor_force_[3 * ForEach_itr] =
      True0_B.CoreSubsys[ForEach_itr].Product9[0];

    /* Product: '<S72>/Product5' */
    phi = cphi * True0_B.CoreSubsys[ForEach_itr].Sum_j[1] *
      True0_B.CoreSubsys[ForEach_itr].Conversionrpmtorads;
    True0_B.CoreSubsys[ForEach_itr].Product5_k[1] = phi;

    /* Sum: '<S60>/Add' incorporates:
     *  Merge: '<S70>/Merge1'
     *  Product: '<S67>/Product8'
     *  Product: '<S72>/Product5'
     *  Sum: '<S69>/Sum'
     */
    phi = ((True0_B.CoreSubsys[ForEach_itr].Product8[1] +
            True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[1])
           + phi) + True0_B.CoreSubsys[ForEach_itr].Sum_h[1];
    True0_B.CoreSubsys[ForEach_itr].Add[1] = phi;

    /* ForEachSliceAssignment generated from: '<S55>/Motor_moment' incorporates:
     *  ForEachSliceAssignment generated from: '<S55>/Motor_force'
     *  Sum: '<S60>/Add'
     */
    i = 3 * ForEach_itr + 1;
    True0_B.ImpAsg_InsertedFor_Motor_moment[i] = phi;

    /* ForEachSliceAssignment generated from: '<S55>/Motor_force' incorporates:
     *  Product: '<S67>/Product9'
     */
    True0_B.ImpAsg_InsertedFor_Motor_force_[i] = True0_B.CoreSubsys[ForEach_itr]
      .Product9[1];

    /* Product: '<S72>/Product5' */
    phi = cphi * True0_B.CoreSubsys[ForEach_itr].Sum_j[2] *
      True0_B.CoreSubsys[ForEach_itr].Conversionrpmtorads;
    True0_B.CoreSubsys[ForEach_itr].Product5_k[2] = phi;

    /* Sum: '<S60>/Add' incorporates:
     *  Merge: '<S70>/Merge1'
     *  Product: '<S67>/Product8'
     *  Product: '<S72>/Product5'
     *  Sum: '<S69>/Sum'
     */
    phi = ((True0_B.CoreSubsys[ForEach_itr].Product8[2] +
            True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[2])
           + phi) + True0_B.CoreSubsys[ForEach_itr].Sum_h[2];
    True0_B.CoreSubsys[ForEach_itr].Add[2] = phi;

    /* ForEachSliceAssignment generated from: '<S55>/Motor_moment' incorporates:
     *  ForEachSliceAssignment generated from: '<S55>/Motor_force'
     *  Sum: '<S60>/Add'
     */
    i = 3 * ForEach_itr + 2;
    True0_B.ImpAsg_InsertedFor_Motor_moment[i] = phi;

    /* ForEachSliceAssignment generated from: '<S55>/Motor_force' incorporates:
     *  Product: '<S67>/Product9'
     */
    True0_B.ImpAsg_InsertedFor_Motor_force_[i] = True0_B.CoreSubsys[ForEach_itr]
      .Product9[2];
  }

  /* End of Outputs for SubSystem: '<S37>/For Each Subsystem' */

  /* Sum: '<S37>/Sum of Elements' incorporates:
   *  ForEachSliceAssignment generated from: '<S55>/Motor_force'
   */
  for (i = 0; i < 3; i++) {
    tmp_0 = i;
    cphi = True0_B.ImpAsg_InsertedFor_Motor_force_[tmp_0];
    tmp_0 = i + 3;
    cphi += True0_B.ImpAsg_InsertedFor_Motor_force_[tmp_0];
    tmp_0 = i + 6;
    cphi += True0_B.ImpAsg_InsertedFor_Motor_force_[tmp_0];
    tmp_0 = i + 9;
    cphi += True0_B.ImpAsg_InsertedFor_Motor_force_[tmp_0];
    True0_B.SumofElements[i] = cphi;
  }

  /* End of Sum: '<S37>/Sum of Elements' */
  if (rtmIsMajorTimeStep((&True0_M))) {
    /* Product: '<S36>/Product1' incorporates:
     *  Constant: '<S36>/Gravity (Inertial axes)'
     *  Inport: '<Root>/mass_real'
     */
    True0_B.ForceofgravityInertialaxes[0] = True0_P.GravityInertialaxes_Value[0]
      * True0_U.mass_real;
    True0_B.ForceofgravityInertialaxes[1] = True0_P.GravityInertialaxes_Value[1]
      * True0_U.mass_real;
    True0_B.ForceofgravityInertialaxes[2] = True0_P.GravityInertialaxes_Value[2]
      * True0_U.mass_real;
  }

  /* Product: '<S36>/Product' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  std::memcpy(&tmp[0], &True0_B.VectorConcatenate[0], 9U * sizeof(real_T));
  phi = True0_B.ForceofgravityInertialaxes[0];
  ctheta = True0_B.ForceofgravityInertialaxes[1];
  theta = True0_B.ForceofgravityInertialaxes[2];
  for (i = 0; i < 3; i++) {
    cphi = tmp[i] * phi;
    cphi += tmp[i + 3] * ctheta;
    cphi += tmp[i + 6] * theta;

    /* Product: '<S36>/Product' */
    True0_B.ForceofgravityBodyaxes[i] = cphi;

    /* Sum: '<S3>/Sum' incorporates:
     *  Product: '<S36>/Product'
     *  Sum: '<S37>/Sum of Elements'
     */
    True0_B.Sum[i] = True0_B.SumofElements[i] + cphi;
  }

  /* End of Product: '<S36>/Product' */

  /* Product: '<S54>/Product' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   *  Inport: '<Root>/Wind_i'
   */
  std::memcpy(&tmp[0], &True0_B.VectorConcatenate[0], 9U * sizeof(real_T));
  phi = True0_U.Wind_i[0];
  ctheta = True0_U.Wind_i[1];
  theta = True0_U.Wind_i[2];
  for (i = 0; i < 3; i++) {
    /* Product: '<S54>/Product' */
    cphi = tmp[i] * phi;
    cphi += tmp[i + 3] * ctheta;
    cphi += tmp[i + 6] * theta;
    True0_B.Product_n[i] = cphi;

    /* Sum: '<S39>/Sum1' incorporates:
     *  Integrator: '<S2>/V_b'
     *  Product: '<S54>/Product'
     */
    cphi = True0_B.V_b[i] - cphi;
    True0_B.TrueairspeedBodyaxes_m[i] = cphi;

    /* Math: '<S53>/transpose' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    True0_B.transpose_i[i] = cphi;
  }

  /* Product: '<S53>/Product' incorporates:
   *  Math: '<S53>/transpose'
   *  Sum: '<S39>/Sum1'
   */
  Product_pk_tmp = True0_B.transpose_i[0];
  phi = True0_B.TrueairspeedBodyaxes_m[0];
  cphi = Product_pk_tmp * phi;
  Product_pk_tmp = True0_B.transpose_i[1];
  phi = True0_B.TrueairspeedBodyaxes_m[1];
  cphi += Product_pk_tmp * phi;
  Product_pk_tmp = True0_B.transpose_i[2];
  phi = True0_B.TrueairspeedBodyaxes_m[2];
  cphi += Product_pk_tmp * phi;

  /* Product: '<S53>/Product' */
  True0_B.Product_f1 = cphi;

  /* Sqrt: '<S42>/Sqrt' */
  True0_B.Sqrt_o = std::sqrt(True0_B.Product_f1);

  /* ComplexToRealImag: '<S42>/Complex to Real-Imag' */
  True0_B.ComplextoRealImag_b = True0_B.Sqrt_o;

  /* If: '<S38>/If' */
  if (rtsiIsModeUpdateTimeStep(&(&True0_M)->solverInfo)) {
    rtAction = static_cast<int8_T>(!(True0_B.ComplextoRealImag_b == 0.0));
    True0_DW.If_ActiveSubsystem = rtAction;
  } else {
    rtAction = True0_DW.If_ActiveSubsystem;
  }

  if (rtAction == 0) {
    /* Outputs for IfAction SubSystem: '<S38>/Zero airspeed' incorporates:
     *  ActionPort: '<S41>/Action Port'
     */
    if (rtmIsMajorTimeStep((&True0_M))) {
      /* Merge: '<S38>/Merge' incorporates:
       *  Constant: '<S41>/Constant'
       */
      True0_B.Forceagainstdirectionofmotiondu[0] = True0_P.Constant_Value[0];
      True0_B.Forceagainstdirectionofmotiondu[1] = True0_P.Constant_Value[1];
      True0_B.Forceagainstdirectionofmotiondu[2] = True0_P.Constant_Value[2];
    }

    /* End of Outputs for SubSystem: '<S38>/Zero airspeed' */
  } else {
    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Math: '<S52>/transpose' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[0];

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */

    /* Math: '<S52>/transpose' */
    True0_B.transpose_j[0] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S52>/Product' incorporates:
     *  Product: '<S47>/Product'
     */
    Product_pk_tmp = cphi * cphi;

    /* Math: '<S52>/transpose' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[1];

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */

    /* Math: '<S52>/transpose' */
    True0_B.transpose_j[1] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S52>/Product' incorporates:
     *  Product: '<S47>/Product'
     */
    Product_pk_tmp += cphi * cphi;

    /* Math: '<S52>/transpose' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[2];

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */

    /* Math: '<S52>/transpose' */
    True0_B.transpose_j[2] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S52>/Product' incorporates:
     *  Product: '<S47>/Product'
     */
    Product_pk_tmp += cphi * cphi;

    /* Product: '<S52>/Product' */
    True0_B.Product_m = Product_pk_tmp;

    /* Product: '<S48>/Divide' incorporates:
     *  Inport: '<Root>/Surface_params'
     */
    True0_B.Divide_i = True0_B.TrueairspeedBodyaxes_m[0] /
      True0_U.Surface_params[0];

    /* Product: '<S48>/Product' */
    True0_B.Product_fu = True0_B.Divide_i * True0_B.Divide_i;

    /* Product: '<S48>/Divide1' incorporates:
     *  Inport: '<Root>/Surface_params'
     */
    True0_B.Divide1 = True0_B.TrueairspeedBodyaxes_m[1] /
      True0_U.Surface_params[1];

    /* Product: '<S48>/Product1' */
    True0_B.Product1_f = True0_B.Divide1 * True0_B.Divide1;

    /* Product: '<S48>/Divide2' incorporates:
     *  Inport: '<Root>/Surface_params'
     */
    True0_B.Divide2 = True0_B.TrueairspeedBodyaxes_m[2] /
      True0_U.Surface_params[2];

    /* Product: '<S48>/Product2' */
    True0_B.Product2_j = True0_B.Divide2 * True0_B.Divide2;

    /* Sum: '<S48>/Add' */
    True0_B.Add_f = (True0_B.Product_fu + True0_B.Product1_f) +
      True0_B.Product2_j;

    /* Sqrt: '<S48>/Reciprocal Sqrt' */
    cphi = True0_B.Add_f;
    if (cphi > 0.0) {
      if (std::isinf(cphi)) {
        /* Sqrt: '<S48>/Reciprocal Sqrt' */
        True0_B.ReciprocalSqrt = 0.0;
      } else {
        /* Sqrt: '<S48>/Reciprocal Sqrt' */
        True0_B.ReciprocalSqrt = 1.0 / std::sqrt(cphi);
      }
    } else if (cphi == 0.0) {
      /* Sqrt: '<S48>/Reciprocal Sqrt' */
      True0_B.ReciprocalSqrt = (rtInf);
    } else {
      /* Sqrt: '<S48>/Reciprocal Sqrt' */
      True0_B.ReciprocalSqrt = (rtNaN);
    }

    /* End of Sqrt: '<S48>/Reciprocal Sqrt' */

    /* Product: '<S49>/Product' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[0];

    /* Product: '<S49>/Product' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    ctheta = cphi * True0_B.ReciprocalSqrt;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Product_fb[0] = ctheta;

    /* Math: '<S51>/transpose' */
    True0_B.transpose_e[0] = ctheta;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S51>/Product' */
    phi = ctheta * ctheta;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */

    /* Math: '<S47>/transpose' */
    True0_B.transpose_h[0] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S49>/Product' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[1];

    /* Product: '<S49>/Product' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    ctheta = cphi * True0_B.ReciprocalSqrt;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Product_fb[1] = ctheta;

    /* Math: '<S51>/transpose' */
    True0_B.transpose_e[1] = ctheta;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S51>/Product' */
    phi += ctheta * ctheta;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */

    /* Math: '<S47>/transpose' */
    True0_B.transpose_h[1] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S49>/Product' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[2];

    /* Product: '<S49>/Product' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    ctheta = cphi * True0_B.ReciprocalSqrt;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Product_fb[2] = ctheta;

    /* Math: '<S51>/transpose' */
    True0_B.transpose_e[2] = ctheta;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S51>/Product' */
    phi += ctheta * ctheta;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */

    /* Math: '<S47>/transpose' */
    True0_B.transpose_h[2] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S51>/Product' */
    True0_B.Product_nx = phi;

    /* Sqrt: '<S50>/Sqrt' */
    True0_B.Sqrt_a = std::sqrt(True0_B.Product_nx);

    /* ComplexToRealImag: '<S50>/Complex to Real-Imag' */
    True0_B.ComplextoRealImag_i = True0_B.Sqrt_a;

    /* Product: '<S40>/Product' incorporates:
     *  Constant: '<S40>/Constant'
     *  Constant: '<S40>/Constant1'
     *  Constant: '<S40>/Constant2'
     */
    True0_B.Product_d = True0_P.Constant_Value_e * True0_P.rho *
      True0_B.Product_m * True0_P.C_D * True0_B.ComplextoRealImag_i;

    /* Abs: '<S40>/Abs' */
    True0_B.Magnitudeofdragforce = std::abs(True0_B.Product_d);

    /* Product: '<S47>/Product' */
    True0_B.Product_dc = Product_pk_tmp;

    /* Sqrt: '<S46>/Sqrt' */
    True0_B.Sqrt_oc = std::sqrt(True0_B.Product_dc);

    /* ComplexToRealImag: '<S46>/Complex to Real-Imag' */
    True0_B.ComplextoRealImag_m = True0_B.Sqrt_oc;

    /* Product: '<S43>/Divide' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[0] / True0_B.ComplextoRealImag_m;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Divide_n[0] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S40>/Product1' incorporates:
     *  Product: '<S43>/Divide'
     */
    cphi *= True0_B.Magnitudeofdragforce;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Product1_m[0] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Merge: '<S38>/Merge' incorporates:
     *  Gain: '<S40>/Drag force opposes direction of airspeed'
     *  Product: '<S40>/Product1'
     */
    True0_B.Forceagainstdirectionofmotiondu[0] =
      True0_P.Dragforceopposesdirectionofairs * cphi;

    /* Product: '<S43>/Divide' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[1] / True0_B.ComplextoRealImag_m;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Divide_n[1] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S40>/Product1' incorporates:
     *  Product: '<S43>/Divide'
     */
    cphi *= True0_B.Magnitudeofdragforce;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Product1_m[1] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Merge: '<S38>/Merge' incorporates:
     *  Gain: '<S40>/Drag force opposes direction of airspeed'
     *  Product: '<S40>/Product1'
     */
    True0_B.Forceagainstdirectionofmotiondu[1] =
      True0_P.Dragforceopposesdirectionofairs * cphi;

    /* Product: '<S43>/Divide' incorporates:
     *  Sum: '<S39>/Sum1'
     */
    cphi = True0_B.TrueairspeedBodyaxes_m[2] / True0_B.ComplextoRealImag_m;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Divide_n[2] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Product: '<S40>/Product1' incorporates:
     *  Product: '<S43>/Divide'
     */
    cphi *= True0_B.Magnitudeofdragforce;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
    True0_B.Product1_m[2] = cphi;

    /* Outputs for IfAction SubSystem: '<S38>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S40>/Action Port'
     */
    /* Merge: '<S38>/Merge' incorporates:
     *  Gain: '<S40>/Drag force opposes direction of airspeed'
     *  Product: '<S40>/Product1'
     */
    True0_B.Forceagainstdirectionofmotiondu[2] =
      True0_P.Dragforceopposesdirectionofairs * cphi;

    /* End of Outputs for SubSystem: '<S38>/Nonzero airspeed' */
  }

  /* End of If: '<S38>/If' */

  /* Product: '<S33>/Product' */
  True0_B.u2v3 = True0_B.omega[1] * True0_B.V_b[2];

  /* Product: '<S33>/Product1' */
  True0_B.u3v1 = True0_B.V_b[0] * True0_B.omega[2];

  /* Product: '<S33>/Product2' */
  True0_B.u1v2 = True0_B.omega[0] * True0_B.V_b[1];

  /* Product: '<S34>/Product' */
  True0_B.u3v2 = True0_B.V_b[1] * True0_B.omega[2];

  /* Product: '<S34>/Product1' */
  True0_B.u1v3 = True0_B.omega[0] * True0_B.V_b[2];

  /* Product: '<S34>/Product2' */
  True0_B.u2v1 = True0_B.V_b[0] * True0_B.omega[1];

  /* Sum: '<S10>/Sum' */
  True0_B.Sum_c[0] = True0_B.u2v3 - True0_B.u3v2;
  True0_B.Sum_c[1] = True0_B.u3v1 - True0_B.u1v3;
  True0_B.Sum_c[2] = True0_B.u1v2 - True0_B.u2v1;

  /* Sum: '<S3>/Sum3' incorporates:
   *  Merge: '<S38>/Merge'
   *  Sum: '<S3>/Sum'
   */
  cphi = True0_B.Sum[0] + True0_B.Forceagainstdirectionofmotiondu[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum3[0] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S3>/Sum1' incorporates:
   *  Inport: '<Root>/Force_disturb'
   *  Sum: '<S3>/Sum3'
   */
  cphi += True0_U.Force_disturb[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum1[0] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S2>/Product1' incorporates:
   *  Inport: '<Root>/mass_real'
   *  Sum: '<S3>/Sum1'
   */
  cphi /= True0_U.mass_real;

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Product1[0] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S2>/Sum1' incorporates:
   *  Product: '<S2>/Product1'
   */
  True0_B.Sum1_o[0] = cphi - True0_B.Sum_c[0];

  /* Sum: '<S3>/Sum3' incorporates:
   *  Merge: '<S38>/Merge'
   *  Sum: '<S3>/Sum'
   */
  cphi = True0_B.Sum[1] + True0_B.Forceagainstdirectionofmotiondu[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum3[1] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S3>/Sum1' incorporates:
   *  Inport: '<Root>/Force_disturb'
   *  Sum: '<S3>/Sum3'
   */
  cphi += True0_U.Force_disturb[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum1[1] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S2>/Product1' incorporates:
   *  Inport: '<Root>/mass_real'
   *  Sum: '<S3>/Sum1'
   */
  cphi /= True0_U.mass_real;

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Product1[1] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S2>/Sum1' incorporates:
   *  Product: '<S2>/Product1'
   */
  True0_B.Sum1_o[1] = cphi - True0_B.Sum_c[1];

  /* Sum: '<S3>/Sum3' incorporates:
   *  Merge: '<S38>/Merge'
   *  Sum: '<S3>/Sum'
   */
  cphi = True0_B.Sum[2] + True0_B.Forceagainstdirectionofmotiondu[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum3[2] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S3>/Sum1' incorporates:
   *  Inport: '<Root>/Force_disturb'
   *  Sum: '<S3>/Sum3'
   */
  cphi += True0_U.Force_disturb[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum1[2] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Product: '<S2>/Product1' incorporates:
   *  Inport: '<Root>/mass_real'
   *  Sum: '<S3>/Sum1'
   */
  cphi /= True0_U.mass_real;

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Product1[2] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S2>/Sum1' incorporates:
   *  Product: '<S2>/Product1'
   */
  True0_B.Sum1_o[2] = cphi - True0_B.Sum_c[2];

  /* Product: '<S5>/Product' incorporates:
   *  Math: '<S5>/Math Function2'
   *  Sum: '<S2>/Sum1'
   */
  std::memcpy(&tmp[0], &True0_B.DCM_bi_c[0], 9U * sizeof(real_T));
  phi = True0_B.Sum1_o[0];
  ctheta = True0_B.Sum1_o[1];
  theta = True0_B.Sum1_o[2];
  for (i = 0; i < 3; i++) {
    /* Product: '<S5>/Product' */
    cphi = tmp[i] * phi;
    cphi += tmp[i + 3] * ctheta;
    cphi += tmp[i + 6] * theta;
    True0_B.Product_e[i] = cphi;
  }

  /* Product: '<S12>/Product' incorporates:
   *  Inport: '<Root>/J_real'
   *  Integrator: '<S2>/omega'
   */
  std::memcpy(&tmp[0], &True0_U.J_real[0], 9U * sizeof(real_T));
  phi = True0_B.omega[0];
  ctheta = True0_B.omega[1];
  theta = True0_B.omega[2];
  for (i = 0; i < 3; i++) {
    /* Product: '<S12>/Product' */
    cphi = tmp[i] * phi;
    cphi += tmp[i + 3] * ctheta;
    cphi += tmp[i + 6] * theta;
    True0_B.Product_na[i] = cphi;

    /* Sum: '<S37>/Sum of Elements1' incorporates:
     *  ForEachSliceAssignment generated from: '<S55>/Motor_moment'
     */
    tmp_0 = i;
    cphi = True0_B.ImpAsg_InsertedFor_Motor_moment[tmp_0];
    tmp_0 = i + 3;
    cphi += True0_B.ImpAsg_InsertedFor_Motor_moment[tmp_0];
    tmp_0 = i + 6;
    cphi += True0_B.ImpAsg_InsertedFor_Motor_moment[tmp_0];
    tmp_0 = i + 9;
    cphi += True0_B.ImpAsg_InsertedFor_Motor_moment[tmp_0];
    True0_B.SumofElements1[i] = cphi;
  }

  /* Product: '<S13>/Product' */
  True0_B.u2v3_m = True0_B.omega[1] * True0_B.Product_na[2];

  /* Product: '<S13>/Product1' */
  True0_B.u3v1_m = True0_B.Product_na[0] * True0_B.omega[2];

  /* Product: '<S13>/Product2' */
  True0_B.u1v2_h = True0_B.omega[0] * True0_B.Product_na[1];

  /* Product: '<S14>/Product' */
  True0_B.u3v2_b = True0_B.Product_na[1] * True0_B.omega[2];

  /* Product: '<S14>/Product1' */
  True0_B.u1v3_k = True0_B.omega[0] * True0_B.Product_na[2];

  /* Product: '<S14>/Product2' */
  True0_B.u2v1_m = True0_B.Product_na[0] * True0_B.omega[1];

  /* Sum: '<S11>/Sum' */
  True0_B.Sum_k[0] = True0_B.u2v3_m - True0_B.u3v2_b;
  True0_B.Sum_k[1] = True0_B.u3v1_m - True0_B.u1v3_k;
  True0_B.Sum_k[2] = True0_B.u1v2_h - True0_B.u2v1_m;

  /* Sum: '<S3>/Sum2' incorporates:
   *  Inport: '<Root>/Moment_disturb'
   *  Sum: '<S37>/Sum of Elements1'
   */
  cphi = True0_B.SumofElements1[0] + True0_U.Moment_disturb[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum2[0] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S6>/Sum1' incorporates:
   *  Sum: '<S11>/Sum'
   *  Sum: '<S3>/Sum2'
   */
  True0_B.Sum1_n[0] = cphi - True0_B.Sum_k[0];

  /* Math: '<S16>/transpose' incorporates:
   *  Integrator: '<S2>/omega'
   */
  True0_B.transpose_g[0] = True0_B.omega[0];

  /* Sum: '<S3>/Sum2' incorporates:
   *  Inport: '<Root>/Moment_disturb'
   *  Sum: '<S37>/Sum of Elements1'
   */
  cphi = True0_B.SumofElements1[1] + True0_U.Moment_disturb[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum2[1] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S6>/Sum1' incorporates:
   *  Sum: '<S11>/Sum'
   *  Sum: '<S3>/Sum2'
   */
  True0_B.Sum1_n[1] = cphi - True0_B.Sum_k[1];

  /* Math: '<S16>/transpose' incorporates:
   *  Integrator: '<S2>/omega'
   */
  True0_B.transpose_g[1] = True0_B.omega[1];

  /* Sum: '<S3>/Sum2' incorporates:
   *  Inport: '<Root>/Moment_disturb'
   *  Sum: '<S37>/Sum of Elements1'
   */
  cphi = True0_B.SumofElements1[2] + True0_U.Moment_disturb[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Sum2[2] = cphi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S6>/Sum1' incorporates:
   *  Sum: '<S11>/Sum'
   *  Sum: '<S3>/Sum2'
   */
  True0_B.Sum1_n[2] = cphi - True0_B.Sum_k[2];

  /* Math: '<S16>/transpose' incorporates:
   *  Integrator: '<S2>/omega'
   */
  True0_B.transpose_g[2] = True0_B.omega[2];

  /* Product: '<S6>/Product' incorporates:
   *  Inport: '<Root>/J_real'
   *  Sum: '<S6>/Sum1'
   */
  rt_mldivide_U1d3x3_U2d_JBYZyA3A(True0_U.J_real, True0_B.Sum1_n,
    True0_B.Product_lc);

  /* Product: '<S19>/Product' */
  True0_B.u2v3_j = True0_B.omega[1] * True0_B.Divide[3];

  /* Product: '<S19>/Product1' */
  True0_B.u3v1_h = True0_B.Divide[1] * True0_B.omega[2];

  /* Product: '<S19>/Product2' */
  True0_B.u1v2_i = True0_B.omega[0] * True0_B.Divide[2];

  /* Product: '<S20>/Product' */
  True0_B.u3v2_d = True0_B.omega[2] * True0_B.Divide[2];

  /* Product: '<S20>/Product1' */
  True0_B.u1v3_p = True0_B.omega[0] * True0_B.Divide[3];

  /* Product: '<S20>/Product2' */
  True0_B.u2v1_b = True0_B.omega[1] * True0_B.Divide[1];

  /* Sum: '<S15>/Sum' */
  True0_B.Sum_o[0] = True0_B.u2v3_j - True0_B.u3v2_d;
  True0_B.Sum_o[1] = True0_B.u3v1_h - True0_B.u1v3_p;
  True0_B.Sum_o[2] = True0_B.u1v2_i - True0_B.u2v1_b;

  /* Product: '<S16>/Product' incorporates:
   *  Math: '<S16>/transpose'
   */
  Product_pk_tmp = True0_B.transpose_g[0];
  phi = True0_B.Divide[1];
  cphi = Product_pk_tmp * phi;

  /* Product: '<S7>/Product' incorporates:
   *  Integrator: '<S2>/omega'
   */
  phi = True0_B.Divide[0] * True0_B.omega[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Product_i[0] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S7>/Subtract' incorporates:
   *  Product: '<S7>/Product'
   */
  phi -= True0_B.Sum_o[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Subtract[0] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Gain: '<S7>/1//2' incorporates:
   *  Sum: '<S7>/Subtract'
   */
  phi *= True0_P.u2_Gain_c;

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.u2_d[0] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* SignalConversion generated from: '<S7>/Q-Integrator' */
  True0_B.TmpSignalConversionAtQIntegrato[1] = phi;

  /* Product: '<S16>/Product' incorporates:
   *  Math: '<S16>/transpose'
   */
  Product_pk_tmp = True0_B.transpose_g[1];
  phi = True0_B.Divide[2];
  cphi += Product_pk_tmp * phi;

  /* Product: '<S7>/Product' incorporates:
   *  Integrator: '<S2>/omega'
   */
  phi = True0_B.Divide[0] * True0_B.omega[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Product_i[1] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S7>/Subtract' incorporates:
   *  Product: '<S7>/Product'
   */
  phi -= True0_B.Sum_o[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Subtract[1] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Gain: '<S7>/1//2' incorporates:
   *  Sum: '<S7>/Subtract'
   */
  phi *= True0_P.u2_Gain_c;

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.u2_d[1] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* SignalConversion generated from: '<S7>/Q-Integrator' */
  True0_B.TmpSignalConversionAtQIntegrato[2] = phi;

  /* Product: '<S16>/Product' incorporates:
   *  Math: '<S16>/transpose'
   */
  Product_pk_tmp = True0_B.transpose_g[2];
  phi = True0_B.Divide[3];
  cphi += Product_pk_tmp * phi;

  /* Product: '<S7>/Product' incorporates:
   *  Integrator: '<S2>/omega'
   */
  phi = True0_B.Divide[0] * True0_B.omega[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Product_i[2] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Sum: '<S7>/Subtract' incorporates:
   *  Product: '<S7>/Product'
   */
  phi -= True0_B.Sum_o[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.Subtract[2] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Gain: '<S7>/1//2' incorporates:
   *  Sum: '<S7>/Subtract'
   */
  phi *= True0_P.u2_Gain_c;

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.u2_d[2] = phi;

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* SignalConversion generated from: '<S7>/Q-Integrator' */
  True0_B.TmpSignalConversionAtQIntegrato[3] = phi;

  /* Product: '<S16>/Product' */
  True0_B.Product_py = cphi;

  /* Gain: '<S7>/-1//2' */
  True0_B.u2 = True0_P.u2_Gain * True0_B.Product_py;

  /* SignalConversion generated from: '<S7>/Q-Integrator' */
  True0_B.TmpSignalConversionAtQIntegrato[0] = True0_B.u2;

  /* Fcn: '<S8>/Fcn' */
  True0_B.Fcn = psi_tmp_0 * 2.0;

  /* Fcn: '<S8>/Fcn1' */
  True0_B.Fcn1 = psi_tmp * 2.0;

  /* Fcn: '<S8>/Fcn2' */
  True0_B.Fcn2 = cpsi * 2.0;

  /* Fcn: '<S8>/Fcn3' */
  True0_B.Fcn3 = spsi;

  /* Fcn: '<S8>/Fcn4' */
  True0_B.Fcn4 = psi;

  /* Trigonometry: '<S8>/Trigonometric Function' */
  psi = True0_B.Fcn1;
  if (psi > 1.0) {
    psi = 1.0;
  } else if (psi < -1.0) {
    psi = -1.0;
  }

  /* Trigonometry: '<S8>/Trigonometric Function' */
  True0_B.TrigonometricFunction = std::asin(psi);

  /* Gain: '<S8>/Gain' incorporates:
   *  Concatenate: '<S8>/Vector Concatenate'
   */
  True0_B.VectorConcatenate_h[1] = True0_P.Gain_Gain_d *
    True0_B.TrigonometricFunction;

  /* Trigonometry: '<S8>/Trigonometric Function1' incorporates:
   *  Concatenate: '<S8>/Vector Concatenate'
   */
  True0_B.VectorConcatenate_h[0] = rt_atan2d_snf(True0_B.Fcn, True0_B.Fcn3);

  /* Trigonometry: '<S8>/Trigonometric Function2' incorporates:
   *  Concatenate: '<S8>/Vector Concatenate'
   */
  True0_B.VectorConcatenate_h[2] = rt_atan2d_snf(True0_B.Fcn2, True0_B.Fcn4);

  /* Integrator: '<S2>/X_i' */
  psi = True0_X.X_i_CSTATE[0];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.X_i[0] = psi;

  /* Outport: '<Root>/X_i' incorporates:
   *  Integrator: '<S2>/X_i'
   */
  True0_Y.X_i[0] = psi;

  /* Outport: '<Root>/V_i' incorporates:
   *  Product: '<S4>/Product'
   */
  True0_Y.V_i[0] = True0_B.Product_b[0];

  /* Outport: '<Root>/V_b' incorporates:
   *  Integrator: '<S2>/V_b'
   */
  True0_Y.V_b[0] = True0_B.V_b[0];

  /* Outport: '<Root>/a_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  True0_Y.a_b[0] = True0_B.Sum1_o[0];

  /* Outport: '<Root>/a_i' incorporates:
   *  Product: '<S5>/Product'
   */
  True0_Y.a_i[0] = True0_B.Product_e[0];

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Integrator: '<S2>/X_i' */
  psi = True0_X.X_i_CSTATE[1];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.X_i[1] = psi;

  /* Outport: '<Root>/X_i' incorporates:
   *  Integrator: '<S2>/X_i'
   */
  True0_Y.X_i[1] = psi;

  /* Outport: '<Root>/V_i' incorporates:
   *  Product: '<S4>/Product'
   */
  True0_Y.V_i[1] = True0_B.Product_b[1];

  /* Outport: '<Root>/V_b' incorporates:
   *  Integrator: '<S2>/V_b'
   */
  True0_Y.V_b[1] = True0_B.V_b[1];

  /* Outport: '<Root>/a_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  True0_Y.a_b[1] = True0_B.Sum1_o[1];

  /* Outport: '<Root>/a_i' incorporates:
   *  Product: '<S5>/Product'
   */
  True0_Y.a_i[1] = True0_B.Product_e[1];

  /* Outputs for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Integrator: '<S2>/X_i' */
  psi = True0_X.X_i_CSTATE[2];

  /* End of Outputs for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  True0_B.X_i[2] = psi;

  /* Outport: '<Root>/X_i' incorporates:
   *  Integrator: '<S2>/X_i'
   */
  True0_Y.X_i[2] = psi;

  /* Outport: '<Root>/V_i' incorporates:
   *  Product: '<S4>/Product'
   */
  True0_Y.V_i[2] = True0_B.Product_b[2];

  /* Outport: '<Root>/V_b' incorporates:
   *  Integrator: '<S2>/V_b'
   */
  True0_Y.V_b[2] = True0_B.V_b[2];

  /* Outport: '<Root>/a_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  True0_Y.a_b[2] = True0_B.Sum1_o[2];

  /* Outport: '<Root>/a_i' incorporates:
   *  Product: '<S5>/Product'
   */
  True0_Y.a_i[2] = True0_B.Product_e[2];

  /* Outport: '<Root>/DCM_ib' incorporates:
   *  Concatenate: '<S32>/Vector Concatenate'
   */
  std::memcpy(&True0_Y.DCM_ib[0], &True0_B.VectorConcatenate[0], 9U * sizeof
              (real_T));

  /* Outport: '<Root>/Quat q' incorporates:
   *  Product: '<S18>/Divide'
   */
  True0_Y.Quatq[0] = True0_B.Divide[0];
  True0_Y.Quatq[1] = True0_B.Divide[1];
  True0_Y.Quatq[2] = True0_B.Divide[2];
  True0_Y.Quatq[3] = True0_B.Divide[3];

  /* Outport: '<Root>/Euler' incorporates:
   *  Concatenate: '<S8>/Vector Concatenate'
   */
  True0_Y.Euler[0] = True0_B.VectorConcatenate_h[0];

  /* Outport: '<Root>/omega' incorporates:
   *  Integrator: '<S2>/omega'
   */
  True0_Y.omega[0] = True0_B.omega[0];

  /* Outport: '<Root>/omega_dot' incorporates:
   *  Product: '<S6>/Product'
   */
  True0_Y.omega_dot[0] = True0_B.Product_lc[0];

  /* Outport: '<Root>/Euler' incorporates:
   *  Concatenate: '<S8>/Vector Concatenate'
   */
  True0_Y.Euler[1] = True0_B.VectorConcatenate_h[1];

  /* Outport: '<Root>/omega' incorporates:
   *  Integrator: '<S2>/omega'
   */
  True0_Y.omega[1] = True0_B.omega[1];

  /* Outport: '<Root>/omega_dot' incorporates:
   *  Product: '<S6>/Product'
   */
  True0_Y.omega_dot[1] = True0_B.Product_lc[1];

  /* Outport: '<Root>/Euler' incorporates:
   *  Concatenate: '<S8>/Vector Concatenate'
   */
  True0_Y.Euler[2] = True0_B.VectorConcatenate_h[2];

  /* Outport: '<Root>/omega' incorporates:
   *  Integrator: '<S2>/omega'
   */
  True0_Y.omega[2] = True0_B.omega[2];

  /* Outport: '<Root>/omega_dot' incorporates:
   *  Product: '<S6>/Product'
   */
  True0_Y.omega_dot[2] = True0_B.Product_lc[2];
  if (rtmIsMajorTimeStep((&True0_M))) {
    /* Matfile logging */
    rt_UpdateTXYLogVars((&True0_M)->rtwLogInfo, ((&True0_M)->Timing.t));
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep((&True0_M))) {
    /* Update for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
    /* Update for Integrator: '<S7>/Q-Integrator' */
    True0_DW.QIntegrator_IWORK = 0;

    /* End of Update for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep((&True0_M))) {
    /* signal main to stop simulation */
    {                                  /* Sample time: [0.0s, 0.0s] */
      if ((rtmGetTFinal((&True0_M))!=-1) &&
          !((rtmGetTFinal((&True0_M))-((((&True0_M)->Timing.clockTick1+(&True0_M)
               ->Timing.clockTickH1* 4294967296.0)) * 0.001)) > ((((&True0_M)
              ->Timing.clockTick1+(&True0_M)->Timing.clockTickH1* 4294967296.0))
            * 0.001) * (DBL_EPSILON))) {
        rtmSetErrorStatus((&True0_M), "Simulation finished");
      }
    }

    rt_ertODEUpdateContinuousStates(&(&True0_M)->solverInfo);

    /* Update absolute time */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick0 and the high bits
     * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++(&True0_M)->Timing.clockTick0)) {
      ++(&True0_M)->Timing.clockTickH0;
    }

    (&True0_M)->Timing.t[0] = rtsiGetSolverStopTime(&(&True0_M)->solverInfo);

    /* Update absolute time */
    /* The "clockTick1" counts the number of times the code of this task has
     * been executed. The resolution of this integer timer is 0.001, which is the step size
     * of the task. Size of "clockTick1" ensures timer will not overflow during the
     * application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick1 and the high bits
     * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
     */
    (&True0_M)->Timing.clockTick1++;
    if (!(&True0_M)->Timing.clockTick1) {
      (&True0_M)->Timing.clockTickH1++;
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void True0::True0_derivatives()
{
  /* local scratch DWork variables */
  int32_T ForEach_itr;
  XDot_True0_T *_rtXdot;
  _rtXdot = ((XDot_True0_T *) (&True0_M)->derivs);

  /* Derivatives for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Derivatives for Integrator: '<S7>/Q-Integrator' incorporates:
   *  SignalConversion generated from: '<S7>/Q-Integrator'
   */
  _rtXdot->QIntegrator_CSTATE[0] = True0_B.TmpSignalConversionAtQIntegrato[0];
  _rtXdot->QIntegrator_CSTATE[1] = True0_B.TmpSignalConversionAtQIntegrato[1];
  _rtXdot->QIntegrator_CSTATE[2] = True0_B.TmpSignalConversionAtQIntegrato[2];
  _rtXdot->QIntegrator_CSTATE[3] = True0_B.TmpSignalConversionAtQIntegrato[3];

  /* Derivatives for Integrator: '<S2>/V_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  _rtXdot->V_b_CSTATE[0] = True0_B.Sum1_o[0];

  /* Derivatives for Integrator: '<S2>/omega' incorporates:
   *  Product: '<S6>/Product'
   */
  _rtXdot->omega_CSTATE[0] = True0_B.Product_lc[0];

  /* Derivatives for Integrator: '<S2>/V_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  _rtXdot->V_b_CSTATE[1] = True0_B.Sum1_o[1];

  /* Derivatives for Integrator: '<S2>/omega' incorporates:
   *  Product: '<S6>/Product'
   */
  _rtXdot->omega_CSTATE[1] = True0_B.Product_lc[1];

  /* Derivatives for Integrator: '<S2>/V_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  _rtXdot->V_b_CSTATE[2] = True0_B.Sum1_o[2];

  /* Derivatives for Integrator: '<S2>/omega' incorporates:
   *  Product: '<S6>/Product'
   */
  _rtXdot->omega_CSTATE[2] = True0_B.Product_lc[2];

  /* Derivatives for Iterator SubSystem: '<S37>/For Each Subsystem' */
  for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
    /* Derivatives for Integrator: '<S61>/Integrator' */
    _rtXdot->CoreSubsys[ForEach_itr].Integrator_CSTATE =
      True0_B.CoreSubsys[ForEach_itr].Switch;
  }

  /* End of Derivatives for SubSystem: '<S37>/For Each Subsystem' */

  /* Derivatives for Integrator: '<S2>/X_i' incorporates:
   *  Product: '<S4>/Product'
   */
  _rtXdot->X_i_CSTATE[0] = True0_B.Product_b[0];
  _rtXdot->X_i_CSTATE[1] = True0_B.Product_b[1];
  _rtXdot->X_i_CSTATE[2] = True0_B.Product_b[2];

  /* End of Derivatives for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
}

/* Model step function for TID2 */
void True0::step2()                    /* Sample time: [0.002s, 0.0s] */
{
  /* Update for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  /* Update for RateTransition: '<S1>/Rate Transition1' incorporates:
   *  Inport: '<Root>/RPM commands'
   */
  True0_DW.RateTransition1_Buffer0[0] = True0_U.RPMcommands[0];
  True0_DW.RateTransition1_Buffer0[1] = True0_U.RPMcommands[1];
  True0_DW.RateTransition1_Buffer0[2] = True0_U.RPMcommands[2];
  True0_DW.RateTransition1_Buffer0[3] = True0_U.RPMcommands[3];

  /* End of Update for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
}

/* Model initialize function */
void True0::initialize()
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  /* Set task counter limit used by the static main program */
  ((&True0_M))->Timing.TaskCounters.cLimit[0] = 1;
  ((&True0_M))->Timing.TaskCounters.cLimit[1] = 1;
  ((&True0_M))->Timing.TaskCounters.cLimit[2] = 2;

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&(&True0_M)->solverInfo, &(&True0_M)
                          ->Timing.simTimeStep);
    rtsiSetTPtr(&(&True0_M)->solverInfo, &rtmGetTPtr((&True0_M)));
    rtsiSetStepSizePtr(&(&True0_M)->solverInfo, &(&True0_M)->Timing.stepSize0);
    rtsiSetdXPtr(&(&True0_M)->solverInfo, &(&True0_M)->derivs);
    rtsiSetContStatesPtr(&(&True0_M)->solverInfo, (real_T **) &(&True0_M)
                         ->contStates);
    rtsiSetNumContStatesPtr(&(&True0_M)->solverInfo, &(&True0_M)
      ->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&(&True0_M)->solverInfo, &(&True0_M)
      ->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&(&True0_M)->solverInfo, &(&True0_M)
      ->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&(&True0_M)->solverInfo, &(&True0_M)
      ->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&(&True0_M)->solverInfo, (&rtmGetErrorStatus((&True0_M))));
    rtsiSetRTModelPtr(&(&True0_M)->solverInfo, (&True0_M));
  }

  rtsiSetSimTimeStep(&(&True0_M)->solverInfo, MAJOR_TIME_STEP);
  (&True0_M)->intgData.y = (&True0_M)->odeY;
  (&True0_M)->intgData.f[0] = (&True0_M)->odeF[0];
  (&True0_M)->intgData.f[1] = (&True0_M)->odeF[1];
  (&True0_M)->intgData.f[2] = (&True0_M)->odeF[2];
  (&True0_M)->contStates = ((X_True0_T *) &True0_X);
  rtsiSetSolverData(&(&True0_M)->solverInfo, static_cast<void *>(&(&True0_M)
    ->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&(&True0_M)->solverInfo, false);
  rtsiSetSolverName(&(&True0_M)->solverInfo,"ode3");
  rtmSetTPtr((&True0_M), &(&True0_M)->Timing.tArray[0]);
  rtmSetTFinal((&True0_M), 10.0);
  (&True0_M)->Timing.stepSize0 = 0.001;
  rtmSetFirstInitCond((&True0_M), 1);

  /* Setup for data logging */
  {
    static RTWLogInfo rt_DataLoggingInfo;
    rt_DataLoggingInfo.loggingInterval = (nullptr);
    (&True0_M)->rtwLogInfo = &rt_DataLoggingInfo;
  }

  /* Setup for data logging */
  {
    rtliSetLogXSignalInfo((&True0_M)->rtwLogInfo, (nullptr));
    rtliSetLogXSignalPtrs((&True0_M)->rtwLogInfo, (nullptr));
    rtliSetLogT((&True0_M)->rtwLogInfo, "tout");
    rtliSetLogX((&True0_M)->rtwLogInfo, "");
    rtliSetLogXFinal((&True0_M)->rtwLogInfo, "");
    rtliSetLogVarNameModifier((&True0_M)->rtwLogInfo, "rt_");
    rtliSetLogFormat((&True0_M)->rtwLogInfo, 1);
    rtliSetLogMaxRows((&True0_M)->rtwLogInfo, 1000);
    rtliSetLogDecimation((&True0_M)->rtwLogInfo, 1);

    /*
     * Set pointers to the data and signal info for each output
     */
    {
      static void * rt_LoggedOutputSignalPtrs[10];
      rt_LoggedOutputSignalPtrs[0] = &True0_Y.X_i[0];
      rt_LoggedOutputSignalPtrs[1] = &True0_Y.V_i[0];
      rt_LoggedOutputSignalPtrs[2] = &True0_Y.V_b[0];
      rt_LoggedOutputSignalPtrs[3] = &True0_Y.a_b[0];
      rt_LoggedOutputSignalPtrs[4] = &True0_Y.a_i[0];
      rt_LoggedOutputSignalPtrs[5] = &True0_Y.DCM_ib[0];
      rt_LoggedOutputSignalPtrs[6] = &True0_Y.Quatq[0];
      rt_LoggedOutputSignalPtrs[7] = &True0_Y.Euler[0];
      rt_LoggedOutputSignalPtrs[8] = &True0_Y.omega[0];
      rt_LoggedOutputSignalPtrs[9] = &True0_Y.omega_dot[0];
      rtliSetLogYSignalPtrs((&True0_M)->rtwLogInfo, ((LogSignalPtrsType)
        rt_LoggedOutputSignalPtrs));
    }

    {
      static int_T rt_LoggedOutputWidths[] {
        3,
        3,
        3,
        3,
        3,
        9,
        4,
        3,
        3,
        3
      };

      static int_T rt_LoggedOutputNumDimensions[] {
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2
      };

      static int_T rt_LoggedOutputDimensions[] {
        3, 1,
        3, 1,
        3, 1,
        3, 1,
        3, 1,
        3, 3,
        4, 1,
        3, 1,
        3, 1,
        3, 1
      };

      static boolean_T rt_LoggedOutputIsVarDims[] {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      };

      static void* rt_LoggedCurrentSignalDimensions[] {
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr),
        (nullptr), (nullptr)
      };

      static int_T rt_LoggedCurrentSignalDimensionsSize[] {
        4, 4,
        4, 4,
        4, 4,
        4, 4,
        4, 4,
        4, 4,
        4, 4,
        4, 4,
        4, 4,
        4, 4
      };

      static BuiltInDTypeId rt_LoggedOutputDataTypeIds[] {
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE,
        SS_DOUBLE
      };

      static int_T rt_LoggedOutputComplexSignals[] {
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0
      };

      static RTWPreprocessingFcnPtr rt_LoggingPreprocessingFcnPtrs[] {
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr),
        (nullptr)
      };

      static const char_T *rt_LoggedOutputLabels[]{
        "<X_i>",
        "<V_i>",
        "<V_b>",
        "<a_b>",
        "<a_i>",
        "<DCM_ib>",
        "<Quat q>",
        "<Euler>",
        "<omega>",
        "<omega_dot>" };

      static const char_T *rt_LoggedOutputBlockNames[]{
        "True0/X_i",
        "True0/V_i",
        "True0/V_b",
        "True0/a_b",
        "True0/a_i",
        "True0/DCM_ib",
        "True0/Quat q",
        "True0/Euler",
        "True0/omega",
        "True0/omega_dot" };

      static RTWLogDataTypeConvert rt_RTWLogDataTypeConvert[] {
        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 },

        { 0, SS_DOUBLE, SS_DOUBLE, 0, 0, 0, 1.0, 0, 0.0 }
      };

      static RTWLogSignalInfo rt_LoggedOutputSignalInfo[] {
        {
          10,
          rt_LoggedOutputWidths,
          rt_LoggedOutputNumDimensions,
          rt_LoggedOutputDimensions,
          rt_LoggedOutputIsVarDims,
          rt_LoggedCurrentSignalDimensions,
          rt_LoggedCurrentSignalDimensionsSize,
          rt_LoggedOutputDataTypeIds,
          rt_LoggedOutputComplexSignals,
          (nullptr),
          rt_LoggingPreprocessingFcnPtrs,

          { rt_LoggedOutputLabels },
          (nullptr),
          (nullptr),
          (nullptr),

          { rt_LoggedOutputBlockNames },

          { (nullptr) },
          (nullptr),
          rt_RTWLogDataTypeConvert
        }
      };

      rtliSetLogYSignalInfo((&True0_M)->rtwLogInfo, rt_LoggedOutputSignalInfo);

      /* set currSigDims field */
      rt_LoggedCurrentSignalDimensions[0] = &rt_LoggedOutputWidths[0];
      rt_LoggedCurrentSignalDimensions[1] = &rt_LoggedOutputWidths[0];
      rt_LoggedCurrentSignalDimensions[2] = &rt_LoggedOutputWidths[1];
      rt_LoggedCurrentSignalDimensions[3] = &rt_LoggedOutputWidths[1];
      rt_LoggedCurrentSignalDimensions[4] = &rt_LoggedOutputWidths[2];
      rt_LoggedCurrentSignalDimensions[5] = &rt_LoggedOutputWidths[2];
      rt_LoggedCurrentSignalDimensions[6] = &rt_LoggedOutputWidths[3];
      rt_LoggedCurrentSignalDimensions[7] = &rt_LoggedOutputWidths[3];
      rt_LoggedCurrentSignalDimensions[8] = &rt_LoggedOutputWidths[4];
      rt_LoggedCurrentSignalDimensions[9] = &rt_LoggedOutputWidths[4];
      rt_LoggedCurrentSignalDimensions[10] = &rt_LoggedOutputWidths[5];
      rt_LoggedCurrentSignalDimensions[11] = &rt_LoggedOutputWidths[5];
      rt_LoggedCurrentSignalDimensions[12] = &rt_LoggedOutputWidths[6];
      rt_LoggedCurrentSignalDimensions[13] = &rt_LoggedOutputWidths[6];
      rt_LoggedCurrentSignalDimensions[14] = &rt_LoggedOutputWidths[7];
      rt_LoggedCurrentSignalDimensions[15] = &rt_LoggedOutputWidths[7];
      rt_LoggedCurrentSignalDimensions[16] = &rt_LoggedOutputWidths[8];
      rt_LoggedCurrentSignalDimensions[17] = &rt_LoggedOutputWidths[8];
      rt_LoggedCurrentSignalDimensions[18] = &rt_LoggedOutputWidths[9];
      rt_LoggedCurrentSignalDimensions[19] = &rt_LoggedOutputWidths[9];
    }

    rtliSetLogY((&True0_M)->rtwLogInfo, "yout");
  }

  /* Matfile logging */
  rt_StartDataLoggingWithStartTime((&True0_M)->rtwLogInfo, 0.0, rtmGetTFinal
    ((&True0_M)), (&True0_M)->Timing.stepSize0, (&rtmGetErrorStatus((&True0_M))));

  {
    /* local scratch DWork variables */
    int32_T ForEach_itr;

    /* Start for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
    /* Start for RateTransition: '<S1>/Rate Transition1' */
    True0_B.RateTransition1[0] = True0_P.rpm_init;
    True0_B.RateTransition1[1] = True0_P.rpm_init;
    True0_B.RateTransition1[2] = True0_P.rpm_init;
    True0_B.RateTransition1[3] = True0_P.rpm_init;

    /* Start for Iterator SubSystem: '<S37>/For Each Subsystem' */
    for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
      /* Start for If: '<S84>/If' */
      True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem = -1;

      /* Start for If: '<S82>/If' */
      True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_l = -1;

      /* Start for If: '<S70>/If' */
      True0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_e = -1;
    }

    /* End of Start for SubSystem: '<S37>/For Each Subsystem' */

    /* Start for If: '<S38>/If' */
    True0_DW.If_ActiveSubsystem = -1;

    /* End of Start for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
  }

  {
    /* local scratch DWork variables */
    int32_T ForEach_itr;

    /* SystemInitialize for Atomic SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */
    /* InitializeConditions for Integrator: '<S7>/Q-Integrator' */
    if (rtmIsFirstInitCond((&True0_M))) {
      True0_X.QIntegrator_CSTATE[0] = 0.0;
      True0_X.QIntegrator_CSTATE[1] = 0.0;
      True0_X.QIntegrator_CSTATE[2] = 0.0;
      True0_X.QIntegrator_CSTATE[3] = 0.0;
    }

    True0_DW.QIntegrator_IWORK = 1;

    /* End of InitializeConditions for Integrator: '<S7>/Q-Integrator' */

    /* InitializeConditions for Integrator: '<S2>/V_b' */
    True0_X.V_b_CSTATE[0] = True0_P.Vb_init[0];
    True0_X.V_b_CSTATE[1] = True0_P.Vb_init[1];
    True0_X.V_b_CSTATE[2] = True0_P.Vb_init[2];

    /* InitializeConditions for RateTransition: '<S1>/Rate Transition1' */
    True0_DW.RateTransition1_Buffer0[0] = True0_P.rpm_init;
    True0_DW.RateTransition1_Buffer0[1] = True0_P.rpm_init;
    True0_DW.RateTransition1_Buffer0[2] = True0_P.rpm_init;
    True0_DW.RateTransition1_Buffer0[3] = True0_P.rpm_init;

    /* InitializeConditions for Integrator: '<S2>/omega' */
    True0_X.omega_CSTATE[0] = True0_P.omega_init[0];

    /* InitializeConditions for Integrator: '<S2>/X_i' */
    True0_X.X_i_CSTATE[0] = True0_P.Xi_init[0];

    /* SystemInitialize for IfAction SubSystem: '<S38>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S38>/Merge' incorporates:
     *  Outport: '<S41>/Drag force'
     */
    True0_B.Forceagainstdirectionofmotiondu[0] = True0_P.Dragforce_Y0[0];

    /* End of SystemInitialize for SubSystem: '<S38>/Zero airspeed' */

    /* InitializeConditions for Integrator: '<S2>/omega' */
    True0_X.omega_CSTATE[1] = True0_P.omega_init[1];

    /* InitializeConditions for Integrator: '<S2>/X_i' */
    True0_X.X_i_CSTATE[1] = True0_P.Xi_init[1];

    /* SystemInitialize for IfAction SubSystem: '<S38>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S38>/Merge' incorporates:
     *  Outport: '<S41>/Drag force'
     */
    True0_B.Forceagainstdirectionofmotiondu[1] = True0_P.Dragforce_Y0[1];

    /* End of SystemInitialize for SubSystem: '<S38>/Zero airspeed' */

    /* InitializeConditions for Integrator: '<S2>/omega' */
    True0_X.omega_CSTATE[2] = True0_P.omega_init[2];

    /* InitializeConditions for Integrator: '<S2>/X_i' */
    True0_X.X_i_CSTATE[2] = True0_P.Xi_init[2];

    /* SystemInitialize for IfAction SubSystem: '<S38>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S38>/Merge' incorporates:
     *  Outport: '<S41>/Drag force'
     */
    True0_B.Forceagainstdirectionofmotiondu[2] = True0_P.Dragforce_Y0[2];

    /* End of SystemInitialize for SubSystem: '<S38>/Zero airspeed' */

    /* SystemInitialize for Iterator SubSystem: '<S37>/For Each Subsystem' */
    for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
      /* InitializeConditions for Integrator: '<S61>/Integrator' */
      True0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE = True0_P.rpm_init;

      /* SystemInitialize for IfAction SubSystem: '<S70>/Zero airspeed in rotor plane' */
      /* SystemInitialize for Merge: '<S70>/Merge' incorporates:
       *  Outport: '<S77>/Thrust direction (Body)'
       */
      True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0] =
        True0_P.CoreSubsys.ThrustdirectionBody_Y0[0];

      /* SystemInitialize for Merge: '<S70>/Merge1' incorporates:
       *  Outport: '<S77>/Hub moment (Body)'
       */
      True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[0] =
        True0_P.CoreSubsys.HubmomentBody_Y0[0];

      /* SystemInitialize for Merge: '<S70>/Merge' incorporates:
       *  Outport: '<S77>/Thrust direction (Body)'
       */
      True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1] =
        True0_P.CoreSubsys.ThrustdirectionBody_Y0[1];

      /* SystemInitialize for Merge: '<S70>/Merge1' incorporates:
       *  Outport: '<S77>/Hub moment (Body)'
       */
      True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[1] =
        True0_P.CoreSubsys.HubmomentBody_Y0[1];

      /* SystemInitialize for Merge: '<S70>/Merge' incorporates:
       *  Outport: '<S77>/Thrust direction (Body)'
       */
      True0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2] =
        True0_P.CoreSubsys.ThrustdirectionBody_Y0[2];

      /* SystemInitialize for Merge: '<S70>/Merge1' incorporates:
       *  Outport: '<S77>/Hub moment (Body)'
       */
      True0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[2] =
        True0_P.CoreSubsys.HubmomentBody_Y0[2];

      /* End of SystemInitialize for SubSystem: '<S70>/Zero airspeed in rotor plane' */

      /* SystemInitialize for IfAction SubSystem: '<S82>/Zero airspeed' */
      /* SystemInitialize for Merge: '<S82>/Merge' incorporates:
       *  Outport: '<S86>/AoA (rad)'
       */
      True0_B.CoreSubsys[ForEach_itr].Angleofattackrad =
        True0_P.CoreSubsys.AoArad_Y0;

      /* End of SystemInitialize for SubSystem: '<S82>/Zero airspeed' */
    }

    /* End of SystemInitialize for SubSystem: '<S37>/For Each Subsystem' */
    /* End of SystemInitialize for SubSystem: '<Root>/True dynamic system representation of a multirotor UAV' */

    /* set "at time zero" to false */
    if (rtmIsFirstInitCond((&True0_M))) {
      rtmSetFirstInitCond((&True0_M), 0);
    }
  }
}

/* Model terminate function */
void True0::terminate()
{
  /* (no terminate code required) */
}

/* Constructor */
True0::True0() :
  True0_U(),
  True0_Y(),
  True0_B(),
  True0_DW(),
  True0_X(),
  True0_M()
{
  /* Currently there is no constructor body generated.*/
}

/* Destructor */
/* Currently there is no destructor body generated.*/
True0::~True0() = default;

/* Real-Time Model get method */
RT_MODEL_True0_T * True0::getRTM()
{
  return (&True0_M);
}
