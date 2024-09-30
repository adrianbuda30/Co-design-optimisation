/*
 * multirotor0.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "multirotor0".
 *
 * Model version              : 14.51
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Mon Aug 28 00:36:10 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: 32-bit Generic
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#include "multirotor0.h"
#include "rtwtypes.h"
#include <cmath>
#include <cstring>
#include "multirotor0_private.h"
#include "rt_defines.h"

extern "C"
{

#include "rt_nonfinite.h"

}

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
void multirotor0::rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
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
  int_T nXc { 21 };

  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  multirotor0_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  this->step0();
  multirotor0_derivatives();

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
  multirotor0_derivatives();

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

real_T rt_atan2d_snf(real_T u0, real_T u1)
{
  real_T y;
  if (std::isnan(u0) || std::isnan(u1)) {
    y = (rtNaN);
  } else if (std::isinf(u0) && std::isinf(u1)) {
    int32_T tmp;
    int32_T tmp_0;
    if (u0 > 0.0) {
      tmp = 1;
    } else {
      tmp = -1;
    }

    if (u1 > 0.0) {
      tmp_0 = 1;
    } else {
      tmp_0 = -1;
    }

    y = std::atan2(static_cast<real_T>(tmp), static_cast<real_T>(tmp_0));
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

void rt_mldivide_U1d3x3_U2d_JBYZyA3A(const real_T u0[9], const real_T u1[3],
  real_T y[3])
{
  real_T A[9];
  real_T a21;
  real_T maxval;
  int32_T r1;
  int32_T r2;
  int32_T r3;
  std::memcpy(&A[0], &u0[0], 9U * sizeof(real_T));
  r1 = 0;
  r2 = 1;
  r3 = 2;
  maxval = std::abs(u0[0]);
  a21 = std::abs(u0[1]);
  if (a21 > maxval) {
    maxval = a21;
    r1 = 1;
    r2 = 0;
  }

  if (std::abs(u0[2]) > maxval) {
    r1 = 2;
    r2 = 1;
    r3 = 0;
  }

  A[r2] = u0[r2] / u0[r1];
  A[r3] /= A[r1];
  A[r2 + 3] -= A[r1 + 3] * A[r2];
  A[r3 + 3] -= A[r1 + 3] * A[r3];
  A[r2 + 6] -= A[r1 + 6] * A[r2];
  A[r3 + 6] -= A[r1 + 6] * A[r3];
  if (std::abs(A[r3 + 3]) > std::abs(A[r2 + 3])) {
    int32_T rtemp;
    rtemp = r2 + 1;
    r2 = r3;
    r3 = rtemp - 1;
  }

  A[r3 + 3] /= A[r2 + 3];
  A[r3 + 6] -= A[r3 + 3] * A[r2 + 6];
  y[1] = u1[r2] - u1[r1] * A[r2];
  y[2] = (u1[r3] - u1[r1] * A[r3]) - A[r3 + 3] * y[1];
  y[2] /= A[r3 + 6];
  y[0] = u1[r1] - A[r1 + 6] * y[2];
  y[1] -= A[r2 + 6] * y[2];
  y[1] /= A[r2 + 3];
  y[0] -= A[r1 + 3] * y[1];
  y[0] /= A[r1];
}

/* Model step function for TID0 */
void multirotor0::step0()              /* Sample time: [0.0s, 0.0s] */
{
  /* local scratch DWork variables */
  int32_T ForEach_itr;
  real_T Rz[36];
  real_T COM_airframe[12];
  real_T COM_propeller[12];
  real_T d_y[12];
  real_T rtb_ImpAsg_InsertedFor_Motor_fo[12];
  real_T rtb_ImpAsg_InsertedFor_Motor_mo[12];
  real_T y[12];
  real_T I_airframe_total[9];
  real_T I_propeller_cm[9];
  real_T I_propeller_total[9];
  real_T Product_tmp[9];
  real_T d_propeller[9];
  real_T rtb_VectorConcatenate[9];
  real_T airframe_mass[4];
  real_T propeller_mass[4];
  real_T rtb_Divide[4];
  real_T rtb_ImpAsg_InsertedFor_RPM_moto[4];
  real_T rtb_Sum_a[3];
  real_T rtb_TrueairspeedBodyaxes[3];
  real_T rtb_TrueairspeedBodyaxes_b[3];
  real_T rtb_TrueairspeedBodyaxes_m[3];
  real_T COM_airframe_0;
  real_T COM_airframe_1;
  real_T COM_airframe_2;
  real_T COM_airframe_3;
  real_T COM_airframe_4;
  real_T COM_airframe_5;
  real_T COM_airframe_6;
  real_T COM_airframe_7;
  real_T COM_airframe_8;
  real_T COM_airframe_9;
  real_T COM_airframe_a;
  real_T COM_airframe_b;
  real_T COM_propeller_0;
  real_T COM_propeller_1;
  real_T COM_propeller_2;
  real_T COM_propeller_3;
  real_T COM_propeller_4;
  real_T COM_propeller_5;
  real_T COM_propeller_6;
  real_T COM_system_inter_idx_0;
  real_T COM_system_inter_idx_1;
  real_T COM_system_inter_idx_2;
  real_T cphi;
  real_T ctheta;
  real_T phi;
  real_T q1;
  real_T rtb_Airspeeddirectionintherot_0;
  real_T rtb_VectorConcatenate_tmp;
  real_T rtb_VectorConcatenate_tmp_0;
  real_T rtb_VectorConcatenate_tmp_1;
  real_T rtb_VectorConcatenate_tmp_2;
  real_T rtb_VectorConcatenate_tmp_3;
  real_T theta;
  int32_T Rz_tmp;
  int32_T i;
  int32_T xpageoffset;
  int8_T b_I[9];
  int8_T rtAction;
  if (rtmIsMajorTimeStep((&multirotor0_M))) {
    /* set solver stop time */
    rtsiSetSolverStopTime(&(&multirotor0_M)->solverInfo,(((&multirotor0_M)
      ->Timing.clockTick0+1)*(&multirotor0_M)->Timing.stepSize0));

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.002s, 0.0s]  */
    ((&multirotor0_M)->Timing.RateInteraction.TID1_2)++;
    if (((&multirotor0_M)->Timing.RateInteraction.TID1_2) > 1) {
      (&multirotor0_M)->Timing.RateInteraction.TID1_2 = 0;
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep((&multirotor0_M))) {
    (&multirotor0_M)->Timing.t[0] = rtsiGetT(&(&multirotor0_M)->solverInfo);
  }

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  if (rtmIsMajorTimeStep((&multirotor0_M))) {
    /* MATLAB Function: '<S8>/MATLAB Function' */
    multirotor0_B.quat_output[0] = 1.0;
    multirotor0_B.quat_output[1] = 0.0;
    multirotor0_B.quat_output[2] = 0.0;
    multirotor0_B.quat_output[3] = 0.0;
  }

  /* Integrator: '<S8>/Q-Integrator' */
  if (multirotor0_DW.QIntegrator_IWORK != 0) {
    multirotor0_X.QIntegrator_CSTATE[0] = multirotor0_B.quat_output[0];
    multirotor0_X.QIntegrator_CSTATE[1] = multirotor0_B.quat_output[1];
    multirotor0_X.QIntegrator_CSTATE[2] = multirotor0_B.quat_output[2];
    multirotor0_X.QIntegrator_CSTATE[3] = multirotor0_B.quat_output[3];
  }

  /* Sqrt: '<S22>/Sqrt' incorporates:
   *  Integrator: '<S8>/Q-Integrator'
   *  Product: '<S23>/Product'
   */
  cphi = std::sqrt(((multirotor0_X.QIntegrator_CSTATE[0] *
                     multirotor0_X.QIntegrator_CSTATE[0] +
                     multirotor0_X.QIntegrator_CSTATE[1] *
                     multirotor0_X.QIntegrator_CSTATE[1]) +
                    multirotor0_X.QIntegrator_CSTATE[2] *
                    multirotor0_X.QIntegrator_CSTATE[2]) +
                   multirotor0_X.QIntegrator_CSTATE[3] *
                   multirotor0_X.QIntegrator_CSTATE[3]);

  /* Product: '<S19>/Divide' incorporates:
   *  Integrator: '<S8>/Q-Integrator'
   */
  rtb_Divide[0] = multirotor0_X.QIntegrator_CSTATE[0] / cphi;
  rtb_Divide[1] = multirotor0_X.QIntegrator_CSTATE[1] / cphi;
  rtb_Divide[2] = multirotor0_X.QIntegrator_CSTATE[2] / cphi;
  rtb_Divide[3] = multirotor0_X.QIntegrator_CSTATE[3] / cphi;

  /* Product: '<S24>/Product' incorporates:
   *  Product: '<S25>/Product'
   */
  rtb_VectorConcatenate_tmp_0 = rtb_Divide[0] * rtb_Divide[0];

  /* Product: '<S24>/Product2' incorporates:
   *  Product: '<S25>/Product2'
   */
  rtb_VectorConcatenate_tmp_1 = rtb_Divide[1] * rtb_Divide[1];

  /* Product: '<S24>/Product3' incorporates:
   *  Product: '<S25>/Product3'
   *  Product: '<S26>/Product3'
   */
  cphi = rtb_Divide[2] * rtb_Divide[2];

  /* Product: '<S24>/Product4' incorporates:
   *  Product: '<S25>/Product4'
   *  Product: '<S26>/Product4'
   */
  phi = rtb_Divide[3] * rtb_Divide[3];

  /* Sum: '<S24>/Add' incorporates:
   *  Fcn: '<S9>/Fcn4'
   *  Product: '<S24>/Product'
   *  Product: '<S24>/Product2'
   *  Product: '<S24>/Product3'
   *  Product: '<S24>/Product4'
   */
  rtb_VectorConcatenate_tmp_3 = ((rtb_VectorConcatenate_tmp_0 +
    rtb_VectorConcatenate_tmp_1) - cphi) - phi;
  rtb_VectorConcatenate[0] = rtb_VectorConcatenate_tmp_3;

  /* Product: '<S29>/Product' incorporates:
   *  Product: '<S27>/Product'
   */
  rtb_VectorConcatenate_tmp = rtb_Divide[1] * rtb_Divide[2];

  /* Product: '<S29>/Product2' incorporates:
   *  Product: '<S27>/Product2'
   */
  ctheta = rtb_Divide[0] * rtb_Divide[3];

  /* Gain: '<S29>/Gain' incorporates:
   *  Product: '<S29>/Product'
   *  Product: '<S29>/Product2'
   *  Sum: '<S29>/Add'
   */
  rtb_VectorConcatenate[1] = (rtb_VectorConcatenate_tmp - ctheta) * 2.0;

  /* Product: '<S31>/Product' incorporates:
   *  Product: '<S28>/Product'
   */
  rtb_VectorConcatenate_tmp_2 = rtb_Divide[1] * rtb_Divide[3];

  /* Product: '<S31>/Product2' incorporates:
   *  Product: '<S28>/Product2'
   */
  theta = rtb_Divide[0] * rtb_Divide[2];

  /* Gain: '<S31>/Gain' incorporates:
   *  Product: '<S31>/Product'
   *  Product: '<S31>/Product2'
   *  Sum: '<S31>/Add'
   */
  rtb_VectorConcatenate[2] = (rtb_VectorConcatenate_tmp_2 + theta) * 2.0;

  /* Sum: '<S27>/Add' incorporates:
   *  Fcn: '<S9>/Fcn2'
   */
  rtb_VectorConcatenate_tmp += ctheta;

  /* Gain: '<S27>/Gain' incorporates:
   *  Sum: '<S27>/Add'
   */
  rtb_VectorConcatenate[3] = rtb_VectorConcatenate_tmp * 2.0;

  /* Sum: '<S25>/Add' incorporates:
   *  Sum: '<S26>/Add'
   */
  rtb_VectorConcatenate_tmp_0 -= rtb_VectorConcatenate_tmp_1;
  rtb_VectorConcatenate[4] = (rtb_VectorConcatenate_tmp_0 + cphi) - phi;

  /* Product: '<S32>/Product' incorporates:
   *  Product: '<S30>/Product'
   */
  rtb_VectorConcatenate_tmp_1 = rtb_Divide[2] * rtb_Divide[3];

  /* Product: '<S32>/Product2' incorporates:
   *  Product: '<S30>/Product2'
   */
  ctheta = rtb_Divide[0] * rtb_Divide[1];

  /* Gain: '<S32>/Gain' incorporates:
   *  Product: '<S32>/Product'
   *  Product: '<S32>/Product2'
   *  Sum: '<S32>/Add'
   */
  rtb_VectorConcatenate[5] = (rtb_VectorConcatenate_tmp_1 - ctheta) * 2.0;

  /* Sum: '<S28>/Add' incorporates:
   *  Fcn: '<S9>/Fcn1'
   */
  rtb_VectorConcatenate_tmp_2 -= theta;

  /* Gain: '<S28>/Gain' incorporates:
   *  Sum: '<S28>/Add'
   */
  rtb_VectorConcatenate[6] = rtb_VectorConcatenate_tmp_2 * 2.0;

  /* Sum: '<S30>/Add' incorporates:
   *  Fcn: '<S9>/Fcn'
   */
  rtb_VectorConcatenate_tmp_1 += ctheta;

  /* Gain: '<S30>/Gain' incorporates:
   *  Sum: '<S30>/Add'
   */
  rtb_VectorConcatenate[7] = rtb_VectorConcatenate_tmp_1 * 2.0;

  /* Sum: '<S26>/Add' incorporates:
   *  Fcn: '<S9>/Fcn3'
   */
  rtb_VectorConcatenate_tmp_0 = (rtb_VectorConcatenate_tmp_0 - cphi) + phi;
  rtb_VectorConcatenate[8] = rtb_VectorConcatenate_tmp_0;
  for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
    /* Math: '<S5>/Math Function2' incorporates:
     *  Concatenate: '<S33>/Vector Concatenate'
     *  Math: '<S6>/Math Function2'
     */
    Product_tmp[3 * xpageoffset] = rtb_VectorConcatenate[xpageoffset];
    Product_tmp[3 * xpageoffset + 1] = rtb_VectorConcatenate[xpageoffset + 3];
    Product_tmp[3 * xpageoffset + 2] = rtb_VectorConcatenate[xpageoffset + 6];
  }

  /* Integrator: '<S2>/V_b' incorporates:
   *  Math: '<S5>/Math Function2'
   */
  cphi = multirotor0_X.V_b_CSTATE[1];
  phi = multirotor0_X.V_b_CSTATE[0];
  ctheta = multirotor0_X.V_b_CSTATE[2];
  for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
    /* Product: '<S5>/Product' incorporates:
     *  Integrator: '<S2>/V_b'
     *  Math: '<S5>/Math Function2'
     */
    multirotor0_B.Product[xpageoffset] = (Product_tmp[xpageoffset + 3] * cphi +
      Product_tmp[xpageoffset] * phi) + Product_tmp[xpageoffset + 6] * ctheta;
  }

  /* RateTransition: '<S1>/Rate Transition1' */
  if (rtmIsMajorTimeStep((&multirotor0_M)) && ((&multirotor0_M)
       ->Timing.RateInteraction.TID1_2 == 1)) {
    /* RateTransition: '<S1>/Rate Transition1' */
    multirotor0_B.RateTransition1[0] = multirotor0_DW.RateTransition1_Buffer0[0];
    multirotor0_B.RateTransition1[1] = multirotor0_DW.RateTransition1_Buffer0[1];
    multirotor0_B.RateTransition1[2] = multirotor0_DW.RateTransition1_Buffer0[2];
    multirotor0_B.RateTransition1[3] = multirotor0_DW.RateTransition1_Buffer0[3];
  }

  /* End of RateTransition: '<S1>/Rate Transition1' */

  /* Product: '<S117>/Product' incorporates:
   *  Inport: '<Root>/wind'
   */
  cphi = multirotor0_U.Wind_i[1];
  phi = multirotor0_U.Wind_i[0];
  ctheta = multirotor0_U.Wind_i[2];
  for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
    /* Product: '<S56>/Product' incorporates:
     *  Concatenate: '<S33>/Vector Concatenate'
     */
    theta = (rtb_VectorConcatenate[xpageoffset + 3] * cphi +
             rtb_VectorConcatenate[xpageoffset] * phi) +
      rtb_VectorConcatenate[xpageoffset + 6] * ctheta;
    rtb_TrueairspeedBodyaxes_b[xpageoffset] = theta;

    /* Sum: '<S58>/Sum1' incorporates:
     *  Integrator: '<S2>/V_b'
     */
    rtb_TrueairspeedBodyaxes[xpageoffset] = multirotor0_X.V_b_CSTATE[xpageoffset]
      - theta;
  }

  /* End of Product: '<S117>/Product' */
  if (rtmIsMajorTimeStep((&multirotor0_M))) {
    /* MATLAB Function: '<S3>/MATLAB Function' incorporates:
     *  Inport: '<Root>/COM_mass_center'
     *  Inport: '<Root>/Motor_arm_angle'
     *  Inport: '<Root>/Surface_params'
     *  Inport: '<Root>/arm_length'
     *  Inport: '<Root>/arm_radius'
     *  Inport: '<Root>/mass_center'
     *  Inport: '<Root>/max_rpm'
     *  Inport: '<Root>/min_rpm'
     *  Inport: '<Root>/prop_diameter'
     *  Inport: '<Root>/prop_height'
     *  Inport: '<Root>/rotation_direction'
     */
    multirotor0_B.Surface_params[0] = multirotor0_U.Surface_params[0];
    multirotor0_B.Surface_params[1] = multirotor0_U.Surface_params[1];
    multirotor0_B.Surface_params[2] = multirotor0_U.Surface_params[2];
    for (i = 0; i < 4; i++) {
      phi = 1.5707963267948966 * static_cast<real_T>(i) + 0.78539816339744828;
      cphi = std::sin(phi);
      phi = std::cos(phi);
      Rz[9 * i] = phi;
      Rz[9 * i + 3] = -cphi;
      Rz[9 * i + 6] = 0.0;
      Rz[9 * i + 1] = cphi;
      Rz[9 * i + 4] = phi;
      Rz[9 * i + 7] = 0.0;
      Rz[9 * i + 2] = 0.0;
      Rz[9 * i + 5] = 0.0;
      Rz[9 * i + 8] = 1.0;
      propeller_mass[i] = multirotor0_U.prop_diameter[i] * 0.054133858267716536;
    }

    cphi = ((propeller_mass[0] + propeller_mass[1]) + propeller_mass[2]) +
      propeller_mass[3];
    airframe_mass[0] = 3.1415926535897931 * multirotor0_U.arm_length[0] *
      (multirotor0_U.arm_radius[0] * multirotor0_U.arm_radius[0]) * 1700.0;
    phi = 3.1415926535897931 * multirotor0_U.arm_length[1] *
      (multirotor0_U.arm_radius[1] * multirotor0_U.arm_radius[1]) * 1700.0;
    airframe_mass[1] = phi;
    airframe_mass[2] = 3.1415926535897931 * multirotor0_U.arm_length[2] *
      (multirotor0_U.arm_radius[2] * multirotor0_U.arm_radius[2]) * 1700.0;
    ctheta = 3.1415926535897931 * multirotor0_U.arm_length[3] *
      (multirotor0_U.arm_radius[3] * multirotor0_U.arm_radius[3]) * 1700.0;
    airframe_mass[3] = ctheta;
    phi = ((phi + airframe_mass[0]) + airframe_mass[2]) + ctheta;
    multirotor0_B.total_mass = (multirotor0_U.mass_center + phi) + cphi;
    for (i = 0; i < 4; i++) {
      theta = multirotor0_U.arm_length[i];
      COM_system_inter_idx_1 = multirotor0_U.prop_height[i];
      COM_system_inter_idx_0 = theta / 2.0;
      for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
        Rz_tmp = 9 * i + xpageoffset;
        ctheta = Rz[Rz_tmp + 3];
        q1 = Rz[Rz_tmp];
        COM_system_inter_idx_2 = Rz[Rz_tmp + 6];
        Rz_tmp = (xpageoffset << 2) + i;
        COM_propeller[Rz_tmp] = (ctheta * 0.0 + q1 * theta) +
          COM_system_inter_idx_2 * COM_system_inter_idx_1;
        COM_airframe[Rz_tmp] = (ctheta * 0.0 + q1 * COM_system_inter_idx_0) +
          COM_system_inter_idx_2 * 0.0;
      }
    }

    q1 = COM_propeller[0];
    COM_system_inter_idx_0 = COM_propeller[1];
    COM_system_inter_idx_1 = COM_propeller[2];
    COM_system_inter_idx_2 = COM_propeller[3];
    COM_propeller_0 = COM_propeller[4];
    rtb_Airspeeddirectionintherot_0 = COM_propeller[5];
    COM_propeller_1 = COM_propeller[6];
    COM_propeller_2 = COM_propeller[7];
    COM_propeller_3 = COM_propeller[8];
    COM_propeller_4 = COM_propeller[9];
    COM_propeller_5 = COM_propeller[10];
    COM_propeller_6 = COM_propeller[11];
    COM_airframe_0 = COM_airframe[0];
    COM_airframe_1 = COM_airframe[1];
    COM_airframe_2 = COM_airframe[2];
    COM_airframe_3 = COM_airframe[3];
    COM_airframe_4 = COM_airframe[4];
    COM_airframe_5 = COM_airframe[5];
    COM_airframe_6 = COM_airframe[6];
    COM_airframe_7 = COM_airframe[7];
    COM_airframe_8 = COM_airframe[8];
    COM_airframe_9 = COM_airframe[9];
    COM_airframe_a = COM_airframe[10];
    COM_airframe_b = COM_airframe[11];
    for (i = 0; i < 4; i++) {
      ctheta = propeller_mass[i];
      theta = airframe_mass[i];
      y[3 * i] = ((ctheta * COM_system_inter_idx_0 + ctheta * q1) + ctheta *
                  COM_system_inter_idx_1) + ctheta * COM_system_inter_idx_2;
      xpageoffset = 3 * i + 1;
      y[xpageoffset] = ((ctheta * rtb_Airspeeddirectionintherot_0 + ctheta *
                         COM_propeller_0) + ctheta * COM_propeller_1) + ctheta *
        COM_propeller_2;
      Rz_tmp = 3 * i + 2;
      y[Rz_tmp] = ((ctheta * COM_propeller_4 + ctheta * COM_propeller_3) +
                   ctheta * COM_propeller_5) + ctheta * COM_propeller_6;
      d_y[3 * i] = ((theta * COM_airframe_1 + theta * COM_airframe_0) + theta *
                    COM_airframe_2) + theta * COM_airframe_3;
      d_y[xpageoffset] = ((theta * COM_airframe_5 + theta * COM_airframe_4) +
                          theta * COM_airframe_6) + theta * COM_airframe_7;
      d_y[Rz_tmp] = ((theta * COM_airframe_9 + theta * COM_airframe_8) + theta *
                     COM_airframe_a) + theta * COM_airframe_b;
    }

    cphi = (cphi + phi) + multirotor0_U.mass_center;
    COM_system_inter_idx_0 = ((y[0] + d_y[0]) + multirotor0_U.mass_center *
      multirotor0_U.COM_mass_center[0]) / cphi;
    COM_system_inter_idx_1 = ((y[1] + d_y[1]) + multirotor0_U.mass_center *
      multirotor0_U.COM_mass_center[1]) / cphi;
    COM_system_inter_idx_2 = ((y[2] + d_y[2]) + multirotor0_U.mass_center *
      multirotor0_U.COM_mass_center[2]) / cphi;
    for (i = 0; i < 4; i++) {
      ctheta = propeller_mass[i];
      theta = multirotor0_U.arm_length[i];
      phi = theta * theta;
      q1 = 0.083333333333333329 * ctheta * phi;
      std::memset(&I_propeller_cm[0], 0, 9U * sizeof(real_T));
      I_propeller_cm[0] = q1;
      COM_propeller_0 = COM_propeller[i] - COM_system_inter_idx_0;
      rtb_Sum_a[0] = COM_propeller_0;
      cphi = COM_propeller_0 * COM_propeller_0;
      I_propeller_cm[4] = q1;
      COM_propeller_0 = COM_propeller[i + 4] - COM_system_inter_idx_1;
      rtb_Sum_a[1] = COM_propeller_0;
      cphi += COM_propeller_0 * COM_propeller_0;
      I_propeller_cm[8] = 0.0;
      COM_propeller_0 = COM_propeller[i + 8] - COM_system_inter_idx_2;
      rtb_Sum_a[2] = COM_propeller_0;
      cphi = (COM_propeller_0 * COM_propeller_0 + cphi) * ctheta;
      for (xpageoffset = 0; xpageoffset < 9; xpageoffset++) {
        b_I[xpageoffset] = 0;
      }

      for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
        b_I[xpageoffset + 3 * xpageoffset] = 1;
        d_propeller[3 * xpageoffset] = rtb_Sum_a[0] * rtb_Sum_a[xpageoffset];
        d_propeller[3 * xpageoffset + 1] = rtb_Sum_a[1] * rtb_Sum_a[xpageoffset];
        d_propeller[3 * xpageoffset + 2] = COM_propeller_0 *
          rtb_Sum_a[xpageoffset];
      }

      theta = airframe_mass[i];
      q1 = multirotor0_U.arm_radius[i];
      q1 *= q1;
      COM_propeller_0 = COM_airframe[i] - COM_system_inter_idx_0;
      rtb_Sum_a[0] = COM_propeller_0;
      for (xpageoffset = 0; xpageoffset < 9; xpageoffset++) {
        I_propeller_total[xpageoffset] = (cphi * static_cast<real_T>
          (b_I[xpageoffset]) + I_propeller_cm[xpageoffset]) - ctheta *
          d_propeller[xpageoffset];
        I_propeller_cm[xpageoffset] = 0.0;
        b_I[xpageoffset] = 0;
      }

      cphi = 0.25 * theta * q1 + 0.083333333333333329 * theta * phi;
      phi = cphi;
      I_propeller_cm[0] = cphi;
      cphi = COM_propeller_0 * COM_propeller_0;
      I_propeller_cm[4] = phi;
      COM_propeller_0 = COM_airframe[i + 4] - COM_system_inter_idx_1;
      rtb_Sum_a[1] = COM_propeller_0;
      cphi += COM_propeller_0 * COM_propeller_0;
      I_propeller_cm[8] = 0.5 * theta * q1;
      COM_propeller_0 = COM_airframe[i + 8] - COM_system_inter_idx_2;
      rtb_Sum_a[2] = COM_propeller_0;
      cphi = (COM_propeller_0 * COM_propeller_0 + cphi) * theta;
      for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
        b_I[xpageoffset + 3 * xpageoffset] = 1;
        d_propeller[3 * xpageoffset] = rtb_Sum_a[0] * rtb_Sum_a[xpageoffset];
        d_propeller[3 * xpageoffset + 1] = rtb_Sum_a[1] * rtb_Sum_a[xpageoffset];
        d_propeller[3 * xpageoffset + 2] = COM_propeller_0 *
          rtb_Sum_a[xpageoffset];
      }

      for (xpageoffset = 0; xpageoffset < 9; xpageoffset++) {
        I_airframe_total[xpageoffset] = (cphi * static_cast<real_T>
          (b_I[xpageoffset]) + I_propeller_cm[xpageoffset]) - theta *
          d_propeller[xpageoffset];
      }
    }

    for (xpageoffset = 0; xpageoffset < 9; xpageoffset++) {
      I_propeller_total[xpageoffset] += I_airframe_total[xpageoffset];
    }

    multirotor0_B.inertial_matrix[0] = I_propeller_total[0];
    multirotor0_B.inertial_matrix[3] = 0.0;
    multirotor0_B.inertial_matrix[6] = 0.0;
    multirotor0_B.inertial_matrix[1] = 0.0;
    multirotor0_B.inertial_matrix[4] = I_propeller_total[4];
    multirotor0_B.inertial_matrix[7] = 0.0;
    multirotor0_B.inertial_matrix[2] = 0.0;
    multirotor0_B.inertial_matrix[5] = 0.0;
    multirotor0_B.inertial_matrix[8] = I_propeller_total[8];
    multirotor0_B.MotorMatrix_real[0] = multirotor0_U.Motor_arm_angle[0];
    multirotor0_B.MotorMatrix_real[4] = multirotor0_U.arm_length[0];
    multirotor0_B.MotorMatrix_real[8] = multirotor0_U.prop_height[0];
    multirotor0_B.MotorMatrix_real[12] = multirotor0_U.rotation_direction[0];
    multirotor0_B.MotorMatrix_real[16] = 1.0;
    multirotor0_B.MotorMatrix_real[20] = 0.01;
    multirotor0_B.MotorMatrix_real[24] = 9.6820000000000012E-5;
    multirotor0_B.MotorMatrix_real[32] = 1.4504E-6;
    multirotor0_B.MotorMatrix_real[28] = 1.0872000000000001E-7;
    multirotor0_B.MotorMatrix_real[36] = 1.6312E-9;
    multirotor0_B.MotorMatrix_real[40] = multirotor0_U.min_rpm[0];
    multirotor0_B.MotorMatrix_real[44] = multirotor0_U.max_rpm[0];
    multirotor0_B.MotorMatrix_real[48] = 0.0;
    multirotor0_B.MotorMatrix_real[52] = 0.0;
    multirotor0_B.MotorMatrix_real[56] = 0.0;
    multirotor0_B.MotorMatrix_real[60] = multirotor0_U.prop_diameter[0];
    multirotor0_B.MotorMatrix_real[64] = propeller_mass[0];
    multirotor0_B.MotorMatrix_real[1] = multirotor0_U.Motor_arm_angle[1];
    multirotor0_B.MotorMatrix_real[5] = multirotor0_U.arm_length[1];
    multirotor0_B.MotorMatrix_real[9] = multirotor0_U.prop_height[1];
    multirotor0_B.MotorMatrix_real[13] = multirotor0_U.rotation_direction[1];
    multirotor0_B.MotorMatrix_real[17] = 1.0;
    multirotor0_B.MotorMatrix_real[21] = 0.01;
    multirotor0_B.MotorMatrix_real[25] = 9.6820000000000012E-5;
    multirotor0_B.MotorMatrix_real[33] = 1.4504E-6;
    multirotor0_B.MotorMatrix_real[29] = 1.0872000000000001E-7;
    multirotor0_B.MotorMatrix_real[37] = 1.6312E-9;
    multirotor0_B.MotorMatrix_real[41] = multirotor0_U.min_rpm[1];
    multirotor0_B.MotorMatrix_real[45] = multirotor0_U.max_rpm[1];
    multirotor0_B.MotorMatrix_real[49] = 0.0;
    multirotor0_B.MotorMatrix_real[53] = 0.0;
    multirotor0_B.MotorMatrix_real[57] = 0.0;
    multirotor0_B.MotorMatrix_real[61] = multirotor0_U.prop_diameter[1];
    multirotor0_B.MotorMatrix_real[65] = propeller_mass[1];
    multirotor0_B.MotorMatrix_real[2] = multirotor0_U.Motor_arm_angle[2];
    multirotor0_B.MotorMatrix_real[6] = multirotor0_U.arm_length[2];
    multirotor0_B.MotorMatrix_real[10] = multirotor0_U.prop_height[2];
    multirotor0_B.MotorMatrix_real[14] = multirotor0_U.rotation_direction[2];
    multirotor0_B.MotorMatrix_real[18] = 1.0;
    multirotor0_B.MotorMatrix_real[22] = 0.01;
    multirotor0_B.MotorMatrix_real[26] = 9.6820000000000012E-5;
    multirotor0_B.MotorMatrix_real[34] = 1.4504E-6;
    multirotor0_B.MotorMatrix_real[30] = 1.0872000000000001E-7;
    multirotor0_B.MotorMatrix_real[38] = 1.6312E-9;
    multirotor0_B.MotorMatrix_real[42] = multirotor0_U.min_rpm[2];
    multirotor0_B.MotorMatrix_real[46] = multirotor0_U.max_rpm[2];
    multirotor0_B.MotorMatrix_real[50] = 0.0;
    multirotor0_B.MotorMatrix_real[54] = 0.0;
    multirotor0_B.MotorMatrix_real[58] = 0.0;
    multirotor0_B.MotorMatrix_real[62] = multirotor0_U.prop_diameter[2];
    multirotor0_B.MotorMatrix_real[66] = propeller_mass[2];
    multirotor0_B.MotorMatrix_real[3] = multirotor0_U.Motor_arm_angle[3];
    multirotor0_B.MotorMatrix_real[7] = multirotor0_U.arm_length[3];
    multirotor0_B.MotorMatrix_real[11] = multirotor0_U.prop_height[3];
    multirotor0_B.MotorMatrix_real[15] = multirotor0_U.rotation_direction[3];
    multirotor0_B.MotorMatrix_real[19] = 1.0;
    multirotor0_B.MotorMatrix_real[23] = 0.01;
    multirotor0_B.MotorMatrix_real[27] = 9.6820000000000012E-5;
    multirotor0_B.MotorMatrix_real[35] = 1.4504E-6;
    multirotor0_B.MotorMatrix_real[31] = 1.0872000000000001E-7;
    multirotor0_B.MotorMatrix_real[39] = 1.6312E-9;
    multirotor0_B.MotorMatrix_real[43] = multirotor0_U.min_rpm[3];
    multirotor0_B.MotorMatrix_real[47] = multirotor0_U.max_rpm[3];
    multirotor0_B.MotorMatrix_real[63] = multirotor0_U.prop_diameter[3];
    multirotor0_B.MotorMatrix_real[67] = propeller_mass[3];
    multirotor0_B.MotorMatrix_real[51] = 0.0;
    multirotor0_B.COM_system[0] = COM_system_inter_idx_0;
    multirotor0_B.MotorMatrix_real[55] = 0.0;
    multirotor0_B.COM_system[1] = COM_system_inter_idx_1;
    multirotor0_B.MotorMatrix_real[59] = 0.0;
    multirotor0_B.COM_system[2] = COM_system_inter_idx_2;

    /* End of MATLAB Function: '<S3>/MATLAB Function' */
  }

  /* Outputs for Iterator SubSystem: '<S39>/For Each Subsystem' incorporates:
   *  ForEach: '<S57>/For Each'
   */
  for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
    /* ForEachSliceSelector generated from: '<S57>/MotorMatrix_real' incorporates:
     *  RelationalOperator: '<S63>/Relational Operator'
     *  RelationalOperator: '<S67>/LowerRelop1'
     */
    q1 = multirotor0_B.MotorMatrix_real[ForEach_itr + 44];

    /* Switch: '<S67>/Switch2' incorporates:
     *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
     *  Integrator: '<S63>/Integrator'
     *  RelationalOperator: '<S67>/LowerRelop1'
     */
    if (multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE > q1) {
      cphi = multirotor0_B.MotorMatrix_real[ForEach_itr + 44];
    } else {
      /* RelationalOperator: '<S67>/UpperRelop' incorporates:
       *  Switch: '<S67>/Switch'
       */
      cphi = multirotor0_B.MotorMatrix_real[ForEach_itr + 40];

      /* Switch: '<S67>/Switch' incorporates:
       *  RelationalOperator: '<S67>/UpperRelop'
       */
      if (!(multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE < cphi)) {
        cphi = multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE;
      }
    }

    /* End of Switch: '<S67>/Switch2' */
    if (rtmIsMajorTimeStep((&multirotor0_M))) {
      /* Product: '<S59>/Product' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       *  ForEachSliceSelector generated from: '<S57>/RPM_commands'
       *  RateTransition: '<S1>/Rate Transition1'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].Product =
        multirotor0_B.MotorMatrix_real[ForEach_itr + 16] *
        multirotor0_B.RateTransition1[ForEach_itr];
    }

    /* Product: '<S59>/Divide' incorporates:
     *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
     *  Sum: '<S59>/Sum1'
     */
    ctheta = (multirotor0_B.CoreSubsys[ForEach_itr].Product - cphi) /
      multirotor0_B.MotorMatrix_real[ForEach_itr + 20];

    /* Switch: '<S63>/Switch' incorporates:
     *  Constant: '<S65>/Constant'
     *  Constant: '<S66>/Constant'
     *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
     *  Integrator: '<S63>/Integrator'
     *  Logic: '<S63>/Logical Operator'
     *  Logic: '<S63>/Logical Operator1'
     *  Logic: '<S63>/Logical Operator2'
     *  RelationalOperator: '<S63>/Relational Operator'
     *  RelationalOperator: '<S63>/Relational Operator1'
     *  RelationalOperator: '<S65>/Compare'
     *  RelationalOperator: '<S66>/Compare'
     */
    if (((multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE <= q1) ||
         (ctheta < 0.0)) && ((ctheta > 0.0) ||
         (multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE >=
          multirotor0_B.MotorMatrix_real[ForEach_itr + 40]))) {
      /* Switch: '<S63>/Switch' */
      multirotor0_B.CoreSubsys[ForEach_itr].Switch = ctheta;
    } else {
      /* Switch: '<S63>/Switch' incorporates:
       *  Constant: '<S63>/Constant'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].Switch = 0.0;
    }

    /* End of Switch: '<S63>/Switch' */

    /* Switch: '<S64>/Switch' */
    multirotor0_B.CoreSubsys[ForEach_itr].Switch_a = 0.0;
    if (rtmIsMajorTimeStep((&multirotor0_M))) {
      /* Gain: '<S61>/Conversion deg to rad' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       */
      phi = 0.017453292519943295 * multirotor0_B.MotorMatrix_real[ForEach_itr];

      /* Abs: '<S61>/Abs' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       */
      ctheta = std::abs(multirotor0_B.MotorMatrix_real[ForEach_itr + 4]);

      /* Sum: '<S61>/Subtract' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       *  Product: '<S61>/Product4'
       *  Reshape: '<S61>/Reshape'
       *  Trigonometry: '<S61>/Trigonometric Function'
       *  Trigonometry: '<S61>/Trigonometric Function1'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[0] =
        std::cos(phi) * ctheta - multirotor0_B.COM_system[0];
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1] =
        std::sin(phi) * ctheta - multirotor0_B.COM_system[1];
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[2] =
        multirotor0_B.MotorMatrix_real[ForEach_itr + 8] -
        multirotor0_B.COM_system[2];

      /* Gain: '<S74>/Conversion deg to rad' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       */
      rtb_Sum_a[0] = multirotor0_B.MotorMatrix_real[ForEach_itr + 48] *
        0.017453292519943295;
      rtb_Sum_a[1] = multirotor0_B.MotorMatrix_real[ForEach_itr + 52] *
        0.017453292519943295;
      rtb_Sum_a[2] = multirotor0_B.MotorMatrix_real[ForEach_itr + 56] *
        0.017453292519943295;

      /* Trigonometry: '<S105>/Trigonometric Function3' incorporates:
       *  Trigonometry: '<S108>/Trigonometric Function3'
       *  Trigonometry: '<S109>/Trigonometric Function'
       *  Trigonometry: '<S111>/Trigonometric Function4'
       *  Trigonometry: '<S112>/Trigonometric Function'
       */
      phi = std::cos(rtb_Sum_a[2]);

      /* Trigonometry: '<S105>/Trigonometric Function1' incorporates:
       *  Trigonometry: '<S106>/Trigonometric Function1'
       *  Trigonometry: '<S110>/Trigonometric Function1'
       *  Trigonometry: '<S113>/Trigonometric Function1'
       */
      ctheta = std::cos(rtb_Sum_a[1]);

      /* Product: '<S105>/Product' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       *  Trigonometry: '<S105>/Trigonometric Function1'
       *  Trigonometry: '<S105>/Trigonometric Function3'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[0] = ctheta * phi;

      /* Trigonometry: '<S108>/Trigonometric Function5' incorporates:
       *  Trigonometry: '<S109>/Trigonometric Function5'
       *  Trigonometry: '<S111>/Trigonometric Function12'
       *  Trigonometry: '<S113>/Trigonometric Function3'
       */
      theta = std::cos(rtb_Sum_a[0]);

      /* Trigonometry: '<S108>/Trigonometric Function1' incorporates:
       *  Trigonometry: '<S107>/Trigonometric Function1'
       *  Trigonometry: '<S111>/Trigonometric Function2'
       */
      COM_system_inter_idx_1 = std::sin(rtb_Sum_a[1]);

      /* Trigonometry: '<S108>/Trigonometric Function12' incorporates:
       *  Trigonometry: '<S110>/Trigonometric Function3'
       *  Trigonometry: '<S111>/Trigonometric Function5'
       *  Trigonometry: '<S112>/Trigonometric Function5'
       */
      COM_system_inter_idx_2 = std::sin(rtb_Sum_a[0]);

      /* Trigonometry: '<S108>/Trigonometric Function' incorporates:
       *  Trigonometry: '<S106>/Trigonometric Function3'
       *  Trigonometry: '<S109>/Trigonometric Function4'
       *  Trigonometry: '<S111>/Trigonometric Function'
       *  Trigonometry: '<S112>/Trigonometric Function3'
       */
      q1 = std::sin(rtb_Sum_a[2]);

      /* Product: '<S108>/Product' incorporates:
       *  Product: '<S109>/Product1'
       *  Trigonometry: '<S108>/Trigonometric Function1'
       *  Trigonometry: '<S108>/Trigonometric Function12'
       */
      COM_system_inter_idx_0 = COM_system_inter_idx_2 * COM_system_inter_idx_1;

      /* Sum: '<S108>/Sum' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       *  Product: '<S108>/Product'
       *  Product: '<S108>/Product1'
       *  Trigonometry: '<S108>/Trigonometric Function'
       *  Trigonometry: '<S108>/Trigonometric Function5'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[1] =
        COM_system_inter_idx_0 * phi - theta * q1;

      /* Product: '<S111>/Product1' incorporates:
       *  Product: '<S112>/Product'
       */
      COM_propeller_0 = theta * COM_system_inter_idx_1;

      /* Sum: '<S111>/Sum' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       *  Product: '<S111>/Product1'
       *  Product: '<S111>/Product2'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[2] =
        COM_propeller_0 * phi + COM_system_inter_idx_2 * q1;

      /* Product: '<S106>/Product' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[3] = ctheta * q1;

      /* Sum: '<S109>/Sum' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       *  Product: '<S109>/Product1'
       *  Product: '<S109>/Product2'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[4] =
        COM_system_inter_idx_0 * q1 + theta * phi;

      /* Sum: '<S112>/Sum' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       *  Product: '<S112>/Product'
       *  Product: '<S112>/Product1'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[5] =
        COM_propeller_0 * q1 - COM_system_inter_idx_2 * phi;

      /* Gain: '<S107>/Gain' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[6] =
        -COM_system_inter_idx_1;

      /* Product: '<S110>/Product' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[7] =
        COM_system_inter_idx_2 * ctheta;

      /* Product: '<S113>/Product' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[8] = theta *
        ctheta;
    }

    /* Sum: '<S57>/Sum1' incorporates:
     *  Integrator: '<S2>/omega'
     *  Product: '<S71>/Product'
     *  Product: '<S71>/Product1'
     *  Product: '<S71>/Product2'
     *  Product: '<S72>/Product'
     *  Product: '<S72>/Product1'
     *  Product: '<S72>/Product2'
     *  Sum: '<S58>/Sum1'
     *  Sum: '<S60>/Sum'
     */
    COM_system_inter_idx_0 = (multirotor0_X.omega_CSTATE[1] *
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[2] -
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1] *
      multirotor0_X.omega_CSTATE[2]) + rtb_TrueairspeedBodyaxes[0];
    COM_system_inter_idx_2 = (multirotor0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[0] * multirotor0_X.omega_CSTATE[2] -
      multirotor0_X.omega_CSTATE[0] * multirotor0_B.CoreSubsys[ForEach_itr].
      VectorfromrealCoGtopropellerBod[2]) + rtb_TrueairspeedBodyaxes[1];
    COM_system_inter_idx_1 = (multirotor0_X.omega_CSTATE[0] *
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1] -
      multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[0] *
      multirotor0_X.omega_CSTATE[1]) + rtb_TrueairspeedBodyaxes[2];

    /* Product: '<S74>/Product' */
    for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
      /* Product: '<S74>/Product' incorporates:
       *  Concatenate: '<S114>/Vector Concatenate'
       */
      rtb_Sum_a[xpageoffset] = (multirotor0_B.CoreSubsys[ForEach_itr].
        VectorConcatenate[xpageoffset + 3] * COM_system_inter_idx_2 +
        multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[xpageoffset] *
        COM_system_inter_idx_0) + multirotor0_B.CoreSubsys[ForEach_itr].
        VectorConcatenate[xpageoffset + 6] * COM_system_inter_idx_1;
    }

    /* End of Product: '<S74>/Product' */

    /* Gain: '<S89>/Gain' */
    multirotor0_B.CoreSubsys[ForEach_itr].Climbspeedv_c = -rtb_Sum_a[2];
    if (rtmIsMajorTimeStep((&multirotor0_M))) {
      /* Outputs for IfAction SubSystem: '<S90>/Vortex ring state -2 <= vc//vh < 0 ' incorporates:
       *  ActionPort: '<S98>/Action Port'
       */
      /* If: '<S90>/If' incorporates:
       *  Constant: '<S77>/Induced velocity at hover'
       *  Product: '<S90>/Divide'
       *  Product: '<S98>/Divide'
       */
      q1 = multirotor0_B.CoreSubsys[ForEach_itr].Climbspeedv_c / 4.0;

      /* End of Outputs for SubSystem: '<S90>/Vortex ring state -2 <= vc//vh < 0 ' */
      if (rtsiIsModeUpdateTimeStep(&(&multirotor0_M)->solverInfo)) {
        if (q1 >= 0.0) {
          rtAction = 0;
        } else if (q1 >= -2.0) {
          rtAction = 1;
        } else {
          rtAction = 2;
        }

        multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem = rtAction;
      } else {
        rtAction = multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem;
      }

      switch (rtAction) {
       case 0:
        /* Outputs for IfAction SubSystem: '<S90>/Normal working state vc//vh >= 0' incorporates:
         *  ActionPort: '<S97>/Action Port'
         */
        /* Gain: '<S97>/Gain' */
        phi = 0.5 * multirotor0_B.CoreSubsys[ForEach_itr].Climbspeedv_c;

        /* Merge: '<S90>/Merge' incorporates:
         *  Product: '<S97>/Product'
         *  Sqrt: '<S97>/Sqrt'
         *  Sum: '<S97>/Sum'
         *  Sum: '<S97>/Sum1'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Merge = std::sqrt(phi * phi + 16.0)
          - phi;

        /* End of Outputs for SubSystem: '<S90>/Normal working state vc//vh >= 0' */
        break;

       case 1:
        /* Outputs for IfAction SubSystem: '<S90>/Vortex ring state -2 <= vc//vh < 0 ' incorporates:
         *  ActionPort: '<S98>/Action Port'
         */
        /* Product: '<S98>/Product' */
        ctheta = q1 * q1;

        /* Gain: '<S98>/Gain1' */
        theta = -1.372 * ctheta;

        /* Product: '<S98>/Product1' */
        ctheta *= q1;

        /* Merge: '<S90>/Merge' incorporates:
         *  Constant: '<S98>/Constant'
         *  Constant: '<S98>/Induced velocity at hover'
         *  Gain: '<S98>/Gain'
         *  Gain: '<S98>/Gain2'
         *  Gain: '<S98>/Gain3'
         *  Product: '<S98>/Product2'
         *  Product: '<S98>/Product3'
         *  Sum: '<S98>/Add'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Merge = ((((-1.125 * q1 + 1.0) +
          theta) + -1.718 * ctheta) + ctheta * q1 * -0.655) * 4.0;

        /* End of Outputs for SubSystem: '<S90>/Vortex ring state -2 <= vc//vh < 0 ' */
        break;

       default:
        /* Outputs for IfAction SubSystem: '<S90>/Windmill braking state vc//vh < -2' incorporates:
         *  ActionPort: '<S99>/Action Port'
         */
        /* Gain: '<S99>/Gain' */
        phi = 0.5 * multirotor0_B.CoreSubsys[ForEach_itr].Climbspeedv_c;

        /* Merge: '<S90>/Merge' incorporates:
         *  Product: '<S99>/Product'
         *  Sqrt: '<S99>/Sqrt'
         *  Sum: '<S99>/Sum'
         *  Sum: '<S99>/Sum1'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Merge = (0.0 - phi) - std::sqrt
          (phi * phi - 16.0);

        /* End of Outputs for SubSystem: '<S90>/Windmill braking state vc//vh < -2' */
        break;
      }

      /* End of If: '<S90>/If' */
    }

    /* Outputs for IfAction SubSystem: '<S76>/Nonzero airspeed in rotor plane' incorporates:
     *  ActionPort: '<S82>/Action Port'
     */
    /* Outputs for IfAction SubSystem: '<S88>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S91>/Action Port'
     */
    /* If: '<S76>/If' incorporates:
     *  If: '<S88>/If'
     *  Math: '<S96>/transpose'
     *  Product: '<S74>/Product'
     *  Product: '<S86>/Product'
     *  Product: '<S87>/Product'
     *  Product: '<S95>/Product'
     *  Product: '<S96>/Product'
     */
    theta = rtb_Sum_a[0] * rtb_Sum_a[0] + rtb_Sum_a[1] * rtb_Sum_a[1];

    /* End of Outputs for SubSystem: '<S88>/Nonzero airspeed' */
    /* End of Outputs for SubSystem: '<S76>/Nonzero airspeed in rotor plane' */

    /* Sqrt: '<S93>/Sqrt' incorporates:
     *  Math: '<S96>/transpose'
     *  Product: '<S74>/Product'
     *  Product: '<S96>/Product'
     */
    ctheta = std::sqrt(rtb_Sum_a[2] * rtb_Sum_a[2] + theta);

    /* If: '<S88>/If' */
    if (rtsiIsModeUpdateTimeStep(&(&multirotor0_M)->solverInfo)) {
      rtAction = static_cast<int8_T>(!(ctheta == 0.0));
      multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_l = rtAction;
    } else {
      rtAction = multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_l;
    }

    if (rtAction == 0) {
      /* Outputs for IfAction SubSystem: '<S88>/Zero airspeed' incorporates:
       *  ActionPort: '<S92>/Action Port'
       */
      if (rtmIsMajorTimeStep((&multirotor0_M))) {
        /* Merge: '<S88>/Merge' incorporates:
         *  Constant: '<S92>/Constant'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Angleofattackrad = 0.0;
      }

      /* End of Outputs for SubSystem: '<S88>/Zero airspeed' */
    } else {
      /* Outputs for IfAction SubSystem: '<S88>/Nonzero airspeed' incorporates:
       *  ActionPort: '<S91>/Action Port'
       */
      /* Merge: '<S88>/Merge' incorporates:
       *  Product: '<S91>/Divide1'
       *  Sqrt: '<S94>/Sqrt'
       *  Trigonometry: '<S91>/Trigonometric Function'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].Angleofattackrad = std::atan(1.0 /
        std::sqrt(theta) * rtb_Sum_a[2]);

      /* End of Outputs for SubSystem: '<S88>/Nonzero airspeed' */
    }

    /* Product: '<S77>/Divide' incorporates:
     *  Constant: '<S77>/Induced velocity at hover'
     *  Product: '<S77>/Product2'
     *  Sum: '<S77>/Sum2'
     *  Trigonometry: '<S77>/Trigonometric Function'
     */
    phi = 4.0 / (multirotor0_B.CoreSubsys[ForEach_itr].Merge - std::sin
                 (multirotor0_B.CoreSubsys[ForEach_itr].Angleofattackrad) *
                 ctheta);

    /* Product: '<S80>/Product5' incorporates:
     *  Product: '<S79>/Product1'
     */
    q1 = cphi * cphi;

    /* Product: '<S73>/Product7' incorporates:
     *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
     *  Product: '<S80>/Product4'
     *  Product: '<S80>/Product5'
     *  Product: '<S80>/Product6'
     *  Sum: '<S80>/Sum1'
     *  Switch: '<S77>/Switch'
     */
    ctheta = (multirotor0_B.MotorMatrix_real[ForEach_itr + 24] * cphi +
              multirotor0_B.MotorMatrix_real[ForEach_itr + 28] * q1) * phi;

    /* If: '<S76>/If' incorporates:
     *  Sqrt: '<S84>/Sqrt'
     */
    if (rtsiIsModeUpdateTimeStep(&(&multirotor0_M)->solverInfo)) {
      rtAction = static_cast<int8_T>(!(std::sqrt(theta) == 0.0));
      multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_e = rtAction;
    } else {
      rtAction = multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_e;
    }

    if (rtAction == 0) {
      /* Outputs for IfAction SubSystem: '<S76>/Zero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S83>/Action Port'
       */
      if (rtmIsMajorTimeStep((&multirotor0_M))) {
        /* Merge: '<S76>/Merge' incorporates:
         *  Constant: '<S83>/Constant'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0]
          = 0.0;

        /* Merge: '<S76>/Merge1' incorporates:
         *  Constant: '<S83>/Constant1'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[0]
          = 0.0;

        /* Merge: '<S76>/Merge' incorporates:
         *  Constant: '<S83>/Constant'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1]
          = 0.0;

        /* Merge: '<S76>/Merge1' incorporates:
         *  Constant: '<S83>/Constant1'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[1]
          = 0.0;

        /* Merge: '<S76>/Merge' incorporates:
         *  Constant: '<S83>/Constant'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2]
          = -1.0;

        /* Merge: '<S76>/Merge1' incorporates:
         *  Constant: '<S83>/Constant1'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[2]
          = 0.0;
      }

      /* End of Outputs for SubSystem: '<S76>/Zero airspeed in rotor plane' */
    } else {
      /* Outputs for IfAction SubSystem: '<S76>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S82>/Action Port'
       */
      /* Sqrt: '<S85>/Sqrt' */
      theta = std::sqrt(theta);

      /* Gain: '<S82>/Conversion deg to rad' incorporates:
       *  Product: '<S82>/Product4'
       */
      COM_system_inter_idx_1 = theta * 0.375 * 0.017453292519943295;

      /* Trigonometry: '<S82>/Trigonometric Function' */
      COM_system_inter_idx_2 = std::sin(COM_system_inter_idx_1);

      /* Product: '<S82>/Divide' */
      COM_propeller_0 = rtb_Sum_a[0] / theta;
      rtb_Airspeeddirectionintherot_0 = COM_propeller_0;

      /* Product: '<S82>/Product2' incorporates:
       *  Gain: '<S82>/Gain'
       *  Product: '<S82>/Divide'
       *  Product: '<S82>/Product'
       */
      COM_system_inter_idx_0 = -COM_propeller_0 * COM_system_inter_idx_2;

      /* Product: '<S82>/Divide' */
      COM_propeller_0 = rtb_Sum_a[1] / theta;

      /* Product: '<S82>/Product2' incorporates:
       *  Gain: '<S82>/Gain'
       *  Product: '<S82>/Divide'
       *  Product: '<S82>/Product'
       */
      COM_system_inter_idx_2 *= -COM_propeller_0;

      /* Gain: '<S82>/Gain1' incorporates:
       *  Trigonometry: '<S82>/Trigonometric Function1'
       */
      theta = -std::cos(COM_system_inter_idx_1);

      /* Product: '<S82>/Product3' incorporates:
       *  Constant: '<S82>/Constant'
       *  Constant: '<S82>/Constant1'
       *  Gain: '<S82>/Gain2'
       *  Product: '<S82>/Divide'
       *  Product: '<S82>/Product1'
       */
      COM_propeller_0 = -COM_propeller_0 * 0.23 * COM_system_inter_idx_1;
      rtb_Airspeeddirectionintherot_0 = rtb_Airspeeddirectionintherot_0 * 0.23 *
        COM_system_inter_idx_1;
      COM_system_inter_idx_1 *= 0.0;
      for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
        /* Product: '<S82>/Product2' incorporates:
         *  Concatenate: '<S114>/Vector Concatenate'
         */
        COM_propeller_1 = multirotor0_B.CoreSubsys[ForEach_itr].
          VectorConcatenate[3 * xpageoffset];
        COM_propeller_2 = multirotor0_B.CoreSubsys[ForEach_itr].
          VectorConcatenate[3 * xpageoffset + 1];
        COM_propeller_3 = multirotor0_B.CoreSubsys[ForEach_itr].
          VectorConcatenate[3 * xpageoffset + 2];

        /* Merge: '<S76>/Merge' incorporates:
         *  Product: '<S82>/Product2'
         *  Reshape: '<S82>/Reshape1'
         */
        multirotor0_B.CoreSubsys[ForEach_itr]
          .NewtiltedthrustdirectionBodyaxe[xpageoffset] = (COM_propeller_2 *
          COM_system_inter_idx_2 + COM_propeller_1 * COM_system_inter_idx_0) +
          COM_propeller_3 * theta;

        /* Merge: '<S76>/Merge1' incorporates:
         *  Product: '<S82>/Product3'
         */
        multirotor0_B.CoreSubsys[ForEach_itr]
          .Momentinthemotorhubduetobending[xpageoffset] = (COM_propeller_2 *
          rtb_Airspeeddirectionintherot_0 + COM_propeller_1 * COM_propeller_0) +
          COM_propeller_3 * COM_system_inter_idx_1;
      }

      /* End of Outputs for SubSystem: '<S76>/Nonzero airspeed in rotor plane' */
    }

    /* Product: '<S73>/Product9' incorporates:
     *  Merge: '<S76>/Merge'
     */
    COM_system_inter_idx_0 = ctheta * multirotor0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[0];
    COM_system_inter_idx_1 = ctheta * multirotor0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[1];
    COM_system_inter_idx_2 = ctheta * multirotor0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[2];
    if (rtmIsMajorTimeStep((&multirotor0_M))) {
      for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
        /* Product: '<S81>/Product9' incorporates:
         *  Concatenate: '<S114>/Vector Concatenate'
         *  Constant: '<S81>/Constant'
         *  Math: '<S81>/Math Function'
         */
        multirotor0_B.CoreSubsys[ForEach_itr].Product9[xpageoffset] =
          (multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[3 *
           xpageoffset + 1] * 0.0 + multirotor0_B.CoreSubsys[ForEach_itr].
           VectorConcatenate[3 * xpageoffset] * 0.0) -
          multirotor0_B.CoreSubsys[ForEach_itr].VectorConcatenate[3 *
          xpageoffset + 2];
      }

      /* Gain: '<S101>/Gain' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       */
      ctheta = multirotor0_B.MotorMatrix_real[ForEach_itr + 60] * 0.5;

      /* Gain: '<S101>/Gain1' incorporates:
       *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
       *  Product: '<S101>/Product7'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].Gain1 = ctheta * ctheta *
        multirotor0_B.MotorMatrix_real[ForEach_itr + 64] * 0.58333333333333337;
    }

    /* Gain: '<S78>/Conversion rpm to rad//s' */
    ctheta = 0.10471975511965977 * cphi;

    /* ForEachSliceAssignment generated from: '<S57>/RPM_motor' */
    rtb_ImpAsg_InsertedFor_RPM_moto[ForEach_itr] = cphi;

    /* ForEachSliceSelector generated from: '<S57>/MotorMatrix_real' incorporates:
     *  Product: '<S73>/Product3'
     *  Product: '<S78>/Product5'
     */
    COM_propeller_0 = multirotor0_B.MotorMatrix_real[ForEach_itr + 12];

    /* Product: '<S73>/Product3' incorporates:
     *  ForEachSliceSelector generated from: '<S57>/MotorMatrix_real'
     *  Product: '<S79>/Product'
     *  Product: '<S79>/Product1'
     *  Sum: '<S79>/Sum'
     */
    theta = (multirotor0_B.MotorMatrix_real[ForEach_itr + 32] * cphi +
             multirotor0_B.MotorMatrix_real[ForEach_itr + 36] * q1) *
      COM_propeller_0;

    /* Product: '<S78>/Product5' */
    q1 = COM_propeller_0 * multirotor0_B.CoreSubsys[ForEach_itr].Gain1;

    /* ForEachSliceAssignment generated from: '<S57>/Motor_moment' incorporates:
     *  Integrator: '<S2>/omega'
     *  Merge: '<S76>/Merge1'
     *  Product: '<S102>/Product'
     *  Product: '<S103>/Product'
     *  Product: '<S115>/Product'
     *  Product: '<S116>/Product'
     *  Product: '<S73>/Product3'
     *  Product: '<S73>/Product8'
     *  Product: '<S78>/Product5'
     *  Product: '<S81>/Product9'
     *  Sum: '<S100>/Sum'
     *  Sum: '<S62>/Add'
     *  Sum: '<S75>/Sum'
     *  Switch: '<S77>/Switch'
     */
    rtb_ImpAsg_InsertedFor_Motor_mo[3 * ForEach_itr] =
      ((multirotor0_X.omega_CSTATE[1] * multirotor0_B.CoreSubsys[ForEach_itr].
        NewtiltedthrustdirectionBodyaxe[2] -
        multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1]
        * multirotor0_X.omega_CSTATE[2]) * q1 * ctheta + (theta *
        multirotor0_B.CoreSubsys[ForEach_itr].Product9[0] * phi +
        multirotor0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[0]))
      + (multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1]
         * COM_system_inter_idx_2 - COM_system_inter_idx_1 *
         multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[2]);

    /* ForEachSliceAssignment generated from: '<S57>/Motor_force' incorporates:
     *  Product: '<S73>/Product9'
     */
    rtb_ImpAsg_InsertedFor_Motor_fo[3 * ForEach_itr] = COM_system_inter_idx_0;

    /* ForEachSliceAssignment generated from: '<S57>/Motor_moment' incorporates:
     *  ForEachSliceAssignment generated from: '<S57>/Motor_force'
     *  Integrator: '<S2>/omega'
     *  Merge: '<S76>/Merge1'
     *  Product: '<S102>/Product1'
     *  Product: '<S103>/Product1'
     *  Product: '<S115>/Product1'
     *  Product: '<S116>/Product1'
     *  Product: '<S73>/Product3'
     *  Product: '<S73>/Product8'
     *  Product: '<S78>/Product5'
     *  Product: '<S81>/Product9'
     *  Sum: '<S100>/Sum'
     *  Sum: '<S62>/Add'
     *  Sum: '<S75>/Sum'
     *  Switch: '<S77>/Switch'
     */
    i = 3 * ForEach_itr + 1;
    rtb_ImpAsg_InsertedFor_Motor_mo[i] = ((multirotor0_B.CoreSubsys[ForEach_itr]
      .NewtiltedthrustdirectionBodyaxe[0] * multirotor0_X.omega_CSTATE[2] -
      multirotor0_X.omega_CSTATE[0] * multirotor0_B.CoreSubsys[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[2]) * q1 * ctheta + (theta *
      multirotor0_B.CoreSubsys[ForEach_itr].Product9[1] * phi +
      multirotor0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[1]))
      + (COM_system_inter_idx_0 * multirotor0_B.CoreSubsys[ForEach_itr].
         VectorfromrealCoGtopropellerBod[2] -
         multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[0]
         * COM_system_inter_idx_2);

    /* ForEachSliceAssignment generated from: '<S57>/Motor_force' incorporates:
     *  Product: '<S73>/Product9'
     */
    rtb_ImpAsg_InsertedFor_Motor_fo[i] = COM_system_inter_idx_1;

    /* ForEachSliceAssignment generated from: '<S57>/Motor_moment' incorporates:
     *  ForEachSliceAssignment generated from: '<S57>/Motor_force'
     *  Integrator: '<S2>/omega'
     *  Merge: '<S76>/Merge1'
     *  Product: '<S102>/Product2'
     *  Product: '<S103>/Product2'
     *  Product: '<S115>/Product2'
     *  Product: '<S116>/Product2'
     *  Product: '<S73>/Product3'
     *  Product: '<S73>/Product8'
     *  Product: '<S78>/Product5'
     *  Product: '<S81>/Product9'
     *  Sum: '<S100>/Sum'
     *  Sum: '<S62>/Add'
     *  Sum: '<S75>/Sum'
     *  Switch: '<S77>/Switch'
     */
    i = 3 * ForEach_itr + 2;
    rtb_ImpAsg_InsertedFor_Motor_mo[i] = ((multirotor0_X.omega_CSTATE[0] *
      multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1] -
      multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0] *
      multirotor0_X.omega_CSTATE[1]) * q1 * ctheta + (theta *
      multirotor0_B.CoreSubsys[ForEach_itr].Product9[2] * phi +
      multirotor0_B.CoreSubsys[ForEach_itr].Momentinthemotorhubduetobending[2]))
      + (multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[0]
         * COM_system_inter_idx_1 - COM_system_inter_idx_0 *
         multirotor0_B.CoreSubsys[ForEach_itr].VectorfromrealCoGtopropellerBod[1]);

    /* ForEachSliceAssignment generated from: '<S57>/Motor_force' incorporates:
     *  Product: '<S73>/Product9'
     */
    rtb_ImpAsg_InsertedFor_Motor_fo[i] = COM_system_inter_idx_2;
  }

  /* End of Outputs for SubSystem: '<S39>/For Each Subsystem' */

  /* Sum: '<S39>/Sum of Elements' incorporates:
   *  ForEachSliceAssignment generated from: '<S57>/Motor_force'
   *  Sum: '<S16>/Sum'
   */
  for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
    rtb_Sum_a[xpageoffset] = ((rtb_ImpAsg_InsertedFor_Motor_fo[xpageoffset + 3]
      + rtb_ImpAsg_InsertedFor_Motor_fo[xpageoffset]) +
      rtb_ImpAsg_InsertedFor_Motor_fo[xpageoffset + 6]) +
      rtb_ImpAsg_InsertedFor_Motor_fo[xpageoffset + 9];
  }

  /* End of Sum: '<S39>/Sum of Elements' */
  if (rtmIsMajorTimeStep((&multirotor0_M))) {
    /* Product: '<S38>/Product1' incorporates:
     *  Constant: '<S38>/Gravity (Inertial axes)'
     */
    multirotor0_B.ForceofgravityInertialaxes[0] = 0.0 * multirotor0_B.total_mass;
    multirotor0_B.ForceofgravityInertialaxes[1] = 0.0 * multirotor0_B.total_mass;
    multirotor0_B.ForceofgravityInertialaxes[2] = 9.80665 *
      multirotor0_B.total_mass;
  }

  /* Sum: '<S41>/Sum1' incorporates:
   *  Integrator: '<S2>/V_b'
   *  Product: '<S56>/Product'
   */
  rtb_TrueairspeedBodyaxes_b[0] = multirotor0_X.V_b_CSTATE[0] -
    rtb_TrueairspeedBodyaxes_b[0];
  rtb_TrueairspeedBodyaxes_b[1] = multirotor0_X.V_b_CSTATE[1] -
    rtb_TrueairspeedBodyaxes_b[1];
  rtb_TrueairspeedBodyaxes_b[2] = multirotor0_X.V_b_CSTATE[2] -
    rtb_TrueairspeedBodyaxes_b[2];

  /* If: '<S40>/If' incorporates:
   *  Math: '<S55>/transpose'
   *  Product: '<S55>/Product'
   *  Sqrt: '<S44>/Sqrt'
   *  Sum: '<S41>/Sum1'
   */
  if (rtsiIsModeUpdateTimeStep(&(&multirotor0_M)->solverInfo)) {
    rtAction = static_cast<int8_T>(!(std::sqrt((rtb_TrueairspeedBodyaxes_b[0] *
      rtb_TrueairspeedBodyaxes_b[0] + rtb_TrueairspeedBodyaxes_b[1] *
      rtb_TrueairspeedBodyaxes_b[1]) + rtb_TrueairspeedBodyaxes_b[2] *
      rtb_TrueairspeedBodyaxes_b[2]) == 0.0));
    multirotor0_DW.If_ActiveSubsystem = rtAction;
  } else {
    rtAction = multirotor0_DW.If_ActiveSubsystem;
  }

  if (rtAction == 0) {
    /* Outputs for IfAction SubSystem: '<S40>/Zero airspeed' incorporates:
     *  ActionPort: '<S43>/Action Port'
     */
    if (rtmIsMajorTimeStep((&multirotor0_M))) {
      /* Merge: '<S40>/Merge' incorporates:
       *  Constant: '<S43>/Constant'
       */
      multirotor0_B.Forceagainstdirectionofmotiondu[0] = 0.0;
      multirotor0_B.Forceagainstdirectionofmotiondu[1] = 0.0;
      multirotor0_B.Forceagainstdirectionofmotiondu[2] = 0.0;
    }

    /* End of Outputs for SubSystem: '<S40>/Zero airspeed' */
  } else {
    /* Outputs for IfAction SubSystem: '<S40>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S42>/Action Port'
     */
    /* Product: '<S50>/Divide' */
    cphi = rtb_TrueairspeedBodyaxes_b[0] / multirotor0_B.Surface_params[0];

    /* Product: '<S50>/Product' */
    phi = cphi * cphi;

    /* Product: '<S50>/Divide1' */
    cphi = rtb_TrueairspeedBodyaxes_b[1] / multirotor0_B.Surface_params[1];

    /* Product: '<S50>/Product1' */
    ctheta = cphi * cphi;

    /* Product: '<S50>/Divide2' */
    cphi = rtb_TrueairspeedBodyaxes_b[2] / multirotor0_B.Surface_params[2];

    /* Sum: '<S50>/Add' incorporates:
     *  Product: '<S50>/Product2'
     */
    cphi = (phi + ctheta) + cphi * cphi;

    /* Sqrt: '<S50>/Reciprocal Sqrt' */
    if (cphi > 0.0) {
      if (std::isinf(cphi)) {
        cphi = 0.0;
      } else {
        cphi = 1.0 / std::sqrt(cphi);
      }
    } else if (cphi == 0.0) {
      cphi = (rtInf);
    } else {
      cphi = (rtNaN);
    }

    /* End of Sqrt: '<S50>/Reciprocal Sqrt' */

    /* Product: '<S51>/Product' incorporates:
     *  Sum: '<S41>/Sum1'
     */
    phi = rtb_TrueairspeedBodyaxes_b[0] * cphi;

    /* Product: '<S53>/Product' incorporates:
     *  Math: '<S53>/transpose'
     *  Product: '<S51>/Product'
     */
    ctheta = phi * phi;

    /* Product: '<S51>/Product' incorporates:
     *  Sum: '<S41>/Sum1'
     */
    phi = rtb_TrueairspeedBodyaxes_b[1] * cphi;

    /* Product: '<S53>/Product' incorporates:
     *  Math: '<S53>/transpose'
     *  Product: '<S51>/Product'
     */
    ctheta += phi * phi;

    /* Product: '<S51>/Product' incorporates:
     *  Sum: '<S41>/Sum1'
     */
    phi = rtb_TrueairspeedBodyaxes_b[2] * cphi;

    /* Product: '<S54>/Product' incorporates:
     *  Product: '<S49>/Product'
     *  Product: '<S51>/Product'
     *  Sum: '<S41>/Sum1'
     */
    q1 = (rtb_TrueairspeedBodyaxes_b[0] * rtb_TrueairspeedBodyaxes_b[0] +
          rtb_TrueairspeedBodyaxes_b[1] * rtb_TrueairspeedBodyaxes_b[1]) +
      rtb_TrueairspeedBodyaxes_b[2] * rtb_TrueairspeedBodyaxes_b[2];

    /* Abs: '<S42>/Abs' incorporates:
     *  Constant: '<S42>/Constant'
     *  Constant: '<S42>/Constant1'
     *  Constant: '<S42>/Constant2'
     *  Math: '<S53>/transpose'
     *  Product: '<S42>/Product'
     *  Product: '<S51>/Product'
     *  Product: '<S53>/Product'
     *  Product: '<S54>/Product'
     *  Sqrt: '<S52>/Sqrt'
     */
    phi = std::abs(q1 * 0.6125 * 0.4 * std::sqrt(phi * phi + ctheta));

    /* Sqrt: '<S48>/Sqrt' */
    cphi = std::sqrt(q1);

    /* Merge: '<S40>/Merge' incorporates:
     *  Gain: '<S42>/Drag force opposes direction of airspeed'
     *  Product: '<S42>/Product1'
     *  Product: '<S45>/Divide'
     *  Sum: '<S41>/Sum1'
     */
    multirotor0_B.Forceagainstdirectionofmotiondu[0] =
      -(rtb_TrueairspeedBodyaxes_b[0] / cphi * phi);
    multirotor0_B.Forceagainstdirectionofmotiondu[1] =
      -(rtb_TrueairspeedBodyaxes_b[1] / cphi * phi);
    multirotor0_B.Forceagainstdirectionofmotiondu[2] =
      -(rtb_TrueairspeedBodyaxes_b[2] / cphi * phi);

    /* End of Outputs for SubSystem: '<S40>/Nonzero airspeed' */
  }

  /* End of If: '<S40>/If' */

  /* Sum: '<S11>/Sum' incorporates:
   *  Integrator: '<S2>/V_b'
   *  Integrator: '<S2>/omega'
   *  Product: '<S34>/Product'
   *  Product: '<S34>/Product1'
   *  Product: '<S34>/Product2'
   *  Product: '<S35>/Product'
   *  Product: '<S35>/Product1'
   *  Product: '<S35>/Product2'
   */
  rtb_TrueairspeedBodyaxes[0] = multirotor0_X.omega_CSTATE[1] *
    multirotor0_X.V_b_CSTATE[2];
  rtb_TrueairspeedBodyaxes[1] = multirotor0_X.V_b_CSTATE[0] *
    multirotor0_X.omega_CSTATE[2];
  rtb_TrueairspeedBodyaxes[2] = multirotor0_X.omega_CSTATE[0] *
    multirotor0_X.V_b_CSTATE[1];
  rtb_TrueairspeedBodyaxes_m[0] = multirotor0_X.V_b_CSTATE[1] *
    multirotor0_X.omega_CSTATE[2];
  rtb_TrueairspeedBodyaxes_m[1] = multirotor0_X.omega_CSTATE[0] *
    multirotor0_X.V_b_CSTATE[2];
  rtb_TrueairspeedBodyaxes_m[2] = multirotor0_X.V_b_CSTATE[0] *
    multirotor0_X.omega_CSTATE[1];

  /* Product: '<S38>/Product' */
  cphi = multirotor0_B.ForceofgravityInertialaxes[1];
  phi = multirotor0_B.ForceofgravityInertialaxes[0];
  ctheta = multirotor0_B.ForceofgravityInertialaxes[2];

  /* Integrator: '<S2>/omega' incorporates:
   *  Product: '<S13>/Product'
   */
  theta = multirotor0_X.omega_CSTATE[1];
  q1 = multirotor0_X.omega_CSTATE[0];
  COM_system_inter_idx_0 = multirotor0_X.omega_CSTATE[2];
  for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
    /* Sum: '<S2>/Sum1' incorporates:
     *  Concatenate: '<S33>/Vector Concatenate'
     *  Inport: '<Root>/force_disturbance'
     *  Merge: '<S40>/Merge'
     *  Product: '<S2>/Product1'
     *  Product: '<S38>/Product'
     *  Sum: '<S11>/Sum'
     *  Sum: '<S16>/Sum'
     *  Sum: '<S4>/Sum'
     *  Sum: '<S4>/Sum1'
     *  Sum: '<S4>/Sum3'
     */
    multirotor0_B.Sum1[xpageoffset] = (((((rtb_VectorConcatenate[xpageoffset + 3]
      * cphi + rtb_VectorConcatenate[xpageoffset] * phi) +
      rtb_VectorConcatenate[xpageoffset + 6] * ctheta) + rtb_Sum_a[xpageoffset])
      + multirotor0_B.Forceagainstdirectionofmotiondu[xpageoffset]) +
      multirotor0_U.Force_disturb[xpageoffset]) / multirotor0_B.total_mass -
      (rtb_TrueairspeedBodyaxes[xpageoffset] -
       rtb_TrueairspeedBodyaxes_m[xpageoffset]);

    /* Product: '<S13>/Product' incorporates:
     *  Integrator: '<S2>/omega'
     *  Product: '<S8>/Product'
     */
    rtb_TrueairspeedBodyaxes_b[xpageoffset] =
      (multirotor0_B.inertial_matrix[xpageoffset + 3] * theta +
       multirotor0_B.inertial_matrix[xpageoffset] * q1) +
      multirotor0_B.inertial_matrix[xpageoffset + 6] * COM_system_inter_idx_0;
  }

  /* Sum: '<S12>/Sum' incorporates:
   *  Integrator: '<S2>/omega'
   *  Product: '<S14>/Product'
   *  Product: '<S14>/Product1'
   *  Product: '<S14>/Product2'
   *  Product: '<S15>/Product'
   *  Product: '<S15>/Product1'
   *  Product: '<S15>/Product2'
   */
  rtb_TrueairspeedBodyaxes[0] = multirotor0_X.omega_CSTATE[1] *
    rtb_TrueairspeedBodyaxes_b[2];
  rtb_TrueairspeedBodyaxes[1] = rtb_TrueairspeedBodyaxes_b[0] *
    multirotor0_X.omega_CSTATE[2];
  rtb_TrueairspeedBodyaxes[2] = multirotor0_X.omega_CSTATE[0] *
    rtb_TrueairspeedBodyaxes_b[1];
  rtb_TrueairspeedBodyaxes_m[0] = rtb_TrueairspeedBodyaxes_b[1] *
    multirotor0_X.omega_CSTATE[2];
  rtb_TrueairspeedBodyaxes_m[1] = multirotor0_X.omega_CSTATE[0] *
    rtb_TrueairspeedBodyaxes_b[2];
  rtb_TrueairspeedBodyaxes_m[2] = rtb_TrueairspeedBodyaxes_b[0] *
    multirotor0_X.omega_CSTATE[1];
  for (xpageoffset = 0; xpageoffset < 3; xpageoffset++) {
    rtb_TrueairspeedBodyaxes_b[xpageoffset] =
      rtb_TrueairspeedBodyaxes[xpageoffset] -
      rtb_TrueairspeedBodyaxes_m[xpageoffset];

    /* Sum: '<S39>/Sum of Elements1' incorporates:
     *  ForEachSliceAssignment generated from: '<S57>/Motor_moment'
     *  Sum: '<S16>/Sum'
     */
    rtb_Sum_a[xpageoffset] = ((rtb_ImpAsg_InsertedFor_Motor_mo[xpageoffset + 3]
      + rtb_ImpAsg_InsertedFor_Motor_mo[xpageoffset]) +
      rtb_ImpAsg_InsertedFor_Motor_mo[xpageoffset + 6]) +
      rtb_ImpAsg_InsertedFor_Motor_mo[xpageoffset + 9];
  }

  /* End of Sum: '<S12>/Sum' */

  /* Product: '<S17>/Product' */
  q1 = 0.0;

  /* SignalConversion generated from: '<S8>/Q-Integrator' incorporates:
   *  Gain: '<S8>/1//2'
   *  Integrator: '<S2>/omega'
   *  Product: '<S20>/Product'
   *  Product: '<S20>/Product1'
   *  Product: '<S20>/Product2'
   *  Product: '<S21>/Product'
   *  Product: '<S21>/Product1'
   *  Product: '<S21>/Product2'
   *  Product: '<S8>/Product'
   *  Sum: '<S16>/Sum'
   *  Sum: '<S8>/Subtract'
   */
  multirotor0_B.TmpSignalConversionAtQIntegrato[1] = (rtb_Divide[0] *
    multirotor0_X.omega_CSTATE[0] - (multirotor0_X.omega_CSTATE[1] * rtb_Divide
    [3] - multirotor0_X.omega_CSTATE[2] * rtb_Divide[2])) * 0.5;
  multirotor0_B.TmpSignalConversionAtQIntegrato[2] = (rtb_Divide[0] *
    multirotor0_X.omega_CSTATE[1] - (rtb_Divide[1] * multirotor0_X.omega_CSTATE
    [2] - multirotor0_X.omega_CSTATE[0] * rtb_Divide[3])) * 0.5;
  multirotor0_B.TmpSignalConversionAtQIntegrato[3] = (rtb_Divide[0] *
    multirotor0_X.omega_CSTATE[2] - (multirotor0_X.omega_CSTATE[0] * rtb_Divide
    [2] - multirotor0_X.omega_CSTATE[1] * rtb_Divide[1])) * 0.5;

  /* Fcn: '<S9>/Fcn1' */
  rtb_VectorConcatenate_tmp_2 *= 2.0;

  /* Trigonometry: '<S9>/Trigonometric Function' */
  if (rtb_VectorConcatenate_tmp_2 > 1.0) {
    rtb_VectorConcatenate_tmp_2 = 1.0;
  } else if (rtb_VectorConcatenate_tmp_2 < -1.0) {
    rtb_VectorConcatenate_tmp_2 = -1.0;
  }

  /* Gain: '<S9>/Gain' incorporates:
   *  Outport: '<Root>/Euler'
   *  Trigonometry: '<S9>/Trigonometric Function'
   */
  multirotor0_Y.Euler[1] = -std::asin(rtb_VectorConcatenate_tmp_2);

  /* Trigonometry: '<S9>/Trigonometric Function1' incorporates:
   *  Fcn: '<S9>/Fcn'
   *  Outport: '<Root>/Euler'
   */
  multirotor0_Y.Euler[0] = rt_atan2d_snf(rtb_VectorConcatenate_tmp_1 * 2.0,
    rtb_VectorConcatenate_tmp_0);

  /* Trigonometry: '<S9>/Trigonometric Function2' incorporates:
   *  Fcn: '<S9>/Fcn2'
   *  Outport: '<Root>/Euler'
   */
  multirotor0_Y.Euler[2] = rt_atan2d_snf(rtb_VectorConcatenate_tmp * 2.0,
    rtb_VectorConcatenate_tmp_3);
  for (i = 0; i < 3; i++) {
    /* Sum: '<S7>/Sum1' incorporates:
     *  Inport: '<Root>/moment_disturbance'
     *  Product: '<S8>/Product'
     *  Sum: '<S16>/Sum'
     *  Sum: '<S4>/Sum2'
     */
    rtb_TrueairspeedBodyaxes[i] = (rtb_Sum_a[i] + multirotor0_U.Moment_disturb[i])
      - rtb_TrueairspeedBodyaxes_b[i];

    /* Product: '<S17>/Product' incorporates:
     *  Integrator: '<S2>/omega'
     */
    q1 += rtb_Divide[i + 1] * multirotor0_X.omega_CSTATE[i];

    /* Outport: '<Root>/X_i' incorporates:
     *  Integrator: '<S2>/X_i'
     */
    multirotor0_Y.X_i[i] = multirotor0_X.X_i_CSTATE[i];

    /* Outport: '<Root>/V_i' incorporates:
     *  Product: '<S5>/Product'
     */
    multirotor0_Y.V_i[i] = multirotor0_B.Product[i];

    /* Outport: '<Root>/V_b' incorporates:
     *  Integrator: '<S2>/V_b'
     */
    multirotor0_Y.V_b[i] = multirotor0_X.V_b_CSTATE[i];

    /* Outport: '<Root>/a_b' incorporates:
     *  Sum: '<S2>/Sum1'
     */
    multirotor0_Y.a_b[i] = multirotor0_B.Sum1[i];

    /* Outport: '<Root>/a_i' incorporates:
     *  Math: '<S6>/Math Function2'
     *  Product: '<S6>/Product'
     *  Sum: '<S2>/Sum1'
     */
    multirotor0_Y.a_i[i] = (Product_tmp[i + 3] * multirotor0_B.Sum1[1] +
      Product_tmp[i] * multirotor0_B.Sum1[0]) + Product_tmp[i + 6] *
      multirotor0_B.Sum1[2];
  }

  /* Product: '<S7>/Product' */
  rt_mldivide_U1d3x3_U2d_JBYZyA3A(multirotor0_B.inertial_matrix,
    rtb_TrueairspeedBodyaxes, multirotor0_B.Product_l);

  /* SignalConversion generated from: '<S8>/Q-Integrator' incorporates:
   *  Gain: '<S8>/-1//2'
   *  Product: '<S17>/Product'
   */
  multirotor0_B.TmpSignalConversionAtQIntegrato[0] = -0.5 * q1;

  /* End of Outputs for SubSystem: '<Root>/multirotor' */

  /* Outport: '<Root>/DCM_ib' incorporates:
   *  Concatenate: '<S33>/Vector Concatenate'
   */
  std::memcpy(&multirotor0_Y.DCM_ib[0], &rtb_VectorConcatenate[0], 9U * sizeof
              (real_T));

  /* Outport: '<Root>/Quat q' incorporates:
   *  Product: '<S19>/Divide'
   */
  multirotor0_Y.Quatq[0] = rtb_Divide[0];
  multirotor0_Y.Quatq[1] = rtb_Divide[1];
  multirotor0_Y.Quatq[2] = rtb_Divide[2];
  multirotor0_Y.Quatq[3] = rtb_Divide[3];

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  /* Outport: '<Root>/omega' incorporates:
   *  Integrator: '<S2>/omega'
   */
  multirotor0_Y.omega[0] = multirotor0_X.omega_CSTATE[0];

  /* End of Outputs for SubSystem: '<Root>/multirotor' */

  /* Outport: '<Root>/omega_dot' incorporates:
   *  Product: '<S7>/Product'
   */
  multirotor0_Y.omega_dot[0] = multirotor0_B.Product_l[0];

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  /* Outport: '<Root>/omega' incorporates:
   *  Integrator: '<S2>/omega'
   */
  multirotor0_Y.omega[1] = multirotor0_X.omega_CSTATE[1];

  /* End of Outputs for SubSystem: '<Root>/multirotor' */

  /* Outport: '<Root>/omega_dot' incorporates:
   *  Product: '<S7>/Product'
   */
  multirotor0_Y.omega_dot[1] = multirotor0_B.Product_l[1];

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  /* Outport: '<Root>/omega' incorporates:
   *  Integrator: '<S2>/omega'
   */
  multirotor0_Y.omega[2] = multirotor0_X.omega_CSTATE[2];

  /* End of Outputs for SubSystem: '<Root>/multirotor' */

  /* Outport: '<Root>/omega_dot' incorporates:
   *  Product: '<S7>/Product'
   */
  multirotor0_Y.omega_dot[2] = multirotor0_B.Product_l[2];

  /* Outport: '<Root>/motor_RPM' incorporates:
   *  ForEachSliceAssignment generated from: '<S57>/RPM_motor'
   */
  multirotor0_Y.motor_RPM[0] = rtb_ImpAsg_InsertedFor_RPM_moto[0];
  multirotor0_Y.motor_RPM[1] = rtb_ImpAsg_InsertedFor_RPM_moto[1];
  multirotor0_Y.motor_RPM[2] = rtb_ImpAsg_InsertedFor_RPM_moto[2];
  multirotor0_Y.motor_RPM[3] = rtb_ImpAsg_InsertedFor_RPM_moto[3];
  if (rtmIsMajorTimeStep((&multirotor0_M))) {
    /* Update for Atomic SubSystem: '<Root>/multirotor' */
    /* Update for Integrator: '<S8>/Q-Integrator' */
    multirotor0_DW.QIntegrator_IWORK = 0;

    /* End of Update for SubSystem: '<Root>/multirotor' */
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep((&multirotor0_M))) {
    rt_ertODEUpdateContinuousStates(&(&multirotor0_M)->solverInfo);

    /* Update absolute time */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     */
    ++(&multirotor0_M)->Timing.clockTick0;
    (&multirotor0_M)->Timing.t[0] = rtsiGetSolverStopTime(&(&multirotor0_M)
      ->solverInfo);

    /* Update absolute time */
    /* The "clockTick1" counts the number of times the code of this task has
     * been executed. The resolution of this integer timer is 0.001, which is the step size
     * of the task. Size of "clockTick1" ensures timer will not overflow during the
     * application lifespan selected.
     */
    (&multirotor0_M)->Timing.clockTick1++;
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void multirotor0::multirotor0_derivatives()
{
  /* local scratch DWork variables */
  int32_T ForEach_itr;
  XDot_multirotor0_T *_rtXdot;
  _rtXdot = ((XDot_multirotor0_T *) (&multirotor0_M)->derivs);

  /* Derivatives for Atomic SubSystem: '<Root>/multirotor' */
  /* Derivatives for Integrator: '<S8>/Q-Integrator' incorporates:
   *  SignalConversion generated from: '<S8>/Q-Integrator'
   */
  _rtXdot->QIntegrator_CSTATE[0] =
    multirotor0_B.TmpSignalConversionAtQIntegrato[0];
  _rtXdot->QIntegrator_CSTATE[1] =
    multirotor0_B.TmpSignalConversionAtQIntegrato[1];
  _rtXdot->QIntegrator_CSTATE[2] =
    multirotor0_B.TmpSignalConversionAtQIntegrato[2];
  _rtXdot->QIntegrator_CSTATE[3] =
    multirotor0_B.TmpSignalConversionAtQIntegrato[3];

  /* Derivatives for Integrator: '<S2>/V_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  _rtXdot->V_b_CSTATE[0] = multirotor0_B.Sum1[0];

  /* Derivatives for Integrator: '<S2>/omega' incorporates:
   *  Product: '<S7>/Product'
   */
  _rtXdot->omega_CSTATE[0] = multirotor0_B.Product_l[0];

  /* Derivatives for Integrator: '<S2>/V_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  _rtXdot->V_b_CSTATE[1] = multirotor0_B.Sum1[1];

  /* Derivatives for Integrator: '<S2>/omega' incorporates:
   *  Product: '<S7>/Product'
   */
  _rtXdot->omega_CSTATE[1] = multirotor0_B.Product_l[1];

  /* Derivatives for Integrator: '<S2>/V_b' incorporates:
   *  Sum: '<S2>/Sum1'
   */
  _rtXdot->V_b_CSTATE[2] = multirotor0_B.Sum1[2];

  /* Derivatives for Integrator: '<S2>/omega' incorporates:
   *  Product: '<S7>/Product'
   */
  _rtXdot->omega_CSTATE[2] = multirotor0_B.Product_l[2];

  /* Derivatives for Iterator SubSystem: '<S39>/For Each Subsystem' */
  for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
    /* Derivatives for Integrator: '<S63>/Integrator' */
    _rtXdot->CoreSubsys[ForEach_itr].Integrator_CSTATE =
      multirotor0_B.CoreSubsys[ForEach_itr].Switch;

    /* Derivatives for Integrator: '<S64>/Integrator' */
    _rtXdot->CoreSubsys[ForEach_itr].Integrator_CSTATE_o =
      multirotor0_B.CoreSubsys[ForEach_itr].Switch_a;
  }

  /* End of Derivatives for SubSystem: '<S39>/For Each Subsystem' */

  /* Derivatives for Integrator: '<S2>/X_i' incorporates:
   *  Product: '<S5>/Product'
   */
  _rtXdot->X_i_CSTATE[0] = multirotor0_B.Product[0];
  _rtXdot->X_i_CSTATE[1] = multirotor0_B.Product[1];
  _rtXdot->X_i_CSTATE[2] = multirotor0_B.Product[2];

  /* End of Derivatives for SubSystem: '<Root>/multirotor' */
}

/* Model step function for TID2 */
void multirotor0::step2()              /* Sample time: [0.002s, 0.0s] */
{
  /* Update for Atomic SubSystem: '<Root>/multirotor' */
  /* Update for RateTransition: '<S1>/Rate Transition1' incorporates:
   *  Inport: '<Root>/RPM commands'
   */
  multirotor0_DW.RateTransition1_Buffer0[0] = multirotor0_U.RPMcommands[0];
  multirotor0_DW.RateTransition1_Buffer0[1] = multirotor0_U.RPMcommands[1];
  multirotor0_DW.RateTransition1_Buffer0[2] = multirotor0_U.RPMcommands[2];
  multirotor0_DW.RateTransition1_Buffer0[3] = multirotor0_U.RPMcommands[3];

  /* End of Update for SubSystem: '<Root>/multirotor' */
}

/* Model initialize function */
void multirotor0::initialize()
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  /* Set task counter limit used by the static main program */
  ((&multirotor0_M))->Timing.TaskCounters.cLimit[0] = 1;
  ((&multirotor0_M))->Timing.TaskCounters.cLimit[1] = 1;
  ((&multirotor0_M))->Timing.TaskCounters.cLimit[2] = 2;

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&(&multirotor0_M)->solverInfo, &(&multirotor0_M)
                          ->Timing.simTimeStep);
    rtsiSetTPtr(&(&multirotor0_M)->solverInfo, &rtmGetTPtr((&multirotor0_M)));
    rtsiSetStepSizePtr(&(&multirotor0_M)->solverInfo, &(&multirotor0_M)
                       ->Timing.stepSize0);
    rtsiSetdXPtr(&(&multirotor0_M)->solverInfo, &(&multirotor0_M)->derivs);
    rtsiSetContStatesPtr(&(&multirotor0_M)->solverInfo, (real_T **)
                         &(&multirotor0_M)->contStates);
    rtsiSetNumContStatesPtr(&(&multirotor0_M)->solverInfo, &(&multirotor0_M)
      ->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&(&multirotor0_M)->solverInfo,
      &(&multirotor0_M)->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&(&multirotor0_M)->solverInfo,
      &(&multirotor0_M)->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&(&multirotor0_M)->solverInfo,
      &(&multirotor0_M)->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&(&multirotor0_M)->solverInfo, (&rtmGetErrorStatus
      ((&multirotor0_M))));
    rtsiSetRTModelPtr(&(&multirotor0_M)->solverInfo, (&multirotor0_M));
  }

  rtsiSetSimTimeStep(&(&multirotor0_M)->solverInfo, MAJOR_TIME_STEP);
  (&multirotor0_M)->intgData.y = (&multirotor0_M)->odeY;
  (&multirotor0_M)->intgData.f[0] = (&multirotor0_M)->odeF[0];
  (&multirotor0_M)->intgData.f[1] = (&multirotor0_M)->odeF[1];
  (&multirotor0_M)->intgData.f[2] = (&multirotor0_M)->odeF[2];
  (&multirotor0_M)->contStates = ((X_multirotor0_T *) &multirotor0_X);
  rtsiSetSolverData(&(&multirotor0_M)->solverInfo, static_cast<void *>
                    (&(&multirotor0_M)->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&(&multirotor0_M)->solverInfo, false);
  rtsiSetSolverName(&(&multirotor0_M)->solverInfo,"ode3");
  rtmSetTPtr((&multirotor0_M), &(&multirotor0_M)->Timing.tArray[0]);
  (&multirotor0_M)->Timing.stepSize0 = 0.001;
  rtmSetFirstInitCond((&multirotor0_M), 1);

  {
    /* local scratch DWork variables */
    int32_T ForEach_itr;

    /* Start for Atomic SubSystem: '<Root>/multirotor' */
    /* Start for RateTransition: '<S1>/Rate Transition1' */
    multirotor0_B.RateTransition1[0] = 3104.5025852;
    multirotor0_B.RateTransition1[1] = 3104.5025852;
    multirotor0_B.RateTransition1[2] = 3104.5025852;
    multirotor0_B.RateTransition1[3] = 3104.5025852;

    /* Start for Iterator SubSystem: '<S39>/For Each Subsystem' */
    for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
      /* Start for If: '<S90>/If' */
      multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem = -1;

      /* Start for If: '<S88>/If' */
      multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_l = -1;

      /* Start for If: '<S76>/If' */
      multirotor0_DW.CoreSubsys[ForEach_itr].If_ActiveSubsystem_e = -1;
    }

    /* End of Start for SubSystem: '<S39>/For Each Subsystem' */

    /* Start for If: '<S40>/If' */
    multirotor0_DW.If_ActiveSubsystem = -1;

    /* End of Start for SubSystem: '<Root>/multirotor' */
  }

  {
    /* local scratch DWork variables */
    int32_T ForEach_itr;

    /* SystemInitialize for Atomic SubSystem: '<Root>/multirotor' */
    /* InitializeConditions for Integrator: '<S8>/Q-Integrator' */
    if (rtmIsFirstInitCond((&multirotor0_M))) {
      multirotor0_X.QIntegrator_CSTATE[0] = 0.0;
      multirotor0_X.QIntegrator_CSTATE[1] = 0.0;
      multirotor0_X.QIntegrator_CSTATE[2] = 0.0;
      multirotor0_X.QIntegrator_CSTATE[3] = 0.0;
    }

    multirotor0_DW.QIntegrator_IWORK = 1;

    /* End of InitializeConditions for Integrator: '<S8>/Q-Integrator' */

    /* InitializeConditions for Integrator: '<S2>/V_b' */
    multirotor0_X.V_b_CSTATE[0] = 0.0;
    multirotor0_X.V_b_CSTATE[1] = 0.0;
    multirotor0_X.V_b_CSTATE[2] = 0.0;

    /* InitializeConditions for RateTransition: '<S1>/Rate Transition1' */
    multirotor0_DW.RateTransition1_Buffer0[0] = 3104.5025852;
    multirotor0_DW.RateTransition1_Buffer0[1] = 3104.5025852;
    multirotor0_DW.RateTransition1_Buffer0[2] = 3104.5025852;
    multirotor0_DW.RateTransition1_Buffer0[3] = 3104.5025852;

    /* InitializeConditions for Integrator: '<S2>/omega' */
    multirotor0_X.omega_CSTATE[0] = 0.0;

    /* InitializeConditions for Integrator: '<S2>/X_i' */
    multirotor0_X.X_i_CSTATE[0] = 0.0;

    /* SystemInitialize for IfAction SubSystem: '<S40>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S40>/Merge' incorporates:
     *  Outport: '<S43>/Drag force'
     */
    multirotor0_B.Forceagainstdirectionofmotiondu[0] = 0.0;

    /* End of SystemInitialize for SubSystem: '<S40>/Zero airspeed' */

    /* InitializeConditions for Integrator: '<S2>/omega' */
    multirotor0_X.omega_CSTATE[1] = 0.0;

    /* InitializeConditions for Integrator: '<S2>/X_i' */
    multirotor0_X.X_i_CSTATE[1] = 0.0;

    /* SystemInitialize for IfAction SubSystem: '<S40>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S40>/Merge' incorporates:
     *  Outport: '<S43>/Drag force'
     */
    multirotor0_B.Forceagainstdirectionofmotiondu[1] = 0.0;

    /* End of SystemInitialize for SubSystem: '<S40>/Zero airspeed' */

    /* InitializeConditions for Integrator: '<S2>/omega' */
    multirotor0_X.omega_CSTATE[2] = 0.0;

    /* InitializeConditions for Integrator: '<S2>/X_i' */
    multirotor0_X.X_i_CSTATE[2] = 0.0;

    /* SystemInitialize for IfAction SubSystem: '<S40>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S40>/Merge' incorporates:
     *  Outport: '<S43>/Drag force'
     */
    multirotor0_B.Forceagainstdirectionofmotiondu[2] = -1.0;

    /* End of SystemInitialize for SubSystem: '<S40>/Zero airspeed' */

    /* SystemInitialize for Iterator SubSystem: '<S39>/For Each Subsystem' */
    for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
      /* InitializeConditions for Integrator: '<S63>/Integrator' */
      multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE = 3104.5025852;

      /* InitializeConditions for Integrator: '<S64>/Integrator' */
      multirotor0_X.CoreSubsys[ForEach_itr].Integrator_CSTATE_o = 3104.5025852;

      /* SystemInitialize for IfAction SubSystem: '<S76>/Zero airspeed in rotor plane' */
      /* SystemInitialize for Merge: '<S76>/Merge' incorporates:
       *  Outport: '<S83>/Thrust direction (Body)'
       */
      multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0] =
        0.0;
      multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1] =
        0.0;
      multirotor0_B.CoreSubsys[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2] =
        -1.0;

      /* End of SystemInitialize for SubSystem: '<S76>/Zero airspeed in rotor plane' */
    }

    /* End of SystemInitialize for SubSystem: '<S39>/For Each Subsystem' */
    /* End of SystemInitialize for SubSystem: '<Root>/multirotor' */

    /* set "at time zero" to false */
    if (rtmIsFirstInitCond((&multirotor0_M))) {
      rtmSetFirstInitCond((&multirotor0_M), 0);
    }
  }
}

/* Model terminate function */
void multirotor0::terminate()
{
  /* (no terminate code required) */
}

/* Constructor */
multirotor0::multirotor0() :
  multirotor0_U(),
  multirotor0_Y(),
  multirotor0_B(),
  multirotor0_DW(),
  multirotor0_X(),
  multirotor0_M()
{
  /* Currently there is no constructor body generated.*/
}

/* Destructor */
/* Currently there is no destructor body generated.*/
multirotor0::~multirotor0() = default;

/* Real-Time Model get method */
RT_MODEL_multirotor0_T * multirotor0::getRTM()
{
  return (&multirotor0_M);
}
