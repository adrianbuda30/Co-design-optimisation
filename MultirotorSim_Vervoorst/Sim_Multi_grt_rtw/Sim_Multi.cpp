/*
 * Sim_Multi.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "Sim_Multi".
 *
 * Model version              : 14.1
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Mon Jul 24 11:57:47 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: 32-bit Generic
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#include "Sim_Multi.h"
#include "rtwtypes.h"
#include "Sim_Multi_private.h"
#include <cmath>
#include <cstring>

extern "C"
{

#include "rt_nonfinite.h"

}

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
void Sim_Multi::rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
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
  int_T nXc { 29 };

  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  Sim_Multi_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  this->step0();
  Sim_Multi_derivatives();

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
  Sim_Multi_derivatives();

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

real_T rt_urand_Upu32_Yd_f_pw_snf(uint32_T *u)
{
  uint32_T hi;
  uint32_T lo;

  /* Uniform random number generator (random number between 0 and 1)

     #define IA      16807                      magic multiplier = 7^5
     #define IM      2147483647                 modulus = 2^31-1
     #define IQ      127773                     IM div IA
     #define IR      2836                       IM modulo IA
     #define S       4.656612875245797e-10      reciprocal of 2^31-1
     test = IA * (seed % IQ) - IR * (seed/IQ)
     seed = test < 0 ? (test + IM) : test
     return (seed*S)
   */
  lo = *u % 127773U * 16807U;
  hi = *u / 127773U * 2836U;
  if (lo < hi) {
    *u = 2147483647U - (hi - lo);
  } else {
    *u = lo - hi;
  }

  return static_cast<real_T>(*u) * 4.6566128752457969E-10;
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
void Sim_Multi::step0()                /* Sample time: [0.0s, 0.0s] */
{
  /* local scratch DWork variables */
  int32_T ForEach_itr;
  real_T rtb_ImpAsg_InsertedFor_Motor_fo[12];
  real_T rtb_ImpAsg_InsertedFor_Motor_mo[12];
  real_T rtb_VectorConcatenate_k[9];
  real_T rtb_Sum_k1[3];
  real_T rtb_TrueairspeedBodyaxes[3];
  real_T rtb_TrueairspeedBodyaxes_b[3];
  real_T rtb_TrueairspeedatpropMotoraxes[3];
  real_T VectorConcatenate;
  real_T VectorConcatenate_tmp;
  real_T VectorConcatenate_tmp_0;
  real_T cphi;
  real_T cpsi;
  real_T ctheta;
  real_T phi;
  real_T phi_tmp;
  real_T rtb_Divide_idx_0;
  real_T rtb_Divide_idx_1;
  real_T rtb_Divide_idx_2;
  real_T rtb_Divide_idx_3;
  real_T rtb_Memory3_idx_0;
  real_T rtb_Memory3_idx_1;
  real_T rtb_Memory3_idx_2;
  real_T rtb_Memory_idx_0;
  real_T rtb_Memory_idx_1;
  real_T rtb_Memory_idx_2;
  real_T rtb_Memory_idx_3;
  real_T rtb_VectorConcatenate_g_tmp;
  real_T rtb_VectorConcatenate_g_tmp_0;
  real_T rtb_VectorConcatenate_g_tmp_1;
  real_T rtb_VectorConcatenate_g_tmp_2;
  real_T rtb_VectorConcatenate_g_tmp_3;
  real_T rtb_VectorConcatenate_g_tmp_4;
  real_T theta;
  int32_T i;
  int8_T rtAction;
  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    /* set solver stop time */
    rtsiSetSolverStopTime(&(&Sim_Multi_M)->solverInfo,(((&Sim_Multi_M)
      ->Timing.clockTick0+1)*(&Sim_Multi_M)->Timing.stepSize0));

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.002s, 0.0s]  */
    ((&Sim_Multi_M)->Timing.RateInteraction.TID1_2)++;
    if (((&Sim_Multi_M)->Timing.RateInteraction.TID1_2) > 1) {
      (&Sim_Multi_M)->Timing.RateInteraction.TID1_2 = 0;
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep((&Sim_Multi_M))) {
    (&Sim_Multi_M)->Timing.t[0] = rtsiGetT(&(&Sim_Multi_M)->solverInfo);
  }

  /* TransferFcn: '<S31>/Transfer Fcn1' */
  Sim_Multi_B.TransferFcn1 = 100.0 * Sim_Multi_X.TransferFcn1_CSTATE;

  /* TransferFcn: '<S31>/Transfer Fcn6' */
  Sim_Multi_B.TransferFcn6 = 100.0 * Sim_Multi_X.TransferFcn6_CSTATE;

  /* TransferFcn: '<S31>/Transfer Fcn4' */
  Sim_Multi_B.TransferFcn4 = 100.0 * Sim_Multi_X.TransferFcn4_CSTATE;

  /* TransferFcn: '<S31>/Transfer Fcn5' */
  Sim_Multi_B.TransferFcn5 = 100.0 * Sim_Multi_X.TransferFcn5_CSTATE;

  /* TransferFcn: '<S31>/Transfer Fcn2' */
  Sim_Multi_B.TransferFcn2 = 100.0 * Sim_Multi_X.TransferFcn2_CSTATE;

  /* TransferFcn: '<S31>/Transfer Fcn3' */
  Sim_Multi_B.TransferFcn3 = 100.0 * Sim_Multi_X.TransferFcn3_CSTATE;

  /* Step: '<S32>/Step2' incorporates:
   *  Step: '<S32>/Step1'
   *  Step: '<S32>/Step6'
   *  Step: '<S33>/Step'
   *  Step: '<S33>/Step1'
   *  Step: '<S33>/Step2'
   */
  phi_tmp = (&Sim_Multi_M)->Timing.t[0];
  if (phi_tmp < 2.0) {
    i = 0;
  } else {
    i = -2;
  }

  /* Gain: '<S32>/Gain2' incorporates:
   *  Gain: '<S32>/Gain1'
   *  Step: '<S32>/Step1'
   *  Step: '<S32>/Step2'
   *  Step: '<S32>/Step6'
   *  Sum: '<S32>/Add2'
   */
  Sim_Multi_B.Gain2 = ((static_cast<real_T>(!(phi_tmp < 1.0)) +
                        static_cast<real_T>(i)) + static_cast<real_T>(!(phi_tmp <
    3.0))) * 0.1 * 500.0;

  /* RateTransition: '<S30>/Rate Transition' */
  if (rtmIsMajorTimeStep((&Sim_Multi_M)) && ((&Sim_Multi_M)
       ->Timing.RateInteraction.TID1_2 == 1)) {
    Sim_Multi_DW.RateTransition_Buffer[0] = Sim_Multi_B.TransferFcn1;
    Sim_Multi_DW.RateTransition_Buffer[1] = Sim_Multi_B.TransferFcn5;
    Sim_Multi_DW.RateTransition_Buffer[2] = Sim_Multi_B.TransferFcn3;
  }

  /* End of RateTransition: '<S30>/Rate Transition' */

  /* Step: '<S33>/Step1' */
  if (phi_tmp < 1.5) {
    i = 0;
  } else {
    i = -2;
  }

  /* Gain: '<S33>/Gain2' incorporates:
   *  Gain: '<S33>/Gain'
   *  Step: '<S33>/Step'
   *  Step: '<S33>/Step1'
   *  Step: '<S33>/Step2'
   *  Sum: '<S33>/Add'
   */
  Sim_Multi_B.Gain2_i = ((static_cast<real_T>(!(phi_tmp < 0.5)) +
    static_cast<real_T>(i)) + static_cast<real_T>(!(phi_tmp < 2.5))) * 0.1 *
    500.0;

  /* Gain: '<S34>/Gain2' incorporates:
   *  Constant: '<S34>/Constant'
   */
  Sim_Multi_B.Gain2_g = 0.0;
  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    /* Memory: '<S36>/Memory' */
    rtb_Memory_idx_0 = Sim_Multi_DW.Memory_PreviousInput[0];
    rtb_Memory_idx_1 = Sim_Multi_DW.Memory_PreviousInput[1];
    rtb_Memory_idx_2 = Sim_Multi_DW.Memory_PreviousInput[2];
    rtb_Memory_idx_3 = Sim_Multi_DW.Memory_PreviousInput[3];

    /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
    /* MATLAB Function: '<S64>/MATLAB Function' */
    Sim_Multi_B.quat_output[0] = 1.0;
    Sim_Multi_B.quat_output[1] = 0.0;
    Sim_Multi_B.quat_output[2] = 0.0;
    Sim_Multi_B.quat_output[3] = 0.0;

    /* End of Outputs for SubSystem: '<Root>/multirotor' */
  }

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  /* Integrator: '<S64>/Q-Integrator' */
  if (Sim_Multi_DW.QIntegrator_IWORK != 0) {
    Sim_Multi_X.QIntegrator_CSTATE[0] = Sim_Multi_B.quat_output[0];
    Sim_Multi_X.QIntegrator_CSTATE[1] = Sim_Multi_B.quat_output[1];
    Sim_Multi_X.QIntegrator_CSTATE[2] = Sim_Multi_B.quat_output[2];
    Sim_Multi_X.QIntegrator_CSTATE[3] = Sim_Multi_B.quat_output[3];
  }

  /* Sqrt: '<S78>/Sqrt' incorporates:
   *  Integrator: '<S64>/Q-Integrator'
   *  Product: '<S79>/Product'
   */
  cphi = std::sqrt(((Sim_Multi_X.QIntegrator_CSTATE[0] *
                     Sim_Multi_X.QIntegrator_CSTATE[0] +
                     Sim_Multi_X.QIntegrator_CSTATE[1] *
                     Sim_Multi_X.QIntegrator_CSTATE[1]) +
                    Sim_Multi_X.QIntegrator_CSTATE[2] *
                    Sim_Multi_X.QIntegrator_CSTATE[2]) +
                   Sim_Multi_X.QIntegrator_CSTATE[3] *
                   Sim_Multi_X.QIntegrator_CSTATE[3]);

  /* Product: '<S75>/Divide' incorporates:
   *  Integrator: '<S64>/Q-Integrator'
   */
  rtb_Divide_idx_0 = Sim_Multi_X.QIntegrator_CSTATE[0] / cphi;
  rtb_Divide_idx_1 = Sim_Multi_X.QIntegrator_CSTATE[1] / cphi;
  rtb_Divide_idx_2 = Sim_Multi_X.QIntegrator_CSTATE[2] / cphi;
  rtb_Divide_idx_3 = Sim_Multi_X.QIntegrator_CSTATE[3] / cphi;

  /* Product: '<S80>/Product' incorporates:
   *  Product: '<S81>/Product'
   */
  cphi = rtb_Divide_idx_0 * rtb_Divide_idx_0;

  /* Product: '<S80>/Product2' incorporates:
   *  Product: '<S81>/Product2'
   */
  rtb_VectorConcatenate_g_tmp_1 = rtb_Divide_idx_1 * rtb_Divide_idx_1;

  /* Product: '<S80>/Product3' incorporates:
   *  Product: '<S81>/Product3'
   *  Product: '<S82>/Product3'
   */
  rtb_VectorConcatenate_g_tmp_2 = rtb_Divide_idx_2 * rtb_Divide_idx_2;

  /* Product: '<S80>/Product4' incorporates:
   *  Product: '<S81>/Product4'
   *  Product: '<S82>/Product4'
   */
  rtb_VectorConcatenate_g_tmp_3 = rtb_Divide_idx_3 * rtb_Divide_idx_3;

  /* Sum: '<S80>/Add' incorporates:
   *  Product: '<S80>/Product'
   *  Product: '<S80>/Product2'
   *  Product: '<S80>/Product3'
   *  Product: '<S80>/Product4'
   */
  rtb_VectorConcatenate_k[0] = ((cphi + rtb_VectorConcatenate_g_tmp_1) -
    rtb_VectorConcatenate_g_tmp_2) - rtb_VectorConcatenate_g_tmp_3;

  /* Product: '<S85>/Product' incorporates:
   *  Product: '<S83>/Product'
   */
  rtb_VectorConcatenate_g_tmp = rtb_Divide_idx_1 * rtb_Divide_idx_2;

  /* Product: '<S85>/Product2' incorporates:
   *  Product: '<S83>/Product2'
   */
  rtb_VectorConcatenate_g_tmp_0 = rtb_Divide_idx_0 * rtb_Divide_idx_3;

  /* Gain: '<S85>/Gain' incorporates:
   *  Product: '<S85>/Product'
   *  Product: '<S85>/Product2'
   *  Sum: '<S85>/Add'
   */
  rtb_VectorConcatenate_k[1] = (rtb_VectorConcatenate_g_tmp -
    rtb_VectorConcatenate_g_tmp_0) * 2.0;

  /* Product: '<S87>/Product' incorporates:
   *  Product: '<S84>/Product'
   */
  rtb_VectorConcatenate_g_tmp_4 = rtb_Divide_idx_1 * rtb_Divide_idx_3;

  /* Product: '<S87>/Product2' incorporates:
   *  Product: '<S84>/Product2'
   */
  phi = rtb_Divide_idx_0 * rtb_Divide_idx_2;

  /* Gain: '<S87>/Gain' incorporates:
   *  Product: '<S87>/Product'
   *  Product: '<S87>/Product2'
   *  Sum: '<S87>/Add'
   */
  rtb_VectorConcatenate_k[2] = (rtb_VectorConcatenate_g_tmp_4 + phi) * 2.0;

  /* Gain: '<S83>/Gain' incorporates:
   *  Sum: '<S83>/Add'
   */
  rtb_VectorConcatenate_k[3] = (rtb_VectorConcatenate_g_tmp +
    rtb_VectorConcatenate_g_tmp_0) * 2.0;

  /* Sum: '<S81>/Add' incorporates:
   *  Sum: '<S82>/Add'
   */
  cphi -= rtb_VectorConcatenate_g_tmp_1;
  rtb_VectorConcatenate_k[4] = (cphi + rtb_VectorConcatenate_g_tmp_2) -
    rtb_VectorConcatenate_g_tmp_3;

  /* Product: '<S88>/Product' incorporates:
   *  Product: '<S86>/Product'
   */
  rtb_VectorConcatenate_g_tmp_1 = rtb_Divide_idx_2 * rtb_Divide_idx_3;

  /* Product: '<S88>/Product2' incorporates:
   *  Product: '<S86>/Product2'
   */
  rtb_VectorConcatenate_g_tmp = rtb_Divide_idx_0 * rtb_Divide_idx_1;

  /* Gain: '<S88>/Gain' incorporates:
   *  Product: '<S88>/Product'
   *  Product: '<S88>/Product2'
   *  Sum: '<S88>/Add'
   */
  rtb_VectorConcatenate_k[5] = (rtb_VectorConcatenate_g_tmp_1 -
    rtb_VectorConcatenate_g_tmp) * 2.0;

  /* Gain: '<S84>/Gain' incorporates:
   *  Sum: '<S84>/Add'
   */
  rtb_VectorConcatenate_k[6] = (rtb_VectorConcatenate_g_tmp_4 - phi) * 2.0;

  /* Gain: '<S86>/Gain' incorporates:
   *  Sum: '<S86>/Add'
   */
  rtb_VectorConcatenate_k[7] = (rtb_VectorConcatenate_g_tmp_1 +
    rtb_VectorConcatenate_g_tmp) * 2.0;

  /* Sum: '<S82>/Add' */
  rtb_VectorConcatenate_k[8] = (cphi - rtb_VectorConcatenate_g_tmp_2) +
    rtb_VectorConcatenate_g_tmp_3;

  /* Integrator: '<S58>/omega' incorporates:
   *  Constant: '<S59>/Constant3'
   *  Product: '<S69>/Product'
   */
  cphi = Sim_Multi_X.omega_CSTATE[1];
  rtb_VectorConcatenate_g_tmp_1 = Sim_Multi_X.omega_CSTATE[0];
  rtb_VectorConcatenate_g_tmp_2 = Sim_Multi_X.omega_CSTATE[2];

  /* Integrator: '<S58>/V_b' incorporates:
   *  Math: '<S61>/Math Function2'
   */
  rtb_VectorConcatenate_g_tmp_3 = Sim_Multi_X.V_b_CSTATE[1];
  rtb_VectorConcatenate_g_tmp = Sim_Multi_X.V_b_CSTATE[0];
  rtb_VectorConcatenate_g_tmp_0 = Sim_Multi_X.V_b_CSTATE[2];
  for (i = 0; i < 3; i++) {
    /* Product: '<S69>/Product' incorporates:
     *  Constant: '<S59>/Constant3'
     *  Integrator: '<S58>/omega'
     *  Sum: '<S72>/Sum'
     */
    rtb_Sum_k1[i] = (Sim_Multi_ConstP.pooled11[i + 3] * cphi +
                     Sim_Multi_ConstP.pooled11[i] *
                     rtb_VectorConcatenate_g_tmp_1) +
      Sim_Multi_ConstP.pooled11[i + 6] * rtb_VectorConcatenate_g_tmp_2;

    /* Product: '<S61>/Product' incorporates:
     *  Concatenate: '<S89>/Vector Concatenate'
     *  Integrator: '<S58>/V_b'
     *  Math: '<S61>/Math Function2'
     */
    Sim_Multi_B.Product[i] = (rtb_VectorConcatenate_k[3 * i + 1] *
      rtb_VectorConcatenate_g_tmp_3 + rtb_VectorConcatenate_k[3 * i] *
      rtb_VectorConcatenate_g_tmp) + rtb_VectorConcatenate_k[3 * i + 2] *
      rtb_VectorConcatenate_g_tmp_0;
  }

  /* Sum: '<S68>/Sum' incorporates:
   *  Integrator: '<S58>/omega'
   *  Product: '<S70>/Product'
   *  Product: '<S70>/Product1'
   *  Product: '<S70>/Product2'
   *  Product: '<S71>/Product'
   *  Product: '<S71>/Product1'
   *  Product: '<S71>/Product2'
   */
  rtb_VectorConcatenate_g_tmp_1 = Sim_Multi_X.omega_CSTATE[1] * rtb_Sum_k1[2];
  rtb_VectorConcatenate_g_tmp_2 = rtb_Sum_k1[0] * Sim_Multi_X.omega_CSTATE[2];
  rtb_VectorConcatenate_g_tmp_3 = Sim_Multi_X.omega_CSTATE[0] * rtb_Sum_k1[1];
  rtb_VectorConcatenate_g_tmp = rtb_Sum_k1[1] * Sim_Multi_X.omega_CSTATE[2];
  rtb_VectorConcatenate_g_tmp_0 = Sim_Multi_X.omega_CSTATE[0] * rtb_Sum_k1[2];
  rtb_VectorConcatenate_g_tmp_4 = rtb_Sum_k1[0] * Sim_Multi_X.omega_CSTATE[1];

  /* RateTransition: '<S5>/Rate Transition1' */
  if (rtmIsMajorTimeStep((&Sim_Multi_M)) && ((&Sim_Multi_M)
       ->Timing.RateInteraction.TID1_2 == 1)) {
    /* RateTransition: '<S5>/Rate Transition1' */
    Sim_Multi_B.RateTransition1[0] = Sim_Multi_DW.RateTransition1_Buffer0[0];
    Sim_Multi_B.RateTransition1[1] = Sim_Multi_DW.RateTransition1_Buffer0[1];
    Sim_Multi_B.RateTransition1[2] = Sim_Multi_DW.RateTransition1_Buffer0[2];
    Sim_Multi_B.RateTransition1[3] = Sim_Multi_DW.RateTransition1_Buffer0[3];
  }

  /* End of RateTransition: '<S5>/Rate Transition1' */
  for (i = 0; i < 3; i++) {
    /* Product: '<S172>/Product' incorporates:
     *  Concatenate: '<S89>/Vector Concatenate'
     *  Constant: '<S59>/Wind vector'
     *  Product: '<S111>/Product'
     */
    cphi = (rtb_VectorConcatenate_k[i + 3] * 0.0 + rtb_VectorConcatenate_k[i] *
            0.0) + rtb_VectorConcatenate_k[i + 6] * 0.0;
    rtb_TrueairspeedBodyaxes_b[i] = cphi;

    /* Sum: '<S113>/Sum1' incorporates:
     *  Integrator: '<S58>/V_b'
     */
    rtb_TrueairspeedBodyaxes[i] = Sim_Multi_X.V_b_CSTATE[i] - cphi;
  }

  /* Outputs for Iterator SubSystem: '<S94>/For Each Subsystem' incorporates:
   *  ForEach: '<S112>/For Each'
   */
  for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
    /* ForEachSliceSelector generated from: '<S112>/MotorMatrix_real' incorporates:
     *  Constant: '<S59>/Constant'
     *  RelationalOperator: '<S118>/Relational Operator'
     *  RelationalOperator: '<S122>/LowerRelop1'
     */
    phi_tmp = Sim_Multi_ConstP.pooled10[ForEach_itr + 44];

    /* Switch: '<S122>/Switch2' incorporates:
     *  Constant: '<S59>/Constant'
     *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
     *  Integrator: '<S118>/Integrator'
     *  RelationalOperator: '<S122>/LowerRelop1'
     */
    if (Sim_Multi_X.CoreSubsys_p[ForEach_itr].Integrator_CSTATE_e > phi_tmp) {
      cphi = Sim_Multi_ConstP.pooled10[ForEach_itr + 44];
    } else {
      /* RelationalOperator: '<S122>/UpperRelop' incorporates:
       *  Switch: '<S122>/Switch'
       */
      cphi = Sim_Multi_ConstP.pooled10[ForEach_itr + 40];

      /* Switch: '<S122>/Switch' incorporates:
       *  RelationalOperator: '<S122>/UpperRelop'
       */
      if (!(Sim_Multi_X.CoreSubsys_p[ForEach_itr].Integrator_CSTATE_e < cphi)) {
        cphi = Sim_Multi_X.CoreSubsys_p[ForEach_itr].Integrator_CSTATE_e;
      }
    }

    /* End of Switch: '<S122>/Switch2' */
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      /* Product: '<S114>/Product' incorporates:
       *  Constant: '<S59>/Constant'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       *  ForEachSliceSelector generated from: '<S112>/RPM_commands'
       *  RateTransition: '<S5>/Rate Transition1'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Product =
        Sim_Multi_ConstP.pooled10[ForEach_itr + 16] *
        Sim_Multi_B.RateTransition1[ForEach_itr];
    }

    /* Product: '<S114>/Divide' incorporates:
     *  Constant: '<S59>/Constant'
     *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
     *  Sum: '<S114>/Sum1'
     */
    phi = (Sim_Multi_B.CoreSubsys_p[ForEach_itr].Product - cphi) /
      Sim_Multi_ConstP.pooled10[ForEach_itr + 20];

    /* Switch: '<S118>/Switch' incorporates:
     *  Constant: '<S120>/Constant'
     *  Constant: '<S121>/Constant'
     *  Constant: '<S59>/Constant'
     *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
     *  Integrator: '<S118>/Integrator'
     *  Logic: '<S118>/Logical Operator'
     *  Logic: '<S118>/Logical Operator1'
     *  Logic: '<S118>/Logical Operator2'
     *  RelationalOperator: '<S118>/Relational Operator'
     *  RelationalOperator: '<S118>/Relational Operator1'
     *  RelationalOperator: '<S120>/Compare'
     *  RelationalOperator: '<S121>/Compare'
     */
    if (((Sim_Multi_X.CoreSubsys_p[ForEach_itr].Integrator_CSTATE_e <= phi_tmp) ||
         (phi < 0.0)) && ((phi > 0.0) || (Sim_Multi_X.CoreSubsys_p[ForEach_itr].
          Integrator_CSTATE_e >= Sim_Multi_ConstP.pooled10[ForEach_itr + 40])))
    {
      /* Switch: '<S118>/Switch' */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Switch = phi;
    } else {
      /* Switch: '<S118>/Switch' incorporates:
       *  Constant: '<S118>/Constant'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Switch = 0.0;
    }

    /* End of Switch: '<S118>/Switch' */

    /* Switch: '<S119>/Switch' */
    Sim_Multi_B.CoreSubsys_p[ForEach_itr].Switch_a = 0.0;
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      /* Gain: '<S116>/Conversion deg to rad' incorporates:
       *  Constant: '<S59>/Constant'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       */
      phi = 0.017453292519943295 * Sim_Multi_ConstP.pooled10[ForEach_itr];

      /* Abs: '<S116>/Abs' incorporates:
       *  Constant: '<S59>/Constant'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       */
      ctheta = std::abs(Sim_Multi_ConstP.pooled10[ForEach_itr + 4]);

      /* Sum: '<S116>/Subtract' incorporates:
       *  Constant: '<S59>/Constant'
       *  Constant: '<S59>/Constant1'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       *  Product: '<S116>/Product4'
       *  Reshape: '<S116>/Reshape'
       *  Trigonometry: '<S116>/Trigonometric Function'
       *  Trigonometry: '<S116>/Trigonometric Function1'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorfromrealCoGtopropellerBod[0] =
        std::cos(phi) * ctheta;
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorfromrealCoGtopropellerBod[1] =
        std::sin(phi) * ctheta;
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorfromrealCoGtopropellerBod[2] =
        Sim_Multi_ConstP.pooled10[ForEach_itr + 8] - 0.001;
    }

    /* Product: '<S135>/Product5' incorporates:
     *  Product: '<S134>/Product1'
     */
    phi_tmp = cphi * cphi;

    /* Product: '<S128>/Product7' incorporates:
     *  Constant: '<S59>/Constant'
     *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
     *  Product: '<S135>/Product4'
     *  Product: '<S135>/Product5'
     *  Product: '<S135>/Product6'
     *  Sum: '<S135>/Sum1'
     */
    phi = Sim_Multi_ConstP.pooled10[ForEach_itr + 24] * cphi +
      Sim_Multi_ConstP.pooled10[ForEach_itr + 28] * phi_tmp;
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      /* Gain: '<S129>/Conversion deg to rad' incorporates:
       *  Constant: '<S59>/Constant'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       *  Memory: '<S36>/Memory3'
       */
      rtb_Memory3_idx_0 = Sim_Multi_ConstP.pooled10[ForEach_itr + 48] *
        0.017453292519943295;
      rtb_Memory3_idx_1 = Sim_Multi_ConstP.pooled10[ForEach_itr + 52] *
        0.017453292519943295;
      rtb_Memory3_idx_2 = Sim_Multi_ConstP.pooled10[ForEach_itr + 56] *
        0.017453292519943295;

      /* Trigonometry: '<S160>/Trigonometric Function3' incorporates:
       *  Trigonometry: '<S163>/Trigonometric Function3'
       *  Trigonometry: '<S164>/Trigonometric Function'
       *  Trigonometry: '<S166>/Trigonometric Function4'
       *  Trigonometry: '<S167>/Trigonometric Function'
       */
      ctheta = std::cos(rtb_Memory3_idx_2);

      /* Trigonometry: '<S160>/Trigonometric Function1' incorporates:
       *  Trigonometry: '<S161>/Trigonometric Function1'
       *  Trigonometry: '<S165>/Trigonometric Function1'
       *  Trigonometry: '<S168>/Trigonometric Function1'
       */
      theta = std::cos(rtb_Memory3_idx_1);

      /* Product: '<S160>/Product' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       *  Trigonometry: '<S160>/Trigonometric Function1'
       *  Trigonometry: '<S160>/Trigonometric Function3'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[0] = theta *
        ctheta;

      /* Trigonometry: '<S163>/Trigonometric Function5' incorporates:
       *  Trigonometry: '<S164>/Trigonometric Function5'
       *  Trigonometry: '<S166>/Trigonometric Function12'
       *  Trigonometry: '<S168>/Trigonometric Function3'
       */
      cpsi = std::cos(rtb_Memory3_idx_0);

      /* Trigonometry: '<S163>/Trigonometric Function1' incorporates:
       *  Trigonometry: '<S162>/Trigonometric Function1'
       *  Trigonometry: '<S166>/Trigonometric Function2'
       */
      rtb_Memory3_idx_1 = std::sin(rtb_Memory3_idx_1);

      /* Trigonometry: '<S163>/Trigonometric Function12' incorporates:
       *  Trigonometry: '<S165>/Trigonometric Function3'
       *  Trigonometry: '<S166>/Trigonometric Function5'
       *  Trigonometry: '<S167>/Trigonometric Function5'
       */
      rtb_Memory3_idx_0 = std::sin(rtb_Memory3_idx_0);

      /* Trigonometry: '<S163>/Trigonometric Function' incorporates:
       *  Trigonometry: '<S161>/Trigonometric Function3'
       *  Trigonometry: '<S164>/Trigonometric Function4'
       *  Trigonometry: '<S166>/Trigonometric Function'
       *  Trigonometry: '<S167>/Trigonometric Function3'
       */
      rtb_Memory3_idx_2 = std::sin(rtb_Memory3_idx_2);

      /* Product: '<S163>/Product' incorporates:
       *  Product: '<S164>/Product1'
       *  Trigonometry: '<S163>/Trigonometric Function1'
       *  Trigonometry: '<S163>/Trigonometric Function12'
       */
      VectorConcatenate_tmp = rtb_Memory3_idx_0 * rtb_Memory3_idx_1;

      /* Sum: '<S163>/Sum' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       *  Product: '<S163>/Product'
       *  Product: '<S163>/Product1'
       *  Trigonometry: '<S163>/Trigonometric Function'
       *  Trigonometry: '<S163>/Trigonometric Function5'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[1] =
        VectorConcatenate_tmp * ctheta - cpsi * rtb_Memory3_idx_2;

      /* Product: '<S166>/Product1' incorporates:
       *  Product: '<S167>/Product'
       */
      VectorConcatenate_tmp_0 = cpsi * rtb_Memory3_idx_1;

      /* Sum: '<S166>/Sum' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       *  Product: '<S166>/Product1'
       *  Product: '<S166>/Product2'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[2] =
        VectorConcatenate_tmp_0 * ctheta + rtb_Memory3_idx_0 * rtb_Memory3_idx_2;

      /* Product: '<S161>/Product' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[3] = theta *
        rtb_Memory3_idx_2;

      /* Sum: '<S164>/Sum' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       *  Product: '<S164>/Product1'
       *  Product: '<S164>/Product2'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[4] =
        VectorConcatenate_tmp * rtb_Memory3_idx_2 + cpsi * ctheta;

      /* Sum: '<S167>/Sum' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       *  Product: '<S167>/Product'
       *  Product: '<S167>/Product1'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[5] =
        VectorConcatenate_tmp_0 * rtb_Memory3_idx_2 - rtb_Memory3_idx_0 * ctheta;

      /* Gain: '<S162>/Gain' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[6] =
        -rtb_Memory3_idx_1;

      /* Product: '<S165>/Product' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[7] =
        rtb_Memory3_idx_0 * theta;

      /* Product: '<S168>/Product' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[8] = cpsi * theta;
    }

    /* Sum: '<S112>/Sum1' incorporates:
     *  Integrator: '<S58>/omega'
     *  Product: '<S126>/Product'
     *  Product: '<S126>/Product1'
     *  Product: '<S126>/Product2'
     *  Product: '<S127>/Product'
     *  Product: '<S127>/Product1'
     *  Product: '<S127>/Product2'
     *  Sum: '<S113>/Sum1'
     *  Sum: '<S115>/Sum'
     */
    rtb_Memory3_idx_1 = (Sim_Multi_X.omega_CSTATE[1] *
                         Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                         VectorfromrealCoGtopropellerBod[2] -
                         Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                         VectorfromrealCoGtopropellerBod[1] *
                         Sim_Multi_X.omega_CSTATE[2]) +
      rtb_TrueairspeedBodyaxes[0];
    cpsi = (Sim_Multi_B.CoreSubsys_p[ForEach_itr].
            VectorfromrealCoGtopropellerBod[0] * Sim_Multi_X.omega_CSTATE[2] -
            Sim_Multi_X.omega_CSTATE[0] * Sim_Multi_B.CoreSubsys_p[ForEach_itr].
            VectorfromrealCoGtopropellerBod[2]) + rtb_TrueairspeedBodyaxes[1];
    theta = (Sim_Multi_X.omega_CSTATE[0] * Sim_Multi_B.CoreSubsys_p[ForEach_itr]
             .VectorfromrealCoGtopropellerBod[1] -
             Sim_Multi_B.CoreSubsys_p[ForEach_itr].
             VectorfromrealCoGtopropellerBod[0] * Sim_Multi_X.omega_CSTATE[1]) +
      rtb_TrueairspeedBodyaxes[2];

    /* Product: '<S129>/Product' */
    for (i = 0; i < 3; i++) {
      /* Product: '<S129>/Product' incorporates:
       *  Concatenate: '<S169>/Vector Concatenate'
       */
      rtb_TrueairspeedatpropMotoraxes[i] = (Sim_Multi_B.CoreSubsys_p[ForEach_itr]
        .VectorConcatenate[i + 3] * cpsi + Sim_Multi_B.CoreSubsys_p[ForEach_itr]
        .VectorConcatenate[i] * rtb_Memory3_idx_1) +
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[i + 6] * theta;
    }

    /* End of Product: '<S129>/Product' */

    /* If: '<S131>/If' incorporates:
     *  Math: '<S142>/transpose'
     *  Product: '<S142>/Product'
     *  Sqrt: '<S139>/Sqrt'
     */
    if (rtsiIsModeUpdateTimeStep(&(&Sim_Multi_M)->solverInfo)) {
      rtAction = static_cast<int8_T>(!(std::sqrt
        (rtb_TrueairspeedatpropMotoraxes[0] * rtb_TrueairspeedatpropMotoraxes[0]
         + rtb_TrueairspeedatpropMotoraxes[1] * rtb_TrueairspeedatpropMotoraxes
         [1]) == 0.0));
      Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem = rtAction;
    } else {
      rtAction = Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem;
    }

    if (rtAction == 0) {
      /* Outputs for IfAction SubSystem: '<S131>/Zero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S138>/Action Port'
       */
      if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
        /* Merge: '<S131>/Merge' incorporates:
         *  Constant: '<S138>/Constant'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0]
          = 0.0;

        /* Merge: '<S131>/Merge1' incorporates:
         *  Constant: '<S138>/Constant1'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[0]
          = 0.0;

        /* Merge: '<S131>/Merge' incorporates:
         *  Constant: '<S138>/Constant'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1]
          = 0.0;

        /* Merge: '<S131>/Merge1' incorporates:
         *  Constant: '<S138>/Constant1'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[1]
          = 0.0;

        /* Merge: '<S131>/Merge' incorporates:
         *  Constant: '<S138>/Constant'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2]
          = -1.0;

        /* Merge: '<S131>/Merge1' incorporates:
         *  Constant: '<S138>/Constant1'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[2]
          = 0.0;
      }

      /* End of Outputs for SubSystem: '<S131>/Zero airspeed in rotor plane' */
    } else {
      /* Outputs for IfAction SubSystem: '<S131>/Nonzero airspeed in rotor plane' incorporates:
       *  ActionPort: '<S137>/Action Port'
       */
      /* Sqrt: '<S140>/Sqrt' incorporates:
       *  Math: '<S141>/transpose'
       *  Product: '<S141>/Product'
       */
      ctheta = std::sqrt(rtb_TrueairspeedatpropMotoraxes[0] *
                         rtb_TrueairspeedatpropMotoraxes[0] +
                         rtb_TrueairspeedatpropMotoraxes[1] *
                         rtb_TrueairspeedatpropMotoraxes[1]);

      /* Gain: '<S137>/Conversion deg to rad' incorporates:
       *  Product: '<S137>/Product4'
       */
      theta = ctheta * 0.0 * 0.017453292519943295;

      /* Trigonometry: '<S137>/Trigonometric Function' */
      cpsi = std::sin(theta);

      /* Product: '<S137>/Divide' */
      rtb_Memory3_idx_0 = rtb_TrueairspeedatpropMotoraxes[0] / ctheta;
      rtb_Memory3_idx_2 = rtb_Memory3_idx_0;

      /* Product: '<S137>/Product2' incorporates:
       *  Gain: '<S137>/Gain'
       *  Product: '<S137>/Divide'
       *  Product: '<S137>/Product'
       */
      rtb_Memory3_idx_1 = -rtb_Memory3_idx_0 * cpsi;

      /* Product: '<S137>/Divide' */
      rtb_Memory3_idx_0 = rtb_TrueairspeedatpropMotoraxes[1] / ctheta;

      /* Product: '<S137>/Product2' incorporates:
       *  Gain: '<S137>/Gain'
       *  Product: '<S137>/Divide'
       *  Product: '<S137>/Product'
       */
      cpsi *= -rtb_Memory3_idx_0;

      /* Gain: '<S137>/Gain1' incorporates:
       *  Trigonometry: '<S137>/Trigonometric Function1'
       */
      ctheta = -std::cos(theta);

      /* Product: '<S137>/Product3' incorporates:
       *  Constant: '<S137>/Constant'
       *  Constant: '<S137>/Constant1'
       *  Gain: '<S137>/Gain2'
       *  Product: '<S137>/Divide'
       *  Product: '<S137>/Product1'
       */
      rtb_Memory3_idx_0 = -rtb_Memory3_idx_0 * 0.23 * theta;
      rtb_Memory3_idx_2 = rtb_Memory3_idx_2 * 0.23 * theta;
      theta *= 0.0;
      for (i = 0; i < 3; i++) {
        /* Product: '<S137>/Product2' incorporates:
         *  Concatenate: '<S169>/Vector Concatenate'
         */
        VectorConcatenate_tmp = Sim_Multi_B.CoreSubsys_p[ForEach_itr].
          VectorConcatenate[3 * i];
        VectorConcatenate_tmp_0 = Sim_Multi_B.CoreSubsys_p[ForEach_itr].
          VectorConcatenate[3 * i + 1];
        VectorConcatenate = Sim_Multi_B.CoreSubsys_p[ForEach_itr].
          VectorConcatenate[3 * i + 2];

        /* Merge: '<S131>/Merge' incorporates:
         *  Product: '<S137>/Product2'
         *  Reshape: '<S137>/Reshape1'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[i]
          = (VectorConcatenate_tmp_0 * cpsi + VectorConcatenate_tmp *
             rtb_Memory3_idx_1) + VectorConcatenate * ctheta;

        /* Merge: '<S131>/Merge1' incorporates:
         *  Product: '<S137>/Product3'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[i]
          = (VectorConcatenate_tmp_0 * rtb_Memory3_idx_2 + VectorConcatenate_tmp
             * rtb_Memory3_idx_0) + VectorConcatenate * theta;
      }

      /* End of Outputs for SubSystem: '<S131>/Nonzero airspeed in rotor plane' */
    }

    /* End of If: '<S131>/If' */

    /* Product: '<S128>/Product9' incorporates:
     *  Merge: '<S131>/Merge'
     */
    theta = phi * Sim_Multi_B.CoreSubsys_p[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[0];
    rtb_Memory3_idx_1 = phi * Sim_Multi_B.CoreSubsys_p[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[1];
    cpsi = phi * Sim_Multi_B.CoreSubsys_p[ForEach_itr].
      NewtiltedthrustdirectionBodyaxe[2];

    /* Sum: '<S134>/Sum' incorporates:
     *  Constant: '<S59>/Constant'
     *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
     *  Product: '<S134>/Product'
     *  Product: '<S134>/Product1'
     */
    phi = Sim_Multi_ConstP.pooled10[ForEach_itr + 32] * cphi +
      Sim_Multi_ConstP.pooled10[ForEach_itr + 36] * phi_tmp;
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      for (i = 0; i < 3; i++) {
        /* Product: '<S136>/Product9' incorporates:
         *  Concatenate: '<S169>/Vector Concatenate'
         *  Constant: '<S136>/Constant'
         *  Math: '<S136>/Math Function'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].Product9[i] =
          (Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[3 * i + 1] *
           0.0 + Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[3 * i] *
           0.0) - Sim_Multi_B.CoreSubsys_p[ForEach_itr].VectorConcatenate[3 * i
          + 2];
      }

      /* Gain: '<S156>/Gain' incorporates:
       *  Constant: '<S59>/Constant'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       */
      ctheta = Sim_Multi_ConstP.pooled10[ForEach_itr + 60] * 0.5;

      /* Gain: '<S156>/Gain1' incorporates:
       *  Constant: '<S59>/Constant'
       *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
       *  Product: '<S156>/Product7'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Gain1 = ctheta * ctheta *
        Sim_Multi_ConstP.pooled10[ForEach_itr + 64] * 0.58333333333333337;
    }

    /* Gain: '<S133>/Conversion rpm to rad//s' */
    cphi *= 0.10471975511965977;

    /* ForEachSliceSelector generated from: '<S112>/MotorMatrix_real' incorporates:
     *  Constant: '<S59>/Constant'
     *  Product: '<S128>/Product3'
     *  Product: '<S133>/Product5'
     */
    phi_tmp = Sim_Multi_ConstP.pooled10[ForEach_itr + 12];

    /* Product: '<S133>/Product5' incorporates:
     *  Constant: '<S59>/Constant'
     *  ForEachSliceSelector generated from: '<S112>/MotorMatrix_real'
     *  Integrator: '<S58>/omega'
     *  Product: '<S157>/Product'
     *  Product: '<S157>/Product1'
     *  Product: '<S157>/Product2'
     *  Product: '<S158>/Product'
     *  Product: '<S158>/Product1'
     *  Product: '<S158>/Product2'
     *  Sum: '<S155>/Sum'
     */
    ctheta = phi_tmp * Sim_Multi_B.CoreSubsys_p[ForEach_itr].Gain1;
    rtb_Sum_k1[0] = (Sim_Multi_X.omega_CSTATE[1] *
                     Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                     NewtiltedthrustdirectionBodyaxe[2] -
                     Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                     NewtiltedthrustdirectionBodyaxe[1] *
                     Sim_Multi_X.omega_CSTATE[2]) * ctheta * cphi;
    rtb_Sum_k1[1] = (Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                     NewtiltedthrustdirectionBodyaxe[0] *
                     Sim_Multi_X.omega_CSTATE[2] - Sim_Multi_X.omega_CSTATE[0] *
                     Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                     NewtiltedthrustdirectionBodyaxe[2]) * ctheta * cphi;
    rtb_Sum_k1[2] = (Sim_Multi_X.omega_CSTATE[0] *
                     Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                     NewtiltedthrustdirectionBodyaxe[1] -
                     Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                     NewtiltedthrustdirectionBodyaxe[0] *
                     Sim_Multi_X.omega_CSTATE[1]) * ctheta * cphi;

    /* If: '<S143>/If' incorporates:
     *  Math: '<S151>/transpose'
     *  Product: '<S129>/Product'
     *  Product: '<S151>/Product'
     *  Sqrt: '<S148>/Sqrt'
     */
    if (rtsiIsModeUpdateTimeStep(&(&Sim_Multi_M)->solverInfo)) {
      rtAction = static_cast<int8_T>(!(std::sqrt
        ((rtb_TrueairspeedatpropMotoraxes[0] * rtb_TrueairspeedatpropMotoraxes[0]
          + rtb_TrueairspeedatpropMotoraxes[1] *
          rtb_TrueairspeedatpropMotoraxes[1]) + rtb_TrueairspeedatpropMotoraxes
         [2] * rtb_TrueairspeedatpropMotoraxes[2]) == 0.0));
      Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_l = rtAction;
    } else {
      rtAction = Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_l;
    }

    if (rtAction == 0) {
      /* Outputs for IfAction SubSystem: '<S143>/Zero airspeed' incorporates:
       *  ActionPort: '<S147>/Action Port'
       */
      if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
        /* Merge: '<S143>/Merge' incorporates:
         *  Constant: '<S147>/Constant'
         */
        Sim_Multi_B.CoreSubsys_p[ForEach_itr].Angleofattackrad = 0.0;
      }

      /* End of Outputs for SubSystem: '<S143>/Zero airspeed' */
    } else {
      /* Outputs for IfAction SubSystem: '<S143>/Nonzero airspeed' incorporates:
       *  ActionPort: '<S146>/Action Port'
       */
      /* Merge: '<S143>/Merge' incorporates:
       *  Math: '<S150>/transpose'
       *  Product: '<S146>/Divide1'
       *  Product: '<S150>/Product'
       *  Sqrt: '<S149>/Sqrt'
       *  Trigonometry: '<S146>/Trigonometric Function'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Angleofattackrad = std::atan(1.0 /
        std::sqrt(rtb_TrueairspeedatpropMotoraxes[0] *
                  rtb_TrueairspeedatpropMotoraxes[0] +
                  rtb_TrueairspeedatpropMotoraxes[1] *
                  rtb_TrueairspeedatpropMotoraxes[1]) *
        rtb_TrueairspeedatpropMotoraxes[2]);

      /* End of Outputs for SubSystem: '<S143>/Nonzero airspeed' */
    }

    /* End of If: '<S143>/If' */

    /* Gain: '<S144>/Gain' */
    Sim_Multi_B.CoreSubsys_p[ForEach_itr].Climbspeedv_c =
      -rtb_TrueairspeedatpropMotoraxes[2];
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      /* Product: '<S145>/Divide' incorporates:
       *  Constant: '<S132>/Induced velocity at hover'
       */
      cphi = Sim_Multi_B.CoreSubsys_p[ForEach_itr].Climbspeedv_c / 4.0;

      /* If: '<S145>/If' */
      if (rtsiIsModeUpdateTimeStep(&(&Sim_Multi_M)->solverInfo)) {
        if (cphi >= 0.0) {
          Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_g = 0;
        } else if (cphi >= -2.0) {
          Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_g = 1;
        } else {
          Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_g = 2;
        }
      }

      /* End of If: '<S145>/If' */
    }

    /* Product: '<S128>/Product3' */
    phi *= phi_tmp;

    /* ForEachSliceAssignment generated from: '<S112>/Motor_moment' incorporates:
     *  Merge: '<S131>/Merge1'
     *  Product: '<S128>/Product3'
     *  Product: '<S136>/Product9'
     *  Product: '<S170>/Product'
     *  Product: '<S171>/Product'
     *  Sum: '<S117>/Add'
     *  Sum: '<S130>/Sum'
     *  Sum: '<S67>/Sum'
     */
    rtb_ImpAsg_InsertedFor_Motor_mo[3 * ForEach_itr] = ((phi *
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Product9[0] +
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[0])
      + rtb_Sum_k1[0]) + (Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                          VectorfromrealCoGtopropellerBod[1] * cpsi -
                          rtb_Memory3_idx_1 *
                          Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                          VectorfromrealCoGtopropellerBod[2]);

    /* ForEachSliceAssignment generated from: '<S112>/Motor_force' incorporates:
     *  Product: '<S128>/Product9'
     */
    rtb_ImpAsg_InsertedFor_Motor_fo[3 * ForEach_itr] = theta;

    /* ForEachSliceAssignment generated from: '<S112>/Motor_moment' incorporates:
     *  ForEachSliceAssignment generated from: '<S112>/Motor_force'
     *  Merge: '<S131>/Merge1'
     *  Product: '<S128>/Product3'
     *  Product: '<S136>/Product9'
     *  Product: '<S170>/Product1'
     *  Product: '<S171>/Product1'
     *  Sum: '<S117>/Add'
     *  Sum: '<S130>/Sum'
     *  Sum: '<S67>/Sum'
     */
    i = 3 * ForEach_itr + 1;
    rtb_ImpAsg_InsertedFor_Motor_mo[i] = ((phi *
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Product9[1] +
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[1])
      + rtb_Sum_k1[1]) + (theta * Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                          VectorfromrealCoGtopropellerBod[2] -
                          Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                          VectorfromrealCoGtopropellerBod[0] * cpsi);

    /* ForEachSliceAssignment generated from: '<S112>/Motor_force' incorporates:
     *  Product: '<S128>/Product9'
     */
    rtb_ImpAsg_InsertedFor_Motor_fo[i] = rtb_Memory3_idx_1;

    /* ForEachSliceAssignment generated from: '<S112>/Motor_moment' incorporates:
     *  ForEachSliceAssignment generated from: '<S112>/Motor_force'
     *  Merge: '<S131>/Merge1'
     *  Product: '<S128>/Product3'
     *  Product: '<S136>/Product9'
     *  Product: '<S170>/Product2'
     *  Product: '<S171>/Product2'
     *  Sum: '<S117>/Add'
     *  Sum: '<S130>/Sum'
     *  Sum: '<S67>/Sum'
     */
    i = 3 * ForEach_itr + 2;
    rtb_ImpAsg_InsertedFor_Motor_mo[i] = ((phi *
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Product9[2] +
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Momentinthemotorhubduetobending[2])
      + rtb_Sum_k1[2]) + (Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                          VectorfromrealCoGtopropellerBod[0] * rtb_Memory3_idx_1
                          - theta * Sim_Multi_B.CoreSubsys_p[ForEach_itr].
                          VectorfromrealCoGtopropellerBod[1]);

    /* ForEachSliceAssignment generated from: '<S112>/Motor_force' incorporates:
     *  Product: '<S128>/Product9'
     */
    rtb_ImpAsg_InsertedFor_Motor_fo[i] = cpsi;
  }

  /* End of Outputs for SubSystem: '<S94>/For Each Subsystem' */

  /* Sum: '<S94>/Sum of Elements1' incorporates:
   *  ForEachSliceAssignment generated from: '<S112>/Motor_moment'
   *  Sum: '<S67>/Sum'
   */
  for (i = 0; i < 3; i++) {
    rtb_Sum_k1[i] = ((rtb_ImpAsg_InsertedFor_Motor_mo[i + 3] +
                      rtb_ImpAsg_InsertedFor_Motor_mo[i]) +
                     rtb_ImpAsg_InsertedFor_Motor_mo[i + 6]) +
      rtb_ImpAsg_InsertedFor_Motor_mo[i + 9];
  }

  /* End of Sum: '<S94>/Sum of Elements1' */

  /* Sum: '<S63>/Sum1' incorporates:
   *  Sum: '<S60>/Sum2'
   *  Sum: '<S67>/Sum'
   *  Sum: '<S68>/Sum'
   */
  rtb_TrueairspeedBodyaxes[0] = rtb_Sum_k1[0] - (rtb_VectorConcatenate_g_tmp_1 -
    rtb_VectorConcatenate_g_tmp);
  rtb_TrueairspeedBodyaxes[1] = rtb_Sum_k1[1] - (rtb_VectorConcatenate_g_tmp_2 -
    rtb_VectorConcatenate_g_tmp_0);
  rtb_TrueairspeedBodyaxes[2] = rtb_Sum_k1[2] - (rtb_VectorConcatenate_g_tmp_3 -
    rtb_VectorConcatenate_g_tmp_4);

  /* Product: '<S63>/Product' incorporates:
   *  Constant: '<S59>/Constant3'
   */
  rt_mldivide_U1d3x3_U2d_JBYZyA3A(Sim_Multi_ConstP.pooled11,
    rtb_TrueairspeedBodyaxes, Sim_Multi_B.Product_l);

  /* Sum: '<S94>/Sum of Elements' incorporates:
   *  ForEachSliceAssignment generated from: '<S112>/Motor_force'
   *  Sum: '<S67>/Sum'
   */
  for (i = 0; i < 3; i++) {
    rtb_Sum_k1[i] = ((rtb_ImpAsg_InsertedFor_Motor_fo[i + 3] +
                      rtb_ImpAsg_InsertedFor_Motor_fo[i]) +
                     rtb_ImpAsg_InsertedFor_Motor_fo[i + 6]) +
      rtb_ImpAsg_InsertedFor_Motor_fo[i + 9];
  }

  /* End of Sum: '<S94>/Sum of Elements' */
  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    /* Product: '<S93>/Product1' incorporates:
     *  Constant: '<S59>/Constant2'
     *  Constant: '<S93>/Gravity (Inertial axes)'
     */
    Sim_Multi_B.ForceofgravityInertialaxes[0] = 0.0;
    Sim_Multi_B.ForceofgravityInertialaxes[1] = 0.0;
    Sim_Multi_B.ForceofgravityInertialaxes[2] = 5.3936575;
  }

  /* Sum: '<S96>/Sum1' incorporates:
   *  Integrator: '<S58>/V_b'
   *  Product: '<S111>/Product'
   */
  rtb_TrueairspeedBodyaxes_b[0] = Sim_Multi_X.V_b_CSTATE[0] -
    rtb_TrueairspeedBodyaxes_b[0];
  rtb_TrueairspeedBodyaxes_b[1] = Sim_Multi_X.V_b_CSTATE[1] -
    rtb_TrueairspeedBodyaxes_b[1];
  rtb_TrueairspeedBodyaxes_b[2] = Sim_Multi_X.V_b_CSTATE[2] -
    rtb_TrueairspeedBodyaxes_b[2];

  /* If: '<S95>/If' incorporates:
   *  Math: '<S110>/transpose'
   *  Product: '<S110>/Product'
   *  Sqrt: '<S99>/Sqrt'
   *  Sum: '<S96>/Sum1'
   */
  if (rtsiIsModeUpdateTimeStep(&(&Sim_Multi_M)->solverInfo)) {
    rtAction = static_cast<int8_T>(!(std::sqrt((rtb_TrueairspeedBodyaxes_b[0] *
      rtb_TrueairspeedBodyaxes_b[0] + rtb_TrueairspeedBodyaxes_b[1] *
      rtb_TrueairspeedBodyaxes_b[1]) + rtb_TrueairspeedBodyaxes_b[2] *
      rtb_TrueairspeedBodyaxes_b[2]) == 0.0));
    Sim_Multi_DW.If_ActiveSubsystem = rtAction;
  } else {
    rtAction = Sim_Multi_DW.If_ActiveSubsystem;
  }

  if (rtAction == 0) {
    /* Outputs for IfAction SubSystem: '<S95>/Zero airspeed' incorporates:
     *  ActionPort: '<S98>/Action Port'
     */
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      /* Merge: '<S95>/Merge' incorporates:
       *  Constant: '<S98>/Constant'
       */
      Sim_Multi_B.Forceagainstdirectionofmotiondu[0] = 0.0;
      Sim_Multi_B.Forceagainstdirectionofmotiondu[1] = 0.0;
      Sim_Multi_B.Forceagainstdirectionofmotiondu[2] = 0.0;
    }

    /* End of Outputs for SubSystem: '<S95>/Zero airspeed' */
  } else {
    /* Outputs for IfAction SubSystem: '<S95>/Nonzero airspeed' incorporates:
     *  ActionPort: '<S97>/Action Port'
     */
    /* Product: '<S105>/Divide' incorporates:
     *  Constant: '<S59>/Surface area params'
     */
    cphi = rtb_TrueairspeedBodyaxes_b[0] / 0.06;

    /* Product: '<S105>/Product' */
    phi = cphi * cphi;

    /* Product: '<S105>/Divide1' incorporates:
     *  Constant: '<S59>/Surface area params'
     */
    cphi = rtb_TrueairspeedBodyaxes_b[1] / 0.06;

    /* Product: '<S105>/Product1' */
    ctheta = cphi * cphi;

    /* Product: '<S105>/Divide2' incorporates:
     *  Constant: '<S59>/Surface area params'
     */
    cphi = rtb_TrueairspeedBodyaxes_b[2] / 0.06;

    /* Sum: '<S105>/Add' incorporates:
     *  Product: '<S105>/Product2'
     */
    cphi = (phi + ctheta) + cphi * cphi;

    /* Sqrt: '<S105>/Reciprocal Sqrt' */
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

    /* End of Sqrt: '<S105>/Reciprocal Sqrt' */

    /* Product: '<S106>/Product' incorporates:
     *  Sum: '<S96>/Sum1'
     */
    rtb_VectorConcatenate_g_tmp_1 = rtb_TrueairspeedBodyaxes_b[0] * cphi;

    /* Product: '<S108>/Product' incorporates:
     *  Math: '<S108>/transpose'
     *  Product: '<S106>/Product'
     */
    rtb_VectorConcatenate_g_tmp_2 = rtb_VectorConcatenate_g_tmp_1 *
      rtb_VectorConcatenate_g_tmp_1;

    /* Product: '<S106>/Product' incorporates:
     *  Sum: '<S96>/Sum1'
     */
    rtb_VectorConcatenate_g_tmp_1 = rtb_TrueairspeedBodyaxes_b[1] * cphi;

    /* Product: '<S108>/Product' incorporates:
     *  Math: '<S108>/transpose'
     *  Product: '<S106>/Product'
     */
    rtb_VectorConcatenate_g_tmp_2 += rtb_VectorConcatenate_g_tmp_1 *
      rtb_VectorConcatenate_g_tmp_1;

    /* Product: '<S106>/Product' incorporates:
     *  Sum: '<S96>/Sum1'
     */
    rtb_VectorConcatenate_g_tmp_1 = rtb_TrueairspeedBodyaxes_b[2] * cphi;

    /* Product: '<S109>/Product' incorporates:
     *  Product: '<S104>/Product'
     *  Product: '<S106>/Product'
     *  Sum: '<S96>/Sum1'
     */
    phi_tmp = (rtb_TrueairspeedBodyaxes_b[0] * rtb_TrueairspeedBodyaxes_b[0] +
               rtb_TrueairspeedBodyaxes_b[1] * rtb_TrueairspeedBodyaxes_b[1]) +
      rtb_TrueairspeedBodyaxes_b[2] * rtb_TrueairspeedBodyaxes_b[2];

    /* Abs: '<S97>/Abs' incorporates:
     *  Constant: '<S97>/Constant'
     *  Constant: '<S97>/Constant1'
     *  Constant: '<S97>/Constant2'
     *  Math: '<S108>/transpose'
     *  Product: '<S106>/Product'
     *  Product: '<S108>/Product'
     *  Product: '<S109>/Product'
     *  Product: '<S97>/Product'
     *  Sqrt: '<S107>/Sqrt'
     */
    phi = std::abs(phi_tmp * 0.6125 * 0.4 * std::sqrt
                   (rtb_VectorConcatenate_g_tmp_1 *
                    rtb_VectorConcatenate_g_tmp_1 +
                    rtb_VectorConcatenate_g_tmp_2));

    /* Sqrt: '<S103>/Sqrt' */
    cphi = std::sqrt(phi_tmp);

    /* Merge: '<S95>/Merge' incorporates:
     *  Gain: '<S97>/Drag force opposes direction of airspeed'
     *  Product: '<S100>/Divide'
     *  Product: '<S97>/Product1'
     *  Sum: '<S96>/Sum1'
     */
    Sim_Multi_B.Forceagainstdirectionofmotiondu[0] =
      -(rtb_TrueairspeedBodyaxes_b[0] / cphi * phi);
    Sim_Multi_B.Forceagainstdirectionofmotiondu[1] =
      -(rtb_TrueairspeedBodyaxes_b[1] / cphi * phi);
    Sim_Multi_B.Forceagainstdirectionofmotiondu[2] =
      -(rtb_TrueairspeedBodyaxes_b[2] / cphi * phi);

    /* End of Outputs for SubSystem: '<S95>/Nonzero airspeed' */
  }

  /* End of If: '<S95>/If' */

  /* SignalConversion generated from: '<S64>/Q-Integrator' incorporates:
   *  Gain: '<S64>/-1//2'
   *  Gain: '<S64>/1//2'
   *  Integrator: '<S58>/omega'
   *  Product: '<S64>/Product'
   *  Product: '<S73>/Product'
   *  Product: '<S76>/Product'
   *  Product: '<S76>/Product1'
   *  Product: '<S76>/Product2'
   *  Product: '<S77>/Product'
   *  Product: '<S77>/Product1'
   *  Product: '<S77>/Product2'
   *  Sum: '<S64>/Subtract'
   *  Sum: '<S72>/Sum'
   */
  Sim_Multi_B.TmpSignalConversionAtQIntegrato[0] = ((Sim_Multi_X.omega_CSTATE[0]
    * rtb_Divide_idx_1 + Sim_Multi_X.omega_CSTATE[1] * rtb_Divide_idx_2) +
    Sim_Multi_X.omega_CSTATE[2] * rtb_Divide_idx_3) * -0.5;
  Sim_Multi_B.TmpSignalConversionAtQIntegrato[1] = (rtb_Divide_idx_0 *
    Sim_Multi_X.omega_CSTATE[0] - (Sim_Multi_X.omega_CSTATE[1] *
    rtb_Divide_idx_3 - Sim_Multi_X.omega_CSTATE[2] * rtb_Divide_idx_2)) * 0.5;
  Sim_Multi_B.TmpSignalConversionAtQIntegrato[2] = (rtb_Divide_idx_0 *
    Sim_Multi_X.omega_CSTATE[1] - (rtb_Divide_idx_1 * Sim_Multi_X.omega_CSTATE[2]
    - Sim_Multi_X.omega_CSTATE[0] * rtb_Divide_idx_3)) * 0.5;
  Sim_Multi_B.TmpSignalConversionAtQIntegrato[3] = (rtb_Divide_idx_0 *
    Sim_Multi_X.omega_CSTATE[2] - (Sim_Multi_X.omega_CSTATE[0] *
    rtb_Divide_idx_2 - Sim_Multi_X.omega_CSTATE[1] * rtb_Divide_idx_1)) * 0.5;

  /* Sum: '<S67>/Sum' incorporates:
   *  Integrator: '<S58>/V_b'
   *  Integrator: '<S58>/omega'
   *  Product: '<S90>/Product'
   *  Product: '<S90>/Product1'
   *  Product: '<S90>/Product2'
   *  Product: '<S91>/Product'
   *  Product: '<S91>/Product1'
   *  Product: '<S91>/Product2'
   */
  rtb_TrueairspeedBodyaxes_b[0] = Sim_Multi_X.omega_CSTATE[1] *
    Sim_Multi_X.V_b_CSTATE[2];
  rtb_TrueairspeedBodyaxes_b[1] = Sim_Multi_X.V_b_CSTATE[0] *
    Sim_Multi_X.omega_CSTATE[2];
  rtb_TrueairspeedBodyaxes_b[2] = Sim_Multi_X.omega_CSTATE[0] *
    Sim_Multi_X.V_b_CSTATE[1];
  rtb_TrueairspeedBodyaxes[0] = Sim_Multi_X.V_b_CSTATE[1] *
    Sim_Multi_X.omega_CSTATE[2];
  rtb_TrueairspeedBodyaxes[1] = Sim_Multi_X.omega_CSTATE[0] *
    Sim_Multi_X.V_b_CSTATE[2];
  rtb_TrueairspeedBodyaxes[2] = Sim_Multi_X.V_b_CSTATE[0] *
    Sim_Multi_X.omega_CSTATE[1];

  /* Product: '<S93>/Product' */
  cphi = Sim_Multi_B.ForceofgravityInertialaxes[1];
  rtb_VectorConcatenate_g_tmp_1 = Sim_Multi_B.ForceofgravityInertialaxes[0];
  rtb_VectorConcatenate_g_tmp_2 = Sim_Multi_B.ForceofgravityInertialaxes[2];
  for (i = 0; i < 3; i++) {
    /* Sum: '<S58>/Sum1' incorporates:
     *  Concatenate: '<S89>/Vector Concatenate'
     *  Constant: '<S59>/Constant2'
     *  Merge: '<S95>/Merge'
     *  Product: '<S58>/Product1'
     *  Sum: '<S60>/Sum'
     *  Sum: '<S60>/Sum3'
     *  Sum: '<S67>/Sum'
     */
    Sim_Multi_B.Sum1_o[i] = ((((rtb_VectorConcatenate_k[i + 3] * cphi +
      rtb_VectorConcatenate_k[i] * rtb_VectorConcatenate_g_tmp_1) +
      rtb_VectorConcatenate_k[i + 6] * rtb_VectorConcatenate_g_tmp_2) +
      rtb_Sum_k1[i]) + Sim_Multi_B.Forceagainstdirectionofmotiondu[i]) / 0.55 -
      (rtb_TrueairspeedBodyaxes_b[i] - rtb_TrueairspeedBodyaxes[i]);

    /* Sum: '<S35>/Sum' incorporates:
     *  Integrator: '<S58>/omega'
     */
    Sim_Multi_B.omega[i] = Sim_Multi_X.omega_CSTATE[i];
  }

  /* End of Product: '<S93>/Product' */
  /* End of Outputs for SubSystem: '<Root>/multirotor' */

  /* Sum: '<S35>/Sum1' incorporates:
   *  Sum: '<S58>/Sum1'
   */
  Sim_Multi_B.a_b[0] = Sim_Multi_B.Sum1_o[0];
  Sim_Multi_B.a_b[1] = Sim_Multi_B.Sum1_o[1];
  Sim_Multi_B.a_b[2] = Sim_Multi_B.Sum1_o[2];

  /* RateTransition: '<S4>/Rate Transition' incorporates:
   *  Memory: '<S36>/Memory1'
   *  Memory: '<S36>/Memory2'
   *  Memory: '<S36>/Memory3'
   */
  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    if ((&Sim_Multi_M)->Timing.RateInteraction.TID1_2 == 1) {
      Sim_Multi_DW.RateTransition_Buffer_b[0] = rtb_Memory_idx_0;
      Sim_Multi_DW.RateTransition_Buffer_b[1] = rtb_Memory_idx_1;
      Sim_Multi_DW.RateTransition_Buffer_b[2] = rtb_Memory_idx_2;
      Sim_Multi_DW.RateTransition_Buffer_b[3] = rtb_Memory_idx_3;
      Sim_Multi_DW.RateTransition_Buffer_b[4] = Sim_Multi_B.omega[0];
      Sim_Multi_DW.RateTransition_Buffer_b[7] =
        Sim_Multi_DW.Memory1_PreviousInput[0];
      Sim_Multi_DW.RateTransition_Buffer_b[10] =
        Sim_Multi_DW.Memory2_PreviousInput[0];
      Sim_Multi_DW.RateTransition_Buffer_b[13] =
        Sim_Multi_DW.Memory3_PreviousInput[0];
      Sim_Multi_DW.RateTransition_Buffer_b[16] = Sim_Multi_B.a_b[0];
      Sim_Multi_DW.RateTransition_Buffer_b[5] = Sim_Multi_B.omega[1];
      Sim_Multi_DW.RateTransition_Buffer_b[8] =
        Sim_Multi_DW.Memory1_PreviousInput[1];
      Sim_Multi_DW.RateTransition_Buffer_b[11] =
        Sim_Multi_DW.Memory2_PreviousInput[1];
      Sim_Multi_DW.RateTransition_Buffer_b[14] =
        Sim_Multi_DW.Memory3_PreviousInput[1];
      Sim_Multi_DW.RateTransition_Buffer_b[17] = Sim_Multi_B.a_b[1];
      Sim_Multi_DW.RateTransition_Buffer_b[6] = Sim_Multi_B.omega[2];
      Sim_Multi_DW.RateTransition_Buffer_b[9] =
        Sim_Multi_DW.Memory1_PreviousInput[2];
      Sim_Multi_DW.RateTransition_Buffer_b[12] =
        Sim_Multi_DW.Memory2_PreviousInput[2];
      Sim_Multi_DW.RateTransition_Buffer_b[15] =
        Sim_Multi_DW.Memory3_PreviousInput[2];
      Sim_Multi_DW.RateTransition_Buffer_b[18] = Sim_Multi_B.a_b[2];
    }

    /* UniformRandomNumber: '<S37>/Uniform Random Number' incorporates:
     *  Memory: '<S36>/Memory1'
     *  Memory: '<S36>/Memory2'
     *  Memory: '<S36>/Memory3'
     */
    Sim_Multi_B.UniformRandomNumber =
      Sim_Multi_DW.UniformRandomNumber_NextOutput;

    /* UniformRandomNumber: '<S38>/Uniform Random Number' */
    Sim_Multi_B.UniformRandomNumber_n =
      Sim_Multi_DW.UniformRandomNumber_NextOutpu_m;

    /* Switch: '<S42>/Switch' incorporates:
     *  Constant: '<S42>/Constant1'
     */
    Sim_Multi_B.Switch = 0.0;

    /* Switch: '<S45>/Switch' incorporates:
     *  Constant: '<S45>/Constant1'
     */
    Sim_Multi_B.Switch_j = 0.0;

    /* Switch: '<S45>/Switch1' incorporates:
     *  Constant: '<S45>/Constant3'
     */
    Sim_Multi_B.Switch1 = 0.0;

    /* Switch: '<S45>/Switch2' incorporates:
     *  Constant: '<S45>/Constant5'
     */
    Sim_Multi_B.Switch2 = 0.0;
  }

  /* End of RateTransition: '<S4>/Rate Transition' */

  /* Sum: '<S36>/Sum' incorporates:
   *  Product: '<S75>/Divide'
   */
  rtb_Memory_idx_0 = rtb_Divide_idx_0 + Sim_Multi_B.Switch;
  rtb_Divide_idx_0 = rtb_Memory_idx_0;

  /* Product: '<S47>/Product' incorporates:
   *  Math: '<S47>/transpose'
   *  Sum: '<S36>/Sum'
   */
  rtb_Memory_idx_1 = rtb_Memory_idx_0 * rtb_Memory_idx_0;

  /* Sum: '<S36>/Sum' incorporates:
   *  Product: '<S75>/Divide'
   */
  rtb_Memory_idx_0 = rtb_Divide_idx_1 + Sim_Multi_B.Switch;
  rtb_Divide_idx_1 = rtb_Memory_idx_0;

  /* Product: '<S47>/Product' incorporates:
   *  Math: '<S47>/transpose'
   *  Sum: '<S36>/Sum'
   */
  rtb_Memory_idx_1 += rtb_Memory_idx_0 * rtb_Memory_idx_0;

  /* Sum: '<S36>/Sum' incorporates:
   *  Product: '<S75>/Divide'
   */
  rtb_Memory_idx_0 = rtb_Divide_idx_2 + Sim_Multi_B.Switch;
  rtb_Divide_idx_2 = rtb_Memory_idx_0;

  /* Product: '<S47>/Product' incorporates:
   *  Math: '<S47>/transpose'
   *  Sum: '<S36>/Sum'
   */
  rtb_Memory_idx_1 += rtb_Memory_idx_0 * rtb_Memory_idx_0;

  /* Sum: '<S36>/Sum' incorporates:
   *  Product: '<S75>/Divide'
   */
  rtb_Memory_idx_0 = rtb_Divide_idx_3 + Sim_Multi_B.Switch;

  /* Sqrt: '<S46>/Sqrt' incorporates:
   *  Math: '<S47>/transpose'
   *  Product: '<S47>/Product'
   *  Sum: '<S36>/Sum'
   */
  cphi = std::sqrt(rtb_Memory_idx_0 * rtb_Memory_idx_0 + rtb_Memory_idx_1);

  /* Product: '<S41>/Divide' incorporates:
   *  Sum: '<S36>/Sum'
   */
  Sim_Multi_B.Divide[0] = rtb_Divide_idx_0 / cphi;
  Sim_Multi_B.Divide[1] = rtb_Divide_idx_1 / cphi;
  Sim_Multi_B.Divide[2] = rtb_Divide_idx_2 / cphi;
  Sim_Multi_B.Divide[3] = rtb_Memory_idx_0 / cphi;

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  /* Sum: '<S36>/Sum1' incorporates:
   *  Integrator: '<S58>/X_i'
   */
  Sim_Multi_B.Sum1[0] = Sim_Multi_X.X_i_CSTATE[0] + Sim_Multi_B.Switch_j;
  Sim_Multi_B.Sum1[1] = Sim_Multi_X.X_i_CSTATE[1] + Sim_Multi_B.Switch1;
  Sim_Multi_B.Sum1[2] = Sim_Multi_X.X_i_CSTATE[2] + Sim_Multi_B.Switch2;

  /* End of Outputs for SubSystem: '<Root>/multirotor' */
  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    /* Switch: '<S44>/Switch' incorporates:
     *  Constant: '<S44>/Constant1'
     */
    Sim_Multi_B.Switch_jm = 0.0;

    /* Switch: '<S44>/Switch1' incorporates:
     *  Constant: '<S44>/Constant3'
     */
    Sim_Multi_B.Switch1_m = 0.0;

    /* Switch: '<S44>/Switch2' incorporates:
     *  Constant: '<S44>/Constant5'
     */
    Sim_Multi_B.Switch2_m = 0.0;

    /* Switch: '<S43>/Switch' incorporates:
     *  Constant: '<S43>/Constant1'
     */
    Sim_Multi_B.Switch_f = 0.0;

    /* Switch: '<S43>/Switch1' incorporates:
     *  Constant: '<S43>/Constant3'
     */
    Sim_Multi_B.Switch1_d = 0.0;

    /* Switch: '<S43>/Switch2' incorporates:
     *  Constant: '<S43>/Constant5'
     */
    Sim_Multi_B.Switch2_n = 0.0;
  }

  /* Sum: '<S36>/Sum2' incorporates:
   *  Product: '<S61>/Product'
   */
  Sim_Multi_B.Sum2[0] = Sim_Multi_B.Product[0] + Sim_Multi_B.Switch_jm;
  Sim_Multi_B.Sum2[1] = Sim_Multi_B.Product[1] + Sim_Multi_B.Switch1_m;
  Sim_Multi_B.Sum2[2] = Sim_Multi_B.Product[2] + Sim_Multi_B.Switch2_m;

  /* Outputs for Atomic SubSystem: '<Root>/multirotor' */
  /* Sum: '<S36>/Sum3' incorporates:
   *  Integrator: '<S58>/V_b'
   */
  Sim_Multi_B.Sum3[0] = Sim_Multi_X.V_b_CSTATE[0] + Sim_Multi_B.Switch_f;
  Sim_Multi_B.Sum3[1] = Sim_Multi_X.V_b_CSTATE[1] + Sim_Multi_B.Switch1_d;
  Sim_Multi_B.Sum3[2] = Sim_Multi_X.V_b_CSTATE[2] + Sim_Multi_B.Switch2_n;

  /* End of Outputs for SubSystem: '<Root>/multirotor' */
  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
      /* Update for Memory: '<S36>/Memory' incorporates:
       *  Product: '<S41>/Divide'
       */
      Sim_Multi_DW.Memory_PreviousInput[0] = Sim_Multi_B.Divide[0];
      Sim_Multi_DW.Memory_PreviousInput[1] = Sim_Multi_B.Divide[1];
      Sim_Multi_DW.Memory_PreviousInput[2] = Sim_Multi_B.Divide[2];
      Sim_Multi_DW.Memory_PreviousInput[3] = Sim_Multi_B.Divide[3];

      /* Update for Memory: '<S36>/Memory1' incorporates:
       *  Sum: '<S36>/Sum1'
       */
      Sim_Multi_DW.Memory1_PreviousInput[0] = Sim_Multi_B.Sum1[0];

      /* Update for Memory: '<S36>/Memory2' incorporates:
       *  Sum: '<S36>/Sum2'
       */
      Sim_Multi_DW.Memory2_PreviousInput[0] = Sim_Multi_B.Sum2[0];

      /* Update for Memory: '<S36>/Memory3' incorporates:
       *  Sum: '<S36>/Sum3'
       */
      Sim_Multi_DW.Memory3_PreviousInput[0] = Sim_Multi_B.Sum3[0];

      /* Update for Memory: '<S36>/Memory1' incorporates:
       *  Sum: '<S36>/Sum1'
       */
      Sim_Multi_DW.Memory1_PreviousInput[1] = Sim_Multi_B.Sum1[1];

      /* Update for Memory: '<S36>/Memory2' incorporates:
       *  Sum: '<S36>/Sum2'
       */
      Sim_Multi_DW.Memory2_PreviousInput[1] = Sim_Multi_B.Sum2[1];

      /* Update for Memory: '<S36>/Memory3' incorporates:
       *  Sum: '<S36>/Sum3'
       */
      Sim_Multi_DW.Memory3_PreviousInput[1] = Sim_Multi_B.Sum3[1];

      /* Update for Memory: '<S36>/Memory1' incorporates:
       *  Sum: '<S36>/Sum1'
       */
      Sim_Multi_DW.Memory1_PreviousInput[2] = Sim_Multi_B.Sum1[2];

      /* Update for Memory: '<S36>/Memory2' incorporates:
       *  Sum: '<S36>/Sum2'
       */
      Sim_Multi_DW.Memory2_PreviousInput[2] = Sim_Multi_B.Sum2[2];

      /* Update for Memory: '<S36>/Memory3' incorporates:
       *  Sum: '<S36>/Sum3'
       */
      Sim_Multi_DW.Memory3_PreviousInput[2] = Sim_Multi_B.Sum3[2];

      /* Update for UniformRandomNumber: '<S37>/Uniform Random Number' */
      Sim_Multi_DW.UniformRandomNumber_NextOutput = rt_urand_Upu32_Yd_f_pw_snf
        (&Sim_Multi_DW.RandSeed) * 0.002 - 0.001;

      /* Update for UniformRandomNumber: '<S38>/Uniform Random Number' */
      Sim_Multi_DW.UniformRandomNumber_NextOutpu_m = rt_urand_Upu32_Yd_f_pw_snf(
        &Sim_Multi_DW.RandSeed_b) * 0.002 - 0.001;
    }

    /* Update for Atomic SubSystem: '<Root>/multirotor' */
    /* Update for Integrator: '<S64>/Q-Integrator' */
    Sim_Multi_DW.QIntegrator_IWORK = 0;

    /* End of Update for SubSystem: '<Root>/multirotor' */
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep((&Sim_Multi_M))) {
    rt_ertODEUpdateContinuousStates(&(&Sim_Multi_M)->solverInfo);

    /* Update absolute time */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     */
    ++(&Sim_Multi_M)->Timing.clockTick0;
    (&Sim_Multi_M)->Timing.t[0] = rtsiGetSolverStopTime(&(&Sim_Multi_M)
      ->solverInfo);

    /* Update absolute time */
    /* The "clockTick1" counts the number of times the code of this task has
     * been executed. The resolution of this integer timer is 0.001, which is the step size
     * of the task. Size of "clockTick1" ensures timer will not overflow during the
     * application lifespan selected.
     */
    (&Sim_Multi_M)->Timing.clockTick1++;
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void Sim_Multi::Sim_Multi_derivatives()
{
  /* local scratch DWork variables */
  int32_T ForEach_itr;
  XDot_Sim_Multi_T *_rtXdot;
  _rtXdot = ((XDot_Sim_Multi_T *) (&Sim_Multi_M)->derivs);

  /* Derivatives for TransferFcn: '<S31>/Transfer Fcn1' */
  _rtXdot->TransferFcn1_CSTATE = -100.0 * Sim_Multi_X.TransferFcn1_CSTATE;
  _rtXdot->TransferFcn1_CSTATE += Sim_Multi_B.TransferFcn6;

  /* Derivatives for TransferFcn: '<S31>/Transfer Fcn6' */
  _rtXdot->TransferFcn6_CSTATE = -100.0 * Sim_Multi_X.TransferFcn6_CSTATE;
  _rtXdot->TransferFcn6_CSTATE += Sim_Multi_B.Gain2_i;

  /* Derivatives for TransferFcn: '<S31>/Transfer Fcn4' */
  _rtXdot->TransferFcn4_CSTATE = -100.0 * Sim_Multi_X.TransferFcn4_CSTATE;
  _rtXdot->TransferFcn4_CSTATE += Sim_Multi_B.Gain2;

  /* Derivatives for TransferFcn: '<S31>/Transfer Fcn5' */
  _rtXdot->TransferFcn5_CSTATE = -100.0 * Sim_Multi_X.TransferFcn5_CSTATE;
  _rtXdot->TransferFcn5_CSTATE += Sim_Multi_B.TransferFcn4;

  /* Derivatives for TransferFcn: '<S31>/Transfer Fcn2' */
  _rtXdot->TransferFcn2_CSTATE = -100.0 * Sim_Multi_X.TransferFcn2_CSTATE;
  _rtXdot->TransferFcn2_CSTATE += Sim_Multi_B.Gain2_g;

  /* Derivatives for TransferFcn: '<S31>/Transfer Fcn3' */
  _rtXdot->TransferFcn3_CSTATE = -100.0 * Sim_Multi_X.TransferFcn3_CSTATE;
  _rtXdot->TransferFcn3_CSTATE += Sim_Multi_B.TransferFcn2;

  /* Derivatives for Atomic SubSystem: '<Root>/multirotor' */
  /* Derivatives for Integrator: '<S64>/Q-Integrator' incorporates:
   *  SignalConversion generated from: '<S64>/Q-Integrator'
   */
  _rtXdot->QIntegrator_CSTATE[0] = Sim_Multi_B.TmpSignalConversionAtQIntegrato[0];
  _rtXdot->QIntegrator_CSTATE[1] = Sim_Multi_B.TmpSignalConversionAtQIntegrato[1];
  _rtXdot->QIntegrator_CSTATE[2] = Sim_Multi_B.TmpSignalConversionAtQIntegrato[2];
  _rtXdot->QIntegrator_CSTATE[3] = Sim_Multi_B.TmpSignalConversionAtQIntegrato[3];

  /* Derivatives for Integrator: '<S58>/V_b' incorporates:
   *  Sum: '<S58>/Sum1'
   */
  _rtXdot->V_b_CSTATE[0] = Sim_Multi_B.Sum1_o[0];

  /* Derivatives for Integrator: '<S58>/omega' incorporates:
   *  Product: '<S63>/Product'
   */
  _rtXdot->omega_CSTATE[0] = Sim_Multi_B.Product_l[0];

  /* Derivatives for Integrator: '<S58>/V_b' incorporates:
   *  Sum: '<S58>/Sum1'
   */
  _rtXdot->V_b_CSTATE[1] = Sim_Multi_B.Sum1_o[1];

  /* Derivatives for Integrator: '<S58>/omega' incorporates:
   *  Product: '<S63>/Product'
   */
  _rtXdot->omega_CSTATE[1] = Sim_Multi_B.Product_l[1];

  /* Derivatives for Integrator: '<S58>/V_b' incorporates:
   *  Sum: '<S58>/Sum1'
   */
  _rtXdot->V_b_CSTATE[2] = Sim_Multi_B.Sum1_o[2];

  /* Derivatives for Integrator: '<S58>/omega' incorporates:
   *  Product: '<S63>/Product'
   */
  _rtXdot->omega_CSTATE[2] = Sim_Multi_B.Product_l[2];

  /* Derivatives for Iterator SubSystem: '<S94>/For Each Subsystem' */
  for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
    /* Derivatives for Integrator: '<S118>/Integrator' */
    _rtXdot->CoreSubsys_p[ForEach_itr].Integrator_CSTATE_e =
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Switch;

    /* Derivatives for Integrator: '<S119>/Integrator' */
    _rtXdot->CoreSubsys_p[ForEach_itr].Integrator_CSTATE_o =
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].Switch_a;
  }

  /* End of Derivatives for SubSystem: '<S94>/For Each Subsystem' */

  /* Derivatives for Integrator: '<S58>/X_i' incorporates:
   *  Product: '<S61>/Product'
   */
  _rtXdot->X_i_CSTATE[0] = Sim_Multi_B.Product[0];
  _rtXdot->X_i_CSTATE[1] = Sim_Multi_B.Product[1];
  _rtXdot->X_i_CSTATE[2] = Sim_Multi_B.Product[2];

  /* End of Derivatives for SubSystem: '<Root>/multirotor' */

  /* Derivatives for Integrator: '<S37>/Integrator' */
  _rtXdot->Integrator_CSTATE = Sim_Multi_B.UniformRandomNumber;

  /* Derivatives for Integrator: '<S38>/Integrator' */
  _rtXdot->Integrator_CSTATE_f = Sim_Multi_B.UniformRandomNumber_n;
}

/* Model step function for TID2 */
void Sim_Multi::step2()                /* Sample time: [0.002s, 0.0s] */
{
  /* local block i/o variables */
  real_T rtb_Product[3];

  /* local scratch DWork variables */
  int32_T ForEach_itr_f;
  real_T rtb_MatrixMultiply1[4];
  real_T rtb_Product_c[3];
  real_T rtb_Sum3_l[3];
  real_T rtb_Switch1[3];
  real_T rtb_Switch1_0[3];
  real_T rtb_Switch1_1[3];
  real_T RateTransition;
  real_T RateTransition_Buffer_b;
  real_T RateTransition_Buffer_b_0;
  real_T RateTransition_Buffer_b_1;
  real_T rtb_Throttlecommandthrustvalue;
  real_T u1;
  real_T u2;

  /* RateTransition: '<S3>/Rate Transition' incorporates:
   *  Constant: '<S30>/Throttle value'
   */
  Sim_Multi_B.RateTransition[3] = 299.9916292904752;

  /* Sum: '<S10>/Sum4' incorporates:
   *  Gain: '<S10>/Unit conversion [stick value] to [N]'
   */
  rtb_Throttlecommandthrustvalue = 0.017979360000000003 *
    Sim_Multi_B.RateTransition[3];

  /* Product: '<S11>/Product1' incorporates:
   *  Constant: '<S2>/Constant7'
   *  RateTransition: '<S4>/Rate Transition'
   */
  RateTransition_Buffer_b = Sim_Multi_DW.RateTransition_Buffer_b[5];
  RateTransition_Buffer_b_0 = Sim_Multi_DW.RateTransition_Buffer_b[4];
  RateTransition_Buffer_b_1 = Sim_Multi_DW.RateTransition_Buffer_b[6];
  for (int32_T i{0}; i < 3; i++) {
    /* RateTransition: '<S3>/Rate Transition' incorporates:
     *  RateTransition: '<S30>/Rate Transition'
     */
    RateTransition = Sim_Multi_DW.RateTransition_Buffer[i];
    Sim_Multi_B.RateTransition[i] = RateTransition;

    /* Product: '<S11>/Product1' incorporates:
     *  Constant: '<S2>/Constant7'
     *  RateTransition: '<S4>/Rate Transition'
     *  Switch: '<S11>/Switch1'
     */
    rtb_Switch1[i] = (Sim_Multi_ConstP.pooled11[i + 3] * RateTransition_Buffer_b
                      + Sim_Multi_ConstP.pooled11[i] * RateTransition_Buffer_b_0)
      + Sim_Multi_ConstP.pooled11[i + 6] * RateTransition_Buffer_b_1;

    /* Gain: '<S10>/Unit conversion [stick value] to [rad//s]' */
    RateTransition *= Sim_Multi_ConstP.Unitconversionstickvaluetorads_[i];

    /* Saturate: '<S11>/Saturation' */
    u1 = Sim_Multi_ConstP.Saturation_LowerSat[i];
    u2 = Sim_Multi_ConstP.Saturation_UpperSat[i];
    if (RateTransition > u2) {
      rtb_Sum3_l[i] = u2;
    } else if (RateTransition < u1) {
      rtb_Sum3_l[i] = u1;
    } else {
      rtb_Sum3_l[i] = RateTransition;
    }

    /* End of Saturate: '<S11>/Saturation' */
  }

  /* Product: '<S13>/Product' */
  RateTransition_Buffer_b = rtb_Sum3_l[1];
  RateTransition_Buffer_b_0 = rtb_Sum3_l[0];
  RateTransition_Buffer_b_1 = rtb_Sum3_l[2];

  /* Sum: '<S11>/Sum3' incorporates:
   *  RateTransition: '<S4>/Rate Transition'
   */
  rtb_Sum3_l[0] -= Sim_Multi_DW.RateTransition_Buffer_b[4];
  rtb_Sum3_l[1] -= Sim_Multi_DW.RateTransition_Buffer_b[5];
  rtb_Sum3_l[2] -= Sim_Multi_DW.RateTransition_Buffer_b[6];

  /* Product: '<S14>/Product' incorporates:
   *  Sum: '<S11>/Sum3'
   */
  RateTransition = rtb_Sum3_l[1];
  u1 = rtb_Sum3_l[0];
  u2 = rtb_Sum3_l[2];
  for (int32_T i{0}; i < 3; i++) {
    /* Product: '<S13>/Product' incorporates:
     *  Product: '<S13>/Product1'
     */
    rtb_Product_c[i] = (Sim_Multi_ConstB.Product1[i + 3] *
                        RateTransition_Buffer_b + Sim_Multi_ConstB.Product1[i] *
                        RateTransition_Buffer_b_0) + Sim_Multi_ConstB.Product1[i
      + 6] * RateTransition_Buffer_b_1;

    /* Product: '<S14>/Product' incorporates:
     *  Product: '<S14>/Product1'
     *  Sum: '<S11>/Sum3'
     */
    rtb_Product[i] = (Sim_Multi_ConstB.Product1_i[i + 3] * RateTransition +
                      Sim_Multi_ConstB.Product1_i[i] * u1) +
      Sim_Multi_ConstB.Product1_i[i + 6] * u2;
  }

  /* Switch: '<S29>/Switch2' incorporates:
   *  RelationalOperator: '<S29>/LowerRelop1'
   *  RelationalOperator: '<S29>/UpperRelop'
   *  Switch: '<S29>/Switch'
   */
  if (rtb_Throttlecommandthrustvalue > 16.181424000000003) {
    rtb_Throttlecommandthrustvalue = 16.181424000000003;
  } else if (rtb_Throttlecommandthrustvalue < 1.7979360000000004) {
    /* Switch: '<S29>/Switch' */
    rtb_Throttlecommandthrustvalue = 1.7979360000000004;
  }

  /* End of Switch: '<S29>/Switch2' */

  /* Sum: '<S12>/Sum' incorporates:
   *  Product: '<S16>/Product'
   *  Product: '<S16>/Product1'
   *  Product: '<S16>/Product2'
   *  Product: '<S17>/Product'
   *  Product: '<S17>/Product1'
   *  Product: '<S17>/Product2'
   *  RateTransition: '<S4>/Rate Transition'
   */
  rtb_Switch1_0[0] = rtb_Switch1[2] * Sim_Multi_DW.RateTransition_Buffer_b[5];
  rtb_Switch1_0[1] = rtb_Switch1[0] * Sim_Multi_DW.RateTransition_Buffer_b[6];
  rtb_Switch1_0[2] = rtb_Switch1[1] * Sim_Multi_DW.RateTransition_Buffer_b[4];
  rtb_Switch1_1[0] = rtb_Switch1[1] * Sim_Multi_DW.RateTransition_Buffer_b[6];
  rtb_Switch1_1[1] = rtb_Switch1[2] * Sim_Multi_DW.RateTransition_Buffer_b[4];
  rtb_Switch1_1[2] = rtb_Switch1[0] * Sim_Multi_DW.RateTransition_Buffer_b[5];

  /* Product: '<S15>/Product' incorporates:
   *  Sum: '<S11>/Sum3'
   */
  RateTransition_Buffer_b = rtb_Sum3_l[1];
  RateTransition_Buffer_b_0 = rtb_Sum3_l[0];
  RateTransition_Buffer_b_1 = rtb_Sum3_l[2];
  for (int32_T i{0}; i < 3; i++) {
    /* Product: '<S8>/Matrix Multiply1' incorporates:
     *  DiscreteIntegrator: '<S11>/Discrete-Time Integrator'
     *  Product: '<S15>/Product'
     *  Product: '<S15>/Product1'
     *  Sum: '<S11>/Sum'
     *  Sum: '<S11>/Sum1'
     *  Sum: '<S11>/Sum2'
     *  Sum: '<S11>/Sum3'
     *  Sum: '<S12>/Sum'
     */
    rtb_MatrixMultiply1[i] = ((((Sim_Multi_ConstB.Product1_g[i + 3] *
      RateTransition_Buffer_b + Sim_Multi_ConstB.Product1_g[i] *
      RateTransition_Buffer_b_0) + Sim_Multi_ConstB.Product1_g[i + 6] *
      RateTransition_Buffer_b_1) + Sim_Multi_DW.DiscreteTimeIntegrator_DSTATE[i])
      - rtb_Product_c[i]) + (rtb_Switch1_0[i] - rtb_Switch1_1[i]);
  }

  /* Product: '<S8>/Matrix Multiply1' incorporates:
   *  Constant: '<S8>/Constant1'
   */
  RateTransition_Buffer_b = rtb_MatrixMultiply1[1];
  RateTransition_Buffer_b_0 = rtb_MatrixMultiply1[0];
  RateTransition_Buffer_b_1 = rtb_MatrixMultiply1[2];
  for (int32_T i{0}; i < 4; i++) {
    /* Product: '<S8>/Matrix Multiply1' incorporates:
     *  SignalConversion generated from: '<S8>/Matrix Multiply1'
     */
    rtb_MatrixMultiply1[i] = ((Sim_Multi_ConstP.Constant1_Value_n[i + 4] *
      RateTransition_Buffer_b + Sim_Multi_ConstP.Constant1_Value_n[i] *
      RateTransition_Buffer_b_0) + Sim_Multi_ConstP.Constant1_Value_n[i + 8] *
      RateTransition_Buffer_b_1) + Sim_Multi_ConstP.Constant1_Value_n[i + 12] *
      rtb_Throttlecommandthrustvalue;
  }

  /* Outputs for Iterator SubSystem: '<S7>/For Each Subsystem' incorporates:
   *  ForEach: '<S24>/For Each'
   */
  for (ForEach_itr_f = 0; ForEach_itr_f < 4; ForEach_itr_f++) {
    /* SignalConversion generated from: '<S25>/ SFunction ' incorporates:
     *  Constant: '<S2>/Constant4'
     *  ForEachSliceSelector generated from: '<S24>/MotorMatrix_nominal'
     *  MATLAB Function: '<S24>/MATLAB Function'
     */
    rtb_Switch1[0] = Sim_Multi_ConstP.pooled10[ForEach_itr_f + 24];
    rtb_Switch1[1] = Sim_Multi_ConstP.pooled10[ForEach_itr_f + 28];

    /* MATLAB Function: '<S24>/MATLAB Function' incorporates:
     *  SignalConversion generated from: '<S25>/ SFunction '
     */
    rtb_Throttlecommandthrustvalue = rtb_Switch1[0] / (2.0 * rtb_Switch1[1]);

    /* Saturate: '<S24>/Saturation limit: no negative thrust' incorporates:
     *  ForEachSliceSelector generated from: '<S24>/Thrust_cmds'
     *  Product: '<S8>/Matrix Multiply1'
     */
    if (rtb_MatrixMultiply1[ForEach_itr_f] <= 0.0) {
      RateTransition_Buffer_b = 0.0;
    } else {
      RateTransition_Buffer_b = rtb_MatrixMultiply1[ForEach_itr_f];
    }

    /* MATLAB Function: '<S24>/MATLAB Function' incorporates:
     *  Saturate: '<S24>/Saturation limit: no negative thrust'
     *  SignalConversion generated from: '<S25>/ SFunction '
     */
    rtb_Throttlecommandthrustvalue = -rtb_Switch1[0] / (2.0 * rtb_Switch1[1]) +
      std::sqrt(rtb_Throttlecommandthrustvalue * rtb_Throttlecommandthrustvalue
                + RateTransition_Buffer_b / rtb_Switch1[1]);

    /* ForEachSliceSelector generated from: '<S24>/MotorMatrix_nominal' incorporates:
     *  Constant: '<S2>/Constant4'
     *  RelationalOperator: '<S26>/LowerRelop1'
     *  Switch: '<S26>/Switch2'
     */
    RateTransition_Buffer_b = Sim_Multi_ConstP.pooled10[ForEach_itr_f + 44];

    /* Switch: '<S26>/Switch2' incorporates:
     *  Constant: '<S2>/Constant4'
     *  ForEachSliceSelector generated from: '<S24>/MotorMatrix_nominal'
     *  RelationalOperator: '<S26>/LowerRelop1'
     */
    if (rtb_Throttlecommandthrustvalue > RateTransition_Buffer_b) {
      /* ForEachSliceAssignment generated from: '<S24>/RPM_cmd_sat' */
      Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[ForEach_itr_f] =
        RateTransition_Buffer_b;
    } else {
      /* RelationalOperator: '<S26>/UpperRelop' incorporates:
       *  Switch: '<S26>/Switch'
       */
      RateTransition_Buffer_b = Sim_Multi_ConstP.pooled10[ForEach_itr_f + 40];

      /* Switch: '<S26>/Switch' incorporates:
       *  RelationalOperator: '<S26>/UpperRelop'
       */
      if (rtb_Throttlecommandthrustvalue < RateTransition_Buffer_b) {
        /* ForEachSliceAssignment generated from: '<S24>/RPM_cmd_sat' */
        Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[ForEach_itr_f] =
          RateTransition_Buffer_b;
      } else {
        /* ForEachSliceAssignment generated from: '<S24>/RPM_cmd_sat' */
        Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[ForEach_itr_f] =
          rtb_Throttlecommandthrustvalue;
      }
    }
  }

  /* End of Outputs for SubSystem: '<S7>/For Each Subsystem' */

  /* Update for DiscreteIntegrator: '<S11>/Discrete-Time Integrator' incorporates:
   *  Product: '<S14>/Product'
   */
  Sim_Multi_DW.DiscreteTimeIntegrator_DSTATE[0] += 0.002 * rtb_Product[0];
  Sim_Multi_DW.DiscreteTimeIntegrator_DSTATE[1] += 0.002 * rtb_Product[1];
  Sim_Multi_DW.DiscreteTimeIntegrator_DSTATE[2] += 0.002 * rtb_Product[2];

  /* Update for Atomic SubSystem: '<Root>/multirotor' */
  /* Update for RateTransition: '<S5>/Rate Transition1' incorporates:
   *  ForEachSliceAssignment generated from: '<S24>/RPM_cmd_sat'
   */
  Sim_Multi_DW.RateTransition1_Buffer0[0] =
    Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[0];
  Sim_Multi_DW.RateTransition1_Buffer0[1] =
    Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[1];
  Sim_Multi_DW.RateTransition1_Buffer0[2] =
    Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[2];
  Sim_Multi_DW.RateTransition1_Buffer0[3] =
    Sim_Multi_B.ImpAsg_InsertedFor_RPM_cmd_sat_[3];

  /* End of Update for SubSystem: '<Root>/multirotor' */
}

/* Model step function for TID3 */
void Sim_Multi::step3()                /* Sample time: [0.01s, 0.0s] */
{
  /* (no output/update code required) */
}

/* Model step function for TID4 */
void Sim_Multi::step4()                /* Sample time: [0.02s, 0.0s] */
{
  /* (no output/update code required) */
}

/* Model step function for TID5 */
void Sim_Multi::step5()                /* Sample time: [0.03s, 0.0s] */
{
  /* (no output/update code required) */
}

/* Model initialize function */
void Sim_Multi::initialize()
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  /* Set task counter limit used by the static main program */
  ((&Sim_Multi_M))->Timing.TaskCounters.cLimit[0] = 1;
  ((&Sim_Multi_M))->Timing.TaskCounters.cLimit[1] = 1;
  ((&Sim_Multi_M))->Timing.TaskCounters.cLimit[2] = 2;
  ((&Sim_Multi_M))->Timing.TaskCounters.cLimit[3] = 10;
  ((&Sim_Multi_M))->Timing.TaskCounters.cLimit[4] = 20;
  ((&Sim_Multi_M))->Timing.TaskCounters.cLimit[5] = 30;

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&(&Sim_Multi_M)->solverInfo, &(&Sim_Multi_M)
                          ->Timing.simTimeStep);
    rtsiSetTPtr(&(&Sim_Multi_M)->solverInfo, &rtmGetTPtr((&Sim_Multi_M)));
    rtsiSetStepSizePtr(&(&Sim_Multi_M)->solverInfo, &(&Sim_Multi_M)
                       ->Timing.stepSize0);
    rtsiSetdXPtr(&(&Sim_Multi_M)->solverInfo, &(&Sim_Multi_M)->derivs);
    rtsiSetContStatesPtr(&(&Sim_Multi_M)->solverInfo, (real_T **) &(&Sim_Multi_M)
                         ->contStates);
    rtsiSetNumContStatesPtr(&(&Sim_Multi_M)->solverInfo, &(&Sim_Multi_M)
      ->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&(&Sim_Multi_M)->solverInfo, &(&Sim_Multi_M
      )->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&(&Sim_Multi_M)->solverInfo,
      &(&Sim_Multi_M)->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&(&Sim_Multi_M)->solverInfo,
      &(&Sim_Multi_M)->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&(&Sim_Multi_M)->solverInfo, (&rtmGetErrorStatus
      ((&Sim_Multi_M))));
    rtsiSetRTModelPtr(&(&Sim_Multi_M)->solverInfo, (&Sim_Multi_M));
  }

  rtsiSetSimTimeStep(&(&Sim_Multi_M)->solverInfo, MAJOR_TIME_STEP);
  (&Sim_Multi_M)->intgData.y = (&Sim_Multi_M)->odeY;
  (&Sim_Multi_M)->intgData.f[0] = (&Sim_Multi_M)->odeF[0];
  (&Sim_Multi_M)->intgData.f[1] = (&Sim_Multi_M)->odeF[1];
  (&Sim_Multi_M)->intgData.f[2] = (&Sim_Multi_M)->odeF[2];
  (&Sim_Multi_M)->contStates = ((X_Sim_Multi_T *) &Sim_Multi_X);
  rtsiSetSolverData(&(&Sim_Multi_M)->solverInfo, static_cast<void *>
                    (&(&Sim_Multi_M)->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&(&Sim_Multi_M)->solverInfo, false);
  rtsiSetSolverName(&(&Sim_Multi_M)->solverInfo,"ode3");
  rtmSetTPtr((&Sim_Multi_M), &(&Sim_Multi_M)->Timing.tArray[0]);
  (&Sim_Multi_M)->Timing.stepSize0 = 0.001;
  rtmSetFirstInitCond((&Sim_Multi_M), 1);

  {
    /* local scratch DWork variables */
    int32_T ForEach_itr;

    /* Start for Atomic SubSystem: '<Root>/multirotor' */
    /* Start for RateTransition: '<S5>/Rate Transition1' */
    Sim_Multi_B.RateTransition1[0] = 3104.5025852;
    Sim_Multi_B.RateTransition1[1] = 3104.5025852;
    Sim_Multi_B.RateTransition1[2] = 3104.5025852;
    Sim_Multi_B.RateTransition1[3] = 3104.5025852;

    /* Start for Iterator SubSystem: '<S94>/For Each Subsystem' */
    for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
      /* Start for If: '<S131>/If' */
      Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem = -1;

      /* Start for If: '<S143>/If' */
      Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_l = -1;

      /* Start for If: '<S145>/If' */
      Sim_Multi_DW.CoreSubsys_p[ForEach_itr].If_ActiveSubsystem_g = -1;
    }

    /* End of Start for SubSystem: '<S94>/For Each Subsystem' */

    /* Start for If: '<S95>/If' */
    Sim_Multi_DW.If_ActiveSubsystem = -1;

    /* End of Start for SubSystem: '<Root>/multirotor' */
  }

  {
    /* local scratch DWork variables */
    int32_T ForEach_itr;

    /* InitializeConditions for TransferFcn: '<S31>/Transfer Fcn1' */
    Sim_Multi_X.TransferFcn1_CSTATE = 0.0;

    /* InitializeConditions for TransferFcn: '<S31>/Transfer Fcn6' */
    Sim_Multi_X.TransferFcn6_CSTATE = 0.0;

    /* InitializeConditions for TransferFcn: '<S31>/Transfer Fcn4' */
    Sim_Multi_X.TransferFcn4_CSTATE = 0.0;

    /* InitializeConditions for TransferFcn: '<S31>/Transfer Fcn5' */
    Sim_Multi_X.TransferFcn5_CSTATE = 0.0;

    /* InitializeConditions for TransferFcn: '<S31>/Transfer Fcn2' */
    Sim_Multi_X.TransferFcn2_CSTATE = 0.0;

    /* InitializeConditions for TransferFcn: '<S31>/Transfer Fcn3' */
    Sim_Multi_X.TransferFcn3_CSTATE = 0.0;

    /* InitializeConditions for Memory: '<S36>/Memory' */
    Sim_Multi_DW.Memory_PreviousInput[0] = 1.0;
    Sim_Multi_DW.Memory_PreviousInput[1] = 0.0;
    Sim_Multi_DW.Memory_PreviousInput[2] = 0.0;
    Sim_Multi_DW.Memory_PreviousInput[3] = 0.0;

    /* InitializeConditions for Integrator: '<S37>/Integrator' */
    Sim_Multi_X.Integrator_CSTATE = 0.0;

    /* InitializeConditions for UniformRandomNumber: '<S37>/Uniform Random Number' */
    Sim_Multi_DW.RandSeed = 43057152U;
    Sim_Multi_DW.UniformRandomNumber_NextOutput = rt_urand_Upu32_Yd_f_pw_snf
      (&Sim_Multi_DW.RandSeed) * 0.002 - 0.001;

    /* InitializeConditions for Integrator: '<S38>/Integrator' */
    Sim_Multi_X.Integrator_CSTATE_f = 0.0;

    /* InitializeConditions for UniformRandomNumber: '<S38>/Uniform Random Number' */
    Sim_Multi_DW.RandSeed_b = 22151168U;
    Sim_Multi_DW.UniformRandomNumber_NextOutpu_m = rt_urand_Upu32_Yd_f_pw_snf
      (&Sim_Multi_DW.RandSeed_b) * 0.002 - 0.001;

    /* SystemInitialize for Atomic SubSystem: '<Root>/multirotor' */
    /* InitializeConditions for Integrator: '<S64>/Q-Integrator' */
    if (rtmIsFirstInitCond((&Sim_Multi_M))) {
      Sim_Multi_X.QIntegrator_CSTATE[0] = 0.0;
      Sim_Multi_X.QIntegrator_CSTATE[1] = 0.0;
      Sim_Multi_X.QIntegrator_CSTATE[2] = 0.0;
      Sim_Multi_X.QIntegrator_CSTATE[3] = 0.0;
    }

    Sim_Multi_DW.QIntegrator_IWORK = 1;

    /* End of InitializeConditions for Integrator: '<S64>/Q-Integrator' */

    /* InitializeConditions for Integrator: '<S58>/V_b' */
    Sim_Multi_X.V_b_CSTATE[0] = 0.0;

    /* InitializeConditions for Integrator: '<S58>/omega' */
    Sim_Multi_X.omega_CSTATE[0] = 0.0;

    /* InitializeConditions for Integrator: '<S58>/V_b' */
    Sim_Multi_X.V_b_CSTATE[1] = 0.0;

    /* InitializeConditions for Integrator: '<S58>/omega' */
    Sim_Multi_X.omega_CSTATE[1] = 0.0;

    /* InitializeConditions for Integrator: '<S58>/V_b' */
    Sim_Multi_X.V_b_CSTATE[2] = 0.0;

    /* InitializeConditions for Integrator: '<S58>/omega' */
    Sim_Multi_X.omega_CSTATE[2] = 0.0;

    /* InitializeConditions for RateTransition: '<S5>/Rate Transition1' */
    Sim_Multi_DW.RateTransition1_Buffer0[0] = 3104.5025852;
    Sim_Multi_DW.RateTransition1_Buffer0[1] = 3104.5025852;
    Sim_Multi_DW.RateTransition1_Buffer0[2] = 3104.5025852;
    Sim_Multi_DW.RateTransition1_Buffer0[3] = 3104.5025852;

    /* InitializeConditions for Integrator: '<S58>/X_i' */
    Sim_Multi_X.X_i_CSTATE[0] = 0.0;

    /* SystemInitialize for IfAction SubSystem: '<S95>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S95>/Merge' incorporates:
     *  Outport: '<S98>/Drag force'
     */
    Sim_Multi_B.Forceagainstdirectionofmotiondu[0] = 0.0;

    /* End of SystemInitialize for SubSystem: '<S95>/Zero airspeed' */

    /* InitializeConditions for Integrator: '<S58>/X_i' */
    Sim_Multi_X.X_i_CSTATE[1] = 0.0;

    /* SystemInitialize for IfAction SubSystem: '<S95>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S95>/Merge' incorporates:
     *  Outport: '<S98>/Drag force'
     */
    Sim_Multi_B.Forceagainstdirectionofmotiondu[1] = 0.0;

    /* End of SystemInitialize for SubSystem: '<S95>/Zero airspeed' */

    /* InitializeConditions for Integrator: '<S58>/X_i' */
    Sim_Multi_X.X_i_CSTATE[2] = 0.0;

    /* SystemInitialize for IfAction SubSystem: '<S95>/Zero airspeed' */
    /* SystemInitialize for Merge: '<S95>/Merge' incorporates:
     *  Outport: '<S98>/Drag force'
     */
    Sim_Multi_B.Forceagainstdirectionofmotiondu[2] = -1.0;

    /* End of SystemInitialize for SubSystem: '<S95>/Zero airspeed' */

    /* SystemInitialize for Iterator SubSystem: '<S94>/For Each Subsystem' */
    for (ForEach_itr = 0; ForEach_itr < 4; ForEach_itr++) {
      /* InitializeConditions for Integrator: '<S118>/Integrator' */
      Sim_Multi_X.CoreSubsys_p[ForEach_itr].Integrator_CSTATE_e = 3104.5025852;

      /* InitializeConditions for Integrator: '<S119>/Integrator' */
      Sim_Multi_X.CoreSubsys_p[ForEach_itr].Integrator_CSTATE_o = 3104.5025852;

      /* SystemInitialize for IfAction SubSystem: '<S131>/Zero airspeed in rotor plane' */
      /* SystemInitialize for Merge: '<S131>/Merge' incorporates:
       *  Outport: '<S138>/Thrust direction (Body)'
       */
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[0] =
        0.0;
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[1] =
        0.0;
      Sim_Multi_B.CoreSubsys_p[ForEach_itr].NewtiltedthrustdirectionBodyaxe[2] =
        -1.0;

      /* End of SystemInitialize for SubSystem: '<S131>/Zero airspeed in rotor plane' */
    }

    /* End of SystemInitialize for SubSystem: '<S94>/For Each Subsystem' */
    /* End of SystemInitialize for SubSystem: '<Root>/multirotor' */

    /* set "at time zero" to false */
    if (rtmIsFirstInitCond((&Sim_Multi_M))) {
      rtmSetFirstInitCond((&Sim_Multi_M), 0);
    }
  }
}

/* Model terminate function */
void Sim_Multi::terminate()
{
  /* (no terminate code required) */
}

/* Constructor */
Sim_Multi::Sim_Multi() :
  Sim_Multi_B(),
  Sim_Multi_DW(),
  Sim_Multi_X(),
  Sim_Multi_M()
{
  /* Currently there is no constructor body generated.*/
}

/* Destructor */
/* Currently there is no destructor body generated.*/
Sim_Multi::~Sim_Multi() = default;

/* Real-Time Model get method */
RT_MODEL_Sim_Multi_T * Sim_Multi::getRTM()
{
  return (&Sim_Multi_M);
}
