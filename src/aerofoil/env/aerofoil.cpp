/*
 * aerofoil.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "aerofoil".
 *
 * Model version              : 2.15
 * Simulink Coder version : 23.2 (R2023b) 01-Aug-2023
 * C++ source code generated on : Fri Jan 12 15:07:58 2024
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Debugging
 * Validation result: Not run
 */q

#include "aerofoil.h"
#include <cstring>
#include "rtwtypes.h"
#include "aerofoil_private.h"

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
void aerofoil::rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
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
  int_T nXc { 12 };

  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  aerofoil_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  this->step();
  aerofoil_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE3_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[1]);
  rtsiSetdX(si, f2);
  this->step();
  aerofoil_derivatives();

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

/* Model step function */
void aerofoil::step()
{
  real_T tmp;
  real_T tmp_0;
  int32_T i;
  int32_T i_0;
  if (rtmIsMajorTimeStep((&aerofoil_M))) {
    /* set solver stop time */
    rtsiSetSolverStopTime(&(&aerofoil_M)->solverInfo,(((&aerofoil_M)
      ->Timing.clockTick0+1)*(&aerofoil_M)->Timing.stepSize0));
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep((&aerofoil_M))) {
    (&aerofoil_M)->Timing.t[0] = rtsiGetT(&(&aerofoil_M)->solverInfo);
  }

  /* Outputs for Atomic SubSystem: '<Root>/aerofoil' */
  /* Integrator: '<S1>/Integrator' incorporates:
   *  Inport: '<Root>/init_state'
   */
  if (aerofoil_DW.Integrator_IWORK != 0) {
    std::memcpy(&aerofoil_X.Integrator_CSTATE[0], &aerofoil_U.init_state[0], 12U
                * sizeof(real_T));
  }

  /* Outport: '<Root>/plunge' incorporates:
   *  Integrator: '<S1>/Integrator'
   */
  aerofoil_Y.plunge = aerofoil_X.Integrator_CSTATE[0];

  /* Outport: '<Root>/pitch' incorporates:
   *  Integrator: '<S1>/Integrator'
   */
  aerofoil_Y.pitch = aerofoil_X.Integrator_CSTATE[1];

  /* Outport: '<Root>/delta' incorporates:
   *  Integrator: '<S1>/Integrator'
   */
  aerofoil_Y.delta = aerofoil_X.Integrator_CSTATE[2];

  /* Outport: '<Root>/plunge_dot' incorporates:
   *  Integrator: '<S1>/Integrator'
   */
  aerofoil_Y.plunge_dot = aerofoil_X.Integrator_CSTATE[3];

  /* Outport: '<Root>/pitch_dot' incorporates:
   *  Integrator: '<S1>/Integrator'
   */
  aerofoil_Y.pitch_dot = aerofoil_X.Integrator_CSTATE[4];

  /* Outport: '<Root>/delta_dot' incorporates:
   *  Integrator: '<S1>/Integrator'
   */
  aerofoil_Y.delta_dot = aerofoil_X.Integrator_CSTATE[5];

  /* Product: '<S1>/Matrix Multiply2' */
  tmp = 0.0;
  for (i = 0; i < 12; i++) {
    /* Sum: '<S1>/Add' incorporates:
     *  Inport: '<Root>/MatrixA'
     *  Inport: '<Root>/MatrixB'
     *  Inport: '<Root>/delta_ddot'
     *  Integrator: '<S1>/Integrator'
     *  Product: '<S1>/Matrix Multiply1'
     *  Product: '<S1>/Matrix Multiply3'
     */
    tmp_0 = 0.0;
    for (i_0 = 0; i_0 < 12; i_0++) {
      tmp_0 += aerofoil_U.MatrixA[12 * i_0 + i] *
        aerofoil_X.Integrator_CSTATE[i_0];
    }

    aerofoil_B.Add[i] = aerofoil_U.MatrixB[i] * aerofoil_U.delta_ddot + tmp_0;

    /* End of Sum: '<S1>/Add' */

    /* Product: '<S1>/Matrix Multiply2' incorporates:
     *  Inport: '<Root>/MatrixC'
     *  Integrator: '<S1>/Integrator'
     */
    tmp += aerofoil_U.MatrixC[i] * aerofoil_X.Integrator_CSTATE[i];
  }

  /* Outport: '<Root>/C_L' incorporates:
   *  Product: '<S1>/Matrix Multiply2'
   */
  aerofoil_Y.C_L = tmp;

  /* End of Outputs for SubSystem: '<Root>/aerofoil' */
  if (rtmIsMajorTimeStep((&aerofoil_M))) {
    /* Update for Atomic SubSystem: '<Root>/aerofoil' */
    /* Update for Integrator: '<S1>/Integrator' */
    aerofoil_DW.Integrator_IWORK = 0;

    /* End of Update for SubSystem: '<Root>/aerofoil' */
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep((&aerofoil_M))) {
    rt_ertODEUpdateContinuousStates(&(&aerofoil_M)->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     */
    ++(&aerofoil_M)->Timing.clockTick0;
    (&aerofoil_M)->Timing.t[0] = rtsiGetSolverStopTime(&(&aerofoil_M)
      ->solverInfo);

    {
      /* Update absolute timer for sample time: [0.01s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.01, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       */
      (&aerofoil_M)->Timing.clockTick1++;
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void aerofoil::aerofoil_derivatives()
{
  XDot_aerofoil_T *_rtXdot;
  _rtXdot = ((XDot_aerofoil_T *) (&aerofoil_M)->derivs);

  /* Derivatives for Atomic SubSystem: '<Root>/aerofoil' */
  /* Derivatives for Integrator: '<S1>/Integrator' incorporates:
   *  Sum: '<S1>/Add'
   */
  std::memcpy(&_rtXdot->Integrator_CSTATE[0], &aerofoil_B.Add[0], 12U * sizeof
              (real_T));

  /* End of Derivatives for SubSystem: '<Root>/aerofoil' */
}

/* Model initialize function */
void aerofoil::initialize()
{
  /* Registration code */
  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M)
                          ->Timing.simTimeStep);
    rtsiSetTPtr(&(&aerofoil_M)->solverInfo, &rtmGetTPtr((&aerofoil_M)));
    rtsiSetStepSizePtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M)
                       ->Timing.stepSize0);
    rtsiSetdXPtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M)->derivs);
    rtsiSetContStatesPtr(&(&aerofoil_M)->solverInfo, (real_T **) &(&aerofoil_M
                         )->contStates);
    rtsiSetNumContStatesPtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M)
      ->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M)
      ->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M)
      ->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&(&aerofoil_M)->solverInfo, &(&aerofoil_M
      )->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&(&aerofoil_M)->solverInfo, (&rtmGetErrorStatus
      ((&aerofoil_M))));
    rtsiSetRTModelPtr(&(&aerofoil_M)->solverInfo, (&aerofoil_M));
  }

  rtsiSetSimTimeStep(&(&aerofoil_M)->solverInfo, MAJOR_TIME_STEP);
  (&aerofoil_M)->intgData.y = (&aerofoil_M)->odeY;
  (&aerofoil_M)->intgData.f[0] = (&aerofoil_M)->odeF[0];
  (&aerofoil_M)->intgData.f[1] = (&aerofoil_M)->odeF[1];
  (&aerofoil_M)->intgData.f[2] = (&aerofoil_M)->odeF[2];
  (&aerofoil_M)->contStates = ((X_aerofoil_T *) &aerofoil_X);
  (&aerofoil_M)->contStateDisabled = ((XDis_aerofoil_T *) &aerofoil_XDis);
  (&aerofoil_M)->Timing.tStart = (0.0);
  rtsiSetSolverData(&(&aerofoil_M)->solverInfo, static_cast<void *>
                    (&(&aerofoil_M)->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&(&aerofoil_M)->solverInfo, false);
  rtsiSetSolverName(&(&aerofoil_M)->solverInfo,"ode3");
  rtmSetTPtr((&aerofoil_M), &(&aerofoil_M)->Timing.tArray[0]);
  (&aerofoil_M)->Timing.stepSize0 = 0.01;
  rtmSetFirstInitCond((&aerofoil_M), 1);

  /* SystemInitialize for Atomic SubSystem: '<Root>/aerofoil' */
  /* InitializeConditions for Integrator: '<S1>/Integrator' */
  if (rtmIsFirstInitCond((&aerofoil_M))) {
    std::memset(&aerofoil_X.Integrator_CSTATE[0], 0, 12U * sizeof(real_T));
  }

  aerofoil_DW.Integrator_IWORK = 1;

  /* End of InitializeConditions for Integrator: '<S1>/Integrator' */
  /* End of SystemInitialize for SubSystem: '<Root>/aerofoil' */

  /* set "at time zero" to false */
  if (rtmIsFirstInitCond((&aerofoil_M))) {
    rtmSetFirstInitCond((&aerofoil_M), 0);
  }
}

/* Model terminate function */
void aerofoil::terminate()
{
  /* (no terminate code required) */
}

/* Constructor */
aerofoil::aerofoil() :
  aerofoil_U(),
  aerofoil_Y(),
  aerofoil_B(),
  aerofoil_DW(),
  aerofoil_X(),
  aerofoil_XDis(),
  aerofoil_M()
{
  /* Currently there is no constructor body generated.*/
}

/* Destructor */
/* Currently there is no destructor body generated.*/
aerofoil::~aerofoil() = default;

/* Real-Time Model get method */
RT_MODEL_aerofoil_T * aerofoil::getRTM()
{
  return (&aerofoil_M);
}
