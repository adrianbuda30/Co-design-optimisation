/*
 * cart_pole_system_dynamics0.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "cart_pole_system_dynamics0".
 *
 * Model version              : 1.23
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Sun Sep 24 11:15:22 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#include "cart_pole_system_dynamics0.h"
#include <cmath>
#include "rtwtypes.h"
#include "cart_pole_system_dynamics0_private.h"

/*
 * This function updates continuous states using the ODE3 fixed-step
 * solver algorithm
 */
void cart_pole_system_dynamics0::rt_ertODEUpdateContinuousStates(RTWSolverInfo
  *si )
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
  int_T nXc { 5 };

  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  cart_pole_system_dynamics0_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE3_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE3_A[0]);
  rtsiSetdX(si, f1);
  this->step();
  cart_pole_system_dynamics0_derivatives();

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
  cart_pole_system_dynamics0_derivatives();

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
void cart_pole_system_dynamics0::step()
{
  real_T TrigonometricFunction1_tmp;
  real_T TrigonometricFunction_tmp;
  if (rtmIsMajorTimeStep((&cart_pole_system_dynamics0_M))) {
    /* set solver stop time */
    if (!((&cart_pole_system_dynamics0_M)->Timing.clockTick0+1)) {
      rtsiSetSolverStopTime(&(&cart_pole_system_dynamics0_M)->solverInfo,
                            (((&cart_pole_system_dynamics0_M)
        ->Timing.clockTickH0 + 1) * (&cart_pole_system_dynamics0_M)
        ->Timing.stepSize0 * 4294967296.0));
    } else {
      rtsiSetSolverStopTime(&(&cart_pole_system_dynamics0_M)->solverInfo,
                            (((&cart_pole_system_dynamics0_M)->Timing.clockTick0
        + 1) * (&cart_pole_system_dynamics0_M)->Timing.stepSize0 +
        (&cart_pole_system_dynamics0_M)->Timing.clockTickH0 *
        (&cart_pole_system_dynamics0_M)->Timing.stepSize0 * 4294967296.0));
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep((&cart_pole_system_dynamics0_M))) {
    (&cart_pole_system_dynamics0_M)->Timing.t[0] = rtsiGetT
      (&(&cart_pole_system_dynamics0_M)->solverInfo);
  }

  /* Outputs for Atomic SubSystem: '<Root>/cart_pole_system_dynamics' */
  /* Integrator: '<S1>/Integrator1' incorporates:
   *  Inport: '<Root>/init_pole_pos'
   */
  if (cart_pole_system_dynamics0_DW.Integrator1_IWORK != 0) {
    cart_pole_system_dynamics0_X.Integrator1_CSTATE =
      cart_pole_system_dynamics0_U.init_pole_pos;
  }

  /* Trigonometry: '<S1>/Trigonometric Function' incorporates:
   *  Integrator: '<S1>/Integrator1'
   *  Trigonometry: '<S1>/Sin1'
   */
  TrigonometricFunction_tmp = std::sin
    (cart_pole_system_dynamics0_X.Integrator1_CSTATE);

  /* Trigonometry: '<S1>/Trigonometric Function' */
  cart_pole_system_dynamics0_B.TrigonometricFunction = TrigonometricFunction_tmp;

  /* Integrator: '<S1>/Integrator' */
  cart_pole_system_dynamics0_B.theta =
    cart_pole_system_dynamics0_X.Integrator_CSTATE;

  /* Math: '<S1>/Math Function1' */
  cart_pole_system_dynamics0_B.MathFunction1 =
    cart_pole_system_dynamics0_B.theta * cart_pole_system_dynamics0_B.theta;

  /* Product: '<S1>/Product1' incorporates:
   *  Constant: '<S1>/Constant2'
   */
  cart_pole_system_dynamics0_B.Product1 =
    cart_pole_system_dynamics0_B.TrigonometricFunction *
    cart_pole_system_dynamics0_P.g;

  /* Math: '<S1>/Math Function' incorporates:
   *  Inport: '<Root>/mass_cart'
   *  Inport: '<Root>/mass_pole'
   *  Sum: '<S1>/Add'
   *
   * About '<S1>/Math Function':
   *  Operator: reciprocal
   */
  cart_pole_system_dynamics0_B.MathFunction = 1.0 /
    (cart_pole_system_dynamics0_U.mass_pole +
     cart_pole_system_dynamics0_U.mass_cart);

  /* Trigonometry: '<S1>/Trigonometric Function1' incorporates:
   *  Integrator: '<S1>/Integrator1'
   *  Trigonometry: '<S1>/Cos1'
   */
  TrigonometricFunction1_tmp = std::cos
    (cart_pole_system_dynamics0_X.Integrator1_CSTATE);

  /* Trigonometry: '<S1>/Trigonometric Function1' */
  cart_pole_system_dynamics0_B.TrigonometricFunction1 =
    TrigonometricFunction1_tmp;

  /* Product: '<S1>/Product7' incorporates:
   *  Constant: '<S1>/Constant'
   *  Inport: '<Root>/length'
   *  Inport: '<Root>/mass_pole'
   *  Math: '<S1>/Math Function2'
   *  Product: '<S1>/Product4'
   *  Sum: '<S1>/Add4'
   */
  cart_pole_system_dynamics0_B.Product7 =
    (cart_pole_system_dynamics0_P.Constant_Value -
     cart_pole_system_dynamics0_B.TrigonometricFunction1 *
     cart_pole_system_dynamics0_B.TrigonometricFunction1 *
     cart_pole_system_dynamics0_B.MathFunction *
     cart_pole_system_dynamics0_U.mass_pole) *
    cart_pole_system_dynamics0_U.length;

  /* Product: '<S1>/Product8' */
  cart_pole_system_dynamics0_B.Product8 =
    cart_pole_system_dynamics0_B.MathFunction1 *
    cart_pole_system_dynamics0_B.TrigonometricFunction;

  /* Integrator: '<S1>/Integrator2' */
  cart_pole_system_dynamics0_B.x =
    cart_pole_system_dynamics0_X.Integrator2_CSTATE;

  /* Outport: '<Root>/cart_position' incorporates:
   *  Integrator: '<S1>/Integrator3'
   */
  cart_pole_system_dynamics0_Y.cart_position =
    cart_pole_system_dynamics0_X.Integrator3_CSTATE;

  /* Outport: '<Root>/pole_angle' incorporates:
   *  Integrator: '<S1>/Integrator1'
   */
  cart_pole_system_dynamics0_Y.pole_angle =
    cart_pole_system_dynamics0_X.Integrator1_CSTATE;

  /* Outport: '<Root>/pole_position' incorporates:
   *  Inport: '<Root>/length'
   *  Integrator: '<S1>/Integrator3'
   *  Product: '<S1>/Product5'
   *  Product: '<S1>/Product6'
   *  Sum: '<S1>/Add2'
   */
  cart_pole_system_dynamics0_Y.pole_position[0] =
    cart_pole_system_dynamics0_U.length * TrigonometricFunction_tmp +
    cart_pole_system_dynamics0_X.Integrator3_CSTATE;
  cart_pole_system_dynamics0_Y.pole_position[1] = TrigonometricFunction1_tmp *
    cart_pole_system_dynamics0_U.length;

  /* Outport: '<Root>/effort' incorporates:
   *  Integrator: '<S1>/Integrator4'
   */
  cart_pole_system_dynamics0_Y.effort =
    cart_pole_system_dynamics0_X.Integrator4_CSTATE;

  /* End of Outputs for SubSystem: '<Root>/cart_pole_system_dynamics' */

  /* Outport: '<Root>/pole_angular_velocity' */
  cart_pole_system_dynamics0_Y.pole_angular_velocity =
    cart_pole_system_dynamics0_B.theta;

  /* Outport: '<Root>/cart_velocity' */
  cart_pole_system_dynamics0_Y.cart_velocity = cart_pole_system_dynamics0_B.x;
  if (rtmIsMajorTimeStep((&cart_pole_system_dynamics0_M))) {
    /* Update for Atomic SubSystem: '<Root>/cart_pole_system_dynamics' */
    /* Product: '<S1>/Product' incorporates:
     *  Inport: '<Root>/length'
     *  Inport: '<Root>/mass_pole'
     */
    cart_pole_system_dynamics0_B.Product = cart_pole_system_dynamics0_U.length *
      cart_pole_system_dynamics0_U.mass_pole;

    /* Product: '<S1>/Product3' */
    cart_pole_system_dynamics0_B.Product3 =
      cart_pole_system_dynamics0_B.TrigonometricFunction *
      cart_pole_system_dynamics0_B.MathFunction1 *
      cart_pole_system_dynamics0_B.Product;

    /* Sum: '<S1>/Add1' incorporates:
     *  Inport: '<Root>/Force'
     */
    cart_pole_system_dynamics0_B.Add1 = (0.0 -
      cart_pole_system_dynamics0_U.Force) -
      cart_pole_system_dynamics0_B.Product3;

    /* Product: '<S1>/Product2' */
    cart_pole_system_dynamics0_B.Product2 = cart_pole_system_dynamics0_B.Add1 *
      cart_pole_system_dynamics0_B.MathFunction *
      cart_pole_system_dynamics0_B.TrigonometricFunction1;

    /* Sum: '<S1>/Add3' */
    cart_pole_system_dynamics0_B.Add3 = cart_pole_system_dynamics0_B.Product1 +
      cart_pole_system_dynamics0_B.Product2;

    /* Product: '<S1>/Divide' */
    cart_pole_system_dynamics0_B.theta_g = cart_pole_system_dynamics0_B.Add3 /
      cart_pole_system_dynamics0_B.Product7;

    /* Product: '<S1>/Product9' */
    cart_pole_system_dynamics0_B.Product9 = cart_pole_system_dynamics0_B.theta_g
      * cart_pole_system_dynamics0_B.TrigonometricFunction1;

    /* Sum: '<S1>/Add6' */
    cart_pole_system_dynamics0_B.Add6 = cart_pole_system_dynamics0_B.Product8 -
      cart_pole_system_dynamics0_B.Product9;

    /* Product: '<S1>/Product10' */
    cart_pole_system_dynamics0_B.Product10 = cart_pole_system_dynamics0_B.Add6 *
      cart_pole_system_dynamics0_B.Product;

    /* Sum: '<S1>/Add5' incorporates:
     *  Inport: '<Root>/Force'
     */
    cart_pole_system_dynamics0_B.Add5 = cart_pole_system_dynamics0_U.Force +
      cart_pole_system_dynamics0_B.Product10;

    /* Product: '<S1>/Product11' */
    cart_pole_system_dynamics0_B.x_h = cart_pole_system_dynamics0_B.Add5 *
      cart_pole_system_dynamics0_B.MathFunction;

    /* Update for Integrator: '<S1>/Integrator1' */
    cart_pole_system_dynamics0_DW.Integrator1_IWORK = 0;

    /* End of Update for SubSystem: '<Root>/cart_pole_system_dynamics' */
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep((&cart_pole_system_dynamics0_M))) {
    rt_ertODEUpdateContinuousStates(&(&cart_pole_system_dynamics0_M)->solverInfo);

    /* Update absolute time for base rate */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick0 and the high bits
     * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++(&cart_pole_system_dynamics0_M)->Timing.clockTick0)) {
      ++(&cart_pole_system_dynamics0_M)->Timing.clockTickH0;
    }

    (&cart_pole_system_dynamics0_M)->Timing.t[0] = rtsiGetSolverStopTime
      (&(&cart_pole_system_dynamics0_M)->solverInfo);

    {
      /* Update absolute timer for sample time: [0.02s, 0.0s] */
      /* The "clockTick1" counts the number of times the code of this task has
       * been executed. The resolution of this integer timer is 0.02, which is the step size
       * of the task. Size of "clockTick1" ensures timer will not overflow during the
       * application lifespan selected.
       * Timer of this task consists of two 32 bit unsigned integers.
       * The two integers represent the low bits Timing.clockTick1 and the high bits
       * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
       */
      (&cart_pole_system_dynamics0_M)->Timing.clockTick1++;
      if (!(&cart_pole_system_dynamics0_M)->Timing.clockTick1) {
        (&cart_pole_system_dynamics0_M)->Timing.clockTickH1++;
      }
    }
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void cart_pole_system_dynamics0::cart_pole_system_dynamics0_derivatives()
{
  XDot_cart_pole_system_dynamic_T *_rtXdot;
  _rtXdot = ((XDot_cart_pole_system_dynamic_T *) (&cart_pole_system_dynamics0_M
             )->derivs);

  /* Derivatives for Atomic SubSystem: '<Root>/cart_pole_system_dynamics' */
  /* Product: '<S1>/Product' incorporates:
   *  Inport: '<Root>/length'
   *  Inport: '<Root>/mass_pole'
   */
  cart_pole_system_dynamics0_B.Product = cart_pole_system_dynamics0_U.length *
    cart_pole_system_dynamics0_U.mass_pole;

  /* Product: '<S1>/Product3' */
  cart_pole_system_dynamics0_B.Product3 =
    cart_pole_system_dynamics0_B.TrigonometricFunction *
    cart_pole_system_dynamics0_B.MathFunction1 *
    cart_pole_system_dynamics0_B.Product;

  /* Sum: '<S1>/Add1' incorporates:
   *  Inport: '<Root>/Force'
   */
  cart_pole_system_dynamics0_B.Add1 = (0.0 - cart_pole_system_dynamics0_U.Force)
    - cart_pole_system_dynamics0_B.Product3;

  /* Product: '<S1>/Product2' */
  cart_pole_system_dynamics0_B.Product2 = cart_pole_system_dynamics0_B.Add1 *
    cart_pole_system_dynamics0_B.MathFunction *
    cart_pole_system_dynamics0_B.TrigonometricFunction1;

  /* Sum: '<S1>/Add3' */
  cart_pole_system_dynamics0_B.Add3 = cart_pole_system_dynamics0_B.Product1 +
    cart_pole_system_dynamics0_B.Product2;

  /* Product: '<S1>/Divide' */
  cart_pole_system_dynamics0_B.theta_g = cart_pole_system_dynamics0_B.Add3 /
    cart_pole_system_dynamics0_B.Product7;

  /* Product: '<S1>/Product9' */
  cart_pole_system_dynamics0_B.Product9 = cart_pole_system_dynamics0_B.theta_g *
    cart_pole_system_dynamics0_B.TrigonometricFunction1;

  /* Sum: '<S1>/Add6' */
  cart_pole_system_dynamics0_B.Add6 = cart_pole_system_dynamics0_B.Product8 -
    cart_pole_system_dynamics0_B.Product9;

  /* Product: '<S1>/Product10' */
  cart_pole_system_dynamics0_B.Product10 = cart_pole_system_dynamics0_B.Add6 *
    cart_pole_system_dynamics0_B.Product;

  /* Sum: '<S1>/Add5' incorporates:
   *  Inport: '<Root>/Force'
   */
  cart_pole_system_dynamics0_B.Add5 = cart_pole_system_dynamics0_U.Force +
    cart_pole_system_dynamics0_B.Product10;

  /* Product: '<S1>/Product11' */
  cart_pole_system_dynamics0_B.x_h = cart_pole_system_dynamics0_B.Add5 *
    cart_pole_system_dynamics0_B.MathFunction;

  /* Derivatives for Integrator: '<S1>/Integrator1' */
  _rtXdot->Integrator1_CSTATE = cart_pole_system_dynamics0_B.theta;

  /* Derivatives for Integrator: '<S1>/Integrator' */
  _rtXdot->Integrator_CSTATE = cart_pole_system_dynamics0_B.theta_g;

  /* Derivatives for Integrator: '<S1>/Integrator3' */
  _rtXdot->Integrator3_CSTATE = cart_pole_system_dynamics0_B.x;

  /* Derivatives for Integrator: '<S1>/Integrator2' */
  _rtXdot->Integrator2_CSTATE = cart_pole_system_dynamics0_B.x_h;

  /* Derivatives for Integrator: '<S1>/Integrator4' incorporates:
   *  Inport: '<Root>/Force'
   */
  _rtXdot->Integrator4_CSTATE = cart_pole_system_dynamics0_U.Force;

  /* End of Derivatives for SubSystem: '<Root>/cart_pole_system_dynamics' */
}

/* Model initialize function */
void cart_pole_system_dynamics0::initialize()
{
  /* Registration code */
  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
                          &(&cart_pole_system_dynamics0_M)->Timing.simTimeStep);
    rtsiSetTPtr(&(&cart_pole_system_dynamics0_M)->solverInfo, &rtmGetTPtr
                ((&cart_pole_system_dynamics0_M)));
    rtsiSetStepSizePtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
                       &(&cart_pole_system_dynamics0_M)->Timing.stepSize0);
    rtsiSetdXPtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
                 &(&cart_pole_system_dynamics0_M)->derivs);
    rtsiSetContStatesPtr(&(&cart_pole_system_dynamics0_M)->solverInfo, (real_T **)
                         &(&cart_pole_system_dynamics0_M)->contStates);
    rtsiSetNumContStatesPtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
      &(&cart_pole_system_dynamics0_M)->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
      &(&cart_pole_system_dynamics0_M)->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&(&cart_pole_system_dynamics0_M)
      ->solverInfo, &(&cart_pole_system_dynamics0_M)->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&(&cart_pole_system_dynamics0_M)
      ->solverInfo, &(&cart_pole_system_dynamics0_M)->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
                          (&rtmGetErrorStatus((&cart_pole_system_dynamics0_M))));
    rtsiSetRTModelPtr(&(&cart_pole_system_dynamics0_M)->solverInfo,
                      (&cart_pole_system_dynamics0_M));
  }

  rtsiSetSimTimeStep(&(&cart_pole_system_dynamics0_M)->solverInfo,
                     MAJOR_TIME_STEP);
  (&cart_pole_system_dynamics0_M)->intgData.y = (&cart_pole_system_dynamics0_M
    )->odeY;
  (&cart_pole_system_dynamics0_M)->intgData.f[0] =
    (&cart_pole_system_dynamics0_M)->odeF[0];
  (&cart_pole_system_dynamics0_M)->intgData.f[1] =
    (&cart_pole_system_dynamics0_M)->odeF[1];
  (&cart_pole_system_dynamics0_M)->intgData.f[2] =
    (&cart_pole_system_dynamics0_M)->odeF[2];
  (&cart_pole_system_dynamics0_M)->contStates = ((X_cart_pole_system_dynamics0_T
    *) &cart_pole_system_dynamics0_X);
  rtsiSetSolverData(&(&cart_pole_system_dynamics0_M)->solverInfo, static_cast<
                    void *>(&(&cart_pole_system_dynamics0_M)->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&(&cart_pole_system_dynamics0_M)
    ->solverInfo, false);
  rtsiSetSolverName(&(&cart_pole_system_dynamics0_M)->solverInfo,"ode3");
  rtmSetTPtr((&cart_pole_system_dynamics0_M), &(&cart_pole_system_dynamics0_M)
             ->Timing.tArray[0]);
  (&cart_pole_system_dynamics0_M)->Timing.stepSize0 = 0.02;
  rtmSetFirstInitCond((&cart_pole_system_dynamics0_M), 1);

  /* SystemInitialize for Atomic SubSystem: '<Root>/cart_pole_system_dynamics' */
  /* InitializeConditions for Integrator: '<S1>/Integrator1' */
  if (rtmIsFirstInitCond((&cart_pole_system_dynamics0_M))) {
    cart_pole_system_dynamics0_X.Integrator1_CSTATE = 0.0;
  }

  cart_pole_system_dynamics0_DW.Integrator1_IWORK = 1;

  /* End of InitializeConditions for Integrator: '<S1>/Integrator1' */

  /* InitializeConditions for Integrator: '<S1>/Integrator' */
  cart_pole_system_dynamics0_X.Integrator_CSTATE =
    cart_pole_system_dynamics0_P.Integrator_IC;

  /* InitializeConditions for Integrator: '<S1>/Integrator3' */
  cart_pole_system_dynamics0_X.Integrator3_CSTATE =
    cart_pole_system_dynamics0_P.Integrator3_IC;

  /* InitializeConditions for Integrator: '<S1>/Integrator2' */
  cart_pole_system_dynamics0_X.Integrator2_CSTATE =
    cart_pole_system_dynamics0_P.Integrator2_IC;

  /* InitializeConditions for Integrator: '<S1>/Integrator4' */
  cart_pole_system_dynamics0_X.Integrator4_CSTATE =
    cart_pole_system_dynamics0_P.Integrator4_IC;

  /* End of SystemInitialize for SubSystem: '<Root>/cart_pole_system_dynamics' */

  /* set "at time zero" to false */
  if (rtmIsFirstInitCond((&cart_pole_system_dynamics0_M))) {
    rtmSetFirstInitCond((&cart_pole_system_dynamics0_M), 0);
  }
}

/* Model terminate function */
void cart_pole_system_dynamics0::terminate()
{
  /* (no terminate code required) */
}

/* Constructor */
cart_pole_system_dynamics0::cart_pole_system_dynamics0() :
  cart_pole_system_dynamics0_U(),
  cart_pole_system_dynamics0_Y(),
  cart_pole_system_dynamics0_B(),
  cart_pole_system_dynamics0_DW(),
  cart_pole_system_dynamics0_X(),
  cart_pole_system_dynamics0_M()
{
  /* Currently there is no constructor body generated.*/
}

/* Destructor */
/* Currently there is no destructor body generated.*/
cart_pole_system_dynamics0::~cart_pole_system_dynamics0() = default;

/* Real-Time Model get method */
RT_MODEL_cart_pole_system_dyn_T * cart_pole_system_dynamics0::getRTM()
{
  return (&cart_pole_system_dynamics0_M);
}
