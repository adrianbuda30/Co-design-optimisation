/*
 * aerofoil.h
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
 */

#ifndef RTW_HEADER_aerofoil_h_
#define RTW_HEADER_aerofoil_h_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "aerofoil_types.h"
#include <cstring>

/* Macros for accessing real-time model data structure */
#ifndef rtmGetContStateDisabled
#define rtmGetContStateDisabled(rtm)   ((rtm)->contStateDisabled)
#endif

#ifndef rtmSetContStateDisabled
#define rtmSetContStateDisabled(rtm, val) ((rtm)->contStateDisabled = (val))
#endif

#ifndef rtmGetContStates
#define rtmGetContStates(rtm)          ((rtm)->contStates)
#endif

#ifndef rtmSetContStates
#define rtmSetContStates(rtm, val)     ((rtm)->contStates = (val))
#endif

#ifndef rtmGetContTimeOutputInconsistentWithStateAtMajorStepFlag
#define rtmGetContTimeOutputInconsistentWithStateAtMajorStepFlag(rtm) ((rtm)->CTOutputIncnstWithState)
#endif

#ifndef rtmSetContTimeOutputInconsistentWithStateAtMajorStepFlag
#define rtmSetContTimeOutputInconsistentWithStateAtMajorStepFlag(rtm, val) ((rtm)->CTOutputIncnstWithState = (val))
#endif

#ifndef rtmGetDerivCacheNeedsReset
#define rtmGetDerivCacheNeedsReset(rtm) ((rtm)->derivCacheNeedsReset)
#endif

#ifndef rtmSetDerivCacheNeedsReset
#define rtmSetDerivCacheNeedsReset(rtm, val) ((rtm)->derivCacheNeedsReset = (val))
#endif

#ifndef rtmGetIntgData
#define rtmGetIntgData(rtm)            ((rtm)->intgData)
#endif

#ifndef rtmSetIntgData
#define rtmSetIntgData(rtm, val)       ((rtm)->intgData = (val))
#endif

#ifndef rtmGetOdeF
#define rtmGetOdeF(rtm)                ((rtm)->odeF)
#endif

#ifndef rtmSetOdeF
#define rtmSetOdeF(rtm, val)           ((rtm)->odeF = (val))
#endif

#ifndef rtmGetOdeY
#define rtmGetOdeY(rtm)                ((rtm)->odeY)
#endif

#ifndef rtmSetOdeY
#define rtmSetOdeY(rtm, val)           ((rtm)->odeY = (val))
#endif

#ifndef rtmGetPeriodicContStateIndices
#define rtmGetPeriodicContStateIndices(rtm) ((rtm)->periodicContStateIndices)
#endif

#ifndef rtmSetPeriodicContStateIndices
#define rtmSetPeriodicContStateIndices(rtm, val) ((rtm)->periodicContStateIndices = (val))
#endif

#ifndef rtmGetPeriodicContStateRanges
#define rtmGetPeriodicContStateRanges(rtm) ((rtm)->periodicContStateRanges)
#endif

#ifndef rtmSetPeriodicContStateRanges
#define rtmSetPeriodicContStateRanges(rtm, val) ((rtm)->periodicContStateRanges = (val))
#endif

#ifndef rtmGetZCCacheNeedsReset
#define rtmGetZCCacheNeedsReset(rtm)   ((rtm)->zCCacheNeedsReset)
#endif

#ifndef rtmSetZCCacheNeedsReset
#define rtmSetZCCacheNeedsReset(rtm, val) ((rtm)->zCCacheNeedsReset = (val))
#endif

#ifndef rtmGetdX
#define rtmGetdX(rtm)                  ((rtm)->derivs)
#endif

#ifndef rtmSetdX
#define rtmSetdX(rtm, val)             ((rtm)->derivs = (val))
#endif

#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   (rtmGetTPtr((rtm))[0])
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

#ifndef rtmGetTStart
#define rtmGetTStart(rtm)              ((rtm)->Timing.tStart)
#endif

/* Block signals (default storage) */
struct B_aerofoil_T {
  real_T Add[12];                      /* '<S1>/Add' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_aerofoil_T {
  int_T Integrator_IWORK;              /* '<S1>/Integrator' */
};

/* Continuous states (default storage) */
struct X_aerofoil_T {
  real_T Integrator_CSTATE[12];        /* '<S1>/Integrator' */
};

/* State derivatives (default storage) */
struct XDot_aerofoil_T {
  real_T Integrator_CSTATE[12];        /* '<S1>/Integrator' */
};

/* State disabled  */
struct XDis_aerofoil_T {
  boolean_T Integrator_CSTATE[12];     /* '<S1>/Integrator' */
};

#ifndef ODE3_INTG
#define ODE3_INTG

/* ODE3 Integration Data */
struct ODE3_IntgData {
  real_T *y;                           /* output */
  real_T *f[3];                        /* derivatives */
};

#endif

/* External inputs (root inport signals with default storage) */
struct ExtU_aerofoil_T {
  real_T MatrixA[144];                 /* '<Root>/MatrixA' */
  real_T MatrixB[12];                  /* '<Root>/MatrixB' */
  real_T MatrixC[12];                  /* '<Root>/MatrixC' */
  real_T delta_ddot;                   /* '<Root>/delta_ddot' */
  real_T init_state[12];               /* '<Root>/init_state' */
};

/* External outputs (root outports fed by signals with default storage) */
struct ExtY_aerofoil_T {
  real_T plunge;                       /* '<Root>/plunge' */
  real_T pitch;                        /* '<Root>/pitch' */
  real_T delta;                        /* '<Root>/delta' */
  real_T plunge_dot;                   /* '<Root>/plunge_dot' */
  real_T pitch_dot;                    /* '<Root>/pitch_dot' */
  real_T delta_dot;                    /* '<Root>/delta_dot' */
  real_T C_L;                          /* '<Root>/C_L' */
};

/* Real-time Model Data Structure */
struct tag_RTM_aerofoil_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;
  X_aerofoil_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  XDis_aerofoil_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[12];
  real_T odeF[3][12];
  ODE3_IntgData intgData;

  /*
   * Sizes:
   * The following substructure contains sizes information
   * for many of the model attributes such as inputs, outputs,
   * dwork, sample times, etc.
   */
  struct {
    int_T numContStates;
    int_T numPeriodicContStates;
    int_T numSampTimes;
  } Sizes;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    uint32_T clockTick0;
    time_T stepSize0;
    uint32_T clockTick1;
    boolean_T firstInitCondFlag;
    time_T tStart;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[2];
  } Timing;
};

/* Class declaration for model aerofoil */
class aerofoil
{
  /* public data and function members */
 public:
  /* Copy Constructor */
  aerofoil(aerofoil const&) = delete;

  /* Assignment Operator */
  aerofoil& operator= (aerofoil const&) & = delete;

  /* Move Constructor */
  aerofoil(aerofoil &&) = delete;

  /* Move Assignment Operator */
  aerofoil& operator= (aerofoil &&) = delete;

  /* Real-Time Model get method */
  RT_MODEL_aerofoil_T * getRTM();

  /* External inputs */
  ExtU_aerofoil_T aerofoil_U;

  /* External outputs */
  ExtY_aerofoil_T aerofoil_Y;

  /* Initial conditions function */
  void initialize();

  /* model step function */
  void step();

  /* model terminate function */
  static void terminate();

  /* Constructor */
  aerofoil();

  /* Destructor */
  ~aerofoil();

  /* private data and function members */
 private:
  /* Block signals */
  B_aerofoil_T aerofoil_B;

  /* Block states */
  DW_aerofoil_T aerofoil_DW;

  /* Block continuous states */
  X_aerofoil_T aerofoil_X;

  /* Block Continuous state disabled vector */
  XDis_aerofoil_T aerofoil_XDis;

  /* Continuous states update member function*/
  void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si );

  /* Derivatives member function */
  void aerofoil_derivatives();

  /* Real-Time Model */
  RT_MODEL_aerofoil_T aerofoil_M;
};

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Note that this particular code originates from a subsystem build,
 * and has its own system numbers different from the parent model.
 * Refer to the system hierarchy for this subsystem below, and use the
 * MATLAB hilite_system command to trace the generated code back
 * to the parent model.  For example,
 *
 * hilite_system('aeroelastic_control_final_2023a_noviterna/aerofoil')    - opens subsystem aeroelastic_control_final_2023a_noviterna/aerofoil
 * hilite_system('aeroelastic_control_final_2023a_noviterna/aerofoil/Kp') - opens and selects block Kp
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'aeroelastic_control_final_2023a_noviterna'
 * '<S1>'   : 'aeroelastic_control_final_2023a_noviterna/aerofoil'
 */
#endif                                 /* RTW_HEADER_aerofoil_h_ */
