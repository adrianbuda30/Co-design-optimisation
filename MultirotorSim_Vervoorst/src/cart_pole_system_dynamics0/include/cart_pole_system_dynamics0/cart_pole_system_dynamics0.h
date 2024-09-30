/*
 * cart_pole_system_dynamics0.h
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "cart_pole_system_dynamics0".
 *
 * Model version              : 1.9
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Sun Sep 10 13:10:08 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#ifndef RTW_HEADER_cart_pole_system_dynamics0_h_
#define RTW_HEADER_cart_pole_system_dynamics0_h_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "cart_pole_system_dynamics0_types.h"
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

/* Block signals (default storage) */
struct B_cart_pole_system_dynamics0_T {
  real_T Cos;                          /* '<S1>/Cos' */
  real_T Sin;                          /* '<S1>/Sin' */
  real_T Add;                          /* '<S1>/Add' */
  real_T Delay3;                       /* '<S1>/Delay3' */
  real_T MathFunction;                 /* '<S1>/Math Function' */
  real_T X;                            /* '<S1>/Divide' */
  real_T Theta;                        /* '<S1>/Integrator' */
  real_T Theta_m;                      /* '<S1>/Integrator1' */
  real_T X_a;                          /* '<S1>/Integrator2' */
  real_T Theta_e;                      /* '<S1>/Product2' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_cart_pole_system_dynamics0_T {
  real_T Delay1_DSTATE;                /* '<S1>/Delay1' */
  real_T Delay_DSTATE;                 /* '<S1>/Delay' */
  real_T Delay2_DSTATE;                /* '<S1>/Delay2' */
  real_T Delay3_DSTATE;                /* '<S1>/Delay3' */
};

/* Continuous states (default storage) */
struct X_cart_pole_system_dynamics0_T {
  real_T Integrator_CSTATE;            /* '<S1>/Integrator' */
  real_T Integrator1_CSTATE;           /* '<S1>/Integrator1' */
  real_T Integrator2_CSTATE;           /* '<S1>/Integrator2' */
  real_T Integrator3_CSTATE;           /* '<S1>/Integrator3' */
};

/* State derivatives (default storage) */
struct XDot_cart_pole_system_dynamic_T {
  real_T Integrator_CSTATE;            /* '<S1>/Integrator' */
  real_T Integrator1_CSTATE;           /* '<S1>/Integrator1' */
  real_T Integrator2_CSTATE;           /* '<S1>/Integrator2' */
  real_T Integrator3_CSTATE;           /* '<S1>/Integrator3' */
};

/* State disabled  */
struct XDis_cart_pole_system_dynamic_T {
  boolean_T Integrator_CSTATE;         /* '<S1>/Integrator' */
  boolean_T Integrator1_CSTATE;        /* '<S1>/Integrator1' */
  boolean_T Integrator2_CSTATE;        /* '<S1>/Integrator2' */
  boolean_T Integrator3_CSTATE;        /* '<S1>/Integrator3' */
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
struct ExtU_cart_pole_system_dynamic_T {
  real_T length;                       /* '<Root>/length' */
  real_T mass_pole;                    /* '<Root>/mass_pole' */
  real_T mass_cart;                    /* '<Root>/mass_cart' */
  real_T force_input_cart;             /* '<Root>/force_input_cart' */
};

/* External outputs (root outports fed by signals with default storage) */
struct ExtY_cart_pole_system_dynamic_T {
  real_T cart_position;                /* '<Root>/cart_position' */
  real_T pole_angle;                   /* '<Root>/pole_angle' */
};

/* Parameters (default storage) */
struct P_cart_pole_system_dynamics0_T_ {
  real_T g;                            /* Variable: g
                                        * Referenced by: '<S1>/Constant2'
                                        */
  real_T Delay1_InitialCondition;      /* Expression: 0
                                        * Referenced by: '<S1>/Delay1'
                                        */
  real_T Delay_InitialCondition;       /* Expression: 0
                                        * Referenced by: '<S1>/Delay'
                                        */
  real_T Delay2_InitialCondition;      /* Expression: 0
                                        * Referenced by: '<S1>/Delay2'
                                        */
  real_T Delay3_InitialCondition;      /* Expression: 0
                                        * Referenced by: '<S1>/Delay3'
                                        */
  real_T Integrator_IC;                /* Expression: 0
                                        * Referenced by: '<S1>/Integrator'
                                        */
  real_T Integrator1_IC;               /* Expression: 0
                                        * Referenced by: '<S1>/Integrator1'
                                        */
  real_T Integrator2_IC;               /* Expression: 0
                                        * Referenced by: '<S1>/Integrator2'
                                        */
  real_T Integrator3_IC;               /* Expression: 0
                                        * Referenced by: '<S1>/Integrator3'
                                        */
};

/* Real-time Model Data Structure */
struct tag_RTM_cart_pole_system_dyna_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;
  X_cart_pole_system_dynamics0_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  XDis_cart_pole_system_dynamic_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[4];
  real_T odeF[3][4];
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
    uint32_T clockTickH0;
    time_T stepSize0;
    uint32_T clockTick1;
    uint32_T clockTickH1;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[2];
  } Timing;
};

/* Class declaration for model cart_pole_system_dynamics0 */
class cart_pole_system_dynamics0 final
{
  /* public data and function members */
 public:
  /* Copy Constructor */
  cart_pole_system_dynamics0(cart_pole_system_dynamics0 const&) = delete;

  /* Assignment Operator */
  cart_pole_system_dynamics0& operator= (cart_pole_system_dynamics0 const&) & =
    delete;

  /* Move Constructor */
  cart_pole_system_dynamics0(cart_pole_system_dynamics0 &&) = delete;

  /* Move Assignment Operator */
  cart_pole_system_dynamics0& operator= (cart_pole_system_dynamics0 &&) = delete;

  /* Real-Time Model get method */
  RT_MODEL_cart_pole_system_dyn_T * getRTM();

  /* Root inports set method */
  void setExternalInputs(const ExtU_cart_pole_system_dynamic_T
    *pExtU_cart_pole_system_dynamic_T)
  {
    cart_pole_system_dynamics0_U = *pExtU_cart_pole_system_dynamic_T;
  }

  /* Root outports get method */
  const ExtY_cart_pole_system_dynamic_T &getExternalOutputs() const
  {
    return cart_pole_system_dynamics0_Y;
  }

  /* Initial conditions function */
  void initialize();

  /* model step function */
  void step();

  /* model terminate function */
  static void terminate();

  /* Constructor */
  cart_pole_system_dynamics0();

  /* Destructor */
  ~cart_pole_system_dynamics0();

  /* private data and function members */
 private:
  /* External inputs */
  ExtU_cart_pole_system_dynamic_T cart_pole_system_dynamics0_U;

  /* External outputs */
  ExtY_cart_pole_system_dynamic_T cart_pole_system_dynamics0_Y;

  /* Block signals */
  B_cart_pole_system_dynamics0_T cart_pole_system_dynamics0_B;

  /* Block states */
  DW_cart_pole_system_dynamics0_T cart_pole_system_dynamics0_DW;

  /* Tunable parameters */
  static P_cart_pole_system_dynamics0_T cart_pole_system_dynamics0_P;

  /* Block continuous states */
  X_cart_pole_system_dynamics0_T cart_pole_system_dynamics0_X;

  /* Continuous states update member function*/
  void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si );

  /* Derivatives member function */
  void cart_pole_system_dynamics0_derivatives();

  /* Real-Time Model */
  RT_MODEL_cart_pole_system_dyn_T cart_pole_system_dynamics0_M;
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
 * hilite_system('CartPole_env/cart_pole_system_dynamics')    - opens subsystem CartPole_env/cart_pole_system_dynamics
 * hilite_system('CartPole_env/cart_pole_system_dynamics/Kp') - opens and selects block Kp
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'CartPole_env'
 * '<S1>'   : 'CartPole_env/cart_pole_system_dynamics'
 */
#endif                            /* RTW_HEADER_cart_pole_system_dynamics0_h_ */
