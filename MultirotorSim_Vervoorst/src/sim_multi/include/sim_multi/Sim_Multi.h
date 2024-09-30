/*
 * Sim_Multi.h
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

#ifndef RTW_HEADER_Sim_Multi_h_
#define RTW_HEADER_Sim_Multi_h_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "Sim_Multi_types.h"

extern "C"
{

#include "rt_nonfinite.h"

}

extern "C"
{

#include "rtGetInf.h"

}

extern "C"
{

#include "rtGetNaN.h"

}

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

#ifndef rtmCounterLimit
#define rtmCounterLimit(rtm, idx)      ((rtm)->Timing.TaskCounters.cLimit[(idx)])
#endif

#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmStepTask
#define rtmStepTask(rtm, idx)          ((rtm)->Timing.TaskCounters.TID[(idx)] == 0)
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

#ifndef rtmTaskCounter
#define rtmTaskCounter(rtm, idx)       ((rtm)->Timing.TaskCounters.TID[(idx)])
#endif

/* Block signals for system '<S94>/For Each Subsystem' */
struct B_CoreSubsys_Sim_Multi_c_T {
  real_T Product;                      /* '<S114>/Product' */
  real_T Switch;                       /* '<S118>/Switch' */
  real_T Switch_a;                     /* '<S119>/Switch' */
  real_T VectorfromrealCoGtopropellerBod[3];/* '<S116>/Subtract' */
  real_T VectorConcatenate[9];         /* '<S169>/Vector Concatenate' */
  real_T Product9[3];                  /* '<S136>/Product9' */
  real_T Gain1;                        /* '<S156>/Gain1' */
  real_T Angleofattackrad;             /* '<S143>/Merge' */
  real_T Climbspeedv_c;                /* '<S144>/Gain' */
  real_T NewtiltedthrustdirectionBodyaxe[3];/* '<S131>/Merge' */
  real_T Momentinthemotorhubduetobending[3];/* '<S131>/Merge1' */
};

/* Block states (default storage) for system '<S94>/For Each Subsystem' */
struct DW_CoreSubsys_Sim_Multi_f_T {
  int8_T If_ActiveSubsystem;           /* '<S131>/If' */
  int8_T If_ActiveSubsystem_l;         /* '<S143>/If' */
  int8_T If_ActiveSubsystem_g;         /* '<S145>/If' */
};

/* Continuous states for system '<S94>/For Each Subsystem' */
struct X_CoreSubsys_Sim_Multi_n_T {
  real_T Integrator_CSTATE_e;          /* '<S118>/Integrator' */
  real_T Integrator_CSTATE_o;          /* '<S119>/Integrator' */
};

/* State derivatives for system '<S94>/For Each Subsystem' */
struct XDot_CoreSubsys_Sim_Multi_n_T {
  real_T Integrator_CSTATE_e;          /* '<S118>/Integrator' */
  real_T Integrator_CSTATE_o;          /* '<S119>/Integrator' */
};

/* State Disabled for system '<S94>/For Each Subsystem' */
struct XDis_CoreSubsys_Sim_Multi_n_T {
  boolean_T Integrator_CSTATE_e;       /* '<S118>/Integrator' */
  boolean_T Integrator_CSTATE_o;       /* '<S119>/Integrator' */
};

/* Block signals (default storage) */
struct B_Sim_Multi_T {
  real_T TransferFcn1;                 /* '<S31>/Transfer Fcn1' */
  real_T TransferFcn6;                 /* '<S31>/Transfer Fcn6' */
  real_T TransferFcn4;                 /* '<S31>/Transfer Fcn4' */
  real_T TransferFcn5;                 /* '<S31>/Transfer Fcn5' */
  real_T TransferFcn2;                 /* '<S31>/Transfer Fcn2' */
  real_T TransferFcn3;                 /* '<S31>/Transfer Fcn3' */
  real_T Gain2;                        /* '<S32>/Gain2' */
  real_T Gain2_i;                      /* '<S33>/Gain2' */
  real_T Gain2_g;                      /* '<S34>/Gain2' */
  real_T omega[3];                     /* '<S35>/Sum' */
  real_T a_b[3];                       /* '<S35>/Sum1' */
  real_T UniformRandomNumber;          /* '<S37>/Uniform Random Number' */
  real_T UniformRandomNumber_n;        /* '<S38>/Uniform Random Number' */
  real_T Switch;                       /* '<S42>/Switch' */
  real_T Divide[4];                    /* '<S41>/Divide' */
  real_T Switch_j;                     /* '<S45>/Switch' */
  real_T Switch1;                      /* '<S45>/Switch1' */
  real_T Switch2;                      /* '<S45>/Switch2' */
  real_T Sum1[3];                      /* '<S36>/Sum1' */
  real_T Switch_jm;                    /* '<S44>/Switch' */
  real_T Switch1_m;                    /* '<S44>/Switch1' */
  real_T Switch2_m;                    /* '<S44>/Switch2' */
  real_T Sum2[3];                      /* '<S36>/Sum2' */
  real_T Switch_f;                     /* '<S43>/Switch' */
  real_T Switch1_d;                    /* '<S43>/Switch1' */
  real_T Switch2_n;                    /* '<S43>/Switch2' */
  real_T Sum3[3];                      /* '<S36>/Sum3' */
  real_T RateTransition[4];            /* '<S3>/Rate Transition' */
  real_T Product[3];                   /* '<S61>/Product' */
  real_T RateTransition1[4];           /* '<S5>/Rate Transition1' */
  real_T Product_l[3];                 /* '<S63>/Product' */
  real_T ForceofgravityInertialaxes[3];/* '<S93>/Product1' */
  real_T TmpSignalConversionAtQIntegrato[4];
  real_T Sum1_o[3];                    /* '<S58>/Sum1' */
  real_T quat_output[4];               /* '<S64>/MATLAB Function' */
  real_T ImpAsg_InsertedFor_RPM_cmd_sat_[4];/* '<S24>/Saturation Dynamic' */
  real_T Forceagainstdirectionofmotiondu[3];/* '<S95>/Merge' */
  B_CoreSubsys_Sim_Multi_c_T CoreSubsys_p[4];/* '<S94>/For Each Subsystem' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_Sim_Multi_T {
  real_T DiscreteTimeIntegrator_DSTATE[3];/* '<S11>/Discrete-Time Integrator' */
  real_T RateTransition_Buffer[3];     /* '<S30>/Rate Transition' */
  real_T Memory_PreviousInput[4];      /* '<S36>/Memory' */
  real_T Memory1_PreviousInput[3];     /* '<S36>/Memory1' */
  real_T Memory2_PreviousInput[3];     /* '<S36>/Memory2' */
  real_T Memory3_PreviousInput[3];     /* '<S36>/Memory3' */
  real_T RateTransition_Buffer_b[19];  /* '<S4>/Rate Transition' */
  real_T UniformRandomNumber_NextOutput;/* '<S37>/Uniform Random Number' */
  real_T UniformRandomNumber_NextOutpu_m;/* '<S38>/Uniform Random Number' */
  real_T RateTransition1_Buffer0[4];   /* '<S5>/Rate Transition1' */
  uint32_T RandSeed;                   /* '<S37>/Uniform Random Number' */
  uint32_T RandSeed_b;                 /* '<S38>/Uniform Random Number' */
  int_T QIntegrator_IWORK;             /* '<S64>/Q-Integrator' */
  int8_T If_ActiveSubsystem;           /* '<S95>/If' */
  DW_CoreSubsys_Sim_Multi_f_T CoreSubsys_p[4];/* '<S94>/For Each Subsystem' */
};

/* Continuous states (default storage) */
struct X_Sim_Multi_T {
  real_T TransferFcn1_CSTATE;          /* '<S31>/Transfer Fcn1' */
  real_T TransferFcn6_CSTATE;          /* '<S31>/Transfer Fcn6' */
  real_T TransferFcn4_CSTATE;          /* '<S31>/Transfer Fcn4' */
  real_T TransferFcn5_CSTATE;          /* '<S31>/Transfer Fcn5' */
  real_T TransferFcn2_CSTATE;          /* '<S31>/Transfer Fcn2' */
  real_T TransferFcn3_CSTATE;          /* '<S31>/Transfer Fcn3' */
  real_T Integrator_CSTATE;            /* '<S37>/Integrator' */
  real_T Integrator_CSTATE_f;          /* '<S38>/Integrator' */
  real_T QIntegrator_CSTATE[4];        /* '<S64>/Q-Integrator' */
  real_T V_b_CSTATE[3];                /* '<S58>/V_b' */
  real_T omega_CSTATE[3];              /* '<S58>/omega' */
  real_T X_i_CSTATE[3];                /* '<S58>/X_i' */
  X_CoreSubsys_Sim_Multi_n_T CoreSubsys_p[4];/* '<S112>/CoreSubsys' */
};

/* State derivatives (default storage) */
struct XDot_Sim_Multi_T {
  real_T TransferFcn1_CSTATE;          /* '<S31>/Transfer Fcn1' */
  real_T TransferFcn6_CSTATE;          /* '<S31>/Transfer Fcn6' */
  real_T TransferFcn4_CSTATE;          /* '<S31>/Transfer Fcn4' */
  real_T TransferFcn5_CSTATE;          /* '<S31>/Transfer Fcn5' */
  real_T TransferFcn2_CSTATE;          /* '<S31>/Transfer Fcn2' */
  real_T TransferFcn3_CSTATE;          /* '<S31>/Transfer Fcn3' */
  real_T Integrator_CSTATE;            /* '<S37>/Integrator' */
  real_T Integrator_CSTATE_f;          /* '<S38>/Integrator' */
  real_T QIntegrator_CSTATE[4];        /* '<S64>/Q-Integrator' */
  real_T V_b_CSTATE[3];                /* '<S58>/V_b' */
  real_T omega_CSTATE[3];              /* '<S58>/omega' */
  real_T X_i_CSTATE[3];                /* '<S58>/X_i' */
  XDot_CoreSubsys_Sim_Multi_n_T CoreSubsys_p[4];/* '<S112>/CoreSubsys' */
};

/* State disabled  */
struct XDis_Sim_Multi_T {
  boolean_T TransferFcn1_CSTATE;       /* '<S31>/Transfer Fcn1' */
  boolean_T TransferFcn6_CSTATE;       /* '<S31>/Transfer Fcn6' */
  boolean_T TransferFcn4_CSTATE;       /* '<S31>/Transfer Fcn4' */
  boolean_T TransferFcn5_CSTATE;       /* '<S31>/Transfer Fcn5' */
  boolean_T TransferFcn2_CSTATE;       /* '<S31>/Transfer Fcn2' */
  boolean_T TransferFcn3_CSTATE;       /* '<S31>/Transfer Fcn3' */
  boolean_T Integrator_CSTATE;         /* '<S37>/Integrator' */
  boolean_T Integrator_CSTATE_f;       /* '<S38>/Integrator' */
  boolean_T QIntegrator_CSTATE[4];     /* '<S64>/Q-Integrator' */
  boolean_T V_b_CSTATE[3];             /* '<S58>/V_b' */
  boolean_T omega_CSTATE[3];           /* '<S58>/omega' */
  boolean_T X_i_CSTATE[3];             /* '<S58>/X_i' */
  XDis_CoreSubsys_Sim_Multi_n_T CoreSubsys_p[4];/* '<S112>/CoreSubsys' */
};

/* Invariant block signals (default storage) */
struct ConstB_Sim_Multi_T {
  real_T Product1[9];                  /* '<S13>/Product1' */
  real_T Product1_i[9];                /* '<S14>/Product1' */
  real_T Product1_g[9];                /* '<S15>/Product1' */
};

#ifndef ODE3_INTG
#define ODE3_INTG

/* ODE3 Integration Data */
struct ODE3_IntgData {
  real_T *y;                           /* output */
  real_T *f[3];                        /* derivatives */
};

#endif

/* Constant parameters (default storage) */
struct ConstP_Sim_Multi_T {
  /* Pooled Parameter (Mixed Expressions)
   * Referenced by:
   *   '<S2>/Constant4'
   *   '<S59>/Constant'
   */
  real_T pooled10[68];

  /* Pooled Parameter (Mixed Expressions)
   * Referenced by:
   *   '<S2>/Constant7'
   *   '<S59>/Constant3'
   */
  real_T pooled11[9];

  /* Expression: MotorMap
   * Referenced by: '<S8>/Constant1'
   */
  real_T Constant1_Value_n[16];

  /* Expression: MaxRate_cmd/500
   * Referenced by: '<S10>/Unit conversion [stick value] to [rad//s]'
   */
  real_T Unitconversionstickvaluetorads_[3];

  /* Expression: MaxRate_cmd
   * Referenced by: '<S11>/Saturation'
   */
  real_T Saturation_UpperSat[3];

  /* Expression: -MaxRate_cmd
   * Referenced by: '<S11>/Saturation'
   */
  real_T Saturation_LowerSat[3];
};

/* Real-time Model Data Structure */
struct tag_RTM_Sim_Multi_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;
  X_Sim_Multi_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  XDis_Sim_Multi_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[29];
  real_T odeF[3][29];
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
    struct {
      uint8_T TID[6];
      uint8_T cLimit[6];
    } TaskCounters;

    struct {
      uint8_T TID1_2;
    } RateInteraction;

    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[6];
  } Timing;
};

extern const ConstB_Sim_Multi_T Sim_Multi_ConstB;/* constant block i/o */

/* Constant parameters (default storage) */
extern const ConstP_Sim_Multi_T Sim_Multi_ConstP;

/* Class declaration for model Sim_Multi */
class Sim_Multi final
{
  /* public data and function members */
 public:
  /* Copy Constructor */
  Sim_Multi(Sim_Multi const&) = delete;

  /* Assignment Operator */
  Sim_Multi& operator= (Sim_Multi const&) & = delete;

  /* Move Constructor */
  Sim_Multi(Sim_Multi &&) = delete;

  /* Move Assignment Operator */
  Sim_Multi& operator= (Sim_Multi &&) = delete;

  /* Real-Time Model get method */
  RT_MODEL_Sim_Multi_T * getRTM();

  /* model start function */
  void start();

  /* Initial conditions function */
  void initialize();

  /* model step function */
  void step0();

  /* model step function */
  void step2();

  /* model step function */
  void step3();

  /* model step function */
  void step4();

  /* model step function */
  void step5();

  /* model terminate function */
  static void terminate();

  /* Constructor */
  Sim_Multi();

  /* Destructor */
  ~Sim_Multi();

  /* private data and function members */
 private:
  /* Block signals */
  B_Sim_Multi_T Sim_Multi_B;

  /* Block states */
  DW_Sim_Multi_T Sim_Multi_DW;

  /* Block continuous states */
  X_Sim_Multi_T Sim_Multi_X;

  /* Continuous states update member function*/
  void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si );

  /* Derivatives member function */
  void Sim_Multi_derivatives();

  /* Real-Time Model */
  RT_MODEL_Sim_Multi_T Sim_Multi_M;
};

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<S26>/Data Type Duplicate' : Unused code path elimination
 * Block '<S26>/Data Type Propagation' : Unused code path elimination
 * Block '<S29>/Data Type Duplicate' : Unused code path elimination
 * Block '<S29>/Data Type Propagation' : Unused code path elimination
 * Block '<S2>/Constant' : Unused code path elimination
 * Block '<S2>/Constant1' : Unused code path elimination
 * Block '<S2>/Constant2' : Unused code path elimination
 * Block '<S2>/Constant3' : Unused code path elimination
 * Block '<S2>/Constant5' : Unused code path elimination
 * Block '<S2>/Constant6' : Unused code path elimination
 * Block '<S2>/Disturbance (forces)' : Unused code path elimination
 * Block '<S2>/Disturbance (torques)' : Unused code path elimination
 * Block '<S2>/Surface area params' : Unused code path elimination
 * Block '<S2>/Wind vector' : Unused code path elimination
 * Block '<S35>/Noisy acc' : Unused code path elimination
 * Block '<S35>/Noisy omega' : Unused code path elimination
 * Block '<S36>/Noisy Vb' : Unused code path elimination
 * Block '<S36>/Noisy Vi' : Unused code path elimination
 * Block '<S36>/Noisy Xi' : Unused code path elimination
 * Block '<S36>/Noisy quat' : Unused code path elimination
 * Block '<S62>/Math Function2' : Unused code path elimination
 * Block '<S62>/Product' : Unused code path elimination
 * Block '<S62>/Reshape' : Unused code path elimination
 * Block '<S62>/Reshape1' : Unused code path elimination
 * Block '<S65>/Reshape' : Unused code path elimination
 * Block '<S59>/Constant4' : Unused code path elimination
 * Block '<S59>/Constant5' : Unused code path elimination
 * Block '<S59>/Constant6' : Unused code path elimination
 * Block '<S59>/Constant7' : Unused code path elimination
 * Block '<S122>/Data Type Duplicate' : Unused code path elimination
 * Block '<S122>/Data Type Propagation' : Unused code path elimination
 * Block '<S125>/Data Type Duplicate' : Unused code path elimination
 * Block '<S125>/Data Type Propagation' : Unused code path elimination
 * Block '<S125>/LowerRelop1' : Unused code path elimination
 * Block '<S125>/Switch' : Unused code path elimination
 * Block '<S125>/Switch2' : Unused code path elimination
 * Block '<S125>/UpperRelop' : Unused code path elimination
 * Block '<S94>/Reshape' : Unused code path elimination
 * Block '<S19>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S21>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S23>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S7>/Rate Transition' : Eliminated since input and output rates are identical
 * Block '<S8>/Rate Transition' : Eliminated since input and output rates are identical
 * Block '<S1>/Reshape' : Reshape block reduction
 * Block '<S32>/Gain' : Eliminated nontunable gain of 1
 * Block '<S32>/Manual Switch' : Eliminated due to constant selection input
 * Block '<S32>/Manual Switch1' : Eliminated due to constant selection input
 * Block '<S33>/Manual Switch' : Eliminated due to constant selection input
 * Block '<S33>/Manual Switch1' : Eliminated due to constant selection input
 * Block '<S34>/Manual Switch' : Eliminated due to constant selection input
 * Block '<S34>/Manual Switch1' : Eliminated due to constant selection input
 * Block '<S35>/Gain' : Eliminated nontunable gain of 1
 * Block '<S35>/Gain1' : Eliminated nontunable gain of 1
 * Block '<S47>/Reshape' : Reshape block reduction
 * Block '<S47>/Reshape1' : Reshape block reduction
 * Block '<S61>/Reshape' : Reshape block reduction
 * Block '<S61>/Reshape1' : Reshape block reduction
 * Block '<S63>/Reshape' : Reshape block reduction
 * Block '<S63>/Reshape ' : Reshape block reduction
 * Block '<S69>/Reshape' : Reshape block reduction
 * Block '<S69>/Reshape1' : Reshape block reduction
 * Block '<S73>/Reshape' : Reshape block reduction
 * Block '<S73>/Reshape1' : Reshape block reduction
 * Block '<S79>/Reshape' : Reshape block reduction
 * Block '<S79>/Reshape1' : Reshape block reduction
 * Block '<S89>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S104>/Reshape' : Reshape block reduction
 * Block '<S104>/Reshape1' : Reshape block reduction
 * Block '<S108>/Reshape' : Reshape block reduction
 * Block '<S108>/Reshape1' : Reshape block reduction
 * Block '<S109>/Reshape' : Reshape block reduction
 * Block '<S109>/Reshape1' : Reshape block reduction
 * Block '<S110>/Reshape' : Reshape block reduction
 * Block '<S110>/Reshape1' : Reshape block reduction
 * Block '<S111>/Reshape' : Reshape block reduction
 * Block '<S111>/Reshape1' : Reshape block reduction
 * Block '<S96>/Reshape' : Reshape block reduction
 * Block '<S93>/Reshape' : Reshape block reduction
 * Block '<S115>/Reshape' : Reshape block reduction
 * Block '<S116>/Reshape1' : Reshape block reduction
 * Block '<S112>/Rate Transition' : Eliminated since input and output rates are identical
 * Block '<S137>/Reshape3' : Reshape block reduction
 * Block '<S141>/Reshape' : Reshape block reduction
 * Block '<S141>/Reshape1' : Reshape block reduction
 * Block '<S142>/Reshape' : Reshape block reduction
 * Block '<S142>/Reshape1' : Reshape block reduction
 * Block '<S150>/Reshape' : Reshape block reduction
 * Block '<S150>/Reshape1' : Reshape block reduction
 * Block '<S151>/Reshape' : Reshape block reduction
 * Block '<S151>/Reshape1' : Reshape block reduction
 * Block '<S133>/Reshape1' : Reshape block reduction
 * Block '<S133>/Reshape2' : Reshape block reduction
 * Block '<S136>/Reshape5' : Reshape block reduction
 * Block '<S169>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S117>/Reshape' : Reshape block reduction
 * Block '<S172>/Reshape' : Reshape block reduction
 * Block '<S172>/Reshape1' : Reshape block reduction
 * Block '<S113>/Reshape' : Reshape block reduction
 * Block '<S11>/Constant2' : Unused code path elimination
 * Block '<S11>/Constant3' : Unused code path elimination
 * Block '<S32>/Add1' : Unused code path elimination
 * Block '<S32>/Constant' : Unused code path elimination
 * Block '<S32>/Step3' : Unused code path elimination
 * Block '<S32>/Step4' : Unused code path elimination
 * Block '<S32>/Step5' : Unused code path elimination
 * Block '<S33>/Constant' : Unused code path elimination
 * Block '<S33>/Step3' : Unused code path elimination
 * Block '<S34>/Add1' : Unused code path elimination
 * Block '<S34>/Add2' : Unused code path elimination
 * Block '<S34>/Gain' : Unused code path elimination
 * Block '<S34>/Gain1' : Unused code path elimination
 * Block '<S34>/Step1' : Unused code path elimination
 * Block '<S34>/Step2' : Unused code path elimination
 * Block '<S34>/Step3' : Unused code path elimination
 * Block '<S34>/Step6' : Unused code path elimination
 * Block '<S34>/Step7' : Unused code path elimination
 * Block '<S34>/Step8' : Unused code path elimination
 * Block '<S39>/Output' : Unused code path elimination
 * Block '<S39>/White Noise' : Unused code path elimination
 * Block '<S37>/Constant' : Unused code path elimination
 * Block '<S37>/Gain1' : Unused code path elimination
 * Block '<S37>/Gain2' : Unused code path elimination
 * Block '<S37>/Sum1' : Unused code path elimination
 * Block '<S40>/Output' : Unused code path elimination
 * Block '<S40>/White Noise' : Unused code path elimination
 * Block '<S38>/Constant' : Unused code path elimination
 * Block '<S38>/Gain1' : Unused code path elimination
 * Block '<S38>/Gain2' : Unused code path elimination
 * Block '<S38>/Sum1' : Unused code path elimination
 * Block '<S48>/Output' : Unused code path elimination
 * Block '<S48>/White Noise' : Unused code path elimination
 * Block '<S42>/Constant' : Unused code path elimination
 * Block '<S42>/Gain2' : Unused code path elimination
 * Block '<S42>/Rate Transition1' : Unused code path elimination
 * Block '<S49>/Output' : Unused code path elimination
 * Block '<S49>/White Noise' : Unused code path elimination
 * Block '<S50>/Output' : Unused code path elimination
 * Block '<S50>/White Noise' : Unused code path elimination
 * Block '<S51>/Output' : Unused code path elimination
 * Block '<S51>/White Noise' : Unused code path elimination
 * Block '<S43>/Constant' : Unused code path elimination
 * Block '<S43>/Constant2' : Unused code path elimination
 * Block '<S43>/Constant4' : Unused code path elimination
 * Block '<S43>/Gain1' : Unused code path elimination
 * Block '<S43>/Gain2' : Unused code path elimination
 * Block '<S43>/Gain3' : Unused code path elimination
 * Block '<S43>/Rate Transition' : Unused code path elimination
 * Block '<S43>/Rate Transition1' : Unused code path elimination
 * Block '<S43>/Rate Transition2' : Unused code path elimination
 * Block '<S52>/Output' : Unused code path elimination
 * Block '<S52>/White Noise' : Unused code path elimination
 * Block '<S53>/Output' : Unused code path elimination
 * Block '<S53>/White Noise' : Unused code path elimination
 * Block '<S54>/Output' : Unused code path elimination
 * Block '<S54>/White Noise' : Unused code path elimination
 * Block '<S44>/Constant' : Unused code path elimination
 * Block '<S44>/Constant2' : Unused code path elimination
 * Block '<S44>/Constant4' : Unused code path elimination
 * Block '<S44>/Gain1' : Unused code path elimination
 * Block '<S44>/Gain2' : Unused code path elimination
 * Block '<S44>/Gain3' : Unused code path elimination
 * Block '<S44>/Rate Transition' : Unused code path elimination
 * Block '<S44>/Rate Transition1' : Unused code path elimination
 * Block '<S44>/Rate Transition2' : Unused code path elimination
 * Block '<S55>/Output' : Unused code path elimination
 * Block '<S55>/White Noise' : Unused code path elimination
 * Block '<S56>/Output' : Unused code path elimination
 * Block '<S56>/White Noise' : Unused code path elimination
 * Block '<S57>/Output' : Unused code path elimination
 * Block '<S57>/White Noise' : Unused code path elimination
 * Block '<S45>/Constant' : Unused code path elimination
 * Block '<S45>/Constant2' : Unused code path elimination
 * Block '<S45>/Constant4' : Unused code path elimination
 * Block '<S45>/Gain1' : Unused code path elimination
 * Block '<S45>/Gain2' : Unused code path elimination
 * Block '<S45>/Gain3' : Unused code path elimination
 * Block '<S45>/Rate Transition' : Unused code path elimination
 * Block '<S45>/Rate Transition1' : Unused code path elimination
 * Block '<S45>/Rate Transition2' : Unused code path elimination
 * Block '<S132>/Constant1' : Unused code path elimination
 * Block '<S132>/Divide' : Unused code path elimination
 * Block '<S132>/Product2' : Unused code path elimination
 * Block '<S132>/Sum2' : Unused code path elimination
 * Block '<S132>/Trigonometric Function' : Unused code path elimination
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'Sim_Multi'
 * '<S1>'   : 'Sim_Multi/Controller'
 * '<S2>'   : 'Sim_Multi/Disturbances & Dynamic Events'
 * '<S3>'   : 'Sim_Multi/Radio commands'
 * '<S4>'   : 'Sim_Multi/State Estimation, Sensors, Noise'
 * '<S5>'   : 'Sim_Multi/multirotor'
 * '<S6>'   : 'Sim_Multi/Controller/Controller Structure'
 * '<S7>'   : 'Sim_Multi/Controller/Mapping thrust to rpm and saturation'
 * '<S8>'   : 'Sim_Multi/Controller/Motor mapping'
 * '<S9>'   : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only'
 * '<S10>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/Conversion stick values to rate commands'
 * '<S11>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller'
 * '<S12>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Cross Product'
 * '<S13>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Kff'
 * '<S14>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Ki'
 * '<S15>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Kp'
 * '<S16>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Cross Product/Subsystem'
 * '<S17>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Cross Product/Subsystem1'
 * '<S18>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Kff/Create diagonal 3x3 matrix'
 * '<S19>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Kff/Create diagonal 3x3 matrix/Subsystem9'
 * '<S20>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Ki/Create diagonal 3x3 matrix'
 * '<S21>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Ki/Create diagonal 3x3 matrix/Subsystem9'
 * '<S22>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Kp/Create diagonal 3x3 matrix'
 * '<S23>'  : 'Sim_Multi/Controller/Controller Structure/Inner-loop control only/PI Rate controller/Gain Kp/Create diagonal 3x3 matrix/Subsystem9'
 * '<S24>'  : 'Sim_Multi/Controller/Mapping thrust to rpm and saturation/For Each Subsystem'
 * '<S25>'  : 'Sim_Multi/Controller/Mapping thrust to rpm and saturation/For Each Subsystem/MATLAB Function'
 * '<S26>'  : 'Sim_Multi/Controller/Mapping thrust to rpm and saturation/For Each Subsystem/Saturation Dynamic'
 * '<S27>'  : 'Sim_Multi/Controller/Motor mapping/Max thrust limit'
 * '<S28>'  : 'Sim_Multi/Controller/Motor mapping/Min thrust limit'
 * '<S29>'  : 'Sim_Multi/Controller/Motor mapping/Saturation Dynamic'
 * '<S30>'  : 'Sim_Multi/Radio commands/Rate commands'
 * '<S31>'  : 'Sim_Multi/Radio commands/Rate commands/Filter'
 * '<S32>'  : 'Sim_Multi/Radio commands/Rate commands/Pitch rate signal'
 * '<S33>'  : 'Sim_Multi/Radio commands/Rate commands/Roll rate signal'
 * '<S34>'  : 'Sim_Multi/Radio commands/Rate commands/Yaw rate signal'
 * '<S35>'  : 'Sim_Multi/State Estimation, Sensors, Noise/Sensor models'
 * '<S36>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation'
 * '<S37>'  : 'Sim_Multi/State Estimation, Sensors, Noise/Sensor models/Acc noise'
 * '<S38>'  : 'Sim_Multi/State Estimation, Sensors, Noise/Sensor models/Gyro noise'
 * '<S39>'  : 'Sim_Multi/State Estimation, Sensors, Noise/Sensor models/Acc noise/Band-Limited White Noise'
 * '<S40>'  : 'Sim_Multi/State Estimation, Sensors, Noise/Sensor models/Gyro noise/Band-Limited White Noise'
 * '<S41>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Normalize'
 * '<S42>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Quat noise'
 * '<S43>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vb noise'
 * '<S44>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vi noise'
 * '<S45>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Xi noise'
 * '<S46>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Normalize/norm'
 * '<S47>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Normalize/norm/dot_product'
 * '<S48>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Quat noise/Band-Limited White Noise'
 * '<S49>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vb noise/Band-Limited White Noise'
 * '<S50>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vb noise/Band-Limited White Noise1'
 * '<S51>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vb noise/Band-Limited White Noise2'
 * '<S52>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vi noise/Band-Limited White Noise'
 * '<S53>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vi noise/Band-Limited White Noise1'
 * '<S54>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Vi noise/Band-Limited White Noise2'
 * '<S55>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Xi noise/Band-Limited White Noise'
 * '<S56>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Xi noise/Band-Limited White Noise1'
 * '<S57>'  : 'Sim_Multi/State Estimation, Sensors, Noise/State estimation/Xi noise/Band-Limited White Noise2'
 * '<S58>'  : 'Sim_Multi/multirotor/6DOF model'
 * '<S59>'  : 'Sim_Multi/multirotor/Disturbances & Dynamic Events'
 * '<S60>'  : 'Sim_Multi/multirotor/Force//Moment computation'
 * '<S61>'  : 'Sim_Multi/multirotor/6DOF model/Body to Inertial'
 * '<S62>'  : 'Sim_Multi/multirotor/6DOF model/Body to Inertial1'
 * '<S63>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot'
 * '<S64>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq.'
 * '<S65>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Euler'
 * '<S66>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix'
 * '<S67>'  : 'Sim_Multi/multirotor/6DOF model/omega x V_b'
 * '<S68>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)'
 * '<S69>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Subsystem'
 * '<S70>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)/Subsystem'
 * '<S71>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)/Subsystem1'
 * '<S72>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Cross Product'
 * '<S73>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Dot product'
 * '<S74>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./MATLAB Function'
 * '<S75>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Normalize'
 * '<S76>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Cross Product/Subsystem'
 * '<S77>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Cross Product/Subsystem1'
 * '<S78>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Normalize/norm'
 * '<S79>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Normalize/norm/dot_product'
 * '<S80>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem'
 * '<S81>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem1'
 * '<S82>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem2'
 * '<S83>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem3'
 * '<S84>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem4'
 * '<S85>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem5'
 * '<S86>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem6'
 * '<S87>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem7'
 * '<S88>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem8'
 * '<S89>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem9'
 * '<S90>'  : 'Sim_Multi/multirotor/6DOF model/omega x V_b/Subsystem'
 * '<S91>'  : 'Sim_Multi/multirotor/6DOF model/omega x V_b/Subsystem1'
 * '<S92>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes'
 * '<S93>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute force of gravity in Body axes'
 * '<S94>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers'
 * '<S95>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force'
 * '<S96>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Total airspeed (Body axes)'
 * '<S97>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed'
 * '<S98>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Zero airspeed'
 * '<S99>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/norm'
 * '<S100>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize'
 * '<S101>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation'
 * '<S102>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/u^2'
 * '<S103>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize/norm'
 * '<S104>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize/norm/dot_product'
 * '<S105>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Scaling factor computation'
 * '<S106>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation'
 * '<S107>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation/norm'
 * '<S108>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation/norm/dot_product'
 * '<S109>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/u^2/dot_product'
 * '<S110>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/norm/dot_product'
 * '<S111>' : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Total airspeed (Body axes)/Inertial to Body'
 * '<S112>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem'
 * '<S113>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/Total airspeed (Body axes)'
 * '<S114>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics'
 * '<S115>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product'
 * '<S116>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Leverarm vector from real CoG to each propeller (Body axes)'
 * '<S117>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)'
 * '<S118>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits'
 * '<S119>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1'
 * '<S120>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Compare To Zero'
 * '<S121>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Compare To Zero1'
 * '<S122>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Saturation Dynamic'
 * '<S123>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1/Compare To Zero'
 * '<S124>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1/Compare To Zero1'
 * '<S125>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1/Saturation Dynamic'
 * '<S126>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product/Subsystem'
 * '<S127>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product/Subsystem1'
 * '<S128>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments'
 * '<S129>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes'
 * '<S130>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product'
 * '<S131>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations'
 * '<S132>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio'
 * '<S133>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque'
 * '<S134>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Hover moment magnitude'
 * '<S135>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Hover thrust magnitude'
 * '<S136>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Thrust direction in Body axes (without blade flapping)'
 * '<S137>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane'
 * '<S138>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Zero airspeed in rotor plane'
 * '<S139>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/norm'
 * '<S140>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane/norm'
 * '<S141>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane/norm/dot_product'
 * '<S142>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/norm/dot_product'
 * '<S143>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude'
 * '<S144>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Compute the climb speed'
 * '<S145>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller'
 * '<S146>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed'
 * '<S147>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Zero airspeed'
 * '<S148>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/norm2'
 * '<S149>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed/norm1'
 * '<S150>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed/norm1/dot_product'
 * '<S151>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/norm2/dot_product'
 * '<S152>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Normal working state vc//vh >= 0'
 * '<S153>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Vortex ring state -2 <= vc//vh < 0 '
 * '<S154>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Windmill braking state vc//vh < -2'
 * '<S155>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product'
 * '<S156>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Propeller moment of inertia'
 * '<S157>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product/Subsystem'
 * '<S158>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product/Subsystem1'
 * '<S159>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor'
 * '<S160>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem'
 * '<S161>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem1'
 * '<S162>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem2'
 * '<S163>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem3'
 * '<S164>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem4'
 * '<S165>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem5'
 * '<S166>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem6'
 * '<S167>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem7'
 * '<S168>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem8'
 * '<S169>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem9'
 * '<S170>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product/Subsystem'
 * '<S171>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product/Subsystem1'
 * '<S172>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/Total airspeed (Body axes)/Inertial to Body'
 */
#endif                                 /* RTW_HEADER_Sim_Multi_h_ */
