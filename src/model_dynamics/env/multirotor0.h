/*
 * multirotor0.h
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

#ifndef RTW_HEADER_multirotor0_h_
#define RTW_HEADER_multirotor0_h_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "multirotor0_types.h"

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

/* Block signals for system '<S39>/For Each Subsystem' */
struct B_CoreSubsys_multirotor0_T {
  real_T Product;                      /* '<S59>/Product' */
  real_T Switch;                       /* '<S63>/Switch' */
  real_T Switch_a;                     /* '<S64>/Switch' */
  real_T VectorfromrealCoGtopropellerBod[3];/* '<S61>/Subtract' */
  real_T VectorConcatenate[9];         /* '<S114>/Vector Concatenate' */
  real_T Climbspeedv_c;                /* '<S89>/Gain' */
  real_T Merge;                        /* '<S90>/Merge' */
  real_T Product9[3];                  /* '<S81>/Product9' */
  real_T Gain1;                        /* '<S101>/Gain1' */
  real_T Angleofattackrad;             /* '<S88>/Merge' */
  real_T NewtiltedthrustdirectionBodyaxe[3];/* '<S76>/Merge' */
  real_T Momentinthemotorhubduetobending[3];/* '<S76>/Merge1' */
};

/* Block states (default storage) for system '<S39>/For Each Subsystem' */
struct DW_CoreSubsys_multirotor0_T {
  int8_T If_ActiveSubsystem;           /* '<S90>/If' */
  int8_T If_ActiveSubsystem_l;         /* '<S88>/If' */
  int8_T If_ActiveSubsystem_e;         /* '<S76>/If' */
};

/* Continuous states for system '<S39>/For Each Subsystem' */
struct X_CoreSubsys_multirotor0_T {
  real_T Integrator_CSTATE;            /* '<S63>/Integrator' */
  real_T Integrator_CSTATE_o;          /* '<S64>/Integrator' */
};

/* State derivatives for system '<S39>/For Each Subsystem' */
struct XDot_CoreSubsys_multirotor0_T {
  real_T Integrator_CSTATE;            /* '<S63>/Integrator' */
  real_T Integrator_CSTATE_o;          /* '<S64>/Integrator' */
};

/* State Disabled for system '<S39>/For Each Subsystem' */
struct XDis_CoreSubsys_multirotor0_T {
  boolean_T Integrator_CSTATE;         /* '<S63>/Integrator' */
  boolean_T Integrator_CSTATE_o;       /* '<S64>/Integrator' */
};

/* Block signals (default storage) */
struct B_multirotor0_T {
  real_T Product[3];                   /* '<S5>/Product' */
  real_T RateTransition1[4];           /* '<S1>/Rate Transition1' */
  real_T ForceofgravityInertialaxes[3];/* '<S38>/Product1' */
  real_T Sum1[3];                      /* '<S2>/Sum1' */
  real_T Product_l[3];                 /* '<S7>/Product' */
  real_T TmpSignalConversionAtQIntegrato[4];
  real_T MotorMatrix_real[68];         /* '<S3>/MATLAB Function' */
  real_T COM_system[3];                /* '<S3>/MATLAB Function' */
  real_T total_mass;                   /* '<S3>/MATLAB Function' */
  real_T inertial_matrix[9];           /* '<S3>/MATLAB Function' */
  real_T Surface_params[3];            /* '<S3>/MATLAB Function' */
  real_T quat_output[4];               /* '<S8>/MATLAB Function' */
  real_T Forceagainstdirectionofmotiondu[3];/* '<S40>/Merge' */
  B_CoreSubsys_multirotor0_T CoreSubsys[4];/* '<S39>/For Each Subsystem' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_multirotor0_T {
  real_T RateTransition1_Buffer0[4];   /* '<S1>/Rate Transition1' */
  int_T QIntegrator_IWORK;             /* '<S8>/Q-Integrator' */
  int8_T If_ActiveSubsystem;           /* '<S40>/If' */
  DW_CoreSubsys_multirotor0_T CoreSubsys[4];/* '<S39>/For Each Subsystem' */
};

/* Continuous states (default storage) */
struct X_multirotor0_T {
  real_T QIntegrator_CSTATE[4];        /* '<S8>/Q-Integrator' */
  real_T V_b_CSTATE[3];                /* '<S2>/V_b' */
  real_T omega_CSTATE[3];              /* '<S2>/omega' */
  real_T X_i_CSTATE[3];                /* '<S2>/X_i' */
  X_CoreSubsys_multirotor0_T CoreSubsys[4];/* '<S57>/CoreSubsys' */
};

/* State derivatives (default storage) */
struct XDot_multirotor0_T {
  real_T QIntegrator_CSTATE[4];        /* '<S8>/Q-Integrator' */
  real_T V_b_CSTATE[3];                /* '<S2>/V_b' */
  real_T omega_CSTATE[3];              /* '<S2>/omega' */
  real_T X_i_CSTATE[3];                /* '<S2>/X_i' */
  XDot_CoreSubsys_multirotor0_T CoreSubsys[4];/* '<S57>/CoreSubsys' */
};

/* State disabled  */
struct XDis_multirotor0_T {
  boolean_T QIntegrator_CSTATE[4];     /* '<S8>/Q-Integrator' */
  boolean_T V_b_CSTATE[3];             /* '<S2>/V_b' */
  boolean_T omega_CSTATE[3];           /* '<S2>/omega' */
  boolean_T X_i_CSTATE[3];             /* '<S2>/X_i' */
  XDis_CoreSubsys_multirotor0_T CoreSubsys[4];/* '<S57>/CoreSubsys' */
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
struct ExtU_multirotor0_T {
  real_T RPMcommands[4];               /* '<Root>/RPM commands' */
  real_T Wind_i[3];                    /* '<Root>/wind' */
  real_T Force_disturb[3];             /* '<Root>/force_disturbance' */
  real_T Moment_disturb[3];            /* '<Root>/moment_disturbance' */
  real_T arm_length[4];                /* '<Root>/arm_length' */
  real_T prop_height[4];               /* '<Root>/prop_height' */
  real_T prop_diameter[4];             /* '<Root>/prop_diameter' */
  real_T rotation_direction[4];        /* '<Root>/rotation_direction' */
  real_T max_rpm[4];                   /* '<Root>/max_rpm' */
  real_T min_rpm[4];                   /* '<Root>/min_rpm' */
  real_T arm_radius[4];                /* '<Root>/arm_radius' */
  real_T Motor_arm_angle[4];           /* '<Root>/Motor_arm_angle' */
  real_T mass_center;                  /* '<Root>/mass_center' */
  real_T COM_mass_center[3];           /* '<Root>/COM_mass_center' */
  real_T Surface_params[3];            /* '<Root>/Surface_params' */
};

struct Init_multirotor0_T{
  real_T pos_init[3];
  real_T vel_init[3];
  real_T omega_init[3];

};
/* External outputs (root outports fed by signals with default storage) */
struct ExtY_multirotor0_T {
  real_T X_i[3];                       /* '<Root>/X_i' */
  real_T V_i[3];                       /* '<Root>/V_i' */
  real_T V_b[3];                       /* '<Root>/V_b' */
  real_T a_b[3];                       /* '<Root>/a_b' */
  real_T a_i[3];                       /* '<Root>/a_i' */
  real_T DCM_ib[9];                    /* '<Root>/DCM_ib' */
  real_T Quatq[4];                     /* '<Root>/Quat q' */
  real_T Euler[3];                     /* '<Root>/Euler' */
  real_T omega[3];                     /* '<Root>/omega' */
  real_T omega_dot[3];                 /* '<Root>/omega_dot' */
  real_T motor_RPM[4];                 /* '<Root>/motor_RPM' */
};

/* Real-time Model Data Structure */
struct tag_RTM_multirotor0_T {
  const char_T *errorStatus;
  RTWSolverInfo solverInfo;
  X_multirotor0_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  XDis_multirotor0_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[21];
  real_T odeF[3][21];
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
      uint8_T TID[3];
      uint8_T cLimit[3];
    } TaskCounters;

    struct {
      uint8_T TID1_2;
    } RateInteraction;

    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[3];
  } Timing;
};

/* Class declaration for model multirotor0 */
class multirotor0
{
  /* public data and function members */
 public:
  /* Copy Constructor */
  multirotor0(multirotor0 const&) = delete;

  /* Assignment Operator */
  multirotor0& operator= (multirotor0 const&) & = delete;

  /* Move Constructor */
  multirotor0(multirotor0 &&) = delete;

  /* Move Assignment Operator */
  multirotor0& operator= (multirotor0 &&) = delete;

  /* Real-Time Model get method */
  RT_MODEL_multirotor0_T * getRTM();

  /* External inputs */
  ExtU_multirotor0_T multirotor0_U;

  Init_multirotor0_T Model_Init;

  /* External outputs */
  ExtY_multirotor0_T multirotor0_Y;

  /* model start function */
  void start();

  /* Initial conditions function */
  void initialize();

  /* model step function */
  void step0();

  /* model step function */
  void step2();

  /* model terminate function */
  static void terminate();

  /* Constructor */
  multirotor0();

  /* Destructor */
  ~multirotor0();

  /* private data and function members */
 private:
  /* Block signals */
  B_multirotor0_T multirotor0_B;

  /* Block states */
  DW_multirotor0_T multirotor0_DW;

  /* Block continuous states */
  X_multirotor0_T multirotor0_X;

  /* Continuous states update member function*/
  void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si );

  /* Derivatives member function */
  void multirotor0_derivatives();

  /* Real-Time Model */
  RT_MODEL_multirotor0_T multirotor0_M;
};

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<S67>/Data Type Duplicate' : Unused code path elimination
 * Block '<S67>/Data Type Propagation' : Unused code path elimination
 * Block '<S70>/Data Type Duplicate' : Unused code path elimination
 * Block '<S70>/Data Type Propagation' : Unused code path elimination
 * Block '<S70>/LowerRelop1' : Unused code path elimination
 * Block '<S70>/Switch' : Unused code path elimination
 * Block '<S70>/Switch2' : Unused code path elimination
 * Block '<S70>/UpperRelop' : Unused code path elimination
 * Block '<S39>/To Workspace' : Unused code path elimination
 * Block '<S1>/To Workspace' : Unused code path elimination
 * Block '<S5>/Reshape' : Reshape block reduction
 * Block '<S5>/Reshape1' : Reshape block reduction
 * Block '<S6>/Reshape' : Reshape block reduction
 * Block '<S6>/Reshape1' : Reshape block reduction
 * Block '<S7>/Reshape' : Reshape block reduction
 * Block '<S7>/Reshape ' : Reshape block reduction
 * Block '<S13>/Reshape' : Reshape block reduction
 * Block '<S13>/Reshape1' : Reshape block reduction
 * Block '<S17>/Reshape' : Reshape block reduction
 * Block '<S17>/Reshape1' : Reshape block reduction
 * Block '<S23>/Reshape' : Reshape block reduction
 * Block '<S23>/Reshape1' : Reshape block reduction
 * Block '<S9>/Reshape' : Reshape block reduction
 * Block '<S33>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S49>/Reshape' : Reshape block reduction
 * Block '<S49>/Reshape1' : Reshape block reduction
 * Block '<S53>/Reshape' : Reshape block reduction
 * Block '<S53>/Reshape1' : Reshape block reduction
 * Block '<S54>/Reshape' : Reshape block reduction
 * Block '<S54>/Reshape1' : Reshape block reduction
 * Block '<S55>/Reshape' : Reshape block reduction
 * Block '<S55>/Reshape1' : Reshape block reduction
 * Block '<S56>/Reshape' : Reshape block reduction
 * Block '<S56>/Reshape1' : Reshape block reduction
 * Block '<S41>/Reshape' : Reshape block reduction
 * Block '<S38>/Reshape' : Reshape block reduction
 * Block '<S60>/Reshape' : Reshape block reduction
 * Block '<S61>/Reshape1' : Reshape block reduction
 * Block '<S57>/Rate Transition' : Eliminated since input and output rates are identical
 * Block '<S82>/Reshape3' : Reshape block reduction
 * Block '<S86>/Reshape' : Reshape block reduction
 * Block '<S86>/Reshape1' : Reshape block reduction
 * Block '<S87>/Reshape' : Reshape block reduction
 * Block '<S87>/Reshape1' : Reshape block reduction
 * Block '<S95>/Reshape' : Reshape block reduction
 * Block '<S95>/Reshape1' : Reshape block reduction
 * Block '<S96>/Reshape' : Reshape block reduction
 * Block '<S96>/Reshape1' : Reshape block reduction
 * Block '<S78>/Reshape1' : Reshape block reduction
 * Block '<S78>/Reshape2' : Reshape block reduction
 * Block '<S81>/Reshape5' : Reshape block reduction
 * Block '<S114>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S62>/Reshape' : Reshape block reduction
 * Block '<S39>/Reshape' : Reshape block reduction
 * Block '<S117>/Reshape' : Reshape block reduction
 * Block '<S117>/Reshape1' : Reshape block reduction
 * Block '<S58>/Reshape' : Reshape block reduction
 * Block '<S77>/Constant' : Unused code path elimination
 * Block '<S77>/Constant1' : Unused code path elimination
 */

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
 * hilite_system('Sim_Multi/multirotor')    - opens subsystem Sim_Multi/multirotor
 * hilite_system('Sim_Multi/multirotor/Kp') - opens and selects block Kp
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'Sim_Multi'
 * '<S1>'   : 'Sim_Multi/multirotor'
 * '<S2>'   : 'Sim_Multi/multirotor/6DOF model'
 * '<S3>'   : 'Sim_Multi/multirotor/Disturbances & Dynamic Events'
 * '<S4>'   : 'Sim_Multi/multirotor/Force//Moment computation'
 * '<S5>'   : 'Sim_Multi/multirotor/6DOF model/Body to Inertial'
 * '<S6>'   : 'Sim_Multi/multirotor/6DOF model/Body to Inertial1'
 * '<S7>'   : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot'
 * '<S8>'   : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq.'
 * '<S9>'   : 'Sim_Multi/multirotor/6DOF model/Quat to Euler'
 * '<S10>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix'
 * '<S11>'  : 'Sim_Multi/multirotor/6DOF model/omega x V_b'
 * '<S12>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)'
 * '<S13>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Subsystem'
 * '<S14>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)/Subsystem'
 * '<S15>'  : 'Sim_Multi/multirotor/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)/Subsystem1'
 * '<S16>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Cross Product'
 * '<S17>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Dot product'
 * '<S18>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./MATLAB Function'
 * '<S19>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Normalize'
 * '<S20>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Cross Product/Subsystem'
 * '<S21>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Cross Product/Subsystem1'
 * '<S22>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Normalize/norm'
 * '<S23>'  : 'Sim_Multi/multirotor/6DOF model/Quat Strapdown Eq./Normalize/norm/dot_product'
 * '<S24>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem'
 * '<S25>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem1'
 * '<S26>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem2'
 * '<S27>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem3'
 * '<S28>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem4'
 * '<S29>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem5'
 * '<S30>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem6'
 * '<S31>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem7'
 * '<S32>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem8'
 * '<S33>'  : 'Sim_Multi/multirotor/6DOF model/Quat to Rotation Matrix/Subsystem9'
 * '<S34>'  : 'Sim_Multi/multirotor/6DOF model/omega x V_b/Subsystem'
 * '<S35>'  : 'Sim_Multi/multirotor/6DOF model/omega x V_b/Subsystem1'
 * '<S36>'  : 'Sim_Multi/multirotor/Disturbances & Dynamic Events/MATLAB Function'
 * '<S37>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes'
 * '<S38>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute force of gravity in Body axes'
 * '<S39>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers'
 * '<S40>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force'
 * '<S41>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Total airspeed (Body axes)'
 * '<S42>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed'
 * '<S43>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Zero airspeed'
 * '<S44>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/norm'
 * '<S45>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize'
 * '<S46>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation'
 * '<S47>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/u^2'
 * '<S48>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize/norm'
 * '<S49>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize/norm/dot_product'
 * '<S50>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Scaling factor computation'
 * '<S51>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation'
 * '<S52>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation/norm'
 * '<S53>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation/norm/dot_product'
 * '<S54>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/u^2/dot_product'
 * '<S55>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/norm/dot_product'
 * '<S56>'  : 'Sim_Multi/multirotor/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Total airspeed (Body axes)/Inertial to Body'
 * '<S57>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem'
 * '<S58>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/Total airspeed (Body axes)'
 * '<S59>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics'
 * '<S60>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product'
 * '<S61>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Leverarm vector from real CoG to each propeller (Body axes)'
 * '<S62>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)'
 * '<S63>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits'
 * '<S64>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1'
 * '<S65>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Compare To Zero'
 * '<S66>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Compare To Zero1'
 * '<S67>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Saturation Dynamic'
 * '<S68>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1/Compare To Zero'
 * '<S69>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1/Compare To Zero1'
 * '<S70>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits1/Saturation Dynamic'
 * '<S71>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product/Subsystem'
 * '<S72>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product/Subsystem1'
 * '<S73>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments'
 * '<S74>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes'
 * '<S75>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product'
 * '<S76>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations'
 * '<S77>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio'
 * '<S78>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque'
 * '<S79>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Hover moment magnitude'
 * '<S80>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Hover thrust magnitude'
 * '<S81>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Thrust direction in Body axes (without blade flapping)'
 * '<S82>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane'
 * '<S83>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Zero airspeed in rotor plane'
 * '<S84>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/norm'
 * '<S85>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane/norm'
 * '<S86>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane/norm/dot_product'
 * '<S87>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/norm/dot_product'
 * '<S88>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude'
 * '<S89>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Compute the climb speed'
 * '<S90>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller'
 * '<S91>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed'
 * '<S92>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Zero airspeed'
 * '<S93>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/norm2'
 * '<S94>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed/norm1'
 * '<S95>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed/norm1/dot_product'
 * '<S96>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/norm2/dot_product'
 * '<S97>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Normal working state vc//vh >= 0'
 * '<S98>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Vortex ring state -2 <= vc//vh < 0 '
 * '<S99>'  : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Windmill braking state vc//vh < -2'
 * '<S100>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product'
 * '<S101>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Propeller moment of inertia'
 * '<S102>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product/Subsystem'
 * '<S103>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product/Subsystem1'
 * '<S104>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor'
 * '<S105>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem'
 * '<S106>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem1'
 * '<S107>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem2'
 * '<S108>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem3'
 * '<S109>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem4'
 * '<S110>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem5'
 * '<S111>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem6'
 * '<S112>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem7'
 * '<S113>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem8'
 * '<S114>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem9'
 * '<S115>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product/Subsystem'
 * '<S116>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product/Subsystem1'
 * '<S117>' : 'Sim_Multi/multirotor/Force//Moment computation/Forces and moments generated by spinning propellers/Total airspeed (Body axes)/Inertial to Body'
 */
#endif                                 /* RTW_HEADER_multirotor0_h_ */
