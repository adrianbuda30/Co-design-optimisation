/*
 * Planar_robot_3DoF_sim.h
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "Planar_robot_3DoF_sim".
 *
 * Model version              : 12.40
 * Simulink Coder version : 9.9 (R2023a) 19-Nov-2022
 * C++ source code generated on : Thu Nov 23 02:24:17 2023
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objective: Debugging
 * Validation result: Not run
 */

#ifndef RTW_HEADER_Planar_robot_3DoF_sim_h_
#define RTW_HEADER_Planar_robot_3DoF_sim_h_
#include <stdlib.h>
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_logging.h"
#include "Planar_robot_3DoF_sim_types.h"
#include <cfloat>

extern "C"
{

#include "rt_nonfinite.h"

}

/* Macros for accessing real-time model data structure */
#ifndef rtmGetFinalTime
#define rtmGetFinalTime(rtm)           ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetRTWLogInfo
#define rtmGetRTWLogInfo(rtm)          ((rtm)->rtwLogInfo)
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

#ifndef rtmGetTFinal
#define rtmGetTFinal(rtm)              ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

/* Block signals (default storage) */
struct B_Planar_robot_3DoF_sim_T {
  real_T j1_torque;                    /* '<Root>/Sine Wave Function3' */
  real_T j2_torque;                    /* '<Root>/Sine Wave Function1' */
  real_T j3_torque;                    /* '<Root>/Sine Wave Function2' */
  real_T joint_torque[3];
  real_T init_pos[3];                  /* '<Root>/init_pos' */
  real_T TmpSignalConversionAtToWorkspac[18];
  real_T q[3];                         /* '<S1>/Discrete-Time Integrator1' */
  real_T dq[3];                        /* '<S1>/Discrete-Time Integrator' */
  real_T TSamp[3];                     /* '<S2>/TSamp' */
  real_T Uk1[3];                       /* '<S2>/UD' */
  real_T Diff[3];                      /* '<S2>/Diff' */
  real_T TSamp_g[3];                   /* '<S3>/TSamp' */
  real_T Uk1_o[3];                     /* '<S3>/UD' */
  real_T Diff_d[3];                    /* '<S3>/Diff' */
  real_T Product5[9];                  /* '<S1>/Product5' */
  real_T Product3[3];                  /* '<S1>/Product3' */
  real_T Sum4[3];                      /* '<S1>/Sum4' */
  real_T Product1[3];                  /* '<S1>/Product1' */
  real_T M[9];                         /* '<S1>/calcSysMatrices' */
  real_T CC[9];                        /* '<S1>/calcSysMatrices' */
  real_T g[3];                         /* '<S1>/calcSysMatrices' */
  real_T pos_tcp[3];                   /* '<S1>/calcSysMatrices' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_Planar_robot_3DoF_sim_T {
  real_T DiscreteTimeIntegrator1_DSTATE[3];/* '<S1>/Discrete-Time Integrator1' */
  real_T DiscreteTimeIntegrator_DSTATE[3];/* '<S1>/Discrete-Time Integrator' */
  real_T UD_DSTATE[3];                 /* '<S2>/UD' */
  real_T UD_DSTATE_o[3];               /* '<S3>/UD' */
  real_T Product5_DWORK1[9];           /* '<S1>/Product5' */
  real_T Product5_DWORK3[9];           /* '<S1>/Product5' */
  real_T Product5_DWORK4[9];           /* '<S1>/Product5' */
  struct {
    void *LoggedData;
  } ToWorkspace_PWORK;                 /* '<Root>/To Workspace' */

  struct {
    void *LoggedData;
  } ToWorkspace1_PWORK;                /* '<Root>/To Workspace1' */

  int32_T Product5_DWORK2[3];          /* '<S1>/Product5' */
};

/* Real-time Model Data Structure */
struct tag_RTM_Planar_robot_3DoF_sim_T {
  const char_T *errorStatus;
  RTWLogInfo *rtwLogInfo;
  RTWSolverInfo solverInfo;

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
    time_T tFinal;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[2];
  } Timing;
};

/* Class declaration for model Planar_robot_3DoF_sim */
class Planar_robot_3DoF_sim final
{
  /* public data and function members */
 public:
  /* Copy Constructor */
  Planar_robot_3DoF_sim(Planar_robot_3DoF_sim const&) = delete;

  /* Assignment Operator */
  Planar_robot_3DoF_sim& operator= (Planar_robot_3DoF_sim const&) & = delete;

  /* Move Constructor */
  Planar_robot_3DoF_sim(Planar_robot_3DoF_sim &&) = delete;

  /* Move Assignment Operator */
  Planar_robot_3DoF_sim& operator= (Planar_robot_3DoF_sim &&) = delete;

  /* Real-Time Model get method */
  RT_MODEL_Planar_robot_3DoF_si_T * getRTM();

  /* model start function */
  void start();

  /* Initial conditions function */
  void initialize();

  /* model step function */
  void step();

  /* model terminate function */
  static void terminate();

  /* Constructor */
  Planar_robot_3DoF_sim();

  /* Destructor */
  ~Planar_robot_3DoF_sim();

  /* private data and function members */
 private:
  /* Block signals */
  B_Planar_robot_3DoF_sim_T Planar_robot_3DoF_sim_B;

  /* Block states */
  DW_Planar_robot_3DoF_sim_T Planar_robot_3DoF_sim_DW;

  /* private member function(s) for subsystem '<Root>'*/
  void Planar_robot_3DoF_sim_eye(real_T b_I[9]);
  void Planar_robot_3DoF_sim_diag(const real_T v[3], real_T d[9]);
  void Planar_robot_3DoF_sim_repmat(real_T b[18]);
  boolean_T Planar_robot_3DoF_sim_all(const boolean_T x[3]);
  void Planar_robot_3DoF_sim_mtimes(const real_T A_data[], const int32_T A_size
    [2], const real_T B[16], real_T C_data[], int32_T C_size[2]);
  void Planar_robot_3DoF_sim_mtimes_p(const real_T A[36], const real_T B_data[],
    const int32_T B_size[2], real_T C_data[], int32_T C_size[2]);
  void Planar_rob_binary_expand_op_ccy(real_T in1[18], int32_T in2, const real_T
    in3[36], const real_T in4[18], const real_T in5[9], const real_T in6[6],
    const real_T in7[18]);
  void Planar_robot_3DoF_sim_mtimes_pn(const real_T A_data[], const int32_T
    A_size[2], const real_T B_data[], const int32_T B_size[2], real_T C_data[],
    int32_T C_size[2]);
  void Planar_robot_3DoF_si_mtimes_pnc(const real_T A_data[], const int32_T
    A_size[2], const real_T B_data[], const int32_T B_size[2], real_T C_data[],
    int32_T C_size[2]);
  void Planar_robo_binary_expand_op_cc(real_T in1[9], int32_T in2, const real_T
    in3_data[], const int32_T in3_size[2], const real_T in4[18]);
  void Planar_robot_3DoF_s_mtimes_pnc5(const real_T A_data[], const int32_T
    A_size[2], const real_T B[36], real_T C_data[], int32_T C_size[2]);
  void Planar_robot_binary_expand_op_c(real_T in1[9], int32_T in2, const real_T
    in3[18], real_T in4, const real_T in5[9], const real_T in6[9], const real_T
    in7[9], const real_T in8_data[], const int32_T in8_size[2], const real_T
    in9[18]);
  void Planar_robot_3DoF__mtimes_pnc5a(const real_T A_data[], const int32_T
    A_size[2], const real_T B[9], real_T C_data[], int32_T C_size[2]);
  void Planar_robot_3DoF_mtimes_pnc5ag(const real_T A_data[], const int32_T
    A_size[2], real_T C_data[], int32_T *C_size);
  void Planar_robot_3_binary_expand_op(real_T in1[3], int32_T in2, const real_T
    in3_data[], const int32_T in3_size[2], const real_T in4[16]);

  /* Real-Time Model */
  RT_MODEL_Planar_robot_3DoF_si_T Planar_robot_3DoF_sim_M;
};

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<Root>/joint_torque' : Unused code path elimination
 * Block '<S2>/Data Type Duplicate' : Unused code path elimination
 * Block '<S3>/Data Type Duplicate' : Unused code path elimination
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
 * '<Root>' : 'Planar_robot_3DoF_sim'
 * '<S1>'   : 'Planar_robot_3DoF_sim/planar_robot3dof_FD'
 * '<S2>'   : 'Planar_robot_3DoF_sim/planar_robot3dof_FD/Discrete Derivative'
 * '<S3>'   : 'Planar_robot_3DoF_sim/planar_robot3dof_FD/Discrete Derivative1'
 * '<S4>'   : 'Planar_robot_3DoF_sim/planar_robot3dof_FD/calcSysMatrices'
 */
#endif                                 /* RTW_HEADER_Planar_robot_3DoF_sim_h_ */
