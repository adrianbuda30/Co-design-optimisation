/*
 * True0.h
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

#ifndef RTW_HEADER_True0_h_
#define RTW_HEADER_True0_h_
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_logging.h"
#include "True0_types.h"

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

#include <cfloat>
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

#ifndef rtmGetFinalTime
#define rtmGetFinalTime(rtm)           ((rtm)->Timing.tFinal)
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

#ifndef rtmGetRTWLogInfo
#define rtmGetRTWLogInfo(rtm)          ((rtm)->rtwLogInfo)
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

#ifndef rtmGetTFinal
#define rtmGetTFinal(rtm)              ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

#ifndef rtmTaskCounter
#define rtmTaskCounter(rtm, idx)       ((rtm)->Timing.TaskCounters.TID[(idx)])
#endif

/* Block signals for system '<S37>/For Each Subsystem' */
struct B_CoreSubsys_True0_T {
  real_T Integrator;                   /* '<S61>/Integrator' */
  real_T Switch2;                      /* '<S64>/Switch2' */
  real_T Product;                      /* '<S57>/Product' */
  real_T Sum1;                         /* '<S57>/Sum1' */
  real_T Divide;                       /* '<S57>/Divide' */
  real_T Switch;                       /* '<S61>/Switch' */
  real_T Conversiondegtorad;           /* '<S59>/Conversion deg to rad' */
  real_T Motorarmxcomponent;           /* '<S59>/Trigonometric Function1' */
  real_T Motorarmycomponent;           /* '<S59>/Trigonometric Function' */
  real_T Abs;                          /* '<S59>/Abs' */
  real_T Motorlocationxyvector[2];     /* '<S59>/Product4' */
  real_T Vectorfromgeometriccentertoprop[3];/* '<S59>/Reshape' */
  real_T VectorfromrealCoGtopropellerBod[3];/* '<S59>/Subtract' */
  real_T u2v3;                         /* '<S65>/Product' */
  real_T u3v1;                         /* '<S65>/Product1' */
  real_T u1v2;                         /* '<S65>/Product2' */
  real_T u3v2;                         /* '<S66>/Product' */
  real_T u1v3;                         /* '<S66>/Product1' */
  real_T u2v1;                         /* '<S66>/Product2' */
  real_T Sum[3];                       /* '<S58>/Sum' */
  real_T Product4;                     /* '<S74>/Product4' */
  real_T Product5;                     /* '<S74>/Product5' */
  real_T Product6;                     /* '<S74>/Product6' */
  real_T Hoverthrustmagnitude;         /* '<S74>/Sum1' */
  real_T Conversiondegtorad_n[3];      /* '<S68>/Conversion deg to rad' */
  real_T TrigonometricFunction1;       /* '<S99>/Trigonometric Function1' */
  real_T TrigonometricFunction3;       /* '<S99>/Trigonometric Function3' */
  real_T TrigonometricFunction12;      /* '<S102>/Trigonometric Function12' */
  real_T TrigonometricFunction1_h;     /* '<S102>/Trigonometric Function1' */
  real_T TrigonometricFunction3_m;     /* '<S102>/Trigonometric Function3' */
  real_T Product_d;                    /* '<S102>/Product' */
  real_T TrigonometricFunction5;       /* '<S102>/Trigonometric Function5' */
  real_T TrigonometricFunction;        /* '<S102>/Trigonometric Function' */
  real_T Product1;                     /* '<S102>/Product1' */
  real_T TrigonometricFunction12_c;    /* '<S105>/Trigonometric Function12' */
  real_T TrigonometricFunction2;       /* '<S105>/Trigonometric Function2' */
  real_T TrigonometricFunction4;       /* '<S105>/Trigonometric Function4' */
  real_T Product1_a;                   /* '<S105>/Product1' */
  real_T TrigonometricFunction5_f;     /* '<S105>/Trigonometric Function5' */
  real_T TrigonometricFunction_o;      /* '<S105>/Trigonometric Function' */
  real_T Product2;                     /* '<S105>/Product2' */
  real_T TrigonometricFunction1_c;     /* '<S100>/Trigonometric Function1' */
  real_T TrigonometricFunction3_e;     /* '<S100>/Trigonometric Function3' */
  real_T TrigonometricFunction12_n;    /* '<S103>/Trigonometric Function12' */
  real_T TrigonometricFunction2_a;     /* '<S103>/Trigonometric Function2' */
  real_T TrigonometricFunction4_h;     /* '<S103>/Trigonometric Function4' */
  real_T Product1_f;                   /* '<S103>/Product1' */
  real_T TrigonometricFunction5_a;     /* '<S103>/Trigonometric Function5' */
  real_T TrigonometricFunction_f;      /* '<S103>/Trigonometric Function' */
  real_T Product2_c;                   /* '<S103>/Product2' */
  real_T TrigonometricFunction12_o;    /* '<S106>/Trigonometric Function12' */
  real_T TrigonometricFunction1_l;     /* '<S106>/Trigonometric Function1' */
  real_T TrigonometricFunction3_n;     /* '<S106>/Trigonometric Function3' */
  real_T Product_e;                    /* '<S106>/Product' */
  real_T TrigonometricFunction5_k;     /* '<S106>/Trigonometric Function5' */
  real_T TrigonometricFunction_f0;     /* '<S106>/Trigonometric Function' */
  real_T Product1_j;                   /* '<S106>/Product1' */
  real_T TrigonometricFunction1_m;     /* '<S101>/Trigonometric Function1' */
  real_T TrigonometricFunction3_mh;    /* '<S104>/Trigonometric Function3' */
  real_T TrigonometricFunction1_a;     /* '<S104>/Trigonometric Function1' */
  real_T TrigonometricFunction3_j;     /* '<S107>/Trigonometric Function3' */
  real_T TrigonometricFunction1_m3;    /* '<S107>/Trigonometric Function1' */
  real_T VectorConcatenate[9];         /* '<S108>/Vector Concatenate' */
  real_T TotallinearvelocityatpropBodyax[3];/* '<S55>/Sum1' */
  real_T TrueairspeedatpropMotoraxes[3];/* '<S68>/Product' */
  real_T Climbspeedv_c;                /* '<S83>/Gain' */
  real_T v_cv_h;                       /* '<S84>/Divide' */
  real_T Merge;                        /* '<S84>/Merge' */
  real_T transpose[3];                 /* '<S90>/transpose' */
  real_T Product_g;                    /* '<S90>/Product' */
  real_T Sqrt;                         /* '<S87>/Sqrt' */
  real_T ComplextoRealImag;            /* '<S87>/Complex to Real-Imag' */
  real_T Angleofattackrad;             /* '<S82>/Merge' */
  real_T ThrustratioTT_h;              /* '<S71>/Switch' */
  real_T Dynamicthrustmagnitude;       /* '<S67>/Product7' */
  real_T transpose_e[2];               /* '<S81>/transpose' */
  real_T Product_eh;                   /* '<S81>/Product' */
  real_T Sqrt_n;                       /* '<S78>/Sqrt' */
  real_T ComplextoRealImag_p;          /* '<S78>/Complex to Real-Imag' */
  real_T NewtiltedthrustdirectionBodyaxe[3];/* '<S70>/Merge' */
  real_T Product9[3];                  /* '<S67>/Product9' */
  real_T Product_i;                    /* '<S73>/Product' */
  real_T Product1_ar;                  /* '<S73>/Product1' */
  real_T Motortorquemagnitude;         /* '<S73>/Sum' */
  real_T MathFunction[9];              /* '<S75>/Math Function' */
  real_T Product9_p[3];                /* '<S75>/Product9' */
  real_T Momentinducedbyaerodynamicdragp[3];/* '<S67>/Product3' */
  real_T Product8[3];                  /* '<S67>/Product8' */
  real_T Momentinthemotorhubduetobending[3];/* '<S70>/Merge1' */
  real_T Gain;                         /* '<S95>/Gain' */
  real_T Product7;                     /* '<S95>/Product7' */
  real_T Gain1;                        /* '<S95>/Gain1' */
  real_T Product_h;                    /* '<S96>/Product' */
  real_T Product1_m;                   /* '<S96>/Product1' */
  real_T Product2_e;                   /* '<S96>/Product2' */
  real_T Product_i4;                   /* '<S97>/Product' */
  real_T Product1_ar5;                 /* '<S97>/Product1' */
  real_T Product2_h;                   /* '<S97>/Product2' */
  real_T Sum_j[3];                     /* '<S94>/Sum' */
  real_T Conversionrpmtorads;          /* '<S72>/Conversion rpm to rad//s' */
  real_T Product5_k[3];                /* '<S72>/Product5' */
  real_T Product_ez;                   /* '<S109>/Product' */
  real_T Product1_jt;                  /* '<S109>/Product1' */
  real_T Product2_p;                   /* '<S109>/Product2' */
  real_T Product_m;                    /* '<S110>/Product' */
  real_T Product1_k;                   /* '<S110>/Product1' */
  real_T Product2_n;                   /* '<S110>/Product2' */
  real_T Sum_h[3];                     /* '<S69>/Sum' */
  real_T Add[3];                       /* '<S60>/Add' */
  real_T Switch_k;                     /* '<S64>/Switch' */
  real_T transpose_h[2];               /* '<S80>/transpose' */
  real_T Product_dg;                   /* '<S80>/Product' */
  real_T Sqrt_f;                       /* '<S79>/Sqrt' */
  real_T ComplextoRealImag_m;          /* '<S79>/Complex to Real-Imag' */
  real_T Switch_e;                     /* '<S76>/Switch' */
  real_T Bladeflappinganglea_1sdeg;    /* '<S76>/Product4' */
  real_T Flappinganglerad;             /* '<S76>/Conversion deg to rad' */
  real_T Airspeeddirectionintherotorplan[2];/* '<S76>/Divide' */
  real_T Gain_m[2];                    /* '<S76>/Gain' */
  real_T TrigonometricFunction1_o;     /* '<S76>/Trigonometric Function1' */
  real_T Gain1_o;                      /* '<S76>/Gain1' */
  real_T Gain2;                        /* '<S76>/Gain2' */
  real_T TrigonometricFunction_p;      /* '<S76>/Trigonometric Function' */
  real_T Product_d5[2];                /* '<S76>/Product' */
  real_T MotorhubmomentMotoraxes[3];   /* '<S76>/Product1' */
  real_T Reshape1[3];                  /* '<S76>/Reshape1' */
  real_T Product2_j[3];                /* '<S76>/Product2' */
  real_T Product3[3];                  /* '<S76>/Product3' */
  real_T TrigonometricFunction_j;      /* '<S71>/Trigonometric Function' */
  real_T Product2_nr;                  /* '<S71>/Product2' */
  real_T Sum2;                         /* '<S71>/Sum2' */
  real_T Divide_p;                     /* '<S71>/Divide' */
  real_T transpose_i[2];               /* '<S89>/transpose' */
  real_T Product_l;                    /* '<S89>/Product' */
  real_T Sqrt_a;                       /* '<S88>/Sqrt' */
  real_T ComplextoRealImag_n;          /* '<S88>/Complex to Real-Imag' */
  real_T Divide1;                      /* '<S85>/Divide1' */
  real_T Gain_l;                       /* '<S93>/Gain' */
  real_T Product_j;                    /* '<S93>/Product' */
  real_T Product1_c;                   /* '<S93>/Product1' */
  real_T Sum1_c;                       /* '<S93>/Sum1' */
  real_T Sqrt_d;                       /* '<S93>/Sqrt' */
  real_T Divide_n;                     /* '<S92>/Divide' */
  real_T Gain_e;                       /* '<S92>/Gain' */
  real_T Product_ge;                   /* '<S92>/Product' */
  real_T Gain1_e;                      /* '<S92>/Gain1' */
  real_T Product1_o;                   /* '<S92>/Product1' */
  real_T Gain2_c;                      /* '<S92>/Gain2' */
  real_T Product2_es;                  /* '<S92>/Product2' */
  real_T Gain3;                        /* '<S92>/Gain3' */
  real_T Add_c;                        /* '<S92>/Add' */
  real_T Gain_a;                       /* '<S91>/Gain' */
  real_T Product_o;                    /* '<S91>/Product' */
  real_T Product1_h;                   /* '<S91>/Product1' */
  real_T Sum1_i;                       /* '<S91>/Sum1' */
  real_T Sqrt_e;                       /* '<S91>/Sqrt' */
  uint8_T Compare;                     /* '<S62>/Compare' */
  uint8_T Compare_j;                   /* '<S63>/Compare' */
  boolean_T LowerRelop1;               /* '<S64>/LowerRelop1' */
  boolean_T RelationalOperator;        /* '<S61>/Relational Operator' */
  boolean_T LogicalOperator;           /* '<S61>/Logical Operator' */
  boolean_T RelationalOperator1;       /* '<S61>/Relational Operator1' */
  boolean_T LogicalOperator1;          /* '<S61>/Logical Operator1' */
  boolean_T LogicalOperator2;          /* '<S61>/Logical Operator2' */
  boolean_T UpperRelop;                /* '<S64>/UpperRelop' */
};

/* Block states (default storage) for system '<S37>/For Each Subsystem' */
struct DW_CoreSubsys_True0_T {
  int8_T If_ActiveSubsystem;           /* '<S84>/If' */
  int8_T If_ActiveSubsystem_l;         /* '<S82>/If' */
  int8_T If_ActiveSubsystem_e;         /* '<S70>/If' */
};

/* Continuous states for system '<S37>/For Each Subsystem' */
struct X_CoreSubsys_True0_T {
  real_T Integrator_CSTATE;            /* '<S61>/Integrator' */
};

/* State derivatives for system '<S37>/For Each Subsystem' */
struct XDot_CoreSubsys_True0_T {
  real_T Integrator_CSTATE;            /* '<S61>/Integrator' */
};

/* State Disabled for system '<S37>/For Each Subsystem' */
struct XDis_CoreSubsys_True0_T {
  boolean_T Integrator_CSTATE;         /* '<S61>/Integrator' */
};

/* Block signals (default storage) */
struct B_True0_T {
  real_T QIntegrator[4];               /* '<S7>/Q-Integrator' */
  real_T transpose[4];                 /* '<S22>/transpose' */
  real_T Product;                      /* '<S22>/Product' */
  real_T Sqrt;                         /* '<S21>/Sqrt' */
  real_T ComplextoRealImag;            /* '<S21>/Complex to Real-Imag' */
  real_T Divide[4];                    /* '<S18>/Divide' */
  real_T Product_o;                    /* '<S23>/Product' */
  real_T Product2;                     /* '<S23>/Product2' */
  real_T Product3;                     /* '<S23>/Product3' */
  real_T Product4;                     /* '<S23>/Product4' */
  real_T Product_p;                    /* '<S28>/Product' */
  real_T Product2_n;                   /* '<S28>/Product2' */
  real_T Add;                          /* '<S28>/Add' */
  real_T Product_l;                    /* '<S30>/Product' */
  real_T Product2_nt;                  /* '<S30>/Product2' */
  real_T Add_c;                        /* '<S30>/Add' */
  real_T Product_c;                    /* '<S26>/Product' */
  real_T Product2_i;                   /* '<S26>/Product2' */
  real_T Add_cz;                       /* '<S26>/Add' */
  real_T Product_lj;                   /* '<S24>/Product' */
  real_T Product2_ib;                  /* '<S24>/Product2' */
  real_T Product3_i;                   /* '<S24>/Product3' */
  real_T Product4_p;                   /* '<S24>/Product4' */
  real_T Product_pk;                   /* '<S31>/Product' */
  real_T Product2_i4;                  /* '<S31>/Product2' */
  real_T Add_h;                        /* '<S31>/Add' */
  real_T Product_lb;                   /* '<S27>/Product' */
  real_T Product2_o;                   /* '<S27>/Product2' */
  real_T Add_i;                        /* '<S27>/Add' */
  real_T Product_f;                    /* '<S29>/Product' */
  real_T Product2_b;                   /* '<S29>/Product2' */
  real_T Add_k;                        /* '<S29>/Add' */
  real_T Product_cs;                   /* '<S25>/Product' */
  real_T Product2_k;                   /* '<S25>/Product2' */
  real_T Product3_k;                   /* '<S25>/Product3' */
  real_T Product4_b;                   /* '<S25>/Product4' */
  real_T VectorConcatenate[9];         /* '<S32>/Vector Concatenate' */
  real_T DCM_bi[9];                    /* '<S4>/Math Function2' */
  real_T V_b[3];                       /* '<S2>/V_b' */
  real_T Product_b[3];                 /* '<S4>/Product' */
  real_T DCM_bi_c[9];                  /* '<S5>/Math Function2' */
  real_T RateTransition1[4];           /* '<S1>/Rate Transition1' */
  real_T omega[3];                     /* '<S2>/omega' */
  real_T Product_oc[3];                /* '<S111>/Product' */
  real_T TrueairspeedBodyaxes[3];      /* '<S56>/Sum1' */
  real_T SumofElements[3];             /* '<S37>/Sum of Elements' */
  real_T ForceofgravityInertialaxes[3];/* '<S36>/Product1' */
  real_T ForceofgravityBodyaxes[3];    /* '<S36>/Product' */
  real_T Sum[3];                       /* '<S3>/Sum' */
  real_T Product_n[3];                 /* '<S54>/Product' */
  real_T TrueairspeedBodyaxes_m[3];    /* '<S39>/Sum1' */
  real_T transpose_i[3];               /* '<S53>/transpose' */
  real_T Product_f1;                   /* '<S53>/Product' */
  real_T Sqrt_o;                       /* '<S42>/Sqrt' */
  real_T ComplextoRealImag_b;          /* '<S42>/Complex to Real-Imag' */
  real_T Forceagainstdirectionofmotiondu[3];/* '<S38>/Merge' */
  real_T Sum3[3];                      /* '<S3>/Sum3' */
  real_T Sum1[3];                      /* '<S3>/Sum1' */
  real_T Product1[3];                  /* '<S2>/Product1' */
  real_T u2v3;                         /* '<S33>/Product' */
  real_T u3v1;                         /* '<S33>/Product1' */
  real_T u1v2;                         /* '<S33>/Product2' */
  real_T u3v2;                         /* '<S34>/Product' */
  real_T u1v3;                         /* '<S34>/Product1' */
  real_T u2v1;                         /* '<S34>/Product2' */
  real_T Sum_c[3];                     /* '<S10>/Sum' */
  real_T Sum1_o[3];                    /* '<S2>/Sum1' */
  real_T Product_e[3];                 /* '<S5>/Product' */
  real_T Product_na[3];                /* '<S12>/Product' */
  real_T u2v3_m;                       /* '<S13>/Product' */
  real_T u3v1_m;                       /* '<S13>/Product1' */
  real_T u1v2_h;                       /* '<S13>/Product2' */
  real_T u3v2_b;                       /* '<S14>/Product' */
  real_T u1v3_k;                       /* '<S14>/Product1' */
  real_T u2v1_m;                       /* '<S14>/Product2' */
  real_T Sum_k[3];                     /* '<S11>/Sum' */
  real_T SumofElements1[3];            /* '<S37>/Sum of Elements1' */
  real_T Sum2[3];                      /* '<S3>/Sum2' */
  real_T Sum1_n[3];                    /* '<S6>/Sum1' */
  real_T Product_lc[3];                /* '<S6>/Product' */
  real_T transpose_g[3];               /* '<S16>/transpose' */
  real_T Product_py;                   /* '<S16>/Product' */
  real_T u2;                           /* '<S7>/-1//2' */
  real_T Product_i[3];                 /* '<S7>/Product' */
  real_T u2v3_j;                       /* '<S19>/Product' */
  real_T u3v1_h;                       /* '<S19>/Product1' */
  real_T u1v2_i;                       /* '<S19>/Product2' */
  real_T u3v2_d;                       /* '<S20>/Product' */
  real_T u1v3_p;                       /* '<S20>/Product1' */
  real_T u2v1_b;                       /* '<S20>/Product2' */
  real_T Sum_o[3];                     /* '<S15>/Sum' */
  real_T Subtract[3];                  /* '<S7>/Subtract' */
  real_T u2_d[3];                      /* '<S7>/1//2' */
  real_T TmpSignalConversionAtQIntegrato[4];
  real_T Fcn;                          /* '<S8>/Fcn' */
  real_T Fcn1;                         /* '<S8>/Fcn1' */
  real_T Fcn2;                         /* '<S8>/Fcn2' */
  real_T Fcn3;                         /* '<S8>/Fcn3' */
  real_T Fcn4;                         /* '<S8>/Fcn4' */
  real_T TrigonometricFunction;        /* '<S8>/Trigonometric Function' */
  real_T VectorConcatenate_h[3];       /* '<S8>/Vector Concatenate' */
  real_T X_i[3];                       /* '<S2>/X_i' */
  real_T ImpAsg_InsertedFor_Motor_moment[12];
              /* '<S55>/Thrust//moment vector for each propeller (Body axes)' */
  real_T ImpAsg_InsertedFor_Motor_force_[12];
              /* '<S55>/Thrust//moment vector for each propeller (Body axes)' */
  real_T transpose_j[3];               /* '<S52>/transpose' */
  real_T Product_m;                    /* '<S52>/Product' */
  real_T Divide_i;                     /* '<S48>/Divide' */
  real_T Product_fu;                   /* '<S48>/Product' */
  real_T Divide1;                      /* '<S48>/Divide1' */
  real_T Product1_f;                   /* '<S48>/Product1' */
  real_T Divide2;                      /* '<S48>/Divide2' */
  real_T Product2_j;                   /* '<S48>/Product2' */
  real_T Add_f;                        /* '<S48>/Add' */
  real_T ReciprocalSqrt;               /* '<S48>/Reciprocal Sqrt' */
  real_T Product_fb[3];                /* '<S49>/Product' */
  real_T transpose_e[3];               /* '<S51>/transpose' */
  real_T Product_nx;                   /* '<S51>/Product' */
  real_T Sqrt_a;                       /* '<S50>/Sqrt' */
  real_T ComplextoRealImag_i;          /* '<S50>/Complex to Real-Imag' */
  real_T Product_d;                    /* '<S40>/Product' */
  real_T Magnitudeofdragforce;         /* '<S40>/Abs' */
  real_T transpose_h[3];               /* '<S47>/transpose' */
  real_T Product_dc;                   /* '<S47>/Product' */
  real_T Sqrt_oc;                      /* '<S46>/Sqrt' */
  real_T ComplextoRealImag_m;          /* '<S46>/Complex to Real-Imag' */
  real_T Divide_n[3];                  /* '<S43>/Divide' */
  real_T Product1_m[3];                /* '<S40>/Product1' */
  real_T quat_output[4];               /* '<S7>/MATLAB Function' */
  B_CoreSubsys_True0_T CoreSubsys[4];  /* '<S37>/For Each Subsystem' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_True0_T {
  real_T RateTransition1_Buffer0[4];   /* '<S1>/Rate Transition1' */
  real_T Product_DWORK1[9];            /* '<S6>/Product' */
  real_T Product_DWORK3[3];            /* '<S6>/Product' */
  int32_T Product_DWORK2[3];           /* '<S6>/Product' */
  int_T QIntegrator_IWORK;             /* '<S7>/Q-Integrator' */
  int8_T If_ActiveSubsystem;           /* '<S38>/If' */
  DW_CoreSubsys_True0_T CoreSubsys[4]; /* '<S37>/For Each Subsystem' */
};

/* Continuous states (default storage) */
struct X_True0_T {
  real_T QIntegrator_CSTATE[4];        /* '<S7>/Q-Integrator' */
  real_T V_b_CSTATE[3];                /* '<S2>/V_b' */
  real_T omega_CSTATE[3];              /* '<S2>/omega' */
  real_T X_i_CSTATE[3];                /* '<S2>/X_i' */
  X_CoreSubsys_True0_T CoreSubsys[4];  /* '<S55>/CoreSubsys' */
};

/* State derivatives (default storage) */
struct XDot_True0_T {
  real_T QIntegrator_CSTATE[4];        /* '<S7>/Q-Integrator' */
  real_T V_b_CSTATE[3];                /* '<S2>/V_b' */
  real_T omega_CSTATE[3];              /* '<S2>/omega' */
  real_T X_i_CSTATE[3];                /* '<S2>/X_i' */
  XDot_CoreSubsys_True0_T CoreSubsys[4];/* '<S55>/CoreSubsys' */
};

/* State disabled  */
struct XDis_True0_T {
  boolean_T QIntegrator_CSTATE[4];     /* '<S7>/Q-Integrator' */
  boolean_T V_b_CSTATE[3];             /* '<S2>/V_b' */
  boolean_T omega_CSTATE[3];           /* '<S2>/omega' */
  boolean_T X_i_CSTATE[3];             /* '<S2>/X_i' */
  XDis_CoreSubsys_True0_T CoreSubsys[4];/* '<S55>/CoreSubsys' */
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
struct ExtU_True0_T {
  real_T MotorMatrix_real[68];         /* '<Root>/MotorMatrix_real' */
  real_T CoG_real[3];                  /* '<Root>/CoG_real' */
  real_T mass_real;                    /* '<Root>/mass_real' */
  real_T J_real[9];                    /* '<Root>/J_real' */
  real_T MotorMatrix_nominal[68];      /* '<Root>/MotorMatrix_nominal' */
  real_T CoG_nominal[3];               /* '<Root>/CoG_nominal' */
  real_T mass_nominal;                 /* '<Root>/mass_nominal' */
  real_T J_nominal[9];                 /* '<Root>/J_nominal' */
  real_T Wind_i[3];                    /* '<Root>/Wind_i' */
  real_T Force_disturb[3];             /* '<Root>/Force_disturb' */
  real_T Moment_disturb[3];            /* '<Root>/Moment_disturb' */
  real_T Surface_params[3];            /* '<Root>/Surface_params' */
  real_T RPMcommands[4];               /* '<Root>/RPM commands' */
};

/* External outputs (root outports fed by signals with default storage) */
struct ExtY_True0_T {
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
};

/* Parameters for system: '<S37>/For Each Subsystem' */
struct P_CoreSubsys_True0_T_ {
  real_T Gain_Gain;                    /* Expression: 0.5
                                        * Referenced by: '<S91>/Gain'
                                        */
  real_T Gain_Gain_b;                  /* Expression: 0.5
                                        * Referenced by: '<S93>/Gain'
                                        */
  real_T AoArad_Y0;                    /* Expression: [0]
                                        * Referenced by: '<S86>/AoA (rad)'
                                        */
  real_T Constant_Value;               /* Expression: 0
                                        * Referenced by: '<S86>/Constant'
                                        */
  real_T ThrustdirectionBody_Y0[3];    /* Expression: [0;0;-1]
                                        * Referenced by: '<S77>/Thrust direction (Body)'
                                        */
  real_T HubmomentBody_Y0[3];          /* Expression: [0;0;0]
                                        * Referenced by: '<S77>/Hub moment (Body)'
                                        */
  real_T Constant_Value_l[3];          /* Expression: [0;0;-1]
                                        * Referenced by: '<S77>/Constant'
                                        */
  real_T Constant1_Value[3];           /* Expression: [0;0;0]
                                        * Referenced by: '<S77>/Constant1'
                                        */
  real_T Bladeflappingdisengaged_Value;/* Expression: 0
                                        * Referenced by: '<S76>/Blade flapping disengaged'
                                        */
  real_T Constant_Value_i;             /* Expression: 0
                                        * Referenced by: '<S76>/Constant'
                                        */
  real_T Switch_Threshold;             /* Expression: 0.5
                                        * Referenced by: '<S76>/Switch'
                                        */
  real_T Gain_Gain_h;                  /* Expression: -1
                                        * Referenced by: '<S76>/Gain'
                                        */
  real_T Gain1_Gain;                   /* Expression: -1
                                        * Referenced by: '<S76>/Gain1'
                                        */
  real_T Gain2_Gain;                   /* Expression: -1
                                        * Referenced by: '<S76>/Gain2'
                                        */
  real_T Constant_Value_e;             /* Expression: 0
                                        * Referenced by: '<S62>/Constant'
                                        */
  real_T Constant_Value_c;             /* Expression: 0
                                        * Referenced by: '<S63>/Constant'
                                        */
  real_T Constant_Value_g;             /* Expression: 0
                                        * Referenced by: '<S61>/Constant'
                                        */
  real_T Gain_Gain_hr;                 /* Expression: -1
                                        * Referenced by: '<S101>/Gain'
                                        */
  real_T Gain_Gain_k;                  /* Expression: -1
                                        * Referenced by: '<S83>/Gain'
                                        */
  real_T Constant_Value_im;            /* Expression: 1
                                        * Referenced by: '<S71>/Constant'
                                        */
  real_T Switch_Threshold_n;           /* Expression: 0.5
                                        * Referenced by: '<S71>/Switch'
                                        */
  real_T Constant_Value_b[3];          /* Expression: [0;0;-1]
                                        * Referenced by: '<S75>/Constant'
                                        */
  real_T Gain_Gain_n;                  /* Expression: 0.5
                                        * Referenced by: '<S95>/Gain'
                                        */
  real_T Gain1_Gain_c;                 /* Expression: 7/12
                                        * Referenced by: '<S95>/Gain1'
                                        */
};

/* Parameters (default storage) */
struct P_True0_T_ {
  real_T Att_init[3];                  /* Variable: Att_init
                                        * Referenced by: '<S2>/Constant'
                                        */
  real_T Blade_flapping;               /* Variable: Blade_flapping
                                        * Referenced by: '<S76>/Constant2'
                                        */
  real_T C_D;                          /* Variable: C_D
                                        * Referenced by: '<S40>/Constant2'
                                        */
  real_T Dyn_thrust;                   /* Variable: Dyn_thrust
                                        * Referenced by: '<S71>/Constant1'
                                        */
  real_T Vb_init[3];                   /* Variable: Vb_init
                                        * Referenced by: '<S2>/V_b'
                                        */
  real_T Xi_init[3];                   /* Variable: Xi_init
                                        * Referenced by: '<S2>/X_i'
                                        */
  real_T d2r;                          /* Variable: d2r
                                        * Referenced by:
                                        *   '<S59>/Conversion deg to rad'
                                        *   '<S68>/Conversion deg to rad'
                                        *   '<S76>/Conversion deg to rad'
                                        */
  real_T k1;                           /* Variable: k1
                                        * Referenced by: '<S92>/Gain'
                                        */
  real_T k2;                           /* Variable: k2
                                        * Referenced by: '<S92>/Gain1'
                                        */
  real_T k3;                           /* Variable: k3
                                        * Referenced by: '<S92>/Gain2'
                                        */
  real_T k4;                           /* Variable: k4
                                        * Referenced by: '<S92>/Gain3'
                                        */
  real_T k_a1s;                        /* Variable: k_a1s
                                        * Referenced by: '<S76>/Blade flapping gain [deg//(m//s)]'
                                        */
  real_T k_beta;                       /* Variable: k_beta
                                        * Referenced by: '<S76>/Constant1'
                                        */
  real_T kappa;                        /* Variable: kappa
                                        * Referenced by: '<S92>/Constant'
                                        */
  real_T omega_init[3];                /* Variable: omega_init
                                        * Referenced by: '<S2>/omega'
                                        */
  real_T rho;                          /* Variable: rho
                                        * Referenced by: '<S40>/Constant1'
                                        */
  real_T rpm2radpersec;                /* Variable: rpm2radpersec
                                        * Referenced by: '<S72>/Conversion rpm to rad//s'
                                        */
  real_T rpm_init;                     /* Variable: rpm_init
                                        * Referenced by:
                                        *   '<S1>/Rate Transition1'
                                        *   '<S61>/Integrator'
                                        */
  real_T v_h;                          /* Variable: v_h
                                        * Referenced by:
                                        *   '<S71>/Induced velocity at hover'
                                        *   '<S91>/Induced velocity at hover'
                                        *   '<S92>/Induced velocity at hover'
                                        *   '<S93>/Induced velocity at hover'
                                        */
  real_T Dragforce_Y0[3];              /* Expression: [0;0;-1]
                                        * Referenced by: '<S41>/Drag force'
                                        */
  real_T Constant_Value[3];            /* Expression: [0;0;0]
                                        * Referenced by: '<S41>/Constant'
                                        */
  real_T Constant_Value_e;             /* Expression: 1/2
                                        * Referenced by: '<S40>/Constant'
                                        */
  real_T Dragforceopposesdirectionofairs;/* Expression: -1
                                          * Referenced by: '<S40>/Drag force opposes direction of airspeed'
                                          */
  real_T Gain_Gain;                    /* Expression: 2
                                        * Referenced by: '<S28>/Gain'
                                        */
  real_T Gain_Gain_e;                  /* Expression: 2
                                        * Referenced by: '<S30>/Gain'
                                        */
  real_T Gain_Gain_ex;                 /* Expression: 2
                                        * Referenced by: '<S26>/Gain'
                                        */
  real_T Gain_Gain_f;                  /* Expression: 2
                                        * Referenced by: '<S31>/Gain'
                                        */
  real_T Gain_Gain_c;                  /* Expression: 2
                                        * Referenced by: '<S27>/Gain'
                                        */
  real_T Gain_Gain_k;                  /* Expression: 2
                                        * Referenced by: '<S29>/Gain'
                                        */
  real_T GravityInertialaxes_Value[3]; /* Expression: [0;0;g]
                                        * Referenced by: '<S36>/Gravity (Inertial axes)'
                                        */
  real_T u2_Gain;                      /* Expression: -0.5
                                        * Referenced by: '<S7>/-1//2'
                                        */
  real_T u2_Gain_c;                    /* Expression: 0.5
                                        * Referenced by: '<S7>/1//2'
                                        */
  real_T Gain_Gain_d;                  /* Expression: -1
                                        * Referenced by: '<S8>/Gain'
                                        */
  P_CoreSubsys_True0_T CoreSubsys;     /* '<S37>/For Each Subsystem' */
};

/* Real-time Model Data Structure */
struct tag_RTM_True0_T {
  const char_T *errorStatus;
  RTWLogInfo *rtwLogInfo;
  RTWSolverInfo solverInfo;
  X_True0_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  XDis_True0_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[17];
  real_T odeF[3][17];
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
    boolean_T firstInitCondFlag;
    struct {
      uint8_T TID[3];
      uint8_T cLimit[3];
    } TaskCounters;

    struct {
      uint8_T TID1_2;
    } RateInteraction;

    time_T tFinal;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *t;
    time_T tArray[3];
  } Timing;
};

/* Class declaration for model True0 */
class True0 final
{
  /* public data and function members */
 public:
  /* Copy Constructor */
  True0(True0 const&) = delete;

  /* Assignment Operator */
  True0& operator= (True0 const&) & = delete;

  /* Move Constructor */
  True0(True0 &&) = delete;

  /* Move Assignment Operator */
  True0& operator= (True0 &&) = delete;

  /* Real-Time Model get method */
  RT_MODEL_True0_T * getRTM();

  /* External inputs */
  ExtU_True0_T True0_U;

  /* External outputs */
  ExtY_True0_T True0_Y;

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
  True0();

  /* Destructor */
  ~True0();

  /* private data and function members */
 private:
  /* Block signals */
  B_True0_T True0_B;

  /* Block states */
  DW_True0_T True0_DW;

  /* Tunable parameters */
  static P_True0_T True0_P;

  /* Block continuous states */
  X_True0_T True0_X;

  /* Continuous states update member function*/
  void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si );

  /* Derivatives member function */
  void True0_derivatives();

  /* Real-Time Model */
  RT_MODEL_True0_T True0_M;
};

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<S64>/Data Type Duplicate' : Unused code path elimination
 * Block '<S64>/Data Type Propagation' : Unused code path elimination
 * Block '<S37>/Reshape' : Unused code path elimination
 * Block '<S4>/Reshape' : Reshape block reduction
 * Block '<S4>/Reshape1' : Reshape block reduction
 * Block '<S5>/Reshape' : Reshape block reduction
 * Block '<S5>/Reshape1' : Reshape block reduction
 * Block '<S6>/Reshape' : Reshape block reduction
 * Block '<S6>/Reshape ' : Reshape block reduction
 * Block '<S12>/Reshape' : Reshape block reduction
 * Block '<S12>/Reshape1' : Reshape block reduction
 * Block '<S16>/Reshape' : Reshape block reduction
 * Block '<S16>/Reshape1' : Reshape block reduction
 * Block '<S22>/Reshape' : Reshape block reduction
 * Block '<S22>/Reshape1' : Reshape block reduction
 * Block '<S8>/Reshape' : Reshape block reduction
 * Block '<S32>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S47>/Reshape' : Reshape block reduction
 * Block '<S47>/Reshape1' : Reshape block reduction
 * Block '<S51>/Reshape' : Reshape block reduction
 * Block '<S51>/Reshape1' : Reshape block reduction
 * Block '<S52>/Reshape' : Reshape block reduction
 * Block '<S52>/Reshape1' : Reshape block reduction
 * Block '<S53>/Reshape' : Reshape block reduction
 * Block '<S53>/Reshape1' : Reshape block reduction
 * Block '<S54>/Reshape' : Reshape block reduction
 * Block '<S54>/Reshape1' : Reshape block reduction
 * Block '<S39>/Reshape' : Reshape block reduction
 * Block '<S36>/Reshape' : Reshape block reduction
 * Block '<S58>/Reshape' : Reshape block reduction
 * Block '<S59>/Reshape1' : Reshape block reduction
 * Block '<S55>/Rate Transition' : Eliminated since input and output rates are identical
 * Block '<S76>/Reshape3' : Reshape block reduction
 * Block '<S80>/Reshape' : Reshape block reduction
 * Block '<S80>/Reshape1' : Reshape block reduction
 * Block '<S81>/Reshape' : Reshape block reduction
 * Block '<S81>/Reshape1' : Reshape block reduction
 * Block '<S89>/Reshape' : Reshape block reduction
 * Block '<S89>/Reshape1' : Reshape block reduction
 * Block '<S90>/Reshape' : Reshape block reduction
 * Block '<S90>/Reshape1' : Reshape block reduction
 * Block '<S72>/Reshape1' : Reshape block reduction
 * Block '<S72>/Reshape2' : Reshape block reduction
 * Block '<S75>/Reshape5' : Reshape block reduction
 * Block '<S108>/Reshape (9) to [3x3] column-major' : Reshape block reduction
 * Block '<S60>/Reshape' : Reshape block reduction
 * Block '<S111>/Reshape' : Reshape block reduction
 * Block '<S111>/Reshape1' : Reshape block reduction
 * Block '<S56>/Reshape' : Reshape block reduction
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
 * hilite_system('Sim_Multi/True dynamic system representation of a multirotor UAV')    - opens subsystem Sim_Multi/True dynamic system representation of a multirotor UAV
 * hilite_system('Sim_Multi/True dynamic system representation of a multirotor UAV/Kp') - opens and selects block Kp
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'Sim_Multi'
 * '<S1>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV'
 * '<S2>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model'
 * '<S3>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation'
 * '<S4>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Body to Inertial'
 * '<S5>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Body to Inertial1'
 * '<S6>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Calculate omega_dot'
 * '<S7>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq.'
 * '<S8>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Euler'
 * '<S9>'   : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix'
 * '<S10>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/omega x V_b'
 * '<S11>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)'
 * '<S12>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Calculate omega_dot/Subsystem'
 * '<S13>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)/Subsystem'
 * '<S14>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Calculate omega_dot/Cross Product omega x (J * omega)/Subsystem1'
 * '<S15>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Cross Product'
 * '<S16>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Dot product'
 * '<S17>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./MATLAB Function'
 * '<S18>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Normalize'
 * '<S19>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Cross Product/Subsystem'
 * '<S20>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Cross Product/Subsystem1'
 * '<S21>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Normalize/norm'
 * '<S22>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat Strapdown Eq./Normalize/norm/dot_product'
 * '<S23>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem'
 * '<S24>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem1'
 * '<S25>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem2'
 * '<S26>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem3'
 * '<S27>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem4'
 * '<S28>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem5'
 * '<S29>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem6'
 * '<S30>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem7'
 * '<S31>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem8'
 * '<S32>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/Quat to Rotation Matrix/Subsystem9'
 * '<S33>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/omega x V_b/Subsystem'
 * '<S34>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/6DOF model/omega x V_b/Subsystem1'
 * '<S35>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes'
 * '<S36>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute force of gravity in Body axes'
 * '<S37>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers'
 * '<S38>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force'
 * '<S39>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Total airspeed (Body axes)'
 * '<S40>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed'
 * '<S41>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Zero airspeed'
 * '<S42>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/norm'
 * '<S43>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize'
 * '<S44>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation'
 * '<S45>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/u^2'
 * '<S46>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize/norm'
 * '<S47>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Normalize/norm/dot_product'
 * '<S48>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Scaling factor computation'
 * '<S49>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation'
 * '<S50>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation/norm'
 * '<S51>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/Surface area computation/Surface area computation/norm/dot_product'
 * '<S52>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/Nonzero airspeed/u^2/dot_product'
 * '<S53>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Aerodynamic drag force/norm/dot_product'
 * '<S54>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Compute aerodynamic drag of the airframe in Body axes/Total airspeed (Body axes)/Inertial to Body'
 * '<S55>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem'
 * '<S56>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/Total airspeed (Body axes)'
 * '<S57>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics'
 * '<S58>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product'
 * '<S59>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Leverarm vector from real CoG to each propeller (Body axes)'
 * '<S60>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)'
 * '<S61>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits'
 * '<S62>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Compare To Zero'
 * '<S63>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Compare To Zero1'
 * '<S64>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Actuator dynamics/Continuous Integrator with dynamic upper//lower limits/Saturation Dynamic'
 * '<S65>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product/Subsystem'
 * '<S66>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Cross Product/Subsystem1'
 * '<S67>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments'
 * '<S68>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes'
 * '<S69>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product'
 * '<S70>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations'
 * '<S71>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio'
 * '<S72>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque'
 * '<S73>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Hover moment magnitude'
 * '<S74>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Hover thrust magnitude'
 * '<S75>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Thrust direction in Body axes (without blade flapping)'
 * '<S76>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane'
 * '<S77>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Zero airspeed in rotor plane'
 * '<S78>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/norm'
 * '<S79>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane/norm'
 * '<S80>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/Nonzero airspeed in rotor plane/norm/dot_product'
 * '<S81>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Blade flapping computations/norm/dot_product'
 * '<S82>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude'
 * '<S83>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Compute the climb speed'
 * '<S84>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller'
 * '<S85>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed'
 * '<S86>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Zero airspeed'
 * '<S87>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/norm2'
 * '<S88>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed/norm1'
 * '<S89>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/Nonzero airspeed/norm1/dot_product'
 * '<S90>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Angle of attack and airspeed magnitude/norm2/dot_product'
 * '<S91>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Normal working state vc//vh >= 0'
 * '<S92>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Vortex ring state -2 <= vc//vh < 0 '
 * '<S93>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Dynamic thrust ratio/Flight modes of the propeller/Windmill braking state vc//vh < -2'
 * '<S94>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product'
 * '<S95>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Propeller moment of inertia'
 * '<S96>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product/Subsystem'
 * '<S97>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Compute dynamic motor thrust and moments/Gyroscopic torque/Cross Product/Subsystem1'
 * '<S98>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor'
 * '<S99>'  : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem'
 * '<S100>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem1'
 * '<S101>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem2'
 * '<S102>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem3'
 * '<S103>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem4'
 * '<S104>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem5'
 * '<S105>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem6'
 * '<S106>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem7'
 * '<S107>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem8'
 * '<S108>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Coordinate transformation Body to Motor axes/Rotation Matrix Body to Motor/Subsystem9'
 * '<S109>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product/Subsystem'
 * '<S110>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/For Each Subsystem/Thrust//moment vector for each propeller (Body axes)/Cross Product/Subsystem1'
 * '<S111>' : 'Sim_Multi/True dynamic system representation of a multirotor UAV/Force//Moment computation/Forces and moments generated by spinning propellers/Total airspeed (Body axes)/Inertial to Body'
 */
#endif                                 /* RTW_HEADER_True0_h_ */
