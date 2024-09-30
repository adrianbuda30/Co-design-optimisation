/*
 * Sim_Multi_data.cpp
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

/* Invariant block signals (default storage) */
const ConstB_Sim_Multi_T Sim_Multi_ConstB{
  {
    0.0027719999999999997,
    0.0,
    0.0,
    0.0,
    0.0026915,
    0.0,
    0.0,
    0.0,
    0.0048877499999999989
  }
  ,                                    /* '<S13>/Product1' */

  {
    0.039599999999999996,
    0.0,
    0.0,
    0.0,
    0.03845,
    0.0,
    0.0,
    0.0,
    0.11025
  }
  ,                                    /* '<S14>/Product1' */

  {
    0.0594,
    0.0,
    0.0,
    0.0,
    0.057675,
    0.0,
    0.0,
    0.0,
    0.165375
  }
  /* '<S15>/Product1' */
};

/* Constant parameters (default storage) */
const ConstP_Sim_Multi_T Sim_Multi_ConstP{
  /* Pooled Parameter (Mixed Expressions)
   * Referenced by:
   *   '<S2>/Constant4'
   *   '<S59>/Constant'
   */
  { 45.0, 135.0, 225.0, 315.0, 0.17, 0.17, 0.17, 0.17, -0.028, -0.028, -0.028,
    -0.028, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01,
    9.6820000000000012E-5, 9.6820000000000012E-5, 9.6820000000000012E-5,
    9.6820000000000012E-5, 1.0872000000000001E-7, 1.0872000000000001E-7,
    1.0872000000000001E-7, 1.0872000000000001E-7, 1.4504E-6, 1.4504E-6,
    1.4504E-6, 1.4504E-6, 1.6312E-9, 1.6312E-9, 1.6312E-9, 1.6312E-9, 0.0, 0.0,
    0.0, 0.0, 6000.0, 6000.0, 6000.0, 6000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.2032, 0.2032, 0.2032, 0.2032, 0.011, 0.011, 0.011,
    0.011 },

  /* Pooled Parameter (Mixed Expressions)
   * Referenced by:
   *   '<S2>/Constant7'
   *   '<S59>/Constant3'
   */
  { 0.00396, 0.0, 0.0, 0.0, 0.003845, 0.0, 0.0, 0.0, 0.00735 },

  /* Expression: MotorMap
   * Referenced by: '<S8>/Constant1'
   */
  { -2.0797258270192569, -2.0797258270192569, 2.0797258270192569,
    2.0797258270192573, 2.0797258270192578, -2.0797258270192573,
    -2.0797258270192582, 2.0797258270192569, 16.666666666666664,
    -16.666666666666664, 16.666666666666664, -16.666666666666664, 0.25,
    0.24999999999999997, 0.24999999999999997, 0.25 },

  /* Expression: MaxRate_cmd/500
   * Referenced by: '<S10>/Unit conversion [stick value] to [rad//s]'
   */
  { 0.012566370614359173, 0.012566370614359173, 0.0041887902047863905 },

  /* Expression: MaxRate_cmd
   * Referenced by: '<S11>/Saturation'
   */
  { 6.2831853071795862, 6.2831853071795862, 2.0943951023931953 },

  /* Expression: -MaxRate_cmd
   * Referenced by: '<S11>/Saturation'
   */
  { -6.2831853071795862, -6.2831853071795862, -2.0943951023931953 }
};
