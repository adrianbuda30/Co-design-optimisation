/*
 * True0_data.cpp
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

#include "True0.h"

/* Block parameters (default storage) */
P_True0_T True0::True0_P{
  /* Variable: Att_init
   * Referenced by: '<S2>/Constant'
   */
  { 0.0, 0.0, 0.0 },

  /* Variable: Blade_flapping
   * Referenced by: '<S76>/Constant2'
   */
  0.0,

  /* Variable: C_D
   * Referenced by: '<S40>/Constant2'
   */
  0.4,

  /* Variable: Dyn_thrust
   * Referenced by: '<S71>/Constant1'
   */
  0.0,

  /* Variable: Vb_init
   * Referenced by: '<S2>/V_b'
   */
  { 0.0, 0.0, 0.0 },

  /* Variable: Xi_init
   * Referenced by: '<S2>/X_i'
   */
  { 0.0, 0.0, 0.0 },

  /* Variable: d2r
   * Referenced by:
   *   '<S59>/Conversion deg to rad'
   *   '<S68>/Conversion deg to rad'
   *   '<S76>/Conversion deg to rad'
   */
  0.017453292519943295,

  /* Variable: k1
   * Referenced by: '<S92>/Gain'
   */
  -1.125,

  /* Variable: k2
   * Referenced by: '<S92>/Gain1'
   */
  -1.372,

  /* Variable: k3
   * Referenced by: '<S92>/Gain2'
   */
  -1.718,

  /* Variable: k4
   * Referenced by: '<S92>/Gain3'
   */
  -0.655,

  /* Variable: k_a1s
   * Referenced by: '<S76>/Blade flapping gain [deg//(m//s)]'
   */
  0.375,

  /* Variable: k_beta
   * Referenced by: '<S76>/Constant1'
   */
  0.23,

  /* Variable: kappa
   * Referenced by: '<S92>/Constant'
   */
  1.0,

  /* Variable: omega_init
   * Referenced by: '<S2>/omega'
   */
  { 0.0, 0.0, 0.0 },

  /* Variable: rho
   * Referenced by: '<S40>/Constant1'
   */
  1.225,

  /* Variable: rpm2radpersec
   * Referenced by: '<S72>/Conversion rpm to rad//s'
   */
  0.10471975511965977,

  /* Variable: rpm_init
   * Referenced by:
   *   '<S1>/Rate Transition1'
   *   '<S61>/Integrator'
   */
  3104.5025852,

  /* Variable: v_h
   * Referenced by:
   *   '<S71>/Induced velocity at hover'
   *   '<S91>/Induced velocity at hover'
   *   '<S92>/Induced velocity at hover'
   *   '<S93>/Induced velocity at hover'
   */
  4.0,

  /* Expression: [0;0;-1]
   * Referenced by: '<S41>/Drag force'
   */
  { 0.0, 0.0, -1.0 },

  /* Expression: [0;0;0]
   * Referenced by: '<S41>/Constant'
   */
  { 0.0, 0.0, 0.0 },

  /* Expression: 1/2
   * Referenced by: '<S40>/Constant'
   */
  0.5,

  /* Expression: -1
   * Referenced by: '<S40>/Drag force opposes direction of airspeed'
   */
  -1.0,

  /* Expression: 2
   * Referenced by: '<S28>/Gain'
   */
  2.0,

  /* Expression: 2
   * Referenced by: '<S30>/Gain'
   */
  2.0,

  /* Expression: 2
   * Referenced by: '<S26>/Gain'
   */
  2.0,

  /* Expression: 2
   * Referenced by: '<S31>/Gain'
   */
  2.0,

  /* Expression: 2
   * Referenced by: '<S27>/Gain'
   */
  2.0,

  /* Expression: 2
   * Referenced by: '<S29>/Gain'
   */
  2.0,

  /* Expression: [0;0;g]
   * Referenced by: '<S36>/Gravity (Inertial axes)'
   */
  { 0.0, 0.0, 9.80665 },

  /* Expression: -0.5
   * Referenced by: '<S7>/-1//2'
   */
  -0.5,

  /* Expression: 0.5
   * Referenced by: '<S7>/1//2'
   */
  0.5,

  /* Expression: -1
   * Referenced by: '<S8>/Gain'
   */
  -1.0,

  /* Start of '<S55>/CoreSubsys' */
  {
    /* Expression: 0.5
     * Referenced by: '<S91>/Gain'
     */
    0.5,

    /* Expression: 0.5
     * Referenced by: '<S93>/Gain'
     */
    0.5,

    /* Expression: [0]
     * Referenced by: '<S86>/AoA (rad)'
     */
    0.0,

    /* Expression: 0
     * Referenced by: '<S86>/Constant'
     */
    0.0,

    /* Expression: [0;0;-1]
     * Referenced by: '<S77>/Thrust direction (Body)'
     */
    { 0.0, 0.0, -1.0 },

    /* Expression: [0;0;0]
     * Referenced by: '<S77>/Hub moment (Body)'
     */
    { 0.0, 0.0, 0.0 },

    /* Expression: [0;0;-1]
     * Referenced by: '<S77>/Constant'
     */
    { 0.0, 0.0, -1.0 },

    /* Expression: [0;0;0]
     * Referenced by: '<S77>/Constant1'
     */
    { 0.0, 0.0, 0.0 },

    /* Expression: 0
     * Referenced by: '<S76>/Blade flapping disengaged'
     */
    0.0,

    /* Expression: 0
     * Referenced by: '<S76>/Constant'
     */
    0.0,

    /* Expression: 0.5
     * Referenced by: '<S76>/Switch'
     */
    0.5,

    /* Expression: -1
     * Referenced by: '<S76>/Gain'
     */
    -1.0,

    /* Expression: -1
     * Referenced by: '<S76>/Gain1'
     */
    -1.0,

    /* Expression: -1
     * Referenced by: '<S76>/Gain2'
     */
    -1.0,

    /* Expression: 0
     * Referenced by: '<S62>/Constant'
     */
    0.0,

    /* Expression: 0
     * Referenced by: '<S63>/Constant'
     */
    0.0,

    /* Expression: 0
     * Referenced by: '<S61>/Constant'
     */
    0.0,

    /* Expression: -1
     * Referenced by: '<S101>/Gain'
     */
    -1.0,

    /* Expression: -1
     * Referenced by: '<S83>/Gain'
     */
    -1.0,

    /* Expression: 1
     * Referenced by: '<S71>/Constant'
     */
    1.0,

    /* Expression: 0.5
     * Referenced by: '<S71>/Switch'
     */
    0.5,

    /* Expression: [0;0;-1]
     * Referenced by: '<S75>/Constant'
     */
    { 0.0, 0.0, -1.0 },

    /* Expression: 0.5
     * Referenced by: '<S95>/Gain'
     */
    0.5,

    /* Expression: 7/12
     * Referenced by: '<S95>/Gain1'
     */
    0.58333333333333337
  }
  /* End of '<S55>/CoreSubsys' */
};
