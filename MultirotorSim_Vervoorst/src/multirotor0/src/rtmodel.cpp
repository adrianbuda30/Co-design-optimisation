/*
 *  rtmodel.cpp:
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

#include "rtmodel.h"

/* Use this function only if you need to maintain compatibility with an existing static main program. */
void multirotor0_step(multirotor0 & multirotor0_Obj, int_T tid)
{
  switch (tid) {
   case 0 :
    multirotor0_Obj.step0();
    break;

   case 2 :
    multirotor0_Obj.step2();
    break;

   default :
    /* do nothing */
    break;
  }
}
