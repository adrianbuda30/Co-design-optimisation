/*
 *  rtmodel.cpp:
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

#include "rtmodel.h"

/* Use this function only if you need to maintain compatibility with an existing static main program. */
void Sim_Multi_step(Sim_Multi & Sim_Multi_Obj, int_T tid)
{
  switch (tid) {
   case 0 :
    Sim_Multi_Obj.step0();
    break;

   case 2 :
    Sim_Multi_Obj.step2();
    break;

   case 3 :
    Sim_Multi_Obj.step3();
    break;

   case 4 :
    Sim_Multi_Obj.step4();
    break;

   case 5 :
    Sim_Multi_Obj.step5();
    break;

   default :
    /* do nothing */
    break;
  }
}
