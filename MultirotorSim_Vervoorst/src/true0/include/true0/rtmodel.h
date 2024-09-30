/*
 *  rtmodel.h:
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

#ifndef RTW_HEADER_rtmodel_h_
#define RTW_HEADER_rtmodel_h_
#include "True0.h"
#define MODEL_CLASSNAME                True0
#define MODEL_STEPNAME                 True0_step
#define GRTINTERFACE                   0

/*
 * ROOT_IO_FORMAT: 0 (Individual arguments)
 * ROOT_IO_FORMAT: 1 (Structure reference)
 * ROOT_IO_FORMAT: 2 (Part of model data structure)
 */
#define ROOT_IO_FORMAT                 2

/* Model wrapper function */
/* Use this function only if you need to maintain compatibility with an existing static main program. */
extern void True0_step(True0 & True0_Obj, int_T tid);

#endif                                 /* RTW_HEADER_rtmodel_h_ */
