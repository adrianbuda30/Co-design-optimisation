#ifndef SIM_MULTI__VISIBILITY_CONTROL_H_
#define SIM_MULTI__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define SIM_MULTI_EXPORT __attribute__ ((dllexport))
    #define SIM_MULTI_IMPORT __attribute__ ((dllimport))
  #else
    #define SIM_MULTI_EXPORT __declspec(dllexport)
    #define SIM_MULTI_IMPORT __declspec(dllimport)
  #endif
  #ifdef SIM_MULTI_BUILDING_LIBRARY
    #define SIM_MULTI_PUBLIC SIM_MULTI_EXPORT
  #else
    #define SIM_MULTI_PUBLIC SIM_MULTI_IMPORT
  #endif
  #define SIM_MULTI_PUBLIC_TYPE SIM_MULTI_PUBLIC
  #define SIM_MULTI_LOCAL
#else
  #define SIM_MULTI_EXPORT __attribute__ ((visibility("default")))
  #define SIM_MULTI_IMPORT
  #if __GNUC__ >= 4
    #define SIM_MULTI_PUBLIC __attribute__ ((visibility("default")))
    #define SIM_MULTI_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define SIM_MULTI_PUBLIC
    #define SIM_MULTI_LOCAL
  #endif
  #define SIM_MULTI_PUBLIC_TYPE
#endif
#endif  // SIM_MULTI__VISIBILITY_CONTROL_H_
// Generated 24-Jul-2023 11:58:09
// Copyright 2019-2020 The MathWorks, Inc.
