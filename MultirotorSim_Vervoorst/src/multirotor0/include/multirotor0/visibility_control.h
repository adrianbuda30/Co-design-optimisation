#ifndef MULTIROTOR0__VISIBILITY_CONTROL_H_
#define MULTIROTOR0__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define MULTIROTOR0_EXPORT __attribute__ ((dllexport))
    #define MULTIROTOR0_IMPORT __attribute__ ((dllimport))
  #else
    #define MULTIROTOR0_EXPORT __declspec(dllexport)
    #define MULTIROTOR0_IMPORT __declspec(dllimport)
  #endif
  #ifdef MULTIROTOR0_BUILDING_LIBRARY
    #define MULTIROTOR0_PUBLIC MULTIROTOR0_EXPORT
  #else
    #define MULTIROTOR0_PUBLIC MULTIROTOR0_IMPORT
  #endif
  #define MULTIROTOR0_PUBLIC_TYPE MULTIROTOR0_PUBLIC
  #define MULTIROTOR0_LOCAL
#else
  #define MULTIROTOR0_EXPORT __attribute__ ((visibility("default")))
  #define MULTIROTOR0_IMPORT
  #if __GNUC__ >= 4
    #define MULTIROTOR0_PUBLIC __attribute__ ((visibility("default")))
    #define MULTIROTOR0_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define MULTIROTOR0_PUBLIC
    #define MULTIROTOR0_LOCAL
  #endif
  #define MULTIROTOR0_PUBLIC_TYPE
#endif
#endif  // MULTIROTOR0__VISIBILITY_CONTROL_H_
// Generated 28-Aug-2023 00:36:28
// Copyright 2019-2020 The MathWorks, Inc.
