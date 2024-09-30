#ifndef TRUE0__VISIBILITY_CONTROL_H_
#define TRUE0__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define TRUE0_EXPORT __attribute__ ((dllexport))
    #define TRUE0_IMPORT __attribute__ ((dllimport))
  #else
    #define TRUE0_EXPORT __declspec(dllexport)
    #define TRUE0_IMPORT __declspec(dllimport)
  #endif
  #ifdef TRUE0_BUILDING_LIBRARY
    #define TRUE0_PUBLIC TRUE0_EXPORT
  #else
    #define TRUE0_PUBLIC TRUE0_IMPORT
  #endif
  #define TRUE0_PUBLIC_TYPE TRUE0_PUBLIC
  #define TRUE0_LOCAL
#else
  #define TRUE0_EXPORT __attribute__ ((visibility("default")))
  #define TRUE0_IMPORT
  #if __GNUC__ >= 4
    #define TRUE0_PUBLIC __attribute__ ((visibility("default")))
    #define TRUE0_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define TRUE0_PUBLIC
    #define TRUE0_LOCAL
  #endif
  #define TRUE0_PUBLIC_TYPE
#endif
#endif  // TRUE0__VISIBILITY_CONTROL_H_
// Generated 23-Jul-2023 21:03:50
// Copyright 2019-2020 The MathWorks, Inc.
