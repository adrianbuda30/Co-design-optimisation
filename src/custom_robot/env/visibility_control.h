#ifndef CALCSYSMATRICES__VISIBILITY_CONTROL_H_
#define CALCSYSMATRICES__VISIBILITY_CONTROL_H_
#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define CALCSYSMATRICES_EXPORT __attribute__ ((dllexport))
    #define CALCSYSMATRICES_IMPORT __attribute__ ((dllimport))
  #else
    #define CALCSYSMATRICES_EXPORT __declspec(dllexport)
    #define CALCSYSMATRICES_IMPORT __declspec(dllimport)
  #endif
  #ifdef CALCSYSMATRICES_BUILDING_LIBRARY
    #define CALCSYSMATRICES_PUBLIC CALCSYSMATRICES_EXPORT
  #else
    #define CALCSYSMATRICES_PUBLIC CALCSYSMATRICES_IMPORT
  #endif
  #define CALCSYSMATRICES_PUBLIC_TYPE CALCSYSMATRICES_PUBLIC
  #define CALCSYSMATRICES_LOCAL
#else
  #define CALCSYSMATRICES_EXPORT __attribute__ ((visibility("default")))
  #define CALCSYSMATRICES_IMPORT
  #if __GNUC__ >= 4
    #define CALCSYSMATRICES_PUBLIC __attribute__ ((visibility("default")))
    #define CALCSYSMATRICES_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define CALCSYSMATRICES_PUBLIC
    #define CALCSYSMATRICES_LOCAL
  #endif
  #define CALCSYSMATRICES_PUBLIC_TYPE
#endif
#endif  // CALCSYSMATRICES__VISIBILITY_CONTROL_H_
// Generated 23-Nov-2023 17:42:16
// Copyright 2019-2020 The MathWorks, Inc.
