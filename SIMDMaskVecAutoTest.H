// ===========================================================================
// 
// SIMDMaskVecAutoTest.H --
// autotest templates for masked operations, Jonas Keller
// 
// This source code file is part of the following software:
// 
//    - the low-level C++ template SIMD library
//    - the SIMD implementation of the MinWarping and the 2D-Warping methods 
//      for local visual homing.
// 
// The software is provided based on the accompanying license agreement
// in the file LICENSE or LICENSE.doc. The software is provided "as is"
// without any warranty by the licensor and without any liability of the
// licensor, and the software may not be distributed by the licensee; see
// the license agreement for details.
// 
// (C) Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
// 
// ===========================================================================

// 05. Nov 22 (Jonas Keller):
// added time measurement for SIMD versions of all tested functions

#ifndef _SIMD_MASK_VEC_AUTO_TEST_H_
#define _SIMD_MASK_VEC_AUTO_TEST_H_

#include "SIMDDefs.H"
#include "SIMDSerialMaskVec.H"
#include "SIMDSerialVec.H"
#include "SIMDTypes.H"
#include "SIMDVecAutoTest.H"
#include "TimeMeasurement.H"

#include <bitset>
#include <iostream>
#include <stdio.h>
#include <string>
#include <time.h>

// 02. Oct 22 (Jonas Keller): added support for different comparators

namespace ns_simd {

#define PRINT_VEC(T, VEC) PRINT(SIMDTypeInfo<T>::defaultFormat(), VEC)
#define PRINT_SERIAL_MASK(T, SIMD_WIDTH, MASK)                                 \
  {                                                                            \
    printf(#MASK "\t= ");                                                      \
    fflush(stdout);                                                            \
    std::string s =                                                            \
        std::bitset<SIMDSerialMask<T, SIMD_WIDTH>::bits>(MASK).to_string();    \
    std::cout << std::string(s.rbegin(), s.rend()) << std::endl << std::flush; \
  }
#define PRINT_SCALAR(T, SCALAR)                                                \
  PRINTVAL(SIMDTypeInfo<T>::defaultFormat(), SCALAR)

// =============================================================================
// test for masked functions with a vector src, a mask and one regular vector
// argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskUnary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src, a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> inputSrc, input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputSrc, inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one regular vector argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZUnary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(inputMask,
                                                                  input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and two regular vector
// arguments
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskBinary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src, a, b;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a, b);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a, b);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> inputSrc, input1, input2;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputSrc, inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and two regular vector arguments
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZBinary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a, b);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a, b);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input1, input2;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with one mask and two vector arguments
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskConditionBinary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a, b);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a, b);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input1, input2;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with one mask and one vector argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskConditionUnary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(inputMask,
                                                                  input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and one regular vector
// argument and one type template parameter
// =============================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct TesterMaskUnaryTemplateType {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<Tout, SIMD_WIDTH> src;
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(src);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<Tout, SIMD_WIDTH> cs =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<Tout, SIMD_WIDTH> cp =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k,
                                                                      a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(Tout, src);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(Tout, cs);
          PRINT_VEC(Tout, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<Tout, SIMD_WIDTH> inputSrc;
    SIMDVec<T, SIMD_WIDTH> input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input);
        doNotOptimize(inputMask);
        SIMDVec<Tout, SIMD_WIDTH> result =
            FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputSrc, inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one regular vector argument
// and one type template parameter
// =============================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct TesterMaskZUnaryTemplateType {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<Tout, SIMD_WIDTH> cs =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<Tout, SIMD_WIDTH> cp =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(Tout, cs);
          PRINT_VEC(Tout, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        SIMDVec<Tout, SIMD_WIDTH> result =
            FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and one scalar argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskScalar {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,s";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      T a = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SCALAR(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> inputSrc, input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    T inputScalar = T();
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input);
        doNotOptimize(inputMask);
        doNotOptimize(inputScalar);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputSrc, inputMask, inputScalar);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one scalar argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZScalar {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,s";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      T a = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SCALAR(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDMask<T, SIMD_WIDTH> inputMask;
    T inputScalar = T();
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputMask);
        doNotOptimize(inputScalar);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(inputMask,
                                                                  inputScalar);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and two vector arguments that return
// a mask
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskCmp {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a, b);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a, b);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cs);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input1, input2;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        SIMDMask<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// 16. Oct 22 (Jonas Keller): added TesterCmpMask

// =============================================================================
// test for functions with two vector arguments that return a mask
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterCmpMask {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec, SIMDSerialMask>(a,
                                                                            b);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(a, b);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cs);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input1, input2;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        SIMDMask<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(input1,
                                                                  input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one vector argument that returns
// an int
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskUnaryInt {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      SIMDSerialVec<T, SIMD_WIDTH> a;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      int cs = FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      int cp = FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (!CMP::cmpScalar(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_SCALAR(int, cs);
          PRINT_SCALAR(int, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        int result = FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
            inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for unary mask functions
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterUnaryK {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialMask>(k);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDMask>(k);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cs);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputMask);
        SIMDMask<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDMask>(inputMask);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for binary mask functions
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterBinaryK {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,m";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k, l;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(l);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialMask>(k, l);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDMask>(k, l);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, l);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cs);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDMask<T, SIMD_WIDTH> inputMask1, inputMask2;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputMask1);
        doNotOptimize(inputMask2);
        SIMDMask<T, SIMD_WIDTH> result =
            FCT<T, SIMD_WIDTH>::template apply<SIMDMask>(inputMask1,
                                                         inputMask2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with src vector, a mask and one vector argument
// with one int template parameter
// =============================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct TesterMaskUnaryTI {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src, a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(src);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input, inputSrc;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input);
        doNotOptimize(inputSrc);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<I, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(
                inputSrc, inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);

    TesterMaskUnaryTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                      FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT, class CMP>
struct TesterMaskUnaryTI<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP> {
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskUnaryTIMinToMax {
  static void test(int repeats, const std::string &pattern) {
    TesterMaskUnaryTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                      FCT<0, T, SIMD_WIDTH>::maxI, FCT>::test(repeats, pattern);
  }
};

// =============================================================================
// test for masked functions with a mask and one vector argument
// with one int template parameter
// =============================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct TesterMaskZUnaryTI {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (!CMP::cmpVec(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDVec<T, SIMD_WIDTH> input;
    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        SIMDVec<T, SIMD_WIDTH> result =
            FCT<I, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(inputMask,
                                                                     input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);

    TesterMaskZUnaryTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                       FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT, class CMP>
struct TesterMaskZUnaryTI<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP> {
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZUnaryTIMinToMax {
  static void test(int repeats, const std::string &pattern) {
    TesterMaskZUnaryTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                       FCT<0, T, SIMD_WIDTH>::maxI, FCT>::test(repeats,
                                                               pattern);
  }
};

// =============================================================================
// test for unary mask functions with one int template parameter
// =============================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct TesterUnaryKTI {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDSerialMask>(k);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDMask>(k);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cs);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    SIMDMask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpec();
    for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (int i = 0; i < SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        doNotOptimize(inputMask);
        SIMDMask<T, SIMD_WIDTH> result =
            FCT<I, T, SIMD_WIDTH>::template apply<SIMDMask>(inputMask);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpec();
    long int time = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);

    TesterUnaryKTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                   FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT, class CMP>
struct TesterUnaryKTI<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP> {
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TesterUnaryKTIMinToMax {
  static void test(int repeats, const std::string &pattern) {
    TesterUnaryKTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                   FCT<0, T, SIMD_WIDTH>::maxI, FCT>::test(repeats, pattern);
  }
};

} // namespace ns_simd

#endif // _SIMD_MASK_VEC_AUTO_TEST_H_
