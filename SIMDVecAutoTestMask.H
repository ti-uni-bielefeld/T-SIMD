// ===========================================================================
//
// SIMDVecAutoTestMask.H --
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

// 22. Jan 23 (Jonas Keller): moved auto test related stuff into auto_test
// namespace

// 15. Mar 23 (Jonas Keller): added ONLY_TIME_MEASUREMENT (see
// SIMDVecAutoTest.H)

#pragma once
#ifndef SIMD_VEC_AUTO_TEST_MASK_H_
#define SIMD_VEC_AUTO_TEST_MASK_H_

#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecAutoTest.H"
#include "SIMDVecAutoTestSerial.H"
#include "SIMDVecAutoTestSerialMask.H"
#include "SIMDVecMask.H"
#include "TimeMeasurement.H"

#include <bitset>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>

// 02. Oct 22 (Jonas Keller): added support for different comparators

namespace simd {
namespace auto_test {

#define PRINT_VEC(T, VEC) PRINT(TypeInfo<T>::defaultFormat(), VEC)
#define PRINT_SERIAL_MASK(T, SIMD_WIDTH, MASK)                                 \
  {                                                                            \
    printf(#MASK "\t= ");                                                      \
    fflush(stdout);                                                            \
    std::string s =                                                            \
      std::bitset<SerialMask<T, SIMD_WIDTH>::bits>(MASK).to_string();          \
    std::cout << std::string(s.rbegin(), s.rend()) << std::endl << std::flush; \
  }
#define PRINT_SCALAR(T, SCALAR) PRINTVAL(TypeInfo<T>::defaultFormat(), SCALAR)

// =============================================================================
// test for masked functions with a vector src, a mask and one regular vector
// argument
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskUnary
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "v,m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> src, a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply(src, k, a);
      SerialVec<T, SIMD_WIDTH> cp =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<T, SIMD_WIDTH>::template apply(
          src.getVec(), k.getMask(), a.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> inputSrc, input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputSrc, inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one regular vector argument
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZUnary
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs = FCT<T, SIMD_WIDTH>::template apply(k, a);
      SerialVec<T, SIMD_WIDTH> cp = SerialVec<T, SIMD_WIDTH>::fromVec(
        FCT<T, SIMD_WIDTH>::template apply(k.getMask(), a.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and two regular vector
// arguments
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskBinary
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "v,m,v,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> src, a, b;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply(src, k, a, b);
      SerialVec<T, SIMD_WIDTH> cp =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<T, SIMD_WIDTH>::template apply(
          src.getVec(), k.getMask(), a.getVec(), b.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> inputSrc, input1, input2;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result = FCT<T, SIMD_WIDTH>::template apply(
          inputSrc, inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and two regular vector arguments
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZBinary
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a, b;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs = FCT<T, SIMD_WIDTH>::template apply(k, a, b);
      SerialVec<T, SIMD_WIDTH> cp =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<T, SIMD_WIDTH>::template apply(
          k.getMask(), a.getVec(), b.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input1, input2;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with one mask and two vector arguments
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskConditionBinary
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a, b;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs = FCT<T, SIMD_WIDTH>::template apply(k, a, b);
      SerialVec<T, SIMD_WIDTH> cp =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<T, SIMD_WIDTH>::template apply(
          k.getMask(), a.getVec(), b.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input1, input2;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with one mask and one vector argument
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskConditionUnary
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs = FCT<T, SIMD_WIDTH>::template apply(k, a);
      SerialVec<T, SIMD_WIDTH> cp = SerialVec<T, SIMD_WIDTH>::fromVec(
        FCT<T, SIMD_WIDTH>::template apply(k.getMask(), a.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and one regular vector
// argument and one type template parameter
// =============================================================================

template <typename Tout, typename T, size_t SIMD_WIDTH,
          template <typename, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterMaskUnaryTemplateType
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "v,m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<Tout, SIMD_WIDTH> src;
      SerialVec<T, SIMD_WIDTH> a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(src);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<Tout, SIMD_WIDTH> cs =
        FCT<Tout, T, SIMD_WIDTH>::template apply(src, k, a);
      SerialVec<Tout, SIMD_WIDTH> cp = SerialVec<Tout, SIMD_WIDTH>::fromVec(
        FCT<Tout, T, SIMD_WIDTH>::template apply(src.getVec(), k.getMask(),
                                                 a.getVec()));
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
#endif

    Vec<Tout, SIMD_WIDTH> inputSrc;
    Vec<T, SIMD_WIDTH> input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input);
        doNotOptimize(inputMask);
        Vec<Tout, SIMD_WIDTH> result =
          FCT<Tout, T, SIMD_WIDTH>::template apply(inputSrc, inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one regular vector argument
// and one type template parameter
// =============================================================================

template <typename Tout, typename T, size_t SIMD_WIDTH,
          template <typename, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterMaskZUnaryTemplateType
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<Tout, SIMD_WIDTH> cs =
        FCT<Tout, T, SIMD_WIDTH>::template apply(k, a);
      SerialVec<Tout, SIMD_WIDTH> cp = SerialVec<Tout, SIMD_WIDTH>::fromVec(
        FCT<Tout, T, SIMD_WIDTH>::template apply(k.getMask(), a.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        Vec<Tout, SIMD_WIDTH> result =
          FCT<Tout, T, SIMD_WIDTH>::template apply(inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and one scalar argument
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskScalar
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "v,m,s";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> src;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      T a = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
      SerialVec<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply(src, k, a);
      SerialVec<T, SIMD_WIDTH> cp = SerialVec<T, SIMD_WIDTH>::fromVec(
        FCT<T, SIMD_WIDTH>::template apply(src.getVec(), k.getMask(), a));
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
#endif

    Vec<T, SIMD_WIDTH> inputSrc, input;
    Mask<T, SIMD_WIDTH> inputMask;
    T inputScalar         = T();
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputSrc);
        doNotOptimize(input);
        doNotOptimize(inputMask);
        doNotOptimize(inputScalar);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputSrc, inputMask, inputScalar);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one scalar argument
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskZScalar
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,s";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      T a = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
      SerialVec<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply<SerialVec>(k, a);
      SerialVec<T, SIMD_WIDTH> cp = SerialVec<T, SIMD_WIDTH>::fromVec(
        FCT<T, SIMD_WIDTH>::template apply<Vec>(k.getMask(), a));
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
#endif

    Mask<T, SIMD_WIDTH> inputMask;
    T inputScalar         = T();
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputMask);
        doNotOptimize(inputScalar);
        Vec<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply<Vec>(inputMask, inputScalar);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and two vector arguments that return
// a mask
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskCmp
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialMask<T, SIMD_WIDTH> k;
      SerialVec<T, SIMD_WIDTH> a, b;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      SerialMask<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply(k, a, b);
      SerialMask<T, SIMD_WIDTH> cp =
        SerialMask<T, SIMD_WIDTH>::fromMask(FCT<T, SIMD_WIDTH>::template apply(
          k.getMask(), a.getVec(), b.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input1, input2;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        doNotOptimize(inputMask);
        Mask<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask, input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// 16. Oct 22 (Jonas Keller): added TesterCmpMask

// =============================================================================
// test for functions with two vector arguments that return a mask
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterCmpMask
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "v,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a, b;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      SerialMask<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply<SerialVec, SerialMask>(a, b);
      SerialMask<T, SIMD_WIDTH> cp = SerialMask<T, SIMD_WIDTH>::fromMask(
        FCT<T, SIMD_WIDTH>::template apply<Vec, Mask>(a.getVec(), b.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input1, input2;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input1);
        doNotOptimize(input2);
        Mask<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply<Vec, Mask>(input1, input2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with a mask and one vector argument that returns
// a bool
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterMaskUnaryBool
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialMask<T, SIMD_WIDTH> k;
      SerialVec<T, SIMD_WIDTH> a;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      const bool cs = FCT<T, SIMD_WIDTH>::template apply(k, a);
      const bool cp =
        FCT<T, SIMD_WIDTH>::template apply(k.getMask(), a.getVec());
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
#endif

    Vec<T, SIMD_WIDTH> input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        size_t result = FCT<T, SIMD_WIDTH>::template apply(inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for unary mask functions
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterUnaryK
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SerialMask<T, SIMD_WIDTH> cs = FCT<T, SIMD_WIDTH>::template apply(k);
      SerialMask<T, SIMD_WIDTH> cp = SerialMask<T, SIMD_WIDTH>::fromMask(
        FCT<T, SIMD_WIDTH>::template apply(k.getMask()));
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
#endif

    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputMask);
        Mask<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for binary mask functions
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterBinaryK
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,m";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialMask<T, SIMD_WIDTH> k, l;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(l);
      SerialMask<T, SIMD_WIDTH> cs = FCT<T, SIMD_WIDTH>::template apply(k, l);
      SerialMask<T, SIMD_WIDTH> cp = SerialMask<T, SIMD_WIDTH>::fromMask(
        FCT<T, SIMD_WIDTH>::template apply(k.getMask(), l.getMask()));
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
#endif

    Mask<T, SIMD_WIDTH> inputMask1, inputMask2;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputMask1);
        doNotOptimize(inputMask2);
        Mask<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask1, inputMask2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for binary mask functions with bool return type
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterBinaryBoolK
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,m";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    SerialMask<T, SIMD_WIDTH> k, l;
    FCT<T, SIMD_WIDTH>::randomizeInput(k);
    FCT<T, SIMD_WIDTH>::randomizeInput(l);
    for (size_t i = 0; i < repeats; i++, trials++) {
      bool cs = FCT<T, SIMD_WIDTH>::template apply(k, l);
      bool cp = FCT<T, SIMD_WIDTH>::template apply(k.getMask(), l.getMask());
      if (!CMP::cmpScalar(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, l);
          PRINT_SCALAR(int, cs);
          PRINT_SCALAR(int, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
#endif

    Mask<T, SIMD_WIDTH> inputMask1, inputMask2;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / 1000; i++) {
      for (size_t j = 0; j < 1000; j++) {
        doNotOptimize(inputMask1);
        doNotOptimize(inputMask2);
        bool result =
          FCT<T, SIMD_WIDTH>::template apply(inputMask1, inputMask2);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for unary mask functions with bool return type
// =============================================================================

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterUnaryBoolK
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();

    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    SerialMask<T, SIMD_WIDTH> k;
    FCT<T, SIMD_WIDTH>::randomizeInput(k);
    for (size_t i = 0; i < repeats; i++, trials++) {
      bool cs = FCT<T, SIMD_WIDTH>::template apply(k);
      bool cp = FCT<T, SIMD_WIDTH>::template apply(k.getMask());
      if (!CMP::cmpScalar(cs, cp)) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, k);
          PRINT_SCALAR(int, cs);
          PRINT_SCALAR(int, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
#endif

    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / 1000; i++) {
      for (size_t j = 0; j < 1000; j++) {
        doNotOptimize(inputMask);
        bool result = FCT<T, SIMD_WIDTH>::template apply(inputMask);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

// =============================================================================
// test for masked functions with src vector, a mask and one vector argument
// with one int template parameter
// =============================================================================

template <typename T, size_t SIMD_WIDTH, size_t I, size_t IMAX,
          template <size_t, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterMaskUnaryTI
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "v,m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> src, a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(src);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs =
        FCT<I, T, SIMD_WIDTH>::template apply(src, k, a);
      SerialVec<T, SIMD_WIDTH> cp =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<I, T, SIMD_WIDTH>::template apply(
          src.getVec(), k.getMask(), a.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input, inputSrc;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        doNotOptimize(inputSrc);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<I, T, SIMD_WIDTH>::template apply(inputSrc, inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);

    TesterMaskUnaryTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                      FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, size_t SIMD_WIDTH, size_t IMAX,
          template <size_t, typename, size_t> class FCT, class CMP>
struct TesterMaskUnaryTI<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP>
{
  static void test(size_t, const std::string &) {}
};

template <typename T, size_t SIMD_WIDTH,
          template <size_t, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterMaskUnaryTIMinToMax
{
  static void test(size_t repeats, const std::string &pattern)
  {
    TesterMaskUnaryTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                      FCT<0, T, SIMD_WIDTH>::maxI, FCT>::test(repeats, pattern);
  }
};

// =============================================================================
// test for masked functions with a mask and one vector argument
// with one int template parameter
// =============================================================================

template <typename T, size_t SIMD_WIDTH, size_t I, size_t IMAX,
          template <size_t, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterMaskZUnaryTI
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m,v";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialVec<T, SIMD_WIDTH> a;
      SerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SerialVec<T, SIMD_WIDTH> cs = FCT<I, T, SIMD_WIDTH>::template apply(k, a);
      SerialVec<T, SIMD_WIDTH> cp = SerialVec<T, SIMD_WIDTH>::fromVec(
        FCT<I, T, SIMD_WIDTH>::template apply(k.getMask(), a.getVec()));
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
#endif

    Vec<T, SIMD_WIDTH> input;
    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        doNotOptimize(inputMask);
        Vec<T, SIMD_WIDTH> result =
          FCT<I, T, SIMD_WIDTH>::template apply(inputMask, input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);

    TesterMaskZUnaryTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                       FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, size_t SIMD_WIDTH, size_t IMAX,
          template <size_t, typename, size_t> class FCT, class CMP>
struct TesterMaskZUnaryTI<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP>
{
  static void test(size_t, const std::string &) {}
};

template <typename T, size_t SIMD_WIDTH,
          template <size_t, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterMaskZUnaryTIMinToMax
{
  static void test(size_t repeats, const std::string &pattern)
  {
    TesterMaskZUnaryTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                       FCT<0, T, SIMD_WIDTH>::maxI, FCT>::test(repeats,
                                                               pattern);
  }
};

// =============================================================================
// test for unary mask functions with one int template parameter
// =============================================================================

template <typename T, size_t SIMD_WIDTH, size_t I, size_t IMAX,
          template <size_t, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterUnaryKTI
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "m";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      SerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SerialMask<T, SIMD_WIDTH> cs = FCT<I, T, SIMD_WIDTH>::template apply(k);
      SerialMask<T, SIMD_WIDTH> cp = SerialMask<T, SIMD_WIDTH>::fromMask(
        FCT<I, T, SIMD_WIDTH>::template apply(k.getMask()));
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
#endif

    Mask<T, SIMD_WIDTH> inputMask;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(inputMask);
        Mask<T, SIMD_WIDTH> result =
          FCT<I, T, SIMD_WIDTH>::template apply(inputMask);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);

    TesterUnaryKTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                   FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, size_t SIMD_WIDTH, size_t IMAX,
          template <size_t, typename, size_t> class FCT, class CMP>
struct TesterUnaryKTI<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP>
{
  static void test(size_t, const std::string &) {}
};

template <typename T, size_t SIMD_WIDTH,
          template <size_t, typename, size_t> class FCT, class CMP = CmpEqual>
struct TesterUnaryKTIMinToMax
{
  static void test(size_t repeats, const std::string &pattern)
  {
    TesterUnaryKTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                   FCT<0, T, SIMD_WIDTH>::maxI, FCT>::test(repeats, pattern);
  }
};

// 28. Dec 22 (Jonas Keller): added TesterIntToMask
template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT,
          class CMP = CmpEqual>
struct TesterIntToMask
{
  static void test(size_t repeats, const std::string &pattern)
  {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) { return; }
    std::string args = "i";
    printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
    size_t errors = 0, trials = 0;
    for (size_t i = 0; i < repeats; i++, trials++) {
      unsigned int k = FCT<T, SIMD_WIDTH>::randomizeInputUInt();
      SerialMask<T, SIMD_WIDTH> cs =
        FCT<T, SIMD_WIDTH>::template apply<SerialMask>(k);
      SerialMask<T, SIMD_WIDTH> cp = SerialMask<T, SIMD_WIDTH>::fromMask(
        FCT<T, SIMD_WIDTH>::template apply<Mask>(k));
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SCALAR(int, k);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cs);
          PRINT_SERIAL_MASK(T, SIMD_WIDTH, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
#endif

    int input             = 0;
    struct timespec start = getTimeSpecMonotonic();
    for (size_t i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
      for (size_t j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
        doNotOptimize(input);
        Mask<T, SIMD_WIDTH> result =
          FCT<T, SIMD_WIDTH>::template apply<Mask>(input);
        doNotOptimize(result);
      }
    }
    struct timespec end = getTimeSpecMonotonic();
    long int time       = timeSpecDiffNsec(end, start);
    printTimeStats(time, repeats);
  }
};

} // namespace auto_test
} // namespace simd

#endif // SIMD_VEC_AUTO_TEST_MASK_H_
