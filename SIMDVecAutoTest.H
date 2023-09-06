// ===========================================================================
//
// SIMDVecAutoTest.H --
// code for auto-test (comparison of Vec and SerialVec behavior)
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

// https://stackoverflow.com/questions/15651488/
// how-to-pass-a-template-function-in-a-template-argument-list
//
// ... you cannot pass a function template as a template
// argument. From 14.3.3: A template-argument for a template
// template-parameter shall be the name of a class template or an
// alias template, expressed as id-expression.

// 01. Aug 22 (Jonas Keller):
// removed SIMD_INLINE from all test functions, speeds up compilation by a
// factor of at least 6 with clang and there is no reason to inline the test
// functions

// 02. Oct 22 (Jonas Keller): added support for different comparators

// 05. Nov 22 (Jonas Keller):
// added time measurement for SIMD versions of all tested functions

// 14. Nov 22 (Jonas Keller):
// added an inner loop to the time measurement with at-compile-time-known
// number of iterations to make the compiler unroll the loop

// 22. Jan 23 (Jonas Keller): moved auto test related stuff into auto_test
// namespace

// 13. May 23 (Jonas Keller): added Double support

#pragma once
#ifndef SIMD_VEC_AUTO_TEST_H_
#define SIMD_VEC_AUTO_TEST_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecAutoTestSerial.H"
#include "SIMDVecExt.H"
#include "TimeMeasurement.H"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

namespace simd {
namespace auto_test {

// how many error cases should be printed to stdout
#define PRINT_ERRORS 1

// 14. Nov 22 (Jonas Keller): added SIMD_TIME_MEASUREMENT_UNROLL
// how many iterations of the time measurement loop should be unrolled
#define SIMD_TIME_MEASUREMENT_UNROLL 16

// 15. Mar 23 (Jonas Keller): added ONLY_TIME_MEASUREMENT
// if true, only time measurements are performed, no correctness checks, which
// speeds up the test significantly
#define ONLY_TIME_MEASUREMENT false

// ===========================================================================
// aux. functions and macros
// ===========================================================================

void printInfo(const std::string &fct, const std::string &args)
{
  printf("testing %s(%s)\n", fct.c_str(), args.c_str());
  fflush(stdout);
}

void printError(const std::string &fct, const std::string &args)
{
  printf("*** error in %s(%s)\n", fct.c_str(), args.c_str());
  fflush(stdout);
}

void printErrorStats(int errors, int trials)
{
  if (errors > 0)
    printf("---------- %d errors / %d trials ----------\n", errors, trials);
}

void printTimeStats(long int time, int trials)
{
  // the "manual" loop unrolling in the time measurement loop rounds the
  // number of iterations down to a multiple of SIMD_TIME_MEASUREMENT_UNROLL,
  // so, to reflect that, the same has to be done with trials
  trials =
    (trials / SIMD_TIME_MEASUREMENT_UNROLL) * SIMD_TIME_MEASUREMENT_UNROLL;
  double avg = (double) time / (double) trials;
  printf("---------- avg. time: %f ns (%d trials) ----------\n", avg, trials);
}

// ===========================================================================
// doNotOptimize
// modified from:
// https://github.com/google/benchmark/blob/d2a8a4ee41b923876c034afb939c4fc03598e622/include/benchmark/benchmark.h#L437-L527
// License: Apache License 2.0
// ===========================================================================

// 05. Nov 22 (Jonas Keller): added doNotOptimize

// The doNotOptimize(...) function can be used to prevent a value or
// expression from being optimized away by the compiler. This function is
// intended to add little to no overhead.
// See: https://youtu.be/nXaxk27zwlk?t=2441
#if !defined(__GNUC__) || defined(__llvm__) || defined(__INTEL_COMPILER)
template <class Tp>
SIMD_INLINE void doNotOptimize(Tp const &value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
SIMD_INLINE void doNotOptimize(Tp &value)
{
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}
#elif (__cplusplus >= 201103L) && (__GNUC__ >= 5)
// Workaround for a bug with full argument copy overhead with GCC.
// See: #1340 and https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105519
template <class Tp>
SIMD_INLINE typename std::enable_if<std::is_trivially_copyable<Tp>::value &&
                                    (sizeof(Tp) <= sizeof(Tp *))>::type
doNotOptimize(Tp const &value)
{
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
SIMD_INLINE typename std::enable_if<!std::is_trivially_copyable<Tp>::value ||
                                    (sizeof(Tp) > sizeof(Tp *))>::type
doNotOptimize(Tp const &value)
{
  asm volatile("" : : "m"(value) : "memory");
}

template <class Tp>
SIMD_INLINE typename std::enable_if<std::is_trivially_copyable<Tp>::value &&
                                    (sizeof(Tp) <= sizeof(Tp *))>::type
doNotOptimize(Tp &value)
{
  asm volatile("" : "+m,r"(value) : : "memory");
}

template <class Tp>
SIMD_INLINE typename std::enable_if<!std::is_trivially_copyable<Tp>::value ||
                                    (sizeof(Tp) > sizeof(Tp *))>::type
doNotOptimize(Tp &value)
{
  asm volatile("" : "+m"(value) : : "memory");
}

#else
// Fallback for GCC < 5. Can add some overhead because the compiler is forced
// to use memory operations instead of operations with registers.
// TODO: Remove if GCC < 5 will be unsupported.
template <class Tp>
SIMD_INLINE void doNotOptimize(Tp const &value)
{
  asm volatile("" : : "m"(value) : "memory");
}

template <class Tp>
SIMD_INLINE void doNotOptimize(Tp &value)
{
  asm volatile("" : "+m"(value) : : "memory");
}
#endif
// #define EXIT break
#define EXIT
// #define EXIT exit(-1)

#define PRINT(FORMAT, VEC)                                                     \
  {                                                                            \
    printf(#VEC "\t= ");                                                       \
    print(FORMAT, VEC.getVec());                                               \
    puts("");                                                                  \
    fflush(stdout);                                                            \
  }

#define PRINTVAL(FORMAT, VAL)                                                  \
  {                                                                            \
    printf(#VAL "\t= ");                                                       \
    printf(FORMAT, VAL);                                                       \
    puts("");                                                                  \
    fflush(stdout);                                                            \
  }

// ===========================================================================
// comparators to provide different comparison methods
// ===========================================================================

struct CmpEqual
{
  // compare bit patterns for all integers
  template <typename T>
  static bool cmpScalar(T expected, T actual)
  {
    static_assert(std::is_integral<T>::value, "");
    return memcmp(&expected, &actual, sizeof(T)) == 0;
  }

  // compare floats while making sure that -0.0f == +0.0f
  static bool cmpScalar(float expected, float actual)
  {
    // if the bits are equal, the floats are equal
    if (memcmp(&expected, &actual, sizeof(float)) == 0) { return true; }
    // if the floats are both +-0, they are equal
    return (std::abs(expected) == 0.0f) && (std::abs(actual) == 0.0f);
  }

  static bool cmpScalar(double expected, double actual)
  {
    // if the bits are equal, the doubles are equal
    if (memcmp(&expected, &actual, sizeof(double)) == 0) { return true; }
    // if the doubles are both +-0, they are equal
    return (std::abs(expected) == 0.0) && (std::abs(actual) == 0.0);
  }

  template <typename T, int SIMD_WIDTH>
  static bool cmpVec(SerialVec<T, SIMD_WIDTH> &expected,
                     SerialVec<T, SIMD_WIDTH> &actual)
  {
    if (memcmp(&expected, &actual, sizeof(SerialVec<T, SIMD_WIDTH>)) == 0) {
      return true;
    }
    for (int i = 0; i < SerialVec<T, SIMD_WIDTH>::elems; ++i) {
      if (!cmpScalar(expected[i], actual[i])) { return false; }
    }
    return true;
  }

  template <typename T, int SIMD_WIDTH>
  static bool cmpVec(SerialVec<T, SIMD_WIDTH> &expected,
                     Vec<T, SIMD_WIDTH> &actual)
  {
    SerialVec<T, SIMD_WIDTH> actualSerial =
      SerialVec<T, SIMD_WIDTH>::fromVec(actual);
    return cmpVec(expected, actualSerial);
  }
};

// allows a relative error of COEFF * 10^EXP_10 * 2^EXP_2 for floats
// a maximum relative error of 1.5*2^-12 for example can be specified
// with CmpRelErr<15, -1, -12>
// (because 15 * 10^-1 * 2^-12 = 1.5 * 2^-12)
template <int64_t COEFF, int64_t EXP_10, int64_t EXP_2>
struct CmpRelError
{
  // for integers, just use the equal comparator
  template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
  static bool cmpScalar(T expected, T actual)
  {
    static_assert(std::is_integral<T>::value, "");
    return CmpEqual::cmpScalar(expected, actual);
  }

  // for floats, compare the relative error, if the floats are not the same
  template <typename T, SIMD_ENABLE_IF(std::is_floating_point<T>::value),
            typename = void>
  static bool cmpScalar(T expected, T actual)
  {
    if (CmpEqual::cmpScalar(expected, actual)) { return true; }
    double maxRelErr = COEFF * std::pow(10, EXP_10) * std::pow(2, EXP_2);
    double relErr    = std::fabs(expected - actual) / std::fabs(expected);
    return relErr <= maxRelErr;
  }

  template <typename T, int SIMD_WIDTH>
  static bool cmpVec(SerialVec<T, SIMD_WIDTH> &expected,
                     SerialVec<T, SIMD_WIDTH> &actual)
  {
    if (memcmp((void *) &expected, (void *) &actual,
               sizeof(SerialVec<T, SIMD_WIDTH>)) == 0) {
      return true;
    }
    for (int i = 0; i < SerialVec<T, SIMD_WIDTH>::elems; ++i) {
      if (!cmpScalar(expected[i], actual[i])) { return false; }
    }
    return true;
  }

  template <typename T, int SIMD_WIDTH>
  static bool cmpVec(SerialVec<T, SIMD_WIDTH> &expected,
                     Vec<T, SIMD_WIDTH> &actual)
  {
    SerialVec<T, SIMD_WIDTH> actualSerial =
      SerialVec<T, SIMD_WIDTH>::fromVec(actual);
    return cmpVec(expected, actualSerial);
  }
};

// ===========================================================================
// tests for different type combinations: no additional template argument
// ===========================================================================

template <
  template <typename, int, template <typename, int> class, class> class Tester,
  int SIMD_WIDTH, template <typename, int> class FCT, class CMP = CmpEqual>
struct TestSignedInt
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<SignedByte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Short, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Int, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <typename, int> class, class> class Tester,
  int SIMD_WIDTH, template <typename, int> class FCT, class CMP = CmpEqual>
struct TestUnsignedInt
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Byte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Word, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <typename, int> class, class> class Tester,
  int SIMD_WIDTH, template <typename, int> class FCT, class CMP = CmpEqual>
struct TestFloat
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Float, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Double, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <typename, int> class, class> class Tester,
  int SIMD_WIDTH, template <typename, int> class FCT, class CMP = CmpEqual>
struct TestInt
{
  static void test(int repeats, const std::string &pattern)
  {
    TestSignedInt<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestUnsignedInt<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <typename, int> class, class> class Tester,
  int SIMD_WIDTH, template <typename, int> class FCT, class CMP = CmpEqual>
struct TestSigned
{
  static void test(int repeats, const std::string &pattern)
  {
    TestSignedInt<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestFloat<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <typename, int> class, class> class Tester,
  int SIMD_WIDTH, template <typename, int> class FCT, class CMP = CmpEqual>
struct TestAll
{
  static void test(int repeats, const std::string &pattern)
  {
    TestInt<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestFloat<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// tests for different type combinations: one additional int template argument
// ===========================================================================

template <template <typename, int, template <int, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestSignedIntTI
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<SignedByte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Short, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Int, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <template <typename, int, template <int, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestUnsignedIntTI
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Byte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Word, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <template <typename, int, template <int, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestFloatTI
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Float, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Double, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <template <typename, int, template <int, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestIntTI
{
  static void test(int repeats, const std::string &pattern)
  {
    TestSignedIntTI<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestUnsignedIntTI<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <template <typename, int, template <int, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestSignedTI
{
  static void test(int repeats, const std::string &pattern)
  {
    TestSignedIntTI<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestFloatTI<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <template <typename, int, template <int, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestAllTI
{
  static void test(int repeats, const std::string &pattern)
  {
    TestIntTI<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestFloatTI<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// tests for different type combinations: two additional int template args.
// ===========================================================================

template <
  template <typename, int, template <int, int, typename, int> class, class>
  class Tester,
  int SIMD_WIDTH, template <int, int, typename, int> class FCT,
  class CMP = CmpEqual>
struct TestSignedIntTII
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<SignedByte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Short, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Int, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <int, int, typename, int> class, class>
  class Tester,
  int SIMD_WIDTH, template <int, int, typename, int> class FCT,
  class CMP = CmpEqual>
struct TestUnsignedIntTII
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Byte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Word, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <int, int, typename, int> class, class>
  class Tester,
  int SIMD_WIDTH, template <int, int, typename, int> class FCT,
  class CMP = CmpEqual>
struct TestFloatTII
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Float, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Double, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <int, int, typename, int> class, class>
  class Tester,
  int SIMD_WIDTH, template <int, int, typename, int> class FCT,
  class CMP = CmpEqual>
struct TestIntTII
{
  static void test(int repeats, const std::string &pattern)
  {
    TestSignedIntTII<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestUnsignedIntTII<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <int, int, typename, int> class, class>
  class Tester,
  int SIMD_WIDTH, template <int, int, typename, int> class FCT,
  class CMP = CmpEqual>
struct TestSignedTII
{
  static void test(int repeats, const std::string &pattern)
  {
    TestSignedIntTII<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestFloatTII<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

template <
  template <typename, int, template <int, int, typename, int> class, class>
  class Tester,
  int SIMD_WIDTH, template <int, int, typename, int> class FCT,
  class CMP = CmpEqual>
struct TestAllTII
{
  static void test(int repeats, const std::string &pattern)
  {
    TestIntTII<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    TestFloatTII<Tester, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// test for different type combinations: one additional template type arg.
// ===========================================================================

template <template <typename, typename, int,
                    template <typename, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <typename, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestAllTT
{
  // 26. Nov 22 (Jonas Keller): Added testAllSeconds to make this easier to
  // read and make sure that all types are tested.
  static void test(int repeats, const std::string &pattern)
  {
    testAllSeconds<Byte>(repeats, pattern);
    testAllSeconds<SignedByte>(repeats, pattern);
    testAllSeconds<Word>(repeats, pattern);
    testAllSeconds<Short>(repeats, pattern);
    testAllSeconds<Int>(repeats, pattern);
    testAllSeconds<Float>(repeats, pattern);
    testAllSeconds<Double>(repeats, pattern);
  }

private:
  template <typename T1>
  static void testAllSeconds(int repeats, const std::string &pattern)
  {
    Tester<T1, Byte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<T1, SignedByte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<T1, Word, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<T1, Short, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<T1, Int, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<T1, Float, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<T1, Double, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// test for different type combs.: one add. templ. type arg. of the same size
// ===========================================================================

// 07. Feb (Jonas Keller): Added TestAllTTSameSize

template <template <typename, typename, int,
                    template <typename, typename, int> class, class>
          class Tester,
          int SIMD_WIDTH, template <typename, typename, int> class FCT,
          class CMP = CmpEqual>
struct TestAllTTSameSize
{
  static void test(int repeats, const std::string &pattern)
  {
    Tester<Byte, Byte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Byte, SignedByte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<SignedByte, Byte, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<SignedByte, SignedByte, SIMD_WIDTH, FCT, CMP>::test(repeats,
                                                               pattern);
    Tester<Word, Word, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Word, Short, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Short, Word, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Short, Short, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Int, Int, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Int, Float, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Float, Int, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Float, Float, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
    Tester<Double, Double, SIMD_WIDTH, FCT, CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// test of nullary commands
// ===========================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct Nullary
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> cs, cp;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        cs = FCT<T, SIMD_WIDTH>::template apply<SerialVec>();
        cp.setVec(FCT<T, SIMD_WIDTH>::template apply<Vec>());
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          Vec<T, SIMD_WIDTH> res = FCT<T, SIMD_WIDTH>::template apply<Vec>();
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of unary commands
// ===========================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct Unary
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, cs, cp;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<T, SIMD_WIDTH>::apply(a);
        cp.setVec(FCT<T, SIMD_WIDTH>::apply(a.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res =
            FCT<T, SIMD_WIDTH>::template apply<Vec>(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of commands with scalar input and vector result
// ===========================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct Scalar
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "s";
#if !ONLY_TIME_MEASUREMENT
    T a;
    SerialVec<T, SIMD_WIDTH> cs, cp;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        a  = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
        cs = FCT<T, SIMD_WIDTH>::template apply<SerialVec>(a);
        cp.setVec(FCT<T, SIMD_WIDTH>::template apply<Vec>(a));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINTVAL(fmt, a);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      T input               = T();
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res =
            FCT<T, SIMD_WIDTH>::template apply<Vec>(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of commands with uint64_t input and vector result
// ===========================================================================

// 09. Oct 22 (Jonas Keller): added ScalarUInt64

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct ScalarUInt64
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "s";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> cs, cp;
    uint64_t a;
    const char *fmt  = TypeInfo<T>::defaultFormat();
    const char *fmtA = "%" PRIu64;
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        a  = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
        cs = FCT<T, SIMD_WIDTH>::template apply<SerialVec>(a);
        cp.setVec(FCT<T, SIMD_WIDTH>::template apply<Vec>(a));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINTVAL(fmtA, a);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      uint64_t input        = uint64_t();
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res =
            FCT<T, SIMD_WIDTH>::template apply<Vec>(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of unary commands with scalar result
// ===========================================================================

// vector if type T, scalar result of type TRES
template <typename TRES, typename T, int SIMD_WIDTH,
          template <typename, int> class FCT, class CMP = CmpEqual>
struct UnaryScalar
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    TRES cs, cp;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtRes = TypeInfo<TRES>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<T, SIMD_WIDTH>::apply(a);
        cp = FCT<T, SIMD_WIDTH>::apply(a.getVec());
        if (!CMP::cmpScalar(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINTVAL(fmtRes, cs);
            PRINTVAL(fmtRes, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          TRES res = FCT<T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryScalarT : public UnaryScalar<T, T, SIMD_WIDTH, FCT, CMP>
{};

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryScalarInt : public UnaryScalar<int, T, SIMD_WIDTH, FCT, CMP>
{};

// 17. Sep 22 (Jonas Keller): added UnaryScalarUInt64
template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryScalarUInt64
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    uint64_t cs, cp;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtRes = "%" PRIu64;
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<T, SIMD_WIDTH>::apply(a);
        cp = FCT<T, SIMD_WIDTH>::apply(a.getVec());
        if (!CMP::cmpScalar(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINTVAL(fmtRes, cs);
            PRINTVAL(fmtRes, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          uint64_t res = FCT<T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of unary commands with add. uint8_t function parameter and vec result
// ===========================================================================

// 19. Dec 22 (Jonas Keller): added UnaryUInt8

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryUInt8
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    SerialVec<T, SIMD_WIDTH> cs, cp;
    uint8_t b;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtRes = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<T, SIMD_WIDTH>::randomizeInput(a);
        b  = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
        cs = FCT<T, SIMD_WIDTH>::apply(a, b);
        cp.setVec(FCT<T, SIMD_WIDTH>::apply(a.getVec(), b));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINTVAL("%" PRIu8, b);
            PRINT(fmtRes, cs);
            PRINT(fmtRes, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res = FCT<T, SIMD_WIDTH>::apply(input, 0);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of unary commands with single type template parameter
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct UnaryTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    SerialVec<Tout, SIMD_WIDTH> cs, cp;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<Tout, T, SIMD_WIDTH>::apply(a);
        cp.setVec(FCT<Tout, T, SIMD_WIDTH>::apply(a.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmtOut, cs);
            PRINT(fmtOut, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<Tout, SIMD_WIDTH> res = FCT<Tout, T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of unary commands with single int template parameter and scalar res.
// ===========================================================================

template <typename T, int SIMD_WIDTH, int I, int N,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct UnaryTemplateIntScalar
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    T cp, cs;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<I, T, SIMD_WIDTH>::apply(a);
        cp = FCT<I, T, SIMD_WIDTH>::apply(a.getVec());
        if (!CMP::cmpScalar(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINTVAL(fmt, cs);
            PRINTVAL(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          T res = FCT<I, T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    UnaryTemplateIntScalar<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, N,
                           FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = maxI
template <typename T, int SIMD_WIDTH, int N,
          template <int, typename, int> class FCT, class CMP>
struct UnaryTemplateIntScalar<T, SIMD_WIDTH, N, N, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryTemplateIntScalarMinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    UnaryTemplateIntScalar<T, SIMD_WIDTH, FCT<1, T, SIMD_WIDTH>::minI,
                           FCT<1, T, SIMD_WIDTH>::maxI, FCT,
                           CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// test of unary commands with single int template parameter and vector res.
// ===========================================================================

template <typename T, int SIMD_WIDTH, int I, int N,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct UnaryTemplateInt
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, cp, cs;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<I, T, SIMD_WIDTH>::apply(a);
        cp.setVec(FCT<I, T, SIMD_WIDTH>::apply(a.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res = FCT<I, T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    UnaryTemplateInt<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, N, FCT>::test(
      repeats, pattern);
  }
};

// end of recursion at I = maxI
template <typename T, int SIMD_WIDTH, int N,
          template <int, typename, int> class FCT, class CMP>
struct UnaryTemplateInt<T, SIMD_WIDTH, N, N, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryTemplateIntMinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    UnaryTemplateInt<T, SIMD_WIDTH, FCT<1, T, SIMD_WIDTH>::minI,
                     FCT<1, T, SIMD_WIDTH>::maxI, FCT, CMP>::test(repeats,
                                                                  pattern);
  }
};

// ===========================================================================
// test of unary commands w. single int template param. and 16 byte vec result
// ===========================================================================

// 10. Oct 22 (Jonas Keller):
// added UnaryTemplateInt16 and UnaryTemplateInt16MinToMax

template <typename T, int SIMD_WIDTH, int I, int N,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct UnaryTemplateInt16
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    SerialVec<T, 16> cp, cs;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
        cs = FCT<I, T, SIMD_WIDTH>::apply(a);
        cp.setVec(FCT<I, T, SIMD_WIDTH>::apply(a.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, 16> res = FCT<I, T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    UnaryTemplateInt16<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, N,
                       FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = maxI
template <typename T, int SIMD_WIDTH, int N,
          template <int, typename, int> class FCT, class CMP>
struct UnaryTemplateInt16<T, SIMD_WIDTH, N, N, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct UnaryTemplateInt16MinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    UnaryTemplateInt16<T, SIMD_WIDTH, FCT<1, T, SIMD_WIDTH>::minI,
                       FCT<1, T, SIMD_WIDTH>::maxI, FCT, CMP>::test(repeats,
                                                                    pattern);
  }
};

// ===========================================================================
// test of binary commands
// ===========================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct Binary
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v,v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, b, cs, cp;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<T, SIMD_WIDTH>::randomizeInput(a);
        FCT<T, SIMD_WIDTH>::randomizeInput(b);
        cs = FCT<T, SIMD_WIDTH>::apply(a, b);
        cp.setVec(FCT<T, SIMD_WIDTH>::apply(a.getVec(), b.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, b);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1, input2;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          Vec<T, SIMD_WIDTH> res = FCT<T, SIMD_WIDTH>::apply(input1, input2);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of binary commands with single type template parameter
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct BinaryTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v,v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, b;
    SerialVec<Tout, SIMD_WIDTH> cs, cp;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
        FCT<Tout, T, SIMD_WIDTH>::randomizeInput(b);
        cs = FCT<Tout, T, SIMD_WIDTH>::apply(a, b);
        cp.setVec(FCT<Tout, T, SIMD_WIDTH>::apply(a.getVec(), b.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, b);
            PRINT(fmtOut, cs);
            PRINT(fmtOut, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1, input2;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          Vec<Tout, SIMD_WIDTH> res =
            FCT<Tout, T, SIMD_WIDTH>::apply(input1, input2);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of commands with unary in and array out w. single type template param.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct UnaryArrayTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args   = "v,*v";
    const int nOutVecs = sizeof(Tout) / sizeof(T);
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<Tout, SIMD_WIDTH>> cs(nOutVecs), cp(nOutVecs);
      std::vector<Vec<Tout, SIMD_WIDTH>,
                  typename Vec<Tout, SIMD_WIDTH>::allocator>
        _cp(nOutVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
        FCT<Tout, T, SIMD_WIDTH>::apply(a, cs.data());
        FCT<Tout, T, SIMD_WIDTH>::apply(a.getVec(), _cp.data());
        bool foundError = false;
        for (int io = 0; io < nOutVecs; io++) {
          cp[io].setVec(_cp[io]);
          if (!CMP::cmpVec(cs[io], cp[io])) foundError = true;
        }
        if (foundError) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            for (int io = 0; io < nOutVecs; io++) {
              PRINT(fmtOut, cs[io]);
              PRINT(fmtOut, cp[io]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<Tout, SIMD_WIDTH> res[nOutVecs];
          FCT<Tout, T, SIMD_WIDTH>::apply(input, res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of commands with array in and vector out
// ===========================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct Array
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args  = "*v";
    const int nInVecs = FCT<T, SIMD_WIDTH>::nInputs;
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> cs, cp;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> a(nInVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _a(nInVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        for (int j = 0; j < nInVecs; j++) {
          FCT<T, SIMD_WIDTH>::randomizeInput(a[j]);
          _a[j] = a[j].getVec();
        }
        cs = FCT<T, SIMD_WIDTH>::apply(a.data());
        cp.setVec(FCT<T, SIMD_WIDTH>::apply(_a.data()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            for (int j = 0; j < nInVecs; j++) PRINT(fmt, a[j]);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[nInVecs];
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res = FCT<T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of commands with array in and vector out w. single templ. type param.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args  = "*v";
    const int nInVecs = FCT<Tout, T, SIMD_WIDTH>::nInputs;
#if !ONLY_TIME_MEASUREMENT
    SerialVec<Tout, SIMD_WIDTH> cs, cp;
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> a(nInVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _a(nInVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        for (int j = 0; j < nInVecs; j++) {
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a[j]);
          _a[j] = a[j].getVec();
        }
        cs = FCT<Tout, T, SIMD_WIDTH>::apply(a.data());
        cp.setVec(FCT<Tout, T, SIMD_WIDTH>::apply(_a.data()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            for (int j = 0; j < nInVecs; j++) PRINT(fmt, a[j]);
            PRINT(fmtOut, cs);
            PRINT(fmtOut, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[nInVecs];
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<Tout, SIMD_WIDTH> res = FCT<Tout, T, SIMD_WIDTH>::apply(input);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of void commands with array in and array out
// ===========================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT,
          class CMP = CmpEqual>
struct ArrayArrayVoid
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "*v,*v";
    const int n      = SIMD_WIDTH / sizeof(T);
#if !ONLY_TIME_MEASUREMENT
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(n), cs(n), cp(n);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(n), _cp(n);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        for (int j = 0; j < n; j++) {
          FCT<T, SIMD_WIDTH>::randomizeInput(as[j]);
          _ap[j] = as[j].getVec();
        }
        // apply function
        FCT<T, SIMD_WIDTH>::apply(as.data(), cs.data());
        FCT<T, SIMD_WIDTH>::apply(_ap.data(), _cp.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < n; j++) {
          // assign Vec -> SerialVec
          cp[j].setVec(_cp[j]);
          if (!CMP::cmpVec(cs[j], cp[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input copy
            for (int j = 0; j < n; j++) PRINT(fmt, as[j]);
            // output
            for (int j = 0; j < n; j++) {
              PRINT(fmt, cs[j]);
              PRINT(fmt, cp[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[n];
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<T, SIMD_WIDTH> res[n];
          FCT<T, SIMD_WIDTH>::apply(input, res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of void commands with array in and array out and single templ. type p.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayArrayVoidTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args     = "*v,*v";
    const int numInVecs  = FCT<Tout, T, SIMD_WIDTH>::numInVecs;
    const int numOutVecs = FCT<Tout, T, SIMD_WIDTH>::numOutVecs;
#if !ONLY_TIME_MEASUREMENT
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(numInVecs);
      std::vector<SerialVec<Tout, SIMD_WIDTH>> cs(numOutVecs), cp(numOutVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(numInVecs);
      std::vector<Vec<Tout, SIMD_WIDTH>,
                  typename Vec<Tout, SIMD_WIDTH>::allocator>
        _cp(numOutVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        for (int j = 0; j < numInVecs; j++) {
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput(as[j]);
          _ap[j] = as[j].getVec();
        }
        // apply function
        FCT<Tout, T, SIMD_WIDTH>::apply(as.data(), cs.data());
        FCT<Tout, T, SIMD_WIDTH>::apply(_ap.data(), _cp.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < numOutVecs; j++) {
          // assign Vec -> SerialVec
          cp[j].setVec(_cp[j]);
          if (!CMP::cmpVec(cs[j], cp[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input
            for (int j = 0; j < numInVecs; j++) PRINT(fmt, as[j]);
            // output
            for (int j = 0; j < numOutVecs; j++) {
              PRINT(fmtOut, cs[j]);
              PRINT(fmtOut, cp[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[numInVecs];
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          Vec<Tout, SIMD_WIDTH> res[numOutVecs];
          FCT<Tout, T, SIMD_WIDTH>::apply(input, res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of void cmds. w. 1 array and float in, array out, single templ. type p.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayDoubleArrayVoidTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args     = "*v,s,*v";
    const int numInVecs  = FCT<Tout, T, SIMD_WIDTH>::numInVecs;
    const int numOutVecs = FCT<Tout, T, SIMD_WIDTH>::numOutVecs;
#if !ONLY_TIME_MEASUREMENT
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(numInVecs);
      std::vector<SerialVec<Tout, SIMD_WIDTH>> cs(numOutVecs), cp(numOutVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(numInVecs);
      std::vector<Vec<Tout, SIMD_WIDTH>,
                  typename Vec<Tout, SIMD_WIDTH>::allocator>
        _cp(numOutVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        const auto d = FCT<Tout, T, SIMD_WIDTH>::randomizeInputFloat();
        for (int j = 0; j < numInVecs; j++) {
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput(as[j]);
          _ap[j] = as[j].getVec();
        }
        // apply function
        FCT<Tout, T, SIMD_WIDTH>::apply(as.data(), d, cs.data());
        FCT<Tout, T, SIMD_WIDTH>::apply(_ap.data(), d, _cp.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < numOutVecs; j++) {
          // assign Vec -> SerialVec
          cp[j].setVec(_cp[j]);
          if (!CMP::cmpVec(cs[j], cp[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input
            printf("d = %g\n", d);
            for (int j = 0; j < numInVecs; j++) PRINT(fmt, as[j]);
            // output
            for (int j = 0; j < numOutVecs; j++) {
              PRINT(fmtOut, cs[j]);
              PRINT(fmtOut, cp[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[numInVecs];
      double inputD         = 1.0;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          doNotOptimize(inputD);
          Vec<Tout, SIMD_WIDTH> res[numOutVecs];
          FCT<Tout, T, SIMD_WIDTH>::apply(input, inputD, res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test void cmds. w. 1 array and 2 float in, array out, single templ. type p.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayDoubleDoubleArrayVoidTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args     = "*v,s,s,*v";
    const int numInVecs  = FCT<Tout, T, SIMD_WIDTH>::numInVecs;
    const int numOutVecs = FCT<Tout, T, SIMD_WIDTH>::numOutVecs;
#if !ONLY_TIME_MEASUREMENT
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(numInVecs);
      std::vector<SerialVec<Tout, SIMD_WIDTH>> cs(numOutVecs), cp(numOutVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(numInVecs);
      std::vector<Vec<Tout, SIMD_WIDTH>,
                  typename Vec<Tout, SIMD_WIDTH>::allocator>
        _cp(numOutVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        const auto d1 = FCT<Tout, T, SIMD_WIDTH>::randomizeInputFloat();
        const auto d2 = FCT<Tout, T, SIMD_WIDTH>::randomizeInputFloat();
        for (int j = 0; j < numInVecs; j++) {
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput(as[j]);
          _ap[j] = as[j].getVec();
        }
        // apply function
        FCT<Tout, T, SIMD_WIDTH>::apply(as.data(), d1, d2, cs.data());
        FCT<Tout, T, SIMD_WIDTH>::apply(_ap.data(), d1, d2, _cp.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < numOutVecs; j++) {
          // assign Vec -> SerialVec
          cp[j].setVec(_cp[j]);
          if (!CMP::cmpVec(cs[j], cp[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input
            printf("d1 = %g, d2 = %g\n", d1, d2);
            for (int j = 0; j < numInVecs; j++) PRINT(fmt, as[j]);
            // output
            for (int j = 0; j < numOutVecs; j++) {
              PRINT(fmtOut, cs[j]);
              PRINT(fmtOut, cp[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[numInVecs];
      double inputD1 = 1.0, inputD2 = 1.0;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          doNotOptimize(inputD1);
          doNotOptimize(inputD2);
          Vec<Tout, SIMD_WIDTH> res[numOutVecs];
          FCT<Tout, T, SIMD_WIDTH>::apply(input, inputD1, inputD2, res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of void cmds. w. 2 arrays and float in, array out, single templ. type p.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayArrayDoubleArrayVoidTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args     = "*v,*v,s,*v";
    const int numInVecs  = FCT<Tout, T, SIMD_WIDTH>::numInVecs;
    const int numOutVecs = FCT<Tout, T, SIMD_WIDTH>::numOutVecs;
#if !ONLY_TIME_MEASUREMENT
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(numInVecs), bs(numInVecs);
      std::vector<SerialVec<Tout, SIMD_WIDTH>> cs(numOutVecs), cp(numOutVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(numInVecs), _bp(numInVecs);
      std::vector<Vec<Tout, SIMD_WIDTH>,
                  typename Vec<Tout, SIMD_WIDTH>::allocator>
        _cp(numOutVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        const auto d = FCT<Tout, T, SIMD_WIDTH>::randomizeInputFloat();
        for (int j = 0; j < numInVecs; j++) {
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput1(as[j]);
          _ap[j] = as[j].getVec();
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput2(bs[j]);
          _bp[j] = bs[j].getVec();
        }
        // apply function
        FCT<Tout, T, SIMD_WIDTH>::apply(as.data(), bs.data(), d, cs.data());
        FCT<Tout, T, SIMD_WIDTH>::apply(_ap.data(), _bp.data(), d, _cp.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < numOutVecs; j++) {
          // assign Vec -> SerialVec
          cp[j].setVec(_cp[j]);
          if (!CMP::cmpVec(cs[j], cp[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input
            printf("d = %g\n", d);
            for (int j = 0; j < numInVecs; j++) {
              PRINT(fmt, as[j]);
              PRINT(fmt, bs[j]);
            }
            // output
            for (int j = 0; j < numOutVecs; j++) {
              PRINT(fmtOut, cs[j]);
              PRINT(fmtOut, cp[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1[numInVecs], input2[numInVecs];
      double inputD         = 1.0;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          doNotOptimize(inputD);
          Vec<Tout, SIMD_WIDTH> res[numOutVecs];
          FCT<Tout, T, SIMD_WIDTH>::apply(input1, input2, inputD, res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test void cmds. w. 2 arrays and 2 float in, array out, single templ. type p.
// ===========================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayArrayDoubleDoubleArrayVoidTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args     = "*v,*v,s,s,*v";
    const int numInVecs  = FCT<Tout, T, SIMD_WIDTH>::numInVecs;
    const int numOutVecs = FCT<Tout, T, SIMD_WIDTH>::numOutVecs;
#if !ONLY_TIME_MEASUREMENT
    const char *fmt    = TypeInfo<T>::defaultFormat();
    const char *fmtOut = TypeInfo<Tout>::defaultFormat();
#endif
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(numInVecs), bs(numInVecs);
      std::vector<SerialVec<Tout, SIMD_WIDTH>> cs(numOutVecs), cp(numOutVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(numInVecs), _bp(numInVecs);
      std::vector<Vec<Tout, SIMD_WIDTH>,
                  typename Vec<Tout, SIMD_WIDTH>::allocator>
        _cp(numOutVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        const auto d1 = FCT<Tout, T, SIMD_WIDTH>::randomizeInputFloat();
        const auto d2 = FCT<Tout, T, SIMD_WIDTH>::randomizeInputFloat();
        for (int j = 0; j < numInVecs; j++) {
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput1(as[j]);
          _ap[j] = as[j].getVec();
          FCT<Tout, T, SIMD_WIDTH>::randomizeInput2(bs[j]);
          _bp[j] = bs[j].getVec();
        }
        // apply function
        FCT<Tout, T, SIMD_WIDTH>::apply(as.data(), bs.data(), d1, d2,
                                        cs.data());
        FCT<Tout, T, SIMD_WIDTH>::apply(_ap.data(), _bp.data(), d1, d2,
                                        _cp.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < numOutVecs; j++) {
          // assign Vec -> SerialVec
          cp[j].setVec(_cp[j]);
          if (!CMP::cmpVec(cs[j], cp[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input
            printf("d1 = %g, d2 = %g\n", d1, d2);
            for (int j = 0; j < numInVecs; j++) {
              PRINT(fmt, as[j]);
              PRINT(fmt, bs[j]);
            }
            // output
            for (int j = 0; j < numOutVecs; j++) {
              PRINT(fmtOut, cs[j]);
              PRINT(fmtOut, cp[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1[numInVecs], input2[numInVecs];
      double inputD1 = 1.0, inputD2 = 1.0;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          doNotOptimize(inputD1);
          doNotOptimize(inputD2);
          Vec<Tout, SIMD_WIDTH> res[numOutVecs];
          FCT<Tout, T, SIMD_WIDTH>::apply(input1, input2, inputD1, inputD2,
                                          res);
          doNotOptimize(res);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};

// ===========================================================================
// test of void commands with array in/out and single int template parameters
// ===========================================================================

template <typename T, int SIMD_WIDTH, int I, int N,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct ArrayVoidTemplateInt
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args  = "*v";
    const int numVecs = FCT<I, T, SIMD_WIDTH>::numVecs;
#if !ONLY_TIME_MEASUREMENT
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      // allocate on the heap to avoid stack overflow
      std::vector<SerialVec<T, SIMD_WIDTH>> as(numVecs), ap(numVecs),
        a(numVecs);
      std::vector<Vec<T, SIMD_WIDTH>, typename Vec<T, SIMD_WIDTH>::allocator>
        _ap(numVecs);
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        for (int j = 0; j < numVecs; j++) {
          FCT<N, T, SIMD_WIDTH>::randomizeInput(as[j]);
          _ap[j] = as[j].getVec();
          // make a copy of the input
          a[j] = as[j];
        }
        // apply function (overwrites results)
        FCT<I, T, SIMD_WIDTH>::apply(as.data());
        FCT<I, T, SIMD_WIDTH>::apply(_ap.data());
        // compare results
        bool equal = true;
        for (int j = 0; j < numVecs; j++) {
          // assign Vec -> SerialVec
          ap[j].setVec(_ap[j]);
          if (!CMP::cmpVec(as[j], ap[j])) equal = false;
        }
        if (!equal) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            // input copy
            for (int j = 0; j < numVecs; j++) PRINT(fmt, a[j]);
            // output
            for (int j = 0; j < numVecs; j++) {
              PRINT(fmt, as[j]);
              PRINT(fmt, ap[j]);
            }
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input[numVecs];
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input);
          FCT<I, T, SIMD_WIDTH>::apply(input);
          doNotOptimize(input);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    ArrayVoidTemplateInt<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, N, FCT,
                         CMP>::test(repeats, pattern);
  }
};

template <typename T, int SIMD_WIDTH, int N,
          template <int, typename, int> class FCT, class CMP>
struct ArrayVoidTemplateInt<T, SIMD_WIDTH, N, N, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct ArrayVoidTemplateIntMinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    ArrayVoidTemplateInt<T, SIMD_WIDTH, FCT<1, T, SIMD_WIDTH>::minI,
                         FCT<1, T, SIMD_WIDTH>::maxI, FCT, CMP>::test(repeats,
                                                                      pattern);
  }
};

// ===========================================================================
// test of binary commands with single int template parameter and vector res.
// ===========================================================================

template <typename T, int SIMD_WIDTH, int I, int N,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct BinaryTemplateInt
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v,v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, b, cp, cs;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
        FCT<I, T, SIMD_WIDTH>::randomizeInput(b);
        cs = FCT<I, T, SIMD_WIDTH>::apply(a, b);
        cp.setVec(FCT<I, T, SIMD_WIDTH>::apply(a.getVec(), b.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, b);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1, input2;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          Vec<T, SIMD_WIDTH> result =
            FCT<I, T, SIMD_WIDTH>::apply(input1, input2);
          doNotOptimize(result);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    BinaryTemplateInt<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, N, FCT,
                      CMP>::test(repeats, pattern);
  }
};

// end of recursion at I = maxI
template <typename T, int SIMD_WIDTH, int N,
          template <int, typename, int> class FCT, class CMP>
struct BinaryTemplateInt<T, SIMD_WIDTH, N, N, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct BinaryTemplateIntMinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    BinaryTemplateInt<T, SIMD_WIDTH, FCT<1, T, SIMD_WIDTH>::minI,
                      FCT<1, T, SIMD_WIDTH>::maxI, FCT, CMP>::test(repeats,
                                                                   pattern);
  }
};

// ===========================================================================
// test of binary commands with two vector outputs and int template arg.
// ===========================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT, class CMP = CmpEqual>
struct BinaryBinaryTemplateInt
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v,v,v&,v&";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, b, cs, cp, ds, dp;
    Vec<T, SIMD_WIDTH> _cp, _dp;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
        FCT<I, T, SIMD_WIDTH>::randomizeInput(b);
        FCT<I, T, SIMD_WIDTH>::apply(a, b, cs, ds);
        FCT<I, T, SIMD_WIDTH>::apply(a.getVec(), b.getVec(), _cp, _dp);
        cp.setVec(_cp);
        dp.setVec(_dp);
        if (!CMP::cmpVec(cs, cp) || !CMP::cmpVec(ds, dp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, b);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
            PRINT(fmt, ds);
            PRINT(fmt, dp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1, input2;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          Vec<T, SIMD_WIDTH> result1, result2;
          FCT<I, T, SIMD_WIDTH>::apply(input1, input2, result1, result2);
          doNotOptimize(result1);
          doNotOptimize(result2);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    BinaryBinaryTemplateInt<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                            FCT, CMP>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT, class CMP>
struct BinaryBinaryTemplateInt<T, SIMD_WIDTH, IMAX, IMAX, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT,
          class CMP = CmpEqual>
struct BinaryBinaryTemplateIntMinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    BinaryBinaryTemplateInt<T, SIMD_WIDTH, FCT<1, T, SIMD_WIDTH>::minI,
                            FCT<1, T, SIMD_WIDTH>::maxI, FCT,
                            CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// test of binary commands with two int template parameters and vector res.
// ===========================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX, int J, int JMAX,
          template <int, int, typename, int> class FCT, class CMP = CmpEqual>
struct BinaryTemplateIntInt
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v,v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, b, cp, cs;
    const char *fmt = TypeInfo<T>::defaultFormat();
#endif
    std::string name = FCT<I, J, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<I, J, T, SIMD_WIDTH>::randomizeInput(a);
        FCT<I, J, T, SIMD_WIDTH>::randomizeInput(b);
        cs = FCT<I, J, T, SIMD_WIDTH>::apply(a, b);
        cp.setVec(FCT<I, J, T, SIMD_WIDTH>::apply(a.getVec(), b.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmt, a);
            PRINT(fmt, b);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1, input2;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          Vec<T, SIMD_WIDTH> result =
            FCT<I, J, T, SIMD_WIDTH>::apply(input1, input2);
          doNotOptimize(result);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
    BinaryTemplateIntInt<T, SIMD_WIDTH, FCT<I, J, T, SIMD_WIDTH>::nextI, IMAX,
                         FCT<I, J, T, SIMD_WIDTH>::nextJ, JMAX, FCT,
                         CMP>::test(repeats, pattern);
  }
};

// end of recursion at J = JMAX for arbitrary I
// assumes that I changes fastest and J changes slowest
template <typename T, int SIMD_WIDTH, int I, int IMAX, int JMAX,
          template <int, int, typename, int> class FCT, class CMP>
struct BinaryTemplateIntInt<T, SIMD_WIDTH, I, IMAX, JMAX, JMAX, FCT, CMP>
{
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH,
          template <int, int, typename, int> class FCT, class CMP = CmpEqual>
struct BinaryTemplateIntIntMinToMax
{
  static void test(int repeats, const std::string &pattern)
  {
    BinaryTemplateIntInt<
      T, SIMD_WIDTH, FCT<0, 0, T, SIMD_WIDTH>::minI,
      FCT<0, 0, T, SIMD_WIDTH>::maxI, FCT<0, 0, T, SIMD_WIDTH>::minJ,
      FCT<0, 0, T, SIMD_WIDTH>::maxJ, FCT, CMP>::test(repeats, pattern);
  }
};

// ===========================================================================
// test of ternary commands with condition arg. & single type template param.
// ===========================================================================

template <typename Tcond, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT, class CMP = CmpEqual>
struct TernaryConditionTemplateType
{
  static void test(int repeats, const std::string &pattern)
  {
    std::string args = "v,v,v";
#if !ONLY_TIME_MEASUREMENT
    SerialVec<T, SIMD_WIDTH> a, b;
    SerialVec<Tcond, SIMD_WIDTH> cond;
    SerialVec<T, SIMD_WIDTH> cs, cp;
    const char *fmt     = TypeInfo<T>::defaultFormat();
    const char *fmtCond = TypeInfo<Tcond>::defaultFormat();
#endif
    std::string name = FCT<Tcond, T, SIMD_WIDTH>::name();
    if (name.find(pattern) != std::string::npos) {
      printInfo(name, args);
#if !ONLY_TIME_MEASUREMENT
      int errors = 0, trials = 0;
      for (int i = 0; i < repeats; i++, trials++) {
        FCT<Tcond, T, SIMD_WIDTH>::randomizeCond(cond);
        FCT<Tcond, T, SIMD_WIDTH>::randomizeInput(a);
        FCT<Tcond, T, SIMD_WIDTH>::randomizeInput(b);
        cs = FCT<Tcond, T, SIMD_WIDTH>::apply(cond, a, b);
        cp.setVec(FCT<Tcond, T, SIMD_WIDTH>::apply(cond.getVec(), a.getVec(),
                                                   b.getVec()));
        if (!CMP::cmpVec(cs, cp)) {
          errors++;
          if (errors <= PRINT_ERRORS) {
            printError(name, args);
            PRINT(fmtCond, cond);
            PRINT(fmt, a);
            PRINT(fmt, b);
            PRINT(fmt, cs);
            PRINT(fmt, cp);
          }
          EXIT;
        }
      }
      printErrorStats(errors, trials);
#endif
      Vec<T, SIMD_WIDTH> input1, input2;
      Vec<Tcond, SIMD_WIDTH> inputCond;
      struct timespec start = getTimeSpecMonotonic();
      for (int i = 0; i < repeats / SIMD_TIME_MEASUREMENT_UNROLL; i++) {
        for (int j = 0; j < SIMD_TIME_MEASUREMENT_UNROLL; j++) {
          doNotOptimize(input1);
          doNotOptimize(input2);
          doNotOptimize(inputCond);
          Vec<T, SIMD_WIDTH> result =
            FCT<Tcond, T, SIMD_WIDTH>::apply(inputCond, input1, input2);
          doNotOptimize(result);
        }
      }
      struct timespec end = getTimeSpecMonotonic();
      long int time       = timeSpecDiffNsec(end, start);
      printTimeStats(time, repeats);
    }
  }
};
} // namespace auto_test
} // namespace simd

#endif
