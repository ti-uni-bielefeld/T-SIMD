// ===========================================================================
//
// SIMDVecAutoTestSerial.H --
// serial implementations of Vec functions
// used for tests, but also useful as a definition of the parallel functions
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

// 22. Jan 23 (Jonas Keller): moved auto test related stuff into auto_test
// namespace

#ifndef _SIMD_VEC_AUTO_TEST_SERIAL_H_
#define _SIMD_VEC_AUTO_TEST_SERIAL_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecAutoTestRandom.H"

#include <algorithm>
#include <assert.h>
#include <functional>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// design considerations
//=======================
//
// separate class SerialVec such that comparison with Vec is
// possible
//
// Float-only functions (mul, div...) are provided only for
// Float
//
// other functions which are not available for all types (neg...) are
// provided with generic parameter T (won't compile for other types)

namespace simd {
namespace auto_test {

// =========================================================================
// SerialVec
// =========================================================================

// 26. Nov 22 (Jonas Keller): Removed implicit conversion from SerialVec
// to Vec and vice versa and added explicit conversion functions
// instead. This is to make sure the right functions are called when using
// SerialVec in the autotests.

template <typename T, int SIMD_WIDTH>
class SerialVec
{
public:
  static constexpr int elements = Vec<T, SIMD_WIDTH>::elems;
  static constexpr int elems    = elements;
  static constexpr int bytes    = SIMD_WIDTH;

  T vec[elements] SIMD_ATTR_ALIGNED(SIMD_WIDTH);

  SerialVec() {}
  template <typename U>
  SerialVec(const SerialVec<U, SIMD_WIDTH> &x)
  {
    memcpy(vec, x.vec, SIMD_WIDTH);
  }
  template <typename U>
  SerialVec &operator=(const SerialVec<U, SIMD_WIDTH> &x)
  {
    memcpy(vec, x.vec, SIMD_WIDTH);
    return *this;
  }
  Vec<T, SIMD_WIDTH> getVec() const { return load<SIMD_WIDTH>(vec); }
  void setVec(const Vec<T, SIMD_WIDTH> &x) { storeu(vec, x); }
  static SerialVec<T, SIMD_WIDTH> fromVec(const Vec<T, SIMD_WIDTH> &x)
  {
    SerialVec<T, SIMD_WIDTH> y;
    y.setVec(x);
    return y;
  }
  T operator[](int i) const
  {
    if ((i >= 0) && (i < elements)) return vec[i];
    return T(0);
  }
  T &operator[](int i)
  {
    assert((i >= 0) && (i < elements));
    return vec[i];
  }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
  void randomize()
  {
    // random init

    // 03. Aug 22 (Jonas Keller):
    // increased the chance of extreme values (min, max, 0, -0)
    // to make sure those are tested as well
    //
    //     for (int i = 0; i < elements; i++)
    // vec[i] = getRandom<T>();

    // 16. Nov 22 (Jonas Keller):
    // moved selection of random special value to getRandomOrSpecialValue()
    // in SIMDVecAutoTestRandom.H

    // randomize with a 1 in 16 chance to fill entire vector with the same
    // random or special value, otherwise fill with random or special values
    // individually
    if ((rand() & 0x0f) == 0) {
      T value = getRandomOrSpecialValue<T>();
      for (int i = 0; i < elements; i++) vec[i] = value;
    } else {
      for (int i = 0; i < elements; i++) vec[i] = getRandomOrSpecialValue<T>();
    }
  }
  void randomizeSorted(SortSlope s)
  {
    randomize();
    if (s == SortSlope::ASCENDING)
      std::sort(vec, vec + elements, std::less<T>());
    else
      std::sort(vec, vec + elements, std::greater<T>());
  }
  void randomizeNonNegative()
  {
    // random init: non-negative, with injection of zeros

    // 03. Aug 22 (Jonas Keller):
    // increased the chance of extreme values (min, max, 0)
    // to make sure those are tested as well
    //
    //     for (int i = 0; i < elements; i++) {
    // if ((rand() & 0x0f) == 0) vec[i] = T(0);
    // else while ((vec[i] = getRandom<T>()) < T(0));

    // 16. Nov 22 (Jonas Keller):
    // moved selection of random special value to getRandomOrSpecialValue()
    // in SIMDVecAutoTestRandom.H

    // randomize with a 1 in 16 chance to fill entire vector with the same
    // random or special value, otherwise fill with random or special values
    // individually
    if ((rand() & 0x0f) == 0) {
      T value;
      while ((value = getRandomOrSpecialValue<T>()) < T(0))
        ;
      for (int i = 0; i < elements; i++) vec[i] = value;
    } else {
      for (int i = 0; i < elements; i++)
        while ((vec[i] = getRandomOrSpecialValue<T>()) < T(0))
          ;
    }
  }
  void randomizeRanges()
  {
    // 03. Aug 22 (Jonas Keller):
    // increased the chance of extreme values (min, max, 0, -0)
    // to make sure those are tested as well
    //
    //     for (int i = 0; i < elements; i++)
    // vec[i] = getRandomRanges<T>();

    // 16. Nov 22 (Jonas Keller):
    // moved selection of random special value to
    // getRandomOrSpecialValueRanges() in SIMDVecAutoTestRandom.H

    // randomize with a 1 in 16 chance to fill entire vector with the same
    // random or special value, otherwise fill with random or special values
    // individually
    if ((rand() & 0x0f) == 0) {
      T value = getRandomOrSpecialValueRanges<T>();
      for (int i = 0; i < elements; i++) vec[i] = value;
    } else {
      for (int i = 0; i < elements; i++)
        vec[i] = getRandomOrSpecialValueRanges<T>();
    }
  }
  void randomizeRangesNonZero()
  {
    // 03. Aug 22 (Jonas Keller):
    // increased the chance of extreme values (min, max, 0, -0)
    // to make sure those are tested as well
    //
    //     for (int i = 0; i < elements; i++)
    // while ((vec[i] = getRandomRanges<T>()) == T(0));

    // 16. Nov 22 (Jonas Keller):
    // moved selection of random special value to
    // getRandomOrSpecialValueRanges() in SIMDVecAutoTestRandom.H

    // randomize with a 1 in 16 chance to fill entire vector with the same
    // random or special value, otherwise fill with random or special values
    // individually
    if ((rand() & 0x0f) == 0) {
      T value;
      while ((value = getRandomOrSpecialValueRanges<T>()) == T(0))
        ;
      for (int i = 0; i < elements; i++) vec[i] = value;
    } else {
      for (int i = 0; i < elements; i++)
        while ((vec[i] = getRandomOrSpecialValueRanges<T>()) == T(0))
          ;
    }
  }
  void randomizeCond()
  {
    // 03. Aug 22 (Jonas Keller):
    // increased the chance of extreme values (all 0s, all 1s)
    // to make sure those are tested as well
    //
    //     for (int i = 0; i < elements; i++)
    // vec[i] = (rand() & 0x01) ? TypeInfo<T>::trueval() : T(0);

    // randomize with a 1 in 16 chance to be all 1s or all 0s
    if ((rand() & 0x0f) == 0) {
      if (rand() & 0x01) {
        setzero();
      } else {
        setones();
      }
    } else {
      for (int i = 0; i < elements; i++)
        vec[i] = (rand() & 0x01) ? TypeInfo<T>::trueval() : T(0);
    }
  }
  void setzero()
  {
    for (int i = 0; i < elements; i++) vec[i] = T(0);
  }
  void setones()
  {
    for (int i = 0; i < elements; i++) vec[i] = TypeInfo<T>::trueval();
  }
};

// =========================================================================
// daz: treat denormal as zero
// =========================================================================

// 02. Oct 22 (Jonas Keller):
// renamed daz to daz_neon and added daz
// daz now always flushes denormals to zero, while daz_neon only does so
// on ARM NEON platforms

// NOTE: there are differences in the behavior of fwaddmul and
// fmuladd depending on the architecture flags; -mavx2 works,
// -march=native on an avx2 shows differences between serial and
// parallel code; use g++ -march=native (or -mavx2) -Q --help=target
// to reveal the differences; it may have something to do with
// fused-multiply-add (on avx this flag is not different and we see
// no differences)

// no daz for integer types
template <typename T>
static SIMD_INLINE T daz(T x)
{
  return x;
}

// daz for float
static SIMD_INLINE Float daz(Float x)
{
  if (std::isnormal(x) || std::isnan(x) || !std::isfinite(x)) return x;
  // x is now either +0.0f, -0.0f or denormal
  // return 0.0f with the same sign as x
  uint32_t asInt;
  memcpy(&asInt, &x, sizeof(float));
  uint32_t sign = asInt & 0x80000000;
  if (sign)
    return -0.0f;
  else
    return 0.0f;
}

// no daz for integer on all architectures
template <typename T>
static SIMD_INLINE T daz_neon(T a)
{
  return a;
}

static SIMD_INLINE Float daz_neon(Float a)
{
#if defined(SIMDVEC_NEON_ENABLE) && __ARM_ARCH < 8
  // use daz for ARM NEON
  return daz(a);
#else
  return a;
#endif
}

// =========================================================================
// auxiliary functions
// =========================================================================

// -------------------------------------------------------------------------
// add
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE T _add(T a, T b)
{
  return daz_neon(daz_neon(a) + daz_neon(b));
}

// -------------------------------------------------------------------------
// _adds
// -------------------------------------------------------------------------

// 09. Mar 23 (Jonas Keller): removed special case of _adds for Int that made
// _adds not be saturated for Int on Intel

// Byte, SignedByte, Word, Short, Int
template <typename T>
static SIMD_INLINE T _adds(T a, T b)
{
  using BiggerInt = int64_t;
  static_assert(std::is_integral<T>::value, "T must be an integer type");
  static_assert(std::is_signed<BiggerInt>::value, "BiggerInt must be signed");
  static_assert(sizeof(BiggerInt) > sizeof(T),
                "BiggerInt must be bigger than T for saturation to work");
  auto tmp = BiggerInt(a) + BiggerInt(b);
  if (tmp > BiggerInt(TypeInfo<T>::max())) {
    return TypeInfo<T>::max();
  } else if (tmp < BiggerInt(TypeInfo<T>::min())) {
    return TypeInfo<T>::min();
  } else {
    return T(tmp);
  }
}

// Float: not saturated
static SIMD_INLINE Float _adds(Float a, Float b)
{
  return daz_neon(daz_neon(a) + daz_neon(b));
}

// -------------------------------------------------------------------------
// sub
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE T _sub(T a, T b)
{
  return daz_neon(daz_neon(a) - daz_neon(b));
}

// -------------------------------------------------------------------------
// _subs
// -------------------------------------------------------------------------

// 09. Mar 23 (Jonas Keller): removed special case of _subs for Int that made
// _subs not be saturated for Int on Intel

// Byte, SignedByte, Word, Short, Int
template <typename T>
static SIMD_INLINE T _subs(T a, T b)
{
  using BiggerInt = int64_t;
  static_assert(std::is_integral<T>::value, "T must be an integer type");
  static_assert(std::is_signed<BiggerInt>::value, "BiggerInt must be signed");
  static_assert(sizeof(BiggerInt) > sizeof(T),
                "BiggerInt must be bigger than T for saturation to work");
  auto tmp = BiggerInt(a) - BiggerInt(b);
  if (tmp > BiggerInt(TypeInfo<T>::max())) {
    return TypeInfo<T>::max();
  } else if (tmp < BiggerInt(TypeInfo<T>::min())) {
    return TypeInfo<T>::min();
  } else {
    return T(tmp);
  }
}

// Float: not saturated
static SIMD_INLINE Float _subs(Float a, Float b)
{
  return daz_neon(daz_neon(a) - daz_neon(b));
}

static SIMD_INLINE Int _cvts_f2i(Float a)
{
#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  // special way to avoid overflow case
  Float clip = MAX_POS_FLOAT_CONVERTIBLE_TO_INT32;
  return Int(std::min(clip, ::rintf(a)));
#elif defined(SIMDVEC_NEON_ENABLE)
  return Int(::trunc(a));
#else
#error "unknown architecture"
#endif
}

// just convert
static SIMD_INLINE Float _cvts_i2f(Int a)
{
  return Float(a);
}

// -------------------------------------------------------------------------
// _mul / _div
// -------------------------------------------------------------------------

static SIMD_INLINE Float _mul(Float a, Float b)
{
  return daz_neon(daz_neon(a) * daz_neon(b));
}

static SIMD_INLINE Float _div(Float a, Float b)
{
  return daz_neon(daz_neon(a) / daz_neon(b));
}

// -------------------------------------------------------------------------
// _min / _max
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE T _min(T a, T b)
{
  return std::min(a, b);
}

static SIMD_INLINE Float _min(Float a, Float b)
{
  return daz_neon(fmin(daz_neon(a), daz_neon(b)));
}

template <typename T>
static SIMD_INLINE T _max(T a, T b)
{
  return std::max(a, b);
}

static SIMD_INLINE Float _max(Float a, Float b)
{
  return daz_neon(fmax(daz_neon(a), daz_neon(b)));
}

// -------------------------------------------------------------------------
// _abs
// -------------------------------------------------------------------------

// Byte, SignedByte, Word, Short, Int
template <typename T>
static SIMD_INLINE T _abs(T a)
{
  return std::abs(a);
}

// no daz also on NEON
static SIMD_INLINE Float _abs(Float a)
{
  return fabsf(a);
}

// -------------------------------------------------------------------------
// _packs
// -------------------------------------------------------------------------

using ::simd::internal::OutputType;

// also accepts same-size input and output types

// from integer types to integer types
template <typename Tout, typename T>
static SIMD_INLINE Tout _packs(T in, OutputType<Tout>)
{
  T minv = T(TypeInfo<Tout>::min());
  T maxv = T(TypeInfo<Tout>::max());
  return (in >= minv) ? ((in <= maxv) ? Tout(in) : Tout(maxv)) : Tout(minv);
}

// from integer type to same integer type
template <typename T>
static SIMD_INLINE T _packs(T in, OutputType<T>)
{
  return in;
}

// from Int to Float
static SIMD_INLINE Float _packs(Int in, OutputType<Float>)
{
  return _cvts_i2f(in);
}

// from Float to integer types
template <typename Tout>
static SIMD_INLINE Tout _packs(Float in, OutputType<Tout>)
{
  // via Int
  return _packs(_cvts_f2i(in), OutputType<Tout>());
}

// from Float to Float (otherwise ambiguous)
static SIMD_INLINE Float _packs(Float in, OutputType<Float>)
{
  return in;
}

// hub
template <typename Tout, typename Tin>
static SIMD_INLINE Tout _packs(Tin in)
{
  return _packs(in, OutputType<Tout>());
}

// -------------------------------------------------------------------------
// _extend
// -------------------------------------------------------------------------

// also accepts same-size input and output types

// from integer types to integer types
template <typename Tout, typename Tin>
static SIMD_INLINE Tout _extend(Tin in, OutputType<Tout>)
{
  return Tout(in);
}

// from integer type to same-size integer type
template <typename T>
static SIMD_INLINE T _extend(T in, OutputType<T>)
{
  return in;
}

// Float -> Int
static SIMD_INLINE Int _extend(Float in, OutputType<Int>)
{
  return _cvts_f2i(in);
}

// Int -> Float
static SIMD_INLINE Float _extend(Int in, OutputType<Float>)
{
  return _cvts_i2f(in);
}

// Float -> Float (otherwise ambiguous)
static SIMD_INLINE Float _extend(Float in, OutputType<Float>)
{
  return in;
}

// hub
template <typename Tout, typename Tin>
static SIMD_INLINE Tout _extend(Tin in)
{
  return _extend(in, OutputType<Tout>());
}

// -------------------------------------------------------------------------
// _convert
// -------------------------------------------------------------------------

using ::simd::internal::CompareEqual;
using ::simd::internal::CompareGreater;
using ::simd::internal::CompareLess;
using ::simd::internal::CompareTypes;

template <typename Tout, typename Tin>
static SIMD_INLINE Tout _convert(CompareLess, OutputType<Tout>, Tin in)
{
  return _packs<Tout>(in);
}

template <typename Tout, typename Tin>
static SIMD_INLINE Tout _convert(CompareEqual, OutputType<Tout>, Tin in)
{
  return in;
}

static SIMD_INLINE Int _convert(CompareEqual, OutputType<Int>, Float in)
{
  return _cvts_f2i(in);
}

static SIMD_INLINE Float _convert(CompareEqual, OutputType<Float>, Int in)
{
  return _cvts_i2f(in);
}

template <typename Tout, typename Tin>
static SIMD_INLINE Tout _convert(CompareGreater, OutputType<Tout>, Tin in)
{
  return Tout(in);
}

// hub
template <typename Tout, typename Tin>
static SIMD_INLINE Tout _convert(Tin in)
{
  return _convert(CompareTypes<Tout, Tin>(), OutputType<Tout>(), in);
}

// =========================================================================
// wrapper for template functions without vector arguments
// =========================================================================

// the problem is that we can't deduce the vector type
// (SerialVec, Vec) in function templates without vector
// arguments, so we have to use this additional layer

// primary template
template <template <typename, int> class VEC>
struct Deductor;

// -------------------------------------------------------------------------
// for Vec, we access the template functions
// -------------------------------------------------------------------------

template <>
struct Deductor<Vec>
{
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _setzero()
  {
    return setzero<T, SIMD_WIDTH>();
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _setones()
  {
    return setones<T, SIMD_WIDTH>();
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _setmin()
  {
    return setmin<T, SIMD_WIDTH>();
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _setmax()
  {
    return setmax<T, SIMD_WIDTH>();
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _setnegunity()
  {
    return setnegunity<T, SIMD_WIDTH>();
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _setunity()
  {
    return setunity<T, SIMD_WIDTH>();
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _set1(T a)
  {
    return set1<T, SIMD_WIDTH>(a);
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _load(const T *const p)
  {
    return load<SIMD_WIDTH, T>(p);
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _loadu(const T *const p)
  {
    return loadu<SIMD_WIDTH, T>(p);
  }
  // 06. Oct 22 (Jonas Keller): added int2msb
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _int2msb(const uint64_t a)
  {
    return int2msb<T, SIMD_WIDTH>(a);
  }
  // 09. Oct 22 (Jonas Keller): added int2bits
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _int2bits(const uint64_t a)
  {
    return int2bits<T, SIMD_WIDTH>(a);
  }
  // 07. Feb 23 (Jonas Keller): added iota
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _iota()
  {
    return iota<T, SIMD_WIDTH>();
  }
};

// -------------------------------------------------------------------------
// for SerialVec, we directly implement the functions
// -------------------------------------------------------------------------

template <>
struct Deductor<SerialVec>
{
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _setzero()
  {
    SerialVec<T, SIMD_WIDTH> v;
    memset(v.vec, 0x00, SIMD_WIDTH);
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _setones()
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) v[i] = TypeInfo<T>::trueval();
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _setmin()
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) v[i] = TypeInfo<T>::min();
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _setmax()
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) v[i] = TypeInfo<T>::max();
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _setnegunity()
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) v[i] = T(-1);
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _setunity()
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) v[i] = T(1);
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _set1(T a)
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) v[i] = a;
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _load(const T *const p)
  {
    SerialVec<T, SIMD_WIDTH> v;
    memcpy(v.vec, p, SIMD_WIDTH);
    return v;
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _loadu(const T *const p)
  {
    SerialVec<T, SIMD_WIDTH> v;
    memcpy(v.vec, p, SIMD_WIDTH);
    return v;
  }
  // 08. Oct 22 (Jonas Keller): added int2msb
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _int2msb(const uint64_t a)
  {
    return _int2msb<SIMD_WIDTH>(a, T());
  }
  // 09. Oct 22 (Jonas Keller): added int2bits
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _int2bits(const uint64_t a)
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++)
      v[i] = (a >> i) & 1 ? TypeInfo<T>::trueval() : T(0);
    return v;
  }
  // 07. Feb 23 (Jonas Keller): added iota
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _iota()
  {
    SerialVec<T, SIMD_WIDTH> v;
    for (int i = 0; i < v.elements; i++) { v[i] = T(i); }
    return v;
  }

private:
  // int2msb
  // integer types
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SerialVec<T, SIMD_WIDTH> _int2msb(const uint64_t a, T)
  {
    T highestBit = T(T(1) << (sizeof(T) * 8 - 1));
    SerialVec<T, SIMD_WIDTH> result;
    for (int i = 0; i < result.elements; i++)
      result[i] = (a & (uint64_t(1) << i)) ? highestBit : 0;
    return result;
  }
  // Float
  template <int SIMD_WIDTH>
  static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> _int2msb(const uint64_t a,
                                                           Float)
  {
    return reinterpret<Float>(_int2msb<SIMD_WIDTH>(a, Int()));
  }
};

// =========================================================================
// template functions
// =========================================================================

template <typename Tdst, typename Tsrc, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Tdst, SIMD_WIDTH> reinterpret(
  const SerialVec<Tsrc, SIMD_WIDTH> &a)
{
  SerialVec<Tdst, SIMD_WIDTH> v;
  memcpy(v.vec, a.vec, SIMD_WIDTH);
  return v;
}

// float -> int
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Int, SIMD_WIDTH> cvts(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Int, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _cvts_f2i(a[i]);
  return c;
}

// int -> float
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> cvts(
  const SerialVec<Int, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _cvts_i2f(a[i]);
  return c;
}

// hub (just for compatibility with Vec version which has
// primary template)
template <typename Tout, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Tout, SIMD_WIDTH> cvts(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  return cvts(a);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void store(T *const p, const SerialVec<T, SIMD_WIDTH> &v)
{
  memcpy(p, v.vec, SIMD_WIDTH);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void storeu(T *const p, const SerialVec<T, SIMD_WIDTH> &v)
{
  store(p, v);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void stream_store(T *const p,
                                     const SerialVec<T, SIMD_WIDTH> &v)
{
  store(p, v);
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE T extract(const SerialVec<T, SIMD_WIDTH> &a)
{
  return a[IMM];
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> add(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _add(a[i], b[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> adds(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _adds(a[i], b[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> sub(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _sub(a[i], b[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> subs(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _subs(a[i], b[i]);
  return c;
}

// no daz also on NEON
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> neg(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = -a[i];
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> mul(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _mul(a[i], b[i]);
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> div(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _div(a[i], b[i]);
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> ceil(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = std::ceil(daz_neon(a[i]));
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> floor(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = std::floor(daz_neon(a[i]));
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> round(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    // std::round has different behavior
    c[i] = ::rintf(a[i]);
  return c;
#elif defined(SIMDVEC_NEON_ENABLE)
  // I'm not sure about this one, this is at least the NEON
  // implementation of the workaround if vrndq is missing;
  // also not sure whether daz_neon() is required
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    // std::round has different behavior
    c[i] = std::floor(daz_neon(a[i]) + 0.5f);
  return c;
#else
#error "unknown architecture"
#endif
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> truncate(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = ::truncf(daz_neon(a[i]));
  return c;
}

// 02. Oct 22 (Jonas Keller): added rcp and rsqrt

// NOTE: hardware implementation is not known
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> rcp(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
#if defined(SIMDVEC_INTEL_ENABLE) &&                                           \
  (defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)) &&               \
  !defined(__AVX512F__)
    // sse and avx version of rcp treats denormals as zero, in the input
    // and output
    c[i] = daz(1.0f / daz(a[i]));
#else
    // TODO: is daz_neon() required here?
    c[i] = daz_neon(1.0f / daz_neon(a[i]));
#endif
  return c;
}

// NOTE: hardware implementation is not known
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> rsqrt(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
#if defined(SIMDVEC_INTEL_ENABLE) &&                                           \
  (defined(__SSE__) || defined(__AVX__) || defined(__AVX2__)) &&               \
  !defined(__AVX512F__)
    // sse and avx version of rsqrt treats denormals as zero, in the input
    // and output
    c[i] = daz(1.0f / std::sqrt(daz(a[i])));
#else
    // TODO: is daz_neon() required here?
    c[i] = daz_neon(1.0f / std::sqrt(daz_neon(a[i])));
#endif
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> sqrt(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = daz_neon(std::sqrt(daz_neon(a[i])));
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> min(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _min(a[i], b[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> max(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _max(a[i], b[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> abs(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = _abs(a[i]);
  return c;
}

// negate a, where b is negative (Float only)
// note: contrary to IEEE 754, this function considers -0.0f to be negative
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> sign(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  // 26. Aug 22 (Jonas Keller):
  // checked sign bit instead of comparison with 0.0f, since
  // IEEE 754 defines -0.0f as non-negative, making -0.0f < 0.0f false
  SerialVec<Float, SIMD_WIDTH> c;
  SerialVec<Int, SIMD_WIDTH> intB = reinterpret<Int>(b);
  for (int i = 0; i < c.elements; i++) {
    // examine sign bit
    c[i] = (intB[i] & 0x80000000) ? -a[i] : a[i];
    // c[i] = (b[i] < 0.0f) ? -a[i] : a[i];
  }
  return c;
}

// for signed integer types (SignedByte, Short, Int)
// which can overflow in absDiff

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<SignedByte, SIMD_WIDTH> absDiff(
  const SerialVec<SignedByte, SIMD_WIDTH> &a,
  const SerialVec<SignedByte, SIMD_WIDTH> &b)
{
  SerialVec<SignedByte, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = std::abs(SignedByte(a[i] - b[i]));
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Short, SIMD_WIDTH> absDiff(
  const SerialVec<Short, SIMD_WIDTH> &a, const SerialVec<Short, SIMD_WIDTH> &b)
{
  SerialVec<Short, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = std::abs(Short(a[i] - b[i]));
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Int, SIMD_WIDTH> absDiff(
  const SerialVec<Int, SIMD_WIDTH> &a, const SerialVec<Int, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = std::abs(Int(a[i] - b[i]));
  return c;
}

// for all other types (Byte, Word, Float) which can't
// overflow in absDiff
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Byte, SIMD_WIDTH> absDiff(
  const SerialVec<Byte, SIMD_WIDTH> &a, const SerialVec<Byte, SIMD_WIDTH> &b)
{
  SerialVec<Byte, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Word, SIMD_WIDTH> absDiff(
  const SerialVec<Word, SIMD_WIDTH> &a, const SerialVec<Word, SIMD_WIDTH> &b)
{
  SerialVec<Word, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> absDiff(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) {
    // 29. Nov 22 (Jonas Keller): applied daz_neon to inputs and result
    Float a_i = daz_neon(a[i]);
    Float b_i = daz_neon(b[i]);
    c[i]      = daz_neon((a_i > b_i) ? (a_i - b_i) : (b_i - a_i));
  }
  return c;
}

template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> unpack(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  int i0 = PART * c.elements / 2, ie = i0 + c.elements / 2;
  for (int i = i0, k = 0; i < ie; i += NUM_ELEMS) {
    for (int j = 0; j < NUM_ELEMS; j++, k++) c[k] = a[i + j];
    for (int j = 0; j < NUM_ELEMS; j++, k++) c[k] = b[i + j];
  }
  return c;
}

// 10. Oct 22 (Jonas Keller): added unpack16
// 128-bit-lane oriented unpack
template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> unpack16(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  constexpr int NUM_LANES = SIMD_WIDTH / 16;
  SerialVec<T, 16> lanesA[NUM_LANES];
  SerialVec<T, 16> lanesB[NUM_LANES];
  SerialVec<T, 16> lanesC[NUM_LANES];
  // copy a and b into lanesA and lanesB
  for (int i = 0; i < NUM_LANES; i++) {
    memcpy(lanesA[i].vec, &a.vec[i * (16 / sizeof(T))], 16);
    memcpy(lanesB[i].vec, &b.vec[i * (16 / sizeof(T))], 16);
  }
  // apply normal unpack to each lane
  for (int i = 0; i < NUM_LANES; i++) {
    lanesC[i] = unpack<PART, NUM_ELEMS>(lanesA[i], lanesB[i]);
  }
  // copy lanesC into c
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < NUM_LANES; i++) {
    memcpy(&c.vec[i * (16 / sizeof(T))], lanesC[i].vec, 16);
  }
  return c;
}

// 10. Oct 22 (Jonas Keller): added extractLane
// extract a 128-bit lane
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, 16> extractLane(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, 16> c;
  memcpy(c.vec, &a.vec[IMM * (16 / sizeof(T))], 16);
  return c;
}

template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
static SIMD_INLINE void zip(const SerialVec<T, SIMD_WIDTH> a,
                            const SerialVec<T, SIMD_WIDTH> b,
                            SerialVec<T, SIMD_WIDTH> &c,
                            SerialVec<T, SIMD_WIDTH> &d)
{
  c = unpack<0, NUM_ELEMS>(a, b);
  d = unpack<1, NUM_ELEMS>(a, b);
}

// 10. Oct 22 (Jonas Keller): added zip16
// 128-bit-lane oriented zip
template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
static SIMD_INLINE void zip16(const SerialVec<T, SIMD_WIDTH> a,
                              const SerialVec<T, SIMD_WIDTH> b,
                              SerialVec<T, SIMD_WIDTH> &c,
                              SerialVec<T, SIMD_WIDTH> &d)
{
  c = unpack16<0, NUM_ELEMS>(a, b);
  d = unpack16<1, NUM_ELEMS>(a, b);
}

template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
static SIMD_INLINE void unzip(const SerialVec<T, SIMD_WIDTH> a,
                              const SerialVec<T, SIMD_WIDTH> b,
                              SerialVec<T, SIMD_WIDTH> &c,
                              SerialVec<T, SIMD_WIDTH> &d)
{
  const int n = a.elements, n2 = n / 2;
  for (int i = 0, k = 0; i < n; i += NUM_ELEMS) {
    for (int j = 0; j < NUM_ELEMS; j++, k++) {
      c[k] = (i < n2) ? a[2 * i + j] : b[2 * (i - n2) + j];
      d[k] =
        (i < n2) ? a[2 * i + NUM_ELEMS + j] : b[2 * (i - n2) + NUM_ELEMS + j];
    }
  }
}

// from integer types
template <typename Tout, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Tout, SIMD_WIDTH> packs(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<Tout, SIMD_WIDTH> c;
  int ce = c.elements, ce2 = ce / 2;
  for (int i = 0; i < ce; i++) {
    T in = (i < ce2) ? a[i] : b[i - ce2];
    c[i] = _packs<Tout>(in);
  }
  return c;
}

// from float types
template <typename Tout, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Tout, SIMD_WIDTH> packs(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> ai, bi;
  ai = cvts(a);
  bi = cvts(b);
  return packs<Tout>(ai, bi);
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void extend(
  const SerialVec<Tin, SIMD_WIDTH> &vIn,
  SerialVec<Tout, SIMD_WIDTH> vOut[sizeof(Tout) / sizeof(Tin)])
{
  int nOut = SerialVec<Tout, SIMD_WIDTH>::elements;
  for (int i = 0; i < vIn.elements; i++)
    vOut[i / nOut][i % nOut] = Tout(vIn[i]);
}

// float -> int (uses saturating cvts)
template <int SIMD_WIDTH>
static SIMD_INLINE void extend(const SerialVec<Float, SIMD_WIDTH> &vIn,
                               SerialVec<Int, SIMD_WIDTH> vOut[1])
{
  vOut[0] = cvts(vIn);
}

// 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> sra(
  const SerialVec<T, SIMD_WIDTH> &a, const uint8_t count)
{
  SerialVec<T, SIMD_WIDTH> c;
  // operator >> on signed types is *usually* an arithmetic right
  // shift, but this is not guaranteed, so we use a workaround
  typedef typename TypeInfo<T>::UnsignedType U;
  U signBitMask = U(1) << (sizeof(T) * 8 - 1);
  for (int i = 0; i < c.elements; i++) {
    U result = U(a[i]);
    if (count >= sizeof(T) * 8) {
      result = (U(a[i]) & signBitMask) != 0 ? ~U(0) : 0;
    } else {
      for (int s = 0; s < count; s++) {
        result = (result >> 1) | (result & signBitMask);
      }
    }
    c[i] = T(result);
  }
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> srl(
  const SerialVec<T, SIMD_WIDTH> &a, const uint8_t count)
{
  SerialVec<T, SIMD_WIDTH> c;
  // on signed types, we force the compiler to use an unsigned shift
  typedef typename TypeInfo<T>::UnsignedType U;
  for (int i = 0; i < c.elements; i++) {
    if (count >= sizeof(T) * 8) {
      c[i] = 0;
    } else {
      c[i] = T(U(a[i]) >> count);
    }
  }
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> sll(
  const SerialVec<T, SIMD_WIDTH> &a, const uint8_t count)
{
  SerialVec<T, SIMD_WIDTH> c;
  // on signed types, we force the compiler to use an unsigned shift
  typedef typename TypeInfo<T>::UnsignedType U;
  for (int i = 0; i < c.elements; i++) {
    if (count >= sizeof(T) * 8) {
      c[i] = 0;
    } else {
      c[i] = T(U(a[i]) << count);
    }
  }
  return c;
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> srai(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  return sra(a, IMM);
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> srli(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  return srl(a, IMM);
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> slli(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  return sll(a, IMM);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> hadd(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  int i = 0, n = c.elements;
  for (; i < n / 2; i++) c[i] = _add(a[2 * i], a[2 * i + 1]);
  for (; i < n; i++) c[i] = _add(b[2 * i - n], b[2 * i - n + 1]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> hadds(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  int i = 0, n = c.elements;
  for (; i < n / 2; i++) c[i] = _adds(a[2 * i], a[2 * i + 1]);
  for (; i < n; i++) c[i] = _adds(b[2 * i - n], b[2 * i - n + 1]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> hsub(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  int i = 0, n = c.elements;
  for (; i < n / 2; i++) c[i] = _sub(a[2 * i], a[2 * i + 1]);
  for (; i < n; i++) c[i] = _sub(b[2 * i - n], b[2 * i - n + 1]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> hsubs(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  int i = 0, n = c.elements;
  for (; i < n / 2; i++) c[i] = _subs(a[2 * i], a[2 * i + 1]);
  for (; i < n; i++) c[i] = _subs(b[2 * i - n], b[2 * i - n + 1]);
  return c;
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> srle(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = a[i + IMM];
  return c;
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> slle(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = a[i - IMM];
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE T elem0(const SerialVec<T, SIMD_WIDTH> &a)
{
  return a[0];
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> alignre(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  int n = c.elements;
  for (int i = 0; i < n; i++) {
    int k = i + IMM;
    c[i]  = (k >= n) ? a[k - n] : b[k];
  }
  return c;
}

template <int N, typename T, int SIMD_WIDTH>
static SIMD_INLINE void swizzle(SerialVec<T, SIMD_WIDTH> v[N])
{
  const int n = SerialVec<T, SIMD_WIDTH>::elements;
  SerialVec<T, SIMD_WIDTH> u[N];
  // copy the N input vectors
  for (int i = 0; i < N; i++) u[i] = v[i];
  // swizzle: j, js in [0,n*N)
  for (int j = 0; j < n * N; j++) {
    int js            = (j % N) * n + std::floor(j / N);
    v[js / n][js % n] = u[j / n][j % n];
  }
}

// 15. Oct 22 (Jonas Keller): renamed _swizzle2 to swizzle2
template <int N, typename T, int SIMD_WIDTH>
static SIMD_INLINE void swizzle2(SerialVec<T, SIMD_WIDTH> v[2 * N])
{
  const int n = SerialVec<T, SIMD_WIDTH>::elements, n2 = 2 * n;
  SerialVec<T, SIMD_WIDTH> u[2 * N];
  // copy the 2*N input vectors
  for (int i = 0; i < 2 * N; i++) u[i] = v[i];
  // swizzle: j, js in [0,2*n*N), copy: [js] <- [j]
  for (int j = 0; j < n2 * N; j++) {
    int js            = (j % N) * n2 + std::floor(j / N);
    v[js / n][js % n] = u[j / n][j % n];
  }
}

// 15. Oct 22 (Jonas Keller): renamed _unswizzle2 to unswizzle2
template <int N, typename T, int SIMD_WIDTH>
static SIMD_INLINE void unswizzle2(SerialVec<T, SIMD_WIDTH> v[2 * N])
{
  const int n = SerialVec<T, SIMD_WIDTH>::elements, n2 = 2 * n;
  SerialVec<T, SIMD_WIDTH> u[2 * N];
  // copy the 2*N input vectors
  for (int i = 0; i < 2 * N; i++) u[i] = v[i];
  // swizzle: j, js in [0,2*n*N), copy: [j] <- [js]
  for (int j = 0; j < n2 * N; j++) {
    int js          = (j % N) * n2 + std::floor(j / N);
    v[j / n][j % n] = u[js / n][js % n];
  }
}

// 07. Feb 23 (Jonas Keller): added swizzle3
template <int N, typename T, int SIMD_WIDTH>
static SIMD_INLINE void swizzle3(SerialVec<T, SIMD_WIDTH> v[2 * N])
{
  // does the same thing as swizzle2
  swizzle2<N>(v);
}

// 15. Oct 22 (Jonas Keller): added swizzle4
template <int N, typename T, int SIMD_WIDTH>
static SIMD_INLINE void swizzle4(SerialVec<T, SIMD_WIDTH> v[2 * N])
{
  // does the same thing as swizzle2
  swizzle2<N>(v);
}

// 15. Oct 22 (Jonas Keller): added unswizzle4
template <int N, typename T, int SIMD_WIDTH>
static SIMD_INLINE void unswizzle4(SerialVec<T, SIMD_WIDTH> v[2 * N])
{
  // does the same thing as unswizzle2
  unswizzle2<N>(v);
}

template <typename Tcond, typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> ifelse(
  const SerialVec<Tcond, SIMD_WIDTH> &cond,
  const SerialVec<T, SIMD_WIDTH> &trueVal,
  const SerialVec<T, SIMD_WIDTH> &falseVal)
{
  SIMD_STATIC_ASSERT(sizeof(Tcond) == sizeof(T));
  SerialVec<T, SIMD_WIDTH> res;
  // avoids comparison with TypeInfo::trueval, which fails for
  // Float (NaN)
  SerialVec<Tcond, SIMD_WIDTH> negCond = not_(cond);
  for (int i = 0; i < res.elements; i++)
    if (negCond[i] == Tcond(0))
      res[i] = trueVal[i];
    else if (cond[i] == Tcond(0))
      res[i] = falseVal[i];
    else {
      fprintf(stderr, "ifelse: invalid condition element\n");
      exit(-1);
    }
  return res;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> cmplt(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (a[i] < b[i]) ? TypeInfo<T>::trueval() : 0;
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> cmple(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (a[i] <= b[i]) ? TypeInfo<T>::trueval() : 0;
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> cmpeq(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (daz_neon(a[i]) == daz_neon(b[i])) ? TypeInfo<T>::trueval() : 0;
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> cmpge(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (a[i] >= b[i]) ? TypeInfo<T>::trueval() : 0;
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> cmpgt(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (a[i] > b[i]) ? TypeInfo<T>::trueval() : 0;
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> cmpneq(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (daz_neon(a[i]) != daz_neon(b[i])) ? TypeInfo<T>::trueval() : 0;
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> and_(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = a[i] & b[i];
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> and_(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> ai(a), bi(b);
  return SerialVec<Float, SIMD_WIDTH>(and_(ai, bi));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> or_(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = a[i] | b[i];
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> or_(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> ai(a), bi(b);
  return SerialVec<Float, SIMD_WIDTH>(or_(ai, bi));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> andnot(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = ~a[i] & b[i];
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> andnot(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> ai(a), bi(b);
  return SerialVec<Float, SIMD_WIDTH>(andnot(ai, bi));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> xor_(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = a[i] ^ b[i];
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> xor_(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> ai(a), bi(b);
  return SerialVec<Float, SIMD_WIDTH>(xor_(ai, bi));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> not_(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = ~a[i];
  return c;
}

template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> not_(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Int, SIMD_WIDTH> ai(a);
  return SerialVec<Float, SIMD_WIDTH>(not_(ai));
}

// int types except Int
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> avg(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = (Int(a[i]) + Int(b[i]) + 1) >> 1;
  return c;
}

// Int
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Int, SIMD_WIDTH> avg(
  const SerialVec<Int, SIMD_WIDTH> &a, const SerialVec<Int, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (uint64_t(a[i]) + uint64_t(b[i]) + 1) >> 1;
  return c;
}

// Float
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> avg(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = daz_neon(0.5f * (daz_neon(a[i]) + daz_neon(b[i])));
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int test_all_zeros(const SerialVec<T, SIMD_WIDTH> &a)
{
  for (int i = 0; i < a.elements; i++)
    if (a[i] != T(0)) return 0;
  return 1;
}

// 26. Aug 22 (Jonas Keller):
// added float version of test_all_zeros since the generic version
// considers -0.0f to be equal to 0.0f (in accordance with IEEE 754),
// but we want to treat -0.0f as non-zero

// note: contrary to IEEE 754, this function considers -0.0f to be negative
template <int SIMD_WIDTH>
static SIMD_INLINE int test_all_zeros(const SerialVec<Float, SIMD_WIDTH> &a)
{
  return test_all_zeros(reinterpret<Int>(a));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int test_all_ones(const SerialVec<T, SIMD_WIDTH> &a)
{
  // avoids comparison with TypeInfo::trueval, which fails for
  // Float (NaN)
  SerialVec<T, SIMD_WIDTH> aNot = not_(a);
  for (int i = 0; i < a.elements; i++)
    if (aNot[i] != T(0)) return 0;
  return 1;
}

// 26. Aug 22 (Jonas Keller):
// added float version of test_all_ones since the generic version
// considers -0.0f to be equal to 0.0f (in accordance with IEEE 754),
// but we want to treat -0.0f as non-zero

// note: contrary to IEEE 754, this function considers -0.0f to be negative
template <int SIMD_WIDTH>
static SIMD_INLINE int test_all_ones(const SerialVec<Float, SIMD_WIDTH> &a)
{
  return test_all_ones(reinterpret<Int>(a));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> reverse(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < a.elements; i++) c[i] = a[a.elements - 1 - i];
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE T hadd(const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c = a;
  for (int n = a.elements; n > 1; n = n / 2)
    for (int i = 0; i < n / 2; i++) c[i] = _add(c[2 * i], c[2 * i + 1]);
  return c[0];
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE T hadds(const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c = a;
  for (int n = a.elements; n > 1; n = n / 2)
    for (int i = 0; i < n / 2; i++) c[i] = _adds(c[2 * i], c[2 * i + 1]);
  return c[0];
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE T hmin(const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c = a;
  for (int n = a.elements; n > 1; n = n / 2)
    for (int i = 0; i < n / 2; i++) c[i] = _min(c[2 * i], c[2 * i + 1]);
  return c[0];
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE T hmax(const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c = a;
  for (int n = a.elements; n > 1; n = n / 2)
    for (int i = 0; i < n / 2; i++) c[i] = _max(c[2 * i], c[2 * i + 1]);
  return c[0];
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void convert(
  const SerialVec<Tin, SIMD_WIDTH> inVecs[numInVecs<Tout, Tin>()],
  SerialVec<Tout, SIMD_WIDTH> outVecs[numOutVecs<Tout, Tin>()])
{
  const int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  const int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  const int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;
  int outIdx            = 0;
  for (int inVec = 0; inVec < numInVecs; inVec++)
    for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
      int outVec               = outIdx / numOutElems;
      int outElem              = outIdx % numOutElems;
      outVecs[outVec][outElem] = _convert<Tout>(inVecs[inVec][inElem]);
    }
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void fdivmul(
  const SerialVec<Tin, SIMD_WIDTH> vecsNum[numInVecs<Tout, Tin>()],
  const SerialVec<Tin, SIMD_WIDTH> vecsDenom[numInVecs<Tout, Tin>()],
  double fac, SerialVec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  SIMD_STATIC_ASSERT(sizeof(Tin) <= sizeof(Float));
  SIMD_STATIC_ASSERT(sizeof(Tout) <= sizeof(Float));
  const int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  const int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  const int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;
  float facF            = fac;
  int outIdx            = 0;
  for (int inVec = 0; inVec < numInVecs; inVec++)
    for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
      int outVec  = outIdx / numOutElems;
      int outElem = outIdx % numOutElems;
      vecsOut[outVec][outElem] =
        _packs<Tout>(_mul(_div(_extend<Float>(vecsNum[inVec][inElem]),
                               _extend<Float>(vecsDenom[inVec][inElem])),
                          facF));
    }
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void fmul(
  const SerialVec<Tin, SIMD_WIDTH> vecsIn[numInVecs<Tout, Tin>()], double fac,
  SerialVec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  SIMD_STATIC_ASSERT(sizeof(Tin) <= sizeof(Float));
  SIMD_STATIC_ASSERT(sizeof(Tout) <= sizeof(Float));
  const int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  const int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  const int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;
  float facF            = fac;
  int outIdx            = 0;
  for (int inVec = 0; inVec < numInVecs; inVec++)
    for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
      int outVec  = outIdx / numOutElems;
      int outElem = outIdx % numOutElems;
      vecsOut[outVec][outElem] =
        _packs<Tout>(_mul(_extend<Float>(vecsIn[inVec][inElem]), facF));
    }
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void faddmul(
  const SerialVec<Tin, SIMD_WIDTH> vecsIn[numInVecs<Tout, Tin>()], double off,
  double fac, SerialVec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  SIMD_STATIC_ASSERT(sizeof(Tin) <= sizeof(Float));
  SIMD_STATIC_ASSERT(sizeof(Tout) <= sizeof(Float));
  const int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  const int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  const int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;
  float facF = fac, offF = off;
  int outIdx = 0;
  for (int inVec = 0; inVec < numInVecs; inVec++)
    for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
      int outVec               = outIdx / numOutElems;
      int outElem              = outIdx % numOutElems;
      vecsOut[outVec][outElem] = _packs<Tout>(
        _mul(_add(_extend<Float>(vecsIn[inVec][inElem]), offF), facF));
    }
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void fmuladd(
  const SerialVec<Tin, SIMD_WIDTH> vecsIn[numInVecs<Tout, Tin>()], double fac,
  double off, SerialVec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  SIMD_STATIC_ASSERT(sizeof(Tin) <= sizeof(Float));
  SIMD_STATIC_ASSERT(sizeof(Tout) <= sizeof(Float));
  const int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  const int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  const int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;
  float facF = fac, offF = off;
  int outIdx = 0;
  for (int inVec = 0; inVec < numInVecs; inVec++)
    for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
      int outVec               = outIdx / numOutElems;
      int outElem              = outIdx % numOutElems;
      vecsOut[outVec][outElem] = _packs<Tout>(
        _add(_mul(_extend<Float>(vecsIn[inVec][inElem]), facF), offF));
    }
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE void fwaddmul(
  const SerialVec<Tin, SIMD_WIDTH> vecsIn1[numInVecs<Tout, Tin>()],
  const SerialVec<Tin, SIMD_WIDTH> vecsIn2[numInVecs<Tout, Tin>()], double w,
  double fac, SerialVec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  SIMD_STATIC_ASSERT(sizeof(Tin) <= sizeof(Float));
  SIMD_STATIC_ASSERT(sizeof(Tout) <= sizeof(Float));
  const int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  const int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  const int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;
  float wF              = w;
  float facF            = fac;
  int outIdx            = 0;
  for (int inVec = 0; inVec < numInVecs; inVec++)
    for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
      int outVec  = outIdx / numOutElems;
      int outElem = outIdx % numOutElems;
      float inF1  = _extend<Float>(vecsIn1[inVec][inElem]);
      float inF2  = _extend<Float>(vecsIn2[inVec][inElem]);
      vecsOut[outVec][outElem] =
        _packs<Tout>(_mul(facF, _add(inF2, _mul(wF, _sub(inF1, inF2)))));
    }
}
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> hadd(
  const SerialVec<T, SIMD_WIDTH> v[SerialVec<T, SIMD_WIDTH>::elems])
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = hadd(v[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> hadds(
  const SerialVec<T, SIMD_WIDTH> v[SerialVec<T, SIMD_WIDTH>::elems])
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = hadds(v[i]);
  return c;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  const int n = SerialVec<T, SIMD_WIDTH>::elements;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) outRows[i][j] = inRows[j][i];
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose2(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose0(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose16(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose3(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose4(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose5(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose1_16_t(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void transpose6_t(
  const SerialVec<T, SIMD_WIDTH> inRows[SerialVec<T, SIMD_WIDTH>::elems],
  SerialVec<T, SIMD_WIDTH> outRows[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as transpose
  transpose(inRows, outRows);
}

template <SortSlope SLOPE, typename T, int SIMD_WIDTH>
static SIMD_INLINE void bitonicSort(
  SerialVec<T, SIMD_WIDTH> vecs[SerialVec<T, SIMD_WIDTH>::elems])
{
  const int n = SerialVec<T, SIMD_WIDTH>::elements;
  for (int i = 0; i < n; i++)
    if (SLOPE == SortSlope::ASCENDING)
      std::sort(vecs[i].vec, vecs[i].vec + n, std::less<T>());
    else
      std::sort(vecs[i].vec, vecs[i].vec + n, std::greater<T>());
}

template <SortSlope SLOPE, typename T, int SIMD_WIDTH>
static SIMD_INLINE void bitonicSort2(
  SerialVec<T, SIMD_WIDTH> vecs[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as bitonicSort
  bitonicSort<SLOPE>(vecs);
}

template <SortSlope SLOPE, typename T, int SIMD_WIDTH>
static SIMD_INLINE void bitonicSortSortedPairs(
  SerialVec<T, SIMD_WIDTH> vecs[SerialVec<T, SIMD_WIDTH>::elems])
{
  const int n = SerialVec<T, SIMD_WIDTH>::elements;
  T twovecs[2 * n];

  for (int i = 0; i < n; i += 2) {
    // join two consecutive vectors
    for (int j = 0; j < n; j++) {
      twovecs[j]     = vecs[i][j];
      twovecs[j + n] = vecs[i + 1][j];
    }
    // sort two vectors together
    if (SLOPE == SortSlope::ASCENDING)
      std::sort(twovecs, twovecs + 2 * n, std::less<T>());
    else
      std::sort(twovecs, twovecs + 2 * n, std::greater<T>());
    // split two consecutive vectors
    for (int j = 0; j < n; j++) {
      vecs[i][j]     = twovecs[j];
      vecs[i + 1][j] = twovecs[j + n];
    }
  }
}

template <SortSlope SLOPE, typename T, int SIMD_WIDTH>
static SIMD_INLINE void bitonicSortSortedPairs2(
  SerialVec<T, SIMD_WIDTH> vecs[SerialVec<T, SIMD_WIDTH>::elems])
{
  // same as bitonicSortSortedPairs
  bitonicSortSortedPairs<SLOPE>(vecs);
}

// synonymous to avg
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> avgru(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  return avg(a, b);
}

// int types except Int
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> avgrd(
  const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = (Int(a[i]) + Int(b[i])) >> 1;
  return c;
}

// Int
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Int, SIMD_WIDTH> avgrd(
  const SerialVec<Int, SIMD_WIDTH> &a, const SerialVec<Int, SIMD_WIDTH> &b)
{
  SerialVec<Int, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = (uint64_t(a[i]) + uint64_t(b[i])) >> 1;
  return c;
}

// Float
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> avgrd(
  const SerialVec<Float, SIMD_WIDTH> &a, const SerialVec<Float, SIMD_WIDTH> &b)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++)
    c[i] = daz_neon(0.5f * (daz_neon(a[i]) + daz_neon(b[i])));
  return c;
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Tout, SIMD_WIDTH> packs(
  const SerialVec<Tin, SIMD_WIDTH> a[sizeof(Tin) / sizeof(Tout)])
{
  SerialVec<Tout, SIMD_WIDTH> c;
  const int numInVecs = sizeof(Tin) / sizeof(Tout);
  int k               = 0;
  for (int i = 0; i < numInVecs; i++)
    for (int j = 0; j < SerialVec<Tin, SIMD_WIDTH>::elements; j++, k++) {
      Tin in = a[i][j];
      c[k]   = _packs<Tout>(in);
    }
  return c;
}

// integer types
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> div2r0(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = T(::trunc(double(a[i]) / 2.0));
  return c;
}

// Float
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> div2r0(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = daz_neon(daz_neon(a[i]) / 2.0f);
  return c;
}

// integer types
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SerialVec<T, SIMD_WIDTH> div2rd(
  const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = T(std::floor(double(a[i]) / 2.0));
  return c;
}

// Float
template <int SIMD_WIDTH>
static SIMD_INLINE SerialVec<Float, SIMD_WIDTH> div2rd(
  const SerialVec<Float, SIMD_WIDTH> &a)
{
  SerialVec<Float, SIMD_WIDTH> c;
  for (int i = 0; i < c.elements; i++) c[i] = daz_neon(daz_neon(a[i]) / 2.0f);
  return c;
}

// ---------------------------------------------------------------------------
// msb2int
// ---------------------------------------------------------------------------

// 27. Aug 22 (Jonas Keller): added msb2int functions

// integer types
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE uint64_t msb2int(const SerialVec<T, SIMD_WIDTH> &a)
{
  T highestBitMask = T(T(1) << (sizeof(T) * 8 - 1));
  uint64_t result  = 0;
  for (int i = 0; i < a.elements; i++)
    if ((a[i] & highestBitMask) != 0) result |= (uint64_t(1) << i);
  return result;
}

// Float
template <int SIMD_WIDTH>
static SIMD_INLINE uint64_t msb2int(const SerialVec<Float, SIMD_WIDTH> &a)
{
  return msb2int(reinterpret<Int>(a));
}

} // namespace auto_test
} // namespace simd

#endif // _SIMD_VEC_AUTO_TEST_SERIAL_H_
