#pragma once
#ifndef SIMD_VEC_AUTO_TEST_RANDOM_H_
#define SIMD_VEC_AUTO_TEST_RANDOM_H_

#include <tsimd.H>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// 16. Nov 22 (Jonas Keller): moved functions for generating random numbers
// from SIMDVecAutoTestSerial.H to this file and added functions for generating
// random numbers where special values have higher probability

// 22. Jan 23 (Jonas Keller): moved auto test related stuff into auto_test
// namespace

// 13. May 23 (Jonas Keller): added Double versions of random functions

namespace simd {
namespace auto_test {

template <typename T>
static SIMD_INLINE T getRandom()
{
  std::array<Byte, sizeof(T)> bytes;
  for (size_t i = 0; i < sizeof(T); ++i) { bytes[i] = std::rand() & 0xff; }
  return internal::bit_cast<T>(bytes);
}

// we only generate finite floats, since the behavior e.g. in SIMD
// min/max differs from std::min/max for NaN (maybe also for
// infinite)
template <>
SIMD_INLINE Float getRandom<Float>()
{
  // restricted selection means: only accept restricted exponent range
  bool restricted = (rand() & 0x01);
  int exponent;
  Float f;
  do {
    f = internal::bit_cast<Float>(getRandom<Int>());
    // determine exponent
    frexpf(f, &exponent);
    // 26. Nov 17 (rm): isfinite -> isnormal
    // 28. Nov 17 (rm): isnormal -> isfinite (now daz handling for NEON)
  } while (!(std::isfinite(f) &&
             // base2-exponent < 23 means: number has fractional digits
             (!restricted || (exponent < 23))));
  return f;
}

template <>
SIMD_INLINE Double getRandom<Double>()
{
  // restricted selection means: only accept restricted exponent range
  bool restricted = (rand() & 0x01);
  int exponent;
  Double d;
  do {
    d = internal::bit_cast<Double>(getRandom<Long>());
    // determine exponent
    frexp(d, &exponent);
  } while (!(std::isfinite(d) &&
             // base2-exponent < 52 means: number has fractional digits
             (!restricted || (exponent < 52))));
  return d;
}

// =========================================================================
// random numbers in different ranges
// =========================================================================

template <typename T>
static SIMD_INLINE T getRandomRanges();

template <>
SIMD_INLINE Byte getRandomRanges<Byte>()
{
  return getRandom<Byte>();
}

template <>
SIMD_INLINE SignedByte getRandomRanges<SignedByte>()
{
  return getRandom<SignedByte>();
}

template <>
SIMD_INLINE Word getRandomRanges<Word>()
{
  if (rand() & 0x01) return getRandomRanges<Byte>();
  return getRandom<Word>();
}

template <>
SIMD_INLINE Short getRandomRanges<Short>()
{
  if (rand() & 0x01) return getRandomRanges<SignedByte>();
  return getRandom<Short>();
}

template <>
SIMD_INLINE Int getRandomRanges<Int>()
{
  if (rand() & 0x01) return getRandomRanges<Short>();
  return getRandom<Int>();
}

template <>
SIMD_INLINE Float getRandomRanges<Float>()
{
  // generate 23 bit mantissa and sign bit (with zero exponent, c = 127)
  const uint32_t i = (getRandom<uint32_t>() & 0x807fffff) | 0x3f800000;
  // f12 is a float from [1,2)
  const Float f12 = internal::bit_cast<Float>(i);
  // generate random exponent from [0,31]
  if (rand() & 0x01) return ldexpf(f12, rand() & 0x1f);
  // or just return completely random float
  return getRandom<Float>();
}

// =========================================================================
// random special value
// =========================================================================

namespace internal {
template <typename T>
static std::vector<T> specialValues();

template <>
std::vector<Byte> specialValues<Byte>()
{
  return {0, 1, SIMDBYTE_MAX / 2, SIMDBYTE_MAX - 1, SIMDBYTE_MAX};
}

template <>
std::vector<SignedByte> specialValues<SignedByte>()
{
  return {SIMDSIGNEDBYTE_MIN,
          SIMDSIGNEDBYTE_MIN + 1,
          SIMDSIGNEDBYTE_MIN / 2,
          -1,
          0,
          1,
          SIMDSIGNEDBYTE_MAX / 2,
          SIMDSIGNEDBYTE_MAX - 1,
          SIMDSIGNEDBYTE_MAX};
}

template <>
std::vector<Word> specialValues<Word>()
{
  return {0, 1, SIMDWORD_MAX / 2, SIMDWORD_MAX - 1, SIMDWORD_MAX};
}

template <>
std::vector<Short> specialValues<Short>()
{
  return {SIMDSHORT_MIN,     SIMDSHORT_MIN + 1, SIMDSHORT_MIN / 2, -1, 0, 1,
          SIMDSHORT_MAX / 2, SIMDSHORT_MAX - 1, SIMDSHORT_MAX};
}

template <>
std::vector<Int> specialValues<Int>()
{
  return {SIMDINT_MIN, SIMDINT_MIN + 1, SIMDINT_MIN / 2, -1,         0,
          1,           SIMDINT_MAX / 2, SIMDINT_MAX - 1, SIMDINT_MAX};
}

template <>
std::vector<Long> specialValues<Long>()
{
  return {SIMDLONG_MIN,     SIMDLONG_MIN + 1, SIMDLONG_MIN / 2, -1, 0, 1,
          SIMDLONG_MAX / 2, SIMDLONG_MAX - 1, SIMDLONG_MAX};
}

template <>
std::vector<Float> specialValues<Float>()
{
  return {SIMDFLOAT_MIN,
          SIMDFLOAT_MIN / 2,
          SIMDFLOAT_MAX / 2,
          SIMDFLOAT_MAX,
          -1.0f,
          -0.5f,
          -0.0f,
          0.0f,
          0.5f,
          1.0f,
          MAX_POS_FLOAT_CONVERTIBLE_TO_INT32,
          MAX_POS_FLOAT_CONVERTIBLE_TO_INT32 + 1};
}

template <>
std::vector<Double> specialValues<Double>()
{
  return {SIMDDOUBLE_MIN,
          SIMDDOUBLE_MIN / 2,
          SIMDDOUBLE_MAX / 2,
          SIMDDOUBLE_MAX,
          -1.0,
          -0.5,
          -0.0,
          0.0,
          0.5,
          1.0};
}

template <>
std::vector<uint64_t> specialValues<uint64_t>()
{
  constexpr auto max = std::numeric_limits<uint64_t>::max();
  return {0, 1, max / 2, max - 1, max};
}

template <>
std::vector<uint32_t> specialValues<uint32_t>()
{
  constexpr auto max = std::numeric_limits<uint32_t>::max();
  return {0, 1, max / 2, max - 1, max};
}
} // namespace internal

template <typename T>
static T getRandomSpecialValue()
{
  std::vector<T> specialValues = internal::specialValues<T>();
  return specialValues[static_cast<size_t>(std::rand()) % specialValues.size()];
}

// returns a special value with 1/16 probability and a random value otherwise
template <typename T>
static T getRandomOrSpecialValue()
{
  if (rand() & 0x0f)
    return getRandom<T>();
  else
    return getRandomSpecialValue<T>();
}

// =========================================================================
// random or special numbers in different ranges
// =========================================================================

template <typename T>
static SIMD_INLINE T getRandomOrSpecialValueRanges();

template <>
SIMD_INLINE Byte getRandomOrSpecialValueRanges<Byte>()
{
  return getRandomOrSpecialValue<Byte>();
}

template <>
SIMD_INLINE SignedByte getRandomOrSpecialValueRanges<SignedByte>()
{
  return getRandomOrSpecialValue<SignedByte>();
}

template <>
SIMD_INLINE Word getRandomOrSpecialValueRanges<Word>()
{
  if (rand() & 0x01) return getRandomOrSpecialValueRanges<Byte>();
  return getRandomOrSpecialValue<Word>();
}

template <>
SIMD_INLINE Short getRandomOrSpecialValueRanges<Short>()
{
  if (rand() & 0x01) return getRandomOrSpecialValueRanges<SignedByte>();
  return getRandomOrSpecialValue<Short>();
}

template <>
SIMD_INLINE Int getRandomOrSpecialValueRanges<Int>()
{
  if (rand() & 0x01) return getRandomOrSpecialValueRanges<Short>();
  return getRandomOrSpecialValue<Int>();
}

template <>
SIMD_INLINE Long getRandomOrSpecialValueRanges<Long>()
{
  if (rand() & 0x01) return getRandomOrSpecialValueRanges<Int>();
  return getRandomOrSpecialValue<Long>();
}

template <>
SIMD_INLINE Float getRandomOrSpecialValueRanges<Float>()
{
  // generate 23 bit mantissa and sign bit (with zero exponent, c = 127)
  const uint32_t i =
    (getRandomOrSpecialValue<uint32_t>() & 0x807fffff) | 0x3f800000;
  // f12 is a float from [1,2)
  const Float f12 = ::simd::internal::bit_cast<Float>(i);
  // generate random exponent from [0,31]
  if (rand() & 0x01) return ldexpf(f12, rand() & 0x1f);
  // or just return completely random float
  return getRandomOrSpecialValue<Float>();
}

template <>
SIMD_INLINE Double getRandomOrSpecialValueRanges<Double>()
{
  // generate 52 bit mantissa and sign bit (with zero exponent, c = 1023)
  const uint64_t i =
    (getRandomOrSpecialValue<uint64_t>() & 0x800fffffffffffffLL) |
    0x3ff0000000000000LL;
  // f12 is a double from [1,2)
  const Double f12 = ::simd::internal::bit_cast<Double>(i);
  // generate random exponent from [0,1023]
  if (rand() & 0x01) return ldexp(f12, rand() & 0x3ff);
  // or just return completely random double
  return getRandomOrSpecialValue<Double>();
}
} // namespace auto_test
} // namespace simd

#endif // SIMD_VEC_AUTO_TEST_RANDOM_H_
