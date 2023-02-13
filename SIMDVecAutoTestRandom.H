#ifndef _SIMD_VEC_AUTO_TEST_RANDOM_H_
#define _SIMD_VEC_AUTO_TEST_RANDOM_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"

#include <cmath>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// 16. Nov 22 (Jonas Keller): moved functions for generating random numbers
// from SIMDVecAutoTestSerial.H to this file and added functions for generating
// random numbers where special values have higher probability

namespace simd {
namespace auto_test {

template <typename T>
static SIMD_INLINE T getRandom();

template <>
SIMD_INLINE Byte getRandom<Byte>()
{
  return rand() & 0xff;
}

template <>
SIMD_INLINE SignedByte getRandom<SignedByte>()
{
  return rand() & 0xff;
}

template <>
SIMD_INLINE Word getRandom<Word>()
{
  Byte b[2];
  b[0] = getRandom<Byte>();
  b[1] = getRandom<Byte>();
  Word w;
  memcpy(&w, b, 2);
  return w;
}

template <>
SIMD_INLINE Short getRandom<Short>()
{
  Byte b[2];
  b[0] = getRandom<Byte>();
  b[1] = getRandom<Byte>();
  Short s;
  memcpy(&s, b, 2);
  return s;
}

template <>
SIMD_INLINE Int getRandom<Int>()
{
  Word w[2];
  w[0] = getRandom<Word>();
  w[1] = getRandom<Word>();
  Int i;
  memcpy(&i, w, 4);
  return i;
}

// we only generate finite floats, since the behavior e.g. in SIMD
// min/max differs from std::min/max for NaN (maybe also for
// infinite)
template <>
SIMD_INLINE Float getRandom<Float>()
{
  Int i;
  Float f;
  int exponent;
  // restricted selection means: only accept restricted exponent range
  bool restricted = (rand() & 0x01);
  do {
    // get random Int
    i = getRandom<Int>();
    // map to float
    memcpy(&f, &i, 4);
    // determine exponent
    frexpf(f, &exponent);
    // 26. Nov 17 (rm): isfinite -> isnormal
    // 28. Nov 17 (rm): isnormal -> isfinite (now daz handling for NEON)
  } while (!(std::isfinite(f) &&
             // base2-exponent < 23 means: number has fractional digits
             (!restricted || (exponent < 23))));
  return f;
}

// 09. Oct 22 (Jonas Keller): added getRandom for uint64_t
template <>
SIMD_INLINE uint64_t getRandom<uint64_t>()
{
  uint64_t i;
  Int s[2];
  s[0] = getRandom<Int>();
  s[1] = getRandom<Int>();
  memcpy(&i, s, 8);
  return i;
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
  Int i = (getRandom<Int>() & 0x807fffff) | 0x3f800000;
  // f12 is a float from [1,2)
  Float f12;
  memcpy(&f12, &i, 4);
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
  Byte v[]       = {0, 1, SIMDBYTE_MAX / 2, SIMDBYTE_MAX - 1, SIMDBYTE_MAX};
  const size_t n = sizeof(v) / sizeof(Byte);
  return std::vector<Byte>(v, v + n);
}

template <>
std::vector<SignedByte> specialValues<SignedByte>()
{
  SignedByte v[] = {SIMDSIGNEDBYTE_MIN,
                    SIMDSIGNEDBYTE_MIN + 1,
                    SIMDSIGNEDBYTE_MIN / 2,
                    -1,
                    0,
                    1,
                    SIMDSIGNEDBYTE_MAX / 2,
                    SIMDSIGNEDBYTE_MAX - 1,
                    SIMDSIGNEDBYTE_MAX};
  const size_t n = sizeof(v) / sizeof(SignedByte);
  return std::vector<SignedByte>(v, v + n);
}

template <>
std::vector<Word> specialValues<Word>()
{
  Word v[]       = {0, 1, SIMDWORD_MAX / 2, SIMDWORD_MAX - 1, SIMDWORD_MAX};
  const size_t n = sizeof(v) / sizeof(Word);
  return std::vector<Word>(v, v + n);
}

template <>
std::vector<Short> specialValues<Short>()
{
  Short v[] = {
    SIMDSHORT_MIN,     SIMDSHORT_MIN + 1, SIMDSHORT_MIN / 2, -1, 0, 1,
    SIMDSHORT_MAX / 2, SIMDSHORT_MAX - 1, SIMDSHORT_MAX};
  const size_t n = sizeof(v) / sizeof(Short);
  return std::vector<Short>(v, v + n);
}

template <>
std::vector<Int> specialValues<Int>()
{
  Int v[] = {SIMDINT_MIN, SIMDINT_MIN + 1, SIMDINT_MIN / 2, -1,         0,
             1,           SIMDINT_MAX / 2, SIMDINT_MAX - 1, SIMDINT_MAX};
  const size_t n = sizeof(v) / sizeof(Int);
  return std::vector<Int>(v, v + n);
}

template <>
std::vector<Float> specialValues<Float>()
{
  Float v[]      = {SIMDFLOAT_MIN,
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
  const size_t n = sizeof(v) / sizeof(Float);
  return std::vector<Float>(v, v + n);
}
} // namespace internal

template <typename T>
static T getRandomSpecialValue()
{
  std::vector<T> specialValues = internal::specialValues<T>();
  return specialValues[((double) rand() / RAND_MAX) * specialValues.size()];
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
SIMD_INLINE Float getRandomOrSpecialValueRanges<Float>()
{
  // generate 23 bit mantissa and sign bit (with zero exponent, c = 127)
  Int i = (getRandomOrSpecialValue<Int>() & 0x807fffff) | 0x3f800000;
  // f12 is a float from [1,2)
  Float f12;
  memcpy(&f12, &i, 4);
  // generate random exponent from [0,31]
  if (rand() & 0x01) return ldexpf(f12, rand() & 0x1f);
  // or just return completely random float
  return getRandomOrSpecialValue<Float>();
}
} // namespace auto_test
} // namespace simd

#endif // _SIMD_VEC_AUTO_TEST_RANDOM_H_
