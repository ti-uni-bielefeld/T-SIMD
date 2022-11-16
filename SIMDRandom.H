#ifndef _SIMD_RANDOM_H_
#define _SIMD_RANDOM_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"

#include <cmath>
#include <random>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// 16. Nov 22 (Jonas Keller): moved functions for generating random numbers
// from SIMDSerialVec.H to this file and added functions for generating
// random numbers where special values have higher probability

namespace ns_simd {

  template <typename T>
  static SIMD_INLINE T
  getRandom();
  
  template <>
  SIMD_INLINE SIMDByte
  getRandom<SIMDByte>()
  {
    return rand() & 0xff;
  }

  template <>
  SIMD_INLINE SIMDSignedByte
  getRandom<SIMDSignedByte>()
  {
    return rand() & 0xff;
  }

  template <>
  SIMD_INLINE SIMDWord
  getRandom<SIMDWord>()
  {
    SIMDByte b[2];
    b[0] = getRandom<SIMDByte>();
    b[1] = getRandom<SIMDByte>();
    SIMDWord w;
    memcpy(&w, b, 2);
    return w;
  }

  template <>
  SIMD_INLINE SIMDShort
  getRandom<SIMDShort>()
  {
    SIMDByte b[2];
    b[0] = getRandom<SIMDByte>();
    b[1] = getRandom<SIMDByte>();
    SIMDShort s;
    memcpy(&s, b, 2);
    return s;
  }

  template <>
  SIMD_INLINE SIMDInt
  getRandom<SIMDInt>()
  {
    SIMDWord w[2];
    w[0] = getRandom<SIMDWord>();
    w[1] = getRandom<SIMDWord>();
    SIMDInt i;
    memcpy(&i, w, 4);
    return i;
  }

  // we only generate finite floats, since the behavior e.g. in SIMD
  // min/max differs from std::min/max for NaN (maybe also for
  // infinite)
  template <>
  SIMD_INLINE SIMDFloat
  getRandom<SIMDFloat>()
  {
    SIMDInt i;
    SIMDFloat f;
    int exponent;
    // restricted selection means: only accept restricted exponent range
    bool restricted = (rand() & 0x01);
    do {
      // get random SIMDInt
      i = getRandom<SIMDInt>();
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
  SIMD_INLINE uint64_t
  getRandom<uint64_t>()
  {
    uint64_t i;
    SIMDInt s[2];
    s[0] = getRandom<SIMDInt>();
    s[1] = getRandom<SIMDInt>();
    memcpy(&i, s, 8);
    return i;
  }

  // =========================================================================
  // random numbers in different ranges
  // =========================================================================

  template <typename T>
  static SIMD_INLINE T
  getRandomRanges();

  template <>
  SIMD_INLINE SIMDByte
  getRandomRanges<SIMDByte>()
  {
    return getRandom<SIMDByte>();
  }

  template <>
  SIMD_INLINE SIMDSignedByte
  getRandomRanges<SIMDSignedByte>()
  {
    return getRandom<SIMDSignedByte>();
  }

  template <>
  SIMD_INLINE SIMDWord
  getRandomRanges<SIMDWord>()
  {
    if (rand() & 0x01) return getRandomRanges<SIMDByte>();
    return getRandom<SIMDWord>();
  }

  template <>
  SIMD_INLINE SIMDShort
  getRandomRanges<SIMDShort>()
  {
    if (rand() & 0x01) return getRandomRanges<SIMDSignedByte>();
    return getRandom<SIMDShort>();
  }

  template <>
  SIMD_INLINE SIMDInt
  getRandomRanges<SIMDInt>()
  {
    if (rand() & 0x01) return getRandomRanges<SIMDShort>();
    return getRandom<SIMDInt>();
  }

  template <>
  SIMD_INLINE SIMDFloat
  getRandomRanges<SIMDFloat>()
  {
    // generate 23 bit mantissa and sign bit (with zero exponent, c = 127)
    SIMDInt i = (getRandom<SIMDInt>() & 0x807fffff) | 0x3f800000;
    // f12 is a float from [1,2)
    SIMDFloat f12;
    memcpy(&f12, &i, 4);
    // generate random exponent from [0,31]
    if (rand() & 0x01) return ldexpf(f12, rand() & 0x1f);
    // or just return completely random float
    return getRandom<SIMDFloat>();
  }

  // =========================================================================
  // random special value
  // =========================================================================
  
  template<typename T>
  static T getRandomSpecialValue()
  {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::vector<T> specialValues = SIMDTypeInfo<T>::specialValues();
    std::uniform_int_distribution<> dis(0, specialValues.size() - 1);
    return specialValues[dis(gen)];
  }

  // returns a special value with 1/16 probability and a random value otherwise
  template<typename T>
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
  static SIMD_INLINE T
  getRandomOrSpecialValueRanges();

  template <>
  SIMD_INLINE SIMDByte
  getRandomOrSpecialValueRanges<SIMDByte>()
  {
    return getRandomOrSpecialValue<SIMDByte>();
  }

  template <>
  SIMD_INLINE SIMDSignedByte
  getRandomOrSpecialValueRanges<SIMDSignedByte>()
  {
    return getRandomOrSpecialValue<SIMDSignedByte>();
  }

  template <>
  SIMD_INLINE SIMDWord
  getRandomOrSpecialValueRanges<SIMDWord>()
  {
    if (rand() & 0x01) return getRandomOrSpecialValueRanges<SIMDByte>();
    return getRandomOrSpecialValue<SIMDWord>();
  }

  template <>
  SIMD_INLINE SIMDShort
  getRandomOrSpecialValueRanges<SIMDShort>()
  {
    if (rand() & 0x01) return getRandomOrSpecialValueRanges<SIMDSignedByte>();
    return getRandomOrSpecialValue<SIMDShort>();
  }

  template <>
  SIMD_INLINE SIMDInt
  getRandomOrSpecialValueRanges<SIMDInt>()
  {
    if (rand() & 0x01) return getRandomOrSpecialValueRanges<SIMDShort>();
    return getRandomOrSpecialValue<SIMDInt>();
  }

  template <>
  SIMD_INLINE SIMDFloat
  getRandomOrSpecialValueRanges<SIMDFloat>()
  {
    // generate 23 bit mantissa and sign bit (with zero exponent, c = 127)
    SIMDInt i = (getRandomOrSpecialValue<SIMDInt>() & 0x807fffff) | 0x3f800000;
    // f12 is a float from [1,2)
    SIMDFloat f12;
    memcpy(&f12, &i, 4);
    // generate random exponent from [0,31]
    if (rand() & 0x01) return ldexpf(f12, rand() & 0x1f);
    // or just return completely random float
    return getRandomOrSpecialValue<SIMDFloat>();
  }
}

#endif
