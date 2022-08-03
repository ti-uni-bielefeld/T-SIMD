// ===========================================================================
// 
// SIMDSerialVec.H --
// serial implementations of SIMDVec functions
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

#ifndef _SIMD_VEC_SERIAL_H_
#define _SIMD_VEC_SERIAL_H_

#include <string.h>
#include <stdint.h>
#include <stdlib.h>
// round, floor, ceil, sqrt, fmin, fmax
#include <cmath>
// min, max
#include <algorithm>
#include <assert.h>
#include "SIMDVecAll.H"
#include <limits>

// design considerations
//=======================
//
// separate class SIMDSerialVec such that comparison with SIMDVec is
// possible
// 
// SIMDFloat-only functions (mul, div...) are provided only for
// SIMDFloat
//
// other functions which are not available for all types (neg...) are
// provided with generic parameter T (won't compile for other types)

namespace ns_simd {

  // =========================================================================
  // random numbers
  // =========================================================================

  template <typename T>
  static SIMD_INLINE T
  getRandom();
  
  template <>
  SIMD_INLINE SIMDByte
  getRandom<SIMDByte>()
  {
    return random() & 0xff;
  }

  template <>
  SIMD_INLINE SIMDSignedByte
  getRandom<SIMDSignedByte>()
  {
    return random() & 0xff;
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
    bool restricted = (random() & 0x01);
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
    if (random() & 0x01) return getRandomRanges<SIMDByte>();
    return getRandom<SIMDWord>();
  }

  template <>
  SIMD_INLINE SIMDShort
  getRandomRanges<SIMDShort>()
  {
    if (random() & 0x01) return getRandomRanges<SIMDSignedByte>();
    return getRandom<SIMDShort>();
  }

  template <>
  SIMD_INLINE SIMDInt
  getRandomRanges<SIMDInt>()
  {
    if (random() & 0x01) return getRandomRanges<SIMDShort>();
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
    if (random() & 0x01) return ldexpf(f12, random() & 0x1f);
    // or just return completely random float
    return getRandom<SIMDFloat>();
  }

  // =========================================================================
  // SIMDSerialVec
  // =========================================================================

  // for all integer types, we can compare bit patterns
  template <typename T>
  static SIMD_INLINE bool
  sameSerialVec(const T *const p1, const T *const p2, int n)
  {
    return !memcmp(p1, p2, n * sizeof(T));
  }

  // for SIMDFloat, we have to make sure that -0.0f == +0.0f
  static SIMD_INLINE bool
  sameSerialVec(const SIMDFloat *const p1, const SIMDFloat *const p2, int n)
  {
    // case 1: if all bit pattern match, the vectors are the same
    if (!memcmp(p1, p2, n * sizeof(SIMDFloat))) return true;
    // case 2: if not all bit pattern match, we look at +-0.0f
    // go through all individual elements
    for (int i = 0; i < n; i++)
      // check whether the bit pattern has a mismatch for this element
      if (memcmp(p1 + i, p2 + i, sizeof(SIMDFloat)))
	// we have a bit pattern mismatch
	// if it is not the case +-0.0f, we have found a vector mismatch
	if (!((fabsf(p1[i]) == 0.0f) && (fabsf(p2[i]) == 0.0f)))
	  return false;
    // no mismatch found
    return true;
  }

  // 31. Jul 18 (rm): comparison of scalar
  template <typename T>
  static SIMD_INLINE bool
  sameScalar(const T& a, const T& b)
  {
    return sameSerialVec(&a, &b, 1);
  }
  
  template <typename T, int SIMD_WIDTH>
  class SIMDSerialVec
  {
  public:
    
    enum { elements = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH), bytes = SIMD_WIDTH};
    
    T vec[elements] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
    
    SIMDSerialVec() 
    {
    }
    template <typename U>
    SIMDSerialVec(const SIMDSerialVec<U,SIMD_WIDTH> &x)
    {
      memcpy(vec, x.vec, SIMD_WIDTH);
    }
    SIMDSerialVec(const SIMDVec<T,SIMD_WIDTH> &x)
    {
      storeu(vec, x);
    }
    template <typename U>
    SIMDSerialVec& operator=(const SIMDSerialVec<U,SIMD_WIDTH> &x)
    {
      memcpy(vec, x.vec, SIMD_WIDTH);
      return *this;
    }
    SIMDSerialVec& operator=(const SIMDVec<T,SIMD_WIDTH> &x) 
    {
      store(vec, x);
      return *this;
    }
    operator SIMDVec<T, SIMD_WIDTH>() const
    {
      return getSIMDVec();
    }
    SIMDVec<T, SIMD_WIDTH> getSIMDVec() const
    {
      return load<SIMD_WIDTH>(vec);
    }
    bool same(const SIMDSerialVec<T,SIMD_WIDTH> &x) const
    {
      return sameSerialVec(vec, x.vec, elements);
    }
    bool operator== (const SIMDSerialVec<T,SIMD_WIDTH> &x) const
    {
      return this->same(x);
    }
    bool operator!= (const SIMDSerialVec<T,SIMD_WIDTH> &x) const
    {
      return !(*this == x);
    }
    T operator[] (int i) const
    {
      if ((i >= 0) && (i < elements))
	return vec[i];
      return T(0);
    }
    T& operator[] (int i)
    {
      assert ((i >= 0) && (i < elements));
      return vec[i];
    }
    void randomize()
    {
      // random init

      // 03. Aug (Jonas Keller):
      // increased the chance of extreme values (min, max, 0, -0)
      // to make sure those are tested as well
      //
      //     for (int i = 0; i < elements; i++)
	    // vec[i] = getRandom<T>();

      // randomize with a 1 in 16 chance to be completely one of min, max, 0 and -0
      if ((random() & 0x0f) == 0) {
        T val;
        switch (random() & 0x03) {
        case 0: val = SIMDTypeInfo<T>::min(); break;
        case 1: val = SIMDTypeInfo<T>::max(); break;
        case 2: val = T(0); break;
        case 3: val = -T(0); break;
        default: assert(false); break;
        }
        for (int i = 0; i < elements; i++)
          vec[i] = val;
      } else {
        for (int i = 0; i < elements; i++) {
          // randomize with a 1 in 16 chance to be one of min, max, 0 and -0
          if ((random() & 0x0f) == 0) {
            switch (random() & 0x03) {
            case 0: vec[i] = SIMDTypeInfo<T>::min(); break;
            case 1: vec[i] = SIMDTypeInfo<T>::max(); break;
            case 2: vec[i] = T(0); break;
            case 3: vec[i] = -T(0); break;
            default: assert(false); break;
            }
          } else {
            vec[i] = getRandom<T>();
          }
        }
      }
    }
    void randomizeNonNegative()
    {
      // random init: non-negative, with injection of zeros

      // 03. Aug (Jonas Keller):
      // increased the chance of extreme values (min, max, 0)
      // to make sure those are tested as well
      //
      //     for (int i = 0; i < elements; i++) {
	    // if ((random() & 0x0f) == 0) vec[i] = T(0);
	    // else while ((vec[i] = getRandom<T>()) < T(0));

      // randomize with a 1 in 16 chance to be completely one of min, max and 0
      if ((random() & 0x0f) == 0) {
        T val;
        switch (random() & 0x03) {
        case 0: val = SIMDTypeInfo<T>::min(); break;
        case 1: val = SIMDTypeInfo<T>::max(); break;
        case 2: case 3: val = T(0); break;
        default: assert(false); break;
        }
        for (int i = 0; i < elements; i++)
          vec[i] = val;
      } else {
        for (int i = 0; i < elements; i++) {
          // randomize with a 1 in 16 chance to be one of min, max, and 0
          if ((random() & 0x0f) == 0) {
            switch (random() & 0x03) {
            case 0: vec[i] = SIMDTypeInfo<T>::min(); break;
            case 1: vec[i] = SIMDTypeInfo<T>::max(); break;
            case 2: case 3: vec[i] = T(0); break;
            default: assert(false); break;
            }
          } else {
            while ((vec[i] = getRandom<T>()) < T(0));
          }
        }
      }
    }
    void randomizeRanges()
    {
      // 03. Aug (Jonas Keller):
      // increased the chance of extreme values (min, max, 0, -0)
      // to make sure those are tested as well
      //
      //     for (int i = 0; i < elements; i++) 
	    // vec[i] = getRandomRanges<T>();

      // randomize with a 1 in 16 chance to be completely one of min, max, 0 and -0
      if ((random() & 0x0f) == 0) {
        T val;
        switch (random() & 0x03) {
        case 0: val = SIMDTypeInfo<T>::min(); break;
        case 1: val = SIMDTypeInfo<T>::max(); break;
        case 2: val = T(0); break;
        case 3: val = -T(0); break;
        default: assert(false); break;
        }
        for (int i = 0; i < elements; i++)
          vec[i] = val;
      } else {
        for (int i = 0; i < elements; i++) {
          // randomize with a 1 in 16 chance to be one of min, max, 0 and -0
          if ((random() & 0x0f) == 0) {
            switch (random() & 0x03) {
            case 0: vec[i] = SIMDTypeInfo<T>::min(); break;
            case 1: vec[i] = SIMDTypeInfo<T>::max(); break;
            case 2: vec[i] = T(0); break;
            case 3: vec[i] = -T(0); break;
            default: assert(false); break;
            }
          } else {
            vec[i] = getRandomRanges<T>();
          }
        }
      }
    }
    void randomizeRangesNonZero()
    {
      // 03. Aug (Jonas Keller):
      // increased the chance of extreme values (min, max, 0, -0)
      // to make sure those are tested as well
      //
      //     for (int i = 0; i < elements; i++)
	    // while ((vec[i] = getRandomRanges<T>()) == T(0));

      // randomize with a 1 in 16 chance to be completely one of min and max
      if ((random() & 0x0f) == 0) {
        T val;
        if (SIMDTypeInfo<T>::min() != 0) {
          val = random() & 0x01 ? SIMDTypeInfo<T>::min()
                                : SIMDTypeInfo<T>::max();
        } else {
          val = SIMDTypeInfo<T>::max();
        }
        for (int i = 0; i < elements; i++)
          vec[i] = val;
      } else {
        for (int i = 0; i < elements; i++) {
          // randomize with a 1 in 16 chance to be one of min and max
          if ((random() & 0x0f) == 0) {
            if (SIMDTypeInfo<T>::min() != 0) {
              vec[i] = random() & 0x01 ? SIMDTypeInfo<T>::min()
                                       : SIMDTypeInfo<T>::max();
            } else {
              vec[i] = SIMDTypeInfo<T>::max();
            }
          } else {
            while ((vec[i] = getRandomRanges<T>()) == T(0));
          }
        }
      }
    }
    void randomizeCond()
    {
      // 03. Aug (Jonas Keller):
      // increased the chance of extreme values (all 0s, all 1s)
      // to make sure those are tested as well
      //
      //     for (int i = 0; i < elements; i++)
	    // vec[i] = (random() & 0x01) ? SIMDTypeInfo<T>::trueval() : T(0);

      // randomize with a 1 in 16 chance to be all 1s or all 0s
      if ((random() & 0x0f) == 0) {
        if (random() & 0x01) {
          setzero();
        } else {
          setones();
        }
      } else {
        for (int i = 0; i < elements; i++)
          vec[i] = (random() & 0x01) ? SIMDTypeInfo<T>::trueval() : T(0);
      }
    }
    void setzero()
    {
      for (int i = 0; i < elements; i++) vec[i] = T(0);
    }
    void setones()
    {
      for (int i = 0; i < elements; i++) vec[i] = SIMDTypeInfo<T>::trueval();
    }
  };
    
  // =========================================================================
  // daz: treat denormal as zero
  // =========================================================================

  // NOTE: there are differences in the behavior of fwaddmul and
  // fmuladd depending on the architecture flags; -mavx2 works,
  // -march=native on an avx2 shows differences between serial and
  // parallel code; use g++ -march=native (or -mavx2) -Q --help=target
  // to reveal the differences; it may have something to do with
  // fused-multiply-add (on avx this flag is not different and we see
  // no differences)

  // no daz for integer on all architectures
  template <typename T>
  static SIMD_INLINE T
  daz(T a)
  {
    return a;
  }

  // no daz for SIMDFloat on Intel
#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  static SIMD_INLINE SIMDFloat
  daz(SIMDFloat a)
  {
    return a;
  }
#endif

  // use daz for ARM NEON
#ifdef SIMDVEC_NEON_ENABLE
  static SIMD_INLINE SIMDFloat
  daz(SIMDFloat a)
  {
    if (std::isnormal(a) || std::isnan(a) || !std::isfinite(a)) return a;
    return 0.0f;
  }
#endif

  // =========================================================================
  // auxiliary functions
  // =========================================================================

  // -------------------------------------------------------------------------
  // add
  // -------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE T
  _add(T a, T b)
  {
    return daz(daz(a) + daz(b));
  }

  // -------------------------------------------------------------------------
  // _adds
  // -------------------------------------------------------------------------
  
  // SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE T
  _adds(T a, T b)
  {
    SIMDInt tmp = SIMDInt(a) + SIMDInt(b);
    return
      (tmp > SIMDInt(SIMDTypeInfo<T>::max())) ? SIMDTypeInfo<T>::max() :
      (tmp < SIMDInt(SIMDTypeInfo<T>::min())) ? SIMDTypeInfo<T>::min() :
      T(tmp);
  }

#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  // SIMDInt: not saturated on Intel
  static SIMD_INLINE SIMDInt
  _adds(SIMDInt a, SIMDInt b)
  {
    return a + b;
  }
#endif

#ifdef SIMDVEC_NEON_ENABLE
  // SIMDInt: saturated on NEON
  static SIMD_INLINE SIMDInt
  _adds(SIMDInt a, SIMDInt b)
  {
    int64_t tmp = int64_t(a) + int64_t(b);
    return
      (tmp > int64_t(SIMDTypeInfo<SIMDInt>::max())) ? 
      SIMDTypeInfo<SIMDInt>::max() :
      (tmp < int64_t(SIMDTypeInfo<SIMDInt>::min())) ? 
      SIMDTypeInfo<SIMDInt>::min() :
      SIMDInt(tmp);
  }
#endif

  // SIMDFloat: not saturated
  static SIMD_INLINE SIMDFloat
  _adds(SIMDFloat a, SIMDFloat b)
  {
    return daz(daz(a) + daz(b));
  }
  
  // -------------------------------------------------------------------------
  // sub
  // -------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE T
  _sub(T a, T b)
  {
    return daz(daz(a) - daz(b));
  }

  // -------------------------------------------------------------------------
  // _subs
  // -------------------------------------------------------------------------
  
  // SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE T
  _subs(T a, T b)
  {
    SIMDInt tmp = SIMDInt(a) - SIMDInt(b);
    return
      (tmp > SIMDInt(SIMDTypeInfo<T>::max())) ? SIMDTypeInfo<T>::max() :
      (tmp < SIMDInt(SIMDTypeInfo<T>::min())) ? SIMDTypeInfo<T>::min() :
      T(tmp);
  }

#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  // SIMDInt: not saturated on Intel
  static SIMD_INLINE SIMDInt
  _subs(SIMDInt a, SIMDInt b)
  {
    return a - b;
  }
#endif

#ifdef SIMDVEC_NEON_ENABLE
  // SIMDInt: saturated on NEON
  static SIMD_INLINE SIMDInt
  _subs(SIMDInt a, SIMDInt b)
  {
    int64_t tmp = int64_t(a) - int64_t(b);
    return
      (tmp > int64_t(SIMDTypeInfo<SIMDInt>::max())) ? 
      SIMDTypeInfo<SIMDInt>::max() :
      (tmp < int64_t(SIMDTypeInfo<SIMDInt>::min())) ? 
      SIMDTypeInfo<SIMDInt>::min() :
      SIMDInt(tmp);
  }
#endif

  // SIMDFloat: not saturated
  static SIMD_INLINE SIMDFloat
  _subs(SIMDFloat a, SIMDFloat b)
  {
    return daz(daz(a) - daz(b));
  }

#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  // special way to avoid overflow case
  static SIMD_INLINE SIMDInt
  _cvts_f2i(SIMDFloat a)
  {
    SIMDFloat clip = MAX_POS_FLOAT_CONVERTIBLE_TO_INT32;
    return SIMDInt(std::min(clip, ::rintf(a)));
  }
#endif

#ifdef SIMDVEC_NEON_ENABLE
  static SIMD_INLINE SIMDInt
  _cvts_f2i(SIMDFloat a)
  {
    return SIMDInt(::trunc(a));
  }
#endif

  // just convert
  static SIMD_INLINE SIMDFloat
  _cvts_i2f(SIMDInt a)
  {
    return SIMDFloat(a);
  }
  
  // -------------------------------------------------------------------------
  // _mul / _div
  // -------------------------------------------------------------------------

  static SIMD_INLINE SIMDFloat
  _mul(SIMDFloat a, SIMDFloat b)
  {
    return daz(daz(a) * daz(b));
  }

  static SIMD_INLINE SIMDFloat
  _div(SIMDFloat a, SIMDFloat b)
  {
    return daz(daz(a) / daz(b));
  }

  // -------------------------------------------------------------------------
  // _min / _max
  // -------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE T
  _min(T a, T b)
  {
    return std::min(a, b);
  }

  static SIMD_INLINE SIMDFloat
  _min(SIMDFloat a, SIMDFloat b)
  {
    return daz(fmin(daz(a), daz(b)));
  }

  template <typename T>
  static SIMD_INLINE T
  _max(T a, T b)
  {
    return std::max(a, b);
  }

  static SIMD_INLINE SIMDFloat
  _max(SIMDFloat a, SIMDFloat b)
  {
    return daz(fmax(daz(a), daz(b)));
  }

  // -------------------------------------------------------------------------
  // _abs
  // -------------------------------------------------------------------------

  // SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort, SIMDInt
  template <typename T>
  static SIMD_INLINE T
  _abs(T a)
  {
    return std::abs(a);
  }

  // no daz also on NEON
  static SIMD_INLINE SIMDFloat
  _abs(SIMDFloat a)
  {
    return fabsf(a);
  }

  // -------------------------------------------------------------------------
  // _packs
  // -------------------------------------------------------------------------

  // also accepts same-size input and output types

  // from integer types to integer types
  template <typename Tout, typename T>
  static SIMD_INLINE Tout
  _packs(T in, OutputType<Tout>)
  {
    T minv = T(SIMDTypeInfo<Tout>::min());
    T maxv = T(SIMDTypeInfo<Tout>::max());
    return (in >= minv) ? ((in <= maxv) ? Tout(in) : Tout(maxv)) : Tout(minv);
  }

  // from integer type to same integer type
  template <typename T>
  static SIMD_INLINE T
  _packs(T in, OutputType<T>)
  {
    return in;
  }

  // from SIMDInt to SIMDFloat
  static SIMD_INLINE SIMDFloat
  _packs(SIMDInt in, OutputType<SIMDFloat>)
  {
    return _cvts_i2f(in);
  }

  // from SIMDFloat to integer types
  template <typename Tout>
  static SIMD_INLINE Tout
  _packs(SIMDFloat in, OutputType<Tout>)
  {
    // via SIMDInt
    return _packs(_cvts_f2i(in), OutputType<Tout>());
  }

  // from SIMDFloat to SIMDFloat (otherwise ambiguous)
  static SIMD_INLINE SIMDFloat
  _packs(SIMDFloat in, OutputType<SIMDFloat>)
  {
    return in;
  }

  // hub
  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _packs(Tin in)
  {
    return _packs(in, OutputType<Tout>());
  }

  // -------------------------------------------------------------------------
  // _extend
  // -------------------------------------------------------------------------

  // also accepts same-size input and output types

  // from integer types to integer types
  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _extend(Tin in, OutputType<Tout>)
  {
    return Tout(in);
  }

  // from integer type to same-size integer type
  template <typename T>
  static SIMD_INLINE T
  _extend(T in, OutputType<T>)
  {
    return in;
  }
  
  // SIMDFloat -> SIMDInt
  static SIMD_INLINE SIMDInt
  _extend(SIMDFloat in, OutputType<SIMDInt>)
  {
    return _cvts_f2i(in);
  }

  // SIMDInt -> SIMDFloat
  static SIMD_INLINE SIMDFloat
  _extend(SIMDInt in, OutputType<SIMDFloat>)
  {
    return _cvts_i2f(in);
  }

  // SIMDFloat -> SIMDFloat (otherwise ambiguous)
  static SIMD_INLINE SIMDFloat
  _extend(SIMDFloat in, OutputType<SIMDFloat>)
  {
    return in;
  }

  // hub
  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _extend(Tin in)
  {
    return _extend(in, OutputType<Tout>());
  }

  // -------------------------------------------------------------------------
  // _convert
  // -------------------------------------------------------------------------

  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _convert(CompareLess, OutputType<Tout>, Tin in)
  {
    return _packs<Tout>(in);
  }

  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _convert(CompareEqual, OutputType<Tout>, Tin in)
  {
    return in;
  }

  static SIMD_INLINE SIMDInt
  _convert(CompareEqual, OutputType<SIMDInt>, SIMDFloat in)
  {
    return _cvts_f2i(in);
  }

  static SIMD_INLINE SIMDFloat
  _convert(CompareEqual, OutputType<SIMDFloat>, SIMDInt in)
  {
    return _cvts_i2f(in);
  }

  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _convert(CompareGreater, OutputType<Tout>, Tin in)
  {
    return Tout(in);
  }

  // hub
  template <typename Tout, typename Tin>
  static SIMD_INLINE Tout
  _convert(Tin in)
  {
    return _convert(CompareTypes<Tout,Tin>(), OutputType<Tout>(), in);
  }
  
  // -------------------------------------------------------------------------
  // shifts
  // -------------------------------------------------------------------------

  // work around for e.g. ARM compiler warning "left/right shift count
  // >= width of type"

  template <int IMM, typename T>
  static SIMD_INLINE T
  _srli(T in, IsPosInRange<true, true>)
  {
    // on signed types, we force the compiler to use an unsigned shift
    typedef typename SIMDTypeInfo<T>::UnsignedType U;
    U u = U(in);
    u = u >> IMM; 
    return T(u);
  }

  template <int IMM, typename T>
  static SIMD_INLINE T
  _srli(T, IsPosInRange<true, false>)
  {
    return T(0);
  }

  // hub
  template <int IMM, typename T>
  static SIMD_INLINE T
  _srli(T in)
  {
    return _srli<IMM>(in, IsPosInBitRange<T, IMM>());
  }
  
  template <int IMM, typename T>
  static SIMD_INLINE T
  _slli(T in, IsPosInRange<true, true>)
  {
    return in << IMM;
  }

  template <int IMM, typename T>
  static SIMD_INLINE T
  _slli(T, IsPosInRange<true, false>)
  {
    return T(0);
  }

  // hub
  template <int IMM, typename T>
  static SIMD_INLINE T
  _slli(T in)
  {
    return _slli<IMM>(in, IsPosInBitRange<T, IMM>());
  }

  // =========================================================================
  // wrapper for template functions without vector arguments
  // =========================================================================

  // the problem is that we can't deduce the vector type
  // (SIMDSerialVec, SIMDVec) in function templates without vector
  // arguments, so we have to use this additional layer

  // primary template
  template <typename T, int SIMD_WIDTH, template <typename, int> class VEC>
  struct Deductor;

  // -------------------------------------------------------------------------
  // for SIMDVec, we access the template functions
  // -------------------------------------------------------------------------

  template <typename T, int SIMD_WIDTH>
  struct Deductor<T, SIMD_WIDTH, SIMDVec>
  {
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _setzero() 
    { 
      return setzero<T,SIMD_WIDTH>(); 
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _setones() 
    { 
      return setones<T,SIMD_WIDTH>(); 
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _setmin() 
    { 
      return setmin<T,SIMD_WIDTH>(); 
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _setmax() 
    { 
      return setmax<T,SIMD_WIDTH>(); 
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _setnegunity() 
    { 
      return setnegunity<T,SIMD_WIDTH>(); 
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _setunity() 
    { 
      return setunity<T,SIMD_WIDTH>(); 
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _set1(T a)
    {
      return set1<T,SIMD_WIDTH>(a);
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _load(const T *const p)
    {
      return load<SIMD_WIDTH,T>(p);
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _loadu(const T *const p)
    {
      return loadu<SIMD_WIDTH,T>(p);
    }
  };

  // -------------------------------------------------------------------------
  // for SIMDSerialVec, we directly implement the functions
  // -------------------------------------------------------------------------

  template <typename T, int SIMD_WIDTH>
  struct Deductor<T, SIMD_WIDTH, SIMDSerialVec>
  {
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _setzero()
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      memset(v.vec, 0x00, SIMD_WIDTH);
      return v;
    }
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _setones()
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      for (int i = 0; i < v.elements; i++)
	v[i] = SIMDTypeInfo<T>::trueval();
      return v;
    }
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _setmin()
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      for (int i = 0; i < v.elements; i++)
	v[i] = SIMDTypeInfo<T>::min();
      return v;
    }
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _setmax()
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      for (int i = 0; i < v.elements; i++)
	v[i] = SIMDTypeInfo<T>::max();
      return v;
    }
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _setnegunity()
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      for (int i = 0; i < v.elements; i++)
	v[i] = T(-1);
      return v;
    }
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _setunity()
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      for (int i = 0; i < v.elements; i++)
	v[i] = T(1);
      return v;
    }
    static SIMD_INLINE SIMDSerialVec<T,SIMD_WIDTH> _set1(T a)
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      for (int i = 0; i < v.elements; i++)
	v[i] = a;
      return v;
    }
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _load(const T *const p)
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      memcpy(v.vec, p, SIMD_WIDTH);
      return v;
    }    
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> _loadu(const T *const p)
    {
      SIMDSerialVec<T,SIMD_WIDTH> v;
      memcpy(v.vec, p, SIMD_WIDTH);
      return v;
    }   
    
  };

  // =========================================================================
  // _swizzle/2, _unswizzle2 wrapper for SIMDVec
  // =========================================================================

  // _swizzle wrapper for SIMDVec (SwizzleTable used internally, for tests only)
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  _swizzle(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    SwizzleTable<N, T, SIMD_WIDTH> t;
    swizzle(t, v);
  }

  // _swizzle2 wrapper for SIMDVec
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  _swizzle2(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    swizzle2<N>(v);
  }

  // _unswizzle2 wrapper for SIMDVec
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  _unswizzle2(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    unswizzle2<N>(v);
  }

  // =========================================================================
  // template functions
  // =========================================================================

  template <typename Tdst, typename Tsrc, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<Tdst,SIMD_WIDTH>
  reinterpret(const SIMDSerialVec<Tsrc,SIMD_WIDTH> &a)
  {
    SIMDSerialVec<Tdst,SIMD_WIDTH> v;
    memcpy(v.vec, a.vec, SIMD_WIDTH);
    return v;
  }

  // float -> int
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDInt,SIMD_WIDTH>
  cvts(const SIMDSerialVec<SIMDFloat,SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDInt,SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _cvts_f2i(a[i]);
    return c;
  }

  // int -> float
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat,SIMD_WIDTH>
  cvts(const SIMDSerialVec<SIMDInt,SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat,SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _cvts_i2f(a[i]);
    return c;
  }

  // hub (just for compatibility with SIMDVec version which has
  // primary template)
  template <typename Tout, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<Tout,SIMD_WIDTH>
  cvts(const SIMDSerialVec<T,SIMD_WIDTH> &a)
  {
    return cvts(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  store(T *const p,
	const SIMDSerialVec<T, SIMD_WIDTH> &v)
  {
    memcpy(p, v.vec, SIMD_WIDTH);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  storeu(T *const p,
	 const SIMDSerialVec<T, SIMD_WIDTH> &v)
  {
    store(p, v);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  streamstore(T *const p, 
	      const SIMDSerialVec<T, SIMD_WIDTH> &v)
  {
    store(p, v);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  extract(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    return a[IMM];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  add(const SIMDSerialVec<T, SIMD_WIDTH> &a,
      const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _add(a[i], b[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  adds(const SIMDSerialVec<T, SIMD_WIDTH> &a,
       const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _adds(a[i], b[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  sub(const SIMDSerialVec<T, SIMD_WIDTH> &a,
      const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _sub(a[i], b[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  subs(const SIMDSerialVec<T, SIMD_WIDTH> &a,
       const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _subs(a[i], b[i]);
    return c;
  }

  // no daz also on NEON
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  neg(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = -a[i];
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  mul(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _mul(a[i], b[i]);
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  div(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _div(a[i], b[i]);
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  ceil(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = std::ceil(daz(a[i]));
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  floor(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = std::floor(daz(a[i]));
    return c;
  }

#if defined(SIMDVEC_INTEL_ENABLE) || defined(SIMDVEC_SANDBOX)
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  round(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      // std::round has different behavior
      c[i] = ::rintf(a[i]);
    return c;
  }
#endif

#ifdef SIMDVEC_NEON_ENABLE
  // I'm not sure about this one, this is at least the NEON
  // implementation of the workaround if vrndq is missing;
  // also not sure whether daz() is required
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  round(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      // std::round has different behavior
      c[i] = std::floor(daz(a[i]) + 0.5f);
    return c;
  }
#endif
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  truncate(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = ::truncf(daz(a[i]));
    return c;
  }

  // TODO: rcp (not possible since hardware implementation is not known)
  // TODO: rsqrt (not possible since hardware implementation is not known)
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  sqrt(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = daz(std::sqrt(daz(a[i])));
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  min(const SIMDSerialVec<T, SIMD_WIDTH> &a,
      const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _min(a[i], b[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  max(const SIMDSerialVec<T, SIMD_WIDTH> &a,
      const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _max(a[i], b[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  abs(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _abs(a[i]);
    return c;
  }

  // negate a, where b is negative (SIMDFloat only)
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH> 
  sign(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
       const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (b[i] < 0.0f) ? -a[i] : a[i];
    return c;
  }


  // for signed integer types (SIMDSignedByte, SIMDShort, SIMDInt)
  // which can overflow in absDiff

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDSignedByte, SIMD_WIDTH> 
  absDiff(const SIMDSerialVec<SIMDSignedByte, SIMD_WIDTH> &a,
	  const SIMDSerialVec<SIMDSignedByte, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDSignedByte, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = std::abs(SIMDSignedByte(a[i] - b[i]));
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDShort, SIMD_WIDTH> 
  absDiff(const SIMDSerialVec<SIMDShort, SIMD_WIDTH> &a,
	  const SIMDSerialVec<SIMDShort, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDShort, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = std::abs(SIMDShort(a[i] - b[i]));
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDInt, SIMD_WIDTH> 
  absDiff(const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &a,
	  const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = std::abs(SIMDInt(a[i] - b[i]));
    return c;
  }

  // for all other types (SIMDByte, SIMDWord, SIMDFloat) which can't
  // overflow in absDiff 
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDByte, SIMD_WIDTH> 
  absDiff(const SIMDSerialVec<SIMDByte, SIMD_WIDTH> &a,
	  const SIMDSerialVec<SIMDByte, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDByte, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDWord, SIMD_WIDTH> 
  absDiff(const SIMDSerialVec<SIMDWord, SIMD_WIDTH> &a,
	  const SIMDSerialVec<SIMDWord, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDWord, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH> 
  absDiff(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
	  const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
    return c;
  }

  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  unpack(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	 const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    int i0 = PART * c.elements / 2, ie = i0 + c.elements / 2;
    for (int i = i0, k = 0; i < ie; i += NUM_ELEMS) {
      for (int j = 0; j < NUM_ELEMS; j++, k++)
	c[k] = a[i + j];
      for (int j = 0; j < NUM_ELEMS; j++, k++)
	c[k] = b[i + j];
    }
    return c;
  }

  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  zip(const SIMDSerialVec<T, SIMD_WIDTH> a,
      const SIMDSerialVec<T, SIMD_WIDTH> b,
      SIMDSerialVec<T, SIMD_WIDTH> &c,
      SIMDSerialVec<T, SIMD_WIDTH> &d)
  {
    c = unpack<0, NUM_ELEMS>(a, b);
    d = unpack<1, NUM_ELEMS>(a, b);
  }

  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  unzip(const SIMDSerialVec<T, SIMD_WIDTH> a,
	const SIMDSerialVec<T, SIMD_WIDTH> b,
	SIMDSerialVec<T, SIMD_WIDTH> &c,
	SIMDSerialVec<T, SIMD_WIDTH> &d)
  {
    const int n = a.elements, n2 = n/2;
    for (int i = 0, k = 0; i < n; i += NUM_ELEMS) {
      for (int j = 0; j < NUM_ELEMS; j++, k++) {
	c[k] = (i < n2) ? a[2*i+j] : b[2*(i-n2)+j];
	d[k] = (i < n2) ? a[2*i+NUM_ELEMS+j] : b[2*(i-n2)+NUM_ELEMS+j];
      }
    }
  }

  // from integer types
  template <typename Tout, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<Tout, SIMD_WIDTH>
  packs(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<Tout, SIMD_WIDTH> c;
    int ce = c.elements, ce2 = ce / 2;
    for (int i = 0; i < ce; i++) {
      T in = (i < ce2) ? a[i] : b[i - ce2];
      c[i] = _packs<Tout>(in);
    }
    return c;
  }

  // from float types
  template <typename Tout, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<Tout, SIMD_WIDTH>
  packs(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
	const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> ai, bi;
    ai = cvts(a);
    bi = cvts(b);
    return packs<Tout>(ai, bi);
  }

  template <typename Tout, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  extend(const SIMDSerialVec<T,SIMD_WIDTH> &vIn,
	 SIMDSerialVec<Tout,SIMD_WIDTH> *const vOut)
  {
    int nOut = SIMDSerialVec<Tout,SIMD_WIDTH>::elements;
    for (int i = 0; i < vIn.elements; i++)
      vOut[i / nOut][i % nOut] = Tout(vIn[i]);
  }

  // float -> int (uses saturating cvts)
  template <int SIMD_WIDTH>
  static SIMD_INLINE void
  extend(const SIMDSerialVec<SIMDFloat,SIMD_WIDTH> &vIn,
	 SIMDSerialVec<SIMDInt,SIMD_WIDTH> *const vOut)
  {
    *vOut = cvts(vIn);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  srai(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    // operator >> on signed types is *usually* an arithmetic right
    // shift, but this is not guaranteed, so we use a workaround
    typedef typename SIMDTypeInfo<T>::UnsignedType U;
    U m = 1u << (sizeof(T)*8-1); 
    for (int i = 0; i < c.elements; i++) {
      U u = U(a[i]);
      for (int s = 0; s < IMM; s++)
	u = (u >> 1) | (u & m);
      c[i] = T(u);
    }
    return c;
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  srli(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++) 
      c[i] = _srli<IMM>(a[i]);
    return c;
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  slli(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = _slli<IMM>(a[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  hadd(const SIMDSerialVec<T, SIMD_WIDTH> &a,
       const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    int i = 0, n = c.elements;
    for (; i < n/2; i++)
      c[i] = _add(a[2*i], a[2*i+1]);
    for (; i < n; i++)
      c[i] = _add(b[2*i-n], b[2*i-n+1]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  hadds(const SIMDSerialVec<T, SIMD_WIDTH> &a,
       const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    int i = 0, n = c.elements;
    for (; i < n/2; i++)
      c[i] = _adds(a[2*i], a[2*i+1]);
    for (; i < n; i++)
      c[i] = _adds(b[2*i-n], b[2*i-n+1]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  hsub(const SIMDSerialVec<T, SIMD_WIDTH> &a,
       const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    int i = 0, n = c.elements;
    for (; i < n/2; i++)
      c[i] = _sub(a[2*i], a[2*i+1]);
    for (; i < n; i++)
      c[i] = _sub(b[2*i-n], b[2*i-n+1]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  hsubs(const SIMDSerialVec<T, SIMD_WIDTH> &a,
       const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    int i = 0, n = c.elements;
    for (; i < n/2; i++)
      c[i] = _subs(a[2*i], a[2*i+1]);
    for (; i < n; i++)
      c[i] = _subs(b[2*i-n], b[2*i-n+1]);
    return c;
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  srle(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = a[i + IMM];
    return c;
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  slle(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = a[i - IMM];
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  elem0(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    return a[0];
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  alignre(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	  const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    int n = c.elements;
    for (int i = 0; i < n; i++) {
      int k = i + IMM;
      c[i] = (k >= n) ? a[k - n] : b[k];
    }
    return c;
  }

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  _swizzle(SIMDSerialVec<T, SIMD_WIDTH> *const v)
  {
    const int n = SIMDSerialVec<T, SIMD_WIDTH>::elements;
    SIMDSerialVec<T, SIMD_WIDTH> u[N];
    // copy the N input vectors
    for (int i = 0; i < N; i++) u[i] = v[i];
    // swizzle: j, js in [0,n*N)
    for (int j = 0; j < n * N; j++) {
      int js = (j % N) * n + std::floor(j / N);
      v[js / n][js % n] = u[j / n][j % n];
    }
  }

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  _swizzle2(SIMDSerialVec<T, SIMD_WIDTH> *const v)
  {
    const int n = SIMDSerialVec<T, SIMD_WIDTH>::elements, n2 = 2 * n;
    SIMDSerialVec<T, SIMD_WIDTH> u[2*N];
    // copy the 2*N input vectors
    for (int i = 0; i < 2*N; i++) u[i] = v[i];
    // swizzle: j, js in [0,2*n*N), copy: [js] <- [j]
    for (int j = 0; j < n2 * N; j++) {
      int js = (j % N) * n2 + std::floor(j / N);
      v[js / n][js % n] = u[j / n][j % n];
    }
  }

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  _unswizzle2(SIMDSerialVec<T, SIMD_WIDTH> *const v)
  {
    const int n = SIMDSerialVec<T, SIMD_WIDTH>::elements, n2 = 2 * n;
    SIMDSerialVec<T, SIMD_WIDTH> u[2*N];
    // copy the 2*N input vectors
    for (int i = 0; i < 2*N; i++) u[i] = v[i];
    // swizzle: j, js in [0,2*n*N), copy: [j] <- [js]
    for (int j = 0; j < n2 * N; j++) {
      int js = (j % N) * n2 + std::floor(j / N);
      v[j / n][j % n] = u[js / n][js % n];
    }
  }

  template <typename Tcond, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  ifelse(const SIMDSerialVec<Tcond,SIMD_WIDTH> &cond,
	 const SIMDSerialVec<T,SIMD_WIDTH> &trueVal,
	 const SIMDSerialVec<T,SIMD_WIDTH> &falseVal)
  {
    SIMD_STATIC_ASSERT(sizeof(Tcond) == sizeof(T));
    SIMDSerialVec<T, SIMD_WIDTH> res;
    // avoids comparison with SIMDTypeInfo::trueval, which fails for
    // SIMDFloat (NaN)
    SIMDSerialVec<Tcond, SIMD_WIDTH> negCond = not(cond);
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
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  cmplt(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (daz(a[i]) < daz(b[i])) ? SIMDTypeInfo<T>::trueval() : 0;
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  cmple(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (daz(a[i]) <= daz(b[i])) ? SIMDTypeInfo<T>::trueval() : 0;
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  cmpeq(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (daz(a[i]) == daz(b[i])) ? SIMDTypeInfo<T>::trueval() : 0;
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  cmpge(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (daz(a[i]) >= daz(b[i])) ? SIMDTypeInfo<T>::trueval() : 0;
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  cmpgt(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (daz(a[i]) > daz(b[i])) ? SIMDTypeInfo<T>::trueval() : 0;
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  cmpneq(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (daz(a[i]) != daz(b[i])) ? SIMDTypeInfo<T>::trueval() : 0;
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  and(const SIMDSerialVec<T, SIMD_WIDTH> &a,
      const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = a[i] & b[i];
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  and(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> ai(a), bi(b);
    return SIMDSerialVec<SIMDFloat, SIMD_WIDTH>(and(ai, bi));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  or(const SIMDSerialVec<T, SIMD_WIDTH> &a,
     const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = a[i] | b[i];
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  or(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
     const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> ai(a), bi(b);
    return SIMDSerialVec<SIMDFloat, SIMD_WIDTH>(or(ai, bi));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  andnot(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	 const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = ~a[i] & b[i];
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  andnot(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
	 const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> ai(a), bi(b);
    return SIMDSerialVec<SIMDFloat, SIMD_WIDTH>(andnot(ai, bi));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  xor(const SIMDSerialVec<T, SIMD_WIDTH> &a,
     const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = a[i] ^ b[i];
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  xor(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> ai(a), bi(b);
    return SIMDSerialVec<SIMDFloat, SIMD_WIDTH>(xor(ai, bi));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  not(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = ~a[i];
    return c;
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  not(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> ai(a);
    return SIMDSerialVec<SIMDFloat, SIMD_WIDTH>(not(ai));
  }

  // int types except SIMDInt
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  avg(const SIMDSerialVec<T, SIMD_WIDTH> &a,
      const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (SIMDInt(a[i]) + SIMDInt(b[i]) + 1) >> 1;
    return c;
  }

  // SIMDInt
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDInt, SIMD_WIDTH> 
  avg(const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &a,
      const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (uint64_t(a[i]) + uint64_t(b[i]) + 1) >> 1;
    return c;
  }

  // SIMDFloat
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  avg(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = daz(0.5f * (daz(a[i]) + daz(b[i])));
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_zeros(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    for (int i = 0; i < a.elements; i++)
      if (a[i] != T(0))
	return 0;
    return 1;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_ones(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    // avoids comparison with SIMDTypeInfo::trueval, which fails for
    // SIMDFloat (NaN)
    SIMDSerialVec<T, SIMD_WIDTH> aNot = not(a);
    for (int i = 0; i < a.elements; i++)
      if (aNot[i] != T(0))
	return 0;
    return 1;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  reverse(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < a.elements; i++)
      c[i] = a[a.elements - 1 - i];
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hadd(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c = a;
    for (int n = a.elements; n > 1; n = n / 2)
      for (int i = 0; i < n / 2; i++)
	c[i] = _add(c[2*i], c[2*i+1]);
    return c[0];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hadds(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c = a;
    for (int n = a.elements; n > 1; n = n / 2)
      for (int i = 0; i < n / 2; i++)
	c[i] = _adds(c[2*i], c[2*i+1]);
    return c[0];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hsub(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c = a;
    for (int n = a.elements; n > 1; n = n / 2)
      for (int i = 0; i < n / 2; i++)
	c[i] = _sub(c[2*i], c[2*i+1]);
    return c[0];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hsubs(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c = a;
    for (int n = a.elements; n > 1; n = n / 2)
      for (int i = 0; i < n / 2; i++)
	c[i] = _subs(c[2*i], c[2*i+1]);
    return c[0];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hmin(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c = a;
    for (int n = a.elements; n > 1; n = n / 2)
      for (int i = 0; i < n / 2; i++)
	c[i] = _min(c[2*i], c[2*i+1]);
    return c[0];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hmax(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c = a;
    for (int n = a.elements; n > 1; n = n / 2)
      for (int i = 0; i < n / 2; i++)
	c[i] = _max(c[2*i], c[2*i+1]);
    return c[0];
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  convert(const SIMDSerialVec<Tin,SIMD_WIDTH> *const inVecs, 
	  SIMDSerialVec<Tout,SIMD_WIDTH> *const outVecs)
  {
    const int numInVecs = NUM_INPUT_SIMDVECS(Tout, Tin);
    const int numInElems = NUM_SIMDVEC_ELEMENTS(Tin, SIMD_WIDTH);
    const int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);
    int outIdx = 0;
    for (int inVec = 0; inVec < numInVecs; inVec++)
      for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
	int outVec = outIdx / numOutElems;
	int outElem = outIdx % numOutElems;
	outVecs[outVec][outElem] = _convert<Tout>(inVecs[inVec][inElem]);
      }
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivmul(const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsNum,
	  const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsDenom,
	  double fac,
	  SIMDSerialVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    const int numInVecs = NUM_INPUT_SIMDVECS(Tout, Tin);
    const int numInElems = NUM_SIMDVEC_ELEMENTS(Tin, SIMD_WIDTH);
    const int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);
    float facF = fac;
    int outIdx = 0;
    for (int inVec = 0; inVec < numInVecs; inVec++)
      for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
	int outVec = outIdx / numOutElems;
	int outElem = outIdx % numOutElems;
	vecsOut[outVec][outElem] = _packs<Tout>
	  (_mul(_div(_extend<SIMDFloat>(vecsNum[inVec][inElem]), 
		     _extend<SIMDFloat>(vecsDenom[inVec][inElem])), 
		facF));
      }
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmul(const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsIn,
       double fac,
       SIMDSerialVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    const int numInVecs = NUM_INPUT_SIMDVECS(Tout, Tin);
    const int numInElems = NUM_SIMDVEC_ELEMENTS(Tin, SIMD_WIDTH);
    const int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);
    float facF = fac;
    int outIdx = 0;
    for (int inVec = 0; inVec < numInVecs; inVec++)
      for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
	int outVec = outIdx / numOutElems;
	int outElem = outIdx % numOutElems;
	vecsOut[outVec][outElem] = _packs<Tout>
	  (_mul(_extend<SIMDFloat>(vecsIn[inVec][inElem]), 
		facF));
      }
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  faddmul(const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double off,
	  double fac,
	  SIMDSerialVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    const int numInVecs = NUM_INPUT_SIMDVECS(Tout, Tin);
    const int numInElems = NUM_SIMDVEC_ELEMENTS(Tin, SIMD_WIDTH);
    const int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);
    float facF = fac, offF = off;
    int outIdx = 0;
    for (int inVec = 0; inVec < numInVecs; inVec++)
      for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
	int outVec = outIdx / numOutElems;
	int outElem = outIdx % numOutElems;
	vecsOut[outVec][outElem] = _packs<Tout>
	  (_mul(_add(_extend<SIMDFloat>(vecsIn[inVec][inElem]), 
		     offF), 
		facF));
      }
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmuladd(const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double fac,
	  double off,
	  SIMDSerialVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    const int numInVecs = NUM_INPUT_SIMDVECS(Tout, Tin);
    const int numInElems = NUM_SIMDVEC_ELEMENTS(Tin, SIMD_WIDTH);
    const int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);
    float facF = fac, offF = off;
    int outIdx = 0;
    for (int inVec = 0; inVec < numInVecs; inVec++)
      for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
	int outVec = outIdx / numOutElems;
	int outElem = outIdx % numOutElems;
	vecsOut[outVec][outElem] = _packs<Tout>
	  (_add(_mul(_extend<SIMDFloat>(vecsIn[inVec][inElem]), 
		     facF), 
		offF));
      }
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fwaddmul(const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsIn1,
	   const SIMDSerialVec<Tin, SIMD_WIDTH> *const vecsIn2,
	   double w,
	   double fac,
	   SIMDSerialVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    const int numInVecs = NUM_INPUT_SIMDVECS(Tout, Tin);
    const int numInElems = NUM_SIMDVEC_ELEMENTS(Tin, SIMD_WIDTH);
    const int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);
    float wF = w;
    float facF = fac;
    int outIdx = 0;
    for (int inVec = 0; inVec < numInVecs; inVec++)
      for (int inElem = 0; inElem < numInElems; inElem++, outIdx++) {
	int outVec = outIdx / numOutElems;
	int outElem = outIdx % numOutElems;
	float inF1 = _extend<SIMDFloat>(vecsIn1[inVec][inElem]);
	float inF2 = _extend<SIMDFloat>(vecsIn2[inVec][inElem]);
	vecsOut[outVec][outElem] = _packs<Tout>
	  (_mul(facF, 
		_add(inF2, 
		     _mul(wF, 
			  _sub(inF1, inF2)))));
      }
  }
  template <typename T ,int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  hadd(const SIMDSerialVec<T,SIMD_WIDTH> *const v)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = hadd(v[i]);
    return c;
  }

  template <typename T ,int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  hadds(const SIMDSerialVec<T,SIMD_WIDTH> *const v)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = hadds(v[i]);
    return c;
  }

  template <typename T ,int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  hsub(const SIMDSerialVec<T,SIMD_WIDTH> *const v)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = hsub(v[i]);
    return c;
  }

  template <typename T ,int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  hsubs(const SIMDSerialVec<T,SIMD_WIDTH> *const v)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = hsubs(v[i]);
    return c;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose(const SIMDSerialVec<T,SIMD_WIDTH> *const inRows,
	    SIMDSerialVec<T,SIMD_WIDTH> *const outRows)
  {
    const int n = SIMDSerialVec<T,SIMD_WIDTH>::elements;
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
	outRows[i][j] = inRows[j][i];
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose2(const SIMDSerialVec<T,SIMD_WIDTH> *const inRows,
	     SIMDSerialVec<T,SIMD_WIDTH> *const outRows)
  {
    // same as transpose
    transpose(inRows, outRows);
  }  
  
  // synonymous to avg
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  avgru(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    return avg(a, b);
  }

  // int types except SIMDInt
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH> 
  avgrd(const SIMDSerialVec<T, SIMD_WIDTH> &a,
	const SIMDSerialVec<T, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (SIMDInt(a[i]) + SIMDInt(b[i])) >> 1;
    return c;
  }

  // SIMDInt
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDInt, SIMD_WIDTH> 
  avgrd(const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &a,
	const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDInt, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = (uint64_t(a[i]) + uint64_t(b[i])) >> 1;
    return c;
  }

  // SIMDFloat
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  avgrd(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a,
	const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = daz(0.5f * (daz(a[i]) + daz(b[i])));
    return c;
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<Tout, SIMD_WIDTH>
  packs(const SIMDSerialVec<Tin, SIMD_WIDTH> *const a)
  {
    SIMDSerialVec<Tout, SIMD_WIDTH> c;
    const int numInVecs = sizeof(Tin) / sizeof(Tout);
    int k = 0;
    for (int i = 0; i < numInVecs; i++)
      for (int j = 0; j < SIMDSerialVec<Tin,SIMD_WIDTH>::elements; j++, k++) {
	Tin in = a[i][j];
	c[k] = _packs<Tout>(in);
      }
    return c;
  }

  // integer types
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  div2r0(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = T(::trunc(double(a[i]) / 2.0));
    return c;
  }

  // SIMDFloat
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  div2r0(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = daz(daz(a[i]) / 2.0f);
    return c;
  }
  
  // integer types
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<T, SIMD_WIDTH>
  div2rd(const SIMDSerialVec<T, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<T, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = T(std::floor(double(a[i]) / 2.0));
    return c;
  }

  // SIMDFloat
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
  div2rd(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    SIMDSerialVec<SIMDFloat, SIMD_WIDTH> c;
    for (int i = 0; i < c.elements; i++)
      c[i] = daz(daz(a[i]) / 2.0f);
    return c;
  }

} // namespace

#endif
