// ===========================================================================
//
// SIMDVecMaskImplIntel64.H --
// Mask class definitions and architecture specific functions
// for Intel 64 byte (512 bit)
// Author: Markus Vieth (Bielefeld University, mvieth@techfak.uni-bielefeld.de)
// Year of creation: 2019
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
// (C) Markus Vieth, Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

// 22. Jan 23 (Jonas Keller): moved internal implementations into internal
// namespace

#pragma once
#ifndef SIMD_VEC_MASK_IMPL_INTEL_64_H_
#define SIMD_VEC_MASK_IMPL_INTEL_64_H_

#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecBaseImplIntel64.H"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(SIMDVEC_INTEL_ENABLE) && defined(_SIMD_VEC_64_AVAIL_) &&           \
  !defined(SIMDVEC_SANDBOX)

namespace simd {
#define CLASS_MASK(TYPE, MASK_SIZE)                                            \
  template <>                                                                  \
  class Mask<TYPE, 64>                                                         \
  {                                                                            \
    __mmask##MASK_SIZE k;                                                      \
                                                                               \
  public:                                                                      \
    Mask()                                                                     \
    {                                                                          \
      k = 0;                                                                   \
    }                                                                          \
    SIMD_INLINE Mask(const __mmask##MASK_SIZE &x)                              \
    {                                                                          \
      k = x;                                                                   \
    }                                                                          \
    explicit SIMD_INLINE Mask(const Vec<TYPE, 64> &x)                          \
    {                                                                          \
      k = msb2int(x);                                                          \
    }                                                                          \
    Mask &operator=(const __mmask##MASK_SIZE &x)                               \
    {                                                                          \
      k = x;                                                                   \
      return *this;                                                            \
    }                                                                          \
    SIMD_INLINE operator __mmask##MASK_SIZE() const                            \
    {                                                                          \
      return k;                                                                \
    }                                                                          \
    explicit SIMD_INLINE operator Vec<TYPE, 64>() const                        \
    {                                                                          \
      return int2bits<TYPE, 64>(k);                                            \
    }                                                                          \
    SIMD_INLINE bool operator[](const uint8_t i) const                         \
    {                                                                          \
      return ((1lu << i) & k) != 0;                                            \
    }                                                                          \
    SIMD_INLINE bool operator==(const Mask<TYPE, 64> &x) const                 \
    {                                                                          \
      return k == x.k;                                                         \
    }                                                                          \
  };

#ifdef __AVX512BW__
CLASS_MASK(Byte, 64)
CLASS_MASK(SignedByte, 64)
CLASS_MASK(Word, 32)
CLASS_MASK(Short, 32)
#endif
CLASS_MASK(Int, 16)
CLASS_MASK(Float, 16)
CLASS_MASK(Long, 8)
CLASS_MASK(Double, 8)

namespace internal {
namespace mask {
#define MASK_SOP(OP, TYPE, SUF)                                                \
  static SIMD_INLINE Vec<TYPE, 64> mask_##OP(                                  \
    const Vec<TYPE, 64> &src, const Mask<TYPE, 64> &k, const Vec<TYPE, 64> &a) \
  {                                                                            \
    return _mm512_mask_##OP##_##SUF(src, k, a);                                \
  }

#define MASKZ_SOP(OP, TYPE, SUF)                                               \
  static SIMD_INLINE Vec<TYPE, 64> maskz_##OP(const Mask<TYPE, 64> &k,         \
                                              const Vec<TYPE, 64> &a)          \
  {                                                                            \
    return _mm512_maskz_##OP##_##SUF(k, a);                                    \
  }

// For operations with one argument. OP is the name of the operation (e.g. add,
// sub, mul), TYPE is the typename (e.g. Word, Float), and SUF is the
// suffix of the intrinsic (e.g. epi8, epi16, ps).
#define GENERATE_SOP(OP, TYPE, SUF)                                            \
  MASK_SOP(OP, TYPE, SUF) MASKZ_SOP(OP, TYPE, SUF)

#define MASK_DOP(OP, TYPE, SUF)                                                \
  static SIMD_INLINE Vec<TYPE, 64> mask_##OP(                                  \
    const Vec<TYPE, 64> &src, const Mask<TYPE, 64> &k, const Vec<TYPE, 64> &a, \
    const Vec<TYPE, 64> &b)                                                    \
  {                                                                            \
    return _mm512_mask_##OP##_##SUF(src, k, a, b);                             \
  }

#define MASKZ_DOP(OP, TYPE, SUF)                                               \
  static SIMD_INLINE Vec<TYPE, 64> maskz_##OP(                                 \
    const Mask<TYPE, 64> &k, const Vec<TYPE, 64> &a, const Vec<TYPE, 64> &b)   \
  {                                                                            \
    return _mm512_maskz_##OP##_##SUF(k, a, b);                                 \
  }

// For operations with two arguments. OP is the name of the operation (e.g. add,
// sub, mul), TYPE is the typename (e.g. Word, Float), and SUF is the
// suffix of the intrinsic (e.g. epi8, epi16, ps).
#define GENERATE_DOP(OP, TYPE, SUF)                                            \
  MASK_DOP(OP, TYPE, SUF) MASKZ_DOP(OP, TYPE, SUF)

// ---------------------------------------------------------------------------
// mask_ifelse v
// ---------------------------------------------------------------------------

// 29. Mar 23 (Jonas Keller): added explicit cast to __m512(i) register to avoid
// compiler errors (can't convert simd::Vec to __v64qi, etc...)

#define MASK_IFELSE(TYPE, SUF, REG)                                            \
  static SIMD_INLINE Vec<TYPE, 64> mask_ifelse(const Mask<TYPE, 64> &cond,     \
                                               const Vec<TYPE, 64> &a,         \
                                               const Vec<TYPE, 64> &b)         \
  {                                                                            \
    return (REG) _mm512_mask_blend_##SUF(cond, (REG) b, (REG) a);              \
  }

#ifdef __AVX512BW__
MASK_IFELSE(Byte, epi8, __m512i)
MASK_IFELSE(SignedByte, epi8, __m512i)
MASK_IFELSE(Word, epi16, __m512i)
MASK_IFELSE(Short, epi16, __m512i)
#endif
MASK_IFELSE(Int, epi32, __m512i)
MASK_IFELSE(Float, ps, __m512)
MASK_IFELSE(Long, epi64, __m512i)
MASK_IFELSE(Double, pd, __m512d)

// ---------------------------------------------------------------------------
// mask_ifelsezero (mask_ifelsezero(cond, a) is the same as mask_ifelse(cond, a,
// setzero()), but may have faster implementations)
// ---------------------------------------------------------------------------

#define MASK_IFELSEZERO(TYPE)                                                  \
  static SIMD_INLINE Vec<TYPE, 64> mask_ifelsezero(                            \
    const Mask<TYPE, 64> &cond, const Vec<TYPE, 64> &trueVal)                  \
  {                                                                            \
    return mask_ifelse(cond, trueVal, ::simd::setzero<TYPE, 64>());            \
  }

#ifdef __AVX512BW__
MASK_IFELSEZERO(Byte)
MASK_IFELSEZERO(SignedByte)
MASK_IFELSEZERO(Word)
MASK_IFELSEZERO(Short)
#endif
MASK_IFELSEZERO(Int)
MASK_IFELSEZERO(Float)
MASK_IFELSEZERO(Long)
MASK_IFELSEZERO(Double)

// ---------------------------------------------------------------------------
// reinterpret_mask v
// ---------------------------------------------------------------------------

// 06. Feb 23 (Jonas Keller): added reinterpret_mask

template <typename Tout, typename Tin>
static SIMD_INLINE Mask<Tout, 64> reinterpret_mask(const Mask<Tin, 64> &k)
{
  static_assert(sizeof(Tout) == sizeof(Tin), "");
  return Mask<Tout, 64>(k.k);
}

// ---------------------------------------------------------------------------
// masked convert (without changes in the number of of elements) v
// ---------------------------------------------------------------------------

// conversion with saturation; we wanted to have a fast solution that
// doesn't trigger the overflow which results in a negative two's
// complement result ("invalid int32": 0x80000000); therefore we clamp
// the positive values at the maximal positive float which is
// convertible to int32 without overflow (0x7fffffbf = 2147483520);
// negative values cannot overflow (they are clamped to invalid int
// which is the most negative int32)
SIMD_INLINE Vec<Int, 64> maskz_cvts(const Mask<Float, 64> &k,
                                    const Vec<Float, 64> &a)
{
  __m512 clip = _mm512_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
  return _mm512_maskz_cvtps_epi32(k, _mm512_maskz_min_ps(k, clip, a));
}

SIMD_INLINE Vec<Int, 64> mask_cvts(const Vec<Int, 64> &src,
                                   const Mask<Float, 64> &k,
                                   const Vec<Float, 64> &a)
{
  __m512 clip = _mm512_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
  return _mm512_mask_cvtps_epi32(src, k, _mm512_maskz_min_ps(k, clip, a));
}

// saturation is not necessary in this case
SIMD_INLINE Vec<Float, 64> maskz_cvts(const Mask<Int, 64> &k,
                                      const Vec<Int, 64> &a)
{
  return _mm512_maskz_cvtepi32_ps(k, a);
}

// saturation is not necessary in this case
SIMD_INLINE Vec<Float, 64> mask_cvts(const Vec<Float, 64> &src,
                                     const Mask<Int, 64> &k,
                                     const Vec<Int, 64> &a)
{
  return _mm512_mask_cvtepi32_ps(src, k, a);
}

// ---------------------------------------------------------------------------
// mask_set1 v
// ---------------------------------------------------------------------------

#define GENERATE_SET1(TYPE, SUF)                                               \
  static SIMD_INLINE Vec<TYPE, 64> mask_set1(                                  \
    const Vec<TYPE, 64> &src, const Mask<TYPE, 64> &k, const TYPE a)           \
  {                                                                            \
    return _mm512_mask_set1_##SUF(src, k, a);                                  \
  }                                                                            \
  static SIMD_INLINE Vec<TYPE, 64> maskz_set1(const Mask<TYPE, 64> &k,         \
                                              const TYPE a)                    \
  {                                                                            \
    return _mm512_maskz_set1_##SUF(k, a);                                      \
  }

#ifdef __AVX512BW__
GENERATE_SET1(Byte, epi8)
GENERATE_SET1(SignedByte, epi8)
GENERATE_SET1(Word, epi16)
GENERATE_SET1(Short, epi16)
#endif
GENERATE_SET1(Int, epi32)
GENERATE_SET1(Long, epi64)
// Workaround for Float, because there is no mask_set1_ps
static SIMD_INLINE Vec<Float, 64> mask_set1(const Vec<Float, 64> &src,
                                            const Mask<Float, 64> &k,
                                            const Float a)
{
  return _mm512_castsi512_ps(
    _mm512_mask_set1_epi32(_mm512_castps_si512(src), k, bit_cast<Int>(a)));
}
static SIMD_INLINE Vec<Float, 64> maskz_set1(const Mask<Float, 64> &k,
                                             const Float a)
{
  return _mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, bit_cast<Int>(a)));
}
// Workaround for Double, because there is no mask_set1_pd
static SIMD_INLINE Vec<Double, 64> mask_set1(const Vec<Double, 64> &src,
                                             const Mask<Double, 64> &k,
                                             const Double a)
{
  return _mm512_castsi512_pd(
    _mm512_mask_set1_epi64(_mm512_castpd_si512(src), k, bit_cast<Long>(a)));
}
static SIMD_INLINE Vec<Double, 64> maskz_set1(const Mask<Double, 64> &k,
                                              const Double a)
{
  return _mm512_castsi512_pd(_mm512_maskz_set1_epi64(k, bit_cast<Long>(a)));
}

// ---------------------------------------------------------------------------
// mask_load v
// ---------------------------------------------------------------------------

#define GENERATE_LOAD(NAME, TYPE, SUF)                                         \
  static SIMD_INLINE Vec<TYPE, 64> mask_load(                                  \
    const Vec<TYPE, 64> &src, const Mask<TYPE, 64> &k, const TYPE *const p)    \
  {                                                                            \
    /* AVX load and store instructions need alignment to 64 byte*/             \
    /* (lower 6 bit need to be zero) */                                        \
    SIMD_CHECK_ALIGNMENT(p, 64);                                               \
    return _mm512_mask_##NAME##_##SUF(src, k, p);                              \
  }                                                                            \
  static SIMD_INLINE Vec<TYPE, 64> maskz_load(const Mask<TYPE, 64> &k,         \
                                              const TYPE *const p)             \
  {                                                                            \
    /* AVX load and store instructions need alignment to 64 byte*/             \
    /* (lower 6 bit need to be zero) */                                        \
    SIMD_CHECK_ALIGNMENT(p, 64);                                               \
    return _mm512_maskz_##NAME##_##SUF(k, p);                                  \
  }

#ifdef __AVX512BW__
// there is no aligned load for 8 and 16 bit types, so we use loadu
GENERATE_LOAD(loadu, Byte, epi8)
GENERATE_LOAD(loadu, SignedByte, epi8)
GENERATE_LOAD(loadu, Word, epi16)
GENERATE_LOAD(loadu, Short, epi16)
#endif

GENERATE_LOAD(load, Int, epi32)
GENERATE_LOAD(load, Float, ps)
GENERATE_LOAD(load, Long, epi64)
GENERATE_LOAD(load, Double, pd)

// ---------------------------------------------------------------------------
// mask_loadu v
// ---------------------------------------------------------------------------

#define GENERATE_LOADU(TYPE, SUF)                                              \
  static SIMD_INLINE Vec<TYPE, 64> mask_loadu(                                 \
    const Vec<TYPE, 64> &src, const Mask<TYPE, 64> &k, const TYPE *const p)    \
  {                                                                            \
    return _mm512_mask_loadu_##SUF(src, k, p);                                 \
  }                                                                            \
  static SIMD_INLINE Vec<TYPE, 64> maskz_loadu(const Mask<TYPE, 64> &k,        \
                                               const TYPE *const p)            \
  {                                                                            \
    return _mm512_maskz_loadu_##SUF(k, p);                                     \
  }

#ifdef __AVX512BW__
GENERATE_LOADU(Byte, epi8)
GENERATE_LOADU(SignedByte, epi8)
GENERATE_LOADU(Word, epi16)
GENERATE_LOADU(Short, epi16)
#endif

GENERATE_LOADU(Int, epi32)
GENERATE_LOADU(Float, ps)
GENERATE_LOADU(Long, epi64)
GENERATE_LOADU(Double, pd)

// ---------------------------------------------------------------------------
// mask_store v
// ---------------------------------------------------------------------------

// There are no *_maskz_store_* intrinsics, only *_mask_store_* intrinsics

#define MASK_STORE(NAME, TYPE, SUF)                                            \
  static SIMD_INLINE void mask_store(TYPE *const p, const Mask<TYPE, 64> &k,   \
                                     const Vec<TYPE, 64> &a)                   \
  {                                                                            \
    /* AVX load and store instructions need alignment to 64 byte*/             \
    /* (lower 6 bit need to be zero) */                                        \
    SIMD_CHECK_ALIGNMENT(p, 64);                                               \
    return _mm512_mask_##NAME##_##SUF(p, k, a);                                \
  }

#ifdef __AVX512BW__
// there is no aligned store for 8 and 16 bit types, so we use storeu
MASK_STORE(storeu, Byte, epi8)
MASK_STORE(storeu, SignedByte, epi8)
MASK_STORE(storeu, Word, epi16)
MASK_STORE(storeu, Short, epi16)
#endif

MASK_STORE(store, Int, epi32)
MASK_STORE(store, Float, ps)
MASK_STORE(store, Long, epi64)
MASK_STORE(store, Double, pd)

// ---------------------------------------------------------------------------
// mask_storeu v
// ---------------------------------------------------------------------------

// There are no *_maskz_storeu_* intrinsics, only *_mask_storeu_* intrinsics

#define MASK_STOREU(TYPE, SUF)                                                 \
  static SIMD_INLINE void mask_storeu(TYPE *const p, const Mask<TYPE, 64> &k,  \
                                      const Vec<TYPE, 64> &a)                  \
  {                                                                            \
    return _mm512_mask_storeu_##SUF(p, k, a);                                  \
  }
#ifdef __AVX512BW__
MASK_STOREU(Byte, epi8)
MASK_STOREU(SignedByte, epi8)
MASK_STOREU(Word, epi16)
MASK_STOREU(Short, epi16)
#endif
MASK_STOREU(Int, epi32)
MASK_STOREU(Float, ps)
MASK_STOREU(Long, epi64)
MASK_STOREU(Double, pd)

// ---------------------------------------------------------------------------
// mask_add v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_DOP(add, Byte, epi8)
GENERATE_DOP(add, SignedByte, epi8)
GENERATE_DOP(add, Word, epi16)
GENERATE_DOP(add, Short, epi16)
#endif
GENERATE_DOP(add, Int, epi32)
GENERATE_DOP(add, Float, ps)
GENERATE_DOP(add, Long, epi64)
GENERATE_DOP(add, Double, pd)

// ---------------------------------------------------------------------------
// mask_adds v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_DOP(adds, Byte, epu8)
GENERATE_DOP(adds, SignedByte, epi8)
GENERATE_DOP(adds, Word, epu16)
GENERATE_DOP(adds, Short, epi16)
#endif

// 09. Mar 23 (Jonas Keller): removed non saturating version of adds for Int and
// Float, use the emulated versions in SIMDVecMaskImplEmu.H instead

// ---------------------------------------------------------------------------
// mask_sub v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_DOP(sub, Byte, epi8)
GENERATE_DOP(sub, SignedByte, epi8)
GENERATE_DOP(sub, Word, epi16)
GENERATE_DOP(sub, Short, epi16)
#endif
GENERATE_DOP(sub, Int, epi32)
GENERATE_DOP(sub, Float, ps)
GENERATE_DOP(sub, Long, epi64)
GENERATE_DOP(sub, Double, pd)

// ---------------------------------------------------------------------------
// mask_subs v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_DOP(subs, Byte, epu8)
GENERATE_DOP(subs, SignedByte, epi8)
GENERATE_DOP(subs, Word, epu16)
GENERATE_DOP(subs, Short, epi16)
#endif

// 09. Mar 23 (Jonas Keller): removed non saturating version of subs for Int and
// Float, use the emulated versions in SIMDVecMaskImplEmu.H instead

// ---------------------------------------------------------------------------
// mask_mul v
// ---------------------------------------------------------------------------

GENERATE_DOP(mul, Float, ps)
GENERATE_DOP(mul, Double, pd)

// ---------------------------------------------------------------------------
// mask_div v
// ---------------------------------------------------------------------------

GENERATE_DOP(div, Float, ps)
GENERATE_DOP(div, Double, pd)

// ---------------------------------------------------------------------------
// masked ceil, floor, round, truncate v
// ---------------------------------------------------------------------------

// 10. Apr 23 (Jonas Keller): added versions for integer types

// versions for integer types do nothing:

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> mask_ceil(const Vec<T, 64> &src,
                                        const Mask<T, 64> &k,
                                        const Vec<T, 64> &a)
{
  return mask_ifelse(k, a, src);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> maskz_ceil(const Mask<T, 64> &k,
                                         const Vec<T, 64> &a)
{
  return mask_ifelsezero(k, a);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> mask_floor(const Vec<T, 64> &src,
                                         const Mask<T, 64> &k,
                                         const Vec<T, 64> &a)
{
  return mask_ifelse(k, a, src);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> maskz_floor(const Mask<T, 64> &k,
                                          const Vec<T, 64> &a)
{
  return mask_ifelsezero(k, a);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> mask_round(const Vec<T, 64> &src,
                                         const Mask<T, 64> &k,
                                         const Vec<T, 64> &a)
{
  return mask_ifelse(k, a, src);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> maskz_round(const Mask<T, 64> &k,
                                          const Vec<T, 64> &a)
{
  return mask_ifelsezero(k, a);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> mask_truncate(const Vec<T, 64> &src,
                                            const Mask<T, 64> &k,
                                            const Vec<T, 64> &a)
{
  return mask_ifelse(k, a, src);
}

template <typename T, SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> maskz_truncate(const Mask<T, 64> &k,
                                             const Vec<T, 64> &a)
{
  return mask_ifelsezero(k, a);
}

// Float versions:

static SIMD_INLINE Vec<Float, 64> mask_ceil(const Vec<Float, 64> &src,
                                            const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a)
{
  return _mm512_mask_roundscale_ps(src, k, a,
                                   _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> maskz_ceil(const Mask<Float, 64> &k,
                                             const Vec<Float, 64> &a)
{
  return _mm512_maskz_roundscale_ps(k, a,
                                    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> mask_floor(const Vec<Float, 64> &src,
                                             const Mask<Float, 64> &k,
                                             const Vec<Float, 64> &a)
{
  return _mm512_mask_roundscale_ps(src, k, a,
                                   _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> maskz_floor(const Mask<Float, 64> &k,
                                              const Vec<Float, 64> &a)
{
  return _mm512_maskz_roundscale_ps(k, a,
                                    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> mask_round(const Vec<Float, 64> &src,
                                             const Mask<Float, 64> &k,
                                             const Vec<Float, 64> &a)
{
  return _mm512_mask_roundscale_ps(
    src, k, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> maskz_round(const Mask<Float, 64> &k,
                                              const Vec<Float, 64> &a)
{
  return _mm512_maskz_roundscale_ps(
    k, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> mask_truncate(const Vec<Float, 64> &src,
                                                const Mask<Float, 64> &k,
                                                const Vec<Float, 64> &a)
{
  return _mm512_mask_roundscale_ps(src, k, a,
                                   _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> maskz_truncate(const Mask<Float, 64> &k,
                                                 const Vec<Float, 64> &a)
{
  return _mm512_maskz_roundscale_ps(k, a,
                                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

// Double versions:

static SIMD_INLINE Vec<Double, 64> mask_ceil(const Vec<Double, 64> &src,
                                             const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a)
{
  return _mm512_mask_roundscale_pd(src, k, a,
                                   _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> maskz_ceil(const Mask<Double, 64> &k,
                                              const Vec<Double, 64> &a)
{
  return _mm512_maskz_roundscale_pd(k, a,
                                    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> mask_floor(const Vec<Double, 64> &src,
                                              const Mask<Double, 64> &k,
                                              const Vec<Double, 64> &a)
{
  return _mm512_mask_roundscale_pd(src, k, a,
                                   _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> maskz_floor(const Mask<Double, 64> &k,
                                               const Vec<Double, 64> &a)
{
  return _mm512_maskz_roundscale_pd(k, a,
                                    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> mask_round(const Vec<Double, 64> &src,
                                              const Mask<Double, 64> &k,
                                              const Vec<Double, 64> &a)
{
  return _mm512_mask_roundscale_pd(
    src, k, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> maskz_round(const Mask<Double, 64> &k,
                                               const Vec<Double, 64> &a)
{
  return _mm512_maskz_roundscale_pd(
    k, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> mask_truncate(const Vec<Double, 64> &src,
                                                 const Mask<Double, 64> &k,
                                                 const Vec<Double, 64> &a)
{
  return _mm512_mask_roundscale_pd(src, k, a,
                                   _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> maskz_truncate(const Mask<Double, 64> &k,
                                                  const Vec<Double, 64> &a)
{
  return _mm512_maskz_roundscale_pd(k, a,
                                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

// ---------------------------------------------------------------------------
// masked elementary mathematical functions v
// ---------------------------------------------------------------------------

// masked estimate of a reciprocal
// NOTE: this has better precision than SSE and AVX versions!
static SIMD_INLINE Vec<Float, 64> mask_rcp(const Vec<Float, 64> &src,
                                           const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a)
{
  return _mm512_mask_rcp14_ps(src, k, a);
}

static SIMD_INLINE Vec<Float, 64> maskz_rcp(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a)
{
  return _mm512_maskz_rcp14_ps(k, a);
}

static SIMD_INLINE Vec<Double, 64> mask_rcp(const Vec<Double, 64> &src,
                                            const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a)
{
  return _mm512_mask_rcp14_pd(src, k, a);
}

static SIMD_INLINE Vec<Double, 64> maskz_rcp(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a)
{
  return _mm512_maskz_rcp14_pd(k, a);
}

// masked estimate of reverse square root
// NOTE: this has better precision than SSE and AVX versions!
static SIMD_INLINE Vec<Float, 64> mask_rsqrt(const Vec<Float, 64> &src,
                                             const Mask<Float, 64> &k,
                                             const Vec<Float, 64> &a)
{
  return _mm512_mask_rsqrt14_ps(src, k, a);
}

static SIMD_INLINE Vec<Float, 64> maskz_rsqrt(const Mask<Float, 64> &k,
                                              const Vec<Float, 64> &a)
{
  return _mm512_maskz_rsqrt14_ps(k, a);
}

static SIMD_INLINE Vec<Double, 64> mask_rsqrt(const Vec<Double, 64> &src,
                                              const Mask<Double, 64> &k,
                                              const Vec<Double, 64> &a)
{
  return _mm512_mask_rsqrt14_pd(src, k, a);
}

static SIMD_INLINE Vec<Double, 64> maskz_rsqrt(const Mask<Double, 64> &k,
                                               const Vec<Double, 64> &a)
{
  return _mm512_maskz_rsqrt14_pd(k, a);
}

// masked square root
GENERATE_SOP(sqrt, Float, ps)
GENERATE_SOP(sqrt, Double, pd)

// ---------------------------------------------------------------------------
// masked_abs v
// ---------------------------------------------------------------------------

// 25. Mar 25 (Jonas Keller): added masked abs for unsigned integers

// unsigned integers: do nothing
template <typename T, SIMD_ENABLE_IF(std::is_unsigned<T>::value
                                       &&std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> mask_abs(const Vec<T, 64> &src,
                                       const Mask<T, 64> &k,
                                       const Vec<T, 64> &a)
{
  return mask_ifelse(k, a, src);
}

template <typename T, SIMD_ENABLE_IF(std::is_unsigned<T>::value
                                       &&std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> maskz_abs(const Mask<T, 64> &k,
                                        const Vec<T, 64> &a)
{
  return mask_ifelsezero(k, a);
}

#ifdef __AVX512BW__
GENERATE_SOP(abs, SignedByte, epi8)
GENERATE_SOP(abs, Short, epi16)
#endif
GENERATE_SOP(abs, Int, epi32)
GENERATE_SOP(abs, Long, epi64)

MASK_SOP(abs, Float, ps)

// There is no _mm512_maskz_abs_ps
static SIMD_INLINE Vec<Float, 64> maskz_abs(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a)
{
  return _mm512_mask_abs_ps(::simd::setzero<Float, 64>(), k, a);
}

MASK_SOP(abs, Double, pd)

// There is no _mm512_maskz_abs_pd
static SIMD_INLINE Vec<Double, 64> maskz_abs(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a)
{
  return _mm512_mask_abs_pd(::simd::setzero<Double, 64>(), k, a);
}

// ---------------------------------------------------------------------------
// mask_and v
// ---------------------------------------------------------------------------

// there is no _mm512_mask_and_epi8 or _mm512_mask_and_epi16
GENERATE_DOP(and, Int, epi32)
GENERATE_DOP(and, Long, epi64)
#ifdef __AVX512DQ__
GENERATE_DOP(and, Float, ps)
GENERATE_DOP(and, Double, pd)
#else
// Workaround with the epi32/64-versions and casts
static SIMD_INLINE Vec<Float, 64> mask_and(const Vec<Float, 64> &src,
                                           const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(_mm512_mask_and_epi32(_mm512_castps_si512(src), k,
                                                   _mm512_castps_si512(a),
                                                   _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Float, 64> maskz_and(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a,
                                            const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(
    _mm512_maskz_and_epi32(k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> mask_and(const Vec<Double, 64> &src,
                                            const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(_mm512_mask_and_epi64(_mm512_castpd_si512(src), k,
                                                   _mm512_castpd_si512(a),
                                                   _mm512_castpd_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> maskz_and(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a,
                                             const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(
    _mm512_maskz_and_epi64(k, _mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
#endif

// ---------------------------------------------------------------------------
// mask_or v
// ---------------------------------------------------------------------------

// there is no _mm512_mask_or_epi8 or _mm512_mask_or_epi16
GENERATE_DOP(or, Int, epi32)
GENERATE_DOP(or, Long, epi64)
#ifdef __AVX512DQ__
GENERATE_DOP(or, Float, ps)
GENERATE_DOP(or, Double, pd)
#else
// Workaround with the epi32/64-versions and casts
static SIMD_INLINE Vec<Float, 64> mask_or(const Vec<Float, 64> &src,
                                          const Mask<Float, 64> &k,
                                          const Vec<Float, 64> &a,
                                          const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(_mm512_mask_or_epi32(_mm512_castps_si512(src), k,
                                                  _mm512_castps_si512(a),
                                                  _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Float, 64> maskz_or(const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(
    _mm512_maskz_or_epi32(k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> mask_or(const Vec<Double, 64> &src,
                                           const Mask<Double, 64> &k,
                                           const Vec<Double, 64> &a,
                                           const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(_mm512_mask_or_epi64(_mm512_castpd_si512(src), k,
                                                  _mm512_castpd_si512(a),
                                                  _mm512_castpd_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> maskz_or(const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(
    _mm512_maskz_or_epi64(k, _mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
#endif

// ---------------------------------------------------------------------------
// mask_andnot v
// ---------------------------------------------------------------------------

// there is no _mm512_mask_andnot_epi8 or _mm512_mask_andnot_epi16
GENERATE_DOP(andnot, Int, epi32)
GENERATE_DOP(andnot, Long, epi64)
#ifdef __AVX512DQ__
GENERATE_DOP(andnot, Float, ps)
GENERATE_DOP(andnot, Double, pd)
#else
// Workaround with the epi32/64-versions and casts
static SIMD_INLINE Vec<Float, 64> mask_andnot(const Vec<Float, 64> &src,
                                              const Mask<Float, 64> &k,
                                              const Vec<Float, 64> &a,
                                              const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(_mm512_mask_andnot_epi32(_mm512_castps_si512(src),
                                                      k, _mm512_castps_si512(a),
                                                      _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Float, 64> maskz_andnot(const Mask<Float, 64> &k,
                                               const Vec<Float, 64> &a,
                                               const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(_mm512_maskz_andnot_epi32(
    k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> mask_andnot(const Vec<Double, 64> &src,
                                               const Mask<Double, 64> &k,
                                               const Vec<Double, 64> &a,
                                               const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(_mm512_mask_andnot_epi64(_mm512_castpd_si512(src),
                                                      k, _mm512_castpd_si512(a),
                                                      _mm512_castpd_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> maskz_andnot(const Mask<Double, 64> &k,
                                                const Vec<Double, 64> &a,
                                                const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(_mm512_maskz_andnot_epi64(
    k, _mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
#endif

// ---------------------------------------------------------------------------
// mask_xor v
// ---------------------------------------------------------------------------

// there is no _mm512_mask_xor_epi8 or _mm512_mask_xor_epi16
GENERATE_DOP(xor, Int, epi32)
GENERATE_DOP(xor, Long, epi64)
#ifdef __AVX512DQ__
GENERATE_DOP(xor, Float, ps)
GENERATE_DOP(xor, Double, pd)
#else
// Workaround with the epi32/64-versions and casts
static SIMD_INLINE Vec<Float, 64> mask_xor(const Vec<Float, 64> &src,
                                           const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(src), k,
                                                   _mm512_castps_si512(a),
                                                   _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Float, 64> maskz_xor(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a,
                                            const Vec<Float, 64> &b)
{
  return _mm512_castsi512_ps(
    _mm512_maskz_xor_epi32(k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> mask_xor(const Vec<Double, 64> &src,
                                            const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(_mm512_mask_xor_epi64(_mm512_castpd_si512(src), k,
                                                   _mm512_castpd_si512(a),
                                                   _mm512_castpd_si512(b)));
}

static SIMD_INLINE Vec<Double, 64> maskz_xor(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a,
                                             const Vec<Double, 64> &b)
{
  return _mm512_castsi512_pd(
    _mm512_maskz_xor_epi64(k, _mm512_castpd_si512(a), _mm512_castpd_si512(b)));
}
#endif

// ---------------------------------------------------------------------------
// mask_not v
// ---------------------------------------------------------------------------

// 08. Apr 23 (Jonas Keller): added mask_not and maskz_not

// There is no masked "not"-intrinsic, so use the masked xor with all ones

// there are no masked xor intrinsics for epi8 and epi16

// Int
static SIMD_INLINE Vec<Int, 64> mask_not(const Vec<Int, 64> &src,
                                         const Mask<Int, 64> &k,
                                         const Vec<Int, 64> &a)
{
  return _mm512_mask_xor_epi32(src, k, a, _mm512_set1_epi32(-1));
}
static SIMD_INLINE Vec<Int, 64> maskz_not(const Mask<Int, 64> &k,
                                          const Vec<Int, 64> &a)
{
  return _mm512_maskz_xor_epi32(k, a, _mm512_set1_epi32(-1));
}

// Long
static SIMD_INLINE Vec<Long, 64> mask_not(const Vec<Long, 64> &src,
                                          const Mask<Long, 64> &k,
                                          const Vec<Long, 64> &a)
{
  return _mm512_mask_xor_epi64(src, k, a, _mm512_set1_epi64(-1));
}
static SIMD_INLINE Vec<Long, 64> maskz_not(const Mask<Long, 64> &k,
                                           const Vec<Long, 64> &a)
{
  return _mm512_maskz_xor_epi64(k, a, _mm512_set1_epi64(-1));
}
#ifdef __AVX512DQ__
// Float
static SIMD_INLINE Vec<Float, 64> mask_not(const Vec<Float, 64> &src,
                                           const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a)
{
  return _mm512_mask_xor_ps(src, k, a,
                            _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
}
static SIMD_INLINE Vec<Float, 64> maskz_not(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a)
{
  return _mm512_maskz_xor_ps(k, a, _mm512_castsi512_ps(_mm512_set1_epi32(-1)));
}

// Double
static SIMD_INLINE Vec<Double, 64> mask_not(const Vec<Double, 64> &src,
                                            const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a)
{
  return _mm512_mask_xor_pd(src, k, a,
                            _mm512_castsi512_pd(_mm512_set1_epi64(-1)));
}
static SIMD_INLINE Vec<Double, 64> maskz_not(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a)
{
  return _mm512_maskz_xor_pd(k, a, _mm512_castsi512_pd(_mm512_set1_epi64(-1)));
}
#else
// Workaround with the epi32/64-versions and casts
static SIMD_INLINE Vec<Float, 64> mask_not(const Vec<Float, 64> &src,
                                           const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a)
{
  return _mm512_castsi512_ps(_mm512_mask_xor_epi32(_mm512_castps_si512(src), k,
                                                   _mm512_castps_si512(a),
                                                   _mm512_set1_epi32(-1)));
}

static SIMD_INLINE Vec<Float, 64> maskz_not(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a)
{
  return _mm512_castsi512_ps(
    _mm512_maskz_xor_epi32(k, _mm512_castps_si512(a), _mm512_set1_epi32(-1)));
}

static SIMD_INLINE Vec<Double, 64> mask_not(const Vec<Double, 64> &src,
                                            const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a)
{
  return _mm512_castsi512_pd(_mm512_mask_xor_epi64(_mm512_castpd_si512(src), k,
                                                   _mm512_castpd_si512(a),
                                                   _mm512_set1_epi64(-1)));
}

static SIMD_INLINE Vec<Double, 64> maskz_not(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a)
{
  return _mm512_castsi512_pd(
    _mm512_maskz_xor_epi64(k, _mm512_castpd_si512(a), _mm512_set1_epi64(-1)));
}
#endif

// ---------------------------------------------------------------------------
// mask_neg (negate = two's complement or unary minus), only signed types v
// ---------------------------------------------------------------------------

#define GENERATE_NEG(TYPE, SUF)                                                \
  static SIMD_INLINE Vec<TYPE, 64> mask_neg(                                   \
    const Vec<TYPE, 64> &src, const Mask<TYPE, 64> &k, const Vec<TYPE, 64> &a) \
  {                                                                            \
    return _mm512_mask_sub_##SUF(src, k, setzero<TYPE, 64>(), a);              \
  }                                                                            \
  static SIMD_INLINE Vec<TYPE, 64> maskz_neg(const Mask<TYPE, 64> &k,          \
                                             const Vec<TYPE, 64> &a)           \
  {                                                                            \
    return _mm512_maskz_sub_##SUF(k, setzero<TYPE, 64>(), a);                  \
  }

#ifdef __AVX512BW__
GENERATE_NEG(SignedByte, epi8)
GENERATE_NEG(Short, epi16)
#endif
GENERATE_NEG(Int, epi32)
GENERATE_NEG(Float, ps)
GENERATE_NEG(Long, epi64)
GENERATE_NEG(Double, pd)

// ---------------------------------------------------------------------------
// mask_min v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_DOP(min, Byte, epu8)
GENERATE_DOP(min, SignedByte, epi8)
GENERATE_DOP(min, Word, epu16)
GENERATE_DOP(min, Short, epi16)
#endif
GENERATE_DOP(min, Int, epi32)
GENERATE_DOP(min, Float, ps)
GENERATE_DOP(min, Long, epi64)
GENERATE_DOP(min, Double, pd)

// ---------------------------------------------------------------------------
// mask_max v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_DOP(max, Byte, epu8)
GENERATE_DOP(max, SignedByte, epi8)
GENERATE_DOP(max, Word, epu16)
GENERATE_DOP(max, Short, epi16)
#endif
GENERATE_DOP(max, Int, epi32)
GENERATE_DOP(max, Float, ps)
GENERATE_DOP(max, Long, epi64)
GENERATE_DOP(max, Double, pd)

// ---------------------------------------------------------------------------
// masked srai (16,32,64 only) v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
template <size_t COUNT>
static SIMD_INLINE Vec<Word, 64> mask_srai(const Vec<Word, 64> &src,
                                           const Mask<Word, 64> &k,
                                           const Vec<Word, 64> &a)
{
  return _mm512_mask_srai_epi16(src, k, a, vec::min(COUNT, 15ul));
}

template <size_t COUNT>
static SIMD_INLINE Vec<Word, 64> maskz_srai(const Mask<Word, 64> &k,
                                            const Vec<Word, 64> &a)
{
  return _mm512_maskz_srai_epi16(k, a, vec::min(COUNT, 15ul));
}

template <size_t COUNT>
static SIMD_INLINE Vec<Short, 64> mask_srai(const Vec<Short, 64> &src,
                                            const Mask<Short, 64> &k,
                                            const Vec<Short, 64> &a)
{
  return _mm512_mask_srai_epi16(src, k, a, vec::min(COUNT, 15ul));
}

template <size_t COUNT>
static SIMD_INLINE Vec<Short, 64> maskz_srai(const Mask<Short, 64> &k,
                                             const Vec<Short, 64> &a)
{
  return _mm512_maskz_srai_epi16(k, a, vec::min(COUNT, 15ul));
}

#endif

template <size_t COUNT>
static SIMD_INLINE Vec<Int, 64> mask_srai(const Vec<Int, 64> &src,
                                          const Mask<Int, 64> &k,
                                          const Vec<Int, 64> &a)
{
  return _mm512_mask_srai_epi32(src, k, a, vec::min(COUNT, 31ul));
}

template <size_t COUNT>
static SIMD_INLINE Vec<Int, 64> maskz_srai(const Mask<Int, 64> &k,
                                           const Vec<Int, 64> &a)
{
  return _mm512_maskz_srai_epi32(k, a, vec::min(COUNT, 31ul));
}

template <size_t COUNT>
static SIMD_INLINE Vec<Long, 64> mask_srai(const Vec<Long, 64> &src,
                                           const Mask<Long, 64> &k,
                                           const Vec<Long, 64> &a)
{
  return _mm512_mask_srai_epi64(src, k, a, vec::min(COUNT, 63ul));
}

template <size_t COUNT>
static SIMD_INLINE Vec<Long, 64> maskz_srai(const Mask<Long, 64> &k,
                                            const Vec<Long, 64> &a)
{
  return _mm512_maskz_srai_epi64(k, a, vec::min(COUNT, 63ul));
}

// ---------------------------------------------------------------------------
// masked srli v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
template <size_t COUNT>
static SIMD_INLINE Vec<Word, 64> mask_srli(const Vec<Word, 64> &src,
                                           const Mask<Word, 64> &k,
                                           const Vec<Word, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_mask_srli_epi16(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi16(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Word, 64> maskz_srli(const Mask<Word, 64> &k,
                                            const Vec<Word, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_maskz_srli_epi16(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Short, 64> mask_srli(const Vec<Short, 64> &src,
                                            const Mask<Short, 64> &k,
                                            const Vec<Short, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_mask_srli_epi16(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi16(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Short, 64> maskz_srli(const Mask<Short, 64> &k,
                                             const Vec<Short, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_maskz_srli_epi16(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

#endif

template <size_t COUNT>
static SIMD_INLINE Vec<Int, 64> mask_srli(const Vec<Int, 64> &src,
                                          const Mask<Int, 64> &k,
                                          const Vec<Int, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 32) {
    return _mm512_mask_srli_epi32(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi32(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Int, 64> maskz_srli(const Mask<Int, 64> &k,
                                           const Vec<Int, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 32) {
    return _mm512_maskz_srli_epi32(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Long, 64> mask_srli(const Vec<Long, 64> &src,
                                           const Mask<Long, 64> &k,
                                           const Vec<Long, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 64) {
    return _mm512_mask_srli_epi64(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi64(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Long, 64> maskz_srli(const Mask<Long, 64> &k,
                                            const Vec<Long, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 64) {
    return _mm512_maskz_srli_epi64(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

// ---------------------------------------------------------------------------
// masked slli v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
template <size_t COUNT>
static SIMD_INLINE Vec<Word, 64> mask_slli(const Vec<Word, 64> &src,
                                           const Mask<Word, 64> &k,
                                           const Vec<Word, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_mask_slli_epi16(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi16(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Word, 64> maskz_slli(const Mask<Word, 64> &k,
                                            const Vec<Word, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_maskz_slli_epi16(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Short, 64> mask_slli(const Vec<Short, 64> &src,
                                            const Mask<Short, 64> &k,
                                            const Vec<Short, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_mask_slli_epi16(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi16(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Short, 64> maskz_slli(const Mask<Short, 64> &k,
                                             const Vec<Short, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 16) {
    return _mm512_maskz_slli_epi16(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

#endif

template <size_t COUNT>
static SIMD_INLINE Vec<Int, 64> mask_slli(const Vec<Int, 64> &src,
                                          const Mask<Int, 64> &k,
                                          const Vec<Int, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 32) {
    return _mm512_mask_slli_epi32(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi32(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Int, 64> maskz_slli(const Mask<Int, 64> &k,
                                           const Vec<Int, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 32) {
    return _mm512_maskz_slli_epi32(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Long, 64> mask_slli(const Vec<Long, 64> &src,
                                           const Mask<Long, 64> &k,
                                           const Vec<Long, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 64) {
    return _mm512_mask_slli_epi64(src, k, a, COUNT);
  } else {
    return _mm512_mask_blend_epi64(k, src, _mm512_setzero_si512());
  }
}

template <size_t COUNT>
static SIMD_INLINE Vec<Long, 64> maskz_slli(const Mask<Long, 64> &k,
                                            const Vec<Long, 64> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < 64) {
    return _mm512_maskz_slli_epi64(k, a, COUNT);
  } else {
    return _mm512_setzero_si512();
  }
}

// 05. Aug 22 (Jonas Keller):
// Improved implementation of masked hadd, hadds, hsub and hsubs,
// implementation uses masked add/adds/sub/subs directly now instead of
// wrapping hadd, hadds, hsub and hsubs with a mask_ifelse(zero).
// Byte and SignedByte are now supported as well.

// ---------------------------------------------------------------------------
// masked hadd v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> mask_hadd(const Vec<T, 64> &src,
                                        const Mask<T, 64> &k,
                                        const Vec<T, 64> &a,
                                        const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return mask_add(src, k, x, y);
}

template <typename T>
static SIMD_INLINE Vec<T, 64> maskz_hadd(const Mask<T, 64> &k,
                                         const Vec<T, 64> &a,
                                         const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return maskz_add(k, x, y);
}

// ---------------------------------------------------------------------------
// masked hadds v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> mask_hadds(const Vec<T, 64> &src,
                                         const Mask<T, 64> &k,
                                         const Vec<T, 64> &a,
                                         const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return mask_adds(src, k, x, y);
}

template <typename T>
static SIMD_INLINE Vec<T, 64> maskz_hadds(const Mask<T, 64> &k,
                                          const Vec<T, 64> &a,
                                          const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return maskz_adds(k, x, y);
}

// ---------------------------------------------------------------------------
// masked hsub v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> mask_hsub(const Vec<T, 64> &src,
                                        const Mask<T, 64> &k,
                                        const Vec<T, 64> &a,
                                        const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return mask_sub(src, k, x, y);
}

template <typename T>
static SIMD_INLINE Vec<T, 64> maskz_hsub(const Mask<T, 64> &k,
                                         const Vec<T, 64> &a,
                                         const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return maskz_sub(k, x, y);
}

// ---------------------------------------------------------------------------
// masked hsubs v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> mask_hsubs(const Vec<T, 64> &src,
                                         const Mask<T, 64> &k,
                                         const Vec<T, 64> &a,
                                         const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return mask_subs(src, k, x, y);
}

template <typename T>
static SIMD_INLINE Vec<T, 64> maskz_hsubs(const Mask<T, 64> &k,
                                          const Vec<T, 64> &a,
                                          const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return maskz_subs(k, x, y);
}

// 16. Oct 22 (Jonas Keller): added overloaded versions of mask_cmp* functions
// that only take two vector parameters and no mask parameter

#define GENERATE_CMP(OP, TYPE, SUF)                                            \
  static SIMD_INLINE Mask<TYPE, 64> mask_##OP(                                 \
    const Mask<TYPE, 64> &k, const Vec<TYPE, 64> &a, const Vec<TYPE, 64> &b)   \
  {                                                                            \
    return _mm512_mask_##OP##_##SUF##_mask(k, a, b);                           \
  }                                                                            \
  static SIMD_INLINE Mask<TYPE, 64> mask_##OP(const Vec<TYPE, 64> &a,          \
                                              const Vec<TYPE, 64> &b)          \
  {                                                                            \
    return _mm512_##OP##_##SUF##_mask(a, b);                                   \
  }

#define GENERATE_CMP_WITH_GENERALIZED_FCT(OP, TYPE, SUF, IMM8)                 \
  static SIMD_INLINE Mask<TYPE, 64> mask_##OP(                                 \
    const Mask<TYPE, 64> &k, const Vec<TYPE, 64> &a, const Vec<TYPE, 64> &b)   \
  {                                                                            \
    return _mm512_mask_cmp_##SUF##_mask(k, a, b, IMM8);                        \
  }                                                                            \
  static SIMD_INLINE Mask<TYPE, 64> mask_##OP(const Vec<TYPE, 64> &a,          \
                                              const Vec<TYPE, 64> &b)          \
  {                                                                            \
    return _mm512_cmp_##SUF##_mask(a, b, IMM8);                                \
  }

// ---------------------------------------------------------------------------
// masked compare < v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_CMP(cmplt, Byte, epu8)
GENERATE_CMP(cmplt, SignedByte, epi8)
GENERATE_CMP(cmplt, Word, epu16)
GENERATE_CMP(cmplt, Short, epi16)
#endif
GENERATE_CMP(cmplt, Int, epi32)
GENERATE_CMP(cmplt, Long, epi64)

GENERATE_CMP_WITH_GENERALIZED_FCT(cmplt, Float, ps, _CMP_LT_OS)
GENERATE_CMP_WITH_GENERALIZED_FCT(cmplt, Double, pd, _CMP_LT_OS)

// ---------------------------------------------------------------------------
// masked compare <= v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_CMP(cmple, Byte, epu8)
GENERATE_CMP(cmple, SignedByte, epi8)
GENERATE_CMP(cmple, Word, epu16)
GENERATE_CMP(cmple, Short, epi16)
#endif
GENERATE_CMP(cmple, Int, epi32)
GENERATE_CMP(cmple, Long, epi64)

GENERATE_CMP_WITH_GENERALIZED_FCT(cmple, Float, ps, _CMP_LE_OS)
GENERATE_CMP_WITH_GENERALIZED_FCT(cmple, Double, pd, _CMP_LE_OS)

// ---------------------------------------------------------------------------
// masked compare == v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_CMP(cmpeq, Byte, epu8)
GENERATE_CMP(cmpeq, SignedByte, epi8)
GENERATE_CMP(cmpeq, Word, epu16)
GENERATE_CMP(cmpeq, Short, epi16)
#endif
GENERATE_CMP(cmpeq, Int, epi32)
GENERATE_CMP(cmpeq, Long, epi64)

GENERATE_CMP_WITH_GENERALIZED_FCT(cmpeq, Float, ps, _CMP_EQ_OQ)
GENERATE_CMP_WITH_GENERALIZED_FCT(cmpeq, Double, pd, _CMP_EQ_OQ)

// ---------------------------------------------------------------------------
// masked compare > v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_CMP(cmpgt, Byte, epu8)
GENERATE_CMP(cmpgt, SignedByte, epi8)
GENERATE_CMP(cmpgt, Word, epu16)
GENERATE_CMP(cmpgt, Short, epi16)
#endif
GENERATE_CMP(cmpgt, Int, epi32)
GENERATE_CMP(cmpgt, Long, epi64)

GENERATE_CMP_WITH_GENERALIZED_FCT(cmpgt, Float, ps, _CMP_GT_OS)
GENERATE_CMP_WITH_GENERALIZED_FCT(cmpgt, Double, pd, _CMP_GT_OS)

// ---------------------------------------------------------------------------
// masked compare >= v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_CMP(cmpge, Byte, epu8)
GENERATE_CMP(cmpge, SignedByte, epi8)
GENERATE_CMP(cmpge, Word, epu16)
GENERATE_CMP(cmpge, Short, epi16)
#endif
GENERATE_CMP(cmpge, Int, epi32)
GENERATE_CMP(cmpge, Long, epi64)

GENERATE_CMP_WITH_GENERALIZED_FCT(cmpge, Float, ps, _CMP_GE_OS)
GENERATE_CMP_WITH_GENERALIZED_FCT(cmpge, Double, pd, _CMP_GE_OS)

// ---------------------------------------------------------------------------
// masked compare != v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
GENERATE_CMP(cmpneq, Byte, epu8)
GENERATE_CMP(cmpneq, SignedByte, epi8)
GENERATE_CMP(cmpneq, Word, epu16)
GENERATE_CMP(cmpneq, Short, epi16)
#endif
GENERATE_CMP(cmpneq, Int, epi32)
GENERATE_CMP(cmpneq, Long, epi64)

GENERATE_CMP_WITH_GENERALIZED_FCT(cmpneq, Float, ps, _CMP_NEQ_OQ)
GENERATE_CMP_WITH_GENERALIZED_FCT(cmpneq, Double, pd, _CMP_NEQ_OQ)

// ---------------------------------------------------------------------------
// masked avg: average with rounding down v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
static SIMD_INLINE Vec<Byte, 64> mask_avg(const Vec<Byte, 64> &src,
                                          const Mask<Byte, 64> &k,
                                          const Vec<Byte, 64> &a,
                                          const Vec<Byte, 64> &b)
{
  return _mm512_mask_avg_epu8(src, k, a, b);
}

static SIMD_INLINE Vec<Byte, 64> maskz_avg(const Mask<Byte, 64> &k,
                                           const Vec<Byte, 64> &a,
                                           const Vec<Byte, 64> &b)
{
  return _mm512_maskz_avg_epu8(k, a, b);
}

static SIMD_INLINE Vec<Word, 64> mask_avg(const Vec<Word, 64> &src,
                                          const Mask<Word, 64> &k,
                                          const Vec<Word, 64> &a,
                                          const Vec<Word, 64> &b)
{
  return _mm512_mask_avg_epu16(src, k, a, b);
}

static SIMD_INLINE Vec<Word, 64> maskz_avg(const Mask<Word, 64> &k,
                                           const Vec<Word, 64> &a,
                                           const Vec<Word, 64> &b)
{
  return _mm512_maskz_avg_epu16(k, a, b);
}
#endif

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average

template <typename T,
          SIMD_ENABLE_IF(std::is_integral<T>::value &&std::is_signed<T>::value)>
static SIMD_INLINE Vec<T, 64> mask_avg(const Vec<T, 64> &src,
                                       const Mask<T, 64> &k,
                                       const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  const auto one = ::simd::set1<T, 64>(1);
  const auto lsb = bit_and(bit_or(a, b), one);
  const auto as  = srai<1>(a);
  const auto bs  = srai<1>(b);
  return mask_add(src, k, lsb, add(as, bs));
}

template <typename T,
          SIMD_ENABLE_IF(std::is_integral<T>::value &&std::is_signed<T>::value)>
static SIMD_INLINE Vec<T, 64> maskz_avg(const Mask<T, 64> &k,
                                        const Vec<T, 64> &a,
                                        const Vec<T, 64> &b)
{
  const auto one = ::simd::set1<T, 64>(1);
  const auto lsb = bit_and(bit_or(a, b), one);
  const auto as  = srai<1>(a);
  const auto bs  = srai<1>(b);
  return maskz_add(k, lsb, add(as, bs));
}

// NOTE: Float version doesn't round!
static SIMD_INLINE Vec<Float, 64> mask_avg(const Vec<Float, 64> &src,
                                           const Mask<Float, 64> &k,
                                           const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b)
{
  return _mm512_mask_mul_ps(src, k, _mm512_maskz_add_ps(k, a, b),
                            _mm512_set1_ps(0.5f));
}

// NOTE: Float version doesn't round!
static SIMD_INLINE Vec<Float, 64> maskz_avg(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a,
                                            const Vec<Float, 64> &b)
{
  return _mm512_maskz_mul_ps(k, _mm512_maskz_add_ps(k, a, b),
                             _mm512_set1_ps(0.5f));
}

// NOTE: Double version doesn't round!
static SIMD_INLINE Vec<Double, 64> mask_avg(const Vec<Double, 64> &src,
                                            const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b)
{
  return _mm512_mask_mul_pd(src, k, _mm512_maskz_add_pd(k, a, b),
                            _mm512_set1_pd(0.5));
}

// NOTE: Double version doesn't round!
static SIMD_INLINE Vec<Double, 64> maskz_avg(const Mask<Double, 64> &k,
                                             const Vec<Double, 64> &a,
                                             const Vec<Double, 64> &b)
{
  return _mm512_maskz_mul_pd(k, _mm512_maskz_add_pd(k, a, b),
                             _mm512_set1_pd(0.5));
}

// ---------------------------------------------------------------------------
// masked test_all_zeros v
// ---------------------------------------------------------------------------

#define TEST_ALL_ZEROS(TYPE, SUF)                                              \
  static SIMD_INLINE bool mask_test_all_zeros(const Mask<TYPE, 64> &k,         \
                                              const Vec<TYPE, 64> &a)          \
  {                                                                            \
    return (_mm512_mask_test_epi##SUF##_mask(k, a, a) == 0);                   \
  }

#ifdef __AVX512BW__
TEST_ALL_ZEROS(Byte, 8)
TEST_ALL_ZEROS(SignedByte, 8)
TEST_ALL_ZEROS(Word, 16)
TEST_ALL_ZEROS(Short, 16)
#endif
TEST_ALL_ZEROS(Int, 32)
TEST_ALL_ZEROS(Long, 64)

static SIMD_INLINE bool mask_test_all_zeros(const Mask<Float, 64> &k,
                                            const Vec<Float, 64> &a)
{
  return (_mm512_mask_test_epi32_mask(k, _mm512_castps_si512(a),
                                      _mm512_castps_si512(a)) == 0);
}

static SIMD_INLINE bool mask_test_all_zeros(const Mask<Double, 64> &k,
                                            const Vec<Double, 64> &a)
{
  return (_mm512_mask_test_epi64_mask(k, _mm512_castpd_si512(a),
                                      _mm512_castpd_si512(a)) == 0);
}

// ---------------------------------------------------------------------------
// masked test_all_ones v
// ---------------------------------------------------------------------------

// already defined in SIMDVecMaskImplEmu.H

// ---------------------------------------------------------------------------
// mask_all_ones v
// ---------------------------------------------------------------------------

#define MASK_ALL_ONES(TYPE, MASK)                                              \
  static SIMD_INLINE Mask<TYPE, 64> mask_all_ones(OutputType<TYPE>,            \
                                                  Integer<64>)                 \
  {                                                                            \
    return MASK;                                                               \
  }

#ifdef __AVX512BW__
MASK_ALL_ONES(Byte, 0xFFFFFFFFFFFFFFFF)
MASK_ALL_ONES(SignedByte, 0xFFFFFFFFFFFFFFFF)
MASK_ALL_ONES(Word, 0xFFFFFFFF)
MASK_ALL_ONES(Short, 0xFFFFFFFF)
#endif
MASK_ALL_ONES(Int, 0xFFFF)
MASK_ALL_ONES(Float, 0xFFFF)
MASK_ALL_ONES(Long, 0xFF)
MASK_ALL_ONES(Double, 0xFF)

/*
Short explanation:
Intrinsics (e.g. _kand_mask16, _kor_mask32) are only available for gcc versions
>= 7. The intrinsics for __mmask32 and __mmask64 are only available under
AVX512BW Intrinsics with a different name and only for __mmask16 (e.g.
_mm512_kand) are available for gcc versions >= 6 If AVX512BW is not available,
the Byte/SignedByte/Word/Short masks are vectors, then the vector functions
are used The last resort (because it is probably slower in most cases) is to
emulate the functions with normal operators (e.g. "+" for kadd, "<<" for
kshiftl, "&" for kand)
*/

#if __GNUC__ >= 7 // TODO other compilers (not really a problem, then the
                  // intrinsics will just not be used)
#define GENERATE_DMASKOP(NAME, TYPE, NUM)                                      \
  static SIMD_INLINE Mask<TYPE, 64> k##NAME(const Mask<TYPE, 64> &a,           \
                                            const Mask<TYPE, 64> &b)           \
  {                                                                            \
    return _k##NAME##_mask##NUM(a, b);                                         \
  }

#define KNOT(TYPE, NUM)                                                        \
  static SIMD_INLINE Mask<TYPE, 64> knot(const Mask<TYPE, 64> &a)              \
  {                                                                            \
    return _knot_mask##NUM(a);                                                 \
  }

// shift with template parameter
#define KSHIFT(R_OR_L, TYPE, NUM)                                              \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Mask<TYPE, 64> kshift##R_OR_L##i(const Mask<TYPE, 64> &a) \
  {                                                                            \
    return _kshift##R_OR_L##i_mask##NUM(a, COUNT);                             \
  }
#ifdef __AVX512BW__
GENERATE_DMASKOP(and, Byte, 64)
GENERATE_DMASKOP(and, SignedByte, 64)
GENERATE_DMASKOP(and, Word, 32)
GENERATE_DMASKOP(and, Short, 32)

GENERATE_DMASKOP(andn, Byte, 64)
GENERATE_DMASKOP(andn, SignedByte, 64)
GENERATE_DMASKOP(andn, Word, 32)
GENERATE_DMASKOP(andn, Short, 32)

GENERATE_DMASKOP(or, Byte, 64)
GENERATE_DMASKOP(or, SignedByte, 64)
GENERATE_DMASKOP(or, Word, 32)
GENERATE_DMASKOP(or, Short, 32)

GENERATE_DMASKOP(xor, Byte, 64)
GENERATE_DMASKOP(xor, SignedByte, 64)
GENERATE_DMASKOP(xor, Word, 32)
GENERATE_DMASKOP(xor, Short, 32)

GENERATE_DMASKOP(xnor, Byte, 64)
GENERATE_DMASKOP(xnor, SignedByte, 64)
GENERATE_DMASKOP(xnor, Word, 32)
GENERATE_DMASKOP(xnor, Short, 32)

GENERATE_DMASKOP(add, Byte, 64)
GENERATE_DMASKOP(add, SignedByte, 64)
GENERATE_DMASKOP(add, Word, 32)
GENERATE_DMASKOP(add, Short, 32)

KNOT(Byte, 64)
KNOT(SignedByte, 64)
KNOT(Word, 32)
KNOT(Short, 32)

KSHIFT(r, Byte, 64)
KSHIFT(r, SignedByte, 64)
KSHIFT(r, Word, 32)
KSHIFT(r, Short, 32)
KSHIFT(l, Byte, 64)
KSHIFT(l, SignedByte, 64)
KSHIFT(l, Word, 32)
KSHIFT(l, Short, 32)
// else-case is further down
#endif // ifdef __AVX512BW__

GENERATE_DMASKOP(and, Int, 16)
GENERATE_DMASKOP(and, Float, 16)
GENERATE_DMASKOP(and, Long, 8)
GENERATE_DMASKOP(and, Double, 8)

GENERATE_DMASKOP(andn, Int, 16)
GENERATE_DMASKOP(andn, Float, 16)
GENERATE_DMASKOP(andn, Long, 8)
GENERATE_DMASKOP(andn, Double, 8)

GENERATE_DMASKOP(or, Int, 16)
GENERATE_DMASKOP(or, Float, 16)
GENERATE_DMASKOP(or, Long, 8)
GENERATE_DMASKOP(or, Double, 8)

GENERATE_DMASKOP(xor, Int, 16)
GENERATE_DMASKOP(xor, Float, 16)
GENERATE_DMASKOP(xor, Long, 8)
GENERATE_DMASKOP(xor, Double, 8)

GENERATE_DMASKOP(xnor, Int, 16)
GENERATE_DMASKOP(xnor, Float, 16)
GENERATE_DMASKOP(xnor, Long, 8)
GENERATE_DMASKOP(xnor, Double, 8)

#ifdef __AVX512DQ__ // _kadd_mask16 and _kadd_mask8 are only available unter
                    // AVX512DQ
GENERATE_DMASKOP(add, Int, 16)
GENERATE_DMASKOP(add, Float, 16)
GENERATE_DMASKOP(add, Long, 8)
GENERATE_DMASKOP(add, Double, 8)
#endif

KNOT(Int, 16)
KNOT(Float, 16)
KNOT(Long, 8)
KNOT(Double, 8)

KSHIFT(r, Int, 16)
KSHIFT(r, Float, 16)
KSHIFT(r, Long, 8)
KSHIFT(r, Double, 8)

KSHIFT(l, Int, 16)
KSHIFT(l, Float, 16)
KSHIFT(l, Long, 8)
KSHIFT(l, Double, 8)
#else
//(__GNUC__ >= 7) is false
#if __GNUC__ >= 6
// At least the intrinsics for 16- and 8-masks (Int, Float, Long and Double) are
// defined.
#define GENERATE_DMASKOP(NAME, TYPE, NUM)                                      \
  static SIMD_INLINE Mask<TYPE, 64> k##NAME(const Mask<TYPE, 64> &a,           \
                                            const Mask<TYPE, 64> &b)           \
  {                                                                            \
    return _mm512_k##NAME(a, b);                                               \
  }

#define KNOT(TYPE, NUM)                                                        \
  static SIMD_INLINE Mask<TYPE, 64> knot(const Mask<TYPE, 64> &a)              \
  {                                                                            \
    return _mm512_knot(a);                                                     \
  }
GENERATE_DMASKOP(and, Int, 16)
GENERATE_DMASKOP(and, Float, 16)
GENERATE_DMASKOP(and, Long, 8)
GENERATE_DMASKOP(and, Double, 8)

GENERATE_DMASKOP(andn, Int, 16)
GENERATE_DMASKOP(andn, Float, 16)
GENERATE_DMASKOP(andn, Long, 8)
GENERATE_DMASKOP(andn, Double, 8)

GENERATE_DMASKOP(or, Int, 16)
GENERATE_DMASKOP(or, Float, 16)
GENERATE_DMASKOP(or, Long, 8)
GENERATE_DMASKOP(or, Double, 8)

GENERATE_DMASKOP(xor, Int, 16)
GENERATE_DMASKOP(xor, Float, 16)
GENERATE_DMASKOP(xor, Long, 8)
GENERATE_DMASKOP(xor, Double, 8)

GENERATE_DMASKOP(xnor, Int, 16)
GENERATE_DMASKOP(xnor, Float, 16)
GENERATE_DMASKOP(xnor, Long, 8)
GENERATE_DMASKOP(xnor, Double, 8)

KNOT(Int, 16)
KNOT(Float, 16)
KNOT(Long, 8)
KNOT(Double, 8)
#endif

template <typename T>
static SIMD_INLINE Mask<T, 64> kand(const Mask<T, 64> &a, const Mask<T, 64> &b)
{
  return (a & b);
}

template <typename T>
static SIMD_INLINE Mask<T, 64> kandn(const Mask<T, 64> &a, const Mask<T, 64> &b)
{
  return (~a) & b;
}

template <typename T>
static SIMD_INLINE Mask<T, 64> kor(const Mask<T, 64> &a, const Mask<T, 64> &b)
{
  return (a | b);
}

template <typename T>
static SIMD_INLINE Mask<T, 64> kxor(const Mask<T, 64> &a, const Mask<T, 64> &b)
{
  return (a ^ b);
}

template <typename T>
static SIMD_INLINE Mask<T, 64> kxnor(const Mask<T, 64> &a, const Mask<T, 64> &b)
{
  return ~(a ^ b);
}

template <typename T>
static SIMD_INLINE Mask<T, 64> kadd(const Mask<T, 64> &a, const Mask<T, 64> &b)
{
  return (a + b);
}

template <typename T>
static SIMD_INLINE Mask<T, 64> knot(const Mask<T, 64> &a)
{
  return ~a;
}

template <size_t COUNT, typename T>
static SIMD_INLINE Mask<T, 64> kshiftri(const Mask<T, 64> &a)
{
  // 04. Aug 22 (Jonas Keller):
  // return zero if COUNT is larger than 63, since then the >> operator is
  // undefined, but kshift should return zero
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
  // since COUNT is a constant, the compiler should optimize away the
  // if-statement
  if (COUNT >= 64) { return 0; }
// we checked that COUNT is not too large above, disable warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
  return ((uint64_t) a) >> ((uint64_t) COUNT);
#pragma GCC diagnostic pop
}

template <size_t COUNT, typename T>
static SIMD_INLINE Mask<T, 64> kshiftli(const Mask<T, 64> &a)
{
  // 04. Aug 22 (Jonas Keller):
  // return zero if COUNT is larger than 63, since then the << operator is
  // undefined, but kshift should return zero
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
  // since COUNT is a constant, the compiler should optimize away the
  // if-statement
  if (COUNT >= 64) { return 0; }
// we checked that COUNT is not too large above, disable warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
  return ((uint64_t) a) << ((uint64_t) COUNT);
#pragma GCC diagnostic pop
}
#endif // if __GNUC__ >= 7

// shift with flexible parameter (not template), probably slower than
// template-version
/*//TODO faster implementation with switch-case possible?
#define SHIFT_CASE(OP, NUM) case : OP<NUM>(a); break;

#define EMULATE_KSHIFT(R_OR_L, OP, TYPE) \
static SIMD_INLINE Mask<TYPE, 64> \
kshift ## R_OR_L ## i (const Mask<TYPE, 64> &a, \
    uint64_t count) \
{ \
  return (a OP count); \
  switch(count) { \
    SHIFT_CASE(OP2, 0) \
    SHIFT_CASE(OP2, 1) \
  } \
}*/

template <typename T>
static SIMD_INLINE Mask<T, 64> kshiftli(const Mask<T, 64> &a, uint64_t count)
{
  // 04. Aug 22 (Jonas Keller):
  // return zero if count is larger than 63, since then the << operator is
  // undefined, but kshift should return zero
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
  if (count >= 64) { return Mask<T, 64>(0); }
  return Mask<T, 64>(((uint64_t) a) << count);
}

template <typename T>
static SIMD_INLINE Mask<T, 64> kshiftri(const Mask<T, 64> &a, uint64_t count)
{
  // 04. Aug 22 (Jonas Keller):
  // return zero if count is larger than 63, since then the >> operator is
  // undefined, but kshift should return zero
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
  if (count >= 64) { return Mask<T, 64>(0); }
  return Mask<T, 64>(((uint64_t) a) >> count);
}

// 07. Aug 23 (Jonas Keller): added mask_test_all_zeros/ones.

template <typename T>
static SIMD_INLINE bool mask_test_all_zeros(const Mask<T, 64> &a)
{
  return a == 0;
}

template <typename T>
static SIMD_INLINE bool mask_test_all_ones(const Mask<T, 64> &a)
{
  return a == mask_all_ones(OutputType<T>(), Integer<64>());
}

// 07. Aug 23 (Jonas Keller): added kcmpeq

template <typename T>
static SIMD_INLINE Mask<T, 64> kcmpeq(const Mask<T, 64> &a,
                                      const Mask<T, 64> &b)
{
  return a == b;
}
} // namespace mask
} // namespace internal
} // namespace simd

#endif

#endif // SIMD_VEC_MASK_IMPL_INTEL_64_H_
