// ===========================================================================
//
// SIMDVecBaseImplNEON16.H --
// encapsulation for ARM NEON vector extension
// inspired by Agner Fog's C++ Vector Class Library
// http://www.agner.org/optimize/#vectorclass
// (VCL License: GNU General Public License Version 3,
//  http://www.gnu.org/licenses/gpl-3.0.en.html)
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

// 22. Jan 23 (Jonas Keller): moved internal implementations into internal
// namespace

// NOTES:
//
// echo | gcc -E -dM -mcpu=cortex-a9 -mfpu=neon - | more
// echo | arm-linux-gnueabihf-gcc -E -dM -mcpu=cortex-a15 -mfpu=neon - | more
//
// -mfpu=neon
// -mfpu=neon-fp16
//
// GCC 4.9:
// GCC now supports Cortex-A12 and the Cortex-R7 through the
// -mcpu=cortex-a12 and -mcpu=cortex-r7 options.
//
// GCC now has tuning for the Cortex-A57 and Cortex-A53 through the
// -mcpu=cortex-a57 and -mcpu=cortex-a53 options.
//
// Initial big.LITTLE tuning support for the combination of Cortex-A57
// and Cortex-A53 was added through the -mcpu=cortex-a57.cortex-a53
// option. Similar support was added for the combination of Cortex-A15
// and Cortex-A7 through the -mcpu=cortex-a15.cortex-a7 option.

#pragma once
#ifndef SIMD_VEC_BASE_IMPL_NEON_16_H_
#define SIMD_VEC_BASE_IMPL_NEON_16_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsNEON.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(SIMDVEC_NEON_ENABLE) && defined(_SIMD_VEC_16_AVAIL_) &&            \
  !defined(SIMDVEC_SANDBOX)

namespace simd {
namespace internal {
namespace base {
// =========================================================================
// type templates
// =========================================================================

// -------------------------------------------------------------------------
// default vector type collection
// -------------------------------------------------------------------------

template <typename T>
struct _NEONRegType;
// clang-format off
template <> struct _NEONRegType<Byte> { using Type = uint8x16_t; };
template <> struct _NEONRegType<SignedByte> { using Type = int8x16_t; };
template <> struct _NEONRegType<Word> { using Type = uint16x8_t; };
template <> struct _NEONRegType<Short> { using Type = int16x8_t; };
template <> struct _NEONRegType<Int> { using Type = int32x4_t; };
template <> struct _NEONRegType<Float> { using Type = float32x4_t; };
#ifdef SIMD_DOUBLE
template <> struct _NEONRegType<Double> { using Type = float64x2_t; };
#endif
// clang-format on

template <typename T>
using NEONRegType = typename _NEONRegType<T>::Type;

// -------------------------------------------------------------------------
// 64bit array type collection
// -------------------------------------------------------------------------

template <size_t N, typename T>
struct SIMDVecNeonArray64;

#define SIMDVEC_NEON_ARRAY64(NUM, T, NEON_T)                                   \
  template <>                                                                  \
  struct SIMDVecNeonArray64<NUM, T>                                            \
  {                                                                            \
    using Type    = NEON_T##x##NUM##_t;                                        \
    using ValType = NEON_T##_t;                                                \
  };

#define SIMDVEC_NEON_ARRAY64_ALLNUM(T, NEON_T)                                 \
  SIMDVEC_NEON_ARRAY64(1, T, NEON_T)                                           \
  SIMDVEC_NEON_ARRAY64(2, T, NEON_T)                                           \
  SIMDVEC_NEON_ARRAY64(3, T, NEON_T)                                           \
  SIMDVEC_NEON_ARRAY64(4, T, NEON_T)

SIMDVEC_NEON_ARRAY64_ALLNUM(Byte, uint8x8)
SIMDVEC_NEON_ARRAY64_ALLNUM(SignedByte, int8x8)
SIMDVEC_NEON_ARRAY64_ALLNUM(Word, uint16x4)
SIMDVEC_NEON_ARRAY64_ALLNUM(Short, int16x4)
SIMDVEC_NEON_ARRAY64_ALLNUM(Int, int32x2)
SIMDVEC_NEON_ARRAY64_ALLNUM(Float, float32x2)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_ARRAY64_ALLNUM(Double, float64x1)
#endif

#undef SIMDVEC_NEON_ARRAY64
#undef SIMDVEC_NEON_ARRAY64_ALLNUM

} // namespace base
} // namespace internal

// =========================================================================
// Vec instantiation for NEON
// =========================================================================

template <typename T>
class Vec<T, 16>
{
  using RegType = internal::base::NEONRegType<T>;
  RegType reg   = {};

public:
  using Type                       = T;
  static constexpr size_t elements = 16 / sizeof(T);
  static constexpr size_t elems    = elements;
  static constexpr size_t bytes    = 16;

  Vec() = default;
  Vec(const RegType &x) { reg = x; }
  Vec &operator=(const RegType &x)
  {
    reg = x;
    return *this;
  }
  operator RegType() const { return reg; }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
  // 05. Sep 23 (Jonas Keller): added allocator
  using allocator = simd_aligned_allocator<Vec<T, bytes>, bytes>;
};

namespace internal {
namespace base {

// =========================================================================
// macros for common functions
// =========================================================================

// -------------------------------------------------------------------------
// binary functions (same input and output type)
// -------------------------------------------------------------------------

// wrapper for arbitrary binary function
#define SIMDVEC_NEON_BINARY(FCT, TYPE, NEON_FCT, NEON_SUF)                     \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       const Vec<TYPE, 16> &b)                 \
  {                                                                            \
    return NEON_FCT##_##NEON_SUF(a, b);                                        \
  }

#define SIMDVEC_NEON_BINARY_ALLINT(FCT, NEON_FCT)                              \
  SIMDVEC_NEON_BINARY(FCT, Byte, NEON_FCT, u8)                                 \
  SIMDVEC_NEON_BINARY(FCT, SignedByte, NEON_FCT, s8)                           \
  SIMDVEC_NEON_BINARY(FCT, Word, NEON_FCT, u16)                                \
  SIMDVEC_NEON_BINARY(FCT, Short, NEON_FCT, s16)                               \
  SIMDVEC_NEON_BINARY(FCT, Int, NEON_FCT, s32)

#ifdef SIMD_DOUBLE
#define SIMDVEC_NEON_BINARY_ALLFLOAT(FCT, NEON_FCT)                            \
  SIMDVEC_NEON_BINARY(FCT, Float, NEON_FCT, f32)                               \
  SIMDVEC_NEON_BINARY(FCT, Double, NEON_FCT, f64)
#else
#define SIMDVEC_NEON_BINARY_ALLFLOAT(FCT, NEON_FCT)                            \
  SIMDVEC_NEON_BINARY(FCT, Float, NEON_FCT, f32)
#endif

#define SIMDVEC_NEON_BINARY_ALL(FCT, NEON_FCT)                                 \
  SIMDVEC_NEON_BINARY_ALLINT(FCT, NEON_FCT)                                    \
  SIMDVEC_NEON_BINARY_ALLFLOAT(FCT, NEON_FCT)

// -------------------------------------------------------------------------
// unary functions
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_UNARY(FCT, TYPE, NEON_FCT, NEON_SUF)                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a)                 \
  {                                                                            \
    return NEON_FCT##_##NEON_SUF(a);                                           \
  }

// #########################################################################
// #########################################################################
// #########################################################################

// =========================================================================
// Vec function instantiations or overloading for NEON
// =========================================================================

// -------------------------------------------------------------------------
// reinterpretation casts
// -------------------------------------------------------------------------

// wrapper for vreinterpretq
#define SIMDVEC_NEON_REINTERP(TDST, NEON_TDST, TSRC, NEON_TSRC)                \
  static SIMD_INLINE Vec<TDST, 16> reinterpret(const Vec<TSRC, 16> &vec,       \
                                               OutputType<TDST>)               \
  {                                                                            \
    return vreinterpretq_##NEON_TDST##_##NEON_TSRC(vec);                       \
  }

// wrapper for all dst types and same source type
#ifdef SIMD_DOUBLE
#define SIMDVEC_NEON_REINTERP_ALLDST(TSRC, NEON_TSRC)                          \
  SIMDVEC_NEON_REINTERP(Byte, u8, TSRC, NEON_TSRC)                             \
  SIMDVEC_NEON_REINTERP(SignedByte, s8, TSRC, NEON_TSRC)                       \
  SIMDVEC_NEON_REINTERP(Word, u16, TSRC, NEON_TSRC)                            \
  SIMDVEC_NEON_REINTERP(Short, s16, TSRC, NEON_TSRC)                           \
  SIMDVEC_NEON_REINTERP(Int, s32, TSRC, NEON_TSRC)                             \
  SIMDVEC_NEON_REINTERP(Float, f32, TSRC, NEON_TSRC)                           \
  SIMDVEC_NEON_REINTERP(Double, f64, TSRC, NEON_TSRC)
#else
#define SIMDVEC_NEON_REINTERP_ALLDST(TSRC, NEON_TSRC)                          \
  SIMDVEC_NEON_REINTERP(Byte, u8, TSRC, NEON_TSRC)                             \
  SIMDVEC_NEON_REINTERP(SignedByte, s8, TSRC, NEON_TSRC)                       \
  SIMDVEC_NEON_REINTERP(Word, u16, TSRC, NEON_TSRC)                            \
  SIMDVEC_NEON_REINTERP(Short, s16, TSRC, NEON_TSRC)                           \
  SIMDVEC_NEON_REINTERP(Int, s32, TSRC, NEON_TSRC)                             \
  SIMDVEC_NEON_REINTERP(Float, f32, TSRC, NEON_TSRC)
#endif

// wrapper for all dst and src types
SIMDVEC_NEON_REINTERP_ALLDST(Byte, u8)
SIMDVEC_NEON_REINTERP_ALLDST(SignedByte, s8)
SIMDVEC_NEON_REINTERP_ALLDST(Word, u16)
SIMDVEC_NEON_REINTERP_ALLDST(Short, s16)
SIMDVEC_NEON_REINTERP_ALLDST(Int, s32)
SIMDVEC_NEON_REINTERP_ALLDST(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_REINTERP_ALLDST(Double, f64)
#endif

#undef SIMDVEC_NEON_REINTERP_ALLDST
#undef SIMDVEC_NEON_REINTERP

// -------------------------------------------------------------------------
// convert (without changes in the number of of elements)
// -------------------------------------------------------------------------

// conversion seems to be saturated in all cases (specified by the
// rounding mode):
// http://stackoverflow.com/questions/24546927/
//  behavior-of-arm-neon-float-integer-conversion-with-overflow

// saturated
// TODO: rounding in cvts (float->int)? +0.5?
// TODO: (NOT the same behavior as in SIMDVecBaseImplIntel16.H
// TODO:  float->int always uses round towards zero = trunc?)
// TODO: cvts: should we saturate in the same way as for Intel?
// TODO: (Intel saturates to max. float which is convertible to int,
// TODO:  NEON saturates to 0x7fffffff)
static SIMD_INLINE Vec<Int, 16> cvts(const Vec<Float, 16> &a, OutputType<Int>)
{
  return vcvtq_s32_f32(a);
}

// saturation is not necessary in this case
static SIMD_INLINE Vec<Float, 16> cvts(const Vec<Int, 16> &a, OutputType<Float>)
{
  return vcvtq_f32_s32(a);
}

// -------------------------------------------------------------------------
// setzero
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_SETZERO(TYPE, NEON_SUF)                                   \
  static SIMD_INLINE Vec<TYPE, 16> setzero(OutputType<TYPE>, Integer<16>)      \
  {                                                                            \
    return vmovq_n##_##NEON_SUF(TYPE(0));                                      \
  }

SIMDVEC_NEON_SETZERO(Byte, u8)
SIMDVEC_NEON_SETZERO(SignedByte, s8)
SIMDVEC_NEON_SETZERO(Word, u16)
SIMDVEC_NEON_SETZERO(Short, s16)
SIMDVEC_NEON_SETZERO(Int, s32)
SIMDVEC_NEON_SETZERO(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_SETZERO(Double, f64)
#endif

#undef SIMDVEC_NEON_SETZERO

// -------------------------------------------------------------------------
// set1
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_SET1(TYPE, NEON_SUF)                                      \
  static SIMD_INLINE Vec<TYPE, 16> set1(TYPE a, Integer<16>)                   \
  {                                                                            \
    return vdupq_n##_##NEON_SUF(a);                                            \
  }

SIMDVEC_NEON_SET1(Byte, u8)
SIMDVEC_NEON_SET1(SignedByte, s8)
SIMDVEC_NEON_SET1(Word, u16)
SIMDVEC_NEON_SET1(Short, s16)
SIMDVEC_NEON_SET1(Int, s32)
SIMDVEC_NEON_SET1(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_SET1(Double, f64)
#endif

#undef SIMDVEC_NEON_SET1

// -------------------------------------------------------------------------
// load
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_LOAD(TYPE, NEON_SUF)                                      \
  static SIMD_INLINE Vec<TYPE, 16> load(const TYPE *const p, Integer<16>)      \
  {                                                                            \
    return vld1q##_##NEON_SUF(p);                                              \
  }                                                                            \
  static SIMD_INLINE Vec<TYPE, 16> loadu(const TYPE *const p, Integer<16>)     \
  {                                                                            \
    return vld1q##_##NEON_SUF(p);                                              \
  }

SIMDVEC_NEON_LOAD(Byte, u8)
SIMDVEC_NEON_LOAD(SignedByte, s8)
SIMDVEC_NEON_LOAD(Word, u16)
SIMDVEC_NEON_LOAD(Short, s16)
SIMDVEC_NEON_LOAD(Int, s32)
SIMDVEC_NEON_LOAD(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_LOAD(Double, f64)
#endif

#undef SIMDVEC_NEON_LOAD

// -------------------------------------------------------------------------
// store
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_STORE(TYPE, NEON_SUF)                                     \
  static SIMD_INLINE void store(TYPE *const p, const Vec<TYPE, 16> &a)         \
  {                                                                            \
    return vst1q##_##NEON_SUF(p, a);                                           \
  }                                                                            \
  static SIMD_INLINE void storeu(TYPE *const p, const Vec<TYPE, 16> &a)        \
  {                                                                            \
    return vst1q##_##NEON_SUF(p, a);                                           \
  }                                                                            \
  static SIMD_INLINE void stream_store(TYPE *const p, const Vec<TYPE, 16> &a)  \
  {                                                                            \
    return vst1q##_##NEON_SUF(p, a);                                           \
  }

SIMDVEC_NEON_STORE(Byte, u8)
SIMDVEC_NEON_STORE(SignedByte, s8)
SIMDVEC_NEON_STORE(Word, u16)
SIMDVEC_NEON_STORE(Short, s16)
SIMDVEC_NEON_STORE(Int, s32)
SIMDVEC_NEON_STORE(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_STORE(Double, f64)
#endif

#undef SIMDVEC_NEON_STORE

// -------------------------------------------------------------------------
// fences
// -------------------------------------------------------------------------

// http://infocenter.arm.com/help/
//   index.jsp?topic=/com.arm.doc.faqs/ka14552.html
// TODO: is this portable to clang?

// NOTE: implemented as full barrier
static SIMD_INLINE void lfence()
{
  SIMD_FULL_MEMBARRIER;
}

// NOTE: implemented as full barrier
static SIMD_INLINE void sfence()
{
  SIMD_FULL_MEMBARRIER;
}

// NOTE: implemented as full barrier
static SIMD_INLINE void mfence()
{
  SIMD_FULL_MEMBARRIER;
}

// -------------------------------------------------------------------------
// extract: with template parameter for immediate argument
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_EXTRACT(TYPE, NEON_SUF)                                   \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE TYPE extract(const Vec<TYPE, 16> &a)                      \
  {                                                                            \
    SIMD_IF_CONSTEXPR (COUNT < Vec<TYPE, 16>::elements) {                      \
      return vgetq_lane##_##NEON_SUF(a, COUNT);                                \
    } else {                                                                   \
      return TYPE(0);                                                          \
    }                                                                          \
  }

SIMDVEC_NEON_EXTRACT(Byte, u8)
SIMDVEC_NEON_EXTRACT(SignedByte, s8)
SIMDVEC_NEON_EXTRACT(Word, u16)
SIMDVEC_NEON_EXTRACT(Short, s16)
SIMDVEC_NEON_EXTRACT(Int, s32)
SIMDVEC_NEON_EXTRACT(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_EXTRACT(Double, f64)
#endif

#undef SIMDVEC_NEON_EXTRACT

// -------------------------------------------------------------------------
// add
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALL(add, vaddq)

// -------------------------------------------------------------------------
// adds
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALLINT(adds, vqaddq)
// float NOT saturated
SIMDVEC_NEON_BINARY(adds, Float, vaddq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_BINARY(adds, Double, vaddq, f64)
#endif

// -------------------------------------------------------------------------
// sub
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALL(sub, vsubq)

// -------------------------------------------------------------------------
// subs
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALLINT(subs, vqsubq)
// float NOT saturated
SIMDVEC_NEON_BINARY(subs, Float, vsubq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_BINARY(subs, Double, vsubq, f64)
#endif

// -------------------------------------------------------------------------
// neg (negate = two's complement or unary minus), only signed types
// -------------------------------------------------------------------------

SIMDVEC_NEON_UNARY(neg, SignedByte, vnegq, s8)
SIMDVEC_NEON_UNARY(neg, Short, vnegq, s16)
SIMDVEC_NEON_UNARY(neg, Int, vnegq, s32)
SIMDVEC_NEON_UNARY(neg, Float, vnegq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_UNARY(neg, Double, vnegq, f64)
#endif

// -------------------------------------------------------------------------
// min
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALL(min, vminq)

// -------------------------------------------------------------------------
// max
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALL(max, vmaxq)

// -------------------------------------------------------------------------
// mul, div
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY(mul, Float, vmulq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_BINARY(mul, Double, vmulq, f64)
#endif

const auto DIV_NEWTON_STEPS = 2;

// adapted from Jens Froemmer's Ba thesis (2014)
static SIMD_INLINE Vec<Float, 16> div(const Vec<Float, 16> &num,
                                      const Vec<Float, 16> &denom)
{
  // get estimate of reciprocal of denom
  float32x4_t reciprocal = vrecpeq_f32(denom);
  // refine estimate using Newton-Raphson steps
  for (size_t i = 0; i < DIV_NEWTON_STEPS; i++)
    reciprocal = vmulq_f32(vrecpsq_f32(denom, reciprocal), reciprocal);
  // num * (1.0 / denom)
  return vmulq_f32(num, reciprocal);
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> div(const Vec<Double, 16> &num,
                                       const Vec<Double, 16> &denom)
{
  // get estimate of reciprocal of denom
  float64x2_t reciprocal = vrecpeq_f64(denom);
  // refine estimate using Newton-Raphson steps
  for (size_t i = 0; i < DIV_NEWTON_STEPS; i++)
    reciprocal = vmulq_f64(vrecpsq_f64(denom, reciprocal), reciprocal);
  // num * (1.0 / denom)
  return vmulq_f64(num, reciprocal);
}
#endif

// -------------------------------------------------------------------------
// ceil, floor, round, truncate
// -------------------------------------------------------------------------

// 25. Mar 23 (Jonas Keller): added versions for integer types

// versions for integer types do nothing:

template <typename T>
static SIMD_INLINE Vec<T, 16> ceil(const Vec<T, 16> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 16> floor(const Vec<T, 16> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 16> round(const Vec<T, 16> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 16> truncate(const Vec<T, 16> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

// http://www.rowleydownload.co.uk/arm/documentation/gnu/gcc/
//   ARM-NEON-Intrinsics.html
// vrnd, only some architectures, see arm_neon.h

#if __ARM_ARCH >= 8

// 10. Apr 19 (rm): BINARY->UNARY, qp -> pq etc., still not tested
SIMDVEC_NEON_UNARY(ceil, Float, vrndpq, f32)
SIMDVEC_NEON_UNARY(floor, Float, vrndmq, f32)
SIMDVEC_NEON_UNARY(round, Float, vrndnq, f32)
SIMDVEC_NEON_UNARY(truncate, Float, vrndq, f32)

#ifdef SIMD_DOUBLE
SIMDVEC_NEON_UNARY(ceil, Double, vrndpq, f64)
SIMDVEC_NEON_UNARY(floor, Double, vrndmq, f64)
SIMDVEC_NEON_UNARY(round, Double, vrndnq, f64)
SIMDVEC_NEON_UNARY(truncate, Double, vrndq, f64)
#endif

#else

static SIMD_INLINE Vec<Float, 16> truncate(const Vec<Float, 16> &a)
{
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  float32x4_t limit = vmovq_n_f32(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  uint32x4_t noRndReq = vcgeq_f32(vabsq_f32(a), limit);
  // truncated result (for |a| < limit)
  float32x4_t aTrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
  // select result
  return vbslq_f32(noRndReq, a, aTrunc);
}

// https://en.wikipedia.org/wiki/Floor_and_ceiling_functions
//
// floor, ceil:
//                 floor(x), x >= 0
// truncate(x) = {
//                 ceil(x), x < 0
//
// floor(x) = ceil(x)  - (x in Z ? 0 : 1)
// ceil(x)  = floor(x) + (x in Z ? 0 : 1)

static SIMD_INLINE Vec<Float, 16> floor(const Vec<Float, 16> &a)
{
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  float32x4_t limit = vmovq_n_f32(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  uint32x4_t noRndReq = vcgeq_f32(vabsq_f32(a), limit);
  // bool mask: true if a is negative
  uint32x4_t isNeg =
    vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(a), 31));
  // truncated result (for |a| < limit)
  float32x4_t aTrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
  // check if a is an integer
  uint32x4_t isNotInt = vmvnq_u32(vceqq_f32(a, aTrunc));
  // constant 1.0
  float32x4_t one = vmovq_n_f32(1.0f);
  // mask which is 1.0f for negative non-integer values, 0.0f otherwise
  float32x4_t oneMask = vreinterpretq_f32_u32(
    vandq_u32(vandq_u32(isNeg, isNotInt), vreinterpretq_u32_f32(one)));
  // if negative, trunc computes ceil, to turn it into floor we sub
  // 1 if aTrunc is non-integer
  aTrunc = vsubq_f32(aTrunc, oneMask);
  // select result (a or aTrunc)
  return vbslq_f32(noRndReq, a, aTrunc);
}

static SIMD_INLINE Vec<Float, 16> ceil(const Vec<Float, 16> &a)
{
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  float32x4_t limit = vmovq_n_f32(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  uint32x4_t noRndReq = vcgeq_f32(vabsq_f32(a), limit);
  // bool mask: true if a is negative
  uint32x4_t isNotNeg =
    vmvnq_u32(vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(a), 31)));
  // truncated result (for |a| < limit)
  float32x4_t aTrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
  // check if a is an integer
  uint32x4_t isNotInt = vmvnq_u32(vceqq_f32(a, aTrunc));
  // constant 1.0
  float32x4_t one = vmovq_n_f32(1.0f);
  // mask which is 1.0f for non-negative non-integer values, 0.0f otherwise
  float32x4_t oneMask = vreinterpretq_f32_u32(
    vandq_u32(vandq_u32(isNotNeg, isNotInt), vreinterpretq_u32_f32(one)));
  // if non-negative, trunc computes floor, to turn it into ceil we
  // add 1 if aTrunc is non-integer
  aTrunc = vaddq_f32(aTrunc, oneMask);
  // select result (a or aTrunc)
  return vbslq_f32(noRndReq, a, aTrunc);
}

// NOTE: rounds ties (*.5) towards infinity, different from Intel
static SIMD_INLINE Vec<Float, 16> round(const Vec<Float, 16> &a)
{
  return floor(add(a, set1(Float(0.5f), Integer<16>())));
}

#endif

// -------------------------------------------------------------------------
// elementary mathematical functions
// -------------------------------------------------------------------------

// estimate of a reciprocal
SIMDVEC_NEON_UNARY(rcp, Float, vrecpeq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_UNARY(rcp, Double, vrecpeq, f64)
#endif

// estimate of a reverse square root
SIMDVEC_NEON_UNARY(rsqrt, Float, vrsqrteq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_UNARY(rsqrt, Double, vrsqrteq, f64)
#endif

const auto SQRT_NEWTON_STEPS = 2;

// square root (may not be very efficient)
static SIMD_INLINE Vec<Float, 16> sqrt(const Vec<Float, 16> &a)
{
  // vector with 0s, vector with 1s
  float32x4_t zero = vmovq_n_f32(0.0f), one = vmovq_n_f32(1.0f);
  // check for 0 to avoid div-by-0 (should also cover -0.0f)
  uint32x4_t isZero = vceqq_f32(a, zero);
  // avoid inf in rev. sqrt, replace 0 by 1
  float32x4_t as = vbslq_f32(isZero, one, a);
  // get estimate of reciprocal sqrt
  float32x4_t rSqrt = vrsqrteq_f32(as);
  // refine estimate using Newton-Raphson steps
  for (size_t i = 0; i < SQRT_NEWTON_STEPS; i++)
    rSqrt = vmulq_f32(vrsqrtsq_f32(as, vmulq_f32(rSqrt, rSqrt)), rSqrt);
  // sqrt(a) = a * (1.0 / sqrt(a))
  float32x4_t res = vmulq_f32(as, rSqrt);
  // select result
  return vbslq_f32(isZero, zero, res);
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> sqrt(const Vec<Double, 16> &a)
{
  // vector with 0s, vector with 1s
  float64x2_t zero = vmovq_n_f64(0.0), one = vmovq_n_f64(1.0);
  // check for 0 to avoid div-by-0 (should also cover -0.0)
  uint64x2_t isZero = vceqq_f64(a, zero);
  // avoid inf in rev. sqrt, replace 0 by 1
  float64x2_t as = vbslq_f64(isZero, one, a);
  // get estimate of reciprocal sqrt
  float64x2_t rSqrt = vrsqrteq_f64(as);
  // refine estimate using Newton-Raphson steps
  for (size_t i = 0; i < SQRT_NEWTON_STEPS; i++)
    rSqrt = vmulq_f64(vrsqrtsq_f64(as, vmulq_f64(rSqrt, rSqrt)), rSqrt);
  // sqrt(a) = a * (1.0 / sqrt(a))
  float64x2_t res = vmulq_f64(as, rSqrt);
  // select result
  return vbslq_f64(isZero, zero, res);
}
#endif

// -------------------------------------------------------------------------
// abs
// -------------------------------------------------------------------------

// 25. Mar 25 (Jonas Keller): added abs for unsigned integers

// unsigned integers
template <typename T>
static SIMD_INLINE Vec<T, 16> abs(const Vec<T, 16> &a)
{
  static_assert(std::is_unsigned<T>::value && std::is_integral<T>::value, "");
  return a;
}

SIMDVEC_NEON_UNARY(abs, SignedByte, vabsq, s8)
SIMDVEC_NEON_UNARY(abs, Short, vabsq, s16)
SIMDVEC_NEON_UNARY(abs, Int, vabsq, s32)
SIMDVEC_NEON_UNARY(abs, Float, vabsq, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_UNARY(abs, Double, vabsq, f64)
#endif

// -------------------------------------------------------------------------
// unpack
// -------------------------------------------------------------------------

// TODO: unpack is inefficient here since vzipq does both unpacklo and
// TODO: unpackhi but only half of the result is used

// via cast to larger datatype
#define SIMDVEC_NEON_UNPACK(TYPE, BYTES, NEON_SUF, NEON_SUF2)                  \
  template <size_t PART>                                                       \
  static SIMD_INLINE Vec<TYPE, 16> unpack(                                     \
    const Vec<TYPE, 16> &a, const Vec<TYPE, 16> &b, Part<PART>, Bytes<BYTES>)  \
  {                                                                            \
    return vreinterpretq_##NEON_SUF##_##NEON_SUF2(                             \
      (vzipq_##NEON_SUF2(vreinterpretq_##NEON_SUF2##_##NEON_SUF(a),            \
                         vreinterpretq_##NEON_SUF2##_##NEON_SUF(b)))           \
        .val[PART]);                                                           \
  }

// via extraction of low or high halfs
// (NOTE: PART and BYTES are needed in argument list)
#define SIMDVEC_NEON_UNPACK_HALFS(TYPE, BYTES, NEON_SUF)                       \
  static SIMD_INLINE Vec<TYPE, 16> unpack(                                     \
    const Vec<TYPE, 16> &a, const Vec<TYPE, 16> &b, Part<0>, Bytes<BYTES>)     \
  {                                                                            \
    return vcombine_##NEON_SUF(vget_low##_##NEON_SUF(a),                       \
                               vget_low##_##NEON_SUF(b));                      \
  }                                                                            \
  static SIMD_INLINE Vec<TYPE, 16> unpack(                                     \
    const Vec<TYPE, 16> &a, const Vec<TYPE, 16> &b, Part<1>, Bytes<BYTES>)     \
  {                                                                            \
    return vcombine_##NEON_SUF(vget_high##_##NEON_SUF(a),                      \
                               vget_high##_##NEON_SUF(b));                     \
  }

SIMDVEC_NEON_UNPACK(Byte, 1, u8, u8)
SIMDVEC_NEON_UNPACK(Byte, 2, u8, u16)
SIMDVEC_NEON_UNPACK(Byte, 4, u8, u32)
SIMDVEC_NEON_UNPACK_HALFS(Byte, 8, u8)
SIMDVEC_NEON_UNPACK(SignedByte, 1, s8, s8)
SIMDVEC_NEON_UNPACK(SignedByte, 2, s8, s16)
SIMDVEC_NEON_UNPACK(SignedByte, 4, s8, s32)
SIMDVEC_NEON_UNPACK_HALFS(SignedByte, 8, s8)
SIMDVEC_NEON_UNPACK(Word, 2, u16, u16)
SIMDVEC_NEON_UNPACK(Word, 4, u16, u32)
SIMDVEC_NEON_UNPACK_HALFS(Word, 8, u16)
SIMDVEC_NEON_UNPACK(Short, 2, s16, s16)
SIMDVEC_NEON_UNPACK(Short, 4, s16, s32)
SIMDVEC_NEON_UNPACK_HALFS(Short, 8, s16)
SIMDVEC_NEON_UNPACK(Int, 4, s32, s32)
SIMDVEC_NEON_UNPACK_HALFS(Int, 8, s32)
SIMDVEC_NEON_UNPACK(Float, 4, f32, f32)
SIMDVEC_NEON_UNPACK_HALFS(Float, 8, f32)

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> unpack(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b, Part<0>,
                                          Bytes<8>)
{
  return vcombine_f64(vget_low_f64(a), vget_low_f64(b));
}
static SIMD_INLINE Vec<Double, 16> unpack(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b, Part<1>,
                                          Bytes<8>)
{
  return vcombine_f64(vget_high_f64(a), vget_high_f64(b));
}
#endif

#undef SIMDVEC_NEON_UNPACK
#undef SIMDVEC_NEON_UNPACK_HALFS

// ---------------------------------------------------------------------------
// unpack16
// ---------------------------------------------------------------------------

// 16-byte-lane oriented unpack: for 16 bytes same as generalized unpack
// unpack blocks of NUM_ELEMS elements of type T
// PART=0: low half of input vectors,
// PART=1: high half of input vectors
template <size_t PART, size_t BYTES, typename T>
static SIMD_INLINE Vec<T, 16> unpack16(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                       Part<PART>, Bytes<BYTES>)
{
  return unpack(a, b, Part<PART>(), Bytes<BYTES>());
}

// ---------------------------------------------------------------------------
// extract 128-bit lane as Vec<T, 16>, does nothing for 16 bytes
// ---------------------------------------------------------------------------

template <size_t LANE_INDEX, typename T>
static SIMD_INLINE Vec<T, 16> extractLane(const Vec<T, 16> &a)
{
  return a;
}

// -------------------------------------------------------------------------
// zip
// -------------------------------------------------------------------------

// a, b passed by-value to avoid problems with identical input/output args.

// via cast to larger datatype
#define SIMDVEC_NEON_ZIP(TYPE, NUM_ELEMS, NEON_SUF, NEON_SUF2, NEONX2_2)       \
  static SIMD_INLINE void zip(const Vec<TYPE, 16> a, const Vec<TYPE, 16> b,    \
                              Vec<TYPE, 16> &c, Vec<TYPE, 16> &d,              \
                              Elements<NUM_ELEMS>)                             \
  {                                                                            \
    NEONX2_2 res;                                                              \
    res = vzipq_##NEON_SUF2(vreinterpretq_##NEON_SUF2##_##NEON_SUF(a),         \
                            vreinterpretq_##NEON_SUF2##_##NEON_SUF(b));        \
    c   = vreinterpretq_##NEON_SUF##_##NEON_SUF2(res.val[0]);                  \
    d   = vreinterpretq_##NEON_SUF##_##NEON_SUF2(res.val[1]);                  \
  }

// via extraction of low or high halfs
// (NOTE: NUM_ELEMS is needed in argument list)
#define SIMDVEC_NEON_ZIP_HALFS(TYPE, NUM_ELEMS, NEON_SUF)                      \
  static SIMD_INLINE void zip(const Vec<TYPE, 16> a, const Vec<TYPE, 16> b,    \
                              Vec<TYPE, 16> &c, Vec<TYPE, 16> &d,              \
                              Elements<NUM_ELEMS>)                             \
  {                                                                            \
    c = vcombine_##NEON_SUF(vget_low_##NEON_SUF(a), vget_low_##NEON_SUF(b));   \
    d = vcombine_##NEON_SUF(vget_high_##NEON_SUF(a), vget_high_##NEON_SUF(b)); \
  }

SIMDVEC_NEON_ZIP(Byte, 1, u8, u8, uint8x16x2_t)
SIMDVEC_NEON_ZIP(Byte, 2, u8, u16, uint16x8x2_t)
SIMDVEC_NEON_ZIP(Byte, 4, u8, u32, uint32x4x2_t)
SIMDVEC_NEON_ZIP_HALFS(Byte, 8, u8)
SIMDVEC_NEON_ZIP(SignedByte, 1, s8, s8, int8x16x2_t)
SIMDVEC_NEON_ZIP(SignedByte, 2, s8, s16, int16x8x2_t)
SIMDVEC_NEON_ZIP(SignedByte, 4, s8, s32, int32x4x2_t)
SIMDVEC_NEON_ZIP_HALFS(SignedByte, 8, s8)
SIMDVEC_NEON_ZIP(Word, 1, u16, u16, uint16x8x2_t)
SIMDVEC_NEON_ZIP(Word, 2, u16, u32, uint32x4x2_t)
SIMDVEC_NEON_ZIP_HALFS(Word, 4, u16)
SIMDVEC_NEON_ZIP(Short, 1, s16, s16, int16x8x2_t)
SIMDVEC_NEON_ZIP(Short, 2, s16, s32, int32x4x2_t)
SIMDVEC_NEON_ZIP_HALFS(Short, 4, s16)
SIMDVEC_NEON_ZIP(Int, 1, s32, s32, int32x4x2_t)
SIMDVEC_NEON_ZIP_HALFS(Int, 2, s32)
SIMDVEC_NEON_ZIP(Float, 1, f32, f32, float32x4x2_t)
SIMDVEC_NEON_ZIP_HALFS(Float, 2, f32)

#ifdef SIMD_DOUBLE
static SIMD_INLINE void zip(const Vec<Double, 16> a, const Vec<Double, 16> b,
                            Vec<Double, 16> &c, Vec<Double, 16> &d, Elements<1>)
{
  c = vcombine_f64(vget_low_f64(a), vget_low_f64(b));
  d = vcombine_f64(vget_high_f64(a), vget_high_f64(b));
}
#endif

template <size_t NUM_ELEMS, typename T>
static SIMD_INLINE void zip(const Vec<T, 16> a, const Vec<T, 16> b,
                            Vec<T, 16> &c, Vec<T, 16> &d)
{
  return zip(a, b, c, d, Elements<NUM_ELEMS>());
}

#undef SIMDVEC_NEON_ZIP
#undef SIMDVEC_NEON_ZIP_HALFS

// ---------------------------------------------------------------------------
// zip16 hub  (16-byte-lane oriented zip): for 16 bytes same as zip
// ---------------------------------------------------------------------------

// a, b are passed by-value to avoid problems with identical input/output args.

template <size_t NUM_ELEMS, typename T>
static SIMD_INLINE void zip16(const Vec<T, 16> a, const Vec<T, 16> b,
                              Vec<T, 16> &l, Vec<T, 16> &h)
{
  zip<NUM_ELEMS, T>(a, b, l, h);
}

// -------------------------------------------------------------------------
// unzip
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// unzip
// -------------------------------------------------------------------------

// a, b passed by-value to avoid problems with identical input/output args.

// via cast to larger datatype
#define SIMDVEC_NEON_UNZIP(TYPE, BYTES, NEON_SUF, NEON_SUF2, NEONX2_2)         \
  static SIMD_INLINE void unzip(const Vec<TYPE, 16> a, const Vec<TYPE, 16> b,  \
                                Vec<TYPE, 16> &c, Vec<TYPE, 16> &d,            \
                                Bytes<BYTES>)                                  \
  {                                                                            \
    NEONX2_2 res;                                                              \
    res = vuzpq_##NEON_SUF2(vreinterpretq_##NEON_SUF2##_##NEON_SUF(a),         \
                            vreinterpretq_##NEON_SUF2##_##NEON_SUF(b));        \
    c   = vreinterpretq_##NEON_SUF##_##NEON_SUF2(res.val[0]);                  \
    d   = vreinterpretq_##NEON_SUF##_##NEON_SUF2(res.val[1]);                  \
  }

// via extraction of low or high halfs
// (NOTE: BYTES is needed in argument list)
#define SIMDVEC_NEON_UNZIP_HALFS(TYPE, BYTES, NEON_SUF)                        \
  static SIMD_INLINE void unzip(const Vec<TYPE, 16> a, const Vec<TYPE, 16> b,  \
                                Vec<TYPE, 16> &c, Vec<TYPE, 16> &d,            \
                                Bytes<BYTES>)                                  \
  {                                                                            \
    c = vcombine_##NEON_SUF(vget_low_##NEON_SUF(a), vget_low_##NEON_SUF(b));   \
    d = vcombine_##NEON_SUF(vget_high_##NEON_SUF(a), vget_high_##NEON_SUF(b)); \
  }

SIMDVEC_NEON_UNZIP(Byte, 1, u8, u8, uint8x16x2_t)
SIMDVEC_NEON_UNZIP(Byte, 2, u8, u16, uint16x8x2_t)
SIMDVEC_NEON_UNZIP(Byte, 4, u8, u32, uint32x4x2_t)
SIMDVEC_NEON_UNZIP_HALFS(Byte, 8, u8)

SIMDVEC_NEON_UNZIP(SignedByte, 1, s8, s8, int8x16x2_t)
SIMDVEC_NEON_UNZIP(SignedByte, 2, s8, s16, int16x8x2_t)
SIMDVEC_NEON_UNZIP(SignedByte, 4, s8, s32, int32x4x2_t)
SIMDVEC_NEON_UNZIP_HALFS(SignedByte, 8, s8)

SIMDVEC_NEON_UNZIP(Word, 2, u16, u16, uint16x8x2_t)
SIMDVEC_NEON_UNZIP(Word, 4, u16, u32, uint32x4x2_t)
SIMDVEC_NEON_UNZIP_HALFS(Word, 8, u16)

SIMDVEC_NEON_UNZIP(Short, 2, s16, s16, int16x8x2_t)
SIMDVEC_NEON_UNZIP(Short, 4, s16, s32, int32x4x2_t)
SIMDVEC_NEON_UNZIP_HALFS(Short, 8, s16)

SIMDVEC_NEON_UNZIP(Int, 4, s32, s32, int32x4x2_t)
SIMDVEC_NEON_UNZIP_HALFS(Int, 8, s32)

SIMDVEC_NEON_UNZIP(Float, 4, f32, f32, float32x4x2_t)
SIMDVEC_NEON_UNZIP_HALFS(Float, 8, f32)

#ifdef SIMD_DOUBLE
static SIMD_INLINE void unzip(const Vec<Double, 16> a, const Vec<Double, 16> b,
                              Vec<Double, 16> &c, Vec<Double, 16> &d, Bytes<8>)
{
  c = vcombine_f64(vget_low_f64(a), vget_low_f64(b));
  d = vcombine_f64(vget_high_f64(a), vget_high_f64(b));
}
#endif

#undef SIMDVEC_NEON_UNZIP
#undef SIMDVEC_NEON_UNZIP_HALFS

// ---------------------------------------------------------------------------
// packs
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE Vec<SignedByte, 16> packs(const Vec<Short, 16> &a,
                                             const Vec<Short, 16> &b,
                                             OutputType<SignedByte>)
{
  return vcombine_s8(vqmovn_s16(a), vqmovn_s16(b));
}

static SIMD_INLINE Vec<Short, 16> packs(const Vec<Int, 16> &a,
                                        const Vec<Int, 16> &b,
                                        OutputType<Short>)
{
  return vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
}

static SIMD_INLINE Vec<Short, 16> packs(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b,
                                        OutputType<Short>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Short>());
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Int, 16> packs(const Vec<Double, 16> &a,
                                      const Vec<Double, 16> &b, OutputType<Int>)
{
  return vcombine_s32(vqmovn_s64(vcvtq_s64_f64(a)),
                      vqmovn_s64(vcvtq_s64_f64(b)));
}

static SIMD_INLINE Vec<Float, 16> packs(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b,
                                        OutputType<Float>)
{
  return vcombine_f32(vcvt_f32_f64(a), vcvt_f32_f64(b));
}
#endif

// signed -> unsigned

static SIMD_INLINE Vec<Byte, 16> packs(const Vec<Short, 16> &a,
                                       const Vec<Short, 16> &b,
                                       OutputType<Byte>)
{
  return vcombine_u8(vqmovun_s16(a), vqmovun_s16(b));
}

static SIMD_INLINE Vec<Word, 16> packs(const Vec<Int, 16> &a,
                                       const Vec<Int, 16> &b, OutputType<Word>)
{
  return vcombine_u16(vqmovun_s32(a), vqmovun_s32(b));
}

static SIMD_INLINE Vec<Word, 16> packs(const Vec<Float, 16> &a,
                                       const Vec<Float, 16> &b,
                                       OutputType<Word>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Word>());
}

// -------------------------------------------------------------------------
// generalized extend: no stage
// -------------------------------------------------------------------------

// from\to
//    SB B S W I F
// SB  x   x   x x
//  B    x x x x x
//  S      x   x x
//  W        x x x
//  I          x x
//  F          x x
//
// combinations:
// - signed   -> extended signed (sign extension)
// - unsigned -> extended unsigned (zero extension)
// - unsigned -> extended signed (zero extension)
// (signed -> extended unsigned is not possible)

// all types
template <typename T>
static SIMD_INLINE void extend(const Vec<T, 16> &vIn, Vec<T, 16> vOut[1])
{
  vOut[0] = vIn;
}

// -------------------------------------------------------------------------
// generalized extend: single stage
// -------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Short, 16> vOut[2])
{
  vOut[0] = vmovl_s8(vget_low_s8(vIn));
  vOut[1] = vmovl_s8(vget_high_s8(vIn));
}

static SIMD_INLINE void extend(const Vec<Short, 16> &vIn, Vec<Int, 16> vOut[2])
{
  vOut[0] = vmovl_s16(vget_low_s16(vIn));
  vOut[1] = vmovl_s16(vget_high_s16(vIn));
}

static SIMD_INLINE void extend(const Vec<Short, 16> &vIn,
                               Vec<Float, 16> vOut[2])
{
  vOut[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vIn)));
  vOut[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vIn)));
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE void extend(const Vec<Int, 16> &vIn, Vec<Double, 16> vOut[2])
{
  vOut[0] = vcvtq_f64_s64(vmovl_s32(vget_low_s32(vIn)));
  vOut[1] = vcvtq_f64_s64(vmovl_s32(vget_high_s32(vIn)));
}

static SIMD_INLINE void extend(const Vec<Float, 16> &vIn,
                               Vec<Double, 16> vOut[2])
{
  vOut[0] = vcvt_f64_f32(vget_low_f32(vIn));
  vOut[1] = vcvt_f64_f32(vget_high_f32(vIn));
}
#endif

// unsigned -> unsigned

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Word, 16> vOut[2])
{
  vOut[0] = vmovl_u8(vget_low_u8(vIn));
  vOut[1] = vmovl_u8(vget_high_u8(vIn));
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Short, 16> vOut[2])
{
  vOut[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vIn)));
  vOut[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vIn)));
}

static SIMD_INLINE void extend(const Vec<Word, 16> &vIn, Vec<Int, 16> vOut[2])
{
  vOut[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vIn)));
  vOut[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vIn)));
}

static SIMD_INLINE void extend(const Vec<Word, 16> &vIn, Vec<Float, 16> vOut[2])
{
  vOut[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vIn)));
  vOut[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vIn)));
}

// -------------------------------------------------------------------------
// generalized extend: two stages
// -------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Int, 16> vOut[4])
{
  Vec<Short, 16> vShort[2];
  extend(vIn, vShort);
  extend(vShort[0], vOut);
  extend(vShort[1], vOut + 2);
}

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Float, 16> vOut[4])
{
  Vec<Short, 16> vShort[2];
  extend(vIn, vShort);
  extend(vShort[0], vOut);
  extend(vShort[1], vOut + 2);
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE void extend(const Vec<Short, 16> &vIn,
                               Vec<Double, 16> vOut[4])
{
  Vec<Int, 16> vInt[2];
  extend(vIn, vInt);
  extend(vInt[0], vOut);
  extend(vInt[1], vOut + 2);
}
#endif

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Int, 16> vOut[4])
{
  Vec<Short, 16> vShort[2];
  extend(vIn, vShort);
  extend(vShort[0], vOut);
  extend(vShort[1], vOut + 2);
}

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Float, 16> vOut[4])
{
  Vec<Short, 16> vShort[2];
  extend(vIn, vShort);
  extend(vShort[0], vOut);
  extend(vShort[1], vOut + 2);
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE void extend(const Vec<Word, 16> &vIn,
                               Vec<Double, 16> vOut[4])
{
  Vec<Int, 16> vInt[2];
  extend(vIn, vInt);
  extend(vInt[0], vOut);
  extend(vInt[1], vOut + 2);
}
#endif

// -------------------------------------------------------------------------
// generalized extend: three stages
// -------------------------------------------------------------------------

// signed -> signed

#ifdef SIMD_DOUBLE
static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Double, 16> vOut[8])
{
  Vec<Int, 16> vInt[4];
  extend(vIn, vInt);
  extend(vInt[0], vOut);
  extend(vInt[1], vOut + 2);
  extend(vInt[2], vOut + 4);
  extend(vInt[3], vOut + 6);
}
#endif

// unsigned -> signed

#ifdef SIMD_DOUBLE
static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn,
                               Vec<Double, 16> vOut[8])
{
  Vec<Int, 16> vInt[4];
  extend(vIn, vInt);
  extend(vInt[0], vOut);
  extend(vInt[1], vOut + 2);
  extend(vInt[2], vOut + 4);
  extend(vInt[3], vOut + 6);
}
#endif

// -------------------------------------------------------------------------
// generalized extend: special case int <-> float
// -------------------------------------------------------------------------

static SIMD_INLINE void extend(const Vec<Int, 16> &vIn, Vec<Float, 16> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Float, 16> &vIn, Vec<Int, 16> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Int>());
}

// -------------------------------------------------------------------------
// shift functions
// -------------------------------------------------------------------------

// it was necessary to introduce a special case COUNT == 0, since this
// is not allowed for the shift intrinsics (just returns the
// argument); since the ARM docs aren't clear in this point, we also
// treat the case COUNT == no-of-bits as special case (in two
// versions: one using FCT on RANGE-1, the other setting result to
// zero)

#define SIMDVEC_NEON_SHIFT(FCT, TYPE, NEON_FCT, NEON_SUF, RANGE)               \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       IsPosNonZeroInRange<true, true>)        \
  {                                                                            \
    return NEON_FCT##_##NEON_SUF(a, COUNT);                                    \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       IsPosNonZeroInRange<false, true>)       \
  {                                                                            \
    return a;                                                                  \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a)                 \
  {                                                                            \
    return FCT<COUNT>(a, IsPosNonZeroInGivenRange<RANGE, COUNT>());            \
  }

// out-of-range implemented with FCT of RANGE-1
#define SIMDVEC_NEON_SHIFT_ARITH(FCT, TYPE, NEON_FCT, NEON_SUF, RANGE)         \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       IsPosNonZeroInRange<true, false>)       \
  {                                                                            \
    return NEON_FCT##_##NEON_SUF(a, (RANGE - 1));                              \
  }                                                                            \
  SIMDVEC_NEON_SHIFT(FCT, TYPE, NEON_FCT, NEON_SUF, RANGE)

// out-of-range implemented with set-to-zero
#define SIMDVEC_NEON_SHIFT_LOGICAL(FCT, TYPE, NEON_FCT, NEON_SUF, RANGE)       \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &,                  \
                                       IsPosNonZeroInRange<true, false>)       \
  {                                                                            \
    return vmovq_n_##NEON_SUF(TYPE(0));                                        \
  }                                                                            \
  SIMDVEC_NEON_SHIFT(FCT, TYPE, NEON_FCT, NEON_SUF, RANGE)

#define SIMDVEC_NEON_SHIFT_REINTER(FCT, TYPE, NFCT, NSUF, NSUF2, RANGE)        \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       IsPosNonZeroInRange<true, true>)        \
  {                                                                            \
    return vreinterpretq_##NSUF##_##NSUF2(                                     \
      NFCT##_##NSUF2(vreinterpretq_##NSUF2##_##NSUF(a), COUNT));               \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       IsPosNonZeroInRange<false, true>)       \
  {                                                                            \
    return a;                                                                  \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a)                 \
  {                                                                            \
    return FCT<COUNT>(a, IsPosNonZeroInGivenRange<RANGE, COUNT>());            \
  }

#define SIMDVEC_NEON_SHIFT_REINTER_ARITH(FCT, TYPE, NFCT, NSUF, NSUF2, RANGE)  \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &a,                 \
                                       IsPosNonZeroInRange<true, false>)       \
  {                                                                            \
    return vreinterpretq_##NSUF##_##NSUF2(                                     \
      NFCT##_##NSUF2(vreinterpretq_##NSUF2##_##NSUF(a), (RANGE - 1)));         \
  }                                                                            \
  SIMDVEC_NEON_SHIFT_REINTER(FCT, TYPE, NFCT, NSUF, NSUF2, RANGE)

#define SIMDVEC_NEON_SHIFT_REINTER_LOGICAL(FCT, TYPE, NFCT, NSUF, NSUF2,       \
                                           RANGE)                              \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> FCT(const Vec<TYPE, 16> &,                  \
                                       IsPosNonZeroInRange<true, false>)       \
  {                                                                            \
    return vmovq_n_##NSUF(TYPE(0));                                            \
  }                                                                            \
  SIMDVEC_NEON_SHIFT_REINTER(FCT, TYPE, NFCT, NSUF, NSUF2, RANGE)

// srai

// requires cast of unsigned types to signed!
// http://stackoverflow.com/questions/18784988/neon-intrinsic-for-arithmetic-shift
// out-of-range case handled with FCT=srai

// 13. Nov 22 (Jonas Keller):
// added missing Byte and SignedByte versions of srai

SIMDVEC_NEON_SHIFT_REINTER_ARITH(srai, Byte, vshrq_n, u8, s8, 8)
SIMDVEC_NEON_SHIFT_ARITH(srai, SignedByte, vshrq_n, s8, 8)
SIMDVEC_NEON_SHIFT_REINTER_ARITH(srai, Word, vshrq_n, u16, s16, 16)
SIMDVEC_NEON_SHIFT_ARITH(srai, Short, vshrq_n, s16, 16)
SIMDVEC_NEON_SHIFT_ARITH(srai, Int, vshrq_n, s32, 32)

// srli

// requires cast of signed types to unsigned!
// http://stackoverflow.com/questions/18784988/neon-intrinsic-for-arithmetic-shift
// out-of-range case handled with set-to-zero

SIMDVEC_NEON_SHIFT_LOGICAL(srli, Byte, vshrq_n, u8, 8)
SIMDVEC_NEON_SHIFT_REINTER_LOGICAL(srli, SignedByte, vshrq_n, s8, u8, 8)
SIMDVEC_NEON_SHIFT_LOGICAL(srli, Word, vshrq_n, u16, 16)
SIMDVEC_NEON_SHIFT_REINTER_LOGICAL(srli, Short, vshrq_n, s16, u16, 16)
SIMDVEC_NEON_SHIFT_REINTER_LOGICAL(srli, Int, vshrq_n, s32, u32, 32)

// slli

// out-of-range case handled with set-to-zero
SIMDVEC_NEON_SHIFT_LOGICAL(slli, Byte, vshlq_n, u8, 8)
SIMDVEC_NEON_SHIFT_LOGICAL(slli, SignedByte, vshlq_n, s8, 8)
SIMDVEC_NEON_SHIFT_LOGICAL(slli, Word, vshlq_n, u16, 16)
SIMDVEC_NEON_SHIFT_LOGICAL(slli, Short, vshlq_n, s16, 16)
SIMDVEC_NEON_SHIFT_LOGICAL(slli, Int, vshlq_n, s32, 32)

#undef SIMDVEC_NEON_SHIFT
#undef SIMDVEC_NEON_SHIFT_ARITH
#undef SIMDVEC_NEON_SHIFT_LOGICAL
#undef SIMDVEC_NEON_SHIFT_REINTER
#undef SIMDVEC_NEON_SHIFT_REINTER_ARITH
#undef SIMDVEC_NEON_SHIFT_REINTER_LOGICAL

// 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

// -------------------------------------------------------------------------
// sra
// -------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> sra(const Vec<Byte, 16> &a,
                                     const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(8)));
  return vreinterpretq_u8_s8(
    vshlq_s8(vreinterpretq_s8_u8(a), vdupq_n_s8(scount)));
}

static SIMD_INLINE Vec<SignedByte, 16> sra(const Vec<SignedByte, 16> &a,
                                           const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(8)));
  return vshlq_s8(a, vdupq_n_s8(scount));
}

static SIMD_INLINE Vec<Word, 16> sra(const Vec<Word, 16> &a,
                                     const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(16)));
  return vreinterpretq_u16_s16(
    vshlq_s16(vreinterpretq_s16_u16(a), vdupq_n_s16(scount)));
}

static SIMD_INLINE Vec<Short, 16> sra(const Vec<Short, 16> &a,
                                      const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(16)));
  return vshlq_s16(a, vdupq_n_s16(scount));
}

static SIMD_INLINE Vec<Int, 16> sra(const Vec<Int, 16> &a, const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(32)));
  return vshlq_s32(a, vdupq_n_s32(scount));
}

// -------------------------------------------------------------------------
// srl
// -------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> srl(const Vec<Byte, 16> &a,
                                     const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(8)));
  return vshlq_u8(a, vdupq_n_s8(scount));
}

static SIMD_INLINE Vec<SignedByte, 16> srl(const Vec<SignedByte, 16> &a,
                                           const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(8)));
  return vreinterpretq_s8_u8(
    vshlq_u8(vreinterpretq_u8_s8(a), vdupq_n_s8(scount)));
}

static SIMD_INLINE Vec<Word, 16> srl(const Vec<Word, 16> &a,
                                     const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(16)));
  return vshlq_u16(a, vdupq_n_s16(scount));
}

static SIMD_INLINE Vec<Short, 16> srl(const Vec<Short, 16> &a,
                                      const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(16)));
  return vreinterpretq_s16_u16(
    vshlq_u16(vreinterpretq_u16_s16(a), vdupq_n_s16(scount)));
}

static SIMD_INLINE Vec<Int, 16> srl(const Vec<Int, 16> &a, const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  int8_t scount = -((int8_t) std::min(count, uint8_t(32)));
  return vreinterpretq_s32_u32(
    vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(scount)));
}

// -------------------------------------------------------------------------
// sll
// -------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> sll(const Vec<Byte, 16> &a,
                                     const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  return vshlq_u8(a, vdupq_n_s8(std::min(count, uint8_t(8))));
}

static SIMD_INLINE Vec<SignedByte, 16> sll(const Vec<SignedByte, 16> &a,
                                           const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  return vshlq_s8(a, vdupq_n_s8(std::min(count, uint8_t(8))));
}

static SIMD_INLINE Vec<Word, 16> sll(const Vec<Word, 16> &a,
                                     const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  return vshlq_u16(a, vdupq_n_s16(std::min(count, uint8_t(16))));
}

static SIMD_INLINE Vec<Short, 16> sll(const Vec<Short, 16> &a,
                                      const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  return vshlq_s16(a, vdupq_n_s16(std::min(count, uint8_t(16))));
}

static SIMD_INLINE Vec<Int, 16> sll(const Vec<Int, 16> &a, const uint8_t count)
{
  if (count == 0) {
    // TODO: is this necessary? what does vshlq do for count==0?
    return a;
  }
  return vshlq_s32(a, vdupq_n_s32(std::min(count, uint8_t(32))));
}

// 26. Sep 22 (Jonas Keller):
// added Byte and SignedByte versions of hadd, hadds, hsub and hsubs
// added Word version of hadds and hsubs

// -------------------------------------------------------------------------
// hadd
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_HADD(TYPE, NEON_SUF)                                      \
  static SIMD_INLINE Vec<TYPE, 16> hadd(const Vec<TYPE, 16> &a,                \
                                        const Vec<TYPE, 16> &b)                \
  {                                                                            \
    return vcombine_##NEON_SUF(                                                \
      vpadd_##NEON_SUF(vget_low_##NEON_SUF(a), vget_high_##NEON_SUF(a)),       \
      vpadd_##NEON_SUF(vget_low_##NEON_SUF(b), vget_high_##NEON_SUF(b)));      \
  }

SIMDVEC_NEON_HADD(Byte, u8)
SIMDVEC_NEON_HADD(SignedByte, s8)
SIMDVEC_NEON_HADD(Word, u16)
SIMDVEC_NEON_HADD(Short, s16)
SIMDVEC_NEON_HADD(Int, s32)
SIMDVEC_NEON_HADD(Float, f32)
#ifdef SIMD_DOUBLE
// vpadd_f64 does not exist, because float64x1_t is just a double, so we use
// the regular plus operator for double
static SIMD_INLINE Vec<Double, 16> hadd(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b)
{
  return vcombine_f64(vget_low_f64(a) + vget_high_f64(a),
                      vget_low_f64(b) + vget_high_f64(b));
}
#endif

#undef SIMDVEC_NEON_HADD

// -------------------------------------------------------------------------
// hadds
// -------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> hadds(const Vec<Byte, 16> &a,
                                       const Vec<Byte, 16> &b)
{
  Vec<Byte, 16> x, y;
  unzip(a, b, x, y, Bytes<1>());
  return adds(x, y);
}

static SIMD_INLINE Vec<SignedByte, 16> hadds(const Vec<SignedByte, 16> &a,
                                             const Vec<SignedByte, 16> &b)
{
  Vec<SignedByte, 16> x, y;
  unzip(a, b, x, y, Bytes<1>());
  return adds(x, y);
}

static SIMD_INLINE Vec<Word, 16> hadds(const Vec<Word, 16> &a,
                                       const Vec<Word, 16> &b)
{
  Vec<Word, 16> x, y;
  unzip(a, b, x, y, Bytes<2>());
  return adds(x, y);
}

static SIMD_INLINE Vec<Short, 16> hadds(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return vcombine_s16(vqmovn_s32(vpaddlq_s16(a)), vqmovn_s32(vpaddlq_s16(b)));
}

static SIMD_INLINE Vec<Int, 16> hadds(const Vec<Int, 16> &a,
                                      const Vec<Int, 16> &b)
{
  return vcombine_s32(vqmovn_s64(vpaddlq_s32(a)), vqmovn_s64(vpaddlq_s32(b)));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 16> hadds(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return hadd(a, b);
}

#ifdef SIMD_DOUBLE
// Double not saturated
static SIMD_INLINE Vec<Double, 16> hadds(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return hadd(a, b);
}
#endif

// -------------------------------------------------------------------------
// hsub
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> hsub(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  Vec<T, 16> x, y;
  unzip(a, b, x, y, Bytes<sizeof(T)>());
  return sub(x, y);
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> hsub(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b)
{
  return vcombine_f64(vget_low_f64(a) - vget_high_f64(a),
                      vget_low_f64(b) - vget_high_f64(b));
}
#endif

// -------------------------------------------------------------------------
// hsubs
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> hsubs(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  Vec<T, 16> x, y;
  unzip(a, b, x, y, Bytes<sizeof(T)>());
  return subs(x, y);
}

#ifdef SIMD_DOUBLE
// Double not saturated
static SIMD_INLINE Vec<Double, 16> hsubs(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return vcombine_f64(vget_low_f64(a) - vget_high_f64(a),
                      vget_low_f64(b) - vget_high_f64(b));
}
#endif

// -------------------------------------------------------------------------
// alignre (moved above srle, slle)
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_ALIGNRE(TYPE, NEON_SUF)                                   \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> alignre(                                    \
    const Vec<TYPE, 16> &, const Vec<TYPE, 16> &l,                             \
    Range<true, 0, Vec<TYPE, 16>::elements>)                                   \
  {                                                                            \
    return l;                                                                  \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> alignre(                                    \
    const Vec<TYPE, 16> &h, const Vec<TYPE, 16> &l,                            \
    Range<false, 0, Vec<TYPE, 16>::elements>)                                  \
  {                                                                            \
    return vextq_##NEON_SUF(l, h, COUNT);                                      \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> alignre(                                    \
    const Vec<TYPE, 16> &h, const Vec<TYPE, 16> &,                             \
    Range<true, Vec<TYPE, 16>::elements, 2 * Vec<TYPE, 16>::elements>)         \
  {                                                                            \
    return h;                                                                  \
  }                                                                            \
  template <size_t COUNT>                                                      \
  static SIMD_INLINE Vec<TYPE, 16> alignre(                                    \
    const Vec<TYPE, 16> &h, const Vec<TYPE, 16> &,                             \
    Range<false, Vec<TYPE, 16>::elements, 2 * Vec<TYPE, 16>::elements>)        \
  {                                                                            \
    return vextq_##NEON_SUF(h, vmovq_n_##NEON_SUF(TYPE(0)),                    \
                            COUNT - Vec<TYPE, 16>::elements);                  \
  }                                                                            \
  template <size_t COUNT, bool AT_LL, size_t LL_INCL, size_t UL_EXCL>          \
  static SIMD_INLINE Vec<TYPE, 16> alignre(const Vec<TYPE, 16> &,              \
                                           const Vec<TYPE, 16> &,              \
                                           Range<AT_LL, LL_INCL, UL_EXCL>)     \
  {                                                                            \
    return vmovq_n_##NEON_SUF(TYPE(0));                                        \
  }

SIMDVEC_NEON_ALIGNRE(Byte, u8)
SIMDVEC_NEON_ALIGNRE(SignedByte, s8)
SIMDVEC_NEON_ALIGNRE(Word, u16)
SIMDVEC_NEON_ALIGNRE(Short, s16)
SIMDVEC_NEON_ALIGNRE(Int, s32)
SIMDVEC_NEON_ALIGNRE(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_ALIGNRE(Double, f64)
#endif

template <size_t COUNT, typename T>
static SIMD_INLINE Vec<T, 16> alignre(const Vec<T, 16> &h, const Vec<T, 16> &l)
{
  return alignre<COUNT>(h, l, SizeRange<COUNT, Vec<T, 16>::elements>());
}

#undef SIMDVEC_NEON_ALIGNRE

// -------------------------------------------------------------------------
// element-wise shift right
// -------------------------------------------------------------------------

// all types, done via alignre
template <size_t COUNT, typename T>
static SIMD_INLINE Vec<T, 16> srle(const Vec<T, 16> &a)
{
  return alignre<COUNT>(setzero(OutputType<T>(), Integer<16>()), a);
}

// -------------------------------------------------------------------------
// element-wise shift left
// -------------------------------------------------------------------------

// all types, done via alignre

template <size_t COUNT, typename T>
static SIMD_INLINE Vec<T, 16> slle(const Vec<T, 16> &a)
{
  SIMD_IF_CONSTEXPR (COUNT < Vec<T, 16>::elements) {
    return alignre<Vec<T, 16>::elements - COUNT>(
      a, setzero(OutputType<T>(), Integer<16>()));
  } else {
    return setzero(OutputType<T>(), Integer<16>());
  }
}

// -------------------------------------------------------------------------
// swizzle
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 16>[1], Integer<1>)
{
  // v remains unchanged
}

template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 16> v[2], Integer<2>)
{
  uint8x8x2_t vu[2];
  for (size_t j = 0; j < 2; j++) {
    const auto vByte = reinterpret(v[j], OutputType<Byte>());
    vu[0].val[j]     = vget_low_u8(vByte);
    vu[1].val[j]     = vget_high_u8(vByte);
  }
  const uint8x8_t table[2] = {
    {0, 2, 4, 6, 8, 10, 12, 14},
    {1, 3, 5, 7, 9, 11, 13, 15},
  };
  for (size_t i = 0; i < 2; i++) {
    v[i] = reinterpret(Vec<Byte, 16>(vcombine_u8(vtbl2_u8(vu[0], table[i]),
                                                 vtbl2_u8(vu[1], table[i]))),
                       OutputType<T>());
  }
}

template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 16> v[3], Integer<3>)
{
  uint8x8x3_t vu[2];
  for (size_t j = 0; j < 3; j++) {
    const auto vByte = reinterpret(v[j], OutputType<Byte>());
    vu[0].val[j]     = vget_low_u8(vByte);
    vu[1].val[j]     = vget_high_u8(vByte);
  }
  const uint8x8_t table[3] = {
    {0, 3, 6, 9, 12, 15, 18, 21},
    {1, 4, 7, 10, 13, 16, 19, 22},
    {2, 5, 8, 11, 14, 17, 20, 23},
  };
  for (size_t i = 0; i < 3; i++) {
    v[i] = reinterpret(Vec<Byte, 16>(vcombine_u8(vtbl3_u8(vu[0], table[i]),
                                                 vtbl3_u8(vu[1], table[i]))),
                       OutputType<T>());
  }
}

template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 16> v[4], Integer<4>)
{
  uint8x8x4_t vu[2];
  for (size_t j = 0; j < 4; j++) {
    const auto vByte = reinterpret(v[j], OutputType<Byte>());
    vu[0].val[j]     = vget_low_u8(vByte);
    vu[1].val[j]     = vget_high_u8(vByte);
  }
  const uint8x8_t table[4] = {
    {0, 4, 8, 12, 16, 20, 24, 28},
    {1, 5, 9, 13, 17, 21, 25, 29},
    {2, 6, 10, 14, 18, 22, 26, 30},
    {3, 7, 11, 15, 19, 23, 27, 31},
  };
  for (size_t i = 0; i < 4; i++) {
    v[i] = reinterpret(Vec<Byte, 16>(vcombine_u8(vtbl4_u8(vu[0], table[i]),
                                                 vtbl4_u8(vu[1], table[i]))),
                       OutputType<T>());
  }
}

// ---------- n = 5 ----------

// swizzle table

// arrays are padded from 24 to 32 elements to keep alignment
static const uint8_t swizzleMask5Lo[9][32] SIMD_ATTR_ALIGNED(16) = {
  {},
  {0, 5, 10, 15, 1, 6, 11, 16, 2,  7,  12, 17,
   3, 8, 13, 18, 4, 9, 14, 19, 99, 99, 99, 99},
  {0, 1, 10, 11, 2, 3, 12, 13, 4,  5,  14, 15,
   6, 7, 16, 17, 8, 9, 18, 19, 99, 99, 99, 99},
  {},
  {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
   12, 13, 14, 15, 16, 17, 18, 19, 99, 99, 99, 99},
  {},
  {},
  {},
  {}, // TODO
};

// arrays are padded from 24 to 32 elements to keep alignment
static const uint8_t swizzleMask5Hi[9][32] SIMD_ATTR_ALIGNED(16) = {
  {},
  {4, 9,  14, 19, 5, 10, 15, 20, 6,  11, 16, 21,
   7, 12, 17, 22, 8, 13, 18, 23, 99, 99, 99, 99},
  {4,  5,  14, 15, 6,  7,  16, 17, 8,  9,  18, 19,
   10, 11, 20, 21, 12, 13, 22, 23, 99, 99, 99, 99},
  {},
  {4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
   16, 17, 18, 19, 20, 21, 22, 23, 99, 99, 99, 99},
  {},
  {},
  {},
  {}, // TODO
};

// n = 5
template <size_t SIZE>
struct SwizzleTable5
{
  // two tables for n=3
  uint8x8x3_t table[2];
  SwizzleTable5()
  {
    for (size_t i = 0; i < 3; i++) {
      // first half (applied to vectors 0,1,2)
      table[0].val[i] = vld1_u8(&swizzleMask5Lo[SIZE][i * 8]);
      // second half (applied to vectors 2,3,4)
      table[1].val[i] = vld1_u8(&swizzleMask5Hi[SIZE][i * 8]);
    }
  }
};

// n = 5
template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 16> v[5], Integer<5>)
{
  //     |  v0l  v0h  |  v1l  v1h  |  v2l  v2h  |  v3l  v3h  |  v4l  v4h  |
  // i=0:
  // k:      0    1       2
  // j:      0    1       2
  //     | vu0.0 v0.1  vu0.2|
  // i=1:
  // k:                   2    3       4
  // j:                   0    1       2
  //                  |vu1.0 vu1.1   vu1.2|
  // i=2:
  // k:                                     5       6    7
  // j:                                     0       1    2
  //                                      |vu2.0  v2.1 vu2.2|
  // i=3:
  // k:                                                  7       8    9
  // j:                                                  0       1    2
  //                                                  |vu3.0  vu3.1 vu3.2 |
  //
  //       n=0:                             n=1:
  //       i=0:         i=1:                i=0:        i=1:
  //       k=0:         k=1:                k=2:        k=3:
  // j=0:
  //     | t.table[0].val[0]|             | t.table[0].val[0]|
  //                  | t.table[1].val[0] |           | t.table[1].val[0] |
  // j=1:
  //     | t.table[0].val[1]|             | t.table[0].val[1]|
  //                  | t.table[1].val[1] |           | t.table[1].val[1] |
  // j=2:
  //     | t.table[0].val[2]|             | t.table[0].val[2]|
  //                  | t.table[1].val[2] |           | t.table[1].val[2] |

  uint8x8x3_t vu[4];
  // input half-vector index starts at k0
  const size_t k0[4] = {0, 2, 5, 7};
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 3; j++) {
      const size_t k         = k0[i] + j;
      const Vec<Byte, 16> vb = reinterpret(v[k >> 1], OutputType<Byte>());
      vu[i].val[j]           = (k & 1) ? vget_high_u8(vb) : vget_low_u8(vb);
    }
  }
  static const SwizzleTable5<sizeof(T)> t;
  uint8x8_t r[2][3][3];
  // n: left/right half of input
  // k: index of vu
  for (size_t n = 0, k = 0; n < 2; n++)
    // i: left/right half of half input
    for (size_t i = 0; i < 2; i++, k++)
      // j: different 3-tables
      for (size_t j = 0; j < 3; j++)
        // apply table
        r[n][i][j] = vtbl3_u8(vu[k], t.table[i].val[j]);
  // zip 4-byte blocks together
  int32x2x2_t z[2][3];
  for (size_t n = 0; n < 2; n++)
    for (size_t j = 0; j < 3; j++)
      z[n][j] = vzip_s32(vreinterpret_s32_u8(r[n][0][j]),
                         vreinterpret_s32_u8(r[n][1][j]));
  // combine left and right halfs
  for (size_t j = 0, k = 0; j < 3; j++) {
    for (size_t lh = 0; lh < 2; lh++) {
      v[k] = reinterpret(
        Vec<Int, 16>(vcombine_s32(z[0][j].val[lh], z[1][j].val[lh])),
        OutputType<T>());
      k++;
      if (k >= 5) break;
    }
  }
}

// -------------------------------------------------------------------------
// compare functions
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_CMP(CMP, TYPE, NEON_SUF, NEON_USUF)                       \
  static SIMD_INLINE Vec<TYPE, 16> cmp##CMP(const Vec<TYPE, 16> &a,            \
                                            const Vec<TYPE, 16> &b)            \
  {                                                                            \
    return vreinterpretq_##NEON_SUF##_##NEON_USUF(                             \
      vc##CMP##q##_##NEON_SUF(a, b));                                          \
  }

#ifdef SIMD_DOUBLE
#define SIMDVEC_NEON_CMP_ALL(CMP)                                              \
  SIMDVEC_NEON_CMP(CMP, Byte, u8, u8)                                          \
  SIMDVEC_NEON_CMP(CMP, SignedByte, s8, u8)                                    \
  SIMDVEC_NEON_CMP(CMP, Word, u16, u16)                                        \
  SIMDVEC_NEON_CMP(CMP, Short, s16, u16)                                       \
  SIMDVEC_NEON_CMP(CMP, Int, s32, u32)                                         \
  SIMDVEC_NEON_CMP(CMP, Float, f32, u32)                                       \
  SIMDVEC_NEON_CMP(CMP, Double, f64, u64)
#else
#define SIMDVEC_NEON_CMP_ALL(CMP)                                              \
  SIMDVEC_NEON_CMP(CMP, Byte, u8, u8)                                          \
  SIMDVEC_NEON_CMP(CMP, SignedByte, s8, u8)                                    \
  SIMDVEC_NEON_CMP(CMP, Word, u16, u16)                                        \
  SIMDVEC_NEON_CMP(CMP, Short, s16, u16)                                       \
  SIMDVEC_NEON_CMP(CMP, Int, s32, u32)                                         \
  SIMDVEC_NEON_CMP(CMP, Float, f32, u32)
#endif

SIMDVEC_NEON_CMP_ALL(lt)
SIMDVEC_NEON_CMP_ALL(le)
SIMDVEC_NEON_CMP_ALL(eq)
SIMDVEC_NEON_CMP_ALL(gt)
SIMDVEC_NEON_CMP_ALL(ge)

#undef SIMDVEC_NEON_CMP_ALL
#undef SIMDVEC_NEON_CMP

// -------------------------------------------------------------------------
// compare !=
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_CMPNEQ(TYPE, NEON_SUF, NEON_USUF)                         \
  static SIMD_INLINE Vec<TYPE, 16> cmpneq(const Vec<TYPE, 16> &a,              \
                                          const Vec<TYPE, 16> &b)              \
  {                                                                            \
    return vreinterpretq_##NEON_SUF##_##NEON_USUF(                             \
      vmvnq_##NEON_USUF(vceqq_##NEON_SUF(a, b)));                              \
  }

SIMDVEC_NEON_CMPNEQ(Byte, u8, u8)
SIMDVEC_NEON_CMPNEQ(SignedByte, s8, u8)
SIMDVEC_NEON_CMPNEQ(Word, u16, u16)
SIMDVEC_NEON_CMPNEQ(Short, s16, u16)
SIMDVEC_NEON_CMPNEQ(Int, s32, u32)
SIMDVEC_NEON_CMPNEQ(Float, f32, u32)
#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> cmpneq(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b)
{
  return vreinterpretq_f64_u32(
    vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(a, b))));
}
#endif

#undef SIMDVEC_NEON_CMPNEQ

// -------------------------------------------------------------------------
// ifelse
// -------------------------------------------------------------------------

// vbslq, unsigned mask
#define SIMDVEC_NEON_IFELSE(T, NEON_SUF, NEON_USUF)                            \
  static SIMD_INLINE Vec<T, 16> ifelse(const Vec<T, 16> &cond,                 \
                                       const Vec<T, 16> &trueVal,              \
                                       const Vec<T, 16> &falseVal)             \
  {                                                                            \
    return vbslq_##NEON_SUF(vreinterpretq_##NEON_USUF##_##NEON_SUF(cond),      \
                            trueVal, falseVal);                                \
  }

SIMDVEC_NEON_IFELSE(Byte, u8, u8)
SIMDVEC_NEON_IFELSE(SignedByte, s8, u8)
SIMDVEC_NEON_IFELSE(Word, u16, u16)
SIMDVEC_NEON_IFELSE(Short, s16, u16)
SIMDVEC_NEON_IFELSE(Int, s32, u32)
SIMDVEC_NEON_IFELSE(Float, f32, u32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_IFELSE(Double, f64, u64)
#endif

#undef SIMDVEC_NEON_IFELSE

// -------------------------------------------------------------------------
// bit_and
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALLINT(bit_and, vandq)

static SIMD_INLINE Vec<Float, 16> bit_and(const Vec<Float, 16> &a,
                                          const Vec<Float, 16> &b)
{
  return vreinterpretq_f32_s32(
    vandq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> bit_and(const Vec<Double, 16> &a,
                                           const Vec<Double, 16> &b)
{
  return vreinterpretq_f64_s64(
    vandq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)));
}
#endif

// -------------------------------------------------------------------------
// bit_or
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALLINT(bit_or, vorrq)

static SIMD_INLINE Vec<Float, 16> bit_or(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b)
{
  return vreinterpretq_f32_s32(
    vorrq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> bit_or(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b)
{
  return vreinterpretq_f64_s64(
    vorrq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)));
}
#endif

// -------------------------------------------------------------------------
// bit_andnot
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> bit_andnot(const Vec<T, 16> &a,
                                         const Vec<T, 16> &b)
{
  return bit_and(bit_not(a), b);
}

// -------------------------------------------------------------------------
// bit_xor
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALLINT(bit_xor, veorq)

static SIMD_INLINE Vec<Float, 16> bit_xor(const Vec<Float, 16> &a,
                                          const Vec<Float, 16> &b)
{
  return vreinterpretq_f32_s32(
    veorq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b)));
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> bit_xor(const Vec<Double, 16> &a,
                                           const Vec<Double, 16> &b)
{
  return vreinterpretq_f64_s64(
    veorq_s64(vreinterpretq_s64_f64(a), vreinterpretq_s64_f64(b)));
}
#endif

// -------------------------------------------------------------------------
// bit_not
// -------------------------------------------------------------------------

SIMDVEC_NEON_UNARY(bit_not, Byte, vmvnq, u8)
SIMDVEC_NEON_UNARY(bit_not, SignedByte, vmvnq, s8)
SIMDVEC_NEON_UNARY(bit_not, Word, vmvnq, u16)
SIMDVEC_NEON_UNARY(bit_not, Short, vmvnq, s16)
SIMDVEC_NEON_UNARY(bit_not, Int, vmvnq, s32)

static SIMD_INLINE Vec<Float, 16> bit_not(const Vec<Float, 16> &a)
{
  return vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(a)));
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> bit_not(const Vec<Double, 16> &a)
{
  return vreinterpretq_f64_s32(vmvnq_s32(vreinterpretq_s32_f64(a)));
}
#endif

// -------------------------------------------------------------------------
// avg: average with rounding up
// -------------------------------------------------------------------------

SIMDVEC_NEON_BINARY_ALLINT(avg, vrhaddq)

static SIMD_INLINE Vec<Float, 16> avg(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return vmulq_n_f32(vaddq_f32(a, b), 0.5f);
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> avg(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return vmulq_n_f64(vaddq_f64(a, b), 0.5);
}
#endif

// -------------------------------------------------------------------------
// test_all_zeros
// -------------------------------------------------------------------------

// from solution suggested by Henri Ylitie
// http://stackoverflow.com/questions/15389539/
//   fastest-way-to-test-a-128-bit-neon-register-
//   for-a-value-of-0-using-intrinsics

static SIMD_INLINE float32x2_t vorr_f32(float32x2_t a, float32x2_t b)
{
  return vreinterpret_f32_s32(
    vorr_s32(vreinterpret_s32_f32(a), vreinterpret_s32_f32(b)));
}

// vpmax has to operate on unsigned (u32), otherwise 0 could be
// the max. of a pair even though the other value is non-zero (neg.)
#define SIMDVEC_NEON_TESTALLZEROS(T, NEON_SUF)                                 \
  static SIMD_INLINE bool test_all_zeros(const Vec<T, 16> &a)                  \
  {                                                                            \
    uint32x4_t au  = vreinterpretq_u32_##NEON_SUF(a);                          \
    uint32x2_t tmp = vorr_u32(vget_low_u32(au), vget_high_u32(au));            \
    return !vget_lane_u32(vpmax_u32(tmp, tmp), 0);                             \
  }

SIMDVEC_NEON_TESTALLZEROS(Byte, u8)
SIMDVEC_NEON_TESTALLZEROS(SignedByte, s8)
SIMDVEC_NEON_TESTALLZEROS(Word, u16)
SIMDVEC_NEON_TESTALLZEROS(Short, s16)
SIMDVEC_NEON_TESTALLZEROS(Int, s32)
SIMDVEC_NEON_TESTALLZEROS(Float, f32)
#ifdef SIMD_DOUBLE
SIMDVEC_NEON_TESTALLZEROS(Double, f64)
#endif

#undef SIMDVEC_NEON_TESTALLZEROS

// -------------------------------------------------------------------------
// test_all_ones
// -------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE bool test_all_ones(const Vec<T, 16> &a)
{
  return test_all_zeros(bit_not(a));
}

// -------------------------------------------------------------------------
// reverse
// -------------------------------------------------------------------------

// https://stackoverflow.com/questions/18760784/reverse-vector-order-in-arm-neon-intrinsics

#define SIMDVEC_NEON_REVERSE(T, NEON_SUF)                                      \
  static SIMD_INLINE Vec<T, 16> reverse(const Vec<T, 16> &a)                   \
  {                                                                            \
    const auto t = vrev64q_##NEON_SUF(a);                                      \
    return vcombine_##NEON_SUF(vget_high_##NEON_SUF(t),                        \
                               vget_low_##NEON_SUF(t));                        \
  }

SIMDVEC_NEON_REVERSE(Byte, u8)
SIMDVEC_NEON_REVERSE(SignedByte, s8)
SIMDVEC_NEON_REVERSE(Word, u16)
SIMDVEC_NEON_REVERSE(Short, s16)
SIMDVEC_NEON_REVERSE(Int, s32)
SIMDVEC_NEON_REVERSE(Float, f32)
#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> reverse(const Vec<Double, 16> &a)
{
  return vcombine_f64(vget_high_f64(a), vget_low_f64(a));
}
#endif

#undef SIMDVEC_NEON_REVERSE

// ---------------------------------------------------------------------------
// msb2int
// ---------------------------------------------------------------------------

// 17. Sep 22 (Jonas Keller): added msb2int functions

static SIMD_INLINE uint64_t msb2int(const Vec<Byte, 16> &a)
{
  // from: https://stackoverflow.com/a/58381188/8461272

  // Example input (half scale):
  // 0x89 FF 1D C0 00 10 99 33

  // Shift out everything but the sign bits
  // 0x01 01 00 01 00 00 01 00
  uint8x16_t high_bits = vshrq_n_u8(a, 7);

  // Merge the even lanes together with vsra. The '??' bytes are garbage.
  // vsri could also be used, but it is slightly slower on aarch64.
  // 0x??03 ??02 ??00 ??01
  uint16x8_t paired16 = vsraq_n_u16(vreinterpretq_u16_u8(high_bits),
                                    vreinterpretq_u16_u8(high_bits), 7);
  // Repeat with wider lanes.
  // 0x??????0B ??????04
  uint32x4_t paired32 = vsraq_n_u32(vreinterpretq_u32_u16(paired16),
                                    vreinterpretq_u32_u16(paired16), 14);
  // 0x??????????????4B
  uint64x2_t paired64 = vsraq_n_u64(vreinterpretq_u64_u32(paired32),
                                    vreinterpretq_u64_u32(paired32), 28);
  // Extract the low 8 bits from each lane and join.
  // 0x4B
  return vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 0) |
         ((int) vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 8) << 8);
}

static SIMD_INLINE uint64_t msb2int(const Vec<SignedByte, 16> &a)
{
  // the same as msb2int(Vec<Byte,16>)
  return msb2int(reinterpret(a, OutputType<Byte>()));
}

static SIMD_INLINE uint64_t msb2int(const Vec<Word, 16> &a)
{
  // analogous to msb2int(Vec<Byte,16>)
  // idea from: https://stackoverflow.com/a/58381188/8461272

  // Shift out everything but the sign bits
  uint16x8_t high_bits = vshrq_n_u16(a, 15);

  // Merge the even lanes together with vsra. The '??' bytes are garbage.
  uint32x4_t paired32 = vsraq_n_u32(vreinterpretq_u32_u16(high_bits),
                                    vreinterpretq_u32_u16(high_bits), 15);
  // Repeat with wider lanes.
  uint64x2_t paired64 = vsraq_n_u64(vreinterpretq_u64_u32(paired32),
                                    vreinterpretq_u64_u32(paired32), 30);
  // Extract the low 4 bits from each lane and join.
  return (vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 0) & 0xf) |
         (vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 8) << 4);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Short, 16> &a)
{
  // the same as msb2int(Vec<Word,16>)
  return msb2int(reinterpret(a, OutputType<Word>()));
}

static SIMD_INLINE uint64_t msb2int(const Vec<Int, 16> &a)
{
  // analogous to msb2int(Vec<Byte,16>)
  // idea from: https://stackoverflow.com/a/58381188/8461272

  // Shift out everything but the sign bits
  uint32x4_t high_bits = vshrq_n_u32(vreinterpretq_u32_s32(a), 31);

  // Merge the even lanes together with vsra. The '??' bytes are garbage.
  uint64x2_t paired64 = vsraq_n_u64(vreinterpretq_u64_u32(high_bits),
                                    vreinterpretq_u64_u32(high_bits), 31);
  // Extract the low 2 bits from each lane and join.
  return (vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 0) & 0x3) |
         ((vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 8) & 0x3) << 2);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Float, 16> &a)
{
  // the same as msb2int(Vec<Int,16>)
  return msb2int(reinterpret(a, OutputType<Int>()));
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE uint64_t msb2int(const Vec<Double, 16> &a)
{
  // shift out everything but the sign bits
  uint64x2_t high_bits = vshrq_n_u64(vreinterpretq_u64_f64(a), 63);
  // extract the low bit from each lane and join
  return vgetq_lane_u8(vreinterpretq_u8_u64(high_bits), 0) |
         (vgetq_lane_u8(vreinterpretq_u8_u64(high_bits), 8) << 1);
}
#endif

// ---------------------------------------------------------------------------
// int2msb
// ---------------------------------------------------------------------------

// 06. Oct 22 (Jonas Keller): added int2msb functions

static SIMD_INLINE Vec<Byte, 16> int2msb(const uint64_t a, OutputType<Byte>,
                                         Integer<16>)
{
  uint8x8_t aVecLo = vdup_n_u8(a & 0xff);
  uint8x8_t aVecHi = vdup_n_u8((a >> 8) & 0xff);
  uint8x16_t aVec  = vcombine_u8(aVecLo, aVecHi);
  // shift the bits to the msb
  int8x16_t shiftAmounts = {7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0};
  uint8x16_t shifted     = vshlq_u8(aVec, shiftAmounts);
  return vandq_u8(shifted, vdupq_n_u8(0x80));
}

static SIMD_INLINE Vec<SignedByte, 16> int2msb(const uint64_t a,
                                               OutputType<SignedByte>,
                                               Integer<16>)
{
  return reinterpret(int2msb(a, OutputType<Byte>(), Integer<16>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Word, 16> int2msb(const uint64_t a, OutputType<Word>,
                                         Integer<16>)
{
  uint16x8_t aVec = vdupq_n_u16(a & 0xff);
  // shift the bits to the msb
  int16x8_t shiftAmounts = {15, 14, 13, 12, 11, 10, 9, 8};
  uint16x8_t shifted     = vshlq_u16(aVec, shiftAmounts);
  return vandq_u16(shifted, vdupq_n_u16(0x8000));
}

static SIMD_INLINE Vec<Short, 16> int2msb(const uint64_t a, OutputType<Short>,
                                          Integer<16>)
{
  return reinterpret(int2msb(a, OutputType<Word>(), Integer<16>()),
                     OutputType<Short>());
}

static SIMD_INLINE Vec<Int, 16> int2msb(const uint64_t a, OutputType<Int>,
                                        Integer<16>)
{
  int32x4_t aVec = vdupq_n_s32(a & 0xf);
  // shift the bits to the msb
  int32x4_t shiftAmounts = {31, 30, 29, 28};
  int32x4_t shifted      = vshlq_s32(aVec, shiftAmounts);
  return vandq_s32(shifted, vdupq_n_s32(0x80000000));
}

static SIMD_INLINE Vec<Float, 16> int2msb(const uint64_t a, OutputType<Float>,
                                          Integer<16>)
{
  return reinterpret(int2msb(a, OutputType<Int>(), Integer<16>()),
                     OutputType<Float>());
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> int2msb(const uint64_t a, OutputType<Double>,
                                           Integer<16>)
{
  int64x2_t aVec = vdupq_n_s64(a & 0x3);
  // shift the bits to the msb
  int64x2_t shiftAmounts = {63, 62};
  int64x2_t shifted      = vshlq_s64(aVec, shiftAmounts);
  int64x2_t result       = vandq_s64(shifted, vdupq_n_s64(0x8000000000000000));
  return vreinterpretq_f64_s64(result);
}
#endif

// ---------------------------------------------------------------------------
// int2bits
// ---------------------------------------------------------------------------

// 09. Oct 22 (Jonas Keller): added int2bits functions

static SIMD_INLINE Vec<Byte, 16> int2bits(const uint64_t a, OutputType<Byte>,
                                          Integer<16>)
{
  uint8x8_t aVecLo = vdup_n_u8(a & 0xff);
  uint8x8_t aVecHi = vdup_n_u8((a >> 8) & 0xff);
  uint8x16_t aVec  = vcombine_u8(aVecLo, aVecHi);
  uint8x16_t sel   = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
  return vtstq_u8(aVec, sel);
}

static SIMD_INLINE Vec<SignedByte, 16> int2bits(const uint64_t a,
                                                OutputType<SignedByte>,
                                                Integer<16>)
{
  return reinterpret(int2bits(a, OutputType<Byte>(), Integer<16>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Word, 16> int2bits(const uint64_t a, OutputType<Word>,
                                          Integer<16>)
{
  uint16x8_t aVec = vdupq_n_u16(a & 0xff);
  uint16x8_t sel  = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
  return vtstq_u16(aVec, sel);
}

static SIMD_INLINE Vec<Short, 16> int2bits(const uint64_t a, OutputType<Short>,
                                           Integer<16>)
{
  return reinterpret(int2bits(a, OutputType<Word>(), Integer<16>()),
                     OutputType<Short>());
}

static SIMD_INLINE Vec<Int, 16> int2bits(const uint64_t a, OutputType<Int>,
                                         Integer<16>)
{
  int32x4_t aVec = vdupq_n_s32(a & 0xf);
  int32x4_t sel  = {0x01, 0x02, 0x04, 0x08};
  return vreinterpretq_s32_u32(vtstq_s32(aVec, sel));
}

static SIMD_INLINE Vec<Float, 16> int2bits(const uint64_t a, OutputType<Float>,
                                           Integer<16>)
{
  return reinterpret(int2bits(a, OutputType<Int>(), Integer<16>()),
                     OutputType<Float>());
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> int2bits(const uint64_t a,
                                            OutputType<Double>, Integer<16>)
{
  int64x2_t aVec = vdupq_n_s64(a & 0xf);
  int64x2_t sel  = {0x01, 0x02};
  return vreinterpretq_f64_u64(vtstq_s64(aVec, sel));
}
#endif

// ---------------------------------------------------------------------------
// iota
// ---------------------------------------------------------------------------

// 30. Jan 23 (Jonas Keller): added iota

static SIMD_INLINE Vec<Byte, 16> iota(OutputType<Byte>, Integer<16>)
{
  uint8x16_t res = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return res;
}

static SIMD_INLINE Vec<SignedByte, 16> iota(OutputType<SignedByte>, Integer<16>)
{
  int8x16_t res = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  return res;
}

static SIMD_INLINE Vec<Word, 16> iota(OutputType<Word>, Integer<16>)
{
  uint16x8_t res = {0, 1, 2, 3, 4, 5, 6, 7};
  return res;
}

static SIMD_INLINE Vec<Short, 16> iota(OutputType<Short>, Integer<16>)
{
  int16x8_t res = {0, 1, 2, 3, 4, 5, 6, 7};
  return res;
}

static SIMD_INLINE Vec<Int, 16> iota(OutputType<Int>, Integer<16>)
{
  int32x4_t res = {0, 1, 2, 3};
  return res;
}

static SIMD_INLINE Vec<Float, 16> iota(OutputType<Float>, Integer<16>)
{
  float32x4_t res = {0.0f, 1.0f, 2.0f, 3.0f};
  return res;
}

#ifdef SIMD_DOUBLE
static SIMD_INLINE Vec<Double, 16> iota(OutputType<Double>, Integer<16>)
{
  float64x2_t res = {0.0, 1.0};
  return res;
}
#endif
} // namespace base
} // namespace internal
} // namespace simd

#endif

#endif // SIMD_VEC_BASE_IMPL_NEON_16_H_
