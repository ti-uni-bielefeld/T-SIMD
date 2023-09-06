// ===========================================================================
//
// SIMDVecBaseImplIntel16.H --
// encapsulation for SSE Intel vector extensions
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
// 13. May 23 (Jonas Keller): added Double support

#ifndef SIMD_VEC_BASE_IMPL_INTEL_16_H_
#define SIMD_VEC_BASE_IMPL_INTEL_16_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDSSSE3Compat.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __SSE2__

namespace simd {

// ===========================================================================
// NOTES:
//
// - setting zero inside the function is not inefficient, see:
//   http://stackoverflow.com/questions/26807285/...
//   ...are-static-static-local-sse-avx-variables-blocking-a-xmm-ymm-register
//
// - for some data types (Int, Float) there are no saturated versions
//   of add/sub instructions; in this case we use the unsaturated version;
//   the user is responsible to avoid overflows
//
// - we could improve performance by using 128-bit instructions from
//   AVX512-VL (e.g. permute instructions); at the moment the idea is that
//   typically the widest vector width is used, so if AVX512 is available,
//   SSE would only rarely be used
//
// ===========================================================================

// ===========================================================================
// Vec integer instantiation for SSE
// ===========================================================================

// partial specialization for SIMD_WIDTH = 16
template <typename T>
class Vec<T, 16>
{
  __m128i xmm = _mm_setzero_si128();

public:
  using Type                    = T;
  static constexpr int elements = 16 / sizeof(T);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 16;

  Vec() = default;
  Vec(const __m128i &x) { xmm = x; }
  Vec &operator=(const __m128i &x)
  {
    xmm = x;
    return *this;
  }
  operator __m128i() const { return xmm; }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
};

// ===========================================================================
// Vec float specialization for SSE
// ===========================================================================

template <>
class Vec<Float, 16>
{
  __m128 xmm = _mm_setzero_ps();

public:
  using Type                    = Float;
  static constexpr int elements = 16 / sizeof(Float);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 16;

  Vec() = default;
  Vec(const __m128 &x) { xmm = x; }
  Vec &operator=(const __m128 &x)
  {
    xmm = x;
    return *this;
  }
  operator __m128() const { return xmm; }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
};

// ===========================================================================
// Vec double specialization for SSE
// ===========================================================================

template <>
class Vec<Double, 16>
{
  __m128d xmm;

public:
  using Type                    = Double;
  static constexpr int elements = 16 / sizeof(Double);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 16;
  Vec()                         = default;
  Vec(const __m128d &x) { xmm = x; }
  Vec &operator=(const __m128d &x)
  {
    xmm = x;
    return *this;
  }
  operator __m128d() const { return xmm; }
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
};

namespace internal {
namespace base {

// ===========================================================================
// auxiliary functions
// ===========================================================================

// These functions either wrap SSE intrinsics (e.g. to handle
// immediate arguments as template parameter), or switch between
// implementations with different SSE* extensions, or provide
// altered or additional functionality.
// Only for use in wrapper functions!

// 01. Apr 23 (Jonas Keller): removed some not really necessary internal
// wrapper functions and inlined them directly into where they were used

// ---------------------------------------------------------------------------
// byte-wise shifts
// ---------------------------------------------------------------------------

// positive and in range: apply shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_si128(__m128i a, IsPosInRange<true, true>)
{
  return _mm_srli_si128(a, IMM);
}

// positive and out of range: return zero vector
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_si128(__m128i, IsPosInRange<true, false>)
{
  return _mm_setzero_si128();
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_si128(__m128i a)
{
  return x_mm_srli_si128<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: apply shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_si128(__m128i a, IsPosInRange<true, true>)
{
  return _mm_slli_si128(a, IMM);
}

// positive and out of range: return zero vector
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_si128(__m128i, IsPosInRange<true, false>)
{
  return _mm_setzero_si128();
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_si128(__m128i a)
{
  return x_mm_slli_si128<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// ---------------------------------------------------------------------------
// element-wise shifts
// ---------------------------------------------------------------------------

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_epi16(__m128i a, IsPosInRange<true, true>)
{
  return _mm_slli_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_epi16(__m128i, IsPosInRange<true, false>)
{
  return _mm_setzero_si128();
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_epi16(__m128i a)
{
  return x_mm_slli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_epi16(__m128i a, IsPosInRange<true, true>)
{
  return _mm_srli_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_epi16(__m128i, IsPosInRange<true, false>)
{
  return _mm_setzero_si128();
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_epi16(__m128i a)
{
  return x_mm_srli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_epi32(__m128i a, IsPosInRange<true, true>)
{
  return _mm_slli_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_epi32(__m128i, IsPosInRange<true, false>)
{
  return _mm_setzero_si128();
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_slli_epi32(__m128i a)
{
  return x_mm_slli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_epi32(__m128i a, IsPosInRange<true, true>)
{
  return _mm_srli_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_epi32(__m128i, IsPosInRange<true, false>)
{
  return _mm_setzero_si128();
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_srli_epi32(__m128i a)
{
  return x_mm_srli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// 16. Oct 22 (Jonas Keller): added x_mm_srai_epi8

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi8(__m128i a, IsPosInRange<true, true>)
{
  __m128i odd         = _mm_srai_epi16(a, IMM);
  __m128i even        = _mm_srai_epi16(_mm_slli_epi16(a, 8), IMM + 8);
  __m128i odd_masked  = _mm_and_si128(odd, _mm_set1_epi16((int16_t) 0xFF00));
  __m128i even_masked = _mm_and_si128(even, _mm_set1_epi16(0x00FF));
  return _mm_or_si128(odd_masked, even_masked);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi8(__m128i a, IsPosInRange<true, false>)
{
  // result should be all ones if a is negative, all zeros otherwise
  return _mm_cmplt_epi8(a, _mm_setzero_si128());
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi8(__m128i a)
{
  return x_mm_srai_epi8<IMM>(a, IsPosInGivenRange<7, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi16(__m128i a, IsPosInRange<true, true>)
{
  return _mm_srai_epi16(a, IMM);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi16(__m128i a, IsPosInRange<true, false>)
{
  return _mm_srai_epi16(a, 15);
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi16(__m128i a)
{
  return x_mm_srai_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi32(__m128i a, IsPosInRange<true, true>)
{
  return _mm_srai_epi32(a, IMM);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi32(__m128i a, IsPosInRange<true, false>)
{
  return _mm_srai_epi32(a, 31);
}

// hub
template <int IMM>
static SIMD_INLINE __m128i x_mm_srai_epi32(__m128i a)
{
  return x_mm_srai_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// ---------------------------------------------------------------------------
// element-wise shift where count does not need to be constant
// ---------------------------------------------------------------------------

// 19. Dec 22 (Jonas Keller): added x_mm_sra_epi8

static SIMD_INLINE __m128i x_mm_sra_epi8(__m128i a, uint8_t count)
{
  // there is no _mm_sra_epi8 intrinsic
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm_cmplt_epi8(a, _mm_setzero_si128());
  }
  __m128i odd = _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m128i even =
    _mm_sra_epi16(_mm_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  __m128i odd_masked  = _mm_and_si128(odd, _mm_set1_epi16((int16_t) 0xFF00));
  __m128i even_masked = _mm_and_si128(even, _mm_set1_epi16(0x00FF));
  return _mm_or_si128(odd_masked, even_masked);
}

// ---------------------------------------------------------------------------
// extract
// ---------------------------------------------------------------------------

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi16(__m128i a, IsPosInRange<true, true>)
{
  return _mm_extract_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi16(__m128i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi16(__m128i a)
{
  // 8 * epi16
  return x_mm_extract_epi16<IMM>(a, IsPosInGivenRange<8, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi8(__m128i a, IsPosInRange<true, true>)
{
#ifdef __SSE4_1__
  return _mm_extract_epi8(a, IMM);
#else
  return (((IMM & 0x1) == 0) ?
            x_mm_extract_epi16<(IMM >> 1)>(a) & 0xff :
            x_mm_extract_epi16<(IMM >> 1)>(_mm_srli_epi16(a, 8)));
#endif
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi8(__m128i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi8(__m128i a)
{
  // 16 * epi8
  return x_mm_extract_epi8<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi32(__m128i a, IsPosInRange<true, true>)
{
  // TODO: extract: is conversion from return type int to Int always safe?
#ifdef __SSE4_1__
  return _mm_extract_epi32(a, IMM);
#else
  // n << 2 = 4 * n
  return _mm_cvtsi128_si32(x_mm_srli_si128<(IMM << 2)>(a));
#endif
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi32(__m128i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm_extract_epi32(__m128i a)
{
  // 4 * epi32
  return x_mm_extract_epi32<IMM>(a, IsPosInGivenRange<4, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm_extract_ps(__m128 a, IsPosInRange<true, true>)
{
#ifdef __SSE4_1__
  return _mm_extract_ps(a, IMM);
#else
  // IMM << 2 = 4 * IMM
  return _mm_cvtsi128_si32(x_mm_srli_si128<(IMM << 2)>(_mm_castps_si128(a)));
#endif
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm_extract_ps(__m128, IsPosInRange<true, false>)
{
  return 0;
}

// hub: yes, really returns an int!
template <int IMM>
static SIMD_INLINE int x_mm_extract_ps(__m128 a)
{
  // 4 * ps
  return x_mm_extract_ps<IMM>(a, IsPosInGivenRange<4, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE long long x_mm_extract_pd(__m128d a,
                                             IsPosInRange<true, true>)
{
#ifdef __SSE4_1__
  return _mm_extract_epi64(_mm_castpd_si128(a), IMM);
#else
  // IMM << 3 = 8 * IMM
  return _mm_cvtsi128_si64(x_mm_srli_si128<(IMM << 3)>(_mm_castpd_si128(a)));
#endif
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE long long x_mm_extract_pd(__m128d, IsPosInRange<true, false>)
{
  return 0;
}

// hub: yes, really returns a long long!
template <int IMM>
static SIMD_INLINE long long x_mm_extract_pd(__m128d a)
{
  // 2 * pd
  return x_mm_extract_pd<IMM>(a, IsPosInGivenRange<2, IMM>());
}

// ---------------------------------------------------------------------------
// abs
// ---------------------------------------------------------------------------

static SIMD_INLINE __m128 x_mm_abs_ps(__m128 a)
{
  // there's no _mm_abs_ps, so we have to emulate it:
  // clear sign bit:
  return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)));
}

static SIMD_INLINE __m128d x_mm_abs_pd(__m128d a)
{
  // there's no _mm_abs_pd, so we have to emulate it:
  // clear sign bit:
  return _mm_and_pd(a, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF)));
}

// ###########################################################################
// ###########################################################################
// ###########################################################################

// ===========================================================================
// Vec function template specialization or overloading for SSE
// ===========================================================================

// ---------------------------------------------------------------------------
// reinterpretation casts
// ---------------------------------------------------------------------------

// 08. Apr 23 (Jonas Keller): used enable_if for cleaner implementation

// between all integer types
template <typename Tdst, typename Tsrc,
          SIMD_ENABLE_IF((!std::is_same<Tdst, Tsrc>::value &&
                          std::is_integral<Tdst>::value &&
                          std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Tdst, 16> reinterpret(const Vec<Tsrc, 16> &vec,
                                             OutputType<Tdst>)
{
  // 26. Nov 22 (Jonas Keller): reinterpret_cast is technically undefined
  // behavior, so just rewrapping the vector register in a new Vec instead
  // return reinterpret_cast<const Vec<Tdst,16>&>(vec);
  return Vec<Tdst, 16>(__m128i(vec));
}

// from float to any integer type
template <typename Tdst, SIMD_ENABLE_IF((std::is_integral<Tdst>::value))>
static SIMD_INLINE Vec<Tdst, 16> reinterpret(const Vec<Float, 16> &vec,
                                             OutputType<Tdst>)
{
  return _mm_castps_si128(vec);
}

// from any integer type to float
template <typename Tsrc, SIMD_ENABLE_IF((std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Float, 16> reinterpret(const Vec<Tsrc, 16> &vec,
                                              OutputType<Float>)
{
  return _mm_castsi128_ps(vec);
}

// from double to any integer type
template <typename Tdst, SIMD_ENABLE_IF((std::is_integral<Tdst>::value))>
static SIMD_INLINE Vec<Tdst, 16> reinterpret(const Vec<Double, 16> &vec,
                                             OutputType<Tdst>)
{
  return _mm_castpd_si128(vec);
}

// from any integer type to double
template <typename Tsrc, SIMD_ENABLE_IF((std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Double, 16> reinterpret(const Vec<Tsrc, 16> &vec,
                                               OutputType<Double>)
{
  return _mm_castsi128_pd(vec);
}

// from float to double
static SIMD_INLINE Vec<Double, 16> reinterpret(const Vec<Float, 16> &vec,
                                               OutputType<Double>)
{
  return _mm_castps_pd(vec);
}

// from double to float
static SIMD_INLINE Vec<Float, 16> reinterpret(const Vec<Double, 16> &vec,
                                              OutputType<Float>)
{
  return _mm_castpd_ps(vec);
}

// between identical types
template <typename T>
static SIMD_INLINE Vec<T, 16> reinterpret(const Vec<T, 16> &vec, OutputType<T>)
{
  return vec;
}

// ---------------------------------------------------------------------------
// convert (without changes in the number of of elements)
// ---------------------------------------------------------------------------

// conversion with saturation; we wanted to have a fast solution that
// doesn't trigger the overflow which results in a negative two's
// complement result ("invalid int32": 0x80000000); therefore we clamp
// the positive values at the maximal positive float which is
// convertible to int32 without overflow (0x7fffffbf = 2147483520);
// negative values cannot overflow (they are clamped to invalid int
// which is the most negative int32)
static SIMD_INLINE Vec<Int, 16> cvts(const Vec<Float, 16> &a, OutputType<Int>)
{
  // TODO: analyze much more complex solution for cvts at
  // TODO: http://stackoverflow.com/questions/9157373/
  // TODO: most-efficient-way-to-convert-vector-of-float-to-vector-of-uint32
  // NOTE: float->int: rounding is affected by MXCSR rounding control bits!
  __m128 clip = _mm_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
  return _mm_cvtps_epi32(_mm_min_ps(clip, a));
}

// saturation is not necessary in this case
static SIMD_INLINE Vec<Float, 16> cvts(const Vec<Int, 16> &a, OutputType<Float>)
{
  return _mm_cvtepi32_ps(a);
}

// ---------------------------------------------------------------------------
// setzero
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> setzero(OutputType<Byte>, Integer<16>)
{
  return _mm_setzero_si128();
}

static SIMD_INLINE Vec<SignedByte, 16> setzero(OutputType<SignedByte>,
                                               Integer<16>)
{
  return _mm_setzero_si128();
}

static SIMD_INLINE Vec<Word, 16> setzero(OutputType<Word>, Integer<16>)
{
  return _mm_setzero_si128();
}

static SIMD_INLINE Vec<Short, 16> setzero(OutputType<Short>, Integer<16>)
{
  return _mm_setzero_si128();
}

static SIMD_INLINE Vec<Int, 16> setzero(OutputType<Int>, Integer<16>)
{
  return _mm_setzero_si128();
}

static SIMD_INLINE Vec<Float, 16> setzero(OutputType<Float>, Integer<16>)
{
  return _mm_setzero_ps();
}

static SIMD_INLINE Vec<Double, 16> setzero(OutputType<Double>, Integer<16>)
{
  return _mm_setzero_pd();
}

// ---------------------------------------------------------------------------
// set1
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> set1(Byte a, Integer<16>)
{
  return _mm_set1_epi8(a);
}

static SIMD_INLINE Vec<SignedByte, 16> set1(SignedByte a, Integer<16>)
{
  return _mm_set1_epi8(a);
}

static SIMD_INLINE Vec<Word, 16> set1(Word a, Integer<16>)
{
  return _mm_set1_epi16(a);
}

static SIMD_INLINE Vec<Short, 16> set1(Short a, Integer<16>)
{
  return _mm_set1_epi16(a);
}

static SIMD_INLINE Vec<Int, 16> set1(Int a, Integer<16>)
{
  return _mm_set1_epi32(a);
}

static SIMD_INLINE Vec<Float, 16> set1(Float a, Integer<16>)
{
  return _mm_set1_ps(a);
}

static SIMD_INLINE Vec<Double, 16> set1(Double a, Integer<16>)
{
  return _mm_set1_pd(a);
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> load(const T *const p, Integer<16>)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  return _mm_load_si128((__m128i *) p);
}

static SIMD_INLINE Vec<Float, 16> load(const Float *const p, Integer<16>)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  return _mm_load_ps(p);
}

static SIMD_INLINE Vec<Double, 16> load(const Double *const p, Integer<16>)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  return _mm_load_pd(p);
}

// ---------------------------------------------------------------------------
// loadu
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> loadu(const T *const p, Integer<16>)
{
  return _mm_loadu_si128((__m128i *) p);
}

static SIMD_INLINE Vec<Float, 16> loadu(const Float *const p, Integer<16>)
{
  return _mm_loadu_ps(p);
}

static SIMD_INLINE Vec<Double, 16> loadu(const Double *const p, Integer<16>)
{
  return _mm_loadu_pd(p);
}

// ---------------------------------------------------------------------------
// store
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void store(T *const p, const Vec<T, 16> &a)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  _mm_store_si128((__m128i *) p, a);
}

// float version
static SIMD_INLINE void store(Float *const p, const Vec<Float, 16> &a)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  _mm_store_ps(p, a);
}

// double version
static SIMD_INLINE void store(Double *const p, const Vec<Double, 16> &a)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  _mm_store_pd(p, a);
}

// ---------------------------------------------------------------------------
// storeu
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void storeu(T *const p, const Vec<T, 16> &a)
{
  _mm_storeu_si128((__m128i *) p, a);
}

// float version
static SIMD_INLINE void storeu(Float *const p, const Vec<Float, 16> &a)
{
  _mm_storeu_ps(p, a);
}

// double version
static SIMD_INLINE void storeu(Double *const p, const Vec<Double, 16> &a)
{
  _mm_storeu_pd(p, a);
}

// ---------------------------------------------------------------------------
// stream_store
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void stream_store(T *const p, const Vec<T, 16> &a)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  _mm_stream_si128((__m128i *) p, a);
}

// float version
static SIMD_INLINE void stream_store(Float *const p, const Vec<Float, 16> &a)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  _mm_stream_ps(p, a);
}

// double version
static SIMD_INLINE void stream_store(Double *const p, const Vec<Double, 16> &a)
{
  // SSE load and store instructions need alignment to 16 byte
  // (lower 4 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 16);
  _mm_stream_pd(p, a);
}

// ---------------------------------------------------------------------------
// fences (defined only here and not in SIMDVec32.H)
// ---------------------------------------------------------------------------

static SIMD_INLINE void lfence()
{
  _mm_lfence();
}

static SIMD_INLINE void sfence()
{
  _mm_sfence();
}

static SIMD_INLINE void mfence()
{
  _mm_mfence();
}

// ---------------------------------------------------------------------------
// extract: with template parameter for immediate argument
// ---------------------------------------------------------------------------

template <int IMM>
static SIMD_INLINE Byte extract(const Vec<Byte, 16> &a)
{
  return x_mm_extract_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE SignedByte extract(const Vec<SignedByte, 16> &a)
{
  return x_mm_extract_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Word extract(const Vec<Word, 16> &a)
{
  return x_mm_extract_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Short extract(const Vec<Short, 16> &a)
{
  return x_mm_extract_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Int extract(const Vec<Int, 16> &a)
{
  return x_mm_extract_epi32<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Float extract(const Vec<Float, 16> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Float, x_mm_extract_ps<IMM>(a), 0)
}

template <int IMM>
static SIMD_INLINE Double extract(const Vec<Double, 16> &a)
{
  Double res;
  long long i = x_mm_extract_pd<IMM>(a);
  memcpy(&res, &i, sizeof(Double));
  return res;
}

// ---------------------------------------------------------------------------
// ifelse
// ---------------------------------------------------------------------------

// elements of cond must be all 1's or all 0's (blendv just tests top
// bit in each byte, but work-around needs this)

template <typename T>
static SIMD_INLINE Vec<T, 16> ifelse(const Vec<T, 16> &cond,
                                     const Vec<T, 16> &trueVal,
                                     const Vec<T, 16> &falseVal)
{
#ifdef __SSE4_1__
  return _mm_blendv_epi8(falseVal, trueVal, cond);
#else
  return _mm_or_si128(_mm_and_si128(cond, trueVal),
                      _mm_andnot_si128(cond, falseVal));
#endif
}

static SIMD_INLINE Vec<Float, 16> ifelse(const Vec<Float, 16> &cond,
                                         const Vec<Float, 16> &trueVal,
                                         const Vec<Float, 16> &falseVal)
{
#ifdef __SSE4_1__
  return _mm_blendv_ps(falseVal, trueVal, cond);
#else
  return _mm_or_ps(_mm_and_ps(cond, trueVal), _mm_andnot_ps(cond, falseVal));
#endif
}

static SIMD_INLINE Vec<Double, 16> ifelse(const Vec<Double, 16> &cond,
                                          const Vec<Double, 16> &trueVal,
                                          const Vec<Double, 16> &falseVal)
{
#ifdef __SSE4_1__
  return _mm_blendv_pd(falseVal, trueVal, cond);
#else
  return _mm_or_pd(_mm_and_pd(cond, trueVal), _mm_andnot_pd(cond, falseVal));
#endif
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> add(const Vec<Byte, 16> &a,
                                     const Vec<Byte, 16> &b)
{
  return _mm_add_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> add(const Vec<SignedByte, 16> &a,
                                           const Vec<SignedByte, 16> &b)
{
  return _mm_add_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> add(const Vec<Word, 16> &a,
                                     const Vec<Word, 16> &b)
{
  return _mm_add_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 16> add(const Vec<Short, 16> &a,
                                      const Vec<Short, 16> &b)
{
  return _mm_add_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> add(const Vec<Int, 16> &a,
                                    const Vec<Int, 16> &b)
{
  return _mm_add_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> add(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return _mm_add_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> add(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return _mm_add_pd(a, b);
}

// ---------------------------------------------------------------------------
// adds
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> adds(const Vec<Byte, 16> &a,
                                      const Vec<Byte, 16> &b)
{
  return _mm_adds_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> adds(const Vec<SignedByte, 16> &a,
                                            const Vec<SignedByte, 16> &b)
{
  return _mm_adds_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> adds(const Vec<Word, 16> &a,
                                      const Vec<Word, 16> &b)
{
  return _mm_adds_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 16> adds(const Vec<Short, 16> &a,
                                       const Vec<Short, 16> &b)
{
  return _mm_adds_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> adds(const Vec<Int, 16> &a,
                                     const Vec<Int, 16> &b)
{
  // 09. Mar 23 (Jonas Keller): added workaround so that this function is
  // saturated

  // _mm_adds_epi32 does not exist, workaround:
  // Hacker's Delight, 2-13 Overflow Detection: "Signed integer overflow of
  // addition occurs if and only if the operands have the same sign and the
  // sum has a sign opposite to that of the operands."
  __m128i sum             = _mm_add_epi32(a, b);
  __m128i opsHaveDiffSign = _mm_xor_si128(a, b);
  __m128i sumHasDiffSign  = _mm_xor_si128(a, sum);
  // indicates when an overflow has occurred
  __m128i overflow =
    _mm_srai_epi32(_mm_andnot_si128(opsHaveDiffSign, sumHasDiffSign), 31);
  // saturated sum for if overflow occurred (0x7FFFFFFF=max positive int, when
  // sign of a (and thus b as well) is 0, 0x80000000=min negative int, when sign
  // of a (and thus b as well) is 1)
  __m128i saturatedSum =
    _mm_xor_si128(_mm_srai_epi32(a, 31), _mm_set1_epi32(0x7FFFFFFF));
  // return saturated sum if overflow occurred, otherwise return sum
  return ifelse(Vec<Int, 16>(overflow), Vec<Int, 16>(saturatedSum),
                Vec<Int, 16>(sum));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 16> adds(const Vec<Float, 16> &a,
                                       const Vec<Float, 16> &b)
{
  return _mm_add_ps(a, b);
}

// Double not saturated
static SIMD_INLINE Vec<Double, 16> adds(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b)
{
  return _mm_add_pd(a, b);
}

// ---------------------------------------------------------------------------
// sub
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> sub(const Vec<Byte, 16> &a,
                                     const Vec<Byte, 16> &b)
{
  return _mm_sub_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> sub(const Vec<SignedByte, 16> &a,
                                           const Vec<SignedByte, 16> &b)
{
  return _mm_sub_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> sub(const Vec<Word, 16> &a,
                                     const Vec<Word, 16> &b)
{
  return _mm_sub_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 16> sub(const Vec<Short, 16> &a,
                                      const Vec<Short, 16> &b)
{
  return _mm_sub_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> sub(const Vec<Int, 16> &a,
                                    const Vec<Int, 16> &b)
{
  return _mm_sub_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> sub(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return _mm_sub_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> sub(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return _mm_sub_pd(a, b);
}

// ---------------------------------------------------------------------------
// subs
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> subs(const Vec<Byte, 16> &a,
                                      const Vec<Byte, 16> &b)
{
  return _mm_subs_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> subs(const Vec<SignedByte, 16> &a,
                                            const Vec<SignedByte, 16> &b)
{
  return _mm_subs_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> subs(const Vec<Word, 16> &a,
                                      const Vec<Word, 16> &b)
{
  return _mm_subs_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 16> subs(const Vec<Short, 16> &a,
                                       const Vec<Short, 16> &b)
{
  return _mm_subs_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> subs(const Vec<Int, 16> &a,
                                     const Vec<Int, 16> &b)
{
  // 09. Mar 23 (Jonas Keller): added workaround so that this function is
  // saturated

  // _mm_subs_epi32 does not exist, workaround:
  // Hacker's Delight, 2-13 Overflow Detection: "[...] overflow in the final
  // value of x−y [...] occurs if and only if x and y have opposite signs and
  // the sign of x−y [...] is opposite to that of x [...]"
  __m128i diff            = _mm_sub_epi32(a, b);
  __m128i opsHaveDiffSign = _mm_xor_si128(a, b);
  __m128i diffHasDiffSign = _mm_xor_si128(a, diff);
  // indicates when an overflow has occurred
  __m128i overflow =
    _mm_srai_epi32(_mm_and_si128(opsHaveDiffSign, diffHasDiffSign), 31);
  // saturated diff for if overflow occurred (0x7FFFFFFF=max positive int, when
  // sign of a (and thus b as well) is 0, 0x80000000=min negative int, when sign
  // of a (and thus b as well) is 1)
  __m128i saturatedDiff =
    _mm_xor_si128(_mm_srai_epi32(a, 31), _mm_set1_epi32(0x7FFFFFFF));
  // return saturated diff if overflow occurred, otherwise return diff
  return ifelse(Vec<Int, 16>(overflow), Vec<Int, 16>(saturatedDiff),
                Vec<Int, 16>(diff));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 16> subs(const Vec<Float, 16> &a,
                                       const Vec<Float, 16> &b)
{
  return _mm_sub_ps(a, b);
}

// Double not saturated
static SIMD_INLINE Vec<Double, 16> subs(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b)
{
  return _mm_sub_pd(a, b);
}

// ---------------------------------------------------------------------------
// neg (negate = two's complement or unary minus), only signed types
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<SignedByte, 16> neg(const Vec<SignedByte, 16> &a)
{
  return _mm_sub_epi8(_mm_setzero_si128(), a);
}

static SIMD_INLINE Vec<Short, 16> neg(const Vec<Short, 16> &a)
{
  return _mm_sub_epi16(_mm_setzero_si128(), a);
}

static SIMD_INLINE Vec<Int, 16> neg(const Vec<Int, 16> &a)
{
  return _mm_sub_epi32(_mm_setzero_si128(), a);
}

static SIMD_INLINE Vec<Float, 16> neg(const Vec<Float, 16> &a)
{
  return _mm_sub_ps(_mm_setzero_ps(), a);
}

static SIMD_INLINE Vec<Double, 16> neg(const Vec<Double, 16> &a)
{
  return _mm_xor_pd(a, _mm_set1_pd(-0.0));
}

// ---------------------------------------------------------------------------
// min
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> min(const Vec<Byte, 16> &a,
                                     const Vec<Byte, 16> &b)
{
  return _mm_min_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> min(const Vec<SignedByte, 16> &a,
                                           const Vec<SignedByte, 16> &b)
{
#ifdef __SSE4_1__
  return _mm_min_epi8(a, b);
#else
  // from Agner Fog's VCL vectori128.h
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // add 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // add 0x80
  __m128i m1      = _mm_min_epu8(a1, b1);      // unsigned min
  return _mm_xor_si128(m1, signbit);           // sub 0x80
#endif
}

static SIMD_INLINE Vec<Word, 16> min(const Vec<Word, 16> &a,
                                     const Vec<Word, 16> &b)
{
#ifdef __SSE4_1__
  return _mm_min_epu16(a, b);
#else
  // from Agner Fog's VCL vectori128.h
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // add 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // add 0x8000
  __m128i m1      = _mm_min_epi16(a1, b1);     // signed min
  return _mm_xor_si128(m1, signbit);           // sub 0x8000
#endif
}

static SIMD_INLINE Vec<Short, 16> min(const Vec<Short, 16> &a,
                                      const Vec<Short, 16> &b)
{
  return _mm_min_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> min(const Vec<Int, 16> &a,
                                    const Vec<Int, 16> &b)
{
#ifdef __SSE4_1__
  return _mm_min_epi32(a, b);
#else
  // from Agner Fog's VCL vectori128.h (modified)
  __m128i gt = _mm_cmpgt_epi32(a, b);
  return _mm_or_si128(_mm_and_si128(gt, b), _mm_andnot_si128(gt, a));
#endif
}

// there is an unsigned version of min for 32 bit but we currently
// don't have an element type for it

static SIMD_INLINE Vec<Float, 16> min(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return _mm_min_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> min(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return _mm_min_pd(a, b);
}

// ---------------------------------------------------------------------------
// max
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> max(const Vec<Byte, 16> &a,
                                     const Vec<Byte, 16> &b)
{
  return _mm_max_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> max(const Vec<SignedByte, 16> &a,
                                           const Vec<SignedByte, 16> &b)
{
#ifdef __SSE4_1__
  return _mm_max_epi8(a, b);
#else
  // from Agner Fog's VCL vectori128.h
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // add 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // add 0x80
  __m128i m1      = _mm_max_epu8(a1, b1);      // unsigned max
  return _mm_xor_si128(m1, signbit);           // sub 0x80
#endif
}

static SIMD_INLINE Vec<Word, 16> max(const Vec<Word, 16> &a,
                                     const Vec<Word, 16> &b)
{
#ifdef __SSE4_1__
  return _mm_max_epu16(a, b);
#else
  // from Agner Fog's VCL vectori128.h
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // add 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // add 0x8000
  __m128i m1      = _mm_max_epi16(a1, b1);     // signed max
  return _mm_xor_si128(m1, signbit);           // sub 0x8000
#endif
}

static SIMD_INLINE Vec<Short, 16> max(const Vec<Short, 16> &a,
                                      const Vec<Short, 16> &b)
{
  return _mm_max_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> max(const Vec<Int, 16> &a,
                                    const Vec<Int, 16> &b)
{
#ifdef __SSE4_1__
  return _mm_max_epi32(a, b);
#else
  // from Agner Fog's VCL vectori128.h
  __m128i gt = _mm_cmpgt_epi32(a, b);
  return _mm_or_si128(_mm_and_si128(gt, a), _mm_andnot_si128(gt, b));
#endif
}

// there is an unsigned version of max for 32 bit but we currently
// don't have an element type for it

static SIMD_INLINE Vec<Float, 16> max(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return _mm_max_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> max(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return _mm_max_pd(a, b);
}

// ---------------------------------------------------------------------------
// mul, div
// ---------------------------------------------------------------------------

// TODO: add mul/div versions for int types? or make special versions of mul
// TODO: and div where the result is scaled?

static SIMD_INLINE Vec<Float, 16> mul(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return _mm_mul_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> mul(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return _mm_mul_pd(a, b);
}

static SIMD_INLINE Vec<Float, 16> div(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  return _mm_div_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> div(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  return _mm_div_pd(a, b);
}

// ---------------------------------------------------------------------------
// ceil, floor, round, truncate
// ---------------------------------------------------------------------------

// 25. Mar 23 (Jonas Keller): added versions for integer types

// TODO: import complex workaround for non-SSE4.1 from Agner Fog's VCL?

// NOTE: behavior for workarounds differs for results of -0.0f and +0.0f

// work-arounds for round, truncate, floor, and ceil all check whether
// rounding is necessary (or whether float is an integer anyhow), this also
// prevents range excess when converting numbers to integer

// workarounds for floor and ceil:
// https://en.wikipedia.org/wiki/Floor_and_ceiling_functions
//
// floor, ceil:
//                 floor(x), x >= 0
// truncate(x) = {
//                 ceil(x), x < 0
//
// floor(x) = ceil(x)  - (x in Z ? 0 : 1)
// ceil(x)  = floor(x) + (x in Z ? 0 : 1)

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

static SIMD_INLINE Vec<Float, 16> ceil(const Vec<Float, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_ceil_ps(a);
#else
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  __m128 limit = _mm_set1_ps(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
  // bool mask: true if a is negative
  __m128 isNeg = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), 31));
  // truncated result (for |a| < limit)
  __m128 aTrunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
  // check if a is an integer
  __m128 isNotInt = _mm_cmpneq_ps(a, aTrunc);
  // constant 1.0
  __m128 one = _mm_set1_ps(1.0f);
  // mask which is 1.0f for non-negative non-integer values, 0.0f otherwise
  __m128 oneMask = _mm_and_ps(_mm_andnot_ps(isNeg, isNotInt), one);
  // if non-negative, trunc computes floor, to turn it into ceil we
  // add 1 if aTrunc is non-integer
  aTrunc = _mm_add_ps(aTrunc, oneMask);
  // select result (a or aTrunc)
  return ifelse(noRndReq, a, aTrunc);
#endif
}

static SIMD_INLINE Vec<Double, 16> ceil(const Vec<Double, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_ceil_pd(a);
#else
  // There is no _mm_cvtepi64_pd in SSE*, which makes the workaround
  // used for the float version not possible here.
  // Another workaround would probably be complicated and slow, so we just
  // ceil serially.
  // TODO: is there a better, vectorized workaround?
  Double inArr[2] SIMD_ATTR_ALIGNED(16);
  _mm_store_pd(inArr, a);
  Double outArr[2] SIMD_ATTR_ALIGNED(16);
  outArr[0] = std::ceil(inArr[0]);
  outArr[1] = std::ceil(inArr[1]);
  return _mm_load_pd(outArr);
#endif
}

static SIMD_INLINE Vec<Float, 16> floor(const Vec<Float, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_floor_ps(a);
#else
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  __m128 limit = _mm_set1_ps(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
  // bool mask: true if a is negative
  __m128 isNeg = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), 31));
  // truncated result (for |a| < limit)
  __m128 aTrunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
  // check if a is an integer
  __m128 isNotInt = _mm_cmpneq_ps(a, aTrunc);
  // constant 1.0
  __m128 one = _mm_set1_ps(1.0f);
  // mask which is 1.0f for negative non-integer values, 0.0f otherwise
  __m128 oneMask = _mm_and_ps(_mm_and_ps(isNeg, isNotInt), one);
  // if negative, trunc computes ceil, to turn it into floor we sub
  // 1 if aTrunc is non-integer
  aTrunc = _mm_sub_ps(aTrunc, oneMask);
  // select result (a or aTrunc)
  return ifelse(noRndReq, a, aTrunc);
#endif
}

static SIMD_INLINE Vec<Double, 16> floor(const Vec<Double, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_floor_pd(a);
#else
  // There is no _mm_cvtepi64_pd in SSE*, which makes the workaround
  // used for the float version not possible here.
  // Another workaround would probably be complicated and slow, so we just
  // floor serially.
  // TODO: is there a better, vectorized workaround?
  Double inArr[2] SIMD_ATTR_ALIGNED(16);
  _mm_store_pd(inArr, a);
  Double outArr[2] SIMD_ATTR_ALIGNED(16);
  outArr[0] = std::floor(inArr[0]);
  outArr[1] = std::floor(inArr[1]);
  return _mm_load_pd(outArr);
#endif
}

static SIMD_INLINE Vec<Float, 16> round(const Vec<Float, 16> &a)
{
#ifdef __SSE4_1__
  // old: use _MM_SET_ROUNDING_MODE to adjust rounding direction
  // return _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
  // new  4. Aug 16 (rm): round to nearest, and suppress exceptions
  return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
  // NOTE: only works if rounding mode is default (rnd. to nearest (even))
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  __m128 limit = _mm_set1_ps(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
  // rounded result (here rounded according to current rounding mode)
  // (for |a| < limit)
  __m128 aRnd = _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
  // select result
  return ifelse(noRndReq, a, aRnd);
#endif
}

static SIMD_INLINE Vec<Double, 16> round(const Vec<Double, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
  // There is no _mm_cvtepi64_pd in SSE*, which makes the workaround
  // used for the float version not possible here.
  // Another workaround would probably be complicated and slow, so we just
  // round serially.
  // TODO: is there a better, vectorized workaround?
  Double inArr[2] SIMD_ATTR_ALIGNED(16);
  _mm_store_pd(inArr, a);
  Double outArr[2] SIMD_ATTR_ALIGNED(16);
  // std::round has different behavior
  outArr[0] = std::rint(inArr[0]);
  outArr[1] = std::rint(inArr[1]);
  return _mm_load_pd(outArr);
#endif
}

static SIMD_INLINE Vec<Float, 16> truncate(const Vec<Float, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
  // if e>=23, floating point number represents an integer, 2^23 = 8388608
  __m128 limit = _mm_set1_ps(8388608.f);
  // bool mask: no rounding required if abs(a) >= limit
  __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
  // truncated result (for |a| < limit) (cvtTps!)
  __m128 aTrunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
  // select result
  return ifelse(noRndReq, a, aTrunc);
#endif
}

static SIMD_INLINE Vec<Double, 16> truncate(const Vec<Double, 16> &a)
{
#ifdef __SSE4_1__
  return _mm_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
  // There is no _mm_cvtepi64_pd in SSE*, which makes the workaround
  // used for the float version not possible here.
  // Another workaround would probably be complicated and slow, so we just
  // truncate serially.
  // TODO: is there a better, vectorized workaround?
  Double inArr[2] SIMD_ATTR_ALIGNED(16);
  _mm_store_pd(inArr, a);
  Double outArr[2] SIMD_ATTR_ALIGNED(16);
  outArr[0] = std::trunc(inArr[0]);
  outArr[1] = std::trunc(inArr[1]);
  return _mm_load_pd(outArr);
#endif
}

// ---------------------------------------------------------------------------
// elementary mathematical functions
// ---------------------------------------------------------------------------

// estimate of a reciprocal
static SIMD_INLINE Vec<Float, 16> rcp(const Vec<Float, 16> &a)
{
  return _mm_rcp_ps(a);
}

// estimate of a reciprocal
static SIMD_INLINE Vec<Double, 16> rcp(const Vec<Double, 16> &a)
{
  // _mm_rcp_pd does not exist, use _mm_div_pd instead
  return _mm_div_pd(_mm_set1_pd(1.0), a);
}

// estimate of a reverse square root
static SIMD_INLINE Vec<Float, 16> rsqrt(const Vec<Float, 16> &a)
{
  return _mm_rsqrt_ps(a);
}

// estimate of a reverse square root
static SIMD_INLINE Vec<Double, 16> rsqrt(const Vec<Double, 16> &a)
{
  // _mm_rsqrt_pd does not exist, use _mm_div_pd and _mm_sqrt_pd instead
  return _mm_div_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(a));
}

// square root
static SIMD_INLINE Vec<Float, 16> sqrt(const Vec<Float, 16> &a)
{
  return _mm_sqrt_ps(a);
}

// square root
static SIMD_INLINE Vec<Double, 16> sqrt(const Vec<Double, 16> &a)
{
  return _mm_sqrt_pd(a);
}

// ---------------------------------------------------------------------------
// abs
// ---------------------------------------------------------------------------

// 25. Mar 25 (Jonas Keller): added abs for unsigned integers

// unsigned integers
template <typename T, SIMD_ENABLE_IF(std::is_unsigned<T>::value
                                       &&std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 16> abs(const Vec<T, 16> &a)
{
  return a;
}

static SIMD_INLINE Vec<SignedByte, 16> abs(const Vec<SignedByte, 16> &a)
{
  return _mm_abs_epi8(a);
}

static SIMD_INLINE Vec<Short, 16> abs(const Vec<Short, 16> &a)
{
  return _mm_abs_epi16(a);
}

static SIMD_INLINE Vec<Int, 16> abs(const Vec<Int, 16> &a)
{
  return _mm_abs_epi32(a);
}

static SIMD_INLINE Vec<Float, 16> abs(const Vec<Float, 16> &a)
{
  return x_mm_abs_ps(a);
}

static SIMD_INLINE Vec<Double, 16> abs(const Vec<Double, 16> &a)
{
  return x_mm_abs_pd(a);
}

// ---------------------------------------------------------------------------
// unpacklo
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<0>, Bytes<1>)
{
  return _mm_unpacklo_epi8(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<0>, Bytes<2>)
{
  return _mm_unpacklo_epi16(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<0>, Bytes<4>)
{
  return _mm_unpacklo_epi32(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<0>, Bytes<8>)
{
  return _mm_unpacklo_epi64(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> unpack(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b, Part<0>,
                                         Bytes<4>)
{
  return _mm_unpacklo_ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> unpack(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b, Part<0>,
                                         Bytes<8>)
{
  // this moves two lower floats from a and b
  return _mm_movelh_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 16> unpack(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b, Part<0>,
                                          Bytes<8>)
{
  return _mm_unpacklo_pd(a, b);
}

// ---------------------------------------------------------------------------
// unpackhi
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<1>, Bytes<1>)
{
  return _mm_unpackhi_epi8(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<1>, Bytes<2>)
{
  return _mm_unpackhi_epi16(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<1>, Bytes<4>)
{
  return _mm_unpackhi_epi32(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> unpack(const Vec<T, 16> &a, const Vec<T, 16> &b,
                                     Part<1>, Bytes<8>)
{
  return _mm_unpackhi_epi64(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> unpack(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b, Part<1>,
                                         Bytes<4>)
{
  return _mm_unpackhi_ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> unpack(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b, Part<1>,
                                         Bytes<8>)
{
  // this moves two upper floats from a and b
  // order b, a
  return _mm_movehl_ps(b, a);
}

// double version
static SIMD_INLINE Vec<Double, 16> unpack(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b, Part<1>,
                                          Bytes<8>)
{
  return _mm_unpackhi_pd(a, b);
}

// contributed by Adam Marschall

// 16-byte-lane oriented unpack: for 16 bytes same as generalized unpack
// unpack blocks of NUM_ELEMS elements of type T
// PART=0: low half of input vectors,
// PART=1: high half of input vectors
template <int PART, int NUM_ELEMS, typename T>
static SIMD_INLINE Vec<T, 16> unpack16(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  return unpack(a, b, Part<PART>(), Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// extract 128-bit lane as Vec<T, 16>, does nothing for 16 bytes
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

template <int IMM, typename T>
static SIMD_INLINE Vec<T, 16> extractLane(const Vec<T, 16> &a)
{
  return a;
}

// ---------------------------------------------------------------------------
// zip (two unpacks similar to ARM NEON vzip, but for different NUM_ELEMS)
// ---------------------------------------------------------------------------

// a, b are passed by-value to avoid problems with identical input/output args.

// here we can directly map zip to unpack<PART,NUM_ELEMS,T>
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void zip(const Vec<T, 16> a, const Vec<T, 16> b,
                            Vec<T, 16> &l, Vec<T, 16> &h)
{
  l = unpack<0, NUM_ELEMS>(a, b);
  h = unpack<1, NUM_ELEMS>(a, b);
}

// ---------------------------------------------------------------------------
// zip16 hub  (16-byte-lane oriented zip): for 16 bytes same as zip
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// a, b are passed by-value to avoid problems with identical
// input/output args.

template <int NUM_ELEMS, typename T>
static SIMD_INLINE void zip16(const Vec<T, 16> a, const Vec<T, 16> b,
                              Vec<T, 16> &l, Vec<T, 16> &h)
{
  zip<NUM_ELEMS, T>(a, b, l, h);
}

// ---------------------------------------------------------------------------
// unzip (similar to ARM NEON vuzp, but for different NUM_ELEMS)
// ---------------------------------------------------------------------------

// solutions by Peter Cordes and Starvin Marvin:
// stackoverflow.com/q/45376193/3852630 and
// stackoverflow.com/a/45385216/3852630 and
// stackoverflow.com/q/20504618/3852630

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 16> a, const Vec<T, 16> b,
                              Vec<T, 16> &l, Vec<T, 16> &h, Bytes<1>)
{
  // mask is hopefully only set once if unzip is used multiple times
  __m128i mask =
    _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
  __m128i atmp = _mm_shuffle_epi8(a, mask);
  __m128i btmp = _mm_shuffle_epi8(b, mask);
  l            = _mm_unpacklo_epi64(atmp, btmp);
  h            = _mm_unpackhi_epi64(atmp, btmp);
}

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 16> a, const Vec<T, 16> b,
                              Vec<T, 16> &l, Vec<T, 16> &h, Bytes<2>)
{
  // mask is hopefully only set once if unzip is used multiple times
  __m128i mask =
    _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
  __m128i atmp = _mm_shuffle_epi8(a, mask);
  __m128i btmp = _mm_shuffle_epi8(b, mask);
  l            = _mm_unpacklo_epi64(atmp, btmp);
  h            = _mm_unpackhi_epi64(atmp, btmp);
}

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 16> a, const Vec<T, 16> b,
                              Vec<T, 16> &l, Vec<T, 16> &h, Bytes<4>)
{
  __m128 aps = _mm_castsi128_ps(a);
  __m128 bps = _mm_castsi128_ps(b);
  l = _mm_castps_si128(_mm_shuffle_ps(aps, bps, _MM_SHUFFLE(2, 0, 2, 0)));
  h = _mm_castps_si128(_mm_shuffle_ps(aps, bps, _MM_SHUFFLE(3, 1, 3, 1)));
}

// all types
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 16> a, const Vec<T, 16> b,
                              Vec<T, 16> &l, Vec<T, 16> &h, Bytes<8>)
{
  l = unpack(a, b, Part<0>(), Bytes<8>());
  h = unpack(a, b, Part<1>(), Bytes<8>());
}

// Float
static SIMD_INLINE void unzip(const Vec<Float, 16> a, const Vec<Float, 16> b,
                              Vec<Float, 16> &l, Vec<Float, 16> &h, Bytes<4>)
{
  l = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
  h = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
}

// ---------------------------------------------------------------------------
// packs
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE Vec<SignedByte, 16> packs(const Vec<Short, 16> &a,
                                             const Vec<Short, 16> &b,
                                             OutputType<SignedByte>)
{
  return _mm_packs_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 16> packs(const Vec<Int, 16> &a,
                                        const Vec<Int, 16> &b,
                                        OutputType<Short>)
{
  return _mm_packs_epi32(a, b);
}

static SIMD_INLINE Vec<Short, 16> packs(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b,
                                        OutputType<Short>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Short>());
}

static SIMD_INLINE Vec<Float, 16> packs(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b,
                                        OutputType<Float>)
{
  return _mm_shuffle_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b),
                        _MM_SHUFFLE(1, 0, 1, 0));
}

// Double to Int

static SIMD_INLINE Vec<Int, 16> packs(const Vec<Double, 16> &a,
                                      const Vec<Double, 16> &b, OutputType<Int>)
{
  auto clip = _mm_set1_pd(std::numeric_limits<Int>::max());
  auto bI   = _mm_castsi128_ps(_mm_cvtpd_epi32(_mm_min_pd(clip, b)));
  auto aI   = _mm_castsi128_ps(_mm_cvtpd_epi32(_mm_min_pd(clip, a)));
  return _mm_castps_si128(_mm_shuffle_ps(aI, bI, _MM_SHUFFLE(1, 0, 1, 0)));
}

// signed -> unsigned

static SIMD_INLINE Vec<Byte, 16> packs(const Vec<Short, 16> &a,
                                       const Vec<Short, 16> &b,
                                       OutputType<Byte>)
{
  return _mm_packus_epi16(a, b);
}

static SIMD_INLINE Vec<Word, 16> packs(const Vec<Int, 16> &a,
                                       const Vec<Int, 16> &b, OutputType<Word>)
{
#ifdef __SSE4_1__
  return _mm_packus_epi32(a, b);
#else
  // mask for lower 16 bit
  __m128i mask = _mm_set1_epi32(0x0000ffff);
  // a >= 0 ? asat = a : asat = 0
  // 23. Nov 17 (rm): 32->31
  __m128i asat = _mm_andnot_si128(_mm_srai_epi32(a, 31), a);
  // cmp/or is used to restrict number to 16 bit
  // srai/slli is used for sign extension of 16 bit number,
  // makes signed saturation (in packs) a no-op, see
  // http://stackoverflow.com/questions/12118910/
  // converting-float-vector-to-16-bit-int-without-saturating
  // e.g.
  // a = 0xffffffff (-1)  -> asat  = 0x00000000
  //                      -> cmpgt = 0x00000000
  //                      -> slli  = 0x00000000
  //                      -> or    = 0x00000000
  //                      -> srai  = 0x00000000
  //                      -> packs = 0x0000
  // a = 0x7fffffff (>=0) -> asat  = 0x7fffffff
  //                      -> cmpgt = 0xffffffff
  //                      -> slli  = 0xffff0000
  //                      -> or    = 0xffffffff
  //                      -> srai  = 0xffffffff
  //                      -> packs = 0xffff
  // a = 0x0000ffff (>=0) -> asat  = 0x0000ffff
  //                      -> cmpgt = 0x00000000
  //                      -> slli  = 0xffff0000
  //                      -> or    = 0xffff0000
  //                      -> srai  = 0xffffffff
  //                      -> packs = 0xffff
  // a = 0x0000fffe (>=0) -> asat  = 0x0000fffe
  //                      -> cmpgt = 0x00000000
  //                      -> slli  = 0xfffe0000
  //                      -> or    = 0xfffe0000
  //                      -> srai  = 0xfffffffe
  //                      -> packs = 0xfffe
  // a = 0x00007fff (>=0) -> asat  = 0x00007fff
  //                      -> cmpgt = 0x00000000
  //                      -> slli  = 0x7fff0000
  //                      -> or    = 0x7fff0000
  //                      -> srai  = 0x00007fff
  //                      -> packs = 0x7fff
  asat = _mm_srai_epi32(
    _mm_or_si128(_mm_slli_epi32(asat, 16), _mm_cmpgt_epi32(asat, mask)), 16);
  // same for b
  // 23. Nov 17 (rm): 32->31
  __m128i bsat = _mm_andnot_si128(_mm_srai_epi32(b, 31), b);
  bsat         = _mm_srai_epi32(
    _mm_or_si128(_mm_slli_epi32(bsat, 16), _mm_cmpgt_epi32(bsat, mask)), 16);
  return _mm_packs_epi32(asat, bsat);
#endif
}

static SIMD_INLINE Vec<Word, 16> packs(const Vec<Float, 16> &a,
                                       const Vec<Float, 16> &b,
                                       OutputType<Word>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Word>());
}

// ---------------------------------------------------------------------------
// generalized extend: no stage
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// generalized extend: single stage
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Short, 16> vOut[2])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi8_epi16(vIn);
  vOut[1] = _mm_cvtepi8_epi16(_mm_srli_si128(vIn, 8));
#else
  vOut[0] = _mm_srai_epi16(_mm_unpacklo_epi8(_mm_undefined_si128(), vIn), 8);
  vOut[1] = _mm_srai_epi16(_mm_unpackhi_epi8(_mm_undefined_si128(), vIn), 8);
#endif
}

static SIMD_INLINE void extend(const Vec<Short, 16> &vIn, Vec<Int, 16> vOut[2])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi16_epi32(vIn);
  vOut[1] = _mm_cvtepi16_epi32(_mm_srli_si128(vIn, 8));
#else
  vOut[0] = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_undefined_si128(), vIn), 16);
  vOut[1] = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_undefined_si128(), vIn), 16);
#endif
}

static SIMD_INLINE void extend(const Vec<Short, 16> &vIn,
                               Vec<Float, 16> vOut[2])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(vIn));
  vOut[1] = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(vIn, 8)));
#else
  vOut[0] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vIn, vIn), 16));
  vOut[1] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vIn, vIn), 16));
#endif
}

// unsigned -> unsigned

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Word, 16> vOut[2])
{
  // there's no _mm_cvtepu8_epu16()
  vOut[0] = _mm_unpacklo_epi8(vIn, _mm_setzero_si128());
  vOut[1] = _mm_unpackhi_epi8(vIn, _mm_setzero_si128());
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Short, 16> vOut[2])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepu8_epi16(vIn);
  vOut[1] = _mm_cvtepu8_epi16(_mm_srli_si128(vIn, 8));
#else
  vOut[0] = _mm_unpacklo_epi8(vIn, _mm_setzero_si128());
  vOut[1] = _mm_unpackhi_epi8(vIn, _mm_setzero_si128());
#endif
}

static SIMD_INLINE void extend(const Vec<Word, 16> &vIn, Vec<Int, 16> vOut[2])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepu16_epi32(vIn);
  vOut[1] = _mm_cvtepu16_epi32(_mm_srli_si128(vIn, 8));
#else
  vOut[0] = _mm_unpacklo_epi16(vIn, _mm_setzero_si128());
  vOut[1] = _mm_unpackhi_epi16(vIn, _mm_setzero_si128());
#endif
}

static SIMD_INLINE void extend(const Vec<Word, 16> &vIn, Vec<Float, 16> vOut[2])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(vIn));
  vOut[1] = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_srli_si128(vIn, 8)));
#else
  vOut[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(vIn, _mm_setzero_si128()));
  vOut[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(vIn, _mm_setzero_si128()));
#endif
}

static SIMD_INLINE void extend(const Vec<Int, 16> &vIn, Vec<Double, 16> vOut[2])
{
  vOut[0] = _mm_cvtepi32_pd(vIn);
  vOut[1] = _mm_cvtepi32_pd(_mm_srli_si128(vIn, 8));
}

static SIMD_INLINE void extend(const Vec<Float, 16> &vIn,
                               Vec<Double, 16> vOut[2])
{
  vOut[0] = _mm_cvtps_pd(vIn);
  vOut[1] =
    _mm_cvtps_pd(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(vIn), 8)));
}

// ---------------------------------------------------------------------------
// generalized extend: two stages
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Int, 16> vOut[4])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi8_epi32(vIn);
  vOut[1] = _mm_cvtepi8_epi32(_mm_srli_si128(vIn, 4));
  vOut[2] = _mm_cvtepi8_epi32(_mm_srli_si128(vIn, 8));
  vOut[3] = _mm_cvtepi8_epi32(_mm_srli_si128(vIn, 12));
#else
  const __m128i lo8    = _mm_unpacklo_epi8(_mm_undefined_si128(), vIn);
  const __m128i hi8    = _mm_unpackhi_epi8(_mm_undefined_si128(), vIn);
  const __m128i lolo16 = _mm_unpacklo_epi16(_mm_undefined_si128(), lo8);
  const __m128i lohi16 = _mm_unpackhi_epi16(_mm_undefined_si128(), lo8);
  const __m128i hilo16 = _mm_unpacklo_epi16(_mm_undefined_si128(), hi8);
  const __m128i hihi16 = _mm_unpackhi_epi16(_mm_undefined_si128(), hi8);
  vOut[0]              = _mm_srai_epi32(lolo16, 24);
  vOut[1]              = _mm_srai_epi32(lohi16, 24);
  vOut[2]              = _mm_srai_epi32(hilo16, 24);
  vOut[3]              = _mm_srai_epi32(hihi16, 24);
#endif
}

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Float, 16> vOut[4])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(vIn));
  vOut[1] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 4)));
  vOut[2] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 8)));
  vOut[3] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 12)));
#else
  const __m128i lo8    = _mm_unpacklo_epi8(_mm_undefined_si128(), vIn);
  const __m128i hi8    = _mm_unpackhi_epi8(_mm_undefined_si128(), vIn);
  const __m128i lolo16 = _mm_unpacklo_epi16(_mm_undefined_si128(), lo8);
  const __m128i lohi16 = _mm_unpackhi_epi16(_mm_undefined_si128(), lo8);
  const __m128i hilo16 = _mm_unpacklo_epi16(_mm_undefined_si128(), hi8);
  const __m128i hihi16 = _mm_unpackhi_epi16(_mm_undefined_si128(), hi8);
  vOut[0]              = _mm_cvtepi32_ps(_mm_srai_epi32(lolo16, 24));
  vOut[1]              = _mm_cvtepi32_ps(_mm_srai_epi32(lohi16, 24));
  vOut[2]              = _mm_cvtepi32_ps(_mm_srai_epi32(hilo16, 24));
  vOut[3]              = _mm_cvtepi32_ps(_mm_srai_epi32(hihi16, 24));
#endif
}

static SIMD_INLINE void extend(const Vec<Short, 16> &vIn,
                               Vec<Double, 16> vOut[4])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_pd(_mm_cvtepi16_epi32(vIn));
  vOut[1] = _mm_cvtepi32_pd(_mm_cvtepi16_epi32(_mm_srli_si128(vIn, 4)));
  vOut[2] = _mm_cvtepi32_pd(_mm_cvtepi16_epi32(_mm_srli_si128(vIn, 8)));
  vOut[3] = _mm_cvtepi32_pd(_mm_cvtepi16_epi32(_mm_srli_si128(vIn, 12)));
#else
  const __m128i lo16 =
    _mm_srai_epi32(_mm_unpacklo_epi16(_mm_undefined_si128(), vIn), 16);
  const __m128i hi16 =
    _mm_srai_epi32(_mm_unpackhi_epi16(_mm_undefined_si128(), vIn), 16);
  vOut[0]           = _mm_cvtepi32_pd(lo16);
  vOut[1]           = _mm_cvtepi32_pd(_mm_srli_si128(lo16, 8));
  vOut[2]           = _mm_cvtepi32_pd(hi16);
  vOut[3]           = _mm_cvtepi32_pd(_mm_srli_si128(hi16, 8));
#endif
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Int, 16> vOut[4])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepu8_epi32(vIn);
  vOut[1] = _mm_cvtepu8_epi32(_mm_srli_si128(vIn, 4));
  vOut[2] = _mm_cvtepu8_epi32(_mm_srli_si128(vIn, 8));
  vOut[3] = _mm_cvtepu8_epi32(_mm_srli_si128(vIn, 12));
#else
  const __m128i lo8 = _mm_unpacklo_epi8(vIn, _mm_setzero_si128());
  const __m128i hi8 = _mm_unpackhi_epi8(vIn, _mm_setzero_si128());
  vOut[0]           = _mm_unpacklo_epi16(lo8, _mm_setzero_si128());
  vOut[1]           = _mm_unpackhi_epi16(lo8, _mm_setzero_si128());
  vOut[2]           = _mm_unpacklo_epi16(hi8, _mm_setzero_si128());
  vOut[3]           = _mm_unpackhi_epi16(hi8, _mm_setzero_si128());
#endif
}

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn, Vec<Float, 16> vOut[4])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(vIn));
  vOut[1] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 4)));
  vOut[2] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 8)));
  vOut[3] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 12)));
#else
  const __m128i lo8 = _mm_unpacklo_epi8(vIn, _mm_setzero_si128());
  const __m128i hi8 = _mm_unpackhi_epi8(vIn, _mm_setzero_si128());
  vOut[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo8, _mm_setzero_si128()));
  vOut[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo8, _mm_setzero_si128()));
  vOut[2] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi8, _mm_setzero_si128()));
  vOut[3] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi8, _mm_setzero_si128()));
#endif
}

static SIMD_INLINE void extend(const Vec<Word, 16> &vIn,
                               Vec<Double, 16> vOut[4])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_pd(_mm_cvtepu16_epi32(vIn));
  vOut[1] = _mm_cvtepi32_pd(_mm_cvtepu16_epi32(_mm_srli_si128(vIn, 4)));
  vOut[2] = _mm_cvtepi32_pd(_mm_cvtepu16_epi32(_mm_srli_si128(vIn, 8)));
  vOut[3] = _mm_cvtepi32_pd(_mm_cvtepu16_epi32(_mm_srli_si128(vIn, 12)));
#else
  const __m128i lo16      = _mm_unpacklo_epi16(vIn, _mm_setzero_si128());
  const __m128i hi16      = _mm_unpackhi_epi16(vIn, _mm_setzero_si128());
  vOut[0]                 = _mm_cvtepi32_pd(lo16);
  vOut[1]                 = _mm_cvtepi32_pd(_mm_srli_si128(lo16, 8));
  vOut[2]                 = _mm_cvtepi32_pd(hi16);
  vOut[3]                 = _mm_cvtepi32_pd(_mm_srli_si128(hi16, 8));
#endif
}

// ---------------------------------------------------------------------------
// generalized extend: three stages
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 16> &vIn,
                               Vec<Double, 16> vOut[8])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(vIn));
  vOut[1] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 2)));
  vOut[2] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 4)));
  vOut[3] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 6)));
  vOut[4] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 8)));
  vOut[5] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 10)));
  vOut[6] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 12)));
  vOut[7] = _mm_cvtepi32_pd(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 14)));
#else
  const __m128i lo8       = _mm_unpacklo_epi8(_mm_undefined_si128(), vIn);
  const __m128i hi8       = _mm_unpackhi_epi8(_mm_undefined_si128(), vIn);
  const __m128i lolo16    = _mm_unpacklo_epi16(_mm_undefined_si128(), lo8);
  const __m128i lohi16    = _mm_unpackhi_epi16(_mm_undefined_si128(), lo8);
  const __m128i hilo16    = _mm_unpacklo_epi16(_mm_undefined_si128(), hi8);
  const __m128i hihi16    = _mm_unpackhi_epi16(_mm_undefined_si128(), hi8);
  const __m128i lolo16ext = _mm_srai_epi32(lolo16, 24);
  const __m128i lohi16ext = _mm_srai_epi32(lohi16, 24);
  const __m128i hilo16ext = _mm_srai_epi32(hilo16, 24);
  const __m128i hihi16ext = _mm_srai_epi32(hihi16, 24);
  vOut[0]                 = _mm_cvtepi32_pd(lolo16ext);
  vOut[1]                 = _mm_cvtepi32_pd(_mm_srli_si128(lolo16ext, 8));
  vOut[2]                 = _mm_cvtepi32_pd(lohi16ext);
  vOut[3]                 = _mm_cvtepi32_pd(_mm_srli_si128(lohi16ext, 8));
  vOut[4]                 = _mm_cvtepi32_pd(hilo16ext);
  vOut[5]                 = _mm_cvtepi32_pd(_mm_srli_si128(hilo16ext, 8));
  vOut[6]                 = _mm_cvtepi32_pd(hihi16ext);
  vOut[7]                 = _mm_cvtepi32_pd(_mm_srli_si128(hihi16ext, 8));
#endif
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 16> &vIn,
                               Vec<Double, 16> vOut[8])
{
#ifdef __SSE4_1__
  vOut[0] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(vIn));
  vOut[1] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 2)));
  vOut[2] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 4)));
  vOut[3] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 6)));
  vOut[4] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 8)));
  vOut[5] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 10)));
  vOut[6] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 12)));
  vOut[7] = _mm_cvtepi32_pd(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 14)));
#else
  const __m128i lo8       = _mm_unpacklo_epi8(vIn, _mm_setzero_si128());
  const __m128i hi8       = _mm_unpackhi_epi8(vIn, _mm_setzero_si128());
  const __m128i lolo16    = _mm_unpacklo_epi16(lo8, _mm_setzero_si128());
  const __m128i lohi16    = _mm_unpackhi_epi16(lo8, _mm_setzero_si128());
  const __m128i hilo16    = _mm_unpacklo_epi16(hi8, _mm_setzero_si128());
  const __m128i hihi16    = _mm_unpackhi_epi16(hi8, _mm_setzero_si128());
  vOut[0]                 = _mm_cvtepi32_pd(lolo16);
  vOut[1]                 = _mm_cvtepi32_pd(_mm_srli_si128(lolo16, 8));
  vOut[2]                 = _mm_cvtepi32_pd(lohi16);
  vOut[3]                 = _mm_cvtepi32_pd(_mm_srli_si128(lohi16, 8));
  vOut[4]                 = _mm_cvtepi32_pd(hilo16);
  vOut[5]                 = _mm_cvtepi32_pd(_mm_srli_si128(hilo16, 8));
  vOut[6]                 = _mm_cvtepi32_pd(hihi16);
  vOut[7]                 = _mm_cvtepi32_pd(_mm_srli_si128(hihi16, 8));
#endif
}

// ---------------------------------------------------------------------------
// generalized extend: special case int <-> float
// ---------------------------------------------------------------------------

static SIMD_INLINE void extend(const Vec<Int, 16> &vIn, Vec<Float, 16> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Float, 16> &vIn, Vec<Int, 16> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Int>());
}

// ---------------------------------------------------------------------------
// srai
// ---------------------------------------------------------------------------

// 16. Oct 22 (Jonas Keller): added missing Byte and SignedByte versions

template <int IMM>
static SIMD_INLINE Vec<Byte, 16> srai(const Vec<Byte, 16> &a)
{
  return x_mm_srai_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<SignedByte, 16> srai(const Vec<SignedByte, 16> &a)
{
  return x_mm_srai_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Word, 16> srai(const Vec<Word, 16> &a)
{
  return x_mm_srai_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 16> srai(const Vec<Short, 16> &a)
{
  return x_mm_srai_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Int, 16> srai(const Vec<Int, 16> &a)
{
  return x_mm_srai_epi32<IMM>(a);
}

// ---------------------------------------------------------------------------
// srli
// ---------------------------------------------------------------------------

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<Byte, 16> srli(const Vec<Byte, 16> &a)
{
  return _mm_and_si128(_mm_set1_epi8((int8_t) (0xff >> IMM)),
                       x_mm_srli_epi32<IMM>(a));
}

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<SignedByte, 16> srli(const Vec<SignedByte, 16> &a)
{
  return _mm_and_si128(_mm_set1_epi8((int8_t) (0xff >> IMM)),
                       x_mm_srli_epi32<IMM>(a));
}

template <int IMM>
static SIMD_INLINE Vec<Word, 16> srli(const Vec<Word, 16> &a)
{
  return x_mm_srli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 16> srli(const Vec<Short, 16> &a)
{
  return x_mm_srli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Int, 16> srli(const Vec<Int, 16> &a)
{
  return x_mm_srli_epi32<IMM>(a);
}

// ---------------------------------------------------------------------------
// slli
// ---------------------------------------------------------------------------

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<Byte, 16> slli(const Vec<Byte, 16> &a)
{
  return _mm_and_si128(_mm_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << IMM))),
                       x_mm_slli_epi32<IMM>(a));
}

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<SignedByte, 16> slli(const Vec<SignedByte, 16> &a)
{
  return _mm_and_si128(_mm_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << IMM))),
                       x_mm_slli_epi32<IMM>(a));
}

template <int IMM>
static SIMD_INLINE Vec<Word, 16> slli(const Vec<Word, 16> &a)
{
  return x_mm_slli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 16> slli(const Vec<Short, 16> &a)
{
  return x_mm_slli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Int, 16> slli(const Vec<Int, 16> &a)
{
  return x_mm_slli_epi32<IMM>(a);
}

// 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

// ---------------------------------------------------------------------------
// sra
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> sra(const Vec<Byte, 16> &a,
                                     const uint8_t count)
{
  // there is no _mm_sra_epi8 intrinsic
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm_cmplt_epi8(a, _mm_setzero_si128());
  }
  __m128i odd = _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m128i even =
    _mm_sra_epi16(_mm_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  return ifelse<Byte>(_mm_set1_epi16((int16_t) 0xFF00), odd, even);
}

static SIMD_INLINE Vec<SignedByte, 16> sra(const Vec<SignedByte, 16> &a,
                                           const uint8_t count)
{
  // there is no _mm_sra_epi8 intrinsic
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm_cmplt_epi8(a, _mm_setzero_si128());
  }
  __m128i odd = _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m128i even =
    _mm_sra_epi16(_mm_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  return ifelse<SignedByte>(_mm_set1_epi16((int16_t) 0xFF00), odd, even);
}

static SIMD_INLINE Vec<Word, 16> sra(const Vec<Word, 16> &a,
                                     const uint8_t count)
{
  return _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 16> sra(const Vec<Short, 16> &a,
                                      const uint8_t count)
{
  return _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Int, 16> sra(const Vec<Int, 16> &a, const uint8_t count)
{
  return _mm_sra_epi32(a, _mm_cvtsi32_si128(count));
}

// ---------------------------------------------------------------------------
// srl
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> srl(const Vec<Byte, 16> &a,
                                     const uint8_t count)
{
  return _mm_and_si128(_mm_srl_epi16(a, _mm_cvtsi32_si128(count)),
                       _mm_set1_epi8((int8_t) (uint8_t) (0xff >> count)));
}

static SIMD_INLINE Vec<SignedByte, 16> srl(const Vec<SignedByte, 16> &a,
                                           const uint8_t count)
{
  return _mm_and_si128(_mm_srl_epi16(a, _mm_cvtsi32_si128(count)),
                       _mm_set1_epi8((int8_t) (uint8_t) (0xff >> count)));
}

static SIMD_INLINE Vec<Word, 16> srl(const Vec<Word, 16> &a,
                                     const uint8_t count)
{
  return _mm_srl_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 16> srl(const Vec<Short, 16> &a,
                                      const uint8_t count)
{
  return _mm_srl_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Int, 16> srl(const Vec<Int, 16> &a, const uint8_t count)
{
  return _mm_srl_epi32(a, _mm_cvtsi32_si128(count));
}

// ---------------------------------------------------------------------------
// sll
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> sll(const Vec<Byte, 16> &a,
                                     const uint8_t count)
{
  return _mm_and_si128(
    _mm_sll_epi16(a, _mm_cvtsi32_si128(count)),
    _mm_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << count))));
}

static SIMD_INLINE Vec<SignedByte, 16> sll(const Vec<SignedByte, 16> &a,
                                           const uint8_t count)
{
  return _mm_and_si128(
    _mm_sll_epi16(a, _mm_cvtsi32_si128(count)),
    _mm_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << count))));
}

static SIMD_INLINE Vec<Word, 16> sll(const Vec<Word, 16> &a,
                                     const uint8_t count)
{
  return _mm_sll_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 16> sll(const Vec<Short, 16> &a,
                                      const uint8_t count)
{
  return _mm_sll_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Int, 16> sll(const Vec<Int, 16> &a, const uint8_t count)
{
  return _mm_sll_epi32(a, _mm_cvtsi32_si128(count));
}

// 19. Sep 22 (Jonas Keller):
// added Byte and SignedByte versions of hadd, hadds, hsub and hsubs
// added Word version of hadds and hsubs

// ---------------------------------------------------------------------------
// hadd
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> hadd(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  Vec<T, 16> x, y;
  unzip(a, b, x, y, Bytes<sizeof(T)>());
  return add(x, y);
}

static SIMD_INLINE Vec<Word, 16> hadd(const Vec<Word, 16> &a,
                                      const Vec<Word, 16> &b)
{
  return _mm_hadd_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 16> hadd(const Vec<Short, 16> &a,
                                       const Vec<Short, 16> &b)
{
  return _mm_hadd_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> hadd(const Vec<Int, 16> &a,
                                     const Vec<Int, 16> &b)
{
  return _mm_hadd_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> hadd(const Vec<Float, 16> &a,
                                       const Vec<Float, 16> &b)
{
  return _mm_hadd_ps(a, b);
}

// _mm_hadd_pd is only available with SSE3, if compiling without SSE3
// use template version above
#ifdef __SSE3__
static SIMD_INLINE Vec<Double, 16> hadd(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b)
{
  return _mm_hadd_pd(a, b);
}
#endif

// ---------------------------------------------------------------------------
// hadds
// ---------------------------------------------------------------------------

// 09. Mar 23 (Jonas Keller): made Int version of hadds saturating

template <typename T>
static SIMD_INLINE Vec<T, 16> hadds(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  Vec<T, 16> x, y;
  unzip(a, b, x, y, Bytes<sizeof(T)>());
  return adds(x, y);
}

static SIMD_INLINE Vec<Short, 16> hadds(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_hadds_epi16(a, b);
}

// Float not saturated
static SIMD_INLINE Vec<Float, 16> hadds(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_hadd_ps(a, b);
}

// _mm_hadd_pd is only available with SSE3, if compiling without SSE3
// use template version above
#ifdef __SSE3__
static SIMD_INLINE Vec<Double, 16> hadds(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_hadd_pd(a, b);
}
#endif

// ---------------------------------------------------------------------------
// hsub
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 16> hsub(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  Vec<T, 16> x, y;
  unzip(a, b, x, y, Bytes<sizeof(T)>());
  return sub(x, y);
}

static SIMD_INLINE Vec<Word, 16> hsub(const Vec<Word, 16> &a,
                                      const Vec<Word, 16> &b)
{
  return _mm_hsub_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 16> hsub(const Vec<Short, 16> &a,
                                       const Vec<Short, 16> &b)
{
  return _mm_hsub_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> hsub(const Vec<Int, 16> &a,
                                     const Vec<Int, 16> &b)
{
  return _mm_hsub_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> hsub(const Vec<Float, 16> &a,
                                       const Vec<Float, 16> &b)
{
  return _mm_hsub_ps(a, b);
}

// _mm_hsub_pd is only available with SSE3, if compiling without SSE3
// use template version above
#ifdef __SSE3__
static SIMD_INLINE Vec<Double, 16> hsub(const Vec<Double, 16> &a,
                                        const Vec<Double, 16> &b)
{
  return _mm_hsub_pd(a, b);
}
#endif

// ---------------------------------------------------------------------------
// hsubs
// ---------------------------------------------------------------------------

// 09. Mar 23 (Jonas Keller): made Int version of hsubs saturating

template <typename T>
static SIMD_INLINE Vec<T, 16> hsubs(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  Vec<T, 16> x, y;
  unzip(a, b, x, y, Bytes<sizeof(T)>());
  return subs(x, y);
}

static SIMD_INLINE Vec<Short, 16> hsubs(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_hsubs_epi16(a, b);
}

// Float not saturated
static SIMD_INLINE Vec<Float, 16> hsubs(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_hsub_ps(a, b);
}

// _mm_hsub_pd is only available with SSE3, if compiling without SSE3
// use template version above
#ifdef __SSE3__
static SIMD_INLINE Vec<Double, 16> hsubs(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_hsub_pd(a, b);
}
#endif

// ---------------------------------------------------------------------------
// element-wise shift right
// ---------------------------------------------------------------------------

// all integer versions
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 16> srle(const Vec<T, 16> &a)
{
  return x_mm_srli_si128<IMM * sizeof(T)>(a);
}

// float version
template <int IMM>
static SIMD_INLINE Vec<Float, 16> srle(const Vec<Float, 16> &a)
{
  return _mm_castsi128_ps(
    x_mm_srli_si128<IMM * sizeof(Float)>(_mm_castps_si128(a)));
}

// double version
template <int IMM>
static SIMD_INLINE Vec<Double, 16> srle(const Vec<Double, 16> &a)
{
  return _mm_castsi128_pd(
    x_mm_srli_si128<IMM * sizeof(Double)>(_mm_castpd_si128(a)));
}

// ---------------------------------------------------------------------------
// element-wise shift left
// ---------------------------------------------------------------------------

// all integer versions
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 16> slle(const Vec<T, 16> &a)
{
  return x_mm_slli_si128<IMM * sizeof(T)>(a);
}

// float version
template <int IMM>
static SIMD_INLINE Vec<Float, 16> slle(const Vec<Float, 16> &a)
{
  return _mm_castsi128_ps(
    x_mm_slli_si128<IMM * sizeof(Float)>(_mm_castps_si128(a)));
}

// double version
template <int IMM>
static SIMD_INLINE Vec<Double, 16> slle(const Vec<Double, 16> &a)
{
  return _mm_castsi128_pd(
    x_mm_slli_si128<IMM * sizeof(Double)>(_mm_castpd_si128(a)));
}

// ---------------------------------------------------------------------------
// extraction of element 0
// ---------------------------------------------------------------------------

static SIMD_INLINE Byte elem0(const Vec<Byte, 16> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Byte, _mm_cvtsi128_si32(a), 0);
}

static SIMD_INLINE SignedByte elem0(const Vec<SignedByte, 16> &a)
{
  SIMD_RETURN_REINTERPRET_INT(SignedByte, _mm_cvtsi128_si32(a), 0);
}

static SIMD_INLINE Word elem0(const Vec<Word, 16> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Word, _mm_cvtsi128_si32(a), 0);
}

static SIMD_INLINE Short elem0(const Vec<Short, 16> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Short, _mm_cvtsi128_si32(a), 0);
}

static SIMD_INLINE Int elem0(const Vec<Int, 16> &a)
{
  // TODO: elem0: is conversion from return type int so Int always safe?
  return _mm_cvtsi128_si32(a);
}

static SIMD_INLINE Float elem0(const Vec<Float, 16> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Float, _mm_cvtsi128_si32(_mm_castps_si128(a)), 0);
}

static SIMD_INLINE Double elem0(const Vec<Double, 16> &a)
{
  Double res;
  long long i = _mm_cvtsi128_si64(_mm_castpd_si128(a));
  memcpy(&res, &i, sizeof(Double));
  return res;
}

// ---------------------------------------------------------------------------
// alignre
// ---------------------------------------------------------------------------

// all integer versions

// IMM in range
template <int IMM, typename T, SIMD_ENABLE_IF(IMM >= 0 && IMM * sizeof(T) < 32)>
static SIMD_INLINE Vec<T, 16> alignre(const Vec<T, 16> &h, const Vec<T, 16> &l)
{
  return _mm_alignr_epi8(h, l, IMM * sizeof(T));
}

// IMM out of range
template <int IMM, typename T, SIMD_ENABLE_IF(IMM * sizeof(T) >= 32),
          typename = void>
static SIMD_INLINE Vec<T, 16> alignre(const Vec<T, 16> &, const Vec<T, 16> &)
{
  return setzero(OutputType<T>(), Integer<16>());
}

// float version

// IMM in range
template <int IMM, SIMD_ENABLE_IF(IMM >= 0 && IMM * sizeof(Float) < 32)>
static SIMD_INLINE Vec<Float, 16> alignre(const Vec<Float, 16> &h,
                                          const Vec<Float, 16> &l)
{
  return _mm_castsi128_ps(_mm_alignr_epi8(
    _mm_castps_si128(h), _mm_castps_si128(l), IMM * sizeof(Float)));
}

// IMM out of range
template <int IMM, SIMD_ENABLE_IF(IMM * sizeof(Float) >= 32), typename = void>
static SIMD_INLINE Vec<Float, 16> alignre(const Vec<Float, 16> &,
                                          const Vec<Float, 16> &)
{
  return setzero(OutputType<Float>(), Integer<16>());
}

// double version
template <int IMM>
static SIMD_INLINE Vec<Double, 16> alignre(const Vec<Double, 16> &h,
                                           const Vec<Double, 16> &l)
{
  return _mm_castsi128_pd(_mm_alignr_epi8(
    _mm_castpd_si128(h), _mm_castpd_si128(l), IMM * sizeof(Double)));
}

// ---------------------------------------------------------------------------
// swizzle
// ---------------------------------------------------------------------------

// swizzle masks (only for 8 and 16 bit element types)

// [masks generated from ~/texte/Talks/SSE/swizzle.c]

// Byte, SignedByte

static SIMD_INLINE __m128i get_swizzle_mask(Integer<2>, Integer<1>)
{
  return _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
}

static SIMD_INLINE __m128i get_swizzle_mask(Integer<3>, Integer<1>)
{
  return _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, -1, -1, -1, -1);
}

static SIMD_INLINE __m128i get_swizzle_mask(Integer<4>, Integer<1>)
{
  return _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
}

static SIMD_INLINE __m128i get_swizzle_mask(Integer<5>, Integer<1>)
{
  return _mm_setr_epi8(0, 5, 1, 6, 2, 7, 3, 8, 4, 9, -1, -1, -1, -1, -1, -1);
}

// Word, Short

static SIMD_INLINE __m128i get_swizzle_mask(Integer<2>, Integer<2>)
{
  return _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
}

static SIMD_INLINE __m128i get_swizzle_mask(Integer<3>, Integer<2>)
{
  return _mm_setr_epi8(0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, -1, -1, -1, -1);
}

static SIMD_INLINE __m128i get_swizzle_mask(Integer<4>, Integer<2>)
{
  return _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
}

static SIMD_INLINE __m128i get_swizzle_mask(Integer<5>, Integer<2>)
{
  return _mm_setr_epi8(0, 1, 10, 11, 2, 3, 12, 13, 4, 5, 14, 15, -1, -1, -1,
                       -1);
}

// hub
template <int N, typename T>
static SIMD_INLINE __m128i get_swizzle_mask()
{
  return get_swizzle_mask(Integer<N>(), Integer<sizeof(T)>());
}

// ---------- swizzle aux functions -----------

// alignoff is the element-wise offset (relates to size of byte)
template <int ALIGNOFF>
static SIMD_INLINE __m128i align_shuffle_128(__m128i lo, __m128i hi,
                                             __m128i mask)
{
  static_assert(ALIGNOFF >= 0 && ALIGNOFF < 32, "");
  return _mm_shuffle_epi8(_mm_alignr_epi8(hi, lo, ALIGNOFF), mask);
}

// ---------- swizzle (AoS to SoA) ----------

// 01. Apr 23 (Jonas Keller): switched from using tag dispatching to using
// enable_if SFINAE, which allows more cases with the same implementation
// to be combined

// -------------------- n = 1 --------------------

// all types
template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 16>[1], Integer<1>)
{
  // v remains unchanged
}

// -------------------- n = 2 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) <= 2 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 16> v[2], Integer<2>)
{
  __m128i s[2];
  s[0] = _mm_shuffle_epi8(v[0], get_swizzle_mask<2, T>());
  s[1] = _mm_shuffle_epi8(v[1], get_swizzle_mask<2, T>());
  v[0] = _mm_unpacklo_epi64(s[0], s[1]);
  v[1] = _mm_unpackhi_epi64(s[0], s[1]);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[2], Integer<2>)
{
  const __m128 v0tmp = reinterpret(v[0], OutputType<Float>());
  const __m128 v1tmp = reinterpret(v[1], OutputType<Float>());
  const Vec<Float, 16> v0TmpOut =
    _mm_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(2, 0, 2, 0));
  const Vec<Float, 16> v1TmpOut =
    _mm_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(3, 1, 3, 1));
  v[0] = reinterpret(v0TmpOut, OutputType<T>());
  v[1] = reinterpret(v1TmpOut, OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[2], Integer<2>)
{
  const __m128d v0tmp = reinterpret(v[0], OutputType<Double>());
  const __m128d v1tmp = reinterpret(v[1], OutputType<Double>());
  const Vec<Double, 16> v0TmpOut =
    _mm_shuffle_pd(v0tmp, v1tmp, _MM_SHUFFLE2(0, 0));
  const Vec<Double, 16> v1TmpOut =
    _mm_shuffle_pd(v0tmp, v1tmp, _MM_SHUFFLE2(1, 1));
  v[0] = reinterpret(v0TmpOut, OutputType<T>());
  v[1] = reinterpret(v1TmpOut, OutputType<T>());
}

// -------------------- n = 3 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) <= 2 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 16> v[3], Integer<3>)
{
  __m128i mask = get_swizzle_mask<3, T>();
  __m128i s0   = align_shuffle_128<0>(v[0], v[1], mask);
  __m128i s1   = align_shuffle_128<12>(v[0], v[1], mask);
  __m128i s2   = align_shuffle_128<8>(v[1], v[2], mask);
  __m128i s3   = align_shuffle_128<4>(v[2], _mm_undefined_si128(), mask);
  __m128i l01  = _mm_unpacklo_epi32(s0, s1);
  __m128i h01  = _mm_unpackhi_epi32(s0, s1);
  __m128i l23  = _mm_unpacklo_epi32(s2, s3);
  __m128i h23  = _mm_unpackhi_epi32(s2, s3);
  v[0]         = _mm_unpacklo_epi64(l01, l23);
  v[1]         = _mm_unpackhi_epi64(l01, l23);
  v[2]         = _mm_unpacklo_epi64(h01, h23);
}

// 32 bit types
// from Stan Melax: "3D Vector Normalization..."
// https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[3], Integer<3>)
{
  const __m128 x0y0z0x1 = reinterpret(v[0], OutputType<Float>());
  const __m128 y1z1x2y2 = reinterpret(v[1], OutputType<Float>());
  const __m128 z2x3y3z3 = reinterpret(v[2], OutputType<Float>());
  const __m128 x2y2x3y3 =
    _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2, 1, 3, 2));
  const __m128 y0z0y1z1 =
    _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1, 0, 2, 1));
  const Vec<Float, 16> x0x1x2x3 =
    _mm_shuffle_ps(x0y0z0x1, x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0));
  const Vec<Float, 16> y0y1y2y3 =
    _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0));
  const Vec<Float, 16> z0z1z2z3 =
    _mm_shuffle_ps(y0z0y1z1, z2x3y3z3, _MM_SHUFFLE(3, 0, 3, 1));
  v[0] = reinterpret(x0x1x2x3, OutputType<T>());
  v[1] = reinterpret(y0y1y2y3, OutputType<T>());
  v[2] = reinterpret(z0z1z2z3, OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void, typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[3], Integer<3>)
{
  const __m128d x0y0         = reinterpret(v[0], OutputType<Double>());
  const __m128d z0x1         = reinterpret(v[1], OutputType<Double>());
  const __m128d y1z1         = reinterpret(v[2], OutputType<Double>());
  const Vec<Double, 16> x0x1 = _mm_shuffle_pd(x0y0, z0x1, _MM_SHUFFLE2(1, 0));
  const Vec<Double, 16> y0y1 = _mm_shuffle_pd(x0y0, y1z1, _MM_SHUFFLE2(0, 1));
  const Vec<Double, 16> z0z1 = _mm_shuffle_pd(z0x1, y1z1, _MM_SHUFFLE2(1, 0));
  v[0]                       = reinterpret(x0x1, OutputType<T>());
  v[1]                       = reinterpret(y0y1, OutputType<T>());
  v[2]                       = reinterpret(z0z1, OutputType<T>());
}

// -------------------- n = 4 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) <= 2 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 16> v[4], Integer<4>)
{
  __m128i mask = get_swizzle_mask<4, T>();
  __m128i s[4];
  s[0]        = _mm_shuffle_epi8(v[0], mask);
  s[1]        = _mm_shuffle_epi8(v[1], mask);
  s[2]        = _mm_shuffle_epi8(v[2], mask);
  s[3]        = _mm_shuffle_epi8(v[3], mask);
  __m128i l01 = _mm_unpacklo_epi32(s[0], s[1]);
  __m128i h01 = _mm_unpackhi_epi32(s[0], s[1]);
  __m128i l23 = _mm_unpacklo_epi32(s[2], s[3]);
  __m128i h23 = _mm_unpackhi_epi32(s[2], s[3]);
  v[0]        = _mm_unpacklo_epi64(l01, l23);
  v[1]        = _mm_unpackhi_epi64(l01, l23);
  v[2]        = _mm_unpacklo_epi64(h01, h23);
  v[3]        = _mm_unpackhi_epi64(h01, h23);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[4], Integer<4>)
{
  __m128 vFloat[4];
  for (int i = 0; i < 4; ++i) {
    vFloat[i] = reinterpret(v[i], OutputType<Float>());
  }
  __m128 s[4];
  s[0] = _mm_shuffle_ps(vFloat[0], vFloat[1], _MM_SHUFFLE(1, 0, 1, 0));
  s[1] = _mm_shuffle_ps(vFloat[0], vFloat[1], _MM_SHUFFLE(3, 2, 3, 2));
  s[2] = _mm_shuffle_ps(vFloat[2], vFloat[3], _MM_SHUFFLE(1, 0, 1, 0));
  s[3] = _mm_shuffle_ps(vFloat[2], vFloat[3], _MM_SHUFFLE(3, 2, 3, 2));
  Vec<Float, 16> vOut[4];
  vOut[0] = _mm_shuffle_ps(s[0], s[2], _MM_SHUFFLE(2, 0, 2, 0));
  vOut[1] = _mm_shuffle_ps(s[0], s[2], _MM_SHUFFLE(3, 1, 3, 1));
  vOut[2] = _mm_shuffle_ps(s[1], s[3], _MM_SHUFFLE(2, 0, 2, 0));
  vOut[3] = _mm_shuffle_ps(s[1], s[3], _MM_SHUFFLE(3, 1, 3, 1));
  for (int i = 0; i < 4; ++i) { v[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[4], Integer<4>)
{
  const __m128d x0y0         = reinterpret(v[0], OutputType<Double>());
  const __m128d z0w0         = reinterpret(v[1], OutputType<Double>());
  const __m128d x1y1         = reinterpret(v[2], OutputType<Double>());
  const __m128d z1w1         = reinterpret(v[3], OutputType<Double>());
  const Vec<Double, 16> x0x1 = _mm_unpacklo_pd(x0y0, x1y1);
  const Vec<Double, 16> y0y1 = _mm_unpackhi_pd(x0y0, x1y1);
  const Vec<Double, 16> z0z1 = _mm_unpacklo_pd(z0w0, z1w1);
  const Vec<Double, 16> w0w1 = _mm_unpackhi_pd(z0w0, z1w1);
  v[0]                       = reinterpret(x0x1, OutputType<T>());
  v[1]                       = reinterpret(y0y1, OutputType<T>());
  v[2]                       = reinterpret(z0z1, OutputType<T>());
  v[3]                       = reinterpret(w0w1, OutputType<T>());
}

// -------------------- n = 5 --------------------

// 8 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) == 1 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 16> v[5], Integer<5>)
{
  __m128i mask    = get_swizzle_mask<5, T>();
  __m128i s0      = align_shuffle_128<0>(v[0], v[1], mask);
  __m128i s1      = align_shuffle_128<10>(v[0], v[1], mask);
  __m128i s2      = align_shuffle_128<4>(v[1], v[2], mask);
  __m128i s3      = align_shuffle_128<14>(v[1], v[2], mask);
  __m128i s4      = align_shuffle_128<8>(v[2], v[3], mask);
  __m128i s5      = align_shuffle_128<2>(v[3], _mm_undefined_si128(), mask);
  __m128i s6      = align_shuffle_128<12>(v[3], v[4], mask);
  __m128i s7      = align_shuffle_128<6>(v[4], _mm_undefined_si128(), mask);
  __m128i l01     = _mm_unpacklo_epi16(s0, s1);
  __m128i h01     = _mm_unpackhi_epi16(s0, s1);
  __m128i l23     = _mm_unpacklo_epi16(s2, s3);
  __m128i h23     = _mm_unpackhi_epi16(s2, s3);
  __m128i l45     = _mm_unpacklo_epi16(s4, s5);
  __m128i h45     = _mm_unpackhi_epi16(s4, s5);
  __m128i l67     = _mm_unpacklo_epi16(s6, s7);
  __m128i h67     = _mm_unpackhi_epi16(s6, s7);
  __m128i ll01l23 = _mm_unpacklo_epi32(l01, l23);
  __m128i hl01l23 = _mm_unpackhi_epi32(l01, l23);
  __m128i ll45l67 = _mm_unpacklo_epi32(l45, l67);
  __m128i hl45l67 = _mm_unpackhi_epi32(l45, l67);
  __m128i lh01h23 = _mm_unpacklo_epi32(h01, h23);
  __m128i lh45h67 = _mm_unpacklo_epi32(h45, h67);
  v[0]            = _mm_unpacklo_epi64(ll01l23, ll45l67);
  v[1]            = _mm_unpackhi_epi64(ll01l23, ll45l67);
  v[2]            = _mm_unpacklo_epi64(hl01l23, hl45l67);
  v[3]            = _mm_unpackhi_epi64(hl01l23, hl45l67);
  v[4]            = _mm_unpacklo_epi64(lh01h23, lh45h67);
}

// 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) == 2 && std::is_integral<T>::value),
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[5], Integer<5>)
{
  __m128i mask = get_swizzle_mask<5, T>();
  __m128i s0   = align_shuffle_128<0>(v[0], v[1], mask);
  __m128i s1   = align_shuffle_128<6>(v[0], v[1], mask);
  __m128i s2   = align_shuffle_128<4>(v[1], v[2], mask);
  __m128i s3   = align_shuffle_128<10>(v[1], v[2], mask);
  __m128i s4   = align_shuffle_128<8>(v[2], v[3], mask);
  __m128i s5   = align_shuffle_128<14>(v[2], v[3], mask);
  __m128i s6   = align_shuffle_128<12>(v[3], v[4], mask);
  __m128i s7   = align_shuffle_128<2>(v[4], _mm_undefined_si128(), mask);
  __m128i l02  = _mm_unpacklo_epi32(s0, s2);
  __m128i h02  = _mm_unpackhi_epi32(s0, s2);
  __m128i l13  = _mm_unpacklo_epi32(s1, s3);
  __m128i l46  = _mm_unpacklo_epi32(s4, s6);
  __m128i h46  = _mm_unpackhi_epi32(s4, s6);
  __m128i l57  = _mm_unpacklo_epi32(s5, s7);
  v[0]         = _mm_unpacklo_epi64(l02, l46);
  v[1]         = _mm_unpackhi_epi64(l02, l46);
  v[2]         = _mm_unpacklo_epi64(h02, h46);
  v[3]         = _mm_unpacklo_epi64(l13, l57);
  v[4]         = _mm_unpackhi_epi64(l13, l57);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> vT[5], Integer<5>)
{
  __m128i v[5];
  for (int i = 0; i < 5; i++) { v[i] = reinterpret(vT[i], OutputType<Int>()); };
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  // v[0]: 0 1 2 3
  // v[1]: 4 x x x
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                   x x x   x
  // 5 6 7 8
  __m128i s2 = _mm_alignr_epi8(v[2], v[1], 4);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                             x  x  x    x
  // 9 x x x
  __m128i s3 = _mm_alignr_epi8(v[3], v[2], 4);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                x  x    x  x
  // 10 11 12 13
  __m128i s4 = _mm_alignr_epi8(v[3], v[2], 8);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                              x  x    x  x
  // 14 x x x
  __m128i s5 = _mm_alignr_epi8(v[4], v[3], 8);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                                 X    X  X  X
  // 15 16 17 18
  __m128i s6 = _mm_alignr_epi8(v[4], v[3], 12);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                                               X X X X
  // 19 x x x
  __m128i s7 = _mm_alignr_epi8(v[0], v[4], 12);
  // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
  __m128i l02 = _mm_unpacklo_epi32(v[0], s2);
  __m128i h02 = _mm_unpackhi_epi32(v[0], s2);
  // 4 x x x / 9 x x x -> 4 9 x x
  __m128i l13 = _mm_unpacklo_epi32(v[1], s3);
  // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
  __m128i l46 = _mm_unpacklo_epi32(s4, s6);
  __m128i h46 = _mm_unpackhi_epi32(s4, s6);
  // 14 x x x / 19 x x x -> 14 19 x x
  __m128i l57 = _mm_unpacklo_epi32(s5, s7);

  const Vec<Int, 16> vOut[5] = {
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    _mm_unpacklo_epi64(l02, l46),
    _mm_unpackhi_epi64(l02, l46),
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    _mm_unpacklo_epi64(h02, h46),
    _mm_unpackhi_epi64(h02, h46),
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    _mm_unpacklo_epi64(l13, l57),
  };
  for (int i = 0; i < 5; ++i) { vT[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void, typename = void>
static SIMD_INLINE void swizzle(Vec<T, 16> v[5], Integer<5>)
{
  const __m128d a0b0         = reinterpret(v[0], OutputType<Double>());
  const __m128d c0d0         = reinterpret(v[1], OutputType<Double>());
  const __m128d e0a1         = reinterpret(v[2], OutputType<Double>());
  const __m128d b1c1         = reinterpret(v[3], OutputType<Double>());
  const __m128d d1e1         = reinterpret(v[4], OutputType<Double>());
  const Vec<Double, 16> a0a1 = _mm_shuffle_pd(a0b0, e0a1, _MM_SHUFFLE2(1, 0));
  const Vec<Double, 16> b0b1 = _mm_shuffle_pd(a0b0, b1c1, _MM_SHUFFLE2(0, 1));
  const Vec<Double, 16> c0c1 = _mm_shuffle_pd(c0d0, b1c1, _MM_SHUFFLE2(1, 0));
  const Vec<Double, 16> d0d1 = _mm_shuffle_pd(c0d0, d1e1, _MM_SHUFFLE2(0, 1));
  const Vec<Double, 16> e0e1 = _mm_shuffle_pd(e0a1, d1e1, _MM_SHUFFLE2(1, 0));
  v[0]                       = reinterpret(a0a1, OutputType<T>());
  v[1]                       = reinterpret(b0b1, OutputType<T>());
  v[2]                       = reinterpret(c0c1, OutputType<T>());
  v[3]                       = reinterpret(d0d1, OutputType<T>());
  v[4]                       = reinterpret(e0e1, OutputType<T>());
}

// ---------------------------------------------------------------------------
// compare <
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
static SIMD_INLINE Vec<Byte, 16> cmplt(const Vec<Byte, 16> &a,
                                       const Vec<Byte, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x80
  return _mm_cmplt_epi8(a1, b1);
}

static SIMD_INLINE Vec<SignedByte, 16> cmplt(const Vec<SignedByte, 16> &a,
                                             const Vec<SignedByte, 16> &b)
{
  return _mm_cmplt_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> cmplt(const Vec<Word, 16> &a,
                                       const Vec<Word, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x8000
  return _mm_cmplt_epi16(a1, b1);
}

static SIMD_INLINE Vec<Short, 16> cmplt(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_cmplt_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> cmplt(const Vec<Int, 16> &a,
                                      const Vec<Int, 16> &b)
{
  return _mm_cmplt_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> cmplt(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_cmplt_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> cmplt(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_cmplt_pd(a, b);
}

// ---------------------------------------------------------------------------
// compare <=
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
static SIMD_INLINE Vec<Byte, 16> cmple(const Vec<Byte, 16> &a,
                                       const Vec<Byte, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x80
  return _mm_or_si128(_mm_cmplt_epi8(a1, b1), _mm_cmpeq_epi8(a1, b1));
}

static SIMD_INLINE Vec<SignedByte, 16> cmple(const Vec<SignedByte, 16> &a,
                                             const Vec<SignedByte, 16> &b)
{
  return _mm_or_si128(_mm_cmplt_epi8(a, b), _mm_cmpeq_epi8(a, b));
}

static SIMD_INLINE Vec<Word, 16> cmple(const Vec<Word, 16> &a,
                                       const Vec<Word, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x8000
  return _mm_or_si128(_mm_cmplt_epi16(a1, b1), _mm_cmpeq_epi16(a1, b1));
}

static SIMD_INLINE Vec<Short, 16> cmple(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_or_si128(_mm_cmplt_epi16(a, b), _mm_cmpeq_epi16(a, b));
}

static SIMD_INLINE Vec<Int, 16> cmple(const Vec<Int, 16> &a,
                                      const Vec<Int, 16> &b)
{
  return _mm_or_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, b));
}

static SIMD_INLINE Vec<Float, 16> cmple(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_cmple_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> cmple(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_cmple_pd(a, b);
}

// ---------------------------------------------------------------------------
// compare ==
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> cmpeq(const Vec<Byte, 16> &a,
                                       const Vec<Byte, 16> &b)
{
  return _mm_cmpeq_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 16> cmpeq(const Vec<SignedByte, 16> &a,
                                             const Vec<SignedByte, 16> &b)
{
  return _mm_cmpeq_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> cmpeq(const Vec<Word, 16> &a,
                                       const Vec<Word, 16> &b)
{
  return _mm_cmpeq_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 16> cmpeq(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_cmpeq_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> cmpeq(const Vec<Int, 16> &a,
                                      const Vec<Int, 16> &b)
{
  return _mm_cmpeq_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> cmpeq(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_cmpeq_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> cmpeq(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_cmpeq_pd(a, b);
}

// ---------------------------------------------------------------------------
// compare >
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
static SIMD_INLINE Vec<Byte, 16> cmpgt(const Vec<Byte, 16> &a,
                                       const Vec<Byte, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x80
  return _mm_cmpgt_epi8(a1, b1);
}

static SIMD_INLINE Vec<SignedByte, 16> cmpgt(const Vec<SignedByte, 16> &a,
                                             const Vec<SignedByte, 16> &b)
{
  return _mm_cmpgt_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 16> cmpgt(const Vec<Word, 16> &a,
                                       const Vec<Word, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x8000
  return _mm_cmpgt_epi16(a1, b1);
}

static SIMD_INLINE Vec<Short, 16> cmpgt(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_cmpgt_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 16> cmpgt(const Vec<Int, 16> &a,
                                      const Vec<Int, 16> &b)
{
  return _mm_cmpgt_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 16> cmpgt(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_cmpgt_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> cmpgt(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_cmpgt_pd(a, b);
}

// ---------------------------------------------------------------------------
// compare >=
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
static SIMD_INLINE Vec<Byte, 16> cmpge(const Vec<Byte, 16> &a,
                                       const Vec<Byte, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x80
  return _mm_or_si128(_mm_cmpgt_epi8(a1, b1), _mm_cmpeq_epi8(a1, b1));
}

static SIMD_INLINE Vec<SignedByte, 16> cmpge(const Vec<SignedByte, 16> &a,
                                             const Vec<SignedByte, 16> &b)
{
  return _mm_or_si128(_mm_cmpgt_epi8(a, b), _mm_cmpeq_epi8(a, b));
}

static SIMD_INLINE Vec<Word, 16> cmpge(const Vec<Word, 16> &a,
                                       const Vec<Word, 16> &b)
{
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // sub 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // sub 0x8000
  return _mm_or_si128(_mm_cmpgt_epi16(a1, b1), _mm_cmpeq_epi16(a1, b1));
}

static SIMD_INLINE Vec<Short, 16> cmpge(const Vec<Short, 16> &a,
                                        const Vec<Short, 16> &b)
{
  return _mm_or_si128(_mm_cmpgt_epi16(a, b), _mm_cmpeq_epi16(a, b));
}

static SIMD_INLINE Vec<Int, 16> cmpge(const Vec<Int, 16> &a,
                                      const Vec<Int, 16> &b)
{
  return _mm_or_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, b));
}

static SIMD_INLINE Vec<Float, 16> cmpge(const Vec<Float, 16> &a,
                                        const Vec<Float, 16> &b)
{
  return _mm_cmpge_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> cmpge(const Vec<Double, 16> &a,
                                         const Vec<Double, 16> &b)
{
  return _mm_cmpge_pd(a, b);
}

// ---------------------------------------------------------------------------
// compare !=
// ---------------------------------------------------------------------------

// there is no cmpneq for integers and no not, so use cmpeq and xor with all
// ones to invert the result

static SIMD_INLINE Vec<Byte, 16> cmpneq(const Vec<Byte, 16> &a,
                                        const Vec<Byte, 16> &b)
{
  return _mm_xor_si128(_mm_cmpeq_epi8(a, b), _mm_set1_epi32(-1));
}

static SIMD_INLINE Vec<SignedByte, 16> cmpneq(const Vec<SignedByte, 16> &a,
                                              const Vec<SignedByte, 16> &b)
{
  return _mm_xor_si128(_mm_cmpeq_epi8(a, b), _mm_set1_epi32(-1));
}

static SIMD_INLINE Vec<Word, 16> cmpneq(const Vec<Word, 16> &a,
                                        const Vec<Word, 16> &b)
{
  return _mm_xor_si128(_mm_cmpeq_epi16(a, b), _mm_set1_epi32(-1));
}

static SIMD_INLINE Vec<Short, 16> cmpneq(const Vec<Short, 16> &a,
                                         const Vec<Short, 16> &b)
{
  return _mm_xor_si128(_mm_cmpeq_epi16(a, b), _mm_set1_epi32(-1));
}

static SIMD_INLINE Vec<Int, 16> cmpneq(const Vec<Int, 16> &a,
                                       const Vec<Int, 16> &b)
{
  return _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_set1_epi32(-1));
}

static SIMD_INLINE Vec<Float, 16> cmpneq(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b)
{
  return _mm_cmpneq_ps(a, b);
}

static SIMD_INLINE Vec<Double, 16> cmpneq(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b)
{
  return _mm_cmpneq_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_and
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> bit_and(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  return _mm_and_si128(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> bit_and(const Vec<Float, 16> &a,
                                          const Vec<Float, 16> &b)
{
  return _mm_and_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 16> bit_and(const Vec<Double, 16> &a,
                                           const Vec<Double, 16> &b)
{
  return _mm_and_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_or
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> bit_or(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  return _mm_or_si128(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> bit_or(const Vec<Float, 16> &a,
                                         const Vec<Float, 16> &b)
{
  return _mm_or_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 16> bit_or(const Vec<Double, 16> &a,
                                          const Vec<Double, 16> &b)
{
  return _mm_or_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_andnot
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> bit_andnot(const Vec<T, 16> &a,
                                         const Vec<T, 16> &b)
{
  return _mm_andnot_si128(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> bit_andnot(const Vec<Float, 16> &a,
                                             const Vec<Float, 16> &b)
{
  return _mm_andnot_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 16> bit_andnot(const Vec<Double, 16> &a,
                                              const Vec<Double, 16> &b)
{
  return _mm_andnot_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_xor
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> bit_xor(const Vec<T, 16> &a, const Vec<T, 16> &b)
{
  return _mm_xor_si128(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 16> bit_xor(const Vec<Float, 16> &a,
                                          const Vec<Float, 16> &b)
{
  return _mm_xor_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 16> bit_xor(const Vec<Double, 16> &a,
                                           const Vec<Double, 16> &b)
{
  return _mm_xor_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_not
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 16> bit_not(const Vec<T, 16> &a)
{
  // there is no _mm_not_si128, so xor with all 1's
  return _mm_xor_si128(a, _mm_set1_epi32(-1));
}

// float version
static SIMD_INLINE Vec<Float, 16> bit_not(const Vec<Float, 16> &a)
{
  // there is no _mm_not_ps, so xor with all 1's
  return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1)));
}

// double version
static SIMD_INLINE Vec<Double, 16> bit_not(const Vec<Double, 16> &a)
{
  // there is no _mm_not_pd, so xor with all 1's
  return _mm_xor_pd(a, _mm_castsi128_pd(_mm_set1_epi32(-1)));
}

// ---------------------------------------------------------------------------
// avg: average with rounding up
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 16> avg(const Vec<Byte, 16> &a,
                                     const Vec<Byte, 16> &b)
{
  return _mm_avg_epu8(a, b);
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<SignedByte, 16> avg(const Vec<SignedByte, 16> &a,
                                           const Vec<SignedByte, 16> &b)
{
  // from Agner Fog's VCL vectori128.h
  __m128i signbit = _mm_set1_epi32(0x80808080);
  __m128i a1      = _mm_xor_si128(a, signbit); // add 0x80
  __m128i b1      = _mm_xor_si128(b, signbit); // add 0x80
  __m128i m1      = _mm_avg_epu8(a1, b1);      // unsigned avg
  return _mm_xor_si128(m1, signbit);           // sub 0x80
}

static SIMD_INLINE Vec<Word, 16> avg(const Vec<Word, 16> &a,
                                     const Vec<Word, 16> &b)
{
  return _mm_avg_epu16(a, b);
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<Short, 16> avg(const Vec<Short, 16> &a,
                                      const Vec<Short, 16> &b)
{
  // from Agner Fog's VCL vectori128.h
  __m128i signbit = _mm_set1_epi32(0x80008000);
  __m128i a1      = _mm_xor_si128(a, signbit); // add 0x8000
  __m128i b1      = _mm_xor_si128(b, signbit); // add 0x8000
  __m128i m1      = _mm_avg_epu16(a1, b1);     // unsigned avg
  return _mm_xor_si128(m1, signbit);           // sub 0x8000
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<Int, 16> avg(const Vec<Int, 16> &a,
                                    const Vec<Int, 16> &b)
{
  Vec<Int, 16> one = set1(Int(1), Integer<16>()), as, bs, lsb;
  lsb              = bit_and(bit_or(a, b), one);
  as               = srai<1>(a);
  bs               = srai<1>(b);
  return add(lsb, add(as, bs));
}

// NOTE: Float version doesn't round!
static SIMD_INLINE Vec<Float, 16> avg(const Vec<Float, 16> &a,
                                      const Vec<Float, 16> &b)
{
  __m128 half = _mm_set1_ps(0.5f);
  return _mm_mul_ps(_mm_add_ps(a, b), half);
}

// NOTE: Double version doesn't round!
static SIMD_INLINE Vec<Double, 16> avg(const Vec<Double, 16> &a,
                                       const Vec<Double, 16> &b)
{
  __m128d half = _mm_set1_pd(0.5);
  return _mm_mul_pd(_mm_add_pd(a, b), half);
}

// ---------------------------------------------------------------------------
// test_all_zeros
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE bool test_all_zeros(const Vec<T, 16> &a)
{
  // reinterpret to int in case T is float
  const auto intA = reinterpret(a, OutputType<Int>());
#ifdef __SSE4_1__
  // 10. Oct 22 (Jonas Keller):
  // replaced unnecessary "_mm_cmpeq_epi8(a, a)" with "a"
  // return _mm_test_all_zeros(a, _mm_cmpeq_epi8(a, a));
  return _mm_test_all_zeros(intA, intA);
#else
  return (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_setzero_si128(), intA)) ==
          0xffff);
#endif
}

// ---------------------------------------------------------------------------
// test_all_ones
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE bool test_all_ones(const Vec<T, 16> &a)
{
  // reinterpret to int in case T is float
  const auto intA = reinterpret(a, OutputType<Int>());
#ifdef __SSE4_1__
  return _mm_test_all_ones(intA);
#else
  __m128i undef = _mm_undefined_si128();
  __m128i ones  = _mm_cmpeq_epi8(undef, undef);
  return _mm_movemask_epi8(_mm_cmpeq_epi8(ones, intA)) == 0xffff;
#endif
}

// ---------------------------------------------------------------------------
// reverse
// ---------------------------------------------------------------------------

// All reverse operations below are courtesy of Yannick Sander.
// modified

static SIMD_INLINE Vec<Byte, 16> reverse(const Vec<Byte, 16> &a)
{
  const __m128i mask =
    _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  // supported by sse3 and higher
  // compat available
  // no faster instruction available in newer instruction sets
  return _mm_shuffle_epi8(a, mask);
}

static SIMD_INLINE Vec<SignedByte, 16> reverse(const Vec<SignedByte, 16> &a)
{
  const __m128i mask =
    _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  return _mm_shuffle_epi8(a, mask);
}

static SIMD_INLINE Vec<Short, 16> reverse(const Vec<Short, 16> &a)
{
  const __m128i mask =
    _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

  return _mm_shuffle_epi8(a, mask);
}

static SIMD_INLINE Vec<Word, 16> reverse(const Vec<Word, 16> &a)
{
  const __m128i mask =
    _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

  return _mm_shuffle_epi8(a, mask);
}

static SIMD_INLINE Vec<Int, 16> reverse(const Vec<Int, 16> &a)
{
  return _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 1, 2, 3));
}

static SIMD_INLINE Vec<Float, 16> reverse(const Vec<Float, 16> &a)
{
  return _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3));
}

static SIMD_INLINE Vec<Double, 16> reverse(const Vec<Double, 16> &a)
{
  return _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1));
}

// ---------------------------------------------------------------------------
// msb2int
// ---------------------------------------------------------------------------

// 26. Aug 22 (Jonas Keller): added msb2int functions

static SIMD_INLINE uint64_t msb2int(const Vec<Byte, 16> &a)
{
  return _mm_movemask_epi8(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<SignedByte, 16> &a)
{
  return _mm_movemask_epi8(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Short, 16> &a)
{
  // move the upper bytes of the 8 shorts to the lower 8 bytes of the vector
  // and set the upper 8 bytes of to 0, so that _mm_movemask_epi8
  // can be used to extract the upper bit of each short
  const __m128i mask =
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
  const __m128i shuffled = _mm_shuffle_epi8(a, mask);
  return _mm_movemask_epi8(shuffled);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Word, 16> &a)
{
  // move the upper bytes of the 8 words to the lower 8 bytes of the vector
  // and set the upper 8 bytes of to 0, so that _mm_movemask_epi8
  // can be used to extract the upper bit of each word
  const __m128i mask =
    _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
  const __m128i shuffled = _mm_shuffle_epi8(a, mask);
  return _mm_movemask_epi8(shuffled);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Int, 16> &a)
{
  return _mm_movemask_ps(_mm_castsi128_ps(a));
}

static SIMD_INLINE uint64_t msb2int(const Vec<Float, 16> &a)
{
  return _mm_movemask_ps(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Double, 16> &a)
{
  return _mm_movemask_pd(a);
}

// ---------------------------------------------------------------------------
// int2msb
// ---------------------------------------------------------------------------

// 06. Oct 22 (Jonas Keller): added int2msb functions

static SIMD_INLINE Vec<Byte, 16> int2msb(const uint64_t a, OutputType<Byte>,
                                         Integer<16>)
{
  // ssse3 version from https://stackoverflow.com/a/72899629
#ifdef __SSSE3__
  __m128i shuffleIndeces = _mm_set_epi64x(0x0101010101010101, 0);
  __m128i aVec = _mm_shuffle_epi8(_mm_cvtsi32_si128(a), shuffleIndeces);
#else
  __m128i maskLo = _mm_set_epi64x(0, 0xffffffffffffffff);
  __m128i aLo    = _mm_and_si128(maskLo, _mm_set1_epi8(a));
  __m128i aHi    = _mm_andnot_si128(maskLo, _mm_set1_epi8(a >> 8));
  __m128i aVec   = _mm_or_si128(aLo, aHi);
#endif
  __m128i sel      = _mm_set1_epi64x(0x8040201008040201);
  __m128i selected = _mm_and_si128(aVec, sel);
  __m128i result   = _mm_cmpeq_epi8(selected, sel);
  return _mm_and_si128(result, _mm_set1_epi8((int8_t) 0x80));
}

static SIMD_INLINE Vec<SignedByte, 16> int2msb(const uint64_t a,
                                               OutputType<SignedByte>,
                                               Integer<16>)
{
  return reinterpret(int2msb(a, OutputType<Byte>(), Integer<16>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Short, 16> int2msb(const uint64_t a, OutputType<Short>,
                                          Integer<16>)
{
  __m128i aVec = _mm_set1_epi16(a);
  __m128i sel  = _mm_set_epi16(0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004,
                               0x0002, 0x0001);
  __m128i selected = _mm_and_si128(aVec, sel);
  __m128i result   = _mm_cmpeq_epi16(selected, sel);
  return _mm_and_si128(result, _mm_set1_epi16((int16_t) 0x8000));
}

static SIMD_INLINE Vec<Word, 16> int2msb(const uint64_t a, OutputType<Word>,
                                         Integer<16>)
{
  return reinterpret(int2msb(a, OutputType<Short>(), Integer<16>()),
                     OutputType<Word>());
}

static SIMD_INLINE Vec<Int, 16> int2msb(const uint64_t a, OutputType<Int>,
                                        Integer<16>)
{
  __m128i aVec = _mm_set1_epi32(a);
  __m128i sel  = _mm_set_epi32(0x00000008, 0x00000004, 0x00000002, 0x00000001);
  __m128i selected = _mm_and_si128(aVec, sel);
  __m128i result   = _mm_cmpeq_epi32(selected, sel);
  return _mm_and_si128(result, _mm_set1_epi32(0x80000000));
}

static SIMD_INLINE Vec<Float, 16> int2msb(const uint64_t a, OutputType<Float>,
                                          Integer<16>)
{
  return reinterpret(int2msb(a, OutputType<Int>(), Integer<16>()),
                     OutputType<Float>());
}

static SIMD_INLINE Vec<Double, 16> int2msb(const uint64_t a, OutputType<Double>,
                                           Integer<16>)
{
  return _mm_set_pd((a & 2) ? -0.0 : 0.0, (a & 1) ? -0.0 : 0.0);
}

// ---------------------------------------------------------------------------
// int2bits
// ---------------------------------------------------------------------------

// 09. Oct 22 (Jonas Keller): added int2bits functions

static SIMD_INLINE Vec<Byte, 16> int2bits(const uint64_t a, OutputType<Byte>,
                                          Integer<16>)
{
  // ssse3 version from https://stackoverflow.com/a/72899629
#ifdef __SSSE3__
  __m128i shuffleIndeces = _mm_set_epi64x(0x0101010101010101, 0);
  __m128i aVec = _mm_shuffle_epi8(_mm_cvtsi32_si128(a), shuffleIndeces);
#else
  __m128i maskLo = _mm_set_epi64x(0, 0xffffffffffffffff);
  __m128i aLo    = _mm_and_si128(maskLo, _mm_set1_epi8(a));
  __m128i aHi    = _mm_andnot_si128(maskLo, _mm_set1_epi8(a >> 8));
  __m128i aVec   = _mm_or_si128(aLo, aHi);
#endif
  __m128i sel      = _mm_set1_epi64x(0x8040201008040201);
  __m128i selected = _mm_and_si128(aVec, sel);
  return _mm_cmpeq_epi8(selected, sel);
}

static SIMD_INLINE Vec<SignedByte, 16> int2bits(const uint64_t a,
                                                OutputType<SignedByte>,
                                                Integer<16>)
{
  return reinterpret(int2bits(a, OutputType<Byte>(), Integer<16>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Short, 16> int2bits(const uint64_t a, OutputType<Short>,
                                           Integer<16>)
{
  __m128i aVec = _mm_set1_epi16(a);
  __m128i sel  = _mm_set_epi16(0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004,
                               0x0002, 0x0001);
  __m128i selected = _mm_and_si128(aVec, sel);
  return _mm_cmpeq_epi16(selected, sel);
}

static SIMD_INLINE Vec<Word, 16> int2bits(const uint64_t a, OutputType<Word>,
                                          Integer<16>)
{
  return reinterpret(int2bits(a, OutputType<Short>(), Integer<16>()),
                     OutputType<Word>());
}

static SIMD_INLINE Vec<Int, 16> int2bits(const uint64_t a, OutputType<Int>,
                                         Integer<16>)
{
  __m128i aVec = _mm_set1_epi32(a);
  __m128i sel  = _mm_set_epi32(0x00000008, 0x00000004, 0x00000002, 0x00000001);
  __m128i selected = _mm_and_si128(aVec, sel);
  return _mm_cmpeq_epi32(selected, sel);
}

static SIMD_INLINE Vec<Float, 16> int2bits(const uint64_t a, OutputType<Float>,
                                           Integer<16>)
{
  return reinterpret(int2bits(a, OutputType<Int>(), Integer<16>()),
                     OutputType<Float>());
}

static SIMD_INLINE Vec<Double, 16> int2bits(const uint64_t a,
                                            OutputType<Double>, Integer<16>)
{
  const auto trueVal = TypeInfo<Double>::trueval();
  return _mm_set_pd((a & 2) ? trueVal : 0.0, (a & 1) ? trueVal : 0.0);
}

// ---------------------------------------------------------------------------
// iota
// ---------------------------------------------------------------------------

// 30. Jan 23 (Jonas Keller): added iota

static SIMD_INLINE Vec<Byte, 16> iota(OutputType<Byte>, Integer<16>)
{
  return _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<SignedByte, 16> iota(OutputType<SignedByte>, Integer<16>)
{
  return _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Short, 16> iota(OutputType<Short>, Integer<16>)
{
  return _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Word, 16> iota(OutputType<Word>, Integer<16>)
{
  return _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Int, 16> iota(OutputType<Int>, Integer<16>)
{
  return _mm_set_epi32(3, 2, 1, 0);
}

static SIMD_INLINE Vec<Float, 16> iota(OutputType<Float>, Integer<16>)
{
  return _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
}

static SIMD_INLINE Vec<Double, 16> iota(OutputType<Double>, Integer<16>)
{
  return _mm_set_pd(1.0, 0.0);
}

} // namespace base
} // namespace internal
} // namespace simd

#endif // __SSE2__

#endif // SIMD_VEC_BASE_IMPL_INTEL_16_H_
