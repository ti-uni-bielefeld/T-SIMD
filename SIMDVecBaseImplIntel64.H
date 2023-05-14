// ===========================================================================
//
// SIMDVecBaseImplIntel64.H --
// encapsulation for AVX512 Intel vector extensions
// inspired by Agner Fog's C++ Vector Class Library
// http://www.agner.org/optimize/#vectorclass
// (VCL License: GNU General Public License Version 3,
//  http://www.gnu.org/licenses/gpl-3.0.en.html)
//
// Changes to unpack, zip and unzip functions in 2022 by
//   Jan-Lukas Wolf (jawolf@techfak.uni-bielefeld.de)
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

#ifndef SIMD_VEC_BASE_IMPL_INTEL_64_H_
#define SIMD_VEC_BASE_IMPL_INTEL_64_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBaseImplIntel16.H"
#include "SIMDVecBaseImplIntel32.H"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#ifdef __AVX512F__

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
// ===========================================================================

// ===========================================================================
// Vec integer specialization for AVX512 v
// ===========================================================================

// partial specialization for SIMD_WIDTH = 64
template <typename T>
class Vec<T, 64>
{
  __m512i zmm = _mm512_setzero_si512();

public:
  using Type                    = T;
  static constexpr int elements = 64 / sizeof(T);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 64;

  Vec() = default;
  Vec(const __m512i &x) { zmm = x; }
  Vec &operator=(const __m512i &x)
  {
    zmm = x;
    return *this;
  }
  operator __m512i() const { return zmm; }
  // for avx512bw emulation
  Vec(const Vec<T, 32> &lo, const Vec<T, 32> &hi)
  {
    zmm = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
  SIMD_INLINE Vec<T, 32> lo() const { return _mm512_castsi512_si256(zmm); }
  SIMD_INLINE Vec<T, 32> hi() const
  {
    return _mm512_extracti64x4_epi64(zmm, 1);
  }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
};

// ===========================================================================
// Vec float specialization for AVX512 v
// ===========================================================================

template <>
class Vec<Float, 64>
{
  __m512 zmm = _mm512_setzero_ps();

public:
  using Type                    = Float;
  static constexpr int elements = 64 / sizeof(Float);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 64;

  Vec() = default;
  Vec(const __m512 &x) { zmm = x; }
  Vec &operator=(const __m512 &x)
  {
    zmm = x;
    return *this;
  }
  operator __m512() const { return zmm; }
  // for avx512bw emulation
  Vec(const Vec<Float, 32> &lo, const Vec<Float, 32> &hi)
  {
    zmm = _mm512_castpd_ps(_mm512_insertf64x4(
      _mm512_castps_pd(_mm512_castps256_ps512(lo)), _mm256_castps_pd(hi), 1));
  }
  SIMD_INLINE Vec<Float, 32> lo() const { return _mm512_castps512_ps256(zmm); }
  // _mm512_extractf32x8_ps only in AVX512DQ
  SIMD_INLINE Vec<Float, 32> hi() const
  {
    return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(zmm), 1));
  }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
};

// ===========================================================================
// Vec double specialization for AVX512 v
// ===========================================================================

template <>
class Vec<Double, 64>
{
  __m512d zmm;

public:
  using Type                    = Double;
  static constexpr int elements = 64 / sizeof(Double);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 64;

  Vec() = default;
  Vec(const __m512d &x) { zmm = x; }
  Vec &operator=(const __m512d &x)
  {
    zmm = x;
    return *this;
  }
  operator __m512d() const { return zmm; }
  // for avx512bw emulation
  Vec(const Vec<Double, 32> &lo, const Vec<Double, 32> &hi)
  {
    zmm = _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1);
  }
  SIMD_INLINE Vec<Double, 32> lo() const { return _mm512_castpd512_pd256(zmm); }
  SIMD_INLINE Vec<Double, 32> hi() const
  {
    return _mm512_extractf64x4_pd(zmm, 1);
  }
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

// These functions either wrap  intrinsics (e.g. to handle
// immediate arguments as template parameter), or switch between
// implementations with different SSE* extensions, or provide
// altered or additional functionality.
// Only for use in wrapper functions!

// 01. Apr 23 (Jonas Keller): removed some not really necessary internal
// wrapper functions and inlined them directly into where they were used

// ---------------------------------------------------------------------------
// element-wise shifts v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_slli_epi16(__m512i a,
                                              IsPosInRange<true, true>)
{
  return _mm512_slli_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m512i x_mm512_slli_epi16(__m512i,
                                              IsPosInRange<true, false>)
{
  return _mm512_setzero_si512();
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_slli_epi16(__m512i a)
{
  return x_mm512_slli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// ----------

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srli_epi16(__m512i a,
                                              IsPosInRange<true, true>)
{
  return _mm512_srli_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srli_epi16(__m512i,
                                              IsPosInRange<true, false>)
{
  return _mm512_setzero_si512();
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srli_epi16(__m512i a)
{
  return x_mm512_srli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// ----------

// 16. Oct 22 (Jonas Keller): added x_mm512_srai_epi8

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi8(__m512i a,
                                             IsPosInRange<true, true>)
{
  __m512i odd    = _mm512_srai_epi16(a, IMM);
  __m512i even   = _mm512_srai_epi16(_mm512_slli_epi16(a, 8), IMM + 8);
  __mmask64 mask = __mmask64(0x5555555555555555);
  return _mm512_mask_blend_epi8(mask, odd, even);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi8(__m512i a,
                                             IsPosInRange<true, false>)
{
  // result should be all ones if a is negative, all zeros otherwise
  return _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi8(__m512i a)
{
  return x_mm512_srai_epi8<IMM>(a, IsPosInGivenRange<7, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi16(__m512i a,
                                              IsPosInRange<true, true>)
{
  return _mm512_srai_epi16(a, IMM);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi16(__m512i a,
                                              IsPosInRange<true, false>)
{
  return _mm512_srai_epi16(a, 15);
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi16(__m512i a)
{
  return x_mm512_srai_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

#endif

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_slli_epi32(__m512i a,
                                              IsPosInRange<true, true>)
{
  return _mm512_slli_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m512i x_mm512_slli_epi32(__m512i,
                                              IsPosInRange<true, false>)
{
  return _mm512_setzero_si512();
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_slli_epi32(__m512i a)
{
  return x_mm512_slli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// ----------

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srli_epi32(__m512i a,
                                              IsPosInRange<true, true>)
{
  return _mm512_srli_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srli_epi32(__m512i,
                                              IsPosInRange<true, false>)
{
  return _mm512_setzero_si512();
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srli_epi32(__m512i a)
{
  return x_mm512_srli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// ----------

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi32(__m512i a,
                                              IsPosInRange<true, true>)
{
  return _mm512_srai_epi32(a, IMM);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi32(__m512i a,
                                              IsPosInRange<true, false>)
{
  return _mm512_srai_epi32(a, 31);
}

// hub
template <int IMM>
static SIMD_INLINE __m512i x_mm512_srai_epi32(__m512i a)
{
  return x_mm512_srai_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// ---------------------------------------------------------------------------
// extract 128-bit-lane as Vec<T, 16>
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// generalized extract of 128-bit-lanes (IMM = 0..3)
// all integer versions
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 16> extractLane(const Vec<T, 64> &a)
{
  return _mm512_extracti32x4_epi32(a, IMM);
}

// generalized extract of 128-bit-lanes (IMM = 0..3)
// float versions
template <int IMM>
static SIMD_INLINE Vec<Float, 16> extractLane(const Vec<Float, 64> &a)
{
  return _mm512_extractf32x4_ps(a, IMM);
}

// generalized extract of 128-bit-lanes (IMM = 0..3)
// double versions
template <int IMM>
static SIMD_INLINE Vec<Double, 16> extractLane(const Vec<Double, 64> &a)
{
#ifdef __AVX512DQ__
  return _mm512_extractf64x2_pd(a, IMM);
#else
  return _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a), IMM));
#endif
}

// ---------------------------------------------------------------------------
// extract v
// ---------------------------------------------------------------------------

// _mm512_extract_epi/_ps etc. don't exist, we emulate that
// NOTE: even in AVX, that was internally done in the extract intrinsics:
// e.g. avxintrin.h:
// int _mm256_extract_epi16 (__m256i __X, int const __N)
// {
//   __m128i __Y = _mm256_extractf128_si256 (__X, __N >> 3);
//   return _mm_extract_epi16 (__Y, __N % 8);
// }

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi16(__m512i a,
                                             IsPosInRange<true, true>)
{
  return x_mm_extract_epi16<(IMM % 8)>(_mm512_extracti32x4_epi32(a, IMM >> 3));
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi16(__m512i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi16(__m512i a)
{
  // 32 * epi16
  return x_mm512_extract_epi16<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// ----------

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi8(__m512i a, IsPosInRange<true, true>)
{
  return _mm_extract_epi8(_mm512_extracti32x4_epi32(a, IMM >> 4), IMM % 16);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi8(__m512i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi8(__m512i a)
{
  // 64 * epi8
  return x_mm512_extract_epi8<IMM>(a, IsPosInGivenRange<64, IMM>());
}

// ----------

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi32(__m512i a,
                                             IsPosInRange<true, true>)
{
  // TODO: extract: is conversion from return type int to Int always safe?
  return _mm_extract_epi32(_mm512_extracti32x4_epi32(a, IMM >> 2), IMM % 4);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi32(__m512i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm512_extract_epi32(__m512i a)
{
  // 16 * epi32
  return x_mm512_extract_epi32<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// ----------

// we stick to the weird definition of returning an int instead of float

template <int IMM>
static SIMD_INLINE int x_mm512_extract_ps(__m512 a, IsPosInRange<true, true>)
{
  // TODO: extract: is conversion from return type int to Int always safe?
  return _mm_extract_ps(_mm512_extractf32x4_ps(a, IMM >> 2), IMM % 4);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm512_extract_ps(__m512, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm512_extract_ps(__m512 a)
{
  // 16 * epi32
  return x_mm512_extract_ps<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE long long x_mm512_extract_epi64(__m512i a,
                                                   IsPosInRange<true, true>)
{
  return _mm_extract_epi64(_mm512_extracti32x4_epi32(a, (IMM >> 1)), IMM % 2);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE long long x_mm512_extract_epi64(__m512i,
                                                   IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE long long x_mm512_extract_epi64(__m512i a)
{
  // 8 * epi64
  return x_mm512_extract_epi64<IMM>(a, IsPosInGivenRange<8, IMM>());
}

// ---------------------------------------------------------------------------
// alignr v
// ---------------------------------------------------------------------------

// 21. Apr 23 (Jonas Keller): replaced IMM range handling via tag dispatch
// with static_assert, since we don't need the range handling anymore,
// we just assert that IMM is in range

#ifdef __AVX512BW__

template <int IMM>
static SIMD_INLINE __m512i x_mm512_alignr_epi8(__m512i h, __m512i l)
{
  static_assert(IMM >= 0 && IMM < 32, "");
  return _mm512_alignr_epi8(h, l, IMM);
}

#else

// non-avx512bw workarounds
// (easy since AVX512BW instructions operate on lanes anyhow)

template <int IMM>
static SIMD_INLINE __m512i x_mm512_alignr_epi8(__m512i h, __m512i l)
{
  static_assert(IMM >= 0 && IMM < 32, "");
  const __m256i lo = _mm256_alignr_epi8(_mm512_castsi512_si256(h),
                                        _mm512_castsi512_si256(l), IMM);
  const __m256i hi = _mm256_alignr_epi8(_mm512_extracti64x4_epi64(h, 1),
                                        _mm512_extracti64x4_epi64(l, 1), IMM);
  return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

#endif

// ---------------------------------------------------------------------------
// insert 16 byte vector a all 4 lanes of a 64 byte vector v
// ---------------------------------------------------------------------------

static SIMD_INLINE __m512i x_mm512_duplicate_si128(__m128i a)
{
  return _mm512_broadcast_i32x4(a);
}

// ---------------------------------------------------------------------------
// transpose8x64 v
// ---------------------------------------------------------------------------

static SIMD_INLINE __m512i x_mm512_transpose8x64_epi64(__m512i a)
{
  return _mm512_permutexvar_epi64(_mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0), a);
}

// ---------------------------------------------------------------------------
// evenodd8x64 v
// ---------------------------------------------------------------------------

static SIMD_INLINE __m512i x_mm512_evenodd8x64_epi64(__m512i a)
{
  return _mm512_permutexvar_epi64(_mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0), a);
}

// ---------------------------------------------------------------------------
// unpack of 2 ps v
// ---------------------------------------------------------------------------

static SIMD_INLINE __m512 x_mm512_unpacklo_2ps(__m512 a, __m512 b)
{
  return _mm512_castpd_ps(
    _mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

static SIMD_INLINE __m512 x_mm512_unpackhi_2ps(__m512 a, __m512 b)
{
  return _mm512_castpd_ps(
    _mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

// ---------------------------------------------------------------------------
// binary functions with non-avx512bw workarounds v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
// avx512bw is available
#define SIMD_X_BW_INT_BINFCT_64(INTRIN)                                        \
  static SIMD_INLINE __m512i x_mm512_##INTRIN(__m512i a, __m512i b)            \
  {                                                                            \
    return _mm512_##INTRIN(a, b);                                              \
  }
#else
// non-avx512bw workaround
#define SIMD_X_BW_INT_BINFCT_64(INTRIN)                                        \
  static SIMD_INLINE __m512i x_mm512_##INTRIN(__m512i a, __m512i b)            \
  {                                                                            \
    __m256i lo =                                                               \
      _mm256_##INTRIN(_mm512_castsi512_si256(a), _mm512_castsi512_si256(b));   \
    __m256i hi = _mm256_##INTRIN(_mm512_extracti64x4_epi64(a, 1),              \
                                 _mm512_extracti64x4_epi64(b, 1));             \
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);              \
  }
#endif

SIMD_X_BW_INT_BINFCT_64(unpacklo_epi8)
SIMD_X_BW_INT_BINFCT_64(unpackhi_epi8)
SIMD_X_BW_INT_BINFCT_64(unpacklo_epi16)
SIMD_X_BW_INT_BINFCT_64(unpackhi_epi16)
SIMD_X_BW_INT_BINFCT_64(shuffle_epi8)
SIMD_X_BW_INT_BINFCT_64(packs_epi16)
SIMD_X_BW_INT_BINFCT_64(packs_epi32)
SIMD_X_BW_INT_BINFCT_64(packus_epi16)
SIMD_X_BW_INT_BINFCT_64(packus_epi32)

// ---------------------------------------------------------------------------
// non-existing avx512 functions emulated via avx v
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// x_mm512_movm_epi32 v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

static SIMD_INLINE __m512i x_mm512_movm_epi32(__mmask16 k)
{
#ifdef __AVX512DQ__
  return _mm512_movm_epi32(k);
#else
  return _mm512_maskz_mov_epi32(k, _mm512_set1_epi32(-1));
#endif
}

// ---------------------------------------------------------------------------
// x_mm512_movm_epi64 v
// ---------------------------------------------------------------------------

static SIMD_INLINE __m512i x_mm512_movm_epi64(__mmask8 k)
{
#ifdef __AVX512DQ__
  return _mm512_movm_epi64(k);
#else
  return _mm512_maskz_mov_epi64(k, _mm512_set1_epi64(-1));
#endif
}

// ---------------------------------------------------------------------------
// x_mm512_set_epi8, x_mm512_set_epi16 v
// ---------------------------------------------------------------------------

// 14. May 23 (Jonas Keller): added x_mm512_set_epi8 and x_mm512_set_epi16

// _mm512_set_epi8 and _mm512_set_epi16 are not available in some gcc versions
// so we use _mm512_set_epi32 to emulate them

static SIMD_INLINE __m512i x_mm512_set_epi8(
  char b63, char b62, char b61, char b60, char b59, char b58, char b57,
  char b56, char b55, char b54, char b53, char b52, char b51, char b50,
  char b49, char b48, char b47, char b46, char b45, char b44, char b43,
  char b42, char b41, char b40, char b39, char b38, char b37, char b36,
  char b35, char b34, char b33, char b32, char b31, char b30, char b29,
  char b28, char b27, char b26, char b25, char b24, char b23, char b22,
  char b21, char b20, char b19, char b18, char b17, char b16, char b15,
  char b14, char b13, char b12, char b11, char b10, char b9, char b8, char b7,
  char b6, char b5, char b4, char b3, char b2, char b1, char b0)
{
  return _mm512_set_epi32(
    (int) b63 << 24 | (int) b62 << 16 | (int) b61 << 8 | (int) b60,
    (int) b59 << 24 | (int) b58 << 16 | (int) b57 << 8 | (int) b56,
    (int) b55 << 24 | (int) b54 << 16 | (int) b53 << 8 | (int) b52,
    (int) b51 << 24 | (int) b50 << 16 | (int) b49 << 8 | (int) b48,
    (int) b47 << 24 | (int) b46 << 16 | (int) b45 << 8 | (int) b44,
    (int) b43 << 24 | (int) b42 << 16 | (int) b41 << 8 | (int) b40,
    (int) b39 << 24 | (int) b38 << 16 | (int) b37 << 8 | (int) b36,
    (int) b35 << 24 | (int) b34 << 16 | (int) b33 << 8 | (int) b32,
    (int) b31 << 24 | (int) b30 << 16 | (int) b29 << 8 | (int) b28,
    (int) b27 << 24 | (int) b26 << 16 | (int) b25 << 8 | (int) b24,
    (int) b23 << 24 | (int) b22 << 16 | (int) b21 << 8 | (int) b20,
    (int) b19 << 24 | (int) b18 << 16 | (int) b17 << 8 | (int) b16,
    (int) b15 << 24 | (int) b14 << 16 | (int) b13 << 8 | (int) b12,
    (int) b11 << 24 | (int) b10 << 16 | (int) b9 << 8 | (int) b8,
    (int) b7 << 24 | (int) b6 << 16 | (int) b5 << 8 | (int) b4,
    (int) b3 << 24 | (int) b2 << 16 | (int) b1 << 8 | (int) b0);
}

static SIMD_INLINE __m512i x_mm512_set_epi16(
  short w31, short w30, short w29, short w28, short w27, short w26, short w25,
  short w24, short w23, short w22, short w21, short w20, short w19, short w18,
  short w17, short w16, short w15, short w14, short w13, short w12, short w11,
  short w10, short w9, short w8, short w7, short w6, short w5, short w4,
  short w3, short w2, short w1, short w0)
{
  return _mm512_set_epi32(
    (int) w31 << 16 | (int) w30, (int) w29 << 16 | (int) w28,
    (int) w27 << 16 | (int) w26, (int) w25 << 16 | (int) w24,
    (int) w23 << 16 | (int) w22, (int) w21 << 16 | (int) w20,
    (int) w19 << 16 | (int) w18, (int) w17 << 16 | (int) w16,
    (int) w15 << 16 | (int) w14, (int) w13 << 16 | (int) w12,
    (int) w11 << 16 | (int) w10, (int) w9 << 16 | (int) w8,
    (int) w7 << 16 | (int) w6, (int) w5 << 16 | (int) w4,
    (int) w3 << 16 | (int) w2, (int) w1 << 16 | (int) w0);
}

// ###########################################################################
// ###########################################################################
// ###########################################################################

// ===========================================================================
// Vec template function specializations or overloading for AVX
// ===========================================================================

// ---------------------------------------------------------------------------
// reinterpretation casts v
// ---------------------------------------------------------------------------

// 08. Apr 23 (Jonas Keller): used enable_if for cleaner implementation

// between all integer types
template <typename Tdst, typename Tsrc,
          SIMD_ENABLE_IF((!std::is_same<Tdst, Tsrc>::value &&
                          std::is_integral<Tdst>::value &&
                          std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Tdst, 64> reinterpret(const Vec<Tsrc, 64> &vec,
                                             OutputType<Tdst>)
{
  // 26. Nov 22 (Jonas Keller): reinterpret_cast is technically undefined
  // behavior, so just rewrapping the vector register in a new Vec instead
  // return reinterpret_cast<const Vec<Tdst,64>&>(vec);
  return Vec<Tdst, 64>(__m512i(vec));
}

// from float to any integer type
template <typename Tdst, SIMD_ENABLE_IF((std::is_integral<Tdst>::value))>
static SIMD_INLINE Vec<Tdst, 64> reinterpret(const Vec<Float, 64> &vec,
                                             OutputType<Tdst>)
{
  return _mm512_castps_si512(vec);
}

// from any integer type to float
template <typename Tsrc, SIMD_ENABLE_IF((std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Float, 64> reinterpret(const Vec<Tsrc, 64> &vec,
                                              OutputType<Float>)
{
  return _mm512_castsi512_ps(vec);
}

// from double to any integer type
template <typename Tdst, SIMD_ENABLE_IF((std::is_integral<Tdst>::value))>
static SIMD_INLINE Vec<Tdst, 64> reinterpret(const Vec<Double, 64> &vec,
                                             OutputType<Tdst>)
{
  return _mm512_castpd_si512(vec);
}

// from any integer type to double
template <typename Tsrc, SIMD_ENABLE_IF((std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Double, 64> reinterpret(const Vec<Tsrc, 64> &vec,
                                               OutputType<Double>)
{
  return _mm512_castsi512_pd(vec);
}

// from float to double
static SIMD_INLINE Vec<Double, 64> reinterpret(const Vec<Float, 64> &vec,
                                               OutputType<Double>)
{
  return _mm512_castps_pd(vec);
}

// from double to float
static SIMD_INLINE Vec<Float, 64> reinterpret(const Vec<Double, 64> &vec,
                                              OutputType<Float>)
{
  return _mm512_castpd_ps(vec);
}

// between identical types
template <typename T>
static SIMD_INLINE Vec<T, 64> reinterpret(const Vec<T, 64> &vec, OutputType<T>)
{
  return vec;
}

// ---------------------------------------------------------------------------
// convert (without changes in the number of of elements) v
// ---------------------------------------------------------------------------

// conversion with saturation; we wanted to have a fast solution that
// doesn't trigger the overflow which results in a negative two's
// complement result ("invalid int32": 0x80000000); therefore we clamp
// the positive values at the maximal positive float which is
// convertible to int32 without overflow (0x7fffffbf = 2147483520);
// negative values cannot overflow (they are clamped to invalid int
// which is the most negative int32)
static SIMD_INLINE Vec<Int, 64> cvts(const Vec<Float, 64> &a, OutputType<Int>)
{
  // TODO: analyze much more complex solution for cvts at
  // TODO: http://stackoverflow.com/questions/9157373/
  // TODO: most-efficient-way-to-convert-vector-of-float-to-vector-of-uint32
  __m512 clip = _mm512_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
  return _mm512_cvtps_epi32(_mm512_min_ps(clip, a));
}

// saturation is not necessary in this case
static SIMD_INLINE Vec<Float, 64> cvts(const Vec<Int, 64> &a, OutputType<Float>)
{
  return _mm512_cvtepi32_ps(a);
}

// ---------------------------------------------------------------------------
// setzero v
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 64> setzero(OutputType<Byte>, Integer<64>)
{
  return _mm512_setzero_si512();
}

static SIMD_INLINE Vec<SignedByte, 64> setzero(OutputType<SignedByte>,
                                               Integer<64>)
{
  return _mm512_setzero_si512();
}

static SIMD_INLINE Vec<Word, 64> setzero(OutputType<Word>, Integer<64>)
{
  return _mm512_setzero_si512();
}

static SIMD_INLINE Vec<Short, 64> setzero(OutputType<Short>, Integer<64>)
{
  return _mm512_setzero_si512();
}

static SIMD_INLINE Vec<Int, 64> setzero(OutputType<Int>, Integer<64>)
{
  return _mm512_setzero_si512();
}

static SIMD_INLINE Vec<Float, 64> setzero(OutputType<Float>, Integer<64>)
{
  return _mm512_setzero_ps();
}

static SIMD_INLINE Vec<Double, 64> setzero(OutputType<Double>, Integer<64>)
{
  return _mm512_setzero_pd();
}

// ---------------------------------------------------------------------------
// set1 v
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 64> set1(Byte a, Integer<64>)
{
  return _mm512_set1_epi8(a);
}

static SIMD_INLINE Vec<SignedByte, 64> set1(SignedByte a, Integer<64>)
{
  return _mm512_set1_epi8(a);
}

static SIMD_INLINE Vec<Word, 64> set1(Word a, Integer<64>)
{
  return _mm512_set1_epi16(a);
}

static SIMD_INLINE Vec<Short, 64> set1(Short a, Integer<64>)
{
  return _mm512_set1_epi16(a);
}

static SIMD_INLINE Vec<Int, 64> set1(Int a, Integer<64>)
{
  return _mm512_set1_epi32(a);
}

static SIMD_INLINE Vec<Float, 64> set1(Float a, Integer<64>)
{
  return _mm512_set1_ps(a);
}

static SIMD_INLINE Vec<Double, 64> set1(Double a, Integer<64>)
{
  return _mm512_set1_pd(a);
}

// ---------------------------------------------------------------------------
// load v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> load(const T *const p, Integer<64>)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  return _mm512_load_si512((__m512i *) p);
}

static SIMD_INLINE Vec<Float, 64> load(const Float *const p, Integer<64>)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  return _mm512_load_ps(p);
}

static SIMD_INLINE Vec<Double, 64> load(const Double *const p, Integer<64>)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  return _mm512_load_pd(p);
}

// ---------------------------------------------------------------------------
// loadu v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> loadu(const T *const p, Integer<64>)
{
  return _mm512_loadu_si512((__m512i *) p);
}

static SIMD_INLINE Vec<Float, 64> loadu(const Float *const p, Integer<64>)
{
  return _mm512_loadu_ps(p);
}

static SIMD_INLINE Vec<Double, 64> loadu(const Double *const p, Integer<64>)
{
  return _mm512_loadu_pd(p);
}

// ---------------------------------------------------------------------------
// store v
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void store(T *const p, const Vec<T, 64> &a)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  _mm512_store_si512((__m512i *) p, a);
}

// float version
static SIMD_INLINE void store(Float *const p, const Vec<Float, 64> &a)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  _mm512_store_ps(p, a);
}

// double version
static SIMD_INLINE void store(Double *const p, const Vec<Double, 64> &a)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  _mm512_store_pd(p, a);
}

// ---------------------------------------------------------------------------
// storeu v
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void storeu(T *const p, const Vec<T, 64> &a)
{
  _mm512_storeu_si512((__m512i *) p, a);
}

// float version
static SIMD_INLINE void storeu(Float *const p, const Vec<Float, 64> &a)
{
  _mm512_storeu_ps(p, a);
}

// double version
static SIMD_INLINE void storeu(Double *const p, const Vec<Double, 64> &a)
{
  _mm512_storeu_pd(p, a);
}

// ---------------------------------------------------------------------------
// stream_store v
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void stream_store(T *const p, const Vec<T, 64> &a)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  _mm512_stream_si512((__m512i *) p, a);
}

// float version
static SIMD_INLINE void stream_store(Float *const p, const Vec<Float, 64> &a)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  _mm512_stream_ps(p, a);
}

// double version
static SIMD_INLINE void stream_store(Double *const p, const Vec<Double, 64> &a)
{
  // AVX load and store instructions need alignment to 64 byte
  // (lower 6 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 64);
  _mm512_stream_pd(p, a);
}

// ---------------------------------------------------------------------------
// extract v
// ---------------------------------------------------------------------------

template <int IMM>
static SIMD_INLINE Byte extract(const Vec<Byte, 64> &a)
{
  return x_mm512_extract_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE SignedByte extract(const Vec<SignedByte, 64> &a)
{
  return x_mm512_extract_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Word extract(const Vec<Word, 64> &a)
{
  return x_mm512_extract_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Short extract(const Vec<Short, 64> &a)
{
  return x_mm512_extract_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Int extract(const Vec<Int, 64> &a)
{
  // TODO: extract: is conversion from return type int to Int always safe?
  return x_mm512_extract_epi32<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Float extract(const Vec<Float, 64> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Float, x_mm512_extract_ps<IMM>(a), 0);
}

template <int IMM>
static SIMD_INLINE Double extract(const Vec<Double, 64> &a)
{
  const uint64_t resInt = x_mm512_extract_epi64<IMM>(_mm512_castpd_si512(a));
  Double res;
  memcpy(&res, &resInt, sizeof(res));
  return res;
}

// ---------------------------------------------------------------------------
// add v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> add(const Vec<Byte, 64> &a,
                                     const Vec<Byte, 64> &b)
{
  return _mm512_add_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 64> add(const Vec<SignedByte, 64> &a,
                                           const Vec<SignedByte, 64> &b)
{
  return _mm512_add_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 64> add(const Vec<Word, 64> &a,
                                     const Vec<Word, 64> &b)
{
  return _mm512_add_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 64> add(const Vec<Short, 64> &a,
                                      const Vec<Short, 64> &b)
{
  return _mm512_add_epi16(a, b);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> add(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(add(a.lo(), b.lo()), add(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> add(const Vec<Int, 64> &a,
                                    const Vec<Int, 64> &b)
{
  return _mm512_add_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 64> add(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  return _mm512_add_ps(a, b);
}

static SIMD_INLINE Vec<Double, 64> add(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  return _mm512_add_pd(a, b);
}

// ---------------------------------------------------------------------------
// adds
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> adds(const Vec<Byte, 64> &a,
                                      const Vec<Byte, 64> &b)
{
  return _mm512_adds_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 64> adds(const Vec<SignedByte, 64> &a,
                                            const Vec<SignedByte, 64> &b)
{
  return _mm512_adds_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 64> adds(const Vec<Word, 64> &a,
                                      const Vec<Word, 64> &b)
{
  return _mm512_adds_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 64> adds(const Vec<Short, 64> &a,
                                       const Vec<Short, 64> &b)
{
  return _mm512_adds_epi16(a, b);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> adds(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(adds(a.lo(), b.lo()), adds(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> adds(const Vec<Int, 64> &a,
                                     const Vec<Int, 64> &b)
{
  // 09. Mar 23 (Jonas Keller): added workaround so that this function is
  // saturated

  // _mm512_adds_epi32 does not exist, workaround:
  // Hacker's Delight, 2-13 Overflow Detection: "Signed integer overflow of
  // addition occurs if and only if the operands have the same sign and the
  // sum has a sign opposite to that of the operands."
  __m512i sum             = _mm512_add_epi32(a, b);
  __m512i opsHaveDiffSign = _mm512_xor_si512(a, b);
  __m512i sumHasDiffSign  = _mm512_xor_si512(a, sum);
  // indicates when an overflow has occurred
  __m512i overflow =
    _mm512_srai_epi32(_mm512_andnot_si512(opsHaveDiffSign, sumHasDiffSign), 31);
  // saturated sum for if overflow occurred (0x7FFFFFFF=max positive int, when
  // sign of a (and thus b as well) is 0, 0x80000000=min negative int, when sign
  // of a (and thus b as well) is 1)
  __m512i saturatedSum =
    _mm512_xor_si512(_mm512_srai_epi32(a, 31), _mm512_set1_epi32(0x7FFFFFFF));
  // return saturated sum if overflow occurred, otherwise return sum
  return _mm512_or_si512(_mm512_andnot_si512(overflow, sum),
                         _mm512_and_si512(overflow, saturatedSum));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 64> adds(const Vec<Float, 64> &a,
                                       const Vec<Float, 64> &b)
{
  return _mm512_add_ps(a, b);
}

// Double not saturated
static SIMD_INLINE Vec<Double, 64> adds(const Vec<Double, 64> &a,
                                        const Vec<Double, 64> &b)
{
  return _mm512_add_pd(a, b);
}

// ---------------------------------------------------------------------------
// sub v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> sub(const Vec<Byte, 64> &a,
                                     const Vec<Byte, 64> &b)
{
  return _mm512_sub_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 64> sub(const Vec<SignedByte, 64> &a,
                                           const Vec<SignedByte, 64> &b)
{
  return _mm512_sub_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 64> sub(const Vec<Word, 64> &a,
                                     const Vec<Word, 64> &b)
{
  return _mm512_sub_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 64> sub(const Vec<Short, 64> &a,
                                      const Vec<Short, 64> &b)
{
  return _mm512_sub_epi16(a, b);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> sub(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(sub(a.lo(), b.lo()), sub(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> sub(const Vec<Int, 64> &a,
                                    const Vec<Int, 64> &b)
{
  return _mm512_sub_epi32(a, b);
}

static SIMD_INLINE Vec<Float, 64> sub(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  return _mm512_sub_ps(a, b);
}

static SIMD_INLINE Vec<Double, 64> sub(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  return _mm512_sub_pd(a, b);
}

// ---------------------------------------------------------------------------
// subs
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> subs(const Vec<Byte, 64> &a,
                                      const Vec<Byte, 64> &b)
{
  return _mm512_subs_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 64> subs(const Vec<SignedByte, 64> &a,
                                            const Vec<SignedByte, 64> &b)
{
  return _mm512_subs_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 64> subs(const Vec<Word, 64> &a,
                                      const Vec<Word, 64> &b)
{
  return _mm512_subs_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 64> subs(const Vec<Short, 64> &a,
                                       const Vec<Short, 64> &b)
{
  return _mm512_subs_epi16(a, b);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> subs(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(subs(a.lo(), b.lo()), subs(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> subs(const Vec<Int, 64> &a,
                                     const Vec<Int, 64> &b)
{
  // 09. Mar 23 (Jonas Keller): added workaround so that this function is
  // saturated

  // _mm512_subs_epi32 does not exist, workaround:
  // Hacker's Delight, 2-13 Overflow Detection: "[...] overflow in the final
  // value of x−y [...] occurs if and only if x and y have opposite signs and
  // the sign of x−y [...] is opposite to that of x [...]"
  __m512i diff            = _mm512_sub_epi32(a, b);
  __m512i opsHaveDiffSign = _mm512_xor_si512(a, b);
  __m512i diffHasDiffSign = _mm512_xor_si512(a, diff);
  // indicates when an overflow has occurred
  __m512i overflow =
    _mm512_srai_epi32(_mm512_and_si512(opsHaveDiffSign, diffHasDiffSign), 31);
  // saturated diff for if overflow occurred (0x7FFFFFFF=max positive int, when
  // sign of a (and thus b as well) is 0, 0x80000000=min negative int, when sign
  // of a (and thus b as well) is 1)
  __m512i saturatedDiff =
    _mm512_xor_si512(_mm512_srai_epi32(a, 31), _mm512_set1_epi32(0x7FFFFFFF));
  // return saturated diff if overflow occurred, otherwise return diff
  return _mm512_or_si512(_mm512_andnot_si512(overflow, diff),
                         _mm512_and_si512(overflow, saturatedDiff));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 64> subs(const Vec<Float, 64> &a,
                                       const Vec<Float, 64> &b)
{
  return _mm512_sub_ps(a, b);
}

// Double not saturated
static SIMD_INLINE Vec<Double, 64> subs(const Vec<Double, 64> &a,
                                        const Vec<Double, 64> &b)
{
  return _mm512_sub_pd(a, b);
}

// ---------------------------------------------------------------------------
// neg (negate = two's complement or unary minus), only signed types v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<SignedByte, 64> neg(const Vec<SignedByte, 64> &a)
{
  return _mm512_sub_epi8(_mm512_setzero_si512(), a);
}

static SIMD_INLINE Vec<Short, 64> neg(const Vec<Short, 64> &a)
{
  return _mm512_sub_epi16(_mm512_setzero_si512(), a);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> neg(const Vec<T, 64> &a)
{
  return Vec<T, 64>(neg(a.lo()), neg(a.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> neg(const Vec<Int, 64> &a)
{
  return _mm512_sub_epi32(_mm512_setzero_si512(), a);
}

static SIMD_INLINE Vec<Float, 64> neg(const Vec<Float, 64> &a)
{
  return _mm512_sub_ps(_mm512_setzero_ps(), a);
}

static SIMD_INLINE Vec<Double, 64> neg(const Vec<Double, 64> &a)
{
  return _mm512_sub_pd(_mm512_setzero_pd(), a);
}

// ---------------------------------------------------------------------------
// min v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> min(const Vec<Byte, 64> &a,
                                     const Vec<Byte, 64> &b)
{
  return _mm512_min_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 64> min(const Vec<SignedByte, 64> &a,
                                           const Vec<SignedByte, 64> &b)
{
  return _mm512_min_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 64> min(const Vec<Word, 64> &a,
                                     const Vec<Word, 64> &b)
{
  return _mm512_min_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 64> min(const Vec<Short, 64> &a,
                                      const Vec<Short, 64> &b)
{
  return _mm512_min_epi16(a, b);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> min(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(min(a.lo(), b.lo()), min(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> min(const Vec<Int, 64> &a,
                                    const Vec<Int, 64> &b)
{
  return _mm512_min_epi32(a, b);
}

// there is an unsigned version of min for 32 bit but we currently
// don't have an element type for it

static SIMD_INLINE Vec<Float, 64> min(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  return _mm512_min_ps(a, b);
}

static SIMD_INLINE Vec<Double, 64> min(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  return _mm512_min_pd(a, b);
}

// ---------------------------------------------------------------------------
// max v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> max(const Vec<Byte, 64> &a,
                                     const Vec<Byte, 64> &b)
{
  return _mm512_max_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 64> max(const Vec<SignedByte, 64> &a,
                                           const Vec<SignedByte, 64> &b)
{
  return _mm512_max_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 64> max(const Vec<Word, 64> &a,
                                     const Vec<Word, 64> &b)
{
  return _mm512_max_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 64> max(const Vec<Short, 64> &a,
                                      const Vec<Short, 64> &b)
{
  return _mm512_max_epi16(a, b);
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> max(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(max(a.lo(), b.lo()), max(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> max(const Vec<Int, 64> &a,
                                    const Vec<Int, 64> &b)
{
  return _mm512_max_epi32(a, b);
}

// there is an unsigned version of max for 32 bit but we currently
// don't have an element type for it

static SIMD_INLINE Vec<Float, 64> max(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  return _mm512_max_ps(a, b);
}

static SIMD_INLINE Vec<Double, 64> max(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  return _mm512_max_pd(a, b);
}

// ---------------------------------------------------------------------------
// mul, div v
// ---------------------------------------------------------------------------

// TODO: add mul/div versions for int types? or make special versions of mul
// TODO: and div where the result is scaled?

static SIMD_INLINE Vec<Float, 64> mul(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  return _mm512_mul_ps(a, b);
}

static SIMD_INLINE Vec<Double, 64> mul(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  return _mm512_mul_pd(a, b);
}

static SIMD_INLINE Vec<Float, 64> div(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  return _mm512_div_ps(a, b);
}

static SIMD_INLINE Vec<Double, 64> div(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  return _mm512_div_pd(a, b);
}

// ---------------------------------------------------------------------------
// ceil, floor, round, truncate v
// ---------------------------------------------------------------------------

// 25. Mar 23 (Jonas Keller): added versions for integer types

// versions for integer types do nothing:

template <typename T>
static SIMD_INLINE Vec<T, 64> ceil(const Vec<T, 64> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 64> floor(const Vec<T, 64> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 64> round(const Vec<T, 64> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 64> truncate(const Vec<T, 64> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

// see Peter Cordes at https://stackoverflow.com/questions/50854991
// _mm512_roundscale_ps:
// imm[7:4] = fraction bits = here 0, imm[0:1] = rounding mode

static SIMD_INLINE Vec<Float, 64> ceil(const Vec<Float, 64> &a)
{
  return _mm512_roundscale_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> ceil(const Vec<Double, 64> &a)
{
  return _mm512_roundscale_pd(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> floor(const Vec<Float, 64> &a)
{
  return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> floor(const Vec<Double, 64> &a)
{
  return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> round(const Vec<Float, 64> &a)
{
  return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> round(const Vec<Double, 64> &a)
{
  return _mm512_roundscale_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 64> truncate(const Vec<Float, 64> &a)
{
  return _mm512_roundscale_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 64> truncate(const Vec<Double, 64> &a)
{
  return _mm512_roundscale_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

// ---------------------------------------------------------------------------
// elementary mathematical functions v
// ---------------------------------------------------------------------------

// estimate of a reciprocal
// NOTE: this has better precision than SSE and AVX versions!

// float version
static SIMD_INLINE Vec<Float, 64> rcp(const Vec<Float, 64> &a)
{
  // 20. Mar 23 (Jonas Keller):
  // use _mm512_rcp28_ps if available, which has even better precision
  // and does not seem to be any slower (at least according to this:
  // https://github.com/tanakamura/instruction-bench/blob/master/knl.log)
#ifdef __AVX512ER__
  return _mm512_rcp28_ps(a);
#else
  return _mm512_rcp14_ps(a);
#endif
}

// double version
static SIMD_INLINE Vec<Double, 64> rcp(const Vec<Double, 64> &a)
{
  // use _mm512_rcp28_pd if available, which has even better precision
  // and does not seem to be any slower (at least according to this:
  // https://github.com/tanakamura/instruction-bench/blob/master/knl.log)
#ifdef __AVX512ER__
  return _mm512_rcp28_pd(a);
#else
  return _mm512_rcp14_pd(a);
#endif
}

// estimate of reverse square root
// NOTE: this has better precision than SSE and AVX versions!

// float version
static SIMD_INLINE Vec<Float, 64> rsqrt(const Vec<Float, 64> &a)
{
  // 20. Mar 23 (Jonas Keller):
  // use _mm512_rsqrt28_ps if available, which has even better precision
  // and does not seem to be any slower (probably)
#ifdef __AVX512ER__
  return _mm512_rsqrt28_ps(a);
#else
  return _mm512_rsqrt14_ps(a);
#endif
}

// double version
static SIMD_INLINE Vec<Double, 64> rsqrt(const Vec<Double, 64> &a)
{
  // use _mm512_rsqrt28_pd if available, which has even better precision
  // and does not seem to be any slower (probably)
#ifdef __AVX512ER__
  return _mm512_rsqrt28_pd(a);
#else
  return _mm512_rsqrt14_pd(a);
#endif
}

// square root

// float version
static SIMD_INLINE Vec<Float, 64> sqrt(const Vec<Float, 64> &a)
{
  return _mm512_sqrt_ps(a);
}

// double version
static SIMD_INLINE Vec<Double, 64> sqrt(const Vec<Double, 64> &a)
{
  return _mm512_sqrt_pd(a);
}

// ---------------------------------------------------------------------------
// abs v
// ---------------------------------------------------------------------------

// 25. Mar 25 (Jonas Keller): added abs for unsigned integers

// unsigned integers
template <typename T, SIMD_ENABLE_IF(std::is_unsigned<T>::value
                                       &&std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 64> abs(const Vec<T, 64> &a)
{
  return a;
}

static SIMD_INLINE Vec<SignedByte, 64> abs(const Vec<SignedByte, 64> &a)
{
#ifdef __AVX512BW__
  return _mm512_abs_epi8(a);
#else
  // non-avx512bw workaround
  return Vec<SignedByte, 64>(abs(a.lo()), abs(a.hi()));
#endif
}

static SIMD_INLINE Vec<Short, 64> abs(const Vec<Short, 64> &a)
{
#ifdef __AVX512BW__
  return _mm512_abs_epi16(a);
#else
  // non-avx512bw workaround
  return Vec<Short, 64>(abs(a.lo()), abs(a.hi()));
#endif
}

static SIMD_INLINE Vec<Int, 64> abs(const Vec<Int, 64> &a)
{
  return _mm512_abs_epi32(a);
}

static SIMD_INLINE Vec<Float, 64> abs(const Vec<Float, 64> &a)
{
  return _mm512_abs_ps(a);
}

static SIMD_INLINE Vec<Double, 64> abs(const Vec<Double, 64> &a)
{
  return _mm512_abs_pd(a);
}

// ---------------------------------------------------------------------------
// unpacklo v (with permutex2var)
// ---------------------------------------------------------------------------

#ifdef __AVX512VBMI__

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<1>)
{
  // element order high to low for idx
  // 95,31,94,30,93,29,92,28,
  // 91,27,90,26,89,25,88,24,
  // 87,23,86,22,85,21,84,20,
  // 83,19,82,18,81,17,80,16,
  // 79,15,78,14,77,13,76,12,
  // 75,11,74,10,73, 9,72, 8,
  // 71, 7,70, 6,69, 5,68, 4,
  // 67, 3,66, 2,65, 1,64, 0
  __m512i idx = _mm512_set_epi32(
    0x5f1f5e1e, 0x5d1d5c1c, 0x5b1b5a1a, 0x59195818, 0x57175616, 0x55155414,
    0x53135212, 0x51115010, 0x4f0f4e0e, 0x4d0d4c0c, 0x4b0b4a0a, 0x49094808,
    0x47074606, 0x45054404, 0x43034202, 0x41014000);
  return _mm512_permutex2var_epi8(a, idx, b);
}

#else

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<1>)
{
  return x_mm512_unpacklo_epi8(x_mm512_transpose8x64_epi64(a),
                               x_mm512_transpose8x64_epi64(b));
}

#endif

#ifdef __AVX512BW__

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<2>)
{
  // element order high to low for idx
  // 47,15,46,14,45,13,44,12,
  // 43,11,42,10,41, 9,40, 8,
  // 39, 7,38, 6,37, 5,36, 4,
  // 35, 3,34, 2,33, 1,32, 0
  __m512i idx = _mm512_set_epi32(
    0x002f000f, 0x002e000e, 0x002d000d, 0x002c000c, 0x002b000b, 0x002a000a,
    0x00290009, 0x00280008, 0x00270007, 0x00260006, 0x00250005, 0x00240004,
    0x00230003, 0x00220002, 0x00210001, 0x00200000);
  return _mm512_permutex2var_epi16(a, idx, b);
}

#else

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<2>)
{
  return x_mm512_unpacklo_epi16(x_mm512_transpose8x64_epi64(a),
                                x_mm512_transpose8x64_epi64(b));
}

#endif

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<4>)
{
  __m512i idx =
    _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  return _mm512_permutex2var_epi32(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<8>)
{
  __m512i idx = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<0>, Bytes<32>)
{
  __m512i idx = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<0>,
                                         Bytes<4>)
{
  __m512i idx =
    _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  return _mm512_permutex2var_ps(a, idx, b);
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<0>,
                                         Bytes<8>)
{
  __m512i idx = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<0>,
                                         Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<0>,
                                         Bytes<32>)
{
  __m512i idx = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b, Part<0>,
                                          Bytes<8>)
{
  __m512i idx = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  return _mm512_permutex2var_pd(a, idx, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b, Part<0>,
                                          Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
  return _mm512_permutex2var_pd(a, idx, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b, Part<0>,
                                          Bytes<32>)
{
  __m512i idx = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
  return _mm512_permutex2var_pd(a, idx, b);
}

// ---------------------------------------------------------------------------
// unpackhi v (with permutex2var)
// ---------------------------------------------------------------------------

#ifdef __AVX512VBMI__

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<1>)
{
  // element order high to low for idx
  // 127,63,126,62,125,61,124,60,
  // 123,59,122,58,121,57,120,56,
  // 119,55,118,54,117,53,116,52,
  // 115,51,114,50,113,49,112,48,
  // 111,47,110,46,109,45,108,44,
  // 107,43,106,42,105,41,104,40,
  // 103,39,102,38,101,37,100,36,
  //  99,35, 98,34, 97,33, 96,32
  __m512i idx = _mm512_set_epi32(
    0x7f3f7e3e, 0x7d3d7c3c, 0x7b3b7a3a, 0x79397838, 0x77377636, 0x75357434,
    0x73337232, 0x71317030, 0x6f2f6e2e, 0x6d2d6c2c, 0x6b2b6a2a, 0x69296828,
    0x67276626, 0x65256424, 0x63236222, 0x61216020);
  return _mm512_permutex2var_epi8(a, idx, b);
}

#else

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<1>)
{
  return x_mm512_unpackhi_epi8(x_mm512_transpose8x64_epi64(a),
                               x_mm512_transpose8x64_epi64(b));
}

#endif

#ifdef __AVX512BW__

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<2>)
{
  // element order high to low for idx
  // 63,31,62,30,61,29,60,28,
  // 59,27,58,26,57,25,56,24,
  // 55,23,54,22,53,21,52,20,
  // 51,19,50,18,49,17,48,16
  __m512i idx = _mm512_set_epi32(
    0x003f001f, 0x003e001e, 0x003d001d, 0x003c001c, 0x003b001b, 0x003a001a,
    0x00390019, 0x00380018, 0x00370017, 0x00360016, 0x00350015, 0x00340014,
    0x00330013, 0x00320012, 0x00310011, 0x00300010);
  return _mm512_permutex2var_epi16(a, idx, b);
}

#else

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<2>)
{
  return x_mm512_unpackhi_epi16(x_mm512_transpose8x64_epi64(a),
                                x_mm512_transpose8x64_epi64(b));
}

#endif

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<4>)
{
  __m512i idx = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10,
                                 25, 9, 24, 8);
  return _mm512_permutex2var_epi32(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<8>)
{
  __m512i idx = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                     Part<1>, Bytes<32>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<1>,
                                         Bytes<4>)
{
  __m512i idx = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10,
                                 25, 9, 24, 8);
  return _mm512_permutex2var_ps(a, idx, b);
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<1>,
                                         Bytes<8>)
{
  __m512i idx = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<1>,
                                         Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b, Part<1>,
                                         Bytes<32>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b, Part<1>,
                                          Bytes<8>)
{
  __m512i idx = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  return _mm512_permutex2var_pd(a, idx, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b, Part<1>,
                                          Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
  return _mm512_permutex2var_pd(a, idx, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b, Part<1>,
                                          Bytes<32>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
  return _mm512_permutex2var_pd(a, idx, b);
}

// ---------------------------------------------------------------------------
// unpack hub v
// ---------------------------------------------------------------------------

// generalized unpack
// unpack blocks of NUM_ELEMS elements of type T
// PART=0: low half of input vectors,
// PART=1: high half of input vectors
template <int PART, int NUM_ELEMS, typename T>
static SIMD_INLINE Vec<T, 64> unpack(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return unpack(a, b, Part<PART>(), Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// 128-bit-lane oriented unpacklo (with direct intrinsic calls)
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<0>, Bytes<1>)
{
  return x_mm512_unpacklo_epi8(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<0>, Bytes<2>)
{
  return x_mm512_unpacklo_epi16(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<0>, Bytes<4>)
{
  return _mm512_unpacklo_epi32(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<0>, Bytes<8>)
{
  return _mm512_unpacklo_epi64(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<0>, Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<0>, Bytes<32>)
{
  return _mm512_shuffle_i32x4(a, b, _MM_SHUFFLE(1, 0, 1, 0));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<0>,
                                           Bytes<4>)
{
  return _mm512_unpacklo_ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<0>,
                                           Bytes<8>)
{
  return _mm512_castpd_ps(
    _mm512_unpacklo_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<0>,
                                           Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<0>,
                                           Bytes<32>)
{
  return _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(1, 0, 1, 0));
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack16(const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b, Part<0>,
                                            Bytes<8>)
{
  return _mm512_unpacklo_pd(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack16(const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b, Part<0>,
                                            Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  return _mm512_permutex2var_pd(a, idx, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack16(const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b, Part<0>,
                                            Bytes<32>)
{
  return _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(1, 0, 1, 0));
}

// ---------------------------------------------------------------------------
// 128-bit-lane oriented unpackhi v
// ---------------------------------------------------------------------------

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<1>, Bytes<1>)
{
  return x_mm512_unpackhi_epi8(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<1>, Bytes<2>)
{
  return x_mm512_unpackhi_epi16(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<1>, Bytes<4>)
{
  return _mm512_unpackhi_epi32(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<1>, Bytes<8>)
{
  return _mm512_unpackhi_epi64(a, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<1>, Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  return _mm512_permutex2var_epi64(a, idx, b);
}

// integer version
template <typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b,
                                       Part<1>, Bytes<32>)
{
  return _mm512_shuffle_i32x4(a, b, _MM_SHUFFLE(3, 2, 3, 2));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<1>,
                                           Bytes<4>)
{
  return _mm512_unpackhi_ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<1>,
                                           Bytes<8>)
{
  return _mm512_castpd_ps(
    _mm512_unpackhi_pd(_mm512_castps_pd(a), _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<1>,
                                           Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  return _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idx, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE Vec<Float, 64> unpack16(const Vec<Float, 64> &a,
                                           const Vec<Float, 64> &b, Part<1>,
                                           Bytes<32>)
{
  return _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(3, 2, 3, 2));
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack16(const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b, Part<1>,
                                            Bytes<8>)
{
  return _mm512_unpackhi_pd(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack16(const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b, Part<1>,
                                            Bytes<16>)
{
  __m512i idx = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  return _mm512_permutex2var_pd(a, idx, b);
}

// double version
static SIMD_INLINE Vec<Double, 64> unpack16(const Vec<Double, 64> &a,
                                            const Vec<Double, 64> &b, Part<1>,
                                            Bytes<32>)
{
  return _mm512_shuffle_f64x2(a, b, _MM_SHUFFLE(3, 2, 3, 2));
}

// ---------------------------------------------------------------------------
// 128-bit-lane oriented unpack hub
// ---------------------------------------------------------------------------

// generalized 128-bit-lane oriented unpack
// unpack blocks of NUM_ELEMS elements of type T
// PART=0: low half of 128-bit lanes of input vectors,
// PART=1: high half of 128-bit lanes of input vectors
template <int PART, int NUM_ELEMS, typename T>
static SIMD_INLINE Vec<T, 64> unpack16(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return unpack16(a, b, Part<PART>(), Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// zip v
// ---------------------------------------------------------------------------

// a, b are passed by-value to avoid problems with identical in/out args.

// here we typically have to transpose the inputs in the same way
// for both output computations, so we define separate functions for
// all T and Bytes<> (combinations of unpack functions above)

// integer version
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h, Bytes<1>)
{
  l = unpack(a, b, Part<0>(), Bytes<1>());
  h = unpack(a, b, Part<1>(), Bytes<1>());
}

// integer version
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h, Bytes<2>)
{
  l = unpack(a, b, Part<0>(), Bytes<2>());
  h = unpack(a, b, Part<1>(), Bytes<2>());
}

// integer version
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h, Bytes<4>)
{
  l = unpack(a, b, Part<0>(), Bytes<4>());
  h = unpack(a, b, Part<1>(), Bytes<4>());
}

// integer version
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h, Bytes<8>)
{
  l = unpack(a, b, Part<0>(), Bytes<8>());
  h = unpack(a, b, Part<1>(), Bytes<8>());
}

// integer version
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h, Bytes<16>)
{
  l = unpack(a, b, Part<0>(), Bytes<16>());
  h = unpack(a, b, Part<1>(), Bytes<16>());
}

// integer version
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h, Bytes<32>)
{
  l = unpack(a, b, Part<0>(), Bytes<32>());
  h = unpack(a, b, Part<1>(), Bytes<32>());
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                            Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<4>)
{
  l = unpack(a, b, Part<0>(), Bytes<4>());
  h = unpack(a, b, Part<1>(), Bytes<4>());
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                            Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<8>)
{
  l = unpack(a, b, Part<0>(), Bytes<8>());
  h = unpack(a, b, Part<1>(), Bytes<8>());
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                            Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<16>)
{
  l = unpack(a, b, Part<0>(), Bytes<16>());
  h = unpack(a, b, Part<1>(), Bytes<16>());
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                            Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<32>)
{
  l = unpack(a, b, Part<0>(), Bytes<32>());
  h = unpack(a, b, Part<1>(), Bytes<32>());
}

// ---------------------------------------------------------------------------
// zip hub v
// ---------------------------------------------------------------------------

// zips blocks of NUM_ELEMS elements of type T
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void zip(const Vec<T, 64> a, const Vec<T, 64> b,
                            Vec<T, 64> &l, Vec<T, 64> &h)
{
  return zip(a, b, l, h, Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// zip16 (16-byte-lane oriented zip)
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// zips blocks of NUM_ELEMS elements of type T
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void zip16(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h)
{
  l = unpack16(a, b, Part<0>(), Bytes<NUM_ELEMS * sizeof(T)>());
  h = unpack16(a, b, Part<1>(), Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// unzip v
// ---------------------------------------------------------------------------

// a, b are passed by-value to avoid problems with identical
// input/output args.

// here we typically have to transpose the inputs in the same way
// for both output computations, so we define separate functions for
// all T and Bytes<> (combinations of unpack functions above)

static SIMD_INLINE __m512i x_mm512_even64_mask()
{
  return _mm512_set_epi64
    // b6   b4   b2   b0  a6 a4 a2 a0
    (8 | 6, 8 | 4, 8 | 2, 8 | 0, 6, 4, 2, 0);
}

static SIMD_INLINE __m512i x_mm512_odd64_mask()
{
  return _mm512_set_epi64
    // b7   b5   b3   b1  a7 a5 a3 a1
    (8 | 7, 8 | 5, 8 | 3, 8 | 1, 7, 5, 3, 1);
}

// _mm512_set_epi8 is missing from some gcc versions, so we use epi32
// https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg188664.html

#ifdef __AVX512VBMI__

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<1>)
{
  // element order for low idx
  // 126,124,122,120,118,116,114,112,
  // 110,108,106,104,102,100, 98, 96,
  //  94, 92, 90, 88, 86, 84, 82, 80,
  //  78, 76, 74, 72, 70, 68, 66, 64,
  //  62, 60, 58, 56, 54, 52, 50, 48,
  //  46, 44, 42, 40, 38, 36, 34, 32,
  //  30, 28, 26, 24, 22, 20, 18, 16,
  //  14, 12, 10,  8,  6,  4,  2,  0
  __m512i idxL = _mm512_set_epi32(
    0x7e7c7a78, 0x76747270, 0x6e6c6a68, 0x66646260, 0x5e5c5a58, 0x56545250,
    0x4e4c4a48, 0x46444240, 0x3e3c3a38, 0x36343230, 0x2e2c2a28, 0x26242220,
    0x1e1c1a18, 0x16141210, 0x0e0c0a08, 0x06040200);
  // element order for high idx
  // 127,125,123,121,119,117,115,113,
  // 111,109,107,105,103,101, 99, 97,
  //  95, 93, 91, 89, 87, 85, 83, 81,
  //  79, 77, 75, 73, 71, 69, 67, 65,
  //  63, 61, 59, 57, 55, 53, 51, 49,
  //  47, 45, 43, 41, 39, 37, 35, 33,
  //  31, 29, 27, 25, 23, 21, 19, 17,
  //  15, 13, 11,  9,  7,  5,  3,  1
  __m512i idxH = _mm512_set_epi32(
    0x7f7d7b79, 0x77757371, 0x6f6d6b69, 0x67656361, 0x5f5d5b59, 0x57555351,
    0x4f4d4b49, 0x47454341, 0x3f3d3b39, 0x37353331, 0x2f2d2b29, 0x27252321,
    0x1f1d1b19, 0x17151311, 0x0f0d0b09, 0x07050301);
  l = _mm512_permutex2var_epi8(a, idxL, b);
  h = _mm512_permutex2var_epi8(a, idxH, b);
}

#else

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<1>)
{
  // mask is hopefully only set once if unzip is used multiple times
  __m512i mask = x_mm512_set_epi8(
    15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5,
    3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8,
    6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
  __m512i atmp = x_mm512_shuffle_epi8(a, mask);
  __m512i btmp = x_mm512_shuffle_epi8(b, mask);
  l = _mm512_permutex2var_epi64(atmp, x_mm512_even64_mask(), btmp);
  h = _mm512_permutex2var_epi64(atmp, x_mm512_odd64_mask(), btmp);
}

#endif

#ifdef __AVX512BW__

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<2>)
{
  // element order for low idx
  // 62,60,58,56,54,52,50,48,
  // 46,44,42,40,38,36,34,32,
  // 30,28,26,24,22,20,18,16,
  // 14,12,10, 8, 6, 4, 2, 0
  __m512i idxL = _mm512_set_epi32(
    0x003e003c, 0x003a0038, 0x00360034, 0x00320030, 0x002e002c, 0x002a0028,
    0x00260024, 0x00220020, 0x001e001c, 0x001a0018, 0x00160014, 0x00120010,
    0x000e000c, 0x000a0008, 0x00060004, 0x00020000);
  // element order for high idx
  // 63,61,59,57,55,53,51,49,
  // 47,45,43,41,39,37,35,33,
  // 31,29,27,25,23,21,19,17,
  // 15,13,11, 9, 7, 5, 3, 1
  __m512i idxH = _mm512_set_epi32(
    0x003f003d, 0x003b0039, 0x00370035, 0x00330031, 0x002f002d, 0x002b0029,
    0x00270025, 0x00230021, 0x001f001d, 0x001b0019, 0x00170015, 0x00130011,
    0x000f000d, 0x000b0009, 0x00070005, 0x00030001);
  l = _mm512_permutex2var_epi16(a, idxL, b);
  h = _mm512_permutex2var_epi16(a, idxH, b);
}

#else

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<2>)
{
  // mask is hopefully only set once if unzip is used multiple times
  __m512i mask = x_mm512_set_epi8(
    15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6,
    3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5,
    4, 1, 0, 15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
  __m512i atmp = x_mm512_shuffle_epi8(a, mask);
  __m512i btmp = x_mm512_shuffle_epi8(b, mask);
  l = _mm512_permutex2var_epi64(atmp, x_mm512_even64_mask(), btmp);
  h = _mm512_permutex2var_epi64(atmp, x_mm512_odd64_mask(), btmp);
}

#endif

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<4>)
{
  __m512i idxL =
    _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idxH =
    _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  l = _mm512_permutex2var_epi32(a, idxL, b);
  h = _mm512_permutex2var_epi32(a, idxH, b);
}

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<8>)
{
  __m512i idxL = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idxH = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  l            = _mm512_permutex2var_epi64(a, idxL, b);
  h            = _mm512_permutex2var_epi64(a, idxH, b);
}

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<16>)
{
  // lanes:                       b2     b0   a2   a0
  __m512i idxL = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
  // lanes:                       b3     b1     a3   a1
  __m512i idxH = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
  l            = _mm512_permutex2var_epi64(a, idxL, b);
  h            = _mm512_permutex2var_epi64(a, idxH, b);
}

// integer version
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h, Bytes<32>)
{
  l = unpack(a, b, Part<0>(), Bytes<32>());
  h = unpack(a, b, Part<1>(), Bytes<32>());
}

// float version
static SIMD_INLINE void unzip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                              Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<4>)
{
  __m512i idxL =
    _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idxH =
    _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  l = _mm512_permutex2var_ps(a, idxL, b);
  h = _mm512_permutex2var_ps(a, idxH, b);
}

// float version
static SIMD_INLINE void unzip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                              Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<8>)
{
  __m512i idxL = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idxH = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  l            = _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idxL, _mm512_castps_pd(b)));
  h = _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idxH, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE void unzip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                              Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<16>)
{
  // lanes:                       b2     b0   a2   a0
  __m512i idxL = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
  // lanes:                       b3     b1     a3   a1
  __m512i idxH = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
  l            = _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idxL, _mm512_castps_pd(b)));
  h = _mm512_castpd_ps(
    _mm512_permutex2var_pd(_mm512_castps_pd(a), idxH, _mm512_castps_pd(b)));
}

// float version
static SIMD_INLINE void unzip(const Vec<Float, 64> a, const Vec<Float, 64> b,
                              Vec<Float, 64> &l, Vec<Float, 64> &h, Bytes<32>)
{
  l = unpack(a, b, Part<0>(), Bytes<32>());
  h = unpack(a, b, Part<1>(), Bytes<32>());
}

// double version
static SIMD_INLINE void unzip(const Vec<Double, 64> a, const Vec<Double, 64> b,
                              Vec<Double, 64> &l, Vec<Double, 64> &h, Bytes<8>)
{
  const __m512i idxL = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  const __m512i idxH = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  l                  = _mm512_permutex2var_pd(a, idxL, b);
  h                  = _mm512_permutex2var_pd(a, idxH, b);
}

// double version
static SIMD_INLINE void unzip(const Vec<Double, 64> a, const Vec<Double, 64> b,
                              Vec<Double, 64> &l, Vec<Double, 64> &h, Bytes<16>)
{
  const __m512i idxL = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
  const __m512i idxH = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
  l                  = _mm512_permutex2var_pd(a, idxL, b);
  h                  = _mm512_permutex2var_pd(a, idxH, b);
}

// ---------------------------------------------------------------------------
// unzip hub v
// ---------------------------------------------------------------------------

// hub
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void unzip(const Vec<T, 64> a, const Vec<T, 64> b,
                              Vec<T, 64> &l, Vec<T, 64> &h)
{
  return unzip(a, b, l, h, Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// packs v
// ---------------------------------------------------------------------------

// ========== signed -> signed ==========

static SIMD_INLINE Vec<SignedByte, 64> packs(const Vec<Short, 64> &a,
                                             const Vec<Short, 64> &b,
                                             OutputType<SignedByte>)
{
  return x_mm512_evenodd8x64_epi64(x_mm512_packs_epi16(a, b));
}

static SIMD_INLINE Vec<Short, 64> packs(const Vec<Int, 64> &a,
                                        const Vec<Int, 64> &b,
                                        OutputType<Short>)
{
  return x_mm512_evenodd8x64_epi64(x_mm512_packs_epi32(a, b));
}

static SIMD_INLINE Vec<Short, 64> packs(const Vec<Float, 64> &a,
                                        const Vec<Float, 64> &b,
                                        OutputType<Short>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Short>());
}

static SIMD_INLINE Vec<Float, 64> packs(const Vec<Double, 64> &a,
                                        const Vec<Double, 64> &b,
                                        OutputType<Float>)
{
  const __m256d low  = _mm256_castps_pd(_mm512_cvtpd_ps(a));
  const __m256d high = _mm256_castps_pd(_mm512_cvtpd_ps(b));
  return _mm512_castpd_ps(
    _mm512_insertf64x4(_mm512_castpd256_pd512(low), high, 1));
}

static SIMD_INLINE Vec<Int, 64> packs(const Vec<Double, 64> &a,
                                      const Vec<Double, 64> &b, OutputType<Int>)
{
  const __m512d clip = _mm512_set1_pd(std::numeric_limits<Int>::max());
  const __m256i low  = _mm512_cvtpd_epi32(_mm512_min_pd(clip, a));
  const __m256i high = _mm512_cvtpd_epi32(_mm512_min_pd(clip, b));
  return _mm512_inserti64x4(_mm512_castsi256_si512(low), high, 1);
}

// ========== signed -> unsigned ==========

// non-avx512bw workaround
static SIMD_INLINE Vec<Byte, 64> packs(const Vec<Short, 64> &a,
                                       const Vec<Short, 64> &b,
                                       OutputType<Byte>)
{
  return x_mm512_evenodd8x64_epi64(x_mm512_packus_epi16(a, b));
}

// non-avx512bw workaround
static SIMD_INLINE Vec<Word, 64> packs(const Vec<Int, 64> &a,
                                       const Vec<Int, 64> &b, OutputType<Word>)
{
  return x_mm512_evenodd8x64_epi64(x_mm512_packus_epi32(a, b));
}

static SIMD_INLINE Vec<Word, 64> packs(const Vec<Float, 64> &a,
                                       const Vec<Float, 64> &b,
                                       OutputType<Word>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Word>());
}

// ---------------------------------------------------------------------------
// generalized extend: no stage v
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
static SIMD_INLINE void extend(const Vec<T, 64> &vIn, Vec<T, 64> vOut[1])
{
  vOut[0] = vIn;
}

// ---------------------------------------------------------------------------
// generalized extend: single stage v
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 64> &vIn,
                               Vec<Short, 64> vOut[2])
{
#ifdef __AVX512BW__
  vOut[0] = _mm512_cvtepi8_epi16(vIn.lo());
  vOut[1] = _mm512_cvtepi8_epi16(vIn.hi());
#else
  {
    const __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vIn.lo()));
    const __m256i hi =
      _mm256_cvtepi8_epi16(_mm256_extractf128_si256(vIn.lo(), 1));
    vOut[0] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
  {
    const __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vIn.hi()));
    const __m256i hi =
      _mm256_cvtepi8_epi16(_mm256_extractf128_si256(vIn.hi(), 1));
    vOut[1] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
#endif
}

static SIMD_INLINE void extend(const Vec<Short, 64> &vIn, Vec<Int, 64> vOut[2])
{
  vOut[0] = _mm512_cvtepi16_epi32(vIn.lo());
  vOut[1] = _mm512_cvtepi16_epi32(vIn.hi());
}

static SIMD_INLINE void extend(const Vec<Short, 64> &vIn,
                               Vec<Float, 64> vOut[2])
{
  vOut[0] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vIn.lo()));
  vOut[1] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vIn.hi()));
}

static SIMD_INLINE void extend(const Vec<Int, 64> &vIn,
                               Vec<Double, 64> vecOut[2])
{
  vecOut[0] = _mm512_cvtepi32_pd(vIn.lo());
  vecOut[1] = _mm512_cvtepi32_pd(vIn.hi());
}

static SIMD_INLINE void extend(const Vec<Float, 64> &vIn,
                               Vec<Double, 64> vecOut[2])
{
  vecOut[0] = _mm512_cvtps_pd(vIn.lo());
  vecOut[1] = _mm512_cvtps_pd(vIn.hi());
}

// unsigned -> unsigned

static SIMD_INLINE void extend(const Vec<Byte, 64> &vIn, Vec<Word, 64> vOut[2])
{
  // there's no _mm512_cvtepu8_epu16()
  Vec<Byte, 64> zero = setzero(OutputType<Byte>(), Integer<64>());
  // 16. Jul 16 (rm): here we avoid to use generalized unpack from
  // SIMDVecExt.H
  vOut[0] = unpack(vIn, zero, Part<0>(), Bytes<1>());
  vOut[1] = unpack(vIn, zero, Part<1>(), Bytes<1>());
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 64> &vIn, Vec<Short, 64> vOut[2])
{
#ifdef __AVX512BW__
  vOut[0] = _mm512_cvtepu8_epi16(vIn.lo());
  vOut[1] = _mm512_cvtepu8_epi16(vIn.hi());
#else
  {
    const __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vIn.lo()));
    const __m256i hi =
      _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vIn.lo(), 1));
    vOut[0] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
  {
    const __m256i lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vIn.hi()));
    const __m256i hi =
      _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vIn.hi(), 1));
    vOut[1] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
#endif
}

static SIMD_INLINE void extend(const Vec<Word, 64> &vIn, Vec<Int, 64> vOut[2])
{
  vOut[0] = _mm512_cvtepu16_epi32(vIn.lo());
  vOut[1] = _mm512_cvtepu16_epi32(vIn.hi());
}

static SIMD_INLINE void extend(const Vec<Word, 64> &vIn, Vec<Float, 64> vOut[2])
{
  vOut[0] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(vIn.lo()));
  vOut[1] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(vIn.hi()));
}

// ---------------------------------------------------------------------------
// generalized extend: two stages v
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 64> &vIn,
                               Vec<Int, 64> vOut[4])
{
  __m256i vInLo256 = vIn.lo();
  vOut[0]          = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(vInLo256));
  vOut[1] = _mm512_cvtepi8_epi32(_mm256_extractf128_si256(vInLo256, 1));
  __m256i vInHi256 = vIn.hi();
  vOut[2]          = _mm512_cvtepi8_epi32(_mm256_castsi256_si128(vInHi256));
  vOut[3] = _mm512_cvtepi8_epi32(_mm256_extractf128_si256(vInHi256, 1));
}

static SIMD_INLINE void extend(const Vec<SignedByte, 64> &vIn,
                               Vec<Float, 64> vOut[4])
{
  Vec<Int, 64> vTmp[4];
  extend(vIn, vTmp);
  for (int i = 0; i < 4; i++) vOut[i] = cvts(vTmp[i], OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Short, 64> &vIn,
                               Vec<Double, 64> vOut[4])
{
  vOut[0] = _mm512_cvtepi32_pd(
    _mm256_cvtepi16_epi32(_mm512_extracti32x4_epi32(vIn, 0)));
  vOut[1] = _mm512_cvtepi32_pd(
    _mm256_cvtepi16_epi32(_mm512_extracti32x4_epi32(vIn, 1)));
  vOut[2] = _mm512_cvtepi32_pd(
    _mm256_cvtepi16_epi32(_mm512_extracti32x4_epi32(vIn, 2)));
  vOut[3] = _mm512_cvtepi32_pd(
    _mm256_cvtepi16_epi32(_mm512_extracti32x4_epi32(vIn, 3)));
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 64> &vIn, Vec<Int, 64> vOut[4])
{
  __m256i vInLo256 = vIn.lo();
  vOut[0]          = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(vInLo256));
  vOut[1] = _mm512_cvtepu8_epi32(_mm256_extractf128_si256(vInLo256, 1));
  __m256i vInHi256 = vIn.hi();
  vOut[2]          = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(vInHi256));
  vOut[3] = _mm512_cvtepu8_epi32(_mm256_extractf128_si256(vInHi256, 1));
}

static SIMD_INLINE void extend(const Vec<Byte, 64> &vIn, Vec<Float, 64> vOut[4])
{
  Vec<Int, 64> vTmp[4];
  extend(vIn, vTmp);
  for (int i = 0; i < 4; i++) vOut[i] = cvts(vTmp[i], OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Word, 64> &vIn,
                               Vec<Double, 64> vOut[4])
{
  vOut[0] = _mm512_cvtepi32_pd(
    _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(vIn, 0)));
  vOut[1] = _mm512_cvtepi32_pd(
    _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(vIn, 1)));
  vOut[2] = _mm512_cvtepi32_pd(
    _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(vIn, 2)));
  vOut[3] = _mm512_cvtepi32_pd(
    _mm256_cvtepu16_epi32(_mm512_extracti32x4_epi32(vIn, 3)));
}

// ---------------------------------------------------------------------------
// generalized extend: three stages
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 64> &vIn,
                               Vec<Double, 64> vOut[8])
{
  const __m128i vIn128[4] = {
    _mm512_extracti32x4_epi32(vIn, 0),
    _mm512_extracti32x4_epi32(vIn, 1),
    _mm512_extracti32x4_epi32(vIn, 2),
    _mm512_extracti32x4_epi32(vIn, 3),
  };

  for (int i = 0; i < 4; i++) {
    vOut[i * 2 + 0] = _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(vIn128[i]));
    vOut[i * 2 + 1] =
      _mm512_cvtepi32_pd(_mm256_cvtepi8_epi32(_mm_srli_si128(vIn128[i], 8)));
  }
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 64> &vIn,
                               Vec<Double, 64> vOut[8])
{
  const __m128i vIn128[4] = {
    _mm512_extracti32x4_epi32(vIn, 0),
    _mm512_extracti32x4_epi32(vIn, 1),
    _mm512_extracti32x4_epi32(vIn, 2),
    _mm512_extracti32x4_epi32(vIn, 3),
  };

  for (int i = 0; i < 4; i++) {
    vOut[i * 2 + 0] = _mm512_cvtepi32_pd(_mm256_cvtepu8_epi32(vIn128[i]));
    vOut[i * 2 + 1] =
      _mm512_cvtepi32_pd(_mm256_cvtepu8_epi32(_mm_srli_si128(vIn128[i], 8)));
  }
}

// ---------------------------------------------------------------------------
// generalized extend: special case int <-> float v
// ---------------------------------------------------------------------------

static SIMD_INLINE void extend(const Vec<Int, 64> &vIn, Vec<Float, 64> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Float, 64> &vIn, Vec<Int, 64> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Int>());
}

// ---------------------------------------------------------------------------
// srai v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__
// 16. Oct 22 (Jonas Keller): added missing Byte and SignedByte versions

template <int IMM>
static SIMD_INLINE Vec<Byte, 64> srai(const Vec<Byte, 64> &a)
{
  return x_mm512_srai_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<SignedByte, 64> srai(const Vec<SignedByte, 64> &a)
{
  return x_mm512_srai_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Word, 64> srai(const Vec<Word, 64> &a)
{
  return x_mm512_srai_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 64> srai(const Vec<Short, 64> &a)
{
  return x_mm512_srai_epi16<IMM>(a);
}

#else

// non-avx512bw workaround
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> srai(const Vec<T, 64> &a)
{
  return Vec<T, 64>(srai<IMM>(a.lo()), srai<IMM>(a.hi()));
}

#endif

template <int IMM>
static SIMD_INLINE Vec<Int, 64> srai(const Vec<Int, 64> &a)
{
  return x_mm512_srai_epi32<IMM>(a);
}

// ---------------------------------------------------------------------------
// srli v
// ---------------------------------------------------------------------------

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<Byte, 64> srli(const Vec<Byte, 64> &a)
{
  return _mm512_and_si512(_mm512_set1_epi8((int8_t) (0xff >> IMM)),
                          x_mm512_srli_epi32<IMM>(a));
}

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<SignedByte, 64> srli(const Vec<SignedByte, 64> &a)
{
  return _mm512_and_si512(_mm512_set1_epi8((int8_t) (0xff >> IMM)),
                          x_mm512_srli_epi32<IMM>(a));
}

#ifdef __AVX512BW__

template <int IMM>
static SIMD_INLINE Vec<Word, 64> srli(const Vec<Word, 64> &a)
{
  return x_mm512_srli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 64> srli(const Vec<Short, 64> &a)
{
  return x_mm512_srli_epi16<IMM>(a);
}

#else

// non-avx512bw workaround
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> srli(const Vec<T, 64> &a)
{
  return Vec<T, 64>(srli<IMM>(a.lo()), srli<IMM>(a.hi()));
}

#endif

template <int IMM>
static SIMD_INLINE Vec<Int, 64> srli(const Vec<Int, 64> &a)
{
  return x_mm512_srli_epi32<IMM>(a);
}

// ---------------------------------------------------------------------------
// slli v
// ---------------------------------------------------------------------------

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<Byte, 64> slli(const Vec<Byte, 64> &a)
{
  return _mm512_and_si512(
    _mm512_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << IMM))),
    x_mm512_slli_epi32<IMM>(a));
}

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<SignedByte, 64> slli(const Vec<SignedByte, 64> &a)
{
  return _mm512_and_si512(
    _mm512_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << IMM))),
    x_mm512_slli_epi32<IMM>(a));
}

#ifdef __AVX512BW__

template <int IMM>
static SIMD_INLINE Vec<Word, 64> slli(const Vec<Word, 64> &a)
{
  return x_mm512_slli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 64> slli(const Vec<Short, 64> &a)
{
  return x_mm512_slli_epi16<IMM>(a);
}

#else

// non-avx512bw workaround
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> slli(const Vec<T, 64> &a)
{
  return Vec<T, 64>(slli<IMM>(a.lo()), slli<IMM>(a.hi()));
}

#endif

template <int IMM>
static SIMD_INLINE Vec<Int, 64> slli(const Vec<Int, 64> &a)
{
  return x_mm512_slli_epi32<IMM>(a);
}

// 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

// ---------------------------------------------------------------------------
// sra
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> sra(const Vec<Byte, 64> &a,
                                     const uint8_t count)
{
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
  }
  __m512i odd = _mm512_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m512i even =
    _mm512_sra_epi16(_mm512_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  __mmask64 mask = __mmask64(0x5555555555555555);
  return _mm512_mask_blend_epi8(mask, odd, even);
}

static SIMD_INLINE Vec<SignedByte, 64> sra(const Vec<SignedByte, 64> &a,
                                           const uint8_t count)
{
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
  }
  __m512i odd = _mm512_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m512i even =
    _mm512_sra_epi16(_mm512_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  __mmask64 mask = __mmask64(0x5555555555555555);
  return _mm512_mask_blend_epi8(mask, odd, even);
}

static SIMD_INLINE Vec<Word, 64> sra(const Vec<Word, 64> &a,
                                     const uint8_t count)
{
  return _mm512_sra_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 64> sra(const Vec<Short, 64> &a,
                                      const uint8_t count)
{
  return _mm512_sra_epi16(a, _mm_cvtsi32_si128(count));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> sra(const Vec<T, 64> &a, const uint8_t count)
{
  return Vec<T, 64>(sra(a.lo(), count), sra(a.hi(), count));
}

#endif

static SIMD_INLINE Vec<Int, 64> sra(const Vec<Int, 64> &a, const uint8_t count)
{
  return _mm512_sra_epi32(a, _mm_cvtsi32_si128(count));
}

// ---------------------------------------------------------------------------
// srl
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 64> srl(const Vec<Byte, 64> &a,
                                     const uint8_t count)
{
  return _mm512_and_si512(_mm512_srl_epi32(a, _mm_cvtsi32_si128(count)),
                          _mm512_set1_epi8((int8_t) (uint8_t) (0xff >> count)));
}

static SIMD_INLINE Vec<SignedByte, 64> srl(const Vec<SignedByte, 64> &a,
                                           const uint8_t count)
{
  return _mm512_and_si512(_mm512_srl_epi32(a, _mm_cvtsi32_si128(count)),
                          _mm512_set1_epi8((int8_t) (uint8_t) (0xff >> count)));
}

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Word, 64> srl(const Vec<Word, 64> &a,
                                     const uint8_t count)
{
  return _mm512_srl_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 64> srl(const Vec<Short, 64> &a,
                                      const uint8_t count)
{
  return _mm512_srl_epi16(a, _mm_cvtsi32_si128(count));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> srl(const Vec<T, 64> &a, const uint8_t count)
{
  return Vec<T, 64>(srl(a.lo(), count), srl(a.hi(), count));
}

#endif

static SIMD_INLINE Vec<Int, 64> srl(const Vec<Int, 64> &a, const uint8_t count)
{
  return _mm512_srl_epi32(a, _mm_cvtsi32_si128(count));
}

// ---------------------------------------------------------------------------
// sll
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 64> sll(const Vec<Byte, 64> &a,
                                     const uint8_t count)
{
  return _mm512_and_si512(
    _mm512_sll_epi32(a, _mm_cvtsi32_si128(count)),
    _mm512_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << count))));
}

static SIMD_INLINE Vec<SignedByte, 64> sll(const Vec<SignedByte, 64> &a,
                                           const uint8_t count)
{
  return _mm512_and_si512(
    _mm512_sll_epi32(a, _mm_cvtsi32_si128(count)),
    _mm512_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << count))));
}

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Word, 64> sll(const Vec<Word, 64> &a,
                                     const uint8_t count)
{
  return _mm512_sll_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 64> sll(const Vec<Short, 64> &a,
                                      const uint8_t count)
{
  return _mm512_sll_epi16(a, _mm_cvtsi32_si128(count));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> sll(const Vec<T, 64> &a, const uint8_t count)
{
  return Vec<T, 64>(sll(a.lo(), count), sll(a.hi(), count));
}

#endif

static SIMD_INLINE Vec<Int, 64> sll(const Vec<Int, 64> &a, const uint8_t count)
{
  return _mm512_sll_epi32(a, _mm_cvtsi32_si128(count));
}

// 05. Aug 22 (Jonas Keller):
// Improved implementation of hadd, hadds, hsub and hsubs,
// implementation does not use emulation via AVX anymore.
// Byte and SignedByte are now supported as well.
// The new implementation is faster for Int and Float, but
// slower for Word and Short for some reason.

// ---------------------------------------------------------------------------
// hadd v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> hadd(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return add(x, y);
}

// ---------------------------------------------------------------------------
// hadds v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> hadds(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return adds(x, y);
}

// ---------------------------------------------------------------------------
// hsub v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> hsub(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return sub(x, y);
}

// ---------------------------------------------------------------------------
// hsubs v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> hsubs(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  Vec<T, 64> x, y;
  unzip<1>(a, b, x, y);
  return subs(x, y);
}

// ---------------------------------------------------------------------------
// extraction of element 0 v
// ---------------------------------------------------------------------------

static SIMD_INLINE Byte elem0(const Vec<Byte, 64> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Byte,
                              _mm_cvtsi128_si32(_mm512_castsi512_si128(a)), 0);
}

static SIMD_INLINE SignedByte elem0(const Vec<SignedByte, 64> &a)
{
  SIMD_RETURN_REINTERPRET_INT(SignedByte,
                              _mm_cvtsi128_si32(_mm512_castsi512_si128(a)), 0);
}

static SIMD_INLINE Word elem0(const Vec<Word, 64> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Word,
                              _mm_cvtsi128_si32(_mm512_castsi512_si128(a)), 0);
}

static SIMD_INLINE Short elem0(const Vec<Short, 64> &a)
{
  SIMD_RETURN_REINTERPRET_INT(Short,
                              _mm_cvtsi128_si32(_mm512_castsi512_si128(a)), 0);
}

static SIMD_INLINE Int elem0(const Vec<Int, 64> &a)
{
  // TODO: elem0: is conversion from return type int so Int always safe?
  return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}

static SIMD_INLINE Float elem0(const Vec<Float, 64> &a)
{
  SIMD_RETURN_REINTERPRET_INT(
    Float, _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_castps_si512(a))),
    0);
}

static SIMD_INLINE Double elem0(const Vec<Double, 64> &a)
{
  const uint64_t intRes =
    _mm_cvtsi128_si64(_mm512_castsi512_si128(_mm512_castpd_si512(a)));
  Double res;
  memcpy(&res, &intRes, sizeof(res));
  return res;
}

// ---------------------------------------------------------------------------
// permute_64_16: permutation of 128-bit lanes, two sources v
// ---------------------------------------------------------------------------

// template parameter:
// - ABi (0/1): select lane i from a (0) or from b (1)
// - Ii (0..3): select lane i from index Ii in either a or b

template <int AB0, int I0, int AB1, int I1, int AB2, int I2, int AB3, int I3,
          typename T>
static SIMD_INLINE Vec<T, 64> permute_64_16(const Vec<T, 64> &a,
                                            const Vec<T, 64> &b)

{
  __m512i mask = _mm512_set_epi64(
    (AB3 << 3) | (2 * I3 + 1), (AB3 << 3) | (2 * I3), (AB2 << 3) | (2 * I2 + 1),
    (AB2 << 3) | (2 * I2), (AB1 << 3) | (2 * I1 + 1), (AB1 << 3) | (2 * I1),
    (AB0 << 3) | (2 * I0 + 1), (AB0 << 3) | (2 * I0));
  // reinterpret as Int in case T is not an integer type
  Vec<Int, 64> res = _mm512_permutex2var_epi64(
    reinterpret(a, OutputType<Int>()), mask, reinterpret(b, OutputType<Int>()));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// alignre v
// ---------------------------------------------------------------------------

// Li, Hi: lanes
// n = IMM * sizeof(T) [#bytes]
//
// input: H0 H1 H2 H3
//        L0 L1 L2 L3         NB
// ==================
// n<16:  L1 L2 L3 H0   L,H   1
//        L0 L1 L2 L3   L,H   0
// ------------------
// n<32:  L2 L3 H0 H1   L,H   2
//        L1 L2 L3 H0   L,H   1
// ------------------
// n<48:  L3 H0 H1 H2   L,H   3
//        L2 L3 H0 H1   L,H   2
// ------------------
// n<64:  H0 H1 H2 H3   L,H   4
//        L3 H0 H1 H2   L,H   3
// ------------------
// n<80:  H1 H2 H3 0    H,0   1
//        H0 H1 H2 H3   H,0   0
// ------------------
// n<96:  H2 H3 0  0    H,0   2
//        H1 H2 H3 0    H,0   1
// ------------------
// n<112: H3 0  0  0    H,0   3
//        H2 H3 0  0    H,0   2
// ------------------
// n<128: 0  0  0  0    H,0   4
//        H3 0  0  0    H,0   3

// align_64_16 v (helper for alignre)

// 16-byte lanes:          AB0 I0 AB1 I1 AB2 I2 AB3 I3
// NB=0: a0 a1 a2 a3         0 0    0 1    0 2    0 3
// NB=1: a1 a2 a3 b0         0 1    0 2    0 3    1 0
// NB=2: a2 a3 b0 b1         0 2    0 3    1 0    1 1
// NB=3: a3 b0 b1 b2         0 3    1 0    1 1    1 2
// NB=4: b0 b1 b2 b3         1 0    1 1    1 2    1 3

// special case
template <typename T>
static SIMD_INLINE Vec<T, 64> align_64_16(const Vec<T, 64> &a,
                                          const Vec<T, 64> &, Integer<0>)
{
  return a;
}

// special case
template <typename T>
static SIMD_INLINE Vec<T, 64> align_64_16(const Vec<T, 64> &,
                                          const Vec<T, 64> &b, Integer<4>)
{
  return b;
}

// general case
template <int NB, typename T>
static SIMD_INLINE Vec<T, 64> align_64_16(const Vec<T, 64> &a,
                                          const Vec<T, 64> &b, Integer<NB>)
{
  return permute_64_16<(NB > 3), (NB % 4), (NB > 2), (NB + 1) % 4, (NB > 1),
                       (NB + 2) % 4, (NB > 0), (NB + 3) % 4>(a, b);
}

// positive and in range
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> alignre(const Vec<T, 64> &h, const Vec<T, 64> &l,
                                      IsPosInRange<true, true>)
{
  const auto byteShift = IMM * sizeof(T);
  const auto laneShift = byteShift / 16;
  const Vec<T, 64> L   = (byteShift < 64) ? l : h;
  const Vec<T, 64> H =
    (byteShift < 64) ? h : setzero(OutputType<T>(), Integer<64>());
  const Vec<T, 64> ll = align_64_16(L, H, Integer<(laneShift % 4)>());
  const Vec<T, 64> hh = align_64_16(L, H, Integer<(laneShift % 4 + 1)>());
  return reinterpret(Vec<Byte, 64>(x_mm512_alignr_epi8<byteShift % 16>(
                       reinterpret(hh, OutputType<Byte>()),
                       reinterpret(ll, OutputType<Byte>()))),
                     OutputType<T>());
}

// positive and out of range
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> alignre(const Vec<T, 64> &, const Vec<T, 64> &,
                                      IsPosInRange<true, false>)
{
  return setzero(OutputType<T>(), Integer<64>());
}

// hub:
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> alignre(const Vec<T, 64> &h, const Vec<T, 64> &l)
{
  return alignre<IMM>(h, l, IsPosInGivenRange<2 * Vec<T, 64>::elements, IMM>());
}

// ---------------------------------------------------------------------------
// srle: element-wise right shift (via alignre) v
// ---------------------------------------------------------------------------

// TODO: srle: solution with byte-wise shift intrinsics instead of align?

// positive and in range
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> srle(const Vec<T, 64> &a,
                                   IsPosInRange<true, true>)
{
  return alignre<IMM>(setzero(OutputType<T>(), Integer<64>()), a);
}

// positive and out of range
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> srle(const Vec<T, 64> &,
                                   IsPosInRange<true, false>)
{
  return setzero(OutputType<T>(), Integer<64>());
}

// hub:
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> srle(const Vec<T, 64> &a)
{
  return srle<IMM>(a, IsPosInGivenRange<Vec<T, 64>::elements, IMM>());
}

// ---------------------------------------------------------------------------
// slle: element-wise left shift (via alignre) v
// ---------------------------------------------------------------------------

// TODO: slle: solution with byte-wise shift intrinsics instead of align?

// positive and in range
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> slle(const Vec<T, 64> &a,
                                   IsPosInRange<true, true>)
{
  return alignre<Vec<T, 64>::elements - IMM>(
    a, setzero(OutputType<T>(), Integer<64>()));
}

// positive and out of range
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> slle(const Vec<T, 64> &,
                                   IsPosInRange<true, false>)
{
  return setzero(OutputType<T>(), Integer<64>());
}

// hub:
// IMM: in elements
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 64> slle(const Vec<T, 64> &a)
{
  return slle<IMM>(a, IsPosInGivenRange<Vec<T, 64>::elements, IMM>());
}

// ---------------------------------------------------------------------------
// swizzle v
// ---------------------------------------------------------------------------

// ---------- swizzle aux functions -----------

// alignoff is the element-wise offset (relates to size of byte)
template <int ALIGNOFF>
static SIMD_INLINE __m512i align_shuffle_512(__m512i lo, __m512i hi,
                                             __m512i mask)
{
  static_assert(ALIGNOFF >= 0 && ALIGNOFF < 32, "");
  return x_mm512_shuffle_epi8(x_mm512_alignr_epi8<ALIGNOFF>(hi, lo), mask);
}

// swizzle_64_16: swizzling of 128-bit lanes (for swizzle) v

// each block (e.g. h2) is a 128-bit lane:
//
// example:
//
//      ----v[0]---|----v[1]---
// n=2: l0 L0 h0 H0 l1 L1 h1 H1
//      --    --    --    --
//         --    --    --    --
//  ->  l0 h0 l1 h1 L0 H0 L1 H1
//      -----------|-----------
//
//
//      ----v[0]---|----v[1]---|----v[2]---
// n=3: l0 L0 h0 H0 l1 L1 h1 H1 l2 L2 h2 H2
//      --       --       --       --
//         --       --       --       --
//            --       --       --       --
//  ->  l0 H0 h1 L2 L0 l1 H1 h2 h0 L1 l2 H2
//      -----------|-----------|-----------
//
//
//      ----v[0]---|----v[1]---|----v[2]---|----v[3]---
// n=4: l0 L0 h0 H0 l1 L1 h1 H1 l2 L2 h2 H2 l3 L3 h3 H3
//      --          --          --          --
//         --          --          --          --
//            --          --          --          --
//               --          --          --          --
//  ->  l0 l1 l2 l3 L0 L1 L2 L3 h0 h1 h2 h3 H0 H1 H2 H3
//      -----------|-----------|-----------|-----------
//
//
//      ----v[0]---|----v[1]---|----v[2]---|----v[3]---|----v[4]---
// n=5: l0 L0 h0 H0 l1 L1 h1 H1 l2 L2 h2 H2 l3 L3 h3 H3 l4 L4 h4 H4
//      --             --             --             --
//         --             --             --             --
//            --             --             --             --
//               --             --             --             --
//                  --             --             --             --
//  ->  l0 L1 h2 H3 L0 h1 H2 l4 h0 H1 l3 L4 H0 l2 L3 h4 l1 L2 h3 H4
//      -----------|-----------|-----------|-----------|-----------

// primary template
template <int N, typename T>
struct Swizzle_64_16;

// N=2
// vIn:  0 1 2 3 | 4 5 6 7
// vOut: 0 2 4 6 | 1 3 5 7
template <typename T>
struct Swizzle_64_16<2, T>
{
  static SIMD_INLINE void _swizzle_64_16(const Vec<T, 64> vIn[2],
                                         Vec<T, 64> vOut[2])
  {
    vOut[0] = permute_64_16<0, 0, 0, 2, 1, 0, 1, 2>(vIn[0], vIn[1]);
    vOut[1] = permute_64_16<0, 1, 0, 3, 1, 1, 1, 3>(vIn[0], vIn[1]);
  }
};

// N=3
// vIn:  0 1 2 3 | 4  5 6  7 | 8 9 10 11
// vTmp: 0 6 1 4 | 7 10 5  8 | 3 9  2 11
// vOut: 0 3 6 9 | 1  4 7 10 | 2 5  8 11
template <typename T>
struct Swizzle_64_16<3, T>
{
  static SIMD_INLINE void _swizzle_64_16(const Vec<T, 64> vIn[3],
                                         Vec<T, 64> vOut[3])
  {
    Vec<T, 64> vTmp[3];
    vTmp[0] = permute_64_16<0, 0, 1, 2, 0, 1, 1, 0>(vIn[0], vIn[1]);
    vTmp[1] = permute_64_16<0, 3, 1, 2, 0, 1, 1, 0>(vIn[1], vIn[2]);
    vTmp[2] = permute_64_16<0, 3, 1, 1, 0, 2, 1, 3>(vIn[0], vIn[2]);

    vOut[0] = permute_64_16<0, 0, 1, 0, 0, 1, 1, 1>(vTmp[0], vTmp[2]);
    vOut[1] = permute_64_16<0, 2, 0, 3, 1, 0, 1, 1>(vTmp[0], vTmp[1]);
    vOut[2] = permute_64_16<1, 2, 0, 2, 0, 3, 1, 3>(vTmp[1], vTmp[2]);
  }
};

// N=4
// vIn:  0 1 2  3 | 4 5 6  7 | 8  9 10 11 | 12 13 14 15
// vTmp: 0 4 1  5 | 2 6 3  7 | 8 12  9 13 | 10 14 11 15
// vOut: 0 4 8 12 | 1 5 9 13 | 2  6 10 14 |  3  7 11 15
template <typename T>
struct Swizzle_64_16<4, T>
{
  static SIMD_INLINE void _swizzle_64_16(const Vec<T, 64> vIn[4],
                                         Vec<T, 64> vOut[4])
  {
    Vec<T, 64> vTmp[4];
    vTmp[0] = permute_64_16<0, 0, 1, 0, 0, 1, 1, 1>(vIn[0], vIn[1]);
    vTmp[1] = permute_64_16<0, 2, 1, 2, 0, 3, 1, 3>(vIn[0], vIn[1]);
    vTmp[2] = permute_64_16<0, 0, 1, 0, 0, 1, 1, 1>(vIn[2], vIn[3]);
    vTmp[3] = permute_64_16<0, 2, 1, 2, 0, 3, 1, 3>(vIn[2], vIn[3]);

    vOut[0] = permute_64_16<0, 0, 0, 1, 1, 0, 1, 1>(vTmp[0], vTmp[2]);
    vOut[1] = permute_64_16<0, 2, 0, 3, 1, 2, 1, 3>(vTmp[0], vTmp[2]);
    vOut[2] = permute_64_16<0, 0, 0, 1, 1, 0, 1, 1>(vTmp[1], vTmp[3]);
    vOut[3] = permute_64_16<0, 2, 0, 3, 1, 2, 1, 3>(vTmp[1], vTmp[3]);
  }
};

// N=5
// vIn:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15 | 16 17 18 19
// vTmp: 5 10  6 11 | 1 16  3 18 | 8 13  9 14 |  7 17  4 19 |  0 15  2 12
// vOut: 0  5 10 15 | 1  6 11 16 | 2  7 12 17 |  3  8 13 18 |  4  9 14 19
template <typename T>
struct Swizzle_64_16<5, T>
{
  static SIMD_INLINE void _swizzle_64_16(const Vec<T, 64> vIn[5],
                                         Vec<T, 64> vOut[5])
  {
    Vec<T, 64> vTmp[5];
    vTmp[0] = permute_64_16<0, 1, 1, 2, 0, 2, 1, 3>(vIn[1], vIn[2]);
    vTmp[1] = permute_64_16<0, 1, 1, 0, 0, 3, 1, 2>(vIn[0], vIn[4]);
    vTmp[2] = permute_64_16<0, 0, 1, 1, 0, 1, 1, 2>(vIn[2], vIn[3]);
    vTmp[3] = permute_64_16<0, 3, 1, 1, 0, 0, 1, 3>(vIn[1], vIn[4]);
    vTmp[4] = permute_64_16<0, 0, 1, 3, 0, 2, 1, 0>(vIn[0], vIn[3]);

    vOut[0] = permute_64_16<1, 0, 0, 0, 0, 1, 1, 1>(vTmp[0], vTmp[4]);
    vOut[1] = permute_64_16<1, 0, 0, 2, 0, 3, 1, 1>(vTmp[0], vTmp[1]);
    vOut[2] = permute_64_16<1, 2, 0, 0, 1, 3, 0, 1>(vTmp[3], vTmp[4]);
    vOut[3] = permute_64_16<0, 2, 1, 0, 1, 1, 0, 3>(vTmp[1], vTmp[2]);
    vOut[4] = permute_64_16<1, 2, 0, 2, 0, 3, 1, 3>(vTmp[2], vTmp[3]);
  }
};

// swizzle lanes (for implementation of swizzle functions)
template <int N, typename T>
static SIMD_INLINE void swizzle_64_16(const Vec<T, 64> vIn[N],
                                      Vec<T, 64> vOut[N])
{
  Swizzle_64_16<N, T>::_swizzle_64_16(vIn, vOut);
}

// ---------- swizzle (AoS to SoA) ----------

// 01. Apr 23 (Jonas Keller): switched from using tag dispatching to using
// enable_if SFINAE, which allows more cases with the same implementation
// to be combined

// -------------------- n = 1 --------------------

// all types
template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 64>[1], Integer<1>)
{
  // v remains unchanged
}

// -------------------- n = 2 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF((sizeof(T) <= 2 && std::is_integral<T>::value))>
static SIMD_INLINE void swizzle(Vec<T, 64> v[2], Integer<2>)
{
  Vec<T, 64> vs[2];
  swizzle_64_16<2>(v, vs);
  const __m512i mask = x_mm512_duplicate_si128(get_swizzle_mask<2, T>());
  const __m512i s[2] = {
    x_mm512_shuffle_epi8(vs[0], mask),
    x_mm512_shuffle_epi8(vs[1], mask),
  };
  v[0] = _mm512_unpacklo_epi64(s[0], s[1]);
  v[1] = _mm512_unpackhi_epi64(s[0], s[1]);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[2], Integer<2>)
{
  const Vec<Float, 64> vFloat[2] = {
    reinterpret(v[0], OutputType<Float>()),
    reinterpret(v[1], OutputType<Float>()),
  };
  Vec<Float, 64> vs[2];
  swizzle_64_16<2>(vFloat, vs);
  const __m512 v0tmp = vs[0];
  const __m512 v1tmp = vs[1];
  const Vec<Float, 64> vOut0 =
    _mm512_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(2, 0, 2, 0));
  const Vec<Float, 64> vOut1 =
    _mm512_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(3, 1, 3, 1));
  v[0] = reinterpret(vOut0, OutputType<T>());
  v[1] = reinterpret(vOut1, OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[2], Integer<2>)
{
  const Vec<Double, 64> vDouble[2] = {
    reinterpret(v[0], OutputType<Double>()),
    reinterpret(v[1], OutputType<Double>()),
  };
  Vec<Double, 64> vs[2];
  swizzle_64_16<2>(vDouble, vs);
  const __m512d v0tmp = vs[0];
  const __m512d v1tmp = vs[1];
  const Vec<Double, 64> vOut0 =
    _mm512_shuffle_pd(v0tmp, v1tmp, _MM_SHUFFLE2(0, 0));
  const Vec<Double, 64> vOut1 =
    _mm512_shuffle_pd(v0tmp, v1tmp, _MM_SHUFFLE2(1, 1));
  v[0] = reinterpret(vOut0, OutputType<T>());
  v[1] = reinterpret(vOut1, OutputType<T>());
}

// -------------------- n = 3 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF((sizeof(T) <= 2 && std::is_integral<T>::value))>
static SIMD_INLINE void swizzle(Vec<T, 64> v[3], Integer<3>)
{
  Vec<T, 64> vs[3];
  swizzle_64_16<3>(v, vs);
  __m512i mask = x_mm512_duplicate_si128(get_swizzle_mask<3, T>());
  __m512i s0   = align_shuffle_512<0>(vs[0], vs[1], mask);
  __m512i s1   = align_shuffle_512<12>(vs[0], vs[1], mask);
  __m512i s2   = align_shuffle_512<8>(vs[1], vs[2], mask);
  __m512i s3   = align_shuffle_512<4>(vs[2], _mm512_undefined_epi32(), mask);
  __m512i l01  = _mm512_unpacklo_epi32(s0, s1);
  __m512i h01  = _mm512_unpackhi_epi32(s0, s1);
  __m512i l23  = _mm512_unpacklo_epi32(s2, s3);
  __m512i h23  = _mm512_unpackhi_epi32(s2, s3);
  v[0]         = _mm512_unpacklo_epi64(l01, l23);
  v[1]         = _mm512_unpackhi_epi64(l01, l23);
  v[2]         = _mm512_unpacklo_epi64(h01, h23);
}

// 32 bit types
// from Stan Melax: "3D Vector Normalization..."
// https://software.intel.com/en-us/articles/3d-vector-normalization-using-512-bit-intel-advanced-vector-extensions-intel-avx
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[3], Integer<3>)
{
  const Vec<Float, 64> vFloat[3] = {
    reinterpret(v[0], OutputType<Float>()),
    reinterpret(v[1], OutputType<Float>()),
    reinterpret(v[2], OutputType<Float>()),
  };
  Vec<Float, 64> vs[3];
  swizzle_64_16<3>(vFloat, vs);
  // x0y0z0x1 = v[0]
  // y1z1x2y2 = v[1]
  // z2x3y3z3 = v[2]
  __m512 x2y2x3y3 = _mm512_shuffle_ps(vs[1], vs[2], _MM_SHUFFLE(2, 1, 3, 2));
  __m512 y0z0y1z1 = _mm512_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(1, 0, 2, 1));
  // x0x1x2x3
  const Vec<Float, 64> vOut0 =
    _mm512_shuffle_ps(vs[0], x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0));
  // y0y1y2y3
  const Vec<Float, 64> vOut1 =
    _mm512_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0));
  // z0z1z2z3
  const Vec<Float, 64> vOut2 =
    _mm512_shuffle_ps(y0z0y1z1, vs[2], _MM_SHUFFLE(3, 0, 3, 1));
  v[0] = reinterpret(vOut0, OutputType<T>());
  v[1] = reinterpret(vOut1, OutputType<T>());
  v[2] = reinterpret(vOut2, OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[3], Integer<3>)
{
  const Vec<Double, 64> vDouble[3] = {
    reinterpret(v[0], OutputType<Double>()),
    reinterpret(v[1], OutputType<Double>()),
    reinterpret(v[2], OutputType<Double>()),
  };
  Vec<Double, 64> vs[3];
  swizzle_64_16<3>(vDouble, vs);
  const __m512d v0tmp = vs[0];
  const __m512d v1tmp = vs[1];
  const __m512d v2tmp = vs[2];
  const Vec<Double, 64> vOut0 =
    _mm512_shuffle_pd(v0tmp, v1tmp, _MM_SHUFFLE2(0, 0));
  const Vec<Double, 64> vOut1 =
    _mm512_shuffle_pd(v0tmp, v1tmp, _MM_SHUFFLE2(1, 1));
  const Vec<Double, 64> vOut2 =
    _mm512_shuffle_pd(v1tmp, v2tmp, _MM_SHUFFLE2(0, 0));
  v[0] = reinterpret(vOut0, OutputType<T>());
  v[1] = reinterpret(vOut1, OutputType<T>());
  v[2] = reinterpret(vOut2, OutputType<T>());
}

// -------------------- n = 4 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF((sizeof(T) <= 2 && std::is_integral<T>::value))>
static SIMD_INLINE void swizzle(Vec<T, 64> v[4], Integer<4>)
{
  Vec<T, 64> vs[4];
  swizzle_64_16<4>(v, vs);
  __m512i mask = x_mm512_duplicate_si128(get_swizzle_mask<4, T>());
  __m512i s[4];
  for (int j = 0; j < 4; j++) s[j] = x_mm512_shuffle_epi8(vs[j], mask);
  __m512i l01 = _mm512_unpacklo_epi32(s[0], s[1]);
  __m512i h01 = _mm512_unpackhi_epi32(s[0], s[1]);
  __m512i l23 = _mm512_unpacklo_epi32(s[2], s[3]);
  __m512i h23 = _mm512_unpackhi_epi32(s[2], s[3]);
  v[0]        = _mm512_unpacklo_epi64(l01, l23);
  v[1]        = _mm512_unpackhi_epi64(l01, l23);
  v[2]        = _mm512_unpacklo_epi64(h01, h23);
  v[3]        = _mm512_unpackhi_epi64(h01, h23);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[4], Integer<4>)
{
  Vec<Int, 64> vInt[4];
  for (int i = 0; i < 4; i++) vInt[i] = reinterpret(v[i], OutputType<Int>());
  Vec<Int, 64> vs[4];
  swizzle_64_16<4>(vInt, vs);
  const __m512i s[4] = {
    _mm512_unpacklo_epi32(vs[0], vs[1]),
    _mm512_unpackhi_epi32(vs[0], vs[1]),
    _mm512_unpacklo_epi32(vs[2], vs[3]),
    _mm512_unpackhi_epi32(vs[2], vs[3]),
  };
  const Vec<Int, 64> vOut[4] = {
    _mm512_unpacklo_epi64(s[0], s[2]),
    _mm512_unpackhi_epi64(s[0], s[2]),
    _mm512_unpacklo_epi64(s[1], s[3]),
    _mm512_unpackhi_epi64(s[1], s[3]),
  };
  for (int i = 0; i < 4; i++) v[i] = reinterpret(vOut[i], OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[4], Integer<4>)
{
  Vec<Double, 64> vDouble[4];
  for (int i = 0; i < 4; i++)
    vDouble[i] = reinterpret(v[i], OutputType<Double>());
  Vec<Double, 64> vs[4];
  swizzle_64_16<4>(vDouble, vs);
  const __m512d s[4] = {
    _mm512_unpacklo_pd(vs[0], vs[1]),
    _mm512_unpackhi_pd(vs[0], vs[1]),
    _mm512_unpacklo_pd(vs[2], vs[3]),
    _mm512_unpackhi_pd(vs[2], vs[3]),
  };
  const Vec<Double, 64> vOut[4] = {
    _mm512_unpacklo_pd(s[0], s[2]),
    _mm512_unpackhi_pd(s[0], s[2]),
    _mm512_unpacklo_pd(s[1], s[3]),
    _mm512_unpackhi_pd(s[1], s[3]),
  };
  for (int i = 0; i < 4; i++) v[i] = reinterpret(vOut[i], OutputType<T>());
}

// -------------------- n = 5 --------------------

// 8 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) == 1 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 64> v[5], Integer<5>)
{
  Vec<T, 64> vs[5];
  swizzle_64_16<5>(v, vs);
  const __m512i mask = x_mm512_duplicate_si128(get_swizzle_mask<5, T>());
  const __m512i s[8] = {
    align_shuffle_512<0>(vs[0], vs[1], mask),
    align_shuffle_512<10>(vs[0], vs[1], mask),
    align_shuffle_512<4>(vs[1], vs[2], mask),
    align_shuffle_512<14>(vs[1], vs[2], mask),
    align_shuffle_512<8>(vs[2], vs[3], mask),
    align_shuffle_512<2>(vs[3], vs[4], mask),
    align_shuffle_512<12>(vs[3], vs[4], mask),
    align_shuffle_512<6>(vs[4], _mm512_undefined_epi32(), mask),
  };
  __m512i l01     = x_mm512_unpacklo_epi16(s[0], s[1]);
  __m512i h01     = x_mm512_unpackhi_epi16(s[0], s[1]);
  __m512i l23     = x_mm512_unpacklo_epi16(s[2], s[3]);
  __m512i h23     = x_mm512_unpackhi_epi16(s[2], s[3]);
  __m512i l45     = x_mm512_unpacklo_epi16(s[4], s[5]);
  __m512i h45     = x_mm512_unpackhi_epi16(s[4], s[5]);
  __m512i l67     = x_mm512_unpacklo_epi16(s[6], s[7]);
  __m512i h67     = x_mm512_unpackhi_epi16(s[6], s[7]);
  __m512i ll01l23 = _mm512_unpacklo_epi32(l01, l23);
  __m512i hl01l23 = _mm512_unpackhi_epi32(l01, l23);
  __m512i ll45l67 = _mm512_unpacklo_epi32(l45, l67);
  __m512i hl45l67 = _mm512_unpackhi_epi32(l45, l67);
  __m512i lh01h23 = _mm512_unpacklo_epi32(h01, h23);
  __m512i lh45h67 = _mm512_unpacklo_epi32(h45, h67);
  v[0]            = _mm512_unpacklo_epi64(ll01l23, ll45l67);
  v[1]            = _mm512_unpackhi_epi64(ll01l23, ll45l67);
  v[2]            = _mm512_unpacklo_epi64(hl01l23, hl45l67);
  v[3]            = _mm512_unpackhi_epi64(hl01l23, hl45l67);
  v[4]            = _mm512_unpacklo_epi64(lh01h23, lh45h67);
}

// 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) == 2 && std::is_integral<T>::value),
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[5], Integer<5>)
{
  Vec<T, 64> vs[5];
  swizzle_64_16<5>(v, vs);
  const __m512i mask = x_mm512_duplicate_si128(get_swizzle_mask<5, T>());
  const __m512i s[8] = {
    align_shuffle_512<0>(vs[0], vs[1], mask),
    align_shuffle_512<6>(vs[0], vs[1], mask),
    align_shuffle_512<4>(vs[1], vs[2], mask),
    align_shuffle_512<10>(vs[1], vs[2], mask),
    align_shuffle_512<8>(vs[2], vs[3], mask),
    align_shuffle_512<14>(vs[2], vs[3], mask),
    align_shuffle_512<12>(vs[3], vs[4], mask),
    align_shuffle_512<2>(vs[4], _mm512_undefined_epi32(), mask),
  };
  __m512i l02 = _mm512_unpacklo_epi32(s[0], s[2]);
  __m512i h02 = _mm512_unpackhi_epi32(s[0], s[2]);
  __m512i l13 = _mm512_unpacklo_epi32(s[1], s[3]);
  __m512i l46 = _mm512_unpacklo_epi32(s[4], s[6]);
  __m512i h46 = _mm512_unpackhi_epi32(s[4], s[6]);
  __m512i l57 = _mm512_unpacklo_epi32(s[5], s[7]);
  v[0]        = _mm512_unpacklo_epi64(l02, l46);
  v[1]        = _mm512_unpackhi_epi64(l02, l46);
  v[2]        = _mm512_unpacklo_epi64(h02, h46);
  v[3]        = _mm512_unpacklo_epi64(l13, l57);
  v[4]        = _mm512_unpackhi_epi64(l13, l57);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[5], Integer<5>)
{
  Vec<Int, 64> vInt[5];
  for (int i = 0; i < 5; i++) {
    vInt[i] = reinterpret(v[i], OutputType<Int>());
  }
  Vec<Int, 64> vs[5];
  swizzle_64_16<5>(vInt, vs);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  // v[0]: 0 1 2 3
  // v[1]: 4 x x x
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                   x x x   x
  // 5 6 7 8
  __m512i s2 = x_mm512_alignr_epi8<4>(vs[2], vs[1]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                             x  x  x    x
  // 9 x x x
  __m512i s3 = x_mm512_alignr_epi8<4>(vs[3], vs[2]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                x  x    x  x
  // 10 11 12 13
  __m512i s4 = x_mm512_alignr_epi8<8>(vs[3], vs[2]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                              x  x    x  x
  // 14 x x x
  __m512i s5 = x_mm512_alignr_epi8<8>(vs[4], vs[3]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                                 X    X  X  X
  // 15 16 17 18
  __m512i s6 = x_mm512_alignr_epi8<12>(vs[4], vs[3]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                                               X X X X
  // 19 x x x
  __m512i s7 = x_mm512_alignr_epi8<12>(vs[0], vs[4]);
  // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
  __m512i l02 = _mm512_unpacklo_epi32(vs[0], s2);
  __m512i h02 = _mm512_unpackhi_epi32(vs[0], s2);
  // 4 x x x / 9 x x x -> 4 9 x x
  __m512i l13 = _mm512_unpacklo_epi32(vs[1], s3);
  // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
  __m512i l46 = _mm512_unpacklo_epi32(s4, s6);
  __m512i h46 = _mm512_unpackhi_epi32(s4, s6);
  // 14 x x x / 19 x x x -> 14 19 x x
  __m512i l57                = _mm512_unpacklo_epi32(s5, s7);
  const Vec<Int, 64> vOut[5] = {
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    _mm512_unpacklo_epi64(l02, l46),
    _mm512_unpackhi_epi64(l02, l46),
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    _mm512_unpacklo_epi64(h02, h46),
    _mm512_unpackhi_epi64(h02, h46),
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    _mm512_unpacklo_epi64(l13, l57),
  };
  for (int i = 0; i < 5; i++) { v[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void, typename = void>
static SIMD_INLINE void swizzle(Vec<T, 64> v[5], Integer<5>)
{
  Vec<Double, 64> vDouble[5];
  for (int i = 0; i < 5; i++) {
    vDouble[i] = reinterpret(v[i], OutputType<Double>());
  }
  // TODO
}

// ---------------------------------------------------------------------------
// comparison functions
// ---------------------------------------------------------------------------

// 28. Mar 23 (Jonas Keller): checked the constants for _mm512_cmp_ps_mask in
// the Float comparison functions, they match the implementation of the SSE
// versions (see cmpps in Intel manual) and added corresponding comments

// ---------------------------------------------------------------------------
// compare < v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> cmplt(const Vec<Byte, 64> &a,
                                       const Vec<Byte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmplt_epu8_mask(a, b));
}

static SIMD_INLINE Vec<SignedByte, 64> cmplt(const Vec<SignedByte, 64> &a,
                                             const Vec<SignedByte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, b));
}

static SIMD_INLINE Vec<Word, 64> cmplt(const Vec<Word, 64> &a,
                                       const Vec<Word, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmplt_epu16_mask(a, b));
}

static SIMD_INLINE Vec<Short, 64> cmplt(const Vec<Short, 64> &a,
                                        const Vec<Short, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmplt_epi16_mask(a, b));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> cmplt(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(cmplt(a.lo(), b.lo()), cmplt(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> cmplt(const Vec<Int, 64> &a,
                                      const Vec<Int, 64> &b)
{
  return x_mm512_movm_epi32(_mm512_cmplt_epi32_mask(a, b));
}

static SIMD_INLINE Vec<Float, 64> cmplt(const Vec<Float, 64> &a,
                                        const Vec<Float, 64> &b)
{
  // same constant as in implementation of _mm_cmplt_ps (see cmpps instruction
  // in Intel manual)
  return _mm512_castsi512_ps(
    x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_LT_OS)));
}

static SIMD_INLINE Vec<Double, 64> cmplt(const Vec<Double, 64> &a,
                                         const Vec<Double, 64> &b)
{
  // same constant as in implementation of _mm_cmplt_pd (see cmppd instruction
  // in Intel manual)
  return _mm512_castsi512_pd(
    x_mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_LT_OS)));
}

// ---------------------------------------------------------------------------
// compare <= v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> cmple(const Vec<Byte, 64> &a,
                                       const Vec<Byte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmple_epu8_mask(a, b));
}

static SIMD_INLINE Vec<SignedByte, 64> cmple(const Vec<SignedByte, 64> &a,
                                             const Vec<SignedByte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmple_epi8_mask(a, b));
}

static SIMD_INLINE Vec<Word, 64> cmple(const Vec<Word, 64> &a,
                                       const Vec<Word, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmple_epu16_mask(a, b));
}

static SIMD_INLINE Vec<Short, 64> cmple(const Vec<Short, 64> &a,
                                        const Vec<Short, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmple_epi16_mask(a, b));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> cmple(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(cmple(a.lo(), b.lo()), cmple(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> cmple(const Vec<Int, 64> &a,
                                      const Vec<Int, 64> &b)
{
  return x_mm512_movm_epi32(_mm512_cmple_epi32_mask(a, b));
}

static SIMD_INLINE Vec<Float, 64> cmple(const Vec<Float, 64> &a,
                                        const Vec<Float, 64> &b)
{
  // same constant as in implementation of _mm_cmple_ps (see cmpps instruction
  // in Intel manual)
  return _mm512_castsi512_ps(
    x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_LE_OS)));
}

static SIMD_INLINE Vec<Double, 64> cmple(const Vec<Double, 64> &a,
                                         const Vec<Double, 64> &b)
{
  // same constant as in implementation of _mm_cmple_pd (see cmppd instruction
  // in Intel manual)
  return _mm512_castsi512_pd(
    x_mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_LE_OS)));
}

// ---------------------------------------------------------------------------
// compare == v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> cmpeq(const Vec<Byte, 64> &a,
                                       const Vec<Byte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpeq_epu8_mask(a, b));
}

static SIMD_INLINE Vec<SignedByte, 64> cmpeq(const Vec<SignedByte, 64> &a,
                                             const Vec<SignedByte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpeq_epi8_mask(a, b));
}

static SIMD_INLINE Vec<Word, 64> cmpeq(const Vec<Word, 64> &a,
                                       const Vec<Word, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpeq_epu16_mask(a, b));
}

static SIMD_INLINE Vec<Short, 64> cmpeq(const Vec<Short, 64> &a,
                                        const Vec<Short, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpeq_epi16_mask(a, b));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> cmpeq(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(cmpeq(a.lo(), b.lo()), cmpeq(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> cmpeq(const Vec<Int, 64> &a,
                                      const Vec<Int, 64> &b)
{
  return x_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(a, b));
}

static SIMD_INLINE Vec<Float, 64> cmpeq(const Vec<Float, 64> &a,
                                        const Vec<Float, 64> &b)
{
  // same constant as in implementation of _mm_cmpeq_ps (see cmpps instruction
  // in Intel manual)
  return _mm512_castsi512_ps(
    x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ)));
}

static SIMD_INLINE Vec<Double, 64> cmpeq(const Vec<Double, 64> &a,
                                         const Vec<Double, 64> &b)
{
  // same constant as in implementation of _mm_cmpeq_pd (see cmppd instruction
  // in Intel manual)
  return _mm512_castsi512_pd(
    x_mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_EQ_OQ)));
}

// ---------------------------------------------------------------------------
// compare > v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> cmpgt(const Vec<Byte, 64> &a,
                                       const Vec<Byte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpgt_epu8_mask(a, b));
}

static SIMD_INLINE Vec<SignedByte, 64> cmpgt(const Vec<SignedByte, 64> &a,
                                             const Vec<SignedByte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(a, b));
}

static SIMD_INLINE Vec<Word, 64> cmpgt(const Vec<Word, 64> &a,
                                       const Vec<Word, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpgt_epu16_mask(a, b));
}

static SIMD_INLINE Vec<Short, 64> cmpgt(const Vec<Short, 64> &a,
                                        const Vec<Short, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpgt_epi16_mask(a, b));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> cmpgt(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(cmpgt(a.lo(), b.lo()), cmpgt(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> cmpgt(const Vec<Int, 64> &a,
                                      const Vec<Int, 64> &b)
{
  return x_mm512_movm_epi32(_mm512_cmpgt_epi32_mask(a, b));
}

static SIMD_INLINE Vec<Float, 64> cmpgt(const Vec<Float, 64> &a,
                                        const Vec<Float, 64> &b)
{
  // same constant as in implementation of _mm_cmplt_ps (see cmpps instruction
  // in Intel manual), except this is > instead of <
  return _mm512_castsi512_ps(
    x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_GT_OS)));
}

static SIMD_INLINE Vec<Double, 64> cmpgt(const Vec<Double, 64> &a,
                                         const Vec<Double, 64> &b)
{
  // same constant as in implementation of _mm_cmplt_pd (see cmppd instruction
  // in Intel manual), except this is > instead of <
  return _mm512_castsi512_pd(
    x_mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_GT_OS)));
}

// ---------------------------------------------------------------------------
// compare >= v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> cmpge(const Vec<Byte, 64> &a,
                                       const Vec<Byte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpge_epu8_mask(a, b));
}

static SIMD_INLINE Vec<SignedByte, 64> cmpge(const Vec<SignedByte, 64> &a,
                                             const Vec<SignedByte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpge_epi8_mask(a, b));
}

static SIMD_INLINE Vec<Word, 64> cmpge(const Vec<Word, 64> &a,
                                       const Vec<Word, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpge_epu16_mask(a, b));
}

static SIMD_INLINE Vec<Short, 64> cmpge(const Vec<Short, 64> &a,
                                        const Vec<Short, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpge_epi16_mask(a, b));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> cmpge(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(cmpge(a.lo(), b.lo()), cmpge(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> cmpge(const Vec<Int, 64> &a,
                                      const Vec<Int, 64> &b)
{
  return x_mm512_movm_epi32(_mm512_cmpge_epi32_mask(a, b));
}

static SIMD_INLINE Vec<Float, 64> cmpge(const Vec<Float, 64> &a,
                                        const Vec<Float, 64> &b)
{
  // same constant as in implementation of _mm_cmple_ps (see cmpps instruction
  // in Intel manual), except this is >= instead of <=
  return _mm512_castsi512_ps(
    x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_GE_OS)));
}

static SIMD_INLINE Vec<Double, 64> cmpge(const Vec<Double, 64> &a,
                                         const Vec<Double, 64> &b)
{
  // same constant as in implementation of _mm_cmple_pd (see cmppd instruction
  // in Intel manual), except this is >= instead of <=
  return _mm512_castsi512_pd(
    x_mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_GE_OS)));
}

// ---------------------------------------------------------------------------
// compare != v
// ---------------------------------------------------------------------------

// https://stackoverflow.com/questions/48099006/
// different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> cmpneq(const Vec<Byte, 64> &a,
                                        const Vec<Byte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpneq_epu8_mask(a, b));
}

static SIMD_INLINE Vec<SignedByte, 64> cmpneq(const Vec<SignedByte, 64> &a,
                                              const Vec<SignedByte, 64> &b)
{
  return _mm512_movm_epi8(_mm512_cmpneq_epi8_mask(a, b));
}

static SIMD_INLINE Vec<Word, 64> cmpneq(const Vec<Word, 64> &a,
                                        const Vec<Word, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpneq_epu16_mask(a, b));
}

static SIMD_INLINE Vec<Short, 64> cmpneq(const Vec<Short, 64> &a,
                                         const Vec<Short, 64> &b)
{
  return _mm512_movm_epi16(_mm512_cmpneq_epi16_mask(a, b));
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> cmpneq(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(cmpneq(a.lo(), b.lo()), cmpneq(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Int, 64> cmpneq(const Vec<Int, 64> &a,
                                       const Vec<Int, 64> &b)
{
  return x_mm512_movm_epi32(_mm512_cmpneq_epi32_mask(a, b));
}

static SIMD_INLINE Vec<Float, 64> cmpneq(const Vec<Float, 64> &a,
                                         const Vec<Float, 64> &b)
{
  // same constant as in implementation of _mm_cmpneq_ps (see cmpps instruction
  // in Intel manual)
  return _mm512_castsi512_ps(
    x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ)));
}

static SIMD_INLINE Vec<Double, 64> cmpneq(const Vec<Double, 64> &a,
                                          const Vec<Double, 64> &b)
{
  // same constant as in implementation of _mm_cmpneq_pd (see cmppd instruction
  // in Intel manual)
  return _mm512_castsi512_pd(
    x_mm512_movm_epi64(_mm512_cmp_pd_mask(a, b, _CMP_NEQ_OQ)));
}

// ---------------------------------------------------------------------------
// ifelse v
// ---------------------------------------------------------------------------

// 10. Apr 23 (Jonas Keller): made two versions of ifelse, one for 8 and 16 bit
// data types, and one for 32 and larger data types, so that for the latter
// the blendv instruction can be used even if avx512bw is not available

// NOTE: only works if cond elements are all 1-bits or all 0-bits

// version for 8 and 16 bit data types
template <typename T, SIMD_ENABLE_IF(sizeof(T) <= 2)>
static SIMD_INLINE Vec<T, 64> ifelse(const Vec<T, 64> &cond,
                                     const Vec<T, 64> &trueVal,
                                     const Vec<T, 64> &falseVal)
{
  // TODO: _mm512_movepi8_mask is slower than _mm512_or_si512, _mm512_and_si512
  // or _mm512_andnot_si512 according to the Intel Intrinsics Guide, maybe use
  // the non-avx512bw workaround always?
  // since _mm512_and_si512 and _mm512_andnot_si512 could potentially be
  // executed in parallel, that might be faster
#ifdef __AVX512BW__
  // cond -> __mask64
  const __mmask64 condReg =
    _mm512_movepi8_mask(reinterpret(cond, OutputType<Byte>()));
  // explicitly cast to __m512i to avoid compiler error with -O0
  const __m512i trueReg   = (__m512i) reinterpret(trueVal, OutputType<Byte>());
  const __m512i falseReg  = (__m512i) reinterpret(falseVal, OutputType<Byte>());
  const Vec<Byte, 64> res = _mm512_mask_blend_epi8(condReg, falseReg, trueReg);
#else
  const Vec<Byte, 64> res = _mm512_or_si512(
    _mm512_and_si512(reinterpret(cond, OutputType<Byte>()),
                     reinterpret(trueVal, OutputType<Byte>())),
    _mm512_andnot_si512(reinterpret(cond, OutputType<Byte>()),
                        reinterpret(falseVal, OutputType<Byte>())));
#endif
  return reinterpret(res, OutputType<T>());
}

// version for 32 bit and larger data types
template <typename T, SIMD_ENABLE_IF(sizeof(T) > 2), typename = void>
static SIMD_INLINE Vec<T, 64> ifelse(const Vec<T, 64> &cond,
                                     const Vec<T, 64> &trueVal,
                                     const Vec<T, 64> &falseVal)
{
  // TODO: _mm512_movepi32_mask is slower than _mm512_or_si512, _mm512_and_si512
  // or _mm512_andnot_si512 according to the Intel Intrinsics Guide, maybe use
  // the non-avx512dq workaround always?
  // since _mm512_and_si512 and _mm512_andnot_si512 could potentially be
  // executed in parallel, that might be faster
#ifdef __AVX512DQ__
  // cond -> __mmask16
  const __mmask16 condReg =
    _mm512_movepi32_mask(reinterpret(cond, OutputType<Int>()));
  // explicitly cast to __m512i to avoid compiler error with -O0
  const __m512i trueReg  = (__m512i) reinterpret(trueVal, OutputType<Int>());
  const __m512i falseReg = (__m512i) reinterpret(falseVal, OutputType<Int>());
  const Vec<Int, 64> res = _mm512_mask_blend_epi32(condReg, falseReg, trueReg);
#else
  const Vec<Int, 64> res = _mm512_or_si512(
    _mm512_and_si512(reinterpret(cond, OutputType<Int>()),
                     reinterpret(trueVal, OutputType<Int>())),
    _mm512_andnot_si512(reinterpret(cond, OutputType<Int>()),
                        reinterpret(falseVal, OutputType<Int>())));
#endif
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// and v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> and_(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  // reinterpret as byte for float and double versions
  const Vec<Byte, 64> res = _mm512_and_si512(
    reinterpret(a, OutputType<Byte>()), reinterpret(b, OutputType<Byte>()));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// or v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> or_(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  // reinterpret as byte for float and double versions
  const Vec<Byte, 64> res = _mm512_or_si512(reinterpret(a, OutputType<Byte>()),
                                            reinterpret(b, OutputType<Byte>()));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// andnot v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> andnot(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  // reinterpret as byte for float and double versions
  const Vec<Byte, 64> res = _mm512_andnot_si512(
    reinterpret(a, OutputType<Byte>()), reinterpret(b, OutputType<Byte>()));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// xor v
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 64> xor_(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  // reinterpret as byte for float and double versions
  const Vec<Byte, 64> res = _mm512_xor_si512(
    reinterpret(a, OutputType<Byte>()), reinterpret(b, OutputType<Byte>()));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// not v
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 64> not_(const Vec<T, 64> &a)
{
  // reinterpret as byte for float and double versions
  // from Agner Fog's VCL vectori256.h operator ~
  const Vec<Byte, 64> res =
    _mm512_xor_si512(reinterpret(a, OutputType<Byte>()), _mm512_set1_epi32(-1));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// avg: average with rounding down v
// ---------------------------------------------------------------------------

#ifdef __AVX512BW__

static SIMD_INLINE Vec<Byte, 64> avg(const Vec<Byte, 64> &a,
                                     const Vec<Byte, 64> &b)
{
  return _mm512_avg_epu8(a, b);
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<SignedByte, 64> avg(const Vec<SignedByte, 64> &a,
                                           const Vec<SignedByte, 64> &b)
{
  // from Agner Fog's VCL vectori128.h
  __m512i signbit = _mm512_set1_epi32(0x80808080);
  __m512i a1      = _mm512_xor_si512(a, signbit); // add 0x80
  __m512i b1      = _mm512_xor_si512(b, signbit); // add 0x80
  __m512i m1      = _mm512_avg_epu8(a1, b1);      // unsigned avg
  return _mm512_xor_si512(m1, signbit);           // sub 0x80
}

static SIMD_INLINE Vec<Word, 64> avg(const Vec<Word, 64> &a,
                                     const Vec<Word, 64> &b)
{
  return _mm512_avg_epu16(a, b);
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<Short, 64> avg(const Vec<Short, 64> &a,
                                      const Vec<Short, 64> &b)
{
  // from Agner Fog's VCL vectori128.h
  __m512i signbit = _mm512_set1_epi32(0x80008000);
  __m512i a1      = _mm512_xor_si512(a, signbit); // add 0x8000
  __m512i b1      = _mm512_xor_si512(b, signbit); // add 0x8000
  __m512i m1      = _mm512_avg_epu16(a1, b1);     // unsigned avg
  return _mm512_xor_si512(m1, signbit);           // sub 0x8000
}

#else

// non-avx512bw workaround
template <typename T>
static SIMD_INLINE Vec<T, 64> avg(const Vec<T, 64> &a, const Vec<T, 64> &b)
{
  return Vec<T, 64>(avg(a.lo(), b.lo()), avg(a.hi(), b.hi()));
}

#endif

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<Int, 64> avg(const Vec<Int, 64> &a,
                                    const Vec<Int, 64> &b)
{
  Vec<Int, 64> one = set1(Int(1), Integer<64>()), as, bs, lsb;
  lsb              = and_(or_(a, b), one);
  as               = srai<1>(a);
  bs               = srai<1>(b);
  return add(lsb, add(as, bs));
}

// NOTE: Float version doesn't round!
static SIMD_INLINE Vec<Float, 64> avg(const Vec<Float, 64> &a,
                                      const Vec<Float, 64> &b)
{
  __m512 half = _mm512_set1_ps(0.5f);
  return _mm512_mul_ps(_mm512_add_ps(a, b), half);
}

// NOTE: Double version doesn't round!
static SIMD_INLINE Vec<Double, 64> avg(const Vec<Double, 64> &a,
                                       const Vec<Double, 64> &b)
{
  __m512d half = _mm512_set1_pd(0.5);
  return _mm512_mul_pd(_mm512_add_pd(a, b), half);
}

// ---------------------------------------------------------------------------
// test_all_zeros v
// ---------------------------------------------------------------------------

static SIMD_INLINE int x_mm512_mask2int(__mmask16 k1)
{
  // not supported by gcc 5.4
  // return _mm512_mask2int(k1);
  // workaround from https://gcc.gnu.org/ml/gcc-patches/2017-04/msg00374.html
  return (int) k1;
}

// all integer versions
template <typename T>
static SIMD_INLINE int test_all_zeros(const Vec<T, 64> &a)
{
  // m[i] = (a[i] & a[i] != 0) = (a[i] != 0)
  __mmask16 m = _mm512_test_epi32_mask(a, a);
  return (x_mm512_mask2int(m) == 0);
}

// float version
// note: contrary to IEEE 754, this function considers -0.0f to be negative
static SIMD_INLINE int test_all_zeros(const Vec<Float, 64> &a)
{
  return test_all_zeros(reinterpret(a, OutputType<Int>()));
}

// double version
// note: contrary to IEEE 754, this function considers -0.0f to be negative
static SIMD_INLINE int test_all_zeros(const Vec<Double, 64> &a)
{
  return test_all_zeros(reinterpret(a, OutputType<Int>()));
}

// ---------------------------------------------------------------------------
// test_all_ones v
// ---------------------------------------------------------------------------

// description of testn intrinsics was not clear, chosen other way
// note: contrary to IEEE 754, this function considers -0.0f to be negative
template <typename T>
static SIMD_INLINE int test_all_ones(const Vec<T, 64> &a)
{
  return test_all_zeros(not_(a));
}

// ---------------------------------------------------------------------------
// reverse
// ---------------------------------------------------------------------------

// 20. Mar 23 (Jonas Keller): implemented reverse masks as functions instead of
// as arrays that need to be loaded first

template <int Bytes>
static SIMD_INLINE __m512i reverse_mask();

template <>
SIMD_INLINE __m512i reverse_mask<1>()
{
  return x_mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
                          42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63);
}

template <>
SIMD_INLINE __m512i reverse_mask<2>()
{
  return x_mm512_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
                          17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29,
                          28, 31, 30, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40,
                          43, 42, 45, 44, 47, 46, 49, 48, 51, 50, 53, 52, 55,
                          54, 57, 56, 59, 58, 61, 60, 63, 62);
}

template <typename T>
static SIMD_INLINE Vec<T, 64> reverse(const Vec<T, 64> &a)
{
#ifdef __AVX512VBMI__
  return _mm512_permutexvar_epi8(reverse_mask<sizeof(T)>(), a);
#else
  Vec<T, 64> r = x_mm512_shuffle_epi8(a, reverse_mask<sizeof(T)>());
  return _mm512_permutexvar_epi64(_mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6), r);
#endif
}

static SIMD_INLINE Vec<Int, 64> reverse(const Vec<Int, 64> &a)
{
  // AVX512F
  const auto mask =
    _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  return _mm512_permutexvar_epi32(mask, a);
}

// float version, slightly changed int version
static SIMD_INLINE Vec<Float, 64> reverse(const Vec<Float, 64> &a)
{
  // AVX512F
  const auto mask =
    _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  return _mm512_permutexvar_ps(mask, a);
}

// double version
static SIMD_INLINE Vec<Double, 64> reverse(const Vec<Double, 64> &a)
{
  // AVX512F
  return _mm512_permutexvar_pd(_mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7), a);
}

// ---------------------------------------------------------------------------
// msb2int
// ---------------------------------------------------------------------------

// 27. Aug 22 (Jonas Keller): added msb2int functions

#ifdef __AVX512DQ__
static SIMD_INLINE uint64_t msb2int(const Vec<Int, 64> &a)
{
  return _mm512_movepi32_mask(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Float, 64> &a)
{
  return _mm512_movepi32_mask(_mm512_castps_si512(a));
}

static SIMD_INLINE uint64_t msb2int(const Vec<Double, 64> &a)
{
  return uint64_t(_mm512_movepi64_mask(_mm512_castpd_si512(a)));
}

#else  // __AVX512DQ__
// workaround for msb2int for 32 bits if AVX512DQ is not available

static SIMD_INLINE uint64_t msb2int(const Vec<Int, 64> &a)
{
  const __m512i mask = _mm512_set1_epi32(uint32_t(0x80000000));
  return _mm512_test_epi32_mask(a, mask);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Float, 64> &a)
{
  const __m512i mask = _mm512_set1_epi32(0x80000000);
  return _mm512_test_epi32_mask(_mm512_castps_si512(a), mask);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Double, 64> &a)
{
  const __m512i mask = _mm512_set1_epi64(0x8000000000000000);
  // _cvtmask8_u32 requires AVX512DQ, so just convert using implicit conversion
  return _mm512_test_epi64_mask(_mm512_castpd_si512(a), mask);
}
#endif // __AVX512DQ__

#ifdef __AVX512BW__
static SIMD_INLINE uint64_t msb2int(const Vec<Byte, 64> &a)
{
  return _mm512_movepi8_mask(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<SignedByte, 64> &a)
{
  return _mm512_movepi8_mask(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Short, 64> &a)
{
  return _mm512_movepi16_mask(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Word, 64> &a)
{
  return _mm512_movepi16_mask(a);
}
#else  // __AVX512BW__
// workaround for msb2int for 8 and 16 bits if AVX512BW is not available

// from:
// https://lemire.me/blog/2018/01/08/how-fast-can-you-bit-interleave-32-bit-integers/
static SIMD_INLINE uint64_t interleave_uint32_with_zeros(uint32_t input)
{
  uint64_t word = input;
  word = (word ^ (word << 16)) & 0x0000ffff0000ffff;
  word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff;
  word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f;
  word = (word ^ (word << 2)) & 0x3333333333333333;
  word = (word ^ (word << 1)) & 0x5555555555555555;
  return word;
}

static SIMD_INLINE uint64_t msb2int(const Vec<Byte, 64> &a)
{
  const uint64_t part3 = msb2int(reinterpret(a, OutputType<Int>()));
  const uint64_t part2 = msb2int(reinterpret(slle<1>(a), OutputType<Int>()));
  const uint64_t part1 = msb2int(reinterpret(slle<2>(a), OutputType<Int>()));
  const uint64_t part0 = msb2int(reinterpret(slle<3>(a), OutputType<Int>()));
  // TODO: is there a more efficient way to interleave with 3 zeros instead of
  // interleaving with 1 zero twice?
  const uint64_t part3_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part3));
  const uint64_t part2_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part2));
  const uint64_t part1_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part1));
  const uint64_t part0_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part0));
  return part0_with_zeros | (part1_with_zeros << 1) | (part2_with_zeros << 2) |
         (part3_with_zeros << 3);
}

static SIMD_INLINE uint64_t msb2int(const Vec<SignedByte, 64> &a)
{
  const uint64_t part3 = msb2int(reinterpret(a, OutputType<Int>()));
  const uint64_t part2 = msb2int(reinterpret(slle<1>(a), OutputType<Int>()));
  const uint64_t part1 = msb2int(reinterpret(slle<2>(a), OutputType<Int>()));
  const uint64_t part0 = msb2int(reinterpret(slle<3>(a), OutputType<Int>()));
  // TODO: is there a more efficient way to interleave with 3 zeros instead of
  // interleaving with 1 zero twice?
  const uint64_t part3_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part3));
  const uint64_t part2_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part2));
  const uint64_t part1_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part1));
  const uint64_t part0_with_zeros =
    interleave_uint32_with_zeros(interleave_uint32_with_zeros(part0));
  return part0_with_zeros | (part1_with_zeros << 1) | (part2_with_zeros << 2) |
         (part3_with_zeros << 3);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Short, 64> &a)
{
  const uint64_t odd = msb2int(reinterpret(a, OutputType<Int>()));
  const uint64_t even = msb2int(reinterpret(slle<1>(a), OutputType<Int>()));
  return interleave_uint32_with_zeros(even) |
         (interleave_uint32_with_zeros(odd) << 1);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Word, 64> &a)
{
  const uint64_t odd = msb2int(reinterpret(a, OutputType<Int>()));
  const uint64_t even = msb2int(reinterpret(slle<1>(a), OutputType<Int>()));
  return interleave_uint32_with_zeros(even) |
         (interleave_uint32_with_zeros(odd) << 1);
}
#endif // __AVX512BW__

// ---------------------------------------------------------------------------
// int2msb
// ---------------------------------------------------------------------------

// 06. Oct 22 (Jonas Keller): added int2msb functions

static SIMD_INLINE Vec<Byte, 64> int2msb(const uint64_t a, OutputType<Byte>,
                                         Integer<64>)
{
#ifdef __AVX512BW__
  return _mm512_maskz_set1_epi8(__mmask64(a), (int8_t) 0x80);
#else
  __m256i shuffleIndeces = _mm256_set_epi64x(
    0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
  __m256i aVecLo = _mm256_shuffle_epi8(_mm256_set1_epi32(a), shuffleIndeces);
  __m256i aVecHi =
    _mm256_shuffle_epi8(_mm256_set1_epi32(a >> 32), shuffleIndeces);
  __m256i sel = _mm256_set1_epi64x(0x8040201008040201);
  __m256i selectedLo = _mm256_and_si256(aVecLo, sel);
  __m256i selectedHi = _mm256_and_si256(aVecHi, sel);
  __m256i resultLo = _mm256_cmpeq_epi8(selectedLo, sel);
  __m256i resultHi = _mm256_cmpeq_epi8(selectedHi, sel);
  __m512i result =
    _mm512_inserti64x4(_mm512_castsi256_si512(resultLo), resultHi, 1);
  return _mm512_and_si512(result, _mm512_set1_epi32(0x80808080));
#endif
}

static SIMD_INLINE Vec<SignedByte, 64> int2msb(const uint64_t a,
                                               OutputType<SignedByte>,
                                               Integer<64>)
{
  return reinterpret(int2msb(a, OutputType<Byte>(), Integer<64>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Short, 64> int2msb(const uint64_t a, OutputType<Short>,
                                          Integer<64>)
{
#ifdef __AVX512BW__
  return _mm512_maskz_set1_epi16(__mmask32(a), (int16_t) 0x8000);
#else
  __m256i sel = _mm256_set_epi16(
    (int16_t) 0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100,
    0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001);
  __m256i aVecLo = _mm256_set1_epi16(a);
  __m256i aVecHi = _mm256_set1_epi16(a >> 16);
  __m256i selectedLo = _mm256_and_si256(aVecLo, sel);
  __m256i selectedHi = _mm256_and_si256(aVecHi, sel);
  __m256i resultLo = _mm256_cmpeq_epi16(selectedLo, sel);
  __m256i resultHi = _mm256_cmpeq_epi16(selectedHi, sel);
  __m512i result =
    _mm512_inserti64x4(_mm512_castsi256_si512(resultLo), resultHi, 1);
  return _mm512_and_si512(result, _mm512_set1_epi32(0x80008000));
#endif
}

static SIMD_INLINE Vec<Word, 64> int2msb(const uint64_t a, OutputType<Word>,
                                         Integer<64>)
{
  return reinterpret(int2msb(a, OutputType<Short>(), Integer<64>()),
                     OutputType<Word>());
}

static SIMD_INLINE Vec<Int, 64> int2msb(const uint64_t a, OutputType<Int>,
                                        Integer<64>)
{
  return _mm512_maskz_set1_epi32(__mmask16(a), 0x80000000);
}

static SIMD_INLINE Vec<Float, 64> int2msb(const uint64_t a, OutputType<Float>,
                                          Integer<64>)
{
  return reinterpret(int2msb(a, OutputType<Int>(), Integer<64>()),
                     OutputType<Float>());
}

static SIMD_INLINE Vec<Double, 64> int2msb(const uint64_t a, OutputType<Double>,
                                           Integer<64>)
{
  return _mm512_castsi512_pd(
    _mm512_maskz_set1_epi64(__mmask8(a), 0x8000000000000000));
}

// ---------------------------------------------------------------------------
// int2bits
// ---------------------------------------------------------------------------

// 09. Oct 22 (Jonas Keller): added int2bits functions

static SIMD_INLINE Vec<Byte, 64> int2bits(const uint64_t a, OutputType<Byte>,
                                          Integer<64>)
{
#ifdef __AVX512BW__
  return _mm512_maskz_set1_epi8(__mmask64(a), (int8_t) 0xff);
#else
  __m256i shuffleIndeces = _mm256_set_epi64x(
    0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
  __m256i aVecLo = _mm256_shuffle_epi8(_mm256_set1_epi32(a), shuffleIndeces);
  __m256i aVecHi =
    _mm256_shuffle_epi8(_mm256_set1_epi32(a >> 32), shuffleIndeces);
  __m256i sel = _mm256_set1_epi64x(0x8040201008040201);
  __m256i selectedLo = _mm256_and_si256(aVecLo, sel);
  __m256i selectedHi = _mm256_and_si256(aVecHi, sel);
  __m256i resultLo = _mm256_cmpeq_epi8(selectedLo, sel);
  __m256i resultHi = _mm256_cmpeq_epi8(selectedHi, sel);
  return _mm512_inserti64x4(_mm512_castsi256_si512(resultLo), resultHi, 1);
#endif
}

static SIMD_INLINE Vec<SignedByte, 64> int2bits(const uint64_t a,
                                                OutputType<SignedByte>,
                                                Integer<64>)
{
  return reinterpret(int2bits(a, OutputType<Byte>(), Integer<64>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Short, 64> int2bits(const uint64_t a, OutputType<Short>,
                                           Integer<64>)
{
#ifdef __AVX512BW__
  return _mm512_maskz_set1_epi16(__mmask32(a), (int16_t) 0xffff);
#else
  __m256i sel = _mm256_set_epi16(
    (int16_t) 0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100,
    0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001);
  __m256i aVecLo = _mm256_set1_epi16(a);
  __m256i aVecHi = _mm256_set1_epi16(a >> 16);
  __m256i selectedLo = _mm256_and_si256(aVecLo, sel);
  __m256i selectedHi = _mm256_and_si256(aVecHi, sel);
  __m256i resultLo = _mm256_cmpeq_epi16(selectedLo, sel);
  __m256i resultHi = _mm256_cmpeq_epi16(selectedHi, sel);
  return _mm512_inserti64x4(_mm512_castsi256_si512(resultLo), resultHi, 1);
#endif
}

static SIMD_INLINE Vec<Word, 64> int2bits(const uint64_t a, OutputType<Word>,
                                          Integer<64>)
{
  return reinterpret(int2bits(a, OutputType<Short>(), Integer<64>()),
                     OutputType<Word>());
}

static SIMD_INLINE Vec<Int, 64> int2bits(const uint64_t a, OutputType<Int>,
                                         Integer<64>)
{
  return _mm512_maskz_set1_epi32(__mmask16(a), 0xffffffff);
}

static SIMD_INLINE Vec<Float, 64> int2bits(const uint64_t a, OutputType<Float>,
                                           Integer<64>)
{
  return reinterpret(int2bits(a, OutputType<Int>(), Integer<64>()),
                     OutputType<Float>());
}

static SIMD_INLINE Vec<Double, 64> int2bits(const uint64_t a,
                                            OutputType<Double>, Integer<64>)
{
  return _mm512_castsi512_pd(
    _mm512_maskz_set1_epi64(__mmask8(a), 0xffffffffffffffff));
}

// ---------------------------------------------------------------------------
// iota
// ---------------------------------------------------------------------------

// 30. Jan 23 (Jonas Keller): added iota

static SIMD_INLINE Vec<Byte, 64> iota(OutputType<Byte>, Integer<64>)
{
  return x_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51,
                          50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38,
                          37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,
                          24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                          11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<SignedByte, 64> iota(OutputType<SignedByte>, Integer<64>)
{
  return x_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51,
                          50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38,
                          37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25,
                          24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                          11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Short, 64> iota(OutputType<Short>, Integer<64>)
{
  return x_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
                           18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,
                           3, 2, 1, 0);
}

static SIMD_INLINE Vec<Word, 64> iota(OutputType<Word>, Integer<64>)
{
  return x_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
                           18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,
                           3, 2, 1, 0);
}

static SIMD_INLINE Vec<Int, 64> iota(OutputType<Int>, Integer<64>)
{
  return _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Float, 64> iota(OutputType<Float>, Integer<64>)
{
  return _mm512_set_ps(15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f, 9.0f, 8.0f,
                       7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
}

static SIMD_INLINE Vec<Double, 64> iota(OutputType<Double>, Integer<64>)
{
  return _mm512_set_pd(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
}

} // namespace base
} // namespace internal
} // namespace simd

#endif // __AVX512F__

#endif // SIMD_VEC_BASE_IMPL_INTEL_64_H_
