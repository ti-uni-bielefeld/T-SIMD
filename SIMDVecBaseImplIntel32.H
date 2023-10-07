// ===========================================================================
//
// SIMDVecBaseImplIntel32.H --
// encapsulation for AVX/AVX2 Intel vector extensions
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

#pragma once
#ifndef SIMD_VEC_BASE_IMPL_INTEL_32_H_
#define SIMD_VEC_BASE_IMPL_INTEL_32_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBaseImplIntel16.H"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(SIMDVEC_INTEL_ENABLE) && defined(_SIMD_VEC_32_AVAIL_) &&           \
  !defined(SIMDVEC_SANDBOX)

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
// - _mm512_alignr_epi32/64 are *not* lane-oriented and could be a better
//   solution than the _epi8 version which *is* lane-oriented
//
// - should we replace set1 with broadcast? probably the compiler
//   generates broadcast anyhow? apparently not without -O3!
//
// - we could improve performance by using 256-bit instructions from
//   AVX512-VL (e.g. permute instructions); at the moment the idea is that
//   typically the widest vector width is used, so if AVX512 is available,
//   AVX/AVX2 would only rarely be used
//
// ===========================================================================

// ===========================================================================
// Vec integer specialization for AVX2
// ===========================================================================

// partial specialization for SIMD_WIDTH = 32
template <typename T>
class Vec<T, 32>
{
  __m256i ymm = _mm256_setzero_si256();

public:
  using Type                    = T;
  static constexpr int elements = 32 / sizeof(T);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 32;

  Vec() = default;
  Vec(const __m256i &x) { ymm = x; }
  Vec &operator=(const __m256i &x)
  {
    ymm = x;
    return *this;
  }
  operator __m256i() const { return ymm; }
  // for avx2 emulation
  Vec(const Vec<T, 16> &lo, const Vec<T, 16> &hi)
  {
    ymm = _mm256_set_m128i(hi, lo);
  }
  SIMD_INLINE Vec<T, 16> lo() const { return _mm256_castsi256_si128(ymm); }
  SIMD_INLINE Vec<T, 16> hi() const { return _mm256_extractf128_si256(ymm, 1); }
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

// ===========================================================================
// Vec float specialization for AVX
// ===========================================================================

template <>
class Vec<Float, 32>
{
  __m256 ymm = _mm256_setzero_ps();

public:
  using Type                    = Float;
  static constexpr int elements = 32 / sizeof(Float);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 32;

  Vec() = default;
  Vec(const __m256 &x) { ymm = x; }
  Vec &operator=(const __m256 &x)
  {
    ymm = x;
    return *this;
  }
  operator __m256() const { return ymm; }
  // for avx2 emulation
  Vec(const Vec<Float, 16> &lo, const Vec<Float, 16> &hi)
  {
    ymm = _mm256_set_m128(hi, lo);
  }
  SIMD_INLINE Vec<Float, 16> lo() const { return _mm256_castps256_ps128(ymm); }
  SIMD_INLINE Vec<Float, 16> hi() const
  {
    return _mm256_extractf128_ps(ymm, 1);
  }
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
  // 05. Sep 23 (Jonas Keller): added allocator
  using allocator = simd_aligned_allocator<Vec<Float, bytes>, bytes>;
};

// ===========================================================================
// Vec double specialization for AVX
// ===========================================================================

template <>
class Vec<Double, 32>
{
  __m256d ymm;

public:
  using Type                    = Double;
  static constexpr int elements = 32 / sizeof(Double);
  static constexpr int elems    = elements;
  static constexpr int bytes    = 32;

  Vec() = default;
  Vec(const __m256d &x) { ymm = x; }
  Vec &operator=(const __m256d &x)
  {
    ymm = x;
    return *this;
  }
  operator __m256d() const { return ymm; }
  // for avx2 emulation
  Vec(const Vec<Double, 16> &lo, const Vec<Double, 16> &hi)
  {
    ymm = _mm256_set_m128d(hi, lo);
  }
  SIMD_INLINE Vec<Double, 16> lo() const { return _mm256_castpd256_pd128(ymm); }
  SIMD_INLINE Vec<Double, 16> hi() const
  {
    return _mm256_extractf128_pd(ymm, 1);
  }
  void *operator new(size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size) { return simd_aligned_malloc(bytes, size); }
  void operator delete[](void *p) { simd_aligned_free(p); }
  using allocator = simd_aligned_allocator<Vec<Double, bytes>, bytes>;
};

namespace internal {
namespace base {
// ===========================================================================
// auxiliary functions
// ===========================================================================

// These functions either wrap AVX intrinsics (e.g. to handle
// immediate arguments as template parameter), or switch between
// implementations with different AVX* extensions, or provide
// altered or additional functionality.
// Only for use in wrapper functions!

// 01. Apr 23 (Jonas Keller): removed some not really necessary internal
// wrapper functions and inlined them directly into where they were used

// ---------------------------------------------------------------------------
// swizzle_32_16: swizzling of 128-bit lanes (for swizzle)
// ---------------------------------------------------------------------------

// rearrange vectors such that lane-oriented processing finds the
// right vectors to combine in corresponding lanes
//
// example: (li,hi are lanes)
//
//      --v0- --v1- --v2-
// N=3: l0 h0 l1 h1 l2 h2
//      --       --
//         --       --
//            --       --
//  ->  l0 h1 h0 l2 l1 h2  (distance = 3 lanes)
//      a0 b1              I=0, a=v0, b=v1
//            a1 b0        I=1, a=v0, b=v1
//                  a0 b1  I=2, a=v1, b=v2
//
//      --v0- --v1- --v2- --v3-
// N=4: l0 h0 l1 h1 l2 h2 l3 h3
//      --          --
//         --          --
//            --          --
//               --          --
//  ->  l0 l2 h0 h2 l1 l3 h1 h3  (distance = 4 lanes)
//      a0 b0                    I=0, a=v0, b=v2
//            a1 b0              I=1, a=v0, b=v2
//                  a0 b1        I=2, a=v1, b=v3
//                        a1 b1  I=3, a=v1, b=v3

// primary template
template <int N, int I = 0>
struct Swizzle_32_16
{
  template <typename T>
  static SIMD_INLINE void _swizzle_32_16(const Vec<T, 32> vIn[N],
                                         Vec<T, 32> vOut[N])
  {
    // example: N=3                                         v     v
    // I=0: permute_32_16(vIn[0], vIn[1], _MM_SHUFFLE(0, 2+ 1, 0, 0));
    // I=1: permute_32_16(vIn[0], vIn[2], _MM_SHUFFLE(0, 2+ 0, 0, 1));
    // I=2: permute_32_16(vIn[1], vIn[2], _MM_SHUFFLE(0, 2+ 1, 0, 0));
    //
    // example: N=4:                                        v     v
    // I=0: permute_32_16(vIn[0], vIn[2], _MM_SHUFFLE(0, 2+ 0, 0, 0));
    // I=1: permute_32_16(vIn[0], vIn[2], _MM_SHUFFLE(0, 2+ 1, 0, 0));
    // I=2: permute_32_16(vIn[1], vIn[3], _MM_SHUFFLE(0, 2+ 0, 0, 1));
    // I=3: permute_32_16(vIn[1], vIn[3], _MM_SHUFFLE(0, 2+ 1, 0, 1));
    //
    // "2+" means: take from second vector
    vOut[I] =
      _mm256_permute2f128_si256(vIn[I / 2], vIn[(I + N) / 2],
                                _MM_SHUFFLE(0, (2 + (I + N) % 2), 0, (I % 2)));
    Swizzle_32_16<N, I + 1>::_swizzle_32_16(vIn, vOut);
  }

  // Float version
  static SIMD_INLINE void _swizzle_32_16(const Vec<Float, 32> vIn[N],
                                         Vec<Float, 32> vOut[N])
  {
    vOut[I] =
      _mm256_permute2f128_ps(vIn[I / 2], vIn[(I + N) / 2],
                             _MM_SHUFFLE(0, (2 + (I + N) % 2), 0, (I % 2)));
    Swizzle_32_16<N, I + 1>::_swizzle_32_16(vIn, vOut);
  }

  // Double version
  static SIMD_INLINE void _swizzle_32_16(const Vec<Double, 32> vIn[N],
                                         Vec<Double, 32> vOut[N])
  {
    vOut[I] =
      _mm256_permute2f128_pd(vIn[I / 2], vIn[(I + N) / 2],
                             _MM_SHUFFLE(0, (2 + (I + N) % 2), 0, (I % 2)));
    Swizzle_32_16<N, I + 1>::_swizzle_32_16(vIn, vOut);
  }
};

// termination
template <int N>
struct Swizzle_32_16<N, N>
{
  template <typename T>
  static SIMD_INLINE void _swizzle_32_16(const Vec<T, 32>[N], Vec<T, 32>[N])
  {}
};

// swizzle lanes (for implementation of swizzle functions)
// from Stan Melax: 3D Vector Normalization... (adapted)
template <int N, typename T>
static SIMD_INLINE void swizzle_32_16(const Vec<T, 32> vIn[N],
                                      Vec<T, 32> vOut[N])
{
  Swizzle_32_16<N>::_swizzle_32_16(vIn, vOut);
}

// ---------------------------------------------------------------------------
// byte-wise shifts
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// positive and in range: apply shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_si256(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_srli_si256(a, IMM);
}

// positive and out of range: return zero vector
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_si256(__m256i,
                                              IsPosInRange<true, false>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_si256(__m256i a)
{
  return x_mm256_srli_si256<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: apply shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_si256(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_slli_si256(a, IMM);
}

// positive and out of range: return zero vector
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_si256(__m256i,
                                              IsPosInRange<true, false>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_si256(__m256i a)
{
  return x_mm256_slli_si256<IMM>(a, IsPosInGivenRange<16, IMM>());
}

#else

// non-avx2 workarounds
// (easy since AVX2 instructions operate on lanes anyhow)

// IMM range handling is done at SSE level

template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_si256(__m256i a)
{
  return _mm256_set_m128i(x_mm_srli_si128<IMM>(_mm256_extractf128_si256(a, 1)),
                          x_mm_srli_si128<IMM>(_mm256_castsi256_si128(a)));
}

template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_si256(__m256i a)
{
  return _mm256_set_m128i(x_mm_slli_si128<IMM>(_mm256_extractf128_si256(a, 1)),
                          x_mm_slli_si128<IMM>(_mm256_castsi256_si128(a)));
}

#endif

// ---------------------------------------------------------------------------
// element-wise shifts
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_epi16(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_slli_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_epi16(__m256i,
                                              IsPosInRange<true, false>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_epi16(__m256i a)
{
  return x_mm256_slli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_epi16(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_srli_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_epi16(__m256i,
                                              IsPosInRange<true, false>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_epi16(__m256i a)
{
  return x_mm256_srli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_epi32(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_slli_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_epi32(__m256i,
                                              IsPosInRange<true, false>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli_epi32(__m256i a)
{
  return x_mm256_slli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_epi32(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_srli_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_epi32(__m256i,
                                              IsPosInRange<true, false>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli_epi32(__m256i a)
{
  return x_mm256_srli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// 16. Oct 22 (Jonas Keller): added x_mm256_srai_epi8

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi8(__m256i a,
                                             IsPosInRange<true, true>)
{
  __m256i odd  = _mm256_srai_epi16(a, IMM);
  __m256i even = _mm256_srai_epi16(_mm256_slli_epi16(a, 8), IMM + 8);
  return _mm256_blendv_epi8(even, odd, _mm256_set1_epi16((int16_t) 0xff00));
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi8(__m256i a,
                                             IsPosInRange<true, false>)
{
  // result should be all ones if a is negative, all zeros otherwise
  return _mm256_cmpgt_epi8(_mm256_setzero_si256(), a);
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi8(__m256i a)
{
  return x_mm256_srai_epi8<IMM>(a, IsPosInGivenRange<7, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi16(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_srai_epi16(a, IMM);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi16(__m256i a,
                                              IsPosInRange<true, false>)
{
  return _mm256_srai_epi16(a, 15);
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi16(__m256i a)
{
  return x_mm256_srai_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi32(__m256i a,
                                              IsPosInRange<true, true>)
{
  return _mm256_srai_epi32(a, IMM);
}

// positive and out of range: maximal shift
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi32(__m256i a,
                                              IsPosInRange<true, false>)
{
  return _mm256_srai_epi32(a, 31);
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srai_epi32(__m256i a)
{
  return x_mm256_srai_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
}

#endif

// ---------------------------------------------------------------------------
// extract
// ---------------------------------------------------------------------------

// NOTE: extract functions for AVX don't tolerate range errors for
// their IMM argument; the reason is that there is actually no
// extract on 32-byte vectors, but only on 16-byte vectors; e.g.
// in /usr/lib/gcc/x86_64-linux-gnu/4.8/include/avxintrin.h we have
// int _mm256_extract_epi16 (__m256i __X, int const __N)
// {
//   __m128i __Y = _mm256_extractf128_si256 (__X, __N >> 3);
//   return _mm_extract_epi16 (__Y, __N % 8);
// }
// which uses _mm256_extractf128_si256 where the immediate argument can only
// be 0 or 1; if this is not case (if IMM is out of range), we get
// error: the last argument must be a 1-bit immediate
// return (__m128i) __builtin_ia32_vextractf128_si256 ((__v8si)__X, __N);

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi16(__m256i a,
                                             IsPosInRange<true, true>)
{
  // strange, Intel intrinsics guide says this is AVX2, but it is
  // already available in avxintrin.h
  return _mm256_extract_epi16(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi16(__m256i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi16(__m256i a)
{
  // 16 * epi16
  return x_mm256_extract_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi8(__m256i a, IsPosInRange<true, true>)
{
  // strange, Intel intrinsics guide says this is AVX2, but it is
  // already available in avxintrin.h
  return _mm256_extract_epi8(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi8(__m256i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi8(__m256i a)
{
  // 32 * epi8
  return x_mm256_extract_epi8<IMM>(a, IsPosInGivenRange<32, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi32(__m256i a,
                                             IsPosInRange<true, true>)
{
  // TODO: extract: is conversion from return type int to Int always safe?
  return _mm256_extract_epi32(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi32(__m256i, IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE int x_mm256_extract_epi32(__m256i a)
{
  // 8 * epi32
  return x_mm256_extract_epi32<IMM>(a, IsPosInGivenRange<8, IMM>());
}

// positive and in range: extract
template <int IMM>
static SIMD_INLINE long long x_mm256_extract_epi64(__m256i a,
                                                   IsPosInRange<true, true>)
{
  return _mm256_extract_epi64(a, IMM);
}

// positive and out of range: return zero
template <int IMM>
static SIMD_INLINE long long x_mm256_extract_epi64(__m256i,
                                                   IsPosInRange<true, false>)
{
  return 0;
}

// hub
template <int IMM>
static SIMD_INLINE long long x_mm256_extract_epi64(__m256i a)
{
  // 4 * epi64
  return x_mm256_extract_epi64<IMM>(a, IsPosInGivenRange<4, IMM>());
}

// _mm256_extract_ps doesn't exist, workaround in extract template function

// ---------------------------------------------------------------------------
// alignr
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// 21. Apr 23 (Jonas Keller): replaced IMM range handling via tag dispatch
// with static_assert, since we don't need the range handling anymore,
// we just assert that IMM is in range

template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr_epi8(__m256i h, __m256i l)
{
  //  2. Jul 18 (rm) BUGFIX: 64 -> 32 (2 lanes only, lane-oriented!)
  // IMM < 32
  static_assert(IMM >= 0 && IMM < 32, "");
  return _mm256_alignr_epi8(h, l, IMM);
}

#else

// non-avx2 workarounds
// (easy since AVX2 instructions operate on lanes anyhow)

template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr_epi8(__m256i h, __m256i l)
{
  static_assert(IMM >= 0 && IMM < 32, "");
  return _mm256_set_m128i(
    _mm_alignr_epi8(_mm256_extractf128_si256(h, 1),
                    _mm256_extractf128_si256(l, 1), IMM),
    _mm_alignr_epi8(_mm256_castsi256_si128(h), _mm256_castsi256_si128(l), IMM));
}

#endif

// ---------------------------------------------------------------------------
// auxiliary function for right shift over full 32 byte
// ---------------------------------------------------------------------------

// (difficulty: _mm256_srli_si256 only works in 128-bit lanes)
// http://stackoverflow.com/questions/25248766/
//        emulating-shifts-on-32-bytes-with-avx
// TODO: finer case distinction using permute4x64?

//  7. Jun 16 (rm): if replaced by tag dispatching
// (reason: all branches are compiles and at least icc complains
// about exceeded ranges in immediates)

// IMM = 0
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli256_si256(__m256i a,
                                                 Range<true, true, 0, 16>)
{
  return a;
}

// IMM = 1..15
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli256_si256(__m256i a,
                                                 Range<true, false, 0, 16>)
{
  // _MM_SHUFFLE(2,0, 0,1) = 0x81, MS-bit set -> setting elements to zero
  // higher lane set to zero (2,0), lower lane taken from higher lane (0,1)
  // a:              HHHHHHHHhhhhhhhh LLLLLLLllllllll
  // _0h:            0000000000000000 HHHHHHHhhhhhhhh (2,0) (0,1)
  __m256i _0h = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(2, 0, 0, 1));
  // e.g. IMM=5
  // a:              HHHHHHHHhhhhhhhh LLLLLLLllllllll
  // _0h:            0000000000000000 HHHHHHHhhhhhhhh
  // alignr H lane:  0000000000000000 HHHHHHHHhhhhhhh
  // selected:                  ----- -----------
  // alignr L lane:  HHHHHHHHhhhhhhhh LLLLLLLLlllllll
  // selected:                  ----- -----------
  // alignr:         00000HHHHHHHHhhh hhhhhLLLLLLLlll
  return x_mm256_alignr_epi8<IMM>(_0h, a);
}

// IMM = 16
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli256_si256(__m256i a,
                                                 Range<true, true, 16, 32>)
{
  // _MM_SHUFFLE(2,0, 0,1) = 0x81, MS-bit set -> setting elements to zero
  // higher lane set to zero (2,0), lower lane taken from higher lane (0,1)
  // a:              HHHHHHHHhhhhhhhh LLLLLLLllllllll
  // _0h:            0000000000000000 HHHHHHHhhhhhhhh (2,0) (0,1)
  __m256i _0h = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(2, 0, 0, 1));
  // _0h:            0000000000000000 HHHHHHHhhhhhhhh (2,0) (0,1)
  return _0h;
}

// IMM = 17..31
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli256_si256(__m256i a,
                                                 Range<true, false, 16, 32>)
{
  // _MM_SHUFFLE(2,0, 0,1) = 0x81, MS-bit set -> setting elements to zero
  // higher lane set to zero (2,0), lower lane taken from higher lane (0,1)
  // a:              HHHHHHHHhhhhhhhh LLLLLLLllllllll
  // _0h:            0000000000000000 HHHHHHHhhhhhhhh (2,0) (0,1)
  __m256i _0h = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(2, 0, 0, 1));
  // e.g. IMM=18 (18-16 = 2)
  // _0h:            0000000000000000 HHHHHHHhhhhhhhh
  // srli:           0000000000000000 00HHHHHHHHhhhhh
  return x_mm256_srli_si256<IMM - 16>(_0h);
}

// IMM >= 32
template <int IMM, bool AT_LOW_LIM, int LOW_LIM_INCL, int UP_LIM_EXCL>
static SIMD_INLINE __m256i x_mm256_srli256_si256(
  __m256i, Range<true, AT_LOW_LIM, LOW_LIM_INCL, UP_LIM_EXCL>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_srli256_si256(__m256i a)
{
  return x_mm256_srli256_si256<IMM>(a, SizeRange<IMM, 16>());
}

// ---------------------------------------------------------------------------
// auxiliary function for left shift over full 32 bytes
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/25248766/
//        emulating-shifts-on-32-bytes-with-avx
// TODO: finer case distinction using permute4x64?

//  7. Jun 16 (rm): if replaced by tag dispatching
// (reason: all branches are compiles and at least icc complains
// about exceeded ranges in immediates)

// IMM = 0
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli256_si256(__m256i a,
                                                 Range<true, true, 0, 16>)
{
  return a;
}

// IMM = 1..15
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli256_si256(__m256i a,
                                                 Range<true, false, 0, 16>)
{
  // _MM_SHUFFLE(0,0, 2,0) = 0x08, MS-bit set -> setting elements to zero
  // higher lane taken from lower lane (0,0), lower lane set to zero (2,0)
  // a:              HHHHHHHHhhhhhhhh LLLLLLLLllllllll
  // _l0:            LLLLLLLLllllllll 0000000000000000 (0,0) (2,0)
  __m256i _l0 = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 2, 0));
  // e.g. IMM = 5: (16-5=11)
  // _l0:            LLLLLLLLllllllll 0000000000000000
  // a:              HHHHHHHHhhhhhhhh LLLLLLLLllllllll
  // alignr H lane:  HHHHHHHHhhhhhhhh LLLLLLLLllllllll
  // selected:            ----------- -----
  // alignr L lane:  LLLLLLLLllllllll 0000000000000000
  // selected:            ----------- -----
  // alignr:         HHHhhhhhhhhLLLLL LLLllllllll00000
  return x_mm256_alignr_epi8<16 - IMM>(a, _l0);
}

// IMM = 16
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli256_si256(__m256i a,
                                                 Range<true, true, 16, 32>)
{
  // _MM_SHUFFLE(0,0, 2,0) = 0x08, MS-bit set -> setting elements to zero
  // higher lane taken from lower lane (0,0), lower lane set to zero (2,0)
  // a:              HHHHHHHHhhhhhhhh LLLLLLLLllllllll
  // _l0:            LLLLLLLLllllllll 0000000000000000 (0,0) (2,0)
  __m256i _l0 = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 2, 0));
  // _l0:            LLLLLLLLllllllll 0000000000000000 (0,0) (2,0)
  return _l0;
}

// IMM = 17..31
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli256_si256(__m256i a,
                                                 Range<true, false, 16, 32>)
{
  // _MM_SHUFFLE(0,0, 2,0) = 0x08, MS-bit set -> setting elements to zero
  // higher lane taken from lower lane (0,0), lower lane set to zero (2,0)
  // a:              HHHHHHHHhhhhhhhh LLLLLLLLllllllll
  // _l0:            LLLLLLLLllllllll 0000000000000000 (0,0) (2,0)
  __m256i _l0 = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 2, 0));
  // e.g. IMM = 18 (18-16=2)
  // _l0:            LLLLLLLLllllllll 0000000000000000
  // slri:           LLLLLLllllllll00 0000000000000000
  return x_mm256_slli_si256<IMM - 16>(_l0);
}

// IMM >= 32
template <int IMM, bool AT_LOW_LIM, int LOW_LIM_INCL, int UP_LIM_EXCL>
static SIMD_INLINE __m256i x_mm256_slli256_si256(
  __m256i, Range<true, AT_LOW_LIM, LOW_LIM_INCL, UP_LIM_EXCL>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_slli256_si256(__m256i a)
{
  return x_mm256_slli256_si256<IMM>(a, SizeRange<IMM, 16>());
}

// ---------------------------------------------------------------------------
// full 32 byte alignr ("alignr256")
// ---------------------------------------------------------------------------

// h:  HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
// l:  LLLLLLLLLLLLLLLL llllllllllllllll
//     000 HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL llllllllllllllll
// 0:                                        ---------------- ----------------
// 5:                                 ------ ---------------- ----------
// 16:                      ---------------- ----------------
// 18:                  --- ---------------- -------------
// 32:     ---------------- ----------------
// 35: --- ---------------- -------------

// modified from emmanualLattia at
// https://idz-smita-idzdev.ssgisp.com/fr-fr/forums/topic/500664

//  7. Jun 16 (rm): if replaced by tag dispatching
// (reason: all branches are compiles and at least icc complains
// about exceeded ranges in immediates)

// IMM = 0
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i, __m256i low,
                                                  Range<true, true, 0, 16>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // IMM == 0:        LLLLLLLLLLLLLLLL llllllllllllllll
  return low;
}

// IMM = 1..15
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i low,
                                                  Range<true, false, 0, 16>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // high0low1:       hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL (0,2) (0,1)
  __m256i high0_low1 =
    _mm256_permute2f128_si256(low, high, _MM_SHUFFLE(0, 2, 0, 1));
  // e.g. IMM = 5
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // high0low1:       hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL (0,2) (0,1)
  // alignr H lane:   hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL
  // selected:                   ----- -----------
  // alignr L lane:   LLLLLLLLLLLLLLLL llllllllllllllll
  // selected:                   ----- -----------
  // alignr:          hhhhhLLLLLLLLLLL LLLLLlllllllllll
  return x_mm256_alignr_epi8<IMM>(high0_low1, low);
}

// IMM = 16
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i low,
                                                  Range<true, true, 16, 32>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // high0low1:       hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL (0,2) (0,1)
  __m256i high0_low1 =
    _mm256_permute2f128_si256(low, high, _MM_SHUFFLE(0, 2, 0, 1));
  // IMM == 16:       hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL
  return high0_low1;
}

// IMM = 17..31
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i low,
                                                  Range<true, false, 16, 32>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // high0low1:       hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL (0,2) (0,1)
  __m256i high0_low1 =
    _mm256_permute2f128_si256(low, high, _MM_SHUFFLE(0, 2, 0, 1));
  // e.g. IMM = 18 (IMM - 16 = 2)
  // high0low1:       hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // alignr H lane:   HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // selected:                      -- --------------
  // alignr L lane:   hhhhhhhhhhhhhhhh LLLLLLLLLLLLLLLL
  // selected:                      -- --------------
  // alignr:          HHhhhhhhhhhhhhhh hhLLLLLLLLLLLLLL
  return x_mm256_alignr_epi8<IMM - 16>(high, high0_low1);
}

// IMM = 32
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i,
                                                  Range<true, true, 32, 48>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  //                  HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  return high;
}

// IMM = 33..47
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i,
                                                  Range<true, false, 32, 48>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // null_high1:      0000000000000000 HHHHHHHHHHHHHHHH (2,0) (0,1)
  __m256i null_high1 =
    _mm256_permute2f128_si256(high, high, _MM_SHUFFLE(2, 0, 0, 1));
  // e.g. IMM = 37 (37-32 = 5)
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // null_high1:      0000000000000000 HHHHHHHHHHHHHHHH
  // alignr H lane    0000000000000000 HHHHHHHHHHHHHHHH
  // selected:                   ----- -----------
  // alignr L lane    HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // selected:                   ----- -----------
  // alignr:          00000HHHHHHHHHHH HHHHHhhhhhhhhhhh
  return x_mm256_alignr_epi8<IMM - 32>(null_high1, high);
}

// IMM == 48
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i,
                                                  Range<true, true, 48, 64>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // null_high1:      0000000000000000 HHHHHHHHHHHHHHHH (2,0) (0,1)
  __m256i null_high1 =
    _mm256_permute2f128_si256(high, high, _MM_SHUFFLE(2, 0, 0, 1));
  // null_high1:      0000000000000000 HHHHHHHHHHHHHHHH
  return null_high1;
}

// IMM = 49..63
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i,
                                                  Range<true, false, 48, 64>)
{
  // high:            HHHHHHHHHHHHHHHH hhhhhhhhhhhhhhhh
  // low:             LLLLLLLLLLLLLLLL llllllllllllllll
  // null_high1:      0000000000000000 HHHHHHHHHHHHHHHH (2,0) (0,1)
  __m256i null_high1 =
    _mm256_permute2f128_si256(high, high, _MM_SHUFFLE(2, 0, 0, 1));
  // e.g. IMM = 50 (50 - 48 = 2)
  // null_high1:      0000000000000000 HHHHHHHHHHHHHHHH
  // zero:            0000000000000000 0000000000000000
  // alignr H lane:   0000000000000000 0000000000000000
  // selected:                      -- --------------
  // alignr L lane:   0000000000000000 HHHHHHHHHHHHHHHH
  // selected:                      -- --------------
  // alignr:          0000000000000000 00HHHHHHHHHHHHHH
  return x_mm256_alignr_epi8<IMM - 48>(_mm256_setzero_si256(), null_high1);
}

// IMM >= 64
template <int IMM, bool AT_LOW_LIM, int LOW_LIM_INCL, int UP_LIM_EXCL>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(
  __m256i, __m256i, Range<true, AT_LOW_LIM, LOW_LIM_INCL, UP_LIM_EXCL>)
{
  return _mm256_setzero_si256();
}

// hub
template <int IMM>
static SIMD_INLINE __m256i x_mm256_alignr256_epi8(__m256i high, __m256i low)
{
  return x_mm256_alignr256_epi8<IMM>(high, low, SizeRange<IMM, 16>());
}

// ---------------------------------------------------------------------------
// insert 16 byte vector a into both lanes of a 32 byte vector
// ---------------------------------------------------------------------------

static SIMD_INLINE __m256i x_mm256_duplicate_si128(__m128i a)
{
  return _mm256_set_m128i(a, a);
}

// ---------------------------------------------------------------------------
// transpose4x64
// ---------------------------------------------------------------------------

// in  = Hh Hl Lh Ll
//        |   X   |
// out = Hh Lh Hl Ll

static SIMD_INLINE __m256i x_mm256_transpose4x64_epi64(__m256i a)
{
#ifdef __AVX2__
  return _mm256_permute4x64_epi64(a, _MM_SHUFFLE(3, 1, 2, 0));
#else
  // non-avx2 workarounds (different versions)

#if 1
  // non-avx2 workaround
  // (more efficient)

  __m256d in, x1, x2;
  // in = Hh Hl Lh Ll
  in = _mm256_castsi256_pd(a);
  // only lower 4 bit are used
  // in = Hh Hl Lh Ll
  //       0  1  0  1  = (0,0,1,1)
  // x1 = Hl Hh Ll Lh
  x1 = _mm256_permute_pd(in, _MM_SHUFFLE(0, 0, 1, 1));
  // all 8 bit are used
  // x1 = Hl Hh Ll Lh
  //       0  0  1  1
  // x2 = Ll Lh Hl Hh
  x2 = _mm256_permute2f128_pd(x1, x1, _MM_SHUFFLE(0, 0, 1, 1));
  // only lower 4 bit are used
  // in = Hh Hl Lh Ll
  // x2 = Ll Lh Hl Hh
  //       0  1  1  0 = (0,0,1,2)
  // ret: Hh Lh Hl Ll
  return _mm256_castpd_si256(_mm256_blend_pd(in, x2, _MM_SHUFFLE(0, 0, 1, 2)));
#else
  // non-avx2 workaround
  // (less efficient)

  __m128i lo    = _mm256_castsi256_si128(a);
  __m128i hi    = _mm256_extractf128_si256(a, 1);
  __m128i loRes = _mm_unpacklo_epi64(lo, hi);
  __m128i hiRes = _mm_unpackhi_epi64(lo, hi);
  return _mm256_set_m128i(hiRes, loRes);
#endif

#endif
}

static SIMD_INLINE __m256 x_mm256_transpose4x64_ps(__m256 a)
{
  return _mm256_castsi256_ps(
    x_mm256_transpose4x64_epi64(_mm256_castps_si256(a)));
}

static SIMD_INLINE __m256d x_mm256_transpose4x64_pd(__m256d a)
{
  return _mm256_castsi256_pd(
    x_mm256_transpose4x64_epi64(_mm256_castpd_si256(a)));
}

// ---------------------------------------------------------------------------
// unpack of 2 ps
// ---------------------------------------------------------------------------

static SIMD_INLINE __m256 x_mm256_unpacklo_2ps(__m256 a, __m256 b)
{
  return _mm256_castpd_ps(
    _mm256_unpacklo_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
}

static SIMD_INLINE __m256 x_mm256_unpackhi_2ps(__m256 a, __m256 b)
{
  return _mm256_castpd_ps(
    _mm256_unpackhi_pd(_mm256_castps_pd(a), _mm256_castps_pd(b)));
}

// ---------------------------------------------------------------------------
// binary functions with non-avx2 workarounds
// ---------------------------------------------------------------------------

#ifdef __AVX2__
// avx2 is available
#define SIMDVEC_INTEL_X_INT_BINFCT_32(INTRIN)                                  \
  static SIMD_INLINE __m256i x_mm256_##INTRIN(__m256i a, __m256i b)            \
  {                                                                            \
    return _mm256_##INTRIN(a, b);                                              \
  }
#else
// non-avx2 workaround
#define SIMDVEC_INTEL_X_INT_BINFCT_32(INTRIN)                                  \
  static SIMD_INLINE __m256i x_mm256_##INTRIN(__m256i a, __m256i b)            \
  {                                                                            \
    return _mm256_set_m128i(                                                   \
      _mm_##INTRIN(_mm256_extractf128_si256(a, 1),                             \
                   _mm256_extractf128_si256(b, 1)),                            \
      _mm_##INTRIN(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b)));     \
  }
#endif

SIMDVEC_INTEL_X_INT_BINFCT_32(unpacklo_epi8)
SIMDVEC_INTEL_X_INT_BINFCT_32(unpackhi_epi8)
SIMDVEC_INTEL_X_INT_BINFCT_32(unpacklo_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(unpackhi_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(shuffle_epi8)
SIMDVEC_INTEL_X_INT_BINFCT_32(packs_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(packs_epi32)
SIMDVEC_INTEL_X_INT_BINFCT_32(packus_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(packus_epi32)
SIMDVEC_INTEL_X_INT_BINFCT_32(hadd_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(hadd_epi32)
SIMDVEC_INTEL_X_INT_BINFCT_32(hadds_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(hsub_epi16)
SIMDVEC_INTEL_X_INT_BINFCT_32(hsub_epi32)
SIMDVEC_INTEL_X_INT_BINFCT_32(hsubs_epi16)

// non-avx2 workarounds via analogous ps, pd functions
#ifdef __AVX2__
// avx2 is available
#define SIMDVEC_INTEL_X_INT_BINFCT_PSPD_32(INTRIN, INTSUFFIX, PSPDSUFFIX)      \
  static SIMD_INLINE __m256i x_mm256_##INTRIN##_##INTSUFFIX(__m256i a,         \
                                                            __m256i b)         \
  {                                                                            \
    return _mm256_##INTRIN##_##INTSUFFIX(a, b);                                \
  }
#else
// non-avx2 workaround
#define SIMDVEC_INTEL_X_INT_BINFCT_PSPD_32(INTRIN, INTSUFFIX, PSPDSUFFIX)      \
  static SIMD_INLINE __m256i x_mm256_##INTRIN##_##INTSUFFIX(__m256i a,         \
                                                            __m256i b)         \
  {                                                                            \
    return _mm256_cast##PSPDSUFFIX##_si256(                                    \
      _mm256_##INTRIN##_##PSPDSUFFIX(_mm256_castsi256##_##PSPDSUFFIX(a),       \
                                     _mm256_castsi256##_##PSPDSUFFIX(b)));     \
  }
#endif

// better non-avx2 workarounds for unpacks (32, 64) via ps, pd
SIMDVEC_INTEL_X_INT_BINFCT_PSPD_32(unpacklo, epi32, ps)
SIMDVEC_INTEL_X_INT_BINFCT_PSPD_32(unpackhi, epi32, ps)
SIMDVEC_INTEL_X_INT_BINFCT_PSPD_32(unpacklo, epi64, pd)
SIMDVEC_INTEL_X_INT_BINFCT_PSPD_32(unpackhi, epi64, pd)

#ifdef __AVX2__
// avx2 is available
#define SIMDVEC_INTEL_X_INT_UNFCT_IMM_PSPD_32(INTRIN, INTSUFFIX, PSPDSUFFIX)   \
  template <int IMM>                                                           \
  static SIMD_INLINE __m256i x_mm256_##INTRIN##_##INTSUFFIX(__m256i a)         \
  {                                                                            \
    return _mm256_##INTRIN##_##INTSUFFIX(a, IMM);                              \
  }
#else
// non-avx2 workaround
#define SIMDVEC_INTEL_X_INT_UNFCT_IMM_PSPD_32(INTRIN, INTSUFFIX, PSPDSUFFIX)   \
  template <int IMM>                                                           \
  static SIMD_INLINE __m256i x_mm256_##INTRIN##_##INTSUFFIX(__m256i a)         \
  {                                                                            \
    __m256 as = _mm256_castsi256##_##PSPDSUFFIX(a);                            \
    return _mm256_cast##PSPDSUFFIX##_si256(                                    \
      _mm256_##INTRIN##_##PSPDSUFFIX(as, as, IMM));                            \
  }
#endif

// non-avx2 workaround for shuffle_epi32
SIMDVEC_INTEL_X_INT_UNFCT_IMM_PSPD_32(shuffle, epi32, ps)

// ###########################################################################
// ###########################################################################
// ###########################################################################

// ===========================================================================
// Vec template function specializations or overloading for AVX
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
static SIMD_INLINE Vec<Tdst, 32> reinterpret(const Vec<Tsrc, 32> &vec,
                                             OutputType<Tdst>)
{
  // 26. Nov 22 (Jonas Keller): reinterpret_cast is technically undefined
  // behavior, so just rewrapping the vector register in a new Vec instead
  // return reinterpret_cast<const Vec<Tdst,32>&>(vec);
  return Vec<Tdst, 32>(__m256i(vec));
}

// from float to any integer type
template <typename Tdst, SIMD_ENABLE_IF((std::is_integral<Tdst>::value))>
static SIMD_INLINE Vec<Tdst, 32> reinterpret(const Vec<Float, 32> &vec,
                                             OutputType<Tdst>)
{
  return _mm256_castps_si256(vec);
}

// from any integer type to float
template <typename Tsrc, SIMD_ENABLE_IF((std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Float, 32> reinterpret(const Vec<Tsrc, 32> &vec,
                                              OutputType<Float>)
{
  return _mm256_castsi256_ps(vec);
}

// from double to any integer type
template <typename Tdst, SIMD_ENABLE_IF((std::is_integral<Tdst>::value))>
static SIMD_INLINE Vec<Tdst, 32> reinterpret(const Vec<Double, 32> &vec,
                                             OutputType<Tdst>)
{
  return _mm256_castpd_si256(vec);
}

// from any integer type to double
template <typename Tsrc, SIMD_ENABLE_IF((std::is_integral<Tsrc>::value))>
static SIMD_INLINE Vec<Double, 32> reinterpret(const Vec<Tsrc, 32> &vec,
                                               OutputType<Double>)
{
  return _mm256_castsi256_pd(vec);
}

// from float to double
static SIMD_INLINE Vec<Double, 32> reinterpret(const Vec<Float, 32> &vec,
                                               OutputType<Double>)
{
  return _mm256_castps_pd(vec);
}

// from double to float
static SIMD_INLINE Vec<Float, 32> reinterpret(const Vec<Double, 32> &vec,
                                              OutputType<Float>)
{
  return _mm256_castpd_ps(vec);
}

// between identical types
template <typename T>
static SIMD_INLINE Vec<T, 32> reinterpret(const Vec<T, 32> &vec, OutputType<T>)
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
static SIMD_INLINE Vec<Int, 32> cvts(const Vec<Float, 32> &a, OutputType<Int>)
{
  // TODO: analyze much more complex solution for cvts at
  // TODO: http://stackoverflow.com/questions/9157373/
  // TODO: most-efficient-way-to-convert-vector-of-float-to-vector-of-uint32
  __m256 clip = _mm256_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
  return _mm256_cvtps_epi32(_mm256_min_ps(clip, a));
}

// saturation is not necessary in this case
static SIMD_INLINE Vec<Float, 32> cvts(const Vec<Int, 32> &a, OutputType<Float>)
{
  return _mm256_cvtepi32_ps(a);
}

// ---------------------------------------------------------------------------
// setzero
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 32> setzero(OutputType<Byte>, Integer<32>)
{
  return _mm256_setzero_si256();
}

static SIMD_INLINE Vec<SignedByte, 32> setzero(OutputType<SignedByte>,
                                               Integer<32>)
{
  return _mm256_setzero_si256();
}

static SIMD_INLINE Vec<Word, 32> setzero(OutputType<Word>, Integer<32>)
{
  return _mm256_setzero_si256();
}

static SIMD_INLINE Vec<Short, 32> setzero(OutputType<Short>, Integer<32>)
{
  return _mm256_setzero_si256();
}

static SIMD_INLINE Vec<Int, 32> setzero(OutputType<Int>, Integer<32>)
{
  return _mm256_setzero_si256();
}

static SIMD_INLINE Vec<Float, 32> setzero(OutputType<Float>, Integer<32>)
{
  return _mm256_setzero_ps();
}

static SIMD_INLINE Vec<Double, 32> setzero(OutputType<Double>, Integer<32>)
{
  return _mm256_setzero_pd();
}

// ---------------------------------------------------------------------------
// set1
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 32> set1(Byte a, Integer<32>)
{
  return _mm256_set1_epi8(a);
}

static SIMD_INLINE Vec<SignedByte, 32> set1(SignedByte a, Integer<32>)
{
  return _mm256_set1_epi8(a);
}

static SIMD_INLINE Vec<Word, 32> set1(Word a, Integer<32>)
{
  return _mm256_set1_epi16(a);
}

static SIMD_INLINE Vec<Short, 32> set1(Short a, Integer<32>)
{
  return _mm256_set1_epi16(a);
}

static SIMD_INLINE Vec<Int, 32> set1(Int a, Integer<32>)
{
  return _mm256_set1_epi32(a);
}

static SIMD_INLINE Vec<Float, 32> set1(Float a, Integer<32>)
{
  return _mm256_set1_ps(a);
}

static SIMD_INLINE Vec<Double, 32> set1(Double a, Integer<32>)
{
  return _mm256_set1_pd(a);
}

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 32> load(const T *const p, Integer<32>)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  return _mm256_load_si256((__m256i *) p);
}

static SIMD_INLINE Vec<Float, 32> load(const Float *const p, Integer<32>)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  return _mm256_load_ps(p);
}

static SIMD_INLINE Vec<Double, 32> load(const Double *const p, Integer<32>)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  return _mm256_load_pd(p);
}

// ---------------------------------------------------------------------------
// loadu
// ---------------------------------------------------------------------------

template <typename T>
static SIMD_INLINE Vec<T, 32> loadu(const T *const p, Integer<32>)
{
  return _mm256_loadu_si256((__m256i *) p);
}

static SIMD_INLINE Vec<Float, 32> loadu(const Float *const p, Integer<32>)
{
  return _mm256_loadu_ps(p);
}

static SIMD_INLINE Vec<Double, 32> loadu(const Double *const p, Integer<32>)
{
  return _mm256_loadu_pd(p);
}

// ---------------------------------------------------------------------------
// store
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void store(T *const p, const Vec<T, 32> &a)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  _mm256_store_si256((__m256i *) p, a);
}

// float version
static SIMD_INLINE void store(Float *const p, const Vec<Float, 32> &a)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  _mm256_store_ps(p, a);
}

// double version
static SIMD_INLINE void store(Double *const p, const Vec<Double, 32> &a)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  _mm256_store_pd(p, a);
}

// ---------------------------------------------------------------------------
// storeu
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void storeu(T *const p, const Vec<T, 32> &a)
{
  _mm256_storeu_si256((__m256i *) p, a);
}

// float version
static SIMD_INLINE void storeu(Float *const p, const Vec<Float, 32> &a)
{
  _mm256_storeu_ps(p, a);
}

// double version
static SIMD_INLINE void storeu(Double *const p, const Vec<Double, 32> &a)
{
  _mm256_storeu_pd(p, a);
}

// ---------------------------------------------------------------------------
// stream_store
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE void stream_store(T *const p, const Vec<T, 32> &a)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  _mm256_stream_si256((__m256i *) p, a);
}

// float version
static SIMD_INLINE void stream_store(Float *const p, const Vec<Float, 32> &a)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  _mm256_stream_ps(p, a);
}

// double version
static SIMD_INLINE void stream_store(Double *const p, const Vec<Double, 32> &a)
{
  // AVX load and store instructions need alignment to 32 byte
  // (lower 5 bit need to be zero)
  SIMD_CHECK_ALIGNMENT(p, 32);
  _mm256_stream_pd(p, a);
}

// ---------------------------------------------------------------------------
// extract
// ---------------------------------------------------------------------------

template <int IMM>
static SIMD_INLINE Byte extract(const Vec<Byte, 32> &a)
{
  return x_mm256_extract_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE SignedByte extract(const Vec<SignedByte, 32> &a)
{
  return x_mm256_extract_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Word extract(const Vec<Word, 32> &a)
{
  return x_mm256_extract_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Short extract(const Vec<Short, 32> &a)
{
  return x_mm256_extract_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Int extract(const Vec<Int, 32> &a)
{
  // TODO: extract: is conversion from return type int to Int always safe?
  return x_mm256_extract_epi32<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Float extract(const Vec<Float, 32> &a)
{
  SIMD_RETURN_REINTERPRET_INT(
    Float, extract<IMM>(reinterpret(a, OutputType<Int>())), 0);
}

template <int IMM>
static SIMD_INLINE Double extract(const Vec<Double, 32> &a)
{
  const uint64_t resInt = x_mm256_extract_epi64<IMM>(_mm256_castpd_si256(a));
  Double res;
  memcpy(&res, &resInt, sizeof(res));
  return res;
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> add(const Vec<Byte, 32> &a,
                                     const Vec<Byte, 32> &b)
{
  return _mm256_add_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> add(const Vec<SignedByte, 32> &a,
                                           const Vec<SignedByte, 32> &b)
{
  return _mm256_add_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> add(const Vec<Word, 32> &a,
                                     const Vec<Word, 32> &b)
{
  return _mm256_add_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 32> add(const Vec<Short, 32> &a,
                                      const Vec<Short, 32> &b)
{
  return _mm256_add_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> add(const Vec<Int, 32> &a,
                                    const Vec<Int, 32> &b)
{
  return _mm256_add_epi32(a, b);
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> add(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(add(a.lo(), b.lo()), add(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> add(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  return _mm256_add_ps(a, b);
}

static SIMD_INLINE Vec<Double, 32> add(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  return _mm256_add_pd(a, b);
}

// ---------------------------------------------------------------------------
// adds
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> adds(const Vec<Byte, 32> &a,
                                      const Vec<Byte, 32> &b)
{
  return _mm256_adds_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> adds(const Vec<SignedByte, 32> &a,
                                            const Vec<SignedByte, 32> &b)
{
  return _mm256_adds_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> adds(const Vec<Word, 32> &a,
                                      const Vec<Word, 32> &b)
{
  return _mm256_adds_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 32> adds(const Vec<Short, 32> &a,
                                       const Vec<Short, 32> &b)
{
  return _mm256_adds_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> adds(const Vec<Int, 32> &a,
                                     const Vec<Int, 32> &b)
{
  // 09. Mar 23 (Jonas Keller): added workaround so that this function is
  // saturated

  // _mm256_adds_epi32 does not exist, workaround:
  // Hacker's Delight, 2-13 Overflow Detection: "Signed integer overflow of
  // addition occurs if and only if the operands have the same sign and the
  // sum has a sign opposite to that of the operands."
  __m256i sum             = _mm256_add_epi32(a, b);
  __m256i opsHaveDiffSign = _mm256_xor_si256(a, b);
  __m256i sumHasDiffSign  = _mm256_xor_si256(a, sum);
  // indicates when an overflow has occurred
  __m256i overflow =
    _mm256_srai_epi32(_mm256_andnot_si256(opsHaveDiffSign, sumHasDiffSign), 31);
  // saturated sum for if overflow occurred (0x7FFFFFFF=max positive int, when
  // sign of a (and thus b as well) is 0, 0x80000000=min negative int, when sign
  // of a (and thus b as well) is 1)
  __m256i saturatedSum =
    _mm256_xor_si256(_mm256_srai_epi32(a, 31), _mm256_set1_epi32(0x7FFFFFFF));
  // return saturated sum if overflow occurred, otherwise return sum
  return _mm256_or_si256(_mm256_andnot_si256(overflow, sum),
                         _mm256_and_si256(overflow, saturatedSum));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> adds(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(adds(a.lo(), b.lo()), adds(a.hi(), b.hi()));
}

#endif

// Float not saturated
static SIMD_INLINE Vec<Float, 32> adds(const Vec<Float, 32> &a,
                                       const Vec<Float, 32> &b)
{
  return _mm256_add_ps(a, b);
}

// Double not saturated
static SIMD_INLINE Vec<Double, 32> adds(const Vec<Double, 32> &a,
                                        const Vec<Double, 32> &b)
{
  return _mm256_add_pd(a, b);
}

// ---------------------------------------------------------------------------
// sub
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> sub(const Vec<Byte, 32> &a,
                                     const Vec<Byte, 32> &b)
{
  return _mm256_sub_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> sub(const Vec<SignedByte, 32> &a,
                                           const Vec<SignedByte, 32> &b)
{
  return _mm256_sub_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> sub(const Vec<Word, 32> &a,
                                     const Vec<Word, 32> &b)
{
  return _mm256_sub_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 32> sub(const Vec<Short, 32> &a,
                                      const Vec<Short, 32> &b)
{
  return _mm256_sub_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> sub(const Vec<Int, 32> &a,
                                    const Vec<Int, 32> &b)
{
  return _mm256_sub_epi32(a, b);
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> sub(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(sub(a.lo(), b.lo()), sub(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> sub(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  return _mm256_sub_ps(a, b);
}

static SIMD_INLINE Vec<Double, 32> sub(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  return _mm256_sub_pd(a, b);
}

// ---------------------------------------------------------------------------
// subs
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> subs(const Vec<Byte, 32> &a,
                                      const Vec<Byte, 32> &b)
{
  return _mm256_subs_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> subs(const Vec<SignedByte, 32> &a,
                                            const Vec<SignedByte, 32> &b)
{
  return _mm256_subs_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> subs(const Vec<Word, 32> &a,
                                      const Vec<Word, 32> &b)
{
  return _mm256_subs_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 32> subs(const Vec<Short, 32> &a,
                                       const Vec<Short, 32> &b)
{
  return _mm256_subs_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> subs(const Vec<Int, 32> &a,
                                     const Vec<Int, 32> &b)
{
  // 09. Mar 23 (Jonas Keller): added workaround so that this function is
  // saturated

  // _mm256_subs_epi32 does not exist, workaround:
  // Hacker's Delight, 2-13 Overflow Detection: "[...] overflow in the final
  // value of x−y [...] occurs if and only if x and y have opposite signs and
  // the sign of x−y [...] is opposite to that of x [...]"
  __m256i diff            = _mm256_sub_epi32(a, b);
  __m256i opsHaveDiffSign = _mm256_xor_si256(a, b);
  __m256i diffHasDiffSign = _mm256_xor_si256(a, diff);
  // indicates when an overflow has occurred
  __m256i overflow =
    _mm256_srai_epi32(_mm256_and_si256(opsHaveDiffSign, diffHasDiffSign), 31);
  // saturated diff for if overflow occurred (0x7FFFFFFF=max positive int, when
  // sign of a (and thus b as well) is 0, 0x80000000=min negative int, when sign
  // of a (and thus b as well) is 1)
  __m256i saturatedDiff =
    _mm256_xor_si256(_mm256_srai_epi32(a, 31), _mm256_set1_epi32(0x7FFFFFFF));
  // return saturated diff if overflow occurred, otherwise return diff
  return _mm256_or_si256(_mm256_andnot_si256(overflow, diff),
                         _mm256_and_si256(overflow, saturatedDiff));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> subs(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(subs(a.lo(), b.lo()), subs(a.hi(), b.hi()));
}

#endif

// Float not saturated
static SIMD_INLINE Vec<Float, 32> subs(const Vec<Float, 32> &a,
                                       const Vec<Float, 32> &b)
{
  return _mm256_sub_ps(a, b);
}

// Double not saturated
static SIMD_INLINE Vec<Double, 32> subs(const Vec<Double, 32> &a,
                                        const Vec<Double, 32> &b)
{
  return _mm256_sub_pd(a, b);
}

// ---------------------------------------------------------------------------
// neg (negate = two's complement or unary minus), only signed types
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<SignedByte, 32> neg(const Vec<SignedByte, 32> &a)
{
  return _mm256_sub_epi8(_mm256_setzero_si256(), a);
}

static SIMD_INLINE Vec<Short, 32> neg(const Vec<Short, 32> &a)
{
  return _mm256_sub_epi16(_mm256_setzero_si256(), a);
}

static SIMD_INLINE Vec<Int, 32> neg(const Vec<Int, 32> &a)
{
  return _mm256_sub_epi32(_mm256_setzero_si256(), a);
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> neg(const Vec<T, 32> &a)
{
  return Vec<T, 32>(neg(a.lo()), neg(a.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> neg(const Vec<Float, 32> &a)
{
  return _mm256_sub_ps(_mm256_setzero_ps(), a);
}

static SIMD_INLINE Vec<Double, 32> neg(const Vec<Double, 32> &a)
{
  return _mm256_sub_pd(_mm256_setzero_pd(), a);
}

// ---------------------------------------------------------------------------
// min
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> min(const Vec<Byte, 32> &a,
                                     const Vec<Byte, 32> &b)
{
  return _mm256_min_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> min(const Vec<SignedByte, 32> &a,
                                           const Vec<SignedByte, 32> &b)
{
  return _mm256_min_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> min(const Vec<Word, 32> &a,
                                     const Vec<Word, 32> &b)
{
  return _mm256_min_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 32> min(const Vec<Short, 32> &a,
                                      const Vec<Short, 32> &b)
{
  return _mm256_min_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> min(const Vec<Int, 32> &a,
                                    const Vec<Int, 32> &b)
{
  return _mm256_min_epi32(a, b);
}

// there is an unsigned version of min for 32 bit but we currently
// don't have an element type for it

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> min(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(min(a.lo(), b.lo()), min(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> min(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  return _mm256_min_ps(a, b);
}

static SIMD_INLINE Vec<Double, 32> min(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  return _mm256_min_pd(a, b);
}

// ---------------------------------------------------------------------------
// max
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> max(const Vec<Byte, 32> &a,
                                     const Vec<Byte, 32> &b)
{
  return _mm256_max_epu8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> max(const Vec<SignedByte, 32> &a,
                                           const Vec<SignedByte, 32> &b)
{
  return _mm256_max_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> max(const Vec<Word, 32> &a,
                                     const Vec<Word, 32> &b)
{
  return _mm256_max_epu16(a, b);
}

static SIMD_INLINE Vec<Short, 32> max(const Vec<Short, 32> &a,
                                      const Vec<Short, 32> &b)
{
  return _mm256_max_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> max(const Vec<Int, 32> &a,
                                    const Vec<Int, 32> &b)
{
  return _mm256_max_epi32(a, b);
}

// there is an unsigned version of max for 32 bit but we currently
// don't have an element type for it

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> max(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(max(a.lo(), b.lo()), max(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> max(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  return _mm256_max_ps(a, b);
}

static SIMD_INLINE Vec<Double, 32> max(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  return _mm256_max_pd(a, b);
}

// ---------------------------------------------------------------------------
// mul, div
// ---------------------------------------------------------------------------

// TODO: add mul/div versions for int types? or make special versions of mul
// TODO: and div where the result is scaled?

static SIMD_INLINE Vec<Float, 32> mul(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  return _mm256_mul_ps(a, b);
}

static SIMD_INLINE Vec<Double, 32> mul(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  return _mm256_mul_pd(a, b);
}

static SIMD_INLINE Vec<Float, 32> div(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  return _mm256_div_ps(a, b);
}

static SIMD_INLINE Vec<Double, 32> div(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  return _mm256_div_pd(a, b);
}

// ---------------------------------------------------------------------------
// ceil, floor, round, truncate
// ---------------------------------------------------------------------------

// 25. Mar 23 (Jonas Keller): added versions for integer types

// versions for integer types do nothing:

template <typename T>
static SIMD_INLINE Vec<T, 32> ceil(const Vec<T, 32> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 32> floor(const Vec<T, 32> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 32> round(const Vec<T, 32> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

template <typename T>
static SIMD_INLINE Vec<T, 32> truncate(const Vec<T, 32> &a)
{
  static_assert(std::is_integral<T>::value, "");
  return a;
}

static SIMD_INLINE Vec<Float, 32> ceil(const Vec<Float, 32> &a)
{
  return _mm256_ceil_ps(a);
}

static SIMD_INLINE Vec<Double, 32> ceil(const Vec<Double, 32> &a)
{
  return _mm256_ceil_pd(a);
}

static SIMD_INLINE Vec<Float, 32> floor(const Vec<Float, 32> &a)
{
  return _mm256_floor_ps(a);
}

static SIMD_INLINE Vec<Double, 32> floor(const Vec<Double, 32> &a)
{
  return _mm256_floor_pd(a);
}

static SIMD_INLINE Vec<Float, 32> round(const Vec<Float, 32> &a)
{
  // old: use _MM_SET_ROUNDING_MODE to adjust rounding direction
  // return _mm256_round_ps(a, _MM_FROUND_CUR_DIRECTION);
  // new  4. Aug 16 (rm): round to nearest, and suppress exceptions
  return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 32> round(const Vec<Double, 32> &a)
{
  return _mm256_round_pd(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Float, 32> truncate(const Vec<Float, 32> &a)
{
  return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

static SIMD_INLINE Vec<Double, 32> truncate(const Vec<Double, 32> &a)
{
  return _mm256_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
}

// ---------------------------------------------------------------------------
// elementary mathematical functions
// ---------------------------------------------------------------------------

// estimate of a reciprocal
static SIMD_INLINE Vec<Float, 32> rcp(const Vec<Float, 32> &a)
{
  return _mm256_rcp_ps(a);
}

static SIMD_INLINE Vec<Double, 32> rcp(const Vec<Double, 32> &a)
{
  // _mm256_rcp_pd does not exist
  return Vec<Double, 32>(rcp(a.lo()), rcp(a.hi()));
}

// estimate of reverse square root
static SIMD_INLINE Vec<Float, 32> rsqrt(const Vec<Float, 32> &a)
{
  return _mm256_rsqrt_ps(a);
}

static SIMD_INLINE Vec<Double, 32> rsqrt(const Vec<Double, 32> &a)
{
  // _mm256_rsqrt_pd does not exist
  return Vec<Double, 32>(rsqrt(a.lo()), rsqrt(a.hi()));
}

// square root
static SIMD_INLINE Vec<Float, 32> sqrt(const Vec<Float, 32> &a)
{
  return _mm256_sqrt_ps(a);
}

static SIMD_INLINE Vec<Double, 32> sqrt(const Vec<Double, 32> &a)
{
  return _mm256_sqrt_pd(a);
}

// ---------------------------------------------------------------------------
// abs
// ---------------------------------------------------------------------------

// 25. Mar 25 (Jonas Keller): added abs for unsigned integers

// unsigned integers
template <typename T, SIMD_ENABLE_IF(std::is_unsigned<T>::value
                                       &&std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, 32> abs(const Vec<T, 32> &a)
{
  return a;
}

static SIMD_INLINE Vec<SignedByte, 32> abs(const Vec<SignedByte, 32> &a)
{
#ifdef __AVX2__
  return _mm256_abs_epi8(a);
#else
  // non-avx2 workaround
  return Vec<SignedByte, 32>(abs(a.lo()), abs(a.hi()));
#endif
}

static SIMD_INLINE Vec<Short, 32> abs(const Vec<Short, 32> &a)
{
#ifdef __AVX2__
  return _mm256_abs_epi16(a);
#else
  // non-avx2 workaround
  return Vec<Short, 32>(abs(a.lo()), abs(a.hi()));
#endif
}

static SIMD_INLINE Vec<Int, 32> abs(const Vec<Int, 32> &a)
{
#ifdef __AVX2__
  return _mm256_abs_epi32(a);
#else
  // non-avx2 workaround
  return Vec<Int, 32>(abs(a.lo()), abs(a.hi()));
#endif
}

static SIMD_INLINE Vec<Float, 32> abs(const Vec<Float, 32> &a)
{
  // there's no _mm256_abs_ps, we have to emulated it:
  // -0.0F is 0x8000000, 0x7fffffff by andnot, sign bit is cleared
  return _mm256_andnot_ps(_mm256_set1_ps(-0.0F), a);
}

static SIMD_INLINE Vec<Double, 32> abs(const Vec<Double, 32> &a)
{
  // there's no _mm256_abs_pd, we have to emulated it:
  // -0.0 is 0x8000000000000000, 0x7fffffffffffffff by andnot, sign bit is
  // cleared
  return _mm256_andnot_pd(_mm256_set1_pd(-0.0), a);
}

// ---------------------------------------------------------------------------
// unpacklo
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<0>, Bytes<1>)
{
  return x_mm256_unpacklo_epi8(x_mm256_transpose4x64_epi64(a),
                               x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<0>, Bytes<2>)
{
  return x_mm256_unpacklo_epi16(x_mm256_transpose4x64_epi64(a),
                                x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<0>, Bytes<4>)
{
  return x_mm256_unpacklo_epi32(x_mm256_transpose4x64_epi64(a),
                                x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<0>, Bytes<8>)
{
  return x_mm256_unpacklo_epi64(x_mm256_transpose4x64_epi64(a),
                                x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<0>, Bytes<16>)
{
  return _mm256_permute2f128_si256(a, b, _MM_SHUFFLE(0, 2, 0, 0));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b, Part<0>,
                                         Bytes<4>)
{
  return _mm256_unpacklo_ps(x_mm256_transpose4x64_ps(a),
                            x_mm256_transpose4x64_ps(b));
}

// float versions
static SIMD_INLINE Vec<Float, 32> unpack(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b, Part<0>,
                                         Bytes<8>)
{
  return x_mm256_unpacklo_2ps(x_mm256_transpose4x64_ps(a),
                              x_mm256_transpose4x64_ps(b));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b, Part<0>,
                                         Bytes<16>)
{
  return _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 2, 0, 0));
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack(const Vec<Double, 32> &a,
                                          const Vec<Double, 32> &b, Part<0>,
                                          Bytes<8>)
{
  return _mm256_unpacklo_pd(x_mm256_transpose4x64_pd(a),
                            x_mm256_transpose4x64_pd(b));
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack(const Vec<Double, 32> &a,
                                          const Vec<Double, 32> &b, Part<0>,
                                          Bytes<16>)
{
  return _mm256_permute2f128_pd(a, b, _MM_SHUFFLE(0, 2, 0, 0));
}

// ---------------------------------------------------------------------------
// unpackhi
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<1>, Bytes<1>)
{
  return x_mm256_unpackhi_epi8(x_mm256_transpose4x64_epi64(a),
                               x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<1>, Bytes<2>)
{
  return x_mm256_unpackhi_epi16(x_mm256_transpose4x64_epi64(a),
                                x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<1>, Bytes<4>)
{
  return x_mm256_unpackhi_epi32(x_mm256_transpose4x64_epi64(a),
                                x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<1>, Bytes<8>)
{
  return x_mm256_unpackhi_epi64(x_mm256_transpose4x64_epi64(a),
                                x_mm256_transpose4x64_epi64(b));
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                     Part<1>, Bytes<16>)
{
  return _mm256_permute2f128_si256(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b, Part<1>,
                                         Bytes<4>)
{
  return _mm256_unpackhi_ps(x_mm256_transpose4x64_ps(a),
                            x_mm256_transpose4x64_ps(b));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b, Part<1>,
                                         Bytes<8>)
{
  return x_mm256_unpackhi_2ps(x_mm256_transpose4x64_ps(a),
                              x_mm256_transpose4x64_ps(b));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b, Part<1>,
                                         Bytes<16>)
{
  return _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack(const Vec<Double, 32> &a,
                                          const Vec<Double, 32> &b, Part<1>,
                                          Bytes<8>)
{
  return _mm256_unpackhi_pd(x_mm256_transpose4x64_pd(a),
                            x_mm256_transpose4x64_pd(b));
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack(const Vec<Double, 32> &a,
                                          const Vec<Double, 32> &b, Part<1>,
                                          Bytes<16>)
{
  return _mm256_permute2f128_pd(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// ---------------------------------------------------------------------------
// 16-byte-lane oriented unpacklo
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<0>, Bytes<1>)
{
  return x_mm256_unpacklo_epi8(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<0>, Bytes<2>)
{
  return x_mm256_unpacklo_epi16(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<0>, Bytes<4>)
{
  return x_mm256_unpacklo_epi32(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<0>, Bytes<8>)
{
  return x_mm256_unpacklo_epi64(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<0>, Bytes<16>)
{
  return _mm256_permute2f128_si256(a, b, _MM_SHUFFLE(0, 2, 0, 0));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack16(const Vec<Float, 32> &a,
                                           const Vec<Float, 32> &b, Part<0>,
                                           Bytes<4>)
{
  return _mm256_unpacklo_ps(a, b);
}

// float versions
static SIMD_INLINE Vec<Float, 32> unpack16(const Vec<Float, 32> &a,
                                           const Vec<Float, 32> &b, Part<0>,
                                           Bytes<8>)
{
  return x_mm256_unpacklo_2ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack16(const Vec<Float, 32> &a,
                                           const Vec<Float, 32> &b, Part<0>,
                                           Bytes<16>)
{
  return _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 2, 0, 0));
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack16(const Vec<Double, 32> &a,
                                            const Vec<Double, 32> &b, Part<0>,
                                            Bytes<8>)
{
  return _mm256_unpacklo_pd(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack16(const Vec<Double, 32> &a,
                                            const Vec<Double, 32> &b, Part<0>,
                                            Bytes<16>)
{
  return _mm256_permute2f128_pd(a, b, _MM_SHUFFLE(0, 2, 0, 0));
}

// ---------------------------------------------------------------------------
// 128-bit-lane oriented unpackhi
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<1>, Bytes<1>)
{
  return x_mm256_unpackhi_epi8(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<1>, Bytes<2>)
{
  return x_mm256_unpackhi_epi16(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<1>, Bytes<4>)
{
  return x_mm256_unpackhi_epi32(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<1>, Bytes<8>)
{
  return x_mm256_unpackhi_epi64(a, b);
}

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> unpack16(const Vec<T, 32> &a, const Vec<T, 32> &b,
                                       Part<1>, Bytes<16>)
{
  return _mm256_permute2f128_si256(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack16(const Vec<Float, 32> &a,
                                           const Vec<Float, 32> &b, Part<1>,
                                           Bytes<4>)
{
  return _mm256_unpackhi_ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack16(const Vec<Float, 32> &a,
                                           const Vec<Float, 32> &b, Part<1>,
                                           Bytes<8>)
{
  return x_mm256_unpackhi_2ps(a, b);
}

// float version
static SIMD_INLINE Vec<Float, 32> unpack16(const Vec<Float, 32> &a,
                                           const Vec<Float, 32> &b, Part<1>,
                                           Bytes<16>)
{
  return _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack16(const Vec<Double, 32> &a,
                                            const Vec<Double, 32> &b, Part<1>,
                                            Bytes<8>)
{
  return _mm256_unpackhi_pd(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 32> unpack16(const Vec<Double, 32> &a,
                                            const Vec<Double, 32> &b, Part<1>,
                                            Bytes<16>)
{
  return _mm256_permute2f128_pd(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// ---------------------------------------------------------------------------
// extract 128-bit-lane as Vec<T, 16>
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// generalized extract of 128-bit-lanes
// IMM=0: first lane of input vector,
// IMM=1: second lane of input vector
// all integer versions
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 16> extractLane(const Vec<T, 32> &a)
{
  return _mm256_extractf128_si256(a, IMM);
}

// generalized extract of 128-bit-lanes
// IMM=0: first lane of input vector,
// IMM=1: second lane of input vector
// float version
template <int IMM>
static SIMD_INLINE Vec<Float, 16> extractLane(const Vec<Float, 32> &a)
{
  return _mm256_extractf128_ps(a, IMM);
}

// generalized extract of 128-bit-lanes
// IMM=0: first lane of input vector,
// IMM=1: second lane of input vector
// double version
template <int IMM>
static SIMD_INLINE Vec<Double, 16> extractLane(const Vec<Double, 32> &a)
{
  return _mm256_extractf128_pd(a, IMM);
}

// ---------------------------------------------------------------------------
// zip
// ---------------------------------------------------------------------------

// a, b are passed by-value to avoid problems with identical
// input/output args.

// here we typically have to transpose the inputs in the same way
// for both output computations, so we define separate functions for
// all T and Bytes<> (combinations of unpack functions above)

// all integer versions
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 32> a, const Vec<T, 32> b,
                            Vec<T, 32> &l, Vec<T, 32> &h, Bytes<1>)
{
  __m256i at = x_mm256_transpose4x64_epi64(a);
  __m256i bt = x_mm256_transpose4x64_epi64(b);
  l          = x_mm256_unpacklo_epi8(at, bt);
  h          = x_mm256_unpackhi_epi8(at, bt);
}

// all integer versions
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 32> a, const Vec<T, 32> b,
                            Vec<T, 32> &l, Vec<T, 32> &h, Bytes<2>)
{
  __m256i at = x_mm256_transpose4x64_epi64(a);
  __m256i bt = x_mm256_transpose4x64_epi64(b);
  l          = x_mm256_unpacklo_epi16(at, bt);
  h          = x_mm256_unpackhi_epi16(at, bt);
}

// all integer versions
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 32> a, const Vec<T, 32> b,
                            Vec<T, 32> &l, Vec<T, 32> &h, Bytes<4>)
{
  __m256i at = x_mm256_transpose4x64_epi64(a);
  __m256i bt = x_mm256_transpose4x64_epi64(b);
  l          = x_mm256_unpacklo_epi32(at, bt);
  h          = x_mm256_unpackhi_epi32(at, bt);
}

// all integer versions
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 32> a, const Vec<T, 32> b,
                            Vec<T, 32> &l, Vec<T, 32> &h, Bytes<8>)
{
  __m256i at = x_mm256_transpose4x64_epi64(a);
  __m256i bt = x_mm256_transpose4x64_epi64(b);
  l          = x_mm256_unpacklo_epi64(at, bt);
  h          = x_mm256_unpackhi_epi64(at, bt);
}

// all integer versions
template <typename T>
static SIMD_INLINE void zip(const Vec<T, 32> a, const Vec<T, 32> b,
                            Vec<T, 32> &l, Vec<T, 32> &h, Bytes<16>)
{
  l = _mm256_permute2f128_si256(a, b, _MM_SHUFFLE(0, 2, 0, 0));
  h = _mm256_permute2f128_si256(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 32> a, const Vec<Float, 32> b,
                            Vec<Float, 32> &l, Vec<Float, 32> &h, Bytes<4>)
{
  __m256 at = x_mm256_transpose4x64_ps(a);
  __m256 bt = x_mm256_transpose4x64_ps(b);
  l         = _mm256_unpacklo_ps(at, bt);
  h         = _mm256_unpackhi_ps(at, bt);
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 32> a, const Vec<Float, 32> b,
                            Vec<Float, 32> &l, Vec<Float, 32> &h, Bytes<8>)
{
  __m256 at = x_mm256_transpose4x64_ps(a);
  __m256 bt = x_mm256_transpose4x64_ps(b);
  l         = x_mm256_unpacklo_2ps(at, bt);
  h         = x_mm256_unpackhi_2ps(at, bt);
}

// float version
static SIMD_INLINE void zip(const Vec<Float, 32> a, const Vec<Float, 32> b,
                            Vec<Float, 32> &l, Vec<Float, 32> &h, Bytes<16>)
{
  l = _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 2, 0, 0));
  h = _mm256_permute2f128_ps(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// double version
static SIMD_INLINE void zip(const Vec<Double, 32> a, const Vec<Double, 32> b,
                            Vec<Double, 32> &l, Vec<Double, 32> &h, Bytes<8>)
{
  __m256d at = x_mm256_transpose4x64_pd(a);
  __m256d bt = x_mm256_transpose4x64_pd(b);
  l          = _mm256_unpacklo_pd(at, bt);
  h          = _mm256_unpackhi_pd(at, bt);
}

// double version
static SIMD_INLINE void zip(const Vec<Double, 32> a, const Vec<Double, 32> b,
                            Vec<Double, 32> &l, Vec<Double, 32> &h, Bytes<16>)
{
  l = _mm256_permute2f128_pd(a, b, _MM_SHUFFLE(0, 2, 0, 0));
  h = _mm256_permute2f128_pd(a, b, _MM_SHUFFLE(0, 3, 0, 1));
}

// ---------------------------------------------------------------------------
// zip hub
// ---------------------------------------------------------------------------

// zips blocks of NUM_ELEMS elements of type T
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void zip(const Vec<T, 32> a, const Vec<T, 32> b,
                            Vec<T, 32> &l, Vec<T, 32> &h)
{
  return zip(a, b, l, h, Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// zip16 hub (16-byte-lane oriented zip)
// ---------------------------------------------------------------------------

// contributed by Adam Marschall

// zips blocks of NUM_ELEMS elements of type T
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void zip16(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h)
{
  l = unpack16(a, b, Part<0>(), Bytes<NUM_ELEMS * sizeof(T)>());
  h = unpack16(a, b, Part<1>(), Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// unzip
// ---------------------------------------------------------------------------

// a, b are passed by-value to avoid problems with identical input/output args.

// here we typically have to transpose the inputs in the same way
// for both output computations, so we define separate functions for
// all T and Bytes<> (combinations of unpack functions above)

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h, Bytes<1>)
{
  // mask is hopefully only set once if unzip is used multiple times
  __m256i mask =
    _mm256_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0, 15,
                    13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
  __m256i atmp = x_mm256_transpose4x64_epi64(x_mm256_shuffle_epi8(a, mask));
  __m256i btmp = x_mm256_transpose4x64_epi64(x_mm256_shuffle_epi8(b, mask));
  l            = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h            = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h, Bytes<2>)
{
  // mask is hopefully only set once if unzip is used multiple times
  __m256i mask =
    _mm256_set_epi8(15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0, 15,
                    14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);
  __m256i atmp = x_mm256_transpose4x64_epi64(x_mm256_shuffle_epi8(a, mask));
  __m256i btmp = x_mm256_transpose4x64_epi64(x_mm256_shuffle_epi8(b, mask));
  l            = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h            = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h, Bytes<4>)
{
  __m256i atmp = x_mm256_transpose4x64_epi64(
    x_mm256_shuffle_epi32<_MM_SHUFFLE(3, 1, 2, 0)>(a));
  __m256i btmp = x_mm256_transpose4x64_epi64(
    x_mm256_shuffle_epi32<_MM_SHUFFLE(3, 1, 2, 0)>(b));
  l = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// all integer versions
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h, Bytes<8>)
{
  __m256i atmp = x_mm256_transpose4x64_epi64(a);
  __m256i btmp = x_mm256_transpose4x64_epi64(b);
  l            = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h            = _mm256_permute2f128_si256(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// all types
template <typename T>
static SIMD_INLINE void unzip(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h, Bytes<16>)
{
  l = unpack(a, b, Part<0>(), Bytes<16>());
  h = unpack(a, b, Part<1>(), Bytes<16>());
}

// float version
static SIMD_INLINE void unzip(const Vec<Float, 32> a, const Vec<Float, 32> b,
                              Vec<Float, 32> &l, Vec<Float, 32> &h, Bytes<4>)
{
  __m256 atmp =
    x_mm256_transpose4x64_ps(_mm256_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 2, 0)));
  __m256 btmp =
    x_mm256_transpose4x64_ps(_mm256_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 2, 0)));
  l = _mm256_permute2f128_ps(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h = _mm256_permute2f128_ps(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// float version
static SIMD_INLINE void unzip(const Vec<Float, 32> a, const Vec<Float, 32> b,
                              Vec<Float, 32> &l, Vec<Float, 32> &h, Bytes<8>)
{
  __m256 atmp = x_mm256_transpose4x64_ps(a);
  __m256 btmp = x_mm256_transpose4x64_ps(b);
  l           = _mm256_permute2f128_ps(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h           = _mm256_permute2f128_ps(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// double version
static SIMD_INLINE void unzip(const Vec<Double, 32> a, const Vec<Double, 32> b,
                              Vec<Double, 32> &l, Vec<Double, 32> &h, Bytes<8>)
{
  __m256d atmp = x_mm256_transpose4x64_pd(a);
  __m256d btmp = x_mm256_transpose4x64_pd(b);
  l            = _mm256_permute2f128_pd(atmp, btmp, _MM_SHUFFLE(0, 2, 0, 0));
  h            = _mm256_permute2f128_pd(atmp, btmp, _MM_SHUFFLE(0, 3, 0, 1));
}

// ---------------------------------------------------------------------------
// unzip hub
// ---------------------------------------------------------------------------

// hub
template <int NUM_ELEMS, typename T>
static SIMD_INLINE void unzip(const Vec<T, 32> a, const Vec<T, 32> b,
                              Vec<T, 32> &l, Vec<T, 32> &h)
{
  return unzip(a, b, l, h, Bytes<NUM_ELEMS * sizeof(T)>());
}

// ---------------------------------------------------------------------------
// packs
// ---------------------------------------------------------------------------

// ========== signed -> signed ==========

static SIMD_INLINE Vec<SignedByte, 32> packs(const Vec<Short, 32> &a,
                                             const Vec<Short, 32> &b,
                                             OutputType<SignedByte>)
{
  return x_mm256_transpose4x64_epi64(x_mm256_packs_epi16(a, b));
}

static SIMD_INLINE Vec<Short, 32> packs(const Vec<Int, 32> &a,
                                        const Vec<Int, 32> &b,
                                        OutputType<Short>)
{
  return x_mm256_transpose4x64_epi64(x_mm256_packs_epi32(a, b));
}

static SIMD_INLINE Vec<Short, 32> packs(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b,
                                        OutputType<Short>)
{
  return packs(cvts(a, OutputType<Int>()), cvts(b, OutputType<Int>()),
               OutputType<Short>());
}

static SIMD_INLINE Vec<Float, 32> packs(const Vec<Double, 32> &a,
                                        const Vec<Double, 32> &b,
                                        OutputType<Float>)
{
  return _mm256_set_m128(_mm256_cvtpd_ps(b), _mm256_cvtpd_ps(a));
}

static SIMD_INLINE Vec<Int, 32> packs(const Vec<Double, 32> &a,
                                      const Vec<Double, 32> &b, OutputType<Int>)
{
  const __m256d clip = _mm256_set1_pd(std::numeric_limits<Int>::max());
  return _mm256_set_m128i(_mm256_cvtpd_epi32(_mm256_min_pd(clip, b)),
                          _mm256_cvtpd_epi32(_mm256_min_pd(clip, a)));
}

// ========== signed -> unsigned ==========

// non-avx2 workaround
static SIMD_INLINE Vec<Byte, 32> packs(const Vec<Short, 32> &a,
                                       const Vec<Short, 32> &b,
                                       OutputType<Byte>)
{
  return x_mm256_transpose4x64_epi64(x_mm256_packus_epi16(a, b));
}

// non-avx2 workaround
static SIMD_INLINE Vec<Word, 32> packs(const Vec<Int, 32> &a,
                                       const Vec<Int, 32> &b, OutputType<Word>)
{
  return x_mm256_transpose4x64_epi64(x_mm256_packus_epi32(a, b));
}

static SIMD_INLINE Vec<Word, 32> packs(const Vec<Float, 32> &a,
                                       const Vec<Float, 32> &b,
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

// 7. Aug 16 (rm):
// tried to remove this to SIMDVecExt.H, but then we get ambiguities with
// non-avx2 workaround

// all types
template <typename T>
static SIMD_INLINE void extend(const Vec<T, 32> &vIn, Vec<T, 32> vOut[1])
{
  vOut[0] = vIn;
}

// ---------------------------------------------------------------------------
// generalized extend: single stage
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 32> &vIn,
                               Vec<Short, 32> vOut[2])
{
  vOut[0] = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vIn));
  vOut[1] = _mm256_cvtepi8_epi16(_mm256_extractf128_si256(vIn, 1));
}

static SIMD_INLINE void extend(const Vec<Short, 32> &vIn, Vec<Int, 32> vOut[2])
{
  vOut[0] = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vIn));
  vOut[1] = _mm256_cvtepi16_epi32(_mm256_extractf128_si256(vIn, 1));
}

static SIMD_INLINE void extend(const Vec<Short, 32> &vIn,
                               Vec<Float, 32> vOut[2])
{
  vOut[0] =
    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(vIn)));
  vOut[1] =
    _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extractf128_si256(vIn, 1)));
}

static SIMD_INLINE void extend(const Vec<Int, 32> &vIn, Vec<Double, 32> vOut[2])
{
  vOut[0] = _mm256_cvtepi32_pd(_mm256_castsi256_si128(vIn));
  vOut[1] = _mm256_cvtepi32_pd(_mm256_extractf128_si256(vIn, 1));
}

static SIMD_INLINE void extend(const Vec<Float, 32> &vIn,
                               Vec<Double, 32> vOut[2])
{
  vOut[0] = _mm256_cvtps_pd(_mm256_castps256_ps128(vIn));
  vOut[1] = _mm256_cvtps_pd(_mm256_extractf128_ps(vIn, 1));
}

// unsigned -> unsigned

static SIMD_INLINE void extend(const Vec<Byte, 32> &vIn, Vec<Word, 32> vOut[2])
{
  // there's no _mm256_cvtepu8_epu16()
  Vec<Byte, 32> zero = setzero(OutputType<Byte>(), Integer<32>());
  // 16. Jul 16 (rm): here we avoid to use generalized unpack from
  // SIMDVecExt.H
  vOut[0] = unpack(vIn, zero, Part<0>(), Bytes<1>());
  vOut[1] = unpack(vIn, zero, Part<1>(), Bytes<1>());
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 32> &vIn, Vec<Short, 32> vOut[2])
{
  vOut[0] = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vIn));
  vOut[1] = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(vIn, 1));
}

static SIMD_INLINE void extend(const Vec<Word, 32> &vIn, Vec<Int, 32> vOut[2])
{
  vOut[0] = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(vIn));
  vOut[1] = _mm256_cvtepu16_epi32(_mm256_extractf128_si256(vIn, 1));
}

static SIMD_INLINE void extend(const Vec<Word, 32> &vIn, Vec<Float, 32> vOut[2])
{
  vOut[0] =
    _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(vIn)));
  vOut[1] =
    _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extractf128_si256(vIn, 1)));
}

// ---------------------------------------------------------------------------
// generalized extend: two stages
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 32> &vIn,
                               Vec<Int, 32> vOut[4])
{
  __m128i vInLo128 = _mm256_castsi256_si128(vIn);
  vOut[0]          = _mm256_cvtepi8_epi32(vInLo128);
  vOut[1]          = _mm256_cvtepi8_epi32(_mm_srli_si128(vInLo128, 8));
  __m128i vInHi128 = _mm256_extractf128_si256(vIn, 1);
  vOut[2]          = _mm256_cvtepi8_epi32(vInHi128);
  vOut[3]          = _mm256_cvtepi8_epi32(_mm_srli_si128(vInHi128, 8));
}

static SIMD_INLINE void extend(const Vec<SignedByte, 32> &vIn,
                               Vec<Float, 32> vOut[4])
{
  Vec<Int, 32> vTmp[4];
  extend(vIn, vTmp);
  for (int i = 0; i < 4; i++) vOut[i] = cvts(vTmp[i], OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Short, 32> &vIn,
                               Vec<Double, 32> vOut[4])
{
  Vec<Int, 32> vTmp[2];
  extend(vIn, vTmp);
  extend(vTmp[0], vOut);
  extend(vTmp[1], vOut + 2);
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 32> &vIn, Vec<Int, 32> vOut[4])
{
  __m128i vInLo128 = _mm256_castsi256_si128(vIn);
  vOut[0]          = _mm256_cvtepu8_epi32(vInLo128);
  vOut[1]          = _mm256_cvtepu8_epi32(_mm_srli_si128(vInLo128, 8));
  __m128i vInHi128 = _mm256_extractf128_si256(vIn, 1);
  vOut[2]          = _mm256_cvtepu8_epi32(vInHi128);
  vOut[3]          = _mm256_cvtepu8_epi32(_mm_srli_si128(vInHi128, 8));
}

static SIMD_INLINE void extend(const Vec<Byte, 32> &vIn, Vec<Float, 32> vOut[4])
{
  Vec<Int, 32> vTmp[4];
  extend(vIn, vTmp);
  for (int i = 0; i < 4; i++) vOut[i] = cvts(vTmp[i], OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Word, 32> &vIn,
                               Vec<Double, 32> vOut[4])
{
  Vec<Int, 32> vTmp[2];
  extend(vIn, vTmp);
  extend(vTmp[0], vOut);
  extend(vTmp[1], vOut + 2);
}

// ---------------------------------------------------------------------------
// generalized extend: three stages
// ---------------------------------------------------------------------------

// signed -> signed

static SIMD_INLINE void extend(const Vec<SignedByte, 32> &vIn,
                               Vec<Double, 32> vOut[8])
{
  Vec<Int, 32> vTmp[4];
  extend(vIn, vTmp);
  extend(vTmp[0], vOut);
  extend(vTmp[1], vOut + 2);
  extend(vTmp[2], vOut + 4);
  extend(vTmp[3], vOut + 6);
}

// unsigned -> signed

static SIMD_INLINE void extend(const Vec<Byte, 32> &vIn,
                               Vec<Double, 32> vOut[8])
{
  Vec<Int, 32> vTmp[4];
  extend(vIn, vTmp);
  extend(vTmp[0], vOut);
  extend(vTmp[1], vOut + 2);
  extend(vTmp[2], vOut + 4);
  extend(vTmp[3], vOut + 6);
}

#else // __AVX2__

// ---------------------------------------------------------------------------
// generalized extend: non-avx2 workaround
// ---------------------------------------------------------------------------

// non-avx2 workaround
template <typename Tout, typename Tin>
static SIMD_INLINE void extend(const Vec<Tin, 32> &vIn,
                               Vec<Tout, 32> vOut[sizeof(Tout) / sizeof(Tin)])
{
  const int nOut = sizeof(Tout) / sizeof(Tin), nOutHalf = nOut / 2;
  Vec<Tout, 16> vOutLo16[nOut], vOutHi16[nOut];
  extend(vIn.lo(), vOutLo16);
  extend(vIn.hi(), vOutHi16);
  for (int i = 0; i < nOutHalf; i++) {
    vOut[i]            = Vec<Tout, 32>(vOutLo16[2 * i], vOutLo16[2 * i + 1]);
    vOut[i + nOutHalf] = Vec<Tout, 32>(vOutHi16[2 * i], vOutHi16[2 * i + 1]);
  }
}

#endif

// ---------------------------------------------------------------------------
// generalized extend: special case int <-> float
// ---------------------------------------------------------------------------

static SIMD_INLINE void extend(const Vec<Int, 32> &vIn, Vec<Float, 32> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Float>());
}

static SIMD_INLINE void extend(const Vec<Float, 32> &vIn, Vec<Int, 32> vOut[1])
{
  vOut[0] = cvts(vIn, OutputType<Int>());
}

// ---------------------------------------------------------------------------
// srai
// ---------------------------------------------------------------------------

#ifdef __AVX2__
// 16. Oct 22 (Jonas Keller): added missing Byte and SignedByte versions

template <int IMM>
static SIMD_INLINE Vec<Byte, 32> srai(const Vec<Byte, 32> &a)
{
  return x_mm256_srai_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<SignedByte, 32> srai(const Vec<SignedByte, 32> &a)
{
  return x_mm256_srai_epi8<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Word, 32> srai(const Vec<Word, 32> &a)
{
  return x_mm256_srai_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 32> srai(const Vec<Short, 32> &a)
{
  return x_mm256_srai_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Int, 32> srai(const Vec<Int, 32> &a)
{
  return x_mm256_srai_epi32<IMM>(a);
}

#else

// non-avx2 workaround
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 32> srai(const Vec<T, 32> &a)
{
  return Vec<T, 32>(srai<IMM>(a.lo()), srai<IMM>(a.hi()));
}

#endif

// ---------------------------------------------------------------------------
// srli
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<Byte, 32> srli(const Vec<Byte, 32> &a)
{
  return _mm256_and_si256(_mm256_set1_epi8((int8_t) (0xff >> IMM)),
                          x_mm256_srli_epi32<IMM>(a));
}

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<SignedByte, 32> srli(const Vec<SignedByte, 32> &a)
{
  return _mm256_and_si256(_mm256_set1_epi8((int8_t) (0xff >> IMM)),
                          x_mm256_srli_epi32<IMM>(a));
}

template <int IMM>
static SIMD_INLINE Vec<Word, 32> srli(const Vec<Word, 32> &a)
{
  return x_mm256_srli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 32> srli(const Vec<Short, 32> &a)
{
  return x_mm256_srli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Int, 32> srli(const Vec<Int, 32> &a)
{
  return x_mm256_srli_epi32<IMM>(a);
}

#else

// non-avx2 workaround
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 32> srli(const Vec<T, 32> &a)
{
  return Vec<T, 32>(srli<IMM>(a.lo()), srli<IMM>(a.hi()));
}

#endif

// ---------------------------------------------------------------------------
// slli
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<Byte, 32> slli(const Vec<Byte, 32> &a)
{
  return _mm256_and_si256(
    _mm256_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << IMM))),
    x_mm256_slli_epi32<IMM>(a));
}

// https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
// License: not specified
template <int IMM>
static SIMD_INLINE Vec<SignedByte, 32> slli(const Vec<SignedByte, 32> &a)
{
  return _mm256_and_si256(
    _mm256_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << IMM))),
    x_mm256_slli_epi32<IMM>(a));
}

template <int IMM>
static SIMD_INLINE Vec<Word, 32> slli(const Vec<Word, 32> &a)
{
  return x_mm256_slli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Short, 32> slli(const Vec<Short, 32> &a)
{
  return x_mm256_slli_epi16<IMM>(a);
}

template <int IMM>
static SIMD_INLINE Vec<Int, 32> slli(const Vec<Int, 32> &a)
{
  return x_mm256_slli_epi32<IMM>(a);
}

#else

// non-avx2 workaround
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 32> slli(const Vec<T, 32> &a)
{
  return Vec<T, 32>(slli<IMM>(a.lo()), slli<IMM>(a.hi()));
}

#endif

// 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

// ---------------------------------------------------------------------------
// sra
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> sra(const Vec<Byte, 32> &a,
                                     const uint8_t count)
{
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm256_cmpgt_epi8(_mm256_setzero_si256(), a);
  }
  __m256i odd = _mm256_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m256i even =
    _mm256_sra_epi16(_mm256_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  return _mm256_blendv_epi8(even, odd, _mm256_set1_epi16((int16_t) 0xff00));
}

static SIMD_INLINE Vec<SignedByte, 32> sra(const Vec<SignedByte, 32> &a,
                                           const uint8_t count)
{
  if (count >= 8) {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm256_cmpgt_epi8(_mm256_setzero_si256(), a);
  }
  __m256i odd = _mm256_sra_epi16(a, _mm_cvtsi32_si128(count));
  __m256i even =
    _mm256_sra_epi16(_mm256_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
  return _mm256_blendv_epi8(even, odd, _mm256_set1_epi16((int16_t) 0xff00));
}

static SIMD_INLINE Vec<Word, 32> sra(const Vec<Word, 32> &a,
                                     const uint8_t count)
{
  return _mm256_sra_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 32> sra(const Vec<Short, 32> &a,
                                      const uint8_t count)
{
  return _mm256_sra_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Int, 32> sra(const Vec<Int, 32> &a, const uint8_t count)
{
  return _mm256_sra_epi32(a, _mm_cvtsi32_si128(count));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> sra(const Vec<T, 32> &a, const uint8_t count)
{
  return Vec<T, 32>(sra(a.lo(), count), sra(a.hi(), count));
}

#endif

// ---------------------------------------------------------------------------
// srl
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> srl(const Vec<Byte, 32> &a,
                                     const uint8_t count)
{
  return _mm256_and_si256(_mm256_srl_epi16(a, _mm_cvtsi32_si128(count)),
                          _mm256_set1_epi8((int8_t) (uint8_t) (0xff >> count)));
}

static SIMD_INLINE Vec<SignedByte, 32> srl(const Vec<SignedByte, 32> &a,
                                           const uint8_t count)
{
  return _mm256_and_si256(_mm256_srl_epi16(a, _mm_cvtsi32_si128(count)),
                          _mm256_set1_epi8((int8_t) (uint8_t) (0xff >> count)));
}

static SIMD_INLINE Vec<Word, 32> srl(const Vec<Word, 32> &a,
                                     const uint8_t count)
{
  return _mm256_srl_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 32> srl(const Vec<Short, 32> &a,
                                      const uint8_t count)
{
  return _mm256_srl_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Int, 32> srl(const Vec<Int, 32> &a, const uint8_t count)
{
  return _mm256_srl_epi32(a, _mm_cvtsi32_si128(count));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> srl(const Vec<T, 32> &a, const uint8_t count)
{
  return Vec<T, 32>(srl(a.lo(), count), srl(a.hi(), count));
}

#endif

// ---------------------------------------------------------------------------
// sll
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> sll(const Vec<Byte, 32> &a,
                                     const uint8_t count)
{
  return _mm256_and_si256(
    _mm256_sll_epi16(a, _mm_cvtsi32_si128(count)),
    _mm256_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << count))));
}

static SIMD_INLINE Vec<SignedByte, 32> sll(const Vec<SignedByte, 32> &a,
                                           const uint8_t count)
{
  return _mm256_and_si256(
    _mm256_sll_epi16(a, _mm_cvtsi32_si128(count)),
    _mm256_set1_epi8((int8_t) (uint8_t) (0xff & (0xff << count))));
}

static SIMD_INLINE Vec<Word, 32> sll(const Vec<Word, 32> &a,
                                     const uint8_t count)
{
  return _mm256_sll_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Short, 32> sll(const Vec<Short, 32> &a,
                                      const uint8_t count)
{
  return _mm256_sll_epi16(a, _mm_cvtsi32_si128(count));
}

static SIMD_INLINE Vec<Int, 32> sll(const Vec<Int, 32> &a, const uint8_t count)
{
  return _mm256_sll_epi32(a, _mm_cvtsi32_si128(count));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> sll(const Vec<T, 32> &a, const uint8_t count)
{
  return Vec<T, 32>(sll(a.lo(), count), sll(a.hi(), count));
}

#endif

// 19. Sep 22 (Jonas Keller):
// added Byte and SignedByte versions of hadd, hadds, hsub and hsubs
// added Word version of hadds and hsubs

// ---------------------------------------------------------------------------
// hadd
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 32> hadd(const Vec<Byte, 32> &a,
                                      const Vec<Byte, 32> &b)
{
  Vec<Byte, 32> x, y;
  unzip<1>(a, b, x, y);
  return add(x, y);
}

static SIMD_INLINE Vec<SignedByte, 32> hadd(const Vec<SignedByte, 32> &a,
                                            const Vec<SignedByte, 32> &b)
{
  Vec<SignedByte, 32> x, y;
  unzip<1>(a, b, x, y);
  return add(x, y);
}

static SIMD_INLINE Vec<Word, 32> hadd(const Vec<Word, 32> &a,
                                      const Vec<Word, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hadd_epi16(a, b));
}

static SIMD_INLINE Vec<Short, 32> hadd(const Vec<Short, 32> &a,
                                       const Vec<Short, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hadd_epi16(a, b));
}

static SIMD_INLINE Vec<Int, 32> hadd(const Vec<Int, 32> &a,
                                     const Vec<Int, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hadd_epi32(a, b));
}

static SIMD_INLINE Vec<Float, 32> hadd(const Vec<Float, 32> &a,
                                       const Vec<Float, 32> &b)
{
  return x_mm256_transpose4x64_ps(_mm256_hadd_ps(a, b));
}

static SIMD_INLINE Vec<Double, 32> hadd(const Vec<Double, 32> &a,
                                        const Vec<Double, 32> &b)
{
  return x_mm256_transpose4x64_pd(_mm256_hadd_pd(a, b));
}

// ---------------------------------------------------------------------------
// hadds
// ---------------------------------------------------------------------------

// 09. Mar 23 (Jonas Keller): made Int version of hadds saturating

template <typename T>
static SIMD_INLINE Vec<T, 32> hadds(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  Vec<T, 32> x, y;
  unzip<1>(a, b, x, y);
  return adds(x, y);
}

static SIMD_INLINE Vec<Short, 32> hadds(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hadds_epi16(a, b));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 32> hadds(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  return x_mm256_transpose4x64_ps(_mm256_hadd_ps(a, b));
}

// Double not saturated
static SIMD_INLINE Vec<Double, 32> hadds(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return x_mm256_transpose4x64_pd(_mm256_hadd_pd(a, b));
}

// ---------------------------------------------------------------------------
// hsub
// ---------------------------------------------------------------------------

static SIMD_INLINE Vec<Byte, 32> hsub(const Vec<Byte, 32> &a,
                                      const Vec<Byte, 32> &b)
{
  Vec<Byte, 32> x, y;
  unzip<1>(a, b, x, y);
  return sub(x, y);
}

static SIMD_INLINE Vec<SignedByte, 32> hsub(const Vec<SignedByte, 32> &a,
                                            const Vec<SignedByte, 32> &b)
{
  Vec<SignedByte, 32> x, y;
  unzip<1>(a, b, x, y);
  return sub(x, y);
}

static SIMD_INLINE Vec<Word, 32> hsub(const Vec<Word, 32> &a,
                                      const Vec<Word, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hsub_epi16(a, b));
}

static SIMD_INLINE Vec<Short, 32> hsub(const Vec<Short, 32> &a,
                                       const Vec<Short, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hsub_epi16(a, b));
}

static SIMD_INLINE Vec<Int, 32> hsub(const Vec<Int, 32> &a,
                                     const Vec<Int, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hsub_epi32(a, b));
}

static SIMD_INLINE Vec<Float, 32> hsub(const Vec<Float, 32> &a,
                                       const Vec<Float, 32> &b)
{
  return x_mm256_transpose4x64_ps(_mm256_hsub_ps(a, b));
}

static SIMD_INLINE Vec<Double, 32> hsub(const Vec<Double, 32> &a,
                                        const Vec<Double, 32> &b)
{
  return x_mm256_transpose4x64_pd(_mm256_hsub_pd(a, b));
}

// ---------------------------------------------------------------------------
// hsubs
// ---------------------------------------------------------------------------

// 09. Mar 23 (Jonas Keller): made Int version of hsubs saturating

template <typename T>
static SIMD_INLINE Vec<T, 32> hsubs(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  Vec<T, 32> x, y;
  unzip<1>(a, b, x, y);
  return subs(x, y);
}

static SIMD_INLINE Vec<Short, 32> hsubs(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return x_mm256_transpose4x64_epi64(x_mm256_hsubs_epi16(a, b));
}

// Float not saturated
static SIMD_INLINE Vec<Float, 32> hsubs(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  return x_mm256_transpose4x64_ps(_mm256_hsub_ps(a, b));
}

// Double not saturated
static SIMD_INLINE Vec<Double, 32> hsubs(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return x_mm256_transpose4x64_pd(_mm256_hsub_pd(a, b));
}

// ---------------------------------------------------------------------------
// element-wise shift right
// ---------------------------------------------------------------------------

template <int IMM, typename T>
static SIMD_INLINE Vec<T, 32> srle(const Vec<T, 32> &a)
{
  const __m256i aInt          = reinterpret(a, OutputType<Int>());
  const Vec<Int, 32> aShifted = x_mm256_srli256_si256<IMM * sizeof(T)>(aInt);
  return reinterpret(aShifted, OutputType<T>());
}

// ---------------------------------------------------------------------------
// element-wise shift left
// ---------------------------------------------------------------------------

template <int IMM, typename T>
static SIMD_INLINE Vec<T, 32> slle(const Vec<T, 32> &a)
{
  const __m256i aInt          = reinterpret(a, OutputType<Int>());
  const Vec<Int, 32> aShifted = x_mm256_slli256_si256<IMM * sizeof(T)>(aInt);
  return reinterpret(aShifted, OutputType<T>());
}

// ---------------------------------------------------------------------------
// alignre
// ---------------------------------------------------------------------------

// all integer versions
template <int IMM, typename T>
static SIMD_INLINE Vec<T, 32> alignre(const Vec<T, 32> &h, const Vec<T, 32> &l)
{
  return x_mm256_alignr256_epi8<IMM * sizeof(T)>(h, l);
}

// float version
template <int IMM>
static SIMD_INLINE Vec<Float, 32> alignre(const Vec<Float, 32> &h,
                                          const Vec<Float, 32> &l)
{
  return _mm256_castsi256_ps(x_mm256_alignr256_epi8<IMM * sizeof(Float)>(
    _mm256_castps_si256(h), _mm256_castps_si256(l)));
}

// double version
template <int IMM>
static SIMD_INLINE Vec<Double, 32> alignre(const Vec<Double, 32> &h,
                                           const Vec<Double, 32> &l)
{
  return _mm256_castsi256_pd(x_mm256_alignr256_epi8<IMM * sizeof(Double)>(
    _mm256_castpd_si256(h), _mm256_castpd_si256(l)));
}

// ---------------------------------------------------------------------------
// swizzle
// ---------------------------------------------------------------------------

// ---------- swizzle aux functions -----------

// alignoff is the element-wise offset (relates to size of byte)
template <int ALIGNOFF>
static SIMD_INLINE __m256i align_shuffle_256(__m256i lo, __m256i hi,
                                             __m256i mask)
{
  static_assert(ALIGNOFF >= 0 && ALIGNOFF < 32, "");
  return x_mm256_shuffle_epi8(x_mm256_alignr_epi8<ALIGNOFF>(hi, lo), mask);
}

// ---------- swizzle (AoS to SoA) ----------

// 01. Apr 23 (Jonas Keller): switched from using tag dispatching to using
// enable_if SFINAE, which allows more cases with the same implementation
// to be combined

// -------------------- n = 1 --------------------

// all types
template <typename T>
static SIMD_INLINE void swizzle(Vec<T, 32>[1], Integer<1>)
{
  // v remains unchanged
}

// -------------------- n = 2 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) <= 2 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 32> v[2], Integer<2>)
{
  Vec<T, 32> vs[2];
  swizzle_32_16<2>(v, vs);
  __m256i mask = x_mm256_duplicate_si128(get_swizzle_mask<2, T>());
  __m256i s[2];
  s[0] = x_mm256_shuffle_epi8(vs[0], mask);
  s[1] = x_mm256_shuffle_epi8(vs[1], mask);
  v[0] = x_mm256_unpacklo_epi64(s[0], s[1]);
  v[1] = x_mm256_unpackhi_epi64(s[0], s[1]);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[2], Integer<2>)
{
  const Vec<Float, 32> vFloat[2] = {
    reinterpret(v[0], OutputType<Float>()),
    reinterpret(v[1], OutputType<Float>()),
  };
  Vec<Float, 32> vs[2];
  swizzle_32_16<2>(vFloat, vs);
  const Vec<Float, 32> vOut[2] = {
    _mm256_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(2, 0, 2, 0)),
    _mm256_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(3, 1, 3, 1)),
  };
  v[0] = reinterpret(vOut[0], OutputType<T>());
  v[1] = reinterpret(vOut[1], OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[2], Integer<2>)
{
  const Vec<Double, 32> vDouble[2] = {
    reinterpret(v[0], OutputType<Double>()),
    reinterpret(v[1], OutputType<Double>()),
  };
  Vec<Double, 32> vs[2];
  swizzle_32_16<2>(vDouble, vs);
  const Vec<Double, 32> vOut[2] = {
    _mm256_shuffle_pd(vs[0], vs[1], 0),
    _mm256_shuffle_pd(vs[0], vs[1], 0xf),
  };
  v[0] = reinterpret(vOut[0], OutputType<T>());
  v[1] = reinterpret(vOut[1], OutputType<T>());
}

// -------------------- n = 3 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) <= 2 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 32> v[3], Integer<3>)
{
  Vec<T, 32> vs[3];
  swizzle_32_16<3>(v, vs);
  __m256i mask = x_mm256_duplicate_si128(get_swizzle_mask<3, T>());
  __m256i s0   = align_shuffle_256<0>(vs[0], vs[1], mask);
  __m256i s1   = align_shuffle_256<12>(vs[0], vs[1], mask);
  __m256i s2   = align_shuffle_256<8>(vs[1], vs[2], mask);
  __m256i s3   = align_shuffle_256<4>(vs[2], _mm256_undefined_si256(), mask);
  __m256i l01  = x_mm256_unpacklo_epi32(s0, s1);
  __m256i h01  = x_mm256_unpackhi_epi32(s0, s1);
  __m256i l23  = x_mm256_unpacklo_epi32(s2, s3);
  __m256i h23  = x_mm256_unpackhi_epi32(s2, s3);
  v[0]         = x_mm256_unpacklo_epi64(l01, l23);
  v[1]         = x_mm256_unpackhi_epi64(l01, l23);
  v[2]         = x_mm256_unpacklo_epi64(h01, h23);
}

// 32 bit types
// from Stan Melax: "3D Vector Normalization..."
// https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[3], Integer<3>)
{
  const Vec<Float, 32> vFloat[3] = {
    reinterpret(v[0], OutputType<Float>()),
    reinterpret(v[1], OutputType<Float>()),
    reinterpret(v[2], OutputType<Float>()),
  };
  Vec<Float, 32> vs[3];
  swizzle_32_16<3>(vFloat, vs);
  // x0y0z0x1 = v[0]
  // y1z1x2y2 = v[1]
  // z2x3y3z3 = v[2]
  const Vec<Float, 32> x2y2x3y3 =
    _mm256_shuffle_ps(vs[1], vs[2], _MM_SHUFFLE(2, 1, 3, 2));
  const Vec<Float, 32> y0z0y1z1 =
    _mm256_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(1, 0, 2, 1));
  const Vec<Float, 32> x0x1x2x3 =
    _mm256_shuffle_ps(vs[0], x2y2x3y3, _MM_SHUFFLE(2, 0, 3, 0));
  const Vec<Float, 32> y0y1y2y3 =
    _mm256_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3, 1, 2, 0));
  const Vec<Float, 32> z0z1z2z3 =
    _mm256_shuffle_ps(y0z0y1z1, vs[2], _MM_SHUFFLE(3, 0, 3, 1));
  v[0] = reinterpret(x0x1x2x3, OutputType<T>());
  v[1] = reinterpret(y0y1y2y3, OutputType<T>());
  v[2] = reinterpret(z0z1z2z3, OutputType<T>());
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[3], Integer<3>)
{
  const Vec<Double, 32> vDouble[3] = {
    reinterpret(v[0], OutputType<Double>()), // x0y0z0x1
    reinterpret(v[1], OutputType<Double>()), // y1z1x2y2
    reinterpret(v[2], OutputType<Double>()), // z2x3y3z3
  };
  Vec<Double, 32> vs[3];
  swizzle_32_16<3>(vDouble, vs);
  // vs[0] = x0y0x2y2
  // vs[1] = z0x1z2x3
  // vs[2] = y1z1y3z3
  const Vec<Double, 32> vOut[3] = {
    // x0x1x2x3
    _mm256_shuffle_pd(vs[0], vs[1], 0xa), // 0b1010
    // y0y1y2y3
    _mm256_shuffle_pd(vs[0], vs[2], 0x5), // 0b0101
    // z0z1z2z3
    _mm256_shuffle_pd(vs[1], vs[2], 0xa), // 0b1010
  };
  v[0] = reinterpret(vOut[0], OutputType<T>());
  v[1] = reinterpret(vOut[1], OutputType<T>());
  v[2] = reinterpret(vOut[2], OutputType<T>());
}

// -------------------- n = 4 --------------------

// 8 and 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF((sizeof(T) <= 2 && std::is_integral<T>::value))>
static SIMD_INLINE void swizzle(Vec<T, 32> v[4], Integer<4>)
{
  Vec<T, 32> vs[4];
  swizzle_32_16<4>(v, vs);
  __m256i mask = x_mm256_duplicate_si128(get_swizzle_mask<4, T>());
  __m256i s[4];
  s[0]        = x_mm256_shuffle_epi8(vs[0], mask);
  s[1]        = x_mm256_shuffle_epi8(vs[1], mask);
  s[2]        = x_mm256_shuffle_epi8(vs[2], mask);
  s[3]        = x_mm256_shuffle_epi8(vs[3], mask);
  __m256i l01 = x_mm256_unpacklo_epi32(s[0], s[1]);
  __m256i h01 = x_mm256_unpackhi_epi32(s[0], s[1]);
  __m256i l23 = x_mm256_unpacklo_epi32(s[2], s[3]);
  __m256i h23 = x_mm256_unpackhi_epi32(s[2], s[3]);
  v[0]        = x_mm256_unpacklo_epi64(l01, l23);
  v[1]        = x_mm256_unpackhi_epi64(l01, l23);
  v[2]        = x_mm256_unpacklo_epi64(h01, h23);
  v[3]        = x_mm256_unpackhi_epi64(h01, h23);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[4], Integer<4>)
{
  Vec<Float, 32> vInt[4];
  for (int i = 0; i < 4; ++i) {
    vInt[i] = reinterpret(v[i], OutputType<Float>());
  }
  Vec<Float, 32> vs[4];
  swizzle_32_16<4>(vInt, vs);
  __m256 s[4];
  s[0] = _mm256_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(1, 0, 1, 0));
  s[1] = _mm256_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(3, 2, 3, 2));
  s[2] = _mm256_shuffle_ps(vs[2], vs[3], _MM_SHUFFLE(1, 0, 1, 0));
  s[3] = _mm256_shuffle_ps(vs[2], vs[3], _MM_SHUFFLE(3, 2, 3, 2));
  Vec<Float, 32> vOut[4];
  vOut[0] = _mm256_shuffle_ps(s[0], s[2], _MM_SHUFFLE(2, 0, 2, 0));
  vOut[1] = _mm256_shuffle_ps(s[0], s[2], _MM_SHUFFLE(3, 1, 3, 1));
  vOut[2] = _mm256_shuffle_ps(s[1], s[3], _MM_SHUFFLE(2, 0, 2, 0));
  vOut[3] = _mm256_shuffle_ps(s[1], s[3], _MM_SHUFFLE(3, 1, 3, 1));
  for (int i = 0; i < 4; ++i) { v[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[4], Integer<4>)
{
  const Vec<Double, 32> vInt[4] = {
    reinterpret(v[0], OutputType<Double>()), // x0y0z0w0
    reinterpret(v[1], OutputType<Double>()), // x1y1z1w1
    reinterpret(v[2], OutputType<Double>()), // x2y2z2w2
    reinterpret(v[3], OutputType<Double>()), // x3y3z3w3
  };
  Vec<Double, 32> vs[4];
  swizzle_32_16<4>(vInt, vs);
  // vs[0] = x0y0x2y2
  // vs[1] = z0w0z2w2
  // vs[2] = x1y1x3y3
  // vs[3] = z1w1z3w3
  const Vec<Double, 32> vOut[4] = {
    // x0x1x2x3
    _mm256_shuffle_pd(vs[0], vs[2], 0x0), // 0b0000
    // y0y1y2y3
    _mm256_shuffle_pd(vs[0], vs[2], 0xF), // 0b1111
    // z0z1z2z3
    _mm256_shuffle_pd(vs[1], vs[3], 0x0), // 0b0000
    // w0w1w2w3
    _mm256_shuffle_pd(vs[1], vs[3], 0xF), // 0b1111
  };
  for (int i = 0; i < 4; ++i) { v[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// -------------------- n = 5 --------------------

// 8 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) == 1 && std::is_integral<T>::value)>
static SIMD_INLINE void swizzle(Vec<T, 32> v[5], Integer<5>)
{
  Vec<T, 32> vs[5];
  swizzle_32_16<5>(v, vs);
  __m256i mask    = x_mm256_duplicate_si128(get_swizzle_mask<5, T>());
  __m256i s0      = align_shuffle_256<0>(vs[0], vs[1], mask);
  __m256i s1      = align_shuffle_256<10>(vs[0], vs[1], mask);
  __m256i s2      = align_shuffle_256<4>(vs[1], vs[2], mask);
  __m256i s3      = align_shuffle_256<14>(vs[1], vs[2], mask);
  __m256i s4      = align_shuffle_256<8>(vs[2], vs[3], mask);
  __m256i s5      = align_shuffle_256<2>(vs[3], vs[4], mask);
  __m256i s6      = align_shuffle_256<12>(vs[3], vs[4], mask);
  __m256i s7      = align_shuffle_256<6>(vs[4], _mm256_undefined_si256(), mask);
  __m256i l01     = x_mm256_unpacklo_epi16(s0, s1);
  __m256i h01     = x_mm256_unpackhi_epi16(s0, s1);
  __m256i l23     = x_mm256_unpacklo_epi16(s2, s3);
  __m256i h23     = x_mm256_unpackhi_epi16(s2, s3);
  __m256i l45     = x_mm256_unpacklo_epi16(s4, s5);
  __m256i h45     = x_mm256_unpackhi_epi16(s4, s5);
  __m256i l67     = x_mm256_unpacklo_epi16(s6, s7);
  __m256i h67     = x_mm256_unpackhi_epi16(s6, s7);
  __m256i ll01l23 = x_mm256_unpacklo_epi32(l01, l23);
  __m256i hl01l23 = x_mm256_unpackhi_epi32(l01, l23);
  __m256i ll45l67 = x_mm256_unpacklo_epi32(l45, l67);
  __m256i hl45l67 = x_mm256_unpackhi_epi32(l45, l67);
  __m256i lh01h23 = x_mm256_unpacklo_epi32(h01, h23);
  __m256i lh45h67 = x_mm256_unpacklo_epi32(h45, h67);
  v[0]            = x_mm256_unpacklo_epi64(ll01l23, ll45l67);
  v[1]            = x_mm256_unpackhi_epi64(ll01l23, ll45l67);
  v[2]            = x_mm256_unpacklo_epi64(hl01l23, hl45l67);
  v[3]            = x_mm256_unpackhi_epi64(hl01l23, hl45l67);
  v[4]            = x_mm256_unpacklo_epi64(lh01h23, lh45h67);
}

// 16 bit integer types
template <typename T,
          SIMD_ENABLE_IF(sizeof(T) == 2 && std::is_integral<T>::value),
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[5], Integer<5>)
{
  Vec<T, 32> vs[5];
  swizzle_32_16<5>(v, vs);
  __m256i mask = x_mm256_duplicate_si128(get_swizzle_mask<5, T>());
  __m256i s0   = align_shuffle_256<0>(vs[0], vs[1], mask);
  __m256i s1   = align_shuffle_256<6>(vs[0], vs[1], mask);
  __m256i s2   = align_shuffle_256<4>(vs[1], vs[2], mask);
  __m256i s3   = align_shuffle_256<10>(vs[1], vs[2], mask);
  __m256i s4   = align_shuffle_256<8>(vs[2], vs[3], mask);
  __m256i s5   = align_shuffle_256<14>(vs[2], vs[3], mask);
  __m256i s6   = align_shuffle_256<12>(vs[3], vs[4], mask);
  __m256i s7   = align_shuffle_256<2>(vs[4], _mm256_undefined_si256(), mask);
  __m256i l02  = x_mm256_unpacklo_epi32(s0, s2);
  __m256i h02  = x_mm256_unpackhi_epi32(s0, s2);
  __m256i l13  = x_mm256_unpacklo_epi32(s1, s3);
  __m256i l46  = x_mm256_unpacklo_epi32(s4, s6);
  __m256i h46  = x_mm256_unpackhi_epi32(s4, s6);
  __m256i l57  = x_mm256_unpacklo_epi32(s5, s7);
  v[0]         = x_mm256_unpacklo_epi64(l02, l46);
  v[1]         = x_mm256_unpackhi_epi64(l02, l46);
  v[2]         = x_mm256_unpacklo_epi64(h02, h46);
  v[3]         = x_mm256_unpacklo_epi64(l13, l57);
  v[4]         = x_mm256_unpackhi_epi64(l13, l57);
}

// 32 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4), typename = void,
          typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[5], Integer<5>)
{
  Vec<Int, 32> vInt[5];
  for (int i = 0; i < 5; i++) {
    vInt[i] = reinterpret(v[i], OutputType<Int>());
  }
  Vec<Int, 32> vs[5];
  swizzle_32_16<5>(vInt, vs);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  // v[0]: 0 1 2 3
  // v[1]: 4 x x x
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                   x x x   x
  // 5 6 7 8
  __m256i s2 = x_mm256_alignr_epi8<4>(vs[2], vs[1]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                             x  x  x    x
  // 9 x x x
  __m256i s3 = x_mm256_alignr_epi8<4>(vs[3], vs[2]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                x  x    x  x
  // 10 11 12 13
  __m256i s4 = x_mm256_alignr_epi8<8>(vs[3], vs[2]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                              x  x    x  x
  // 14 x x x
  __m256i s5 = x_mm256_alignr_epi8<8>(vs[4], vs[3]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                                 X    X  X  X
  // 15 16 17 18
  __m256i s6 = x_mm256_alignr_epi8<12>(vs[4], vs[3]);
  // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
  //                                                               X X X X
  // 19 x x x
  __m256i s7 = x_mm256_alignr_epi8<12>(vs[0], vs[4]);
  // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
  __m256i l02 = x_mm256_unpacklo_epi32(vs[0], s2);
  __m256i h02 = x_mm256_unpackhi_epi32(vs[0], s2);
  // 4 x x x / 9 x x x -> 4 9 x x
  __m256i l13 = x_mm256_unpacklo_epi32(vs[1], s3);
  // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
  __m256i l46 = x_mm256_unpacklo_epi32(s4, s6);
  __m256i h46 = x_mm256_unpackhi_epi32(s4, s6);
  // 14 x x x / 19 x x x -> 14 19 x x
  __m256i l57                = x_mm256_unpacklo_epi32(s5, s7);
  const Vec<Int, 32> vOut[5] = {
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    x_mm256_unpacklo_epi64(l02, l46),
    x_mm256_unpackhi_epi64(l02, l46),
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    x_mm256_unpacklo_epi64(h02, h46),
    x_mm256_unpackhi_epi64(h02, h46),
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    x_mm256_unpacklo_epi64(l13, l57),
  };
  for (int i = 0; i < 5; i++) { v[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// 64 bit types
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void,
          typename = void, typename = void>
static SIMD_INLINE void swizzle(Vec<T, 32> v[5], Integer<5>)
{
  const Vec<Double, 32> vDouble[5] = {
    reinterpret(v[0], OutputType<Double>()), // x0y0z0w0
    reinterpret(v[1], OutputType<Double>()), // v0x1y1z1
    reinterpret(v[2], OutputType<Double>()), // w1v1x2y2
    reinterpret(v[3], OutputType<Double>()), // z2w2v2x3
    reinterpret(v[4], OutputType<Double>()), // y3z3w3v3
  };
  Vec<Double, 32> vs[5];
  swizzle_32_16<5>(vDouble, vs);
  // vs[0] = x0y0x2y2
  // vs[1] = z0w0z2w2
  // vs[2] = v0x1v2x3
  // vs[3] = y1z1y3z3
  // vs[4] = w1v1w3v3
  const Vec<Double, 32> vOut[5] = {
    // x0x1x2x3
    _mm256_shuffle_pd(vs[0], vs[2], 0xa), // 0b1010
    // y0y1y2y3
    _mm256_shuffle_pd(vs[0], vs[3], 0x5), // 0b0101
    // z0z1z2z3
    _mm256_shuffle_pd(vs[1], vs[3], 0xa), // 0b1010
    // w0w1w2w3
    _mm256_shuffle_pd(vs[1], vs[4], 0x5), // 0b0101
    // v0v1v2v3
    _mm256_shuffle_pd(vs[2], vs[4], 0xa), // 0b1010
  };
  for (int i = 0; i < 5; i++) { v[i] = reinterpret(vOut[i], OutputType<T>()); }
}

// ---------------------------------------------------------------------------
// comparison functions
// ---------------------------------------------------------------------------

// 28. Mar 23 (Jonas Keller): checked the constants for _mm256_cmp_ps in the
// Float comparison functions, they match the implementation of the SSE versions
// (see cmpps in Intel manual) and added corresponding comments

// ---------------------------------------------------------------------------
// compare <
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/
//   sse-compare-packed-unsigned-bytes

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> cmplt(const Vec<Byte, 32> &a,
                                       const Vec<Byte, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80808080);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x80
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x80
  return _mm256_cmpgt_epi8(b1, a1);
}

static SIMD_INLINE Vec<SignedByte, 32> cmplt(const Vec<SignedByte, 32> &a,
                                             const Vec<SignedByte, 32> &b)
{
  return _mm256_cmpgt_epi8(b, a);
}

static SIMD_INLINE Vec<Word, 32> cmplt(const Vec<Word, 32> &a,
                                       const Vec<Word, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80008000);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x8000
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x8000
  return _mm256_cmpgt_epi16(b1, a1);
}

static SIMD_INLINE Vec<Short, 32> cmplt(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return _mm256_cmpgt_epi16(b, a);
}

static SIMD_INLINE Vec<Int, 32> cmplt(const Vec<Int, 32> &a,
                                      const Vec<Int, 32> &b)
{
  return _mm256_cmpgt_epi32(b, a);
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> cmplt(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(cmplt(a.lo(), b.lo()), cmplt(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> cmplt(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  // same constant as in implementation of _mm_cmplt_ps (see cmpps instruction
  // in Intel manual)
  return _mm256_cmp_ps(a, b, _CMP_LT_OS);
}

static SIMD_INLINE Vec<Double, 32> cmplt(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return _mm256_cmp_pd(a, b, _CMP_LT_OS);
}

// ---------------------------------------------------------------------------
// compare <=
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/
//   sse-compare-packed-unsigned-bytes

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> cmple(const Vec<Byte, 32> &a,
                                       const Vec<Byte, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80808080);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x80
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x80
  return _mm256_or_si256(_mm256_cmpgt_epi8(b1, a1), _mm256_cmpeq_epi8(a1, b1));
}

static SIMD_INLINE Vec<SignedByte, 32> cmple(const Vec<SignedByte, 32> &a,
                                             const Vec<SignedByte, 32> &b)
{
  return _mm256_or_si256(_mm256_cmpgt_epi8(b, a), _mm256_cmpeq_epi8(a, b));
}

static SIMD_INLINE Vec<Word, 32> cmple(const Vec<Word, 32> &a,
                                       const Vec<Word, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80008000);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x8000
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x8000
  return _mm256_or_si256(_mm256_cmpgt_epi16(b1, a1),
                         _mm256_cmpeq_epi16(a1, b1));
}

static SIMD_INLINE Vec<Short, 32> cmple(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return _mm256_or_si256(_mm256_cmpgt_epi16(b, a), _mm256_cmpeq_epi16(a, b));
}

static SIMD_INLINE Vec<Int, 32> cmple(const Vec<Int, 32> &a,
                                      const Vec<Int, 32> &b)
{
  return _mm256_or_si256(_mm256_cmpgt_epi32(b, a), _mm256_cmpeq_epi32(a, b));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> cmple(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(cmple(a.lo(), b.lo()), cmple(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> cmple(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  // same constant as in implementation of _mm_cmple_ps (see cmpps instruction
  // in Intel manual)
  return _mm256_cmp_ps(a, b, _CMP_LE_OS);
}

static SIMD_INLINE Vec<Double, 32> cmple(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return _mm256_cmp_pd(a, b, _CMP_LE_OS);
}

// ---------------------------------------------------------------------------
// compare ==
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> cmpeq(const Vec<Byte, 32> &a,
                                       const Vec<Byte, 32> &b)
{
  return _mm256_cmpeq_epi8(a, b);
}

static SIMD_INLINE Vec<SignedByte, 32> cmpeq(const Vec<SignedByte, 32> &a,
                                             const Vec<SignedByte, 32> &b)
{
  return _mm256_cmpeq_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> cmpeq(const Vec<Word, 32> &a,
                                       const Vec<Word, 32> &b)
{
  return _mm256_cmpeq_epi16(a, b);
}

static SIMD_INLINE Vec<Short, 32> cmpeq(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return _mm256_cmpeq_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> cmpeq(const Vec<Int, 32> &a,
                                      const Vec<Int, 32> &b)
{
  return _mm256_cmpeq_epi32(a, b);
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> cmpeq(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(cmpeq(a.lo(), b.lo()), cmpeq(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> cmpeq(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  // same constant as in implementation of _mm_cmpeq_ps (see cmpps instruction
  // in Intel manual)
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}

static SIMD_INLINE Vec<Double, 32> cmpeq(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);
}

// ---------------------------------------------------------------------------
// compare >
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/
//   sse-compare-packed-unsigned-bytes

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> cmpgt(const Vec<Byte, 32> &a,
                                       const Vec<Byte, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80808080);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x80
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x80
  return _mm256_cmpgt_epi8(a1, b1);
}

static SIMD_INLINE Vec<SignedByte, 32> cmpgt(const Vec<SignedByte, 32> &a,
                                             const Vec<SignedByte, 32> &b)
{
  return _mm256_cmpgt_epi8(a, b);
}

static SIMD_INLINE Vec<Word, 32> cmpgt(const Vec<Word, 32> &a,
                                       const Vec<Word, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80008000);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x8000
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x8000
  return _mm256_cmpgt_epi16(a1, b1);
}

static SIMD_INLINE Vec<Short, 32> cmpgt(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return _mm256_cmpgt_epi16(a, b);
}

static SIMD_INLINE Vec<Int, 32> cmpgt(const Vec<Int, 32> &a,
                                      const Vec<Int, 32> &b)
{
  return _mm256_cmpgt_epi32(a, b);
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> cmpgt(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(cmpgt(a.lo(), b.lo()), cmpgt(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> cmpgt(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  // same constant as in implementation of _mm_cmplt_ps (see cmpps instruction
  // in Intel manual)
  return _mm256_cmp_ps(b, a, _CMP_LT_OS);
}

static SIMD_INLINE Vec<Double, 32> cmpgt(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return _mm256_cmp_pd(b, a, _CMP_LT_OS);
}

// ---------------------------------------------------------------------------
// compare >=
// ---------------------------------------------------------------------------

// http://stackoverflow.com/questions/16204663/
//  sse-compare-packed-unsigned-bytes

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> cmpge(const Vec<Byte, 32> &a,
                                       const Vec<Byte, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80808080);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x80
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x80
  return _mm256_or_si256(_mm256_cmpgt_epi8(a1, b1), _mm256_cmpeq_epi8(a1, b1));
}

static SIMD_INLINE Vec<SignedByte, 32> cmpge(const Vec<SignedByte, 32> &a,
                                             const Vec<SignedByte, 32> &b)
{
  return _mm256_or_si256(_mm256_cmpgt_epi8(a, b), _mm256_cmpeq_epi8(a, b));
}

static SIMD_INLINE Vec<Word, 32> cmpge(const Vec<Word, 32> &a,
                                       const Vec<Word, 32> &b)
{
  __m256i signbit = _mm256_set1_epi32(0x80008000);
  __m256i a1      = _mm256_xor_si256(a, signbit); // sub 0x8000
  __m256i b1      = _mm256_xor_si256(b, signbit); // sub 0x8000
  return _mm256_or_si256(_mm256_cmpgt_epi16(a1, b1),
                         _mm256_cmpeq_epi16(a1, b1));
}

static SIMD_INLINE Vec<Short, 32> cmpge(const Vec<Short, 32> &a,
                                        const Vec<Short, 32> &b)
{
  return _mm256_or_si256(_mm256_cmpgt_epi16(a, b), _mm256_cmpeq_epi16(a, b));
}

static SIMD_INLINE Vec<Int, 32> cmpge(const Vec<Int, 32> &a,
                                      const Vec<Int, 32> &b)
{
  return _mm256_or_si256(_mm256_cmpgt_epi32(a, b), _mm256_cmpeq_epi32(a, b));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> cmpge(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(cmpge(a.lo(), b.lo()), cmpge(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> cmpge(const Vec<Float, 32> &a,
                                        const Vec<Float, 32> &b)
{
  // same constant as in implementation of _mm_cmple_ps (see cmpps instruction
  // in Intel manual)
  return _mm256_cmp_ps(b, a, _CMP_LE_OS);
}

static SIMD_INLINE Vec<Double, 32> cmpge(const Vec<Double, 32> &a,
                                         const Vec<Double, 32> &b)
{
  return _mm256_cmp_pd(b, a, _CMP_LE_OS);
}

// ---------------------------------------------------------------------------
// compare !=
// ---------------------------------------------------------------------------

#ifdef __AVX2__

// there is no cmpneq for integers and no not, so use cmpeq and xor with all
// ones to invert the result

static SIMD_INLINE Vec<Byte, 32> cmpneq(const Vec<Byte, 32> &a,
                                        const Vec<Byte, 32> &b)
{
  return _mm256_xor_si256(_mm256_cmpeq_epi8(a, b), _mm256_set1_epi32(-1));
}

static SIMD_INLINE Vec<SignedByte, 32> cmpneq(const Vec<SignedByte, 32> &a,
                                              const Vec<SignedByte, 32> &b)
{
  return _mm256_xor_si256(_mm256_cmpeq_epi8(a, b), _mm256_set1_epi32(-1));
}

static SIMD_INLINE Vec<Word, 32> cmpneq(const Vec<Word, 32> &a,
                                        const Vec<Word, 32> &b)
{
  return _mm256_xor_si256(_mm256_cmpeq_epi16(a, b), _mm256_set1_epi32(-1));
}

static SIMD_INLINE Vec<Short, 32> cmpneq(const Vec<Short, 32> &a,
                                         const Vec<Short, 32> &b)
{
  return _mm256_xor_si256(_mm256_cmpeq_epi16(a, b), _mm256_set1_epi32(-1));
}

static SIMD_INLINE Vec<Int, 32> cmpneq(const Vec<Int, 32> &a,
                                       const Vec<Int, 32> &b)
{
  return _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_set1_epi32(-1));
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> cmpneq(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(cmpneq(a.lo(), b.lo()), cmpneq(a.hi(), b.hi()));
}

#endif

static SIMD_INLINE Vec<Float, 32> cmpneq(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b)
{
  // same constant as in implementation of _mm_cmpneq_ps (see cmpps instruction
  // in Intel manual)
  return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ);
}

static SIMD_INLINE Vec<Double, 32> cmpneq(const Vec<Double, 32> &a,
                                          const Vec<Double, 32> &b)
{
  return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ);
}

// ---------------------------------------------------------------------------
// ifelse
// ---------------------------------------------------------------------------

// 10. Apr 23 (Jonas Keller): made two versions of ifelse, one for 8 and 16 bit
// data types, and one for 32 and larger data types, so that for the latter
// the blendv instruction can be used even if avx2 is not available

// version for 8 and 16 bit data types
template <typename T, SIMD_ENABLE_IF(sizeof(T) <= 2)>
static SIMD_INLINE Vec<T, 32> ifelse(const Vec<T, 32> &cond,
                                     const Vec<T, 32> &trueVal,
                                     const Vec<T, 32> &falseVal)
{
#ifdef __AVX2__
  const Vec<Byte, 32> res =
    _mm256_blendv_epi8(reinterpret(falseVal, OutputType<Byte>()),
                       reinterpret(trueVal, OutputType<Byte>()),
                       reinterpret(cond, OutputType<Byte>()));
#else
  // non-avx2 workaround
  const Vec<Float, 32> res =
    _mm256_or_ps(_mm256_and_ps(reinterpret(cond, OutputType<Float>()),
                               reinterpret(trueVal, OutputType<Float>())),
                 _mm256_andnot_ps(reinterpret(cond, OutputType<Float>()),
                                  reinterpret(falseVal, OutputType<Float>())));
#endif
  return reinterpret(res, OutputType<T>());
}

// version for 32 bit or larger data types
template <typename T, SIMD_ENABLE_IF(sizeof(T) > 2), typename = void>
static SIMD_INLINE Vec<T, 32> ifelse(const Vec<T, 32> &cond,
                                     const Vec<T, 32> &trueVal,
                                     const Vec<T, 32> &falseVal)
{
  const Vec<Float, 32> res =
    _mm256_blendv_ps(reinterpret(falseVal, OutputType<Float>()),
                     reinterpret(trueVal, OutputType<Float>()),
                     reinterpret(cond, OutputType<Float>()));
  return reinterpret(res, OutputType<T>());
}

// ---------------------------------------------------------------------------
// bit_and
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> bit_and(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
#ifdef __AVX2__
  return _mm256_and_si256(a, b);
#else
  // non-avx2 workaround
  return _mm256_castps_si256(
    _mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// float version
static SIMD_INLINE Vec<Float, 32> bit_and(const Vec<Float, 32> &a,
                                          const Vec<Float, 32> &b)
{
  return _mm256_and_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 32> bit_and(const Vec<Double, 32> &a,
                                           const Vec<Double, 32> &b)
{
  return _mm256_and_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_or
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> bit_or(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
#ifdef __AVX2__
  return _mm256_or_si256(a, b);
#else
  // non-avx2 workaround
  return _mm256_castps_si256(
    _mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// float version
static SIMD_INLINE Vec<Float, 32> bit_or(const Vec<Float, 32> &a,
                                         const Vec<Float, 32> &b)
{
  return _mm256_or_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 32> bit_or(const Vec<Double, 32> &a,
                                          const Vec<Double, 32> &b)
{
  return _mm256_or_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_andnot
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> bit_andnot(const Vec<T, 32> &a,
                                         const Vec<T, 32> &b)
{
#ifdef __AVX2__
  return _mm256_andnot_si256(a, b);
#else
  // non-avx2 workaround
  return _mm256_castps_si256(
    _mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// float version
static SIMD_INLINE Vec<Float, 32> bit_andnot(const Vec<Float, 32> &a,
                                             const Vec<Float, 32> &b)
{
  return _mm256_andnot_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 32> bit_andnot(const Vec<Double, 32> &a,
                                              const Vec<Double, 32> &b)
{
  return _mm256_andnot_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_xor
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> bit_xor(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
#ifdef __AVX2__
  return _mm256_xor_si256(a, b);
#else
  // non-avx2 workaround
  return _mm256_castps_si256(
    _mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// float version
static SIMD_INLINE Vec<Float, 32> bit_xor(const Vec<Float, 32> &a,
                                          const Vec<Float, 32> &b)
{
  return _mm256_xor_ps(a, b);
}

// double version
static SIMD_INLINE Vec<Double, 32> bit_xor(const Vec<Double, 32> &a,
                                           const Vec<Double, 32> &b)
{
  return _mm256_xor_pd(a, b);
}

// ---------------------------------------------------------------------------
// bit_not
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE Vec<T, 32> bit_not(const Vec<T, 32> &a)
{
#ifdef __AVX2__
  return _mm256_xor_si256(a, _mm256_set1_epi32(-1));
#else
  // non-avx2 workaround
  return _mm256_castps_si256(_mm256_xor_ps(
    _mm256_castsi256_ps(a), _mm256_castsi256_ps(_mm256_set1_epi32(-1))));
#endif
}

// float version
static SIMD_INLINE Vec<Float, 32> bit_not(const Vec<Float, 32> &a)
{
  return _mm256_xor_ps(a, _mm256_castsi256_ps(_mm256_set1_epi32(-1)));
}

// double version
static SIMD_INLINE Vec<Double, 32> bit_not(const Vec<Double, 32> &a)
{
  return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_set1_epi32(-1)));
}

// ---------------------------------------------------------------------------
// avg: average with rounding down
// ---------------------------------------------------------------------------

#ifdef __AVX2__

static SIMD_INLINE Vec<Byte, 32> avg(const Vec<Byte, 32> &a,
                                     const Vec<Byte, 32> &b)
{
  return _mm256_avg_epu8(a, b);
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<SignedByte, 32> avg(const Vec<SignedByte, 32> &a,
                                           const Vec<SignedByte, 32> &b)
{
  // from Agner Fog's VCL vectori128.h
  __m256i signbit = _mm256_set1_epi32(0x80808080);
  __m256i a1      = _mm256_xor_si256(a, signbit); // add 0x80
  __m256i b1      = _mm256_xor_si256(b, signbit); // add 0x80
  __m256i m1      = _mm256_avg_epu8(a1, b1);      // unsigned avg
  return _mm256_xor_si256(m1, signbit);           // sub 0x80
}

static SIMD_INLINE Vec<Word, 32> avg(const Vec<Word, 32> &a,
                                     const Vec<Word, 32> &b)
{
  return _mm256_avg_epu16(a, b);
}

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<Short, 32> avg(const Vec<Short, 32> &a,
                                      const Vec<Short, 32> &b)
{
  // from Agner Fog's VCL vectori128.h
  __m256i signbit = _mm256_set1_epi32(0x80008000);
  __m256i a1      = _mm256_xor_si256(a, signbit); // add 0x8000
  __m256i b1      = _mm256_xor_si256(b, signbit); // add 0x8000
  __m256i m1      = _mm256_avg_epu16(a1, b1);     // unsigned avg
  return _mm256_xor_si256(m1, signbit);           // sub 0x8000
}

#else

// non-avx2 workaround
template <typename T>
static SIMD_INLINE Vec<T, 32> avg(const Vec<T, 32> &a, const Vec<T, 32> &b)
{
  return Vec<T, 32>(avg(a.lo(), b.lo()), avg(a.hi(), b.hi()));
}

#endif

// Paul R at
// http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
static SIMD_INLINE Vec<Int, 32> avg(const Vec<Int, 32> &a,
                                    const Vec<Int, 32> &b)
{
  Vec<Int, 32> one = set1(Int(1), Integer<32>()), as, bs, lsb;
  lsb              = bit_and(bit_or(a, b), one);
  as               = srai<1>(a);
  bs               = srai<1>(b);
  return add(lsb, add(as, bs));
}

// NOTE: Float version doesn't round!
static SIMD_INLINE Vec<Float, 32> avg(const Vec<Float, 32> &a,
                                      const Vec<Float, 32> &b)
{
  __m256 half = _mm256_set1_ps(0.5f);
  return _mm256_mul_ps(_mm256_add_ps(a, b), half);
}

// NOTE: Double version doesn't round!
static SIMD_INLINE Vec<Double, 32> avg(const Vec<Double, 32> &a,
                                       const Vec<Double, 32> &b)
{
  __m256d half = _mm256_set1_pd(0.5);
  return _mm256_mul_pd(_mm256_add_pd(a, b), half);
}

// ---------------------------------------------------------------------------
// test_all_zeros
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE bool test_all_zeros(const Vec<T, 32> &a)
{
  // 2nd argument: generates vector full of ones (epiX: X irrelevant)
  // 10. Oct 22 (Jonas Keller):
  // replaced unnecessary "_mm256_cmpeq_epi8(a, a)" with "a"
  // return _mm256_testz_si256(a, _mm256_cmpeq_epi8(a, a));
  return _mm256_testz_si256(a, a);
  // 13. Nov. 22 (Jonas Keller):
  // removed workaround for when AVX2 is not available, since it is not
  // needed anymore now that _mm256_cmpeq_epi8 is not used above anymore
}

// float version
// NOTE: _mm256_testz_ps only tests the sign bits!
// 28. Sep 17 (rm) BUGFIX: using _mm256_testz_si256 instead
static SIMD_INLINE bool test_all_zeros(const Vec<Float, 32> &a)
{
  // _CMP_EQ_UQ! (11..11 is NaN, not treated differently for Unordered)
  // 10. Oct 22 (Jonas Keller):
  // replaced unnecessary "_mm256_cmp_ps(a, a, _CMP_EQ_UQ)" with "a"
  // return _mm256_testz_si256
  //   (_mm256_castps_si256(a),
  //    _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_EQ_UQ)));
  return _mm256_testz_si256(_mm256_castps_si256(a), _mm256_castps_si256(a));
}

// double version
static SIMD_INLINE int test_all_zeros(const Vec<Double, 32> &a)
{
  return _mm256_testz_si256(_mm256_castpd_si256(a), _mm256_castpd_si256(a));
}

// ---------------------------------------------------------------------------
// test_all_ones
// ---------------------------------------------------------------------------

// all integer versions
template <typename T>
static SIMD_INLINE bool test_all_ones(const Vec<T, 32> &a)
{
  // 2nd argument: generates vector full of ones (epiX: X irrelevant)
#ifdef __AVX2__
  return _mm256_testc_si256(a, _mm256_cmpeq_epi8(a, a));
#else
  // _CMP_EQ_UQ! (11..11 is NaN, not treated differently for Unordered)
  return _mm256_testc_si256(
    a, _mm256_castps_si256(_mm256_cmp_ps(_mm256_castsi256_ps(a),
                                         _mm256_castsi256_ps(a), _CMP_EQ_UQ)));
#endif
}

// float version
// NOTE: _mm256_testc_ps only tests the sign bits!
// 28. Sep 17 (rm) BUGFIX: using _mm256_testc_si256 instead
static SIMD_INLINE bool test_all_ones(const Vec<Float, 32> &a)
{
  // _CMP_EQ_UQ! (11..11 is NaN, not treated differently for Unordered)
  return _mm256_testc_si256(
    _mm256_castps_si256(a),
    _mm256_castps_si256(_mm256_cmp_ps(a, a, _CMP_EQ_UQ)));
}

// double version
static SIMD_INLINE int test_all_ones(const Vec<Double, 32> &a)
{
  // _CMP_EQ_UQ! (11..11 is NaN, not treated differently for Unordered)
  return _mm256_testc_si256(
    _mm256_castpd_si256(a),
    _mm256_castpd_si256(_mm256_cmp_pd(a, a, _CMP_EQ_UQ)));
}

// ---------------------------------------------------------------------------
// reverse
// ---------------------------------------------------------------------------

// All reverse operations below are courtesy of Yannick Sander
// modified

static SIMD_INLINE Vec<Byte, 32> reverse(const Vec<Byte, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask =
    _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1,
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  /**
   * _mm256_shuffle_epi8 reverses the upper and lower lane of a individually
   * the to lanes have to be swapped as well to perform a full reverse
   */
  __m256i shuffled_lanes = _mm256_shuffle_epi8(a, mask);
  __m256i swapped_lanes =
    _mm256_permute4x64_epi64(shuffled_lanes, _MM_SHUFFLE(1, 0, 3, 2));

  return swapped_lanes;

#else // AVX fallback
  const Vec<Byte, 16> h = _mm256_extractf128_si256(a, 0);
  const Vec<Byte, 16> l = _mm256_extractf128_si256(a, 1);
  return _mm256_set_m128i(reverse(h), reverse(l));
#endif
}

static SIMD_INLINE Vec<SignedByte, 32> reverse(const Vec<SignedByte, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask =
    _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1,
                    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

  /**
   * _mm256_shuffle_epi8 reverses the upper and lower lane of a individually
   * the to lanes have to be swapped as well to perform a full reverse
   */
  __m256i shuffled_lanes = _mm256_shuffle_epi8(a, mask);
  __m256i swapped_lanes =
    _mm256_permute4x64_epi64(shuffled_lanes, _MM_SHUFFLE(1, 0, 3, 2));

  return swapped_lanes;

#else // AVX fallback
  const Vec<SignedByte, 16> h = _mm256_extractf128_si256(a, 0);
  const Vec<SignedByte, 16> l = _mm256_extractf128_si256(a, 1);
  return _mm256_set_m128i(reverse(h), reverse(l));
#endif
}

static SIMD_INLINE Vec<Short, 32> reverse(const Vec<Short, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask =
    _mm256_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);

  __m256i shuffled_lanes = _mm256_shuffle_epi8(a, mask);
  __m256i swapped_lanes =
    _mm256_permute4x64_epi64(shuffled_lanes, _MM_SHUFFLE(1, 0, 3, 2));

  return swapped_lanes;
#else // AVX fallback
  const Vec<Short, 16> h = _mm256_extractf128_si256(a, 0);
  const Vec<Short, 16> l = _mm256_extractf128_si256(a, 1);
  return _mm256_set_m128i(reverse(h), reverse(l));
#endif
}

static SIMD_INLINE Vec<Word, 32> reverse(const Vec<Word, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask =
    _mm256_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17,
                    16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30);

  __m256i shuffled_lanes = _mm256_shuffle_epi8(a, mask);
  __m256i swapped_lanes =
    _mm256_permute4x64_epi64(shuffled_lanes, _MM_SHUFFLE(1, 0, 3, 2));

  return swapped_lanes;
#else // AVX fallback
  const Vec<Word, 16> h = _mm256_extractf128_si256(a, 0);
  const Vec<Word, 16> l = _mm256_extractf128_si256(a, 1);
  return _mm256_set_m128i(reverse(h), reverse(l));
#endif
}

static SIMD_INLINE Vec<Int, 32> reverse(const Vec<Int, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_permutevar8x32_epi32(a, mask);
#else // AVX fallback
  const Vec<Int, 16> h = _mm256_extractf128_si256(a, 0);
  const Vec<Int, 16> l = _mm256_extractf128_si256(a, 1);
  return _mm256_set_m128i(reverse(h), reverse(l));
#endif
}

static SIMD_INLINE Vec<Float, 32> reverse(const Vec<Float, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_permutevar8x32_ps(a, mask);
#else // AVX fallback
  const Vec<Float, 16> h = _mm256_extractf128_ps(a, 0);
  const Vec<Float, 16> l = _mm256_extractf128_ps(a, 1);
  return _mm256_set_m128(reverse(h), reverse(l));
#endif
}

static SIMD_INLINE Vec<Double, 32> reverse(const Vec<Double, 32> &a)
{
#ifdef __AVX2__
  const __m256i mask = _mm256_set_epi32(1, 0, 3, 2, 5, 4, 7, 6);
  return _mm256_castps_pd(_mm256_permutevar8x32_ps(_mm256_castpd_ps(a), mask));
#else // AVX fallback
  const Vec<Double, 16> h = _mm256_extractf128_pd(a, 0);
  const Vec<Double, 16> l = _mm256_extractf128_pd(a, 1);
  return _mm256_set_m128d(reverse(h), reverse(l));
#endif
}

// ---------------------------------------------------------------------------
// msb2int
// ---------------------------------------------------------------------------

// 26. Aug 22 (Jonas Keller): added msb2int functions

// 16. Aug 23 (Jonas Keller): fixed bug in msb2int for Byte and SignedByte
// caused by trying to cast an int to uint64_t which internally first casts to
// int64_t and then to uint64_t, which causes sign extension

template <typename T,
          SIMD_ENABLE_IF(std::is_integral<T>::value && sizeof(T) == 1)>
static SIMD_INLINE uint64_t msb2int(const Vec<T, 32> &a)
{
  // first convert to uint32_t to avoid sign extension
#ifdef __AVX2__
  const auto res = _mm256_movemask_epi8(a);
#else
  const auto res =
    _mm_movemask_epi8(a.lo()) | (_mm_movemask_epi8(a.hi()) << 16);
#endif
  // prevent sign extension when casting to uint64_t by first casting to uint
  return uint64_t(uint(res));
}

template <typename T,
          SIMD_ENABLE_IF(std::is_integral<T>::value && sizeof(T) == 2),
          typename = void>
static SIMD_INLINE uint64_t msb2int(const Vec<T, 32> &a)
{
#ifdef __AVX2__
  return uint64_t(_mm256_movemask_epi8(a));
#else
  return uint64_t(_mm_movemask_epi8(a.lo()) |
                  (_mm_movemask_epi8(a.hi()) << 16));
#endif
}

static SIMD_INLINE uint64_t msb2int(const Vec<Short, 32> &a)
{
  // there is no _mm256_movemask_epi16, so use _mm256_movemask_epi8
  // and discard the even bits
  // discarding the even bytes in a with a shuffle does not work,
  // since shuffle shuffles within 128 bit lanes
  // TODO: better way to do this?
#ifdef __AVX2__
  uint64_t x = _mm256_movemask_epi8(a);
#else
  uint64_t x = _mm_movemask_epi8(a.lo()) | (_mm_movemask_epi8(a.hi()) << 16);
#endif
  // idea from: https://stackoverflow.com/a/45695465/8461272
  // x = 0b a.b. c.d. e.f. g.h. i.j. k.l. m.n. o.p.
  // where a,b,c,d,... are the bits we care about and . represents
  // the bits we don't care about

  x >>= 1;
  // x = 0b .a.b .c.d .e.f .g.h .i.j .k.l .m.n .o.p

  x = ((x & 0x44444444) >> 1) | (x & 0x11111111);
  // x = 0b ..ab ..cd ..ef ..gh ..ij ..kl ..mn ..op

  x = ((x & 0x30303030) >> 2) | (x & 0x03030303);
  // x = 0b .... abcd .... efgh .... ijkl .... mnop

  x = ((x & 0x0F000F00) >> 4) | (x & 0x000F000F);
  // x = 0b .... .... abcd efgh .... .... ijkl mnop

  x = ((x & 0x00FF0000) >> 8) | (x & 0x000000FF);
  // x = 0b .... .... .... .... abcd efgh ijkl mnop

  return x;
}

static SIMD_INLINE uint64_t msb2int(const Vec<Int, 32> &a)
{
  return _mm256_movemask_ps(_mm256_castsi256_ps(a));
}

static SIMD_INLINE uint64_t msb2int(const Vec<Float, 32> &a)
{
  return _mm256_movemask_ps(a);
}

static SIMD_INLINE uint64_t msb2int(const Vec<Double, 32> &a)
{
  return _mm256_movemask_pd(a);
}

// ---------------------------------------------------------------------------
// int2msb
// ---------------------------------------------------------------------------

// 06. Oct 22 (Jonas Keller): added int2msb functions

static SIMD_INLINE Vec<Byte, 32> int2msb(const uint64_t a, OutputType<Byte>,
                                         Integer<32>)
{
#ifdef __AVX2__
  __m256i shuffleIndeces = _mm256_set_epi64x(
    0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
  __m256i aVec     = _mm256_shuffle_epi8(_mm256_set1_epi32(a), shuffleIndeces);
  __m256i sel      = _mm256_set1_epi64x(0x8040201008040201);
  __m256i selected = _mm256_and_si256(aVec, sel);
  __m256i result   = _mm256_cmpeq_epi8(selected, sel);
  return _mm256_and_si256(result, _mm256_set1_epi8((int8_t) 0x80));
#else
  __m128i shuffleIndeces = _mm_set_epi64x(0x0101010101010101, 0);
  __m128i aVecLo = _mm_shuffle_epi8(_mm_cvtsi32_si128(a), shuffleIndeces);
  __m128i aVecHi = _mm_shuffle_epi8(_mm_cvtsi32_si128(a >> 16), shuffleIndeces);
  __m128i sel    = _mm_set1_epi64x(0x8040201008040201);
  __m128i selectedLo = _mm_and_si128(aVecLo, sel);
  __m128i selectedHi = _mm_and_si128(aVecHi, sel);
  __m128i resultLo   = _mm_cmpeq_epi8(selectedLo, sel);
  __m128i resultHi   = _mm_cmpeq_epi8(selectedHi, sel);
  __m256i result     = _mm256_set_m128i(resultHi, resultLo);
  return _mm256_castps_si256(
    _mm256_and_ps(_mm256_castsi256_ps(result),
                  _mm256_castsi256_ps(_mm256_set1_epi8((int8_t) 0x80))));
#endif
}

static SIMD_INLINE Vec<SignedByte, 32> int2msb(const uint64_t a,
                                               OutputType<SignedByte>,
                                               Integer<32>)
{
  return reinterpret(int2msb(a, OutputType<Byte>(), Integer<32>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Short, 32> int2msb(const uint64_t a, OutputType<Short>,
                                          Integer<32>)
{
#ifdef __AVX2__
  __m256i aVec = _mm256_set1_epi16(a);
  __m256i sel  = _mm256_set_epi16(
    (int16_t) 0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100,
    0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001);
  __m256i selected = _mm256_and_si256(aVec, sel);
  __m256i result   = _mm256_cmpeq_epi16(selected, sel);
  return _mm256_and_si256(result, _mm256_set1_epi16((int16_t) 0x8000));
#else
  __m128i aVec  = _mm_set1_epi16(a);
  __m128i selLo = _mm_set_epi16(0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004,
                                0x0002, 0x0001);
  __m128i selHi = _mm_set_epi16((int16_t) 0x8000, 0x4000, 0x2000, 0x1000,
                                0x0800, 0x0400, 0x0200, 0x0100);
  __m128i selectedLo = _mm_and_si128(aVec, selLo);
  __m128i selectedHi = _mm_and_si128(aVec, selHi);
  __m128i resultLo   = _mm_cmpeq_epi16(selectedLo, selLo);
  __m128i resultHi   = _mm_cmpeq_epi16(selectedHi, selHi);
  __m256i result     = _mm256_set_m128i(resultHi, resultLo);
  return _mm256_castps_si256(
    _mm256_and_ps(_mm256_castsi256_ps(result),
                  _mm256_castsi256_ps(_mm256_set1_epi16((int16_t) 0x8000))));
#endif
}

static SIMD_INLINE Vec<Word, 32> int2msb(const uint64_t a, OutputType<Word>,
                                         Integer<32>)
{
  return reinterpret(int2msb(a, OutputType<Short>(), Integer<32>()),
                     OutputType<Word>());
}

static SIMD_INLINE Vec<Int, 32> int2msb(const uint64_t a, OutputType<Int>,
                                        Integer<32>)
{
#ifdef __AVX2__
  __m256i aVec = _mm256_set1_epi32(a);
  __m256i sel =
    _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
  __m256i selected = _mm256_and_si256(aVec, sel);
  __m256i result   = _mm256_cmpeq_epi32(selected, sel);
  return _mm256_and_si256(result, _mm256_set1_epi32(0x80000000));
#else
  __m128i aVec       = _mm_set1_epi32(a);
  __m128i selLo      = _mm_set_epi32(0x08, 0x04, 0x02, 0x01);
  __m128i selHi      = _mm_set_epi32(0x80, 0x40, 0x20, 0x10);
  __m128i selectedLo = _mm_and_si128(aVec, selLo);
  __m128i selectedHi = _mm_and_si128(aVec, selHi);
  __m256i result     = _mm256_set_m128i(_mm_cmpeq_epi32(selectedHi, selHi),
                                        _mm_cmpeq_epi32(selectedLo, selLo));
  return _mm256_castps_si256(
    _mm256_and_ps(_mm256_castsi256_ps(result),
                  _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))));
#endif
}

static SIMD_INLINE Vec<Float, 32> int2msb(const uint64_t a, OutputType<Float>,
                                          Integer<32>)
{
  return reinterpret(int2msb(a, OutputType<Int>(), Integer<32>()),
                     OutputType<Float>());
}

static SIMD_INLINE Vec<Double, 32> int2msb(const uint64_t a, OutputType<Double>,
                                           Integer<32>)
{
#ifdef __AVX2__
  const __m256i aVec     = _mm256_set1_epi64x(a);
  const __m256i sel      = _mm256_set_epi64x(8, 4, 2, 1);
  const __m256i selected = _mm256_and_si256(aVec, sel);
  const __m256i result   = _mm256_cmpeq_epi64(selected, sel);
  return _mm256_castsi256_pd(
    _mm256_and_si256(result, _mm256_set1_epi64x(0x8000000000000000)));
#else
  const __m128i aVec       = _mm_set1_epi64x(a);
  const __m128i selLo      = _mm_set_epi64x(2, 1);
  const __m128i selHi      = _mm_set_epi64x(8, 4);
  const __m128i selectedLo = _mm_and_si128(aVec, selLo);
  const __m128i selectedHi = _mm_and_si128(aVec, selHi);
  const __m256i result = _mm256_set_m128i(_mm_cmpeq_epi64(selectedHi, selHi),
                                          _mm_cmpeq_epi64(selectedLo, selLo));
  return _mm256_and_pd(
    _mm256_castsi256_pd(result),
    _mm256_castsi256_pd(_mm256_set1_epi64x(0x8000000000000000)));
#endif
}

// ---------------------------------------------------------------------------
// int2bits
// ---------------------------------------------------------------------------

// 09. Oct 22 (Jonas Keller): added int2bits

static SIMD_INLINE Vec<Byte, 32> int2bits(const uint64_t a, OutputType<Byte>,
                                          Integer<32>)
{
#ifdef __AVX2__
  __m256i shuffleIndeces = _mm256_set_epi64x(
    0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0);
  __m256i aVec     = _mm256_shuffle_epi8(_mm256_set1_epi32(a), shuffleIndeces);
  __m256i sel      = _mm256_set1_epi64x(0x8040201008040201);
  __m256i selected = _mm256_and_si256(aVec, sel);
  return _mm256_cmpeq_epi8(selected, sel);
#else
  return _mm256_set_m128i(int2bits(a >> 16, OutputType<Byte>(), Integer<16>()),
                          int2bits(a, OutputType<Byte>(), Integer<16>()));
#endif
}

static SIMD_INLINE Vec<SignedByte, 32> int2bits(const uint64_t a,
                                                OutputType<SignedByte>,
                                                Integer<32>)
{
  return reinterpret(int2bits(a, OutputType<Byte>(), Integer<32>()),
                     OutputType<SignedByte>());
}

static SIMD_INLINE Vec<Short, 32> int2bits(const uint64_t a, OutputType<Short>,
                                           Integer<32>)
{
#ifdef __AVX2__
  __m256i aVec = _mm256_set1_epi16(a);
  __m256i sel  = _mm256_set_epi16(
    (int16_t) 0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100,
    0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001);
  __m256i selected = _mm256_and_si256(aVec, sel);
  return _mm256_cmpeq_epi16(selected, sel);
#else
  return _mm256_set_m128i(int2bits(a >> 8, OutputType<Short>(), Integer<16>()),
                          int2bits(a, OutputType<Short>(), Integer<16>()));
#endif
}

static SIMD_INLINE Vec<Word, 32> int2bits(const uint64_t a, OutputType<Word>,
                                          Integer<32>)
{
  return reinterpret(int2bits(a, OutputType<Short>(), Integer<32>()),
                     OutputType<Word>());
}

static SIMD_INLINE Vec<Int, 32> int2bits(const uint64_t a, OutputType<Int>,
                                         Integer<32>)
{
#ifdef __AVX2__
  __m256i aVec = _mm256_set1_epi32(a);
  __m256i sel =
    _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
  __m256i selected = _mm256_and_si256(aVec, sel);
  return _mm256_cmpeq_epi32(selected, sel);
#else
  return _mm256_set_m128i(int2bits(a >> 4, OutputType<Int>(), Integer<16>()),
                          int2bits(a, OutputType<Int>(), Integer<16>()));
#endif
}

static SIMD_INLINE Vec<Float, 32> int2bits(const uint64_t a, OutputType<Float>,
                                           Integer<32>)
{
  return reinterpret(int2bits(a, OutputType<Int>(), Integer<32>()),
                     OutputType<Float>());
}

static SIMD_INLINE Vec<Double, 32> int2bits(const uint64_t a,
                                            OutputType<Double>, Integer<32>)
{
#ifdef __AVX2__
  const __m256i aVec     = _mm256_set1_epi64x(a);
  const __m256i sel      = _mm256_set_epi64x(8, 4, 2, 1);
  const __m256i selected = _mm256_and_si256(aVec, sel);
  const __m256i result   = _mm256_cmpeq_epi64(selected, sel);
  return _mm256_castsi256_pd(result);
#else
  const __m128i aVec       = _mm_set1_epi64x(a);
  const __m128i selLo      = _mm_set_epi64x(2, 1);
  const __m128i selHi      = _mm_set_epi64x(8, 4);
  const __m128i selectedLo = _mm_and_si128(aVec, selLo);
  const __m128i selectedHi = _mm_and_si128(aVec, selHi);
  const __m256i result = _mm256_set_m128i(_mm_cmpeq_epi64(selectedHi, selHi),
                                          _mm_cmpeq_epi64(selectedLo, selLo));
  return _mm256_castsi256_pd(result);
#endif
}

// ---------------------------------------------------------------------------
// iota
// ---------------------------------------------------------------------------

// 30. Jan 23 (Jonas Keller): added iota

static SIMD_INLINE Vec<Byte, 32> iota(OutputType<Byte>, Integer<32>)
{
  return _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                         17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                         1, 0);
}

static SIMD_INLINE Vec<SignedByte, 32> iota(OutputType<SignedByte>, Integer<32>)
{
  return _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                         17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                         1, 0);
}

static SIMD_INLINE Vec<Short, 32> iota(OutputType<Short>, Integer<32>)
{
  return _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Word, 32> iota(OutputType<Word>, Integer<32>)
{
  return _mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Int, 32> iota(OutputType<Int>, Integer<32>)
{
  return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
}

static SIMD_INLINE Vec<Float, 32> iota(OutputType<Float>, Integer<32>)
{
  return _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
}

static SIMD_INLINE Vec<Double, 32> iota(OutputType<Double>, Integer<32>)
{
  return _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
}

} // namespace base
} // namespace internal
} // namespace simd

#endif

#endif // SIMD_VEC_BASE_IMPL_INTEL_32_H_
