// ===========================================================================
//
// SIMDIntrinsIntel.H --
// includes include files for vector intrinsics on Intel CPUs
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

#ifndef SIMD_INTRINS_INTEL_H_
#define SIMD_INTRINS_INTEL_H_

#include "SIMDDefs.H"

#ifdef SIMDVEC_INTEL_ENABLE

// 30. Aug 22 (Jonas Keller):
// gcc warns that the value returned by the _mm*_undefined_* intrinsics is
// used uninitialized, which is exactly what these intrinsics are for,
// so disabling the warning when compiling with gcc
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__llvm__) && !defined(__INTEL_COMPILER)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
#include <x86intrin.h>
#pragma GCC diagnostic pop

// ---------------------------------------------------------------------------
// some definitions are missing for -O0 in some versions of gcc (e.g. 5.4)
// ---------------------------------------------------------------------------

// 15. Nov 22 (Jonas Keller): moved this to here from SIMDVecBaseImplIntel64.H

// bug seems to be fixed in avx512bwintrin.h in gcc 5.5.0

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) &&  \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 50500 &&   \
  !defined(__OPTIMIZE__) && defined(__AVX512BW__)

// _mm512_pack[u]s_epi32 doesn't need a define (no int arguments),
// but is not available without optimization (error in include file)

extern __inline __m512i
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm512_packs_epi32(__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_packssdw512_mask(
    (__v16si) __A, (__v16si) __B, (__v32hi) _mm512_setzero_hi(),
    (__mmask32) -1);
}

extern __inline __m512i
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm512_packus_epi32(__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_packusdw512_mask(
    (__v16si) __A, (__v16si) __B, (__v32hi) _mm512_setzero_hi(),
    (__mmask32) -1);
}

#endif

// ---------------------------------------------------------------------------
// _mm256_set_m128 and friends are missing in gcc below version 8
// ---------------------------------------------------------------------------

// 14. May 23 (Jonas Keller):
// added this fix for missing _mm256_set_m128 and friends in gcc < 8

// implementation from
// https://github.com/gcc-mirror/gcc/blob/master/gcc/config/i386/avxintrin.h

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) &&  \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 80000 &&   \
  defined(__AVX__)

extern __inline __m256
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm256_set_m128(__m128 __H, __m128 __L)
{
  return _mm256_insertf128_ps(_mm256_castps128_ps256(__L), __H, 1);
}

extern __inline __m256d
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm256_set_m128d(__m128d __H, __m128d __L)
{
  return _mm256_insertf128_pd(_mm256_castpd128_pd256(__L), __H, 1);
}

extern __inline __m256i
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm256_set_m128i(__m128i __H, __m128i __L)
{
  return _mm256_insertf128_si256(_mm256_castsi128_si256(__L), __H, 1);
}

extern __inline __m256
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm256_setr_m128(__m128 __L, __m128 __H)
{
  return _mm256_set_m128(__H, __L);
}

extern __inline __m256d
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm256_setr_m128d(__m128d __L, __m128d __H)
{
  return _mm256_set_m128d(__H, __L);
}

extern __inline __m256i
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm256_setr_m128i(__m128i __L, __m128i __H)
{
  return _mm256_set_m128i(__H, __L);
}

#endif

#endif // SIMDVEC_INTEL_ENABLE

#endif // SIMD_INTRINS_INTEL_H_
