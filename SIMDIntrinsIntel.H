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

#pragma once
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

// ---------------------------------------------------------------------------
// _mm512_set_epi8 and _mm512_set_epi16 missing in gcc below version 9
// ---------------------------------------------------------------------------

// 29. Sep 23 (Jonas Keller):
// added this fix for missing _mm512_set_epi8 and _mm512_set_epi16 in gcc < 9

// implementation from
// https://github.com/gcc-mirror/gcc/blob/master/gcc/config/i386/avx512fintrin.h

#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER) &&  \
  (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 90000 &&   \
  defined(__AVX512F__)

extern __inline __m512i
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm512_set_epi8(char __q63, char __q62, char __q61, char __q60, char __q59,
                  char __q58, char __q57, char __q56, char __q55, char __q54,
                  char __q53, char __q52, char __q51, char __q50, char __q49,
                  char __q48, char __q47, char __q46, char __q45, char __q44,
                  char __q43, char __q42, char __q41, char __q40, char __q39,
                  char __q38, char __q37, char __q36, char __q35, char __q34,
                  char __q33, char __q32, char __q31, char __q30, char __q29,
                  char __q28, char __q27, char __q26, char __q25, char __q24,
                  char __q23, char __q22, char __q21, char __q20, char __q19,
                  char __q18, char __q17, char __q16, char __q15, char __q14,
                  char __q13, char __q12, char __q11, char __q10, char __q09,
                  char __q08, char __q07, char __q06, char __q05, char __q04,
                  char __q03, char __q02, char __q01, char __q00)
{
  return __extension__(__m512i)(__v64qi) {
    __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07, __q08, __q09, __q10,
    __q11, __q12, __q13, __q14, __q15, __q16, __q17, __q18, __q19, __q20, __q21,
    __q22, __q23, __q24, __q25, __q26, __q27, __q28, __q29, __q30, __q31, __q32,
    __q33, __q34, __q35, __q36, __q37, __q38, __q39, __q40, __q41, __q42, __q43,
    __q44, __q45, __q46, __q47, __q48, __q49, __q50, __q51, __q52, __q53, __q54,
    __q55, __q56, __q57, __q58, __q59, __q60, __q61, __q62, __q63};
}

extern __inline __m512i
  __attribute__((__gnu_inline__, __always_inline__, __artificial__))
  _mm512_set_epi16(short __q31, short __q30, short __q29, short __q28,
                   short __q27, short __q26, short __q25, short __q24,
                   short __q23, short __q22, short __q21, short __q20,
                   short __q19, short __q18, short __q17, short __q16,
                   short __q15, short __q14, short __q13, short __q12,
                   short __q11, short __q10, short __q09, short __q08,
                   short __q07, short __q06, short __q05, short __q04,
                   short __q03, short __q02, short __q01, short __q00)
{
  return __extension__(__m512i)(__v32hi) {
    __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07, __q08, __q09, __q10,
    __q11, __q12, __q13, __q14, __q15, __q16, __q17, __q18, __q19, __q20, __q21,
    __q22, __q23, __q24, __q25, __q26, __q27, __q28, __q29, __q30, __q31};
}

#endif

#endif // SIMDVEC_INTEL_ENABLE

#endif // SIMD_INTRINS_INTEL_H_
