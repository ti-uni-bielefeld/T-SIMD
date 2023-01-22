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

#ifndef _SIMD_INTRINS_INTEL_H_
#define _SIMD_INTRINS_INTEL_H_

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
    (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 50500 && \
    !defined(__OPTIMIZE__) && defined(__AVX512BW__)

// _mm512_pack[u]s_epi32 doesn't need a define (no int arguments),
// but is not available without optimization (error in include file)

extern __inline __m512i
__attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
_mm512_packs_epi32 (__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_packssdw512_mask ((__v16si) __A,
                (__v16si) __B,
                (__v32hi)
                _mm512_setzero_hi (),
                (__mmask32) -1);
}

extern __inline __m512i
__attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
_mm512_packus_epi32 (__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_packusdw512_mask ((__v16si) __A,
                (__v16si) __B,
                (__v32hi)
                _mm512_setzero_hi (),
                (__mmask32) -1);
}

#endif

#endif // SIMDVEC_INTEL_ENABLE

#endif // _SIMD_INTRINS_INTEL_H_

