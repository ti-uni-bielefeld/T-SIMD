// ===========================================================================
// 
// SIMD.H --
// includes include files for vector intrinsics
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

#ifndef _SIMD_H_
#define _SIMD_H_

// TODO: - may be a bit confusing: SIMD.H is included from SIMDVecAll.H since
// TODO:   we need the *ENABLE variables to select the right SIMDVec*.H 
// TOOD:   include files, but it is also included from SIMDVec*.H since we
// TODO:   need the include files included here depending on the architecture

#if defined(__i386__) || defined(__x86_64__)

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

#define SIMDVEC_INTEL_ENABLE 1
#endif

// 31. Mar 22 (rm): gcc on ARM doesn't define __arm__
// #if defined(__arm__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMDVEC_NEON_ENABLE 1
#endif

// TODO: - check whether either SIMDVEC_INTEL_ENABLE od
// TODO:   SIMDVEC_NEON_ENABLE are defined? (problem: why doesn't #error
// TODO:   stop compilation?)

// 30. Aug 22 (Jonas Keller): added simd_aligned_malloc and simd_aligned_free
#ifdef _WIN32
#include <malloc.h>
void *simd_aligned_malloc(size_t alignment, size_t size) {
  return _aligned_malloc(size, alignment);
}
void simd_aligned_free(void *ptr) {
  _aligned_free(ptr);
}
#else
#include <stdlib.h>
void *simd_aligned_malloc(size_t alignment, size_t size) {
  void *ptr = NULL;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return NULL;
  }
  return ptr;
}
void simd_aligned_free(void *ptr) {
  free(ptr);
}
#endif

#endif // _SIMD_H_

