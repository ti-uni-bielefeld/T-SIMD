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

// 03. Aug 22 (Jonas Keller):
// gcc warns that the value returned by the _mm*_undefined_* intrinsics is
// used uninitialized, which is exactly what these intrinsics are for,
// so disabling the warning
#pragma GCC diagnostic push
// clang doesn't know -Wmaybe-uninitialized, so it warns,
// so disabling warning -Wunknown-warning-option
// however, gcc doesn't know -Wunknown-warning-option, so it warns,
// so disabling warning -Wpragmas
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
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

#endif // _SIMD_H_

