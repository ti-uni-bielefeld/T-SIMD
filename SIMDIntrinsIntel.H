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

#endif // SIMDVEC_INTEL_ENABLE

#endif // _SIMD_INTRINS_INTEL_H_

