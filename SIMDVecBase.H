// ===========================================================================
// 
// SIMDVecBase.H --
// common include file for base-level classes and functions
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

#ifndef _SIMDVEC_BASE_H_
#define _SIMDVEC_BASE_H_

#ifdef SIMDVEC_SANDBOX

// test templates
#include "SIMDVecSandbox.H"

#else

#include "SIMD.H"

// architecture-dependent files:

// Intel
#ifdef SIMDVEC_INTEL_ENABLE
// order of includes from newer to older vector extensions is
// important for the definition of NATIVE_SIMD_WIDTH!

// specialized templates and overloaded functions for AVX512
// TODO: 31. Mar 22 (rm) remove this later, only for tests
// #ifdef USE_OLD_SIMDVECINTEL64
// #include "SIMDVecIntel64_OldUnpack.H"
// #else
#include "SIMDVecIntel64.H"
// #endif

// specialized templates and overloaded functions for AVX
#include "SIMDVecIntel32.H"

// specialized templates and overloaded functions for SSE
#include "SIMDVecIntel16.H"

// 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
#include "SIMDMaskIntel64.H"
#include "SIMDMaskIntel32.H"
#include "SIMDMaskIntel16.H"
#include "SIMDMaskEmu.H"

#endif // SIMDVEC_INTEL_ENABLE

// ARM NEON
#ifdef SIMDVEC_NEON_ENABLE
#include "SIMDVecNEON16.H"
// 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
#include "SIMDMaskNEON16.H"
#include "SIMDMaskEmu.H"

#endif // SIMDVEC_INTEL_ENABLE

#endif // else SIMDVEC_SANDBOX

#endif // _SIMDVEC_BASE_H_
