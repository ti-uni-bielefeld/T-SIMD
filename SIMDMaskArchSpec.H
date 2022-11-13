// ===========================================================================
// 
// SIMDMaskArchSpec.H --
// common include file for architecture specific mask classes and masked
// functions
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

#ifndef _SIMD_MASK_ARCH_SPEC_H_
#define _SIMD_MASK_ARCH_SPEC_H_

#ifdef SIMDVEC_SANDBOX

// test templates
#include "SIMDMaskSandbox.H"

#else // SIMDVEC_SANDBOX

#include "SIMDDefs.H"

// architecture-dependent files:

// Intel
#ifdef SIMDVEC_INTEL_ENABLE
// 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
#include "SIMDMaskIntel16.H"
#include "SIMDMaskIntel32.H"
#include "SIMDMaskIntel64.H"
#endif // SIMDVEC_INTEL_ENABLE

// ARM NEON
#ifdef SIMDVEC_NEON_ENABLE
// 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
#include "SIMDMaskNEON16.H"
#endif // SIMDVEC_NEON_ENABLE

#endif // else SIMDVEC_SANDBOX

#endif // _SIMD_MASK_ARCH_SPEC_H_
