// ===========================================================================
//
// SIMDVec.H --
// generic template for SIMDVec
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

#ifndef _SIMD_VEC_H_
#define _SIMD_VEC_H_

#include "SIMDDefs.H"

/**
 * @brief Namespace for T-SIMD.
 */
namespace ns_simd {

// TODO: - absdiff also for unsigned types -> SSE lecture vecintrin66
// TODO: - bitwise shift: what about float?
// TODO: - loadr / storer
// TODO: - saturated workarounds for adds/hadds for SIMDInt?
// TODO: - element-wise rotation in a vector using alignre
// TODO: - functions for rsqrt, rcp Newton *steps*?
// TODO: - add to names rcp, rsqrt something with "estimate"?
// TODO: - NEON has a "set1" with immediate arguments (vmovq_n), so it would
// TODO:   be nice to have a set1const function with template argument, but
// TODO:   this only works for integers since float template parameter are not
// TODO:   allowed by the standard

// ===========================================================================
// generic template for SIMDVec and SIMDMask
// ===========================================================================

// 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth

// specialized for type of elements and number of bytes in the SIMD vector
/**
 * @ingroup types
 * @brief SIMD vector class, holds multiple elements of the same type.
 *
 * @tparam T the type of the vector elements
 * @tparam SIMD_WIDTH the number of bytes in the SIMD vector
 */
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
class SIMDVec;

/**
 * @ingroup types
 * @brief SIMD mask class consisting of as many bits as the corresponding
 *        SIMDVec has elements.
 *
 * @tparam T the type of the elements of the corresponding SIMDVec
 * @tparam SIMD_WIDTH the number of bytes in the corresponding SIMDVec
 */
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
class SIMDMask;
} // namespace ns_simd

#endif
