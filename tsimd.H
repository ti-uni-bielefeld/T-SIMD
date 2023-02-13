// ===========================================================================
//
// tsimd.H --
// different access to SIMD* types and functions, only works with C++11
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

/**
 * @file tsimd.H
 * @brief Main file to include for using the SIMD library. Requires C++11.
 *
 * This file contains aliases for the SIMD types and the ns_simd namespace
 * that might be faster to type than the original names.
 *
 * Use SIMDVecAll.H instead if C++11 is not available.
 */

#ifndef _TSIMD_H_
#define _TSIMD_H_

#include "SIMDVecAll.H"

namespace ns_simd {

// we can use element types simd::Byte instead of SIMDByte etc.
/**
 * @ingroup types
 * @defgroup aliases Aliases For SIMD Types
 * @brief Shorter names for SIMD types. Requires C++11.
 *
 * The file tsimd.H must be included to use these aliases.
 *
 * @{
 */
typedef SIMDByte Byte; ///< Alias for @ref SIMDByte. Requires C++11.
typedef SIMDSignedByte
  SignedByte;            ///< Alias for @ref SIMDSignedByte. Requires C++11.
typedef SIMDWord Word;   ///< Alias for @ref SIMDWord. Requires C++11.
typedef SIMDShort Short; ///< Alias for @ref SIMDShort. Requires C++11.
typedef SIMDInt Int;     ///< Alias for @ref SIMDInt. Requires C++11.
typedef SIMDFloat Float; ///< Alias for @ref SIMDFloat. Requires C++11.

// we can use simd::Vec instead of SIMDVec (requires C++11)
// 15. Oct 21 (rm): add NATIVE_SIMD_WIDTH
/// Alias for SIMDVec. Requires C++11.
template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
using Vec = SIMDVec<T, SIMD_WIDTH>;

// we can use simd::Vecs instead of SIMDVecs (requires C++11)
// 15. Oct 21 (rm): add NATIVE_SIMD_WIDTH
/// Alias for SIMDVecs. Requires C++11.
template <int NUM, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
using Vecs = SIMDVecs<NUM, T, SIMD_WIDTH>;

// 05. Aug 22 (Jonas Keller): add simd::Mask alias for SIMDMask
// we can use simd::Mask instead of SIMDMask (requires C++11)
/// Alias for SIMDMask. Requires C++11.
template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
using Mask = SIMDMask<T, SIMD_WIDTH>;

// we can use simd::TypeInfo instead of SIMDTypeInfo (requires C++11)
/// Alias for SIMDTypeInfo. Requires C++11.
template <typename T>
using TypeInfo = SIMDTypeInfo<T>;

// we can use simd::Format instead of SIMDFormat (requires C++11)
/// Alias for SIMDFormat. Requires C++11.
template <typename T>
using Format = SIMDFormat<T>;

// we can use simd::Decimal instead of SIMDDecimal (requires C++11)
/// Alias for SIMDDecimal. Requires C++11.
template <typename T>
using Decimal = SIMDDecimal<T>;
/** @} */

} // namespace ns_simd

// some IDEs don't see this if it comes first (compiles, but IDE has problems)
// shifted here:

// introduce alias namespace "simd"

#ifdef DOXYGEN
// doxygen doesn't understand namespace aliases
/**
 * @ingroup aliases
 * @brief Alias for @ref ns_simd. Requires C++11.
 */
namespace simd {}
#else
namespace simd = ns_simd;
#endif

#endif
