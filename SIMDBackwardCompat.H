// ===========================================================================
//
// SIMDBackwardCompat.H --
// aliases for backward compatibility
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
// (C) Jonas Keller, Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

// 13. Feb 23 (Jonas Keller): added SIMDBackwardCompat.H for backward
// compatibility of renamed types and functions

// 09. Mar 23 (Jonas Keller): added doxygen documentation

/**
 * @file SIMDBackwardCompat.H
 * @brief Aliases for backward compatibility.
 */

#ifndef SIMD_BACKWARD_COMPAT_H_
#define SIMD_BACKWARD_COMPAT_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecs.H"

namespace simd {

/**
 * @addtogroup group_aliases
 * @{
 */

using SIMDByte       = Byte;       ///< Alias for Byte.
using SIMDSignedByte = SignedByte; ///< Alias for SignedByte.
using SIMDWord       = Word;       ///< Alias for Word.
using SIMDShort      = Short;      ///< Alias for Short.
using SIMDInt        = Int;        ///< Alias for Int.
using SIMDFloat      = Float;      ///< Alias for Float.

typedef SIMDByte *SIMDBytePtr;             ///< Pointer to SIMDByte.
typedef SIMDSignedByte *SIMDSignedBytePtr; ///< Pointer to SIMDSignedByte.
typedef SIMDWord *SIMDWordPtr;             ///< Pointer to SIMDWord.
typedef SIMDShort *SIMDShortPtr;           ///< Pointer to SIMDShort.
typedef SIMDInt *SIMDIntPtr;               ///< Pointer to SIMDInt.
typedef SIMDFloat *SIMDFloatPtr;           ///< Pointer to SIMDFloat.

/// Alias for Vec. @deprecated Use Vec instead.
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
using SIMDVec = Vec<T, SIMD_WIDTH>;

/// Alias for Vecs. @deprecated Use Vecs instead.
template <int NUM, typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
using SIMDVecs = Vecs<NUM, T, SIMD_WIDTH>;

/// Alias for Mask. @deprecated Use Mask instead.
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
using SIMDMask = Mask<T, SIMD_WIDTH>;

/// Alias for TypeInfo. @deprecated Use TypeInfo instead.
template <typename T>
using SIMDTypeInfo = TypeInfo<T>;

/// Alias for Format. @deprecated Use Format instead.
template <typename T>
using SIMDFormat = Format<T>;

/// Alias for Decimal. @deprecated Use Decimal instead.
template <typename T>
using SIMDDecimal = Decimal<T>;

/// Alias for NumVecs. @deprecated Use NumVecs instead.
template <typename Tout, typename Tin>
using NumSIMDVecs = NumVecs<Tout, Tin>;

/// Alias for Vec<T, SIMD_WIDTH>::elems. @deprecated Use Vec::elems instead.
template <typename T, int SIMD_WIDTH>
static constexpr SIMD_INLINE int numSIMDVecElements()
{
  return Vec<T, SIMD_WIDTH>::elems;
}

/// Alias for numInVecs(). @deprecated Use numInVecs() instead.
template <typename Tout, typename Tin>
static constexpr SIMD_INLINE int numInputSIMDVecs()
{
  return numInVecs<Tout, Tin>();
}

/// Alias for numOutVecs(). @deprecated Use numOutVecs() instead.
template <typename Tout, typename Tin>
static constexpr SIMD_INLINE int numOutputSIMDVecs()
{
  return numOutVecs<Tout, Tin>();
}

/** @} */
} // namespace simd

// some IDEs don't see this if it comes first (compiles, but IDE has problems)
// shifted here:

#ifdef DOXYGEN
// doxygen doesn't understand namespace aliases
/**
 * @ingroup group_aliases
 * @brief Alias for @ref simd.
 * @deprecated Use @ref simd instead.
 */
namespace ns_simd {}
#else
namespace ns_simd = simd;
#endif

#endif
