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

/**
 * @file SIMDBackwardCompat.H
 * @brief Aliases for backward compatibility.
 */

#ifndef _SIMD_BACKWARD_COMPAT_H_
#define _SIMD_BACKWARD_COMPAT_H_

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

/// Alias for Vec.
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
using SIMDVec = Vec<T, SIMD_WIDTH>;

/// Alias for Vecs.
template <int NUM, typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
using SIMDVecs = Vecs<NUM, T, SIMD_WIDTH>;

/// Alias for Mask.
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
using SIMDMask = Mask<T, SIMD_WIDTH>;

/// Alias for TypeInfo.
template <typename T>
using SIMDTypeInfo = TypeInfo<T>;

/// Alias for Format.
template <typename T>
using SIMDFormat = Format<T>;

/// Alias for Decimal.
template <typename T>
using SIMDDecimal = Decimal<T>;

/// Alias for NumVecs.
template <typename Tout, typename Tin>
using NumSIMDVecs = NumVecs<Tout, Tin>;

/// Alias for Vec::elems.
template <typename T, int SIMD_WIDTH>
static constexpr SIMD_INLINE int numSIMDVecElements()
{
  return Vec<T, SIMD_WIDTH>::elems;
}

/// Alias for numInVecs().
template <typename Tout, typename Tin>
static constexpr SIMD_INLINE int numInputSIMDVecs()
{
  return numInVecs<Tout, Tin>();
}

/// Alias for numOutVecs().
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
 */
namespace ns_simd {}
#else
namespace ns_simd = simd;
#endif

#endif
