// ===========================================================================
//
// SIMDVec.H --
// generic template for Vec
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

// 13. Feb 23 (Jonas Keller): removed "SIMD"-prefix from most types
// (versions with "SIMD"-prefix are still available for backward
// compatibility in SIMDBackwardCompat.H)

// 09. Mar 23 (Jonas Keller): added doxygen documentation

#ifndef SIMD_VEC_H_
#define SIMD_VEC_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"

#include <algorithm>
#include <type_traits>

/**
 * @brief Namespace for T-SIMD.
 */
namespace simd {

// TODO: - absdiff also for unsigned types -> SSE lecture vecintrin66
// TODO: - bitwise shift: what about float?
// TODO: - loadr / storer
// TODO: - element-wise rotation in a vector using alignre
// TODO: - functions for rsqrt, rcp Newton *steps*?
// TODO: - add to names rcp, rsqrt something with "estimate"?
// TODO: - NEON has a "set1" with immediate arguments (vmovq_n), so it would
// TODO:   be nice to have a set1const function with template argument, but
// TODO:   this only works for integers since float template parameter are not
// TODO:   allowed by the standard

// ===========================================================================
// generic template for Vec and Mask
// ===========================================================================

// 30. Sep 19 (rm): Mask support contributed by Markus Vieth

// specialized for type of elements and number of bytes in the SIMD vector
/**
 * @ingroup group_types
 * @brief SIMD vector class, holds multiple elements of the same type.
 *
 * @tparam T type of the vector elements
 * @tparam SIMD_WIDTH number of bytes in the SIMD vector
 */
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
class Vec
#ifdef DOXYGEN
{
public:
  /**
   * @brief Number of elements in the vector.
   */
  static constexpr int elements = SIMD_WIDTH / sizeof(T);

  /**
   * @brief Number of elements in the vector. Alias for @ref elements.
   */
  static constexpr int elems = elements;

  /**
   * @brief Number of bytes in the vector.
   */
  static constexpr int bytes = SIMD_WIDTH;
}
#endif
;

/**
 * @ingroup group_types
 * @brief SIMD mask class consisting of as many bits as the corresponding
 *        Vec has elements.
 *
 * @tparam T type of the elements of the corresponding Vec
 * @tparam SIMD_WIDTH number of bytes in the corresponding Vec
 */
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
class Mask
#ifdef DOXYGEN
{
public:
  /**
   * @brief Constructs a Mask from a Vec.
   *
   * The Mask bits are set to 1 if the corresponding Vec element has its
   * most significant bit set, otherwise the Mask bit is set to 0.
   */
  explicit SIMD_INLINE Mask<T, SIMD_WIDTH>(const Vec<T, SIMD_WIDTH> &x);

  /**
   * @brief Constructs a Mask from an integer.
   *
   * The lowest bits of the integer are used to set the Mask bits.
   */
  explicit SIMD_INLINE Mask<T, SIMD_WIDTH>(const uint64_t x);

  /**
   * @brief Converts the Mask to a Vec.
   *
   * The bits of the Vec elements are set to all 1s if the corresponding
   * Mask bit is set, otherwise the Vec element bits are set to 0.
   *
   * @return converted Vec from the Mask
   */
  explicit SIMD_INLINE operator Vec<T, SIMD_WIDTH>() const { return mask; };

  /**
   * @brief Converts the Mask to an integer.
   *
   * The lowest bits of the integer are set to the Mask bits.
   *
   * @return converted integer from the Mask
   */
  explicit SIMD_INLINE operator uint64_t() const;

  /**
   * @brief Returns the Mask bit at the given index.
   *
   * @param i index of the Mask bit
   * @return whether the Mask bit at the given index is set
   */
  SIMD_INLINE bool operator[](const int i) const;

  /**
   * @brief Compares the Mask with another Mask.
   *
   * @param other Mask to compare with
   * @return whether the Mask is equal to the other Mask
   */
  SIMD_INLINE bool operator==(const Mask<T, SIMD_WIDTH> &other) const;
}
#endif
;

// ===========================================================================
// helper functions for templates converting from Tin to Tout
// ===========================================================================

// numInVecs() and numOutVecs() assume that either
// - a single vector is extended into multiple vectors or
// - multiple vectors are packed into a single vector
//
// numSIMDVecsElements encodes the number of elements in *all* input / all
// output vectors
//
// removed: numSIMDVecElements encodes the number of elements in each vector
// (or use Vec::elements instead)
//
// removed: numInputSIMDVecElements/numOutputSIMDVecElements encode
// the number of elements in *each* input / output vector

/**
 * @ingroup group_type_conversion
 * @brief Number of input vectors for functions that potentially change the
 * size of the elements but not the number of elements.
 *
 * @tparam Tout output type
 * @tparam Tin input type
 * @return number of input vectors
 */
template <typename Tout, typename Tin>
static constexpr SIMD_INLINE int numInVecs()
{
  return (sizeof(Tout) < sizeof(Tin)) ? (sizeof(Tin) / sizeof(Tout)) : 1;
}

/**
 * @ingroup group_type_conversion
 * @brief Number of output vectors for functions that potentially change the
 * size of the elements but not the number of elements.
 *
 * @tparam Tout output type
 * @tparam Tin input type
 * @return number of output vectors
 */
template <typename Tout, typename Tin>
static constexpr SIMD_INLINE int numOutVecs()
{
  return (sizeof(Tout) > sizeof(Tin)) ? (sizeof(Tout) / sizeof(Tin)) : 1;
}

/**
 * @ingroup group_type_conversion
 * @brief Number of elements in all input vectors for functions that potentially
 * change the size of the elements but not the number of elements.
 *
 * @tparam Tout output type
 * @tparam Tin input type
 * @return number of elements in all input vectors
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static constexpr SIMD_INLINE int numSIMDVecsElements()
{
  return (sizeof(Tout) > sizeof(Tin)) ? Vec<Tin, SIMD_WIDTH>::elems :
                                        Vec<Tout, SIMD_WIDTH>::elems;
}

// 13. May 23 (Jonas Keller): added BigEnoughFloat

namespace internal {
namespace vec {
// std::max is not constexpr in C++11, so we need to provide our own, since
// we need it in a constexpr context
template <typename T>
constexpr const T &max(const T &a, const T &b)
{
  return (a < b) ? b : a;
}
} // namespace vec
} // namespace internal

/**
 * @ingroup group_fops
 * @brief Smallest floating point type that is at least as big as the
 * input and output types.
 *
 * @tparam Tout output type
 * @tparam Tin input type
 */
template <typename Tout, typename Tin>
using BigEnoughFloat =
  typename std::conditional<internal::vec::max(sizeof(Tout), sizeof(Tin)) <=
                              sizeof(Float),
                            Float, Double>::type;

// 22. Jan 23 (Jonas Keller): removed primary template functions, as they are
// not needed anymore due to a wrapper layer

} // namespace simd

#endif
