// ===========================================================================
//
// SIMDVecMask.H --
// mask classes and masked functions
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

// 22. Jan 23 (Jonas Keller): introduced wrapper layer that wraps the internal
// architecture-specific implementations

// 02. Mar 23 (Jonas Keller): added doxygen documentation

#ifndef _SIMD_VEC_MASK_H_
#define _SIMD_VEC_MASK_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <stdint.h>

#ifdef SIMDVEC_SANDBOX

// test templates
#include "SIMDVecMaskImplSandbox.H"

#else // SIMDVEC_SANDBOX

#ifdef SIMDVEC_INTEL_ENABLE
#include "SIMDVecMaskImplIntel64.H"
#endif // SIMDVEC_INTEL_ENABLE

#include "SIMDVecMaskImplEmu.H"

#endif // else SIMDVEC_SANDBOX

namespace simd {

/**
 * @addtogroup group_mask
 * @{
 */

/**
 * @brief Masked version of set1(const T).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_set1(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const T a)
{
  return internal::mask::mask_set1(src, k, a);
}

/**
 * @brief Zero-masked version of set1(const T).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_set1(const Mask<T, SIMD_WIDTH> &k,
                                                 const T a)
{
  return internal::mask::maskz_set1(k, a);
}

/**
 * @brief Masked version of cvts(const Vec<Tin, SIMD_WIDTH> &).
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> mask_cvts(
  const Vec<Tout, SIMD_WIDTH> &src, const Mask<Tin, SIMD_WIDTH> &k,
  const Vec<Tin, SIMD_WIDTH> &a)
{
  return internal::mask::mask_cvts(src, k, a);
}

/**
 * @brief Zero-masked version of cvts(const Vec<Tin, SIMD_WIDTH> &).
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> maskz_cvts(
  const Mask<Tin, SIMD_WIDTH> &k, const Vec<Tin, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_cvts(k, a);
}

/** @} */

/**
 * @addtogroup group_mask_functions
 * @{
 */

// 06. Feb 23 (Jonas Keller): added reinterpret_mask

/**
 * @brief Reinterprets a Mask of one type as a Mask of another type.
 *
 * The size of the element type of the Mask must be the same.
 *
 * @tparam Tout element type of the reinterpreted Mask
 * @tparam Tin element type of the Mask to reinterpret
 * @param a Mask to reinterpret
 * @return reinterpreted Mask
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE Mask<Tout, SIMD_WIDTH> reinterpret_mask(
  const Mask<Tin, SIMD_WIDTH> &a)
{
  return internal::mask::reinterpret_mask<Tout, Tin, SIMD_WIDTH>(a);
}

/**
 * @brief Adds two Mask's together as if they were integers.
 *
 * @param a first Mask
 * @param b second Mask
 * @return sum of the two Mask's
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kadd(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kadd(a, b);
}

/**
 * @brief Computes the bitwise AND of two Mask's.
 *
 * @param a first Mask
 * @param b second Mask
 * @return bitwise AND of the two Mask's
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kand(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kand(a, b);
}

/**
 * @brief Computes bitwise ANDNOT of two Mask's.
 *
 * The result is computed as (not a) and b.
 *
 * @param a first Mask
 * @param b second Mask
 * @return bitwise ANDNOT of the two Mask's
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kandn(const Mask<T, SIMD_WIDTH> &a,
                                             const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kandn(a, b);
}

/**
 * @brief Computes the bitwise OR of two Mask's.
 *
 * @param a first Mask
 * @param b second Mask
 * @return bitwise OR of the two Mask's
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kor(const Mask<T, SIMD_WIDTH> &a,
                                           const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kor(a, b);
}

/**
 * @brief Computes the bitwise XOR of two Mask's.
 *
 * @param a first Mask
 * @param b second Mask
 * @return bitwise XOR of the two Mask's
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kxor(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kxor(a, b);
}

/**
 * @brief Computes the bitwise XNOR of two Mask's.
 *
 * @param a first Mask
 * @param b second Mask
 * @return bitwise XNOR of the two Mask's
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kxnor(const Mask<T, SIMD_WIDTH> &a,
                                             const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kxnor(a, b);
}

/**
 * @brief Computes the bitwise NOT of a Mask.
 *
 * @param a Mask
 * @return bitwise NOT of the Mask
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> knot(const Mask<T, SIMD_WIDTH> &a)
{
  return internal::mask::knot(a);
}

/**
 * @brief Shifts the bits of a Mask to the right by a
 *        constant number of bits.
 *
 * @param a Mask
 * @tparam COUNT number of bits to shift
 * @return shifted Mask
 */
template <unsigned int COUNT, typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftri(const Mask<T, SIMD_WIDTH> &a)
{
  return internal::mask::kshiftri<COUNT>(a);
}

/**
 * @brief Shifts the bits of a Mask to the left by a
 *        constant number of bits.
 *
 * @param a Mask
 * @tparam COUNT number of bits to shift
 * @return shifted Mask
 */
template <unsigned int COUNT, typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftli(const Mask<T, SIMD_WIDTH> &a)
{
  return internal::mask::kshiftli<COUNT>(a);
}

/**
 * @brief Shifts the bits of a Mask to the right by a
 *        variable number of bits.
 *
 * @param a Mask
 * @param count number of bits to shift
 * @return shifted Mask
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftri(const Mask<T, SIMD_WIDTH> &a,
                                                const uint64_t count)
{
  return internal::mask::kshiftri(a, count);
}

/**
 * @brief Shifts the bits of a Mask to the left by a
 *        variable number of bits.
 *
 * @param a Mask
 * @param count number of bits to shift
 * @return shifted Mask
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftli(const Mask<T, SIMD_WIDTH> &a,
                                                const uint64_t count)
{
  return internal::mask::kshiftli(a, count);
}

/**
 * @brief Creates a Mask with all elements set to true.
 *
 * @return Mask with all elements set to true
 */
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_all_ones()
{
  return internal::mask::mask_all_ones(internal::OutputType<T>(),
                                       internal::Integer<SIMD_WIDTH>());
}

// 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
// mask_set_true/false_low/high.

/**
 * @brief Sets the lower @p x bits of a Mask to true.
 *
 * The remaining bits are set to false.
 *
 * @param x number of bits to set to true
 * @return Mask with the lower @p x bits set to true
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_true_low(const unsigned int x)
{
  return internal::mask::mask_set_true_low(x, internal::OutputType<T>(),
                                           internal::Integer<SIMD_WIDTH>());
}

/**
 * @brief Sets the upper @p x bits of a Mask to true.
 *
 * The remaining bits are set to false.
 *
 * @param x number of bits to set to true
 * @return Mask with the upper @p x bits set to true
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_true_high(const unsigned int x)
{
  return internal::mask::mask_set_true_high(x, internal::OutputType<T>(),
                                            internal::Integer<SIMD_WIDTH>());
}

/**
 * @brief Sets the lower @p x bits of a Mask to false.
 *
 * The remaining bits are set to true.
 *
 * @param x number of bits to set to false
 * @return Mask with the lower @p x bits set to false
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_false_low(const unsigned int x)
{
  return internal::mask::mask_set_false_low(x, internal::OutputType<T>(),
                                            internal::Integer<SIMD_WIDTH>());
}

/**
 * @brief Sets the upper @p x bits of a Mask to false.
 *
 * The remaining bits are set to true.
 *
 * @param x number of bits to set to false
 * @return Mask with the upper @p x bits set to false
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_false_high(const unsigned int x)
{
  return internal::mask::mask_set_false_high(x, internal::OutputType<T>(),
                                             internal::Integer<SIMD_WIDTH>());
}

/** @} */

/**
 * @addtogroup group_mask_memory
 * @{
 */

/**
 * @brief Masked versions of load(const T *const).
 *
 * Does not touch memory if the corresponding mask element is false, so may
 * be used to prevent page faults where a non-masked load would cause a
 * fault.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_load(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const T *const p)
{
  return internal::mask::mask_load(src, k, p);
}

/**
 * @brief Zero-masked version of load(const T *const).
 *
 * Does not touch memory if the corresponding mask element is false, so may
 * be used to prevent page faults where a non-masked load would cause a
 * fault.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_load(const Mask<T, SIMD_WIDTH> &k,
                                                 const T *const p)
{
  return internal::mask::maskz_load(k, p);
}

/**
 * @brief Masked version of loadu(const T *const).
 *
 * Does not touch memory if the corresponding mask element is false, so may
 * be used to prevent page faults where a non-masked load would cause a
 * fault.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_loadu(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const T *const p)
{
  return internal::mask::mask_loadu(src, k, p);
}

/**
 * @brief Zero-masked version of loadu(const T *const).
 *
 * Does not touch memory if the corresponding mask element is false, so may
 * be used to prevent page faults where a non-masked load would cause a
 * fault.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_loadu(const Mask<T, SIMD_WIDTH> &k,
                                                  const T *const p)
{
  return internal::mask::maskz_loadu(k, p);
}

/**
 * @brief Masked version of store(T *const, const Vec<T, SIMD_WIDTH> &).
 *
 * Only stores the elements for which the corresponding mask element is true.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_store(T *const p, const Mask<T, SIMD_WIDTH> &k,
                                   const Vec<T, SIMD_WIDTH> &a)
{
  internal::mask::mask_store(p, k, a);
}

/**
 * @brief Masked version of storeu(T *const, const Vec<T, SIMD_WIDTH> &).
 *
 * Only stores the elements for which the corresponding mask element is true.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_storeu(T *const p, const Mask<T, SIMD_WIDTH> &k,
                                    const Vec<T, SIMD_WIDTH> &a)
{
  internal::mask::mask_storeu(p, k, a);
}

/** @} */

/**
 * @addtogroup group_mask_arithmetic
 * @{
 */

/**
 * @brief Masked version of
 * add(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_add(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_add(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * add(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_add(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_add(k, a, b);
}

/**
 * @brief Masked version of
 * adds(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_adds(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_adds(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * adds(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_adds(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a,
                                                 const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_adds(k, a, b);
}

/**
 * @brief Masked version of
 * sub(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_sub(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_sub(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * sub(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_sub(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_sub(k, a, b);
}

/**
 * @brief Masked version of
 * subs(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_subs(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_subs(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * subs(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_subs(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a,
                                                 const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_subs(k, a, b);
}

/**
 * @brief Masked version of
 * mul(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_mul(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_mul(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * mul(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_mul(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_mul(k, a, b);
}

/**
 * @brief Masked version of
 * div(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_div(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_div(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * div(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_div(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_div(k, a, b);
}

/**
 * @brief Masked version of
 * avg(const Vec<T, SIMD_WIDTH> &a, const Vec<T, SIMD_WIDTH> &b).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_avg(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_avg(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * avg(const Vec<T, SIMD_WIDTH> &a, const Vec<T, SIMD_WIDTH> &b).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_avg(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_avg(k, a, b);
}

/**
 * @brief Masked version of
 * div2r0(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_div2r0(const Vec<T, SIMD_WIDTH> &src,
                                                  const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_div2r0(src, k, a);
}

/**
 * @brief Zero-masked version of
 * div2r0(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_div2r0(const Mask<T, SIMD_WIDTH> &k,
                                                   const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_div2r0(k, a);
}

/**
 * @brief Masked version of
 * div2rd(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_div2rd(const Vec<T, SIMD_WIDTH> &src,
                                                  const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_div2rd(src, k, a);
}

/**
 * @brief Zero-masked version of
 * div2rd(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_div2rd(const Mask<T, SIMD_WIDTH> &k,
                                                   const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_div2rd(k, a);
}

/** @} */

/**
 * @addtogroup group_mask_horizontal
 * @{
 */

/**
 * @brief Masked version of
 * hadd(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hadd(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hadd(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hadd(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hadd(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a,
                                                 const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hadd(k, a, b);
}

/**
 * @brief Masked version of
 * hadds(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hadds(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a,
                                                 const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hadds(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hadds(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hadds(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hadds(k, a, b);
}

/**
 * @brief Masked version of
 * hsub(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hsub(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hsub(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hsub(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hsub(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a,
                                                 const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hsub(k, a, b);
}

/**
 * @brief Masked version of
 * hsubs(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hsubs(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a,
                                                 const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hsubs(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hsubs(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hsubs(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hsubs(k, a, b);
}

/** @} */

/**
 * @addtogroup group_mask_math_functions
 * @{
 */

/**
 * @brief Masked version of
 * rcp(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_rcp(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_rcp(src, k, a);
}

/**
 * @brief Zero-masked version of
 * rcp(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_rcp(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_rcp(k, a);
}

/**
 * @brief Masked version of
 * rsqrt(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_rsqrt(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_rsqrt(src, k, a);
}

/**
 * @brief Zero-masked version of
 * rsqrt(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_rsqrt(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_rsqrt(k, a);
}

/**
 * @brief Masked version of
 * sqrt(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_sqrt(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_sqrt(src, k, a);
}

/**
 * @brief Zero-masked version of
 * sqrt(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_sqrt(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_sqrt(k, a);
}

/** @} */

/**
 * @addtogroup group_mask_special_math
 * @{
 */

/**
 * @brief Masked version of
 * min(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_min(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_min(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * min(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_min(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_min(k, a, b);
}

/**
 * @brief Masked version of
 * max(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_max(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_max(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * max(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_max(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_max(k, a, b);
}

/**
 * @brief Masked version of
 * neg(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_neg(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_neg(src, k, a);
}

/**
 * @brief Zero-masked version of
 * neg(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_neg(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_neg(k, a);
}

/**
 * @brief Masked version of
 * abs(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_abs(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_abs(src, k, a);
}

/**
 * @brief Zero-masked version of
 * abs(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_abs(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_abs(k, a);
}

/**
 * @brief Masked version of
 * ceil(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ceil(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_ceil(src, k, a);
}

/**
 * @brief Zero-masked version of
 * ceil(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_ceil(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_ceil(k, a);
}

/**
 * @brief Masked version of
 * floor(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_floor(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_floor(src, k, a);
}

/**
 * @brief Zero-masked version of
 * floor(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_floor(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_floor(k, a);
}

/**
 * @brief Masked version of
 * round(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_round(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_round(src, k, a);
}

/**
 * @brief Zero-masked version of
 * round(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_round(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_round(k, a);
}

/**
 * @brief Masked version of
 * truncate(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_truncate(
  const Vec<T, SIMD_WIDTH> &src, const Mask<T, SIMD_WIDTH> &k,
  const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_truncate(src, k, a);
}

/**
 * @brief Zero-masked version of
 * truncate(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_truncate(
  const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_truncate(k, a);
}

/** @} */

/**
 * @addtogroup group_mask_logic
 * @{
 */

/**
 * @brief Masked version of
 * and_(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_and(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_and(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * and_(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_and(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_and(k, a, b);
}

/**
 * @brief Masked version of
 * or_(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_or(const Vec<T, SIMD_WIDTH> &src,
                                              const Mask<T, SIMD_WIDTH> &k,
                                              const Vec<T, SIMD_WIDTH> &a,
                                              const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_or(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * or_(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_or(const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_or(k, a, b);
}

/**
 * @brief Masked version of
 * andnot(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_andnot(const Vec<T, SIMD_WIDTH> &src,
                                                  const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_andnot(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * andnot(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_andnot(const Mask<T, SIMD_WIDTH> &k,
                                                   const Vec<T, SIMD_WIDTH> &a,
                                                   const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_andnot(k, a, b);
}

/**
 * @brief Masked version of
 * xor_(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_xor(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a,
                                               const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_xor(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * xor_(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_xor(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_xor(k, a, b);
}

/**
 * @brief Masked version of
 * not_(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_not(const Vec<T, SIMD_WIDTH> &src,
                                               const Mask<T, SIMD_WIDTH> &k,
                                               const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_not(src, k, a);
}

/**
 * @brief Zero-masked version of
 * not_(const Vec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_not(const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_not(k, a);
}

/** @} */

/**
 * @addtogroup group_mask_shift
 * @{
 */

/**
 * @brief Masked version of
 * srai(const Vec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_srai(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_srai<IMM>(src, k, a);
}

/**
 * @brief Zero-masked version of
 * srai(const Vec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_srai(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_srai<IMM>(k, a);
}

/**
 * @brief Masked version of
 * srli(const Vec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_srli(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_srli<IMM>(src, k, a);
}

/**
 * @brief Zero-masked version of
 * srli(const Vec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_srli(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_srli<IMM>(k, a);
}

/**
 * @brief Masked version of
 * slli(const Vec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_slli(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_slli<IMM>(src, k, a);
}

/**
 * @brief Zero-masked version of
 * slli(const Vec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_slli(const Mask<T, SIMD_WIDTH> &k,
                                                 const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_slli<IMM>(k, a);
}

// TODO: add masked versions of sra, srl and sll

/** @} */

/**
 * @addtogroup group_mask_cmp
 * @{
 */

/**
 * @brief Masked comparison between corresponding elements of two Vec's
 * for less-than ( @c < ).
 *
 * Bits in the resulting Mask are zeroed out if the corresponding bit in
 * the given Mask is not set.
 *
 * @param k Mask to use for masking the results
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the masked results of the comparisons
 * @sa cmplt(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmple(const Vec<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmplt(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmplt(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two Vec's
 * for less-than-or-equal ( @c <= ).
 *
 * Bits in the resulting Mask are zeroed out if the corresponding bit in
 * the given Mask is not set.
 *
 * @param k Mask to use for masking the results
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the masked results of the comparisons
 * @sa cmple(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmplt(const Vec<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmple(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmple(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two Vec's
 * for equality ( @c == ).
 *
 * Bits in the resulting Mask are zeroed out if the corresponding bit in
 * the given Mask is not set.
 *
 * @param k Mask to use for masking the results
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the masked results of the comparisons
 * @sa cmpeq(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpeq(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpeq(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two Vec's
 * for greater-than ( @c > ).
 *
 * Bits in the resulting Mask are zeroed out if the corresponding bit in
 * the given Mask is not set.
 *
 * @param k Mask to use for masking the results
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the masked results of the comparisons
 * @sa cmpgt(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpge(const Vec<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpgt(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpgt(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two Vec's
 * for greater-than-or-equal ( @c >= ).
 *
 * Bits in the resulting Mask are zeroed out if the corresponding bit in
 * the given Mask is not set.
 *
 * @param k Mask to use for masking the results
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the masked results of the comparisons
 * @sa cmpge(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpgt(const Vec<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpge(const Mask<T, SIMD_WIDTH> &k,
                                                  const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpge(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two Vec's
 * for inequality ( @c != ).
 *
 * Bits in the resulting Mask are zeroed out if the corresponding bit in
 * the given Mask is not set.
 *
 * @param k Mask to use for masking the results
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the masked results of the comparisons
 * @sa cmpneq(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpneq(const Vec<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpneq(const Mask<T, SIMD_WIDTH> &k,
                                                   const Vec<T, SIMD_WIDTH> &a,
                                                   const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpneq(k, a, b);
}

/**
 * @brief Compares corresponding elements of two Vec's for less-than
 * ( @c < ).
 *
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the results of the comparisons
 * @sa cmplt(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmplt(const Mask<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmplt(const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmplt(a, b);
}

/**
 * @brief Compares corresponding elements of two Vec's for
 * less-than-or-equal ( @c <= ).
 *
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the results of the comparisons
 * @sa cmple(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmple(const Mask<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmple(const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmple(a, b);
}

/**
 * @brief Compares corresponding elements of two Vec's for equality
 * ( @c == ).
 *
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the results of the comparisons
 * @sa cmpeq(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpeq(const Mask<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpeq(const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpeq(a, b);
}

/**
 * @brief Compares corresponding elements of two Vec's for greater-than
 * ( @c > ).
 *
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the results of the comparisons
 * @sa cmpgt(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpgt(const Mask<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpgt(const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpgt(a, b);
}

/**
 * @brief Compares corresponding elements of two Vec's for
 * greater-than-or-equal ( @c >= ).
 *
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the results of the comparisons
 * @sa cmpge(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpge(const Mask<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpge(const Vec<T, SIMD_WIDTH> &a,
                                                  const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpge(a, b);
}

/**
 * @brief Compares corresponding elements of two Vec's for inequality
 * ( @c != ).
 *
 * @param a first Vec
 * @param b second Vec
 * @return Mask with the results of the comparisons
 * @sa cmpneq(const Vec<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 * @sa mask_cmpneq(const Mask<T, SIMD_WIDTH> &, const Vec<T,
 * SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpneq(const Vec<T, SIMD_WIDTH> &a,
                                                   const Vec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpneq(a, b);
}

/**
 * @brief Tests if all elements of an Vec are zero, while
 * ignoring elements where the corresponding bit in an Mask is
 * zero.
 *
 * @param k Mask to use for the test
 * @param a Vec to test
 * @return 1 if all elements (except those ignored by the mask) are zero, 0
 * otherwise
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_zeros(const Mask<T, SIMD_WIDTH> &k,
                                           const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_test_all_zeros(k, a);
}

/**
 * @brief Tests if all bits of all elements of an Vec are one,
 * while ignoring elements where the corresponding bit in an Mask
 * is zero.
 *
 * @param k Mask to use for the test
 * @param a Vec to test
 * @return 1 if all bits of all elements (except those ignored by the mask)
 * are one, 0 otherwise
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_ones(const Mask<T, SIMD_WIDTH> &k,
                                          const Vec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_test_all_ones(k, a);
}

/**
 * @brief Selects elements from two Vec's based on a condition
 * Mask.
 *
 * @param cond condition mask
 * @param trueVal Vec to select from if the condition is true
 * @param falseVal Vec to select from if the condition is false
 * @return Vec containing the selected elements
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ifelse(
  const Mask<T, SIMD_WIDTH> &cond, const Vec<T, SIMD_WIDTH> &trueVal,
  const Vec<T, SIMD_WIDTH> &falseVal)
{
  return internal::mask::mask_ifelse(cond, trueVal, falseVal);
}

/**
 * @brief Selects elements from a Vec and zero based on a
 * condition Mask.
 *
 * @param cond condition mask
 * @param trueVal Vec to select from if the condition is true
 * @return Vec containing the selected elements
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ifelsezero(
  const Mask<T, SIMD_WIDTH> &cond, const Vec<T, SIMD_WIDTH> &trueVal)
{
  return internal::mask::mask_ifelsezero(cond, trueVal);
}

/** @} */
} // namespace simd

#endif // _SIMD_VEC_MASK_H_
