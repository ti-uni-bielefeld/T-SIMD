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

namespace ns_simd {
/**
 * @defgroup mask Masked Functions
 * @{
 */

// ===========================================================================
// functions without a group
// ===========================================================================

/**
 * @brief Masked version of set1(const T).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_set1(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const T a)
{
  return internal::mask::mask_set1(src, k, a);
}

/**
 * @brief Zero-masked version of set1(const T).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_set1(
  const SIMDMask<T, SIMD_WIDTH> &k, const T a)
{
  return internal::mask::maskz_set1(k, a);
}

/**
 * @brief Selects elements from two SIMDVec's based on a condition
 * SIMDMask.
 *
 * @param cond The condition mask.
 * @param trueVal The SIMD vector to select from if the condition is true.
 * @param falseVal The SIMD vector to select from if the condition is false.
 * @return A SIMDVec containing the selected elements.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_ifelse(
  const SIMDMask<T, SIMD_WIDTH> &cond, const SIMDVec<T, SIMD_WIDTH> &trueVal,
  const SIMDVec<T, SIMD_WIDTH> &falseVal)
{
  return internal::mask::mask_ifelse(cond, trueVal, falseVal);
}

/**
 * @brief Selects elements from a SIMDVec and zero based on a
 * condition SIMDMask.
 *
 * @param cond The condition mask.
 * @param trueVal The SIMD vector to select from if the condition is true.
 * @return A SIMDVec containing the selected elements.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_ifelsezero(
  const SIMDMask<T, SIMD_WIDTH> &cond, const SIMDVec<T, SIMD_WIDTH> &trueVal)
{
  return internal::mask::mask_ifelsezero(cond, trueVal);
}

/**
 * @brief Masked version of cvts(const SIMDVec<Tin, SIMD_WIDTH> &).
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> mask_cvts(
  const SIMDVec<Tout, SIMD_WIDTH> &src, const SIMDMask<Tin, SIMD_WIDTH> &k,
  const SIMDVec<Tin, SIMD_WIDTH> &a)
{
  return internal::mask::mask_cvts(src, k, a);
}

/**
 * @brief Zero-masked version of cvts(const SIMDVec<Tin, SIMD_WIDTH> &).
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> maskz_cvts(
  const SIMDMask<Tin, SIMD_WIDTH> &k, const SIMDVec<Tin, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_cvts(k, a);
}

/**
 * @brief Masked version of
 * div2r0(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_div2r0(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_div2r0(src, k, a);
}

/**
 * @brief Zero-masked version of
 * div2r0(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_div2r0(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_div2r0(k, a);
}

/**
 * @brief Masked version of
 * div2rd(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_div2rd(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_div2rd(src, k, a);
}

/**
 * @brief Zero-masked version of
 * div2rd(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_div2rd(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_div2rd(k, a);
}

/**
 * @defgroup mask_functions Pure Mask Functions
 * @brief Functions that operate on masks themselves.
 * @{
 */

/**
 * @brief Reinterprets a SIMDMask of one type as a SIMDMask of another type.
 *
 * The size of the element type of the SIMDMask must be the same.
 *
 * @tparam Tout The element type of the reinterpreted SIMDMask.
 * @tparam Tin The element type of the SIMDMask to reinterpret.
 * @param a The SIMDMask to reinterpret.
 * @return The reinterpreted SIMDMask.
 */
template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<Tout, SIMD_WIDTH> reinterpret_mask(
  const SIMDMask<Tin, SIMD_WIDTH> &a)
{
  return internal::mask::reinterpret_mask<Tout, Tin, SIMD_WIDTH>(a);
}

/**
 * @brief Adds two SIMDMask's together as if they were integers.
 *
 * @param a The first SIMDMask.
 * @param b The second SIMDMask.
 * @return The sum of the two SIMDMask's.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kadd(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kadd(a, b);
}

/**
 * @name Bitwise functions
 * @{
 */

/**
 * @brief Computes the bitwise AND of two SIMDMask's.
 *
 * @param a The first SIMDMask.
 * @param b The second SIMDMask.
 * @return The bitwise AND of the two SIMDMask's.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kand(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kand(a, b);
}

/**
 * @brief Computes bitwise ANDNOT of two SIMDMask's.
 *
 * The result is computed as (not a) and b.
 *
 * @param a The first SIMDMask.
 * @param b The second SIMDMask.
 * @return The bitwise ANDNOT of the two SIMDMask's.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kandn(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kandn(a, b);
}

/**
 * @brief Computes the bitwise OR of two SIMDMask's.
 *
 * @param a The first SIMDMask.
 * @param b The second SIMDMask.
 * @return The bitwise OR of the two SIMDMask's.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kor(const SIMDMask<T, SIMD_WIDTH> &a,
                                               const SIMDMask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kor(a, b);
}

/**
 * @brief Computes the bitwise XOR of two SIMDMask's.
 *
 * @param a The first SIMDMask.
 * @param b The second SIMDMask.
 * @return The bitwise XOR of the two SIMDMask's.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kxor(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kxor(a, b);
}

/**
 * @brief Computes the bitwise XNOR of two SIMDMask's.
 *
 * @param a The first SIMDMask.
 * @param b The second SIMDMask.
 * @return The bitwise XNOR of the two SIMDMask's.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kxnor(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  return internal::mask::kxnor(a, b);
}

/**
 * @brief Computes the bitwise NOT of a SIMDMask.
 *
 * @param a The SIMDMask.
 * @return The bitwise NOT of the SIMDMask.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> knot(
  const SIMDMask<T, SIMD_WIDTH> &a)
{
  return internal::mask::knot(a);
}

/** @} */

/**
 * @name Shift functions
 * @{
 */

/**
 * @brief Shifts the bits of a SIMDMask to the right by a
 *        constant number of bits.
 *
 * @param a The SIMDMask.
 * @tparam COUNT The number of bits to shift.
 * @return The shifted SIMDMask.
 */
template <unsigned int COUNT, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftri(
  const SIMDMask<T, SIMD_WIDTH> &a)
{
  return internal::mask::kshiftri<COUNT>(a);
}

/**
 * @brief Shifts the bits of a SIMDMask to the left by a
 *        constant number of bits.
 *
 * @param a The SIMDMask.
 * @tparam COUNT The number of bits to shift.
 * @return The shifted SIMDMask.
 */
template <unsigned int COUNT, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftli(
  const SIMDMask<T, SIMD_WIDTH> &a)
{
  return internal::mask::kshiftli<COUNT>(a);
}

/**
 * @brief Shifts the bits of a SIMDMask to the right by a
 *        variable number of bits.
 *
 * @param a The SIMDMask.
 * @param count The number of bits to shift.
 * @return The shifted SIMDMask.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftri(
  const SIMDMask<T, SIMD_WIDTH> &a, const uint64_t count)
{
  return internal::mask::kshiftri(a, count);
}

/**
 * @brief Shifts the bits of a SIMDMask to the left by a
 *        variable number of bits.
 *
 * @param a The SIMDMask.
 * @param count The number of bits to shift.
 * @return The shifted SIMDMask.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftli(
  const SIMDMask<T, SIMD_WIDTH> &a, const uint64_t count)
{
  return internal::mask::kshiftli(a, count);
}

/** @} */

/**
 * @name Mask initialization functions
 * @{
 */

/**
 * @brief Creates a SIMDMask with all elements set to true.
 *
 * @return A SIMDMask with all elements set to true.
 */
template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_all_ones()
{
  return internal::mask::mask_all_ones(internal::OutputType<T>(),
                                       internal::Integer<SIMD_WIDTH>());
}

// 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
// mask_set_true/false_low/high.

/**
 * @brief Sets the lower @p x bits of a SIMDMask to true.
 *
 * The remaining bits are set to false.
 *
 * @param x The number of bits to set to true.
 * @return The SIMDMask with the lower @p x bits set to true.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_true_low(
  const unsigned int x)
{
  return internal::mask::mask_set_true_low(x, internal::OutputType<T>(),
                                           internal::Integer<SIMD_WIDTH>());
}

/**
 * @brief Sets the upper @p x bits of a SIMDMask to true.
 *
 * The remaining bits are set to false.
 *
 * @param x The number of bits to set to true.
 * @return The SIMDMask with the upper @p x bits set to true.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_true_high(
  const unsigned int x)
{
  return internal::mask::mask_set_true_high(x, internal::OutputType<T>(),
                                            internal::Integer<SIMD_WIDTH>());
}

/**
 * @brief Sets the lower @p x bits of a SIMDMask to false.
 *
 * The remaining bits are set to true.
 *
 * @param x The number of bits to set to false.
 * @return The SIMDMask with the lower @p x bits set to false.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_false_low(
  const unsigned int x)
{
  return internal::mask::mask_set_false_low(x, internal::OutputType<T>(),
                                            internal::Integer<SIMD_WIDTH>());
}

/**
 * @brief Sets the upper @p x bits of a SIMDMask to false.
 *
 * The remaining bits are set to true.
 *
 * @param x The number of bits to set to false.
 * @return The SIMDMask with the upper @p x bits set to false.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_false_high(
  const unsigned int x)
{
  return internal::mask::mask_set_false_high(x, internal::OutputType<T>(),
                                             internal::Integer<SIMD_WIDTH>());
}

/** @} */

/** @} */ // end of group mask_functions

/**
 * @defgroup mask_memory Masked Memory Functions
 * @brief Functions for masked loading and storing SIMDVec's from and to
 * memory.
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
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_load(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
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
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_load(
  const SIMDMask<T, SIMD_WIDTH> &k, const T *const p)
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
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_loadu(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
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
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_loadu(
  const SIMDMask<T, SIMD_WIDTH> &k, const T *const p)
{
  return internal::mask::maskz_loadu(k, p);
}

/**
 * @brief Masked version of store(T *const, const SIMDVec<T, SIMD_WIDTH> &).
 *
 * Only stores the elements for which the corresponding mask element is true.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_store(T *const p, const SIMDMask<T, SIMD_WIDTH> &k,
                                   const SIMDVec<T, SIMD_WIDTH> &a)
{
  internal::mask::mask_store(p, k, a);
}

/**
 * @brief Masked version of storeu(T *const, const SIMDVec<T, SIMD_WIDTH> &).
 *
 * Only stores the elements for which the corresponding mask element is true.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_storeu(T *const p,
                                    const SIMDMask<T, SIMD_WIDTH> &k,
                                    const SIMDVec<T, SIMD_WIDTH> &a)
{
  internal::mask::mask_storeu(p, k, a);
}

/** @} */ // end of group mask_memory

/**
 * @defgroup mask_arithmetic Masked Arithmetic Functions
 * @brief Masked arithmetic functions on SIMDVec's, like addition,
 * multiplication, etc.
 * @{
 */

/**
 * @brief Masked version of
 * add(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_add(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_add(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * add(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_add(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_add(k, a, b);
}

/**
 * @brief Masked version of
 * adds(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_adds(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_adds(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * adds(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_adds(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_adds(k, a, b);
}

/**
 * @brief Masked version of
 * sub(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_sub(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_sub(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * sub(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_sub(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_sub(k, a, b);
}

/**
 * @brief Masked version of
 * subs(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_subs(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_subs(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * subs(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_subs(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_subs(k, a, b);
}

/**
 * @brief Masked version of
 * mul(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_mul(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_mul(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * mul(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_mul(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_mul(k, a, b);
}

/**
 * @brief Masked version of
 * div(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_div(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_div(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * div(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_div(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_div(k, a, b);
}

/**
 * @brief Masked version of
 * avg(const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_avg(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_avg(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * avg(const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_avg(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_avg(k, a, b);
}

/**
 * @name Horizontal functions
 * @{
 */

/**
 * @brief Masked version of
 * hadd(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hadd(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hadd(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hadd(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hadd(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hadd(k, a, b);
}

/**
 * @brief Masked version of
 * hadds(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hadds(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hadds(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hadds(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hadds(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hadds(k, a, b);
}

/**
 * @brief Masked version of
 * hsub(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hsub(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hsub(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hsub(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hsub(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hsub(k, a, b);
}

/**
 * @brief Masked version of
 * hsubs(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hsubs(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_hsubs(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * hsubs(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hsubs(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_hsubs(k, a, b);
}

/** @} */

/** @} */ // end of group mask_arithmetic

/**
 * @defgroup mask_math_functions Masked Elementary Math Functions
 * @brief Masked elementary mathematical functions on SIMDVec's, such as sqrt,
 * rcp, etc.
 * @{
 */

/**
 * @brief Masked version of
 * rcp(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_rcp(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_rcp(src, k, a);
}

/**
 * @brief Zero-masked version of
 * rcp(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_rcp(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_rcp(k, a);
}

/**
 * @brief Masked version of
 * rsqrt(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_rsqrt(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_rsqrt(src, k, a);
}

/**
 * @brief Zero-masked version of
 * rsqrt(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_rsqrt(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_rsqrt(k, a);
}

/**
 * @brief Masked version of
 * sqrt(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_sqrt(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_sqrt(src, k, a);
}

/**
 * @brief Zero-masked version of
 * sqrt(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_sqrt(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_sqrt(k, a);
}

/** @} */ // end of group mask_math_functions

/**
 * @defgroup mask_special_math Masked Special Math Functions
 * @brief Masked special math functions on SIMDVec's, such as min, max,
 * rounding, etc.
 * @{
 */

/**
 * @brief Masked version of
 * min(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_min(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_min(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * min(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_min(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_min(k, a, b);
}

/**
 * @brief Masked version of
 * max(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_max(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_max(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * max(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_max(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_max(k, a, b);
}

/**
 * @brief Masked version of
 * neg(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_neg(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_neg(src, k, a);
}

/**
 * @brief Zero-masked version of
 * neg(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_neg(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_neg(k, a);
}

/**
 * @brief Masked version of
 * abs(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_abs(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_abs(src, k, a);
}

/**
 * @brief Zero-masked version of
 * abs(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_abs(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_abs(k, a);
}

/**
 * @brief Masked version of
 * ceil(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_ceil(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_ceil(src, k, a);
}

/**
 * @brief Zero-masked version of
 * ceil(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_ceil(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_ceil(k, a);
}

/**
 * @brief Masked version of
 * floor(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_floor(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_floor(src, k, a);
}

/**
 * @brief Zero-masked version of
 * floor(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_floor(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_floor(k, a);
}

/**
 * @brief Masked version of
 * round(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_round(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_round(src, k, a);
}

/**
 * @brief Zero-masked version of
 * round(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_round(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_round(k, a);
}

/**
 * @brief Masked version of
 * truncate(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_truncate(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_truncate(src, k, a);
}

/**
 * @brief Zero-masked version of
 * truncate(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_truncate(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_truncate(k, a);
}

/** @} */ // end of group mask_special_math

/**
 * @defgroup mask_logic Masked Logical Functions
 * @brief Masked logical functions on SIMDVec's.
 * @{
 */

/**
 * @brief Masked version of
 * and_(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_and(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_and(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * and_(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_and(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_and(k, a, b);
}

/**
 * @brief Masked version of
 * or_(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_or(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_or(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * or_(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_or(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_or(k, a, b);
}

/**
 * @brief Masked version of
 * andnot(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_andnot(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_andnot(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * andnot(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_andnot(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_andnot(k, a, b);
}

/**
 * @brief Masked version of
 * xor_(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_xor(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_xor(src, k, a, b);
}

/**
 * @brief Zero-masked version of
 * xor_(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_xor(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::maskz_xor(k, a, b);
}

/**
 * @brief Masked version of
 * not_(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_not(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_not(src, k, a);
}

/**
 * @brief Zero-masked version of
 * not_(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_not(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_not(k, a);
}

/** @} */ // end of group mask_logic

/**
 * @defgroup mask_shift Masked Bitwise Shift Functions
 * @brief Masked bitwise shift functions on SIMDVec's.
 * @{
 */

/**
 * @brief Masked version of
 * srai(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_srai(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_srai<IMM>(src, k, a);
}

/**
 * @brief Zero-masked version of
 * srai(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_srai(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_srai<IMM>(k, a);
}

/**
 * @brief Masked version of
 * srli(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_srli(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_srli<IMM>(src, k, a);
}

/**
 * @brief Zero-masked version of
 * srli(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_srli(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_srli<IMM>(k, a);
}

/**
 * @brief Masked version of
 * slli(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_slli(
  const SIMDVec<T, SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k,
  const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_slli<IMM>(src, k, a);
}

/**
 * @brief Zero-masked version of
 * slli(const SIMDVec<T, SIMD_WIDTH> &).
 */
template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_slli(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::maskz_slli<IMM>(k, a);
}

// TODO: add masked versions of sra, srl and sll

/** @} */ // end of group mask_shift

/**
 * @defgroup mask_cmp Masked Comparison Functions
 * @brief Functions for masked comparing elements of SIMDVec's.
 * @{
 */

/**
 * @brief Masked comparison between corresponding elements of two SIMDVec's
 * for less-than ( @c < ).
 *
 * Bits in the resulting SIMDMask are zeroed out if the corresponding bit in
 * the given SIMDMask is not set.
 *
 * @param k The SIMDMask to use for masking the results.
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the masked results of the comparisons.
 * @sa cmplt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmple(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmplt(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmplt(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two SIMDVec's
 * for less-than-or-equal ( @c <= ).
 *
 * Bits in the resulting SIMDMask are zeroed out if the corresponding bit in
 * the given SIMDMask is not set.
 *
 * @param k The SIMDMask to use for masking the results.
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the masked results of the comparisons.
 * @sa cmple(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmplt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmple(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmple(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two SIMDVec's
 * for equality ( @c == ).
 *
 * Bits in the resulting SIMDMask are zeroed out if the corresponding bit in
 * the given SIMDMask is not set.
 *
 * @param k The SIMDMask to use for masking the results.
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the masked results of the comparisons.
 * @sa cmpeq(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpeq(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpeq(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two SIMDVec's
 * for greater-than ( @c > ).
 *
 * Bits in the resulting SIMDMask are zeroed out if the corresponding bit in
 * the given SIMDMask is not set.
 *
 * @param k The SIMDMask to use for masking the results.
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the masked results of the comparisons.
 * @sa cmpgt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpge(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpgt(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpgt(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two SIMDVec's
 * for greater-than-or-equal ( @c >= ).
 *
 * Bits in the resulting SIMDMask are zeroed out if the corresponding bit in
 * the given SIMDMask is not set.
 *
 * @param k The SIMDMask to use for masking the results.
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the masked results of the comparisons.
 * @sa cmpge(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpgt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpge(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpge(k, a, b);
}

/**
 * @brief Masked comparison between corresponding elements of two SIMDVec's
 * for inequality ( @c != ).
 *
 * Bits in the resulting SIMDMask are zeroed out if the corresponding bit in
 * the given SIMDMask is not set.
 *
 * @param k The SIMDMask to use for masking the results.
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the masked results of the comparisons.
 * @sa cmpneq(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpneq(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpneq(
  const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH> &a,
  const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpneq(k, a, b);
}

/**
 * @brief Compares corresponding elements of two SIMDVec's for less-than
 * ( @c < ).
 *
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the results of the comparisons.
 * @sa cmplt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmplt(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmplt(
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmplt(a, b);
}

/**
 * @brief Compares corresponding elements of two SIMDVec's for
 * less-than-or-equal ( @c <= ).
 *
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the results of the comparisons.
 * @sa cmple(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmple(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmple(
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmple(a, b);
}

/**
 * @brief Compares corresponding elements of two SIMDVec's for equality
 * ( @c == ).
 *
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the results of the comparisons.
 * @sa cmpeq(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpeq(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpeq(
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpeq(a, b);
}

/**
 * @brief Compares corresponding elements of two SIMDVec's for greater-than
 * ( @c > ).
 *
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the results of the comparisons.
 * @sa cmpgt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpgt(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpgt(
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpgt(a, b);
}

/**
 * @brief Compares corresponding elements of two SIMDVec's for
 * greater-than-or-equal ( @c >= ).
 *
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the results of the comparisons.
 * @sa cmpge(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpge(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpge(
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpge(a, b);
}

/**
 * @brief Compares corresponding elements of two SIMDVec's for inequality
 * ( @c != ).
 *
 * @param a The first SIMDVec.
 * @param b The second SIMDVec.
 * @return A SIMDMask with the results of the comparisons.
 * @sa cmpneq(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 * @sa mask_cmpneq(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
 * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpneq(
  const SIMDVec<T, SIMD_WIDTH> &a, const SIMDVec<T, SIMD_WIDTH> &b)
{
  return internal::mask::mask_cmpneq(a, b);
}

/**
 * @brief Tests if all elements of an SIMDVec are zero, while
 * ignoring elements where the corresponding bit in an SIMDMask is
 * zero.
 *
 * @param k The SIMDMask to use for the test.
 * @param a The SIMDVec to test.
 * @return 1 if all elements (except those ignored by the mask) are zero,
 * 0 otherwise.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_zeros(const SIMDMask<T, SIMD_WIDTH> &k,
                                           const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_test_all_zeros(k, a);
}

/**
 * @brief Tests if all bits of all elements of an SIMDVec are one,
 * while ignoring elements where the corresponding bit in an SIMDMask
 * is zero.
 *
 * @param k The SIMDMask to use for the test.
 * @param a The SIMDVec to test.
 * @return 1 if all bits of all elements (except those ignored by the mask)
 * are one, 0 otherwise.
 */
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_ones(const SIMDMask<T, SIMD_WIDTH> &k,
                                          const SIMDVec<T, SIMD_WIDTH> &a)
{
  return internal::mask::mask_test_all_ones(k, a);
}

/** @} */ // end of group mask_cmp

/** @} */ // end of group mask
} // namespace ns_simd

#endif // _SIMD_VEC_MASK_H_
