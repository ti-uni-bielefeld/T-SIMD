// ===========================================================================
// 
// SIMDVecBase.H --
// base-level classes and functions
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

#ifndef _SIMD_VEC_BASE_H_
#define _SIMD_VEC_BASE_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <stdint.h>

#ifdef SIMDVEC_SANDBOX

// test templates
#include "SIMDVecBaseImplSandbox.H"

#else // SIMDVEC_SANDBOX

// architecture-dependent files:

// Intel
#ifdef SIMDVEC_INTEL_ENABLE
#include "SIMDVecBaseImplIntel16.H"
#include "SIMDVecBaseImplIntel32.H"
#include "SIMDVecBaseImplIntel64.H"
#endif // SIMDVEC_INTEL_ENABLE

// ARM NEON
#ifdef SIMDVEC_NEON_ENABLE
#include "SIMDVecBaseImplNEON16.H"
#endif // SIMDVEC_NEON_ENABLE

#endif // else SIMDVEC_SANDBOX

namespace ns_simd {
  /**
   * @defgroup base Base Functions
   * @brief Base-level functions on SIMDVec's.
   * @{
   */

  // ===========================================================================
  // functions without a group
  // ===========================================================================

  /**
   * @brief Reinterprets a given SIMDVec as a SIMDVec with a different element
   * type.
   *
   * @tparam Tout The element type of the resulting SIMDVec.
   * @tparam Tin The element type of the given SIMDVec.
   * @param a The SIMDVec to reinterpret.
   * @return The reinterpreted SIMDVec.
   */
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  reinterpret(const SIMDVec<Tin, SIMD_WIDTH> &a) {
    return internal::base::reinterpret(a, internal::OutputType<Tout>());
  }

  /**
   * @brief Returns a SIMDVec with all elements set to zero.
   * @return A SIMDVec with all elements set to zero.
   */
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setzero()
  {
    return internal::base::setzero(internal::OutputType<T>(),
                             internal::Integer<SIMD_WIDTH>());
  }

  /**
   * @brief Returns a SIMDVec with all elements set to the same value.
   * @param a The value to set all elements to.
   * @return A SIMDVec with all elements set to the same value.
   */
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  set1(T a)
  {
    return internal::base::set1(a, internal::Integer<SIMD_WIDTH>());
  }

  // 30. Jan 23 (Jonas Keller): added iota

  /**
   * @brief Creates a SIMDVec with sequentially increasing numbers, starting
   * with 0.
   *
   * The sequence starts from the lowest element of the SIMDVec, i.e. the lowest
   * element of the SIMDVec is set to 0, the next element to 1, and so on.
   *
   * @return A SIMDVec containing sequentially increasing numbers.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  iota()
  {
    return internal::base::iota(internal::OutputType<T>(),
                                internal::Integer<SIMD_WIDTH>());
  }

  /**
   * @brief Selects elements from two SIMDVec's based on a condition SIMDVec.
   *
   * The element type of the condition vector must have the same size as the
   * element type of the true and false value vectors.
   *
   * @param cond The condition SIMDVec. Each element must be either
   * all 1 bits or all 0 bits, representing true or false, respectively.
   * @param trueVal The SIMDVec to select from if the condition is true.
   * @param falseVal The SIMDVec to select from if the condition is false.
   * @return A SIMDVec containing the selected elements.
   */
  template <typename Tcond, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  ifelse(const SIMDVec<Tcond, SIMD_WIDTH> &cond,
	 const SIMDVec<T, SIMD_WIDTH> &trueVal,
	 const SIMDVec<T, SIMD_WIDTH> &falseVal)
  {
    SIMD_STATIC_ASSERT_MSG(
      sizeof(Tcond) == sizeof(T),
      "condition and value types must have the same size");
    return internal::base::ifelse(reinterpret<T>(cond), trueVal, falseVal);
  }

  // 27. Aug 22 (Jonas Keller): added msb2int

  /**
   * @brief Collects the most significant bit of each element of a SIMDVec into
   * an integer.
   *
   * @param a The SIMDVec.
   * @return An integer containing the most significant bit of each element
   * of the input SIMDVec padded with zeros to 64 bits.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::msb2int(a);
  }

  // 09. Oct 22 (Jonas Keller): added int2msb

  /**
   * @brief Sets the most significant bit of each element of a SIMDVec to the
   * corresponding bit of an integer.
   *
   * The bottom n bits of the input integer are used, where n is the number of
   * elements in the SIMDVec. All other bits of the input are ignored.
   *
   * All other bits of the output elements are set to zero.
   *
   * @param a The integer.
   * @return A SIMDVec with the most significant bit of each element set to the
   * corresponding bit of the input integer.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2msb(const uint64_t a)
  {
    return internal::base::int2msb(a, internal::OutputType<T>(),
                             internal::Integer<SIMD_WIDTH>());
  }

  // 09. Oct 22 (Jonas Keller): added int2bits

  /**
   * @brief Sets all bits of each element of a SIMDVec to the corresponding bit
   * of an integer.
   *
   * The bottom n bits of the input integer are used, where n is the number of
   * elements in the SIMDVec. All other bits of the input are ignored.
   *
   * @param a The integer.
   * @return A SIMDVec with all bits of each element set to the corresponding
   * bit of the input integer.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2bits(const uint64_t a)
  {
    return internal::base::int2bits(a, internal::OutputType<T>(),
                              internal::Integer<SIMD_WIDTH>());
  }

  /**
   * @defgroup memory Memory Functions
   * @brief Functions for loading and storing SIMDVec's from and to memory.
   * @{
   */

  /**
   * @name Load functions
   * @{
   */

  /**
   * @brief Loads a SIMDVec from aligned memory.
   *
   * The memory location must be aligned to the @p SIMD_WIDTH.
   *
   * @note The template argument order of @p T and @p SIMD_WIDTH is reversed so
   * that @p T can be omitted in instantiation.
   *
   * @param[in] p A pointer to the aligned memory location to load from.
   * @return A SIMDVec with the loaded values.
   */
  template <int SIMD_WIDTH_DEFAULT_NATIVE_FUNC, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  load(const T *const p)
  {
    return internal::base::load(p, internal::Integer<SIMD_WIDTH>());
  }

  /**
   * @brief Loads a SIMDVec from unaligned memory.
   *
   * In contrast to load(const T *const p), the memory location does not need to
   * be aligned to any boundary.
   *
   * @note The template argument order of @p T and @p SIMD_WIDTH is reversed so
   * that @p T can be omitted in instantiation.
   *
   * @param[in] p A pointer to the memory location to load from.
   * @return A SIMDVec with the loaded values.
   */
  template <int SIMD_WIDTH_DEFAULT_NATIVE_FUNC, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  loadu(const T *const p)
  {
    return internal::base::loadu(p, internal::Integer<SIMD_WIDTH>());
  }

  /** @} */

  /**
   * @name Store functions
   * @{
   */

  /**
   * @brief Stores a SIMDVec to aligned memory.
   *
   * The memory location must be aligned to the SIMD_WIDTH.
   *
   * @param[out] p A pointer to the aligned memory location to store to.
   * @param a The SIMDVec to store.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  store(T *const p,
	const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::base::store(p, a);
  }

  /**
   * @brief Stores a SIMDVec to unaligned memory.
   *
   * In contrast to store(), the memory location does not need to be aligned to
   * any boundary.
   *
   * @param[out] p A pointer to the memory location to store to.
   * @param a The SIMDVec to store.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  storeu(T *const p,
	 const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::base::storeu(p, a);
  }

  /**
   * @brief Stores a SIMDVec to aligned memory using a non-temporal memory
   * hint.
   *
   * This function uses the @c _mm*_stream_* intrinsics on Intel and regular
   * store intrinsics on NEON. A call to sfence() may be required in order for
   * other threads/processors to see the stored values. This function may
   * improve performance on some architectures compared to store().
   *
   * The memory location must be aligned to the @p SIMD_WIDTH.
   *
   * @param[out] p A pointer to the aligned memory location to store to.
   * @param a The SIMDVec to store.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  stream_store(T *const p,
	      const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::base::stream_store(p, a);
  }

  /** @} */

  /**
   * @name Fences
   * Memory fences.
   * @{
   */

  /**
   * @brief Load fence.
   *
   * Forces strong memory ordering (serialization) between load instructions
   * preceding this instruction and load instructions following this
   * instruction, ensuring the system completes all previous loads before
   * executing subsequent loads.
   */
  static SIMD_INLINE void
  lfence()
  {
    internal::base::lfence();
  }

  /**
   * @brief Store fence.
   *
   * Forces strong memory ordering (serialization) between store instructions
   * preceding this instruction and store instructions following this
   * instruction, ensuring the system completes all previous stores before
   * executing subsequent stores.
   */
  static SIMD_INLINE void
  sfence()
  {
    internal::base::sfence();
  }

  /**
   * @brief Full memory fence.
   *
   * Forces strong memory ordering (serialization) between load and store
   * instructions preceding this instruction and load and store instructions
   * following this instruction, ensuring that the system completes all
   * previous memory accesses before executing subsequent memory accesses.
   */
  static SIMD_INLINE void
  mfence()
  {
    internal::base::mfence();
  }

  /** @} */

  /** @} */ // end of group memory

  /**
   * @defgroup arithmetic Arithmetic Functions
   * @brief Arithmetic functions on SIMDVec's, such as addition, multiplication,
   * etc.
   * @{
   */

  /**
   * @brief Adds the elements of two SIMDVec's.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the sums of the elements of the two
   * input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  add(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::add(a, b);
  }

  /**
   * @brief Adds the elements of two SIMDVec's using saturated
   * arithmetic.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the saturated sums of the elements of
   * the two input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  adds(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::adds(a, b);
  }

  /**
   * @brief Subtracts the elements of two SIMDVec's.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the differences of the elements of the
   * two input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sub(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::sub(a, b);
  }

  /**
   * @brief Subtracts the elements of two SIMDVec's using saturated
   * arithmetic.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the saturated differences of the
   * elements of the two input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  subs(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::subs(a, b);
  }

  /**
   * @brief Multiplies the elements of two SIMDVec's.
   *
   * @note This function is only available for floating point types.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the products of the elements of the
   * two input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mul(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isFloatingPoint,
                           "mul() is only available for floating point types");
    return internal::base::mul(a, b);
  }

  /**
   * @brief Divides the elements of two SIMDVec's.
   *
   * @note This function is only available for floating point types.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the quotients of the elements of the
   * two input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  div(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isFloatingPoint,
                           "div() is only available for floating point types");
    return internal::base::div(a, b);
  }

  /**
   * @brief Computes the average of the elements of two SIMDVec's, rounded up.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the rounded up average of the elements of
   * the two input SIMDVec's.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  avg(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::avg(a, b);
  }

  /**
   * @name Horizontal functions
   * @{
   */

  /**
   * @brief Horizontally adds adjacent elements of two SIMDVec's.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the results of the horizontal
   * additions.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hadd(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hadd(a, b);
  }

  /**
   * @brief Horizontally adds adjacent elements of two SIMDVec's
   * with saturation.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the results of the horizontal
   * saturated additions.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hadds(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hadds(a, b);
  }

  /**
   * @brief Horizontally subtracts adjacent elements of two SIMDVec's.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the results of the horizontal subtractions.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hsub(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hsub(a, b);
  }

  /**
   * @brief Horizontally subtracts adjacent elements of two SIMDVec's
   * with saturation.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the results of the horizontal saturated
   * subtractions.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hsubs(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hsubs(a, b);
  }

  /** @} */

  /** @} */ // end of group arithmetic

  /**
   * @defgroup math_functions Elementary Math Functions
   * @brief Elementary mathematical functions on SIMDVec's, such as sqrt, rcp,
   * etc.
   * @{
   */

  /**
   * @brief Computes the approximate reciprocal of the elements of a
   * SIMDVec.
   *
   * This function is only available for floating point types.
   *
   * @param a The SIMDVec to compute the reciprocal of.
   * @return A SIMDVec containing the approximate reciprocal of the
   * elements of the input vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  rcp(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isFloatingPoint,
                           "rcp() is only available for floating point types");
    return internal::base::rcp(a);
  }

  /**
   * @brief Computes the approximate reciprocal square root of the elements of a
   * SIMDVec.
   *
   * This function is only available for floating point types.
   *
   * @param a The SIMDVec to compute the reciprocal square root of.
   * @return A SIMDVec containing the approximate reciprocal square
   * root of the elements of the input vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  rsqrt(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isFloatingPoint,
                           "rsqrt() is only available for floating point types");
    return internal::base::rsqrt(a);
  }

  /**
   * @brief Computes the square root of the elements of a SIMDVec.
   *
   * This function is only available for floating point types.
   *
   * @note This function may only compute an approximation on some
   * architectures.
   *
   * @param a The SIMDVec to compute the square root of.
   * @return A SIMDVec containing the square root of the elements of
   * the input vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sqrt(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isFloatingPoint,
                           "sqrt() is only available for floating point types");
    return internal::base::sqrt(a);
  }

  /** @} */ // end of group math_functions

  /**
   * @defgroup special_math Special Math Functions
   * @brief Special math functions on SIMDVec's, such as min, max, rounding,
   * etc.
   * @{
   */

  /**
   * @brief Computes the minimum of the elements of two SIMDVec's.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the minimum of the elements of the
   * input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  min(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::min(a, b);
  }

  /**
   * @brief Computes the maximum of the elements of two SIMDVec's.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the maximum of the elements of the
   * input vectors.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  max(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::max(a, b);
  }

  /**
   * @brief Negates the elements of a SIMDVec.
   *
   * This function is only available for signed types.
   *
   * @param a The SIMDVec to negate.
   * @return A SIMDVec containing the negated elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  neg(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isSigned,
                           "neg() is only available for signed types");
    return internal::base::neg(a);
  }

  /**
   * @brief Computes the absolute value of the elements of a SIMDVec.
   *
   * This function is only available for signed types.
   *
   * @param a The SIMDVec to compute the absolute value of.
   * @return A SIMDVec containing the absolute value of the elements
   * of the input vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  abs(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isSigned,
                           "abs() is only available for signed types");
    return internal::base::abs(a);
  }

  /**
   * @brief Rounds the elements of a SIMDVec up to the nearest integer.
   *
   * This function is only available for floating point types.
   *
   * @param a The SIMDVec to round.
   * @return A SIMDVec containing the rounded elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  ceil(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(SIMDTypeInfo<T>::isFloatingPoint,
                           "ceil() is only available for floating point types");
    return internal::base::ceil(a);
  }

  /**
   * @brief Rounds the elements of a SIMDVec down to the nearest
   * integer.
   *
   * This function is only available for floating point types.
   *
   * @param a The SIMDVec to round.
   * @return A SIMDVec containing the rounded elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  floor(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(
      SIMDTypeInfo<T>::isFloatingPoint,
      "floor() is only available for floating point types");
    return internal::base::floor(a);
  }

  /**
   * @brief Rounds the elements of a SIMDVec to the nearest integer.
   *
   * This function is only available for floating point types.
   *
   * @param a The SIMDVec to round.
   * @return A SIMDVec containing the rounded elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  round(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(
      SIMDTypeInfo<T>::isFloatingPoint,
      "round() is only available for floating point types");
    return internal::base::round(a);
  }

  /**
   * @brief Truncates the elements of a SIMDVec to the nearest integer
   * i.e. rounds towards zero.
   *
   * This function is only available for floating point types.
   *
   * @param a The SIMDVec to truncate.
   * @return A SIMDVec containing the truncated elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  truncate(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(
      SIMDTypeInfo<T>::isFloatingPoint,
      "truncate() is only available for floating point types");
    return internal::base::truncate(a);
  }

  /** @} */ // end of group special_math

  /**
   * @defgroup logic Logical Functions
   * @brief Logical functions on SIMDVec's.
   * @{
   */

  /**
   * @brief Computes the bitwise AND of two SIMDVec's.
   *
   * @note This function is called @c "and_" instead of @c "and" because the
   * latter is a reserved word in C++.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the bitwise AND of the two
   * input SIMDVec's.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  and_(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::and_(a, b);
  }

  /**
   * @brief Computes the bitwise OR of two SIMDVec's.
   *
   * @note This function is called @c "or_" instead of @c "or" because the
   * latter is a reserved word in C++.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the bitwise OR of the two
   * input SIMDVec's.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  or_(const SIMDVec<T, SIMD_WIDTH> &a,
     const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::or_(a, b);
  }

  /**
   * @brief Computes the bitwise ANDNOT of two SIMDVec's.
   *
   * The result is computed as (not @p a ) and @p b .
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the bitwise ANDNOT of the two
   * input SIMDVec's.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  andnot(const SIMDVec<T, SIMD_WIDTH> &a,
	 const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::andnot(a, b);
  }

  /**
   * @brief Computes the bitwise XOR of two SIMDVec's.
   *
   * @note This function is called @c "xor_" instead of @c "xor" because the
   * latter is a reserved word in C++.
   *
   * @param a The first SIMDVec.
   * @param b The second SIMDVec.
   * @return A SIMDVec containing the bitwise XOR of the two
   * input SIMDVec's.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  xor_(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::xor_(a, b);
  }

  /**
   * @brief Computes the bitwise NOT of a SIMDVec.
   *
   * @note This function is called @c "not_" instead of @c "not" because the
   * latter is a reserved word in C++.
   *
   * @param a The SIMDVec.
   * @return A SIMDVec containing the bitwise NOT of the input
   * SIMDVec.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  not_(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::not_(a);
  }

  /** @} */ // end of group logic

  /**
   * @defgroup shift Bitwise Shift Functions
   * @brief Bitwise shift functions on SIMDVec's.
   * @{
   */

  /**
   * @brief Shifts the elements of a SIMDVec right by a constant
   * number of bits while shifting in the sign bit.
   *
   * @sa sra()
   *
   * @tparam COUNT The number of bits to shift by.
   * @param a The SIMDVec to shift.
   * @return A SIMDVec containing the shifted elements of the input
   * vector.
   */
  template <int COUNT, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srai(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::srai<COUNT>(a);
  }

  /**
   * @brief Shifts the elements of a SIMDVec right by a constant
   * number of bits while shifting in zeros.
   *
   * @sa srl()
   *
   * @tparam COUNT The number of bits to shift by.
   * @param a The SIMDVec to shift.
   * @return A SIMDVec containing the shifted elements of the input
   * vector.
   */
  template <int COUNT, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srli(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::srli<COUNT>(a);
  }

  /**
   * @brief Shifts the elements of a SIMDVec left by a constant
   * number of bits while shifting in zeros.
   *
   * @sa sll()
   *
   * @tparam COUNT The number of bits to shift by.
   * @param a The SIMDVec to shift.
   * @return A SIMDVec containing the shifted elements of the input
   * vector.
   */
  template <int COUNT, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  slli(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::slli<COUNT>(a);
  }

  // 12. Jan 23 (Jonas Keller): added sra, srl and sll functions

  /**
   * @brief Shifts the elements of a SIMDVec right by a variable
   * number of bits while shifting in the sign bit.
   *
   * @sa srai()
   *
   * @param a The SIMDVec to shift.
   * @param count The number of bits to shift by.
   * @return A SIMDVec containing the shifted elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sra(const SIMDVec<T, SIMD_WIDTH> &a, const uint8_t count)
  {
    return internal::base::sra(a, count);
  }

  /**
   * @brief Shifts the elements of a SIMDVec right by a variable
   * number of bits while shifting in zeros.
   *
   * @sa srli()
   *
   * @param a The SIMDVec to shift.
   * @param count The number of bits to shift by.
   * @return A SIMDVec containing the shifted elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srl(const SIMDVec<T, SIMD_WIDTH> &a, const uint8_t count)
  {
    return internal::base::srl(a, count);
  }

  /**
   * @brief Shifts the elements of a SIMDVec left by a variable
   * number of bits while shifting in zeros.
   *
   * @sa slli()
   *
   * @param a The SIMDVec to shift.
   * @param count The number of bits to shift by.
   * @return A SIMDVec containing the shifted elements of the input
   * vector.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sll(const SIMDVec<T, SIMD_WIDTH> &a, const uint8_t count)
  {
    return internal::base::sll(a, count);
  }

  /** @} */ // end of group shift

  /**
   * @defgroup cmp Comparison Functions
   * @brief Functions for comparing elements of SIMDVec's.
   * @{
   */

  /**
   * @brief Compares corresponding elements of two SIMDVec's for less-than
   * ( @c < ).
   *
   * @param a The first SIMDVec to compare.
   * @param b The second SIMDVec to compare.
   * @return A SIMDVec containing the results of the comparison,
   * where each element is all 1 bits or all 0 bits, depending on whether
   * the corresponding comparison returned true or false, respectively.
   * @sa mask_cmplt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &)
   * @sa mask_cmplt(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmplt(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmplt(a, b);
  }

  /**
   * @brief Compares corresponding elements of two SIMDVec's for
   * less-than-or-equal ( @c <= ).
   *
   * @param a The first SIMDVec to compare.
   * @param b The second SIMDVec to compare.
   * @return A SIMDVec containing the results of the comparison,
   * where each element is all 1 bits or all 0 bits, depending on whether
   * the corresponding comparison returned true or false, respectively.
   * @sa mask_cmple(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &)
   * @sa mask_cmple(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmple(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmple(a, b);
  }

  /**
   * @brief Compares corresponding elements of two SIMDVec's for
   * equality ( @c == ).
   *
   * @param a The first SIMDVec to compare.
   * @param b The second SIMDVec to compare.
   * @return A SIMDVec containing the results of the comparison,
   * where each element is all 1 bits or all 0 bits, depending on whether
   * the corresponding comparison returned true or false, respectively.
   * @sa mask_cmpeq(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &)
   * @sa mask_cmpeq(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpeq(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpeq(a, b);
  }

  /**
   * @brief Compares corresponding elements of two SIMDVec's for
   * greater-than-or-equal ( @c >= ).
   *
   * @param a The first SIMDVec to compare.
   * @param b The second SIMDVec to compare.
   * @return A SIMDVec containing the results of the comparison,
   * where each element is all 1 bits or all 0 bits, depending on whether
   * the corresponding comparison returned true or false, respectively.
   * @sa mask_cmpge(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &)
   * @sa mask_cmpge(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpge(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpge(a, b);
  }

  /**
   * @brief Compares corresponding elements of two SIMDVec's for
   * greater-than ( @c > ).
   *
   * @param a The first SIMDVec to compare.
   * @param b The second SIMDVec to compare.
   * @return A SIMDVec containing the results of the comparison,
   * where each element is all 1 bits or all 0 bits, depending on whether
   * the corresponding comparison returned true or false, respectively.
   * @sa mask_cmpgt(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &)
   * @sa mask_cmpgt(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpgt(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpgt(a, b);
  }

  /**
   * @brief Compares corresponding elements of two SIMDVec's for
   * inequality ( @c != ).
   *
   * @param a The first SIMDVec to compare.
   * @param b The second SIMDVec to compare.
   * @return A SIMDVec containing the results of the comparison,
   * where each element is all 1 bits or all 0 bits, depending on whether
   * the corresponding comparison returned true or false, respectively.
   * @sa mask_cmpne(const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &)
   * @sa mask_cmpne(const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T,
   * SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpneq(const SIMDVec<T, SIMD_WIDTH> &a,
	 const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpneq(a, b);
  }

  /**
   * @brief Tests if all bits of a SIMDVec are zero.
   *
   * @param a The SIMDVec to test.
   * @return 1 if all bits are zero, 0 otherwise.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::test_all_zeros(a);
  }

  /**
   * @brief Tests if all bits of a SIMDVec are one.
   *
   * @param a The SIMDVec to test.
   * @return 1 if all bits are one, 0 otherwise.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::test_all_ones(a);
  }

  /** @} */ // end of group cmp

  /**
   * @defgroup permutation Permutation Functions
   * @brief Functions on SIMDVec's for reordering elements and other related
   * functions.
   * @{
   */

  /**
   * @brief Extracts a single value from a SIMDVec.
   *
   * @tparam INDEX The index of the value to extract.
   * @param a The SIMDVec to extract from.
   * @return The extracted value.
   */
  template <int INDEX, typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  extract(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::extract<INDEX>(a);
  }

  /**
   * @brief Extracts the lowest element of a SIMDVec.
   *
   * Semantically equivalent to @c extract<0>(a) , but may be more efficient.
   *
   * @sa extract()
   *
   * @param a The SIMDVec to extract from.
   * @return The lowest element of the SIMDVec.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  elem0(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::elem0(a);
  }

  /**
   * @brief Extracts a 16-byte lane from a SIMDVec as a SIMDVec < T, 16 >.
   *
   * @tparam LANE_INDEX The lane to extract. Must be in the range
   * [0, @p SIMD_WIDTH / 16).
   * @param a The SIMDVec to extract a lane from.
   * @return The extracted lane as a SIMDVec < T, 16 >.
   */
  template <int LANE_INDEX, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, 16>
  extractLane(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::extractLane<LANE_INDEX>(a);
  }

  /**
   * @brief Reverses the order of the elements of a SIMDVec.
   *
   * @param a The SIMDVec to reverse.
   * @return A SIMDVec containing the elements of the input
   * SIMDVec in reverse order.
   */
  template <typename T, int SIMD_WIDTH>
  static SIMDVec<T, SIMD_WIDTH>
  reverse(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::reverse(a);
  }

  // swizzle (AoS to SoA), in-place, with template parameter N
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    internal::base::swizzle(v, internal::Integer<N>(),
                            internal::TypeIsIntSize<T>());
  }

  /**
   * @brief Interleaves blocks of elements from the high or low half of two
   * SIMDVec's.
   *
   * This function interleaves blocks of elements from the high or low
   * half of two SIMDVec's, starting with the lowest block of the
   * selected half of the first input SIMDVec.
   *
   * To get both halves of the input vectors interleaved, use zip().
   *
   * Example: TODO?
   *
   * @tparam PART Selects which half of the input vectors to use. A value
   * of 0 selects the low half, a value of 1 selects the high half.
   * @tparam NUM_ELEMS The number of elements in a block. This must be a
   * power of two (including 1) and be at most half of one SIMDVec.
   * @param a The first input SIMDVec.
   * @param b The second input SIMDVec.
   * @return The interleaved SIMDVec.
   */
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  unpack(const SIMDVec<T, SIMD_WIDTH> &a,
	 const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::unpack(a, b, internal::Part<PART>(),
                            internal::Bytes<NUM_ELEMS * sizeof(T)>());
  }

  /**
   * @brief Interleaves blocks of elements from the high or low half of each
   * 16-byte lane of two SIMDVec's.
   *
   * This function interleaves blocks of elements from the high or low
   * half of each 16-byte lane of two SIMDVec's, starting with the
   * lowest block of the selected half of the first input SIMDVec.
   *
   * This function is the lane-oriented equivalent of unpack().
   *
   * To get both halves of the input vectors interleaved, use zip16().
   *
   * Example: TODO?
   *
   * @tparam PART Selects which half of the lanes of the input vectors to use.
   * A value of 0 selects the low half, a value of 1 selects the high half.
   * @tparam NUM_ELEMS The number of elements in a block. This must be a
   * power of two (including 1) and be at most half of one SIMDVec.
   * @param a The first input SIMDVec.
   * @param b The second input SIMDVec.
   * @return The interleaved SIMDVec.
   */
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  unpack16(const SIMDVec<T, SIMD_WIDTH> &a,
     const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::unpack16<PART, NUM_ELEMS>(a, b);
  }

  /**
   * @brief Interleaves blocks of elements of two SIMDVec's.
   *
   * This function interleaves blocks of elements from two
   * SIMDVec's, starting with the lowest block of the first input
   * SIMDVec. The interleaved blocks are returned in two output
   * SIMDVec's.
   *
   * This function is semantically equivalent to calling unpack() twice for
   * both parts, but may be more efficient on some platforms (such as arm
   * NEON).
   *
   * Example: TODO?
   *
   * @sa unpack()
   * @sa unzip()
   *
   * @tparam NUM_ELEMS The number of elements in a block. This must be a
   * power of two (including 1) and be at most half of one SIMDVec.
   * @param a The first input SIMDVec.
   * @param b The second input SIMDVec.
   * @param[out] l The output SIMDVec containing the low half of the
   * interleaved blocks.
   * @param[out] h The output SIMDVec containing the high half of the
   * interleaved blocks.
   */
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  zip(const SIMDVec<T, SIMD_WIDTH> a,
      const SIMDVec<T, SIMD_WIDTH> b,
      SIMDVec<T, SIMD_WIDTH> &l,
      SIMDVec<T, SIMD_WIDTH> &h)
  {
    return internal::base::zip<NUM_ELEMS>(a, b, l, h);
  }

  /**
   * @brief Interleaves blocks of elements of each 16-byte lane of two
   * SIMDVec's.
   *
   * This function interleaves blocks of elements from the high or low
   * half of each 16-byte lane of two SIMDVec's, starting with the
   * lowest block of the selected half of the first input SIMDVec. The
   * interleaved blocks are returned in two output SIMDVec's.
   *
   * This function is the lane-oriented equivalent of zip().
   *
   * This function is semantically equivalent to calling unpack16() twice for
   * both parts, but may be more efficient on some platforms (such as arm
   * NEON).
   *
   * Example: TODO?
   *
   * @sa unpack16()
   *
   * @tparam NUM_ELEMS The number of elements in a block. This must be a
   * power of two (including 1) and be at most half of one SIMDVec.
   * @param a The first input SIMDVec.
   * @param b The second input SIMDVec.
   * @param[out] l The output SIMDVec containing the low half of the
   * interleaved blocks.
   * @param[out] h The output SIMDVec containing the high half of the
   * interleaved blocks.
   */
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  zip16(const SIMDVec<T, SIMD_WIDTH> a,
      const SIMDVec<T, SIMD_WIDTH> b,
      SIMDVec<T, SIMD_WIDTH> &l,
      SIMDVec<T, SIMD_WIDTH> &h)
  {
    return internal::base::zip16<NUM_ELEMS>(a, b, l, h);
  }

  /**
   * @brief Deinterleaves blocks of elements two SIMDVec's.
   *
   * This function deinterleaves blocks of elements from two
   * SIMDVec's. This is the inverse of zip().
   *
   * Example: TODO?
   *
   * @sa zip()
   *
   * @tparam NUM_ELEMS The number of elements in a block. This must be a
   * power of two (including 1) and be at most half of one SIMDVec.
   * @param a The first input SIMDVec.
   * @param b The second input SIMDVec.
   * @param[out] l The output SIMDVec containing the low half of the
   * deinterleaved blocks.
   * @param[out] h The output SIMDVec containing the high half of the
   * deinterleaved blocks.
   */
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, SIMD_WIDTH> a,
      const SIMDVec<T, SIMD_WIDTH> b,
      SIMDVec<T, SIMD_WIDTH> &l,
      SIMDVec<T, SIMD_WIDTH> &h)
  {
    return internal::base::unzip(a, b, l, h,
                           internal::Bytes<NUM_ELEMS * sizeof(T)>());
  }

  /**
   * @brief Shifts a SIMDVec right by a constant number of elements,
   * shifting in zero elements.
   *
   * @tparam COUNT The number of elements to shift by.
   * @param a The SIMDVec to shift.
   * @return The shifted SIMDVec.
   */
  template <int COUNT, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srle(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::srle<COUNT>(a);
  }

  /**
   * @brief Shifts a SIMDVec left by a constant number of elements,
   * shifting in zero elements.
   *
   * @tparam COUNT The number of elements to shift by.
   * @param a The SIMDVec to shift.
   * @return The shifted SIMDVec.
   */
  template <int COUNT, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  slle(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::slle<COUNT>(a);
  }

  /**
   * @brief Concatenates two SIMDVec's, shifts the result right by a
   * constant number of elements, and returns the low half of the
   * result.
   *
   * @tparam COUNT The number of elements to shift by.
   * @param h The first SIMDVec. The high half of the concatenated SIMDVec.
   * @param l The second SIMDVec. The low half of the concatenated SIMDVec.
   * @return The low half of the shifted concatenated SIMDVec.
   */
  template <int COUNT, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  alignre(const SIMDVec<T, SIMD_WIDTH> &h,
	  const SIMDVec<T, SIMD_WIDTH> &l)
  {
    return internal::base::alignre<COUNT>(h, l);
  }

  /** @} */ // end of group permutation

  /**
   * @defgroup type_conversion Type Conversion Functions
   * @brief Functions for converting SIMDVec's between different types.
   * @{
   */

  /**
   * @brief Packs two SIMDVec's into one by converting the elements
   * into the next smaller type with saturation.
   *
   * Only supported for signed->signed and signed->unsigned
   * conversions. SIMDFloat is converted to an integer type, as there
   * is no 16-bit floating point type.
   *
   * @sa extend()
   *
   * @tparam Tout The type of the resulting SIMDVec. Must be the next
   * smaller type of the elements of the input SIMDVec's.
   * @tparam Tin The type of the input SIMDVec's.
   * @param a, b The input SIMDVec's.
   * @return A SIMDVec with the elements of a and b packed into one.
   */
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  packs(const SIMDVec<Tin, SIMD_WIDTH> &a,
	const SIMDVec<Tin, SIMD_WIDTH> &b)
  {
    SIMD_STATIC_ASSERT_MSG(
      (SIMDTypeInfo<Tin>::isSigned && SIMDTypeInfo<Tout>::isSigned) ||
        (SIMDTypeInfo<Tin>::isSigned && !SIMDTypeInfo<Tout>::isSigned),
      "packs() only supports signed->signed and signed->unsigned");
    return internal::base::packs(a, b, internal::OutputType<Tout>());
  }

  /**
   * @brief Extends the elements of a SIMDVec to a larger or equally
   * sized type.
   *
   * Supported are signed->signed, unsigned->unsigned and unsigned->signed
   * conversions, but unsigned->signed conversions are only supported
   * if the destination type is larger than the source type.
   *
   * The values are zero-extended or sign-extended, depending on the type.
   *
   * Multiple output vectors are produced, where the amount of output
   * vectors is <tt>sizeof(Tout) / sizeof(Tin)</tt>.
   *
   * @sa packs()
   *
   * @tparam Tout The type to extend to.
   * @tparam Tin The type to extend from.
   * @param vIn The input SIMDVec.
   * @param[out] vOut The output SIMDVec's.
   */
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  extend(const SIMDVec<Tin,SIMD_WIDTH> &vIn,
	 SIMDVec<Tout,SIMD_WIDTH> vOut[sizeof(Tout) / sizeof(Tin)])
  {
    SIMD_STATIC_ASSERT_MSG(
      (SIMDTypeInfo<Tin>::isSigned && SIMDTypeInfo<Tout>::isSigned) ||
        (!SIMDTypeInfo<Tin>::isSigned && !SIMDTypeInfo<Tout>::isSigned) ||
        (!SIMDTypeInfo<Tin>::isSigned && SIMDTypeInfo<Tout>::isSigned &&
         sizeof(Tout) > sizeof(Tin)),
      "extend() only supports signed->signed, unsigned->unsigned and "
      "unsigned->signed (with destination type larger than source type) "
      "conversions");
    return internal::base::extend(vIn, vOut);
  }

  /**
   * @brief Converts the elements of a SIMDVec between @ref SIMDInt
   * and @ref SIMDFloat types
   *
   * The conversion @ref SIMDFloat -> @ref SIMDInt is saturated.
   *
   * @tparam Tout The type of the resulting SIMDVec. Must be one of
   * SIMDInt or @ref SIMDFloat, but not the same as @p Tin.
   * @tparam Tin The type of the input SIMDVec. Must be one of
   * @ref SIMDInt or @ref SIMDFloat, but not the same as @p Tout.
   * @param a The input SIMDVec.
   */
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  cvts(const SIMDVec<Tin, SIMD_WIDTH> &a)
  {
    SIMD_STATIC_ASSERT_MSG(
      ((SIMD_TYPE_EQ(Tout, SIMDInt) && SIMD_TYPE_EQ(Tin, SIMDFloat)) ||
       (SIMD_TYPE_EQ(Tout, SIMDFloat) && SIMD_TYPE_EQ(Tin, SIMDInt))),
      "Tout and Tin must be one of SIMDInt or SIMDFloat, but not the same");
    return internal::base::cvts(a, internal::OutputType<Tout>());
  }

  /** @} */ // end of group type_conversion

  /** @} */ // end of group base
}

#endif // _SIMD_VEC_BASE_H_
