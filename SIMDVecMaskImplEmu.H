// ===========================================================================
//
// SIMDVecMaskImplEmu.H --
// emulated mask functions
// Author: Markus Vieth (Bielefeld University, mvieth@techfak.uni-bielefeld.de)
// Year of creation: 2019
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
// (C) Markus Vieth, Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

// 22. Jan 23 (Jonas Keller): moved internal implementations into internal
// namespace

// 01. Feb 23 (Jonas Keller): implemented the emulated mask functions in a more
// efficient way, described below and also optimized other small things.
// 30. Nov 22 (Jonas Keller):
// NOTE:
// The float versions of the emulated mask functions in this file as well as in
// SIMDVecMaskImplIntel64.H are not as fast as they could be, as they are
// implemented such that they match the not emulated ones in flag and exception
// behavior as well. This is done by masking the inputs of the masked functions,
// which for example leads to the following code for masked addition:
/*
template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH>
maskz_add(const Mask<Float, SIMD_WIDTH> &k,
          const Vec<Float, SIMD_WIDTH> &a,
          const Vec<Float, SIMD_WIDTH> &b)
{
  return add(mask_ifelse(k, a, setzero<Float, SIMD_WIDTH>()),
             mask_ifelse(k, b, setzero<Float, SIMD_WIDTH>()));
}
template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH>
mask_add(const Vec<Float, SIMD_WIDTH> &src,
         const Mask<Float, SIMD_WIDTH> &k,
         const Vec<Float, SIMD_WIDTH> &a,
         const Vec<Float, SIMD_WIDTH> &b)
{
  return mask_ifelse(k, maskz_add(k, a, b), src);
}
*/
// which calls mask_ifelse 3 times for one call of mask_add, instead of
/*
template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH>
maskz_add(const Mask<Float, SIMD_WIDTH> &k,
          const Vec<Float, SIMD_WIDTH> &a,
          const Vec<Float, SIMD_WIDTH> &b)
{
  return mask_ifelse(k, add(a, b), setzero<Float, SIMD_WIDTH>());
}
template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH>
mask_add(const Vec<Float, SIMD_WIDTH> &src,
         const Mask<Float, SIMD_WIDTH> &k,
         const Vec<Float, SIMD_WIDTH> &a,
         const Vec<Float, SIMD_WIDTH> &b)
{
  return mask_ifelse(k, add(a, b), k);
}
*/
// which calls mask_ifelse only once for one call of mask_add.
//
// The second version would however for example set the denormal flag if an
// input is denormalized, even if the corresponding mask bit is not set, which
// is different from the behavior of the not emulated mask functions.
//
// It may be worth considering to implement the emulated mask functions
// analogous to the second version to improve performance. This would
// change the flag/exception behavior of the emulated mask functions.
// However, the flag/exception behavior is probably not correct in the
// whole library anyway, and probably also different on ARM.
// Additionally, the T-SIMD does not provide an architecture independent
// way to use flags or exceptions, so emulating them does not make much
// sense anyway.

#pragma once
#ifndef SIMD_VEC_MASK_IMPL_EMU_H_
#define SIMD_VEC_MASK_IMPL_EMU_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"

#ifdef SIMDVEC_INTEL_ENABLE
#include "SIMDVecMaskImplIntel64.H"
#endif // SIMDVEC_INTEL_ENABLE

#include <cstdint>

namespace simd {
// 05. Feb 23 (Jonas Keller): introduced generic emulated Mask class using the
// Vec class

/// @cond
template <typename T, int SIMD_WIDTH>
class Mask
{
  Vec<T, SIMD_WIDTH> mask;

public:
  Mask() = default;
  explicit SIMD_INLINE Mask(const Vec<T, SIMD_WIDTH> &x) : mask(x) {}
  SIMD_INLINE Mask(const uint64_t x) : mask(int2bits<T, SIMD_WIDTH>(x)) {}
  explicit SIMD_INLINE operator Vec<T, SIMD_WIDTH>() const { return mask; }
  SIMD_INLINE operator uint64_t() const { return msb2int<T, SIMD_WIDTH>(mask); }
  SIMD_INLINE bool operator[](const int i) const
  {
    if (i < 0 || i >= SIMD_WIDTH) { return false; }
    uint8_t mask_array[SIMD_WIDTH] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
    store((T *) mask_array, mask);
    // check first byte of i-th element (the bytes of a single element should
    // be the same)
    return mask_array[i * sizeof(T)] != 0;
  }
  SIMD_INLINE bool operator==(const Mask<T, SIMD_WIDTH> &other) const
  {
    return test_all_zeros(
      bit_xor(reinterpret<Int>(mask), reinterpret<Int>(other.mask)));
  }
  // define operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  void *operator new(size_t size)
  {
    return simd_aligned_malloc(SIMD_WIDTH, size);
  }
  void operator delete(void *p) { simd_aligned_free(p); }
  void *operator new[](size_t size)
  {
    return simd_aligned_malloc(SIMD_WIDTH, size);
  }
  void operator delete[](void *p) { simd_aligned_free(p); }
};
/// @endcond

namespace internal {
namespace mask {
#define EMULATE_SOP_NAME(OP, OP_NAME)                                          \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_##OP_NAME(                       \
    const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a)                 \
  {                                                                            \
    return mask::mask_ifelsezero(k, OP(a));                                    \
  }                                                                            \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_##OP_NAME(                        \
    const Vec<T, SIMD_WIDTH> &src, const Mask<T, SIMD_WIDTH> &k,               \
    const Vec<T, SIMD_WIDTH> &a)                                               \
  {                                                                            \
    return mask::mask_ifelse(k, OP(a), src);                                   \
  }

#define EMULATE_SOP(OP) EMULATE_SOP_NAME(OP, OP)

#define EMULATE_DOP_NAME(OP, OP_NAME)                                          \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_##OP_NAME(                       \
    const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a,                 \
    const Vec<T, SIMD_WIDTH> &b)                                               \
  {                                                                            \
    return mask::mask_ifelsezero(k, OP(a, b));                                 \
  }                                                                            \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_##OP_NAME(                        \
    const Vec<T, SIMD_WIDTH> &src, const Mask<T, SIMD_WIDTH> &k,               \
    const Vec<T, SIMD_WIDTH> &a, const Vec<T, SIMD_WIDTH> &b)                  \
  {                                                                            \
    return mask::mask_ifelse(k, OP(a, b), src);                                \
  }

#define EMULATE_DOP(OP) EMULATE_DOP_NAME(OP, OP)

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ifelse(
  const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &trueVal,
  const Vec<T, SIMD_WIDTH> &falseVal)
{
  return ifelse((Vec<T, SIMD_WIDTH>) k, trueVal, falseVal);
}

// 04. Aug 22 (Jonas Keller): added mask_ifelsezero
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ifelsezero(
  const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &trueVal)
{
  return bit_and((Vec<T, SIMD_WIDTH>) k, trueVal);
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE Mask<Tout, SIMD_WIDTH> reinterpret_mask(
  const Mask<Tin, SIMD_WIDTH> &k)
{
  SIMD_STATIC_ASSERT(sizeof(Tout) == sizeof(Tin));
  return Mask<Tout, SIMD_WIDTH>(reinterpret<Tout>((Vec<Tin, SIMD_WIDTH>) k));
}

// The types of the masks are kind of arbitrary
template <int SIMD_WIDTH>
SIMD_INLINE Vec<Int, SIMD_WIDTH> maskz_cvts(const Mask<Float, SIMD_WIDTH> &k,
                                            const Vec<Float, SIMD_WIDTH> &a)
{
  return mask::mask_ifelsezero(mask::reinterpret_mask<Int>(k),
                               ::simd::cvts<Int>(a));
}

template <int SIMD_WIDTH>
SIMD_INLINE Vec<Int, SIMD_WIDTH> mask_cvts(const Vec<Int, SIMD_WIDTH> &src,
                                           const Mask<Float, SIMD_WIDTH> &k,
                                           const Vec<Float, SIMD_WIDTH> &a)
{
  return mask::mask_ifelse(mask::reinterpret_mask<Int>(k), ::simd::cvts<Int>(a),
                           src);
}

template <int SIMD_WIDTH>
SIMD_INLINE Vec<Float, SIMD_WIDTH> maskz_cvts(const Mask<Int, SIMD_WIDTH> &k,
                                              const Vec<Int, SIMD_WIDTH> &a)
{
  return mask::mask_ifelsezero(mask::reinterpret_mask<Float>(k),
                               ::simd::cvts<Float>(a));
}

template <int SIMD_WIDTH>
SIMD_INLINE Vec<Float, SIMD_WIDTH> mask_cvts(const Vec<Float, SIMD_WIDTH> &src,
                                             const Mask<Int, SIMD_WIDTH> &k,
                                             const Vec<Int, SIMD_WIDTH> &a)
{
  return mask::mask_ifelse(mask::reinterpret_mask<Float>(k),
                           ::simd::cvts<Float>(a), src);
}

// =======================================================================
// emulated load/store
// =======================================================================

// 04. Feb 23 (Jonas Keller): improved implementation of masked load/store
// functions

template <int SIMD_WIDTH, typename T>
static SIMD_INLINE bool is_within_same_page(const T *const p)
{
  const uintptr_t PAGE_SIZE = 4096; // smallest page size I found
  const uintptr_t begin_page =
    reinterpret_cast<uintptr_t>(p) & ~(PAGE_SIZE - 1);
  // 29. Aug 23 (Jonas Keller): fixed wrong calculation of end_page
  const uintptr_t end_page =
    // reinterpret_cast<uintptr_t>(p + Vec<T, SIMD_WIDTH>::elems - 1) &
    // ~(PAGE_SIZE - 1);
    (reinterpret_cast<uintptr_t>(p) + SIMD_WIDTH - 1) & ~(PAGE_SIZE - 1);
  return begin_page == end_page;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_load(const Mask<T, SIMD_WIDTH> &k,
                                                 const T *const p)
{
  // if k is all zeros nothing should be loaded
  if (test_all_zeros((Vec<T, SIMD_WIDTH>) k)) {
    return setzero<T, SIMD_WIDTH>();
  }
  // If p till p+Vec<T, SIMD_WIDTH>::elems-1 is within the same page,
  // there is no risk of a page fault, so we load the whole vector and mask
  // it. Otherwise, we load the vector element-wise.
  if (is_within_same_page<SIMD_WIDTH>(p)) {
    return mask::mask_ifelsezero(k, load<SIMD_WIDTH>(p));
  }
  if (test_all_ones((Vec<T, SIMD_WIDTH>) k)) {
    // if k is all ones, we can load the whole vector
    return load<SIMD_WIDTH>(p);
  }
  T result[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
    result[i] = k[i] ? p[i] : 0;
  }
  return load<SIMD_WIDTH>(result);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_load(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const T *const p)
{
  // if k is all zeros nothing should be loaded
  if (test_all_zeros((Vec<T, SIMD_WIDTH>) k)) { return src; }
  // If p till p+Vec<T, SIMD_WIDTH>::elems-1 is within the same page,
  // there is no risk of a page fault, so we load the whole vector and mask
  // it. Otherwise, we load the vector element-wise.
  if (is_within_same_page<SIMD_WIDTH>(p)) {
    return mask::mask_ifelse(k, load<SIMD_WIDTH>(p), src);
  }
  if (test_all_ones((Vec<T, SIMD_WIDTH>) k)) {
    // if k is all ones, we can load the whole vector
    return load<SIMD_WIDTH>(p);
  }
  T result[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  T src_arr[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  store(src_arr, src);
  for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
    result[i] = k[i] ? p[i] : src_arr[i];
  }
  return load<SIMD_WIDTH>(result);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_loadu(const Mask<T, SIMD_WIDTH> &k,
                                                  const T *const p)
{
  // if k is all zeros nothing should be loaded
  if (test_all_zeros((Vec<T, SIMD_WIDTH>) k)) {
    return setzero<T, SIMD_WIDTH>();
  }
  // If p till p+Vec<T, SIMD_WIDTH>::elems-1 is within the same page,
  // there is no risk of a page fault, so we load the whole vector and mask
  // it. Otherwise, we load the vector element-wise.
  if (is_within_same_page<SIMD_WIDTH>(p)) {
    return mask::mask_ifelsezero(k, loadu<SIMD_WIDTH>(p));
  }
  if (test_all_ones((Vec<T, SIMD_WIDTH>) k)) {
    // if k is all ones, we can load the whole vector
    return loadu<SIMD_WIDTH>(p);
  }
  T result[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
    result[i] = k[i] ? p[i] : 0;
  }
  return load<SIMD_WIDTH>(result);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_loadu(const Vec<T, SIMD_WIDTH> &src,
                                                 const Mask<T, SIMD_WIDTH> &k,
                                                 const T *const p)
{
  // if k is all zeros nothing should be loaded
  if (test_all_zeros((Vec<T, SIMD_WIDTH>) k)) { return src; }
  // If p till p+Vec<T, SIMD_WIDTH>::elems-1 is within the same page,
  // there is no risk of a page fault, so we load the whole vector and mask
  // it. Otherwise, we load the vector element-wise.
  if (is_within_same_page<SIMD_WIDTH>(p)) {
    return mask::mask_ifelse(k, loadu<SIMD_WIDTH>(p), src);
  }
  if (test_all_ones((Vec<T, SIMD_WIDTH>) k)) {
    // if k is all ones, we can load the whole vector
    return loadu<SIMD_WIDTH>(p);
  }
  T result[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  T src_arr[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  store(src_arr, src);
  for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
    result[i] = k[i] ? p[i] : src_arr[i];
  }
  return load<SIMD_WIDTH>(result);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_store(T *const p, const Mask<T, SIMD_WIDTH> &k,
                                   const Vec<T, SIMD_WIDTH> &a)
{
  // if k is all zeros nothing should be stored
  if (test_all_zeros((Vec<T, SIMD_WIDTH>) k)) { return; }
  // If p till p+Vec<T, SIMD_WIDTH>::elems-1 is within the same page,
  // there is no risk of a page fault, so we load the whole vector, mask it
  // and store it back. Otherwise, we store the vector element-wise.
  if (is_within_same_page<SIMD_WIDTH>(p)) {
    store(p, mask::mask_ifelse(k, a, load<SIMD_WIDTH>(p)));
    return;
  }
  if (test_all_ones((Vec<T, SIMD_WIDTH>) k)) {
    // if k is all ones, we can store the whole vector
    store(p, a);
    return;
  }
  T a_arr[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  store(a_arr, a);
  for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
    if (k[i]) { p[i] = a_arr[i]; }
  }
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_storeu(T *const p, const Mask<T, SIMD_WIDTH> &k,
                                    const Vec<T, SIMD_WIDTH> &a)
{
  // if k is all zeros nothing should be stored
  if (test_all_zeros((Vec<T, SIMD_WIDTH>) k)) { return; }
  // If p till p+Vec<T, SIMD_WIDTH>::elems-1 is within the same page,
  // there is no risk of a page fault, so we load the whole vector, mask it
  // and store it back. Otherwise, we store the vector element-wise.
  if (is_within_same_page<SIMD_WIDTH>(p)) {
    storeu(p, mask::mask_ifelse(k, a, loadu<SIMD_WIDTH>(p)));
    return;
  }
  if (test_all_ones((Vec<T, SIMD_WIDTH>) k)) {
    // if k is all ones, we can store the whole vector
    storeu(p, a);
    return;
  }
  T a_arr[Vec<T, SIMD_WIDTH>::elems] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  store(a_arr, a);
  for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
    if (k[i]) { p[i] = a_arr[i]; }
  }
}

// maskz_store(u) does not exist/does not make sense

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_set1(const Mask<T, SIMD_WIDTH> &k,
                                                 const T a)
{
  return mask::mask_ifelsezero(k, ::simd::set1<T, SIMD_WIDTH>(a));
}
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_set1(const Vec<T, SIMD_WIDTH> &src,
                                                const Mask<T, SIMD_WIDTH> &k,
                                                const T a)
{
  return mask::mask_ifelse(k, ::simd::set1<T, SIMD_WIDTH>(a), src);
}

EMULATE_DOP(add)
EMULATE_DOP(adds)
EMULATE_DOP(sub)
EMULATE_DOP(subs)

EMULATE_DOP(mul)
EMULATE_DOP(div)

// ---------------------------------------------------------------------------
// masked ceil, floor, round, truncate v
// ---------------------------------------------------------------------------

EMULATE_SOP(ceil)
EMULATE_SOP(floor)
EMULATE_SOP(round)
EMULATE_SOP(truncate)

// ---------------------------------------------------------------------------
// masked elementary mathematical functions v
// ---------------------------------------------------------------------------

EMULATE_SOP(rcp)
EMULATE_SOP(rsqrt)
EMULATE_SOP(sqrt)

EMULATE_SOP(abs)

EMULATE_DOP_NAME(bit_and, and)
EMULATE_DOP_NAME(bit_or, or)
EMULATE_DOP_NAME(bit_andnot, andnot)
EMULATE_DOP_NAME(bit_xor, xor)
EMULATE_SOP_NAME(bit_not, not )
EMULATE_SOP(neg)
EMULATE_DOP(min)
EMULATE_DOP(max)
EMULATE_SOP(div2r0)
EMULATE_SOP(div2rd)

#define EMULATE_SHIFT(OP)                                                      \
  template <int IMM, typename T, int SIMD_WIDTH>                               \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_##OP(                            \
    const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a)                 \
  {                                                                            \
    return mask::mask_ifelsezero(k, OP<IMM>(a));                               \
  }                                                                            \
  template <int IMM, typename T, int SIMD_WIDTH>                               \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_##OP(                             \
    const Vec<T, SIMD_WIDTH> &src, const Mask<T, SIMD_WIDTH> &k,               \
    const Vec<T, SIMD_WIDTH> &a)                                               \
  {                                                                            \
    return mask::mask_ifelse(k, OP<IMM>(a), src);                              \
  }
EMULATE_SHIFT(srai)
EMULATE_SHIFT(srli)
EMULATE_SHIFT(slli)

EMULATE_DOP(hadd)
EMULATE_DOP(hadds)
EMULATE_DOP(hsub)
EMULATE_DOP(hsubs)

// TODO mask parameters?

// 16. Oct 22 (Jonas Keller): added overloaded versions of mask_cmp* functions
// that only take two vector parameters and no mask parameter
#define EMULATE_CMP(OP)                                                        \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_##OP(                            \
    const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a,                 \
    const Vec<T, SIMD_WIDTH> &b)                                               \
  {                                                                            \
    return Mask<T, SIMD_WIDTH>(mask::mask_ifelsezero(k, OP(a, b)));            \
  }                                                                            \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_##OP(                            \
    const Vec<T, SIMD_WIDTH> &a, const Vec<T, SIMD_WIDTH> &b)                  \
  {                                                                            \
    return Mask<T, SIMD_WIDTH>(OP(a, b));                                      \
  }

EMULATE_CMP(cmplt)
EMULATE_CMP(cmple)
EMULATE_CMP(cmpeq)
EMULATE_CMP(cmpgt)
EMULATE_CMP(cmpge)
EMULATE_CMP(cmpneq)

EMULATE_DOP(avg)

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_zeros(const Mask<T, SIMD_WIDTH> &k,
                                           const Vec<T, SIMD_WIDTH> &a)
{
  return test_all_zeros(mask::mask_ifelsezero(k, a));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_ones(const Mask<T, SIMD_WIDTH> &k,
                                          const Vec<T, SIMD_WIDTH> &a)
{
  return mask::mask_test_all_zeros(
    k, bit_not(a)); // test_all_ones(mask_ifelse<T, SIMD_WIDTH>(k, a, ()
                    // set1<Byte, SIMD_WIDTH>(0xFF)));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_all_ones(OutputType<T>,
                                                     Integer<SIMD_WIDTH>)
{
  return (Mask<T, SIMD_WIDTH>) set1<T, SIMD_WIDTH>(TypeInfo<T>::trueval());
}

#define EMULATE_DMASKOP(NAME)                                                  \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMD_INLINE Mask<T, SIMD_WIDTH> k##NAME(const Mask<T, SIMD_WIDTH> &a, \
                                                 const Mask<T, SIMD_WIDTH> &b) \
  {                                                                            \
    return (Mask<T, SIMD_WIDTH>) NAME##_((Vec<T, SIMD_WIDTH>) a,               \
                                         (Vec<T, SIMD_WIDTH>) b);              \
  }

EMULATE_DMASKOP(and)

// EMULATE_DMASKOP(andn)
//  function name should be "kandn" but the vector function is "bit_andnot"
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kandn(const Mask<T, SIMD_WIDTH> &a,
                                             const Mask<T, SIMD_WIDTH> &b)
{
  return (Mask<T, SIMD_WIDTH>) bit_andnot((Vec<T, SIMD_WIDTH>) a,
                                          (Vec<T, SIMD_WIDTH>) b);
}

EMULATE_DMASKOP(or)
EMULATE_DMASKOP(xor)

// EMULATE_DMASKOP(xnor)
//  there is not xnor-function for vectors, so we have to do: bit_not(bit_xor(a,
//  b))
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kxnor(const Mask<T, SIMD_WIDTH> &a,
                                             const Mask<T, SIMD_WIDTH> &b)
{
  return (Mask<T, SIMD_WIDTH>) bit_not(
    bit_xor((Vec<T, SIMD_WIDTH>) a, (Vec<T, SIMD_WIDTH>) b));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kadd(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  Mask<T, SIMD_WIDTH> ret;
  ret = (((uintmax_t) a) + ((uintmax_t) b));
  return ret;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> knot(const Mask<T, SIMD_WIDTH> &a)
{
  return (Mask<T, SIMD_WIDTH>) bit_not((Vec<T, SIMD_WIDTH>) a);
}

// shift with flexible parameter (not template), probably slower than
// template-version
// TODO faster implementation with switch-case possible?
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftri(const Mask<T, SIMD_WIDTH> &a,
                                                uintmax_t count)
{
  // 04. Aug 22 (Jonas Keller):
  // return zero if count is larger than sizeof(uintmax_t)*8 - 1, since then
  // the >> operator is undefined, but kshift should return zero
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
  if (count >= sizeof(uintmax_t) * 8) { return Mask<T, SIMD_WIDTH>(0); }
  return (Mask<T, SIMD_WIDTH>) (((uintmax_t) a) >> count);
}
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftli(const Mask<T, SIMD_WIDTH> &a,
                                                uintmax_t count)
{
  // 04. Aug 22 (Jonas Keller):
  // return zero if count is larger than sizeof(uintmax_t)*8 - 1, since then
  // the << operator is undefined, but kshift should return zero
  // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
  if (count >= sizeof(uintmax_t) * 8) { return Mask<T, SIMD_WIDTH>(0); }
  return (Mask<T, SIMD_WIDTH>) (((uintmax_t) a) << count);
}

// shift with template parameter
template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftri(const Mask<T, SIMD_WIDTH> &a)
{
  return (Mask<T, SIMD_WIDTH>) srle<IMM>((Vec<T, SIMD_WIDTH>) a);
}
template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftli(const Mask<T, SIMD_WIDTH> &a)
{
  return (Mask<T, SIMD_WIDTH>) slle<IMM>((Vec<T, SIMD_WIDTH>) a);
}

// 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
// mask_set_true/false_low/high.

template <bool UP, typename T, int SIMD_WIDTH>
struct MaskSetBuffer
{
  T buffer[Vec<T, SIMD_WIDTH>::elems * 2];
  MaskSetBuffer()
  {
    for (int i = 0; i < Vec<T, SIMD_WIDTH>::elems; i++) {
      buffer[i] = UP ? 0 : TypeInfo<T>::trueval();
    }
    for (int i = Vec<T, SIMD_WIDTH>::elems; i < Vec<T, SIMD_WIDTH>::elems * 2;
         i++) {
      buffer[i] = UP ? TypeInfo<T>::trueval() : 0;
    }
  }
};

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_true_low(const unsigned int x,
                                                         OutputType<T>,
                                                         Integer<SIMD_WIDTH>)
{
  if (x >= Vec<T, SIMD_WIDTH>::elems) {
    return mask_all_ones(OutputType<T>(), Integer<SIMD_WIDTH>());
  }
  static MaskSetBuffer<false, T, SIMD_WIDTH> buffer;
  return Mask<T, SIMD_WIDTH>(
    loadu<SIMD_WIDTH>(buffer.buffer + Vec<T, SIMD_WIDTH>::elems - x));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_true_high(const unsigned int x,
                                                          OutputType<T>,
                                                          Integer<SIMD_WIDTH>)
{
  if (x >= Vec<T, SIMD_WIDTH>::elems) {
    return mask_all_ones(OutputType<T>(), Integer<SIMD_WIDTH>());
  }
  static MaskSetBuffer<true, T, SIMD_WIDTH> buffer;
  return Mask<T, SIMD_WIDTH>(loadu<SIMD_WIDTH>(buffer.buffer + x));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_false_low(const unsigned int x,
                                                          OutputType<T>,
                                                          Integer<SIMD_WIDTH>)
{
  if (x >= Vec<T, SIMD_WIDTH>::elems) { return Mask<T, SIMD_WIDTH>(0); }
  static MaskSetBuffer<true, T, SIMD_WIDTH> buffer;
  return Mask<T, SIMD_WIDTH>(
    loadu<SIMD_WIDTH>(buffer.buffer + Vec<T, SIMD_WIDTH>::elems - x));
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_false_high(const unsigned int x,
                                                           OutputType<T>,
                                                           Integer<SIMD_WIDTH>)
{
  if (x >= Vec<T, SIMD_WIDTH>::elems) { return Mask<T, SIMD_WIDTH>(0); }
  static MaskSetBuffer<false, T, SIMD_WIDTH> buffer;
  return Mask<T, SIMD_WIDTH>(loadu<SIMD_WIDTH>(buffer.buffer + x));
}

// 07. Aug 23 (Jonas Keller): added ktest_all_zeros/ones.

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE bool ktest_all_zeros(const Mask<T, SIMD_WIDTH> &a)
{
  return test_all_zeros((Vec<T, SIMD_WIDTH>) a);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE bool ktest_all_ones(const Mask<T, SIMD_WIDTH> &a)
{
  return test_all_ones((Vec<T, SIMD_WIDTH>) a);
}

// 07. Aug 23 (Jonas Keller): added kcmpeq

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE bool kcmpeq(const Mask<T, SIMD_WIDTH> &a,
                               const Mask<T, SIMD_WIDTH> &b)
{
  return internal::mask::ktest_all_zeros(internal::mask::kxor(a, b));
}

} // namespace mask
} // namespace internal
} // namespace simd

#endif // SIMD_VEC_MASK_IMPL_EMU_H_
