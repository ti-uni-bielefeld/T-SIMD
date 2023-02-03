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

// 01. Feb 23 (Jonas Keller): implemented the emulated mask functions in a more
// efficient way, described below and also optimized other small things.
// 30. Nov 22 (Jonas Keller):
// NOTE:
// The float versions of the emulated mask functions in this file as well as in
// SIMDVecMaskImplIntel64.H are not as fast as they could be, as they are implemented
// such that they match the not emulated ones in flag and exception behavior as
// well. This is done by masking the inputs of the masked functions, which for
// example leads to the following code for masked addition:
/*
template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
maskz_add(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
          const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
          const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
{
  return add(mask_ifelse(k, a, setzero<SIMDFloat, SIMD_WIDTH>()),
             mask_ifelse(k, b, setzero<SIMDFloat, SIMD_WIDTH>()));
}
template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
mask_add(const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
         const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
         const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
         const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
{
  return mask_ifelse(k, maskz_add(k, a, b), src);
}
*/
// which calls mask_ifelse 3 times for one call of mask_add, instead of
/*
template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
maskz_add(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
          const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
          const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
{
  return mask_ifelse(k, add(a, b), setzero<SIMDFloat, SIMD_WIDTH>());
}
template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
mask_add(const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
         const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
         const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
         const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
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

#ifndef _SIMD_VEC_MASK_IMPL_EMU_H_
#define _SIMD_VEC_MASK_IMPL_EMU_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecMaskImplArchSpec.H"

#include <math.h>
#include <stdint.h>

namespace ns_simd {
namespace internal {
namespace mask {
  #define EMULATE_SOP(OP) \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask::mask_ifelsezero(k, OP (a)); \
  } \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  mask_ ## OP(const SIMDVec<T, SIMD_WIDTH> &src, \
      const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask::mask_ifelse(k, OP (a), src); \
  }

  #define EMULATE_DOP(OP) \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return mask::mask_ifelsezero(k, OP (a, b)); \
  } \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  mask_ ## OP(const SIMDVec<T, SIMD_WIDTH> &src, \
      const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return mask::mask_ifelse(k, OP (a, b), src); \
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ifelse(const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &trueVal,
	  const SIMDVec<T, SIMD_WIDTH> &falseVal)
  {
    return ifelse((SIMDVec<T, SIMD_WIDTH>) k, trueVal, falseVal);
  }

  // 04. Aug 22 (Jonas Keller): added mask_ifelsezero
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ifelsezero(const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &trueVal)
  {
    return and((SIMDVec<T, SIMD_WIDTH>) k, trueVal);
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<SIMDInt, SIMD_WIDTH>
  reinterpret_mask(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k)//TODO better name? more types?
  {
      return SIMDMask<SIMDInt, SIMD_WIDTH>(reinterpret<SIMDInt>((SIMDVec<SIMDFloat, SIMD_WIDTH>) k));
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<SIMDFloat, SIMD_WIDTH>
  reinterpret_mask(const SIMDMask<SIMDInt, SIMD_WIDTH> &k)//TODO better name? more types?
  {
      return SIMDMask<SIMDFloat, SIMD_WIDTH>(reinterpret<SIMDFloat>((SIMDVec<SIMDInt, SIMD_WIDTH>) k));
  }
  
  //The types of the masks are kind of arbitrary
  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return mask::mask_ifelsezero(reinterpret_mask(k), ::ns_simd::cvts<SIMDInt>(a));
  }
  
  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDInt, SIMD_WIDTH> &src,
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return mask::mask_ifelse(reinterpret_mask(k), ::ns_simd::cvts<SIMDInt>(a), src);
  }

  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDInt, SIMD_WIDTH> &k,
      const SIMDVec<SIMDInt, SIMD_WIDTH> &a)
  {
    return mask::mask_ifelsezero(reinterpret_mask(k), ::ns_simd::cvts<SIMDFloat>(a));
  }

  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
      const SIMDMask<SIMDInt, SIMD_WIDTH> &k,
      const SIMDVec<SIMDInt, SIMD_WIDTH> &a)
  {
    return mask::mask_ifelse(reinterpret_mask(k), ::ns_simd::cvts<SIMDFloat>(a), src);
  }

  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE bool
  is_within_same_page(const T *const p)
  {
    const uintptr_t PAGE_SIZE = 4096; // smallest page size I found
    const uintptr_t begin_page =
        reinterpret_cast<uintptr_t>(p) & ~(PAGE_SIZE - 1);
    const uintptr_t end_page =
        reinterpret_cast<uintptr_t>(p + SIMDVec<T, SIMD_WIDTH>::elems - 1) &
        ~(PAGE_SIZE - 1);
    return begin_page == end_page;
  }

  template<typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_load(const SIMDMask<T, SIMD_WIDTH> &k, const T *const p)
  {
    // if k is all zeros nothing should be loaded
    if (test_all_zeros((SIMDVec<T, SIMD_WIDTH>)k)) {
      return setzero<T, SIMD_WIDTH>();
    }
    // If p till p+SIMDVec<T, SIMD_WIDTH>::elems-1 is within the same page,
    // there is no risk of a page fault, so we load the whole vector and mask
    // it. Otherwise, we load the vector element-wise.
    if (is_within_same_page<SIMD_WIDTH>(p)) {
      return mask::mask_ifelsezero(k, load<SIMD_WIDTH>(p));
    }
    SIMDVec<T, SIMD_WIDTH> result = setzero<T, SIMD_WIDTH>();
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elems; i++) {
      if (k[i]) {
        ((T *)&result)[i] = p[i];
      }
    }
    return result;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_load(const SIMDVec<T, SIMD_WIDTH> &src,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const T *const p)
  {
    // if k is all zeros nothing should be loaded
    if (test_all_zeros((SIMDVec<T, SIMD_WIDTH>)k)) {
      return src;
    }
    // If p till p+SIMDVec<T, SIMD_WIDTH>::elems-1 is within the same page,
    // there is no risk of a page fault, so we load the whole vector and mask
    // it. Otherwise, we load the vector element-wise.
    if (is_within_same_page<SIMD_WIDTH>(p)) {
      return mask::mask_ifelse(k, load<SIMD_WIDTH>(p), src);
    }
    SIMDVec<T, SIMD_WIDTH> result;
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elems; i++) {
      ((T *)&result)[i] = k[i] ? p[i] : ((T *)&src)[i];
    }
    return result;
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_loadu(const SIMDMask<T, SIMD_WIDTH> &k,
      const T *const p)
  {
    // if k is all zeros nothing should be loaded
    if (test_all_zeros((SIMDVec<T, SIMD_WIDTH>)k)) {
      return setzero<T, SIMD_WIDTH>();
    }
    // If p till p+SIMDVec<T, SIMD_WIDTH>::elems-1 is within the same page,
    // there is no risk of a page fault, so we load the whole vector and mask
    // it. Otherwise, we load the vector element-wise.
    if (is_within_same_page<SIMD_WIDTH>(p)) {
      return mask::mask_ifelsezero(k, loadu<SIMD_WIDTH>(p));
    }
    SIMDVec<T, SIMD_WIDTH> result = setzero<T, SIMD_WIDTH>();
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elems; i++) {
      if (k[i]) {
        ((T *)&result)[i] = p[i];
      }
    }
    return result;
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_loadu(const SIMDVec<T, SIMD_WIDTH> &src,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const T *const p)
  {
    // if k is all zeros nothing should be loaded
    if (test_all_zeros((SIMDVec<T, SIMD_WIDTH>)k)) {
      return src;
    }
    // If p till p+SIMDVec<T, SIMD_WIDTH>::elems-1 is within the same page,
    // there is no risk of a page fault, so we load the whole vector and mask
    // it. Otherwise, we load the vector element-wise.
    if (is_within_same_page<SIMD_WIDTH>(p)) {
      return mask::mask_ifelse(k, loadu<SIMD_WIDTH>(p), src);
    }
    SIMDVec<T, SIMD_WIDTH> result;
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elems; i++) {
      ((T *)&result)[i] = k[i] ? p[i] : ((T *)&src)[i];
    }
    return result;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_store(T *const p,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &a) {
    // if k is all zeros nothing should be stored
    if (test_all_zeros((SIMDVec<T, SIMD_WIDTH>)k)) {
      return;
    }
    // If p till p+SIMDVec<T, SIMD_WIDTH>::elems-1 is within the same page,
    // there is no risk of a page fault, so we load the whole vector, mask it
    // and store it back. Otherwise, we store the vector element-wise.
    if (is_within_same_page<SIMD_WIDTH>(p)) {
      store(p, mask::mask_ifelse(k, a, load<SIMD_WIDTH>(p)));
      return;
    }
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elems; i++) {
      if (k[i]) {
        p[i] = ((T *)&a)[i];
      }
    }
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_storeu(T *const p,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &a) {
    // if k is all zeros nothing should be stored
    if (test_all_zeros((SIMDVec<T, SIMD_WIDTH>)k)) {
      return;
    }
    // If p till p+SIMDVec<T, SIMD_WIDTH>::elems-1 is within the same page,
    // there is no risk of a page fault, so we load the whole vector, mask it
    // and store it back. Otherwise, we store the vector element-wise.
    if (is_within_same_page<SIMD_WIDTH>(p)) {
      storeu(p, mask::mask_ifelse(k, a, loadu<SIMD_WIDTH>(p)));
      return;
    }
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elems; i++) {
      if (k[i]) {
        p[i] = ((T *)&a)[i];
      }
    }
  }

  // maskz_store(u) does not exist/does not make sense

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_set1 (const SIMDMask<T, SIMD_WIDTH> &k,
      const T a)
  {
    return mask::mask_ifelsezero(k, ::ns_simd::set1<T, SIMD_WIDTH>(a));
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_set1 (const SIMDVec<T, SIMD_WIDTH> &src,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const T a)
  {
    return mask::mask_ifelse(k, ::ns_simd::set1<T, SIMD_WIDTH>(a), src);
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
  
  EMULATE_DOP(and)
  EMULATE_DOP(or)
  EMULATE_DOP(andnot)
  EMULATE_DOP(xor)
  EMULATE_SOP(not)
  EMULATE_SOP(neg)
  EMULATE_DOP(min)
  EMULATE_DOP(max)
  EMULATE_SOP(div2r0)
  EMULATE_SOP(div2rd)
  
  #define EMULATE_SHIFT(OP) \
  template <int IMM, typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask::mask_ifelsezero(k, OP <IMM>(a)); \
  } \
  template <int IMM, typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  mask_ ## OP(const SIMDVec<T, SIMD_WIDTH> &src, \
      const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask::mask_ifelse(k, OP <IMM>(a), src); \
  }
  EMULATE_SHIFT(srai)
  EMULATE_SHIFT(srli)
  EMULATE_SHIFT(slli)

  EMULATE_DOP(hadd)
  EMULATE_DOP(hadds)
  EMULATE_DOP(hsub)
  EMULATE_DOP(hsubs)

  //TODO mask parameters?

  // 16. Oct 22 (Jonas Keller): added overloaded versions of mask_cmp* functions
  // that only take two vector parameters and no mask parameter
#define EMULATE_CMP(OP) \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> \
  mask_ ## OP (const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return SIMDMask<T, SIMD_WIDTH>(mask::mask_ifelsezero(k, OP(a, b))); \
  } \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> \
  mask_ ## OP (const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return SIMDMask<T, SIMD_WIDTH>(OP(a, b)); \
  }
  
  EMULATE_CMP(cmplt)
  EMULATE_CMP(cmple)
  EMULATE_CMP(cmpeq)
  EMULATE_CMP(cmpgt)
  EMULATE_CMP(cmpge)
  EMULATE_CMP(cmpneq)
  
  EMULATE_DOP(avg)
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_zeros(const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return test_all_zeros(mask::mask_ifelsezero(k, a));
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_ones(const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return mask::mask_test_all_zeros(k, not(a)); //test_all_ones(mask_ifelse<T, SIMD_WIDTH>(k, a, () set1<SIMDByte, SIMD_WIDTH>(0xFF)));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_all_ones(OutputType<T>, Integer<SIMD_WIDTH>)
  {
    return (SIMDMask<T, SIMD_WIDTH>)set1<T, SIMD_WIDTH>(
        SIMDTypeInfo<T>::trueval());
  }
  
  #define EMULATE_DMASKOP(NAME) \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> \
  k ## NAME (const SIMDMask<T, SIMD_WIDTH> &a, \
      const SIMDMask<T, SIMD_WIDTH> &b) \
  { \
    return (SIMDMask<T, SIMD_WIDTH>) NAME ((SIMDVec<T, SIMD_WIDTH>) a, (SIMDVec<T, SIMD_WIDTH>) b); \
  }

  EMULATE_DMASKOP(and)
  
  //EMULATE_DMASKOP(andn)
  // function name should be "kandn" but the vector function is "andnot"
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kandn (const SIMDMask<T, SIMD_WIDTH> &a,
      const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return (SIMDMask<T, SIMD_WIDTH>) andnot ((SIMDVec<T, SIMD_WIDTH>) a, (SIMDVec<T, SIMD_WIDTH>) b);
  }
  
  EMULATE_DMASKOP(or)
  EMULATE_DMASKOP(xor)
  
  //EMULATE_DMASKOP(xnor)
  // there is not xnor-function for vectors, so we have to do: not(xor(a, b))
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kxnor (const SIMDMask<T, SIMD_WIDTH> &a,
      const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return (SIMDMask<T, SIMD_WIDTH>) not(xor((SIMDVec<T, SIMD_WIDTH>) a, (SIMDVec<T, SIMD_WIDTH>) b));
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kadd (const SIMDMask<T, SIMD_WIDTH> &a,
      const SIMDMask<T, SIMD_WIDTH> &b)
  {
    SIMDMask<T, SIMD_WIDTH> ret;
    ret=(((uintmax_t) a)+((uintmax_t) b));
    return ret;
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  knot(const SIMDMask<T, SIMD_WIDTH> &a)
  {
    return (SIMDMask<T, SIMD_WIDTH>) not((SIMDVec<T, SIMD_WIDTH>) a);
  }

  //shift with flexible parameter (not template), probably slower than template-version
  //TODO faster implementation with switch-case possible?
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftri (const SIMDMask<T, SIMD_WIDTH> &a,
      uintmax_t count)
  {
    // 04. Aug 22 (Jonas Keller):
    // return zero if count is larger than sizeof(uintmax_t)*8 - 1, since then
    // the >> operator is undefined, but kshift should return zero
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
    if (count >= sizeof(uintmax_t)*8) {
      return 0;
    }
    return (SIMDMask<T, SIMD_WIDTH>) (((uintmax_t) a) >> count);
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftli (const SIMDMask<T, SIMD_WIDTH> &a,
      uintmax_t count)
  {
    // 04. Aug 22 (Jonas Keller):
    // return zero if count is larger than sizeof(uintmax_t)*8 - 1, since then
    // the << operator is undefined, but kshift should return zero
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
    if (count >= sizeof(uintmax_t)*8) {
      return 0;
    }
    return (SIMDMask<T, SIMD_WIDTH>) (((uintmax_t) a) << count);
  }
  
  //shift with template parameter
  template <unsigned int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftri (const SIMDMask<T, SIMD_WIDTH> &a)
  {
    return (SIMDMask<T, SIMD_WIDTH>) srle<IMM>((SIMDVec<T, SIMD_WIDTH>) a);
  }
  template <unsigned int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftli (const SIMDMask<T, SIMD_WIDTH> &a)
  {
    return (SIMDMask<T, SIMD_WIDTH>) slle<IMM>((SIMDVec<T, SIMD_WIDTH>) a);
  }

  // 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
  // mask_set_true/false_low/high.

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_set_true_low(const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>) {
    if (x >= SIMDVec<T, SIMD_WIDTH>::elems) {
      return mask_all_ones(OutputType<T>(), Integer<SIMD_WIDTH>());
    }
    return mask::mask_cmplt(iota<T, SIMD_WIDTH>(), set1<T, SIMD_WIDTH>(T(x)));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_set_true_high(const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>) {
    if (x >= SIMDVec<T, SIMD_WIDTH>::elems) {
      return mask_all_ones(OutputType<T>(), Integer<SIMD_WIDTH>());
    }
    return mask::mask_cmplt(reverse(iota<T, SIMD_WIDTH>()),
                            set1<T, SIMD_WIDTH>(T(x)));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_set_false_low(const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>) {
    if (x >= SIMDVec<T, SIMD_WIDTH>::elems) {
      return SIMDMask<T, SIMD_WIDTH>(0);
    }
    return mask::mask_cmpge(iota<T, SIMD_WIDTH>(), set1<T, SIMD_WIDTH>(T(x)));
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_set_false_high(const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>) {
    if (x >= SIMDVec<T, SIMD_WIDTH>::elems) {
      return SIMDMask<T, SIMD_WIDTH>(0);
    }
    return mask::mask_cmpge(reverse(iota<T, SIMD_WIDTH>()),
                            set1<T, SIMD_WIDTH>(T(x)));
  }
} // namespace mask
} // namespace internal
} // namespace ns_simd

#endif // _SIMD_VEC_MASK_IMPL_EMU_H_
