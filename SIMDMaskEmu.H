// ===========================================================================
// 
// SIMDMaskEmu.H --
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

#ifndef _SIMDMASKEMU_H_
#define _SIMDMASKEMU_H_

#include "SIMDDefs.H"
#include "SIMDMaskArchSpec.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"

#include <stdint.h>

namespace ns_simd {
  #define EMULATE_SOP_INT(OP) \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask_ifelse(k, OP (a), setzero<T,SIMD_WIDTH>()); \
  } \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  mask_ ## OP(const SIMDVec<T, SIMD_WIDTH> &src, \
      const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask_ifelse(k, OP (a), src); \
  }

  #define EMULATE_SOP_FLOAT(OP) \
  template <int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<SIMDFloat, SIMD_WIDTH> &k, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a) \
  { \
    return mask_ifelse(k, OP (a), setzero<SIMDFloat,SIMD_WIDTH>()); \
  } \
  template <int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH> \
  mask_ ## OP (const SIMDVec<SIMDFloat, SIMD_WIDTH> &src, \
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a) \
  { \
    return mask_ifelse(k, OP (a), src); \
  }

  #define EMULATE_SOP(OP) EMULATE_SOP_INT(OP) EMULATE_SOP_FLOAT(OP)

  #define EMULATE_DOP_INT(OP) \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return mask_ifelse(k, OP (a, b), setzero<T,SIMD_WIDTH>()); \
  } \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  mask_ ## OP(const SIMDVec<T, SIMD_WIDTH> &src, \
      const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return mask_ifelse(k, OP (a, b), src); \
  }

  #define EMULATE_DOP_FLOAT(OP) \
  template <int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH> \
  maskz_ ## OP (const SIMDMask<SIMDFloat, SIMD_WIDTH> &k, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &b) \
  { \
    return OP (mask_ifelse(k, a, setzero<SIMDFloat,SIMD_WIDTH>()), mask_ifelse(k, b, setzero<SIMDFloat,SIMD_WIDTH>())); \
  } \
  template <int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH> \
  mask_ ## OP (const SIMDVec<SIMDFloat, SIMD_WIDTH> &src, \
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &b) \
  { \
    return mask_ifelse(k, maskz_ ## OP (k, a, b), src); \
  }

  #define EMULATE_DOP(OP) EMULATE_DOP_INT(OP) EMULATE_DOP_FLOAT(OP)
  
  
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
    return ifelse((SIMDVec<T, SIMD_WIDTH>) k, trueVal, setzero<T,SIMD_WIDTH>());
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
    return cvts<SIMDInt, SIMDFloat, SIMD_WIDTH>(mask_ifelse(k, a, setzero<SIMDFloat,SIMD_WIDTH>()));
  }
  
  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDInt, SIMD_WIDTH> &src,
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return mask_ifelse(reinterpret_mask(k), cvts<SIMDInt, SIMDFloat, SIMD_WIDTH>(mask_ifelse(k, a, setzero<SIMDFloat, SIMD_WIDTH>())), src);
  }

  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDInt, SIMD_WIDTH> &k,
      const SIMDVec<SIMDInt, SIMD_WIDTH> &a)
  {
    return cvts<SIMDFloat, SIMDInt, SIMD_WIDTH>(mask_ifelse(k, a, setzero<SIMDInt,SIMD_WIDTH>()));
  }

  template <int SIMD_WIDTH>
  SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
      const SIMDMask<SIMDInt, SIMD_WIDTH> &k,
      const SIMDVec<SIMDInt, SIMD_WIDTH> &a)
  {
    return mask_ifelse(reinterpret_mask(k), cvts<SIMDFloat, SIMDInt, SIMD_WIDTH>(mask_ifelse(k, a, setzero<SIMDInt,SIMD_WIDTH>())), src);
  }

  //maskz_load is implemented with architecture specific intrinsics (in SIMDMaskIntel16/32/Neon16.H)

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_load(const SIMDVec<T, SIMD_WIDTH> &src,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const T *const p)
  {
    return mask_ifelse(k, maskz_load(k, p), src);//There might be some potential for optimization here: if, e.g. the maskz uses inserts on a setzero, then the src could be used instead (saves the mask_ifelse)
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_loadu(const SIMDMask<T, SIMD_WIDTH> &k,
      const T *const p)
  {
    return maskz_load(k, p);//The functions maskz_load uses (maskload, loadl, loadh) do not need any alignment
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_loadu(const SIMDVec<T, SIMD_WIDTH> &src,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const T *const p)
  {
    return mask_ifelse(k, maskz_loadu(k, p), src);//There might be some potential for optimization here: if, e.g. the maskz uses inserts on a setzero, then the src could be used instead (saves the mask_ifelse)
  }

  //maskz_storeu does not exist/does not make sense
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_storeu(T *const p,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &a) {
      return mask_store(p, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_set1 (const SIMDMask<T, SIMD_WIDTH> &k,
      const T a)
  {
    return mask_ifelse(k, set1<T, SIMD_WIDTH>(a), setzero<T, SIMD_WIDTH>());
  }
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_set1 (const SIMDVec<T, SIMD_WIDTH> &src,
      const SIMDMask<T, SIMD_WIDTH> &k,
      const T a)
  {
    return mask_ifelse(k, set1<T, SIMD_WIDTH>(a), src);
  }

  EMULATE_DOP(add)
  EMULATE_DOP(adds)
  EMULATE_DOP(sub)
  EMULATE_DOP(subs)

  EMULATE_DOP_FLOAT(mul)

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_div(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    return div(mask_ifelse(k, a, setzero<SIMDFloat,SIMD_WIDTH>()), mask_ifelse(k, b, set1<SIMDFloat,SIMD_WIDTH>(1.0f))); // "div" is special because the "neutral" operation must be 0/1, not 0/0
  }
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_div(const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    return mask_ifelse(k, maskz_div(k, a, b), src);
  }
  
  // ---------------------------------------------------------------------------
  // masked ceil, floor, round, truncate v
  // ---------------------------------------------------------------------------

  EMULATE_SOP_FLOAT(ceil)
  EMULATE_SOP_FLOAT(floor)
  EMULATE_SOP_FLOAT(round)
  EMULATE_SOP_FLOAT(truncate)

  // ---------------------------------------------------------------------------
  // masked elementary mathematical functions v
  // ---------------------------------------------------------------------------

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_rcp (const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return rcp(mask_ifelse(k, a, set1<SIMDFloat,SIMD_WIDTH>(INFINITY)));
  }
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_rcp (const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return mask_ifelse(k, maskz_rcp (k, a), src);
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_rsqrt (const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return rsqrt (mask_ifelse(k, a, set1<SIMDFloat,SIMD_WIDTH>(INFINITY)));
  }
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_rsqrt (const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return mask_ifelse(k, maskz_rsqrt (k, a), src);
  }
  
  EMULATE_SOP_FLOAT(sqrt)
  
  EMULATE_SOP(abs)
  
  EMULATE_DOP(and)
  EMULATE_DOP(or)
  EMULATE_DOP(andnot)
  EMULATE_DOP(xor)
  EMULATE_SOP_INT(not)
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_not (const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return not(mask_ifelse(k, a, reinterpret<SIMDFloat>(set1<SIMDInt,SIMD_WIDTH>(0xffffffff))));
  }
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_not (const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
      const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return mask_ifelse(k, maskz_not (k, a), src);
  }
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
    return mask_ifelse(k, OP <IMM>(a), setzero<T,SIMD_WIDTH>()); \
  } \
  template <int IMM, typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> \
  mask_ ## OP(const SIMDVec<T, SIMD_WIDTH> &src, \
      const SIMDMask<T, SIMD_WIDTH> &k, \
      const SIMDVec<T, SIMD_WIDTH> &a) \
  { \
    return mask_ifelse(k, OP <IMM>(a), src); \
  }
  EMULATE_SHIFT(srai)
  EMULATE_SHIFT(srli)
  EMULATE_SHIFT(slli)

  // 04. Aug 22 (Jonas Keller):
  // Removed float specialization of hadd, hadds, hsub and hsubs,
  // instead use generic int version for floats as well.
  // Special float implementation causes wrong results for
  // hadd, hadds, hsub and hsubs.
  //
  // EMULATE_DOP(hadd)
  // EMULATE_DOP(hadds)
  // EMULATE_DOP(hsub)
  // EMULATE_DOP(hsubs)
  EMULATE_DOP_INT(hadd)
  EMULATE_DOP_INT(hadds)
  EMULATE_DOP_INT(hsub)
  EMULATE_DOP_INT(hsubs)

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
    return SIMDMask<T, SIMD_WIDTH>(mask_ifelse(k, OP(a, b), setzero<T, SIMD_WIDTH>())); \
  } \
  template <int SIMD_WIDTH> \
  static SIMD_INLINE SIMDMask<SIMDFloat, SIMD_WIDTH> \
  mask_ ## OP (const SIMDMask<SIMDFloat, SIMD_WIDTH> &k, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &b) \
  { \
    return SIMDMask<SIMDFloat, SIMD_WIDTH>(mask_ifelse(k, OP(mask_ifelse(k, a, setzero<SIMDFloat, SIMD_WIDTH>()), mask_ifelse(k, b, setzero<SIMDFloat, SIMD_WIDTH>())), setzero<SIMDFloat, SIMD_WIDTH>())); \
  } \
  template <typename T, int SIMD_WIDTH> \
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> \
  mask_ ## OP (const SIMDVec<T, SIMD_WIDTH> &a, \
      const SIMDVec<T, SIMD_WIDTH> &b) \
  { \
    return SIMDMask<T, SIMD_WIDTH>(OP(a, b)); \
  } \
  template <int SIMD_WIDTH> \
  static SIMD_INLINE SIMDMask<SIMDFloat, SIMD_WIDTH> \
  mask_ ## OP (const SIMDVec<SIMDFloat, SIMD_WIDTH> &a, \
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &b) \
  { \
    return SIMDMask<SIMDFloat, SIMD_WIDTH>(OP(a, b)); \
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
    return test_all_zeros(mask_ifelse<T, SIMD_WIDTH>(k, a, setzero<T, SIMD_WIDTH>()));
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_ones(const SIMDMask<T, SIMD_WIDTH> &k,
      const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return mask_test_all_zeros(k, not(a)); //test_all_ones(mask_ifelse<T, SIMD_WIDTH>(k, a, () set1<SIMDByte, SIMD_WIDTH>(0xFF)));
  }
  template <int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_ones(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
      const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return test_all_ones(mask_ifelse<SIMDFloat, SIMD_WIDTH>(k, a, reinterpret<SIMDFloat>(set1<SIMDByte, SIMD_WIDTH>(0xFF))));
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

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  setTrueLeft(const unsigned int x)
  {
    return kshiftli(mask_all_ones<T, SIMD_WIDTH>(), x);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  setTrueRight(const unsigned int x)
  {
    return kshiftri(mask_all_ones<T, SIMD_WIDTH>(), x);
  }
} //namespace ns_simd
#endif // _SIMDMASKEMU_H_
