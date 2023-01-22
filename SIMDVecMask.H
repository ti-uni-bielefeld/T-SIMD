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
// (C) Ralf Möller
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

#ifdef SIMDVEC_SANDBOX

// test templates
#include "SIMDVecMaskImplSandbox.H"

#else // SIMDVEC_SANDBOX

#include "SIMDVecMaskImplArchSpec.H"
#include "SIMDVecMaskImplEmu.H"

#endif // else SIMDVEC_SANDBOX

namespace ns_simd {

  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_all_ones()
  {
    return internal::mask::mask_all_ones(internal::OutputType<T>(),
                                   internal::Integer<SIMD_WIDTH>());
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ifelse(const SIMDMask<T, SIMD_WIDTH> &cond,
              const SIMDVec<T, SIMD_WIDTH> &trueVal,
              const SIMDVec<T, SIMD_WIDTH> &falseVal)
  {
    return internal::mask::mask_ifelse(cond, trueVal, falseVal);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ifelsezero(const SIMDMask<T, SIMD_WIDTH> &cond,
                  const SIMDVec<T, SIMD_WIDTH> &trueVal)
  {
    return internal::mask::mask_ifelsezero(cond, trueVal);
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDInt, SIMD_WIDTH> &src,
            const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
            const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_cvts(src, k, a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDFloat, SIMD_WIDTH> &k,
             const SIMDVec<SIMDFloat, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_cvts(k, a);
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDFloat, SIMD_WIDTH> &src,
            const SIMDMask<SIMDInt, SIMD_WIDTH> &k,
            const SIMDVec<SIMDInt, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_cvts(src, k, a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDInt, SIMD_WIDTH> &k,
             const SIMDVec<SIMDInt, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_cvts(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_set1(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const T &a)
  {
    return internal::mask::mask_set1(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_set1(const SIMDMask<T, SIMD_WIDTH> &k,
             const T &a)
  {
    return internal::mask::maskz_set1(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_load(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const T *const p)
  {
    return internal::mask::mask_load(src, k, p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_load(const SIMDMask<T, SIMD_WIDTH> &k,
             const T *const p)
  {
    return internal::mask::maskz_load(k, p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_loadu(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const T *const p)
  {
    return internal::mask::mask_loadu(src, k, p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_loadu(const SIMDMask<T, SIMD_WIDTH> &k,
             const T *const p)
  {
    return internal::mask::maskz_loadu(k, p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_store(T *const p,
             const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::mask::mask_store(p, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_storeu(T *const p,
              const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::mask::mask_storeu(p, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_add(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_add(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_add(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_add(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_adds(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_adds(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_adds(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_adds(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_sub(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_sub(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_sub(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_sub(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_subs(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_subs(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_subs(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_subs(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_mul(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_mul(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_mul(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_mul(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_div(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_div(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_div(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_div(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ceil(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_ceil(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_ceil(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_ceil(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_floor(const SIMDVec<T, SIMD_WIDTH> &src,
             const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_floor(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_floor(const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_floor(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_round(const SIMDVec<T, SIMD_WIDTH> &src,
             const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_round(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_round(const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_round(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_truncate(const SIMDVec<T, SIMD_WIDTH> &src,
                const SIMDMask<T, SIMD_WIDTH> &k,
                const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_truncate(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_truncate(const SIMDMask<T, SIMD_WIDTH> &k,
                 const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_truncate(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_rcp(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_rcp(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_rcp(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_rcp(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_rsqrt(const SIMDVec<T, SIMD_WIDTH> &src,
             const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_rsqrt(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_rsqrt(const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_rsqrt(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_sqrt(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_sqrt(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_sqrt(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_sqrt(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_abs(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_abs(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_abs(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_abs(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_and(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_and(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_and(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_and(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_or(const SIMDVec<T, SIMD_WIDTH> &src,
          const SIMDMask<T, SIMD_WIDTH> &k,
          const SIMDVec<T, SIMD_WIDTH> &a,
          const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_or(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_or(const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_or(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_andnot(const SIMDVec<T, SIMD_WIDTH> &src,
              const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a,
              const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_andnot(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_andnot(const SIMDMask<T, SIMD_WIDTH> &k,
               const SIMDVec<T, SIMD_WIDTH> &a,
               const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_andnot(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_xor(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_xor(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_xor(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_xor(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_not(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_not(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_not(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_not(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_neg(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_neg(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_neg(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_neg(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_min(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_min(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_min(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_min(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_max(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_max(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_max(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_max(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_div2r0(const SIMDVec<T, SIMD_WIDTH> &src,
              const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_div2r0(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_div2r0(const SIMDMask<T, SIMD_WIDTH> &k,
               const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_div2r0(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_div2rd(const SIMDVec<T, SIMD_WIDTH> &src,
              const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_div2rd(src, k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_div2rd(const SIMDMask<T, SIMD_WIDTH> &k,
               const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_div2rd(k, a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_srai(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_srai<IMM>(src, k, a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_srai(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_srai<IMM>(k, a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_srli(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_srli<IMM>(src, k, a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_srli(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_srli<IMM>(k, a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_slli(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_slli<IMM>(src, k, a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_slli(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::maskz_slli<IMM>(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hadd(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_hadd(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hadd(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_hadd(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hadds(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_hadds(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hadds(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_hadds(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hsub(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_hsub(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hsub(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_hsub(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hsubs(const SIMDVec<T, SIMD_WIDTH> &src,
            const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_hsubs(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hsubs(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_hsubs(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmplt(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmplt(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmple(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmple(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpeq(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpeq(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpgt(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpgt(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpge(const SIMDMask<T, SIMD_WIDTH> &k,
             const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpge(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpneq(const SIMDMask<T, SIMD_WIDTH> &k,
              const SIMDVec<T, SIMD_WIDTH> &a,
              const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpneq(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmplt(const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmplt(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmple(const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmple(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpeq(const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpeq(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpgt(const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpgt(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpge(const SIMDVec<T, SIMD_WIDTH> &a,
             const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpge(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpneq(const SIMDVec<T, SIMD_WIDTH> &a,
              const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_cmpneq(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_avg(const SIMDVec<T, SIMD_WIDTH> &src,
           const SIMDMask<T, SIMD_WIDTH> &k,
           const SIMDVec<T, SIMD_WIDTH> &a,
           const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::mask_avg(src, k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_avg(const SIMDMask<T, SIMD_WIDTH> &k,
            const SIMDVec<T, SIMD_WIDTH> &a,
            const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::mask::maskz_avg(k, a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_zeros(const SIMDMask<T, SIMD_WIDTH> &k,
                      const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::mask::mask_test_all_zeros(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_ones(const SIMDMask<T, SIMD_WIDTH> & k,
                     const SIMDVec<T, SIMD_WIDTH> & a)
  {
    return internal::mask::mask_test_all_ones(k, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kadd(const SIMDMask<T, SIMD_WIDTH> &a,
       const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return internal::mask::kadd(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kand(const SIMDMask<T, SIMD_WIDTH> &a,
       const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return internal::mask::kand(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kandn(const SIMDMask<T, SIMD_WIDTH> &a,
        const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return internal::mask::kandn(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kor(const SIMDMask<T, SIMD_WIDTH> &a,
      const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return internal::mask::kor(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kxor(const SIMDMask<T, SIMD_WIDTH> &a,
       const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return internal::mask::kxor(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kxnor(const SIMDMask<T, SIMD_WIDTH> &a,
        const SIMDMask<T, SIMD_WIDTH> &b)
  {
    return internal::mask::kxnor(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  knot(const SIMDMask<T, SIMD_WIDTH> &a)
  {
    return internal::mask::knot(a);
  }

  template <unsigned int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftri(const SIMDMask<T, SIMD_WIDTH> &a)
  {
    return internal::mask::kshiftri<IMM>(a);
  }

  template <unsigned int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftli(const SIMDMask<T, SIMD_WIDTH> &a)
  {
    return internal::mask::kshiftli<IMM>(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftri(const SIMDMask<T, SIMD_WIDTH> &a,
           const uint64_t count)
  {
    return internal::mask::kshiftri(a, count);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftli(const SIMDMask<T, SIMD_WIDTH> &a,
           const uint64_t count)
  {
    return internal::mask::kshiftli(a, count);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  setTrueLeft(const unsigned int x)
  {
    return internal::mask::setTrueLeft<T, SIMD_WIDTH>(x);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  setTrueRight(const unsigned int x)
  {
    return internal::mask::setTrueRight<T, SIMD_WIDTH>(x);
  }
}

#endif // _SIMD_VEC_MASK_H_
