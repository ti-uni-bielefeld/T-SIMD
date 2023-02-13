// ===========================================================================
//
// SIMDVecMaskImplSandbox.H --
// test template functions for Mask and associated function templates
// these functions just print out template parameters and some arguments
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

// 04. Aug 22 (Jonas Keller): added masked functions

#ifndef _SIMD_VEC_MASK_IMPL_SANDBOX_H_
#define _SIMD_VEC_MASK_IMPL_SANDBOX_H_

// TODO: should this only contain level 0 functions?

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <stdint.h>
#include <stdio.h>

#ifdef SIMDVEC_SANDBOX

namespace simd {
// ===========================================================================
// generic template for Mask
// ===========================================================================

template <typename T, int SIMD_WIDTH>
class Mask
{
public:
  Mask() {}
  Mask(const uint64_t &) {}
  operator uint64_t() const { return 0; }
};

namespace internal {
namespace mask {
// ===========================================================================
// masked functions
// ===========================================================================

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_all_ones(OutputType<T>,
                                                     Integer<SIMD_WIDTH>)
{
  printf("mask_all_ones<%s,%d>()\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ifelse(
  const Mask<T, SIMD_WIDTH> &cond, const Vec<T, SIMD_WIDTH> &trueVal,
  const Vec<T, SIMD_WIDTH> &falseVal)
{
  (void) cond;
  (void) trueVal;
  (void) falseVal;
  printf("mask_ifelse<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// mask_ifelsezero
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ifelsezero(
  const Mask<T, SIMD_WIDTH> &cond, const Vec<T, SIMD_WIDTH> &trueVal)
{
  (void) cond;
  (void) trueVal;
  printf("mask_ifelsezero<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE Mask<Tout, SIMD_WIDTH> reinterpret_mask(
  const Mask<Tin, SIMD_WIDTH> &)
{
  printf("reinterpret_mask<%s,%s,%d>(M)\n", TypeInfo<Tout>::name(),
         TypeInfo<Tin>::name(), SIMD_WIDTH);
  return Mask<Tout, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Int, SIMD_WIDTH> mask_cvts(
  const Vec<Int, SIMD_WIDTH> &, const Mask<Float, SIMD_WIDTH> &,
  const Vec<Float, SIMD_WIDTH> &)
{
  printf("mask_cvts<%s,%s,%d>(V,M,V)\n", TypeInfo<Int>::name(),
         TypeInfo<Float>::name(), SIMD_WIDTH);
  return Vec<Int, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Int, SIMD_WIDTH> maskz_cvts(
  const Mask<Float, SIMD_WIDTH> &, const Vec<Float, SIMD_WIDTH> &)
{
  printf("maskz_cvts<%s,%s,%d>(M,V)\n", TypeInfo<Int>::name(),
         TypeInfo<Float>::name(), SIMD_WIDTH);
  return Vec<Int, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH> mask_cvts(
  const Vec<Float, SIMD_WIDTH> &, const Mask<Int, SIMD_WIDTH> &,
  const Vec<Int, SIMD_WIDTH> &)
{
  printf("mask_cvts<%s,%s,%d>(V,M,V)\n", TypeInfo<Float>::name(),
         TypeInfo<Int>::name(), SIMD_WIDTH);
  return Vec<Float, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH> maskz_cvts(
  const Mask<Int, SIMD_WIDTH> &, const Vec<Int, SIMD_WIDTH> &)
{
  printf("maskz_cvts<%s,%s,%d>(M,V)\n", TypeInfo<Float>::name(),
         TypeInfo<Int>::name(), SIMD_WIDTH);
  return Vec<Float, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_set1(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const T &)
{
  printf("mask_set1<%s,%d>(V,M,T)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_set1(const Mask<T, SIMD_WIDTH> &,
                                                 const T &)
{
  printf("maskz_set1<%s,%d>(M,T)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_load(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const T *const p)
{
  printf("mask_load<%s,%d>(V,M,%p)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_load(const Mask<T, SIMD_WIDTH> &,
                                                 const T *const p)
{
  printf("maskz_load<%s,%d>(M,%p)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_loadu(const Vec<T, SIMD_WIDTH> &,
                                                 const Mask<T, SIMD_WIDTH> &,
                                                 const T *const p)
{
  printf("mask_loadu<%s,%d>(V,M,%p)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_loadu(const Mask<T, SIMD_WIDTH> &,
                                                  const T *const p)
{
  printf("maskz_loadu<%s,%d>(M,%p)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_store(T *const p, const Mask<T, SIMD_WIDTH> &,
                                   const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_store<%s,%d>(%p,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_storeu(T *const p, const Mask<T, SIMD_WIDTH> &,
                                    const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_storeu<%s,%d>(%p,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_add(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_add<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_add(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_add<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_adds(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_adds<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_adds(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_adds<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_sub(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_sub<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_sub(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_sub<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_subs(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_subs<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_subs(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_subs<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_mul(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_mul<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_mul(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_mul<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_div(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_div<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_div(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_div<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_ceil(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_ceil<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_ceil(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_ceil<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_floor(const Vec<T, SIMD_WIDTH> &,
                                                 const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_floor<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_floor(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_floor<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_round(const Vec<T, SIMD_WIDTH> &,
                                                 const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_round<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_round(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_round<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_truncate(const Vec<T, SIMD_WIDTH> &,
                                                    const Mask<T, SIMD_WIDTH> &,
                                                    const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_truncate<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_truncate(
  const Mask<T, SIMD_WIDTH> &, const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_truncate<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_rcp(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_rcp<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_rcp(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_rcp<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_rsqrt(const Vec<T, SIMD_WIDTH> &,
                                                 const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_rsqrt<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_rsqrt(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_rsqrt<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_sqrt(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_sqrt<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_sqrt(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_sqrt<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_abs(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_abs<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_abs(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_abs<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_and(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_and<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_and(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_and<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_or(const Vec<T, SIMD_WIDTH> &,
                                              const Mask<T, SIMD_WIDTH> &,
                                              const Vec<T, SIMD_WIDTH> &,
                                              const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_or<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_or(const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_or<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_andnot(const Vec<T, SIMD_WIDTH> &,
                                                  const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_andnot<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_andnot(const Mask<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_andnot<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_xor(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_xor<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_xor(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_xor<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_not(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_not<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_not(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_not<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_neg(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_neg<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_neg(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_neg<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_min(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_min<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_min(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_min<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_max(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_max<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_max(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_max<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_div2r0(const Vec<T, SIMD_WIDTH> &,
                                                  const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_div2r0<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_div2r0(const Mask<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_div2r0<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_div2rd(const Vec<T, SIMD_WIDTH> &,
                                                  const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_div2rd<%s,%d>(V,M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_div2rd(const Mask<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_div2rd<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_srai(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_srai<%d,%s,%d>(V,M,V)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_srai(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_srai<%d,%s,%d>(M,V)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_srli(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_srli<%d,%s,%d>(V,M,V)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_srli(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_srli<%d,%s,%d>(M,V)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_slli(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_slli<%d,%s,%d>(V,M,V)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_slli(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_slli<%d,%s,%d>(M,V)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hadd(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_hadd<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hadd(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_hadd<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hadds(const Vec<T, SIMD_WIDTH> &,
                                                 const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_hadds<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hadds(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_hadds<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hsub(const Vec<T, SIMD_WIDTH> &,
                                                const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_hsub<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hsub(const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_hsub<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_hsubs(const Vec<T, SIMD_WIDTH> &,
                                                 const Mask<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_hsubs<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_hsubs(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_hsubs<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmplt(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmplt<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmple(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmple<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpeq(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpeq<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpgt(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpgt<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpge(const Mask<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpge<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpneq(const Mask<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpneq<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

// 12. Jan 23 (Jonas Keller): added overloaded versions of mask_cmp*
// that only take two vector parameters and no mask parameter

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmplt(const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmplt<%s,%d>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmple(const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmple<%s,%d>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpeq(const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpeq<%s,%d>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpgt(const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpgt<%s,%d>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpge(const Vec<T, SIMD_WIDTH> &,
                                                  const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpge<%s,%d>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_cmpneq(const Vec<T, SIMD_WIDTH> &,
                                                   const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpneq<%s,%d>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mask_avg(const Vec<T, SIMD_WIDTH> &,
                                               const Mask<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &)
{
  printf("mask_avg<%s,%d>(V,M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> maskz_avg(const Mask<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &,
                                                const Vec<T, SIMD_WIDTH> &)
{
  printf("maskz_avg<%s,%d>(M,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_zeros(const Mask<T, SIMD_WIDTH> &k,
                                           const Vec<T, SIMD_WIDTH> &a)
{
  (void) k;
  (void) a;
  printf("mask_test_all_zeros<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return 0;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_ones(const Mask<T, SIMD_WIDTH> &k,
                                          const Vec<T, SIMD_WIDTH> &a)
{
  (void) k;
  (void) a;
  printf("mask_test_all_ones<%s,%d>(M,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return 0;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kadd(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kadd<%s,%d>(M,M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kand(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kand<%s,%d>(M,M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kandn(const Mask<T, SIMD_WIDTH> &a,
                                             const Mask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kandn<%s,%d>(M,M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kor(const Mask<T, SIMD_WIDTH> &a,
                                           const Mask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kor<%s,%d>(M,M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kxor(const Mask<T, SIMD_WIDTH> &a,
                                            const Mask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kxor<%s,%d>(M,M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kxnor(const Mask<T, SIMD_WIDTH> &a,
                                             const Mask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kxnor<%s,%d>(M,M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> knot(const Mask<T, SIMD_WIDTH> &a)
{
  (void) a;
  printf("knot<%s,%d>(M)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Mask<T, SIMD_WIDTH>();
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftri(const Mask<T, SIMD_WIDTH> &a)
{
  (void) a;
  printf("kshiftri<%d, %s,%d>(M,%d)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH,
         IMM);
  return Mask<T, SIMD_WIDTH>();
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftli(const Mask<T, SIMD_WIDTH> &a)
{
  (void) a;
  printf("kshiftli<%d, %s,%d>(M,%d)\n", IMM, TypeInfo<T>::name(), SIMD_WIDTH,
         IMM);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftri(const Mask<T, SIMD_WIDTH> &a,
                                                const uint64_t count)
{
  (void) a;
  printf("kshiftri<%s,%d>(M,%d)\n", TypeInfo<T>::name(), SIMD_WIDTH, count);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> kshiftli(const Mask<T, SIMD_WIDTH> &a,
                                                const uint64_t count)
{
  (void) a;
  printf("kshiftli<%s,%d>(M,%d)\n", TypeInfo<T>::name(), SIMD_WIDTH, count);
  return Mask<T, SIMD_WIDTH>();
}

// 12. Jan 23 (Jonas Keller): add setTrueRight and setTrueLeft
// 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
// mask_set_true/false_low/high.

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_true_low(const unsigned int x,
                                                         OutputType<T>,
                                                         Integer<SIMD_WIDTH>)
{
  printf("mask_set_true_low<%s,%d>(%d)\n", TypeInfo<T>::name(), SIMD_WIDTH, x);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_true_high(const unsigned int x,
                                                          OutputType<T>,
                                                          Integer<SIMD_WIDTH>)
{
  printf("mask_set_true_high<%s,%d>(%d)\n", TypeInfo<T>::name(), SIMD_WIDTH, x);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_false_low(const unsigned int x,
                                                          OutputType<T>,
                                                          Integer<SIMD_WIDTH>)
{
  printf("mask_set_false_low<%s,%d>(%d)\n", TypeInfo<T>::name(), SIMD_WIDTH, x);
  return Mask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE Mask<T, SIMD_WIDTH> mask_set_false_high(const unsigned int x,
                                                           OutputType<T>,
                                                           Integer<SIMD_WIDTH>)
{
  printf("mask_set_false_high<%s,%d>(%d)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         x);
  return Mask<T, SIMD_WIDTH>();
}
} // namespace mask
} // namespace internal
} // namespace simd

#endif // SIMDVEC_SANDBOX

#endif // _SIMD_VEC_MASK_IMPL_SANDBOX_H_
