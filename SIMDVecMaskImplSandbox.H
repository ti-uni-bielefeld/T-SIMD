// ===========================================================================
//
// SIMDVecMaskImplSandbox.H --
// test template functions for SIMDMask and associated function templates
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

namespace ns_simd {
// ===========================================================================
// generic template for SIMDMask
// ===========================================================================

template <typename T, int SIMD_WIDTH>
class SIMDMask
{
public:
  SIMDMask() {}
  SIMDMask(const uint64_t &) {}
  operator uint64_t() const { return 0; }
};

namespace internal {
namespace mask {
// ===========================================================================
// masked functions
// ===========================================================================

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_all_ones(OutputType<T>,
                                                         Integer<SIMD_WIDTH>)
{
  printf("mask_all_ones<%s,%d>()\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_ifelse(
  const SIMDMask<T, SIMD_WIDTH> &cond, const SIMDVec<T, SIMD_WIDTH> &trueVal,
  const SIMDVec<T, SIMD_WIDTH> &falseVal)
{
  (void) cond;
  (void) trueVal;
  (void) falseVal;
  printf("mask_ifelse<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

// mask_ifelsezero
template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_ifelsezero(
  const SIMDMask<T, SIMD_WIDTH> &cond, const SIMDVec<T, SIMD_WIDTH> &trueVal)
{
  (void) cond;
  (void) trueVal;
  printf("mask_ifelsezero<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename Tout, typename Tin, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<Tout, SIMD_WIDTH> reinterpret_mask(
  const SIMDMask<Tin, SIMD_WIDTH> &)
{
  printf("reinterpret_mask<%s,%s,%d>(M)\n", SIMDTypeInfo<Tout>::name(),
         SIMDTypeInfo<Tin>::name(), SIMD_WIDTH);
  return SIMDMask<Tout, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH> mask_cvts(
  const SIMDVec<SIMDInt, SIMD_WIDTH> &, const SIMDMask<SIMDFloat, SIMD_WIDTH> &,
  const SIMDVec<SIMDFloat, SIMD_WIDTH> &)
{
  printf("mask_cvts<%s,%s,%d>(V,M,V)\n", SIMDTypeInfo<SIMDInt>::name(),
         SIMDTypeInfo<SIMDFloat>::name(), SIMD_WIDTH);
  return SIMDVec<SIMDInt, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH> maskz_cvts(
  const SIMDMask<SIMDFloat, SIMD_WIDTH> &,
  const SIMDVec<SIMDFloat, SIMD_WIDTH> &)
{
  printf("maskz_cvts<%s,%s,%d>(M,V)\n", SIMDTypeInfo<SIMDInt>::name(),
         SIMDTypeInfo<SIMDFloat>::name(), SIMD_WIDTH);
  return SIMDVec<SIMDInt, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH> mask_cvts(
  const SIMDVec<SIMDFloat, SIMD_WIDTH> &, const SIMDMask<SIMDInt, SIMD_WIDTH> &,
  const SIMDVec<SIMDInt, SIMD_WIDTH> &)
{
  printf("mask_cvts<%s,%s,%d>(V,M,V)\n", SIMDTypeInfo<SIMDFloat>::name(),
         SIMDTypeInfo<SIMDInt>::name(), SIMD_WIDTH);
  return SIMDVec<SIMDFloat, SIMD_WIDTH>();
}

template <int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH> maskz_cvts(
  const SIMDMask<SIMDInt, SIMD_WIDTH> &, const SIMDVec<SIMDInt, SIMD_WIDTH> &)
{
  printf("maskz_cvts<%s,%s,%d>(M,V)\n", SIMDTypeInfo<SIMDFloat>::name(),
         SIMDTypeInfo<SIMDInt>::name(), SIMD_WIDTH);
  return SIMDVec<SIMDFloat, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_set1(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &, const T &)
{
  printf("mask_set1<%s,%d>(V,M,T)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_set1(
  const SIMDMask<T, SIMD_WIDTH> &, const T &)
{
  printf("maskz_set1<%s,%d>(M,T)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_load(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const T *const p)
{
  printf("mask_load<%s,%d>(V,M,%p)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_load(
  const SIMDMask<T, SIMD_WIDTH> &, const T *const p)
{
  printf("maskz_load<%s,%d>(M,%p)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_loadu(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const T *const p)
{
  printf("mask_loadu<%s,%d>(V,M,%p)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_loadu(
  const SIMDMask<T, SIMD_WIDTH> &, const T *const p)
{
  printf("maskz_loadu<%s,%d>(M,%p)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_store(T *const p, const SIMDMask<T, SIMD_WIDTH> &,
                                   const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_store<%s,%d>(%p,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE void mask_storeu(T *const p, const SIMDMask<T, SIMD_WIDTH> &,
                                    const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_storeu<%s,%d>(%p,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_add(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_add<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_add(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_add<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_adds(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_adds<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_adds(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_adds<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_sub(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_sub<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_sub(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_sub<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_subs(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_subs<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_subs(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_subs<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_mul(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_mul<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_mul(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_mul<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_div(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_div<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_div(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_div<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_ceil(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_ceil<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_ceil(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_ceil<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_floor(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_floor<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_floor(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_floor<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_round(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_round<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_round(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_round<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_truncate(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_truncate<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_truncate(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_truncate<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_rcp(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_rcp<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_rcp(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_rcp<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_rsqrt(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_rsqrt<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_rsqrt(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_rsqrt<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_sqrt(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_sqrt<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_sqrt(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_sqrt<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_abs(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_abs<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_abs(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_abs<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_and(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_and<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_and(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_and<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_or(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_or<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_or(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_or<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_andnot(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_andnot<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_andnot(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_andnot<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_xor(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_xor<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_xor(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_xor<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_not(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_not<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_not(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_not<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_neg(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_neg<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_neg(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_neg<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_min(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_min<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_min(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_min<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_max(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_max<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_max(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_max<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_div2r0(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_div2r0<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_div2r0(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_div2r0<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_div2rd(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_div2rd<%s,%d>(V,M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_div2rd(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_div2rd<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_srai(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_srai<%d,%s,%d>(V,M,V)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_srai(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_srai<%d,%s,%d>(M,V)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_srli(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_srli<%d,%s,%d>(V,M,V)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_srli(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_srli<%d,%s,%d>(M,V)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_slli(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_slli<%d,%s,%d>(V,M,V)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_slli(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_slli<%d,%s,%d>(M,V)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hadd(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_hadd<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hadd(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_hadd<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hadds(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_hadds<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hadds(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_hadds<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hsub(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_hsub<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hsub(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_hsub<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_hsubs(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_hsubs<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_hsubs(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_hsubs<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmplt(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmplt<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmple(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmple<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpeq(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpeq<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpgt(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpgt<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpge(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpge<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpneq(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpneq<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

// 12. Jan 23 (Jonas Keller): added overloaded versions of mask_cmp*
// that only take two vector parameters and no mask parameter

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmplt(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmplt<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmple(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmple<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpeq(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpeq<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpgt(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpgt<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpge(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpge<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_cmpneq(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_cmpneq<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> mask_avg(
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDMask<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("mask_avg<%s,%d>(V,M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> maskz_avg(
  const SIMDMask<T, SIMD_WIDTH> &, const SIMDVec<T, SIMD_WIDTH> &,
  const SIMDVec<T, SIMD_WIDTH> &)
{
  printf("maskz_avg<%s,%d>(M,V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDVec<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_zeros(const SIMDMask<T, SIMD_WIDTH> &k,
                                           const SIMDVec<T, SIMD_WIDTH> &a)
{
  (void) k;
  (void) a;
  printf("mask_test_all_zeros<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return 0;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE int mask_test_all_ones(const SIMDMask<T, SIMD_WIDTH> &k,
                                          const SIMDVec<T, SIMD_WIDTH> &a)
{
  (void) k;
  (void) a;
  printf("mask_test_all_ones<%s,%d>(M,V)\n", SIMDTypeInfo<T>::name(),
         SIMD_WIDTH);
  return 0;
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kadd(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kadd<%s,%d>(M,M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kand(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kand<%s,%d>(M,M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kandn(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kandn<%s,%d>(M,M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kor(const SIMDMask<T, SIMD_WIDTH> &a,
                                               const SIMDMask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kor<%s,%d>(M,M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kxor(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kxor<%s,%d>(M,M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kxnor(
  const SIMDMask<T, SIMD_WIDTH> &a, const SIMDMask<T, SIMD_WIDTH> &b)
{
  (void) a;
  (void) b;
  printf("kxnor<%s,%d>(M,M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> knot(
  const SIMDMask<T, SIMD_WIDTH> &a)
{
  (void) a;
  printf("knot<%s,%d>(M)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftri(
  const SIMDMask<T, SIMD_WIDTH> &a)
{
  (void) a;
  printf("kshiftri<%d, %s,%d>(M,%d)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH, IMM);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftli(
  const SIMDMask<T, SIMD_WIDTH> &a)
{
  (void) a;
  printf("kshiftli<%d, %s,%d>(M,%d)\n", IMM, SIMDTypeInfo<T>::name(),
         SIMD_WIDTH, IMM);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftri(
  const SIMDMask<T, SIMD_WIDTH> &a, const uint64_t count)
{
  (void) a;
  printf("kshiftri<%s,%d>(M,%d)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH, count);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> kshiftli(
  const SIMDMask<T, SIMD_WIDTH> &a, const uint64_t count)
{
  (void) a;
  printf("kshiftli<%s,%d>(M,%d)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH, count);
  return SIMDMask<T, SIMD_WIDTH>();
}

// 12. Jan 23 (Jonas Keller): add setTrueRight and setTrueLeft
// 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
// mask_set_true/false_low/high.

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_true_low(
  const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>)
{
  printf("mask_set_true_low<%s,%d>(%d)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         x);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_true_high(
  const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>)
{
  printf("mask_set_true_high<%s,%d>(%d)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         x);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_false_low(
  const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>)
{
  printf("mask_set_false_low<%s,%d>(%d)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH,
         x);
  return SIMDMask<T, SIMD_WIDTH>();
}

template <typename T, int SIMD_WIDTH>
static SIMD_INLINE SIMDMask<T, SIMD_WIDTH> mask_set_false_high(
  const unsigned int x, OutputType<T>, Integer<SIMD_WIDTH>)
{
  printf("mask_set_false_high<%s,%d>(%d)\n", SIMDTypeInfo<T>::name(),
         SIMD_WIDTH, x);
  return SIMDMask<T, SIMD_WIDTH>();
}
} // namespace mask
} // namespace internal
} // namespace ns_simd

#endif // SIMDVEC_SANDBOX

#endif // _SIMD_VEC_MASK_IMPL_SANDBOX_H_
