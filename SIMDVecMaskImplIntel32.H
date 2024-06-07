// ===========================================================================
//
// SIMDVecMaskImplIntel32.H --
// Mask class definitions and architecture specific functions
// for Intel 32 byte (256 bit)
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

// 05. Jun 24 (Jonas Keller): added Intel32 implementations of masked load and
// store functions for 32 bit and 64 bit data types

#pragma once
#ifndef SIMD_VEC_MASK_IMPL_INTEL_32_H_
#define SIMD_VEC_MASK_IMPL_INTEL_32_H_

#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecMaskImplEmu.H"

#if defined(SIMDVEC_INTEL_ENABLE) && defined(_SIMD_VEC_32_AVAIL_) &&           \
  !defined(SIMDVEC_SANDBOX)
namespace simd {
namespace internal {
namespace mask {
template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4)>
static SIMD_INLINE Vec<T, 32> maskz_loadu(const Mask<T, 32> &k,
                                          const T *const p)
{
  return reinterpret<T>(Vec<Float, 32>(_mm256_maskload_ps(
    reinterpret_cast<const Float *>(p), reinterpret<Int>((Vec<T, 32>) k))));
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void>
static SIMD_INLINE Vec<T, 32> maskz_loadu(const Mask<T, 32> &k,
                                          const T *const p)
{
  return reinterpret<T>(Vec<Double, 32>(_mm256_maskload_pd(
    reinterpret_cast<const Double *>(p), reinterpret<Long>((Vec<T, 32>) k))));
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4 || sizeof(T) == 8)>
static SIMD_INLINE Vec<T, 32> mask_loadu(const Vec<T, 32> &src,
                                         const Mask<T, 32> &k, const T *const p)
{
  return mask::mask_ifelse(k, mask::maskz_loadu(k, p), src);
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4 || sizeof(T) == 8)>
static SIMD_INLINE Vec<T, 32> maskz_load(const Mask<T, 32> &k, const T *const p)
{
  return mask::maskz_loadu(k, p);
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4 || sizeof(T) == 8)>
static SIMD_INLINE Vec<T, 32> mask_load(const Vec<T, 32> &src,
                                        const Mask<T, 32> &k, const T *const p)
{
  return mask::mask_loadu(src, k, p);
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4)>
static SIMD_INLINE void mask_storeu(T *const p, const Mask<T, 32> &k,
                                    const Vec<T, 32> &a)
{
  _mm256_maskstore_ps(reinterpret_cast<Float *>(p),
                      reinterpret<Int>((Vec<T, 32>) k), reinterpret<Float>(a));
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 8), typename = void>
static SIMD_INLINE void mask_storeu(T *const p, const Mask<T, 32> &k,
                                    const Vec<T, 32> &a)
{
  _mm256_maskstore_pd(reinterpret_cast<Double *>(p),
                      reinterpret<Long>((Vec<T, 32>) k),
                      reinterpret<Double>(a));
}

template <typename T, SIMD_ENABLE_IF(sizeof(T) == 4 || sizeof(T) == 8)>
static SIMD_INLINE void mask_store(T *const p, const Mask<T, 32> &k,
                                   const Vec<T, 32> &a)
{
  mask::mask_storeu(p, k, a);
}

} // namespace mask
} // namespace internal
} // namespace simd
#endif

#endif // SIMD_VEC_MASK_IMPL_INTEL_32_H_
