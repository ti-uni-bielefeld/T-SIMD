// ===========================================================================
// 
// SIMDVecMaskImplIntel32.H --
// Mask class definitions and architecture specific functions
// for Intel 32 byte (256 bit)
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

#ifndef _SIMD_VEC_MASK_IMPL_INTEL_32_H_
#define _SIMD_VEC_MASK_IMPL_INTEL_32_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecBaseImplIntel32.H"
#include "SIMDVecMaskImplIntel16.H"

#include <stddef.h>
#include <stdint.h>

#ifdef __AVX__

namespace ns_simd {
  // 10. Oct 22 (Jonas Keller):
  // Combined all integer mask classes into one generic one and used the
  // int2bits and msb2int functions for conversion from/to an integer.
  template <typename T>
  class SIMDMask<T, 32>
  {
  public:
    __m256i k;
    SIMDMask() { k = _mm256_setzero_si256(); }
    SIMDMask(const __m256i &x) { k = x; }
    SIMDMask(const uint64_t x) { k = int2bits<T, 32>(x); }
    SIMDMask &operator=(const __m256i &x) {
      k = x;
      return *this;
    }
    SIMDMask &operator=(const uint64_t x) {
      k = int2bits<T, 32>(x);
      return *this;
    }
    operator __m256i() const { return k; }
    operator uint64_t() const { return msb2int(SIMDVec<T, 32>(k)); }
    bool operator[](const uint8_t i) const {
      if (i >= 32) {
        return false;
      }
      return ((1 << i) & msb2int(SIMDVec<T, 32>(k))) != 0;
    }
    bool operator==(const SIMDMask<T, 32> &x) const {
      return test_all_zeros(
          xor(SIMDVec<SIMDByte, 32>(k), SIMDVec<SIMDByte, 32>(x)));
    }
    // 29. Nov 22 (Jonas Keller):
    // defined operators new and delete to ensure proper alignment, since
    // the default new and delete are not guaranteed to do so before C++17
    void *operator new(size_t size) const
    { return simd_aligned_malloc(sizeof(__m256i), size); }
    void operator delete(void *p) const
    { simd_aligned_free(p); }
    void *operator new[](size_t size) const
    { return simd_aligned_malloc(sizeof(__m256i), size); }
    void operator delete[](void *p) const
    { simd_aligned_free(p); }
  };

  template <>
  class SIMDMask<SIMDFloat, 32>
  {
  public:
    __m256 k;
    SIMDMask() { k = _mm256_setzero_ps(); }
    SIMDMask(const __m256 &x) { k = x; }
    SIMDMask(const uint64_t x) { k = int2bits<SIMDFloat, 32>(x); }
    SIMDMask &operator=(const __m256 &x) {
      k = x;
      return *this;
    }
    SIMDMask &operator=(const uint64_t x) {
      k = int2bits<SIMDFloat, 32>(x);
      return *this;
    }
    operator __m256() const { return k; }
    operator uint64_t() const { return msb2int(SIMDVec<SIMDFloat, 32>(k)); }
    bool operator[](const uint8_t i) const {
      if (i >= 8) {
        return false;
      }
      return ((1 << i) & msb2int(SIMDVec<SIMDFloat, 32>(k))) != 0;
    }
    bool operator==(const SIMDMask<SIMDFloat, 32> &x) const {
      return ::ns_simd::test_all_zeros<SIMDFloat, 32>(_mm256_xor_ps(k, x));
    }
    // 29. Nov 22 (Jonas Keller):
    // defined operators new and delete to ensure proper alignment, since
    // the default new and delete are not guaranteed to do so before C++17
    void *operator new(size_t size) const
    { return simd_aligned_malloc(sizeof(__m256), size); }
    void operator delete(void *p) const
    { simd_aligned_free(p); }
    void *operator new[](size_t size) const
    { return simd_aligned_malloc(sizeof(__m256), size); }
    void operator delete[](void *p) const
    { simd_aligned_free(p); }
  };
} // namespace ns_simd

#endif // ifdef __AVX__

#endif // _SIMD_VEC_MASK_IMPL_INTEL_32_H_
