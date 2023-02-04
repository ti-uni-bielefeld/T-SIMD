// ===========================================================================
// 
// SIMDVecMaskImplIntel16.H --
// Mask class definitions and architecture specific functions
// for Intel 16 byte (128 bit)
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

#ifndef _SIMD_VEC_MASK_IMPL_INTEL_16_H_
#define _SIMD_VEC_MASK_IMPL_INTEL_16_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecBaseImplIntel16.H"

#include <stddef.h>
#include <stdint.h>

#ifdef __SSE2__

namespace ns_simd {
  // 10. Oct 22 (Jonas Keller):
  // Combined all integer mask classes into one generic one and used the
  // int2bits and msb2int functions for conversion from/to an integer.
  template <typename T>
  class SIMDMask<T, 16>
  {
  public:
    __m128i k;
    SIMDMask() { k = _mm_setzero_si128(); }
    SIMDMask(const __m128i &x) { k = x; }
    SIMDMask(const uint64_t x) { k = int2bits<T, 16>(x); }
    SIMDMask &operator=(const __m128i &x) {
      k = x;
      return *this;
    }
    SIMDMask &operator=(const uint64_t x) {
      k = int2bits<T, 16>(x);
      return *this;
    }
    operator __m128i() const { return k; }
    operator uint64_t() const { return msb2int(SIMDVec<T, 16>(k)); }
    bool operator[](const uint8_t i) const {
      if (i >= 16) {
        return false;
      }
      return ((1 << i) & msb2int(SIMDVec<T, 16>(k))) != 0;
    }
    bool operator==(const SIMDMask<T, 16> &x) const {
      return _mm_movemask_epi8(_mm_cmpeq_epi8(k, x)) == 0xffff;
    }
    // 29. Nov 22 (Jonas Keller):
    // defined operators new and delete to ensure proper alignment, since
    // the default new and delete are not guaranteed to do so before C++17
    void *operator new(size_t size)
    { return simd_aligned_malloc(sizeof(__m128i), size); }
    void operator delete(void *p)
    { simd_aligned_free(p); }
    void *operator new[](size_t size)
    { return simd_aligned_malloc(sizeof(__m128i), size); }
    void operator delete[](void *p)
    { simd_aligned_free(p); }
  };

  template <>
  class SIMDMask<SIMDFloat, 16>
  {
  public:
    __m128 k;
    SIMDMask() { k = _mm_setzero_ps(); }
    SIMDMask(const __m128 &x) { k = x; }
    SIMDMask(const uint64_t x) { k = int2bits<SIMDFloat, 16>(x); }
    SIMDMask &operator=(const __m128 &x) {
      k = x;
      return *this;
    }
    SIMDMask &operator=(const uint64_t x) {
      k = int2bits<SIMDFloat, 16>(x);
      return *this;
    }
    operator __m128() const { return k; }
    operator uint64_t() const { return msb2int(SIMDVec<SIMDFloat, 16>(k)); }
    bool operator[](const uint8_t i) const {
      if (i >= 16) {
        return false;
      }
      return ((1 << i) & msb2int(SIMDVec<SIMDFloat, 16>(k))) != 0;
    }
    bool operator==(const SIMDMask<SIMDFloat, 16> &x) const {
      return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_castps_si128(k),
                                              _mm_castps_si128(x))) == 0xffff;
    }
    // 29. Nov 22 (Jonas Keller):
    // defined operators new and delete to ensure proper alignment, since
    // the default new and delete are not guaranteed to do so before C++17
    void *operator new(size_t size)
    { return simd_aligned_malloc(sizeof(__m128), size); }
    void operator delete(void *p)
    { simd_aligned_free(p); }
    void *operator new[](size_t size)
    { return simd_aligned_malloc(sizeof(__m128), size); }
    void operator delete[](void *p)
    { simd_aligned_free(p); }
  };
} // namespace ns_simd

#endif // ifdef __SSE2__

#endif // _SIMD_VEC_MASK_IMPL_INTEL_16_H_
