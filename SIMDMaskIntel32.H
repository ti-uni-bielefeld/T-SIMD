// ===========================================================================
// 
// SIMDMaskIntel32.H --
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

#ifndef _SIMDMASKINTEL32_H_
#define _SIMDMASKINTEL32_H_

#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDMaskIntel16.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecIntel32.H"

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
    bool operator==(const SIMDMask<T, 32> &x) {
      return test_all_zeros(
          xor(SIMDVec<SIMDByte, 32>(k), SIMDVec<SIMDByte, 32>(x)));
    }
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
    bool operator==(const SIMDMask<SIMDFloat, 32> &x) {
      return test_all_zeros(_mm256_xor_ps(k, x));
    }
  };
  
  template <>
  SIMD_INLINE SIMDMask<SIMDByte, 32>
  mask_all_ones()
  {
    return _mm256_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDSignedByte, 32>
  mask_all_ones()
  {
    return _mm256_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDWord, 32>
  mask_all_ones()
  {
    return _mm256_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDShort, 32>
  mask_all_ones()
  {
    return _mm256_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDInt, 32>
  mask_all_ones()
  {
    return _mm256_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDFloat, 32>
  mask_all_ones()
  {
    return _mm256_castsi256_ps(_mm256_set1_epi8(-1));
  }
  
//#ifdef __AVX__
  #define CHECK_AND_INSERT(OP, INDEX) \
  if(k[INDEX]) { \
    result = OP (result, p[INDEX], INDEX); \
  }
  
  #define CHECK_AND_INSERT2(MASKNAME, OP, INDEX) \
  if(MASKNAME [INDEX]) { \
    result = OP (result, p[INDEX], INDEX); \
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte, 32>
  maskz_load(const SIMDMask<SIMDSignedByte, 32> &k,
      const SIMDSignedByte *const p)
  {
    SIMDVec<SIMDSignedByte, 32> result=setzero<SIMDSignedByte, 32>();
    CHECK_AND_INSERT(_mm256_insert_epi8, 0)
    CHECK_AND_INSERT(_mm256_insert_epi8, 1)
    CHECK_AND_INSERT(_mm256_insert_epi8, 2)
    CHECK_AND_INSERT(_mm256_insert_epi8, 3)
    CHECK_AND_INSERT(_mm256_insert_epi8, 4)
    CHECK_AND_INSERT(_mm256_insert_epi8, 5)
    CHECK_AND_INSERT(_mm256_insert_epi8, 6)
    CHECK_AND_INSERT(_mm256_insert_epi8, 7)
    CHECK_AND_INSERT(_mm256_insert_epi8, 8)
    CHECK_AND_INSERT(_mm256_insert_epi8, 9)
    CHECK_AND_INSERT(_mm256_insert_epi8, 10)
    CHECK_AND_INSERT(_mm256_insert_epi8, 11)
    CHECK_AND_INSERT(_mm256_insert_epi8, 12)
    CHECK_AND_INSERT(_mm256_insert_epi8, 13)
    CHECK_AND_INSERT(_mm256_insert_epi8, 14)
    CHECK_AND_INSERT(_mm256_insert_epi8, 15)
    CHECK_AND_INSERT(_mm256_insert_epi8, 16)
    CHECK_AND_INSERT(_mm256_insert_epi8, 17)
    CHECK_AND_INSERT(_mm256_insert_epi8, 18)
    CHECK_AND_INSERT(_mm256_insert_epi8, 19)
    CHECK_AND_INSERT(_mm256_insert_epi8, 20)
    CHECK_AND_INSERT(_mm256_insert_epi8, 21)
    CHECK_AND_INSERT(_mm256_insert_epi8, 22)
    CHECK_AND_INSERT(_mm256_insert_epi8, 23)
    CHECK_AND_INSERT(_mm256_insert_epi8, 24)
    CHECK_AND_INSERT(_mm256_insert_epi8, 25)
    CHECK_AND_INSERT(_mm256_insert_epi8, 26)
    CHECK_AND_INSERT(_mm256_insert_epi8, 27)
    CHECK_AND_INSERT(_mm256_insert_epi8, 28)
    CHECK_AND_INSERT(_mm256_insert_epi8, 29)
    CHECK_AND_INSERT(_mm256_insert_epi8, 30)
    CHECK_AND_INSERT(_mm256_insert_epi8, 31)
    return result;
  }
  static SIMD_INLINE SIMDVec<SIMDByte, 32>
  maskz_load(const SIMDMask<SIMDByte, 32> &k,
      const SIMDByte *const p)
  {
    return reinterpret<SIMDByte>(maskz_load(SIMDMask<SIMDSignedByte, 32>((__m256i) k), (const SIMDSignedByte *) p));
  }
  static SIMD_INLINE SIMDVec<SIMDShort, 32>
  maskz_load(const SIMDMask<SIMDShort, 32> &k,
      const SIMDShort *const p)
  {
//#ifdef BENCH_METHOD1
#if __GNUC__ < 8
    SIMDVec<SIMDShort, 32> result=setzero<SIMDShort, 32>();
    CHECK_AND_INSERT(_mm256_insert_epi16, 0)
    CHECK_AND_INSERT(_mm256_insert_epi16, 1)
    CHECK_AND_INSERT(_mm256_insert_epi16, 2)
    CHECK_AND_INSERT(_mm256_insert_epi16, 3)
    CHECK_AND_INSERT(_mm256_insert_epi16, 4)
    CHECK_AND_INSERT(_mm256_insert_epi16, 5)
    CHECK_AND_INSERT(_mm256_insert_epi16, 6)
    CHECK_AND_INSERT(_mm256_insert_epi16, 7)
    CHECK_AND_INSERT(_mm256_insert_epi16, 8)
    CHECK_AND_INSERT(_mm256_insert_epi16, 9)
    CHECK_AND_INSERT(_mm256_insert_epi16, 10)
    CHECK_AND_INSERT(_mm256_insert_epi16, 11)
    CHECK_AND_INSERT(_mm256_insert_epi16, 12)
    CHECK_AND_INSERT(_mm256_insert_epi16, 13)
    CHECK_AND_INSERT(_mm256_insert_epi16, 14)
    CHECK_AND_INSERT(_mm256_insert_epi16, 15)
    return result;
#endif
/*#endif
#ifdef BENCH_METHOD2
    if(test_all_ones(SIMDVec<SIMDShort, 32>(k))) { return _mm256_load_si256((__m256i*) p); }
    if(test_all_zeros(SIMDVec<SIMDShort, 32>(k))) { return setzero<SIMDShort, 32>(); }
    return _mm256_set_epi16((k[15]?p[15]:0), (k[14]?p[14]:0), (k[13]?p[13]:0), (k[12]?p[12]:0), (k[11]?p[11]:0), (k[10]?p[10]:0), (k[9]?p[9]:0), (k[8]?p[8]:0), (k[7]?p[7]:0), (k[6]?p[6]:0), (k[5]?p[5]:0), (k[4]?p[4]:0), (k[3]?p[3]:0), (k[2]?p[2]:0), (k[1]?p[1]:0), (k[0]?p[0]:0));
#endif
#ifdef BENCH_METHOD3
    if(test_all_ones(SIMDVec<SIMDShort, 32>(k))) { return _mm256_load_si256((__m256i*) p); }
    if(test_all_zeros(SIMDVec<SIMDShort, 32>(k))) { return setzero<SIMDShort, 32>(); }
    
    __m256i mask32=_mm256_cmpeq_epi32(k, _mm256_set1_epi32(0xffffffff));
    __m256i result=_mm256_maskload_epi32 ((int const *) p, mask32);
    SIMDMask<SIMDShort, 32> residuals_mask=SIMDMask<SIMDShort, 32>(andnot(SIMDVec<SIMDShort, 32>(mask32), SIMDVec<SIMDShort, 32>(k)));
    if(test_all_zeros(SIMDVec<SIMDShort, 32>(residuals_mask))) { return result; }
    
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 0)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 1)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 2)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 3)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 4)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 5)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 6)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 7)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 8)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 9)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 10)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 11)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 12)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 13)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 14)
    CHECK_AND_INSERT2(residuals_mask, _mm256_insert_epi16, 15)
    return result;
#endif
#ifdef BENCH_METHOD4*/
#if __GNUC__ >= 8
    return _mm256_set_m128i(maskz_load(_mm256_extractf128_si256(k, 1), p+8), maskz_load(_mm256_extractf128_si256(k, 0), p));
#endif
//#endif
  }
  static SIMD_INLINE SIMDVec<SIMDWord, 32>
  maskz_load(const SIMDMask<SIMDWord, 32> &k,
      const SIMDWord *const p)
  {
    return reinterpret<SIMDWord>(maskz_load(SIMDMask<SIMDShort, 32>((__m256i) k), (const SIMDShort *) p));
  }
  static SIMD_INLINE SIMDVec<SIMDInt, 32>
  maskz_load(const SIMDMask<SIMDInt, 32> &k,
      const SIMDInt *const p)
  {
//#ifdef BENCH_METHOD1
#ifndef __AVX2__
    SIMDVec<SIMDInt, 32> result=setzero<SIMDInt, 32>();
    CHECK_AND_INSERT(_mm256_insert_epi32, 0)
    CHECK_AND_INSERT(_mm256_insert_epi32, 1)
    CHECK_AND_INSERT(_mm256_insert_epi32, 2)
    CHECK_AND_INSERT(_mm256_insert_epi32, 3)
    CHECK_AND_INSERT(_mm256_insert_epi32, 4)
    CHECK_AND_INSERT(_mm256_insert_epi32, 5)
    CHECK_AND_INSERT(_mm256_insert_epi32, 6)
    CHECK_AND_INSERT(_mm256_insert_epi32, 7)
    return result;
#endif
/*#endif
#ifdef BENCH_METHOD2
    return _mm256_set_ps((k[7]?p[7]:0), (k[6]?p[6]:0), (k[5]?p[5]:0), (k[4]?p[4]:0), (k[3]?p[3]:0), (k[2]?p[2]:0), (k[1]?p[1]:0), (k[0]?p[0]:0));
#endif
#ifdef BENCH_METHOD3*/
#ifdef __AVX2__
    return _mm256_maskload_epi32(p, k);
#endif
//#endif
  }
  static SIMD_INLINE SIMDVec<SIMDFloat, 32>
  maskz_load(const SIMDMask<SIMDFloat, 32> &k,
      const SIMDFloat *const p)
  {
    return reinterpret<SIMDFloat>(maskz_load(_mm256_castps_si256(k), (const SIMDInt *) p));
  }
  
  #define CHECK_AND_STORE(INDEX) \
  if(k [INDEX]) { p[INDEX] = extract<INDEX> (a); }
  
  #define CHECK_AND_STORE4 CHECK_AND_STORE(0) CHECK_AND_STORE(1) CHECK_AND_STORE(2) CHECK_AND_STORE(3)
  
  #define CHECK_AND_STORE8 CHECK_AND_STORE4 CHECK_AND_STORE(4) CHECK_AND_STORE(5) CHECK_AND_STORE(6) CHECK_AND_STORE(7)
  
  #define CHECK_AND_STORE16 CHECK_AND_STORE8 CHECK_AND_STORE(8) CHECK_AND_STORE(9) CHECK_AND_STORE(10) CHECK_AND_STORE(11) CHECK_AND_STORE(12) CHECK_AND_STORE(13) CHECK_AND_STORE(14) CHECK_AND_STORE(15) 
  
  #define CHECK_AND_STORE32 CHECK_AND_STORE16 CHECK_AND_STORE(16) CHECK_AND_STORE(17) CHECK_AND_STORE(18) CHECK_AND_STORE(19) CHECK_AND_STORE(20) CHECK_AND_STORE(21) CHECK_AND_STORE(22) CHECK_AND_STORE(23) CHECK_AND_STORE(24) CHECK_AND_STORE(25) CHECK_AND_STORE(26) CHECK_AND_STORE(27) CHECK_AND_STORE(28) CHECK_AND_STORE(29) CHECK_AND_STORE(30) CHECK_AND_STORE(31) 
  
  static SIMD_INLINE void
  mask_store(SIMDByte *const p,
      const SIMDMask<SIMDByte, 32> &k,
      const SIMDVec<SIMDByte, 32> &a)
  {
    CHECK_AND_STORE32
  }
  static SIMD_INLINE void
  mask_store(SIMDSignedByte *const p,
      const SIMDMask<SIMDSignedByte, 32> &k,
      const SIMDVec<SIMDSignedByte, 32> &a)
  {
    CHECK_AND_STORE32
  }
  static SIMD_INLINE void
  mask_store(SIMDWord *const p,
      const SIMDMask<SIMDWord, 32> &k,
      const SIMDVec<SIMDWord, 32> &a)
  {
    CHECK_AND_STORE16
  }
  static SIMD_INLINE void
  mask_store(SIMDShort *const p,
      const SIMDMask<SIMDShort, 32> &k,
      const SIMDVec<SIMDShort, 32> &a)
  {
    CHECK_AND_STORE16
  }
  static SIMD_INLINE void
  mask_store(SIMDInt *const p,
      const SIMDMask<SIMDInt, 32> &k,
      const SIMDVec<SIMDInt, 32> &a)
  {
#ifdef __AVX2__
    _mm256_maskstore_epi32(p, k, a);
#else
    CHECK_AND_STORE8
#endif
  }
  static SIMD_INLINE void
  mask_store(SIMDFloat *const p,
      const SIMDMask<SIMDFloat, 32> &k,
      const SIMDVec<SIMDFloat, 32> &a)
  {
#ifdef __AVX__
    _mm256_maskstore_ps (p, _mm256_castps_si256(k), a);
#else
    CHECK_AND_STORE8
#endif
  }
//#endif // __AVX__
} //namespace ns_simd
#endif // ifdef __AVX__
#endif // _SIMDMASKINTEL32_H_
