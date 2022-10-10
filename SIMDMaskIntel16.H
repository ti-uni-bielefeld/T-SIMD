// ===========================================================================
// 
// SIMDMaskIntel16.H --
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

#ifndef _SIMDMASKINTEL16_H_
#define _SIMDMASKINTEL16_H_

#ifdef __SSE2__
#include "SIMDVecIntel16.H"
#include <emmintrin.h> //SSE2

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
    bool operator==(const SIMDMask<T, 16> &x) {
      return _mm_movemask_epi8(_mm_cmpeq_epi8(k, x)) == 0xffff;
    }
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
    bool operator==(const SIMDMask<SIMDFloat, 16> &x) {
      return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_castps_si128(k),
                                              _mm_castps_si128(x))) == 0xffff;
    }
  };
  
  template <>
  SIMD_INLINE SIMDMask<SIMDByte, 16>
  mask_all_ones()
  {
    return _mm_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDSignedByte, 16>
  mask_all_ones()
  {
    return _mm_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDWord, 16>
  mask_all_ones()
  {
    return _mm_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDShort, 16>
  mask_all_ones()
  {
    return _mm_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDInt, 16>
  mask_all_ones()
  {
    return _mm_set1_epi8(-1);
  }
  
  template <>
  SIMD_INLINE SIMDMask<SIMDFloat, 16>
  mask_all_ones()
  {
    return _mm_castsi128_ps(_mm_set1_epi8(-1));
  }
  
  static SIMD_INLINE __m128i
  _mm_loadh_epi64 (__m128i a, __m128i const* mem_addr)
  {
    return _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(a), (__m64 const*) mem_addr));
  }
  
  #define CHECK_AND_INSERT(OP, INDEX) \
  if(k[INDEX]) { \
    result = OP (result, p[INDEX], INDEX); \
  }
  
  #define CHECK_AND_INSERT2(MASKNAME, OP, INDEX) \
  if(MASKNAME [INDEX]) { \
    result = OP (result, p[INDEX], INDEX); \
  }
  
  static SIMD_INLINE SIMDVec<SIMDSignedByte, 16>
  maskz_load(const SIMDMask<SIMDSignedByte, 16> &k,
      const SIMDSignedByte *const p)
  {
#ifdef __SSE4_1__
    SIMDVec<SIMDSignedByte, 16> result=setzero<SIMDSignedByte, 16>();
    CHECK_AND_INSERT(_mm_insert_epi8, 0)
    CHECK_AND_INSERT(_mm_insert_epi8, 1)
    CHECK_AND_INSERT(_mm_insert_epi8, 2)
    CHECK_AND_INSERT(_mm_insert_epi8, 3)
    CHECK_AND_INSERT(_mm_insert_epi8, 4)
    CHECK_AND_INSERT(_mm_insert_epi8, 5)
    CHECK_AND_INSERT(_mm_insert_epi8, 6)
    CHECK_AND_INSERT(_mm_insert_epi8, 7)
    CHECK_AND_INSERT(_mm_insert_epi8, 8)
    CHECK_AND_INSERT(_mm_insert_epi8, 9)
    CHECK_AND_INSERT(_mm_insert_epi8, 10)
    CHECK_AND_INSERT(_mm_insert_epi8, 11)
    CHECK_AND_INSERT(_mm_insert_epi8, 12)
    CHECK_AND_INSERT(_mm_insert_epi8, 13)
    CHECK_AND_INSERT(_mm_insert_epi8, 14)
    CHECK_AND_INSERT(_mm_insert_epi8, 15)
    return result;
#else
#ifdef __SSE2__
    return _mm_set_epi8((k[15]?p[15]:0), (k[14]?p[14]:0), (k[13]?p[13]:0), (k[12]?p[12]:0), (k[11]?p[11]:0), (k[10]?p[10]:0), (k[9]?p[9]:0), (k[8]?p[8]:0), (k[7]?p[7]:0), (k[6]?p[6]:0), (k[5]?p[5]:0), (k[4]?p[4]:0), (k[3]?p[3]:0), (k[2]?p[2]:0), (k[1]?p[1]:0), (k[0]?p[0]:0));
#else
    #error "maskz_load<SIMDSignedByte,16> not available because of missing target features"
#endif
#endif
  }
  static SIMD_INLINE SIMDVec<SIMDByte, 16>
  maskz_load(const SIMDMask<SIMDByte, 16> &k,
      const SIMDByte *const p)
  {
    return reinterpret<SIMDByte>(maskz_load(SIMDMask<SIMDSignedByte, 16>((__m128i) k), (const SIMDSignedByte *) p));
  }
  static SIMD_INLINE SIMDVec<SIMDShort, 16>
  maskz_load(const SIMDMask<SIMDShort, 16> &k,
      const SIMDShort *const p)
  {
/*#ifdef BENCH_METHOD1
    SIMDVec<SIMDShort, 16> result=setzero<SIMDShort, 16>();
    CHECK_AND_INSERT(_mm_insert_epi16, 0)
    CHECK_AND_INSERT(_mm_insert_epi16, 1)
    CHECK_AND_INSERT(_mm_insert_epi16, 2)
    CHECK_AND_INSERT(_mm_insert_epi16, 3)
    CHECK_AND_INSERT(_mm_insert_epi16, 4)
    CHECK_AND_INSERT(_mm_insert_epi16, 5)
    CHECK_AND_INSERT(_mm_insert_epi16, 6)
    CHECK_AND_INSERT(_mm_insert_epi16, 7)
    return result;
#endif
#ifdef BENCH_METHOD2
    return _mm_set_epi16((k[7]?p[7]:0), (k[6]?p[6]:0), (k[5]?p[5]:0), (k[4]?p[4]:0), (k[3]?p[3]:0), (k[2]?p[2]:0), (k[1]?p[1]:0), (k[0]?p[0]:0));
#endif
#ifdef BENCH_METHOD3*/
    //if(test_all_ones(SIMDVec<SIMDShort, 16>(k))) { return load<16, SIMDShort>(p); }
    //if(test_all_zeros(SIMDVec<SIMDShort, 16>(k))) { return setzero<SIMDShort, 16>(); }
    SIMDVec<SIMDShort, 16> result=setzero<SIMDShort, 16>();
    /*if(_mm_extract_epi64 (k, 0)==0xffffffffffffffff) { result=_mm_loadl_epi64((__m128i const*) p); } else {
        result=setzero<SIMDShort, 16>();
        CHECK_AND_INSERT(_mm_insert_epi16, 0)
        CHECK_AND_INSERT(_mm_insert_epi16, 1)
        CHECK_AND_INSERT(_mm_insert_epi16, 2)
        CHECK_AND_INSERT(_mm_insert_epi16, 3)
    }
    if(_mm_extract_epi64 (k, 1)==0xffffffffffffffff) { result=_mm_loadh_epi64(result, (__m128i const*) (p+4)); } else {
        CHECK_AND_INSERT(_mm_insert_epi16, 4)
        CHECK_AND_INSERT(_mm_insert_epi16, 5)
        CHECK_AND_INSERT(_mm_insert_epi16, 6)
        CHECK_AND_INSERT(_mm_insert_epi16, 7)
    }*/
    uint64_t mask=k;
    if((0x0f&mask)==0x0f) {
        if((0xf0&mask)==0xf0) {
            result=load<16, SIMDShort>(p);
        } else {
            result=_mm_loadl_epi64((__m128i const*) p);
            CHECK_AND_INSERT(_mm_insert_epi16, 4)
            CHECK_AND_INSERT(_mm_insert_epi16, 5)
            CHECK_AND_INSERT(_mm_insert_epi16, 6)
            CHECK_AND_INSERT(_mm_insert_epi16, 7)
        }
    } else {
        if((0xf0&mask)==0xf0) {
            CHECK_AND_INSERT(_mm_insert_epi16, 0)
            CHECK_AND_INSERT(_mm_insert_epi16, 1)
            CHECK_AND_INSERT(_mm_insert_epi16, 2)
            CHECK_AND_INSERT(_mm_insert_epi16, 3)
            result=_mm_loadh_epi64(result, (__m128i const*) (p+4));
        } else {
            CHECK_AND_INSERT(_mm_insert_epi16, 0)
            CHECK_AND_INSERT(_mm_insert_epi16, 1)
            CHECK_AND_INSERT(_mm_insert_epi16, 2)
            CHECK_AND_INSERT(_mm_insert_epi16, 3)
            CHECK_AND_INSERT(_mm_insert_epi16, 4)
            CHECK_AND_INSERT(_mm_insert_epi16, 5)
            CHECK_AND_INSERT(_mm_insert_epi16, 6)
            CHECK_AND_INSERT(_mm_insert_epi16, 7)
        }
    }
    return result;
//#endif
  }
  static SIMD_INLINE SIMDVec<SIMDWord, 16>
  maskz_load(const SIMDMask<SIMDWord, 16> &k,
      const SIMDWord *const p)
  {
    return reinterpret<SIMDWord>(maskz_load(SIMDMask<SIMDShort, 16>((__m128i) k), (const SIMDShort *) p));
  }
  static SIMD_INLINE SIMDVec<SIMDInt, 16>
  maskz_load(const SIMDMask<SIMDInt, 16> &k,
      const SIMDInt *const p)
  {
//#ifdef BENCH_METHOD1
#ifndef __AVX2__
#ifdef __SSE4_1__
    SIMDVec<SIMDInt, 16> result=setzero<SIMDInt, 16>();
    CHECK_AND_INSERT(_mm_insert_epi32, 0)
    CHECK_AND_INSERT(_mm_insert_epi32, 1)
    CHECK_AND_INSERT(_mm_insert_epi32, 2)
    CHECK_AND_INSERT(_mm_insert_epi32, 3)
    return result;
#endif
#endif
//#endif
//#ifdef BENCH_METHOD2
#ifndef __AVX2__
#ifndef __SSE4_1__
#ifdef __SSE2__
    return _mm_set_epi32((k[3]?p[3]:0), (k[2]?p[2]:0), (k[1]?p[1]:0), (k[0]?p[0]:0));
#else
    #error "maskz_load<SIMDInt,16> not available because of missing target features"
#endif
#endif
#endif
/*#endif
#ifdef BENCH_METHOD3
    SIMDVec<SIMDInt, 16> result=setzero<SIMDInt, 16>();
    if(k[0]) {
      if(k[1]) {
        if(k[2]) {
          if(k[3]) {
            result=load<16, SIMDInt>(p);
          } else {
            result=_mm_loadl_epi64((__m128i const*) p);
            result=_mm_insert_epi32(result, p[2], 2);
          }
        } else {
          if(k[3]) {
            result=_mm_loadl_epi64((__m128i const*) p);
            result=_mm_insert_epi32(result, p[3], 3);
          } else {
            result=_mm_loadl_epi64((__m128i const*) p);
          }
        }
      } else {
        if(k[2]) {
          if(k[3]) {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[0], 0);
            result=_mm_loadh_epi64(result, (__m128i const*) (p+2));
          } else {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[0], 0);
            result=_mm_insert_epi32(result, p[2], 2);
          }
        } else {
          if(k[3]) {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[0], 0);
            result=_mm_insert_epi32(result, p[3], 3);
          } else {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[0], 0);
          }
        }
      }
    } else {
      if(k[1]) {
        if(k[2]) {
          if(k[3]) {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[1], 1);
            result=_mm_loadh_epi64(result, (__m128i const*) (p+2));
          } else {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[1], 1);
            result=_mm_insert_epi32(result, p[2], 2);
          }
        } else {
          if(k[3]) {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[1], 1);
            result=_mm_insert_epi32(result, p[3], 3);
          } else {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[1], 1);
          }
        }
      } else {
        if(k[2]) {
          if(k[3]) {
            result=setzero<SIMDInt, 16>();
            result=_mm_loadh_epi64(result, (__m128i const*) (p+2));
          } else {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[2], 2);
          }
        } else {
          if(k[3]) {
            result=setzero<SIMDInt, 16>();
            result=_mm_insert_epi32(result, p[3], 3);
          } else {
            result=setzero<SIMDInt, 16>();
          }
        }
      }
    }
    return result;
#endif
#ifdef BENCH_METHOD4*/
#ifdef __AVX2__
    return _mm_maskload_epi32 (p, k);
#endif
//#endif
  }
  static SIMD_INLINE SIMDVec<SIMDFloat, 16>
  maskz_load(const SIMDMask<SIMDFloat, 16> &k,
      const SIMDFloat *const p)
  {
    return reinterpret<SIMDFloat>(maskz_load(_mm_castps_si128(k), (const SIMDInt *) p));
  }
  
  
  #define CHECK_AND_STORE(INDEX) \
  if(k [INDEX]) { p[INDEX] = extract<INDEX> (a); }
  
  #define CHECK_AND_STORE4 CHECK_AND_STORE(0) CHECK_AND_STORE(1) CHECK_AND_STORE(2) CHECK_AND_STORE(3)
  
  #define CHECK_AND_STORE8 CHECK_AND_STORE4 CHECK_AND_STORE(4) CHECK_AND_STORE(5) CHECK_AND_STORE(6) CHECK_AND_STORE(7)
  
  #define CHECK_AND_STORE16 CHECK_AND_STORE8 CHECK_AND_STORE(8) CHECK_AND_STORE(9) CHECK_AND_STORE(10) CHECK_AND_STORE(11) CHECK_AND_STORE(12) CHECK_AND_STORE(13) CHECK_AND_STORE(14) CHECK_AND_STORE(15) 
  
  #define CHECK_AND_STORE32 CHECK_AND_STORE16 CHECK_AND_STORE(16) CHECK_AND_STORE(17) CHECK_AND_STORE(18) CHECK_AND_STORE(19) CHECK_AND_STORE(20) CHECK_AND_STORE(21) CHECK_AND_STORE(22) CHECK_AND_STORE(23) CHECK_AND_STORE(24) CHECK_AND_STORE(25) CHECK_AND_STORE(26) CHECK_AND_STORE(27) CHECK_AND_STORE(28) CHECK_AND_STORE(29) CHECK_AND_STORE(30) CHECK_AND_STORE(31) 
  
  static SIMD_INLINE void
  mask_store(SIMDByte *const p,
      const SIMDMask<SIMDByte, 16> &k,
      const SIMDVec<SIMDByte, 16> &a)
  {
    CHECK_AND_STORE16
    //_mm_maskmoveu_si128(a, k, p); TODO This might be faster. Untested
  }
  static SIMD_INLINE void
  mask_store(SIMDSignedByte *const p,
      const SIMDMask<SIMDSignedByte, 16> &k,
      const SIMDVec<SIMDSignedByte, 16> &a)
  {
    CHECK_AND_STORE16
    //_mm_maskmoveu_si128(a, k, p); TODO This might be faster. Untested
  }
  static SIMD_INLINE void
  mask_store(SIMDWord *const p,
      const SIMDMask<SIMDWord, 16> &k,
      const SIMDVec<SIMDWord, 16> &a)
  {
    CHECK_AND_STORE8
    //_mm_maskmoveu_si128(a, k, p); TODO This might be faster. Untested
  }
  static SIMD_INLINE void
  mask_store(SIMDShort *const p,
      const SIMDMask<SIMDShort, 16> &k,
      const SIMDVec<SIMDShort, 16> &a)
  {
    CHECK_AND_STORE8
    //_mm_maskmoveu_si128(a, k, p); TODO This might be faster. Untested
  }
  static SIMD_INLINE void
  mask_store(SIMDInt *const p,
      const SIMDMask<SIMDInt, 16> &k,
      const SIMDVec<SIMDInt, 16> &a)
  {
#ifdef __AVX2__
    _mm_maskstore_epi32(p, k, a);
#else
    CHECK_AND_STORE4
#endif
  }
  static SIMD_INLINE void
  mask_store(SIMDFloat *const p,
      const SIMDMask<SIMDFloat, 16> &k,
      const SIMDVec<SIMDFloat, 16> &a)
  {
#ifdef __AVX__
    _mm_maskstore_ps (p, _mm_castps_si128(k), a);
#else
    CHECK_AND_STORE4
#endif
  }
} //namespace ns_simd
#endif // ifdef __SSE2__
#endif // _SIMDMASKINTEL16_H_
