// ===========================================================================
// 
// SIMDVecMaskImplIntel64.H --
// Mask class definitions and architecture specific functions
// for Intel 64 byte (512 bit)
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

// 01. Feb 23 (Jonas Keller): implemented emulated float versions of some
// functions in a more efficient way and also optimized some other small things

#ifndef _SIMD_VEC_MASK_IMPL_INTEL_64_H_
#define _SIMD_VEC_MASK_IMPL_INTEL_64_H_

#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecBaseImplIntel64.H"

#include <stdint.h>
#include <string.h>

#ifdef __AVX512F__

namespace ns_simd {
  #define CLASS_MASK(TYPE, MASK_SIZE) \
  template <> \
  class SIMDMask<TYPE, 64> \
  { \
  public: \
    __mmask ## MASK_SIZE k; \
    SIMDMask() { k=0; } \
    SIMDMask(const __mmask ## MASK_SIZE &x) { k = x; } \
    SIMDMask& operator=(const __mmask ## MASK_SIZE &x) { k = x; return *this; } \
    operator __mmask ## MASK_SIZE () const { return k; } \
    bool operator[](const uint8_t i) const { \
      return ((1lu<<i)&k)!=0; \
    } \
    bool operator==(const SIMDMask<TYPE, 64> &x) const { \
      return k==x; \
    } \
  };

  // 10. Oct 22 (Jonas Keller):
  // Used the int2bits and msb2int functions for conversion from/to an integer.
  // 29. Nov 22 (Jonas Keller):
  // defined operators new and delete to ensure proper alignment, since
  // the default new and delete are not guaranteed to do so before C++17
  #define EMULATE_CLASS_MASK(TYPE, MASK_SIZE) \
  template <> \
  class SIMDMask<TYPE, 64> \
  { \
  public: \
    __m512i k; \
    SIMDMask() { k=setzero<TYPE, 64>(); } \
    SIMDMask(const __m512i &x) { k = x; } \
    SIMDMask(const uint64_t &x) { k= int2bits<TYPE, 64>(x); } \
    SIMDMask& operator=(const __m512i &x) { k=x; return *this; } \
    SIMDMask& operator=(const uint64_t &x) { k= int2bits<TYPE, 64>(x); return *this; } \
    operator uint64_t() const { return msb2int(SIMDVec<TYPE, 64>(k)); } \
    operator __m512i() const { return k; } \
    bool operator[](const uint8_t i) const { \
      return ((uint64_t(1)<<i)&msb2int(SIMDVec<TYPE, 64>(k)))!=0; \
    } \
    bool operator==(const SIMDMask<TYPE, 64> &x) const {\
      return _mm512_cmpeq_epi64_mask(k, x)==0xff; \
    } \
    void *operator new(size_t size) \
    { return simd_aligned_malloc(sizeof(__m256i), size); } \
    void operator delete(void *p) \
    { simd_aligned_free(p); } \
    void *operator new[](size_t size) \
    { return simd_aligned_malloc(sizeof(__m256i), size); } \
    void operator delete[](void *p) \
    { simd_aligned_free(p); } \
  };

#ifdef __AVX512BW__
  CLASS_MASK(SIMDByte, 64)
  CLASS_MASK(SIMDSignedByte, 64)
  CLASS_MASK(SIMDWord, 32)
  CLASS_MASK(SIMDShort, 32)
#else
  EMULATE_CLASS_MASK(SIMDByte, 64)
  EMULATE_CLASS_MASK(SIMDSignedByte, 64)
  EMULATE_CLASS_MASK(SIMDWord, 32)
  EMULATE_CLASS_MASK(SIMDShort, 32)
#endif
  CLASS_MASK(SIMDInt, 16)
  CLASS_MASK(SIMDFloat, 16)

namespace internal {
namespace mask {

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  setTrueLeft(const unsigned int x)
  {
    return kshiftli(mask_all_ones(OutputType<T>(), Integer<64>()), x);
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  setTrueRight(const unsigned int x)
  {
    return kshiftri(mask_all_ones(OutputType<T>(), Integer<64>()), x);
  }




  #define MASK_SOP(OP, TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ ## OP (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return _mm512_mask_ ## OP ## _ ## SUF (src, k, a); \
  }

  #define MASKZ_SOP(OP, TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return _mm512_maskz_ ## OP ## _ ## SUF (k, a); \
  }

  //For operations with one argument. OP is the name of the operation (e.g. add, sub, mul), TYPE is the typename (e.g. SIMDWord, SIMDFloat), and SUF is the suffix of the intrinsic (e.g. epi8, epi16, ps).
  #define GENERATE_SOP(OP, TYPE, SUF) MASK_SOP(OP, TYPE, SUF) MASKZ_SOP(OP, TYPE, SUF)


  #define MASK_DOP(OP, TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ ## OP (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_mask_ ## OP ## _ ## SUF (src, k, a, b); \
  }

  #define MASKZ_DOP(OP, TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_maskz_ ## OP ## _ ## SUF (k, a, b); \
  }

  //For operations with two arguments. OP is the name of the operation (e.g. add, sub, mul), TYPE is the typename (e.g. SIMDWord, SIMDFloat), and SUF is the suffix of the intrinsic (e.g. epi8, epi16, ps).
  #define GENERATE_DOP(OP, TYPE, SUF) MASK_DOP(OP, TYPE, SUF) MASKZ_DOP(OP, TYPE, SUF)



  #define EMULATE_SOP_TYPE(OP, TYPE) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return mask_ifelsezero(k, OP (a)); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ ## OP(const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return mask_ifelse(k, OP (a), src); \
  }

  #define EMULATE_DOP_TYPE(OP, TYPE) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return mask_ifelsezero(k, OP (a, b)); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ ## OP(const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return mask_ifelse(k, OP (a, b), src); \
  }

  // ---------------------------------------------------------------------------
  // mask_ifelse v
  // ---------------------------------------------------------------------------

  #define MASK_IFELSE(TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ifelse(const SIMDMask<TYPE, 64> &cond, \
	 const SIMDVec<TYPE, 64> &a, \
	 const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_mask_blend_ ## SUF (cond, b, a); \
  }
  #define EMULATE_MASK_IFELSE(TYPE) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ifelse(const SIMDMask<TYPE, 64> &cond, \
	 const SIMDVec<TYPE, 64> &trueVal, \
	 const SIMDVec<TYPE, 64> &falseVal) \
  { \
    return ifelse((SIMDVec<TYPE, 64>) cond, trueVal, falseVal); \
  }

#ifdef __AVX512BW__
  MASK_IFELSE(SIMDByte, epi8)
  MASK_IFELSE(SIMDSignedByte, epi8)
  MASK_IFELSE(SIMDWord, epi16)
  MASK_IFELSE(SIMDShort, epi16)
#else
  EMULATE_MASK_IFELSE(SIMDByte)
  EMULATE_MASK_IFELSE(SIMDSignedByte)
  EMULATE_MASK_IFELSE(SIMDWord)
  EMULATE_MASK_IFELSE(SIMDShort)
#endif
  MASK_IFELSE(SIMDInt, epi32)
  MASK_IFELSE(SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_ifelsezero (mask_ifelsezero(cond, a) is the same as mask_ifelse(cond, a, setzero()), but may have faster implementations)
  // ---------------------------------------------------------------------------

  #define MASK_IFELSEZERO(TYPE) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ifelsezero(const SIMDMask<TYPE, 64> &cond, \
	 const SIMDVec<TYPE, 64> &trueVal) \
  { \
    return mask_ifelse(cond, trueVal, ::ns_simd::setzero<TYPE, 64>()); \
  }

  #define MASK_IFELSEZEROV(TYPE) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_ifelsezero(const SIMDMask<TYPE, 64> &cond, \
	 const SIMDVec<TYPE, 64> &trueVal) \
  { \
    return _mm512_and_si512(cond, trueVal); \
  }
#ifdef __AVX512BW__
  MASK_IFELSEZERO(SIMDByte)
  MASK_IFELSEZERO(SIMDSignedByte)
  MASK_IFELSEZERO(SIMDWord)
  MASK_IFELSEZERO(SIMDShort)
#else
  MASK_IFELSEZEROV(SIMDByte)
  MASK_IFELSEZEROV(SIMDSignedByte)
  MASK_IFELSEZEROV(SIMDWord)
  MASK_IFELSEZEROV(SIMDShort)
#endif
  MASK_IFELSEZERO(SIMDInt)
  MASK_IFELSEZERO(SIMDFloat)

  // ---------------------------------------------------------------------------
  // masked convert (without changes in the number of of elements) v
  // ---------------------------------------------------------------------------

  // conversion with saturation; we wanted to have a fast solution that
  // doesn't trigger the overflow which results in a negative two's
  // complement result ("invalid int32": 0x80000000); therefore we clamp
  // the positive values at the maximal positive float which is
  // convertible to int32 without overflow (0x7fffffbf = 2147483520);
  // negative values cannot overflow (they are clamped to invalid int
  // which is the most negative int32)
  SIMD_INLINE SIMDVec<SIMDInt, 64>
  maskz_cvts(const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    __m512 clip = _mm512_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
    return _mm512_maskz_cvtps_epi32(k, _mm512_maskz_min_ps(k, clip, a));
  }

  SIMD_INLINE SIMDVec<SIMDInt, 64>
  mask_cvts(const SIMDVec<SIMDInt, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    __m512 clip = _mm512_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
    return _mm512_mask_cvtps_epi32(src, k, _mm512_maskz_min_ps(k, clip, a));
  }

  // saturation is not necessary in this case
  SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_cvts(const SIMDMask<SIMDInt, 64> &k,
      const SIMDVec<SIMDInt, 64> &a)
  {
    return _mm512_maskz_cvtepi32_ps(k, a);
  }

  // saturation is not necessary in this case
  SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_cvts(const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDInt, 64> &k,
      const SIMDVec<SIMDInt, 64> &a)
  {
    return _mm512_mask_cvtepi32_ps(src, k, a);
  }

  // ---------------------------------------------------------------------------
  // mask_set1 v
  // ---------------------------------------------------------------------------

  #define GENERATE_SET1(TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_set1 (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const TYPE a) \
  { \
    return _mm512_mask_set1_ ## SUF (src, k, a); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_set1 (const SIMDMask<TYPE, 64> &k, \
      const TYPE a) \
  { \
    return _mm512_maskz_set1_ ## SUF (k, a); \
  }

#ifdef __AVX512BW__
  GENERATE_SET1(SIMDByte, epi8)
  GENERATE_SET1(SIMDSignedByte, epi8)
  GENERATE_SET1(SIMDWord, epi16)
  GENERATE_SET1(SIMDShort, epi16)
#else
  #define EMULATE_SET1(TYPE) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_set1 (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const TYPE a) \
  { \
    return mask_ifelse(k, set1<TYPE, 64>(a), src); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_set1 (const SIMDMask<TYPE, 64> &k, \
      const TYPE a) \
  { \
    /*return mask_ifelse(k, set1(a), setzero<TYPE, 64>());*/ \
    return _mm512_castps_si512(base::x_mm512_and_ps(_mm512_castsi512_ps(k), _mm512_castsi512_ps(set1<TYPE, 64>(a)))); \
  }
  EMULATE_SET1(SIMDByte)
  EMULATE_SET1(SIMDSignedByte)
  EMULATE_SET1(SIMDWord)
  EMULATE_SET1(SIMDShort)
#endif
  GENERATE_SET1(SIMDInt, epi32)
  //Workaround for SIMDFloat, because there is no mask_set1_ps
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_set1 (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDFloat a)
  {
    SIMDInt a_int;
    memcpy(&a_int, &a, sizeof(SIMDInt));
    return _mm512_castsi512_ps (_mm512_mask_set1_epi32 (_mm512_castps_si512 (src), k, a_int));//Reinterpetation
    //return mask_ifelse(k, set1<SIMDFloat, 64>(a), src);
  }
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_set1 (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDFloat a)
  {
    SIMDInt a_int;
    memcpy(&a_int, &a, sizeof(SIMDInt));
    return _mm512_castsi512_ps (_mm512_maskz_set1_epi32 (k, a_int));//Reinterpetation
    //return mask_ifelse(k, set1<SIMDFloat, 64>(a), setzero<SIMDFloat, 64>());
  }

  // ---------------------------------------------------------------------------
  // mask_load v
  // ---------------------------------------------------------------------------

#ifdef SIMD_ALIGN_CHK
  #define GENERATE_LOAD(NAME, TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_load (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const TYPE *const p) \
  { \
    /* AVX load and store instructions need alignment to 64 byte*/ \
    /* (lower 6 bit need to be zero) */ \
    assert((((uintptr_t) p) & 0x3f) == 0); \
    return _mm512_mask_ ## NAME ## _ ## SUF (src, k, p); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_load (const SIMDMask<TYPE, 64> &k, \
      const TYPE *const p) \
  { \
    /* AVX load and store instructions need alignment to 64 byte*/ \
    /* (lower 6 bit need to be zero) */ \
    assert((((uintptr_t) p) & 0x3f) == 0); \
    return _mm512_maskz_ ## NAME ## _ ## SUF (k, p); \
  }
#else
  #define GENERATE_LOAD(NAME, TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_load (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const TYPE *const p) \
  { \
    return _mm512_mask_ ## NAME ## _ ## SUF (src, k, p); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_load (const SIMDMask<TYPE, 64> &k, \
      const TYPE *const p) \
  { \
    return _mm512_maskz_ ## NAME ## _ ## SUF (k, p); \
  }
#endif

#ifdef __AVX512BW__
  // there is no aligned load for 8 and 16 bit types, so we use loadu
  GENERATE_LOAD(loadu, SIMDByte, epi8)
  GENERATE_LOAD(loadu, SIMDSignedByte, epi8)
  GENERATE_LOAD(loadu, SIMDWord, epi16)
  GENERATE_LOAD(loadu, SIMDShort, epi16)
  // use emulated loads in SIMDVecMaskImplEmu.H for 8 and 16 bit types if
  // avx512bw is not available
#endif

  GENERATE_LOAD(load, SIMDInt, epi32)
  GENERATE_LOAD(load, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_loadu v
  // ---------------------------------------------------------------------------

  #define GENERATE_LOADU(TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_loadu (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const TYPE *const p) \
  { \
    return _mm512_mask_loadu_ ## SUF (src, k, p); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_loadu (const SIMDMask<TYPE, 64> &k, \
      const TYPE *const p) \
  { \
    return _mm512_maskz_loadu_ ## SUF (k, p); \
  }

#ifdef __AVX512BW__
  GENERATE_LOADU(SIMDByte, epi8)
  GENERATE_LOADU(SIMDSignedByte, epi8)
  GENERATE_LOADU(SIMDWord, epi16)
  GENERATE_LOADU(SIMDShort, epi16)
  // use emulated loads in SIMDVecMaskImplEmu.H for 8 and 16 bit types if
  // avx512bw is not available
#endif

  GENERATE_LOADU(SIMDInt, epi32)
  GENERATE_LOADU(SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_store v
  // ---------------------------------------------------------------------------

  // There are no *_maskz_store_* intrinsics, only *_mask_store_* intrinsics

#ifdef SIMD_ALIGN_CHK
  #define MASK_STORE(NAME, TYPE, SUF) \
  static SIMD_INLINE void \
  mask_store (TYPE *const p, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    /* AVX load and store instructions need alignment to 64 byte*/ \
    /* (lower 6 bit need to be zero) */ \
    assert((((uintptr_t) p) & 0x3f) == 0); \
    return _mm512_mask_ ## NAME ## _ ## SUF (p, k, a); \
  }
#else
  #define MASK_STORE(NAME, TYPE, SUF) \
  static SIMD_INLINE void \
  mask_store (TYPE *const p, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return _mm512_mask_ ## NAME ## _ ## SUF (p, k, a); \
  }
#endif

#ifdef __AVX512BW__
  // there is no aligned store for 8 and 16 bit types, so we use storeu
  MASK_STORE(storeu, SIMDByte, epi8)
  MASK_STORE(storeu, SIMDSignedByte, epi8)
  MASK_STORE(storeu, SIMDWord, epi16)
  MASK_STORE(storeu, SIMDShort, epi16)
  // use emulated stores in SIMDVecMaskImplEmu.H for 8 and 16 bit types if
  // avx512bw is not available
#endif

  MASK_STORE(store, SIMDInt, epi32)
  MASK_STORE(store, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_storeu v
  // ---------------------------------------------------------------------------

  // There are no *_maskz_storeu_* intrinsics, only *_mask_storeu_* intrinsics

  #define MASK_STOREU(TYPE, SUF) \
  static SIMD_INLINE void \
  mask_storeu (TYPE *const p, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return _mm512_mask_storeu_ ## SUF (p, k, a); \
  }
#ifdef __AVX512BW__
  MASK_STOREU(SIMDByte, epi8)
  MASK_STOREU(SIMDSignedByte, epi8)
  MASK_STOREU(SIMDWord, epi16)
  MASK_STOREU(SIMDShort, epi16)
  // use emulated stores in SIMDVecMaskImplEmu.H for 8 and 16 bit types if
  // avx512bw is not available
#endif

  MASK_STOREU(SIMDInt, epi32)
  MASK_STOREU(SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_add v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_DOP(add, SIMDByte, epi8)
  GENERATE_DOP(add, SIMDSignedByte, epi8)
  GENERATE_DOP(add, SIMDWord, epi16)
  GENERATE_DOP(add, SIMDShort, epi16)
#else
  EMULATE_DOP_TYPE(add, SIMDByte)
  EMULATE_DOP_TYPE(add, SIMDSignedByte)
  EMULATE_DOP_TYPE(add, SIMDWord)
  EMULATE_DOP_TYPE(add, SIMDShort)
#endif
  GENERATE_DOP(add, SIMDInt, epi32)
  GENERATE_DOP(add, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_adds v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_DOP(adds, SIMDByte, epu8)
  GENERATE_DOP(adds, SIMDSignedByte, epi8)
  GENERATE_DOP(adds, SIMDWord, epu16)
  GENERATE_DOP(adds, SIMDShort, epi16)
#else
  EMULATE_DOP_TYPE(adds, SIMDByte)
  EMULATE_DOP_TYPE(adds, SIMDSignedByte)
  EMULATE_DOP_TYPE(adds, SIMDWord)
  EMULATE_DOP_TYPE(adds, SIMDShort)
#endif

#ifndef SIMD_STRICT_SATURATION
  #define ADDS_WORKAROUND(TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_adds (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_mask_add_ ## SUF (src, k, a, b); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_adds (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_maskz_add_ ## SUF (k, a, b); \
  }

  // NOT SATURATED!
  ADDS_WORKAROUND(SIMDInt, epi32)
  ADDS_WORKAROUND(SIMDFloat, ps)
#endif

  // ---------------------------------------------------------------------------
  // mask_sub v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_DOP(sub, SIMDByte, epi8)
  GENERATE_DOP(sub, SIMDSignedByte, epi8)
  GENERATE_DOP(sub, SIMDWord, epi16)
  GENERATE_DOP(sub, SIMDShort, epi16)
#else
  EMULATE_DOP_TYPE(sub, SIMDByte)
  EMULATE_DOP_TYPE(sub, SIMDSignedByte)
  EMULATE_DOP_TYPE(sub, SIMDWord)
  EMULATE_DOP_TYPE(sub, SIMDShort)
#endif
  GENERATE_DOP(sub, SIMDInt, epi32)
  GENERATE_DOP(sub, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_subs v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_DOP(subs, SIMDByte, epu8)
  GENERATE_DOP(subs, SIMDSignedByte, epi8)
  GENERATE_DOP(subs, SIMDWord, epu16)
  GENERATE_DOP(subs, SIMDShort, epi16)
#else
  EMULATE_DOP_TYPE(subs, SIMDByte)
  EMULATE_DOP_TYPE(subs, SIMDSignedByte)
  EMULATE_DOP_TYPE(subs, SIMDWord)
  EMULATE_DOP_TYPE(subs, SIMDShort)
#endif

#ifndef SIMD_STRICT_SATURATION
  #define SUBS_WORKAROUND(TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_subs (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_mask_sub_ ## SUF (src, k, a, b); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_subs (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_maskz_sub_ ## SUF (k, a, b); \
  }

  // NOT SATURATED!
  SUBS_WORKAROUND(SIMDInt, epi32)
  SUBS_WORKAROUND(SIMDFloat, ps)
#endif

  // ---------------------------------------------------------------------------
  // mask_mul v
  // ---------------------------------------------------------------------------

  GENERATE_DOP(mul, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_div v
  // ---------------------------------------------------------------------------

  GENERATE_DOP(div, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // masked ceil, floor, round, truncate v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_ceil (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_roundscale_ps(src, k, a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_ceil (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_maskz_roundscale_ps(k, a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_floor (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_roundscale_ps(src, k, a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_floor (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_maskz_roundscale_ps(k, a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_round (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_roundscale_ps(src, k, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_round (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_maskz_roundscale_ps(k, a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_truncate (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_roundscale_ps(src, k, a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_truncate (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_maskz_roundscale_ps(k, a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  }

  // ---------------------------------------------------------------------------
  // masked elementary mathematical functions v
  // ---------------------------------------------------------------------------

  // masked estimate of a reciprocal
  // NOTE: this has better precision than SSE and AVX versions!
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_rcp (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_rcp14_ps (src, k, a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_rcp (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_maskz_rcp14_ps (k, a);
  }

  // masked estimate of reverse square root
  // NOTE: this has better precision than SSE and AVX versions!
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_rsqrt (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_rsqrt14_ps (src, k, a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_rsqrt (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_maskz_rsqrt14_ps (k, a);
  }

  // masked square root
  GENERATE_SOP(sqrt, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // masked_abs (integer: signed only) v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_SOP(abs, SIMDSignedByte, epi8)
  GENERATE_SOP(abs, SIMDShort, epi16)
#else
  EMULATE_SOP_TYPE(abs, SIMDSignedByte)
  EMULATE_SOP_TYPE(abs, SIMDShort)
#endif
  GENERATE_SOP(abs, SIMDInt, epi32)

#if defined(GCC_VERSION) && GCC_VERSION < 70000
//_mm512_mask_abs_ps is not defined
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_abs (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_castsi512_ps(_mm512_mask_andnot_epi32(_mm512_castps_si512(src), k, _mm512_castps_si512(_mm512_set1_ps(-0.0F)), _mm512_castps_si512(a)));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_abs (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_castsi512_ps(_mm512_maskz_andnot_epi32(k, _mm512_castps_si512(_mm512_set1_ps(-0.0F)), _mm512_castps_si512(a)));
  }
#else
//_mm512_mask_abs_ps is defined
  MASK_SOP(abs, SIMDFloat, ps)

  //There is no _mm512_maskz_abs_ps
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_abs (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return _mm512_mask_abs_ps (::ns_simd::setzero<SIMDFloat, 64>(), k, a);
  }
#endif

  // ---------------------------------------------------------------------------
  // mask_and v
  // ---------------------------------------------------------------------------

  //workaround for SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort, because there is no _mm512_mask_and_epi8 or _mm512_mask_and_epi16
  EMULATE_DOP_TYPE(and, SIMDByte)
  EMULATE_DOP_TYPE(and, SIMDSignedByte)
  EMULATE_DOP_TYPE(and, SIMDWord)
  EMULATE_DOP_TYPE(and, SIMDShort)

  GENERATE_DOP(and, SIMDInt, epi32)
#ifdef __AVX512DQ__
  GENERATE_DOP(and, SIMDFloat, ps)
#else
  //Workaround with the epi32-version and casts
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_and (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_mask_and_epi32 (_mm512_castps_si512(src), k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_and (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_maskz_and_epi32 (k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }
#endif

  // ---------------------------------------------------------------------------
  // mask_or v
  // ---------------------------------------------------------------------------

  //workaround for SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort, because there is no _mm512_mask_or_epi8 or _mm512_mask_or_epi16
  EMULATE_DOP_TYPE(or, SIMDByte)
  EMULATE_DOP_TYPE(or, SIMDSignedByte)
  EMULATE_DOP_TYPE(or, SIMDWord)
  EMULATE_DOP_TYPE(or, SIMDShort)
  GENERATE_DOP(or, SIMDInt, epi32)
#ifdef __AVX512DQ__
  GENERATE_DOP(or, SIMDFloat, ps)
#else
  //Workaround with the epi32-version and casts
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_or (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_mask_or_epi32 (_mm512_castps_si512(src), k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_or (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_maskz_or_epi32 (k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }
#endif

  // ---------------------------------------------------------------------------
  // mask_andnot v
  // ---------------------------------------------------------------------------

  // Workaround for SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort, because there is no _mm512_mask_andnot_epi8 or _mm512_mask_andnot_epi16
  EMULATE_DOP_TYPE(andnot, SIMDByte)
  EMULATE_DOP_TYPE(andnot, SIMDSignedByte)
  EMULATE_DOP_TYPE(andnot, SIMDWord)
  EMULATE_DOP_TYPE(andnot, SIMDShort)
  GENERATE_DOP(andnot, SIMDInt, epi32)
#ifdef __AVX512DQ__
  GENERATE_DOP(andnot, SIMDFloat, ps)
#else
  //Workaround with the epi32-version and casts
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_andnot (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_mask_andnot_epi32 (_mm512_castps_si512(src), k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_andnot (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_maskz_andnot_epi32 (k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }
#endif

  // ---------------------------------------------------------------------------
  // mask_xor v
  // ---------------------------------------------------------------------------

  // Workaround for SIMDByte, SIMDSignedByte, SIMDWord, SIMDShort, because there is no _mm512_mask_xor_epi8 or _mm512_mask_xor_epi16
  EMULATE_DOP_TYPE(xor, SIMDByte)
  EMULATE_DOP_TYPE(xor, SIMDSignedByte)
  EMULATE_DOP_TYPE(xor, SIMDWord)
  EMULATE_DOP_TYPE(xor, SIMDShort)
  GENERATE_DOP(xor, SIMDInt, epi32)
#ifdef __AVX512DQ__
  GENERATE_DOP(xor, SIMDFloat, ps)
#else
  //Workaround with the epi32-version and casts
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  mask_xor (const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_mask_xor_epi32 (_mm512_castps_si512(src), k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  maskz_xor (const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a,
      const SIMDVec<SIMDFloat, 64> &b)
  {
    return _mm512_castsi512_ps(_mm512_maskz_xor_epi32 (k, _mm512_castps_si512(a), _mm512_castps_si512(b)));
  }
#endif

  // ---------------------------------------------------------------------------
  // mask_not v
  // ---------------------------------------------------------------------------

  // There is absolutely no "not"-intrinsic
  EMULATE_SOP_TYPE(not, SIMDByte)
  EMULATE_SOP_TYPE(not, SIMDSignedByte)
  EMULATE_SOP_TYPE(not, SIMDWord)
  EMULATE_SOP_TYPE(not, SIMDShort)
  EMULATE_SOP_TYPE(not, SIMDInt)
  EMULATE_SOP_TYPE(not, SIMDFloat)

  // ---------------------------------------------------------------------------
  // mask_neg (negate = two's complement or unary minus), only signed types v
  // ---------------------------------------------------------------------------

  #define GENERATE_NEG(TYPE, SUF) \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  mask_neg (const SIMDVec<TYPE, 64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return _mm512_mask_sub_ ## SUF (src, k, setzero<TYPE, 64>(), a); \
  } \
  static SIMD_INLINE SIMDVec<TYPE, 64> \
  maskz_neg (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return _mm512_maskz_sub_ ## SUF (k, setzero<TYPE, 64>(), a); \
  }

#ifdef __AVX512BW__
  GENERATE_NEG(SIMDSignedByte, epi8)
  GENERATE_NEG(SIMDShort, epi16)
#else
  EMULATE_SOP_TYPE(neg, SIMDSignedByte)
  EMULATE_SOP_TYPE(neg, SIMDShort)
#endif
  GENERATE_NEG(SIMDInt, epi32)
  GENERATE_NEG(SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_min v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_DOP(min, SIMDByte, epu8)
  GENERATE_DOP(min, SIMDSignedByte, epi8)
  GENERATE_DOP(min, SIMDWord, epu16)
  GENERATE_DOP(min, SIMDShort, epi16)
#else
  EMULATE_DOP_TYPE(min, SIMDByte)
  EMULATE_DOP_TYPE(min, SIMDSignedByte)
  EMULATE_DOP_TYPE(min, SIMDWord)
  EMULATE_DOP_TYPE(min, SIMDShort)
#endif
  GENERATE_DOP(min, SIMDInt, epi32)
  GENERATE_DOP(min, SIMDFloat, ps)

  // ---------------------------------------------------------------------------
  // mask_max v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_DOP(max, SIMDByte, epu8)
  GENERATE_DOP(max, SIMDSignedByte, epi8)
  GENERATE_DOP(max, SIMDWord, epu16)
  GENERATE_DOP(max, SIMDShort, epi16)
#else
  EMULATE_DOP_TYPE(max, SIMDByte)
  EMULATE_DOP_TYPE(max, SIMDSignedByte)
  EMULATE_DOP_TYPE(max, SIMDWord)
  EMULATE_DOP_TYPE(max, SIMDShort)
#endif
  GENERATE_DOP(max, SIMDInt, epi32)
  GENERATE_DOP(max, SIMDFloat, ps)

  // 13. Nov. 2022 (Jonas Keller):
  // removed emulation of masked div2r0 and div2rd, they are already emulated
  // in SIMDVecMaskImplEmu.H


  // For all shifts
  //ARG1, ARG2, and ARG3 can be used to control whether the parameter should be named (or not, to suppress compiler's unused warning). Default: a, k, a
  #define MAKE_X_SHIFT(OP, BITS, MASK_SIZE) \
  /* positive and in range: shift */ \
  template <int IMM> \
  static SIMD_INLINE __m512i \
  x_mm512_mask_ ## OP ## _epi ## BITS (__m512i src, __mmask ## MASK_SIZE k, __m512i a, IsPosInRange<true, true>) \
  { \
    return _mm512_mask_ ## OP ## _epi ## BITS (src, k, a, IMM); \
  } \
  /* hub */ \
  template <int IMM> \
  static SIMD_INLINE __m512i \
  x_mm512_mask_ ## OP ## _epi ## BITS (__m512i src, __mmask ## MASK_SIZE k, __m512i a) \
  { \
    return x_mm512_mask_ ## OP ## _epi ## BITS <IMM>(src, k, a, IsPosInGivenRange<BITS, IMM>()); \
  } \
  /* positive and in range: shift */ \
  template <int IMM> \
  static SIMD_INLINE __m512i \
  x_mm512_maskz_ ## OP ## _epi ## BITS (__mmask ## MASK_SIZE k, __m512i a, IsPosInRange<true, true>) \
  { \
    return _mm512_maskz_ ## OP ## _epi ## BITS (k, a, IMM); \
  } \
  /* hub */ \
  template <int IMM> \
  static SIMD_INLINE __m512i \
  x_mm512_maskz_ ## OP ## _epi ## BITS (__mmask ## MASK_SIZE k, __m512i a) \
  { \
    return x_mm512_maskz_ ## OP ## _epi ## BITS <IMM>(k, a, IsPosInGivenRange<BITS, IMM>()); \
  }

  #define GENERATE_SHIFT(OP, TYPE, SUF) \
  template <int IMM> \
  static SIMD_INLINE SIMDVec<TYPE,64> \
  mask_ ## OP (const SIMDVec<TYPE,64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE,64> &a) \
  { \
    return x_mm512_mask_ ## OP ## _ ## SUF <IMM>(src, k, a); \
  } \
  template <int IMM> \
  static SIMD_INLINE SIMDVec<TYPE,64> \
  maskz_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE,64> &a) \
  { \
    return x_mm512_maskz_ ## OP ## _ ## SUF <IMM>(k, a); \
  }

  #define EMULATE_SHIFT_64(OP, TYPE) \
  template <int IMM> \
  static SIMD_INLINE SIMDVec<TYPE,64> \
  mask_ ## OP (const SIMDVec<TYPE,64> &src, \
      const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE,64> &a) \
  { \
    return mask_ifelse(k, OP <IMM>(a), src); \
  } \
  template <int IMM> \
  static SIMD_INLINE SIMDVec<TYPE,64> \
  maskz_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE,64> &a) \
  { \
    return mask_ifelsezero(k, OP <IMM>(a)); \
  }

  // ---------------------------------------------------------------------------
  // masked srai (16/32 only) v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  /* positive and out of range */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_mask_srai_epi16(__m512i src, __mmask32 k, __m512i a, IsPosInRange<true, false>)
  {
    return _mm512_mask_srai_epi16(src, k, a, 15);
  }
  /* positive and out of range: maximal shift */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_maskz_srai_epi16(__mmask32 k, __m512i a, IsPosInRange<true, false>)
  {
    return _mm512_maskz_srai_epi16(k, a, 15);
  }
  MAKE_X_SHIFT(srai, 16, 32)
  GENERATE_SHIFT(srai, SIMDWord, epi16)
  GENERATE_SHIFT(srai, SIMDShort, epi16)
#else
  EMULATE_SHIFT_64(srai, SIMDWord)
  EMULATE_SHIFT_64(srai, SIMDShort)
#endif
  /* positive and out of range */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_mask_srai_epi32(__m512i src, __mmask16 k, __m512i a, IsPosInRange<true, false>)
  {
    return _mm512_mask_srai_epi32(src, k, a, 31);
  }
  /* positive and out of range: maximal shift */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_maskz_srai_epi32(__mmask16 k, __m512i a, IsPosInRange<true, false>)
  {
    // 02. Aug 22 (Jonas Keller):
    // fixed wrong intrinsic
    // return _mm512_maskz_srai_epi16(k, a, 31);
    return _mm512_maskz_srai_epi32(k, a, 31);
  }
  MAKE_X_SHIFT(srai, 32, 16)
  GENERATE_SHIFT(srai, SIMDInt, epi32)

  // ---------------------------------------------------------------------------
  // masked srli v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  /* positive and out of range */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_mask_srli_epi16(__m512i src, __mmask32 k, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_mask_set1_epi16(src, k, 0);
  }
  /* positive and out of range: maximal shift */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_maskz_srli_epi16(__mmask32, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }
  MAKE_X_SHIFT(srli, 16, 32)
  GENERATE_SHIFT(srli, SIMDWord, epi16)
  GENERATE_SHIFT(srli, SIMDShort, epi16)
#else
  EMULATE_SHIFT_64(srli, SIMDWord)
  EMULATE_SHIFT_64(srli, SIMDShort)
#endif
  /* positive and out of range */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_mask_srli_epi32(__m512i src, __mmask16 k, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_mask_set1_epi32(src, k, 0);
  }
  /* positive and out of range: maximal shift */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_maskz_srli_epi32(__mmask16, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }
  MAKE_X_SHIFT(srli, 32, 16)
  GENERATE_SHIFT(srli, SIMDInt, epi32)

  // ---------------------------------------------------------------------------
  // masked slli v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  /* positive and out of range */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_mask_slli_epi16(__m512i src, __mmask32 k, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_mask_set1_epi16(src, k, 0);
  }
  /* positive and out of range: maximal shift */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_maskz_slli_epi16(__mmask32, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }
  MAKE_X_SHIFT(slli, 16, 32)
  GENERATE_SHIFT(slli, SIMDWord, epi16)
  GENERATE_SHIFT(slli, SIMDShort, epi16)
#else
  EMULATE_SHIFT_64(slli, SIMDWord)
  EMULATE_SHIFT_64(slli, SIMDShort)
#endif
  /* positive and out of range */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_mask_slli_epi32(__m512i src, __mmask16 k, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_mask_set1_epi32(src, k, 0);
  }
  /* positive and out of range: maximal shift */
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_maskz_slli_epi32(__mmask16, __m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }
  MAKE_X_SHIFT(slli, 32, 16)
  GENERATE_SHIFT(slli, SIMDInt, epi32)


  // 05. Aug 22 (Jonas Keller):
  // Improved implementation of masked hadd, hadds, hsub and hsubs,
  // implementation uses masked add/adds/sub/subs directly now instead of
  // wrapping hadd, hadds, hsub and hsubs with a mask_ifelse(zero).
  // SIMDByte and SIMDSignedByte are now supported as well.

  // ---------------------------------------------------------------------------
  // masked hadd v
  // ---------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  mask_hadd(const SIMDVec<T, 64> &src,
            const SIMDMask<T, 64> &k,
            const SIMDVec<T, 64> &a,
            const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return mask_add(src, k, x, y);
  }

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  maskz_hadd(const SIMDMask<T, 64> &k,
             const SIMDVec<T, 64> &a,
             const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return maskz_add(k, x, y);
  }

  // ---------------------------------------------------------------------------
  // masked hadds v
  // ---------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  mask_hadds(const SIMDVec<T, 64> &src,
             const SIMDMask<T, 64> &k,
             const SIMDVec<T, 64> &a,
             const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return mask_adds(src, k, x, y);
  }

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  maskz_hadds(const SIMDMask<T, 64> &k,
              const SIMDVec<T, 64> &a,
              const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return maskz_adds(k, x, y);
  }

  // ---------------------------------------------------------------------------
  // masked hsub v
  // ---------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  mask_hsub(const SIMDVec<T, 64> &src,
            const SIMDMask<T, 64> &k,
            const SIMDVec<T, 64> &a,
            const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return mask_sub(src, k, x, y);
  }

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  maskz_hsub(const SIMDMask<T, 64> &k,
             const SIMDVec<T, 64> &a,
             const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return maskz_sub(k, x, y);
  }

  // ---------------------------------------------------------------------------
  // masked hsubs v
  // ---------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  mask_hsubs(const SIMDVec<T, 64> &src,
             const SIMDMask<T, 64> &k,
             const SIMDVec<T, 64> &a,
             const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return mask_subs(src, k, x, y);
  }

  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  maskz_hsubs(const SIMDMask<T, 64> &k,
              const SIMDVec<T, 64> &a,
              const SIMDVec<T, 64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return maskz_subs(k, x, y);
  }


  // 16. Oct 22 (Jonas Keller): added overloaded versions of mask_cmp* functions
  // that only take two vector parameters and no mask parameter

  #define GENERATE_CMP(OP, TYPE, SUF) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  mask_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_mask_ ## OP ## _ ## SUF ## _mask (k, a, b); \
  } \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  mask_ ## OP (const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_ ## OP ## _ ## SUF ## _mask (a, b); \
  }

  #define GENERATE_CMP_WITH_GENERALIZED_FCT(OP, IMM8) \
  static SIMD_INLINE SIMDMask<SIMDFloat, 64> \
  mask_ ## OP (const SIMDMask<SIMDFloat, 64> &k, \
      const SIMDVec<SIMDFloat, 64> &a, \
      const SIMDVec<SIMDFloat, 64> &b) \
  { \
    return _mm512_mask_cmp_ps_mask (k, a, b, IMM8); \
  } \
  static SIMD_INLINE SIMDMask<SIMDFloat, 64> \
  mask_ ## OP (const SIMDVec<SIMDFloat, 64> &a, \
      const SIMDVec<SIMDFloat, 64> &b) \
  { \
    return _mm512_cmp_ps_mask (a, b, IMM8); \
  }

  #define EMULATE_CMP_64(OP, TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  mask_ ## OP (const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return _mm512_and_si512(k, OP (a, b)); \
  } \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  mask_ ## OP (const SIMDVec<TYPE, 64> &a, \
      const SIMDVec<TYPE, 64> &b) \
  { \
    return OP (a, b).zmm; \
  }

  // ---------------------------------------------------------------------------
  // masked compare < v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_CMP(cmplt, SIMDByte, epu8)
  GENERATE_CMP(cmplt, SIMDSignedByte, epi8)
  GENERATE_CMP(cmplt, SIMDWord, epu16)
  GENERATE_CMP(cmplt, SIMDShort, epi16)
#else
  EMULATE_CMP_64(cmplt, SIMDByte)
  EMULATE_CMP_64(cmplt, SIMDSignedByte)
  EMULATE_CMP_64(cmplt, SIMDWord)
  EMULATE_CMP_64(cmplt, SIMDShort)
#endif
  GENERATE_CMP(cmplt, SIMDInt, epi32)

//#if defined(GCC_VERSION) && GCC_VERSION < 80000
  GENERATE_CMP_WITH_GENERALIZED_FCT(cmplt, _CMP_LT_OS)
//#else
//  GENERATE_CMP(cmplt, SIMDFloat, ps)
//#endif

  // ---------------------------------------------------------------------------
  // masked compare <= v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_CMP(cmple, SIMDByte, epu8)
  GENERATE_CMP(cmple, SIMDSignedByte, epi8)
  GENERATE_CMP(cmple, SIMDWord, epu16)
  GENERATE_CMP(cmple, SIMDShort, epi16)
#else
  EMULATE_CMP_64(cmple, SIMDByte)
  EMULATE_CMP_64(cmple, SIMDSignedByte)
  EMULATE_CMP_64(cmple, SIMDWord)
  EMULATE_CMP_64(cmple, SIMDShort)
#endif
  GENERATE_CMP(cmple, SIMDInt, epi32)

//#if defined(GCC_VERSION) && GCC_VERSION < 80000
  GENERATE_CMP_WITH_GENERALIZED_FCT(cmple, _CMP_LE_OS)
//#else
//  GENERATE_CMP(cmple, SIMDFloat, ps)
//#endif

  // ---------------------------------------------------------------------------
  // masked compare == v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_CMP(cmpeq, SIMDByte, epu8)
  GENERATE_CMP(cmpeq, SIMDSignedByte, epi8)
  GENERATE_CMP(cmpeq, SIMDWord, epu16)
  GENERATE_CMP(cmpeq, SIMDShort, epi16)
#else
  EMULATE_CMP_64(cmpeq, SIMDByte)
  EMULATE_CMP_64(cmpeq, SIMDSignedByte)
  EMULATE_CMP_64(cmpeq, SIMDWord)
  EMULATE_CMP_64(cmpeq, SIMDShort)
#endif
  GENERATE_CMP(cmpeq, SIMDInt, epi32)

//#if defined(GCC_VERSION) && GCC_VERSION < 80000
  GENERATE_CMP_WITH_GENERALIZED_FCT(cmpeq, _CMP_EQ_OQ)
//#else
//  GENERATE_CMP(cmpeq, SIMDFloat, ps)
//#endif

  // ---------------------------------------------------------------------------
  // masked compare > v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_CMP(cmpgt, SIMDByte, epu8)
  GENERATE_CMP(cmpgt, SIMDSignedByte, epi8)
  GENERATE_CMP(cmpgt, SIMDWord, epu16)
  GENERATE_CMP(cmpgt, SIMDShort, epi16)
#else
  EMULATE_CMP_64(cmpgt, SIMDByte)
  EMULATE_CMP_64(cmpgt, SIMDSignedByte)
  EMULATE_CMP_64(cmpgt, SIMDWord)
  EMULATE_CMP_64(cmpgt, SIMDShort)
#endif
  GENERATE_CMP(cmpgt, SIMDInt, epi32)

  //There is no _mm512_mask_cmpgt_ps_mask
  GENERATE_CMP_WITH_GENERALIZED_FCT(cmpgt, _CMP_GT_OS)

  // ---------------------------------------------------------------------------
  // masked compare >= v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_CMP(cmpge, SIMDByte, epu8)
  GENERATE_CMP(cmpge, SIMDSignedByte, epi8)
  GENERATE_CMP(cmpge, SIMDWord, epu16)
  GENERATE_CMP(cmpge, SIMDShort, epi16)
#else
  EMULATE_CMP_64(cmpge, SIMDByte)
  EMULATE_CMP_64(cmpge, SIMDSignedByte)
  EMULATE_CMP_64(cmpge, SIMDWord)
  EMULATE_CMP_64(cmpge, SIMDShort)
#endif
  GENERATE_CMP(cmpge, SIMDInt, epi32)

  //There is no _mm512_mask_cmpge_ps_mask
  GENERATE_CMP_WITH_GENERALIZED_FCT(cmpge, _CMP_GE_OS)

  // ---------------------------------------------------------------------------
  // masked compare != v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  GENERATE_CMP(cmpneq, SIMDByte, epu8)
  GENERATE_CMP(cmpneq, SIMDSignedByte, epi8)
  GENERATE_CMP(cmpneq, SIMDWord, epu16)
  GENERATE_CMP(cmpneq, SIMDShort, epi16)
#else
  EMULATE_CMP_64(cmpneq, SIMDByte)
  EMULATE_CMP_64(cmpneq, SIMDSignedByte)
  EMULATE_CMP_64(cmpneq, SIMDWord)
  EMULATE_CMP_64(cmpneq, SIMDShort)
#endif
  GENERATE_CMP(cmpneq, SIMDInt, epi32)

//#if defined(GCC_VERSION) && GCC_VERSION < 80000
  GENERATE_CMP_WITH_GENERALIZED_FCT(cmpneq, _CMP_NEQ_OQ)
//#else
//  GENERATE_CMP(cmpneq, SIMDFloat, ps)
//#endif

  // ---------------------------------------------------------------------------
  // masked avg: average with rounding down v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  static SIMD_INLINE SIMDVec<SIMDByte,64>
  mask_avg(const SIMDVec<SIMDByte, 64> &src, \
      const SIMDMask<SIMDByte, 64> &k, \
      const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_mask_avg_epu8(src, k, a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  maskz_avg(const SIMDMask<SIMDByte, 64> &k, \
      const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_maskz_avg_epu8(k, a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  mask_avg(const SIMDVec<SIMDWord, 64> &src,
      const SIMDMask<SIMDWord, 64> &k,
      const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_mask_avg_epu16(src, k, a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  maskz_avg(const SIMDMask<SIMDWord, 64> &k,
      const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_maskz_avg_epu16(k, a, b);
  }
#else
  EMULATE_DOP_TYPE(avg, SIMDByte)
  EMULATE_DOP_TYPE(avg, SIMDWord)
#endif
  EMULATE_DOP_TYPE(avg, SIMDSignedByte)
  EMULATE_DOP_TYPE(avg, SIMDShort)

  //EMULATE_DOP_TYPE(avg, SIMDInt)
  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  mask_avg(const SIMDVec<SIMDInt, 64> &src,
      const SIMDMask<SIMDInt, 64> &k,
      const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    SIMDVec<SIMDInt,64> one = ::ns_simd::set1<SIMDInt,64>(1), as, bs, lsb;
    lsb = and(or(a, b), one);
    as = srai<1>(a);
    bs = srai<1>(b);
    return mask_add(src, k, lsb, add(as, bs));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  maskz_avg(const SIMDMask<SIMDInt, 64> &k,
      const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    SIMDVec<SIMDInt,64> one = ::ns_simd::set1<SIMDInt,64>(1), as, bs, lsb;
    lsb = and(or(a, b), one);
    as = srai<1>(a);
    bs = srai<1>(b);
    return maskz_add(k, lsb, add(as, bs));
  }

  // NOTE: SIMDFloat version doesn't round!
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  mask_avg(const SIMDVec<SIMDFloat, 64> &src,
      const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_mask_mul_ps(src, k, _mm512_maskz_add_ps(k, a, b), _mm512_set1_ps(0.5f));
  }

  // NOTE: SIMDFloat version doesn't round!
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  maskz_avg(const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_maskz_mul_ps(k, _mm512_maskz_add_ps(k, a, b), _mm512_set1_ps(0.5f));
  }

  // ---------------------------------------------------------------------------
  // masked test_all_zeros v
  // ---------------------------------------------------------------------------

  #define TEST_ALL_ZEROS(TYPE, SUF) \
  static SIMD_INLINE int \
  mask_test_all_zeros(const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return (_mm512_mask_test_epi ## SUF ## _mask(k, a, a) == 0); \
  }

  #define EMULATE_TEST_ALL_ZEROS(TYPE) \
  static SIMD_INLINE int \
  mask_test_all_zeros(const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return test_all_zeros(mask_ifelsezero(k, a)); \
  }

#ifdef __AVX512BW__
  TEST_ALL_ZEROS(SIMDByte, 8)
  TEST_ALL_ZEROS(SIMDSignedByte, 8)
  TEST_ALL_ZEROS(SIMDWord, 16)
  TEST_ALL_ZEROS(SIMDShort, 16)
#else
  EMULATE_TEST_ALL_ZEROS(SIMDByte)
  EMULATE_TEST_ALL_ZEROS(SIMDSignedByte)
  EMULATE_TEST_ALL_ZEROS(SIMDWord)
  EMULATE_TEST_ALL_ZEROS(SIMDShort)
#endif
  TEST_ALL_ZEROS(SIMDInt, 32)

  static SIMD_INLINE int
  mask_test_all_zeros(const SIMDMask<SIMDFloat, 64> &k,
      const SIMDVec<SIMDFloat, 64> &a)
  {
    return (_mm512_mask_test_epi32_mask(k, _mm512_castps_si512(a), _mm512_castps_si512(a)) == 0);
  }

  // ---------------------------------------------------------------------------
  // masked test_all_ones v
  // ---------------------------------------------------------------------------

  // description of testn intrinsics was not clear, chosen other way

  // 02. Aug 22 (Jonas Keller): Was implemented using a template parameter for
  // the type, but that was ambiguous with the emulated version in SIMDVecMaskImplEmu.H,
  // so defining it for every type explicitly instead to resolve the ambiguity.

  #define TEST_ALL_ONES(TYPE) \
  static SIMD_INLINE int \
  mask_test_all_ones(const SIMDMask<TYPE, 64> &k, \
      const SIMDVec<TYPE, 64> &a) \
  { \
    return mask_test_all_zeros(k, not(a)); \
  }

  TEST_ALL_ONES(SIMDByte)
  TEST_ALL_ONES(SIMDSignedByte)
  TEST_ALL_ONES(SIMDWord)
  TEST_ALL_ONES(SIMDShort)
  TEST_ALL_ONES(SIMDInt)
  TEST_ALL_ONES(SIMDFloat)


  #define MASK_ALL_ONES(TYPE, MASK) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  mask_all_ones(OutputType<TYPE>, Integer<64>) \
  { \
    return MASK; \
  }

#ifdef __AVX512BW__
  MASK_ALL_ONES(SIMDByte, 0xFFFFFFFFFFFFFFFF)
  MASK_ALL_ONES(SIMDSignedByte, 0xFFFFFFFFFFFFFFFF)
  MASK_ALL_ONES(SIMDWord, 0xFFFFFFFF)
  MASK_ALL_ONES(SIMDShort, 0xFFFFFFFF)
  // use emulated functions in SIMDVecMaskImplEmu.H for 8 and 16 bit types if
  // avx512bw is not available
#endif

  MASK_ALL_ONES(SIMDInt, 0xFFFF)
  MASK_ALL_ONES(SIMDFloat, 0xFFFF)

/*
Short explanation:
Intrinsics (e.g. _kand_mask16, _kor_mask32) are only available for gcc versions >= 7. The intrinsics for __mmask32 and __mmask64 are only available under AVX512BW
Intrinsics with a different name and only for __mmask16 (e.g. _mm512_kand) are available for gcc versions >= 6
If AVX512BW is not available, the SIMDByte/SignedByte/Word/Short masks are vectors, then the vector functions are used
The last resort (because it is probably slower in most cases) is to emulate the functions with normal operators (e.g. "+" for kadd, "<<" for kshiftl, "&" for kand)
*/

#define EMULATE_KADD(TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  kadd (const SIMDMask<TYPE, 64> &a, \
      const SIMDMask<TYPE, 64> &b) \
  { \
    return (((uint64_t) a) + ((uint64_t) b)); \
  }

#if __GNUC__ >= 7 //TODO other compilers (not really a problem, then the intrinsics will just not be used)
  #define GENERATE_DMASKOP(NAME, TYPE, NUM) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  k ## NAME (const SIMDMask<TYPE, 64> &a, \
      const SIMDMask<TYPE, 64> &b) \
  { \
    return _k ## NAME ## _mask ## NUM (a, b); \
  }

  #define KNOT(TYPE, NUM) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  knot(const SIMDMask<TYPE, 64> &a) \
  { \
    return _knot_mask ## NUM (a); \
  }

  //shift with template parameter
  #define KSHIFT(R_OR_L, TYPE, NUM) \
  template <unsigned int IMM> \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  kshift ## R_OR_L ## i (const SIMDMask<TYPE, 64> &a) \
  { \
    return _kshift ## R_OR_L ## i_mask ## NUM (a, IMM); \
  }
#ifdef __AVX512BW__
  GENERATE_DMASKOP(and, SIMDByte, 64)
  GENERATE_DMASKOP(and, SIMDSignedByte, 64)
  GENERATE_DMASKOP(and, SIMDWord, 32)
  GENERATE_DMASKOP(and, SIMDShort, 32)

  GENERATE_DMASKOP(andn, SIMDByte, 64)
  GENERATE_DMASKOP(andn, SIMDSignedByte, 64)
  GENERATE_DMASKOP(andn, SIMDWord, 32)
  GENERATE_DMASKOP(andn, SIMDShort, 32)

  GENERATE_DMASKOP(or, SIMDByte, 64)
  GENERATE_DMASKOP(or, SIMDSignedByte, 64)
  GENERATE_DMASKOP(or, SIMDWord, 32)
  GENERATE_DMASKOP(or, SIMDShort, 32)

  GENERATE_DMASKOP(xor, SIMDByte, 64)
  GENERATE_DMASKOP(xor, SIMDSignedByte, 64)
  GENERATE_DMASKOP(xor, SIMDWord, 32)
  GENERATE_DMASKOP(xor, SIMDShort, 32)

  GENERATE_DMASKOP(xnor, SIMDByte, 64)
  GENERATE_DMASKOP(xnor, SIMDSignedByte, 64)
  GENERATE_DMASKOP(xnor, SIMDWord, 32)
  GENERATE_DMASKOP(xnor, SIMDShort, 32)

  GENERATE_DMASKOP(add, SIMDByte, 64)
  GENERATE_DMASKOP(add, SIMDSignedByte, 64)
  GENERATE_DMASKOP(add, SIMDWord, 32)
  GENERATE_DMASKOP(add, SIMDShort, 32)

  KNOT(SIMDByte, 64)
  KNOT(SIMDSignedByte, 64)
  KNOT(SIMDWord, 32)
  KNOT(SIMDShort, 32)
  
  KSHIFT(r, SIMDByte, 64)
  KSHIFT(r, SIMDSignedByte, 64)
  KSHIFT(r, SIMDWord, 32)
  KSHIFT(r, SIMDShort, 32)
  KSHIFT(l, SIMDByte, 64)
  KSHIFT(l, SIMDSignedByte, 64)
  KSHIFT(l, SIMDWord, 32)
  KSHIFT(l, SIMDShort, 32)
//else-case is further down
#endif //ifdef __AVX512BW__

  GENERATE_DMASKOP(and, SIMDInt, 16)
  GENERATE_DMASKOP(and, SIMDFloat, 16)

  GENERATE_DMASKOP(andn, SIMDInt, 16)
  GENERATE_DMASKOP(andn, SIMDFloat, 16)

  GENERATE_DMASKOP(or, SIMDInt, 16)
  GENERATE_DMASKOP(or, SIMDFloat, 16)

  GENERATE_DMASKOP(xor, SIMDInt, 16)
  GENERATE_DMASKOP(xor, SIMDFloat, 16)

  GENERATE_DMASKOP(xnor, SIMDInt, 16)
  GENERATE_DMASKOP(xnor, SIMDFloat, 16)

#ifdef __AVX512DQ__ //_kadd_mask16 is only available unter AVX512DQ
  GENERATE_DMASKOP(add, SIMDInt, 16)
  GENERATE_DMASKOP(add, SIMDFloat, 16)
#else
  EMULATE_KADD(SIMDInt)
  EMULATE_KADD(SIMDFloat)
#endif

  KNOT(SIMDInt, 16)
  KNOT(SIMDFloat, 16)
  
  KSHIFT(r, SIMDInt, 16)
  KSHIFT(r, SIMDFloat, 16)
  KSHIFT(l, SIMDInt, 16)
  KSHIFT(l, SIMDFloat, 16)
#else
//(__GNUC__ >= 7) is false
#if __GNUC__ >= 6
//At least the intrinsics for 16-masks (SIMDInt and SIMDFloat) are defined.
  #define GENERATE_DMASKOP(NAME, TYPE, NUM) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  k ## NAME (const SIMDMask<TYPE, 64> &a, \
      const SIMDMask<TYPE, 64> &b) \
  { \
    return _mm512_k ## NAME (a, b); \
  }

  #define KNOT(TYPE, NUM) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  knot(const SIMDMask<TYPE, 64> &a) \
  { \
    return _mm512_knot (a); \
  }
  GENERATE_DMASKOP(and, SIMDInt, 16)
  GENERATE_DMASKOP(and, SIMDFloat, 16)

  GENERATE_DMASKOP(andn, SIMDInt, 16)
  GENERATE_DMASKOP(andn, SIMDFloat, 16)

  GENERATE_DMASKOP(or, SIMDInt, 16)
  GENERATE_DMASKOP(or, SIMDFloat, 16)

  GENERATE_DMASKOP(xor, SIMDInt, 16)
  GENERATE_DMASKOP(xor, SIMDFloat, 16)

  GENERATE_DMASKOP(xnor, SIMDInt, 16)
  GENERATE_DMASKOP(xnor, SIMDFloat, 16)

  KNOT(SIMDInt, 16)
  KNOT(SIMDFloat, 16)
#endif

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kand (const SIMDMask<T, 64> &a,
      const SIMDMask<T, 64> &b)
  {
    return (a & b);
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kandn (const SIMDMask<T, 64> &a,
      const SIMDMask<T, 64> &b)
  {
    return (mask_all_ones(OutputType<T>(), Integer<64>()) & ((~a) & b));
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kor (const SIMDMask<T, 64> &a,
      const SIMDMask<T, 64> &b)
  {
    return (a | b);
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kxor (const SIMDMask<T, 64> &a,
      const SIMDMask<T, 64> &b)
  {
    return (a ^ b);
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kxnor (const SIMDMask<T, 64> &a,
      const SIMDMask<T, 64> &b)
  {
    return (mask_all_ones(OutputType<T>(), Integer<64>()) & (~(a ^ b)));
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kadd (const SIMDMask<T, 64> &a,
      const SIMDMask<T, 64> &b)
  {
    return (a + b);
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  knot(const SIMDMask<T, 64> &a)
  {
    return (mask_all_ones(OutputType<T>(), Integer<64>()) & (~a));
  }

  template <unsigned int IMM, typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kshiftri (const SIMDMask<T, 64> &a)
  {
    // 04. Aug 22 (Jonas Keller):
    // return zero if IMM is larger than 63, since then the >> operator is
    // undefined, but kshift should return zero
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
    // since IMM is a constant, the compiler should optimize away the if-statement
    if (IMM >= 64) {
      return 0;
    }
    // we checked that IMM is not too large above, disable warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wshift-count-overflow"
    return ((uint64_t) a) >> ((uint64_t) IMM);
    #pragma GCC diagnostic pop
  }

  template <unsigned int IMM, typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kshiftli (const SIMDMask<T, 64> &a)
  {
    // 04. Aug 22 (Jonas Keller):
    // return zero if IMM is larger than 63, since then the << operator is
    // undefined, but kshift should return zero
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
    // since IMM is a constant, the compiler should optimize away the if-statement
    if (IMM >= 64) {
      return 0;
    }
    // we checked that IMM is not too large above, disable warning
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wshift-count-overflow"
    return ((uint64_t) a) << ((uint64_t) IMM);
    #pragma GCC diagnostic pop
  }
#endif //if __GNUC__ >= 7
#ifndef __AVX512BW__
//Masks are vectors
  #define EMULATE_DMASKOP_64(NAME, TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  k ## NAME (const SIMDMask<TYPE, 64> &a, \
      const SIMDMask<TYPE, 64> &b) \
  { \
    return (SIMDMask<TYPE, 64>) NAME ((SIMDVec<TYPE, 64>) a, (SIMDVec<TYPE, 64>) b); \
  }
  EMULATE_DMASKOP_64(and, SIMDByte)
  EMULATE_DMASKOP_64(and, SIMDSignedByte)
  EMULATE_DMASKOP_64(and, SIMDWord)
  EMULATE_DMASKOP_64(and, SIMDShort)

  EMULATE_DMASKOP_64(or, SIMDByte)
  EMULATE_DMASKOP_64(or, SIMDSignedByte)
  EMULATE_DMASKOP_64(or, SIMDWord)
  EMULATE_DMASKOP_64(or, SIMDShort)

  EMULATE_DMASKOP_64(xor, SIMDByte)
  EMULATE_DMASKOP_64(xor, SIMDSignedByte)
  EMULATE_DMASKOP_64(xor, SIMDWord)
  EMULATE_DMASKOP_64(xor, SIMDShort)

  EMULATE_KADD(SIMDByte)
  EMULATE_KADD(SIMDSignedByte)
  EMULATE_KADD(SIMDWord)
  EMULATE_KADD(SIMDShort)

  #define EMULATE_KNOT(TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  knot(const SIMDMask<TYPE, 64> &a) \
  { \
    return (SIMDMask<TYPE, 64>) not((SIMDVec<TYPE, 64>) a); \
  }

  EMULATE_KNOT(SIMDByte)
  EMULATE_KNOT(SIMDSignedByte)
  EMULATE_KNOT(SIMDWord)
  EMULATE_KNOT(SIMDShort)

  // function name should be "kandn" but the vector function is "andnot"
  #define EMULATE_ANDN(TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  kandn (const SIMDMask<TYPE, 64> &a, \
      const SIMDMask<TYPE, 64> &b) \
  { \
    return (SIMDMask<TYPE, 64>) andnot ((SIMDVec<TYPE, 64>) a, (SIMDVec<TYPE, 64>) b); \
  }
  EMULATE_ANDN(SIMDByte)
  EMULATE_ANDN(SIMDSignedByte)
  EMULATE_ANDN(SIMDWord)
  EMULATE_ANDN(SIMDShort)

  // there is not xnor-function for vectors, so we have to do: not(xor(a, b))
  #define EMULATE_XNOR(TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  kxnor (const SIMDMask<TYPE, 64> &a, \
      const SIMDMask<TYPE, 64> &b) \
  { \
    return (SIMDMask<TYPE, 64>) not(xor((SIMDVec<TYPE, 64>) a, (SIMDVec<TYPE, 64>) b)); \
  }
  EMULATE_XNOR(SIMDByte)
  EMULATE_XNOR(SIMDSignedByte)
  EMULATE_XNOR(SIMDWord)
  EMULATE_XNOR(SIMDShort)
  
  #define EMULATE_KSHIFT(R_OR_L, TYPE, NUM) \
  template <unsigned int IMM> \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  kshift ## R_OR_L ## i (const SIMDMask<TYPE, 64> &a) \
  { \
    return (SIMDMask<TYPE, 64>) (s ## R_OR_L ## le<IMM>((SIMDVec<TYPE, 64>) a)); \
  }
  
  EMULATE_KSHIFT(r, SIMDByte, 64)
  EMULATE_KSHIFT(r, SIMDSignedByte, 64)
  EMULATE_KSHIFT(r, SIMDWord, 32)
  EMULATE_KSHIFT(r, SIMDShort, 32)
  EMULATE_KSHIFT(l, SIMDByte, 64)
  EMULATE_KSHIFT(l, SIMDSignedByte, 64)
  EMULATE_KSHIFT(l, SIMDWord, 32)
  EMULATE_KSHIFT(l, SIMDShort, 32)
#endif //ifndef __AVX512BW__
  //shift with flexible parameter (not template), probably slower than template-version
  /*//TODO faster implementation with switch-case possible?
  #define SHIFT_CASE(OP, NUM) case : OP<NUM>(a); break;

  #define EMULATE_KSHIFT(R_OR_L, OP, TYPE) \
  static SIMD_INLINE SIMDMask<TYPE, 64> \
  kshift ## R_OR_L ## i (const SIMDMask<TYPE, 64> &a, \
      uint64_t count) \
  { \
    return (a OP count); \
    switch(count) { \
      SHIFT_CASE(OP2, 0) \
      SHIFT_CASE(OP2, 1) \
    } \
  }*/

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kshiftli (const SIMDMask<T, 64> &a,
      uint64_t count)
  {
    // 04. Aug 22 (Jonas Keller):
    // return zero if count is larger than 63, since then the << operator is
    // undefined, but kshift should return zero
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
    if (count >= 64) {
      return 0;
    }
    return (((uint64_t) a) << count);
  }

  template <typename T>
  static SIMD_INLINE SIMDMask<T, 64>
  kshiftri (const SIMDMask<T, 64> &a,
      uint64_t count)
  {
    // 04. Aug 22 (Jonas Keller):
    // return zero if count is larger than 63, since then the >> operator is
    // undefined, but kshift should return zero
    // https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=kshift
    if (count >= 64) {
      return 0;
    }
    return (((uint64_t) a) >> count);
  }
} // namespace mask
} // namespace internal
} // namespace ns_simd

#endif // __AVX512F__

#endif // _SIMD_VEC_MASK_IMPL_INTEL_64_H_
