// ===========================================================================
// 
// SIMDMaskNEON16.H --
// Mask class definitions and architecture specific functions
// for Arm Neon 16 byte (128 bit)
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

#ifndef _SIMDMASKNEON16_H_
#define _SIMDMASKNEON16_H_

#include "SIMDDefs.H"
#include "SIMDIntrinsNEON.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <stdint.h>

#ifdef SIMDVEC_NEON_ENABLE

namespace ns_simd {
  #define TEST_AND_SET(SUF, NUM, INDEX) \
    if(((1<<INDEX)&x)!=0) { k=vsetq_lane_ ## SUF (NUM, k, INDEX); }

  #define GET_AND_SET_BIT(SUF, INDEX) \
    if(vgetq_lane_ ## SUF (x, INDEX)!=0) { res|=(1<<INDEX); }

  #define CASE_SWITCH(SUF, INDEX) \
    case INDEX: return vgetq_lane_ ## SUF (k, INDEX)!=0;

  template <>
  class SIMDMask<SIMDByte, 16>
  {
  public:
    uint8x16_t k;
    SIMDMask() { k=vdupq_n_u8(0); }
    SIMDMask(const uint8x16_t &x) { k = x; }
    SIMDMask(const uint16_t x) { mask_from_int(x); }
    SIMDMask& operator=(const uint8x16_t &x) { k = x; return *this; }
    SIMDMask& operator=(const uint16_t x) { mask_from_int(x); return *this; }
    operator uint8x16_t() const { return k; }
    operator uint16_t() const { return mask_to_int(k); }
    bool operator[](const uint8_t i) const {
      switch(i) {
        CASE_SWITCH(u8, 0)
        CASE_SWITCH(u8, 1)
        CASE_SWITCH(u8, 2)
        CASE_SWITCH(u8, 3)
        CASE_SWITCH(u8, 4)
        CASE_SWITCH(u8, 5)
        CASE_SWITCH(u8, 6)
        CASE_SWITCH(u8, 7)
        CASE_SWITCH(u8, 8)
        CASE_SWITCH(u8, 9)
        CASE_SWITCH(u8, 10)
        CASE_SWITCH(u8, 11)
        CASE_SWITCH(u8, 12)
        CASE_SWITCH(u8, 13)
        CASE_SWITCH(u8, 14)
        CASE_SWITCH(u8, 15)
        default: return false;
      }
    }
    bool operator==(const SIMDMask<SIMDByte, 16> &x) {
      return test_all_ones(cmpeq(SIMDVec<SIMDByte,16>(x), SIMDVec<SIMDByte,16>(k)));
    }
    SIMD_INLINE void
    mask_from_int(const uint16_t x) {
      k=vdupq_n_u8(0);
      TEST_AND_SET(u8, 0xFF, 0)
      TEST_AND_SET(u8, 0xFF, 1)
      TEST_AND_SET(u8, 0xFF, 2)
      TEST_AND_SET(u8, 0xFF, 3)
      TEST_AND_SET(u8, 0xFF, 4)
      TEST_AND_SET(u8, 0xFF, 5)
      TEST_AND_SET(u8, 0xFF, 6)
      TEST_AND_SET(u8, 0xFF, 7)
      TEST_AND_SET(u8, 0xFF, 8)
      TEST_AND_SET(u8, 0xFF, 9)
      TEST_AND_SET(u8, 0xFF, 10)
      TEST_AND_SET(u8, 0xFF, 11)
      TEST_AND_SET(u8, 0xFF, 12)
      TEST_AND_SET(u8, 0xFF, 13)
      TEST_AND_SET(u8, 0xFF, 14)
      TEST_AND_SET(u8, 0xFF, 15)
    }
    static SIMD_INLINE uint16_t
    mask_to_int(const uint8x16_t x) {
      uint16_t res=0;
      GET_AND_SET_BIT(u8, 0)
      GET_AND_SET_BIT(u8, 1)
      GET_AND_SET_BIT(u8, 2)
      GET_AND_SET_BIT(u8, 3)
      GET_AND_SET_BIT(u8, 4)
      GET_AND_SET_BIT(u8, 5)
      GET_AND_SET_BIT(u8, 6)
      GET_AND_SET_BIT(u8, 7)
      GET_AND_SET_BIT(u8, 8)
      GET_AND_SET_BIT(u8, 9)
      GET_AND_SET_BIT(u8, 10)
      GET_AND_SET_BIT(u8, 11)
      GET_AND_SET_BIT(u8, 12)
      GET_AND_SET_BIT(u8, 13)
      GET_AND_SET_BIT(u8, 14)
      GET_AND_SET_BIT(u8, 15)
      return res;
    }
  };

  template <>
  class SIMDMask<SIMDSignedByte, 16>
  {
  public:
    int8x16_t k;
    SIMDMask() { k=vdupq_n_s8(0); }
    SIMDMask(const int8x16_t &x) { k = x; }
    SIMDMask(const uint16_t x) { mask_from_int(x); }
    SIMDMask& operator=(const int8x16_t &x) { k = x; return *this; }
    SIMDMask& operator=(const uint16_t x) { mask_from_int(x); return *this; }
    operator int8x16_t() const { return k; }
    operator uint16_t() const { return mask_to_int(k); }
    bool operator[](const uint8_t i) const {
      switch(i) {
        CASE_SWITCH(s8, 0)
        CASE_SWITCH(s8, 1)
        CASE_SWITCH(s8, 2)
        CASE_SWITCH(s8, 3)
        CASE_SWITCH(s8, 4)
        CASE_SWITCH(s8, 5)
        CASE_SWITCH(s8, 6)
        CASE_SWITCH(s8, 7)
        CASE_SWITCH(s8, 8)
        CASE_SWITCH(s8, 9)
        CASE_SWITCH(s8, 10)
        CASE_SWITCH(s8, 11)
        CASE_SWITCH(s8, 12)
        CASE_SWITCH(s8, 13)
        CASE_SWITCH(s8, 14)
        CASE_SWITCH(s8, 15)
        default: return false;
      }
    }
    bool operator==(const SIMDMask<SIMDSignedByte, 16> &x) {
      return test_all_ones(cmpeq(SIMDVec<SIMDSignedByte,16>(x), SIMDVec<SIMDSignedByte,16>(k)));
    }
    SIMD_INLINE void
    mask_from_int(const uint16_t x) {
      k=vdupq_n_s8(0);
      TEST_AND_SET(s8, -1, 0)
      TEST_AND_SET(s8, -1, 1)
      TEST_AND_SET(s8, -1, 2)
      TEST_AND_SET(s8, -1, 3)
      TEST_AND_SET(s8, -1, 4)
      TEST_AND_SET(s8, -1, 5)
      TEST_AND_SET(s8, -1, 6)
      TEST_AND_SET(s8, -1, 7)
      TEST_AND_SET(s8, -1, 8)
      TEST_AND_SET(s8, -1, 9)
      TEST_AND_SET(s8, -1, 10)
      TEST_AND_SET(s8, -1, 11)
      TEST_AND_SET(s8, -1, 12)
      TEST_AND_SET(s8, -1, 13)
      TEST_AND_SET(s8, -1, 14)
      TEST_AND_SET(s8, -1, 15)
    }
    static SIMD_INLINE uint16_t
    mask_to_int(const int8x16_t x) {
      uint16_t res=0;
      GET_AND_SET_BIT(s8, 0)
      GET_AND_SET_BIT(s8, 1)
      GET_AND_SET_BIT(s8, 2)
      GET_AND_SET_BIT(s8, 3)
      GET_AND_SET_BIT(s8, 4)
      GET_AND_SET_BIT(s8, 5)
      GET_AND_SET_BIT(s8, 6)
      GET_AND_SET_BIT(s8, 7)
      GET_AND_SET_BIT(s8, 8)
      GET_AND_SET_BIT(s8, 9)
      GET_AND_SET_BIT(s8, 10)
      GET_AND_SET_BIT(s8, 11)
      GET_AND_SET_BIT(s8, 12)
      GET_AND_SET_BIT(s8, 13)
      GET_AND_SET_BIT(s8, 14)
      GET_AND_SET_BIT(s8, 15)
      return res;
    }
  };

  template <>
  class SIMDMask<SIMDWord, 16>
  {
  public:
    uint16x8_t k;
    SIMDMask() { k=vdupq_n_u16(0); }
    SIMDMask(const uint16x8_t &x) { k = x; }
    SIMDMask(const uint8_t x) { mask_from_int(x); }
    SIMDMask& operator=(const uint16x8_t &x) { k = x; return *this; }
    SIMDMask& operator=(const uint8_t x) { mask_from_int(x); return *this; }
    operator uint16x8_t() const { return k; }
    operator uint8_t() const { return mask_to_int(k); }
    bool operator[](const uint8_t i) const {
      switch(i) {
        CASE_SWITCH(u16, 0)
        CASE_SWITCH(u16, 1)
        CASE_SWITCH(u16, 2)
        CASE_SWITCH(u16, 3)
        CASE_SWITCH(u16, 4)
        CASE_SWITCH(u16, 5)
        CASE_SWITCH(u16, 6)
        CASE_SWITCH(u16, 7)
        default: return false;
      }
    }
    bool operator==(const SIMDMask<SIMDWord, 16> &x) {
      return test_all_ones(cmpeq(SIMDVec<SIMDWord,16>(x), SIMDVec<SIMDWord,16>(k)));
    }
    SIMD_INLINE void
    mask_from_int(const uint8_t x) {
      k=vdupq_n_u16(0);
      TEST_AND_SET(u16, 0xFFFF, 0)
      TEST_AND_SET(u16, 0xFFFF, 1)
      TEST_AND_SET(u16, 0xFFFF, 2)
      TEST_AND_SET(u16, 0xFFFF, 3)
      TEST_AND_SET(u16, 0xFFFF, 4)
      TEST_AND_SET(u16, 0xFFFF, 5)
      TEST_AND_SET(u16, 0xFFFF, 6)
      TEST_AND_SET(u16, 0xFFFF, 7)
    }
    static SIMD_INLINE uint8_t
    mask_to_int(const uint16x8_t x) {
      uint8_t res=0;
      GET_AND_SET_BIT(u16, 0)
      GET_AND_SET_BIT(u16, 1)
      GET_AND_SET_BIT(u16, 2)
      GET_AND_SET_BIT(u16, 3)
      GET_AND_SET_BIT(u16, 4)
      GET_AND_SET_BIT(u16, 5)
      GET_AND_SET_BIT(u16, 6)
      GET_AND_SET_BIT(u16, 7)
      return res;
    }
  };

  template <>
  class SIMDMask<SIMDShort, 16>
  {
  public:
    int16x8_t k;
    SIMDMask() { k=vdupq_n_s16(0); }
    SIMDMask(const int16x8_t &x) { k = x; }
    SIMDMask(const uint8_t x) { mask_from_int(x); }
    SIMDMask& operator=(const int16x8_t &x) { k = x; return *this; }
    SIMDMask& operator=(const uint8_t x) { mask_from_int(x); return *this; }
    operator int16x8_t() const { return k; }
    operator uint8_t() const { return mask_to_int(k); }
    bool operator[](const uint8_t i) const {
      switch(i) {
        CASE_SWITCH(s16, 0)
        CASE_SWITCH(s16, 1)
        CASE_SWITCH(s16, 2)
        CASE_SWITCH(s16, 3)
        CASE_SWITCH(s16, 4)
        CASE_SWITCH(s16, 5)
        CASE_SWITCH(s16, 6)
        CASE_SWITCH(s16, 7)
        default: return false;
      }
    }
    bool operator==(const SIMDMask<SIMDShort, 16> &x) {
      return test_all_ones(cmpeq(SIMDVec<SIMDShort,16>(x), SIMDVec<SIMDShort,16>(k)));
    }
    SIMD_INLINE void
    mask_from_int(const uint8_t x) {
      k=vdupq_n_s16(0);
      TEST_AND_SET(s16, -1, 0)
      TEST_AND_SET(s16, -1, 1)
      TEST_AND_SET(s16, -1, 2)
      TEST_AND_SET(s16, -1, 3)
      TEST_AND_SET(s16, -1, 4)
      TEST_AND_SET(s16, -1, 5)
      TEST_AND_SET(s16, -1, 6)
      TEST_AND_SET(s16, -1, 7)
    }
    static SIMD_INLINE uint8_t
    mask_to_int(const int16x8_t x) {
      uint8_t res=0;
      GET_AND_SET_BIT(s16, 0)
      GET_AND_SET_BIT(s16, 1)
      GET_AND_SET_BIT(s16, 2)
      GET_AND_SET_BIT(s16, 3)
      GET_AND_SET_BIT(s16, 4)
      GET_AND_SET_BIT(s16, 5)
      GET_AND_SET_BIT(s16, 6)
      GET_AND_SET_BIT(s16, 7)
      return res;
    }
  };

  template <>
  class SIMDMask<SIMDInt, 16>
  {
  public:
    int32x4_t k;
    SIMDMask() { k=vdupq_n_s32(0); }
    SIMDMask(const int32x4_t &x) { k = x; }
    SIMDMask(const uint8_t x) { mask_from_int(x); }
    SIMDMask& operator=(const int32x4_t &x) { k = x; return *this; }
    SIMDMask& operator=(const uint8_t x) { mask_from_int(x); return *this; }
    operator int32x4_t() const { return k; }
    operator uint8_t() const { return mask_to_int(k); }
    bool operator[](const uint8_t i) const {
      switch(i) {
        CASE_SWITCH(s32, 0)
        CASE_SWITCH(s32, 1)
        CASE_SWITCH(s32, 2)
        CASE_SWITCH(s32, 3)
        default: return false;
      }
    }
    bool operator==(const SIMDMask<SIMDInt, 16> &x) {
      return test_all_ones(cmpeq(SIMDVec<SIMDInt,16>(x), SIMDVec<SIMDInt,16>(k)));
    }
    SIMD_INLINE void
    mask_from_int(const uint8_t x) {
      k=vdupq_n_s32(0);
      TEST_AND_SET(s32, -1, 0)
      TEST_AND_SET(s32, -1, 1)
      TEST_AND_SET(s32, -1, 2)
      TEST_AND_SET(s32, -1, 3)
    }
    SIMD_INLINE uint8_t
    static mask_to_int(const int32x4_t x) {
      uint8_t res=0;
      GET_AND_SET_BIT(s32, 0)
      GET_AND_SET_BIT(s32, 1)
      GET_AND_SET_BIT(s32, 2)
      GET_AND_SET_BIT(s32, 3)
      return res;
    }
  };

  template <>
  class SIMDMask<SIMDFloat, 16>
  {
  public:
    float32x4_t k;
    SIMDMask() { k=vdupq_n_f32(0.0); }
    SIMDMask(const float32x4_t &x) { k = x; }
    SIMDMask(const uint8_t x) { mask_from_int(x); }
    SIMDMask& operator=(const float32x4_t &x) { k = x; return *this; }
    SIMDMask& operator=(const uint8_t x) { mask_from_int(x); return *this; }
    operator float32x4_t() const { return k; }
    operator uint8_t() const { return mask_to_int(k); }
    bool operator[](const uint8_t i) const {
      switch(i) {
        CASE_SWITCH(f32, 0)
        CASE_SWITCH(f32, 1)
        CASE_SWITCH(f32, 2)
        CASE_SWITCH(f32, 3)
        default: return false;
      }
    }
    bool operator==(const SIMDMask<SIMDFloat, 16> &x) {
      return test_all_ones(cmpeq(SIMDVec<SIMDInt,16>(vreinterpretq_s32_f32(x)), SIMDVec<SIMDInt,16>(vreinterpretq_s32_f32(k)))); //reinterprets because nan==nan is false but should be true
    }
    SIMD_INLINE void
    mask_from_int(const uint8_t x) {
      int32x4_t ktemp=vdupq_n_s32(0);
      if(((1<<0)&x)!=0) { ktemp=vsetq_lane_s32 (-1, ktemp, 0); }
      if(((1<<1)&x)!=0) { ktemp=vsetq_lane_s32 (-1, ktemp, 1); }
      if(((1<<2)&x)!=0) { ktemp=vsetq_lane_s32 (-1, ktemp, 2); }
      if(((1<<3)&x)!=0) { ktemp=vsetq_lane_s32 (-1, ktemp, 3); }
      k=vreinterpretq_f32_s32(ktemp);
    }
    static SIMD_INLINE uint8_t
    mask_to_int(const float32x4_t x) {
      int32x4_t ktemp=vreinterpretq_s32_f32(x);
      uint8_t res=0;
      if(vgetq_lane_s32(ktemp, 0)!=0) { res|=(1<<0); }
      if(vgetq_lane_s32(ktemp, 1)!=0) { res|=(1<<1); }
      if(vgetq_lane_s32(ktemp, 2)!=0) { res|=(1<<2); }
      if(vgetq_lane_s32(ktemp, 3)!=0) { res|=(1<<3); }
      return res;
    }
  };

  template <>
  SIMD_INLINE SIMDMask<SIMDByte, 16>
  mask_all_ones()
  {
    return vdupq_n_u8(0xFF);
  }

  template <>
  SIMD_INLINE SIMDMask<SIMDSignedByte, 16>
  mask_all_ones()
  {
    return vdupq_n_s8(-1);
  }

  template <>
  SIMD_INLINE SIMDMask<SIMDWord, 16>
  mask_all_ones()
  {
    return vdupq_n_u16(0xFFFF);
  }

  template <>
  SIMD_INLINE SIMDMask<SIMDShort, 16>
  mask_all_ones()
  {
    return vdupq_n_s16(-1);
  }

  template <>
  SIMD_INLINE SIMDMask<SIMDInt, 16>
  mask_all_ones()
  {
    return vdupq_n_s32(-1);
  }

  template <>
  SIMD_INLINE SIMDMask<SIMDFloat, 16>
  mask_all_ones()
  {
    return vreinterpretq_f32_s32(vdupq_n_s32(-1));
  }
  
  #define CHECK_AND_INSERT(OP, INDEX) \
  if(k[INDEX]) { \
    result = OP (p[INDEX], result, INDEX); /* argument order is different than for Intel */ \
  }
  
  static SIMD_INLINE SIMDVec<SIMDByte, 16>
  maskz_load(const SIMDMask<SIMDByte, 16> &k,
      const SIMDByte *const p)
  {
    SIMDVec<SIMDByte, 16> result=setzero<SIMDByte, 16>();
    CHECK_AND_INSERT(vsetq_lane_u8, 0)
    CHECK_AND_INSERT(vsetq_lane_u8, 1)
    CHECK_AND_INSERT(vsetq_lane_u8, 2)
    CHECK_AND_INSERT(vsetq_lane_u8, 3)
    CHECK_AND_INSERT(vsetq_lane_u8, 4)
    CHECK_AND_INSERT(vsetq_lane_u8, 5)
    CHECK_AND_INSERT(vsetq_lane_u8, 6)
    CHECK_AND_INSERT(vsetq_lane_u8, 7)
    CHECK_AND_INSERT(vsetq_lane_u8, 8)
    CHECK_AND_INSERT(vsetq_lane_u8, 9)
    CHECK_AND_INSERT(vsetq_lane_u8, 10)
    CHECK_AND_INSERT(vsetq_lane_u8, 11)
    CHECK_AND_INSERT(vsetq_lane_u8, 12)
    CHECK_AND_INSERT(vsetq_lane_u8, 13)
    CHECK_AND_INSERT(vsetq_lane_u8, 14)
    CHECK_AND_INSERT(vsetq_lane_u8, 15)
    return result;
  }
  static SIMD_INLINE SIMDVec<SIMDSignedByte, 16>
  maskz_load(const SIMDMask<SIMDSignedByte, 16> &k,
      const SIMDSignedByte *const p)
  {
    SIMDVec<SIMDSignedByte, 16> result=setzero<SIMDSignedByte, 16>();
    CHECK_AND_INSERT(vsetq_lane_s8, 0)
    CHECK_AND_INSERT(vsetq_lane_s8, 1)
    CHECK_AND_INSERT(vsetq_lane_s8, 2)
    CHECK_AND_INSERT(vsetq_lane_s8, 3)
    CHECK_AND_INSERT(vsetq_lane_s8, 4)
    CHECK_AND_INSERT(vsetq_lane_s8, 5)
    CHECK_AND_INSERT(vsetq_lane_s8, 6)
    CHECK_AND_INSERT(vsetq_lane_s8, 7)
    CHECK_AND_INSERT(vsetq_lane_s8, 8)
    CHECK_AND_INSERT(vsetq_lane_s8, 9)
    CHECK_AND_INSERT(vsetq_lane_s8, 10)
    CHECK_AND_INSERT(vsetq_lane_s8, 11)
    CHECK_AND_INSERT(vsetq_lane_s8, 12)
    CHECK_AND_INSERT(vsetq_lane_s8, 13)
    CHECK_AND_INSERT(vsetq_lane_s8, 14)
    CHECK_AND_INSERT(vsetq_lane_s8, 15)
    return result;
  }
  static SIMD_INLINE SIMDVec<SIMDWord, 16>
  maskz_load(const SIMDMask<SIMDWord, 16> &k,
      const SIMDWord *const p)
  {
    SIMDVec<SIMDWord, 16> result=setzero<SIMDWord, 16>();
    CHECK_AND_INSERT(vsetq_lane_u16, 0)
    CHECK_AND_INSERT(vsetq_lane_u16, 1)
    CHECK_AND_INSERT(vsetq_lane_u16, 2)
    CHECK_AND_INSERT(vsetq_lane_u16, 3)
    CHECK_AND_INSERT(vsetq_lane_u16, 4)
    CHECK_AND_INSERT(vsetq_lane_u16, 5)
    CHECK_AND_INSERT(vsetq_lane_u16, 6)
    CHECK_AND_INSERT(vsetq_lane_u16, 7)
    return result;
  }
  static SIMD_INLINE SIMDVec<SIMDShort, 16>
  maskz_load(const SIMDMask<SIMDShort, 16> &k,
      const SIMDShort *const p)
  {
    SIMDVec<SIMDShort, 16> result=setzero<SIMDShort, 16>();
    CHECK_AND_INSERT(vsetq_lane_s16, 0)
    CHECK_AND_INSERT(vsetq_lane_s16, 1)
    CHECK_AND_INSERT(vsetq_lane_s16, 2)
    CHECK_AND_INSERT(vsetq_lane_s16, 3)
    CHECK_AND_INSERT(vsetq_lane_s16, 4)
    CHECK_AND_INSERT(vsetq_lane_s16, 5)
    CHECK_AND_INSERT(vsetq_lane_s16, 6)
    CHECK_AND_INSERT(vsetq_lane_s16, 7)
    return result;
  }
  static SIMD_INLINE SIMDVec<SIMDInt, 16>
  maskz_load(const SIMDMask<SIMDInt, 16> &k,
      const SIMDInt *const p)
  {
    SIMDVec<SIMDInt, 16> result=setzero<SIMDInt, 16>();
    CHECK_AND_INSERT(vsetq_lane_s32, 0)
    CHECK_AND_INSERT(vsetq_lane_s32, 1)
    CHECK_AND_INSERT(vsetq_lane_s32, 2)
    CHECK_AND_INSERT(vsetq_lane_s32, 3)
    return result;
  }
  static SIMD_INLINE SIMDVec<SIMDFloat, 16>
  maskz_load(const SIMDMask<SIMDFloat, 16> &k,
      const SIMDFloat *const p)
  {
    SIMDVec<SIMDFloat, 16> result=setzero<SIMDFloat, 16>();
    CHECK_AND_INSERT(vsetq_lane_f32, 0)
    CHECK_AND_INSERT(vsetq_lane_f32, 1)
    CHECK_AND_INSERT(vsetq_lane_f32, 2)
    CHECK_AND_INSERT(vsetq_lane_f32, 3)
    return result;
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
  }
  static SIMD_INLINE void
  mask_store(SIMDSignedByte *const p,
      const SIMDMask<SIMDSignedByte, 16> &k,
      const SIMDVec<SIMDSignedByte, 16> &a)
  {
    CHECK_AND_STORE16
  }
  static SIMD_INLINE void
  mask_store(SIMDWord *const p,
      const SIMDMask<SIMDWord, 16> &k,
      const SIMDVec<SIMDWord, 16> &a)
  {
    CHECK_AND_STORE8
  }
  static SIMD_INLINE void
  mask_store(SIMDShort *const p,
      const SIMDMask<SIMDShort, 16> &k,
      const SIMDVec<SIMDShort, 16> &a)
  {
    CHECK_AND_STORE8
  }
  static SIMD_INLINE void
  mask_store(SIMDInt *const p,
      const SIMDMask<SIMDInt, 16> &k,
      const SIMDVec<SIMDInt, 16> &a)
  {
    CHECK_AND_STORE4
  }
  static SIMD_INLINE void
  mask_store(SIMDFloat *const p,
      const SIMDMask<SIMDFloat, 16> &k,
      const SIMDVec<SIMDFloat, 16> &a)
  {
    CHECK_AND_STORE4
  }
} //namespace ns_simd
#endif //ifdef SIMDVEC_NEON_ENABLE
#endif // _SIMDMASKNEON16_H_
