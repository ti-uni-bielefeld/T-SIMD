// ===========================================================================
//
// SIMDIntrinsNEON.H --
// includes include files for vector intrinsics on ARM CPUs
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

#pragma once
#ifndef SIMD_INTRINS_NEON_H_
#define SIMD_INTRINS_NEON_H_

#include "SIMDDefs.H"

#ifdef SIMDVEC_NEON_ENABLE
#include <arm_neon.h>

// -------------------------------------------------------------------------
// single element array type (for generalizations)
// -------------------------------------------------------------------------

// defined as the other array types in arm_neon.h
// but with type conversion constructor and operator=
#define SIMDVEC_NEON_64X1(NEON_T)                                              \
  struct NEON_T##x1_t                                                          \
  {                                                                            \
    NEON_T##_t val[1];                                                         \
    NEON_T##x1_t() {}                                                          \
    NEON_T##x1_t(const NEON_T##_t &x)                                          \
    {                                                                          \
      val[0] = x;                                                              \
    }                                                                          \
    NEON_T##x1_t &operator=(const NEON_T##_t &x)                               \
    {                                                                          \
      val[0] = x;                                                              \
      return *this;                                                            \
    }                                                                          \
  };

SIMDVEC_NEON_64X1(uint8x8)
SIMDVEC_NEON_64X1(int8x8)
SIMDVEC_NEON_64X1(uint16x4)
SIMDVEC_NEON_64X1(int16x4)
SIMDVEC_NEON_64X1(int32x2)
SIMDVEC_NEON_64X1(float32x2)
#ifdef SIMD_64BIT_TYPES
SIMDVEC_NEON_64X1(int64x1)
SIMDVEC_NEON_64X1(float64x1)
#endif

#undef SIMDVEC_NEON_64X1

// -------------------------------------------------------------------------
// vreinterpret[q] with same input and output type (not avail. as intrinsic)
// -------------------------------------------------------------------------

#define SIMDVEC_NEON_VREINTERPRET_SAME(T, NEON_SUF)                            \
  static SIMD_INLINE T vreinterpretq_##NEON_SUF##_##NEON_SUF(T a)              \
  {                                                                            \
    return a;                                                                  \
  }

SIMDVEC_NEON_VREINTERPRET_SAME(uint8x16_t, u8)
SIMDVEC_NEON_VREINTERPRET_SAME(int8x16_t, s8)
SIMDVEC_NEON_VREINTERPRET_SAME(uint16x8_t, u16)
SIMDVEC_NEON_VREINTERPRET_SAME(int16x8_t, s16)
SIMDVEC_NEON_VREINTERPRET_SAME(uint32x4_t, u32)
SIMDVEC_NEON_VREINTERPRET_SAME(int32x4_t, s32)
SIMDVEC_NEON_VREINTERPRET_SAME(float32x4_t, f32)
#ifdef SIMD_64BIT_TYPES
SIMDVEC_NEON_VREINTERPRET_SAME(uint64x2_t, u64)
SIMDVEC_NEON_VREINTERPRET_SAME(int64x2_t, s64)
SIMDVEC_NEON_VREINTERPRET_SAME(float64x2_t, f64)
#endif

#endif

#endif // SIMD_INTRINS_NEON_H_
