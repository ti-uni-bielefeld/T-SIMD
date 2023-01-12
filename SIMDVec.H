// ===========================================================================
// 
// SIMDVec.H --
// generic template for SIMDVec and associated function templates
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

#ifndef _SIMD_VEC_H_
#define _SIMD_VEC_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"

#include <stdint.h>

namespace ns_simd {

  // TODO: - absdiff also for unsigned types -> SSE lecture vecintrin66
  // TODO: - bitwise shift: what about float?
  // TODO: - loadr / storer
  // TODO: - saturated workarounds for adds/hadds for SIMDInt?
  // TODO: - element-wise rotation in a vector using alignre
  // TODO: - functions for rsqrt, rcp Newton *steps*?
  // TODO: - add to names rcp, rsqrt something with "estimate"?
  // TODO: - NEON has a "set1" with immediate arguments (vmovq_n), so it would
  // TODO:   be nice to have a set1const function with template argument, but
  // TODO:   this only works for integers since float template parameter are not
  // TODO:   allowed by the standard

  // ===========================================================================
  // generic template for SIMDVec and SIMDMask
  // ===========================================================================

  // 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth

  // specialized for type of elements and number of bytes in the SIMD vector
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
  class SIMDVec;
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
  class SIMDMask;
  
  // ===========================================================================
  // primary template functions on SIMDVec
  // ===========================================================================
  
  // 14. Jul 16 (rm):
  // here we just list primary template functions;
  // overloaded functions are not listed, but for these functions we
  // have template versions in SIMDVecSandBox.H
  // (note that functions marked as "hub" are not found in SIMDVecSandBox.H)

  // 31. Jul 17 (rm): 

  // TODO: are these primary templates necessary for cvts or
  // TODO: could we have overloaded template functions instead?
  // TODO: what about packs?

  // hub:
  // apply as: x = reinterpret<Tdst>(y);
  // TODO: should a reference be returned? probably not, see specializations,
  // TODO: there we return the results of cast intrinsics
  template <typename Tdst, typename Tsrc, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tdst,SIMD_WIDTH>
  reinterpret(const SIMDVec<Tsrc,SIMD_WIDTH>& vec)
  {
    // return SIMDVec<Tdst,SIMD_WIDTH>(vec);
    // return reinterpret_cast<const SIMDVec<Tdst,SIMD_WIDTH>&>(vec);
    return reinterpret(vec, OutputType<Tdst>());
  }

  // conversion without changes in the number of vector elements
  // saturated version (but saturation only concerns float->int)
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  cvts(const SIMDVec<Tin, SIMD_WIDTH> &a);

  // unifies packs and packus, depending on Tout
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  packs(const SIMDVec<Tin, SIMD_WIDTH> &a,
	const SIMDVec<Tin, SIMD_WIDTH> &b);

  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  setzero();

  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  set1(T a);

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH_DEFAULT_NATIVE_FUNC, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  load(const T *const p);

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH_DEFAULT_NATIVE_FUNC, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  loadu(const T *const p);

  // swizzle table
  template <int N, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  struct SwizzleTable;

  // 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_all_ones();

  // 06. Oct 22 (Jonas Keller): added int2msb
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2msb(const uint64_t a);

  // 09. Oct 22 (Jonas Keller): added int2bits
  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2bits(const uint64_t a);
  
  // hub:
  // swizzle (AoS to SoA)
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<N, T, SIMD_WIDTH> &t,
	  SIMDVec<T, SIMD_WIDTH> *const v)
  {
    swizzle(t, v,
	    Integer<N>(),
	    TypeIsIntSize<T>());
  }

}

#endif
