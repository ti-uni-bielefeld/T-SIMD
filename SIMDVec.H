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

// Please do not include this file directly! NATIVE_SIMD_WIDTH has to
// be defined according to the architecture in the files that are
// including SIMDVec.H.

#ifndef _SIMD_VEC_H_
#define _SIMD_VEC_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include <stdio.h>

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
#ifdef NATIVE_SIMD_WIDTH
  template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  class SIMDVec;
  template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  class SIMDMask;
#else
  template <typename T, int SIMD_WIDTH>
  class SIMDVec;
  template <typename T, int SIMD_WIDTH>
  class SIMDMask;
#endif
  
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

#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)

  // NATIVE_SIMD_WIDTH is defined and C++11 is available
  // (default template function arguments require C++11)

  template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  setzero();

  template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  set1(T a);

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH = NATIVE_SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  load(const T *const p);

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH = NATIVE_SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  loadu(const T *const p);

  // swizzle table
  template <int N, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  struct SwizzleTable;

  // 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
  template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_all_ones();

#else
  
  // NATIVE_SIMD_WIDTH is undefined or no C++11 available
  // (default template function arguments are not supported in C++98)
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  setzero();

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  set1(T a);

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  load(const T *const p);

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  loadu(const T *const p);

  // swizzle table
  template <int N, typename T, int SIMD_WIDTH>
  struct SwizzleTable;

  // 30. Sep 19 (rm): SIMDMask support contributed by Markus Vieth
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_all_ones();

#endif
  
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
