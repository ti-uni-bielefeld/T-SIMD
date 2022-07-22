// ===========================================================================
// 
// tsimd.H --
// different access to SIMD* types and functions, only works with C++11
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

#ifndef _TSIMD_H_
#define _TSIMD_H_

#include "SIMDVecAll.H"

// simd not allowed here
namespace ns_simd {

  // we can use element types simd::Byte instead of SIMDByte etc.
  typedef SIMDByte Byte;
  typedef SIMDSignedByte SignedByte;
  typedef SIMDWord Word;
  typedef SIMDShort Short;
  typedef SIMDInt Int;
  typedef SIMDFloat Float;

  // we can use simd::Vec instead of SIMDVec (requires C++11)
  // 15. Oct 21 (rm): add NATIVE_SIMD_WIDTH
  template <typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  using Vec = SIMDVec<T, SIMD_WIDTH>;

  // we can use simd::Vecs instead of SIMDVecs (requires C++11)
  // 15. Oct 21 (rm): add NATIVE_SIMD_WIDTH
  template <int NUM, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
  using Vecs = SIMDVecs<NUM, T, SIMD_WIDTH>;
  
  // we can use simd::TypeInfo instead of SIMDTypeInfo (requires C++11)
  template <typename T>
  using TypeInfo = SIMDTypeInfo<T>;

  // we can use simd::Format instead of SIMDFormat (requires C++11)
  template <typename T>
  using Format = SIMDFormat<T>;

  // we can use simd::Decimal instead of SIMDDecimal (requires C++11)
  template <typename T>
  using Decimal = SIMDDecimal<T>;
 
} // namespace ns_simd

// some IDEs don't see this if it comes first (compiles, but IDE has problems)
// shifted here:

// introduce alias namespace "simd"
namespace simd = ns_simd;

#endif
