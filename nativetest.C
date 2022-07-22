// ===========================================================================
// 
// nativetest.C --
// test of NATIVE_SIMD_WIDTH
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

// This is a small test of NATIVE_SIMD_WIDTH, the default SIMD_WIDTH
// for SIMDVec and some functions (those with primary templates). The
// SIMD_WIDTH parameter can be omitted and is selected automatically.
// To fully exploit this feature, at least C++11 is required
// (otherwise the function templates cannot have default parameters).

#include <stdio.h>
#include "SIMDVecAll.H"

using namespace ns_simd;

int
main()
{
#ifdef _SIMD_VEC_64_AVAIL_
  printf("_SIMD_VEC_64_AVAIL_\n");
#endif
#ifdef _SIMD_VEC_64_FULL_AVAIL_
  printf("_SIMD_VEC_64_FULL_AVAIL_\n");
#endif
#ifdef _SIMD_VEC_32_AVAIL_
  printf("_SIMD_VEC_32_AVAIL_\n");
#endif
#ifdef _SIMD_VEC_32_FULL_AVAIL_
  printf("_SIMD_VEC_32_FULL_AVAIL_\n");
#endif
#ifdef _SIMD_VEC_16_AVAIL_
  printf("_SIMD_VEC_16_AVAIL_\n");
#endif
  printf("NATIVE_SIMD_WIDTH = %d\n", NATIVE_SIMD_WIDTH);
  SIMDVec<SIMDFloat> a, b, c;
  const int n = SIMDVec<SIMDFloat>::elements;
  SIMDFloat buf[n];
  for (int i = 0; i < n; i++) buf[i] = i;
  
#if __cplusplus >= 201103L
  puts("C++11");
  a = setzero<SIMDFloat>();
  b = loadu(buf);
#else
  puts("C++98");
  a = setzero<SIMDFloat,NATIVE_SIMD_WIDTH>();
  b = loadu<NATIVE_SIMD_WIDTH>(buf);
#endif
  c = a + b;
  printf("a = "); print("%g ", a); puts("");
  printf("b = "); print("%g ", b); puts("");
  printf("c = "); print("%g ", c); puts("");
}
