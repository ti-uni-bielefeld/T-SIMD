// ===========================================================================
//
// test of NATIVE_SIMD_WIDTH
//
// This source code file is part of the following software:
//
//    - the low-level C++ template SIMD library
//    - the SIMD implementation of the MinWarping and the 2D-Warping methods
//      for local visual homing.
//
// The software is provided based on the accompanying license agreement in the
// file LICENSE.md.
// The software is provided "as is" without any warranty by the licensor and
// without any liability of the licensor, and the software may not be
// distributed by the licensee; see the license agreement for details.
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

#include <tsimd.H>

#include <cstdio>

using namespace ns_simd;

int main()
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
  const auto n = SIMDVec<SIMDFloat>::elements;
  SIMDFloat buf[n];
  for (size_t i = 0; i < n; i++) buf[i] = i;
  a = setzero<SIMDFloat>();
  b = loadu(buf);
  c = a + b;
  printf("a = ");
  print("%g ", a);
  puts("");
  printf("b = ");
  print("%g ", b);
  puts("");
  printf("c = ");
  print("%g ", c);
  puts("");
}
