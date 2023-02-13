// ===========================================================================
//
// tsimdtest.C --
// test of simd:: access to all types and functions
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

// This is a small test of an alternative way to access types and
// functions via the namespace "simd". Instead of
// SIMDVec<SIMDFloat,SW> we can now use simd::Vec<simd::Float,SW>, and
// instead of using ns_simd and calling setzero<SIMDFloat,SW> we can
// now use simd::setzero<simd::Float,SW>().
// This feature requires at least C++11.

#include "tsimd.H"

#include <stdio.h>

#define SW 16

int main()
{
  simd::Vec<simd::Float, SW> a, b, c;
  a = simd::setzero<simd::Float, SW>();
  b = simd::setzero<simd::Float, SW>();
  c = simd::add(a, b);
  simd::print("%g ", c);
  puts("");
  return 0;
}
