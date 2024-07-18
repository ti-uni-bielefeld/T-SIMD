// ===========================================================================
//
// SIMDExperimental.H --
// place to put experimental code
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

// NOTE: This file contains experimental code which hasn't been tested
// NOTE: at all or hasn't been tested in depth.

#pragma once
#ifndef SIMD_EXPERIMENTAL_H_
#define SIMD_EXPERIMENTAL_H_

#include "tsimd.H"

namespace simd {

// ========================================================================
// integration of values in simd::Vec
// ========================================================================

// example:
//        0    1    2    3    4    5    6    7
// +      -    0    1    2    3    4    5    6    slle(1)
//      --------------------------------------
//        0  0-1  1-2  2-3  3-4  4-5  5-6  6-7
// +      -    -    0  0-1  1-2  2-3  3-4  4-5    slle(2)
//      --------------------------------------
//        0  0-1  0-2  0-3  1-4  2-5  3-6  4-7
// +      -    -    -    -    0  0-1  0-2  0-3    slle(4)
//      --------------------------------------
//        0  0-1  0-2  0-3  0-4  0-5  0-6  0-7
//
// problem: slle has immediate argument, T-SIMD: template parameter

// primary template
template <typename T, size_t SIMD_WIDTH, int SHIFT, int END_SHIFT>
struct HInt
{
public:
  static SIMD_INLINE simd::Vec<T, SIMD_WIDTH> integrate(
    const simd::Vec<T, SIMD_WIDTH> &v)
  {
    return HInt<T, SIMD_WIDTH, 2 * SHIFT, END_SHIFT>::integrate(
      simd::add(v, simd::slle<SHIFT>(v)));
  }
};

// termination template
template <typename T, size_t SIMD_WIDTH, int END_SHIFT>
struct HInt<T, SIMD_WIDTH, END_SHIFT, END_SHIFT>
{
public:
  static SIMD_INLINE simd::Vec<T, SIMD_WIDTH> integrate(
    const simd::Vec<T, SIMD_WIDTH> &v)
  {
    return v;
  }
};

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE simd::Vec<T, SIMD_WIDTH> integrate(
  const simd::Vec<T, SIMD_WIDTH> &v)
{
  return HInt<T, SIMD_WIDTH, 1, simd::Vec<T, SIMD_WIDTH>::elems>::integrate(v);
}

} // namespace simd

#endif
