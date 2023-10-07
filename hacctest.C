// ===========================================================================
//
// hacctest.C
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

#include "tsimd.H"

using T = simd::Word;

constexpr auto SW     = 16;
constexpr auto FACTOR = 3;

int main()
{
  srand48(time(NULL));

  constexpr auto elems = simd::Vec<T, SW>::elems;
  T array[FACTOR * elems][elems];
  for (size_t i = 0; i < FACTOR * elems; i++) {
    T s = 0;
    for (size_t j = 0; j < elems; j++) {
      array[i][j] = T(drand48() * 4);
      s += array[i][j];
    }
    printf("%d,", s);
  }
  puts("");

  simd::HAcc<simd::HAdd, T, SW> hAcc;
  // const auto num = elems;
  const auto num = elems - 2;
  for (size_t i = 0; i < num; i++) {
    const simd::Vec<T, SW> v = simd::loadu<SW>(array[i]);
    hAcc.push(v);
  }
  hAcc.finish();
  if (hAcc.isDone()) {
    simd::print("%d,", hAcc.get());
    puts("");
  }

  T sum[FACTOR * elems];
  for (size_t i = 0; i < FACTOR * elems; i++) sum[i] = T(0);
  simd::HAccStore<simd::HAdd, T, SW> hAccStore(sum);
  // const auto num2 = FACTOR * elems;
  const auto num2 = (FACTOR - 1) * elems - 2;
  for (size_t i = 0; i < num2; i++) {
    const simd::Vec<T, SW> v = simd::loadu<SW>(array[i]);
    hAccStore.push(v);
  }
  hAccStore.finish();
  for (size_t i = 0; i < FACTOR * elems; i++) printf("%d,", sum[i]);
  puts("");

  return 0;
}
