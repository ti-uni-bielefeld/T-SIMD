// ===========================================================================
//
// styleExamples.C --
// examples of different styles of using the T-SIMD library
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

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// #define MAX_SIMD_WIDTH 16

#include "tsimd.H"

// ===========================================================================

// for classic style and modern style:
// SW = SIMD width = number of bytes in a single SIMD vector

// use fixed SIMD_WIDTH (16, 32, 64)
static constexpr size_t SW = 16;

// alternative: use widest SIMD_WIDTH available
// #define SW NATIVE_SIMD_WIDTH

// ===========================================================================
// Vector class
// ===========================================================================

template <typename T>
struct Vector
{
  T *data;
  size_t size;

  Vector() : data(0), size(0) {}
  Vector(size_t size, size_t alignment) : size(size)
  {
    data = (T *) simd_aligned_malloc(alignment, size * sizeof(T));
    assert(data != nullptr);
    // 20. Sep 22 (Jonas Keller): use memset instead of bzero, since bzero
    // is not available on Windows
    // bzero(data, size * sizeof(T));
    memset(data, 0, size * sizeof(T));
  }
  ~Vector() { simd_aligned_free(data); }
};

// ===========================================================================
// native style, automatic selection of widest possible simd vectors
// ===========================================================================

// vector element types could be specified e.g. with simd::Float

template <typename T>
T vectorSumNative(const Vector<T> &input)
{
  simd::Vec<T> vecSum = simd::setzero<T>();
  // you can use vecSum.elements or simd::Vec<T>::elements
  printf("native style:  elements %zu, bytes %zu\n", vecSum.elements,
         vecSum.bytes);
  // parallel part
  const size_t numVecAdds = input.size / vecSum.elements;
  for (size_t i = 0; i < numVecAdds; i++) {
    vecSum = simd::add(vecSum, simd::load(&input.data[i * vecSum.elements]));
  }
  // postamble
  T seqSum = 0;
  for (size_t i = numVecAdds * vecSum.elements; i < input.size; i++) {
    seqSum += input.data[i];
  }
  return simd::hadd(vecSum) + seqSum;
}

// ===========================================================================
// modern style, like native style, but explicit specification of SIMD_WIDTH
// ===========================================================================

// vector element types could be specified e.g. with simd::Float

template <size_t SIMD_WIDTH, typename T>
T vectorSumModern(const Vector<T> &input)
{
  simd::Vec<T, SIMD_WIDTH> vecSum = simd::setzero<T, SIMD_WIDTH>();
  // you can use vecSum.elements or simd::Vec<T>::elements
  printf("native style:  elements %zu, bytes %zu\n", vecSum.elements,
         vecSum.bytes);
  // parallel part
  const size_t numVecAdds = input.size / vecSum.elements;
  for (size_t i = 0; i < numVecAdds; i++) {
    vecSum = simd::add(
      vecSum, simd::load<SIMD_WIDTH>(&input.data[i * vecSum.elements]));
  }
  // postamble
  T seqSum = 0;
  for (size_t i = numVecAdds * vecSum.elements; i < input.size; i++) {
    seqSum += input.data[i];
  }
  return simd::hadd(vecSum) + seqSum;
}

// ===========================================================================
// classic style, explicit specification of SIMD_WIDTH
// ===========================================================================

// this is the style used in the WarpingSIMD implementation

// no explicit namespace specification for types and functions
using namespace simd;

// vector element types could be specified e.g. with SIMDFloat

template <size_t SIMD_WIDTH, typename T>
T vectorSumClassic(const Vector<T> &input)
{
  SIMDVec<T, SIMD_WIDTH> vecSum = setzero<T, SIMD_WIDTH>();
  // you can use vecSum.elements or SIMDVec<T>::elements
  printf("native style:  elements %zu, bytes %zu\n", vecSum.elements,
         vecSum.bytes);
  // parallel part
  const size_t numVecAdds = input.size / vecSum.elements;
  for (size_t i = 0; i < numVecAdds; i++) {
    vecSum = add(vecSum, load<SIMD_WIDTH>(&input.data[i * vecSum.elements]));
  }
  // postamble
  T seqSum = 0;
  for (size_t i = numVecAdds * vecSum.elements; i < input.size; i++) {
    seqSum += input.data[i];
  }
  return hadd(vecSum) + seqSum;
}

// ===========================================================================
// main
// ===========================================================================

// for simplicity, we use a fixed alignment here
enum { ALIGN = 64 };

int main(int argc, char *argv[])
{
  printf("NATIVE_SIMD_WIDTH = %d\n", NATIVE_SIMD_WIDTH);

  if (argc != 2) {
    fprintf(stderr, "styleExamples <size>\n");
    exit(-1);
  }
  const size_t size = atol(argv[1]);

  // random input vector
  // 20. Sep 22 (Jonas Keller): use srand and rand instead of srand48 and
  // drand48, since srand48 and drand48 are not available on Windows
  // srand48(time(NULL));
  srand(time(nullptr));
  Vector<float> input(size, ALIGN);
  float sum = 0;
  for (size_t i = 0; i < input.size; i++) {
    // float x = drand48();
    // + 1.0f since drand48 returns values in [0,1)
    const float x = (float) rand() / ((float) RAND_MAX + 1.0f);
    sum += x;
    input.data[i] = x;
  }
  printf("results may differ slightly due to different order of summation\n");

  printf("sequential:    sum = %.20g\n", sum);

  // native style: T is deduced from input
  printf("native style:  sum = %.20g\n", vectorSumNative(input));

  // modern style: SW is passed as SIMD_WIDTH, T is deduced from input
  printf("modern style:  sum = %.20g\n", vectorSumModern<SW>(input));

  // classic style: SW is passed as SIMD_WIDTH, T is deduced from input
  printf("classic style: sum = %.20g\n", vectorSumClassic<SW>(input));

  return 0;
}
