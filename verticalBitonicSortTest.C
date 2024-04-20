// ===========================================================================
//
// verticalBitonicSortTest.C --
// test of verticalBitonicSort function
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

// contributed by Ebba Stina Siebold, modified

#include "tsimd.H"
// #include "TimeMeasurement.H"

// =========== DEFINITIONS ============
// number of bytes in a simd vector
const int SIMD_WIDTH = 64;
// iterations 10000, 1000000
const int ITERATIONS_VALIDATION = 1000;
const int ITERATIONS            = 100000;
//=====================================

namespace SortHelp {
template <class T>
T *alloc_array(int length)
{
  return (T *) malloc(sizeof(T) * length);
}

template <class T>
void random_init_array(T *data, int length)
{
  T range_min = std::numeric_limits<T>::min() / 10;
  T range_max = std::numeric_limits<T>::max() / 10;
  for (int i = 0; i < length; i++) {
    data[i] =
      T(float(rand()) / float(RAND_MAX) * (range_max - range_min) + range_min);
  }
}

template <class T>
void copy_array(T *original, T *copy, int length)
{
  for (int i = 0; i < length; i++) { copy[i] = original[i]; }
}

template <class T>
void print_array(T *data, int length)
{
  std::cout << "[";
  for (int i = 0; i < length - 1; i++) { std::cout << data[i] << ", "; }
  std::cout << data[length - 1] << "]\n";
}

template <class T, int SIMD_WIDTH>
void print_vecs(T *data, int length)
{
  // number of elements in a simd vector
  constexpr int SIMD_ELEMS = SIMD_WIDTH / sizeof(T);
  int i;
  for (i = 0; i < length / SIMD_ELEMS; i++) {
    std::cout << "[";
    for (int j = i * SIMD_ELEMS; j < SIMD_ELEMS * (1 + i) - 1; j++) {
      std::cout << data[j] << ", ";
    }
    std::cout << data[SIMD_ELEMS * (i + 1) - 1] << "]\n";
  }
  if (i * SIMD_ELEMS != length) {
    std::cout << "[";
    for (i = i * SIMD_ELEMS; i < length - 1; i++) {
      std::cout << data[i] << ", ";
    }
    std::cout << data[length - 1] << "]\n";
  }
}

template <class T, int SIMD_WIDTH>
void test_equal_array(T *array1, T *array2, int length)
{
  for (int i = 0; i < length; i++) {
    if (array1[i] != array2[i]) {
      printf("Difference in arrays (n=%d) at index %d.\n", length, i);
      printf("Array 1:\n");
      print_vecs<T, SIMD_WIDTH>(array1, length);
      printf("Array 2:\n");
      print_vecs<T, SIMD_WIDTH>(array2, length);
      return;
    }
  }
}
} // namespace SortHelp

//======= TESTS AND BENCHMARKS ========

template <class T, int LENGTH>
void validate_vertical(int iterations)
{
  for (int i = 0; i < iterations; i++) {
    T *original      = SortHelp::alloc_array<T>(LENGTH);
    T *copy_vertical = SortHelp::alloc_array<T>(LENGTH);
    T *copy_stdsort  = SortHelp::alloc_array<T>(LENGTH);
    SortHelp::random_init_array<T>(original, LENGTH);

    // sort with bitonic sort
    SortHelp::copy_array<T>(original, copy_vertical, LENGTH);
    simd::verticalBitonicSort<LENGTH, simd::SortSlope::ASCENDING, T,
                              SIMD_WIDTH>(copy_vertical);

    // sort with std::sort
    SortHelp::copy_array<T>(original, copy_stdsort, LENGTH);
    std::sort(&copy_stdsort[0], &copy_stdsort[LENGTH]);

    SortHelp::test_equal_array<T, SIMD_WIDTH>(copy_stdsort, copy_vertical,
                                              LENGTH);

    free(original);
    free(copy_vertical);
    free(copy_stdsort);
  }
}

template <class T>
void validate_vertical_all(int iterations)
{
  constexpr int SIMD_ELEMS = SIMD_WIDTH / sizeof(T);
  constexpr int MIN_LENGTH = SIMD_ELEMS * SIMD_ELEMS;
  validate_vertical<T, MIN_LENGTH>(iterations);
  validate_vertical<T, MIN_LENGTH * 2>(iterations);
  validate_vertical<T, MIN_LENGTH * 4>(iterations);
  validate_vertical<T, MIN_LENGTH * 8>(iterations);
}

template <class T, int LENGTH>
void benchmark_vertical(int iterations)
{
  clock_t time;
  double time_total = 0;
  T *data           = SortHelp::alloc_array<T>(LENGTH);
  for (int i = 0; i < iterations; i++) {
    SortHelp::random_init_array<T>(data, LENGTH);

    time = clock();
    simd::verticalBitonicSort<LENGTH, simd::SortSlope::ASCENDING, T,
                              SIMD_WIDTH>(data);
    time = clock() - time;
    time_total += ((double) time) / CLOCKS_PER_SEC;
  }
  free(data);
  printf("vertical bitonic sort: %fs\n", time_total);
}

template <class T, int LENGTH>
void benchmark_stdsort(int iterations)
{
  clock_t time;
  double time_total = 0;
  T *data           = SortHelp::alloc_array<T>(LENGTH);
  for (int i = 0; i < iterations; i++) {
    SortHelp::random_init_array<T>(data, LENGTH);

    time = clock();
    std::sort(&data[0], &data[LENGTH]);
    time = clock() - time;
    time_total += ((double) time) / CLOCKS_PER_SEC;
  }
  free(data);
  printf("std::sort: %fs\n", time_total);
}

template <class T>
void benchmark_vertical_all(int iterations)
{
  constexpr int SIMD_ELEMS = SIMD_WIDTH / sizeof(T);
  constexpr int MIN_LENGTH = SIMD_ELEMS * SIMD_ELEMS;
  benchmark_vertical<T, MIN_LENGTH>(iterations);
  benchmark_vertical<T, MIN_LENGTH * 2>(iterations);
  benchmark_vertical<T, MIN_LENGTH * 4>(iterations);
  benchmark_vertical<T, MIN_LENGTH * 8>(iterations);
}

template <class T>
void benchmark_stdsort_all(int iterations)
{
  constexpr int SIMD_ELEMS = SIMD_WIDTH / sizeof(T);
  constexpr int MIN_LENGTH = SIMD_ELEMS * SIMD_ELEMS;
  benchmark_stdsort<T, MIN_LENGTH>(iterations);
  benchmark_stdsort<T, MIN_LENGTH * 2>(iterations);
  benchmark_stdsort<T, MIN_LENGTH * 4>(iterations);
  benchmark_stdsort<T, MIN_LENGTH * 8>(iterations);
}

//=====================================
int main(int, char **)
{
  // validation of the vertical bitonic sort function
  constexpr int iterations_validation = ITERATIONS_VALIDATION;
  printf("VALIDATION IN PROGRESS...\n");
  printf("SIGNED BYTE...\n");
  validate_vertical_all<int8_t>(iterations_validation);
  printf("BYTE...\n");
  validate_vertical_all<uint8_t>(iterations_validation);
  printf("SHORT...\n");
  validate_vertical_all<int16_t>(iterations_validation);
  printf("WORD...\n");
  validate_vertical_all<uint16_t>(iterations_validation);
  printf("INT...\n");
  validate_vertical_all<int>(iterations_validation);
  printf("FLOAT...\n");
  validate_vertical_all<float>(iterations_validation);
  printf("DOUBLE...\n");
  validate_vertical_all<double>(iterations_validation);
  printf("LONG...\n");
  validate_vertical_all<long>(iterations_validation);
  printf("VALIDATION DONE\n");

  // Benchmarks of all possible types and sorting functions
  constexpr int iterations = ITERATIONS;
  printf("\nBENCHMARKS\n");
  printf("VERTICAL BITONIC SORT\n");
  printf("signed byte\n");
  benchmark_vertical_all<int8_t>(iterations);
  printf("\nbyte\n");
  benchmark_vertical_all<uint8_t>(iterations);
  printf("\nshort\n");
  benchmark_vertical_all<int16_t>(iterations);
  printf("\nword\n");
  benchmark_vertical_all<uint16_t>(iterations);
  printf("\nint\n");
  benchmark_vertical_all<int>(iterations);
  printf("\nfloat\n");
  benchmark_vertical_all<float>(iterations);
  printf("\ndouble\n");
  benchmark_vertical_all<double>(iterations);
  printf("\nlong\n");
  benchmark_vertical_all<long>(iterations);

  printf("\nSTD::SORT\n");
  printf("signed byte\n");
  benchmark_vertical_all<int8_t>(iterations);
  printf("\nbyte\n");
  benchmark_vertical_all<uint8_t>(iterations);
  printf("\nshort\n");
  benchmark_vertical_all<int16_t>(iterations);
  printf("\nword\n");
  benchmark_vertical_all<uint16_t>(iterations);
  printf("\nint\n");
  benchmark_vertical_all<int>(iterations);
  printf("\nfloat\n");
  benchmark_vertical_all<float>(iterations);
  printf("\ndouble\n");
  benchmark_vertical_all<double>(iterations);
  printf("\nlong\n");
  benchmark_vertical_all<long>(iterations);
}
