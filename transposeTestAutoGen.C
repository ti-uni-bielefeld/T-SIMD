// ===========================================================================
// 
// transposeTestAutoGen.C --
// test of auto-generated transpose functions
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

#define MAX_SIMD_WIDTH 64

#include <stdlib.h>
#include "tsimd.H"
#include "TimeMeasurement.H"

// NOTE: byte may not be enough to fill each element with a unique value
// for large matrices

//#define TYPE simd::Float
//#define TFORMAT "%3g "

//#define TYPE simd::Byte
//#define TFORMAT "%3d "

#define TYPE simd::Double
#define TFORMAT "%3g "

#define barrier() __asm__ __volatile__("": : :"memory")

// ===========================================================================

template <typename T>
void
transposeCheck(simd::Vec<T> *rows1, simd::Vec<T> *rows2)
{
  constexpr int n = simd::Vec<T>::elements;
  T buf1[n * n], buf2[n * n];
  for (int i = 0; i < n; i++) {
    storeu(buf1 + i * n, rows1[i]);
    storeu(buf2 + i * n, rows2[i]);
  }
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (buf1[i * n + j] != buf2[j * n + i]) {
        fprintf(stderr, "!!!!!!!!!!! transpose error !!!!!!!!!!!!!!\n");
        return;
      }
}

// ===========================================================================

// also odd number of repeats
#define TEST_TRANSPOSE(TFUNCNAME)                                \
  puts("========== " #TFUNCNAME "  ==========");                 \
  t0 = simd::getTimeSpec();                                      \
  for (long r = 0; r < (repeats | 1); r++) {                     \
    barrier();                                                   \
    TFUNCNAME(inRows, outRows);                                  \
    barrier();                                                   \
  }                                                              \
  dt = simd::timeSpecDiffUsec(simd::getTimeSpec(), t0);          \
  printf("TIME dt = %10.0f for " #TFUNCNAME "\n", dt);           \
  fflush(stdout);                                                \
  for (unsigned int i = 0; i < elems; i++) {                     \
    printf("outRows[%2d] = ", i);                                \
    print(TFORMAT, outRows[i]);                                  \
    printf("\n");                                                \
  }                                                              \
  puts("");                                                      \
  transposeCheck(inRows, outRows);

// test of single-arg versions
// odd number of repeats
#define TEST_TRANSPOSE_1ARG(TFUNCNAME)                           \
  puts("========== " #TFUNCNAME "  ==========");                 \
  for (int i = 0; i < elems; i++)                                \
    rows[i] = inRows[i];                                         \
  t0 = simd::getTimeSpec();                                      \
  for (long r = 0; r < (repeats | 1); r++) {                     \
    barrier();                                                   \
    TFUNCNAME(rows);                                             \
    barrier();                                                   \
  }                                                              \
  dt = simd::timeSpecDiffUsec(simd::getTimeSpec(), t0);          \
  printf("TIME dt = %10.0f for " #TFUNCNAME " (1 arg)\n", dt);   \
  fflush(stdout);                                                \
  for (unsigned int i = 0; i < elems; i++) {                     \
    printf("rows[%2d] = ", i);                                   \
    print(TFORMAT, rows[i]);                                     \
    printf("\n");                                                \
  }                                                              \
  puts("");                                                      \
  transposeCheck(inRows, rows);

// test of single-arg versions, emulated with two-arg versions
// odd number of repeats
// use only for auto-generated inplc functions
#define TEST_TRANSPOSE_1ARG_EMU(TFUNCNAME)                              \
  puts("========== " #TFUNCNAME "  ==========");                        \
  for (int i = 0; i < elems; i++)                                       \
    rows[i] = inRows[i];                                                \
  t0 = simd::getTimeSpec();                                             \
  for (long r = 0; r < (repeats | 1); r++) {                            \
    barrier();                                                          \
    TFUNCNAME(rows, rows);                                              \
    barrier();                                                          \
  }                                                                     \
  dt = simd::timeSpecDiffUsec(simd::getTimeSpec(), t0);                 \
  printf("TIME dt = %10.0f for " #TFUNCNAME " (1 arg, emu)\n", dt);     \
  fflush(stdout);                                                       \
  for (unsigned int i = 0; i < elems; i++) {                            \
    printf("rows[%2d] = ", i);                                          \
    print(TFORMAT, rows[i]);                                            \
    printf("\n");                                                       \
  }                                                                     \
  puts("");                                                             \
  transposeCheck(inRows, rows);

int
main(int argc, char *argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: transposeTest <repeats>\n");
    exit(-1);
  }
  long repeats = atol(argv[1]);
  printf("repeats = %ld\n", repeats);
  
  constexpr int elems = simd::Vec<TYPE>::elements; 
  simd::Vec<TYPE> inRows[elems], outRows[elems], rows[elems];
  TYPE buf[elems * elems];
  struct timespec t0;
  double dt;
  
  for (unsigned int i = 0; i < elems * elems; i++) buf[i] = TYPE(i);
  for (unsigned int i = 0; i < elems; i++) {
    inRows[i] = simd::loadu(buf + i * elems);
    printf("inRows[%2d] = ", i);
    print(TFORMAT, inRows[i]);
    printf("\n");
  }
  puts("");

  TEST_TRANSPOSE(simd::internal::ext::transpose_a);
  TEST_TRANSPOSE(simd::internal::ext::transpose_b);
  TEST_TRANSPOSE(simd::internal::ext::transpose_c);
  TEST_TRANSPOSE(simd::internal::ext::transpose_d);
  TEST_TRANSPOSE(simd::internal::ext::transpose_e);
  TEST_TRANSPOSE(simd::internal::ext::transpose_f);
  TEST_TRANSPOSE(simd::internal::ext::transpose_g);
  TEST_TRANSPOSE(simd::internal::ext::transpose_h);
  TEST_TRANSPOSE(simd::internal::ext::transpose_i);
  
  TEST_TRANSPOSE(simd::internal::ext::transpose1inplc);
  TEST_TRANSPOSE(simd::internal::ext::transpose1inplcLane);
  TEST_TRANSPOSE(simd::internal::ext::transpose2inplc);
  // TEST_TRANSPOSE(simd::internal::ext::transpose2inplcLaneA);
  TEST_TRANSPOSE(simd::internal::ext::transpose2inplcLane);

  TEST_TRANSPOSE_1ARG(simd::internal::ext::transpose1inplc);
  TEST_TRANSPOSE_1ARG(simd::internal::ext::transpose1inplcLane);
  TEST_TRANSPOSE_1ARG(simd::internal::ext::transpose2inplc);
  // TEST_TRANSPOSE(simd::internal::ext::transpose2inplcLaneA);
  TEST_TRANSPOSE_1ARG(simd::internal::ext::transpose2inplcLane);

  TEST_TRANSPOSE_1ARG_EMU(simd::internal::ext::transpose1inplc);
  TEST_TRANSPOSE_1ARG_EMU(simd::internal::ext::transpose1inplcLane);
  TEST_TRANSPOSE_1ARG_EMU(simd::internal::ext::transpose2inplc);
  // TEST_TRANSPOSE(simd::internal::ext::transpose2inplcLaneA);
  TEST_TRANSPOSE_1ARG_EMU(simd::internal::ext::transpose2inplcLane);

  return 0;
}
