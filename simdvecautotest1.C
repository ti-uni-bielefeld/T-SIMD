// ===========================================================================
//
// simdvecautotest1.C --
// automatic test of Vec level 1 templates
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

// TODO: auto-test for Vecs functions

// https://stackoverflow.com/questions/15651488/
// how-to-pass-a-template-function-in-a-template-argument-list
//
// ... you cannot pass a function template as a template
// argument. From 14.3.3: A template-argument for a template
// template-parameter shall be the name of a class template or an
// alias template, expressed as id-expression.

#include "SIMDDefs.H"
#include "SIMDVecAll.H"
#include "SIMDVecAutoTest.H"
#include "SIMDVecAutoTestWrapper.H"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace simd;
using namespace auto_test;

// SW = SIMD width = number of bytes in a single SIMD vector
#define SW NATIVE_SIMD_WIDTH

enum { REPEATS1 = 100000 };

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char *argv[])
{
  srand(time(nullptr));
  // default values
  int repeats1        = REPEATS1;
  std::string pattern = "";
  // overwrite default values from command line
  if (argc >= 2) pattern = argv[1];
  if (argc == 3) repeats1 = atoi(argv[2]);
  printf("pattern \"%s\", repeats1 = %d\n", pattern.c_str(), repeats1);

  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<Byte>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<SignedByte>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<Word>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<Short>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<Int>());
  // for (int i = 0; i<100000; i++) printf("%g\n", getRandom<Float>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandomRanges<Byte>());
  // for (int i = 0; i<100000; i++) printf("%d\n",
  // getRandomRanges<SignedByte>()); for (int i = 0; i<100000; i++)
  // printf("%d\n", getRandomRanges<Word>()); for (int i = 0; i<100000; i++)
  // printf("%d\n", getRandomRanges<Short>()); for (int i = 0; i<100000;
  // i++) printf("%d\n", getRandomRanges<Int>()); for (int i = 0; i<100000;
  // i++) printf("%g\n", getRandomRanges<Float>()); gnuplot> plot "test.dat"
  // using ($1):(1) smooth freq with boxes / or use test.gp exit(0);

  // ------------------------------------------------------------------------
  // level 1
  // ------------------------------------------------------------------------

  // TODO: multi-vector store and load
  // TODO: load*_store*
  ArrayArrayVoidTemplateType<SignedByte, SignedByte, SW, Convert>::test(
    repeats1, pattern);
  ArrayArrayVoidTemplateType<Short, SignedByte, SW, Convert>::test(repeats1,
                                                                   pattern);
  ArrayArrayVoidTemplateType<Int, SignedByte, SW, Convert>::test(repeats1,
                                                                 pattern);
  ArrayArrayVoidTemplateType<Float, SignedByte, SW, Convert>::test(repeats1,
                                                                   pattern);
  ArrayArrayVoidTemplateType<Byte, Byte, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Short, Byte, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Word, Byte, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Int, Byte, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Float, Byte, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SignedByte, Short, SW, Convert>::test(repeats1,
                                                                   pattern);
  ArrayArrayVoidTemplateType<Byte, Short, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Short, Short, SW, Convert>::test(repeats1,
                                                              pattern);
  ArrayArrayVoidTemplateType<Int, Short, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Float, Short, SW, Convert>::test(repeats1,
                                                              pattern);
  ArrayArrayVoidTemplateType<Word, Word, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Int, Word, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Float, Word, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SignedByte, Int, SW, Convert>::test(repeats1,
                                                                 pattern);
  ArrayArrayVoidTemplateType<Byte, Int, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Short, Int, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Word, Int, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Int, Int, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Float, Int, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SignedByte, Float, SW, Convert>::test(repeats1,
                                                                   pattern);
  ArrayArrayVoidTemplateType<Byte, Float, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Short, Float, SW, Convert>::test(repeats1,
                                                              pattern);
  ArrayArrayVoidTemplateType<Word, Float, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Int, Float, SW, Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<Float, Float, SW, Convert>::test(repeats1,
                                                              pattern);
  // NOTE: errors occur in fdivmul on NEON since div is an approximation
  TestAllTT<ArrayArrayDoubleArrayVoidTemplateType, SW, Fdivmul>::test(repeats1,
                                                                      pattern);
  TestAllTT<ArrayDoubleArrayVoidTemplateType, SW, Fmul>::test(repeats1,
                                                              pattern);
  TestAllTT<ArrayDoubleDoubleArrayVoidTemplateType, SW, Faddmul>::test(repeats1,
                                                                       pattern);
  // NOTE: errors occur in fmuladd if fused-multiply-add is available and used
  // by compiler (-mfma) NOTE: check flags with g++ -mavx2 (or -march=native) -Q
  // --help=target
  TestAllTT<ArrayDoubleDoubleArrayVoidTemplateType, SW, Fmuladd>::test(repeats1,
                                                                       pattern);
  // NOTE: errors occur in fwaddmul if fused-multiply-add is available and used
  // by compiler (-mfma) NOTE: check flags with g++ -mavx2 (or -march=native) -Q
  // --help=target
  TestAllTT<ArrayArrayDoubleDoubleArrayVoidTemplateType, SW, Fwaddmul>::test(
    repeats1, pattern);
  TestAll<Array, SW, HaddMulti>::test(repeats1, pattern);
  TestAll<Array, SW, HaddsMulti>::test(repeats1, pattern);
  TestAll<UnaryScalarT, SW, HaddSingle>::test(repeats1, pattern);
  TestAll<UnaryScalarT, SW, HaddsSingle>::test(repeats1, pattern);
  TestAll<UnaryScalarT, SW, HminSingle>::test(repeats1, pattern);
  TestAll<UnaryScalarT, SW, HmaxSingle>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_a>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_e>::test(repeats1, pattern);
  // contributed by Adam Marschall
  TestAll<ArrayArrayVoid, SW, Transpose_b>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_d>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_f>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_g>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_h>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_c>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid, SW, Transpose_i>::test(repeats1, pattern);
  // 15. Oct 22 (Jonas Keller): added test for _Swizzle
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Swizzle>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Swizzle2_a>::test(repeats1,
                                                                pattern);
  // 07. Feb 23 (Jonas Keller): added test for swizzle2_b
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Swizzle2_b>::test(repeats1,
                                                                pattern);
  // 15. Oct 22 (Jonas Keller): added test for swizzle2_c
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Swizzle2_c>::test(repeats1,
                                                                pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Unswizzle_a>::test(repeats1,
                                                                 pattern);
  // 15. Oct 22 (Jonas Keller): added test for unswizzle_b
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Unswizzle_b>::test(repeats1,
                                                                 pattern);
  // 04. Dec 22 (Jonas Keller): The bitonic sorts flush denormals to zero on
  // ARMv7, since the min and max operations flush denormals to zero on ARMv7.
  // This results in autotest errors on ARMv7.
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, BitonicSort>::test(repeats1,
                                                                 pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, BitonicSort2>::test(repeats1,
                                                                  pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, BitonicSortSortedPairs>::test(
    repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, BitonicSortSortedPairs2>::test(
    repeats1, pattern);
  TestAll<Binary, SW, Avgru>::test(repeats1, pattern);
  TestAll<Binary, SW, Avgrd>::test(repeats1, pattern);
  TestAll<Nullary, SW, Setones>::test(1, pattern);
  TestAll<Nullary, SW, Setmin>::test(1, pattern);
  TestAll<Nullary, SW, Setmax>::test(1, pattern);
  TestSigned<Nullary, SW, Setnegunity>::test(1, pattern);
  TestAll<Nullary, SW, Setunity>::test(1, pattern);
  ArrayTemplateType<SignedByte, SignedByte, SW, PacksMulti>::test(repeats1,
                                                                  pattern);
  ArrayTemplateType<SignedByte, Short, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Byte, Short, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Short, Short, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SignedByte, Int, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Byte, Int, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Short, Int, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Word, Int, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Int, Int, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Float, Int, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SignedByte, Float, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Byte, Float, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Short, Float, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Word, Float, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Int, Float, SW, PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<Float, Float, SW, PacksMulti>::test(repeats1, pattern);
  TestAll<Unary, SW, Div2r0>::test(repeats1, pattern);
  TestAll<Unary, SW, Div2rd>::test(repeats1, pattern);
  Binary<Float, SW, Sign>::test(repeats1, pattern);
  TestAll<Binary, SW, AbsDiff>::test(repeats1, pattern);
  // TODO: operators
  // TODO: Vecs
  printf("simdvecautotest1 complete\n");
  return 0;
}
