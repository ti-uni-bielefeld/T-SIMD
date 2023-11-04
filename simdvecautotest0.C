// ===========================================================================
//
// simdvecautotest0.C --
// automatic test of Vec level 0 templates
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

// https://stackoverflow.com/questions/15651488/
// how-to-pass-a-template-function-in-a-template-argument-list
//
// ... you cannot pass a function template as a template
// argument. From 14.3.3: A template-argument for a template
// template-parameter shall be the name of a class template or an
// alias template, expressed as id-expression.

// 13. May 23 (Jonas Keller): added tests for Double versions of functions

#include "SIMDVecAll.H"
#include "SIMDVecAutoTest.H"
#include "SIMDVecAutoTestWrapper.H"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace simd;
using namespace auto_test;

// SW = SIMD width = number of bytes in a single SIMD vector
#define SW NATIVE_SIMD_WIDTH

static constexpr size_t REPEATS1 = 100000;

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char *argv[])
{
  srand(static_cast<unsigned int>(time(nullptr)));

  const std::string pattern = argc >= 2 ? argv[1] : "";
  const size_t repeats1 =
    argc >= 3 ? static_cast<size_t>(atoi(argv[2])) : REPEATS1;

  printf("pattern \"%s\", repeats1 = %zu\n", pattern.c_str(), repeats1);

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
  // level 0
  // ------------------------------------------------------------------------

  TestAllTT<UnaryTemplateType, SW, Reinterpret>::test(repeats1, pattern);
  TestAll<Nullary, SW, Setzero>::test(repeats1, pattern);
  TestAll<Scalar, SW, Set1>::test(repeats1, pattern);
  TestAllTI<UnaryTemplateIntScalarMinToMax, SW, Extract>::test(repeats1,
                                                               pattern);
  TestAll<Binary, SW, Add>::test(repeats1, pattern);
  TestAll<Binary, SW, Adds>::test(repeats1, pattern);
  TestAll<Binary, SW, Sub>::test(repeats1, pattern);
  TestAll<Binary, SW, Subs>::test(repeats1, pattern);
  TestSigned<Unary, SW, Neg>::test(repeats1, pattern);
  TestFloat<Binary, SW, Mul>::test(repeats1, pattern);
  TestAll<Unary, SW, Ceil>::test(repeats1, pattern);
  TestAll<Unary, SW, Floor>::test(repeats1, pattern);
  TestAll<Unary, SW, Round>::test(repeats1, pattern);
  TestAll<Unary, SW, Truncate>::test(repeats1, pattern);
  // TODO: div and sqrt are approximations on NEON, figure out max relative
  // error
  TestFloat<Binary, SW, Div>::test(repeats1, pattern);
  TestFloat<Unary, SW, Sqrt>::test(repeats1, pattern);
  // 02. Oct 22 (Jonas Keller): added consideration of relative error for rcp
  // and rsqrt
#if SIMDVEC_INTEL_ENABLE
#if (defined(__SSE__) || defined(__AVX__)) && !defined(__AVX512F__)
  // for sse and avx the relative error is 1.5*2^-12
  TestFloat<Unary, SW, Rcp, CmpRelError<15, -1, -12>>::test(repeats1, pattern);
  TestFloat<Unary, SW, Rsqrt, CmpRelError<15, -1, -12>>::test(repeats1,
                                                              pattern);
#elif defined(__AVX512F__)
  // for avx512 the relative error is 2^-14
  TestFloat<Unary, SW, Rcp, CmpRelError<1, 0, -14>>::test(repeats1, pattern);
  TestFloat<Unary, SW, Rsqrt, CmpRelError<1, 0, -14>>::test(repeats1, pattern);
#endif
#else
  // TODO: rcp and rsqrt are approximations on NEON, figure out max relative
  // error
  TestFloat<Unary, SW, Rcp>::test(repeats1, pattern);
  TestFloat<Unary, SW, Rsqrt>::test(repeats1, pattern);
#endif
  TestAll<Binary, SW, Min>::test(repeats1, pattern);
  TestAll<Binary, SW, Max>::test(repeats1, pattern);
  TestAll<Unary, SW, Abs>::test(repeats1, pattern);
  TestAllTII<BinaryTemplateIntIntMinToMax, SW, Unpack>::test(repeats1, pattern);
  // 10. Oct 22 (Jonas Keller): added test for Unpack16
  TestAllTII<BinaryTemplateIntIntMinToMax, SW, Unpack16>::test(repeats1,
                                                               pattern);
  // 10. Oct 22 (Jonas Keller): added test for ExtractLane
  TestAllTI<UnaryTemplateInt16MinToMax, SW, ExtractLane>::test(repeats1,
                                                               pattern);
  TestAllTI<BinaryBinaryTemplateIntMinToMax, SW, Zip>::test(repeats1, pattern);
  // 10. Oct 22 (Jonas Keller): added test for Zip16
  TestAllTI<BinaryBinaryTemplateIntMinToMax, SW, Zip16>::test(repeats1,
                                                              pattern);
  TestAllTI<BinaryBinaryTemplateIntMinToMax, SW, Unzip>::test(repeats1,
                                                              pattern);
  TestAllTTOneSmaller<BinaryTemplateType, SW, Packs>::test(repeats1, pattern);
  TestAllTTSameSizeOrLarger<UnaryArrayTemplateType, SW, Extend>::test(repeats1,
                                                                      pattern);
  TestIntTI<UnaryTemplateIntMinToMax, SW, Srai>::test(repeats1, pattern);
  TestIntTI<UnaryTemplateIntMinToMax, SW, Srli>::test(repeats1, pattern);
  TestIntTI<UnaryTemplateIntMinToMax, SW, Slli>::test(repeats1, pattern);
  // 19. Dec 22 (Jonas Keller): added test for Sra, Srl and Sll
  TestInt<UnaryUInt8, SW, Sra>::test(repeats1, pattern);
  TestInt<UnaryUInt8, SW, Srl>::test(repeats1, pattern);
  TestInt<UnaryUInt8, SW, Sll>::test(repeats1, pattern);
  UnaryTemplateType<Int, Float, SW, Cvts>::test(repeats1, pattern);
  UnaryTemplateType<Float, Int, SW, Cvts>::test(repeats1, pattern);
#ifdef SIMD_64BIT_TYPES
  UnaryTemplateType<Long, Double, SW, Cvts>::test(repeats1, pattern);
  UnaryTemplateType<Double, Long, SW, Cvts>::test(repeats1, pattern);
#endif
  TestAll<Binary, SW, Hadd>::test(repeats1, pattern);
  TestAll<Binary, SW, Hadds>::test(repeats1, pattern);
  TestAll<Binary, SW, Hsub>::test(repeats1, pattern);
  TestAll<Binary, SW, Hsubs>::test(repeats1, pattern);
  TestAllTI<UnaryTemplateIntMinToMax, SW, Srle>::test(repeats1, pattern);
  TestAllTI<UnaryTemplateIntMinToMax, SW, Slle>::test(repeats1, pattern);
  TestAllTI<BinaryTemplateIntMinToMax, SW, Alignre>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax, SW, Swizzle>::test(repeats1, pattern);
  TestAllTTSameSize<TernaryConditionTemplateType, SW, Ifelse>::test(repeats1,
                                                                    pattern);
  TestAll<Binary, SW, Cmplt>::test(repeats1, pattern);
  TestAll<Binary, SW, Cmple>::test(repeats1, pattern);
  TestAll<Binary, SW, Cmpeq>::test(repeats1, pattern);
  TestAll<Binary, SW, Cmpge>::test(repeats1, pattern);
  TestAll<Binary, SW, Cmpgt>::test(repeats1, pattern);
  TestAll<Binary, SW, Cmpneq>::test(repeats1, pattern);
  TestAll<Binary, SW, And>::test(repeats1, pattern);
  TestAll<Binary, SW, Or>::test(repeats1, pattern);
  TestAll<Binary, SW, Andnot>::test(repeats1, pattern);
  TestAll<Binary, SW, Xor>::test(repeats1, pattern);
  TestAll<Unary, SW, Not>::test(repeats1, pattern);
  TestAll<Binary, SW, Avg>::test(repeats1, pattern);
  TestAll<UnaryScalarBool, SW, Test_all_zeros>::test(repeats1, pattern);
  TestAll<UnaryScalarBool, SW, Test_all_ones>::test(repeats1, pattern);
  TestAll<Unary, SW, Reverse>::test(repeats1, pattern);
  // 27. Aug 22 (Jonas Keller): added msb2int
  TestAll<UnaryScalarUInt64, SW, Msb2int>::test(repeats1, pattern);
  // 06. Oct 22 (Jonas Keller): added int2msb
  TestAll<ScalarUInt64, SW, Int2msb>::test(repeats1, pattern);
  // 09. Oct 22 (Jonas Keller): added int2bits
  TestAll<ScalarUInt64, SW, Int2bits>::test(repeats1, pattern);
  // 07. Feb 23 (Jonas Keller): added test for iota
  TestAll<Nullary, SW, Iota>::test(repeats1, pattern);
  printf("simdvecautotest0 complete\n");
  return 0;
}
