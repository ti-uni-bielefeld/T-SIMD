// ===========================================================================
// 
// simdvecautotest0.C --
// automatic test of SIMDVec level 0 templates
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

#include <time.h>
#include "SIMDVecAutoTest.H"

using namespace ns_simd;

// SW = SIMD width = number of bytes in a single SIMD vector

#ifdef _SIMD_VEC_64_AVAIL_
# define SW 64
# define SA 64
#else
# ifdef _SIMD_VEC_32_AVAIL_
#  define SW 32
#  define SA 32
# else
#  define SW 16
#  define SA 16
# endif
#endif

#define REPEATS1 100000

// ===========================================================================
// main
// ===========================================================================

int
main(int argc, char *argv[])
{
  srand(time(NULL));
  // default values
  int repeats1 = REPEATS1;
  std::string pattern = "";
  // overwrite default values from command line
  if (argc >= 2) pattern = argv[1];
  if (argc == 3) repeats1 = atoi(argv[2]);
  printf("pattern \"%s\", repeats1 = %d\n", pattern.c_str(), repeats1);

  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<SIMDByte>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<SIMDSignedByte>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<SIMDWord>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<SIMDShort>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandom<SIMDInt>());
  // for (int i = 0; i<100000; i++) printf("%g\n", getRandom<SIMDFloat>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandomRanges<SIMDByte>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandomRanges<SIMDSignedByte>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandomRanges<SIMDWord>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandomRanges<SIMDShort>());
  // for (int i = 0; i<100000; i++) printf("%d\n", getRandomRanges<SIMDInt>());
  // for (int i = 0; i<100000; i++) printf("%g\n", getRandomRanges<SIMDFloat>());
  // gnuplot> plot "test.dat" using ($1):(1) smooth freq with boxes / or use test.gp
  // exit(0);

  // ------------------------------------------------------------------------
  // level 0
  // ------------------------------------------------------------------------

  TestAllTT<UnaryTemplateType,SW,Reinterpret>::test(repeats1, pattern);
  TestAll<Nullary,SW,Setzero>::test(1, pattern);
  TestAll<Scalar,SW,Set1>::test(repeats1, pattern);
  // NOTE: load and store are used in SIMDSerialVec, test would probably be flawed
  // NOTE: not tested: load, loadu, store, storeu, streamstore
  TestAllTI<UnaryTemplateIntScalarMinToMax,SW,Extract>::test(repeats1, pattern);
  TestAll<Binary,SW,Add>::test(repeats1, pattern);
  TestAll<Binary,SW,Adds>::test(repeats1, pattern);
  TestAll<Binary,SW,Sub>::test(repeats1, pattern);
  TestAll<Binary,SW,Subs>::test(repeats1, pattern);
  TestSigned<Unary,SW,Neg>::test(repeats1, pattern);
  TestFloat<Binary,SW,Mul>::test(repeats1, pattern);
  TestFloat<Unary,SW,Ceil>::test(repeats1, pattern);
  TestFloat<Unary,SW,Floor>::test(repeats1, pattern);
  TestFloat<Unary,SW,Round>::test(repeats1, pattern);
  TestFloat<Unary,SW,Truncate>::test(repeats1, pattern);
  // TODO: div and sqrt are approximations on NEON, figure out max relative error
  TestFloat<Binary,SW,Div>::test(repeats1, pattern);
  TestFloat<Unary,SW,Sqrt>::test(repeats1, pattern);
  // 02. Oct 22 (Jonas Keller): added consideration of relative error for rcp and rsqrt
#if SIMDVEC_INTEL_ENABLE
#if (defined(__SSE__) || defined(__AVX__)) && !defined(__AVX512F__)
  // for sse and avx the relative error is 1.5*2^-12
  Unary<SIMDFloat,SW,Rcp,CmpRelError<15, -1, -12> >::test(repeats1, pattern);
  Unary<SIMDFloat,SW,Rsqrt,CmpRelError<15, -1, -12> >::test(repeats1, pattern);
#elif defined(__AVX512F__)
  // for avx512 the relative error is 2^-14
  Unary<SIMDFloat,SW,Rcp,CmpRelError<1, 0, -14> >::test(repeats1, pattern);
  Unary<SIMDFloat,SW,Rsqrt,CmpRelError<1, 0, -14> >::test(repeats1, pattern);
#endif
#else
  // TODO: rcp and rsqrt are approximations on NEON, figure out max relative error
  Unary<SIMDFloat,SW,Rcp>::test(repeats1, pattern);
  Unary<SIMDFloat,SW,Rsqrt>::test(repeats1, pattern);
#endif
  TestAll<Binary,SW,Min>::test(repeats1, pattern);
  TestAll<Binary,SW,Max>::test(repeats1, pattern);
  TestSignedInt<Unary,SW,Abs>::test(repeats1, pattern);
  TestAllTII<BinaryTemplateIntIntMinToMax,SW,Unpack_>::test(repeats1, pattern);
  TestAllTI<BinaryBinaryTemplateIntMinToMax,SW,Zip_>::test(repeats1, pattern);
  TestAllTI<BinaryBinaryTemplateIntMinToMax,SW,Unzip>::test(repeats1, pattern);
  BinaryTemplateType<SIMDSignedByte,SIMDShort,SW,Packs>::test(repeats1, pattern);
  BinaryTemplateType<SIMDShort,SIMDInt,SW,Packs>::test(repeats1, pattern);
  BinaryTemplateType<SIMDShort,SIMDFloat,SW,Packs>::test(repeats1, pattern);
  BinaryTemplateType<SIMDByte,SIMDShort,SW,Packs>::test(repeats1, pattern);
  BinaryTemplateType<SIMDWord,SIMDInt,SW,Packs>::test(repeats1, pattern);
  BinaryTemplateType<SIMDWord,SIMDFloat,SW,Packs>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDSignedByte,SIMDSignedByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDShort,SIMDSignedByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDInt,SIMDSignedByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDFloat,SIMDSignedByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDByte,SIMDByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDShort,SIMDByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDWord,SIMDByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDInt,SIMDByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDFloat,SIMDByte,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDShort,SIMDShort,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDInt,SIMDShort,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDFloat,SIMDShort,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDWord,SIMDWord,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDInt,SIMDWord,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDFloat,SIMDWord,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDInt,SIMDInt,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDFloat,SIMDInt,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDInt,SIMDFloat,SW,Extend>::test(repeats1, pattern);
  UnaryArrayTemplateType<SIMDFloat,SIMDFloat,SW,Extend>::test(repeats1, pattern);
  UnaryTemplateIntMinToMax<SIMDWord,SW,Srai>::test(repeats1, pattern);
  UnaryTemplateIntMinToMax<SIMDShort,SW,Srai>::test(repeats1, pattern);
  UnaryTemplateIntMinToMax<SIMDInt,SW,Srai>::test(repeats1, pattern);
  TestIntTI<UnaryTemplateIntMinToMax,SW,Srli>::test(repeats1, pattern);
  TestIntTI<UnaryTemplateIntMinToMax,SW,Slli>::test(repeats1, pattern);
  UnaryTemplateType<SIMDInt,SIMDFloat,SW,Cvts>::test(repeats1, pattern);
  UnaryTemplateType<SIMDFloat,SIMDInt,SW,Cvts>::test(repeats1, pattern);
  TestAll<Binary,SW,Hadd>::test(repeats1, pattern);
  TestAll<Binary,SW,Hadds>::test(repeats1, pattern);
  TestAll<Binary,SW,Hsub>::test(repeats1, pattern);
  TestAll<Binary,SW,Hsubs>::test(repeats1, pattern);
  TestAllTI<UnaryTemplateIntMinToMax,SW,Srle>::test(repeats1, pattern);
  TestAllTI<UnaryTemplateIntMinToMax,SW,Slle>::test(repeats1, pattern);
  TestAll<UnaryScalarT,SW,Elem0>::test(repeats1, pattern);
  TestAllTI<BinaryTemplateIntMinToMax,SW,Alignre>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,_Swizzle>::test(repeats1, pattern);
  TernaryConditionTemplateType<SIMDByte,SIMDByte,SW,Ifelse>::test(repeats1, pattern);
  TernaryConditionTemplateType<SIMDSignedByte,SIMDSignedByte,SW,Ifelse>::test(repeats1, pattern);
  TernaryConditionTemplateType<SIMDWord,SIMDWord,SW,Ifelse>::test(repeats1, pattern);
  TernaryConditionTemplateType<SIMDShort,SIMDShort,SW,Ifelse>::test(repeats1, pattern);
  TernaryConditionTemplateType<SIMDInt,SIMDInt,SW,Ifelse>::test(repeats1, pattern);
  TernaryConditionTemplateType<SIMDFloat,SIMDFloat,SW,Ifelse>::test(repeats1, pattern);
  TestAll<Binary,SW,Cmplt>::test(repeats1, pattern);
  TestAll<Binary,SW,Cmple>::test(repeats1, pattern);
  TestAll<Binary,SW,Cmpeq>::test(repeats1, pattern);
  TestAll<Binary,SW,Cmpge>::test(repeats1, pattern);
  TestAll<Binary,SW,Cmpgt>::test(repeats1, pattern);
  TestAll<Binary,SW,Cmpneq>::test(repeats1, pattern);
  TestAll<Binary,SW,And>::test(repeats1, pattern);
  TestAll<Binary,SW,Or>::test(repeats1, pattern);
  TestAll<Binary,SW,Andnot>::test(repeats1, pattern);
  TestAll<Binary,SW,Xor>::test(repeats1, pattern);
  TestAll<Unary,SW,Not>::test(repeats1, pattern);
  TestAll<Binary,SW,Avg>::test(repeats1, pattern);
  TestAll<UnaryScalarInt,SW,Test_all_zeros>::test(repeats1, pattern);
  TestAll<UnaryScalarInt,SW,Test_all_ones>::test(repeats1, pattern);
  TestAll<Unary,SW,Reverse>::test(repeats1, pattern);
  // 27. Aug 22 (Jonas Keller): added msb2int
  TestAll<UnaryScalarUInt64,SW,Msb2int>::test(repeats1, pattern);
  // 06. Oct 22 (Jonas Keller): added int2msb
  TestAll<ScalarUInt64,SW,Int2msb>::test(repeats1, pattern);
  // 09. Oct 22 (Jonas Keller): added int2bits
  TestAll<ScalarUInt64,SW,Int2bits>::test(repeats1, pattern);
  printf("simdvecautotest0 complete\n");
  return 0;
}
