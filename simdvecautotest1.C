// ===========================================================================
// 
// simdvecautotest1.C --
// automatic test of SIMDVec level 1 templates
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

// TODO: auto-test for SIMDVecs functions

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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

using namespace ns_simd;
using namespace auto_test;

// SW = SIMD width = number of bytes in a single SIMD vector
#define SW NATIVE_SIMD_WIDTH

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
  // level 1
  // ------------------------------------------------------------------------

  // TODO: multi-vector store and load
  // TODO: load*_store*
  ArrayArrayVoidTemplateType<SIMDSignedByte,SIMDSignedByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDShort,SIMDSignedByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDInt,SIMDSignedByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDFloat,SIMDSignedByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDByte,SIMDByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDShort,SIMDByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDWord,SIMDByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDInt,SIMDByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDFloat,SIMDByte,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDSignedByte,SIMDShort,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDByte,SIMDShort,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDShort,SIMDShort,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDInt,SIMDShort,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDFloat,SIMDShort,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDWord,SIMDWord,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDInt,SIMDWord,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDFloat,SIMDWord,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDSignedByte,SIMDInt,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDByte,SIMDInt,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDShort,SIMDInt,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDWord,SIMDInt,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDInt,SIMDInt,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDFloat,SIMDInt,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDSignedByte,SIMDFloat,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDByte,SIMDFloat,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDShort,SIMDFloat,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDWord,SIMDFloat,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDInt,SIMDFloat,SW,Convert>::test(repeats1, pattern);
  ArrayArrayVoidTemplateType<SIMDFloat,SIMDFloat,SW,Convert>::test(repeats1, pattern);
  // NOTE: errors occur in fdivmul on NEON since div is an approximation
  TestAllTT<ArrayArrayDoubleArrayVoidTemplateType,SW,Fdivmul>::test(repeats1, pattern);
  TestAllTT<ArrayDoubleArrayVoidTemplateType,SW,Fmul>::test(repeats1, pattern);
  TestAllTT<ArrayDoubleDoubleArrayVoidTemplateType,SW,Faddmul>::test(repeats1, pattern);
  // NOTE: errors occur in fmuladd if fused-multiply-add is available and used by compiler (-mfma)
  // NOTE: check flags with g++ -mavx2 (or -march=native) -Q --help=target
  TestAllTT<ArrayDoubleDoubleArrayVoidTemplateType,SW,Fmuladd>::test(repeats1, pattern);
  // NOTE: errors occur in fwaddmul if fused-multiply-add is available and used by compiler (-mfma)
  // NOTE: check flags with g++ -mavx2 (or -march=native) -Q --help=target
  TestAllTT<ArrayArrayDoubleDoubleArrayVoidTemplateType,SW,Fwaddmul>::test(repeats1, pattern);
  Array<SIMDWord,SW,HaddMulti>::test(repeats1, pattern);
  Array<SIMDShort,SW,HaddMulti>::test(repeats1, pattern);
  Array<SIMDInt,SW,HaddMulti>::test(repeats1, pattern);
  Array<SIMDFloat,SW,HaddMulti>::test(repeats1, pattern);
  Array<SIMDShort,SW,HaddsMulti>::test(repeats1, pattern);
  Array<SIMDInt,SW,HaddsMulti>::test(repeats1, pattern);
  Array<SIMDFloat,SW,HaddsMulti>::test(repeats1, pattern);
  Array<SIMDWord,SW,HsubMulti>::test(repeats1, pattern);
  Array<SIMDShort,SW,HsubMulti>::test(repeats1, pattern);
  Array<SIMDInt,SW,HsubMulti>::test(repeats1, pattern);
  Array<SIMDFloat,SW,HsubMulti>::test(repeats1, pattern);
  Array<SIMDShort,SW,HsubsMulti>::test(repeats1, pattern);
  Array<SIMDInt,SW,HsubsMulti>::test(repeats1, pattern);
  Array<SIMDFloat,SW,HsubsMulti>::test(repeats1, pattern);
  UnaryScalarT<SIMDWord,SW,HaddSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDShort,SW,HaddSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDInt,SW,HaddSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDFloat,SW,HaddSingle>::test(repeats1, pattern);  
  UnaryScalarT<SIMDShort,SW,HaddsSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDInt,SW,HaddsSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDFloat,SW,HaddsSingle>::test(repeats1, pattern);  
  UnaryScalarT<SIMDWord,SW,HsubSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDShort,SW,HsubSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDInt,SW,HsubSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDFloat,SW,HsubSingle>::test(repeats1, pattern);  
  UnaryScalarT<SIMDShort,SW,HsubsSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDInt,SW,HsubsSingle>::test(repeats1, pattern);
  UnaryScalarT<SIMDFloat,SW,HsubsSingle>::test(repeats1, pattern);  
  TestAll<UnaryScalarT,SW,HminSingle>::test(repeats1, pattern);
  TestAll<UnaryScalarT,SW,HmaxSingle>::test(repeats1, pattern);
  // TODO: partial transpose
  TestAll<ArrayArrayVoid,SW,Transpose>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose2>::test(repeats1, pattern);
  // contributed by Adam Marschall
  TestAll<ArrayArrayVoid,SW,Transpose0>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose16>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose3>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose4>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose5>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose1_16>::test(repeats1, pattern);
  TestAll<ArrayArrayVoid,SW,Transpose6>::test(repeats1, pattern);
  // 15. Oct 22 (Jonas Keller): added test for _Swizzle
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,Swizzle>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,Swizzle2>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,Unswizzle2>::test(repeats1, pattern);
  // 15. Oct 22 (Jonas Keller): added test for swizzle4 and unswizzle4
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,Swizzle4>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,Unswizzle4>::test(repeats1, pattern);
  // 04. Dec 22 (Jonas Keller): The bitonic sorts flush denormals to zero on
  // ARMv7, since the min and max operations flush denormals to zero on ARMv7.
  // This results in autotest errors on ARMv7.
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,BitonicSort>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,BitonicSort2>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,BitonicSortSortedPairs>::test(repeats1, pattern);
  TestAllTI<ArrayVoidTemplateIntMinToMax,SW,BitonicSortSortedPairs2>::test(repeats1, pattern);
  TestAll<Binary,SW,Avgru>::test(repeats1, pattern);
  TestAll<Binary,SW,Avgrd>::test(repeats1, pattern);
  TestAll<Nullary,SW,Setones>::test(1, pattern);
  TestAll<Nullary,SW,Setmin>::test(1, pattern);
  TestAll<Nullary,SW,Setmax>::test(1, pattern);
  TestSigned<Nullary,SW,Setnegunity>::test(1, pattern);
  TestAll<Nullary,SW,Setunity>::test(1, pattern);
  ArrayTemplateType<SIMDSignedByte,SIMDSignedByte,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDSignedByte,SIMDShort,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDByte,SIMDShort,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDShort,SIMDShort,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDSignedByte,SIMDInt,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDByte,SIMDInt,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDShort,SIMDInt,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDWord,SIMDInt,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDInt,SIMDInt,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDFloat,SIMDInt,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDSignedByte,SIMDFloat,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDByte,SIMDFloat,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDShort,SIMDFloat,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDWord,SIMDFloat,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDInt,SIMDFloat,SW,PacksMulti>::test(repeats1, pattern);
  ArrayTemplateType<SIMDFloat,SIMDFloat,SW,PacksMulti>::test(repeats1, pattern);
  TestAll<Unary,SW,Div2r0>::test(repeats1, pattern);
  TestAll<Unary,SW,Div2rd>::test(repeats1, pattern);
  Binary<SIMDFloat,SW,Sign>::test(repeats1, pattern);
  TestAll<Binary,SW,AbsDiff>::test(repeats1, pattern);
  // TODO: operators
  // TODO: SIMDVecs
  printf("simdvecautotest1 complete\n");
  return 0;
}
