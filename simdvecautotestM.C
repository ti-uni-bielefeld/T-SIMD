// ===========================================================================
// 
// simdvecautotestM.H --
// autotest for masked operations, Jonas Keller
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

#include "SIMDDefs.H"
#include "SIMDVecAll.H"
#include "SIMDVecAutoTest.H"
#include "SIMDVecAutoTestMask.H"
#include "SIMDVecAutoTestWrapperMask.H"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>

using namespace ns_simd;
using namespace auto_test;

// SW = SIMD width = number of bytes in a single SIMD vector
#define SW NATIVE_SIMD_WIDTH

#define REPEATS1 100000

int main(int argc, char *argv[]) {
  srand(time(NULL));
  // default values
  int repeats1 = REPEATS1;
  std::string pattern = "";
  // overwrite default values from command line
  if (argc >= 2)
    pattern = argv[1];
  if (argc == 3)
    repeats1 = atoi(argv[2]);
  printf("pattern \"%s\", repeats1 = %d\n", pattern.c_str(), repeats1);

  TestAll<TesterMaskConditionBinary, SW, Mask_ifelse>::test(repeats1, pattern);
  TestAll<TesterMaskConditionUnary, SW, Mask_ifelsezero>::test(repeats1,
                                                               pattern);
  TesterMaskUnaryTemplateType<SIMDInt, SIMDFloat, SW, Mask_cvts>::test(repeats1,
                                                                       pattern);
  TesterMaskZUnaryTemplateType<SIMDInt, SIMDFloat, SW, MaskZ_cvts>::test(
      repeats1, pattern);
  TesterMaskUnaryTemplateType<SIMDFloat, SIMDInt, SW, Mask_cvts>::test(repeats1,
                                                                       pattern);
  TesterMaskZUnaryTemplateType<SIMDFloat, SIMDInt, SW, MaskZ_cvts>::test(
      repeats1, pattern);
  TestAll<TesterMaskScalar, SW, Mask_set1>::test(repeats1, pattern);
  TestAll<TesterMaskZScalar, SW, MaskZ_set1>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_add>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_add>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_adds>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_adds>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_sub>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_sub>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_subs>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_subs>::test(repeats1, pattern);
  TestFloat<TesterMaskBinary, SW, Mask_mul>::test(repeats1, pattern);
  TestFloat<TesterMaskZBinary, SW, MaskZ_mul>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_ceil>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_ceil>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_floor>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_floor>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_round>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_round>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_truncate>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_truncate>::test(repeats1, pattern);
  // TODO: div and sqrt are approximations on NEON, figure out max relative error
  TestFloat<TesterMaskBinary, SW, Mask_div>::test(repeats1, pattern);
  TestFloat<TesterMaskZBinary, SW, MaskZ_div>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_sqrt>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_sqrt>::test(repeats1, pattern);
  // 02. Oct 22 (Jonas Keller): added consideration of relative error for rcp and rsqrt
#if SIMDVEC_INTEL_ENABLE
#if (defined(__SSE__) || defined(__AVX__)) && !defined(__AVX512F__)
  // for sse and avx the relative error is 1.5*2^-12
  TestFloat<TesterMaskUnary, SW, Mask_rcp,CmpRelError<15, -1, -12> >::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_rcp,CmpRelError<15, -1, -12> >::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_rsqrt,CmpRelError<15, -1, -12> >::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_rsqrt,CmpRelError<15, -1, -12> >::test(repeats1, pattern);
#elif defined(__AVX512F__)
  // for avx512 the relative error is 2^-14
  TestFloat<TesterMaskUnary, SW, Mask_rcp,CmpRelError<1, 0, -14> >::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_rcp,CmpRelError<1, 0, -14> >::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_rsqrt,CmpRelError<1, 0, -14> >::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_rsqrt,CmpRelError<1, 0, -14> >::test(repeats1, pattern);
#endif
#else
  // TODO: rcp and rsqrt are approximations on NEON, figure out max relative error
  TestFloat<TesterMaskUnary, SW, Mask_rcp>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_rcp>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_rsqrt>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_rsqrt>::test(repeats1, pattern);
#endif
  TestSigned<TesterMaskUnary, SW, Mask_abs>::test(repeats1, pattern);
  TestSigned<TesterMaskZUnary, SW, MaskZ_abs>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_and>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_and>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_or>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_or>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_andnot>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_andnot>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_xor>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_xor>::test(repeats1, pattern);
  TestAll<TesterMaskUnary, SW, Mask_not>::test(repeats1, pattern);
  TestAll<TesterMaskZUnary, SW, MaskZ_not>::test(repeats1, pattern);
  TestSigned<TesterMaskUnary, SW, Mask_neg>::test(repeats1, pattern);
  TestSigned<TesterMaskZUnary, SW, MaskZ_neg>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_min>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_min>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_max>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_max>::test(repeats1, pattern);

  TestAll<TesterMaskUnary, SW, Mask_div2r0>::test(repeats1, pattern);
  TestAll<TesterMaskZUnary, SW, MaskZ_div2r0>::test(repeats1, pattern);
  TestAll<TesterMaskUnary, SW, Mask_div2rd>::test(repeats1, pattern);
  TestAll<TesterMaskZUnary, SW, MaskZ_div2rd>::test(repeats1, pattern);

  TestIntTI<TesterMaskUnaryTIMinToMax, SW, Mask_srai>::test(repeats1, pattern);
  TestIntTI<TesterMaskZUnaryTIMinToMax, SW, MaskZ_srai>::test(repeats1, pattern);
  TestIntTI<TesterMaskUnaryTIMinToMax, SW, Mask_srli>::test(repeats1, pattern);
  TestIntTI<TesterMaskZUnaryTIMinToMax, SW, MaskZ_srli>::test(repeats1, pattern);
  TestIntTI<TesterMaskUnaryTIMinToMax, SW, Mask_slli>::test(repeats1, pattern);
  TestIntTI<TesterMaskZUnaryTIMinToMax, SW, MaskZ_slli>::test(repeats1, pattern);

  TestAll<TesterMaskBinary, SW, Mask_hadd>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hadd>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_hadds>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hadds>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_hsub>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hsub>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_hsubs>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hsubs>::test(repeats1, pattern);

  TestAll<TesterMaskCmp, SW, Mask_cmplt>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmple>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpeq>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpgt>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpge>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpneq>::test(repeats1, pattern);

  // 16. Oct 22 (Jonas Keller): added tests for overloaded versions of
  // mask_cmp* functions that only take two vector parameters and no mask
  // parameter
  TestAll<TesterCmpMask, SW, Mask_cmplt>::test(repeats1, pattern);
  TestAll<TesterCmpMask, SW, Mask_cmple>::test(repeats1, pattern);
  TestAll<TesterCmpMask, SW, Mask_cmpeq>::test(repeats1, pattern);
  TestAll<TesterCmpMask, SW, Mask_cmpgt>::test(repeats1, pattern);
  TestAll<TesterCmpMask, SW, Mask_cmpge>::test(repeats1, pattern);
  TestAll<TesterCmpMask, SW, Mask_cmpneq>::test(repeats1, pattern);

  TestAll<TesterMaskBinary, SW, Mask_avg>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_avg>::test(repeats1, pattern);

  TestAll<TesterMaskUnaryInt, SW, Mask_test_all_zeros>::test(repeats1, pattern);
  TestAll<TesterMaskUnaryInt, SW, Mask_test_all_ones>::test(repeats1, pattern);

  TestAll<TesterBinaryK, SW, Kand>::test(repeats1, pattern);
  TestAll<TesterBinaryK, SW, Kandn>::test(repeats1, pattern);
  TestAll<TesterBinaryK, SW, Kor>::test(repeats1, pattern);
  TestAll<TesterBinaryK, SW, Kxor>::test(repeats1, pattern);
  TestAll<TesterBinaryK, SW, Kxnor>::test(repeats1, pattern);
  TestAll<TesterBinaryK, SW, Kadd>::test(repeats1, pattern);
  TestAll<TesterUnaryK, SW, Knot>::test(repeats1, pattern);

  TestAllTI<TesterUnaryKTIMinToMax, SW, Kshiftli>::test(repeats1, pattern);
  TestAllTI<TesterUnaryKTIMinToMax, SW, Kshiftri>::test(repeats1, pattern);

  // 28. Dec 22 (Jonas Keller): added tests for setTrueLeft and setTrueRight
  // 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
  // mask_set_true/false_low/high.
  TestAll<TesterIntToMask, SW, Mask_set_true_low>::test(repeats1, pattern);
  TestAll<TesterIntToMask, SW, Mask_set_true_high>::test(repeats1, pattern);
  TestAll<TesterIntToMask, SW, Mask_set_false_low>::test(repeats1, pattern);
  TestAll<TesterIntToMask, SW, Mask_set_false_high>::test(repeats1, pattern);

  printf("simdvecautotestM complete\n");
  return 0;
}
