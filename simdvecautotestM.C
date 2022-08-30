#include "SIMDMaskVecAutoTest.H"
#include "SIMDMaskVecWrapper.H"
#include "SIMDTypes.H"
#include "SIMDVecAutoTest.H"
#include <time.h>

using namespace ns_simd;

// SW = SIMD width = number of bytes in a single SIMD vector

#ifdef _SIMD_VEC_64_AVAIL_
#define SW 64
#define SA 64
#else
#ifdef _SIMD_VEC_32_AVAIL_
#define SW 32
#define SA 32
#else
#define SW 16
#define SA 16
#endif
#endif

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
  TestFloat<TesterMaskBinary, SW, Mask_div>::test(repeats1, pattern);
  TestFloat<TesterMaskZBinary, SW, MaskZ_div>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_ceil>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_ceil>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_floor>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_floor>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_round>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_round>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_truncate>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_truncate>::test(repeats1, pattern);
  TestFloat<TesterMaskUnary, SW, Mask_sqrt>::test(repeats1, pattern);
  TestFloat<TesterMaskZUnary, SW, MaskZ_sqrt>::test(repeats1, pattern);
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

  // TODO: masked div2r0 and div2rd are not implemented for SIMDSignedByte
  TestList<TesterMaskUnary, SW, Mask_div2r0, SIMDByte, SIMDWord, SIMDShort,
           SIMDInt, SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskZUnary, SW, MaskZ_div2r0, SIMDByte, SIMDWord, SIMDShort,
           SIMDInt, SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskUnary, SW, Mask_div2rd, SIMDByte, SIMDWord, SIMDShort,
           SIMDInt, SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskZUnary, SW, MaskZ_div2rd, SIMDByte, SIMDWord, SIMDShort,
           SIMDInt, SIMDFloat>::test(repeats1, pattern);

  TestListTI<TesterMaskUnaryTIMinToMax, SW, Mask_srai, SIMDWord, SIMDShort,
             SIMDInt>::test(repeats1, pattern);
  TestListTI<TesterMaskZUnaryTIMinToMax, SW, MaskZ_srai, SIMDWord, SIMDShort,
             SIMDInt>::test(repeats1, pattern);
  TestListTI<TesterMaskUnaryTIMinToMax, SW, Mask_srli, SIMDWord, SIMDShort,
             SIMDInt>::test(repeats1, pattern);
  TestListTI<TesterMaskZUnaryTIMinToMax, SW, MaskZ_srli, SIMDWord, SIMDShort,
             SIMDInt>::test(repeats1, pattern);
  TestListTI<TesterMaskUnaryTIMinToMax, SW, Mask_slli, SIMDWord, SIMDShort,
             SIMDInt>::test(repeats1, pattern);
  TestListTI<TesterMaskZUnaryTIMinToMax, SW, MaskZ_slli, SIMDWord, SIMDShort,
             SIMDInt>::test(repeats1, pattern);

#ifdef _SIMD_VEC_64_AVAIL_
  TestAll<TesterMaskBinary, SW, Mask_hadd>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hadd>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_hadds>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hadds>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_hsub>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hsub>::test(repeats1, pattern);
  TestAll<TesterMaskBinary, SW, Mask_hsubs>::test(repeats1, pattern);
  TestAll<TesterMaskZBinary, SW, MaskZ_hsubs>::test(repeats1, pattern);
#else
  TestList<TesterMaskBinary, SW, Mask_hadd, SIMDWord, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskZBinary, SW, MaskZ_hadd, SIMDWord, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskBinary, SW, Mask_hadds, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskZBinary, SW, MaskZ_hadds, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskBinary, SW, Mask_hsub, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskZBinary, SW, MaskZ_hsub, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskBinary, SW, Mask_hsubs, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
  TestList<TesterMaskZBinary, SW, MaskZ_hsubs, SIMDShort, SIMDInt,
           SIMDFloat>::test(repeats1, pattern);
#endif

  TestAll<TesterMaskCmp, SW, Mask_cmplt>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmple>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpeq>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpgt>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpge>::test(repeats1, pattern);
  TestAll<TesterMaskCmp, SW, Mask_cmpneq>::test(repeats1, pattern);

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

  printf("simdvecautotestM complete\n");
  return 0;
}
