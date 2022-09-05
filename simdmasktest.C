// ===========================================================================
//
// simdmasktest.C --
// test of SIMDMask templates
// Author: Markus Vieth (Bielefeld University, mvieth@techfak.uni-bielefeld.de)
// Year of creation: 2019
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
// (C) Markus Vieth, Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

// TODO: 31. Mar 22 (rm)
// TODO: there are warnings when this is compiled on an ARM architecture
// TODO: where uinttest_t is defined below as uint32_t:
// TODO: - %lx printf format for uinttest_t arguments doesn't fit
// TODO: - uinttest_t mask=0xfffffffffff overflows
// TODO: - TEST_TEMPLATE_SHIFT: compiler complains that shifts are larger
// TODO:   than width of type

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// 30. aug 22 (Jonas Keller): add workaround for gettimeofday on windows
// from https://stackoverflow.com/a/26085827/8461272
#ifdef _WIN32
#include <Windows.h>
int gettimeofday(struct timeval *tp, struct timezone *tzp) {
  // Note: some broken versions only have 8 trailing zero's, the correct epoch
  // has 9 trailing zero's This magic number is the number of 100 nanosecond
  // intervals since January 1, 1601 (UTC) until 00:00:00 January 1, 1970
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

  SYSTEMTIME system_time;
  FILETIME file_time;
  uint64_t time;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
}
// max and min are defined as macros in Windows.h, which causes problems
#undef max
#undef min
#else
#include <sys/time.h> //gettimeofday
#endif
#include "SIMDVecAll.H"

#define PRINT_PASS 0

#ifdef SIMDVEC_SANDBOX
int main() {
  printf("simdmasktest cannot be used in sandbox mode\n");
  return 0;
}
#else

using namespace ns_simd;

#ifdef __SSE__
unsigned int getcsr() { return _mm_getcsr(); }
void setcsr(unsigned int a) { _mm_setcsr(a); }
#else
unsigned int getcsr() { return 0; }
void setcsr(unsigned int) {
  // Do nothing
}
#endif

int dummy(int i) { return i; }

#ifdef SIMDVEC_NEON_ENABLE
template <typename T, int SIMD_WIDTH>
bool vectorsEqual(const SIMDVec<T, SIMD_WIDTH> &a,
                  const SIMDVec<T, SIMD_WIDTH> &b) {
  return test_all_ones(cmpeq(a, b));
}
bool vectorsEqual(const SIMDVec<SIMDFloat, 16> &a,
                  const SIMDVec<SIMDFloat, 16> &b) {
  return vectorsEqual(reinterpret<SIMDInt>(a), reinterpret<SIMDInt>(b));
}
typedef uint32_t uinttest_t;
#else
typedef uint64_t uinttest_t;
#endif

#ifdef __SSE__
bool vectorsEqual(const __m128i a, const __m128i b) {
  return _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) == 0xffff;
}

bool vectorsEqual(const __m128 a, const __m128 b) {
  return _mm_movemask_epi8(_mm_cmpeq_epi8(_mm_castps_si128(a),
                                          _mm_castps_si128(b))) == 0xffff;
}
#endif

#ifdef __AVX2__
bool vectorsEqual(const __m256i a, const __m256i b) {
  return ((unsigned int)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, b))) ==
         0xffffffff;
}

bool vectorsEqual(const __m256 a, const __m256 b) {
  return ((unsigned int)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
             _mm256_castps_si256(a), _mm256_castps_si256(b)))) == 0xffffffff;
}
#else
#ifdef __AVX__
bool vectorsEqual(__m256i a, __m256i b) {
  return test_all_ones(
      cmpeq(SIMDVec<SIMDByte, 32>(a), SIMDVec<SIMDByte, 32>(b)));
}

bool vectorsEqual(__m256 a, __m256 b) {
  return test_all_ones(cmpeq(SIMDVec<SIMDByte, 32>(_mm256_castps_si256(a)),
                             SIMDVec<SIMDByte, 32>(_mm256_castps_si256(b))));
}
#endif
#endif

#ifdef __AVX512F__
bool vectorsEqual(const __m512i a, const __m512i b) {
  // return _mm512_cmpeq_epi8_mask(a,
  // b)==0xffffffffffffffff;//test_all_zeros(cmpeq(a, b));
  return _mm512_cmpeq_epi64_mask(a, b) == 0xff;
  // return (vectorsEqual(SIMDVec<SIMDByte,64>(a).lo(),
  // SIMDVec<SIMDByte,64>(b).lo())) &&
  // (vectorsEqual(SIMDVec<SIMDByte,64>(a).hi(), SIMDVec<SIMDByte,64>(b).hi()));
}

bool vectorsEqual(const __m512 a, const __m512 b) {
  // return _mm512_cmpeq_epi8_mask(_mm512_castps_si512(a),
  // _mm512_castps_si512(b))==0xffffffffffffffff;//test_all_zeros(cmpeq(a, b));
  return (vectorsEqual(SIMDVec<SIMDFloat, 64>(a).lo(),
                       SIMDVec<SIMDFloat, 64>(b).lo())) &&
         (vectorsEqual(SIMDVec<SIMDFloat, 64>(a).hi(),
                       SIMDVec<SIMDFloat, 64>(b).hi()));
}
#endif

template <typename T, int SIMD_WIDTH> SIMDVec<T, SIMD_WIDTH> getRandomVector() {
  void *buf = malloc(SIMD_WIDTH);
  // getrandom(buf, 64, 0);
  uint8_t i;
  for (i = 0; i < SIMD_WIDTH; i++) {
    ((uint8_t *)buf)[i] = rand() & 0xff;
  }
  SIMDVec<T, SIMD_WIDTH> vec = loadu<SIMD_WIDTH, T>((T *)buf);
  free(buf);
  return vec;
}

uinttest_t random64() {
  return ((((uinttest_t)rand()) * RAND_MAX) + rand()); // rand()*1000000+rand();
}

// Returns a number 2^a where a is a random number smaller than size. In other
// words: the number contains at most "size" many successive ones.
uinttest_t random_continuous(uint8_t size) {
  long int a = (rand() % (size - 1));
  // printf("random_continuous: %li\n", a);
  // return ((intmax_t) -1)>>(63-a);
  return (1 << a) - 1;
}

// Test any function with two operands ("NAME" is the name of the function, e.g.
// "add" or "and")
#define TEST_FUNCTION(NAME, OP, TYPE, SIMD_WIDTH)                              \
  {                                                                            \
    unsigned int csr0, csr1, csr2;                                             \
    SIMDVec<TYPE, SIMD_WIDTH> src = setzero<TYPE, SIMD_WIDTH>(),               \
                              a = getRandomVector<TYPE, SIMD_WIDTH>(),         \
                              b = getRandomVector<TYPE, SIMD_WIDTH>(), res1,   \
                              res2;                                            \
    /* First test */                                                           \
    SIMDMask<TYPE, SIMD_WIDTH> k = mask_all_ones<TYPE, SIMD_WIDTH>();          \
    csr0 = getcsr();                                                           \
    res1 = mask_##NAME(src, k, a, b);                                          \
    csr1 = getcsr();                                                           \
    setcsr(csr0);                                                              \
    res2 = NAME(a, b);                                                         \
    csr2 = getcsr();                                                           \
    if (!vectorsEqual(res1, res2) ||                                           \
        csr1 != csr2) /* TODO problem with flags */                            \
    {                                                                          \
      printf("Problem with %s for %s test 1, csr1=%x, csr2=%x \na=", #NAME,    \
             #TYPE, csr1, csr2);                                               \
      print("%e ", a);                                                         \
      printf("\nb=");                                                          \
      print("%e ", b);                                                         \
      printf("\nres1=");                                                       \
      print("%e ", res1);                                                      \
      printf("\nres2=");                                                       \
      print("%e ", res2);                                                      \
      printf("\nres1=");                                                       \
      print("%x ", reinterpret<SIMDInt>(res1));                                \
      printf("\nres2=");                                                       \
      print("%x ", reinterpret<SIMDInt>(res2));                                \
      puts("");                                                                \
    } else {                                                                   \
      if (PRINT_PASS)                                                          \
        printf("Pass %s for %s \n", #NAME, #TYPE);                             \
    }                                                                          \
    /* Second test */                                                          \
    SIMDMask<TYPE, SIMD_WIDTH> kzero;                                          \
    setcsr(csr0);                                                              \
    res1 = maskz_##NAME(kzero, a, b);                                          \
    if (!vectorsEqual(res1, setzero<TYPE, SIMD_WIDTH>()) ||                    \
        getcsr() != csr0) {                                                    \
      printf("Problem with %s for %s test 2, csr=%x, csr0=%x \na=", #NAME,     \
             #TYPE, getcsr(), csr0);                                           \
      print("%e ", a);                                                         \
      printf("\nb=");                                                          \
      print("%e ", b);                                                         \
      printf("\nres1=");                                                       \
      print("%e ", res1);                                                      \
      printf("\n");                                                            \
    }                                                                          \
    setcsr(csr0);                                                              \
  }

// Test any function with one operand ("NAME" is the name of the function, e.g.
// "sqrt" or "abs")
#define TEST_FUNCTION_SOP(NAME, OP, TYPE, SIMD_WIDTH, FORMAT)                  \
  {                                                                            \
    /*printf("Testing function %s for type %s\n", #NAME, #TYPE);*/             \
    unsigned int csr0, csr1, csr2;                                             \
    csr0 = getcsr();                                                           \
    SIMDVec<TYPE, SIMD_WIDTH> src = setzero<TYPE, SIMD_WIDTH>(),               \
                              a = getRandomVector<TYPE, SIMD_WIDTH>(), res1,   \
                              res2;                                            \
    /* First test */                                                           \
    SIMDMask<TYPE, SIMD_WIDTH> k = mask_all_ones<TYPE, SIMD_WIDTH>();          \
    res1 = mask_##NAME(src, k, a);                                             \
    csr1 = getcsr();                                                           \
    setcsr(csr0);                                                              \
    res2 = NAME(a);                                                            \
    csr2 = getcsr();                                                           \
    /*print("%x ", reinterpret<SIMDInt>(cmpeq(res1, res2))); puts("");         \
    printf("\nres1="); print("%x ", reinterpret<SIMDInt>(res1));               \
    printf("\nres2="); print("%x ", reinterpret<SIMDInt>(res2)); puts("");     \
    print("%e ", res1); printf("\nres2="); print("%e ", res2); puts("");*/     \
    if (!vectorsEqual(res1, res2) ||                                           \
        csr1 != csr2) /* TODO problem with flags */                            \
    {                                                                          \
      printf("Problem with %s for %s test 1, csr1=%x, csr2=%x \na=", #NAME,    \
             #TYPE, csr1, csr2);                                               \
      print("%e ", a);                                                         \
      printf("\nres1=");                                                       \
      print("%e ", res1);                                                      \
      printf("\nres2=");                                                       \
      print("%e ", res2);                                                      \
      printf("\nres1=");                                                       \
      print("%x ", reinterpret<SIMDInt>(res1));                                \
      printf("\nres2=");                                                       \
      print("%x ", reinterpret<SIMDInt>(res2));                                \
      puts("");                                                                \
    } else {                                                                   \
      if (PRINT_PASS)                                                          \
        printf("Pass %s for %s \n", #NAME, #TYPE);                             \
    }                                                                          \
    /* Second test */                                                          \
    SIMDMask<TYPE, SIMD_WIDTH> kzero;                                          \
    setcsr(csr0);                                                              \
    res1 = maskz_##NAME(kzero, a);                                             \
    if (!vectorsEqual(res1, setzero<TYPE, SIMD_WIDTH>()) ||                    \
        getcsr() != csr0) {                                                    \
      printf("Problem with %s for %s test 2, csr=%x, csr0=%x \na=", #NAME,     \
             #TYPE, getcsr(), csr0);                                           \
      print("%e ", a);                                                         \
      printf("\nres1=");                                                       \
      print("%e ", res1);                                                      \
      printf("\nzeros=");                                                      \
      print("%e ", setzero<TYPE, SIMD_WIDTH>());                               \
      puts("");                                                                \
    }                                                                          \
    setcsr(csr0);                                                              \
    /* Third test */                                                           \
    unsigned int i;                                                            \
    if (strcmp(#OP, "dummy") != 0) {                                           \
      uinttest_t mask = random64();                                            \
      SIMDMask<TYPE, SIMD_WIDTH> rand_mask;                                    \
      rand_mask = mask;                                                        \
      TYPE *arg = (TYPE *)malloc(SIMD_WIDTH),                                  \
           *result1 = (TYPE *)malloc(SIMD_WIDTH),                              \
           *result2 = (TYPE *)malloc(SIMD_WIDTH);                              \
      for (i = 0; i < SIMD_WIDTH; i++) {                                       \
        ((uint8_t *)arg)[i] = rand() & 0xff;                                   \
      }                                                                        \
      storeu(result1, maskz_##NAME(rand_mask, loadu<SIMD_WIDTH>(arg)));        \
      /*print("%i ", maskz_ ## NAME(k, loadu<SIMD_WIDTH>(arg))); puts("");*/   \
      for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                        \
        if ((mask & (1lu << i)) != 0) {                                        \
          result2[i] = OP(arg[i]);                                             \
        } else {                                                               \
          result2[i] = 0;                                                      \
        }                                                                      \
      }                                                                        \
      if (memcmp(result1, result2, SIMD_WIDTH) != 0) {                         \
        printf("Problem with %s for %s test 3, csr1=%x, csr2=%x result 1=",    \
               #NAME, #TYPE, csr1, csr2);                                      \
        for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                      \
          printf(FORMAT, result1[i]);                                          \
        }                                                                      \
        printf(" result2=");                                                   \
        for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                      \
          printf(FORMAT, result2[i]);                                          \
        }                                                                      \
        printf(" mask=%lx=%lx\n", mask, (uinttest_t)rand_mask);                \
      } else {                                                                 \
        if (PRINT_PASS)                                                        \
          printf("Pass test 2 %s for %s \n", #NAME, #TYPE);                    \
      }                                                                        \
    }                                                                          \
    setcsr(csr0);                                                              \
  }

// Test any function with one operand ("NAME" is the name of the function, e.g.
// "sqrt" or "abs")
#define TEST_FUNCTION_CMP(NAME, OP, TYPE, SIMD_WIDTH, FORMAT)                  \
  {                                                                            \
    /*printf("Testing function %s for type %s\n", #NAME, #TYPE);*/             \
    unsigned int csr0;                                                         \
    csr0 = getcsr();                                                           \
    SIMDVec<TYPE, SIMD_WIDTH> a = getRandomVector<TYPE, SIMD_WIDTH>(),         \
                              b = getRandomVector<TYPE, SIMD_WIDTH>();         \
    SIMDMask<TYPE, SIMD_WIDTH> res1, res2;                                     \
    /* Second test */                                                          \
    SIMDMask<TYPE, SIMD_WIDTH> kzero;                                          \
    setcsr(csr0);                                                              \
    res1 = mask_##NAME(kzero, a, b);                                           \
    if (res1 != 0 || getcsr() != csr0) {                                       \
      printf("Problem with %s for %s test 2, csr=%x, csr0=%x \na=", #NAME,     \
             #TYPE, getcsr(), csr0);                                           \
      print("%e ", a);                                                         \
      printf("\nres1=%lx", (uinttest_t)res1);                                  \
      printf("\n");                                                            \
    }                                                                          \
    setcsr(csr0);                                                              \
    /* Third test */                                                           \
    unsigned int i;                                                            \
    if (strcmp(#OP, "dummy") != 0) {                                           \
      uinttest_t mask = 0xfffffffffff; /*random64();*/                         \
      SIMDMask<TYPE, SIMD_WIDTH> rand_mask;                                    \
      rand_mask = mask;                                                        \
      TYPE *arg1 = (TYPE *)malloc(SIMD_WIDTH),                                 \
           *arg2 = (TYPE *)malloc(SIMD_WIDTH);                                 \
      uinttest_t result1 = 0, result2 = 0;                                     \
      for (i = 0; i < SIMD_WIDTH; i++) {                                       \
        ((uint8_t *)arg1)[i] = rand() & 0xff;                                  \
        ((uint8_t *)arg2)[i] = rand() & 0xff;                                  \
      }                                                                        \
      result1 = mask_##NAME(rand_mask, loadu<SIMD_WIDTH>(arg1),                \
                            loadu<SIMD_WIDTH>(arg2));                          \
      /*print("%i ", maskz_ ## NAME(k, loadu<SIMD_WIDTH>(arg))); puts("");*/   \
      for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                        \
        if ((mask & (1lu << i)) != 0 && (arg1[i] OP arg2[i])) {                \
          result2 |= 1lu << i;                                                 \
        }                                                                      \
      }                                                                        \
      if (result1 != result2) {                                                \
        printf("Problem with %s for %s test 3, result 1=%lx, result2=%lx  "    \
               "mask=%lx=%lx\narg1=",                                          \
               #NAME, #TYPE, result1, result2, mask, (uinttest_t)rand_mask);   \
        for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                      \
          printf(FORMAT, arg1[i]);                                             \
        }                                                                      \
        printf("\narg2=");                                                     \
        for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                      \
          printf(FORMAT, arg2[i]);                                             \
        }                                                                      \
        puts("");                                                              \
      } else {                                                                 \
        if (PRINT_PASS)                                                        \
          printf("Pass test 2 %s for %s \n", #NAME, #TYPE);                    \
      }                                                                        \
    }                                                                          \
    setcsr(csr0);                                                              \
  }

// Test any function with one operand ("NAME" is the name of the function, e.g.
// "sqrt" or "abs")
#define TEST_FUNCTION_SHIFT(NAME, OP, TYPE, SIMD_WIDTH, FORMAT, COUNT)         \
  {                                                                            \
    /*printf("Testing function %s for type %s\n", #NAME, #TYPE);*/             \
    unsigned int csr0, csr1, csr2;                                             \
    SIMDVec<TYPE, SIMD_WIDTH> src = setzero<TYPE, SIMD_WIDTH>(),               \
                              a = getRandomVector<TYPE, SIMD_WIDTH>(), res1,   \
                              res2;                                            \
    /* First test */                                                           \
    SIMDMask<TYPE, SIMD_WIDTH> k = mask_all_ones<TYPE, SIMD_WIDTH>();          \
    csr0 = getcsr();                                                           \
    res1 = mask_##NAME<COUNT>(src, k, a);                                      \
    csr1 = getcsr();                                                           \
    setcsr(csr0);                                                              \
    res2 = NAME<COUNT>(a);                                                     \
    csr2 = getcsr();                                                           \
    /*print("%x ", reinterpret<SIMDInt>(cmpeq(res1, res2))); puts("");         \
    printf("\nres1="); print("%x ", reinterpret<SIMDInt>(res1));               \
    printf("\nres2="); print("%x ", reinterpret<SIMDInt>(res2)); puts("");     \
    print("%e ", res1); printf("\nres2="); print("%e ", res2); puts("");*/     \
    if (!vectorsEqual(res1, res2) ||                                           \
        csr1 != csr2) /* TODO problem with flags */                            \
    {                                                                          \
      printf("Problem with %s for %s test 1, csr1=%x, csr2=%x \na=", #NAME,    \
             #TYPE, csr1, csr2);                                               \
      print("%e ", a);                                                         \
      printf("\nres1=");                                                       \
      print("%e ", res1);                                                      \
      printf("\nres2=");                                                       \
      print("%e ", res2);                                                      \
      printf("\nres1=");                                                       \
      print("%x ", reinterpret<SIMDInt>(res1));                                \
      printf("\nres2=");                                                       \
      print("%x ", reinterpret<SIMDInt>(res2));                                \
      puts("");                                                                \
    } else {                                                                   \
      if (PRINT_PASS)                                                          \
        printf("Pass %s for %s \n", #NAME, #TYPE);                             \
    }                                                                          \
    /* Second test */                                                          \
    SIMDMask<TYPE, SIMD_WIDTH> kzero;                                          \
    setcsr(csr0);                                                              \
    res1 = maskz_##NAME<COUNT>(kzero, a);                                      \
    if (!vectorsEqual(res1, setzero<TYPE, SIMD_WIDTH>()) ||                    \
        getcsr() != csr0) {                                                    \
      printf("Problem with %s for %s, csr=%x test 2, csr0=%x \na=", #NAME,     \
             #TYPE, getcsr(), csr0);                                           \
      print("%e ", a);                                                         \
      printf("\nres1=");                                                       \
      print("%e ", res1);                                                      \
      printf("\n");                                                            \
    }                                                                          \
    setcsr(csr0);                                                              \
    /* Third test */                                                           \
    /*unsigned int i;                                                          \
    if(strcmp(#OP, "dummy")!=0) {                                              \
    uinttest_t mask=random64();                                                \
    SIMDMask<TYPE, SIMD_WIDTH> rand_mask;                                      \
    rand_mask=mask;                                                            \
    TYPE * arg=(TYPE *) malloc(SIMD_WIDTH), * result1=(TYPE *)                 \
    malloc(SIMD_WIDTH), * result2=(TYPE *) malloc(SIMD_WIDTH); for(i=0;                                                                      \
    i<SIMD_WIDTH; i++) {                                                       \
        ((uint8_t *) arg)[i] = rand()&0xff;                                    \
    }                                                                          \
    storeu(result1, maskz_ ## NAME<COUNT>(rand_mask, loadu<SIMD_WIDTH>(arg))); \
    for(i=0; i<SIMD_WIDTH/sizeof(TYPE); i++) {                                 \
     if((mask&(1lu<<i))!=0) {                                                  \
       result2[i]=(arg[i]) OP COUNT;                                           \
     } else {                                                                  \
       result2[i]=0;                                                           \
     }                                                                         \
    }                                                                          \
    if(memcmp(result1, result2, SIMD_WIDTH)!=0)                                \
    { printf("Problem with %s for %s test 3, csr1=%x, csr2=%x result 1=",      \
    #NAME, #TYPE, csr1, csr2); for(i=0; i<SIMD_WIDTH/sizeof(TYPE); i++) {                                    \
    printf(FORMAT, result1[i]); } printf(" result2="); for(i=0;                                                                      \
    i<SIMD_WIDTH/sizeof(TYPE); i++) { printf(FORMAT, result2[i]); } printf("   \
    mask=%lx=%lx\n", mask, (uinttest_t) rand_mask); }                          \
    }*/                                                                        \
    setcsr(csr0);                                                              \
  }

#define TEST_FUNCTION_MASK_TEST_ALL_(NAME, TARGET, TYPE, SIMD_WIDTH, FORMAT)   \
  {                                                                            \
    /*printf("Testing function %s for type %s\n", #NAME, #TYPE);*/             \
    /* Third test */                                                           \
    unsigned int i;                                                            \
    uinttest_t mask = random64();                                              \
    SIMDMask<TYPE, SIMD_WIDTH> rand_mask;                                      \
    rand_mask = mask;                                                          \
    TYPE *arg = (TYPE *)malloc(SIMD_WIDTH);                                    \
    int result1, result2 = 1;                                                  \
    for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                          \
      arg[i] = ((rand() & 1) == 0 ? TARGET : 2);                               \
    }                                                                          \
    result1 = mask_test_all_##NAME(rand_mask, loadu<SIMD_WIDTH>(arg));         \
    for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                          \
      if ((mask & (1lu << i)) != 0) {                                          \
        if (arg[i] != TARGET) {                                                \
          result2 = 0;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
    if (result1 != result2) {                                                  \
      printf("Problem with mask_test_all_ %s for %s test 3, result 1=%i, "     \
             "result 2=%i",                                                    \
             #NAME, #TYPE, result1, result2);                                  \
      printf(" mask=%lx=%lx\n arg=", mask, (uinttest_t)rand_mask);             \
      for (i = 0; i < SIMD_WIDTH / sizeof(TYPE); i++) {                        \
        printf(FORMAT, arg[i]);                                                \
      }                                                                        \
      puts("");                                                                \
    } else {                                                                   \
      if (PRINT_PASS)                                                          \
        printf("Pass test 2 %s for %s \n", #NAME, #TYPE);                      \
    }                                                                          \
  }

// Test load function: 1. Test if "maskz_load" does the same as "load" if all
// mask bits are ones. 2. Test if "maskz_load" does not access the memory and
// returns only zeros if all mask bits are zeros. 3. Test if maskz_load with a
// random mask returns the same result as load and mask_ifelse
#define TEST_FUNCTION_LOAD_TYPE(TYPE, SIMD_WIDTH, FORMAT)                      \
  {                                                                            \
    unsigned int csr0, csr1, csr2;                                             \
    TYPE *buffer = (TYPE *)simd_aligned_malloc(SIMD_WIDTH, SIMD_WIDTH);        \
    if (buffer != NULL) {                                                      \
      uint8_t i;                                                               \
      for (i = 0; i < SIMD_WIDTH; i++) {                                       \
        ((uint8_t *)buffer)[i] = rand() & 0xff;                                \
      }                                                                        \
      SIMDVec<TYPE, SIMD_WIDTH> res1, res2;                                    \
      /* First test */                                                         \
      SIMDMask<TYPE, SIMD_WIDTH> k = mask_all_ones<TYPE, SIMD_WIDTH>();        \
      csr0 = getcsr();                                                         \
      /*printf("Performing %s for %s\n", "load", #TYPE);*/                     \
      res1 = maskz_load(k, buffer);                                            \
      csr1 = getcsr();                                                         \
      setcsr(csr0);                                                            \
      res2 = load<SIMD_WIDTH, TYPE>(buffer);                                   \
      csr2 = getcsr();                                                         \
      /*printf("%u\n", SIMD_WIDTH/sizeof(TYPE));*/                             \
      if (!vectorsEqual(res1, res2) || csr1 != csr2) {                         \
        printf("Problem with %s for %s, csr1=%x, csr2=%x \n", "load", #TYPE,   \
               csr1, csr2);                                                    \
        printf("res1=");                                                       \
        print(FORMAT, res1);                                                   \
        printf("\nres2=");                                                     \
        print(FORMAT, res2);                                                   \
        printf("\nres1=");                                                     \
        print("%x ", reinterpret<SIMDInt>(res1));                              \
        printf("\nres2=");                                                     \
        print("%x ", reinterpret<SIMDInt>(res2));                              \
        puts("");                                                              \
      } else {                                                                 \
        if (PRINT_PASS)                                                        \
          printf("Pass %s for %s \n", "load", #TYPE);                          \
      }                                                                        \
      /* Second test */                                                        \
      SIMDMask<TYPE, SIMD_WIDTH> kzero;                                        \
      setcsr(csr0);                                                            \
      res1 = maskz_load(kzero, (TYPE *)0);                                     \
      if (!vectorsEqual(res1, setzero<TYPE, SIMD_WIDTH>()) ||                  \
          getcsr() != csr0) {                                                  \
        printf("Problem with %s for %s, csr=%x, csr0=%x \n", "load", #TYPE,    \
               getcsr(), csr0);                                                \
        printf("res1=");                                                       \
        print(FORMAT, res1);                                                   \
        printf("\n");                                                          \
      }                                                                        \
      setcsr(csr0);                                                            \
      /* Third test */                                                         \
      SIMDMask<TYPE, SIMD_WIDTH> krand;                                        \
      krand = random64();                                                      \
      res1 = maskz_load(krand, buffer);                                        \
      csr1 = getcsr();                                                         \
      setcsr(csr0);                                                            \
      res2 = mask_ifelse(krand, load<SIMD_WIDTH, TYPE>(buffer),                \
                         setzero<TYPE, SIMD_WIDTH>());                         \
      csr2 = getcsr();                                                         \
      if (!vectorsEqual(res1, res2) || csr1 != csr2) {                         \
        printf("Problem with %s for %s, csr1=%x, csr2=%x \n", "load", #TYPE,   \
               csr1, csr2);                                                    \
        printf("res1=");                                                       \
        print(FORMAT, res1);                                                   \
        printf("\nres2=");                                                     \
        print(FORMAT, res2);                                                   \
        printf("\nres1=");                                                     \
        print("%x ", reinterpret<SIMDInt>(res1));                              \
        printf("\nres2=");                                                     \
        print("%x ", reinterpret<SIMDInt>(res2));                              \
        puts("");                                                              \
      } else {                                                                 \
        if (PRINT_PASS)                                                        \
          printf("Pass %s for %s \n", "load", #TYPE);                          \
      }                                                                        \
      setcsr(csr0);                                                            \
      simd_aligned_free(buffer);                                               \
    } else {                                                                   \
      puts("Error simd_aligned_malloc");                                       \
    }                                                                          \
  }

// Test store function: 1. Test if "mask_store" does the same as "store" if all
// mask bits are ones. 2. Test if mask_store does not write anything if all mask
// bits are zeros
#define TEST_FUNCTION_STORE_TYPE(TYPE, SIMD_WIDTH)                             \
  {                                                                            \
    SIMDVec<TYPE, SIMD_WIDTH> a = getRandomVector<TYPE, SIMD_WIDTH>();         \
    TYPE *buffer1 = (TYPE *)simd_aligned_malloc(SIMD_WIDTH, SIMD_WIDTH);       \
    TYPE *buffer2 = (TYPE *)simd_aligned_malloc(SIMD_WIDTH, SIMD_WIDTH);       \
    if (buffer1 != NULL && buffer2 != NULL) {                                  \
      SIMDMask<TYPE, SIMD_WIDTH> k = mask_all_ones<TYPE, SIMD_WIDTH>();        \
      unsigned int csr0 = getcsr();                                            \
      /*printf("Performing %s for %s\n", "store", #TYPE);*/                    \
      mask_store(buffer1, k, a);                                               \
      unsigned int csr1 = getcsr();                                            \
      setcsr(csr0);                                                            \
      store(buffer2, a);                                                       \
      unsigned int csr2 = getcsr();                                            \
      if (memcmp(buffer1, buffer2, SIMD_WIDTH) != 0 ||                         \
          csr1 != csr2) /* TODO problem with flags */                          \
      {                                                                        \
        printf("Problem with %s for %s, csr1=%x, csr2=%x \n", "store", #TYPE,  \
               csr1, csr2);                                                    \
      } else {                                                                 \
        if (PRINT_PASS)                                                        \
          printf("Pass %s for %s \n", "store", #TYPE);                         \
      }                                                                        \
      SIMDMask<TYPE, SIMD_WIDTH> kzero;                                        \
      setcsr(csr0);                                                            \
      mask_store(                                                              \
          (TYPE *)0, kzero,                                                    \
          a); /* As long as there is no segfault here, this is a success. */   \
      setcsr(csr0);                                                            \
      simd_aligned_free(buffer1);                                              \
      simd_aligned_free(buffer2);                                              \
    } else {                                                                   \
      puts("Error simd_aligned_malloc");                                       \
    }                                                                          \
  }

template <typename Tin, typename Tout, int SIMD_WIDTH> void test_cvts() {
  unsigned int csr0, csr1, csr2;
  csr0 = getcsr();
  SIMDVec<Tin, SIMD_WIDTH> a = getRandomVector<Tin, SIMD_WIDTH>();
  SIMDVec<Tout, SIMD_WIDTH> res1, res2,
      src = getRandomVector<Tout, SIMD_WIDTH>();
  /* First test */
  SIMDMask<Tin, SIMD_WIDTH> k = mask_all_ones<Tin, SIMD_WIDTH>();
  res1 = mask_cvts(src, k, a);
  csr1 = getcsr();
  setcsr(csr0);
  res2 = cvts<Tout>(a);
  csr2 = getcsr();
  /*print("%x ", reinterpret<SIMDInt>(cmpeq(res1, res2))); puts("");
  printf("\nres1="); print("%x ", reinterpret<SIMDInt>(res1));
  printf("\nres2="); print("%x ", reinterpret<SIMDInt>(res2)); puts("");
  print("%e ", res1); printf("\nres2="); print("%e ", res2); puts("");*/
  if (!vectorsEqual(res1, res2) || csr1 != csr2) /* TODO problem with flags */
  {
    printf("Problem with %s for %s test 1, csr1=%x, csr2=%x \na=", "cvts",
           __PRETTY_FUNCTION__, csr1, csr2);
    print("%e ", a);
    printf("\nres1=");
    print("%e ", res1);
    printf("\nres2=");
    print("%e ", res2);
    printf("\nres1=");
    print("%x ", reinterpret<SIMDInt>(res1));
    printf("\nres2=");
    print("%x ", reinterpret<SIMDInt>(res2));
    puts("");
  } else {
    if (PRINT_PASS)
      printf("Pass cvts for Tin %s Tout %s\n", SIMDTypeInfo<Tin>::name(),
             SIMDTypeInfo<Tout>::name());
  }
  /* Second test */
  SIMDMask<Tin, SIMD_WIDTH> kzero;
  setcsr(csr0);
  res1 = mask_cvts(src, kzero, a);
  if (!vectorsEqual(res1, src /*setzero<Tout, SIMD_WIDTH>()*/) ||
      getcsr() != csr0) {
    printf("Problem with %s for %s test 2, csr=%x, csr0=%x \na=", "cvts",
           __PRETTY_FUNCTION__, getcsr(), csr0);
    print("%e ", a);
    printf("\nres1=");
    print("%e ", res1);
    printf("\n");
  }
  setcsr(csr0);
}

#define TEST_FUNCTION_LOAD_ALL_TYPES                                           \
  TEST_FUNCTION_LOAD_TYPE(SIMDByte, NATIVE_SIMD_WIDTH, "%3u ")                 \
  TEST_FUNCTION_LOAD_TYPE(SIMDSignedByte, NATIVE_SIMD_WIDTH, "%3i ")           \
      TEST_FUNCTION_LOAD_TYPE(SIMDWord, NATIVE_SIMD_WIDTH, "%6u ")             \
          TEST_FUNCTION_LOAD_TYPE(SIMDShort, NATIVE_SIMD_WIDTH, "%6i ")        \
              TEST_FUNCTION_LOAD_TYPE(SIMDInt, NATIVE_SIMD_WIDTH, "%8i ")      \
                  TEST_FUNCTION_LOAD_TYPE(SIMDFloat, NATIVE_SIMD_WIDTH, "%e ")

#define TEST_FUNCTION_STORE_ALL_TYPES                                          \
  TEST_FUNCTION_STORE_TYPE(SIMDByte, NATIVE_SIMD_WIDTH)                        \
  TEST_FUNCTION_STORE_TYPE(SIMDSignedByte, NATIVE_SIMD_WIDTH)                  \
      TEST_FUNCTION_STORE_TYPE(SIMDWord, NATIVE_SIMD_WIDTH)                    \
          TEST_FUNCTION_STORE_TYPE(SIMDShort, NATIVE_SIMD_WIDTH)               \
              TEST_FUNCTION_STORE_TYPE(SIMDInt, NATIVE_SIMD_WIDTH)             \
                  TEST_FUNCTION_STORE_TYPE(SIMDFloat, NATIVE_SIMD_WIDTH)

#define TEST_FUNCTION_ALL_INTS(NAME, OP)                                       \
  TEST_FUNCTION(NAME, OP, SIMDByte, NATIVE_SIMD_WIDTH)                         \
  TEST_FUNCTION(NAME, OP, SIMDSignedByte, NATIVE_SIMD_WIDTH)                   \
      TEST_FUNCTION(NAME, OP, SIMDWord, NATIVE_SIMD_WIDTH)                     \
          TEST_FUNCTION(NAME, OP, SIMDShort, NATIVE_SIMD_WIDTH)                \
              TEST_FUNCTION(NAME, OP, SIMDInt, NATIVE_SIMD_WIDTH)

#define TEST_FUNCTION_ALL_TYPES(NAME, OP)                                      \
  TEST_FUNCTION_ALL_INTS(NAME, OP)                                             \
  TEST_FUNCTION(NAME, OP, SIMDFloat, NATIVE_SIMD_WIDTH)

#define TEST_FUNCTION_ALL_INTS_SOP(NAME, OP)                                   \
  TEST_FUNCTION_SOP(NAME, OP, SIMDByte, NATIVE_SIMD_WIDTH, "%u ")              \
  TEST_FUNCTION_SOP(NAME, OP, SIMDSignedByte, NATIVE_SIMD_WIDTH, "%i ")        \
      TEST_FUNCTION_SOP(NAME, OP, SIMDWord, NATIVE_SIMD_WIDTH, "%u ")          \
          TEST_FUNCTION_SOP(NAME, OP, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")     \
              TEST_FUNCTION_SOP(NAME, OP, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")

#define TEST_FUNCTION_ALL_TYPES_SOP(NAME, OP)                                  \
  TEST_FUNCTION_ALL_INTS_SOP(NAME, OP)                                         \
  TEST_FUNCTION_SOP(NAME, OP, SIMDFloat, NATIVE_SIMD_WIDTH, "%e ")

#define TEST_FUNCTION_ALL_TYPES_CMP(NAME, OP)                                  \
  TEST_FUNCTION_CMP(NAME, OP, SIMDByte, NATIVE_SIMD_WIDTH, "%u ")              \
  TEST_FUNCTION_CMP(NAME, OP, SIMDSignedByte, NATIVE_SIMD_WIDTH, "%i ")        \
      TEST_FUNCTION_CMP(NAME, OP, SIMDWord, NATIVE_SIMD_WIDTH, "%u ")          \
          TEST_FUNCTION_CMP(NAME, OP, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")     \
              TEST_FUNCTION_CMP(NAME, OP, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")   \
                  TEST_FUNCTION_CMP(NAME, OP, SIMDFloat, NATIVE_SIMD_WIDTH,    \
                                    "%e ")

#define TEST_FUNCTION_ALL_TYPES_SHIFT(NAME, OP, COUNT)                         \
  TEST_FUNCTION_SHIFT(NAME, OP, SIMDByte, NATIVE_SIMD_WIDTH, "%u ", COUNT)     \
  TEST_FUNCTION_SHIFT(NAME, OP, SIMDSignedByte, NATIVE_SIMD_WIDTH, "%i ",      \
                      COUNT)                                                   \
      TEST_FUNCTION_SHIFT(NAME, OP, SIMDWord, NATIVE_SIMD_WIDTH, "%u ", COUNT) \
          TEST_FUNCTION_SHIFT(NAME, OP, SIMDShort, NATIVE_SIMD_WIDTH, "%i ",   \
                              COUNT)                                           \
              TEST_FUNCTION_SHIFT(NAME, OP, SIMDInt, NATIVE_SIMD_WIDTH, "%i ", \
                                  COUNT)

/*template <typename T, int SIMD_WIDTH>
void test_function(SIMDVec<T, SIMD_WIDTH> (*function) (const SIMDVec<T,
SIMD_WIDTH> &src, const SIMDMask<T, SIMD_WIDTH> &k, const SIMDVec<T, SIMD_WIDTH>
&a, const SIMDVec<T, SIMD_WIDTH> &b)) { _mm_getcsr()
}*/

template <typename T, int SIMD_WIDTH> void benchmark() {
  unsigned int i;
  SIMDVec<T, SIMD_WIDTH> sum = setzero<T, SIMD_WIDTH>();
  SIMDVec<T, SIMD_WIDTH> result = setzero<T, SIMD_WIDTH>();
  SIMDMask<T, SIMD_WIDTH> kzeros;
  SIMDMask<T, SIMD_WIDTH> kones = mask_all_ones<T, SIMD_WIDTH>();
  SIMDMask<T, SIMD_WIDTH> krand;
  SIMDMask<T, SIMD_WIDTH> krand2;
  T *buffer = (T *)simd_aligned_malloc(SIMD_WIDTH, SIMD_WIDTH);
  if (buffer == NULL) {
    puts("Error simd_aligned_malloc");
    return;
  }
  struct timeval start, end;
  srand(time(NULL));
  for (i = 0; i < SIMD_WIDTH; i++) {
    ((uint8_t *)buffer)[i] = rand() & 0xff;
  }

  gettimeofday(&start, NULL);
  for (i = 0; i < 0x8fffffff; i++) {
    krand = random64();
    krand2 =
        random_continuous(4); // TODO change parameter according to width of T
#ifdef BENCH_MASK_ZEROS
    result = maskz_load(kzeros, buffer);
#endif
#ifdef BENCH_MASK_ONES
    result = maskz_load(kones, buffer);
#endif
#ifdef BENCH_MASK_RAND
    result = maskz_load(krand, buffer);
#endif
#ifdef BENCH_MASK_RAND2
    result = maskz_load(krand2, buffer);
#endif
    sum = add(sum, result);
  }
  gettimeofday(&end, NULL);
  print("%i ", sum); // TODO change format according to type T
  // printf("%f seconds elapsed\n",
  // ((end.tv_sec-start.tv_sec)+(end.tv_usec-start.tv_usec)/1000000.0));
  printf("%.1f", ((end.tv_sec - start.tv_sec) +
                  (end.tv_usec - start.tv_usec) / 1000000.0));
}

#define TEST_TEMPLATE_SHIFT(R_OR_L, I, OP)                                     \
  kshift##R_OR_L##i_mask = ones & (a OP I);                                    \
  if (!(kshift##R_OR_L##i_mask == kshift##R_OR_L##i<I>(a_mask))) {             \
    printf("ERROR: template kshift %s i, i=%i, should be: %lx, is: %lx\n",     \
           #R_OR_L, I, (uinttest_t)kshift##R_OR_L##i_mask,                     \
           (uinttest_t)kshift##R_OR_L##i<I>(a_mask));                          \
  }

template <typename T, int SIMD_WIDTH> void test_mask_functions() {
  // printf("begin %s\n", __PRETTY_FUNCTION__);
  // kand, kandn, kor, kxor, kxnor, kadd, knot, kshiftli, kshiftri
  uinttest_t i;
  SIMDMask<T, SIMD_WIDTH> a_mask, b_mask, kand_mask, kandn_mask, kor_mask,
      kxor_mask, kxnor_mask, kadd_mask, knot_mask, kshiftli_mask, kshiftri_mask;
  uinttest_t ones =
      (SIMD_WIDTH / sizeof(T) == 64
           ? -1
           : ((1ul << (SIMD_WIDTH / sizeof(T))) -
              1)); // Conditional operator removes compiler warning
  uinttest_t a = ones & random64(), b = ones & random64();
  a_mask = a;
  b_mask = b;
  if (a != ((uinttest_t)a_mask) || b != ((uinttest_t)b_mask)) {
    printf("ERROR: mask conversion: a=%lx %lx, b=%lx %lx, ones=%lx, "
           "SIMD_WIDTH/sizeof(T)=%lu, %s\n",
           a, (uinttest_t)a_mask, b, (uinttest_t)b_mask, ones,
           SIMD_WIDTH / sizeof(T), __PRETTY_FUNCTION__);
  }
  for (i = 0; i < 64; i++) {
    if ((a_mask[i]) != (((a >> i) & 1) == 1)) {
      printf("ERROR: operator[], %s\n", __PRETTY_FUNCTION__);
    }
  }
  // printf("a=%lx %lx, b=%lx %lx, ones=%lx, SIMD_WIDTH/sizeof(T)=%lu\n", a,
  // (uinttest_t) a_mask, b, (uinttest_t) b_mask, ones, SIMD_WIDTH/sizeof(T));
  // SIMDVec<T, SIMD_WIDTH> vec((__m512i) a_mask);
  // print("%x ", vec); puts("");
  kand_mask = ones & (a & b);
  if (!(kand_mask == kand(a_mask, b_mask)) ||
      (ones & (a & b)) != (kand(a_mask, b_mask))) {
    printf("ERROR: kand %lx %lx\n", (uinttest_t)kand_mask,
           (uinttest_t)kand(a_mask, b_mask));
  }
  kandn_mask = ones & ((~a) & b);
  if (!(kandn_mask == kandn(a_mask, b_mask)) ||
      (ones & ((~a) & b)) != (kandn(a_mask, b_mask))) {
    printf("ERROR: kandn %lx %lx\n", (uinttest_t)kandn_mask,
           (uinttest_t)kandn(a_mask, b_mask));
  }
  kor_mask = ones & (a | b);
  if (!(kor_mask == kor(a_mask, b_mask))) {
    printf("ERROR: kor %lx %lx\n", (uinttest_t)kor_mask,
           (uinttest_t)kor(a_mask, b_mask));
  }
  kxor_mask = ones & (a ^ b);
  if (!(kxor_mask == kxor(a_mask, b_mask))) {
    printf("ERROR: kxor %lx %lx\n", (uinttest_t)kxor_mask,
           (uinttest_t)kxor(a_mask, b_mask));
  }
  kxnor_mask = ones & (~(a ^ b));
  if (!(kxnor_mask == kxnor(a_mask, b_mask))) {
    printf("ERROR: kxnor %lx %lx\n", (uinttest_t)kxnor_mask,
           (uinttest_t)kxnor(a_mask, b_mask));
  }
  kadd_mask = ones & (a + b);
  if (!(kadd_mask == kadd(a_mask, b_mask))) {
    printf("ERROR: kadd %lx %lx\n", (uinttest_t)kadd_mask,
           (uinttest_t)kadd(a_mask, b_mask));
  }
  knot_mask = ones & (~a);
  if (!(knot_mask == knot(a_mask))) {
    printf("ERROR: knot %lx %lx\n", (uinttest_t)knot_mask,
           (uinttest_t)knot(a_mask));
  }
  for (i = 0; i < 64; i++) {
    kshiftli_mask = ones & (a << i);
    if (!(kshiftli_mask == kshiftli(a_mask, i))) {
      printf("ERROR: kshiftli, i=%lu, should be: %lx, is: %lx\n", i,
             (uinttest_t)kshiftli_mask, (uinttest_t)kshiftli(a_mask, i));
    }
  }
  for (i = 0; i < 64; i++) {
    kshiftri_mask = ones & (a >> i);
    if (!(kshiftri_mask == kshiftri(a_mask, i))) {
      printf("ERROR: kshiftri, i=%lu, should be: %lx, is: %lx\n", i,
             (uinttest_t)kshiftri_mask, (uinttest_t)kshiftri(a_mask, i));
    }
  }
  TEST_TEMPLATE_SHIFT(l, 0, <<)
  TEST_TEMPLATE_SHIFT(l, 1, <<)
  TEST_TEMPLATE_SHIFT(l, 2, <<)
  TEST_TEMPLATE_SHIFT(l, 3, <<)
  TEST_TEMPLATE_SHIFT(l, 4, <<)
  TEST_TEMPLATE_SHIFT(l, 5, <<)
  TEST_TEMPLATE_SHIFT(l, 6, <<)
  TEST_TEMPLATE_SHIFT(l, 7, <<)
  TEST_TEMPLATE_SHIFT(l, 8, <<)
  TEST_TEMPLATE_SHIFT(l, 9, <<)
  TEST_TEMPLATE_SHIFT(l, 10, <<)
  TEST_TEMPLATE_SHIFT(l, 11, <<)
  TEST_TEMPLATE_SHIFT(l, 12, <<)
  TEST_TEMPLATE_SHIFT(l, 13, <<)
  TEST_TEMPLATE_SHIFT(l, 14, <<)
  TEST_TEMPLATE_SHIFT(l, 15, <<)
  TEST_TEMPLATE_SHIFT(l, 16, <<)
  TEST_TEMPLATE_SHIFT(l, 17, <<)
  TEST_TEMPLATE_SHIFT(l, 18, <<)
  TEST_TEMPLATE_SHIFT(l, 19, <<)
  TEST_TEMPLATE_SHIFT(l, 20, <<)
  TEST_TEMPLATE_SHIFT(l, 21, <<)
  TEST_TEMPLATE_SHIFT(l, 22, <<)
  TEST_TEMPLATE_SHIFT(l, 23, <<)
  TEST_TEMPLATE_SHIFT(l, 24, <<)
  TEST_TEMPLATE_SHIFT(l, 25, <<)
  TEST_TEMPLATE_SHIFT(l, 26, <<)
  TEST_TEMPLATE_SHIFT(l, 27, <<)
  TEST_TEMPLATE_SHIFT(l, 28, <<)
  TEST_TEMPLATE_SHIFT(l, 29, <<)
  TEST_TEMPLATE_SHIFT(l, 30, <<)
  TEST_TEMPLATE_SHIFT(l, 31, <<)
  TEST_TEMPLATE_SHIFT(l, 32, <<)
  TEST_TEMPLATE_SHIFT(l, 33, <<)
  TEST_TEMPLATE_SHIFT(l, 34, <<)
  TEST_TEMPLATE_SHIFT(l, 35, <<)
  TEST_TEMPLATE_SHIFT(l, 36, <<)
  TEST_TEMPLATE_SHIFT(l, 37, <<)
  TEST_TEMPLATE_SHIFT(l, 38, <<)
  TEST_TEMPLATE_SHIFT(l, 39, <<)
  TEST_TEMPLATE_SHIFT(l, 40, <<)
  TEST_TEMPLATE_SHIFT(l, 41, <<)
  TEST_TEMPLATE_SHIFT(l, 42, <<)
  TEST_TEMPLATE_SHIFT(l, 43, <<)
  // ...
  TEST_TEMPLATE_SHIFT(r, 0, >>)
  TEST_TEMPLATE_SHIFT(r, 1, >>)
  TEST_TEMPLATE_SHIFT(r, 2, >>)
  TEST_TEMPLATE_SHIFT(r, 3, >>)
  TEST_TEMPLATE_SHIFT(r, 4, >>)
  TEST_TEMPLATE_SHIFT(r, 5, >>)
  TEST_TEMPLATE_SHIFT(r, 6, >>)
  TEST_TEMPLATE_SHIFT(r, 7, >>)
  TEST_TEMPLATE_SHIFT(r, 8, >>)
  TEST_TEMPLATE_SHIFT(r, 9, >>)
  TEST_TEMPLATE_SHIFT(r, 10, >>)
  TEST_TEMPLATE_SHIFT(r, 11, >>)
  TEST_TEMPLATE_SHIFT(r, 12, >>)
  TEST_TEMPLATE_SHIFT(r, 13, >>)
  TEST_TEMPLATE_SHIFT(r, 14, >>)
  TEST_TEMPLATE_SHIFT(r, 15, >>)
  TEST_TEMPLATE_SHIFT(r, 16, >>)
  TEST_TEMPLATE_SHIFT(r, 17, >>)
  TEST_TEMPLATE_SHIFT(r, 18, >>)
  TEST_TEMPLATE_SHIFT(r, 19, >>)
  TEST_TEMPLATE_SHIFT(r, 20, >>)
  TEST_TEMPLATE_SHIFT(r, 21, >>)
  TEST_TEMPLATE_SHIFT(r, 22, >>)
  TEST_TEMPLATE_SHIFT(r, 23, >>)
  TEST_TEMPLATE_SHIFT(r, 24, >>)
  TEST_TEMPLATE_SHIFT(r, 25, >>)
  TEST_TEMPLATE_SHIFT(r, 26, >>)
  TEST_TEMPLATE_SHIFT(r, 27, >>)
  TEST_TEMPLATE_SHIFT(r, 28, >>)
  TEST_TEMPLATE_SHIFT(r, 29, >>)
  TEST_TEMPLATE_SHIFT(r, 30, >>)
  TEST_TEMPLATE_SHIFT(r, 31, >>)
  TEST_TEMPLATE_SHIFT(r, 32, >>)
  TEST_TEMPLATE_SHIFT(r, 33, >>)
  TEST_TEMPLATE_SHIFT(r, 34, >>)
  TEST_TEMPLATE_SHIFT(r, 35, >>)
  TEST_TEMPLATE_SHIFT(r, 36, >>)
  TEST_TEMPLATE_SHIFT(r, 37, >>)
  TEST_TEMPLATE_SHIFT(r, 38, >>)
  TEST_TEMPLATE_SHIFT(r, 39, >>)
  TEST_TEMPLATE_SHIFT(r, 40, >>)
  TEST_TEMPLATE_SHIFT(r, 41, >>)
  TEST_TEMPLATE_SHIFT(r, 42, >>)
  TEST_TEMPLATE_SHIFT(r, 43, >>)
  // ...
}

void testsuite() {
  // TODO: tests cvts, set1
  /*SIMDMask<SIMDWord, 16> mask;
  mask=0x5b;
  print("%u ", SIMDVec<SIMDWord, 16>(mask)); puts("");
  printf("%x\n", _mm_movemask_epi8(mask));
  printf("%i %i %i %i %i %i %i %i\n", mask[0], mask[1], mask[2], mask[3],
  mask[4], mask[5], mask[6], mask[7]);*/
  // puts("Testsuite start");
  // SIMDVec<SIMDFloat, 64> test=getRandomVector<SIMDFloat, 64>();
  // print("%x ", test); puts("");
  //(0.0/1.0=ZE), (FLT_MAX+FLT_MAX=OE,PE), (FLT_MAX*FLT_MAX=OE,PE)
  //(1.0,1.0,-1.0) (0.0,1.0e30,-1.0)
  test_mask_functions<SIMDByte, NATIVE_SIMD_WIDTH>();
  test_mask_functions<SIMDSignedByte, NATIVE_SIMD_WIDTH>();
  test_mask_functions<SIMDWord, NATIVE_SIMD_WIDTH>();
  test_mask_functions<SIMDShort, NATIVE_SIMD_WIDTH>();
  test_mask_functions<SIMDInt, NATIVE_SIMD_WIDTH>();
  test_mask_functions<SIMDFloat, NATIVE_SIMD_WIDTH>();
  TEST_FUNCTION_LOAD_ALL_TYPES
  TEST_FUNCTION_STORE_ALL_TYPES
  // TEST_FUNCTION_ALL_TYPES(set1)
  TEST_FUNCTION_ALL_TYPES(add, +)
  TEST_FUNCTION_ALL_TYPES(adds, +)
  TEST_FUNCTION_ALL_TYPES(sub, -)
  TEST_FUNCTION_ALL_TYPES(subs, -)
  TEST_FUNCTION(mul, *, SIMDFloat, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(div, /, SIMDFloat, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION_SOP(ceil, ceilf, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(floor, floorf, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(round, dummy, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ") // roundf
  TEST_FUNCTION_SOP(truncate, truncf, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(rcp, dummy, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(rsqrt, dummy, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(sqrt, sqrtf, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(abs, abs, SIMDSignedByte, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(abs, abs, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(abs, abs, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(abs, fabsf, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_ALL_TYPES(and, &)
  TEST_FUNCTION_ALL_TYPES(or, |)
  TEST_FUNCTION_ALL_TYPES(andnot, &)
  TEST_FUNCTION_ALL_TYPES(xor, ^)
  TEST_FUNCTION_ALL_TYPES_SOP(not, dummy)
  TEST_FUNCTION_SOP(neg, -, SIMDSignedByte, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(neg, -, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(neg, -, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(neg, dummy, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ") //-
  TEST_FUNCTION_ALL_TYPES(min, dummy)
  TEST_FUNCTION_ALL_TYPES(max, dummy)
  TEST_FUNCTION_SOP(div2r0, dummy, SIMDByte, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_SOP(div2r0, dummy, SIMDWord, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_SOP(div2r0, dummy, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(div2r0, dummy, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(div2r0, dummy, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SOP(div2rd, dummy, SIMDByte, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_SOP(div2rd, dummy, SIMDWord, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_SOP(div2rd, dummy, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(div2rd, dummy, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_SOP(div2rd, dummy, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_SHIFT(srai, dummy, SIMDWord, NATIVE_SIMD_WIDTH, "%u ", 3)
  TEST_FUNCTION_SHIFT(srai, dummy, SIMDShort, NATIVE_SIMD_WIDTH, "%i ", 3)
  TEST_FUNCTION_SHIFT(srai, dummy, SIMDInt, NATIVE_SIMD_WIDTH, "%i ",
                      3) // srai not defined for SIMDByte and SIMDSignedByte
  TEST_FUNCTION_ALL_TYPES_SHIFT(srli, >>, 3)
  TEST_FUNCTION_ALL_TYPES_SHIFT(slli, <<, 3)
  TEST_FUNCTION(hadd, dummy, SIMDWord, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hadd, dummy, SIMDShort, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hadd, dummy, SIMDInt, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hadd, dummy, SIMDFloat, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hadds, dummy, SIMDShort, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hadds, dummy, SIMDInt, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hadds, dummy, SIMDFloat, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsub, dummy, SIMDWord, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsub, dummy, SIMDShort, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsub, dummy, SIMDInt, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsub, dummy, SIMDFloat, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsubs, dummy, SIMDShort, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsubs, dummy, SIMDInt, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION(hsubs, dummy, SIMDFloat, NATIVE_SIMD_WIDTH)
  TEST_FUNCTION_ALL_TYPES(avg, x)
  TEST_FUNCTION_MASK_TEST_ALL_(zeros, 0, SIMDByte, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_MASK_TEST_ALL_(zeros, 0, SIMDSignedByte, NATIVE_SIMD_WIDTH,
                               "%i ")
  TEST_FUNCTION_MASK_TEST_ALL_(zeros, 0, SIMDWord, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_MASK_TEST_ALL_(zeros, 0, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_MASK_TEST_ALL_(zeros, 0, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_MASK_TEST_ALL_(zeros, 0.0f, SIMDFloat, NATIVE_SIMD_WIDTH, "%g ")
  TEST_FUNCTION_MASK_TEST_ALL_(ones, 0xff, SIMDByte, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_MASK_TEST_ALL_(ones, -1, SIMDSignedByte, NATIVE_SIMD_WIDTH,
                               "%i ")
  TEST_FUNCTION_MASK_TEST_ALL_(ones, 0xffff, SIMDWord, NATIVE_SIMD_WIDTH, "%u ")
  TEST_FUNCTION_MASK_TEST_ALL_(ones, -1, SIMDShort, NATIVE_SIMD_WIDTH, "%i ")
  TEST_FUNCTION_MASK_TEST_ALL_(ones, -1, SIMDInt, NATIVE_SIMD_WIDTH, "%i ")
  // TEST_FUNCTION_TEST_ALL(ones, , SIMDFloat, NATIVE_SIMD_WIDTH, "%g ") TODO
  TEST_FUNCTION_ALL_TYPES_CMP(cmplt, <)
  TEST_FUNCTION_ALL_TYPES_CMP(cmple, <=)
  TEST_FUNCTION_ALL_TYPES_CMP(cmpgt, >)
  TEST_FUNCTION_ALL_TYPES_CMP(cmpge, >=)
  TEST_FUNCTION_ALL_TYPES_CMP(cmpeq, ==)
  // TEST_FUNCTION_ALL_TYPES_CMP(cmpneq, !=)
  test_cvts<SIMDInt, SIMDFloat, NATIVE_SIMD_WIDTH>();
  test_cvts<SIMDFloat, SIMDInt, NATIVE_SIMD_WIDTH>();

  // printf("Testsuite done");
}

int main() {
  // printf("RAND_MAX=%u\n", RAND_MAX);
  srand(time(NULL));
  /*SIMDVec<SIMDShort, 64> a=getRandomVector<SIMDShort, 64>(),
  b=getRandomVector<SIMDShort, 64>(); uinttest_t mask=random64();
  printf("%lx\n", mask);
  SIMDMask<SIMDShort, 64> k=mask;
  print("%u ", SIMDVec<SIMDShort, 64>(_mm512_maskz_sub_epi16(mask, a, b)));
  puts(""); print("%u ", mask_ifelsezero(k, sub (a, b))); puts("");*/
  /*__m512i a=getRandomVector<SIMDSignedByte, NATIVE_SIMD_WIDTH>();
  __mmask64 k=random64();
  print("%i ", SIMDVec<SIMDSignedByte, NATIVE_SIMD_WIDTH>(a)); puts("");
  printf("%x \n", k);
  print("%i ", SIMDVec<SIMDSignedByte, NATIVE_SIMD_WIDTH>(_mm512_maskz_abs_epi8
  (k, a))); puts(""); exit(0);*/
  unsigned int i;
  for (i = 0; i < 10000; i++) {
    testsuite();
  }
  // benchmark<SIMDInt, 16>();
  exit(EXIT_SUCCESS);
}

#endif
