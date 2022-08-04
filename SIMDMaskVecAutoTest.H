#ifndef _SIMD_MASK_VEC_AUTO_TEST_H_
#define _SIMD_MASK_VEC_AUTO_TEST_H_

#include "SIMDMaskVecWrapper.H"
#include "SIMDSerialMaskVec.H"
#include "SIMDVecAutoTest.H"
#include <bitset>
#include <iostream>
#include <string>

namespace ns_simd {

#define PRINT_VEC(T, VEC) PRINT(SIMDTypeInfo<T>::defaultFormat(), VEC)
#define PRINT_SERIAL_MASK(MASK)                                                \
  {                                                                            \
    printf(#MASK "\t= ");                                                      \
    fflush(stdout);                                                            \
    std::string s = std::bitset<MASK.bits>(MASK).to_string();                  \
    std::cout << std::string(s.rbegin(), s.rend()) << std::endl << std::flush; \
  }
#define PRINT_SCALAR(T, SCALAR)                                                \
  PRINTVAL(SIMDTypeInfo<T>::defaultFormat(), SCALAR)

// =============================================================================
// test for masked functions with a vector src, a mask and one regular vector
// argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskUnary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src, a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a mask and one regular vector argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskZUnary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and two regular vector
// arguments
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskBinary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src, a, b;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a, b);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a, b);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a mask and two regular vector arguments
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskZBinary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a, b);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a, b);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with one mask and two vector arguments
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskConditionBinary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a, b);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a, b);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with one mask and one vector argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskConditionUnary {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and one regular vector
// argument and one type template parameter
// =============================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT>
struct TesterMaskUnaryTemplateType {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<Tout, SIMD_WIDTH> src;
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(src);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<Tout, SIMD_WIDTH> cs =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<Tout, SIMD_WIDTH> cp =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k,
                                                                      a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(Tout, src);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(Tout, cs);
          PRINT_VEC(Tout, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a mask and one regular vector argument
// and one type template parameter
// =============================================================================

template <typename Tout, typename T, int SIMD_WIDTH,
          template <typename, typename, int> class FCT>
struct TesterMaskZUnaryTemplateType {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<Tout, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<Tout, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<Tout, SIMD_WIDTH> cs =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<Tout, SIMD_WIDTH> cp =
          FCT<Tout, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(Tout, cs);
          PRINT_VEC(Tout, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a vector src, a mask and one scalar argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskScalar {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,s";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(src);
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      T a = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(k);
          PRINT_SCALAR(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a mask and one scalar argument
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskZScalar {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,s";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      T a = FCT<T, SIMD_WIDTH>::randomizeInputScalar();
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_SCALAR(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with a mask and two vector arguments that return
// a mask
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskCmp {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      SIMDSerialVec<T, SIMD_WIDTH> a, b;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      FCT<T, SIMD_WIDTH>::randomizeInput(b);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a, b);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a, b);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, b);
          PRINT_SERIAL_MASK(cs);
          PRINT_SERIAL_MASK(cp);
        }
        EXIT;
      }
    }
  }
};

// =============================================================================
// test for masked functions with a mask and one vector argument that returns
// an int
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterMaskUnaryInt {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      SIMDSerialVec<T, SIMD_WIDTH> a;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(a);
      int cs = FCT<T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      int cp = FCT<T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_SCALAR(int, cs);
          PRINT_SCALAR(int, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for unary mask functions
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterUnaryK {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialMask>(k);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDMask>(k);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_SERIAL_MASK(cs);
          PRINT_SERIAL_MASK(cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for binary mask functions
// =============================================================================

template <typename T, int SIMD_WIDTH, template <typename, int> class FCT>
struct TesterBinaryK {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,m";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k, l;
      FCT<T, SIMD_WIDTH>::randomizeInput(k);
      FCT<T, SIMD_WIDTH>::randomizeInput(l);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<T, SIMD_WIDTH>::template apply<SIMDSerialMask>(k, l);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<T, SIMD_WIDTH>::template apply<SIMDMask>(k, l);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_SERIAL_MASK(l);
          PRINT_SERIAL_MASK(cs);
          PRINT_SERIAL_MASK(cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);
  }
};

// =============================================================================
// test for masked functions with src vector, a mask and one vector argument
// with one int template parameter
// =============================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT>
struct TesterMaskUnaryTI {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "v,m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> src, a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(src);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(src, k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(src, k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_VEC(T, src);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    TesterMaskUnaryTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                      FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT>
struct TesterMaskUnaryTI<T, SIMD_WIDTH, IMAX, IMAX, FCT> {
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT>
using TesterMaskUnaryTIMinToMax =
    TesterMaskUnaryTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                      FCT<0, T, SIMD_WIDTH>::maxI, FCT>;

// =============================================================================
// test for masked functions with a mask and one vector argument
// with one int template parameter
// =============================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT>
struct TesterMaskZUnaryTI {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m,v";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialVec<T, SIMD_WIDTH> a;
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(a);
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialVec<T, SIMD_WIDTH> cs =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDSerialVec>(k, a);
      SIMDSerialVec<T, SIMD_WIDTH> cp =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDVec, SIMDMask>(k, a);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_VEC(T, a);
          PRINT_VEC(T, cs);
          PRINT_VEC(T, cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    TesterMaskZUnaryTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                       FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT>
struct TesterMaskZUnaryTI<T, SIMD_WIDTH, IMAX, IMAX, FCT> {
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT>
using TesterMaskZUnaryTIMinToMax =
    TesterMaskZUnaryTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                       FCT<0, T, SIMD_WIDTH>::maxI, FCT>;

// =============================================================================
// test for unary mask functions with one int template parameter
// =============================================================================

template <typename T, int SIMD_WIDTH, int I, int IMAX,
          template <int, typename, int> class FCT>
struct TesterUnaryKTI {
  static void test(int repeats, const std::string &pattern) {
    std::string name = FCT<I, T, SIMD_WIDTH>::name();
    if (name.find(pattern) == std::string::npos) {
      return;
    }
    std::string args = "m";
    printInfo(name, args);
    int errors = 0, trials = 0;
    for (int i = 0; i < repeats; i++, trials++) {
      SIMDSerialMask<T, SIMD_WIDTH> k;
      FCT<I, T, SIMD_WIDTH>::randomizeInput(k);
      SIMDSerialMask<T, SIMD_WIDTH> cs =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDSerialMask>(k);
      SIMDSerialMask<T, SIMD_WIDTH> cp =
          FCT<I, T, SIMD_WIDTH>::template apply<SIMDMask>(k);
      if (cs != cp) {
        errors++;
        if (errors <= PRINT_ERRORS) {
          printError(name, args);
          PRINT_SERIAL_MASK(k);
          PRINT_SERIAL_MASK(cs);
          PRINT_SERIAL_MASK(cp);
        }
        EXIT;
      }
    }
    printErrorStats(errors, trials);

    TesterUnaryKTI<T, SIMD_WIDTH, FCT<I, T, SIMD_WIDTH>::nextI, IMAX,
                   FCT>::test(repeats, pattern);
  }
};

// end of recursion at I = IMAX
template <typename T, int SIMD_WIDTH, int IMAX,
          template <int, typename, int> class FCT>
struct TesterUnaryKTI<T, SIMD_WIDTH, IMAX, IMAX, FCT> {
  static void test(int, const std::string &) {}
};

template <typename T, int SIMD_WIDTH, template <int, typename, int> class FCT>
using TesterUnaryKTIMinToMax =
    TesterUnaryKTI<T, SIMD_WIDTH, FCT<0, T, SIMD_WIDTH>::minI,
                   FCT<0, T, SIMD_WIDTH>::maxI, FCT>;

} // namespace ns_simd

#endif // _SIMD_MASK_VEC_AUTO_TEST_H_
