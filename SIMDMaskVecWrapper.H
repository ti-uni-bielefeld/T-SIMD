// ===========================================================================
// 
// SIMDMaskVecWrapper.H --
// wrapper templates for autotest of masked operations, Jonas Keller
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

#ifndef _SIMD_MASK_VEC_WRAPPER_H_
#define _SIMD_MASK_VEC_WRAPPER_H_

#include "SIMDSerialMaskVec.H"
#include "SIMDSerialVec.H"
#include "SIMDVecWrapper.H"
#include <iostream>
#include <string>

namespace ns_simd {

template <typename T, int SIMD_WIDTH> struct Mask_ifelse {
  static std::string name() { return t2s<T, SIMD_WIDTH>("mask_ifelse"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static Vec<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &cond,
                                  const Vec<T, SIMD_WIDTH> &trueVal,
                                  const Vec<T, SIMD_WIDTH> &falseVal) {
    return mask_ifelse(cond, trueVal, falseVal);
  }
};

template <typename T, int SIMD_WIDTH> struct Mask_ifelsezero {
  static std::string name() { return t2s<T, SIMD_WIDTH>("mask_ifelsezero"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static Vec<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &cond,
                                  const Vec<T, SIMD_WIDTH> &trueVal) {
    return mask_ifelsezero(cond, trueVal);
  }
};

template <typename Tout, typename T, int SIMD_WIDTH> struct Mask_cvts {
  static std::string name() { return tt2s<Tout, T, SIMD_WIDTH>("mask_cvts"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  static void randomizeInput(SIMDSerialVec<Tout, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static Vec<Tout, SIMD_WIDTH> apply(const Vec<Tout, SIMD_WIDTH> &src,
                                     const Mask<T, SIMD_WIDTH> &k,
                                     const Vec<T, SIMD_WIDTH> &a) {
    return mask_cvts(src, k, a);
  }
};

template <typename Tout, typename T, int SIMD_WIDTH> struct MaskZ_cvts {
  static std::string name() { return tt2s<Tout, T, SIMD_WIDTH>("maskz_cvts"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static Vec<Tout, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &k,
                                     const Vec<T, SIMD_WIDTH> &a) {
    return maskz_cvts(k, a);
  }
};

template <typename T, int SIMD_WIDTH> struct Mask_set1 {
  static std::string name() { return t2s<T, SIMD_WIDTH>("mask_set1"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  static T randomizeInputScalar() { return getRandom<T>(); }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &src,
                                  const Mask<T, SIMD_WIDTH> &k, const T a) {
    return mask_set1(src, k, a);
  }
};

template <typename T, int SIMD_WIDTH> struct MaskZ_set1 {
  static std::string name() { return t2s<T, SIMD_WIDTH>("maskz_set1"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static T randomizeInputScalar() { return getRandom<T>(); }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static Vec<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &k, const T a) {
    return maskz_set1(k, a);
  }
};

// TODO: load/store ?

#define GENERATE_WRAPPER_MASK_UNARY(OP)                                        \
  template <typename T, int SIMD_WIDTH> struct Mask_##OP {                     \
    static std::string name() { return t2s<T, SIMD_WIDTH>("mask_" #OP); }      \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &src,             \
                                    const Mask<T, SIMD_WIDTH> &k,              \
                                    const Vec<T, SIMD_WIDTH> &a) {             \
      return mask_##OP(src, k, a);                                             \
    }                                                                          \
  };

#define GENERATE_WRAPPER_MASKZ_UNARY(OP)                                       \
  template <typename T, int SIMD_WIDTH> struct MaskZ_##OP {                    \
    static std::string name() { return t2s<T, SIMD_WIDTH>("maskz_" #OP); }     \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Vec<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &k,              \
                                    const Vec<T, SIMD_WIDTH> &a) {             \
      return maskz_##OP(k, a);                                                 \
    }                                                                          \
  };

#define GENERATE_WRAPPER_MASK_MASKZ_UNARY(OP)                                  \
  GENERATE_WRAPPER_MASK_UNARY(OP)                                              \
  GENERATE_WRAPPER_MASKZ_UNARY(OP)

#define GENERATE_WRAPPER_MASK_SHIFT(OP)                                        \
  template <int IMM, typename T, int SIMD_WIDTH> struct Mask_##OP {            \
    static std::string name() {                                                \
      return ti2s<IMM, T, SIMD_WIDTH>("mask_" #OP);                            \
    }                                                                          \
    static SIMD_CONSTEXPR int minI = 0;                                             \
    static SIMD_CONSTEXPR int maxI = sizeof(T) * 8 + 1;                             \
    static SIMD_CONSTEXPR int nextI = IMM + 1;                                      \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &src,             \
                                    const Mask<T, SIMD_WIDTH> &k,              \
                                    const Vec<T, SIMD_WIDTH> &a) {             \
      return mask_##OP<IMM>(src, k, a);                                        \
    }                                                                          \
  };

#define GENERATE_WRAPPER_MASKZ_SHIFT(OP)                                       \
  template <int IMM, typename T, int SIMD_WIDTH> struct MaskZ_##OP {           \
    static std::string name() {                                                \
      return ti2s<IMM, T, SIMD_WIDTH>("maskz_" #OP);                           \
    }                                                                          \
    static SIMD_CONSTEXPR int minI = 0;                                             \
    static SIMD_CONSTEXPR int maxI = sizeof(T) * 8 + 1;                             \
    static SIMD_CONSTEXPR int nextI = IMM + 1;                                      \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Vec<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &k,              \
                                    const Vec<T, SIMD_WIDTH> &a) {             \
      return maskz_##OP<IMM>(k, a);                                            \
    }                                                                          \
  };

#define GENERATE_WRAPPER_MASK_MASKZ_SHIFT(OP)                                  \
  GENERATE_WRAPPER_MASK_SHIFT(OP)                                              \
  GENERATE_WRAPPER_MASKZ_SHIFT(OP)

#define GENERATE_WRAPPER_MASK_BINARY(OP)                                       \
  template <typename T, int SIMD_WIDTH> struct Mask_##OP {                     \
    static std::string name() { return t2s<T, SIMD_WIDTH>("mask_" #OP); }      \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Vec<T, SIMD_WIDTH>                                                  \
    apply(const Vec<T, SIMD_WIDTH> &src, const Mask<T, SIMD_WIDTH> &k,         \
          const Vec<T, SIMD_WIDTH> &a, const Vec<T, SIMD_WIDTH> &b) {          \
      return mask_##OP(src, k, a, b);                                          \
    }                                                                          \
  };

#define GENERATE_WRAPPER_MASKZ_BINARY(OP)                                      \
  template <typename T, int SIMD_WIDTH> struct MaskZ_##OP {                    \
    static std::string name() { return t2s<T, SIMD_WIDTH>("maskz_" #OP); }     \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Vec<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &k,              \
                                    const Vec<T, SIMD_WIDTH> &a,               \
                                    const Vec<T, SIMD_WIDTH> &b) {             \
      return maskz_##OP(k, a, b);                                              \
    }                                                                          \
  };

#define GENERATE_WRAPPER_MASK_MASKZ_BINARY(OP)                                 \
  GENERATE_WRAPPER_MASK_BINARY(OP)                                             \
  GENERATE_WRAPPER_MASKZ_BINARY(OP)

GENERATE_WRAPPER_MASK_MASKZ_BINARY(add)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(adds)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(sub)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(subs)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(mul)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(div)

GENERATE_WRAPPER_MASK_MASKZ_UNARY(ceil)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(floor)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(round)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(truncate)

// 02. Oct 22 (Jonas Keller): added rcp and rsqrt
GENERATE_WRAPPER_MASK_MASKZ_UNARY(rcp)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(rsqrt)

GENERATE_WRAPPER_MASK_MASKZ_UNARY(sqrt)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(abs)

GENERATE_WRAPPER_MASK_MASKZ_BINARY(and)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(or)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(andnot)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(xor)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(not )

GENERATE_WRAPPER_MASK_MASKZ_UNARY(neg)

GENERATE_WRAPPER_MASK_MASKZ_BINARY(min)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(max)

GENERATE_WRAPPER_MASK_MASKZ_UNARY(div2r0)
GENERATE_WRAPPER_MASK_MASKZ_UNARY(div2rd)

GENERATE_WRAPPER_MASK_MASKZ_SHIFT(srai)
GENERATE_WRAPPER_MASK_MASKZ_SHIFT(srli)
GENERATE_WRAPPER_MASK_MASKZ_SHIFT(slli)

GENERATE_WRAPPER_MASK_MASKZ_BINARY(hadd)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(hadds)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(hsub)
GENERATE_WRAPPER_MASK_MASKZ_BINARY(hsubs)


// 16. Oct 22 (Jonas Keller): added overloaded versions of mask_cmp* functions
// that only take two vector parameters and no mask parameter

#define GENERATE_WRAPPER_MASK_CMP(OP)                                          \
  template <typename T, int SIMD_WIDTH> struct Mask_##OP {                     \
    static std::string name() { return t2s<T, SIMD_WIDTH>("mask_" #OP); }      \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {            \
      vec.randomize();                                                         \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Mask<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &k,             \
                                     const Vec<T, SIMD_WIDTH> &a,              \
                                     const Vec<T, SIMD_WIDTH> &b) {            \
      return mask_##OP(k, a, b);                                               \
    }                                                                          \
    template <template <typename, int> class Vec,                              \
              template <typename, int> class Mask>                             \
    static Mask<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &a,              \
                                     const Vec<T, SIMD_WIDTH> &b) {            \
      return mask_##OP(a, b);                                                  \
    }                                                                          \
  };

GENERATE_WRAPPER_MASK_CMP(cmplt)
GENERATE_WRAPPER_MASK_CMP(cmple)
GENERATE_WRAPPER_MASK_CMP(cmpeq)
GENERATE_WRAPPER_MASK_CMP(cmpgt)
GENERATE_WRAPPER_MASK_CMP(cmpge)
GENERATE_WRAPPER_MASK_CMP(cmpneq)

GENERATE_WRAPPER_MASK_MASKZ_BINARY(avg)

template <typename T, int SIMD_WIDTH> struct Mask_test_all_zeros {
  static std::string name() {
    return t2s<T, SIMD_WIDTH>("mask_test_all_zeros");
  }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static int apply(const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a) {
    return mask_test_all_zeros(k, a);
  }
};

template <typename T, int SIMD_WIDTH> struct Mask_test_all_ones {
  static std::string name() { return t2s<T, SIMD_WIDTH>("mask_test_all_ones"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  static void randomizeInput(SIMDSerialVec<T, SIMD_WIDTH> &vec) {
    vec.randomize();
  }
  template <template <typename, int> class Vec,
            template <typename, int> class Mask>
  static int apply(const Mask<T, SIMD_WIDTH> &k, const Vec<T, SIMD_WIDTH> &a) {
    return mask_test_all_ones(k, a);
  }
};

#define GENERATE_WRAPPER_BINARY_K(OP_STRUCT, OP)                               \
  template <typename T, int SIMD_WIDTH> struct OP_STRUCT {                     \
    static std::string name() { return t2s<T, SIMD_WIDTH>(#OP); }              \
    static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {          \
      mask.randomize();                                                        \
    }                                                                          \
    template <template <typename, int> class Mask>                             \
    static Mask<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &a,             \
                                     const Mask<T, SIMD_WIDTH> &b) {           \
      return OP(a, b);                                                         \
    }                                                                          \
  };

GENERATE_WRAPPER_BINARY_K(Kand, kand)
GENERATE_WRAPPER_BINARY_K(Kandn, kandn)
GENERATE_WRAPPER_BINARY_K(Kor, kor)
GENERATE_WRAPPER_BINARY_K(Kxor, kxor)
GENERATE_WRAPPER_BINARY_K(Kxnor, kxnor)
GENERATE_WRAPPER_BINARY_K(Kadd, kadd)

template <typename T, int SIMD_WIDTH> struct Knot {
  static std::string name() { return t2s<T, SIMD_WIDTH>("knot"); }
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  template <template <typename, int> class Mask>
  static Mask<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &a) {
    return knot(a);
  }
};

template <int IMM, typename T, int SIMD_WIDTH> struct Kshiftli {
  static std::string name() { return ti2s<IMM, T, SIMD_WIDTH>("kshiftli"); }
  static SIMD_CONSTEXPR int minI = 0;
  static SIMD_CONSTEXPR int maxI = SIMDSerialMask<T, SIMD_WIDTH>::bits + 1;
  static SIMD_CONSTEXPR int nextI = IMM + 1;
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  template <template <typename, int> class Mask>
  static Mask<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &a) {
    return kshiftli<IMM>(a);
  }
};

template <int IMM, typename T, int SIMD_WIDTH> struct Kshiftri {
  static std::string name() { return ti2s<IMM, T, SIMD_WIDTH>("kshiftri"); }
  static SIMD_CONSTEXPR int minI = 0;
  static SIMD_CONSTEXPR int maxI = SIMDSerialMask<T, SIMD_WIDTH>::bits + 1;
  static SIMD_CONSTEXPR int nextI = IMM + 1;
  static void randomizeInput(SIMDSerialMask<T, SIMD_WIDTH> &mask) {
    mask.randomize();
  }
  template <template <typename, int> class Mask>
  static Mask<T, SIMD_WIDTH> apply(const Mask<T, SIMD_WIDTH> &a) {
    return kshiftri<IMM>(a);
  }
};

} // namespace ns_simd

#endif // _SIMD_MASK_VEC_WRAPPER_H_
