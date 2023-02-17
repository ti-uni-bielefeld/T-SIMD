// ===========================================================================
//
// SIMDVecAutoTestSerialMask.H --
// serial implementations of masked operations for autotest, Jonas Keller
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

#ifndef _SIMD_VEC_AUTO_TEST_SERIAL_MASK_H_
#define _SIMD_VEC_AUTO_TEST_SERIAL_MASK_H_

#include "SIMDVecAutoTestSerial.H"
#include "SIMDVecExt.H"
#include "SIMDVecMask.H"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

namespace simd {
namespace auto_test {

template <int Bits>
struct _UInt;
template <>
struct _UInt<4>
{
  typedef uint8_t type;
};
template <>
struct _UInt<8>
{
  typedef uint8_t type;
};
template <>
struct _UInt<16>
{
  typedef uint16_t type;
};
template <>
struct _UInt<32>
{
  typedef uint32_t type;
};
template <>
struct _UInt<64>
{
  typedef uint64_t type;
};

template <int Bits>
struct UInt
{
  typename _UInt<Bits>::type value;
  UInt() {}
  UInt(const typename _UInt<Bits>::type &v) : value(maskValue(v)) {}
  operator typename _UInt<Bits>::type() const { return value; }

private:
  typename _UInt<Bits>::type maskValue(typename _UInt<Bits>::type x) const
  {
    assert(Bits >= 4 && Bits <= 64);
    if (Bits == 4) { return x & 0x0f; }
    return x;
  }
};

template <int Bits>
UInt<Bits> getRandomUInt()
{
  return getRandomUInt<Bits / 2>() |
         ((UInt<Bits>) getRandomUInt<Bits / 2>() << (Bits / 2));
}
template <>
UInt<4> getRandomUInt<4>()
{
  return rand() & 0xF;
}

// 26. Nov 22 (Jonas Keller): Removed implicit conversion from SerialMask to
// Mask and vice versa and added explicit conversion functions instead.
// This is to make sure the right functions are called when using SerialVec
// in the autotests.

template <typename T, int SIMD_WIDTH>
class SerialMask
{
public:
  static constexpr int bits = Vec<T, SIMD_WIDTH>::elems;
  UInt<bits> mask;

  SerialMask() : mask(0) {}
  SerialMask(const UInt<bits> &_mask) : mask(_mask) {}
  SerialMask(const typename _UInt<bits>::type &_mask) : mask(_mask) {}
  SerialMask(const SerialVec<T, SIMD_WIDTH> &x)
  {
    mask = 0;
    // avoids comparison with TypeInfo::trueval, which fails for
    // Float (NaN)
    SerialVec<T, SIMD_WIDTH> negX = not_(x);
    for (int i = 0; i < x.elements; i++) {
      if (negX[i] == T(0)) { mask = mask | (UInt<bits>(1) << i); }
    }
  }

  operator UInt<bits>() const { return mask; }
  operator typename _UInt<bits>::type() const { return mask; }

  bool operator[](const uint8_t i) const
  {
    if (i >= bits) { return false; }
    return (mask & (UInt<bits>(1) << i)) != 0;
  }

  void setTrue(const uint8_t i)
  {
    if (i < bits) { mask = mask | (UInt<bits>(1) << i); }
  }

  void randomize()
  {
    // randomize with a 1 in 16 chance to be all 1s or all 0s
    // to make sure these cases are also tested
    if ((rand() & 0x0f) == 0) {
      if (rand() & 0x01) {
        mask = 0;
      } else {
        mask = ~UInt<bits>(0);
      }
    } else {
      mask = getRandomUInt<bits>();
    }
  }

  Mask<T, SIMD_WIDTH> getMask() const { return Mask<T, SIMD_WIDTH>(mask); }
  void setMask(const Mask<T, SIMD_WIDTH> &x) { mask = x; }
  static SerialMask<T, SIMD_WIDTH> fromMask(const Mask<T, SIMD_WIDTH> &x)
  {
    return SerialMask<T, SIMD_WIDTH>(uint64_t(x));
  }
};

// ===========================================================================
// wrapper for template functions without vector arguments
// ===========================================================================

// 28. Dec 22 (Jonas Keller): Added MaskDeductor

// 30. Jan 23 (Jonas Keller): removed setTrueLeft/Right and replaced them with
// mask_set_true/false_low/high.

template <template <typename, int> class MASK>
struct MaskDeductor;

template <>
struct MaskDeductor<Mask>
{
  template <typename T, int SIMD_WIDTH>
  static Mask<T, SIMD_WIDTH> _mask_set_true_low(const unsigned int x)
  {
    return mask_set_true_low<T, SIMD_WIDTH>(x);
  }

  template <typename T, int SIMD_WIDTH>
  static Mask<T, SIMD_WIDTH> _mask_set_true_high(const unsigned int x)
  {
    return mask_set_true_high<T, SIMD_WIDTH>(x);
  }

  template <typename T, int SIMD_WIDTH>
  static Mask<T, SIMD_WIDTH> _mask_set_false_low(const unsigned int x)
  {
    return mask_set_false_low<T, SIMD_WIDTH>(x);
  }

  template <typename T, int SIMD_WIDTH>
  static Mask<T, SIMD_WIDTH> _mask_set_false_high(const unsigned int x)
  {
    return mask_set_false_high<T, SIMD_WIDTH>(x);
  }
};

template <>
struct MaskDeductor<SerialMask>
{
  template <typename T, int SIMD_WIDTH>
  static SerialMask<T, SIMD_WIDTH> _mask_set_true_low(const unsigned int x)
  {
    if (x >= SerialMask<T, SIMD_WIDTH>::bits) {
      return SerialMask<T, SIMD_WIDTH>(
        ~UInt<SerialMask<T, SIMD_WIDTH>::bits>(0));
    }
    SerialMask<T, SIMD_WIDTH> result;
    for (unsigned int i = 0; i < x; i++) { result.setTrue(i); }
    return result;
  }

  template <typename T, int SIMD_WIDTH>
  static SerialMask<T, SIMD_WIDTH> _mask_set_true_high(const unsigned int x)
  {
    if (x >= SerialMask<T, SIMD_WIDTH>::bits) {
      return SerialMask<T, SIMD_WIDTH>(
        ~UInt<SerialMask<T, SIMD_WIDTH>::bits>(0));
    }
    SerialMask<T, SIMD_WIDTH> result;
    for (unsigned int i = SerialMask<T, SIMD_WIDTH>::bits - x;
         i < SerialMask<T, SIMD_WIDTH>::bits; i++) {
      result.setTrue(i);
    }
    return result;
  }

  template <typename T, int SIMD_WIDTH>
  static SerialMask<T, SIMD_WIDTH> _mask_set_false_low(const unsigned int x)
  {
    return ~_mask_set_true_low<T, SIMD_WIDTH>(x);
  }

  template <typename T, int SIMD_WIDTH>
  static SerialMask<T, SIMD_WIDTH> _mask_set_false_high(const unsigned int x)
  {
    return ~_mask_set_true_high<T, SIMD_WIDTH>(x);
  }
};

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> mask_ifelse(
  const SerialMask<T, SIMD_WIDTH> &cond,
  const SerialVec<T, SIMD_WIDTH> &trueVal,
  const SerialVec<T, SIMD_WIDTH> &falseVal)
{
  SerialVec<T, SIMD_WIDTH> result;
  for (int i = 0; i < result.elements; i++) {
    result[i] = cond[i] ? trueVal[i] : falseVal[i];
  }
  return result;
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> mask_ifelsezero(
  const SerialMask<T, SIMD_WIDTH> &cond,
  const SerialVec<T, SIMD_WIDTH> &trueVal)
{
  return mask_ifelse(cond, trueVal,
                     Deductor<SerialVec>::_setzero<T, SIMD_WIDTH>());
}

template <typename Tout, typename T, int SIMD_WIDTH>
static SerialVec<Tout, SIMD_WIDTH> mask_cvts(
  const SerialVec<Tout, SIMD_WIDTH> &src, const SerialMask<T, SIMD_WIDTH> &k,
  const SerialVec<T, SIMD_WIDTH> &a)
{
  return mask_ifelse(SerialMask<Tout, SIMD_WIDTH>(k.mask), cvts(a), src);
}

template <typename Tout, typename T, int SIMD_WIDTH>
static SerialVec<Tout, SIMD_WIDTH> maskz_cvts(
  const SerialMask<T, SIMD_WIDTH> &k, const SerialVec<T, SIMD_WIDTH> &a)
{
  return mask_cvts(Deductor<SerialVec>::_setzero<Tout, SIMD_WIDTH>(), k, a);
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> mask_set1(const SerialVec<T, SIMD_WIDTH> &src,
                                          const SerialMask<T, SIMD_WIDTH> &k,
                                          const T a)
{
  return mask_ifelse(k, Deductor<SerialVec>::_set1<T, SIMD_WIDTH>(a), src);
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> maskz_set1(const SerialMask<T, SIMD_WIDTH> &k,
                                           const T a)
{
  return mask_set1(Deductor<SerialVec>::_setzero<T, SIMD_WIDTH>(), k, a);
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> mask_loadu(const SerialVec<T, SIMD_WIDTH> &src,
                                           const SerialMask<T, SIMD_WIDTH> &k,
                                           const T *p)
{
  SerialVec<T, SIMD_WIDTH> result;
  for (int i = 0; i < result.elements; i++) {
    result[i] = k[i] ? p[i] : src[i];
  }
  return result;
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> maskz_loadu(const SerialMask<T, SIMD_WIDTH> &k,
                                            const T *p)
{
  return mask_loadu(Deductor<SerialVec>::_setzero<T, SIMD_WIDTH>(), k, p);
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> mask_load(const SerialVec<T, SIMD_WIDTH> &src,
                                          const SerialMask<T, SIMD_WIDTH> &k,
                                          const T *p)
{
  // TODO: make sure p is aligned?
  return mask_loadu(src, k, p);
}

template <typename T, int SIMD_WIDTH>
static SerialVec<T, SIMD_WIDTH> maskz_load(const SerialMask<T, SIMD_WIDTH> &k,
                                           const T *p)
{
  // TODO: make sure p is aligned?
  return maskz_loadu(k, p);
}

template <typename T, int SIMD_WIDTH>
static void mask_storeu(T *const p, const SerialMask<T, 64> &k,
                        const SerialVec<T, 64> &a)
{
  for (int i = 0; i < a.elements; i++) {
    if (k[i]) { p[i] = a[i]; }
  }
}

template <typename T, int SIMD_WIDTH>
static void mask_store(T *const p, const SerialMask<T, 64> &k,
                       const SerialVec<T, 64> &a)
{
  // TODO: make sure p is aligned?
  mask_storeu(p, k, a);
}

#define GENERATE_SERIAL_MASK_UNARY_NAME(OP, OP_NAME)                           \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialVec<T, SIMD_WIDTH> mask_##OP_NAME(                              \
    const SerialVec<T, SIMD_WIDTH> &src, const SerialMask<T, SIMD_WIDTH> &k,   \
    const SerialVec<T, SIMD_WIDTH> &a)                                         \
  {                                                                            \
    return mask_ifelse(k, OP(a), src);                                         \
  }

#define GENERATE_SERIAL_MASK_UNARY(OP) GENERATE_SERIAL_MASK_UNARY_NAME(OP, OP)

#define GENERATE_SERIAL_MASKZ_UNARY_NAME(OP, OP_NAME)                          \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialVec<T, SIMD_WIDTH> maskz_##OP_NAME(                             \
    const SerialMask<T, SIMD_WIDTH> &k, const SerialVec<T, SIMD_WIDTH> &a)     \
  {                                                                            \
    return mask_ifelsezero(k, OP(a));                                          \
  }

#define GENERATE_SERIAL_MASKZ_UNARY(OP) GENERATE_SERIAL_MASKZ_UNARY_NAME(OP, OP)

#define GENERATE_SERIAL_MASK_MASKZ_UNARY_NAME(OP, OP_NAME)                     \
  GENERATE_SERIAL_MASK_UNARY_NAME(OP, OP_NAME)                                 \
  GENERATE_SERIAL_MASKZ_UNARY_NAME(OP, OP_NAME)

#define GENERATE_SERIAL_MASK_MASKZ_UNARY(OP)                                   \
  GENERATE_SERIAL_MASK_UNARY(OP)                                               \
  GENERATE_SERIAL_MASKZ_UNARY(OP)

#define GENERATE_SERIAL_MASK_UNARY_TI(OP)                                      \
  template <int IMM, typename T, int SIMD_WIDTH>                               \
  static SerialVec<T, SIMD_WIDTH> mask_##OP(                                   \
    const SerialVec<T, SIMD_WIDTH> &src, const SerialMask<T, SIMD_WIDTH> &k,   \
    const SerialVec<T, SIMD_WIDTH> &a)                                         \
  {                                                                            \
    return mask_ifelse(k, OP<IMM>(a), src);                                    \
  }

#define GENERATE_SERIAL_MASKZ_UNARY_TI(OP)                                     \
  template <int IMM, typename T, int SIMD_WIDTH>                               \
  static SerialVec<T, SIMD_WIDTH> maskz_##OP(                                  \
    const SerialMask<T, SIMD_WIDTH> &k, const SerialVec<T, SIMD_WIDTH> &a)     \
  {                                                                            \
    return mask_ifelsezero(k, OP<IMM>(a));                                     \
  }

#define GENERATE_SERIAL_MASK_MASKZ_UNARY_TI(OP)                                \
  GENERATE_SERIAL_MASK_UNARY_TI(OP)                                            \
  GENERATE_SERIAL_MASKZ_UNARY_TI(OP)

#define GENERATE_SERIAL_MASK_BINARY_NAME(OP, OP_NAME)                          \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialVec<T, SIMD_WIDTH> mask_##OP_NAME(                              \
    const SerialVec<T, SIMD_WIDTH> &src, const SerialMask<T, SIMD_WIDTH> &k,   \
    const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)      \
  {                                                                            \
    return mask_ifelse(k, OP(a, b), src);                                      \
  }

#define GENERATE_SERIAL_MASK_BINARY(OP) GENERATE_SERIAL_MASK_BINARY_NAME(OP, OP)

#define GENERATE_SERIAL_MASKZ_BINARY_NAME(OP, OP_NAME)                         \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialVec<T, SIMD_WIDTH> maskz_##OP_NAME(                             \
    const SerialMask<T, SIMD_WIDTH> &k, const SerialVec<T, SIMD_WIDTH> &a,     \
    const SerialVec<T, SIMD_WIDTH> &b)                                         \
  {                                                                            \
    return mask_ifelsezero(k, OP(a, b));                                       \
  }

#define GENERATE_SERIAL_MASKZ_BINARY(OP)                                       \
  GENERATE_SERIAL_MASKZ_BINARY_NAME(OP, OP)

#define GENERATE_SERIAL_MASK_MASKZ_BINARY_NAME(OP, OP_NAME)                    \
  GENERATE_SERIAL_MASK_BINARY_NAME(OP, OP_NAME)                                \
  GENERATE_SERIAL_MASKZ_BINARY_NAME(OP, OP_NAME)

#define GENERATE_SERIAL_MASK_MASKZ_BINARY(OP)                                  \
  GENERATE_SERIAL_MASK_BINARY(OP)                                              \
  GENERATE_SERIAL_MASKZ_BINARY(OP)

GENERATE_SERIAL_MASK_MASKZ_BINARY(add)
GENERATE_SERIAL_MASK_MASKZ_BINARY(adds)
GENERATE_SERIAL_MASK_MASKZ_BINARY(sub)
GENERATE_SERIAL_MASK_MASKZ_BINARY(subs)
GENERATE_SERIAL_MASK_MASKZ_BINARY(mul)
GENERATE_SERIAL_MASK_MASKZ_BINARY(div)

GENERATE_SERIAL_MASK_MASKZ_UNARY(ceil)
GENERATE_SERIAL_MASK_MASKZ_UNARY(floor)
GENERATE_SERIAL_MASK_MASKZ_UNARY(round)
GENERATE_SERIAL_MASK_MASKZ_UNARY(truncate)

// 02. Oct 22 (Jonas Keller): added rcp and rsqrt
GENERATE_SERIAL_MASK_MASKZ_UNARY(rcp)
GENERATE_SERIAL_MASK_MASKZ_UNARY(rsqrt)

GENERATE_SERIAL_MASK_MASKZ_UNARY(sqrt)
GENERATE_SERIAL_MASK_MASKZ_UNARY(abs)

GENERATE_SERIAL_MASK_MASKZ_BINARY_NAME(and_, and)
GENERATE_SERIAL_MASK_MASKZ_BINARY_NAME(or_, or)
GENERATE_SERIAL_MASK_MASKZ_BINARY(andnot)
GENERATE_SERIAL_MASK_MASKZ_BINARY_NAME(xor_, xor)
GENERATE_SERIAL_MASK_MASKZ_UNARY_NAME(not_, not )

GENERATE_SERIAL_MASK_MASKZ_UNARY(neg)

GENERATE_SERIAL_MASK_MASKZ_BINARY(min)
GENERATE_SERIAL_MASK_MASKZ_BINARY(max)

GENERATE_SERIAL_MASK_MASKZ_UNARY(div2r0)
GENERATE_SERIAL_MASK_MASKZ_UNARY(div2rd)

GENERATE_SERIAL_MASK_MASKZ_UNARY_TI(srai)
GENERATE_SERIAL_MASK_MASKZ_UNARY_TI(srli)
GENERATE_SERIAL_MASK_MASKZ_UNARY_TI(slli)

GENERATE_SERIAL_MASK_MASKZ_BINARY(hadd)
GENERATE_SERIAL_MASK_MASKZ_BINARY(hadds)
GENERATE_SERIAL_MASK_MASKZ_BINARY(hsub)
GENERATE_SERIAL_MASK_MASKZ_BINARY(hsubs)

// 16. Oct 22 (Jonas Keller): added overloaded versions of mask_cmp* functions
// that only take two vector parameters and no mask parameter

#define GENERATE_SERIAL_MASK_CMP(OP)                                           \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialMask<T, SIMD_WIDTH> mask_##OP(                                  \
    const SerialMask<T, SIMD_WIDTH> &k, const SerialVec<T, SIMD_WIDTH> &a,     \
    const SerialVec<T, SIMD_WIDTH> &b)                                         \
  {                                                                            \
    return mask_ifelsezero(k, OP(a, b));                                       \
  }                                                                            \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialMask<T, SIMD_WIDTH> mask_##OP(                                  \
    const SerialVec<T, SIMD_WIDTH> &a, const SerialVec<T, SIMD_WIDTH> &b)      \
  {                                                                            \
    return OP(a, b);                                                           \
  }

GENERATE_SERIAL_MASK_CMP(cmplt)
GENERATE_SERIAL_MASK_CMP(cmple)
GENERATE_SERIAL_MASK_CMP(cmpeq)
GENERATE_SERIAL_MASK_CMP(cmpgt)
GENERATE_SERIAL_MASK_CMP(cmpge)
GENERATE_SERIAL_MASK_CMP(cmpneq)

GENERATE_SERIAL_MASK_MASKZ_BINARY(avg)

template <typename T, int SIMD_WIDTH>
static int mask_test_all_zeros(const SerialMask<T, SIMD_WIDTH> &k,
                               const SerialVec<T, SIMD_WIDTH> &a)
{
  return test_all_zeros(mask_ifelsezero(k, a));
}

template <typename T, int SIMD_WIDTH>
static int mask_test_all_ones(const SerialMask<T, SIMD_WIDTH> &k,
                              const SerialVec<T, SIMD_WIDTH> &a)
{
  SerialVec<T, SIMD_WIDTH> ones;
  ones.setones();
  return test_all_ones(mask_ifelse(k, a, ones));
}

#define GENERATE_SERIAL_BINARY_K(OP_NAME, OP)                                  \
  template <typename T, int SIMD_WIDTH>                                        \
  static SerialMask<T, SIMD_WIDTH> OP_NAME(const SerialMask<T, SIMD_WIDTH> &a, \
                                           const SerialMask<T, SIMD_WIDTH> &b) \
  {                                                                            \
    return OP;                                                                 \
  }

GENERATE_SERIAL_BINARY_K(kand, a &b)
GENERATE_SERIAL_BINARY_K(kandn, (~a) & b)
GENERATE_SERIAL_BINARY_K(kor, a | b)
GENERATE_SERIAL_BINARY_K(kxor, a ^ b)
GENERATE_SERIAL_BINARY_K(kxnor, ~(a ^ b))
GENERATE_SERIAL_BINARY_K(kadd, a + b)

template <typename T, int SIMD_WIDTH>
static SerialMask<T, SIMD_WIDTH> knot(const SerialMask<T, SIMD_WIDTH> &a)
{
  return ~a;
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SerialMask<T, SIMD_WIDTH> kshiftli(const SerialMask<T, SIMD_WIDTH> &a)
{
  if (IMM >= SerialMask<T, SIMD_WIDTH>::bits) { return 0; }
  // we checked that IMM is not too large above, disable warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#ifdef __INTEL_COMPILER
#pragma warning(suppress : 63) // shift count is too large
#endif
  return a << IMM;
#pragma GCC diagnostic pop
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SerialMask<T, SIMD_WIDTH> kshiftri(const SerialMask<T, SIMD_WIDTH> &a)
{
  if (IMM >= SerialMask<T, SIMD_WIDTH>::bits) { return 0; }
  // we checked that IMM is not too large above, disable warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
#ifdef __INTEL_COMPILER
#pragma warning(suppress : 63) // shift count is too large
#endif
  return a >> IMM;
#pragma GCC diagnostic pop
}

template <typename T, int SIMD_WIDTH>
static SerialMask<T, SIMD_WIDTH> kshiftli(const SerialMask<T, SIMD_WIDTH> &a,
                                          unsigned int count)
{
  if (count >= SerialMask<T, SIMD_WIDTH>::bits) { return 0; }
  return a << count;
}

template <typename T, int SIMD_WIDTH>
static SerialMask<T, SIMD_WIDTH> kshiftri(const SerialMask<T, SIMD_WIDTH> &a,
                                          unsigned int count)
{
  if (count >= SerialMask<T, SIMD_WIDTH>::bits) { return 0; }
  return a >> count;
}

} // namespace auto_test
} // namespace simd

#endif // _SIMD_VEC_AUTO_TEST_SERIAL_MASK_H_
