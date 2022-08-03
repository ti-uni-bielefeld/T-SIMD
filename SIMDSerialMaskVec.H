#ifndef _SIMD_MASK_VEC_SERIAL_H_
#define _SIMD_MASK_VEC_SERIAL_H_

#include "SIMDSerialVec.H"
#include "SIMDVec.H"

namespace ns_simd {

template <int Bits> struct _UInt;
template <> struct _UInt<8> { using type = uint8_t; };
template <> struct _UInt<16> { using type = uint16_t; };
template <> struct _UInt<32> { using type = uint32_t; };
template <> struct _UInt<64> { using type = uint64_t; };

template <int Bits> using UInt = typename _UInt<Bits>::type;

template <int Bits> UInt<Bits> getRandomUInt();
template <> UInt<8> getRandomUInt<8>() { return random() & 0xFF; }
template <> UInt<16> getRandomUInt<16>() { return random() & 0xFFFF; }
template <> UInt<32> getRandomUInt<32>() { return random(); }
template <> UInt<64> getRandomUInt<64>() {
  return ((UInt<64>)random() << 32) | random();
}

template <typename T, int SIMD_WIDTH> class SIMDSerialMask {
public:
  enum { bits = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH) };
  UInt<bits> mask;

  SIMDSerialMask() : mask(0) {}
  SIMDSerialMask(const UInt<bits> &_mask) : mask(_mask) {}
  SIMDSerialMask(const SIMDSerialVec<T, SIMD_WIDTH> &x) {
    mask = 0;
    // avoids comparison with SIMDTypeInfo::trueval, which fails for
    // SIMDFloat (NaN)
    SIMDSerialVec<T, SIMD_WIDTH> negX = not(x);
    for (int i = 0; i < x.elements; i++) {
      if (negX[i] == T(0)) {
        mask |= (UInt<bits>(1) << i);
      }
    }
  }
  SIMDSerialMask(const SIMDMask<T, SIMD_WIDTH> &x) : mask(x) {}
  SIMDSerialMask<T, SIMD_WIDTH> &operator=(const UInt<bits> &_mask) {
    mask = _mask;
    return *this;
  }
  SIMDSerialMask<T, SIMD_WIDTH> &operator=(const SIMDMask<T, SIMD_WIDTH> &x) {
    mask = x;
    return *this;
  }

  operator UInt<bits>() const { return mask; }
  operator SIMDMask<T, SIMD_WIDTH>() const { return getSIMDMask(); }
  bool operator==(const SIMDSerialMask<T, SIMD_WIDTH> &other) const {
    return mask == other.mask;
  }

  bool operator[](const uint8_t i) const {
    if (i >= bits) {
      return false;
    }
    return (mask & (UInt<bits>(1) << i)) != 0;
  }

  void randomize() {
    // randomize with a 1 in 16 chance to be all 1s or all 0s
    // to make sure these cases are also tested
    if ((random() & 0x0f) == 0) {
      if (random() & 0x01) {
        mask = 0;
      } else {
        mask = -1;
      }
    } else {
      mask = getRandomUInt<bits>();
    }
  }

  SIMDMask<T, SIMD_WIDTH> getSIMDMask() const {
    return SIMDMask<T, SIMD_WIDTH>(mask);
  }
};

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
mask_ifelse(const SIMDSerialMask<T, SIMD_WIDTH> &cond,
            const SIMDSerialVec<T, SIMD_WIDTH> &trueVal,
            const SIMDSerialVec<T, SIMD_WIDTH> &falseVal) {
  SIMDSerialVec<T, SIMD_WIDTH> result;
  for (int i = 0; i < result.elements; i++) {
    result[i] = cond[i] ? trueVal[i] : falseVal[i];
  }
  return result;
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
mask_ifelsezero(const SIMDSerialMask<T, SIMD_WIDTH> &cond,
                const SIMDSerialVec<T, SIMD_WIDTH> &trueVal) {
  return mask_ifelse(cond, trueVal,
                     Deductor<T, SIMD_WIDTH, SIMDSerialVec>::_setzero());
}

template <int SIMD_WIDTH>
static SIMDSerialVec<SIMDInt, SIMD_WIDTH>
mask_cvts(const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &src,
          const SIMDSerialMask<SIMDFloat, SIMD_WIDTH> &k,
          const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a) {
  return mask_ifelse(SIMDSerialMask<SIMDInt, SIMD_WIDTH>(k.mask), cvts(a), src);
}

template <int SIMD_WIDTH>
static SIMDSerialVec<SIMDInt, SIMD_WIDTH>
maskz_cvts(const SIMDSerialMask<SIMDFloat, SIMD_WIDTH> &k,
           const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &a) {
  return mask_cvts(Deductor<SIMDInt, SIMD_WIDTH, SIMDSerialVec>::_setzero(), k,
                   a);
}

template <int SIMD_WIDTH>
static SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
mask_cvts(const SIMDSerialVec<SIMDFloat, SIMD_WIDTH> &src,
          const SIMDSerialMask<SIMDInt, SIMD_WIDTH> &k,
          const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &a) {
  return mask_ifelse(SIMDSerialMask<SIMDFloat, SIMD_WIDTH>(k.mask), cvts(a),
                     src);
}

template <int SIMD_WIDTH>
static SIMDSerialVec<SIMDFloat, SIMD_WIDTH>
maskz_cvts(const SIMDSerialMask<SIMDInt, SIMD_WIDTH> &k,
           const SIMDSerialVec<SIMDInt, SIMD_WIDTH> &a) {
  return mask_cvts(Deductor<SIMDFloat, SIMD_WIDTH, SIMDSerialVec>::_setzero(),
                   k, a);
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
mask_set1(const SIMDSerialVec<T, SIMD_WIDTH> &src,
          const SIMDSerialMask<T, SIMD_WIDTH> &k, const T a) {
  return mask_ifelse(k, Deductor<T, SIMD_WIDTH, SIMDSerialVec>::_set1(a), src);
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
maskz_set1(const SIMDSerialMask<T, SIMD_WIDTH> &k, const T a) {
  return mask_set1(Deductor<T, SIMD_WIDTH, SIMDSerialVec>::_setzero(), k, a);
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
mask_loadu(const SIMDSerialVec<T, SIMD_WIDTH> &src,
           const SIMDSerialMask<T, SIMD_WIDTH> &k, const T *p) {
  SIMDSerialVec<T, SIMD_WIDTH> result;
  for (int i = 0; i < result.elements; i++) {
    result[i] = k[i] ? p[i] : src[i];
  }
  return result;
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
maskz_loadu(const SIMDSerialMask<T, SIMD_WIDTH> &k, const T *p) {
  return mask_loadu(Deductor<T, SIMD_WIDTH, SIMDSerialVec>::_setzero(), k, p);
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
mask_load(const SIMDSerialVec<T, SIMD_WIDTH> &src,
          const SIMDSerialMask<T, SIMD_WIDTH> &k, const T *p) {
  // TODO: make sure p is aligned?
  return mask_loadu(src, k, p);
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialVec<T, SIMD_WIDTH>
maskz_load(const SIMDSerialMask<T, SIMD_WIDTH> &k, const T *p) {
  // TODO: make sure p is aligned?
  return maskz_loadu(k, p);
}

template <typename T, int SIMD_WIDTH>
static void mask_storeu(T *const p, const SIMDSerialMask<T, 64> &k,
                        const SIMDSerialVec<T, 64> &a) {
  for (int i = 0; i < a.elements; i++) {
    if (k[i]) {
      p[i] = a[i];
    }
  }
}

template <typename T, int SIMD_WIDTH>
static void mask_store(T *const p, const SIMDSerialMask<T, 64> &k,
                       const SIMDSerialVec<T, 64> &a) {
  // TODO: make sure p is aligned?
  mask_storeu(p, k, a);
}

#define GENERATE_SERIAL_MASK_UNARY(OP)                                         \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMDSerialVec<T, SIMD_WIDTH> mask_##OP(                               \
      const SIMDSerialVec<T, SIMD_WIDTH> &src,                                 \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a) {                                 \
    return mask_ifelse(k, OP(a), src);                                         \
  }

#define GENERATE_SERIAL_MASKZ_UNARY(OP)                                        \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMDSerialVec<T, SIMD_WIDTH> maskz_##OP(                              \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a) {                                 \
    return mask_ifelsezero(k, OP(a));                                          \
  }

#define GENERATE_SERIAL_MASK_MASKZ_UNARY(OP)                                   \
  GENERATE_SERIAL_MASK_UNARY(OP)                                               \
  GENERATE_SERIAL_MASKZ_UNARY(OP)

#define GENERATE_SERIAL_MASK_UNARY_TI(OP)                                      \
  template <int IMM, typename T, int SIMD_WIDTH>                               \
  static SIMDSerialVec<T, SIMD_WIDTH> mask_##OP(                               \
      const SIMDSerialVec<T, SIMD_WIDTH> &src,                                 \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a) {                                 \
    return mask_ifelse(k, OP<IMM>(a), src);                                    \
  }

#define GENERATE_SERIAL_MASKZ_UNARY_TI(OP)                                     \
  template <int IMM, typename T, int SIMD_WIDTH>                               \
  static SIMDSerialVec<T, SIMD_WIDTH> maskz_##OP(                              \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a) {                                 \
    return mask_ifelsezero(k, OP<IMM>(a));                                     \
  }

#define GENERATE_SERIAL_MASK_MASKZ_UNARY_TI(OP)                                \
  GENERATE_SERIAL_MASK_UNARY_TI(OP)                                            \
  GENERATE_SERIAL_MASKZ_UNARY_TI(OP)

#define GENERATE_SERIAL_MASK_BINARY(OP)                                        \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMDSerialVec<T, SIMD_WIDTH> mask_##OP(                               \
      const SIMDSerialVec<T, SIMD_WIDTH> &src,                                 \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a,                                   \
      const SIMDSerialVec<T, SIMD_WIDTH> &b) {                                 \
    return mask_ifelse(k, OP(a, b), src);                                      \
  }

#define GENERATE_SERIAL_MASKZ_BINARY(OP)                                       \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMDSerialVec<T, SIMD_WIDTH> maskz_##OP(                              \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a,                                   \
      const SIMDSerialVec<T, SIMD_WIDTH> &b) {                                 \
    return mask_ifelsezero(k, OP(a, b));                                       \
  }

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

// TODO: rcp (not possible since hardware implementation is not known)
// TODO: rsqrt (not possible since hardware implementation is not known)

GENERATE_SERIAL_MASK_MASKZ_UNARY(sqrt)
GENERATE_SERIAL_MASK_MASKZ_UNARY(abs)

GENERATE_SERIAL_MASK_MASKZ_BINARY(and)
GENERATE_SERIAL_MASK_MASKZ_BINARY(or)
GENERATE_SERIAL_MASK_MASKZ_BINARY(andnot)
GENERATE_SERIAL_MASK_MASKZ_BINARY(xor)
GENERATE_SERIAL_MASK_MASKZ_UNARY(not )

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

#define GENERATE_SERIAL_MASK_CMP(OP)                                           \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMDSerialMask<T, SIMD_WIDTH> mask_##OP(                              \
      const SIMDSerialMask<T, SIMD_WIDTH> &k,                                  \
      const SIMDSerialVec<T, SIMD_WIDTH> &a,                                   \
      const SIMDSerialVec<T, SIMD_WIDTH> &b) {                                 \
    return mask_ifelsezero(k, OP(a, b));                                       \
  }

GENERATE_SERIAL_MASK_CMP(cmplt)
GENERATE_SERIAL_MASK_CMP(cmple)
GENERATE_SERIAL_MASK_CMP(cmpeq)
GENERATE_SERIAL_MASK_CMP(cmpgt)
GENERATE_SERIAL_MASK_CMP(cmpge)
GENERATE_SERIAL_MASK_CMP(cmpneq)

GENERATE_SERIAL_MASK_MASKZ_BINARY(avg)

template <typename T, int SIMD_WIDTH>
static int mask_test_all_zeros(const SIMDSerialMask<T, SIMD_WIDTH> &k,
                               const SIMDSerialVec<T, SIMD_WIDTH> &a) {
  return test_all_zeros(mask_ifelsezero(k, a));
}

template <typename T, int SIMD_WIDTH>
static int mask_test_all_ones(const SIMDSerialMask<T, SIMD_WIDTH> &k,
                              const SIMDSerialVec<T, SIMD_WIDTH> &a) {
  return test_all_ones(mask_ifelsezero(k, a));
}

#define GENERATE_SERIAL_BINARY_K(OP_NAME, OP)                                  \
  template <typename T, int SIMD_WIDTH>                                        \
  static SIMDSerialMask<T, SIMD_WIDTH> OP_NAME(                                \
      const SIMDSerialMask<T, SIMD_WIDTH> &a,                                  \
      const SIMDSerialMask<T, SIMD_WIDTH> &b) {                                \
    return OP;                                                                 \
  }

GENERATE_SERIAL_BINARY_K(kand, a &b)
GENERATE_SERIAL_BINARY_K(kandn, (~a) & b)
GENERATE_SERIAL_BINARY_K(kor, a | b)
GENERATE_SERIAL_BINARY_K(kxor, a ^ b)
GENERATE_SERIAL_BINARY_K(kxnor, ~(a ^ b))
GENERATE_SERIAL_BINARY_K(kadd, a + b)

template <typename T, int SIMD_WIDTH>
static SIMDSerialMask<T, SIMD_WIDTH>
knot(const SIMDSerialMask<T, SIMD_WIDTH> &a) {
  return ~a;
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMDSerialMask<T, SIMD_WIDTH>
kshiftli(const SIMDSerialMask<T, SIMD_WIDTH> &a) {
  if (IMM >= SIMDSerialMask<T, SIMD_WIDTH>::bits) {
    return 0;
  }
  // we checked that IMM is not too large above, disable warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
  return a << IMM;
#pragma GCC diagnostic pop
}

template <unsigned int IMM, typename T, int SIMD_WIDTH>
static SIMDSerialMask<T, SIMD_WIDTH>
kshiftri(const SIMDSerialMask<T, SIMD_WIDTH> &a) {
  if (IMM >= SIMDSerialMask<T, SIMD_WIDTH>::bits) {
    return 0;
  }
  // we checked that IMM is not too large above, disable warning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshift-count-overflow"
  return a >> IMM;
#pragma GCC diagnostic pop
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialMask<T, SIMD_WIDTH>
kshiftli(const SIMDSerialMask<T, SIMD_WIDTH> &a, unsigned int count) {
  if (count >= SIMDSerialMask<T, SIMD_WIDTH>::bits) {
    return 0;
  }
  return a << count;
}

template <typename T, int SIMD_WIDTH>
static SIMDSerialMask<T, SIMD_WIDTH>
kshiftri(const SIMDSerialMask<T, SIMD_WIDTH> &a, unsigned int count) {
  if (count >= SIMDSerialMask<T, SIMD_WIDTH>::bits) {
    return 0;
  }
  return a >> count;
}

} // namespace ns_simd

#endif // _SIMD_MASK_VEC_SERIAL_H_
