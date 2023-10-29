// ===========================================================================
//
// SIMDVecBaseImplSandbox.H --
// test template functions for Vec and base level ("level-0") function
// templates these functions just print out template parameters and some
// arguments
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

// 22. Jan 23 (Jonas Keller): moved sandbox implementations into internal
// namespace

#pragma once
#ifndef SIMD_VEC_BASE_IMPL_SANDBOX_H_
#define SIMD_VEC_BASE_IMPL_SANDBOX_H_

// TODO: should this only contain level 0 functions?

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <cstdint>
#include <cstdio>

#ifdef SIMDVEC_SANDBOX

namespace simd {

// ===========================================================================
// generic template for Vec
// ===========================================================================

template <typename T, size_t SIMD_WIDTH>
class Vec
{
public:
  using Type                       = T;
  static constexpr size_t elements = SIMD_WIDTH / sizeof(T);
  static constexpr size_t elems    = elements;
  static constexpr size_t bytes    = SIMD_WIDTH;
  Vec() {}
  // 05. Sep 23 (Jonas Keller): added allocator
  using allocator = simd_aligned_allocator<Vec<T, bytes>, bytes>;
};

namespace internal {
namespace base {

// ===========================================================================
// reinterpretation cast
// ===========================================================================

template <typename Tdst, typename Tsrc, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tdst, SIMD_WIDTH> reinterpret(
  const Vec<Tsrc, SIMD_WIDTH> &, OutputType<Tdst>)
{
  printf("reinterpret<%s,%s,%zu>(V)\n", TypeInfo<Tdst>::name(),
         TypeInfo<Tsrc>::name(), SIMD_WIDTH);
  return Vec<Tdst, SIMD_WIDTH>();
}

// ===========================================================================
// generic functions on Vec
// ===========================================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setzero(OutputType<T>,
                                              Integer<SIMD_WIDTH>)
{
  printf("setzero<%s,%zu>()\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> set1(T, Integer<SIMD_WIDTH>)
{
  printf("set1<%s,%zu>()\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// template argument order reversed so that T can be omitted in instantiation
template <size_t SIMD_WIDTH, typename T>
static SIMD_INLINE Vec<T, SIMD_WIDTH> load(const T *const p,
                                           Integer<SIMD_WIDTH>)
{
  printf("load<%zu,%s>(%p)\n", SIMD_WIDTH, TypeInfo<T>::name(), (void *) p);
  return Vec<T, SIMD_WIDTH>();
}

// template argument order reversed so that T can be omitted in instantiation
template <size_t SIMD_WIDTH, typename T>
static SIMD_INLINE Vec<T, SIMD_WIDTH> loadu(const T *const p,
                                            Integer<SIMD_WIDTH>)
{
  printf("loadu<%zu,%s>(%p)\n", SIMD_WIDTH, TypeInfo<T>::name(), (void *) p);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void store(T *const p, const Vec<T, SIMD_WIDTH> &)
{
  printf("store<%s,%zu>(%p,V)\n", TypeInfo<T>::name(), SIMD_WIDTH, (void *) p);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void storeu(T *const p, const Vec<T, SIMD_WIDTH> &)
{
  printf("storeu<%s,%zu>(%p,V)\n", TypeInfo<T>::name(), SIMD_WIDTH, (void *) p);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void stream_store(T *const p, const Vec<T, SIMD_WIDTH> &)
{
  printf("stream_store<%s,%zu>(%p,V)\n", TypeInfo<T>::name(), SIMD_WIDTH,
         (void *) p);
}

static SIMD_INLINE void lfence()
{
  printf("lfence\n");
}

static SIMD_INLINE void sfence()
{
  printf("sfence\n");
}

static SIMD_INLINE void mfence()
{
  printf("mfence\n");
}

template <size_t INDEX, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE T extract(const Vec<T, SIMD_WIDTH> &)
{
  printf("extract<%zu,%s,%zu>(V)\n", INDEX, TypeInfo<T>::name(), SIMD_WIDTH);
  return T(0);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> add(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("add<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> adds(const Vec<T, SIMD_WIDTH> &,
                                           const Vec<T, SIMD_WIDTH> &)
{
  printf("adds<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> sub(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("sub<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> subs(const Vec<T, SIMD_WIDTH> &,
                                           const Vec<T, SIMD_WIDTH> &)
{
  printf("subs<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> neg(const Vec<T, SIMD_WIDTH> &)
{
  printf("neg<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> mul(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("mul<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> div(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("div<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> ceil(const Vec<T, SIMD_WIDTH> &)
{
  printf("ceil<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> floor(const Vec<T, SIMD_WIDTH> &)
{
  printf("floor<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> round(const Vec<T, SIMD_WIDTH> &)
{
  printf("round<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> truncate(const Vec<T, SIMD_WIDTH> &)
{
  printf("truncate<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> rcp(const Vec<T, SIMD_WIDTH> &)
{
  printf("rcp<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> rsqrt(const Vec<T, SIMD_WIDTH> &)
{
  printf("rsqrt<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> sqrt(const Vec<T, SIMD_WIDTH> &)
{
  printf("sqrt<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> min(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("min<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> max(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("max<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> abs(const Vec<T, SIMD_WIDTH> &)
{
  printf("abs<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// unpack NUM_ELEMS elements of type T
// PART=0: low half of input vectors,
// PART=1: high half of input vectors
template <size_t PART, size_t BYTES, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> unpack(const Vec<T, SIMD_WIDTH> &,
                                             const Vec<T, SIMD_WIDTH> &,
                                             Part<PART>, Bytes<BYTES>)
{
  printf("unpack<PART=%zu,BYTES=%zu,%s,%zu>(V,V)\n", PART, BYTES,
         TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// unpack16
template <size_t PART, size_t BYTES, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> unpack16(const Vec<T, SIMD_WIDTH> &,
                                               const Vec<T, SIMD_WIDTH> &,
                                               Part<PART>, Bytes<BYTES>)
{
  printf("unpack16<PART=%zu,BYTES=%zu,%s,%zu>(V,V)\n", PART, BYTES,
         TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// extractLane
template <size_t LANE_INDEX, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, 16> extractLane(const Vec<T, SIMD_WIDTH> &)
{
  printf("extractLane<IMM=%zu,%s,%zu>(V)\n", LANE_INDEX, TypeInfo<T>::name(),
         SIMD_WIDTH);
  return Vec<T, 16>();
}

// zip (2 x unpack) NUM_ELEMS elements of type T
template <size_t NUM_ELEMS, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void zip(const Vec<T, SIMD_WIDTH>, const Vec<T, SIMD_WIDTH>,
                            Vec<T, SIMD_WIDTH> &, Vec<T, SIMD_WIDTH> &)
{
  printf("zip<NUM_ELEMS=%zu,%s,%zu>(V,V,V&,V&)\n", NUM_ELEMS,
         TypeInfo<T>::name(), SIMD_WIDTH);
}

// zip16
template <size_t NUM_ELEMS, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void zip16(const Vec<T, SIMD_WIDTH>,
                              const Vec<T, SIMD_WIDTH>, Vec<T, SIMD_WIDTH> &,
                              Vec<T, SIMD_WIDTH> &)
{
  printf("zip16<NUM_ELEMS=%zu,%s,%zu>(V,V,V&,V&)\n", NUM_ELEMS,
         TypeInfo<T>::name(), SIMD_WIDTH);
}

// unzip (inverse of zip)
template <size_t BYTES, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void unzip(const Vec<T, SIMD_WIDTH>,
                              const Vec<T, SIMD_WIDTH>, Vec<T, SIMD_WIDTH> &,
                              Vec<T, SIMD_WIDTH> &, Bytes<BYTES>)
{
  printf("unzip<BYTES=%zu,%s,%zu>(V,V,V&,V&)\n", BYTES, TypeInfo<T>::name(),
         SIMD_WIDTH);
}

// unifies packs and packus, depending on Tout
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(const Vec<Tin, SIMD_WIDTH> &,
                                               const Vec<Tin, SIMD_WIDTH> &,
                                               OutputType<Tout>)
{
  printf("packs<%s,%s,%zu>(V,V)\n", TypeInfo<Tout>::name(),
         TypeInfo<Tin>::name(), SIMD_WIDTH);
  return Vec<Tout, SIMD_WIDTH>();
}

// generalized version of unpack-based type conversion: includes
// multistage extension (zero-extend or sign-extend, depending on
// type)
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void extend(
  const Vec<Tin, SIMD_WIDTH> &,
  Vec<Tout, SIMD_WIDTH>[sizeof(Tout) / sizeof(Tin)])
{
  printf("extend<%s,%s,%zu>>(V,V*)\n", TypeInfo<Tout>::name(),
         TypeInfo<Tin>::name(), SIMD_WIDTH);
}

template <size_t COUNT, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> srai(const Vec<T, SIMD_WIDTH> &)
{
  printf("srai<%zu,%s,%zu>(V)\n", COUNT, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <size_t COUNT, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> srli(const Vec<T, SIMD_WIDTH> &)
{
  printf("srli<%zu,%s,%zu>(V)\n", COUNT, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <size_t COUNT, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> slli(const Vec<T, SIMD_WIDTH> &)
{
  printf("slli<%zu,%s,%zu>(V)\n", COUNT, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// 12. Jan 23 (Jonas Keller): added sra, srl and sll functions

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> sra(const Vec<T, SIMD_WIDTH> &,
                                          const uint8_t)
{
  printf("sra<%s,%zu>(V, U8)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> srl(const Vec<T, SIMD_WIDTH> &,
                                          const uint8_t)
{
  printf("srl<%s,%zu>(V, U8)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> sll(const Vec<T, SIMD_WIDTH> &,
                                          const uint8_t)
{
  printf("sll<%s,%zu>(V, U8)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// conversion without changes in the number of vector elements
// saturated version
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> cvts(const Vec<Tin, SIMD_WIDTH> &,
                                              OutputType<Tout>)
{
  printf("cvts<%s,%s,%zu>(V)\n", TypeInfo<Tout>::name(), TypeInfo<Tin>::name(),
         SIMD_WIDTH);
  return Vec<Tout, SIMD_WIDTH>();
}

// horizontal addition
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hadd(const Vec<T, SIMD_WIDTH> &,
                                           const Vec<T, SIMD_WIDTH> &)
{
  printf("hadd<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// horizontal addition (with saturation)
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hadds(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("hadds<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// horizontal subtraction
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hsub(const Vec<T, SIMD_WIDTH> &,
                                           const Vec<T, SIMD_WIDTH> &)
{
  printf("hsub<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// horizontal subtraction (with saturation)
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hsubs(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("hsubs<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// logical right shift by n elements, filling in zeros
template <size_t COUNT, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> srle(const Vec<T, SIMD_WIDTH> &)
{
  printf("srle<%zu,%s,%zu>(V)\n", COUNT, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// logical left shift by n elements, filling in zeros
template <size_t COUNT, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> slle(const Vec<T, SIMD_WIDTH> &)
{
  printf("slle<%zu,%s,%zu>(V)\n", COUNT, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// alignr (with number of elements in imm):
template <size_t COUNT, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> alignre(const Vec<T, SIMD_WIDTH> &,
                                              const Vec<T, SIMD_WIDTH> &)
{
  printf("alignre<%zu,%s,%zu>(V,V)\n", COUNT, TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// swizzle (AoS to SoA), in-place, with template parameter N
template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle(Vec<T, SIMD_WIDTH>[N], Integer<N>)
{
  printf("swizzle<%zu,%s,%zu>(V,V*)\n", N, TypeInfo<T>::name(), SIMD_WIDTH);
}

// ifelse
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> ifelse(const Vec<T, SIMD_WIDTH> &,
                                             const Vec<T, SIMD_WIDTH> &,
                                             const Vec<T, SIMD_WIDTH> &)
{
  printf("ifelse<%s,%zu>(V,V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// compare <
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> cmplt(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("cmplt<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// compare <=
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> cmple(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("cmple<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// compare ==
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> cmpeq(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("cmpeq<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// compare >=
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> cmpge(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("cmpge<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// compare >
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> cmpgt(const Vec<T, SIMD_WIDTH> &,
                                            const Vec<T, SIMD_WIDTH> &)
{
  printf("cmpgt<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// compare !=
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> cmpneq(const Vec<T, SIMD_WIDTH> &,
                                             const Vec<T, SIMD_WIDTH> &)
{
  printf("cmpneq<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// bitwise and
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> bit_and(const Vec<T, SIMD_WIDTH> &,
                                              const Vec<T, SIMD_WIDTH> &)
{
  printf("and<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// bitwise or
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> bit_or(const Vec<T, SIMD_WIDTH> &,
                                             const Vec<T, SIMD_WIDTH> &)
{
  printf("or<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// bitwise andnot: (not a) and b
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> bit_andnot(const Vec<T, SIMD_WIDTH> &,
                                                 const Vec<T, SIMD_WIDTH> &)
{
  printf("bit_andnot<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// bitwise xor
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> bit_xor(const Vec<T, SIMD_WIDTH> &,
                                              const Vec<T, SIMD_WIDTH> &)
{
  printf("xor<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// bitwise not (work-arounds required)
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> bit_not(const Vec<T, SIMD_WIDTH> &)
{
  printf("not<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> avg(const Vec<T, SIMD_WIDTH> &,
                                          const Vec<T, SIMD_WIDTH> &)
{
  printf("avg<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// test if all bits are zeros (false)
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE bool test_all_zeros(const Vec<T, SIMD_WIDTH> &)
{
  printf("test_all_zeros<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return 0;
}

// test if all bits are ones (true)
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE bool test_all_ones(const Vec<T, SIMD_WIDTH> &)
{
  printf("test_all_ones<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return 0;
}

// TODO: not level 0, should this be in the sandbox?
// negate a, where b is negative, Float only
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> sign(const Vec<T, SIMD_WIDTH> &,
                                           const Vec<T, SIMD_WIDTH> &)
{
  printf("sign<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// TODO: not level 0, should this be in the sandbox?
// Computes elementwise absolute difference of vectors
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> absDiff(const Vec<T, SIMD_WIDTH> &,
                                              const Vec<T, SIMD_WIDTH> &)
{
  printf("absDiff<%s,%zu>(V,V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// reverse order
template <typename T, size_t SIMD_WIDTH>
static Vec<T, SIMD_WIDTH> reverse(const Vec<T, SIMD_WIDTH> &)
{
  printf("reverse<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// 27. Aug 22 (Jonas Keller): added msb2int
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE uint64_t msb2int(const Vec<T, SIMD_WIDTH> &)
{
  printf("msb2int<%s,%zu>(V)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return 0;
}

// 09. Oct 22 (Jonas Keller): added int2msb
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> int2msb(const uint64_t, OutputType<T>,
                                              Integer<SIMD_WIDTH>)
{
  printf("int2msb<%s,%zu>(U64)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// 09. Oct 22 (Jonas Keller): added int2bits
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> int2bits(const uint64_t, OutputType<T>,
                                               Integer<SIMD_WIDTH>)
{
  printf("int2bits<%s,%zu>(U64)\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}

// 30. Jan 23 (Jonas Keller): added iota

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> iota(OutputType<T>, Integer<SIMD_WIDTH>)
{
  printf("iota<%s,%zu>()\n", TypeInfo<T>::name(), SIMD_WIDTH);
  return Vec<T, SIMD_WIDTH>();
}
} // namespace base
} // namespace internal
} // namespace simd

#endif // SIMDVEC_SANDBOX

#endif // SIMD_VEC_BASE_IMPL_SANDBOX_H_
