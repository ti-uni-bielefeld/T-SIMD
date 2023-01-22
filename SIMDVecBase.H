// ===========================================================================
// 
// SIMDVecBase.H --
// base-level classes and functions
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

#ifndef _SIMD_VEC_BASE_H_
#define _SIMD_VEC_BASE_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <stdint.h>

#ifdef SIMDVEC_SANDBOX

// test templates
#include "SIMDVecBaseImplSandbox.H"

#else // SIMDVEC_SANDBOX

// architecture-dependent files:

// Intel
#ifdef SIMDVEC_INTEL_ENABLE
#include "SIMDVecBaseImplIntel16.H"
#include "SIMDVecBaseImplIntel32.H"
#include "SIMDVecBaseImplIntel64.H"
#endif // SIMDVEC_INTEL_ENABLE

// ARM NEON
#ifdef SIMDVEC_NEON_ENABLE
#include "SIMDVecBaseImplNEON16.H"
#endif // SIMDVEC_NEON_ENABLE

#endif // else SIMDVEC_SANDBOX

namespace ns_simd {

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  reinterpret(const SIMDVec<Tin, SIMD_WIDTH> &a)
  {
    return internal::base::reinterpret(a, internal::OutputType<Tout>());
  }

  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setzero()
  {
    return internal::base::setzero(internal::OutputType<T>(),
                             internal::Integer<SIMD_WIDTH>());
  }

  template <typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  set1(T a)
  {
    return internal::base::set1(a, internal::Integer<SIMD_WIDTH>());
  }

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH_DEFAULT_NATIVE_FUNC, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  load(const T *const p)
  {
    return internal::base::load(p, internal::Integer<SIMD_WIDTH>());
  }

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH_DEFAULT_NATIVE_FUNC, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  loadu(const T *const p)
  {
    return internal::base::loadu(p, internal::Integer<SIMD_WIDTH>());
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  store(T *const p,
	const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::base::store(p, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  storeu(T *const p,
	 const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::base::storeu(p, a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  stream_store(T *const p,
	      const SIMDVec<T, SIMD_WIDTH> &a)
  {
    internal::base::stream_store(p, a);
  }

  static SIMD_INLINE void
  lfence()
  {
    internal::base::lfence();
  }

  static SIMD_INLINE void
  sfence()
  {
    internal::base::sfence();
  }

  static SIMD_INLINE void
  mfence()
  {
    internal::base::mfence();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  extract(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::extract<IMM>(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  add(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::add(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  adds(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::adds(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sub(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::sub(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  subs(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::subs(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  neg(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::neg(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mul(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::mul(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  div(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::div(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  ceil(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::ceil(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  floor(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::floor(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  round(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::round(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  truncate(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::truncate(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  rcp(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::rcp(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  rsqrt(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::rsqrt(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sqrt(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::sqrt(a);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  min(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::min(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  max(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::max(a, b);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  abs(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::abs(a);
  }

  // unpack NUM_ELEMS elements of type T
  // PART=0: low half of input vectors,
  // PART=1: high half of input vectors
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  unpack(const SIMDVec<T, SIMD_WIDTH> &a,
	 const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::unpack(a, b, internal::Part<PART>(),
                            internal::Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // unpack16
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  unpack16(const SIMDVec<T, SIMD_WIDTH> &a,
     const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::unpack16<PART, NUM_ELEMS>(a, b);
  }

  // extractLane
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, 16>
  extractLane(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::extractLane<IMM>(a);
  }

  // zip (2 x unpack) NUM_ELEMS elements of type T
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  zip(const SIMDVec<T, SIMD_WIDTH> a,
      const SIMDVec<T, SIMD_WIDTH> b,
      SIMDVec<T, SIMD_WIDTH> &l,
      SIMDVec<T, SIMD_WIDTH> &h)
  {
    return internal::base::zip<NUM_ELEMS>(a, b, l, h);
  }

  // zip16
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  zip16(const SIMDVec<T, SIMD_WIDTH> a,
      const SIMDVec<T, SIMD_WIDTH> b,
      SIMDVec<T, SIMD_WIDTH> &l,
      SIMDVec<T, SIMD_WIDTH> &h)
  {
    return internal::base::zip16<NUM_ELEMS>(a, b, l, h);
  }

  // unzip (inverse of zip)
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, SIMD_WIDTH> a,
      const SIMDVec<T, SIMD_WIDTH> b,
      SIMDVec<T, SIMD_WIDTH> &l,
      SIMDVec<T, SIMD_WIDTH> &h)
  {
    return internal::base::unzip(a, b, l, h,
                           internal::Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // unifies packs and packus, depending on Tout
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  packs(const SIMDVec<Tin, SIMD_WIDTH> &a,
	const SIMDVec<Tin, SIMD_WIDTH> &b)
  {
    return internal::base::packs(a, b, internal::OutputType<Tout>());
  }

  // generalized version of unpack-based type conversion: includes
  // multistage extension (zero-extend or sign-extend, depending on
  // type)
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  extend(const SIMDVec<Tin,SIMD_WIDTH> &vIn,
	 SIMDVec<Tout,SIMD_WIDTH> *const vOut)
  {
    return internal::base::extend(vIn, vOut);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srai(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::srai<IMM>(a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srli(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::srli<IMM>(a);
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  slli(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::slli<IMM>(a);
  }

  // 12. Jan 23 (Jonas Keller): added sra, srl and sll functions

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sra(const SIMDVec<T, SIMD_WIDTH> &a, const uint8_t count)
  {
    return internal::base::sra(a, count);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srl(const SIMDVec<T, SIMD_WIDTH> &a, const uint8_t count)
  {
    return internal::base::srl(a, count);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sll(const SIMDVec<T, SIMD_WIDTH> &a, const uint8_t count)
  {
    return internal::base::sll(a, count);
  }

  // conversion without changes in the number of vector elements
  // saturated version (but saturation only concerns float->int)
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  cvts(const SIMDVec<Tin, SIMD_WIDTH> &a)
  {
    return internal::base::cvts(a, internal::OutputType<Tout>());
  }

  // horizontal addition
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hadd(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hadd(a, b);
  }

  // horizontal addition (with saturation)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hadds(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hadds(a, b);
  }

  // horizontal subtraction
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hsub(const SIMDVec<T, SIMD_WIDTH> &a,
       const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hsub(a, b);
  }

  // horizontal subtraction (with saturation)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hsubs(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::hsubs(a, b);
  }

  // logical right shift by n elements, filling in zeros
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srle(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::srle<IMM>(a);
  }

  // logical left shift by n elements, filling in zeros
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  slle(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::slle<IMM>(a);
  }

  // extraction of lowest element
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  elem0(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::elem0(a);
  }

  // alignr (with number of elements in imm):
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  alignre(const SIMDVec<T, SIMD_WIDTH> &h,
	  const SIMDVec<T, SIMD_WIDTH> &l)
  {
    return internal::base::alignre<IMM>(h, l);
  }

  // swizzle (AoS to SoA), in-place, with template parameter N
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<N, T, SIMD_WIDTH> &t,
	  SIMDVec<T, SIMD_WIDTH> *const v)
  {
    internal::base::swizzle(t, v, internal::Integer<N>(),
                            internal::TypeIsIntSize<T>());
  }

  // ifelse
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  ifelse(const SIMDVec<T, SIMD_WIDTH> &cond,
	 const SIMDVec<T, SIMD_WIDTH> &trueVal,
	 const SIMDVec<T, SIMD_WIDTH> &falseVal)
  {
    return internal::base::ifelse(cond, trueVal, falseVal);
  }

  // compare <
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmplt(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmplt(a, b);
  }

  // compare <=
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmple(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmple(a, b);
  }

  // compare ==
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpeq(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpeq(a, b);
  }

  // compare >=
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpge(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpge(a, b);
  }

  // compare >
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpgt(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpgt(a, b);
  }

  // compare !=
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpneq(const SIMDVec<T, SIMD_WIDTH> &a,
	 const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::cmpneq(a, b);
  }

  // bitwise and
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  and(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::and(a, b);
  }

  // bitwise or
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  or(const SIMDVec<T, SIMD_WIDTH> &a,
     const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::or(a, b);
  }

  // bitwise andnot: (not a) and b
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  andnot(const SIMDVec<T, SIMD_WIDTH> &a,
	 const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::andnot(a, b);
  }

  // bitwise xor
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  xor(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::xor(a, b);
  }


  // bitwise not (work-arounds required)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  not(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::not(a);
  }

  // avg
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  avg(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return internal::base::avg(a, b);
  }

  // test if all bits are zeros (false)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::test_all_zeros(a);
  }

  // test if all bits are ones (true)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::test_all_ones(a);
  }

  // reverse order
  template <typename T, int SIMD_WIDTH>
  static SIMDVec<T, SIMD_WIDTH>
  reverse(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::reverse(a);
  }

  // 27. Aug 22 (Jonas Keller): added msb2int
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<T, SIMD_WIDTH> &a)
  {
    return internal::base::msb2int(a);
  }

  // 09. Oct 22 (Jonas Keller): added int2msb
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2msb(const uint64_t a)
  {
    return internal::base::int2msb(a, internal::OutputType<T>(),
                             internal::Integer<SIMD_WIDTH>());
  }

  // 09. Oct 22 (Jonas Keller): added int2bits
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2bits(const uint64_t a)
  {
    return internal::base::int2bits(a, internal::OutputType<T>(),
                              internal::Integer<SIMD_WIDTH>());
  }
}

#endif // _SIMD_VEC_BASE_H_
