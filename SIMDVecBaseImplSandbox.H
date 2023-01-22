// ===========================================================================
// 
// SIMDVecBaseImplSandbox.H --
// test template functions for SIMDVec and base level ("level-0") function
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

#ifndef _SIMD_VEC_BASE_IMPL_SANDBOX_H_
#define _SIMD_VEC_BASE_IMPL_SANDBOX_H_

// TODO: should this only contain level 0 functions?

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <stdint.h>
#include <stdio.h>

namespace ns_simd {

  // ===========================================================================
  // generic template for SIMDVec
  // ===========================================================================

  template <typename T, int SIMD_WIDTH>
  class SIMDVec
  {
  public:
    typedef T Type;
    enum { elements = SIMD_WIDTH / sizeof(T), bytes = SIMD_WIDTH };
    // shorter version:
    enum { elems = elements };
    SIMDVec() {}
  };

  // swizzle table (empty)
  template <int N, typename T, int SIMD_WIDTH>
  struct SwizzleTable {};

namespace internal {
namespace base {

  // ===========================================================================
  // reinterpretation cast
  // ===========================================================================

  // hub in SIMDVec.H
  template <typename Tdst, typename Tsrc, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tdst,SIMD_WIDTH>
  reinterpret(const SIMDVec<Tsrc,SIMD_WIDTH>&,
	      OutputType<Tdst>)
  {
    printf("reinterpret<%s,%s,%d>(V)\n", 
	   SIMDTypeInfo<Tdst>::name(),
	   SIMDTypeInfo<Tsrc>::name(),
	   SIMD_WIDTH);
    return SIMDVec<Tdst,SIMD_WIDTH>();
  }

  // ===========================================================================
  // generic functions on SIMDVec
  // ===========================================================================

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  setzero(OutputType<T>, Integer<SIMD_WIDTH>)
  {
    printf("setzero<%s,%d>()\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  set1(T, Integer<SIMD_WIDTH>)
  {
    printf("set1<%s,%d>()\n", 
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  load(const T *const p, Integer<SIMD_WIDTH>)
  {
    printf("load<%d,%s>(%p)\n", SIMD_WIDTH, SIMDTypeInfo<T>::name(), (void*) p);
    return SIMDVec<T,SIMD_WIDTH>();  
  }

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  loadu(const T *const p, Integer<SIMD_WIDTH>)
  {
    printf("loadu<%d,%s>(%p)\n", SIMD_WIDTH, SIMDTypeInfo<T>::name(), (void*) p);
    return SIMDVec<T,SIMD_WIDTH>();  
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  store(T *const p,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("store<%s,%d>(%p,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*) p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  storeu(T *const p, 
	 const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("storeu<%s,%d>(%p,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*) p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  stream_store(T *const p, 
	      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("stream_store<%s,%d>(%p,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*) p);
  }

  static SIMD_INLINE void
  lfence()
  {
    printf("lfence\n");
  }

  static SIMD_INLINE void
  sfence()
  {
    printf("sfence\n");
  }

  static SIMD_INLINE void
  mfence()
  {
    printf("mfence\n");
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  extract(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("extract<%d,%s,%d>(V)\n", IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return T(0);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  add(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("add<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  adds(const SIMDVec<T, SIMD_WIDTH> &,
       const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("adds<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  sub(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("sub<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  subs(const SIMDVec<T, SIMD_WIDTH> &,
       const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("subs<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  neg(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("neg<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  mul(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("mul<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  div(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("div<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  ceil(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("ceil<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  floor(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("floor<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  round(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("round<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  truncate(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("truncate<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  rcp(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("rcp<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  rsqrt(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("rsqrt<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sqrt(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("sqrt<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  min(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("min<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  max(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("max<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  abs(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("abs<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // unpack NUM_ELEMS elements of type T 
  // PART=0: low half of input vectors,
  // PART=1: high half of input vectors
  template <int PART, int BYTES, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  unpack(const SIMDVec<T, SIMD_WIDTH> &,
	 const SIMDVec<T, SIMD_WIDTH> &,
   Part<PART>, Bytes<BYTES>)
  {
    printf("unpack<PART=%d,BYTES=%d,%s,%d>(V,V)\n",
	   PART, BYTES, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();  
  }

  // unpack16
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  unpack16(const SIMDVec<T, SIMD_WIDTH> &,
     const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("unpack16<PART=%d,NUM_ELEMS=%d,%s,%d>(V,V)\n",
     PART, NUM_ELEMS, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();  
  }

  // extractLane
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, 16>
  extractLane(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("extractLane<IMM=%d,%s,%d>(V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,16>();  
  }

  // zip (2 x unpack) NUM_ELEMS elements of type T 
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  zip(const SIMDVec<T, SIMD_WIDTH>,
      const SIMDVec<T, SIMD_WIDTH>,
      SIMDVec<T, SIMD_WIDTH> &,
      SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("zip<NUM_ELEMS=%d,%s,%d>(V,V,V&,V&)\n",
	   NUM_ELEMS, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  }

  // zip16
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  zip16(const SIMDVec<T, SIMD_WIDTH>,
      const SIMDVec<T, SIMD_WIDTH>,
      SIMDVec<T, SIMD_WIDTH> &,
      SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("zip16<NUM_ELEMS=%d,%s,%d>(V,V,V&,V&)\n",
     NUM_ELEMS, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  }

  // unzip (inverse of zip)
  template <int BYTES, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, SIMD_WIDTH>,
	const SIMDVec<T, SIMD_WIDTH>,
	SIMDVec<T, SIMD_WIDTH> &,
	SIMDVec<T, SIMD_WIDTH> &,
  Bytes<BYTES>)
  {
    printf("unzip<BYTES=%d,%s,%d>(V,V,V&,V&)\n",
	   BYTES, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  }

  // unifies packs and packus, depending on Tout
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  packs(const SIMDVec<Tin, SIMD_WIDTH> &,
	const SIMDVec<Tin, SIMD_WIDTH> &,
  OutputType<Tout>)
  {
    printf("packs<%s,%s,%d>(V,V)\n", 
	   SIMDTypeInfo<Tout>::name(), 
	   SIMDTypeInfo<Tin>::name(), 
	   SIMD_WIDTH);
    return SIMDVec<Tout,SIMD_WIDTH>();
  }

  // generalized version of unpack-based type conversion: includes
  // multistage extension (zero-extend or sign-extend, depending on
  // type)
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void 
  extend(const SIMDVec<Tin,SIMD_WIDTH> &,
	 SIMDVec<Tout,SIMD_WIDTH> *const)
  {
    printf("extend<%s,%s,%d>>(V,V*)\n",
	   SIMDTypeInfo<Tout>::name(), 
	   SIMDTypeInfo<Tin>::name(),
	   SIMD_WIDTH);
  }	 

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  srai(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("srai<%d,%s,%d>(V)\n",
	   IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  srli(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("srli<%d,%s,%d>(V)\n",
	   IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  slli(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("slli<%d,%s,%d>(V)\n",
	   IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // 12. Jan 23 (Jonas Keller): added sra, srl and sll functions

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sra(const SIMDVec<T, SIMD_WIDTH>&, const uint8_t)
  {
    printf("sra<%s,%d>(V, U8)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srl(const SIMDVec<T, SIMD_WIDTH>&, const uint8_t)
  {
    printf("srl<%s,%d>(V, U8)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sll(const SIMDVec<T, SIMD_WIDTH>&, const uint8_t)
  {
    printf("sll<%s,%d>(V, U8)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // conversion without changes in the number of vector elements
  // saturated version
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  cvts(const SIMDVec<Tin, SIMD_WIDTH> &, OutputType<Tout>)
  {
    printf("cvts<%s,%s,%d>(V)\n", 
	   SIMDTypeInfo<Tout>::name(),
	   SIMDTypeInfo<Tin>::name(), 
	   SIMD_WIDTH);
    return SIMDVec<Tout,SIMD_WIDTH>();
  }

  // horizontal addition
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hadd(const SIMDVec<T, SIMD_WIDTH> &,
       const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("hadd<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // horizontal addition (with saturation)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hadds(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("hadds<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // horizontal subtraction
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hsub(const SIMDVec<T, SIMD_WIDTH> &,
       const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("hsub<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // horizontal subtraction (with saturation)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  hsubs(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("hsubs<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // logical right shift by n elements, filling in zeros
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  srle(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("srle<%d,%s,%d>(V)\n",
	   IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // logical left shift by n elements, filling in zeros
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  slle(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("slle<%d,%s,%d>(V)\n",
	   IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // extraction of lowest element
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  elem0(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("elem0<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return T(0);
  }

  // alignr (with number of elements in imm):
  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  alignre(const SIMDVec<T, SIMD_WIDTH> &,
	  const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("alignre<%d,%s,%d>(V,V)\n",
	   IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // swizzle (AoS to SoA), in-place, with template parameter N
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<N, T, SIMD_WIDTH> &,
	  SIMDVec<T, SIMD_WIDTH> *const,
	  Integer<N>,
	  TypeIsIntSize<T>)
  {
    printf("swizzle<%d,%s,%d>(V,V*)\n",
	   N, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  }

  // ifelse
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  ifelse(const SIMDVec<T, SIMD_WIDTH> &,
	 const SIMDVec<T, SIMD_WIDTH> &,
	 const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("ifelse<%s,%d>(V,V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // compare <
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmplt(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("cmplt<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // compare <=
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmple(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("cmple<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // compare ==
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpeq(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("cmpeq<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // compare >=
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpge(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("cmpge<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // compare >
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpgt(const SIMDVec<T, SIMD_WIDTH> &,
	const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("cmpgt<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // compare !=
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  cmpneq(const SIMDVec<T, SIMD_WIDTH> &,
	 const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("cmpneq<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // bitwise and
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  and(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("and<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // bitwise or
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  or(const SIMDVec<T, SIMD_WIDTH> &,
     const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("or<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // bitwise andnot: (not a) and b 
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  andnot(const SIMDVec<T, SIMD_WIDTH> &,
	 const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("andnot<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // bitwise xor
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  xor(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("xor<%s,%d>(V,V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }


  // bitwise not (work-arounds required)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  not(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("not<%s,%d>(V)\n",
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  avg(const SIMDVec<T, SIMD_WIDTH> &,
      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("avg<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // test if all bits are zeros (false)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("test_all_zeros<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return 0;
  }

  // test if all bits are ones (true)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("test_all_ones<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return 0;
  }

  // TODO: not level 0, should this be in the sandbox?
  // negate a, where b is negative, SIMDFloat only
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  sign(const SIMDVec<T, SIMD_WIDTH> &,
       const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("sign<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // TODO: not level 0, should this be in the sandbox?
  // Computes elementwise absolute difference of vectors
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  absDiff(const SIMDVec<T, SIMD_WIDTH> &,
	  const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("absDiff<%s,%d>(V,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // reverse order
  template <typename T, int SIMD_WIDTH>
  static SIMDVec<T, SIMD_WIDTH>
  reverse(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("reverse<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // 27. Aug 22 (Jonas Keller): added msb2int
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("msb2int<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return 0;
  }

  // 09. Oct 22 (Jonas Keller): added int2msb
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2msb(const uint64_t, OutputType<T>, Integer<SIMD_WIDTH>)
  {
    printf("int2msb<%s,%d>(U64)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // 09. Oct 22 (Jonas Keller): added int2bits
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  int2bits(const uint64_t, OutputType<T>, Integer<SIMD_WIDTH>)
  {
    printf("int2bits<%s,%d>(U64)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }
} // namespace base
} // namespace internal
} // namespace ns_simd

#endif // _SIMD_VEC_BASE_IMPL_SANDBOX_H_
