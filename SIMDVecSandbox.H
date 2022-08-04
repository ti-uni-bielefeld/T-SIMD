// ===========================================================================
// 
// SIMDVecSandbox.H --
// test template functions for SIMDVec and associated function templates
// these functions just print out template parameters and some arguments
// contains all "level-0" functions (SIMDVec*16/32.H)
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

// 04. Aug 22 (Jonas Keller): added masked functions

#ifndef _SIMD_VEC_SANDBOX_H_
#define _SIMD_VEC_SANDBOX_H_

// determine NATIVE_SIMD_WIDTH: in this case we take a default value
#define NATIVE_SIMD_WIDTH 16

#include <cstdint>
#include <stdio.h>
#include "SIMDVec.H"

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
    SIMDVec() {}
  };

  // ===========================================================================
  // generic template for SIMDMask
  // ===========================================================================

  template <typename T, int SIMD_WIDTH>
  class SIMDMask {
  public:
    SIMDMask() {}
    SIMDMask(const uint64_t &) {}
    operator uint64_t() const { return 0; }
  };

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
  setzero()
  {
    printf("setzero<%s,%d>()\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  set1(T)
  {
    printf("set1<%s,%d>()\n", 
	   SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  load(const T *const p)
  {
    printf("load<%d,%s>(%p)\n", SIMD_WIDTH, SIMDTypeInfo<T>::name(), (void*) p);
    return SIMDVec<T,SIMD_WIDTH>();  
  }

  // template argument order reversed so that T can be omitted in instantiation
  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  loadu(const T *const p)
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
  streamstore(T *const p, 
	      const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("streamstore<%s,%d>(%p,V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*) p);
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
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH> 
  unpack(const SIMDVec<T, SIMD_WIDTH> &,
	 const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("unpack<PART=%d,NUM_ELEMS=%d,%s,%d>(V,V)\n",
	   PART, NUM_ELEMS, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();  
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

  // unzip (inverse of zip)
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, SIMD_WIDTH>,
	const SIMDVec<T, SIMD_WIDTH>,
	SIMDVec<T, SIMD_WIDTH> &,
	SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("unzip<NUM_ELEMS=%d,%s,%d>(V,V,V&,V&)\n",
	   NUM_ELEMS, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
  }

  // unifies packs and packus, depending on Tout
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  packs(const SIMDVec<Tin, SIMD_WIDTH> &,
	const SIMDVec<Tin, SIMD_WIDTH> &)
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

  // conversion without changes in the number of vector elements
  // saturated version
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH> 
  cvts(const SIMDVec<Tin, SIMD_WIDTH> &)
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

  // swizzle table (empty)
  template <int N, typename T, int SIMD_WIDTH>
  struct SwizzleTable {};

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

  // reverse order
  template <typename T, int SIMD_WIDTH>
  static SIMDVec<T, SIMD_WIDTH>
  reverse(const SIMDVec<T, SIMD_WIDTH> &)
  {
    printf("reverse<%s,%d>(V)\n", SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  // ===========================================================================
  // masked functions
  // ===========================================================================

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ifelse(const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_ifelse<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ifelsezero(const SIMDMask<T, SIMD_WIDTH>&,
                  const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_ifelsezero<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDInt, SIMD_WIDTH>&,
            const SIMDMask<SIMDFloat, SIMD_WIDTH>&,
            const SIMDVec<SIMDFloat, SIMD_WIDTH>&)
  {
    printf("mask_cvts<%s,%s,%d>(V,M,V)\n",
     SIMDTypeInfo<SIMDInt>::name(), SIMDTypeInfo<SIMDFloat>::name(), SIMD_WIDTH);
    return SIMDVec<SIMDInt,SIMD_WIDTH>();
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDFloat, SIMD_WIDTH>&,
             const SIMDVec<SIMDFloat, SIMD_WIDTH>&)
  {
    printf("maskz_cvts<%s,%s,%d>(M,V)\n",
     SIMDTypeInfo<SIMDInt>::name(), SIMDTypeInfo<SIMDFloat>::name(), SIMD_WIDTH);
    return SIMDVec<SIMDInt,SIMD_WIDTH>();
  }
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  mask_cvts(const SIMDVec<SIMDFloat, SIMD_WIDTH>&,
            const SIMDMask<SIMDInt, SIMD_WIDTH>&,
            const SIMDVec<SIMDInt, SIMD_WIDTH>&)
  {
    printf("mask_cvts<%s,%s,%d>(V,M,V)\n",
     SIMDTypeInfo<SIMDFloat>::name(), SIMDTypeInfo<SIMDInt>::name(), SIMD_WIDTH);
    return SIMDVec<SIMDFloat,SIMD_WIDTH>();
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  maskz_cvts(const SIMDMask<SIMDInt, SIMD_WIDTH>&,
             const SIMDVec<SIMDInt, SIMD_WIDTH>&)
  {
    printf("maskz_cvts<%s,%s,%d>(M,V)\n",
     SIMDTypeInfo<SIMDFloat>::name(), SIMDTypeInfo<SIMDInt>::name(), SIMD_WIDTH);
    return SIMDVec<SIMDFloat,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_set1(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const T&)
  {
    printf("mask_set1<%s,%d>(V,M,T)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_set1(const SIMDMask<T, SIMD_WIDTH>&,
             const T&)
  {
    printf("maskz_set1<%s,%d>(M,T)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_load(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const T *const p)
  {
    printf("mask_load<%s,%d>(V,M,%p)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*)p);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_load(const SIMDMask<T, SIMD_WIDTH>&,
             const T *const p)
  {
    printf("maskz_load<%s,%d>(M,%p)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*)p);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_loadu(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const T *const p)
  {
    printf("mask_loadu<%s,%d>(V,M,%p)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*)p);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_loadu(const SIMDMask<T, SIMD_WIDTH>&,
             const T *const p)
  {
    printf("maskz_loadu<%s,%d>(M,%p)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*)p);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_store(T *const p,
             const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_store<%s,%d>(%p,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*)p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  mask_storeu(T *const p,
              const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_storeu<%s,%d>(%p,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, (void*)p);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_add(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_add<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_add(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_add<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_adds(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_adds<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_adds(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_adds<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_sub(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_sub<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_sub(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_sub<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_subs(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_subs<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_subs(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_subs<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_mul(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_mul<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_mul(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_mul<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_div(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_div<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_div(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_div<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_ceil(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_ceil<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_ceil(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_ceil<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_floor(const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_floor<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_floor(const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_floor<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_round(const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_round<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_round(const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_round<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_truncate(const SIMDVec<T, SIMD_WIDTH>&,
                const SIMDMask<T, SIMD_WIDTH>&,
                const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_truncate<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_truncate(const SIMDMask<T, SIMD_WIDTH>&,
                 const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_truncate<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_rcp(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_rcp<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_rcp(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_rcp<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_rsqrt(const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_rsqrt<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_rsqrt(const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_rsqrt<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_sqrt(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_sqrt<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_sqrt(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_sqrt<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_abs(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_abs<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_abs(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_abs<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_and(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_and<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_and(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_and<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_or(const SIMDVec<T, SIMD_WIDTH>&,
          const SIMDMask<T, SIMD_WIDTH>&,
          const SIMDVec<T, SIMD_WIDTH>&,
          const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_or<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_or(const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_or<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_andnot(const SIMDVec<T, SIMD_WIDTH>&,
              const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_andnot<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_andnot(const SIMDMask<T, SIMD_WIDTH>&,
               const SIMDVec<T, SIMD_WIDTH>&,
               const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_andnot<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_xor(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_xor<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_xor(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_xor<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_not(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_not<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_not(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_not<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_neg(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_neg<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_neg(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_neg<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_min(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_min<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_min(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_min<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_max(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_max<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_max(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_max<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_div2r0(const SIMDVec<T, SIMD_WIDTH>&,
              const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_div2r0<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_div2r0(const SIMDMask<T, SIMD_WIDTH>&,
               const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_div2r0<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_div2rd(const SIMDVec<T, SIMD_WIDTH>&,
              const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_div2rd<%s,%d>(V,M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_div2rd(const SIMDMask<T, SIMD_WIDTH>&,
               const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_div2rd<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_srai(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_srai<%d,%s,%d>(V,M,V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_srai(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_srai<%d,%s,%d>(M,V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_srli(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_srli<%d,%s,%d>(V,M,V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_srli(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_srli<%d,%s,%d>(M,V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_slli(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_slli<%d,%s,%d>(V,M,V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_slli(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_slli<%d,%s,%d>(M,V)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hadd(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_hadd<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hadd(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_hadd<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hadds(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_hadds<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hadds(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_hadds<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hsub(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_hsub<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hsub(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_hsub<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_hsubs(const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_hsubs<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_hsubs(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_hsubs<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmplt(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_cmplt<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmple(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_cmple<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpeq(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_cmpeq<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpgt(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_cmpgt<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpge(const SIMDMask<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&,
             const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_cmpge<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  mask_cmpneq(const SIMDMask<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&,
              const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_cmpneq<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  mask_avg(const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDMask<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&,
           const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_avg<%s,%d>(V,M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  maskz_avg(const SIMDMask<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&,
            const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("maskz_avg<%s,%d>(M,V,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDVec<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_zeros(const SIMDMask<T, SIMD_WIDTH>&,
                      const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_test_all_zeros<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return 0;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  mask_test_all_ones(const SIMDMask<T, SIMD_WIDTH>&,
                     const SIMDVec<T, SIMD_WIDTH>&)
  {
    printf("mask_test_all_ones<%s,%d>(M,V)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return 0;
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kadd(const SIMDMask<T, SIMD_WIDTH>&,
       const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kadd<%s,%d>(M,M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kand(const SIMDMask<T, SIMD_WIDTH>&,
       const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kand<%s,%d>(M,M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kandn(const SIMDMask<T, SIMD_WIDTH>&,
        const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kandn<%s,%d>(M,M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kor(const SIMDMask<T, SIMD_WIDTH>&,
      const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kor<%s,%d>(M,M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kxor(const SIMDMask<T, SIMD_WIDTH>&,
       const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kxor<%s,%d>(M,M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kxnor(const SIMDMask<T, SIMD_WIDTH>&,
        const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kxnor<%s,%d>(M,M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  knot(const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("knot<%s,%d>(M)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <unsigned int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftri(const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kshiftri<%d, %s,%d>(M,%d)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH, IMM);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <unsigned int IMM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftli(const SIMDMask<T, SIMD_WIDTH>&)
  {
    printf("kshiftli<%d, %s,%d>(M,%d)\n",
     IMM, SIMDTypeInfo<T>::name(), SIMD_WIDTH, IMM);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftri(const SIMDMask<T, SIMD_WIDTH>&,
           const uint64_t count)
  {
    printf("kshiftri<%s,%d>(M,%d)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, count);
    return SIMDMask<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDMask<T, SIMD_WIDTH>
  kshiftli(const SIMDMask<T, SIMD_WIDTH>&,
           const uint64_t count)
  {
    printf("kshiftli<%s,%d>(M,%d)\n",
     SIMDTypeInfo<T>::name(), SIMD_WIDTH, count);
    return SIMDMask<T,SIMD_WIDTH>();
  }
}

#endif
