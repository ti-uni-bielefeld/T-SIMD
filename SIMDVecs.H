// ===========================================================================
// 
// SIMDVecs.H --
// multiple SIMDVec in a template class plus some functions 
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

#ifndef _SIMD_VECS_H_
#define _SIMD_VECS_H_

#include "SIMDVecExt.H"

namespace ns_simd {

  // ===========================================================================
  // SIMDVecs: for functions operating on multiple SIMDVec
  // ===========================================================================
  
  // class to compute number of input and output vectors of conversion functions
  // (convert, packs, extend, f*)
  template <typename Tout, typename Tin>
  class NumSIMDVecs
  {	    
  public:
    enum { 
      in  = (sizeof(Tout) < sizeof(Tin)) ? (sizeof(Tin) / sizeof(Tout)) : 1,
      out = (sizeof(Tout) > sizeof(Tin)) ? (sizeof(Tout) / sizeof(Tin)) : 1
    };
  };
  
  // multiple SIMDVec
  template <int NUM, typename T, int SIMD_WIDTH>
  class SIMDVecs
  {
  public:
    enum { 
      vectors  = NUM,
      elements = NUM * SIMDVec<T,SIMD_WIDTH>::elements,
      bytes    = NUM * SIMDVec<T,SIMD_WIDTH>::bytes
    };
    SIMDVec<T,SIMD_WIDTH> vec[NUM];
  };
  
  // wrapper functions working on SIMDVecs

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  convert(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
			 Tin,SIMD_WIDTH> &inVecs,
	  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,
		   Tout,SIMD_WIDTH> &outVecs)
  {
    convert(inVecs.vec, outVecs.vec);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivmul(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsNum,
	  const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsDenom,
	  double fac,
	  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,
	           Tout,SIMD_WIDTH> &vecsOut)
  {
    fdivmul(vecsNum.vec, vecsDenom.vec, fac, vecsOut.vec);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmul(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsIn,
	  double fac,
	  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,
	           Tout,SIMD_WIDTH> &vecsOut)
  {
    fmul(vecsIn.vec, fac, vecsOut.vec);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  faddmul(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsIn,
	  double off,
	  double fac,
	  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,
	           Tout,SIMD_WIDTH> &vecsOut)
  {
    faddmul(vecsIn.vec, off, fac, vecsOut.vec);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmuladd(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsIn,
	  double fac,
	  double off,
	  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,
	           Tout,SIMD_WIDTH> &vecsOut)
  {
    fmuladd(vecsIn.vec, fac, off, vecsOut.vec);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fwaddmul(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsIn1,
	   const SIMDVecs<NumSIMDVecs<Tout,Tin>::in,
	                 Tin,SIMD_WIDTH> &vecsIn2,
	  double w,
	  double fac,
	  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,
	           Tout,SIMD_WIDTH> &vecsOut)
  {
    fwaddmul(vecsIn1.vec, vecsIn2.vec, w, fac, vecsOut.vec);
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  load(const T *const p, SIMDVecs<NUM,T,SIMD_WIDTH> &inVecs)
  {
    load(p, inVecs.vec, inVecs.vectors);
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  loadu(const T *const p, SIMDVecs<NUM,T,SIMD_WIDTH> &inVecs)
  {
    loadu(p, inVecs.vec, inVecs.vectors);
  }
  
  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  store(T *const p, const SIMDVecs<NUM,T, SIMD_WIDTH> &outVecs)
  {
    store(p, outVecs.vec, outVecs.vectors);
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  storeu(T *const p, const SIMDVecs<NUM,T, SIMD_WIDTH> &outVecs)
  {
    storeu(p, outVecs.vec, outVecs.vectors);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  packs(const SIMDVecs<NumSIMDVecs<Tout,Tin>::in, Tin, SIMD_WIDTH> &a)
  {
    return packs<Tout>(a.vec);
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  extend(const SIMDVec<Tin,SIMD_WIDTH> &vIn,
	 SIMDVecs<NumSIMDVecs<Tout,Tin>::out, Tout,SIMD_WIDTH> &vOut)
  {
    extend(vIn, vOut.vec);
  }
  
  // with SIMDVecs as return value
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE
  SIMDVecs<NumSIMDVecs<Tout,Tin>::out,Tout,SIMD_WIDTH> 
  extend(const SIMDVec<Tin,SIMD_WIDTH> &vIn)
  {
    SIMDVecs<NumSIMDVecs<Tout,Tin>::out,Tout,SIMD_WIDTH> vOut;
    extend(vIn, vOut);
    return vOut;
  }

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<N, T, SIMD_WIDTH> &t,
	  SIMDVecs<N, T, SIMD_WIDTH> &v)
  {
    swizzle(t, v.vec);
  }

  // inRows passed by-value to allow in-place transpose
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> inRows,
	    SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> &outRows)
  {
    transpose(inRows.vec, outRows.vec);
  }

  // 10. Feb 16 (rm): this was originally called transpose(), however,
  // icc 15 (but not gcc 4.8) complained that for the special case of
  // NUM_TRANSPOSE_ROWS = SIMD_WIDTH / sizeof(T) it is not clear
  // whether to use this function or transpose() above; I therefore
  // renamed it to transposePartial
  // inRows passed by-value to allow in-place transpose
  template <int NUM_TRANSPOSE_ROWS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transposePartial(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> inRows,
		   SIMDVecs<NUM_TRANSPOSE_ROWS,T,SIMD_WIDTH> &outRows)
  {
    transpose<NUM_TRANSPOSE_ROWS>(inRows.vec, outRows.vec);
  }

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle2(SIMDVecs<2*N, T, SIMD_WIDTH> &v)
  {
    swizzle2(v.vec);
  }

  // inRows passed by-value to allow in-place transpose
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose2(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> inRows,
	     SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> &outRows)
  {
    transpose2(inRows.vec, outRows.vec);
  }
 
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  unswizzle2(SIMDVecs<2*N, T, SIMD_WIDTH> &v)
  {
    unswizzle2(v.vec);
  }
 
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hadd(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> &v)
  {
    return hadd(v.vec);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hadds(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> &v)
  {
    return hadds(v.vec);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hsub(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> &v)
  {
    return hsub(v.vec);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hsubs(const SIMDVecs<(SIMD_WIDTH/sizeof(T)),T,SIMD_WIDTH> &v)
  {
    return hsubs(v.vec);
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  add(const SIMDVecs<NUM, T, SIMD_WIDTH> &a,
      const SIMDVecs<NUM, T, SIMD_WIDTH> &b)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = add(a.vec[v], b.vec[v]);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  adds(const SIMDVecs<NUM, T, SIMD_WIDTH> &a,
      const SIMDVecs<NUM, T, SIMD_WIDTH> &b)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = adds(a.vec[v], b.vec[v]);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  sub(const SIMDVecs<NUM, T, SIMD_WIDTH> &a,
      const SIMDVecs<NUM, T, SIMD_WIDTH> &b)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = sub(a.vec[v], b.vec[v]);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  subs(const SIMDVecs<NUM, T, SIMD_WIDTH> &a,
      const SIMDVecs<NUM, T, SIMD_WIDTH> &b)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = subs(a.vec[v], b.vec[v]);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  min(const SIMDVecs<NUM, T, SIMD_WIDTH> &a,
      const SIMDVecs<NUM, T, SIMD_WIDTH> &b)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = min(a.vec[v], b.vec[v]);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  max(const SIMDVecs<NUM, T, SIMD_WIDTH> &a,
      const SIMDVecs<NUM, T, SIMD_WIDTH> &b)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = max(a.vec[v], b.vec[v]);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  setzero()
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = setzero<T,SIMD_WIDTH>();
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  setzero(SIMDVecs<NUM, T, SIMD_WIDTH> &res)
  {
    for (int v = 0; v < NUM; v++) res.vec[v] = setzero<T,SIMD_WIDTH>();
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVecs<NUM, T, SIMD_WIDTH>
  set1(T a)
  {
    SIMDVecs<NUM, T, SIMD_WIDTH> res;
    for (int v = 0; v < NUM; v++) res.vec[v] = set1<T,SIMD_WIDTH>(a);
    return res;
  }

  template <int NUM, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  set1(SIMDVecs<NUM, T, SIMD_WIDTH> &res, T a)
  {
    for (int v = 0; v < NUM; v++) res.vec[v] = set1<T,SIMD_WIDTH>(a);
  }

  // TODO: add more SIMDVecs functions if needed

} // namespace

#endif
