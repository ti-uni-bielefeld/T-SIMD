// ===========================================================================
// 
// SIMDVecExt.H --
// extension commands combining multiple 1st-level vector template functions
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

#ifndef _SIMD_VEC_EXT_H_
#define _SIMD_VEC_EXT_H_

#include "Compare2Type.H"
#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>

namespace ns_simd {

  // ===========================================================================
  // helper functions for templates converting from Tin to Tout
  // ===========================================================================

  // numInputSIMDVecs() and numOutputSIMDVecs() assume that either 
  // - a single vector is extended into multiple vectors or
  // - multiple vectors are packed into a single vector
  //
  // numSIMDVecsElements encodes the number of elements in *all* input / all
  // output vectors
  //
  // numSIMDVecElements encodes the number of elements in each vector
  // (or use SIMDVec::elements instead)
  //
  // removed: numInputSIMDVecElements/numOutputSIMDVecElements encode
  // the number of elements in *each* input / output vector

  // TODO: in C++11 we could make these functions constexpr such that
  // TODO: they can be used to compute template arguments
  // TODO: since code should also run on C++98, we use macros instead

#define NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH) (SIMD_WIDTH / sizeof(T))

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE int
  numSIMDVecElements()
  {
    return NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH);
  }

#define NUM_INPUT_SIMDVECS(TOUT,TIN)					\
  ((sizeof(TOUT) < sizeof(TIN)) ? (sizeof(TIN) / sizeof(TOUT)) : 1)
  
  template <typename Tout, typename Tin>
  static SIMD_INLINE int 
  numInputSIMDVecs()
  {
    return NUM_INPUT_SIMDVECS(Tout,Tin);
  }

#define NUM_OUTPUT_SIMDVECS(TOUT,TIN)		\
  ((sizeof(TOUT) > sizeof(TIN)) ? (sizeof(TOUT) / sizeof(TIN)) : 1)

  template <typename Tout, typename Tin>
  static SIMD_INLINE int
  numOutputSIMDVecs()
  {
    return NUM_OUTPUT_SIMDVECS(Tout,Tin);
  }
  
#define NUM_SIMDVECS_ELEMENTS(TOUT,TIN,SIMD_WIDTH)	\
  ((sizeof(TOUT) > sizeof(TIN)) ?			\
   NUM_SIMDVEC_ELEMENTS(TIN,SIMD_WIDTH) :		\
   NUM_SIMDVEC_ELEMENTS(TOUT,SIMD_WIDTH))
   
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE int
  numSIMDVecsElements()
  {
    return NUM_SIMDVECS_ELEMENTS(Tout,Tin,SIMD_WIDTH);
  }
  
  // 22. Sep 22 (Jonas Keller): this was implemented using a constexpr function from
  // https://stackoverflow.com/questions/23781506/compile-time-computing-of-number-of-bits-needed-to-encode-n-different-states
  // but C++98 does not support constexpr, so we use template metaprogramming instead
  // from https://hbfs.wordpress.com/2016/03/22/log2-with-c-metaprogramming/
  template <unsigned int x> struct floorlog2 {
    static const unsigned int value = 1 + floorlog2<x / 2>::value;
  };
  template <> struct floorlog2<1> { static const unsigned int value = 0; };

  // determine NATIVE_SIMD_REG_COUNT
  // https://stackoverflow.com/questions/62419256/how-can-i-determine-how-many-avx-registers-my-processor-has
#ifdef __x86_64__
#ifdef __AVX512VL__
#define NATIVE_SIMD_REG_COUNT 32
#else
#define NATIVE_SIMD_REG_COUNT 16
#endif
#else
#define NATIVE_SIMD_REG_COUNT 8
#endif

  // ===========================================================================
  // print functions (for tests)
  // ===========================================================================
  
  // 04. Aug 22 (Jonas Keller):
  // removed treatZero(), not needed anymore because of change below
  //
  // // integer types don't have negative zero
  // template <typename T>
  // static SIMD_INLINE T
  // treatZero(T in)
  // {
  //   return in;
  // }
  // 
  // // SIMDFloat: map -0.0f to 0.0f
  // static SIMD_INLINE SIMDFloat
  // treatZero(SIMDFloat in)
  // {
  //   return (in == -0.0f) ? 0.0f : in;
  // }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  fprint(FILE *f, const char *format, const SIMDVec<T,SIMD_WIDTH> &vec)
  {
    // buffer
    // 19. Jul 16 (rm)
    const int elems = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); //SIMD_WIDTH/sizeof(T)
    // T buf[SIMD_WIDTH];
    T buf[elems];
    // store vector (unaligned, not time-critical)
    storeu(buf, vec);
    // print elements of vector to f
    for (int i = 0; i < elems; i++)
      // 04. Aug 22 (Jonas Keller):
      // removed mapping from -0.0f to 0.0f,
      // for debugging you want to see -0.0f
      // fprintf(f, format, treatZero(buf[i]));
      fprintf(f, format, buf[i]);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  print(const char *format, const SIMDVec<T,SIMD_WIDTH> &vec)
  {
    fprint(stdout, format, vec);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  fprint(FILE *f,
	 const char *format, const char *separator, 
	 const SIMDVec<T,SIMD_WIDTH> &vec)
  {
    // 09. Jan 23 (Jonas Keller): used std::string instead of strcpy and strcat
    // to avoid potential buffer overflows
    //char fmtSep[256];
    //strcat(strcpy(fmtSep, format), separator);
    std::string fmtSep = std::string(format) + std::string(separator);
    fprint(f, fmtSep.c_str(), vec);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  print(const char *format, const char *separator, 
	const SIMDVec<T,SIMD_WIDTH> &vec)
  {
    fprint(stdout, format, separator, vec);
  }

  // ===========================================================================
  // multi-vector store and load
  // ===========================================================================

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  load(const T *const p, SIMDVec<T, SIMD_WIDTH> *inVecs, int numInVecs)
  {
    const int vecElemsIn 
      = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); 
    for (int vi = 0, off = 0; vi < numInVecs; vi++, off += vecElemsIn)
      inVecs[vi] = load<SIMD_WIDTH>(p + off);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  loadu(const T *const p, SIMDVec<T, SIMD_WIDTH> *inVecs, int numInVecs)
  {
    const int vecElemsIn 
      = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); 
    for (int vi = 0, off = 0; vi < numInVecs; vi++, off += vecElemsIn)
      inVecs[vi] = loadu<SIMD_WIDTH>(p + off);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  store(T *const p, const SIMDVec<T, SIMD_WIDTH> *outVecs, int numOutVecs)
  {
    const int vecElemsOut 
      = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); 
    for (int vo = 0, off = 0; vo < numOutVecs; vo++, off += vecElemsOut)
      store(p + off, outVecs[vo]);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  storeu(T *const p, const SIMDVec<T, SIMD_WIDTH> *outVecs, int numOutVecs)
  {
    const int vecElemsOut 
      = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); 
    for (int vo = 0, off = 0; vo < numOutVecs; vo++, off += vecElemsOut)
      storeu(p + off, outVecs[vo]);
  }

  // store single vector multiple times
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  store(T *const p, const SIMDVec<T, SIMD_WIDTH> &outVec, int numOutVecs)
  {
    const int vecElemsOut 
      = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); 
    for (int vo = 0, off = 0; vo < numOutVecs; vo++, off += vecElemsOut)
      store(p + off, outVec);
  }

  // store single vector multiple times (unaligned)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  storeu(T *const p, const SIMDVec<T, SIMD_WIDTH> &outVec, int numOutVecs)
  {
    const int vecElemsOut 
      = NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH); 
    for (int vo = 0, off = 0; vo < numOutVecs; vo++, off += vecElemsOut)
      storeu(p + off, outVec);
  }

  // -------------------- different store functions ----------------------------

  template <typename T, int SIMD_WIDTH>
  class Store
  {
  public:
    static SIMD_INLINE void
    _store(T *const p, const SIMDVec<T, SIMD_WIDTH> &outVec)
    {
      return store(p, outVec);
    }
  };

  template <typename T, int SIMD_WIDTH>
  class StoreU
  {
  public:
    static SIMD_INLINE void
    _store(T *const p, const SIMDVec<T, SIMD_WIDTH> &outVec)
    {
      return storeu(p, outVec);
    }
  };

  // ---------------------------------------------------------------------------
  // Meta Template Class Store16
  // used to store matrix after transposed with Transpose<Unpack16>
  //
  // TODO: Currently storing complete quadratic matrix. Integrate numOutVecs?
  //
  // gcc error with inline template function: inlining failed in call to
  // always_inline ‘void ns_simd::storeu16(..) [..]’: recursive inlining
  // ---------------------------------------------------------------------------

  template< template <typename, int> class Store,
      typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW, int STORE_STOP,
      int STORE_WIDTH, int SRC_OFF, int DST_OFF>
  class Store16
  {

  public:
    static SIMD_INLINE void
    _store16(T *const p, const SIMDVec<T, SIMD_WIDTH> *outVecs)
    {
      // printf("STORE_WIDTH=%d, SRC_OFFSET=%d, DST_OFFSET=%d\n",
      // STORE_WIDTH, SRC_OFFSET, DST_OFFSET);
      Store16<Store, T, SIMD_WIDTH,
          NUMROWS, ROW, STORE_STOP,
          STORE_WIDTH / 2, SRC_OFF,
          2 * DST_OFF>::_store16(p, outVecs);
      Store16<Store, T, SIMD_WIDTH,
          NUMROWS, ROW, STORE_STOP,
          STORE_WIDTH / 2, SRC_OFF + SIMD_WIDTH / STORE_WIDTH,
          2 * DST_OFF + STORE_STOP>::_store16(p, outVecs);
    }
  };

  template< template <typename, int> class Store,
      typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW, int STORE_STOP,
      int SRC_OFF, int DST_OFF>
  class Store16<Store, T, SIMD_WIDTH,
      NUMROWS, ROW, STORE_STOP,
      16, SRC_OFF, DST_OFF>
  {
    enum { STEP = SIMD_WIDTH / 16 };
    enum { VO = SRC_OFF + ROW * STEP };
    enum { OFF = (DST_OFF + ROW) * NUMROWS };

  public:
    static SIMD_INLINE void
    _store16(T *const p, const SIMDVec<T, SIMD_WIDTH> *outVecs)
    {
      // printf("VO=%d\n", OFF=%d\n", VO, OFF);
      Store<T, SIMD_WIDTH>::_store(p + OFF, outVecs[VO]);
      Store16<Store, T, SIMD_WIDTH,
          NUMROWS, ROW + 1, STORE_STOP,
          16, SRC_OFF, DST_OFF>::_store16(p, outVecs);
    }
  };

  template< template <typename, int> class Store,
      typename T, int SIMD_WIDTH,
      int NUMROWS, int STORE_STOP,
      int SRC_OFF, int DST_OFF>
  class Store16<Store, T, SIMD_WIDTH,
      NUMROWS, STORE_STOP, STORE_STOP,
      16, SRC_OFF, DST_OFF>
  {

  public:
    static SIMD_INLINE void
    _store16(T *const, const SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // -------------------- store16 functions ------------------------------------

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  store16(T *const p, const SIMDVec<T, SIMD_WIDTH> *outVecs)
  {
    const int numRows = SIMD_WIDTH / sizeof(T);
    const int storeStop = 16 / sizeof(T);
    Store16<Store, T, SIMD_WIDTH, numRows, 0, storeStop, SIMD_WIDTH, 0, 0>
        ::_store16(p, outVecs);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  storeu16(T *const p, const SIMDVec<T, SIMD_WIDTH> *outVecs)
  {
    const int numRows = SIMD_WIDTH / sizeof(T);
    const int storeStop = 16 / sizeof(T);
    Store16<StoreU, T, SIMD_WIDTH, numRows, 0, storeStop, SIMD_WIDTH, 0, 0>
        ::_store16(p, outVecs);
  }

  // ===========================================================================
  // copy (load and store)
  // ===========================================================================

  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE void
  load_store(const T *const src, T *const dst)
  {
    SIMDVec<T, SIMD_WIDTH> copy = load<SIMD_WIDTH>(src);
    store(dst, copy);
  }

  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE void
  loadu_store(const T *const src, T *const dst)
  {
    SIMDVec<T, SIMD_WIDTH> copy = loadu<SIMD_WIDTH>(src);
    store(dst, copy);
  }

  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE void
  load_storeu(const T *const src, T *const dst)
  {
    SIMDVec<T, SIMD_WIDTH> copy = load<SIMD_WIDTH>(src);
    storeu(dst, copy);
  }

  template <int SIMD_WIDTH, typename T>
  static SIMD_INLINE void
  loadu_storeu(const T *const src, T *const dst)
  {
    SIMDVec<T, SIMD_WIDTH> copy = loadu<SIMD_WIDTH>(src);
    storeu(dst, copy);
  }

  // ===========================================================================
  // generalized convert (using extend and packs)
  // ===========================================================================

  // here we distinguish between the three cases (type size comparison)
  // using the CompareTypes mechanisms (from Compare2Type.H);
  // the alternative would be to have branches for the 3 different cases,
  // however, all branches are compiled, even if the compiler can select
  // between the three different branches at compile time,
  // this may lead to problems, since packs and extend are not defined for
  // all combinations of input and output types, and are not defined for
  // the same combinations, thus compilation may fail since extend or packs
  // cannot be instatiated, even though they wouldn't actually be used

  // from\to (combines generalized packs, extend)
  //    SB B S W I F 
  // SB  x   x   x x
  //  B    x x x x x
  //  S  x x x   x x
  //  W        x x x 
  //  I  x x x x x x
  //  F  x x x x x x

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  convert(CompareLess,
	  const SIMDVec<Tin,SIMD_WIDTH> *const inVecs, 
	  SIMDVec<Tout,SIMD_WIDTH> *const outVecs)
  {
    *outVecs = packs<Tout>(inVecs);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  convert(CompareEqual,
	  const SIMDVec<Tin,SIMD_WIDTH> *const inVecs, 
	  SIMDVec<Tout,SIMD_WIDTH> *const outVecs)
  {
    extend(*inVecs, outVecs);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  convert(CompareGreater,
	  const SIMDVec<Tin,SIMD_WIDTH> *const inVecs, 
	  SIMDVec<Tout,SIMD_WIDTH> *const outVecs)
  {
    extend(*inVecs, outVecs);
  }

  // hub
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  convert(const SIMDVec<Tin,SIMD_WIDTH> *const inVecs, 
	  SIMDVec<Tout,SIMD_WIDTH> *const outVecs)
  {
    convert(CompareTypes<Tout,Tin>(), inVecs, outVecs);
  }

  // ===========================================================================
  // float-based operations on arbitrary input and output types
  // ===========================================================================

  // TODO: is there a way to fuse the three cases in fdivmul/...?

  // ---------------------------------------------------------------------------
  // divide then multiply with float constant in float arithmetic
  // ---------------------------------------------------------------------------

  // TODO: fdivmul: better fmuldiv = first multiply then divide?

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivmul(CompareLess,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsNum,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsDenom,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nIn = sizeof(Tin) / sizeof(Tout);
    const int fanIn = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> numF[fanIn], denomF[fanIn];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[nIn * fanIn];
    for (int i = 0, k = 0; i < nIn; i++) {
      extend(vecsNum[i], numF);
      extend(vecsDenom[i], denomF);
      for (int j = 0; j < fanIn; j++, k++)
	resF[k] = mul(div(numF[j], denomF[j]), facF);
    }
    *vecsOut = packs<Tout>(resF);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivmul(CompareGreater,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsNum,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsDenom,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nOut = sizeof(Tout) / sizeof(Tin);
    const int fanOut = sizeof(SIMDFloat) / sizeof(Tout);
    SIMDVec<SIMDFloat,SIMD_WIDTH> numF[nOut * fanOut], denomF[nOut * fanOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanOut];
    extend(*vecsNum, numF);
    extend(*vecsDenom, denomF);
    for (int i = 0, k = 0; i < nOut; i++) {
      for (int j = 0; j < fanOut; j++, k++)
	resF[j] = mul(div(numF[k], denomF[k]), facF);
      vecsOut[i] = packs<Tout>(resF);
    }
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivmul(CompareEqual,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsNum,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsDenom,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    // sizeof(Tout) == sizeof(Tin)
    const int fanInOut = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> numF[fanInOut], denomF[fanInOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanInOut];
    extend(*vecsNum, numF);
    extend(*vecsDenom, denomF);
    for (int j = 0; j < fanInOut; j++)
      resF[j] = mul(div(numF[j], denomF[j]), facF);
    *vecsOut = packs<Tout>(resF);
  }

  // hub
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivmul(const SIMDVec<Tin, SIMD_WIDTH> *const vecsNum,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsDenom,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    fdivmul(CompareTypes<Tout,Tin>(), vecsNum, vecsDenom, fac, vecsOut);
  }
  
  // ---------------------------------------------------------------------------
  // divide, apply multidimensional sigmoid and then multiply with float
  // constant in float arithmetic (derived from fdivmul)
  // sigmoid(x) = ((y(x,a)/(1-y(x,a)**4)**0.25))+1)/2)
  // y(x,a) = sum_d(a*w[d]*(x[d]-w0[d]))
  // a = -0.433 from fitting this to 1/(1+exp(y(x,1))
  // ---------------------------------------------------------------------------

  template <int DIM, int NVEC, typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivMsigmoidmul(CompareLess,
                  const SIMDVec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
                  const SIMDVec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC],
                  const double w[DIM], const double w0[DIM],
                  double fac,
                  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    const int nIn = sizeof(Tin) / sizeof(Tout);
    const int fanIn = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac/2.0f),
                                  oneF = set1<SIMDFloat,SIMD_WIDTH>(1.0f),
                                  wF[DIM], w0F[DIM], yF, y4F,
                                  numF[DIM][fanIn], denomF[DIM][fanIn],
                                  resF[nIn * fanIn];
    for (int d = 0; d < DIM; d++) {
      wF[d] = set1<SIMDFloat,SIMD_WIDTH>(-0.433f*w[d]);
      w0F[d] = set1<SIMDFloat,SIMD_WIDTH>(w0[d]);
    }
    //i: index of input vector
    //j: index of extended input vector
    //k: index of output vectors
    //TODO: sometimes i < nIn does not work with -O2 is always true?
    for (int i = 0, k = 0; i < nIn; /*i++*/) {
      for (int d = 0; d < DIM; d++) {
        extend(vecsNum[d][i], numF[d]);
        extend(vecsDenom[d][i], denomF[d]);
      }
      for (int j = 0; j < fanIn; j++, k++){
        yF = setzero<SIMDFloat,SIMD_WIDTH>();
        for (int d = 0; d < DIM; d++) {
          yF = add(yF, mul(wF[d],sub(div(numF[d][j], denomF[d][j]),w0F[d])));
        }
        y4F = mul(yF,yF);
        y4F = mul(y4F,y4F);
        resF[k] = mul(add(div(yF,sqrt(sqrt(add(oneF, y4F)))),oneF),facF);
      }
      i++;
    }
    *vecsOut = packs<Tout>(resF);
  }

  template <int DIM, int NVEC, typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivMsigmoidmul(CompareGreater,
                  const SIMDVec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
                  const SIMDVec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC],
                  const double w[DIM], const double w0[DIM],
                  double fac,
                  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
      const int nOut = sizeof(Tout) / sizeof(Tin);
      const int fanOut = sizeof(SIMDFloat) / sizeof(Tout);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac/2.0f),
                                  oneF = set1<SIMDFloat,SIMD_WIDTH>(1.0f),
                                  wF[DIM], w0F[DIM], yF, y4F,
                                  numF[DIM][nOut * fanOut],
                                  denomF[DIM][nOut * fanOut],
                                  resF[fanOut];
    for (int d = 0; d < DIM; d++) {
      wF[d] = set1<SIMDFloat,SIMD_WIDTH>(-0.433f*w[d]);
      w0F[d] = set1<SIMDFloat,SIMD_WIDTH>(w0[d]);
      extend(*vecsNum[d], numF[d]);
      extend(*vecsDenom[d], denomF[d]);
    }
    //i: index of output vector
    //j: index of partial output vectors
    //k: index of input vector
    for (int i = 0, k = 0; i < nOut; i++) {
      for (int j = 0; j < fanOut; j++, k++) {
        yF = setzero<SIMDFloat,SIMD_WIDTH>();
        for (int d = 0; d < DIM; d++) {
          yF = add(yF, mul(wF[d],sub(div(numF[d][k], denomF[d][k]),w0F[d])));
        }
        y4F = mul(yF,yF);
        y4F = mul(y4F,y4F);
        resF[j] = mul(add(div(yF,sqrt(sqrt(add(oneF, y4F)))),oneF),facF);
      }
      vecsOut[i] = packs<Tout>(resF);
    }
  }

  template <int DIM, int NVEC, typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivMsigmoidmul(CompareEqual,
                  const SIMDVec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
                  const SIMDVec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC],
                  const double w[DIM], const double w0[DIM],
                  double fac,
                  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac/2.0f),
                                  oneF = set1<SIMDFloat,SIMD_WIDTH>(1.0f),
                                  wF[DIM], w0F[DIM], yF, y4F;
    const int fanInOut = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> numF[DIM][fanInOut], denomF[DIM][fanInOut],
                                  resF[fanInOut];
    for (int d = 0; d < DIM; d++) {
      wF[d] = set1<SIMDFloat,SIMD_WIDTH>(-0.433f*w[d]);
      w0F[d] = set1<SIMDFloat,SIMD_WIDTH>(w0[d]);
      extend(*vecsNum[d], numF[d]);
      extend(*vecsDenom[d], denomF[d]);
    }
    //j: index of extended input/output vector
    for (int j = 0; j < fanInOut; j++){
      yF = setzero<SIMDFloat,SIMD_WIDTH>();
      for (int d = 0; d < DIM; d++) {
        yF = add(yF, mul(wF[d],sub(div(numF[d][j], denomF[d][j]),w0F[d])));
      }
      y4F = mul(yF,yF);
      y4F = mul(y4F,y4F);
      resF[j] = mul(add(div(yF,sqrt(sqrt(add(oneF, y4F)))),oneF),facF);
    }
    *vecsOut = packs<Tout>(resF);
  }
  
  // hub
  template <int DIM, int NVEC, typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fdivMsigmoidmul(const SIMDVec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
                  const SIMDVec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC],
                  const double w[DIM], const double w0[DIM],
                  double fac,
                  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    fdivMsigmoidmul<DIM, NVEC>(CompareTypes<Tout,Tin>(),
                               vecsNum, vecsDenom, w, w0, fac, vecsOut);
  }

  // ---------------------------------------------------------------------------
  // multiply with float constant in float arithmetic
  // ---------------------------------------------------------------------------

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmul(CompareLess,
       const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
       double fac,
       SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nIn = sizeof(Tin) / sizeof(Tout);
    const int fanIn = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[fanIn];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[nIn * fanIn];
    for (int i = 0, k = 0; i < nIn; i++) {
      extend(vecsIn[i], inF);
      for (int j = 0; j < fanIn; j++, k++)
	resF[k] = mul(inF[j], facF);
    }
    *vecsOut = packs<Tout>(resF);
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmul(CompareGreater, 
       const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
       double fac,
       SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nOut = sizeof(Tout) / sizeof(Tin);
    const int fanOut = sizeof(SIMDFloat) / sizeof(Tout);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[nOut * fanOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanOut];
    extend(*vecsIn, inF);
    for (int i = 0, k = 0; i < nOut; i++) {
      for (int j = 0; j < fanOut; j++, k++)
	resF[j] = mul(inF[k], facF);
      vecsOut[i] = packs<Tout>(resF);
    }
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmul(CompareEqual, 
       const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
       double fac,
       SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    // sizeof(Tout) == sizeof(Tin)
    const int fanInOut = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[fanInOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanInOut];
    extend(*vecsIn, inF);
    for (int j = 0; j < fanInOut; j++)
      resF[j] = mul(inF[j], facF);
    *vecsOut = packs<Tout>(resF);
  }

  // hub
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmul(const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
       double fac,
       SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    fmul(CompareTypes<Tout,Tin>(), vecsIn, fac, vecsOut);
  }

  // ---------------------------------------------------------------------------
  // add then multiply with float constant in float arithmetic
  // ---------------------------------------------------------------------------

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  faddmul(CompareLess,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double off,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> offF = set1<SIMDFloat,SIMD_WIDTH>(off);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nIn = sizeof(Tin) / sizeof(Tout);
    const int fanIn = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[fanIn];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[nIn * fanIn];
    for (int i = 0, k = 0; i < nIn; i++) {
      extend(vecsIn[i], inF);
      for (int j = 0; j < fanIn; j++, k++)
	resF[k] = mul(add(inF[j], offF), facF);
    }
    *vecsOut = packs<Tout>(resF);
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  faddmul(CompareGreater,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double off,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> offF = set1<SIMDFloat,SIMD_WIDTH>(off);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nOut = sizeof(Tout) / sizeof(Tin);
    const int fanOut = sizeof(SIMDFloat) / sizeof(Tout);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[nOut * fanOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanOut];
    extend(*vecsIn, inF);
    for (int i = 0, k = 0; i < nOut; i++) {
      for (int j = 0; j < fanOut; j++, k++)
	resF[j] = mul(add(inF[k], offF), facF);
      vecsOut[i] = packs<Tout>(resF);
    }
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  faddmul(CompareEqual,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double off,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> offF = set1<SIMDFloat,SIMD_WIDTH>(off);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    // sizeof(Tout) == sizeof(Tin)
    const int fanInOut = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[fanInOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanInOut];
    extend(*vecsIn, inF);
    for (int j = 0; j < fanInOut; j++)
      resF[j] = mul(add(inF[j], offF), facF);
    *vecsOut = packs<Tout>(resF);
  }
  
  // hub
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  faddmul(const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double off,
	  double fac,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    faddmul(CompareTypes<Tout,Tin>(), vecsIn, off, fac, vecsOut);
  }

  // ---------------------------------------------------------------------------
  // multiply then add with float constant in float arithmetic
  // ---------------------------------------------------------------------------

  // better for conversion of zero-centered data to unsigned pixel format

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmuladd(CompareLess,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double fac,
	  double off,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> offF = set1<SIMDFloat,SIMD_WIDTH>(off);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nIn = sizeof(Tin) / sizeof(Tout);
    const int fanIn = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[fanIn];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[nIn * fanIn];
    for (int i = 0, k = 0; i < nIn; i++) {
      extend(vecsIn[i], inF);
      for (int j = 0; j < fanIn; j++, k++)
	resF[k] = add(mul(inF[j], facF), offF);
    }
    *vecsOut = packs<Tout>(resF);
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmuladd(CompareGreater,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double fac,
	  double off,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> offF = set1<SIMDFloat,SIMD_WIDTH>(off);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nOut = sizeof(Tout) / sizeof(Tin);
    const int fanOut = sizeof(SIMDFloat) / sizeof(Tout);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[nOut * fanOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanOut];
    extend(*vecsIn, inF);
    for (int i = 0, k = 0; i < nOut; i++) {
      for (int j = 0; j < fanOut; j++, k++)
	resF[j] = add(mul(inF[k], facF), offF);
      vecsOut[i] = packs<Tout>(resF);
    }
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmuladd(CompareEqual,
	  const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double fac,
	  double off,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> offF = set1<SIMDFloat,SIMD_WIDTH>(off);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    // sizeof(Tout) == sizeof(Tin)
    const int fanInOut = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF[fanInOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanInOut];
    extend(*vecsIn, inF);
    for (int j = 0; j < fanInOut; j++)
      resF[j] = add(mul(inF[j], facF), offF);
    *vecsOut = packs<Tout>(resF);
  }
  
  // hub
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fmuladd(const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn,
	  double fac,
	  double off,
	  SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    fmuladd(CompareTypes<Tout,Tin>(), vecsIn, fac, off, vecsOut);
  }

  // ---------------------------------------------------------------------------
  // multiply with float constant in float arithmetic
  // ---------------------------------------------------------------------------

  // fac * [v2 + w * (v1 - v2)] = fac * [w * v1 + (1 - w) * v2], w in [0,1]
  // w: weight factor (in [0,1])
  // fac: scale factor

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fwaddmul(CompareLess,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn1,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn2,
	   double w,
	   double fac,
	   SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> wF = set1<SIMDFloat,SIMD_WIDTH>(w);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nIn = sizeof(Tin) / sizeof(Tout);
    const int fanIn = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF1[fanIn], inF2[fanIn];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[nIn * fanIn];
    for (int i = 0, k = 0; i < nIn; i++) {
      extend(vecsIn1[i], inF1);
      extend(vecsIn2[i], inF2);
      for (int j = 0; j < fanIn; j++, k++)
	resF[k] = mul(facF, add(inF2[j], mul(wF, sub(inF1[j], inF2[j]))));
    }
    *vecsOut = packs<Tout>(resF);
  }

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fwaddmul(CompareGreater,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn1,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn2,
	   double w,
	   double fac,
	   SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> wF = set1<SIMDFloat,SIMD_WIDTH>(w);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    const int nOut = sizeof(Tout) / sizeof(Tin);
    const int fanOut = sizeof(SIMDFloat) / sizeof(Tout);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF1[nOut * fanOut], inF2[nOut * fanOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanOut];
    extend(*vecsIn1, inF1);
    extend(*vecsIn2, inF2);
    for (int i = 0, k = 0; i < nOut; i++) {
      for (int j = 0; j < fanOut; j++, k++)
	resF[j] = mul(facF, add(inF2[k], mul(wF, sub(inF1[k], inF2[k]))));
      vecsOut[i] = packs<Tout>(resF);
    }
  }
  
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fwaddmul(CompareEqual,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn1,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn2,
	   double w,
	   double fac,
	   SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    // we assume that sizeof(Tout),sizeof(Tin) <= sizeof(SIMDFloat)
    typeSizeLEQ<Tin,SIMDFloat>();
    typeSizeLEQ<Tout,SIMDFloat>();
    SIMDVec<SIMDFloat,SIMD_WIDTH> wF = set1<SIMDFloat,SIMD_WIDTH>(w);
    SIMDVec<SIMDFloat,SIMD_WIDTH> facF = set1<SIMDFloat,SIMD_WIDTH>(fac);
    // sizeof(Tout) == sizeof(Tin)
    const int fanInOut = sizeof(SIMDFloat) / sizeof(Tin);
    SIMDVec<SIMDFloat,SIMD_WIDTH> inF1[fanInOut], inF2[fanInOut];
    SIMDVec<SIMDFloat,SIMD_WIDTH> resF[fanInOut];
    extend(*vecsIn1, inF1);
    extend(*vecsIn2, inF2);
    for (int j = 0; j < fanInOut; j++)
      resF[j] = mul(facF, add(inF2[j], mul(wF, sub(inF1[j], inF2[j]))));
    *vecsOut = packs<Tout>(resF);
  }

  // hub
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE void
  fwaddmul(const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn1,
	   const SIMDVec<Tin, SIMD_WIDTH> *const vecsIn2,
	   double w,
	   double fac,
	   SIMDVec<Tout, SIMD_WIDTH> *const vecsOut)
  {
    fwaddmul(CompareTypes<Tout,Tin>(), vecsIn1, vecsIn2, w, fac, vecsOut);
  }

  // ===========================================================================
  // horizontal add/adds/sub/subs: generic form for multiple vector inputs
  // ===========================================================================

  // TODO: is there an easy way to implement multivec horizontal min/max?
  // TODO: (Hackers delight: min/max via doz = hsubs?)

  // primary template
  // num: number of elements processed
  // i0, i1: indices of lowest elements of block
  template <typename T, int SIMD_WIDTH, int num, int i0, int i1>
  class Horizontal
  {
  public:

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> 
    _hadd(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hadd(Horizontal<T, SIMD_WIDTH, num/2, i0, i0 + num/4>::_hadd(v),
		  Horizontal<T, SIMD_WIDTH, num/2, i1, i1 + num/4>::_hadd(v));
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> 
    _hadds(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hadds(Horizontal<T, SIMD_WIDTH, num/2, i0, i0 + num/4>::_hadds(v),
		   Horizontal<T, SIMD_WIDTH, num/2, i1, i1 + num/4>::_hadds(v));
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> 
    _hsub(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hsub(Horizontal<T, SIMD_WIDTH, num/2, i0, i0 + num/4>::_hsub(v),
		  Horizontal<T, SIMD_WIDTH, num/2, i1, i1 + num/4>::_hsub(v));
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH> 
    _hsubs(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hsubs(Horizontal<T, SIMD_WIDTH, num/2, i0, i0 + num/4>::_hsubs(v),
		   Horizontal<T, SIMD_WIDTH, num/2, i1, i1 + num/4>::_hsubs(v));
    }

  };

  // partial specialization to end the recursion
  template <typename T, int SIMD_WIDTH, int i0, int i1>
  class Horizontal<T, SIMD_WIDTH, 2, i0, i1>
  {
  public:

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hadd(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hadd(v[i0], v[i1]);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hadds(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hadds(v[i0], v[i1]);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hsub(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hsub(v[i0], v[i1]);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hsubs(const SIMDVec<T,SIMD_WIDTH> *const v)
    {
      return hsubs(v[i0], v[i1]);
    }

  };

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hadd(const SIMDVec<T,SIMD_WIDTH> *const v)
  {
    return Horizontal<T, SIMD_WIDTH, 
		      SIMD_WIDTH / sizeof(T),
		      0, 
		      (SIMD_WIDTH / sizeof(T)) / 2>::_hadd(v);
  }
  
  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hadds(const SIMDVec<T,SIMD_WIDTH> *const v)
  {
    return Horizontal<T, SIMD_WIDTH, 
		      SIMD_WIDTH / sizeof(T),
		      0, 
		      (SIMD_WIDTH / sizeof(T)) / 2>::_hadds(v);
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hsub(const SIMDVec<T,SIMD_WIDTH> *const v)
  {
    return Horizontal<T, SIMD_WIDTH, 
		      SIMD_WIDTH / sizeof(T), 
		      0, 
		      (SIMD_WIDTH / sizeof(T)) / 2>::_hsub(v);
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
  hsubs(const SIMDVec<T,SIMD_WIDTH> *const v)
  {
    return Horizontal<T, SIMD_WIDTH, 
		      SIMD_WIDTH / sizeof(T), 
		      0, 
		      (SIMD_WIDTH / sizeof(T)) / 2>::_hsubs(v);
  }

  // ===========================================================================
  // horizontal operations (generic form for single vector input)
  // ===========================================================================

  // these operations are not fully parallel!

  // example: SIMD_WIDTH = 16, T = float
  // elem0(Horizontal1<float,16,2>::_hadd(v));
  //       u = Horizontal1<float,16,1>::_hadd(v);
  //           hadd(v, v)
  //       hadd(u, u)

  template <typename T, int SIMD_WIDTH, int NUM>
  class Horizontal1
  {
  public:

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hadd(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      SIMDVec<T,SIMD_WIDTH> u = Horizontal1<T, SIMD_WIDTH, NUM/2>::_hadd(v);
      return hadd(u, u);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hadds(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      SIMDVec<T,SIMD_WIDTH> u = Horizontal1<T, SIMD_WIDTH, NUM/2>::_hadds(v);
      return hadds(u, u);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hsub(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      SIMDVec<T,SIMD_WIDTH> u = Horizontal1<T, SIMD_WIDTH, NUM/2>::_hsub(v);
      return hsub(u, u);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hsubs(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      SIMDVec<T,SIMD_WIDTH> u = Horizontal1<T, SIMD_WIDTH, NUM/2>::_hsubs(v);
      return hsubs(u, u);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hmin(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return Horizontal1<T, SIMD_WIDTH, NUM/2>::_hmin(min(srle<NUM>(v), v));
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hmax(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return Horizontal1<T, SIMD_WIDTH, NUM/2>::_hmax(max(srle<NUM>(v), v));
    }

  };

  template <typename T, int SIMD_WIDTH>
  class Horizontal1<T, SIMD_WIDTH, 1>
  {
  public:

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hadd(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return hadd(v, v);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hadds(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return hadds(v, v);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hsub(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return hsub(v, v);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hsubs(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return hsubs(v, v);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hmin(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return min(srle<1>(v), v);
    }

    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _hmax(const SIMDVec<T,SIMD_WIDTH> &v)
    {
      return max(srle<1>(v), v);
    }

  };

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hadd(const SIMDVec<T,SIMD_WIDTH> &v)
  {
    return elem0(Horizontal1<T, SIMD_WIDTH, SIMD_WIDTH/sizeof(T)/2>::_hadd(v));
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hadds(const SIMDVec<T,SIMD_WIDTH> &v)
  {
    return elem0(Horizontal1<T, SIMD_WIDTH, SIMD_WIDTH/sizeof(T)/2>::_hadds(v));
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hsub(const SIMDVec<T,SIMD_WIDTH> &v)
  {
    return elem0(Horizontal1<T, SIMD_WIDTH, SIMD_WIDTH/sizeof(T)/2>::_hsub(v));
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hsubs(const SIMDVec<T,SIMD_WIDTH> &v)
  {
    return elem0(Horizontal1<T, SIMD_WIDTH, SIMD_WIDTH/sizeof(T)/2>::_hsubs(v));
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hmin(const SIMDVec<T,SIMD_WIDTH> &v)
  {
    return elem0(Horizontal1<T, SIMD_WIDTH, SIMD_WIDTH/sizeof(T)/2>::_hmin(v));
  }

  // function template
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE T
  hmax(const SIMDVec<T,SIMD_WIDTH> &v)
  {
    return elem0(Horizontal1<T, SIMD_WIDTH, SIMD_WIDTH/sizeof(T)/2>::_hmax(v));
  }

  // ===========================================================================
  // transpose
  // ===========================================================================

  // -------------------- different unpack functions ---------------------------

  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  class Unpack
  {
  public:
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _unpack(const SIMDVec<T, SIMD_WIDTH> &a,
	    const SIMDVec<T, SIMD_WIDTH> &b)
    {
      return unpack<PART,NUM_ELEMS>(a, b);
    }
  };

  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  class Unpack16
  {
  public:
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _unpack(const SIMDVec<T, SIMD_WIDTH> &a,
      const SIMDVec<T, SIMD_WIDTH> &b)
    {
      return unpack16<PART,NUM_ELEMS>(a, b);
    }
  };

  // ------------------------ transpose a single row ---------------------------

  // primary template
  template <template <int, int, typename, int> class Unpack,
	    typename T, int SIMD_WIDTH, 
	    // INDEX: index of first input element to unpack
	    // NLOHI: low/high unpack selector index
	    // ELEMS: number of elements to unpack in this stage
	    int INDEX, int NLOHI, int ELEMS>
  class Transpose1
  {
    enum { PART = (NLOHI & 0x01), NEXT = (NLOHI >> 1) };
    enum { LIDX = INDEX, RIDX = INDEX + ELEMS };
    enum { HALF = ELEMS / 2 };

  public:
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _transpose1(const SIMDVec<T,SIMD_WIDTH> *const inRows)
    {
      // printf("_transpose1("
      //       "INDEX=%d NLOHI=%d ELEMS=%d PART=%d LIDX=%d RIDX=%d HALF=%d)\n", 
      //       INDEX, NLOHI, ELEMS, PART, LIDX, RIDX, HALF);
      // TODO: T,SIMD_WIDTH necessary or can it be deduced from arguments?
      return Unpack<PART,ELEMS,T,SIMD_WIDTH>::_unpack
	(Transpose1<Unpack,T,SIMD_WIDTH,LIDX,NEXT,HALF>::_transpose1(inRows),
	 Transpose1<Unpack,T,SIMD_WIDTH,RIDX,NEXT,HALF>::_transpose1(inRows));
    }
  };

  // partial specialization to end the iteration (ELEMS=1)
  template <template <int, int, typename, int> class Unpack,
	    typename T, int SIMD_WIDTH, int INDEX, int NLOHI>
  class Transpose1<Unpack, T, SIMD_WIDTH, INDEX, NLOHI, 1>
  {
    enum { PART = (NLOHI & 0x01) };
    
  public:
    static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>
    _transpose1(const SIMDVec<T,SIMD_WIDTH> *const inRows)
    {
      // printf("_transpose1(INDEX=%d NLOHI=%d *ELEMS=%d PART=%d)\n", 
      // 	   INDEX, NLOHI, 1, PART);
      // TODO: T,SIMD_WIDTH necessary or can it be deduced from arguments?
      return Unpack<PART,1,T,SIMD_WIDTH>::_unpack(inRows[INDEX], 
						  inRows[INDEX+1]);
    }
  };

  // ----------------------- transpose multiple rows --------------------------

  // primary template
  template <template <int, int, typename, int> class Unpack,
	    typename T, int SIMD_WIDTH, 
	    // NUMROWS: total number of rows
	    // NUM_TRANSPOSE_ROWS: number of rows to transpose
	    // ROW: index of row to transpose
	    int NUMROWS, int NUM_TRANSPOSE_ROWS, int ROW>
  class Transpose
  {
  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
	       SIMDVec<T,SIMD_WIDTH> *const outRows)
    {
      // printf("\n_transpose(NUMROWS=%d,ROW=%d)\n", NUMROWS, ROW);
      // transpose single row with index ROW
      outRows[ROW] = 
	// INDEX=0, NLOWHI=ROW, ELEMS=NUMROWS/2
	Transpose1<Unpack,T,SIMD_WIDTH,
		   0,ROW,NUMROWS/2>::_transpose1(inRows);
      // transpose next row
      // NUMROWS=NUMROWS, ROW=ROW+1
      Transpose<Unpack,T,SIMD_WIDTH,
		NUMROWS,NUM_TRANSPOSE_ROWS,ROW+1>::_transpose(inRows, outRows);
    }
  };

  // partial specialization to end the iteration
  template <template <int, int, typename, int> class Unpack,
	    typename T, int SIMD_WIDTH, int NUMROWS, int NUM_TRANSPOSE_ROWS>
  class Transpose<Unpack,T,SIMD_WIDTH,
		  NUMROWS,NUM_TRANSPOSE_ROWS,NUM_TRANSPOSE_ROWS>
  {
  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const,
	       SIMDVec<T,SIMD_WIDTH> *const)
    {
    }
  };
  
  // function template: partial transpose
  template <int NUM_TRANSPOSE_ROWS, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
	    SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    Transpose<Unpack,T,SIMD_WIDTH,
	      // NUMROWS, NUM_TRANSPOSE_ROWS, ROW
	      SIMD_WIDTH / sizeof(T),
	      NUM_TRANSPOSE_ROWS, 0>::_transpose(inRows, outRows);
  }

  // function template: full transpose
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
	    SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    transpose<SIMD_WIDTH/sizeof(T)>(inRows, outRows);
  }

  // ===========================================================================
  // copy matrix
  // ===========================================================================

  // primary template
  template <typename T, int SIMD_WIDTH,
      int ROW, int ROW_STOP>
  class CopyMatrix
  {

  public:
    static SIMD_INLINE void
    _copy(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      v2[ROW] = v[ROW];
      CopyMatrix<T, SIMD_WIDTH, ROW + 1, ROW_STOP>::_copy(v, v2);
    }
  };

  // partial specialization to end the iteration
  template <typename T, int SIMD_WIDTH, int ROW_STOP>
  class CopyMatrix<T, SIMD_WIDTH, ROW_STOP, ROW_STOP>
  {

  public:
    static SIMD_INLINE void
    _copy(SIMDVec<T, SIMD_WIDTH> *, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // ===========================================================================
  // Transpose Post-Process
  // ===========================================================================

  // ------------------------ transpose post-process 16 ------------------------
  //            Used to post-process transposed matrix using unpack16

  // primary template
  template <typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW, int ROW_STOP,
      int TRANSPOSE_WIDTH, int SRC_OFF, int DST_OFF>
  class TransposePostprocess16
  {

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
               SIMDVec<T,SIMD_WIDTH> *const outRows)
    {
      // printf("%s", "\nTransposePostprocess16\n");
      // printf("TRANSPOSE_WIDTH=%d\n", TRANSPOSE_WIDTH);
      TransposePostprocess16<T, SIMD_WIDTH,
          NUMROWS, ROW, ROW_STOP,
          TRANSPOSE_WIDTH / 2, SRC_OFF,
          2 * DST_OFF>::_transpose(inRows, outRows);
      TransposePostprocess16<T, SIMD_WIDTH,
          NUMROWS, ROW, ROW_STOP,
          TRANSPOSE_WIDTH / 2, SRC_OFF + SIMD_WIDTH / TRANSPOSE_WIDTH,
          2 * DST_OFF + ROW_STOP>::_transpose(inRows, outRows);
    }
  };

  // partial specialization
  template <typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW, int ROW_STOP,
      int SRC_OFF, int DST_OFF>
  class TransposePostprocess16<T, SIMD_WIDTH,
      NUMROWS, ROW, ROW_STOP,
      16, SRC_OFF, DST_OFF>
  {
    enum { STEP = SIMD_WIDTH / 16 };
    enum { SRC_ROW = SRC_OFF + ROW * STEP };
    enum { DST_ROW = DST_OFF + ROW };

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
               SIMDVec<T,SIMD_WIDTH> *const outRows)
    {
      // printf("%s", "\nTransposePostprocess16\n");
      // printf("TRANSPOSE_WIDTH=%d\n", 16);
      // printf("SRC_ROW=%d DST_ROW=%d\n", SRC_ROW, DST_ROW);
      outRows[DST_ROW] = inRows[SRC_ROW];
      TransposePostprocess16<T, SIMD_WIDTH,
          NUMROWS, ROW + 1, ROW_STOP,
          16, SRC_OFF, DST_OFF>::_transpose(inRows, outRows);
    }
  };

  // partial specialization to end the iteration
  template  <typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW_STOP,
      int SRC_OFF, int DST_OFF>
  class TransposePostprocess16<T, SIMD_WIDTH,
      NUMROWS, ROW_STOP, ROW_STOP,
      16, SRC_OFF, DST_OFF>
  {
  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const,
               SIMDVec<T,SIMD_WIDTH> *const)
    {
    }
  };

  // ------------------------ transpose post-process hub -----------------------

  // primary template
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH>
  class TransposePostprocess
  {

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const,
               SIMDVec<T,SIMD_WIDTH> *const)
    {
    }
  };

  // partial specialization to post-process Transpose<Unpack16>
  template <typename T, int SIMD_WIDTH>
  class TransposePostprocess<Unpack16, T, SIMD_WIDTH>
  {
    enum { NUMROWS = SIMD_WIDTH / sizeof(T) };
    enum { ROW_STOP = 16 / sizeof(T) };

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
               SIMDVec<T,SIMD_WIDTH> *const outRows)
    {
      // printf("%s", "\nTransposePostprocess\n");
      // printf("SIMD_WIDTH=%d TYPE=%s\n", SIMD_WIDTH, SIMDTypeInfo<T>::name());
      TransposePostprocess16<T, SIMD_WIDTH,
          NUMROWS, 0, ROW_STOP, SIMD_WIDTH, 0, 0>::_transpose(
          inRows, outRows);
    }
  };

  // ===========================================================================
  // transpose0: Transpose<Unpack16> + post-process
  // ===========================================================================

  // function template: full transpose
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose0(const SIMDVec<T,SIMD_WIDTH> *const inRows,
             SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    SIMDVec<T,SIMD_WIDTH> tempRows[SIMDVec<T,SIMD_WIDTH>::elements];
    Transpose<Unpack16,T,SIMD_WIDTH,
        SIMD_WIDTH / sizeof(T), SIMD_WIDTH / sizeof(T), 0>
        ::_transpose(inRows, tempRows);
    TransposePostprocess<Unpack16, T, SIMD_WIDTH>::_transpose(tempRows, outRows);
  }

  // ===========================================================================
  // transpose1_16: Transpose<Unpack16> - needs store16
  // ===========================================================================

  // function template: full transpose
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose1_16(const SIMDVec<T,SIMD_WIDTH> *const inRows,
                SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    Transpose<Unpack16,T,SIMD_WIDTH,
        SIMD_WIDTH / sizeof(T), SIMD_WIDTH / sizeof(T), 0>
        ::_transpose(inRows, outRows);
  }

  // function template: full transpose (for tests, includes store16)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose1_16_t(const SIMDVec<T,SIMD_WIDTH> *const inRows,
                  SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    Transpose<Unpack16,T,SIMD_WIDTH,
        SIMD_WIDTH / sizeof(T), SIMD_WIDTH / sizeof(T), 0>
        ::_transpose(inRows, outRows);

    // post-process with store16 ...
    const int n = SIMD_WIDTH / sizeof(T);
    T outArray[n * n];
    storeu16(outArray, outRows);

    // ... and reload to outRows
    loadu(outArray, outRows, n);
  }

  // ===========================================================================
  // Transpose16: Template Class to transpose multiple rows with integrated
  // Unpack16 post-process
  // Uses Transpose1 to transpose single rows.
  // ===========================================================================

  // ----------------------- transpose multiple rows --------------------------

  // primary template
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      // NUMROWS: total number of rows
      // ROW: index of row to transpose
      int NUMROWS, int ROW, int ROW_STOP,
      int TRANSPOSE_WIDTH, int SRC_OFF, int DST_OFF>
  class Transpose16
  {
  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const inRows,
               SIMDVec<T,SIMD_WIDTH> *const outRows)
    {
      Transpose16<Unpack, T, SIMD_WIDTH,
          NUMROWS,  ROW, ROW_STOP,
          TRANSPOSE_WIDTH / 2, SRC_OFF,
          2 * DST_OFF>::_transpose(inRows, outRows);
      Transpose16<Unpack, T, SIMD_WIDTH,
          NUMROWS,  ROW, ROW_STOP,
          TRANSPOSE_WIDTH / 2, SRC_OFF + SIMD_WIDTH / TRANSPOSE_WIDTH,
          2 * DST_OFF + ROW_STOP>::_transpose(inRows, outRows);
    }
  };

  // partial specialization to end first iteration
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW, int ROW_STOP,
      int SRC_OFF, int DST_OFF>
  class Transpose16<Unpack, T, SIMD_WIDTH,
      NUMROWS, ROW, ROW_STOP,
      16, SRC_OFF, DST_OFF>
  {
    enum { STEP = SIMD_WIDTH / 16 };
    enum { SRC_ROW = SRC_OFF + ROW * STEP };
    enum { DST_ROW = DST_OFF + ROW };

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *inRows,
               SIMDVec<T,SIMD_WIDTH> *outRows)
    {
      // printf("\n_transpose0(SRC=%d,DST=%d)", SRC_ROW, DST_ROW);
      // printf("\n   ROW=%d,OFF=%d,STEP=%d", ROW, TRANSPOSE_OFFSET, STEP);
      // transpose single row with index SRC_ROW
      outRows[DST_ROW] =
          Transpose1<Unpack,T,SIMD_WIDTH,
              // INDEX=0, NLOWHI=SRC_ROW, ELEMS=NUMROWS/2
              0,SRC_ROW,NUMROWS/2>::_transpose1(inRows);
      // transpose next row
      // NUMROWS=NUMROWS, ROW=ROW+1
      Transpose16<Unpack, T, SIMD_WIDTH,
          NUMROWS, ROW + 1, ROW_STOP,
          16, SRC_OFF, DST_OFF>::_transpose(
          inRows, outRows);
    }
  };

  // partial specialization to end the iteration
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int NUMROWS, int ROW_STOP,
      int SRC_OFF,  int DST_OFF>
  class Transpose16<Unpack, T, SIMD_WIDTH,
      NUMROWS, ROW_STOP, ROW_STOP,
      16, SRC_OFF, DST_OFF>
  {
  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T,SIMD_WIDTH> *const,
               SIMDVec<T,SIMD_WIDTH> *const)
    {
    }
  };

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose16(const SIMDVec<T,SIMD_WIDTH> *const inRows,
              SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    Transpose16<Unpack16,T,SIMD_WIDTH,
        // NUMROWS, ROW, ROW_STOP
        SIMD_WIDTH / sizeof(T), 0, 16 / sizeof(T),
        // TRANSPOSE_WIDTH, SRC_OFF, DST_OFF
        SIMD_WIDTH, 0, 0>::_transpose(inRows, outRows);
  }

  // ===========================================================================
  // swizzle2 (deinterleave)
  // ===========================================================================
  
  // generalized from Marat Dukhan's solution referred to at
  // https://stackoverflow.com/a/15377386/3852630
  // takes 2*N input elements

  // TODO: swizzling chunks of multiple elements (useful?)
  // TODO: could be possible by starting loop at sizeof(T) and
  // TODO: using zip<NUM_ELEMS>

  // v has to have length 2*N
  // FINALBLKSIZE template argument is required since function is also
  // used for transpose2
  template <int N, int FINALBLKSIZE, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle2(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    SIMDVec<T, SIMD_WIDTH> v2[2*N];
    for (int blkSize = 1; blkSize <= FINALBLKSIZE; blkSize *= 2) {
      // zip
      for (int src = 0, dst = 0; src < N; src++, dst += 2)
	zip<1>(v[src], v[src+N], v2[dst], v2[dst+1]);
      // copy result back to v
      // TODO: swizzle2: check code produced by compiler for copying
      for (int i = 0; i < 2*N; i++)
	v[i] = v2[i];
    }
  }

  // v has to have length 2*N
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle2(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    swizzle2<N, SIMDVec<T, SIMD_WIDTH>::elements>(v);
  }

  // ===========================================================================
  // transpose2
  // ===========================================================================

  // is called transpose2 as it is based on swizzle2, but expects same
  // number of vectors as transpose (not twice the number)
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose2(const SIMDVec<T,SIMD_WIDTH> *const inRows,
	     SIMDVec<T,SIMD_WIDTH> *const outRows)
  {
    SIMD_CONSTEXPR int n = SIMDVec<T,SIMD_WIDTH>::elements, N = n / 2;
    for (int i = 0; i < n; i++)
      outRows[i] = inRows[i];
    swizzle2<N,N>(outRows);
  }
  
  // ===========================================================================
  // swizzle3 (deinterleave)
  // ===========================================================================

  // contributed by Adam Marschall
  
  // generalized from Marat Dukhan's solution referred to at
  // https://stackoverflow.com/a/15377386/3852630
  // takes 2*N input elements

  // v has to have length 2*N
  // FINALBLKSIZE template argument is required since function is also
  // used for transpose3
  template<int N, int FINALBLKSIZE, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle3(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    SIMDVec<T, SIMD_WIDTH> v2[2 * N];
    int origReps = log2(FINALBLKSIZE) + 1;
    int finalReps = std::floor(origReps / 2.0);
    // printf("origReps=%d finalReps=%d\n", origReps, finalReps);

    for (int rep = 0; rep < finalReps; rep++) {
      // zip there ...
      for (int src = 0, dst = 0; src < N; src++, dst += 2)
        zip<1>(v[src], v[src + N], v2[dst], v2[dst + 1]);

      // ... and zip back again
      for (int src = 0, dst = 0; src < N; src++, dst += 2)
        zip<1>(v2[src], v2[src + N], v[dst], v[dst + 1]);
    }

    // skip post-amble in case of even origReps
    if (origReps % 2 == 0)
      return;

    // zip there ...
    for (int src = 0, dst = 0; src < N; src++, dst += 2)
      zip<1>(v[src], v[src + N], v2[dst], v2[dst + 1]);

    // ...and copy back again
    for (int i = 0; i < 2 * N; i++)
      v[i] = v2[i];
  }

  // v has to have length 2*N
  template<int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle3(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    swizzle3<N, SIMDVec<T, SIMD_WIDTH>::elements>(v);
  }

  // ===========================================================================
  // transpose3
  // ===========================================================================

  // contributed by Adam Marschall

  // called transpose3 as it is based on swizzle2, but expects same
  // number of vectors as transpose (not twice the number)
  template<typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose3(const SIMDVec<T, SIMD_WIDTH> *const inRows,
             SIMDVec<T, SIMD_WIDTH> *const outRows)
  {
    const int n = SIMDVec<T, SIMD_WIDTH>::elements, N = n / 2;
    for (int i = 0; i < n; i++)
      outRows[i] = inRows[i];
    swizzle3<N, N>(outRows);
  }

  // ===========================================================================
  // Swizzle4 meta template
  // ===========================================================================

  // 12. Oct 22 (Jonas Keller): renamed Swizzle3* classes to Swizzle4*

  // contributed by Adam Marschall

  // -------------------- different zip functions ------------------------------

  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  class Zip
  {
  public:
    static SIMD_INLINE void
    _zip(SIMDVec<T, SIMD_WIDTH> a,
         SIMDVec<T, SIMD_WIDTH> b,
         SIMDVec<T, SIMD_WIDTH> &l,
         SIMDVec<T, SIMD_WIDTH> &h)
    {
      zip<NUM_ELEMS, T>(a, b, l, h);
    }
  };

  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  class Zip16
  {
  public:
    static SIMD_INLINE void
    _zip(SIMDVec<T, SIMD_WIDTH> a,
         SIMDVec<T, SIMD_WIDTH> b,
         SIMDVec<T, SIMD_WIDTH> &l,
         SIMDVec<T, SIMD_WIDTH> &h)
    {
      zip16<NUM_ELEMS, T>(a, b, l, h);
    }
  };

  // ------------------------ swizzle matrix once ------------------------------

  // primary template
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH,
      int N, int SRC, int DST>
  class Swizzle4Once
  {
    enum { SRC2 = SRC + N, DST2 = DST + 1 };

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      // printf("%s\n", "SwizzleOnce");
      // printf("  SRC=%d, SRC2=%d, DST=%d, DS2T=%d\n", SRC, SRC2, DST, DST2);
      Zip<1,T,SIMD_WIDTH>::_zip(v[SRC], v[SRC2], v2[DST], v2[DST2]);
      Swizzle4Once<Zip, T, SIMD_WIDTH, N, SRC + 1, DST + 2>::_swizzle(v, v2);
    }
  };

  // partial specialization to end the iteration
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH,
      int N, int DST>
  class Swizzle4Once<Zip, T, SIMD_WIDTH, N, N, DST>
  {
  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> */*v*/, SIMDVec<T, SIMD_WIDTH> */*v2*/)
    {
      // for (int i = 0; i < 2 * N; i++) {
      //   print("%4d", v2[i]);
      //   puts("");
      // }
    }
  };

  // ------------------------ swizzle matrix multiple times --------------------

  // primary template
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH,
      int N, int REP, int FINAL_REPS, int ODD>
  class Swizzle4Multiple
  {

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      // printf("%s\n", "SwizzleMultiple");
      // printf("  REP=%d, FINAL_REPS=%d\n", REP, FINAL_REPS);
      Swizzle4Once<Zip, T, SIMD_WIDTH, N, 0, 0>::_swizzle(v, v2);
      Swizzle4Once<Zip, T, SIMD_WIDTH, N, 0, 0>::_swizzle(v2, v);
      Swizzle4Multiple<Zip, T, SIMD_WIDTH, N,
          REP + 1, FINAL_REPS, ODD>::_swizzle(v, v2);
    }
  };

  // partial specialization to end the iteration without swizzle post-amble
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH,
      int N, int FINAL_REPS, int ODD>
  class Swizzle4Multiple<Zip, T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, ODD>
  {
  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // partial specialization to end the iteration with swizzle post-amble
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH,
      int N, int FINAL_REPS>
  class Swizzle4Multiple<Zip, T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, 1>
  {
    enum { ROW_STOP = 2 * N };
  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      // printf("%s\n", "SwizzlePostamble");
      Swizzle4Once<Zip, T, SIMD_WIDTH, N, 0, 0>::_swizzle(v, v2);
      CopyMatrix<T, SIMD_WIDTH, 0, ROW_STOP>::_copy(v2, v);
    }
  };

  // ------------------------ swizzle main meta template -----------------------

  // generalized from Marat Dukhan's solution referred to at
  // https://stackoverflow.com/a/15377386/3852630
  // takes 2*N input elements

  // primary template
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH, int N, int FINALBLKSIZE>
  class Swizzle4
  {
    enum { ORIG_REPS = (int) floorlog2<FINALBLKSIZE>::value + 1 };
    enum { FINAL_REPS = ORIG_REPS / 2 };
    enum { ODD = (ORIG_REPS & 0x01) };

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v) {
      SIMDVec<T, SIMD_WIDTH> v2[2 * N];
      // printf("%s\n", "Swizzle");
      // printf("  N=%d, FINALBLKSIZE=%d\n", N, FINALBLKSIZE);
      // printf("  ORIG_REPS=%d, FINAL_REPS=%d, ODD=%d\n", ORIG_REPS, FINAL_REPS, ODD);
      Swizzle4Multiple<Zip, T, SIMD_WIDTH, N,
          0, FINAL_REPS, ODD>::_swizzle(v, v2);
    }
  };

  // ===========================================================================
  // swizzle4 wrapper function
  // ===========================================================================

  // 15. Oct 22 (Jonas Keller): added swizzle4 wrapper function

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  swizzle4(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    Swizzle4<Zip, T, SIMD_WIDTH, N, SIMDVec<T, SIMD_WIDTH>::elements>::_swizzle(v);
  }

  // ===========================================================================
  // Unswizzle4
  // ===========================================================================

  // 15. Oct 22 (Jonas Keller): added Unswizzle4 classes

  // Note: Unlike the Swizzle4 classes, the Unswizzle4 classes do not have a
  // template-template parameter for the Zip class.
  // In the Swizzle4 classes, the Zip template parameter is used to choose
  // between the zip and zip16 functions, which is needed by the Transpose4
  // classes. The Unswizzle4 classes are not used by Transpose4, so the Zip
  // template parameter is not needed.

  template <typename T, int SIMD_WIDTH, int N, int SRC, int DST>
  class Unswizzle4Once
  {
  public:
    static SIMD_INLINE void
    _unswizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      unzip<1,T>(v[SRC], v[SRC + 1], v2[DST], v2[DST + N]);
      Unswizzle4Once<T, SIMD_WIDTH, N, SRC + 2, DST + 1>::_unswizzle(v, v2);
    }
  };

  // partial specialization to end the iteration
  template <typename T, int SIMD_WIDTH, int N, int SRC>
  class Unswizzle4Once<T, SIMD_WIDTH, N, SRC, N>
  {
  public:
    static SIMD_INLINE void
    _unswizzle(SIMDVec<T, SIMD_WIDTH> *, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  template <typename T, int SIMD_WIDTH, int N, int REP, int FINAL_REPS, int ODD>
  class Unswizzle4Multiple
  {
  public:
    static SIMD_INLINE void
    _unswizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      Unswizzle4Once<T, SIMD_WIDTH, N, 0, 0>::_unswizzle(v, v2);
      Unswizzle4Once<T, SIMD_WIDTH, N, 0, 0>::_unswizzle(v2, v);
      Unswizzle4Multiple<T, SIMD_WIDTH, N,
          REP + 1, FINAL_REPS, ODD>::_unswizzle(v, v2);
    }
  };

  // partial specialization to end the iteration without unswizzle post-amble
  template <typename T, int SIMD_WIDTH, int N, int FINAL_REPS, int ODD>
  class Unswizzle4Multiple<T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, ODD>
  {
  public:
    static SIMD_INLINE void
    _unswizzle(SIMDVec<T, SIMD_WIDTH> *, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // partial specialization to end the iteration with unswizzle post-amble
  template <typename T, int SIMD_WIDTH, int N, int FINAL_REPS>
  class Unswizzle4Multiple<T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, 1>
  {
  public:
    static SIMD_INLINE void
    _unswizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      Unswizzle4Once<T, SIMD_WIDTH, N, 0, 0>::_unswizzle(v, v2);
      CopyMatrix<T, SIMD_WIDTH, 0, 2 * N>::_copy(v2, v);
    }
  };

  // ------------------------ unswizzle main meta template ---------------------

  template <typename T, int SIMD_WIDTH, int N>
  class Unswizzle4
  {
    enum { FINALBLKSIZE = SIMDVec<T, SIMD_WIDTH>::elements };
    enum { ORIG_REPS = (int) floorlog2<FINALBLKSIZE>::value + 1 };
    enum { FINAL_REPS = ORIG_REPS / 2 };
    enum { ODD = (ORIG_REPS & 0x01) };

  public:
    static SIMD_INLINE void
    _unswizzle(SIMDVec<T, SIMD_WIDTH> *v)
    {
      SIMDVec<T, SIMD_WIDTH> v2[2 * N];
      Unswizzle4Multiple<T, SIMD_WIDTH, N,
          0, FINAL_REPS, ODD>::_unswizzle(v, v2);
    }
  };

  // ===========================================================================
  // unswizzle4 wrapper function
  // ===========================================================================

  // 15. Oct 22 (Jonas Keller): added unswizzle4 wrapper function

  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  unswizzle4(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    Unswizzle4<T, SIMD_WIDTH, N>::_unswizzle(v);
  }

  // ===========================================================================
  // transpose4 Swizzle4<Zip>
  // ===========================================================================

  // contributed by Adam Marschall

  // called transpose4 as it is based on swizzle3, but expects same
  // number of vectors as transpose (not twice the number)
  template<typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose4(const SIMDVec<T, SIMD_WIDTH> *const inRows,
             SIMDVec<T, SIMD_WIDTH> *const outRows)
  {
    const int n = SIMDVec<T, SIMD_WIDTH>::elements, N = n / 2;
    for (int i = 0; i < n; i++)
      outRows[i] = inRows[i];
    Swizzle4<Zip, T, SIMD_WIDTH, N, N>::_swizzle(outRows);
  }

  // ===========================================================================
  // transpose5: Swizzle4<Zip16> + Swizzle post-process
  // ===========================================================================

  // contributed by Adam Marschall
  
  // ------------------------ swizzle post-process 16 once ---------------------

  // primary template
  template <typename T, int SIMD_WIDTH,
      int N, int SRC, int DST, int LANE_ELEMS>
  class Swizzle4Postprocess16Once
  {
    enum { SRC2 = SRC + 1, DST2 = DST + N };

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      // printf("%s\n", "SwizzlePostprocess16Once");
      // printf("  SRC=%d, SRC2=%d, DST=%d, DS2T=%d\n", SRC, SRC2, DST, DST2);
      Zip16<LANE_ELEMS,T,SIMD_WIDTH>
          ::_zip(v[SRC], v[SRC2], v2[DST], v2[DST2]);
      Swizzle4Postprocess16Once<T, SIMD_WIDTH,
          N, SRC + 2, DST + 1, LANE_ELEMS>::_swizzle(v, v2);
    }
  };

  // partial specialization to end the iteration
  template <typename T, int SIMD_WIDTH,
      int N, int SRC, int LANE_ELEMS>
  class Swizzle4Postprocess16Once<T, SIMD_WIDTH, N, SRC, N, LANE_ELEMS>
  {
  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> */*v*/, SIMDVec<T, SIMD_WIDTH> */*v2*/)
    {
      // for (int i = 0; i < 2 * N; i++) {
      //   print("%4d", v2[i]);
      //   puts("");
      // }
    }
  };

  // ------------------------ swizzle post-process 16 --------------------------

  // primary template
  template <typename T, int SIMD_WIDTH,
      int N, int LANE_ELEMS, int REP, int FINAL_REPS, int ODD>
  class Swizzle4Postprocess16
  {

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      // printf("%s\n", "SwizzlePostprocess16");
      // printf("  REP=%d, FINAL_REPS=%d, LANE_ELEMS=%d\n", REP, FINAL_REPS, LANE_ELEMS);
      Swizzle4Postprocess16Once<T, SIMD_WIDTH, N, 0, 0, LANE_ELEMS>
          ::_swizzle(v, v2);
      Swizzle4Postprocess16Once<T, SIMD_WIDTH, N, 0, 0, LANE_ELEMS * 2>
          ::_swizzle(v2, v);
      Swizzle4Postprocess16<T, SIMD_WIDTH, N,
          LANE_ELEMS * 4, REP + 1, FINAL_REPS, ODD>::_swizzle(v, v2);
    }
  };

  // partial specialization to end the iteration without post-process post-amble
  template <typename T, int SIMD_WIDTH,
      int N, int LANE_ELEMS, int FINAL_REPS, int ODD>
  class Swizzle4Postprocess16<T, SIMD_WIDTH,
      N, LANE_ELEMS, FINAL_REPS, FINAL_REPS, ODD>
  {

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // partial specialization to end the iteration with post-process post-amble
  template <typename T, int SIMD_WIDTH,
      int N, int LANE_ELEMS, int FINAL_REPS>
  class Swizzle4Postprocess16<T, SIMD_WIDTH,
      N, LANE_ELEMS, FINAL_REPS, FINAL_REPS, 1>
  {
    enum { ROW_STOP = 2 * N };

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v, SIMDVec<T, SIMD_WIDTH> *v2)
    {
      Swizzle4Postprocess16Once<T, SIMD_WIDTH, N, 0, 0, LANE_ELEMS>
          ::_swizzle(v, v2);
      CopyMatrix<T, SIMD_WIDTH, 0, ROW_STOP>::_copy(v2, v);
    }
  };

  // ------------------------ swizzle post-process hub -------------------------

  // primary template
  template <template <int, typename, int> class Zip,
      typename T, int SIMD_WIDTH, int N>
  class Swizzle4Postprocess
  {

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // partial specialization to post-process Swizzle<Zip16>
  template <typename T, int SIMD_WIDTH, int N>
  class Swizzle4Postprocess<Zip16, T, SIMD_WIDTH, N>
  {
    enum { ORIG_REPS = (int) floorlog2<SIMD_WIDTH>::value - 4 };
    enum { FINAL_REPS = ORIG_REPS / 2 };
    enum { ODD = (ORIG_REPS & 0x01) };
    enum { LANE_ELEMS = 16 / sizeof(T) };

  public:
    static SIMD_INLINE void
    _swizzle(SIMDVec<T, SIMD_WIDTH> *v)
    {
      SIMDVec<T, SIMD_WIDTH> v2[2 * N];
      Swizzle4Postprocess16<T, SIMD_WIDTH, N, LANE_ELEMS, 0, FINAL_REPS, ODD>
          ::_swizzle(v, v2);
    }
  };
  // ------------------------ transpose5 function call -------------------------

  // called transpose5 as it is based on swizzle2, but expects same
  // number of vectors as transpose (not twice the number)
  template<typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose5(const SIMDVec<T, SIMD_WIDTH> *const inRows,
             SIMDVec<T, SIMD_WIDTH> *const outRows)
  {
    const int n = SIMDVec<T, SIMD_WIDTH>::elements, N = n / 2;
    for (int i = 0; i < n; i++)
      outRows[i] = inRows[i];
    Swizzle4<Zip16, T, SIMD_WIDTH, N, N>::_swizzle(outRows);
    Swizzle4Postprocess<Zip16, T, SIMD_WIDTH, N>::_swizzle(outRows);
  }

  // ===========================================================================
  // transpose6: register-count based transpose
  // ===========================================================================

  // contributed by Adam Marschall

  // primary template: unpack repetition
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int PROCESS_ROW, int PROCESS_ROWS, int UNPACK_ELEMS,
      int UNPACK_REP, int UNPACK_REPS, int SUB_BASE, int SUB>
  class TransposeRcUnpackSingle
  {
    enum { UNPACK_PART = ((PROCESS_ROW >> (UNPACK_REPS-UNPACK_REP-1)) & 0x01) };
    enum { UNPACK_PART_NEXT = (((PROCESS_ROW+1) >> (UNPACK_REPS-UNPACK_REP-1)) & 0x01) };
    enum { SRC1 = (PROCESS_ROW - SUB) * 2 };
    enum { SRC2 = (PROCESS_ROW - SUB) * 2 + 1 };

    // wrong numbering of rows
    // enum { HALF_ROWS = PROCESS_ROWS / 2 };
    // enum { UNPACK_PART = PROCESS_ROW / HALF_ROWS };
    // enum { SRC1 = ((PROCESS_ROW * 2) % PROCESS_ROWS) };
    // enum { SRC2 = ((PROCESS_ROW * 2) % PROCESS_ROWS) + 1 };

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T, SIMD_WIDTH> *const inRows,
               SIMDVec<T, SIMD_WIDTH> *const outRows)
    {
      // printf("%2d <- Unpack<%d, %d, %s, %d>(%2d, %2d) SUB_BASE: %d, SUB: %d\n",
      //    PROCESS_ROW, UNPACK_PART, UNPACK_ELEMS,
      //    SIMDTypeInfo<T>::name(), SIMD_WIDTH, SRC1, SRC2, SUB_BASE, SUB);
      outRows[PROCESS_ROW] = Unpack<UNPACK_PART, UNPACK_ELEMS, T, SIMD_WIDTH>
          ::_unpack(inRows[SRC1], inRows[SRC2]);
      TransposeRcUnpackSingle<Unpack, T, SIMD_WIDTH,
          PROCESS_ROW + 1, PROCESS_ROWS, UNPACK_ELEMS, UNPACK_REP, UNPACK_REPS,
          SUB_BASE,
          SUB + (UNPACK_PART_NEXT == 1 && (PROCESS_ROW + 1) % SUB_BASE == 0
                 ? SUB_BASE
                 : 0)>
          ::_transpose(inRows, outRows);
    }
  };

  // partial specialisation to end iteration PROCESS_REP
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int PROCESS_ROWS, int UNPACK_ELEMS,
      int UNPACK_REP, int UNPACK_REPS, int SUB_BASE, int SUB>
  class TransposeRcUnpackSingle <Unpack,
      T, SIMD_WIDTH,
      PROCESS_ROWS, PROCESS_ROWS, UNPACK_ELEMS,
      UNPACK_REP, UNPACK_REPS, SUB_BASE, SUB>
  {

  public:
    static SIMD_INLINE void
    _transpose(const SIMDVec<T, SIMD_WIDTH> * const /*inRows*/,
               SIMDVec<T, SIMD_WIDTH> * const /*outRows*/)
    {
      // printf("%2d\n", PROCESS_ROWS);
      // for (int i = 0; i < PROCESS_ROWS; i++) {
      //   print("%5d", outRows[i]);
      //   puts("");
      // }
      // puts("");
    }
  };

  // ---------------------------------------------------------------------------

  // primary template: unpack repetition
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int UNPACK_REP, int UNPACK_REPS, int PROCESS_ROWS, int UNPACK_ELEMS,
      int SUB_BASE, int UNPACK_ODD>
  class TransposeRcUnpackMultiple
  {

  public:
    static SIMD_INLINE void
    _transpose(SIMDVec<T, SIMD_WIDTH> * const inRows,
               SIMDVec<T, SIMD_WIDTH> * const outRows)
    {
      // printf("\nTransposeRcUnpackMultiple %2d %s  %d/%d\n",
      //        SIMD_WIDTH, SIMDTypeInfo<T>::name(), UNPACK_REP+1, UNPACK_REPS);
      TransposeRcUnpackSingle<Unpack, T, SIMD_WIDTH,
          0, PROCESS_ROWS, UNPACK_ELEMS, UNPACK_REP, UNPACK_REPS, SUB_BASE, 0>
          ::_transpose(inRows, outRows);
      TransposeRcUnpackMultiple<Unpack, T, SIMD_WIDTH,
          UNPACK_REP + 1, UNPACK_REPS, PROCESS_ROWS,
          UNPACK_ELEMS * 2, SUB_BASE / 2, UNPACK_ODD>
          ::_transpose(outRows, inRows);
    }
  };

  // partial specialisation to end iteration UNPACK_REP
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int UNPACK_REPS, int PROCESS_ROWS, int UNPACK_ELEMS,
      int SUB_BASE, int UNPACK_ODD>
  class TransposeRcUnpackMultiple <Unpack,
      T, SIMD_WIDTH,
      UNPACK_REPS, UNPACK_REPS, PROCESS_ROWS, UNPACK_ELEMS, SUB_BASE,
      UNPACK_ODD>
  {

  public:
    static SIMD_INLINE void
    _transpose(SIMDVec<T, SIMD_WIDTH> * const,
               SIMDVec<T, SIMD_WIDTH> * const)
    {
    }
  };

  // partial specialisation to end iteration UNPACK_REP
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int UNPACK_REPS, int PROCESS_ROWS, int UNPACK_ELEMS,
      int SUB_BASE>
  class TransposeRcUnpackMultiple <Unpack,
      T, SIMD_WIDTH,
      UNPACK_REPS, UNPACK_REPS, PROCESS_ROWS, UNPACK_ELEMS, SUB_BASE, 0>
  {

  public:
    static SIMD_INLINE void
    _transpose(SIMDVec<T, SIMD_WIDTH> * const inRows,
               SIMDVec<T, SIMD_WIDTH> * const outRows)
    {
      // printf("\nTransposeRcUnpackMultiple %2d %s  %d/%d Copy Matrix\n",
      //        SIMD_WIDTH, SIMDTypeInfo<T>::name(), UNPACK_REPS, UNPACK_REPS);
      CopyMatrix<T, SIMD_WIDTH, 0, PROCESS_ROWS>::_copy(inRows, outRows);
    }
  };

  // ---------------------------------------------------------------------------

  // primary template: store all registers lane-wise
  template <typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS,
      int STORE_OFF, int VO, int LANE>
  class TransposeRcStoreLane
  {
    enum { VEC_ELEMS_OUT =  NUM_SIMDVEC_ELEMENTS(T,SIMD_WIDTH) };

  public:
    static SIMD_INLINE void
    _store(T *const outArray, SIMDVec<T, SIMD_WIDTH> * outRows)
    {
      storeu(outArray + STORE_OFF, extractLane<LANE>(outRows[VO]));
      TransposeRcStoreLane<T, SIMD_WIDTH,
          PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS,
          STORE_OFF + PROCESS_ROWS * VEC_ELEMS_OUT, VO, LANE + 1>
          ::_store(outArray, outRows);
    }
  };

  // partial specialisation to end iteration LANE=PROCESS_REPS
  template <typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS,
      int STORE_OFF, int VO>
  class TransposeRcStoreLane <T, SIMD_WIDTH,
      PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS,
      STORE_OFF, VO, PROCESS_REPS>
  {

  public:
    static SIMD_INLINE void
    _store(T *const, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // ---------------------------------------------------------------------------

  // primary template: store all registers lane-wise
  template <typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS,
      int STORE_OFF, int VO>
  class TransposeRcStoreLanes
  {
    enum { VEC_ELEMS_OUT =  NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH) };

  public:
    static SIMD_INLINE void
    _store(T *const  outArray, SIMDVec<T, SIMD_WIDTH> *  outRows)
    {
      TransposeRcStoreLane<T, SIMD_WIDTH,
          PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS,
          STORE_OFF, VO, 0>::_store(outArray, outRows);
      TransposeRcStoreLanes<T, SIMD_WIDTH,
          PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS,
          STORE_OFF + VEC_ELEMS_OUT, VO + 1>
          ::_store(outArray, outRows);
    }
  };

  // partial specialisation to end iteration VO=PROCESS_ROWS
  template <typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS,
      int STORE_OFF>
  class TransposeRcStoreLanes <T, SIMD_WIDTH,
      PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS, STORE_OFF, PROCESS_ROWS>
  {

  public:
    static SIMD_INLINE void
    _store(T *const, SIMDVec<T, SIMD_WIDTH> *)
    {
    }
  };

  // ---------------------------------------------------------------------------

  // primary template: store hub
  // decides whether to store directly (Store16) or to store lane-wise
  template <typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS>
  class TransposeRcStore
  {
    enum { ELEMS_PER_LANE = 16 / sizeof(T) };
    enum { STORE_OFF = PROCESS_REP * ELEMS_PER_LANE };

  public:
    static SIMD_INLINE void
    _store(T *const outArray, SIMDVec<T, SIMD_WIDTH> * outRows)
    {
      TransposeRcStoreLanes<T, SIMD_WIDTH,
          PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS, STORE_OFF, 0>
          ::_store(outArray, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_ROWS, int UNPACK_REPS>
  class TransposeRcStore <T, SIMD_WIDTH,
      PROCESS_REP, 1, PROCESS_ROWS, UNPACK_REPS>
  {

  public:
    static SIMD_INLINE void
    _store(T *const outArray, SIMDVec<T, SIMD_WIDTH> * outRows)
    {
      // printf("\nStore16 PROCESS_ROWS=%d\n", PROCESS_ROWS);
      Store16<Store, T, SIMD_WIDTH,
          PROCESS_ROWS, 0, 16 / sizeof(T),
          SIMD_WIDTH, 0, 0>::_store16(outArray, outRows);
    }
  };

  // ---------------------------------------------------------------------------


  // primary template: main repetition
  // loads, transposes, stores chunk of matrix
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int PROCESS_REP, int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS,
      int UNPACK_ODD>
  class TransposeRcRep
  {
    enum { LOAD_OFF = PROCESS_REP * PROCESS_ROWS * SIMD_WIDTH / sizeof(T) };
    enum { SUB_BASE = 1 << (floorlog2<PROCESS_ROWS>::value - 1) };

  public:
    static SIMD_INLINE void
    _transpose(const T *const inArray, T *const outArray)
    {
      // printf("\nTransposeRcRep %2d %s  %d/%d\n",
      //         SIMD_WIDTH, SIMDTypeInfo<T>::name(), PROCESS_REP+1, PROCESS_REPS);
      SIMDVec<T, SIMD_WIDTH> inRows[PROCESS_ROWS];
      SIMDVec<T, SIMD_WIDTH> outRows[PROCESS_ROWS];
      load(inArray + LOAD_OFF, inRows, PROCESS_ROWS);
      // for (int i = 0; i < PROCESS_ROWS; i++) {
      //   print("%5d", inRows[i]);
      //   puts("");
      // }
      // puts("");
      TransposeRcUnpackMultiple<Unpack, T, SIMD_WIDTH,
          0, UNPACK_REPS, PROCESS_ROWS, 1, SUB_BASE, UNPACK_ODD>
          ::_transpose(inRows, outRows);
      // for (int i = 0; i < PROCESS_ROWS; i++) {
      //   print("%5d", outRows[i]);
      //   puts("");
      // }
      // puts("");
      TransposeRcStore<T, SIMD_WIDTH,
          PROCESS_REP, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS>
          ::_store(outArray, outRows);
      TransposeRcRep<Unpack, T, SIMD_WIDTH,
          PROCESS_REP + 1, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS, UNPACK_ODD>
      ::_transpose(inArray, outArray);
    }
  };

  // partial specialisation to end iteration PROCESS_REP
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH,
      int PROCESS_REPS, int PROCESS_ROWS, int UNPACK_REPS,
      int UNPACK_ODD>
  class TransposeRcRep <Unpack,
      T, SIMD_WIDTH,
      PROCESS_REPS, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS, UNPACK_ODD>
  {

  public:
    static SIMD_INLINE void
    _transpose(const T *const, T *const)
    {
    }
  };

  // ---------------------------------------------------------------------------

  // primary template: main entrance
  template <template <int, int, typename, int> class Unpack,
      typename T, int SIMD_WIDTH>
  class TransposeRc
  {
    enum { SIMD_REGS = NATIVE_SIMD_REG_COUNT / 2 };
    enum { NUM_ROWS = SIMD_WIDTH / sizeof(T) };
    enum { PROCESS_REPS = (int) NUM_ROWS > (int) SIMD_REGS
           ? SIMD_WIDTH / 16
           : 1 };
    enum { PROCESS_ROWS = NUM_ROWS / PROCESS_REPS };
    enum { UNPACK_REPS = PROCESS_REPS == 1
           ? floorlog2<PROCESS_ROWS>::value
           : floorlog2<16 / sizeof(T)>::value };
    enum { UNPACK_ODD = (UNPACK_REPS & 0x01) };

  public:
    static SIMD_INLINE void
    _transpose(const T *const inArray, T *const outArray)
    {
      // printf("TransposeRc Process Rows: %d \n", PROCESS_ROWS);
      TransposeRcRep<Unpack, T, SIMD_WIDTH,
          0, PROCESS_REPS, PROCESS_ROWS, UNPACK_REPS, UNPACK_ODD>
          ::_transpose(inArray, outArray);
      // printf("%s","\n");
    }
  };

  // ---------------------------------------------------------------------------

  // function template: full transpose
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose6(const T *const inArray, T *const outArray)
  {
    TransposeRc<Unpack16,T,SIMD_WIDTH>::_transpose(inArray, outArray);
  }

  // function template: full transpose for tests
  template<typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  transpose6_t(const SIMDVec<T, SIMD_WIDTH> *const inRows,
               SIMDVec<T, SIMD_WIDTH> *const outRows)
  {
    // 15. Oct 22 (Jonas Keller):
    // This previously allocated two temporary arrays of type T* on the heap,
    // stored inRows and outRows into them, and then called transpose6 on those.
    // This is not necessary, we can just cast inRows and outRows to T* and call
    // transpose6 directly.
    transpose6<T, SIMD_WIDTH>((T*) inRows, (T*) outRows);
  }

  // ===========================================================================
  // unswizzle2 (interleave)
  // ===========================================================================
  
  // v has to have length 2*N
  template <int N, typename T, int SIMD_WIDTH>
  static SIMD_INLINE void
  unswizzle2(SIMDVec<T, SIMD_WIDTH> *const v)
  {
    const int finalBlkSize = SIMDVec<T, SIMD_WIDTH>::elements;
    SIMDVec<T, SIMD_WIDTH> v2[2*N];
    for (int blkSize = 1; blkSize <= finalBlkSize; blkSize *= 2) {
      // zip
      for (int dst = 0, src = 0; dst < N; dst++, src += 2)
	unzip<1>(v[src], v[src+1], v2[dst], v2[dst+N]);
      // copy result back to v
      // TODO: unswizzle2: check code produced by compiler for copying
      for (int i = 0; i < 2*N; i++)
	v[i] = v2[i];
    }
  }
 
  // ===========================================================================
  // avgru: synonym f. average with rounding up
  // ===========================================================================

  // this is just a synonym for avg which is compatible with the auxiliary avgrd
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  avgru(const SIMDVec<T, SIMD_WIDTH> &a,
	const SIMDVec<T, SIMD_WIDTH> &b)
  {
    return avg(a, b);
  }

  // ===========================================================================
  // avgrd: average with rounding down
  // ===========================================================================

  // 30. Jul 17 (rm): removed unnecessary tag dispatching for avgrd()

  // int types
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  avgrd(const SIMDVec<T, SIMD_WIDTH> &a,
        const SIMDVec<T, SIMD_WIDTH> &b)
  {
    SIMDVec<T, SIMD_WIDTH> one = set1<T, SIMD_WIDTH>(1), as, bs, lsb;
    lsb = and(and(a, b), one);
    as = div2rd(a);
    bs = div2rd(b);
    return add(lsb, add(as, bs));
  }

  // NOTE: no rounding for floating-point types
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  avgrd(const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
        const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    return mul(add(a, b), set1<SIMDFloat, SIMD_WIDTH>(0.5));
  }

  // ===========================================================================
  // setones: set all bits to 1
  // ===========================================================================

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setones()
  {
    SIMDVec<T, SIMD_WIDTH> zero = setzero<T, SIMD_WIDTH>();
    return cmpeq(zero, zero);
  }

  // ===========================================================================
  // setmin / setmax: set all elements min./max. value of type without set1()
  // setunity: set all elements to +1
  // setnegunity: set all elements to -1
  // ===========================================================================

  // ---------- signed int ----------

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setmin(IsIntIsSigned<true,true>)
  {
    return slli<8*sizeof(T)-1>(setones<T,SIMD_WIDTH>());
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setmax(IsIntIsSigned<true,true>)
  {
    return srli<1>(setones<T,SIMD_WIDTH>());
  }

  // only for signed int
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setnegunity(IsIntIsSigned<true, true>)
  {
    return setones<T,SIMD_WIDTH>();
  }

  // ---------- unsigned int ----------

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setmin(IsIntIsSigned<true,false>)
  {
    return setzero<T,SIMD_WIDTH>();
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setmax(IsIntIsSigned<true,false>)
  {
    return setones<T,SIMD_WIDTH>();
  }

  // ----------- int ------------

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setunity(IsFloatingPoint<false>)
  {
    return srli<8*sizeof(T)-1>(setones<T,SIMD_WIDTH>());
  }

  // ----------- float ----------

  // here we need set1()

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  setmin(IsIntIsSigned<false,true>)
  {
    return set1<SIMDFloat, SIMD_WIDTH>(SIMDTypeInfo<SIMDFloat>::min());
  }
  
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  setmax(IsIntIsSigned<false,true>)
  {
    return set1<SIMDFloat, SIMD_WIDTH>(SIMDTypeInfo<SIMDFloat>::max());
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  setunity(IsFloatingPoint<true>)
  {
    return set1<SIMDFloat, SIMD_WIDTH>(1.0f);
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  setnegunity(IsIntIsSigned<false,true>)
  {
    return set1<SIMDFloat, SIMD_WIDTH>(-1.0f);
  }
  
  // ---------- hubs ----------

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setmin()
  {
    return setmin<T,SIMD_WIDTH>(TypeIsIntIsSigned<T>());
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setmax()
  {
    return setmax<T,SIMD_WIDTH>(TypeIsIntIsSigned<T>());
  }

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setunity()
  {
    return setunity<T,SIMD_WIDTH>(TypeIsFloatingPoint<T>());
  }

  // only for signed types
  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  setnegunity()
  {
    return setnegunity<T,SIMD_WIDTH>(TypeIsIntIsSigned<T>());
  }

  // ===========================================================================
  // generalized packs
  // ===========================================================================

  // from\to
  //    SB B S W I F 
  // SB  x
  //  B     
  //  S  x x x
  //  W         
  //  I  x x x x x x
  //  F  x x x x x x
  //
  // input is only signed
  // same-size input and output is allowed

  // hub
  // generalized version of packs: includes multistage packing
  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  packs(const SIMDVec<Tin, SIMD_WIDTH> *const a)
  {
    return packs(a, 
		 OutputType<Tout>(), 
		 Compression<sizeof(Tin)/sizeof(Tout)>());
  }

  // no stage

  template <typename T, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<T, SIMD_WIDTH>
  packs(const SIMDVec<T, SIMD_WIDTH> *const a, 
	OutputType<T>, Compression<1>)
  {
    return *a;
  }
  
  // single stage

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  packs(const SIMDVec<Tin, SIMD_WIDTH> *const a, 
	OutputType<Tout>, Compression<2>)
  {
    return packs<Tout>(a[0], a[1]);
  }

  // two stages

  template <typename Tout, typename Tin, int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<Tout, SIMD_WIDTH>
  packs(const SIMDVec<Tin, SIMD_WIDTH> *const a, 
	OutputType<Tout>, Compression<4>)
  {
    // always via SIMDShort
    return packs<Tout>(packs<SIMDShort>(a[0], a[1]),
		       packs<SIMDShort>(a[2], a[3]));
  }

  // special case: int <-> float

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat,SIMD_WIDTH>
  packs(const SIMDVec<SIMDInt,SIMD_WIDTH> *const a, 
	OutputType<SIMDFloat>, Compression<1>)
  {
    return cvts<SIMDFloat>(*a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt,SIMD_WIDTH>
  packs(const SIMDVec<SIMDFloat,SIMD_WIDTH> *const a, 
	OutputType<SIMDInt>, Compression<1>)
  {
    return cvts<SIMDInt>(*a);
  }
  
  // ===========================================================================
  // ifelse with casts for compatible conditions
  // ===========================================================================
  
  // SIMDByte <-> SIMDSignedByte

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDByte,SIMD_WIDTH>
  ifelse(const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &cond,
	 const SIMDVec<SIMDByte,SIMD_WIDTH> &trueVal,
	 const SIMDVec<SIMDByte,SIMD_WIDTH> &falseVal)
  {
    return ifelse(reinterpret<SIMDByte>(cond), trueVal, falseVal);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,SIMD_WIDTH>
  ifelse(const SIMDVec<SIMDByte,SIMD_WIDTH> &cond,
	 const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &trueVal,
	 const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &falseVal)
  {
    return ifelse(reinterpret<SIMDSignedByte>(cond), trueVal, falseVal);
  }

  // SIMDShort <-> SIMDWord
  
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDWord,SIMD_WIDTH>
  ifelse(const SIMDVec<SIMDShort,SIMD_WIDTH> &cond,
	 const SIMDVec<SIMDWord,SIMD_WIDTH> &trueVal,
	 const SIMDVec<SIMDWord,SIMD_WIDTH> &falseVal)
  {
    return ifelse(reinterpret<SIMDWord>(cond), trueVal, falseVal);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDShort,SIMD_WIDTH>
  ifelse(const SIMDVec<SIMDWord,SIMD_WIDTH> &cond,
	 const SIMDVec<SIMDShort,SIMD_WIDTH> &trueVal,
	 const SIMDVec<SIMDShort,SIMD_WIDTH> &falseVal)
  {
    return ifelse(reinterpret<SIMDShort>(cond), trueVal, falseVal);
  }

  // SIMDFloat <-> SIMDInt

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt,SIMD_WIDTH>
  ifelse(const SIMDVec<SIMDFloat,SIMD_WIDTH> &cond,
	 const SIMDVec<SIMDInt,SIMD_WIDTH> &trueVal,
	 const SIMDVec<SIMDInt,SIMD_WIDTH> &falseVal)
  {
    return ifelse(reinterpret<SIMDInt>(cond), trueVal, falseVal);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat,SIMD_WIDTH>
  ifelse(const SIMDVec<SIMDInt,SIMD_WIDTH> &cond,
	 const SIMDVec<SIMDFloat,SIMD_WIDTH> &trueVal,
	 const SIMDVec<SIMDFloat,SIMD_WIDTH> &falseVal)
  {
    return ifelse(reinterpret<SIMDFloat>(cond), trueVal, falseVal);
  }

  // ===========================================================================
  // div2r0: integer div. by 2 with round to 0 (for integers)
  // ===========================================================================

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDByte,SIMD_WIDTH>
  div2r0(const SIMDVec<SIMDByte,SIMD_WIDTH> &a)
  {
    return srli<1>(a); 
  }

  // 16. Oct 22 (Jonas Keller): added missing version for SIMDSignedByte
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,SIMD_WIDTH>
  div2r0(const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &a)
  {
    // add 1 if number is negative
    return srai<1>(add(a, srli<7>(a)));
  }
  

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDWord,SIMD_WIDTH>
  div2r0(const SIMDVec<SIMDWord,SIMD_WIDTH> &a)
  {
    return srli<1>(a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDShort,SIMD_WIDTH>
  div2r0(const SIMDVec<SIMDShort,SIMD_WIDTH> &a)
  {
    // add 1 if number is negative
    return srai<1>(add(a, srli<15>(a)));
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt,SIMD_WIDTH>
  div2r0(const SIMDVec<SIMDInt,SIMD_WIDTH> &a)
  {
    // add 1 if number is negative
    return srai<1>(add(a, srli<31>(a)));
  }

  // NOTE: no rounding for float
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat,SIMD_WIDTH>
  div2r0(const SIMDVec<SIMDFloat,SIMD_WIDTH> &a)
  {
    return mul(set1<SIMDFloat,SIMD_WIDTH>(0.5f), a);
  }

  // ===========================================================================
  // div2rd: integer division by two with rounding down (for integers)
  // ===========================================================================

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDByte,SIMD_WIDTH>
  div2rd(const SIMDVec<SIMDByte,SIMD_WIDTH> &a)
  {
    return srli<1>(a); 
  }

  // 16. Oct 22 (Jonas Keller): added missing version for SIMDSignedByte
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,SIMD_WIDTH>
  div2rd(const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &a)
  {
    return srai<1>(a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDWord,SIMD_WIDTH>
  div2rd(const SIMDVec<SIMDWord,SIMD_WIDTH> &a)
  {
    return srli<1>(a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDShort,SIMD_WIDTH>
  div2rd(const SIMDVec<SIMDShort,SIMD_WIDTH> &a)
  {
    return srai<1>(a);
  }

  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt,SIMD_WIDTH>
  div2rd(const SIMDVec<SIMDInt,SIMD_WIDTH> &a)
  {
    return srai<1>(a);
  }

  // NOTE: no rounding for float
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat,SIMD_WIDTH>
  div2rd(const SIMDVec<SIMDFloat,SIMD_WIDTH> &a)
  {
    return mul(set1<SIMDFloat,SIMD_WIDTH>(0.5f), a);
  }
  	  
  // ===========================================================================
  // sign function (SIMDFloat only)
  // ===========================================================================

  // contributed by Benedikt Volkmer
  // negate a, where b is negative
  // note: contrary to IEEE 754, this function considers -0.0f to be negative
  template <int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  sign(const SIMDVec<SIMDFloat, SIMD_WIDTH> &a,
       const SIMDVec<SIMDFloat, SIMD_WIDTH> &b)
  {
    // -0.0F aka. 0x80000000 aka. 1000...b
    return xor(a, and(set1<SIMDFloat, SIMD_WIDTH>(-0.0F), b));
  }

  // ===========================================================================
  // absDiff function
  // ===========================================================================

  // contributed by Benedikt Volkmer
  // 23. Mar 22 (rm): removed SFINAE enable_if construct
  // (not compatible with C++98)
  // Computes elementwise absolute difference of vectors

  // Use these overloads of the function template if Type is unsigned
  
  template<int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDByte, SIMD_WIDTH>
  absDiff(const SIMDVec<SIMDByte, SIMD_WIDTH> &v1,
	  const SIMDVec<SIMDByte, SIMD_WIDTH> &v2) {
    // Trick working around non-existing abs() for unsigned Type
    return or(subs(v1, v2), subs(v2, v1));
  }
  
  template<int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDWord, SIMD_WIDTH>
  absDiff(const SIMDVec<SIMDWord, SIMD_WIDTH> &v1,
	  const SIMDVec<SIMDWord, SIMD_WIDTH> &v2) {
    // Trick working around non-existing abs() for unsigned Type
    return or(subs(v1, v2), subs(v2, v1));
  }
  
  // Use these overloads of the function template if Type is signed

  template<int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDSignedByte, SIMD_WIDTH>
  absDiff(const SIMDVec<SIMDSignedByte, SIMD_WIDTH> &v1,
	  const SIMDVec<SIMDSignedByte, SIMD_WIDTH> &v2) {
    return abs(sub(v1, v2));
  }
  
  template<int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDShort, SIMD_WIDTH>
  absDiff(const SIMDVec<SIMDShort, SIMD_WIDTH> &v1,
	  const SIMDVec<SIMDShort, SIMD_WIDTH> &v2) {
    return abs(sub(v1, v2));
  }

  template<int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDInt, SIMD_WIDTH>
  absDiff(const SIMDVec<SIMDInt, SIMD_WIDTH> &v1,
	  const SIMDVec<SIMDInt, SIMD_WIDTH> &v2) {
    return abs(sub(v1, v2));
  }

  template<int SIMD_WIDTH>
  static SIMD_INLINE SIMDVec<SIMDFloat, SIMD_WIDTH>
  absDiff(const SIMDVec<SIMDFloat, SIMD_WIDTH> &v1,
	  const SIMDVec<SIMDFloat, SIMD_WIDTH> &v2) {
    return abs(sub(v1, v2));
  }

  // ===========================================================================
  // bitonic sort
  // ===========================================================================

  // code contributed by Lukas Schiermeier and Moritz Breipohl, modified

  const int ASCENDING = 0;
  const int DESCENDING = 1;

  // compare-and-swap
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE>
  class Cas;

  // specialization for DESCENDING
  template <typename T, int SIMD_WIDTH>
  class Cas<DESCENDING, T, SIMD_WIDTH>
  {
  public:
    static void
    compareAndSwap(SIMDVec<T, SIMD_WIDTH> &a, SIMDVec<T, SIMD_WIDTH> &b)
    {
      SIMDVec<T, SIMD_WIDTH> temp = min(a, b);
      a = max(a, b);
      b = temp;
    }
  };

  // specialization for ASCENDING
  template <typename T, int SIMD_WIDTH>
  class Cas<ASCENDING, T, SIMD_WIDTH>
  {
  public:
    static void
    compareAndSwap(SIMDVec<T, SIMD_WIDTH> &a, SIMDVec<T, SIMD_WIDTH> &b)
    {
      SIMDVec<T, SIMD_WIDTH> temp = max(a, b);
      a = min(a, b);
      b = temp;
    }
  };

  // in-place sorting of multiple arbitrary vectors;
  // transVecs have to be transposed vectors (same number of elements
  // as in SIMDVec), are still transposed afterwards
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicSortTransposed(SIMDVec<T, SIMD_WIDTH> *const transVecs)
  {
    SIMD_CONSTEXPR unsigned int numVecs = SIMDVec<T, SIMD_WIDTH>::elements;
    /* Dependent Loops */
    for (unsigned int blkSize = 2; blkSize <= numVecs; blkSize *= 2) {
      /*
       * Bitonic Core
       * Independent Loops
       */
      for (unsigned int blkStart = 0; blkStart < numVecs; blkStart += blkSize) {
	unsigned int halfBlk = blkSize / 2;
	unsigned int leftCounter = blkStart;
	unsigned int rightCounter = blkStart + (blkSize - 1);
	/* Independent Loops */
	for (unsigned int i = 0; i < halfBlk; i++) {
	  Cas<SLOPE,T,SIMD_WIDTH>::compareAndSwap
	    (transVecs[leftCounter], transVecs[rightCounter]);
	  leftCounter++;
	  rightCounter--;
	}
	/*
	 * This loop is skipped for blkSize < 4
	 * Builds the second half of the bitonic core.
	 *
	 * Dependent Loops
	 */
	for (unsigned int step = blkSize / 4; step > 0; step /= 2) {
	  /* Independent Loops */
	  for (unsigned int jump = 0; jump < blkSize; jump += step * 2) {
	    leftCounter = blkStart + jump;
	    rightCounter = blkStart + jump + step;	    
	    /* Independent Loops */
	    for (unsigned int k = 0; k < step; k++) {
	      Cas<SLOPE,T,SIMD_WIDTH>::compareAndSwap
		(transVecs[leftCounter], transVecs[rightCounter]);
	      leftCounter++;
	      rightCounter++;
	    }
	  }
	}
      }
    }
  }

  // post-fusion stage of bitonic sort, used to sort pairs of sorted vectors
  // which were fused (one reversed) and then sorted such that the pair
  // is sorted over the two vectors
  // in-place sorting; transVecs have to be transposed vectors (same
  // number of elements as in SIMDVec), are still transposed
  // afterwards
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicSortReducedTransposed(SIMDVec<T, SIMD_WIDTH> *const transVecs)
  {
    SIMD_CONSTEXPR unsigned int numVecs = SIMDVec<T, SIMD_WIDTH>::elements;
    for (unsigned int step = numVecs / 2; step > 0; step /= 2) {
      /* Independent Loops */
      for (unsigned int jump = 0; jump < numVecs; jump += step * 2) {
	unsigned int leftCounter = jump;
	unsigned int rightCounter = jump + step;	
	/* Independent Loops */
	for (unsigned int k = 0; k < step; k++) {
	  Cas<SLOPE,T,SIMD_WIDTH>::compareAndSwap
	    (transVecs[leftCounter], transVecs[rightCounter]);
	  leftCounter++;
	  rightCounter++;
	}
      }
    }
  }

  // with transpose
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicSort(SIMDVec<T, SIMD_WIDTH> *const vecs)
  {
    SIMDVec<T, SIMD_WIDTH> transVecs[SIMDVec<T, SIMD_WIDTH>::elements];
    transpose(vecs, transVecs);
    bitonicSortTransposed<SLOPE>(transVecs);
    transpose(transVecs, vecs);
  }

  // with transpose2
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicSort2(SIMDVec<T, SIMD_WIDTH> *const vecs)
  {
    SIMDVec<T, SIMD_WIDTH> transVecs[SIMDVec<T, SIMD_WIDTH>::elements];
    transpose2(vecs, transVecs);
    bitonicSortTransposed<SLOPE>(transVecs);
    transpose2(transVecs, vecs);
  }

  // second vector is reversed and fused with first vector
  // we don't have to reverse b after the compare-swap since it is
  // bitonic
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicFusion(SIMDVec<T, SIMD_WIDTH> &a, SIMDVec<T, SIMD_WIDTH> &b)
  {
    b = reverse(b);
    Cas<SLOPE,T,SIMD_WIDTH>::compareAndSwap(a, b);
  }
  
  // given sorted vectors as inputs, it fuses each consecutive pair
  // such it is completely sorted over the pair
  // with transpose
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicSortSortedPairs(SIMDVec<T, SIMD_WIDTH> *const vecs)
  {
    SIMDVec<T, SIMD_WIDTH> transVecs[SIMDVec<T, SIMD_WIDTH>::elements];
    // second vector of each pair is reversed and fused with first vector
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elements; i += 2)
      bitonicFusion<SLOPE>(vecs[i], vecs[i + 1]);
    transpose(vecs, transVecs);
    bitonicSortReducedTransposed<SLOPE>(transVecs);
    transpose(transVecs, vecs);
  }

  // given sorted vectors as inputs, it fuses each consecutive pair
  // such it is completely sorted over the pair
  // with transpose2
  template <int SLOPE, typename T, int SIMD_WIDTH_DEFAULT_NATIVE_FUNC>
  static SIMD_INLINE void
  bitonicSortSortedPairs2(SIMDVec<T, SIMD_WIDTH> *const vecs)
  {
    SIMDVec<T, SIMD_WIDTH> transVecs[SIMDVec<T, SIMD_WIDTH>::elements];
    // second vector of each pair is reversed and fused with first vector
    for (int i = 0; i < SIMDVec<T, SIMD_WIDTH>::elements; i += 2)
      bitonicFusion<SLOPE>(vecs[i], vecs[i + 1]);
    transpose2(vecs, transVecs);
    bitonicSortReducedTransposed<SLOPE>(transVecs);
    transpose2(transVecs, vecs);
  }
  
  // ===========================================================================
  // operators
  // ===========================================================================
  
  // C++ Coding Standards p.49 (item 27)

#define SIMDVEC_BINOPEQ(OP,FCT)					\
  template <typename T, int SIMD_WIDTH>				\
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>			\
  OP (SIMDVec<T,SIMD_WIDTH> &a,					\
      const SIMDVec<T,SIMD_WIDTH> &b)				\
  {								\
    a = FCT (a, b);						\
    return a;							\
  }
  
#define SIMDVEC_BINOP(OP,FCT)					\
  template <typename T, int SIMD_WIDTH>				\
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>			\
  OP (const SIMDVec<T,SIMD_WIDTH> &a,				\
      const SIMDVec<T,SIMD_WIDTH> &b)				\
  {								\
    return FCT (a, b);						\
  }
  
#define SIMDVEC_UNOP(OP,FCT)				        \
  template <typename T, int SIMD_WIDTH>				\
  static SIMD_INLINE SIMDVec<T,SIMD_WIDTH>			\
  OP (const SIMDVec<T,SIMD_WIDTH> &a)				\
  {								\
    return FCT (a);						\
  }

  // limitations:
  // - mul, div only for SIMDFloat
  // - neg only for signed types

  SIMDVEC_BINOP(operator+,adds)
  SIMDVEC_BINOP(operator-,subs)
  SIMDVEC_BINOP(operator*,mul)
  SIMDVEC_BINOP(operator/,div)
  SIMDVEC_BINOP(operator&,and)
  SIMDVEC_BINOP(operator|,or)
  SIMDVEC_BINOP(operator^,xor)

  SIMDVEC_BINOPEQ(operator+=,adds)
  SIMDVEC_BINOPEQ(operator-=,subs)
  SIMDVEC_BINOPEQ(operator*=,mul)
  SIMDVEC_BINOPEQ(operator/=,div)
  SIMDVEC_BINOPEQ(operator&=,and)
  SIMDVEC_BINOPEQ(operator|=,or)
  SIMDVEC_BINOPEQ(operator^=,xor)

  SIMDVEC_BINOP(operator>,cmpgt)
  SIMDVEC_BINOP(operator>=,cmpge)
  SIMDVEC_BINOP(operator==,cmpeq)
  SIMDVEC_BINOP(operator!=,cmpneq)
  SIMDVEC_BINOP(operator<=,cmple)
  SIMDVEC_BINOP(operator<,cmplt)
  
  SIMDVEC_UNOP(operator-,neg)
  SIMDVEC_UNOP(operator~,not)
  
} // namespace

#endif
