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

#include "SIMDVecBase.H"
#include "Compare2Type.H"
#include <assert.h>
#include <string.h>
#include <cmath>

// ===========================================================================
// NOTES:
//
// - conflicting reports: transpose2 sometimes about the same speed as
//   transpose, sometimes transpose2 is much slower?
// 
// ===========================================================================


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
    char fmtSep[256];
    // 30. Aug 22 (Jonas Keller): clang++ on windows warns that strcat and
    // strcpy are deprecated and suggests to use strcpy_s and strncat_s
    // instead
#ifdef _WIN32
    strcpy_s(fmtSep, format);
    strcat_s(fmtSep, separator);
#else
    strcat(strcpy(fmtSep, format), separator);
#endif
    fprint(f, fmtSep, vec);
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

  /*
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
  */

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

  /* TODO: div2r0 can be implemented if srai exists for SIMDSignedByte
     template <int SIMD_WIDTH>
     static SIMD_INLINE SIMDVec<SIMDSignedByte,SIMD_WIDTH>
     div2r0(const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &a)
     {
     }
  */

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

  /* TODO: div2rd can be implemented if srai exists for SIMDSignedByte
     template <int SIMD_WIDTH>
     static SIMD_INLINE SIMDVec<SIMDSignedByte,SIMD_WIDTH>
     div2rd(const SIMDVec<SIMDSignedByte,SIMD_WIDTH> &a)
     {
     }
  */

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
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
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
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
  static SIMD_INLINE void
  bitonicSortTransposed(SIMDVec<T, SIMD_WIDTH> *const transVecs)
  {
    SIMD_CONSTEXPR uint numVecs = SIMDVec<T, SIMD_WIDTH>::elements;
    /* Dependent Loops */
    for (uint blkSize = 2; blkSize <= numVecs; blkSize *= 2) {
      /*
       * Bitonic Core
       * Independent Loops
       */
      for (uint blkStart = 0; blkStart < numVecs; blkStart += blkSize) {
	uint halfBlk = blkSize / 2;
	uint leftCounter = blkStart;
	uint rightCounter = blkStart + (blkSize - 1);
	/* Independent Loops */
	for (uint i = 0; i < halfBlk; i++) {
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
	for (uint step = blkSize / 4; step > 0; step /= 2) {
	  /* Independent Loops */
	  for (uint jump = 0; jump < blkSize; jump += step * 2) {
	    leftCounter = blkStart + jump;
	    rightCounter = blkStart + jump + step;	    
	    /* Independent Loops */
	    for (uint k = 0; k < step; k++) {
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
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
  static SIMD_INLINE void
  bitonicSortReducedTransposed(SIMDVec<T, SIMD_WIDTH> *const transVecs)
  {
    SIMD_CONSTEXPR uint numVecs = SIMDVec<T, SIMD_WIDTH>::elements;
    for (uint step = numVecs / 2; step > 0; step /= 2) {
      /* Independent Loops */
      for (uint jump = 0; jump < numVecs; jump += step * 2) {
	uint leftCounter = jump;
	uint rightCounter = jump + step;	
	/* Independent Loops */
	for (uint k = 0; k < step; k++) {
	  Cas<SLOPE,T,SIMD_WIDTH>::compareAndSwap
	    (transVecs[leftCounter], transVecs[rightCounter]);
	  leftCounter++;
	  rightCounter++;
	}
      }
    }
  }

  // with transpose
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
  static SIMD_INLINE void
  bitonicSort(SIMDVec<T, SIMD_WIDTH> *const vecs)
  {
    SIMDVec<T, SIMD_WIDTH> transVecs[SIMDVec<T, SIMD_WIDTH>::elements];
    transpose(vecs, transVecs);
    bitonicSortTransposed<SLOPE>(transVecs);
    transpose(transVecs, vecs);
  }

  // with transpose2
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
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
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
  static SIMD_INLINE void
  bitonicFusion(SIMDVec<T, SIMD_WIDTH> &a, SIMDVec<T, SIMD_WIDTH> &b)
  {
    b = reverse(b);
    Cas<SLOPE,T,SIMD_WIDTH>::compareAndSwap(a, b);
  }
  
  // given sorted vectors as inputs, it fuses each consecutive pair
  // such it is completely sorted over the pair
  // with transpose
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
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
#if defined(NATIVE_SIMD_WIDTH) && (__cplusplus >= 201103L)
  template <int SLOPE, typename T, int SIMD_WIDTH = NATIVE_SIMD_WIDTH>
#else
  template <int SLOPE, typename T, int SIMD_WIDTH>
#endif
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
