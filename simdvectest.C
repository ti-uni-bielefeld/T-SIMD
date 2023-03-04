// ===========================================================================
//
// simdvectest.C --
// test of Vec templates
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

#include "SIMDVecAll.H"

#include <stdio.h>

using namespace simd;

// SW = SIMD width = number of bytes in a single SIMD vector

#define SW NATIVE_SIMD_WIDTH

#define LINE1                                                                  \
  puts("--------------------------------------------------------------------"  \
       "-");

#define LINE2                                                                  \
  puts("===================================================================="  \
       "=");

#define SETFCT(TYPE, FORMAT)                                                   \
  {                                                                            \
    puts(#TYPE);                                                               \
    Vec<TYPE, SW> vmin, vmax, vunity, vones;                                   \
    vmin   = setmin<TYPE, SW>();                                               \
    vmax   = setmax<TYPE, SW>();                                               \
    vunity = setunity<TYPE, SW>();                                             \
    vones  = setones<TYPE, SW>();                                              \
    printf("min:   ");                                                         \
    print(FORMAT, vmin);                                                       \
    puts("");                                                                  \
    printf("max:   ");                                                         \
    print(FORMAT, vmax);                                                       \
    puts("");                                                                  \
    printf("unity: ");                                                         \
    print(FORMAT, vunity);                                                     \
    puts("");                                                                  \
    printf("ones:  ");                                                         \
    print(FORMAT, vones);                                                      \
    puts("");                                                                  \
    puts("");                                                                  \
  }

#define SETSIGNEDFCT(TYPE, FORMAT)                                             \
  {                                                                            \
    puts(#TYPE);                                                               \
    Vec<TYPE, SW> vnegunity;                                                   \
    vnegunity = setnegunity<TYPE, SW>();                                       \
    printf("negunity: ");                                                      \
    print(FORMAT, vnegunity);                                                  \
    puts("");                                                                  \
    puts("");                                                                  \
  }

#define NUMERIC                                                                \
  {                                                                            \
    const int elems = Vec<Float, SW>::elements;                                \
    Vec<Float, SW> x, y;                                                       \
    Float bufx[elems], bufy[elems];                                            \
    Float xi = 1.234e-10, yi = 33.3;                                           \
    for (int i = 0; i < elems; i++) {                                          \
      bufx[i] = xi;                                                            \
      bufy[i] = yi;                                                            \
      xi *= 55.3;                                                              \
      yi += 34.2;                                                              \
    }                                                                          \
    x = loadu<SW>(bufx);                                                       \
    y = loadu<SW>(bufy);                                                       \
    printf("x        = ");                                                     \
    print("%14.8g ", x);                                                       \
    puts("");                                                                  \
    printf("y        = ");                                                     \
    print("%14.8g ", y);                                                       \
    puts("");                                                                  \
    printf("div(x,y) = ");                                                     \
    print("%14.8g ", div(x, y));                                               \
    puts("");                                                                  \
    printf("div(y,x) = ");                                                     \
    print("%14.8g ", div(y, x));                                               \
    puts("");                                                                  \
    printf("sqrt(x)  = ");                                                     \
    print("%14.8g ", sqrt(x));                                                 \
    puts("");                                                                  \
    printf("sqrt(y)  = ");                                                     \
    print("%14.8g ", sqrt(y));                                                 \
    puts("");                                                                  \
  }

#define ROUNDING(X0, XOFF)                                                     \
  {                                                                            \
    const int elems = Vec<Float, SW>::elements;                                \
    Vec<Float, SW> x;                                                          \
    Float bufx[elems];                                                         \
    Float xi = X0;                                                             \
    for (int i = 0; i < elems; i++) {                                          \
      bufx[i] = xi;                                                            \
      xi += XOFF;                                                              \
    }                                                                          \
    x = loadu<SW>(bufx);                                                       \
    puts("");                                                                  \
    printf("x        = ");                                                     \
    print("%14.8g ", x);                                                       \
    puts("");                                                                  \
    printf("round(x) = ");                                                     \
    print("%14.8g ", round(x));                                                \
    puts("");                                                                  \
    printf("ceil(x)  = ");                                                     \
    print("%14.8g ", ceil(x));                                                 \
    puts("");                                                                  \
    printf("floor(x) = ");                                                     \
    print("%14.8g ", floor(x));                                                \
    puts("");                                                                  \
    printf("trunc(x) = ");                                                     \
    print("%14.8g ", truncate(x));                                             \
    puts("");                                                                  \
  }

#define HOR(CMD, TYPE, FORMAT, A0, AS, B0, BS)                                 \
  {                                                                            \
    puts(#TYPE);                                                               \
    const int elems = Vec<TYPE, SW>::elements;                                 \
    Vec<TYPE, SW> a, b;                                                        \
    TYPE bufa[elems], bufb[elems];                                             \
    TYPE av = A0, bv = B0;                                                     \
    for (unsigned i = 0; i < elems; i++, av += AS, bv += BS) {                 \
      bufa[i] = av;                                                            \
      bufb[i] = bv;                                                            \
    }                                                                          \
    a = loadu<SW>(bufa);                                                       \
    b = loadu<SW>(bufb);                                                       \
    printf("a = ");                                                            \
    print(FORMAT, a);                                                          \
    puts("");                                                                  \
    printf("b = ");                                                            \
    print(FORMAT, b);                                                          \
    puts("");                                                                  \
    printf(#CMD "(a,b) = ");                                                   \
    print(FORMAT, CMD(a, b));                                                  \
    puts("");                                                                  \
    LINE1;                                                                     \
  }

#define PACKS(Tout, Tin, outFormat, inFormat, inStart, inFac, inOff)           \
  {                                                                            \
    const int numVecsIn  = sizeof(Tin) / sizeof(Tout);                         \
    const int numElemsIn = SW / sizeof(Tin);                                   \
    printf("%d " #Tin " -> " #Tout "\n", numVecsIn);                           \
    Vec<Tout, SW> out;                                                         \
    Vec<Tin, SW> in[numVecsIn];                                                \
    Tin buf[numElemsIn] __attribute__((aligned(SW)));                          \
    Tin val = inStart, fac = 1;                                                \
    for (int iVec = 0; iVec < numVecsIn; iVec++) {                             \
      for (int iElem = 0; iElem < numElemsIn; iElem++) {                       \
        buf[iElem] = val * fac;                                                \
        val += inOff;                                                          \
        fac *= inFac;                                                          \
      }                                                                        \
      in[iVec] = load<SW>(buf);                                                \
      printf("in[%d] = ", iVec);                                               \
      print(inFormat, in[iVec]);                                               \
      printf("\n");                                                            \
    }                                                                          \
    out = packs<Tout>(in);                                                     \
    printf("out   = ");                                                        \
    print(outFormat, out);                                                     \
    printf("\n");                                                              \
    LINE1;                                                                     \
  }

#define EXTEND(Tout, Tin, outFormat, inFormat, inStart, inFac, inOff)          \
  {                                                                            \
    const int numVecsOut = sizeof(Tout) / sizeof(Tin);                         \
    const int numElemsIn = SW / sizeof(Tin);                                   \
    printf(#Tin " -> %d " #Tout "\n", numVecsOut);                             \
    Vec<Tout, SW> out[numVecsOut];                                             \
    Vec<Tin, SW> in;                                                           \
    Tin buf[numElemsIn] __attribute__((aligned(SW)));                          \
    Tin val = inStart, fac = 1;                                                \
    for (int iElem = 0; iElem < numElemsIn; iElem++) {                         \
      buf[iElem] = val * fac;                                                  \
      val += inOff;                                                            \
      fac *= inFac;                                                            \
    }                                                                          \
    in = load<SW>(buf);                                                        \
    printf("in     = ");                                                       \
    print(inFormat, in);                                                       \
    printf("\n");                                                              \
    extend(in, out);                                                           \
    for (int oVec = 0; oVec < numVecsOut; oVec++) {                            \
      printf("out[%d] = ", oVec);                                              \
      print(outFormat, out[oVec]);                                             \
      printf("\n");                                                            \
    }                                                                          \
    LINE1;                                                                     \
  }

#define UNPACK(T, lohi, num, format)                                           \
  {                                                                            \
    const int numElems = SW / sizeof(T);                                       \
    printf("unpack<%d,%d> 2x %d " #T " elements\n", lohi, num, numElems);      \
    T inBuf[2][numElems] __attribute__((aligned(SW)));                         \
    for (int i = 0; i < numElems; i++) {                                       \
      inBuf[0][i] = i;                                                         \
      inBuf[1][i] = numElems + i;                                              \
    }                                                                          \
    Vec<T, SW> inVec[2], outVec;                                               \
    for (int i = 0; i < 2; i++) {                                              \
      inVec[i] = load<SW>(inBuf[i]);                                           \
      printf("inVec[%d] = ", i);                                               \
      print(format, inVec[i]);                                                 \
      printf("\n");                                                            \
    }                                                                          \
    outVec = unpack<lohi, num>(inVec[0], inVec[1]);                            \
    printf("outVec   = ");                                                     \
    print(format, outVec);                                                     \
    printf("\n");                                                              \
  }

#define FXMUL(OUTTYPE, INTYPE, OUTFORMAT, INFORMAT)                            \
  {                                                                            \
    const int nInVecs        = numInputSIMDVecs<OUTTYPE, INTYPE>();            \
    const int nOutVecs       = numOutputSIMDVecs<OUTTYPE, INTYPE>();           \
    const int nElemsPerInVec = (SW / sizeof(INTYPE));                          \
    const int nInElems       = numSIMDVecsElements<OUTTYPE, INTYPE, SW>();     \
    printf("\nINTYPE = " #INTYPE ", OUTTYPE = " #OUTTYPE "\n");                \
    printf("sizeof(INTYPE)=%d, sizeof(OUTTYPE)=%d\n", (int) sizeof(INTYPE),    \
           (int) sizeof(OUTTYPE));                                             \
    printf("nInVecs=%d, nOutVecs=%d, nInElems = %d\n", nInVecs, nOutVecs,      \
           nInElems);                                                          \
    INTYPE inBuf1[nInElems] __attribute__((aligned(SW)));                      \
    INTYPE inBuf2[nInElems] __attribute__((aligned(SW)));                      \
    for (int i = 0; i < nInElems; i++) {                                       \
      inBuf1[i] = i + 1;                                                       \
      inBuf2[i] = i + 2;                                                       \
    }                                                                          \
    Vec<INTYPE, SW> inVecs1[nInVecs], inVecs2[nInVecs];                        \
    for (int i = 0; i < nInVecs; i++) {                                        \
      inVecs1[i] = load<SW>(inBuf1 + i * nElemsPerInVec);                      \
      inVecs2[i] = load<SW>(inBuf2 + i * nElemsPerInVec);                      \
      printf("inVecs1[%d]: ", i);                                              \
      print(INFORMAT, inVecs1[i]);                                             \
      puts("");                                                                \
      printf("inVecs2[%d]: ", i);                                              \
      print(INFORMAT, inVecs2[i]);                                             \
      puts("");                                                                \
    }                                                                          \
    Vec<OUTTYPE, SW> outVecs[nOutVecs];                                        \
    puts("----- fmul(inVecs1, 3.5, outVecs) -----");                           \
    fmul(inVecs1, 3.5, outVecs);                                               \
    for (int i = 0; i < nOutVecs; i++) {                                       \
      printf("outVecs[%d]: ", i);                                              \
      print(OUTFORMAT, outVecs[i]);                                            \
      puts("");                                                                \
    }                                                                          \
    puts("----- faddmul(inVecs1, 1.5, 3.5, outVecs) -----");                   \
    faddmul(inVecs1, 1.5, 3.5, outVecs);                                       \
    for (int i = 0; i < nOutVecs; i++) {                                       \
      printf("outVecs[%d]: ", i);                                              \
      print(OUTFORMAT, outVecs[i]);                                            \
      puts("");                                                                \
    }                                                                          \
    puts("----- fdivmul(inVecs1, inVecs2, 35.0, outVecs) -----");              \
    fdivmul(inVecs1, inVecs2, 35.0, outVecs);                                  \
    for (int i = 0; i < nOutVecs; i++) {                                       \
      printf("outVecs[%d]: ", i);                                              \
      print(OUTFORMAT, outVecs[i]);                                            \
      puts("");                                                                \
    }                                                                          \
    puts("----- fwaddmul(inVecs1, inVecs2, 0.2, 35.0, outVecs) -----");        \
    fwaddmul(inVecs1, inVecs2, 0.2, 35.0, outVecs);                            \
    for (int i = 0; i < nOutVecs; i++) {                                       \
      printf("outVecs[%d]: ", i);                                              \
      print(OUTFORMAT, outVecs[i]);                                            \
      puts("");                                                                \
    }                                                                          \
  }

#define CONVERT(OUTTYPE, INTYPE, OUTFORMAT, INFORMAT)                          \
  {                                                                            \
    const int nInVecs        = numInputSIMDVecs<OUTTYPE, INTYPE>();            \
    const int nOutVecs       = numOutputSIMDVecs<OUTTYPE, INTYPE>();           \
    const int nElemsPerInVec = (SW / sizeof(INTYPE));                          \
    const int nInElems       = nInVecs * nElemsPerInVec;                       \
    printf("\nINTYPE = " #INTYPE ", OUTTYPE = " #OUTTYPE "\n");                \
    printf("sizeof(INTYPE)=%d, sizeof(OUTTYPE)=%d\n", (int) sizeof(INTYPE),    \
           (int) sizeof(OUTTYPE));                                             \
    printf("nInVecs=%d, nOutVecs=%d, nInElems = %d\n", nInVecs, nOutVecs,      \
           nInElems);                                                          \
    INTYPE inBuf[nInElems] __attribute__((aligned(SW)));                       \
    for (int i = 0; i < nInElems; i++) { inBuf[i] = i + 1; }                   \
    Vec<INTYPE, SW> inVecs[nInVecs];                                           \
    for (int i = 0; i < nInVecs; i++) {                                        \
      inVecs[i] = load<SW>(inBuf + i * nElemsPerInVec);                        \
      printf("inVecs[%d]: ", i);                                               \
      print(INFORMAT, inVecs[i]);                                              \
      puts("");                                                                \
    }                                                                          \
    Vec<OUTTYPE, SW> outVecs[nOutVecs];                                        \
    puts("----- convert(inVecs, outVecs) -----");                              \
    convert(inVecs, outVecs);                                                  \
    for (int i = 0; i < nOutVecs; i++) {                                       \
      printf("outVecs[%d]: ", i);                                              \
      print(OUTFORMAT, outVecs[i]);                                            \
      puts("");                                                                \
    }                                                                          \
  }

#define CONVERTVECS(OUTTYPE, INTYPE, OUTFORMAT, INFORMAT)                      \
  {                                                                            \
    const int nInVecs        = numInputSIMDVecs<OUTTYPE, INTYPE>();            \
    const int nOutVecs       = numOutputSIMDVecs<OUTTYPE, INTYPE>();           \
    const int nElemsPerInVec = numSIMDVecElements<INTYPE, SW>();               \
    const int nInElems       = nInVecs * nElemsPerInVec;                       \
    printf("\nINTYPE = " #INTYPE ", OUTTYPE = " #OUTTYPE "\n");                \
    printf("sizeof(INTYPE)=%d, sizeof(OUTTYPE)=%d\n", (int) sizeof(INTYPE),    \
           (int) sizeof(OUTTYPE));                                             \
    printf("nInVecs=%d, nOutVecs=%d, nInElems = %d\n", nInVecs, nOutVecs,      \
           nInElems);                                                          \
    INTYPE inBuf[nInElems] __attribute__((aligned(SW)));                       \
    for (int i = 0; i < nInElems; i++) { inBuf[i] = i + 1; }                   \
    Vecs<NumVecs<OUTTYPE, INTYPE>::in, INTYPE, SW> inVecs;                     \
    for (int i = 0; i < nInVecs; i++) {                                        \
      inVecs.vec[i] = load<SW>(inBuf + i * nElemsPerInVec);                    \
      printf("inVecs.vec[%d]: ", i);                                           \
      print(INFORMAT, inVecs.vec[i]);                                          \
      puts("");                                                                \
    }                                                                          \
    Vecs<NumVecs<OUTTYPE, INTYPE>::out, OUTTYPE, SW> outVecs;                  \
    puts("----- convert(inVecs, outVecs) -----");                              \
    convert(inVecs, outVecs);                                                  \
    for (int i = 0; i < nOutVecs; i++) {                                       \
      printf("outVecs.vec[%d]: ", i);                                          \
      print(OUTFORMAT, outVecs.vec[i]);                                        \
      puts("");                                                                \
    }                                                                          \
  }

#define HMINMAX(TYPE, FORMAT, INIT)                                            \
  {                                                                            \
    puts("\n" #TYPE);                                                          \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[nElems], val = TYPE(INIT);                                        \
    for (int i = 0; i < nElems; i++, val -= TYPE(1)) buf[i] = val;             \
    Vec<TYPE, SW> in = loadu<SW>(buf);                                         \
    printf("in = ");                                                           \
    print(FORMAT, in);                                                         \
    puts("");                                                                  \
    printf("hmin(in) = " FORMAT "\n", hmin(in));                               \
    printf("hmax(in) = " FORMAT "\n", hmax(in));                               \
  }

#define NEG(TYPE, FORMAT, INIT, STEP)                                          \
  {                                                                            \
    puts(#TYPE);                                                               \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[nElems], val = TYPE(INIT);                                        \
    for (int i = 0; i < nElems; i++, val += TYPE(STEP)) buf[i] = val;          \
    Vec<TYPE, SW> in = loadu<SW>(buf);                                         \
    printf("in           = ");                                                 \
    print(FORMAT, in);                                                         \
    puts("");                                                                  \
    printf("neg(in)      = ");                                                 \
    print(FORMAT, neg(in));                                                    \
    puts("");                                                                  \
    printf("neg(neg(in)) = ");                                                 \
    print(FORMAT, neg(neg(in)));                                               \
    puts("");                                                                  \
    puts("");                                                                  \
  }

#define HOR1(FCT, TYPE, FORMAT, INIT)                                          \
  {                                                                            \
    puts("\n" #TYPE);                                                          \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[nElems], val = TYPE(INIT);                                        \
    for (int i = 0; i < nElems; i++, val -= TYPE(1)) buf[i] = val;             \
    Vec<TYPE, SW> in = loadu<SW>(buf);                                         \
    printf("in = ");                                                           \
    print(FORMAT, in);                                                         \
    puts("");                                                                  \
    printf(#FCT "(in) = " FORMAT "\n", FCT(in));                               \
  }

#define HORn(FCT, TYPE, FORMAT)                                                \
  {                                                                            \
    puts("\n" #TYPE);                                                          \
    const int nElems = Vec<TYPE, SW>::elements;                                \
    Vec<TYPE, SW> v[nElems], res;                                              \
    TYPE buf[nElems];                                                          \
    for (unsigned i = 0; i < nElems; i++) {                                    \
      for (unsigned int j = 0; j < nElems; j++)                                \
        buf[j] = (2 * nElems - (i + j)) * 100;                                 \
      v[i] = loadu<SW>(buf);                                                   \
      printf("v[%2d] = ", i);                                                  \
      print(FORMAT, v[i]);                                                     \
      puts("");                                                                \
    }                                                                          \
    printf(#FCT "(v) = ");                                                     \
    print(FORMAT, FCT(v));                                                     \
    puts("");                                                                  \
  }

#define SHIFT(CMD, N, TYPE, FORMAT, VAL0, VALS)                                \
  {                                                                            \
    puts("\n" #TYPE);                                                          \
    const int nElems = Vec<TYPE, SW>::elements;                                \
    TYPE buf[nElems], val = VAL0;                                              \
    for (int i = 0; i < nElems; i++, val += VALS) buf[i] = val;                \
    Vec<TYPE, SW> in = loadu<SW>(buf), out;                                    \
    printf("in          = ");                                                  \
    print(FORMAT, in);                                                         \
    puts("");                                                                  \
    printf(#CMD "<%d>(in) = ", N);                                             \
    print(FORMAT, CMD<N>(in));                                                 \
    puts("");                                                                  \
    LINE1;                                                                     \
  }

#define SLLE_SRLE(TYPE, FORMAT, INIT, SHIFT)                                   \
  {                                                                            \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[nElems], val = TYPE(INIT);                                        \
    for (int i = 0; i < nElems; i++, val += TYPE(1)) { buf[i] = val; }         \
    Vec<TYPE, SW> in = loadu<SW>(buf), out;                                    \
    printf("in          = ");                                                  \
    print(FORMAT, in);                                                         \
    puts("");                                                                  \
    printf("slle(in,%2d) = ", SHIFT);                                          \
    print(FORMAT, slle<SHIFT>(in));                                            \
    puts("");                                                                  \
    printf("srle(in,%2d) = ", SHIFT);                                          \
    print(FORMAT, srle<SHIFT>(in));                                            \
    puts("");                                                                  \
    puts("");                                                                  \
  }

#define EXTRACT(TYPE, FORMAT, INIT, IDX)                                       \
  {                                                                            \
    const int nElems = Vec<TYPE, SW>::elements;                                \
    TYPE buf[nElems], val = TYPE(INIT);                                        \
    for (int i = 0; i < nElems; i++, val += TYPE(1)) buf[i] = val;             \
    Vec<TYPE, SW> in = loadu<SW>(buf);                                         \
    printf("in          = ");                                                  \
    print(FORMAT, in);                                                         \
    puts("");                                                                  \
    printf("in[" #IDX "] = " FORMAT "\n", extract<IDX>(in));                   \
    printf("elem0 = " FORMAT "\n", elem0(in));                                 \
  }

#define MIN_MAX(TYPE, FORMAT, INIT0, INIT1)                                    \
  {                                                                            \
    puts("\n" #TYPE);                                                          \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[2][nElems];                                                       \
    for (int i = 0; i < nElems; i++) {                                         \
      buf[0][i] = TYPE(INIT0 + i);                                             \
      buf[1][i] = TYPE(INIT1 - i);                                             \
    }                                                                          \
    Vec<TYPE, SW> a    = loadu<SW>(buf[0]);                                    \
    Vec<TYPE, SW> b    = loadu<SW>(buf[1]);                                    \
    Vec<TYPE, SW> cmin = min(a, b);                                            \
    Vec<TYPE, SW> cmax = max(a, b);                                            \
    printf("a        = ");                                                     \
    print(FORMAT, a);                                                          \
    puts("");                                                                  \
    printf("b        = ");                                                     \
    print(FORMAT, b);                                                          \
    puts("");                                                                  \
    printf("min(a,b) = ");                                                     \
    print(FORMAT, cmin);                                                       \
    puts("");                                                                  \
    printf("max(a,b) = ");                                                     \
    print(FORMAT, cmax);                                                       \
    puts("");                                                                  \
  }

#define ALIGNRE(TYPE, FORMAT, N)                                               \
  {                                                                            \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[2 * nElems], val = TYPE(0);                                       \
    for (int i = 0; i < 2 * nElems; i++, val += TYPE(1)) buf[i] = val;         \
    Vec<TYPE, SW> in[2];                                                       \
    in[0] = loadu<SW>(buf);                                                    \
    in[1] = loadu<SW>(buf + nElems);                                           \
    printf("\nin[0]  = ");                                                     \
    print(FORMAT, in[0]);                                                      \
    printf("\nin[1]  = ");                                                     \
    print(FORMAT, in[1]);                                                      \
    printf("\nal(%2d) = ", N);                                                 \
    print(FORMAT, alignre<N>(in[1], in[0]));                                   \
    puts("");                                                                  \
  }

#define SWIZZLE(TYPE, N, FORMAT)                                               \
  {                                                                            \
    printf("swizzle " #TYPE " " #N "\n\n");                                    \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[N * nElems];                                                      \
    for (int i = 0; i < N * nElems; i++) buf[i] = TYPE(i);                     \
    Vec<TYPE, SW> v[N];                                                        \
    TYPE *bufp = buf;                                                          \
    for (int i = 0; i < N; i++, bufp += nElems) {                              \
      v[i] = loadu<SW>(bufp);                                                  \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
    swizzle<N>(v);                                                             \
    for (int i = 0; i < N; i++, bufp += nElems) {                              \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
  }

#define SWIZZLE2(TYPE, N, FORMAT)                                              \
  {                                                                            \
    printf("swizzle2 " #TYPE " " #N "\n\n");                                   \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[2 * N * nElems];                                                  \
    for (int i = 0; i < 2 * N * nElems; i++) buf[i] = TYPE(i);                 \
    Vec<TYPE, SW> v[2 * N];                                                    \
    TYPE *bufp = buf;                                                          \
    for (int i = 0; i < 2 * N; i++, bufp += nElems) {                          \
      v[i] = loadu<SW>(bufp);                                                  \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
    swizzle2<N>(v);                                                            \
    for (int i = 0; i < 2 * N; i++, bufp += nElems) {                          \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
    unswizzle2<N>(v);                                                          \
    for (int i = 0; i < 2 * N; i++, bufp += nElems) {                          \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
  }

#define UNSWIZZLE2(TYPE, N, FORMAT)                                            \
  {                                                                            \
    printf("unswizzle2 " #TYPE " " #N "\n\n");                                 \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE buf[2 * N * nElems];                                                  \
    for (int i = 0; i < 2 * N * nElems; i++) buf[i] = TYPE(i);                 \
    Vec<TYPE, SW> v[2 * N];                                                    \
    TYPE *bufp = buf;                                                          \
    for (int i = 0; i < 2 * N; i++, bufp += nElems) {                          \
      v[i] = loadu<SW>(bufp);                                                  \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
    unswizzle2<N>(v);                                                          \
    for (int i = 0; i < 2 * N; i++, bufp += nElems) {                          \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
    swizzle2<N>(v);                                                            \
    for (int i = 0; i < 2 * N; i++, bufp += nElems) {                          \
      printf("v[%d] = ", i);                                                   \
      print(FORMAT, v[i]);                                                     \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
  }

// independent of SW
#define SWIZZLE_32_16(TYPE, N, FORMAT)                                         \
  {                                                                            \
    printf("swizzle_32_16 " #TYPE " " #N "\n\n");                              \
    const int nElems = 32 / sizeof(TYPE);                                      \
    TYPE buf[N * nElems];                                                      \
    for (int i = 0; i < N; i++)                                                \
      for (int j = 0; j < nElems / 2; j++) {                                   \
        buf[i * nElems + j]              = 2 * i;                              \
        buf[i * nElems + nElems / 2 + j] = 2 * i + 1;                          \
      }                                                                        \
    Vec<TYPE, 32> vIn[N], vOut[N];                                             \
    TYPE *bufp = buf;                                                          \
    for (int i = 0; i < N; i++, bufp += nElems) {                              \
      vIn[i] = loadu<32>(bufp);                                                \
      printf("vIn[%d] =  ", i);                                                \
      print(FORMAT, vIn[i]);                                                   \
      printf("\n");                                                            \
    }                                                                          \
    puts("");                                                                  \
    internal::base::swizzle_32_16<N>(vIn, vOut);                               \
    for (int i = 0; i < N; i++, bufp += nElems) {                              \
      printf("vOut[%d] = ", i);                                                \
      print(FORMAT, vOut[i]);                                                  \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
  }

#define CMP(TYPE, FORMAT)                                                      \
  {                                                                            \
    puts("\n" #TYPE);                                                          \
    const int nElems = SW / sizeof(TYPE);                                      \
    TYPE val1, val2;                                                           \
    if (SIMDTypeInfo<TYPE>::isSigned) {                                        \
      val1 = TYPE(-nElems / 2);                                                \
      val2 = TYPE(nElems / 2);                                                 \
    } else {                                                                   \
      val1 = TYPE(0);                                                          \
      val2 = TYPE(nElems);                                                     \
    }                                                                          \
    TYPE buf1[nElems], buf2[nElems];                                           \
    for (int i = 0; i < nElems; i++, val1 += TYPE(1), val2 -= TYPE(1)) {       \
      buf1[i] = val1;                                                          \
      buf2[i] = val2;                                                          \
    }                                                                          \
    Vec<TYPE, SW> in1 = loadu<SW>(buf1);                                       \
    Vec<TYPE, SW> in2 = loadu<SW>(buf2);                                       \
    printf("in1        = ");                                                   \
    print(FORMAT, in1);                                                        \
    puts("");                                                                  \
    printf("in2        = ");                                                   \
    print(FORMAT, in2);                                                        \
    puts("");                                                                  \
    Vec<TYPE, SW> out;                                                         \
    printf("in1 <  in2 = ");                                                   \
    print(FORMAT, cmplt(in1, in2));                                            \
    puts("");                                                                  \
    printf("in1 <= in2 = ");                                                   \
    print(FORMAT, cmple(in1, in2));                                            \
    puts("");                                                                  \
    printf("in1 == in2 = ");                                                   \
    print(FORMAT, cmpeq(in1, in2));                                            \
    puts("");                                                                  \
    printf("in1 >= in2 = ");                                                   \
    print(FORMAT, cmpge(in1, in2));                                            \
    puts("");                                                                  \
    printf("in1 >  in2 = ");                                                   \
    print(FORMAT, cmpgt(in1, in2));                                            \
    puts("");                                                                  \
    printf("in1 != in2 = ");                                                   \
    print(FORMAT, cmpneq(in1, in2));                                           \
    puts("");                                                                  \
    puts("");                                                                  \
    printf("if(in1<in2) =");                                                   \
    print(FORMAT, ifelse(cmplt(in1, in2), in1, in2));                          \
    puts("");                                                                  \
  }

#define AVG(AVGFCT, TYPE, FORMAT)                                              \
  {                                                                            \
    printf(#AVGFCT " " #TYPE "\n");                                            \
    TYPE loLimit, hiLimit;                                                     \
    if (SIMDTypeInfo<TYPE>::isSigned) {                                        \
      loLimit = TYPE(-4);                                                      \
      hiLimit = TYPE(4);                                                       \
    } else {                                                                   \
      loLimit = TYPE(0);                                                       \
      hiLimit = TYPE(4);                                                       \
    }                                                                          \
    Vec<TYPE, SW> a, b, r;                                                     \
    for (TYPE j = loLimit; j <= hiLimit; j++) printf(FORMAT, j);               \
    printf("\n\n");                                                            \
    for (TYPE i = loLimit; i <= hiLimit; i++) {                                \
      a = set1<TYPE, SW>(i);                                                   \
      for (TYPE j = loLimit; j <= hiLimit; j++) {                              \
        b = set1<TYPE, SW>(j);                                                 \
        r = AVGFCT(a, b);                                                      \
        printf(FORMAT, elem0(r));                                              \
      }                                                                        \
      printf("  |  " FORMAT "\n", i);                                          \
    }                                                                          \
  }

#define DIV2(TYPE, FORMAT, LOVAL)                                              \
  {                                                                            \
    printf("div2* " #TYPE "\n");                                               \
    Vec<TYPE, SW> a, r0, rd;                                                   \
    TYPE v = TYPE(LOVAL);                                                      \
    TYPE buf[Vec<TYPE, SW>::elements];                                         \
    for (int i = 0; i < Vec<TYPE, SW>::elements; i++, v++) buf[i] = v;         \
    a  = loadu<SW>(buf);                                                       \
    r0 = div2r0(a);                                                            \
    rd = div2rd(a);                                                            \
    printf("a     ");                                                          \
    print(FORMAT, a);                                                          \
    puts("");                                                                  \
    printf("divr0 ");                                                          \
    print(FORMAT, r0);                                                         \
    puts("");                                                                  \
    printf("divrd ");                                                          \
    print(FORMAT, rd);                                                         \
    puts("");                                                                  \
  }

int main()
{
  printf("SW = %d\n\n", SW);

  puts("\n*********** test of SIMD_ALIGN_CHK **********\n");

  Vec<Byte, SW> v = setzero<Byte, SW>();
  Byte ba[2 * SW];
  store(ba, v);
  // this triggers the assertion if SIMD_ALIGN_CHK is defined, but
  // does not always lead to a seg fault:
  // store(ba+1, v);

  puts("\n*********** test of hadd/hsub  ***********\n");

  HOR(hadd, Word, "%u ", 30000, 1000, 5000, 10000);
  HOR(hadd, Short, "%d ", 30000, 1000, 5000, 10000);
  HOR(hadd, Int, "%d ", 30000, 1000, 5000, 10000);
  HOR(hadd, Float, "%.0f ", 30000, 1000, 5000, 10000);

  HOR(hsub, Word, "%u ", 30000, 1000, 5000, 10000);
  HOR(hsub, Short, "%d ", 30000, 1000, 5000, 10000);
  HOR(hsub, Int, "%d ", 30000, 1000, 5000, 10000);
  HOR(hsub, Float, "%.0f ", 30000, 1000, 5000, 10000);

  HOR(hadds, Short, "%d ", 30000, 1000, 5000, 10000);
  HOR(hadds, Int, "%d ", 30000, 1000, 5000, 10000);
  HOR(hadds, Float, "%.0f ", 30000, 1000, 5000, 10000);

  HOR(hsubs, Short, "%d ", 30000, 1000, 5000, 10000);
  HOR(hsubs, Int, "%d ", 30000, 1000, 5000, 10000);
  HOR(hsubs, Float, "%.0f ", 30000, 1000, 5000, 10000);

  puts("\n*********** test of min/max  ***********\n");

  MIN_MAX(Byte, "%2d ", 10, 20);
  MIN_MAX(SignedByte, "%2d ", -5, 10);
  MIN_MAX(Word, "%2d ", 10, 20);
  MIN_MAX(Short, "%2d ", -5, 5);
  MIN_MAX(Int, "%2d ", -2, 2);
  MIN_MAX(Float, "%2g ", -2, 2);

  puts("\n*********** test of f*mul  ***********\n");

  FXMUL(Byte, Byte, "%d ", "%d ");
  FXMUL(Byte, Word, "%d ", "%d ");
  FXMUL(Word, Byte, "%d ", "%d ");

  puts("\n*********** test of convert ***********\n");

  CONVERT(Short, Short, "%d ", "%d ");
  CONVERT(Int, Short, "%d ", "%d ");
  CONVERT(Short, Int, "%d ", "%d ");

  puts("\n*********** test of hmin/hmax ***********\n");

  HMINMAX(Byte, "%d ", 33);
  HMINMAX(SignedByte, "%d ", 5);
  HMINMAX(Word, "%d ", 33);
  HMINMAX(Short, "%d ", 5);
  HMINMAX(Int, "%d ", 2);
  HMINMAX(Float, "%g ", 2);

  puts("\n*********** test of horizontal fct(vector)  ***********\n");

  HOR1(hadd, Word, "%u ", 33);
  HOR1(hadd, Short, "%d ", 5);
  HOR1(hadd, Int, "%d ", 33);
  HOR1(hadd, Float, "%g ", 5);

  puts("\n*********** test of horizontal fct(vectors)  ***********\n");

  HORn(hadd, Word, "%u ");
  HORn(hadd, Short, "%d ");
  HORn(hadd, Int, "%d ");
  HORn(hadd, Float, "%g ");

  HORn(hadds, Short, "%d ");
  HORn(hadds, Int, "%d ");
  HORn(hadds, Float, "%g ");

  puts("\n*********** test of cmp* ***********\n");

  CMP(Byte, "%4d");
  CMP(SignedByte, "%4d");
  CMP(Word, "%6d");
  CMP(Short, "%6d");
  CMP(Int, "%4d");
  CMP(Float, "%6g");

  puts("\n*********** test of srai/srli/slli **********\n");

  SHIFT(srai, 1, Word, "0x%04hx ", 0x0000, 0x2000);
  SHIFT(srai, 1, Short, "0x%04hx ", -10, 2);
  SHIFT(srai, 1, Int, "0x%08x ", -10, 2);

  SHIFT(srli, 1, Byte, "0x%02hhx ", 0x00, 0x20);
  SHIFT(srli, 1, SignedByte, "0x%02hhx ", -10, 2);
  SHIFT(srli, 1, Word, "0x%04hx ", 0x0000, 0x2000);
  SHIFT(srli, 1, Short, "0x%04hx ", -10, 2);
  SHIFT(srli, 1, Int, "0x%08x ", -10, 2);

  SHIFT(slli, 1, Byte, "0x%02hhx ", 0x00, 0x20);
  SHIFT(slli, 1, SignedByte, "0x%02hhx ", -10, 2);
  SHIFT(slli, 1, Word, "0x%04hx ", 0x0000, 0x2000);
  SHIFT(slli, 1, Short, "0x%04hx ", -10, 2);
  SHIFT(slli, 1, Int, "0x%08x ", -10, 2);

  puts("\n*********** test of srle/slle **********\n");

  SLLE_SRLE(Byte, "%02x ", 0, 0);
  SLLE_SRLE(Byte, "%02x ", 0, 1);
  SLLE_SRLE(Byte, "%02x ", 0, 2);
  SLLE_SRLE(Byte, "%02x ", 0, 3);
  SLLE_SRLE(Byte, "%02x ", 0, 4);
  SLLE_SRLE(Byte, "%02x ", 0, 5);
  SLLE_SRLE(Byte, "%02x ", 0, 6);
  SLLE_SRLE(Byte, "%02x ", 0, 7);
  SLLE_SRLE(Byte, "%02x ", 0, 8);
  SLLE_SRLE(Byte, "%02x ", 0, 9);
  SLLE_SRLE(Byte, "%02x ", 0, 10);
  SLLE_SRLE(Byte, "%02x ", 0, 11);
  SLLE_SRLE(Byte, "%02x ", 0, 12);
  SLLE_SRLE(Byte, "%02x ", 0, 13);
  SLLE_SRLE(Byte, "%02x ", 0, 14);
  SLLE_SRLE(Byte, "%02x ", 0, 15);
  SLLE_SRLE(Byte, "%02x ", 0, 16);
  SLLE_SRLE(Byte, "%02x ", 0, 17);
  SLLE_SRLE(Byte, "%02x ", 0, 18);
  SLLE_SRLE(Byte, "%02x ", 0, 19);
  SLLE_SRLE(Byte, "%02x ", 0, 20);
  SLLE_SRLE(Byte, "%02x ", 0, 21);
  SLLE_SRLE(Byte, "%02x ", 0, 22);
  SLLE_SRLE(Byte, "%02x ", 0, 23);
  SLLE_SRLE(Byte, "%02x ", 0, 24);
  SLLE_SRLE(Byte, "%02x ", 0, 25);
  SLLE_SRLE(Byte, "%02x ", 0, 26);
  SLLE_SRLE(Byte, "%02x ", 0, 27);
  SLLE_SRLE(Byte, "%02x ", 0, 28);
  SLLE_SRLE(Byte, "%02x ", 0, 29);
  SLLE_SRLE(Byte, "%02x ", 0, 30);
  SLLE_SRLE(Byte, "%02x ", 0, 31);
  SLLE_SRLE(Byte, "%02x ", 0, 32);
  SLLE_SRLE(Float, "%2g ", 0, 2);

  puts("\n*********** test of extract and elem0 ************\n");

  EXTRACT(Byte, "%02x ", 10, 3);
  EXTRACT(Word, "%02x ", 10, 3);
  EXTRACT(Float, "%2g ", 10, 3);

  puts("\n*********** test of alignre ************\n");

  ALIGNRE(Byte, "%02x ", 0);
  ALIGNRE(Byte, "%02x ", 5);
  ALIGNRE(Byte, "%02x ", 16);
  ALIGNRE(Byte, "%02x ", 18);
  ALIGNRE(Byte, "%02x ", 26);
  ALIGNRE(Byte, "%02x ", 32);
  ALIGNRE(Byte, "%02x ", 37);
  ALIGNRE(Byte, "%02x ", 48);
  ALIGNRE(Byte, "%02x ", 50);
  ALIGNRE(Byte, "%02x ", 58);
  ALIGNRE(Byte, "%02x ", 64);
  ALIGNRE(Byte, "%02x ", 68);

  puts("\n*********** test of swizzle ************\n");

  SWIZZLE(Byte, 1, "%3d ");
  SWIZZLE(Byte, 2, "%3d ");
  SWIZZLE(Byte, 3, "%3d ");
  SWIZZLE(Byte, 4, "%3d ");
  SWIZZLE(Byte, 5, "%3d ");

  SWIZZLE(SignedByte, 1, "%3d ");
  SWIZZLE(SignedByte, 2, "%3d ");
  SWIZZLE(SignedByte, 3, "%3d ");
  SWIZZLE(SignedByte, 4, "%3d ");
  SWIZZLE(SignedByte, 5, "%3d ");

  SWIZZLE(Word, 1, "%3d ");
  SWIZZLE(Word, 2, "%3d ");
  SWIZZLE(Word, 3, "%3d ");
  SWIZZLE(Word, 4, "%3d ");
  SWIZZLE(Word, 5, "%3d ");

  SWIZZLE(Short, 1, "%3d ");
  SWIZZLE(Short, 2, "%3d ");
  SWIZZLE(Short, 3, "%3d ");
  SWIZZLE(Short, 4, "%3d ");
  SWIZZLE(Short, 5, "%3d ");

  SWIZZLE(Int, 1, "%3d ");
  SWIZZLE(Int, 2, "%3d ");
  SWIZZLE(Int, 3, "%3d ");
  SWIZZLE(Int, 4, "%3d ");
  SWIZZLE(Int, 5, "%3d ");

  SWIZZLE(Float, 1, "%3g ");
  SWIZZLE(Float, 2, "%3g ");
  SWIZZLE(Float, 3, "%3g ");
  SWIZZLE(Float, 4, "%3g ");
  SWIZZLE(Float, 5, "%3g ");

  puts("\n*********** test of swizzle2 ************\n");

  // here we could also test higher values than 5

  SWIZZLE2(Byte, 1, "%3d ");
  SWIZZLE2(Byte, 2, "%3d ");
  SWIZZLE2(Byte, 3, "%3d ");
  SWIZZLE2(Byte, 4, "%3d ");
  SWIZZLE2(Byte, 5, "%3d ");

  SWIZZLE2(SignedByte, 1, "%3d ");
  SWIZZLE2(SignedByte, 2, "%3d ");
  SWIZZLE2(SignedByte, 3, "%3d ");
  SWIZZLE2(SignedByte, 4, "%3d ");
  SWIZZLE2(SignedByte, 5, "%3d ");

  SWIZZLE2(Word, 1, "%3d ");
  SWIZZLE2(Word, 2, "%3d ");
  SWIZZLE2(Word, 3, "%3d ");
  SWIZZLE2(Word, 4, "%3d ");
  SWIZZLE2(Word, 5, "%3d ");

  SWIZZLE2(Short, 1, "%3d ");
  SWIZZLE2(Short, 2, "%3d ");
  SWIZZLE2(Short, 3, "%3d ");
  SWIZZLE2(Short, 4, "%3d ");
  SWIZZLE2(Short, 5, "%3d ");

  SWIZZLE2(Int, 1, "%3d ");
  SWIZZLE2(Int, 2, "%3d ");
  SWIZZLE2(Int, 3, "%3d ");
  SWIZZLE2(Int, 4, "%3d ");
  SWIZZLE2(Int, 5, "%3d ");

  SWIZZLE2(Float, 1, "%3g ");
  SWIZZLE2(Float, 2, "%3g ");
  SWIZZLE2(Float, 3, "%3g ");
  SWIZZLE2(Float, 4, "%3g ");
  SWIZZLE2(Float, 5, "%3g ");

  puts("\n*********** test of unswizzle2 ************\n");

  // here we could also test higher values than 5

  UNSWIZZLE2(Byte, 1, "%3d ");
  UNSWIZZLE2(Byte, 2, "%3d ");
  UNSWIZZLE2(Byte, 3, "%3d ");
  UNSWIZZLE2(Byte, 4, "%3d ");
  UNSWIZZLE2(Byte, 5, "%3d ");

  UNSWIZZLE2(SignedByte, 1, "%3d ");
  UNSWIZZLE2(SignedByte, 2, "%3d ");
  UNSWIZZLE2(SignedByte, 3, "%3d ");
  UNSWIZZLE2(SignedByte, 4, "%3d ");
  UNSWIZZLE2(SignedByte, 5, "%3d ");

  UNSWIZZLE2(Word, 1, "%3d ");
  UNSWIZZLE2(Word, 2, "%3d ");
  UNSWIZZLE2(Word, 3, "%3d ");
  UNSWIZZLE2(Word, 4, "%3d ");
  UNSWIZZLE2(Word, 5, "%3d ");

  UNSWIZZLE2(Short, 1, "%3d ");
  UNSWIZZLE2(Short, 2, "%3d ");
  UNSWIZZLE2(Short, 3, "%3d ");
  UNSWIZZLE2(Short, 4, "%3d ");
  UNSWIZZLE2(Short, 5, "%3d ");

  UNSWIZZLE2(Int, 1, "%3d ");
  UNSWIZZLE2(Int, 2, "%3d ");
  UNSWIZZLE2(Int, 3, "%3d ");
  UNSWIZZLE2(Int, 4, "%3d ");
  UNSWIZZLE2(Int, 5, "%3d ");

  UNSWIZZLE2(Float, 1, "%3g ");
  UNSWIZZLE2(Float, 2, "%3g ");
  UNSWIZZLE2(Float, 3, "%3g ");
  UNSWIZZLE2(Float, 4, "%3g ");
  UNSWIZZLE2(Float, 5, "%3g ");

  puts("\n*********** test of swizzle_32_16 ************\n");

#if defined(_SIMD_VEC_32_AVAIL_) && !defined(SIMDVEC_SANDBOX)
  SWIZZLE_32_16(Byte, 1, "%2d ");
  SWIZZLE_32_16(Byte, 2, "%2d ");
  SWIZZLE_32_16(Byte, 3, "%2d ");
  SWIZZLE_32_16(Byte, 4, "%2d ");
  SWIZZLE_32_16(Byte, 5, "%2d ");

  SWIZZLE_32_16(Word, 1, "%2d ");
  SWIZZLE_32_16(Word, 2, "%2d ");
  SWIZZLE_32_16(Word, 3, "%2d ");
  SWIZZLE_32_16(Word, 4, "%2d ");
  SWIZZLE_32_16(Word, 5, "%2d ");

  SWIZZLE_32_16(Int, 1, "%2d ");
  SWIZZLE_32_16(Int, 2, "%2d ");
  SWIZZLE_32_16(Int, 3, "%2d ");
  SWIZZLE_32_16(Int, 4, "%2d ");
  SWIZZLE_32_16(Int, 5, "%2d ");

  SWIZZLE_32_16(Float, 1, "%2g ");
  SWIZZLE_32_16(Float, 2, "%2g ");
  SWIZZLE_32_16(Float, 3, "%2g ");
  SWIZZLE_32_16(Float, 4, "%2g ");
  SWIZZLE_32_16(Float, 5, "%2g ");
#endif

  puts("\n*********** test of reinterpret functions  ***********\n");

  const int nFloats = SW / sizeof(Float);
  Float buf[nFloats] __attribute__((aligned(SW)));
  for (int i = 0; i < nFloats; i++) buf[i] = (Float) i;
  Vec<Float, SW> fv;
  fv = load<SW>(buf);
  print("%3g ", fv);
  puts("");
  Vec<Int, SW> iv;
  iv = reinterpret<Int>(fv);
  print("%08x ", iv);
  puts("");
  Vec<Short, SW> sv;
  sv = reinterpret<Short>(iv);
  print("%04x ", sv);
  puts("");
  Vec<Word, SW> wv;
  wv = reinterpret<Word>(sv);
  print("%04x ", sv);
  puts("");
  Vec<Float, SW> fv1;
  fv1 = reinterpret<Float>(sv);
  print("%3g ", fv1);
  puts("");

  puts("\n*********** test of generic packs functions ***********\n");

  LINE2;
  PACKS(SignedByte, SignedByte, "%3d ", "%3d ", 0, -1, 8);

  LINE2;
  PACKS(SignedByte, Short, "%7d ", "%7d ", 0, -1, 20);
  PACKS(Byte, Short, "%7u ", "%7d ", 0, -1, 40);
  PACKS(Short, Short, "%7d ", "%7d ", 0, -1, 2000);

  LINE2;
  PACKS(SignedByte, Int, "%7d ", "%7d ", 0, -1, 20);
  PACKS(Byte, Int, "%7u ", "%7d ", 0, -1, 40);
  PACKS(Short, Int, "%7d ", "%7d ", 0, -1, 6000);
  PACKS(Word, Int, "%7u ", "%7d ", 0, -1, 15000);
  PACKS(Int, Int, "%7d ", "%7d ", 0, -1, 1000000);
  PACKS(Float, Int, "%7g ", "%7d ", 0, -1, 100);

  LINE2;
  PACKS(SignedByte, Float, "%7d ", "%7g ", 0, -1, 20);
  PACKS(Byte, Float, "%7u ", "%7g ", 0, -1, 40);
  PACKS(Short, Float, "%7d ", "%7g ", 0, -1, 6000);
  PACKS(Word, Float, "%7u ", "%7g ", 0, -1, 15000);
  PACKS(Int, Float, "%7d ", "%7g ", 0, -1, 1000);
  PACKS(Float, Float, "%7g ", "%7g ", 0, -1, 100);

  puts("\n*********** test of generic extend functions ***********\n");

  LINE2;
  EXTEND(SignedByte, SignedByte, "%4d ", "%4d ", 0, -1, 8);
  EXTEND(Short, SignedByte, "%4d ", "%4d ", 0, -1, 8);
  EXTEND(Int, SignedByte, "%4d ", "%4d ", 0, -1, 8);
  EXTEND(Float, SignedByte, "%4g ", "%4d ", 0, -1, 8);

  LINE2;
  EXTEND(Byte, Byte, "%4u ", "%4u ", 0, 1, 15);
  EXTEND(Short, Byte, "%4d ", "%4u ", 0, 1, 15);
  EXTEND(Word, Byte, "%4u ", "%4u ", 0, 1, 15);
  EXTEND(Int, Byte, "%4d ", "%4u ", 0, 1, 15);
  EXTEND(Float, Byte, "%4g ", "%4u ", 0, 1, 15);

  LINE2;
  EXTEND(Short, Short, "%7d ", "%7d ", 0, -1, 4000);
  EXTEND(Int, Short, "%7d ", "%7d ", 0, -1, 4000);
  EXTEND(Float, Short, "%7g ", "%7d ", 0, -1, 4000);

  LINE2;
  EXTEND(Word, Word, "%7u ", "%7u ", 0, 1, 8000);
  EXTEND(Int, Word, "%7d ", "%7u ", 0, 1, 8000);
  EXTEND(Float, Word, "%7g ", "%7u ", 0, 1, 8000);

  LINE2;
  EXTEND(Int, Int, "%7d ", "%7d ", 0, -1, 100);
  EXTEND(Float, Int, "%7g ", "%7d ", 0, -1, 100);

  LINE2;
  EXTEND(Int, Float, "%7d ", "%7g ", 0, -1, 100);
  EXTEND(Float, Float, "%7g ", "%7g ", 0, -1, 100);

  puts("\n*********** test of unpack functions ***********\n");

  LINE2;
  puts("unpack low");
  LINE2;

  UNPACK(Byte, 0, 1, "%02d ");
  UNPACK(Byte, 0, 2, "%02d ");
  UNPACK(Byte, 0, 4, "%02d ");
  UNPACK(Byte, 0, 8, "%02d ");
#if SW == 32
  UNPACK(Byte, 0, 16, "%02d ");
#endif
  LINE1;

  UNPACK(SignedByte, 0, 1, "%02d ");
  UNPACK(SignedByte, 0, 2, "%02d ");
  UNPACK(SignedByte, 0, 4, "%02d ");
  UNPACK(SignedByte, 0, 8, "%02d ");
#if SW == 32
  UNPACK(SignedByte, 0, 16, "%02d ");
#endif
  LINE1;

  UNPACK(Word, 0, 1, "%02d ");
  UNPACK(Word, 0, 2, "%02d ");
  UNPACK(Word, 0, 4, "%02d ");
#if SW == 32
  UNPACK(Word, 0, 8, "%02d ");
#endif
  LINE1;

  UNPACK(Short, 0, 1, "%02d ");
  UNPACK(Short, 0, 2, "%02d ");
  UNPACK(Short, 0, 4, "%02d ");
#if SW == 32
  UNPACK(Short, 0, 8, "%02d ");
#endif
  LINE1;

  UNPACK(Int, 0, 1, "%02d ");
  UNPACK(Int, 0, 2, "%02d ");
#if SW == 32
  UNPACK(Int, 0, 4, "%02d ");
#endif
  LINE1;

  UNPACK(Float, 0, 1, "%2g ");
  UNPACK(Float, 0, 2, "%2g ");
#if SW == 32
  UNPACK(Float, 0, 4, "%2g ");
#endif

  LINE2;
  puts("unpack high");
  LINE2;

  UNPACK(Byte, 1, 1, "%02d ");
  UNPACK(Byte, 1, 2, "%02d ");
  UNPACK(Byte, 1, 4, "%02d ");
  UNPACK(Byte, 1, 8, "%02d ");
#if SW == 32
  UNPACK(Byte, 1, 16, "%02d ");
#endif
  LINE1;

  UNPACK(SignedByte, 1, 1, "%02d ");
  UNPACK(SignedByte, 1, 2, "%02d ");
  UNPACK(SignedByte, 1, 4, "%02d ");
  UNPACK(SignedByte, 1, 8, "%02d ");
#if SW == 32
  UNPACK(SignedByte, 1, 16, "%02d ");
#endif
  LINE1;

  UNPACK(Word, 1, 1, "%02d ");
  UNPACK(Word, 1, 2, "%02d ");
  UNPACK(Word, 1, 4, "%02d ");
#if SW == 32
  UNPACK(Word, 1, 8, "%02d ");
#endif
  LINE1;

  UNPACK(Short, 1, 1, "%02d ");
  UNPACK(Short, 1, 2, "%02d ");
  UNPACK(Short, 1, 4, "%02d ");
#if SW == 32
  UNPACK(Short, 1, 8, "%02d ");
#endif
  LINE1;

  UNPACK(Int, 1, 1, "%02d ");
  UNPACK(Int, 1, 2, "%02d ");
#if SW == 32
  UNPACK(Int, 1, 4, "%02d ");
#endif
  LINE1;

  UNPACK(Float, 1, 1, "%2g ");
  UNPACK(Float, 1, 2, "%2g ");
#if SW == 32
  UNPACK(Float, 1, 4, "%2g ");
#endif
  LINE1;

  puts("\n*********** test of div2* ***********\n");

  DIV2(Byte, "%2u ", 0);
  LINE1;
  // DIV2(SignedByte, "%2d ", -4);
  // LINE1;
  DIV2(Word, "%2u ", 0);
  LINE1;
  DIV2(Short, "%2d ", -4);
  LINE1;
  DIV2(Int, "%2d ", -4);

  puts("\n*********** test of avg* ***********\n");

  AVG(avg, Byte, "%2u ");
  LINE1;
  AVG(avg, SignedByte, "%2d ");
  LINE1;
  AVG(avg, Word, "%2u ");
  LINE1;
  AVG(avg, Short, "%2d ");
  LINE1;
  AVG(avg, Int, "%2d ");
  LINE1;
  AVG(avg, Float, "%5g ");
  LINE2;
  AVG(avgrd, Byte, "%2u ");
  LINE1;
  // AVG(avgrd, SignedByte, "%2d ");
  // LINE1;
  AVG(avgrd, Word, "%2u ");
  LINE1;
  AVG(avgrd, Short, "%2d ");
  LINE1;
  AVG(avgrd, Int, "%2d ");
  LINE1;
  AVG(avgrd, Float, "%5g ");

  puts("\n*********** test of Vecs functions ***********\n");

#define INTYPE  Word
#define OUTTYPE Float
  Vecs<NumVecs<OUTTYPE, INTYPE>::in, INTYPE, SW> inVecs;
  Vecs<NumVecs<OUTTYPE, INTYPE>::out, OUTTYPE, SW> outVecs;
  printf("inVecs:  vectors = %d, elements = %d\n", inVecs.vectors,
         inVecs.elements);
  printf("outVecs: vectors = %d, elements = %d\n", outVecs.vectors,
         outVecs.elements);

  CONVERTVECS(Short, Int, "%d ", "%d ");
  fdivmul(inVecs, inVecs, 1.23, outVecs);
  fmul(inVecs, 1.23, outVecs);
  faddmul(inVecs, 10, 1.23, outVecs);
  fmuladd(inVecs, 1.23, 10, outVecs);
  fwaddmul(inVecs, inVecs, 0.5, 43.21, outVecs);

  Vecs<4, Float, SW> vecs;
  Float buffer[Vecs<4, Float, SW>::elements] SIMD_ATTR_ALIGNED(SW);
  load(buffer, vecs);
  loadu(buffer, vecs);
  store(buffer, vecs);
  storeu(buffer, vecs);
#define LARGETYPE Float
#define SMALLTYPE Word
  Vecs<NumVecs<LARGETYPE, SMALLTYPE>::out, LARGETYPE, SW> largeTypeVecs;
  Vec<SMALLTYPE, SW> smallTypeVec;
  setzero(largeTypeVecs);
  smallTypeVec = packs<SMALLTYPE>(largeTypeVecs);
  extend(smallTypeVec, largeTypeVecs);
#define NSWIZZLE    3
#define SWIZZLETYPE Word
  Vecs<NSWIZZLE, SWIZZLETYPE, SW> swizzleVecs;
  setzero(swizzleVecs);
  swizzle<NSWIZZLE>(swizzleVecs);
#define TRANSPOSETYPE    Word
#define NTRANSPOSE       (SW / sizeof(TRANSPOSETYPE))
#define NUMTRANSPOSEROWS 5
  // e.g. NTRANSPOSE-1 is passed: error!
  Vecs<NTRANSPOSE, TRANSPOSETYPE, SW> inRows, outRows;
  setzero(inRows);
  Vecs<NUMTRANSPOSEROWS, TRANSPOSETYPE, SW> outRowsPart;
  // 30. Sep 22 (rm): was called transpose1, moved back to transpose
  transpose(inRows, outRows);
  transposePartial(inRows, outRowsPart);
#define HORTYPE Short
#define NHOR    (SW / sizeof(HORTYPE))
  Vecs<NHOR, HORTYPE, SW> horInVecs;
  setzero(horInVecs);
  Vec<HORTYPE, SW> horOutVec;
  horOutVec = hadd(horInVecs);
  horOutVec = hadds(horInVecs);

  puts("\n*********** test of print functions ***********\n");

  Vec<Float, SW> vf = set1<Float, SW>(1.234);
  Vec<Int, SW> vi   = set1<Int, SW>(98765);
  print(SIMDFormat<Float>(5, 2).format, " sep ", vf);
  printf("\n");
  print("%d", ", ", vi);
  printf("\n");

  puts("\n*********** test of test_all_* functions ***********\n");

  Vec<Word, SW> zeros = setzero<Word, SW>();
  Vec<Word, SW> ones  = set1<Word, SW>(0xffff);
  Vec<Word, SW> some  = set1<Word, SW>(0x3333);
  printf("zeros = ");
  print("%04x ", zeros);
  puts("");
  printf("ones  = ");
  print("%04x ", ones);
  puts("");
  printf("some  = ");
  print("%04x ", some);
  puts("");

  printf("test_all_zeros(zeros) = %d\n", test_all_zeros(zeros));
  printf("test_all_zeros(some)  = %d\n", test_all_zeros(some));
  printf("test_all_zeros(ones)  = %d\n", test_all_zeros(ones));
  printf("test_all_ones(zeros)  = %d\n", test_all_ones(zeros));
  printf("test_all_ones(some)   = %d\n", test_all_ones(some));
  printf("test_all_ones(ones)   = %d\n", test_all_ones(ones));

  puts("\n*********** test of logic functions ***********\n");

  Vec<Float, SW> _zero = setzero<Float, SW>();
  Vec<Float, SW> _ones = setones<Float, SW>();
  printf("xor_(zero, zero) = ");
  print("%g ", xor_(_zero, _zero));
  puts("");
  printf("xor_(zero, ones) = ");
  print("%g ", xor_(_zero, _ones));
  puts("");
  printf("xor_(ones, zero) = ");
  print("%g ", xor_(_ones, _zero));
  puts("");
  printf("xor_(ones, ones) = ");
  print("%g ", xor_(_ones, _ones));
  puts("");

  puts("\n*********** test of cvts ************\n");

  Float testVals[3] = {Float(SIMDINT_MIN), 2147483520.0f, Float(SIMDINT_MAX)};

  for (int i = 0; i < 3; i++) {
    for (int off = -200; off <= 200; off += 100) {
      Float v              = testVals[i] + Float(off);
      Vec<Float, SW> ffvec = set1<Float, SW>(v);
      printf("\n%12.0f (%d, %g)\n", v, i, Float(off));
      print("%12.0f ", ffvec);
      puts("");
      print("%12d ", cvts<Int>(ffvec));
      puts("");
    }
  }

  puts("\n*********** test of numeric functions  ************\n");
  NUMERIC;
  ROUNDING(-8388610.f, 0.5f);
  ROUNDING(-8388608.f, 0.5f);
  ROUNDING(-8388606.f, 0.5f);
  ROUNDING(-2.f, 0.25f);
  ROUNDING(-1.f, 0.25f);
  ROUNDING(-0.5f, 0.25f);
  ROUNDING(0.5f, 0.25f);
  ROUNDING(1.f, 0.25f);
  ROUNDING(2.f, 0.25f);
  ROUNDING(8388606.f, 0.5f);
  ROUNDING(8388608.f, 0.5f);
  ROUNDING(8388610.f, 0.5f);

  puts("\n*********** test of set functions  ************\n");
  SETFCT(Byte, "%u ");
  SETFCT(SignedByte, "%d ");
  SETFCT(Word, "%u ");
  SETFCT(Short, "%d ");
  SETFCT(Int, "%d ");
  SETFCT(Float, "%g ");
  SETSIGNEDFCT(SignedByte, "%d ");
  SETSIGNEDFCT(Short, "%d ");
  SETSIGNEDFCT(Int, "%d ");
  SETSIGNEDFCT(Float, "%g ");

  puts("\n********** test of neg **********\n");
  NEG(SignedByte, "%4d ", 0, 5);
  NEG(Short, "%4d ", 0, 5);
  NEG(Int, "%4d ", 0, 5);
  NEG(Float, "%6g ", 0.0, 1.234);

  return 0;
}
