// ===========================================================================
// 
// SIMDVecIntel64.H --
// encapsulation for AVX512 Intel vector extensions
// inspired by Agner Fog's C++ Vector Class Library
// http://www.agner.org/optimize/#vectorclass
// (VCL License: GNU General Public License Version 3,
//  http://www.gnu.org/licenses/gpl-3.0.en.html)
//
// Changes to unpack, zip and unzip functions in 2022 by
//   Jan-Lukas Wolf (jawolf@techfak.uni-bielefeld.de)
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

// TODO:
// [ ] more efficient hadd/hsub implementation

#ifndef _SIMD_VEC_INTEL_64_H_
#define _SIMD_VEC_INTEL_64_H_

// TODO: remove
// #define WHICH_SIMDVECINTEL64 "new"

#ifdef __AVX512F__
// this can be used to check whether 64-byte support is available
#define _SIMD_VEC_64_AVAIL_
#ifdef __AVX512BW__
// this can be used to check whether full 64-byte support is available
#define _SIMD_VEC_64_FULL_AVAIL_
#endif

// determine NATIVE_SIMD_WIDTH
// we are the first to define NATIVE_SIMD_WIDTH, nobody else will do it later
#ifndef NATIVE_SIMD_WIDTH
// the user specified a maximal width
# ifdef MAX_SIMD_WIDTH
// the maximal width is smaller than what would be possible, so we
// take the reduced width (but it is ignored if not 32 or 16)
#  if (MAX_SIMD_WIDTH == 32) || (MAX_SIMD_WIDTH == 16)
#   define NATIVE_SIMD_WIDTH MAX_SIMD_WIDTH
// the maximal width is not smaller than what would be possible, so we
// take what is possible
#  else
#   define NATIVE_SIMD_WIDTH 64
#  endif
// the user didn't specify a maximal width, we take what is possible
# else
#  define NATIVE_SIMD_WIDTH 64
# endif
#endif

#include "SIMDVec.H"
#include "SIMD.H"
#include "SIMDVecIntel32.H"
#include <assert.h>

namespace ns_simd {

  // ===========================================================================
  // NOTES:
  //
  // - setting zero inside the function is not inefficient, see:
  //   http://stackoverflow.com/questions/26807285/...
  //   ...are-static-static-local-sse-avx-variables-blocking-a-xmm-ymm-register
  //
  // - for some data types (SIMDInt, SIMDFloat) there are no saturated versions
  //   of add/sub instructions; in this case we use the unsaturated version;
  //   the user is responsible to avoid overflows
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // x_mm512_hi_si256/_ps, x_mm512_lo_si256/_ps: extract high / low 256-bit v
  // ---------------------------------------------------------------------------

  // these aux. functions are used in the class definition, that's why
  // they already appear here
  
  static SIMD_INLINE __m256i
  x_mm512_hi_si256(__m512i a)
  {
    return _mm512_extracti64x4_epi64(a, 1);
  }
  
  static SIMD_INLINE __m256i
  x_mm512_lo_si256(__m512i a)
  {
    return _mm512_castsi512_si256(a);
  }
  
  static SIMD_INLINE __m512i
  x_mm512_combine_si256(__m256i lo, __m256i hi)
  {
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
  }
  
  // ----------

  static SIMD_INLINE __m256
  x_mm512_hi_ps256(__m512 a)
  {
    return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a), 1));
  }
  
  static SIMD_INLINE __m256
  x_mm512_lo_ps256(__m512 a)
  {
    return _mm512_castps512_ps256(a);
  }
  
  static SIMD_INLINE __m512
  x_mm512_combine_ps(__m256 lo, __m256 hi)
  {
    return _mm512_castpd_ps
      (_mm512_insertf64x4
       (_mm512_castps_pd(_mm512_castps256_ps512(lo)),
	_mm256_castps_pd(hi), 1));
  }
  
  // ===========================================================================
  // SIMDVec integer specialization for AVX512 v
  // ===========================================================================

  // partial specialization for SIMD_WIDTH = 64
  template <typename T>
  class SIMDVec<T, 64>
  {
  public:
    typedef T Type;
    __m512i zmm;
    enum { elements = 64 / sizeof(T), bytes = 64 };
    // shorter version:
    enum { elems = elements };
    SIMDVec() {}
    SIMDVec(const __m512i &x) { zmm = x; }
    SIMDVec& operator=(const __m512i &x) { zmm = x; return *this; }
    operator __m512i() const { return zmm; }
    // for avx512bw emulation
    SIMDVec(const SIMDVec<T,32> &lo, const SIMDVec<T,32> &hi)
    { zmm = x_mm512_combine_si256(lo, hi); }
    SIMD_INLINE SIMDVec<T,32> lo() const 
    { return x_mm512_lo_si256(zmm); }
    SIMD_INLINE SIMDVec<T,32> hi() const 
    { return x_mm512_hi_si256(zmm); }
  };

  // ===========================================================================
  // SIMDVec float specialization for AVX v
  // ===========================================================================

  template <>
  class SIMDVec<SIMDFloat, 64>
  {
  public:
    typedef SIMDFloat Type;
    __m512 zmm;
    enum { elements = 64 / sizeof(SIMDFloat), bytes = 64 };
    // shorter version:
    enum { elems = elements };
    SIMDVec() {}
    SIMDVec(const __m512 &x) { zmm = x; }
    SIMDVec& operator=(const __m512 &x) { zmm = x; return *this; }
    operator __m512() const { return zmm; }
    // for avx512bw emulation
    SIMDVec(const SIMDVec<SIMDFloat,32> &lo, const SIMDVec<SIMDFloat,32> &hi)
    { zmm = x_mm512_combine_ps(lo, hi); }
    SIMD_INLINE SIMDVec<SIMDFloat,32> lo() const 
    { return x_mm512_lo_ps256(zmm); }
    // _mm512_extractf32x8_ps only in AVX512DQ
    SIMD_INLINE SIMDVec<SIMDFloat,32> hi() const
    { return x_mm512_hi_ps256(zmm); }
  };

  // ===========================================================================
  // macros v
  // ===========================================================================
  
  // cast from ps to si512, perform OP on si512 and cast back to ps
#define SIMDVEC_INTEL_PS_SI_PS_64(OP,A,B)				\
  (_mm512_castsi512_ps							\
   (_mm512_ ## OP ## _si512						\
    (_mm512_castps_si512 (A), _mm512_castps_si512 (B))))

  // ===========================================================================
  // auxiliary functions
  // ===========================================================================

  // These functions either wrap  intrinsics (e.g. to handle
  // immediate arguments as template parameter), or switch between
  // implementations with different SSE* extensions, or provide
  // altered or additional functionality.
  // Only for use in wrapper functions!

  // ---------------------------------------------------------------------------
  // some definitions are missing for -O0 in some versions of gcc (e.g. 5.4)
  // ---------------------------------------------------------------------------

  // bug seems to be fixed in avx512bwintrin.h in gcc 5.5.0
  
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define GCC_VERSION (__GNUC__ * 10000 \
                     + __GNUC_MINOR__ * 100 \
                     + __GNUC_PATCHLEVEL__)
# if GCC_VERSION < 50500
#  ifndef __OPTIMIZE__
#   ifdef __AVX512BW__
  
  // _mm512_pack[u]s_epi32 doesn't need a define (no int arguments),
  // but is not available without optimization (error in include file)
  
  extern __inline __m512i
  __attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
  _mm512_packs_epi32 (__m512i __A, __m512i __B)
  {
    return (__m512i) __builtin_ia32_packssdw512_mask ((__v16si) __A,
						      (__v16si) __B,
						      (__v32hi)
						      _mm512_setzero_hi (),
						      (__mmask32) -1);
  }
  
  extern __inline __m512i
  __attribute__ ((__gnu_inline__, __always_inline__, __artificial__))
  _mm512_packus_epi32 (__m512i __A, __m512i __B)
  {
    return (__m512i) __builtin_ia32_packusdw512_mask ((__v16si) __A,
						      (__v16si) __B,
						      (__v32hi)
						      _mm512_setzero_hi (),
						      (__mmask32) -1);
  }
  
#   endif
#  endif
# endif
#endif
  
  // ---------------------------------------------------------------------------
  // permute1_64_16: permutation of 128-bit lanes, single source
  // ---------------------------------------------------------------------------
  
  // template parameter:
  // - Ii (0..3): select lane i from index Ii in a
 
  template <int I0,
	    int I1,
	    int I2,
	    int I3>
  static SIMD_INLINE __m512i
  x_mm512_lane_mask_1()
  {
    return _mm512_set_epi64((2*I3+1),
			    (2*I3),
			    (2*I2+1),
			    (2*I2),
			    (2*I1+1),
			    (2*I1),
			    (2*I0+1),
			    (2*I0));
  }
  
  // all integer versions
  template <int I0,
	    int I1,
	    int I2,
	    int I3,
	    typename T>
  static SIMD_INLINE SIMDVec<T,64>
  permute_64_16_1(const SIMDVec<T,64> &a)
  {
    __m512i m = x_mm512_lane_mask_1<I0,I1,I2,I3>();
    return _mm512_permutexvar_epi64(m, a);
  }

  // float version
  template <int I0,
	    int I1,
	    int I2,
	    int I3>
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  permute_64_16_1(const SIMDVec<SIMDFloat,64> &a)
  {
    __m512i m = x_mm512_lane_mask_1<I0,I1,I2,I3>();
    return _mm512_castpd_ps
      (_mm512_permutexvar_pd(m, _mm512_castps_pd(a)));
  }

  // ---------------------------------------------------------------------------
  // permute_64_16: permutation of 128-bit lanes, two sources v
  // ---------------------------------------------------------------------------

  // template parameter:
  // - ABi (0/1): select lane i from a (0) or from b (1)
  // - Ii (0..3): select lane i from index Ii in either a or b
 
  template <int AB0, int I0,
	    int AB1, int I1,
	    int AB2, int I2,
	    int AB3, int I3>
  static SIMD_INLINE __m512i
  x_mm512_lane_mask()
  {
    return _mm512_set_epi64((AB3<<3)|(2*I3+1),
			    (AB3<<3)|(2*I3),
			    (AB2<<3)|(2*I2+1),
			    (AB2<<3)|(2*I2),
			    (AB1<<3)|(2*I1+1),
			    (AB1<<3)|(2*I1),
			    (AB0<<3)|(2*I0+1),
			    (AB0<<3)|(2*I0));
  }
  
  // all integer versions
  template <int AB0, int I0,
	    int AB1, int I1,
	    int AB2, int I2,
	    int AB3, int I3,
	    typename T>
  static SIMD_INLINE SIMDVec<T,64>
  permute_64_16(const SIMDVec<T,64> &a, 
		const SIMDVec<T,64> &b)

  {
    __m512i m = x_mm512_lane_mask<AB0,I0,AB1,I1,AB2,I2,AB3,I3>();
    return _mm512_permutex2var_epi64(a, m, b);
  }

  // float version
  template <int AB0, int I0,
	    int AB1, int I1,
	    int AB2, int I2,
	    int AB3, int I3>
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  permute_64_16(const SIMDVec<SIMDFloat,64> &a,
		const SIMDVec<SIMDFloat,64> &b)
  {
    __m512i m = x_mm512_lane_mask<AB0,I0,AB1,I1,AB2,I2,AB3,I3>();
    return _mm512_castpd_ps
      (_mm512_permutex2var_pd(_mm512_castps_pd(a), m, _mm512_castps_pd(b)));
  }

  // ---------------------------------------------------------------------------
  // align_64_16 v
  // ---------------------------------------------------------------------------

  // special case
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  align_64_16(const SIMDVec<T,64> &a,
	      const SIMDVec<T,64> &,
	      Integer<0>)
  {
    return a;
  }

  // special case
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  align_64_16(const SIMDVec<T,64> &,
	      const SIMDVec<T,64> &b,
	      Integer<4>)
  {
    return b;
  }

  // general case
  template <int NB, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  align_64_16(const SIMDVec<T,64> &a,
	      const SIMDVec<T,64> &b,
	      Integer<NB>)
  {
    return permute_64_16<
      (NB>3),(NB%4),
      (NB>2),(NB+1)%4,
      (NB>1),(NB+2)%4,
      (NB>0),(NB+3)%4>(a, b);
  }

  // hub
  template <int NB, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  align_64_16(const SIMDVec<T,64> &a,
	      const SIMDVec<T,64> &b)
  {
    // 16-byte lanes:          AB0 I0 AB1 I1 AB2 I2 AB3 I3
    // NB=0: a0 a1 a2 a3         0 0    0 1    0 2    0 3
    // NB=1: a1 a2 a3 b0         0 1    0 2    0 3    1 0
    // NB=2: a2 a3 b0 b1         0 2    0 3    1 0    1 1
    // NB=3: a3 b0 b1 b2         0 3    1 0    1 1    1 2
    // NB=4: b0 b1 b2 b3         1 0    1 1    1 2    1 3
    return align_64_16(a, b, Integer<NB>());
  }
  
  // ---------------------------------------------------------------------------
  // swizzle_64_16: swizzling of 128-bit lanes (for swizzle) v
  // ---------------------------------------------------------------------------

  // each block (e.g. h2) is a 128-bit lane:
  // 
  // example:
  //      
  //      ----v[0]---|----v[1]---
  // n=2: l0 L0 h0 H0 l1 L1 h1 H1
  //      --    --    --    --
  //         --    --    --    --
  //  ->  l0 h0 l1 h1 L0 H0 L1 H1
  //      -----------|-----------
  //
  //
  //      ----v[0]---|----v[1]---|----v[2]---
  // n=3: l0 L0 h0 H0 l1 L1 h1 H1 l2 L2 h2 H2
  //      --       --       --       --
  //         --       --       --       --
  //            --       --       --       --
  //  ->  l0 H0 h1 L2 L0 l1 H1 h2 h0 L1 l2 H2
  //      -----------|-----------|-----------
  //
  //
  //      ----v[0]---|----v[1]---|----v[2]---|----v[3]---
  // n=4: l0 L0 h0 H0 l1 L1 h1 H1 l2 L2 h2 H2 l3 L3 h3 H3
  //      --          --          --          --
  //         --          --          --          --
  //            --          --          --          --
  //               --          --          --          --
  //  ->  l0 l1 l2 l3 L0 L1 L2 L3 h0 h1 h2 h3 H0 H1 H2 H3
  //      -----------|-----------|-----------|-----------
  //
  //
  //      ----v[0]---|----v[1]---|----v[2]---|----v[3]---|----v[4]---
  // n=5: l0 L0 h0 H0 l1 L1 h1 H1 l2 L2 h2 H2 l3 L3 h3 H3 l4 L4 h4 H4
  //      --             --             --             --
  //         --             --             --             --
  //            --             --             --             --
  //               --             --             --             --
  //                  --             --             --             --
  //  ->  l0 L1 h2 H3 L0 h1 H2 l4 h0 H1 l3 L4 H0 l2 L3 h4 l1 L2 h3 H4
  //      -----------|-----------|-----------|-----------|-----------
  
  // primary template
  template <int N, typename T>
  class Swizzle_64_16;

  // N=2
  // vIn:  0 1 2 3 | 4 5 6 7
  // vOut: 0 2 4 6 | 1 3 5 7
  template <typename T>
  class Swizzle_64_16<2,T>
  {
  public:
    static SIMD_INLINE void
    _swizzle_64_16(const SIMDVec<T,64> *const vIn,
		   SIMDVec<T,64> *const vOut)
    {
      vOut[0] = permute_64_16< 0,0, 0,2, 1,0, 1,2 >(vIn[0], vIn[1]);
      vOut[1] = permute_64_16< 0,1, 0,3, 1,1, 1,3 >(vIn[0], vIn[1]);
    }
  };
  
  // N=3
  // vIn:  0 1 2 3 | 4  5 6  7 | 8 9 10 11
  // vTmp: 0 6 1 4 | 7 10 5  8 | 3 9  2 11
  // vOut: 0 3 6 9 | 1  4 7 10 | 2 5  8 11 
  template <typename T>
  class Swizzle_64_16<3,T>
  {
  public:
    static SIMD_INLINE void
    _swizzle_64_16(const SIMDVec<T,64> *const vIn,
		   SIMDVec<T,64> *const vOut)
    {
      SIMDVec<T,64> vTmp[3];
      vTmp[0] = permute_64_16< 0,0, 1,2, 0,1, 1,0 >(vIn[0], vIn[1]);
      vTmp[1] = permute_64_16< 0,3, 1,2, 0,1, 1,0 >(vIn[1], vIn[2]);
      vTmp[2] = permute_64_16< 0,3, 1,1, 0,2, 1,3 >(vIn[0], vIn[2]);

      vOut[0] = permute_64_16< 0,0, 1,0, 0,1, 1,1 >(vTmp[0], vTmp[2]);
      vOut[1] = permute_64_16< 0,2, 0,3, 1,0, 1,1 >(vTmp[0], vTmp[1]);
      vOut[2] = permute_64_16< 1,2, 0,2, 0,3, 1,3 >(vTmp[1], vTmp[2]);
    }
  };

  // N=4
  // vIn:  0 1 2  3 | 4 5 6  7 | 8  9 10 11 | 12 13 14 15
  // vTmp: 0 4 1  5 | 2 6 3  7 | 8 12  9 13 | 10 14 11 15
  // vOut: 0 4 8 12 | 1 5 9 13 | 2  6 10 14 |  3  7 11 15
  template <typename T>
  class Swizzle_64_16<4,T>
  {
  public:
    static SIMD_INLINE void
    _swizzle_64_16(const SIMDVec<T,64> *const vIn,
		   SIMDVec<T,64> *const vOut)
    {
      SIMDVec<T,64> vTmp[4];
      vTmp[0] = permute_64_16< 0,0, 1,0, 0,1, 1,1 >(vIn[0], vIn[1]);
      vTmp[1] = permute_64_16< 0,2, 1,2, 0,3, 1,3 >(vIn[0], vIn[1]);
      vTmp[2] = permute_64_16< 0,0, 1,0, 0,1, 1,1 >(vIn[2], vIn[3]);
      vTmp[3] = permute_64_16< 0,2, 1,2, 0,3, 1,3 >(vIn[2], vIn[3]);
      
      vOut[0] = permute_64_16< 0,0, 0,1, 1,0, 1,1 >(vTmp[0], vTmp[2]);
      vOut[1] = permute_64_16< 0,2, 0,3, 1,2, 1,3 >(vTmp[0], vTmp[2]);
      vOut[2] = permute_64_16< 0,0, 0,1, 1,0, 1,1 >(vTmp[1], vTmp[3]);
      vOut[3] = permute_64_16< 0,2, 0,3, 1,2, 1,3 >(vTmp[1], vTmp[3]);
    }
  };

  // N=5
  // vIn:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15 | 16 17 18 19
  // vTmp: 5 10  6 11 | 1 16  3 18 | 8 13  9 14 |  7 17  4 19 |  0 15  2 12
  // vOut: 0  5 10 15 | 1  6 11 16 | 2  7 12 17 |  3  8 13 18 |  4  9 14 19
  template <typename T>
  class Swizzle_64_16<5,T>
  {
  public:
    static SIMD_INLINE void
    _swizzle_64_16(const SIMDVec<T,64> *const vIn,
		   SIMDVec<T,64> *const vOut)
    {
      SIMDVec<T,64> vTmp[5];
      vTmp[0] = permute_64_16< 0,1, 1,2, 0,2, 1,3 >(vIn[1], vIn[2]);
      vTmp[1] = permute_64_16< 0,1, 1,0, 0,3, 1,2 >(vIn[0], vIn[4]);
      vTmp[2] = permute_64_16< 0,0, 1,1, 0,1, 1,2 >(vIn[2], vIn[3]);
      vTmp[3] = permute_64_16< 0,3, 1,1, 0,0, 1,3 >(vIn[1], vIn[4]);
      vTmp[4] = permute_64_16< 0,0, 1,3, 0,2, 1,0 >(vIn[0], vIn[3]);
      
      vOut[0] = permute_64_16< 1,0, 0,0, 0,1, 1,1 >(vTmp[0], vTmp[4]);
      vOut[1] = permute_64_16< 1,0, 0,2, 0,3, 1,1 >(vTmp[0], vTmp[1]);
      vOut[2] = permute_64_16< 1,2, 0,0, 1,3, 0,1 >(vTmp[3], vTmp[4]);
      vOut[3] = permute_64_16< 0,2, 1,0, 1,1, 0,3 >(vTmp[1], vTmp[2]);
      vOut[4] = permute_64_16< 1,2, 0,2, 0,3, 1,3 >(vTmp[2], vTmp[3]);
    }
  };

  // swizzle lanes (for implementation of swizzle functions)
  template <int N, typename T>
  static SIMD_INLINE void
  swizzle_64_16(const SIMDVec<T,64> *const vIn,
		SIMDVec<T,64> *const vOut)
  {
    Swizzle_64_16<N,T>::_swizzle_64_16(vIn, vOut);
  }

  // ---------------------------------------------------------------------------
  // element-wise shifts v
  // ---------------------------------------------------------------------------
  
#ifdef __AVX512BW__

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_slli_epi16(__m512i a, IsPosInRange<true, true>)
  {
    return _mm512_slli_epi16(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_slli_epi16(__m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_slli_epi16(__m512i a)
  {
    return x_mm512_slli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // ----------
  
  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srli_epi16(__m512i a, IsPosInRange<true, true>)
  {
    return _mm512_srli_epi16(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srli_epi16(__m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srli_epi16(__m512i a)
  {
    return x_mm512_srli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // ----------
  
  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srai_epi16(__m512i a, IsPosInRange<true, true>)
  {
    return _mm512_srai_epi16(a, IMM);
  }

  // positive and out of range: maximal shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srai_epi16(__m512i a, IsPosInRange<true, false>)
  {
    return _mm512_srai_epi16(a, 15);
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srai_epi16(__m512i a)
  {
    return x_mm512_srai_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

#endif

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_slli_epi32(__m512i a, IsPosInRange<true, true>)
  {
    return _mm512_slli_epi32(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_slli_epi32(__m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_slli_epi32(__m512i a)
  {
    return x_mm512_slli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // ----------
  
  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srli_epi32(__m512i a, IsPosInRange<true, true>)
  {
    return _mm512_srli_epi32(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srli_epi32(__m512i, IsPosInRange<true, false>)
  {
    return _mm512_setzero_si512();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srli_epi32(__m512i a)
  {
    return x_mm512_srli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // ----------
  
  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srai_epi32(__m512i a, IsPosInRange<true, true>)
  {
    return _mm512_srai_epi32(a, IMM);
  }

  // positive and out of range: maximal shift
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srai_epi32(__m512i a, IsPosInRange<true, false>)
  {
    return _mm512_srai_epi32(a, 31);
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_srai_epi32(__m512i a)
  {
    return x_mm512_srai_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // ---------------------------------------------------------------------------
  // extract 16 byte portions (IMM = 0..3), used for x_mm512_extract_epi/_ps v
  // ---------------------------------------------------------------------------

  template <int IMM>
  static SIMD_INLINE __m128
  x_mm512_extractf32x4_ps(__m512 a)
  {
    return _mm512_extractf32x4_ps(a, IMM);
  }

  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm512_extracti32x4_epi32(__m512i a)
  {
    return _mm512_extracti32x4_epi32(a, IMM);
  }
 
  // ---------------------------------------------------------------------------
  // extract 128-bit-lane as SIMDVec<T, 16>
  // ---------------------------------------------------------------------------

  // generalized extract of 128-bit-lanes (IMM = 0..3)
  // all integer versions
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  extractLane(const SIMDVec<T, 64> &a)
  {
    return x_mm512_extracti32x4_epi32<IMM>(a);
  }

  // generalized extract of 128-bit-lanes (IMM = 0..3)
  // float versions
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDFloat, 16>
  extractLane(const SIMDVec<SIMDFloat, 64> &a)
  {
    return x_mm512_extractf32x4_ps<IMM>(a);
  }

  // ---------------------------------------------------------------------------
  // extract v
  // ---------------------------------------------------------------------------
  
  // _mm512_extract_epi/_ps etc. don't exist, we emulate that
  // NOTE: even in AVX, that was internally done in the extract intrinsics:
  // e.g. avxintrin.h:
  // int _mm256_extract_epi16 (__m256i __X, int const __N)
  // {
  //   __m128i __Y = _mm256_extractf128_si256 (__X, __N >> 3);
  //   return _mm_extract_epi16 (__Y, __N % 8);
  // }

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi16(__m512i a, IsPosInRange<true, true>)
  {
    return x_mm_extract_epi16<(IMM % 8)>
      (x_mm512_extracti32x4_epi32<(IMM >> 3)>(a));
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi16(__m512i, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi16(__m512i a)
  {
    // 32 * epi16
    return x_mm512_extract_epi16<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // ----------

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi8(__m512i a, IsPosInRange<true, true>)
  {
    return _mm_extract_epi8
      (x_mm512_extracti32x4_epi32<(IMM >> 4)>(a),
       IMM % 16);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi8(__m512i, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi8(__m512i a)
  {
    // 64 * epi8
    return x_mm512_extract_epi8<IMM>(a, IsPosInGivenRange<64, IMM>());
  }

  // ----------

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi32(__m512i a, IsPosInRange<true, true>)
  {
    // TODO: extract: is conversion from return type int to SIMDInt always safe?
    return _mm_extract_epi32
      (x_mm512_extracti32x4_epi32<(IMM >> 2)>(a),
       IMM % 4);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi32(__m512i, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_epi32(__m512i a)
  {
    // 16 * epi32
    return x_mm512_extract_epi32<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // ----------

  // we stick to the weird definition of returning an int instead of float
  
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_ps(__m512 a, IsPosInRange<true, true>)
  {
    // TODO: extract: is conversion from return type int to SIMDInt always safe?
    return _mm_extract_ps
      (x_mm512_extractf32x4_ps<(IMM >> 2)>(a),
       IMM % 4);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_ps(__m512, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm512_extract_ps(__m512 a)
  {
    // 16 * epi32
    return x_mm512_extract_ps<IMM>(a, IsPosInGivenRange<16, IMM>());
  }
  
  // ---------------------------------------------------------------------------
  // alignr v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  // positive, zero (not non-zero), and in range: return l
  // (this case was introduced for swizzle functions)
  template <int IMM>
  static SIMD_INLINE __m512i 
  x_mm512_alignr_epi8(__m512i, __m512i l, 
		      IsPosNonZeroInRange<true, false, true>)
  {
    return l;
  }
  
  // positive, non-zero, and in range: run align
  template <int IMM>
  static SIMD_INLINE __m512i 
  x_mm512_alignr_epi8(__m512i h, __m512i l, 
		      IsPosNonZeroInRange<true, true, true>)
  {
    return _mm512_alignr_epi8(h, l, IMM);	
  }

  // positive, non-zero, and out of range: return zero vector
  template <int IMM>
  static SIMD_INLINE __m512i 
  x_mm512_alignr_epi8(__m512i, __m512i, 
		      IsPosNonZeroInRange<true, true, false>)
  {
    return _mm512_setzero_si512();	
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m512i 
  x_mm512_alignr_epi8(__m512i h, __m512i l)
  {
    // IMM < 32
    return x_mm512_alignr_epi8<IMM>(h, l, 
				    IsPosNonZeroInGivenRange<32, IMM>());	
  }

#else

  // non-avx512bw workarounds
  // (easy since AVX512BW instructions operate on lanes anyhow)

  // IMM range handling is done at SSE level
  
  template <int IMM>
  static SIMD_INLINE __m512i
  x_mm512_alignr_epi8(__m512i h, __m512i l)
  {
    return x_mm512_combine_si256
      (x_mm256_alignr_epi8<IMM>(x_mm512_lo_si256(h), x_mm512_lo_si256(l)),
       x_mm256_alignr_epi8<IMM>(x_mm512_hi_si256(h), x_mm512_hi_si256(l)));
  }
  
#endif
  
  // ---------------------------------------------------------------------------
  // insert 16 byte vector a all 4 lanes of a 64 byte vector v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE __m512i
  x_mm512_duplicate_si128(__m128i a)
  {
    return _mm512_broadcast_i32x4(a);
  }

  // ---------------------------------------------------------------------------
  // workarounds for logical functions on float vecs (AVX512DQ only) v
  // ---------------------------------------------------------------------------

#ifdef __AVX512DQ__
#define SIMD_X_LOGICPS_64(OP)			\
  static SIMD_INLINE __m512			\
  x_mm512_ ## OP ## _ps(__m512 a, __m512 b)	\
  {						\
    return _mm512_ ## OP ## _ps(a, b);		\
  }
#else
#define SIMD_X_LOGICPS_64(OP)			\
  static SIMD_INLINE __m512			\
  x_mm512_ ## OP ## _ps(__m512 a, __m512 b)	\
  {						\
    return SIMDVEC_INTEL_PS_SI_PS_64(OP,a,b);	\
  }
#endif

  SIMD_X_LOGICPS_64(and)
  SIMD_X_LOGICPS_64(or)
  SIMD_X_LOGICPS_64(xor)
  SIMD_X_LOGICPS_64(andnot)
  
  // ---------------------------------------------------------------------------
  // not (missing from instruction set) v
  // ---------------------------------------------------------------------------

  // from Agner Fog's VCL vectori256.h operator ~
  static SIMD_INLINE
  __m512i x_mm512_not_si512(__m512i a)
  {
    return _mm512_xor_si512(a, _mm512_set1_epi32(-1));
  }

  static SIMD_INLINE
  __m512 x_mm512_not_ps(__m512 a)
  {
    return _mm512_castsi512_ps
      (_mm512_xor_si512(_mm512_castps_si512(a), _mm512_set1_epi32(-1)));
  }

  // ---------------------------------------------------------------------------
  // abs
  // ---------------------------------------------------------------------------

  // strange, Intel Instrinsics Guide says it exists, but it doesn't
  static SIMD_INLINE __m512
  x_mm512_abs_ps(__m512 a)
  {
    // there's no _mm_abs_ps, we have to emulated it:
    // -0.0F is 0x8000000, 0x7fffffff by andnot, sign bit is cleared
    return x_mm512_andnot_ps(_mm512_set1_ps(-0.0F), a);
  }

  // ---------------------------------------------------------------------------
  // transpose8x64 v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE __m512i
  x_mm512_transpose8x64_epi64(__m512i a)
  {
    return _mm512_permutexvar_epi64(_mm512_set_epi64(7,3,6,2,5,1,4,0), a);
  }

  static SIMD_INLINE __m512
  x_mm512_transpose8x64_ps(__m512 a)
  {
    return 
      _mm512_castsi512_ps
      (x_mm512_transpose8x64_epi64(_mm512_castps_si512(a)));
  }

  // ---------------------------------------------------------------------------
  // evenodd8x64 v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE __m512i
  x_mm512_evenodd8x64_epi64(__m512i a)
  {
    return _mm512_permutexvar_epi64(_mm512_set_epi64(7,5,3,1,6,4,2,0), a);
  }
  
  static SIMD_INLINE __m512
  x_mm512_evenodd8x64_ps(__m512 a)
  {
    return 
      _mm512_castsi512_ps
      (x_mm512_evenodd8x64_epi64(_mm512_castps_si512(a)));
  }

  // ---------------------------------------------------------------------------
  // unpack of 2 ps v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE __m512
  x_mm512_unpacklo_2ps(__m512 a, __m512 b)
  {
    return _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a),
					       _mm512_castps_pd(b)));
  }

  static SIMD_INLINE __m512
  x_mm512_unpackhi_2ps(__m512 a, __m512 b)
  {
    return _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(a),
					       _mm512_castps_pd(b)));
  }

  // ---------------------------------------------------------------------------
  // binary functions with non-avx512bw workarounds v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__
  // avx512bw is available
#define SIMD_X_BW_INT_BINFCT_64(INTRIN)	    \
  static SIMD_INLINE __m512i		    \
  x_mm512_ ## INTRIN (__m512i a, __m512i b) \
  {					    \
    return _mm512_ ## INTRIN (a, b);	    \
  }
#else
  // non-avx512bw workaround
#define SIMD_X_BW_INT_BINFCT_64(INTRIN)					\
  static SIMD_INLINE __m512i						\
  x_mm512_ ## INTRIN (__m512i a, __m512i b)				\
  {									\
    return x_mm512_combine_si256					\
      (_mm256_ ## INTRIN (x_mm512_lo_si256(a), x_mm512_lo_si256(b)),	\
       _mm256_ ## INTRIN (x_mm512_hi_si256(a), x_mm512_hi_si256(b)));	\
  }									
#endif
  
  SIMD_X_BW_INT_BINFCT_64(unpacklo_epi8)
  SIMD_X_BW_INT_BINFCT_64(unpackhi_epi8)
  SIMD_X_BW_INT_BINFCT_64(unpacklo_epi16)
  SIMD_X_BW_INT_BINFCT_64(unpackhi_epi16)
  SIMD_X_BW_INT_BINFCT_64(shuffle_epi8)
  SIMD_X_BW_INT_BINFCT_64(packs_epi16)
  SIMD_X_BW_INT_BINFCT_64(packs_epi32)
  SIMD_X_BW_INT_BINFCT_64(packus_epi16)
  SIMD_X_BW_INT_BINFCT_64(packus_epi32)

  // ---------------------------------------------------------------------------
  // unary functions with non-avx512bw workarounds v
  // ---------------------------------------------------------------------------

  // UP: __m256i -> __m512i
  
#ifdef __AVX512BW__
  // avx512bw is available
#define SIMD_X_BW_INT_UNFCT_UP_64(INTRIN)   \
  static SIMD_INLINE __m512i		    \
  x_mm512_ ## INTRIN (__m256i a)	    \
  {					    \
    return _mm512_ ## INTRIN (a);	    \
  }
#else
  // non-avx512bw workaround
#define SIMD_X_BW_INT_UNFCT_UP_64(INTRIN)	\
  static SIMD_INLINE __m512i			\
  x_mm512_ ## INTRIN (__m256i a)		\
  {						\
    return x_mm512_combine_si256		\
      (_mm256_ ## INTRIN (x_mm256_lo128i(a)),	\
       _mm256_ ## INTRIN (x_mm256_hi128i(a)));	\
  }									
#endif
  
  SIMD_X_BW_INT_UNFCT_UP_64(cvtepi8_epi16)
  SIMD_X_BW_INT_UNFCT_UP_64(cvtepu8_epi16)
  
  // ---------------------------------------------------------------------------
  // non-existing avx512 functions emulated via avx v
  // ---------------------------------------------------------------------------

#define SIMD_X_AVX_INT_BINFCT_64(INTRIN)				\
  static SIMD_INLINE __m512i						\
  x_mm512_ ## INTRIN (__m512i a, __m512i b)				\
  {									\
    return x_mm512_combine_si256					\
      (_mm256_ ## INTRIN (x_mm512_lo_si256(a), x_mm512_lo_si256(b)),	\
       _mm256_ ## INTRIN (x_mm512_hi_si256(a), x_mm512_hi_si256(b)));	\
  }									

#define SIMD_X_AVX_PS_BINFCT_64(INTRIN)					\
  static SIMD_INLINE __m512						\
  x_mm512_ ## INTRIN (__m512 a, __m512 b)				\
  {									\
    return x_mm512_combine_ps						\
      (_mm256_ ## INTRIN (x_mm512_lo_ps256(a), x_mm512_lo_ps256(b)),	\
       _mm256_ ## INTRIN (x_mm512_hi_ps256(a), x_mm512_hi_ps256(b)));	\
  }									

  SIMD_X_AVX_INT_BINFCT_64(hadd_epi16)
  SIMD_X_AVX_INT_BINFCT_64(hadd_epi32)
  SIMD_X_AVX_PS_BINFCT_64(hadd_ps)
  SIMD_X_AVX_INT_BINFCT_64(hadds_epi16)
  SIMD_X_AVX_INT_BINFCT_64(hsub_epi16)
  SIMD_X_AVX_INT_BINFCT_64(hsub_epi32)
  SIMD_X_AVX_PS_BINFCT_64(hsub_ps)
  SIMD_X_AVX_INT_BINFCT_64(hsubs_epi16)

  // ---------------------------------------------------------------------------
  // x_mm512_movm_* v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__
  // no non-avx512 workaround needed, only used below if avx512bw is available
  
  static SIMD_INLINE __m512i
  x_mm512_movm_epi8(__mmask64 k)
  {
    return _mm512_movm_epi8(k);
  }

  static SIMD_INLINE __m512i
  x_mm512_movm_epi16(__mmask32 k)
  {
    return _mm512_movm_epi16(k);
  }
#endif

#ifdef __AVX512DQ__
  static SIMD_INLINE __m512i
  x_mm512_movm_epi32(__mmask16 k)
  {
    return _mm512_movm_epi32(k);
  }
#else
# ifdef __AVX512BW__
  static SIMD_INLINE __m512i
  x_mm512_movm_epi32(__mmask16 k)
  {
    return _mm512_maskz_mov_epi32(k, _mm512_set1_epi32(-1));
  }
# else
  static SIMD_INLINE __m512i
  x_mm512_movm_epi32(__mmask16 k)
  {
    return _mm512_castps_si512
      (_mm512_maskz_mov_ps
       (k, _mm512_castsi512_ps(_mm512_set1_epi32(-1))));
  }
# endif
#endif

  // ---------------------------------------------------------------------------
  // necessary to avoid bugs when compiling with -O0
  // ---------------------------------------------------------------------------

  // avx512bw uses defines when -O0 is passed; these seem to have
  // difficulties when SIMDVec<> is passed, so we have this
  // intermediate layer
  static SIMD_INLINE __m512i
  x_mm512_mask_blend_epi8(__mmask64 k, __m512i a, __m512i b)
  {
    return _mm512_mask_blend_epi8(k, a, b);
  }

  // ###########################################################################
  // ###########################################################################
  // ###########################################################################

  // ===========================================================================
  // SIMDVec template function specializations or overloading for AVX
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // reinterpretation casts v
  // ---------------------------------------------------------------------------

  // between all integer types
  template <typename Tdst, typename Tsrc>
  static SIMD_INLINE SIMDVec<Tdst,64>
  reinterpret(const SIMDVec<Tsrc,64>& vec, OutputType<Tdst>)
  {
    return reinterpret_cast<const SIMDVec<Tdst,64>&>(vec);
  }

  // from float to any integer type
  template <typename Tdst>
  static SIMD_INLINE SIMDVec<Tdst,64>
  reinterpret(const SIMDVec<SIMDFloat,64>& vec, OutputType<Tdst>)
  {
    return _mm512_castps_si512(vec);
  }

  // from any integer type to float
  template <typename Tsrc>
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  reinterpret(const SIMDVec<Tsrc,64>& vec, OutputType<SIMDFloat>)
  {
    return _mm512_castsi512_ps(vec);
  }

  // between float and float
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  reinterpret(const SIMDVec<SIMDFloat,64>& vec, OutputType<SIMDFloat>)
  {
    return vec;
  }

  // ---------------------------------------------------------------------------
  // convert (without changes in the number of of elements) v
  // ---------------------------------------------------------------------------

  // conversion with saturation; we wanted to have a fast solution that
  // doesn't trigger the overflow which results in a negative two's
  // complement result ("invalid int32": 0x80000000); therefore we clamp
  // the positive values at the maximal positive float which is
  // convertible to int32 without overflow (0x7fffffbf = 2147483520);
  // negative values cannot overflow (they are clamped to invalid int
  // which is the most negative int32)
  template <>
  SIMD_INLINE SIMDVec<SIMDInt,64> 
  cvts(const SIMDVec<SIMDFloat,64> &a)
  {
    // TODO: analyze much more complex solution for cvts at
    // TODO: http://stackoverflow.com/questions/9157373/
    // TODO: most-efficient-way-to-convert-vector-of-float-to-vector-of-uint32
    __m512 clip = _mm512_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32); 
    return _mm512_cvtps_epi32(_mm512_min_ps(clip, a));
  }

  // saturation is not necessary in this case
  template <>
  SIMD_INLINE SIMDVec<SIMDFloat,64> 
  cvts(const SIMDVec<SIMDInt,64> &a)
  {
    return _mm512_cvtepi32_ps(a);
  }

  // ---------------------------------------------------------------------------
  // setzero v
  // ---------------------------------------------------------------------------

  template <>
  SIMD_INLINE SIMDVec<SIMDByte,64> 
  setzero()
  { 
    return _mm512_setzero_si512(); 
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  setzero()
  {
    return _mm512_setzero_si512();
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDWord,64> 
  setzero()
  {
    return _mm512_setzero_si512();
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDShort,64> 
  setzero()
  {
    return _mm512_setzero_si512();
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDInt,64> 
  setzero()
  {
    return _mm512_setzero_si512();
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDFloat,64>
  setzero()
  {
    return _mm512_setzero_ps();
  }

  // ---------------------------------------------------------------------------
  // set1 v
  // ---------------------------------------------------------------------------

  template <>
  SIMD_INLINE SIMDVec<SIMDByte,64> 
  set1(SIMDByte a)
  {
    return _mm512_set1_epi8(a);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  set1(SIMDSignedByte a)
  {
    return _mm512_set1_epi8(a);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDWord,64> 
  set1(SIMDWord a)
  {
    return _mm512_set1_epi16(a);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDShort,64> 
  set1(SIMDShort a)
  {
    return _mm512_set1_epi16(a);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDInt,64> 
  set1(SIMDInt a)
  {
    return _mm512_set1_epi32(a);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDFloat,64>
  set1(SIMDFloat a)
  {
    return _mm512_set1_ps(a);
  }

  // ---------------------------------------------------------------------------
  // load v
  // ---------------------------------------------------------------------------

  template <>
  SIMD_INLINE SIMDVec<SIMDByte,64> 
  load(const SIMDByte *const p)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    return _mm512_load_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  load(const SIMDSignedByte *const p)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    return _mm512_load_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDWord,64> 
  load(const SIMDWord *const p)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    return _mm512_load_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDShort,64> 
  load(const SIMDShort *const p)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    return _mm512_load_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDInt,64> 
  load(const SIMDInt *const p)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    return _mm512_load_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDFloat,64>
  load(const SIMDFloat *const p)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    return _mm512_load_ps(p);
  }

  // ---------------------------------------------------------------------------
  // loadu v
  // ---------------------------------------------------------------------------

  template <>
  SIMD_INLINE SIMDVec<SIMDByte,64> 
  loadu(const SIMDByte *const p)
  {
    return _mm512_loadu_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  loadu(const SIMDSignedByte *const p)
  {
    return _mm512_loadu_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDWord,64> 
  loadu(const SIMDWord *const p)
  {
    return _mm512_loadu_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDShort,64> 
  loadu(const SIMDShort *const p)
  {
    return _mm512_loadu_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDInt,64> 
  loadu(const SIMDInt *const p)
  {
    return _mm512_loadu_si512((__m512i*) p);
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDFloat,64>
  loadu(const SIMDFloat *const p)
  {
    return _mm512_loadu_ps(p);
  }

  // ---------------------------------------------------------------------------
  // store v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE void 
  store(T *const p, const SIMDVec<T,64> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    _mm512_store_si512((__m512i*) p, a);
  }

  // float version
  static SIMD_INLINE void
  store(SIMDFloat *const p, const SIMDVec<SIMDFloat,64> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    _mm512_store_ps(p, a);
  }

  // ---------------------------------------------------------------------------
  // storeu v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE void 
  storeu(T *const p, const SIMDVec<T,64> &a)
  {
    _mm512_storeu_si512((__m512i*) p, a);
  }

  // float version
  static SIMD_INLINE void
  storeu(SIMDFloat *const p, const SIMDVec<SIMDFloat,64> &a)
  {
    _mm512_storeu_ps(p, a);
  }

  // ---------------------------------------------------------------------------
  // stream_store v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE void 
  stream_store(T *const p, const SIMDVec<T,64> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    _mm512_stream_si512((__m512i*) p, a);
  }

  // float version
  static SIMD_INLINE void
  stream_store(SIMDFloat *const p, const SIMDVec<SIMDFloat,64> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // AVX load and store instructions need alignment to 64 byte
    // (lower 6 bit need to be zero)
    assert((((uintptr_t) p) & 0x3f) == 0);
#endif
    _mm512_stream_ps(p, a);
  }

  // ---------------------------------------------------------------------------
  // extract v
  // ---------------------------------------------------------------------------
  
  template <int IMM>
  static SIMD_INLINE SIMDByte
  extract(const SIMDVec<SIMDByte,64> &a)
  {
    return x_mm512_extract_epi8<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDSignedByte
  extract(const SIMDVec<SIMDSignedByte,64> &a)
  {
    return x_mm512_extract_epi8<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDWord
  extract(const SIMDVec<SIMDWord,64> &a)
  {
    return x_mm512_extract_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDShort
  extract(const SIMDVec<SIMDShort,64> &a)
  {
    return x_mm512_extract_epi16<IMM>(a);
  }
	
  template <int IMM>
  static SIMD_INLINE SIMDInt
  extract(const SIMDVec<SIMDInt,64> &a)
  {
    // TODO: extract: is conversion from return type int to SIMDInt always safe?
    return x_mm512_extract_epi32<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDFloat
  extract(const SIMDVec<SIMDFloat,64> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDFloat,x_mm512_extract_ps<IMM>(a),0);
  }

  // ---------------------------------------------------------------------------
  // add v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64> 
  add(const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_add_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  add(const SIMDVec<SIMDSignedByte,64> &a,
      const SIMDVec<SIMDSignedByte,64> &b)
  {
    return _mm512_add_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  add(const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_add_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  add(const SIMDVec<SIMDShort,64> &a,
      const SIMDVec<SIMDShort,64> &b)
  {
    return _mm512_add_epi16(a, b);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  add(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(add(a.lo(), b.lo()),
			 add(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64> 
  add(const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    return _mm512_add_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  add(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_add_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // adds (integer: signed, unsigned; 8/16 only, 32 without saturation) v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64> 
  adds(const SIMDVec<SIMDByte,64> &a,
       const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_adds_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  adds(const SIMDVec<SIMDSignedByte,64> &a,
       const SIMDVec<SIMDSignedByte,64> &b)
  {
    return _mm512_adds_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  adds(const SIMDVec<SIMDWord,64> &a,
       const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_adds_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  adds(const SIMDVec<SIMDShort,64> &a,
       const SIMDVec<SIMDShort,64> &b)
  {
    return _mm512_adds_epi16(a, b);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  adds(const SIMDVec<T,64> &a,
       const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(adds(a.lo(), b.lo()),
			 adds(a.hi(), b.hi()));
  }
  
#endif

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  adds(const SIMDVec<SIMDInt,64> &a,
       const SIMDVec<SIMDInt,64> &b)
  {
    return _mm512_add_epi32(a, b);
  }
#endif

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  adds(const SIMDVec<SIMDFloat,64> &a,
       const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_add_ps(a, b);
  }
#endif

  // ---------------------------------------------------------------------------
  // sub v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64> 
  sub(const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_sub_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  sub(const SIMDVec<SIMDSignedByte,64> &a,
      const SIMDVec<SIMDSignedByte,64> &b)
  {
    return _mm512_sub_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  sub(const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_sub_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  sub(const SIMDVec<SIMDShort,64> &a,
      const SIMDVec<SIMDShort,64> &b)
  {
    return _mm512_sub_epi16(a, b);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  sub(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(sub(a.lo(), b.lo()),
			 sub(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64> 
  sub(const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    return _mm512_sub_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  sub(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_sub_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // subs (integer: signed, unsigned; 8/16 only, 32 without saturation) v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64> 
  subs(const SIMDVec<SIMDByte,64> &a,
       const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_subs_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  subs(const SIMDVec<SIMDSignedByte,64> &a,
       const SIMDVec<SIMDSignedByte,64> &b)
  {
    return _mm512_subs_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  subs(const SIMDVec<SIMDWord,64> &a,
       const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_subs_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  subs(const SIMDVec<SIMDShort,64> &a,
       const SIMDVec<SIMDShort,64> &b)
  {
    return _mm512_subs_epi16(a, b);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  subs(const SIMDVec<T,64> &a,
       const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(subs(a.lo(), b.lo()),
			 subs(a.hi(), b.hi()));
  }
  
#endif

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  subs(const SIMDVec<SIMDInt,64> &a,
       const SIMDVec<SIMDInt,64> &b)
  {
    return _mm512_sub_epi32(a, b);
  }
#endif

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  subs(const SIMDVec<SIMDFloat,64> &a,
       const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_sub_ps(a, b);
  }
#endif

  // ---------------------------------------------------------------------------
  // neg (negate = two's complement or unary minus), only signed types v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  neg(const SIMDVec<SIMDSignedByte,64> &a)
  {
    return _mm512_sub_epi8(_mm512_setzero_si512(), a);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  neg(const SIMDVec<SIMDShort,64> &a)
  {
    return _mm512_sub_epi16(_mm512_setzero_si512(), a);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  neg(const SIMDVec<T,64> &a)
  {
    return SIMDVec<T,64>(neg(a.lo()),
			 neg(a.hi()));
  }

#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  neg(const SIMDVec<SIMDInt,64> &a)
  {
    return _mm512_sub_epi32(_mm512_setzero_si512(), a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  neg(const SIMDVec<SIMDFloat,64> &a)
  {
    return _mm512_sub_ps(_mm512_setzero_ps(), a);
  }

  // ---------------------------------------------------------------------------
  // min v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64> 
  min(const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_min_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  min(const SIMDVec<SIMDSignedByte,64> &a,
      const SIMDVec<SIMDSignedByte,64> &b)
  {
    return _mm512_min_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  min(const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_min_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  min(const SIMDVec<SIMDShort,64> &a,
      const SIMDVec<SIMDShort,64> &b)
  {
    return _mm512_min_epi16(a, b);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  min(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(min(a.lo(), b.lo()),
			 min(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64> 
  min(const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    return _mm512_min_epi32(a, b);
  }

  // there is an unsigned version of min for 32 bit but we currently
  // don't have an element type for it

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  min(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_min_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // max v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64> 
  max(const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_max_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  max(const SIMDVec<SIMDSignedByte,64> &a,
      const SIMDVec<SIMDSignedByte,64> &b)
  {
    return _mm512_max_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  max(const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_max_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  max(const SIMDVec<SIMDShort,64> &a,
      const SIMDVec<SIMDShort,64> &b)
  {
    return _mm512_max_epi16(a, b);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  max(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(max(a.lo(), b.lo()),
			 max(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64> 
  max(const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    return _mm512_max_epi32(a, b);
  }

  // there is an unsigned version of max for 32 bit but we currently
  // don't have an element type for it

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  max(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_max_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // mul, div v
  // ---------------------------------------------------------------------------

  // TODO: add mul/div versions for int types? or make special versions of mul
  // TODO: and div where the result is scaled?

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  mul(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_mul_ps(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  div(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_div_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // ceil, floor, round, truncate v
  // ---------------------------------------------------------------------------

  // TODO: add versions of ceil/floor/round for int types?

  // see Peter Cordes at https://stackoverflow.com/questions/50854991
  // _mm512_roundscale_ps:
  // imm[7:4] = fraction bits = here 0, imm[0:1] = rounding mode
  
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  ceil(const SIMDVec<SIMDFloat,64> &a)
  {
    return _mm512_roundscale_ps(a, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  floor(const SIMDVec<SIMDFloat,64> &a)
  {
    return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  round(const SIMDVec<SIMDFloat,64> &a)
  {
    return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  truncate(const SIMDVec<SIMDFloat,64> &a)
  {
    return _mm512_roundscale_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  }

  // ---------------------------------------------------------------------------
  // elementary mathematical functions v
  // ---------------------------------------------------------------------------

  // estimate of a reciprocal
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  rcp(const SIMDVec<SIMDFloat,64> &a)
  {
    // NOTE: this has better precision than SSE and AVX versions!
    return _mm512_rcp14_ps(a);
  }

  // estimate of reverse square root
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  rsqrt(const SIMDVec<SIMDFloat,64> &a)
  {
    // NOTE: this has better precision than SSE and AVX versions!
    return _mm512_rsqrt14_ps(a);
  }

  // square root
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  sqrt(const SIMDVec<SIMDFloat,64> &a)
  {
    return _mm512_sqrt_ps(a);
  }

  // ---------------------------------------------------------------------------
  // abs (integer: signed only) v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  abs(const SIMDVec<SIMDSignedByte,64> &a)
  {
    return _mm512_abs_epi8(a);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  abs(const SIMDVec<SIMDShort,64> &a)
  {
    return _mm512_abs_epi16(a);
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  abs(const SIMDVec<T,64> &a)
  {
    return SIMDVec<T,64>(abs(a.lo()),
			 abs(a.hi()));
  }

#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64> 
  abs(const SIMDVec<SIMDInt,64> &a)
  {
    return _mm512_abs_epi32(a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  abs(const SIMDVec<SIMDFloat,64> &a)
  {
    return x_mm512_abs_ps(a);
  }

  // ---------------------------------------------------------------------------
  // unpacklo v (with permutex2var)
  // ---------------------------------------------------------------------------

#ifdef __AVX512VBMI__

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<1>)
  {
    // element order high to low for idx
    // 95,31,94,30,93,29,92,28,
    // 91,27,90,26,89,25,88,24,
    // 87,23,86,22,85,21,84,20,
    // 83,19,82,18,81,17,80,16,
    // 79,15,78,14,77,13,76,12,
    // 75,11,74,10,73, 9,72, 8,
    // 71, 7,70, 6,69, 5,68, 4,
    // 67, 3,66, 2,65, 1,64, 0
    __m512i idx = _mm512_set_epi32(0x5f1f5e1e, 0x5d1d5c1c,
                                   0x5b1b5a1a, 0x59195818,
                                   0x57175616, 0x55155414,
                                   0x53135212, 0x51115010,
                                   0x4f0f4e0e, 0x4d0d4c0c,
                                   0x4b0b4a0a, 0x49094808,
                                   0x47074606, 0x45054404,
                                   0x43034202, 0x41014000);
    return _mm512_permutex2var_epi8(a, idx, b);
  }

#else

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<1>)
  {
    return x_mm512_unpacklo_epi8(x_mm512_transpose8x64_epi64(a),
                                 x_mm512_transpose8x64_epi64(b));
  }

#endif

#ifdef __AVX512BW__

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<2>)
  {
    // element order high to low for idx
    // 47,15,46,14,45,13,44,12,
    // 43,11,42,10,41, 9,40, 8,
    // 39, 7,38, 6,37, 5,36, 4,
    // 35, 3,34, 2,33, 1,32, 0
    __m512i idx = _mm512_set_epi32(
                    0x002f000f, 0x002e000e, 0x002d000d, 0x002c000c,
                    0x002b000b, 0x002a000a, 0x00290009, 0x00280008,
                    0x00270007, 0x00260006, 0x00250005, 0x00240004,
                    0x00230003, 0x00220002, 0x00210001, 0x00200000);
    return _mm512_permutex2var_epi16(a, idx, b);
  }

#else

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<2>)
  {
    return x_mm512_unpacklo_epi16(x_mm512_transpose8x64_epi64(a),
                                  x_mm512_transpose8x64_epi64(b));
  }

#endif

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<4>)
  {
    __m512i idx = _mm512_set_epi32(23,7,22,6,21,5,20,4,
                                   19,3,18,2,17,1,16,0);
    return _mm512_permutex2var_epi32(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<8>)
  {
    __m512i idx = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(11,10,3,2,9,8,1,0);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<32>)
  {
    __m512i idx = _mm512_set_epi64(11,10,9,8,3,2,1,0);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<4>)
  {
    __m512i idx = _mm512_set_epi32(23,7,22,6,21,5,20,4,
                                   19,3,18,2,17,1,16,0);
    return _mm512_permutex2var_ps(a, idx, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<8>)
  {
    __m512i idx = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(11,10,3,2,9,8,1,0);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<32>)
  {
    __m512i idx = _mm512_set_epi64(11,10,9,8,3,2,1,0);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // ---------------------------------------------------------------------------
  // unpackhi v (with permutex2var)
  // ---------------------------------------------------------------------------

#ifdef __AVX512VBMI__

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<1>)
  {
    // element order high to low for idx
    // 127,63,126,62,125,61,124,60,
    // 123,59,122,58,121,57,120,56,
    // 119,55,118,54,117,53,116,52,
    // 115,51,114,50,113,49,112,48,
    // 111,47,110,46,109,45,108,44,
    // 107,43,106,42,105,41,104,40,
    // 103,39,102,38,101,37,100,36,
    //  99,35, 98,34, 97,33, 96,32
    __m512i idx = _mm512_set_epi32(0x7f3f7e3e, 0x7d3d7c3c,
                                   0x7b3b7a3a, 0x79397838,
                                   0x77377636, 0x75357434,
                                   0x73337232, 0x71317030,
                                   0x6f2f6e2e, 0x6d2d6c2c,
                                   0x6b2b6a2a, 0x69296828,
                                   0x67276626, 0x65256424,
                                   0x63236222, 0x61216020);
    return _mm512_permutex2var_epi8(a, idx, b);
  }

#else

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<1>)
  {
    return x_mm512_unpackhi_epi8(x_mm512_transpose8x64_epi64(a),
                                 x_mm512_transpose8x64_epi64(b));
  }

#endif

#ifdef __AVX512BW__

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<2>)
  {
    // element order high to low for idx
    // 63,31,62,30,61,29,60,28,
    // 59,27,58,26,57,25,56,24,
    // 55,23,54,22,53,21,52,20,
    // 51,19,50,18,49,17,48,16
    __m512i idx = _mm512_set_epi32(
                    0x003f001f, 0x003e001e, 0x003d001d, 0x003c001c,
                    0x003b001b, 0x003a001a, 0x00390019, 0x00380018,
                    0x00370017, 0x00360016, 0x00350015, 0x00340014,
                    0x00330013, 0x00320012, 0x00310011, 0x00300010);
    return _mm512_permutex2var_epi16(a, idx, b);
  }

#else

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<2>)
  {
    return x_mm512_unpackhi_epi16(x_mm512_transpose8x64_epi64(a),
                                  x_mm512_transpose8x64_epi64(b));
  }

#endif

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<4>)
  {
    __m512i idx = _mm512_set_epi32(31,15,30,14,29,13,28,12,
                                   27,11,26,10,25,9,24,8);
    return _mm512_permutex2var_epi32(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
         const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<8>)
  {
    __m512i idx = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(15,14,7,6,13,12,5,4);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<32>)
  {
    __m512i idx = _mm512_set_epi64(15,14,13,12,7,6,5,4);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<4>)
  {
    __m512i idx = _mm512_set_epi32(31,15,30,14,29,13,28,12,
                                   27,11,26,10,25,9,24,8);
    return _mm512_permutex2var_ps(a, idx, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<8>)
  {
    __m512i idx = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(15,14,7,6,13,12,5,4);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<32>)
  {
    __m512i idx = _mm512_set_epi64(15,14,13,12,7,6,5,4);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // ---------------------------------------------------------------------------
  // unpack hub v
  // ---------------------------------------------------------------------------

  // generalized unpack
  // unpack blocks of NUM_ELEMS elements of type T 
  // PART=0: low half of input vectors,
  // PART=1: high half of input vectors
  template <int PART, int NUM_ELEMS, typename T>
  static SIMD_INLINE SIMDVec<T, 64> 
  unpack(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b)
  {
    return unpack(a, b, Part<PART>(), Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // ---------------------------------------------------------------------------
  // 128-bit-lane oriented unpacklo (with direct intrinsic calls)
  // ---------------------------------------------------------------------------

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<1>)
  {
    return x_mm512_unpacklo_epi8(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<2>)
  {
    return x_mm512_unpacklo_epi16(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<4>)
  {
    return _mm512_unpacklo_epi32(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<8>)
  {
    return _mm512_unpacklo_epi64(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(13,12,5,4,9,8,1,0);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<0>,
	 Bytes<32>)
  {
    return _mm512_shuffle_i32x4(a, b, _MM_SHUFFLE(1,0,1,0));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<4>)
  {
    return _mm512_unpacklo_ps(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<8>)
  {
    return _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(a),
                                               _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(13,12,5,4,9,8,1,0);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<0>,
	 Bytes<32>)
  {
    return _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(1,0,1,0));
  }

  // ---------------------------------------------------------------------------
  // 128-bit-lane oriented unpackhi v
  // ---------------------------------------------------------------------------

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<1>)
  {
    return x_mm512_unpackhi_epi8(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<2>)
  {
    return x_mm512_unpackhi_epi16(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<4>)
  {
    return _mm512_unpackhi_epi32(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
   const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<8>)
  {
    return _mm512_unpackhi_epi64(a, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(15,14,7,6,11,10,3,2);
    return _mm512_permutex2var_epi64(a, idx, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b,
	 Part<1>,
	 Bytes<32>)
  {
    return _mm512_shuffle_i32x4(a, b, _MM_SHUFFLE(3,2,3,2));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<4>)
  {
    return _mm512_unpackhi_ps(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<8>)
  {
    return _mm512_castpd_ps(
             _mm512_unpackhi_pd(_mm512_castps_pd(a),
                                _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<16>)
  {
    __m512i idx = _mm512_set_epi64(15,14,7,6,11,10,3,2);
    return _mm512_castpd_ps(
             _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                    idx,
                                    _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat, 64>
  unpack16(const SIMDVec<SIMDFloat, 64> &a,
	 const SIMDVec<SIMDFloat, 64> &b,
	 Part<1>,
	 Bytes<32>)
  {
    return _mm512_shuffle_f32x4(a, b, _MM_SHUFFLE(3,2,3,2));
  }

  // ---------------------------------------------------------------------------
  // 128-bit-lane oriented unpack hub
  // ---------------------------------------------------------------------------

  // generalized 128-bit-lane oriented unpack
  // unpack blocks of NUM_ELEMS elements of type T
  // PART=0: low half of 128-bit lanes of input vectors,
  // PART=1: high half of 128-bit lanes of input vectors
  template <int PART, int NUM_ELEMS, typename T>
  static SIMD_INLINE SIMDVec<T, 64>
  unpack16(const SIMDVec<T, 64> &a,
	 const SIMDVec<T, 64> &b)
  {
    return unpack16(a, b, Part<PART>(), Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // ---------------------------------------------------------------------------
  // zip v
  // ---------------------------------------------------------------------------

  // a, b are passed by-value to avoid problems with identical in/out args.

  // here we typically have to transpose the inputs in the same way
  // for both output computations, so we define separate functions for
  // all T and Bytes<> (combinations of unpack functions above)

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h,
      Bytes<1>)
  {
    l = unpack(a, b, Part<0>(), Bytes<1>());
    h = unpack(a, b, Part<1>(), Bytes<1>());
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h,
      Bytes<2>)
  {
    l = unpack(a, b, Part<0>(), Bytes<2>());
    h = unpack(a, b, Part<1>(), Bytes<2>());
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h,
      Bytes<4>)
  {
    l = unpack(a, b, Part<0>(), Bytes<4>());
    h = unpack(a, b, Part<1>(), Bytes<4>());
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h,
      Bytes<8>)
  {
    l = unpack(a, b, Part<0>(), Bytes<8>());
    h = unpack(a, b, Part<1>(), Bytes<8>());
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h,
      Bytes<16>)
  {
    l = unpack(a, b, Part<0>(), Bytes<16>());
    h = unpack(a, b, Part<1>(), Bytes<16>());
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h,
      Bytes<32>)
  {
    l = unpack(a, b, Part<0>(), Bytes<32>());
    h = unpack(a, b, Part<1>(), Bytes<32>());
  }

  // float version
  static SIMD_INLINE void
  zip(const SIMDVec<SIMDFloat, 64> a,
      const SIMDVec<SIMDFloat, 64> b,
      SIMDVec<SIMDFloat, 64> &l,
      SIMDVec<SIMDFloat, 64> &h,
      Bytes<4>)
  {
    l = unpack(a, b, Part<0>(), Bytes<4>());
    h = unpack(a, b, Part<1>(), Bytes<4>());
  }

  // float version
  static SIMD_INLINE void
  zip(const SIMDVec<SIMDFloat, 64> a,
      const SIMDVec<SIMDFloat, 64> b,
      SIMDVec<SIMDFloat, 64> &l,
      SIMDVec<SIMDFloat, 64> &h,
      Bytes<8>)
  {
    l = unpack(a, b, Part<0>(), Bytes<8>());
    h = unpack(a, b, Part<1>(), Bytes<8>());
  }

  // float version
  static SIMD_INLINE void
  zip(const SIMDVec<SIMDFloat, 64> a,
      const SIMDVec<SIMDFloat, 64> b,
      SIMDVec<SIMDFloat, 64> &l,
      SIMDVec<SIMDFloat, 64> &h,
      Bytes<16>)
  {
    l = unpack(a, b, Part<0>(), Bytes<16>());
    h = unpack(a, b, Part<1>(), Bytes<16>());
  }

  // float version
  static SIMD_INLINE void
  zip(const SIMDVec<SIMDFloat, 64> a,
      const SIMDVec<SIMDFloat, 64> b,
      SIMDVec<SIMDFloat, 64> &l,
      SIMDVec<SIMDFloat, 64> &h,
      Bytes<32>)
  {
    l = unpack(a, b, Part<0>(), Bytes<32>());
    h = unpack(a, b, Part<1>(), Bytes<32>());
  }

  // ---------------------------------------------------------------------------
  // zip hub v
  // ---------------------------------------------------------------------------

  // zips blocks of NUM_ELEMS elements of type T 
  template <int NUM_ELEMS, typename T>
  static SIMD_INLINE void
  zip(const SIMDVec<T, 64> a,
      const SIMDVec<T, 64> b,
      SIMDVec<T, 64> &l,
      SIMDVec<T, 64> &h)
  {
    return zip(a, b, l, h, Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // ---------------------------------------------------------------------------
  // zip16 (16-byte-lane oriented zip)
  // ---------------------------------------------------------------------------

  // zips blocks of NUM_ELEMS elements of type T
  template <int NUM_ELEMS, typename T>
  static SIMD_INLINE void
  zip16(const SIMDVec<T, 64> a,
        const SIMDVec<T, 64> b,
        SIMDVec<T, 64> &l,
        SIMDVec<T, 64> &h)
  {
    l = unpack16(a, b, Part<0>(), Bytes<NUM_ELEMS * sizeof(T)>());
    h = unpack16(a, b, Part<1>(), Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // ---------------------------------------------------------------------------
  // unzip v
  // ---------------------------------------------------------------------------

  // a, b are passed by-value to avoid problems with identical
  // input/output args.

  // here we typically have to transpose the inputs in the same way
  // for both output computations, so we define separate functions for
  // all T and Bytes<> (combinations of unpack functions above)

  static SIMD_INLINE __m512i
  x_mm512_even64_mask()
  {
    return _mm512_set_epi64
      // b6   b4   b2   b0  a6 a4 a2 a0
      ( 8|6, 8|4, 8|2, 8|0,  6, 4, 2, 0);    
  }

  static SIMD_INLINE __m512i
  x_mm512_odd64_mask()
  {
    return _mm512_set_epi64
      // b7   b5   b3   b1  a7 a5 a3 a1
      ( 8|7, 8|5, 8|3, 8|1,  7, 5, 3, 1);
  }

  // _mm512_set_epi8 is missing from some gcc versions, so we use epi32
  // https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg188664.html

#ifdef __AVX512VBMI__

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h,
	Bytes<1>)
  {
    // element order for low idx
    // 126,124,122,120,118,116,114,112,
    // 110,108,106,104,102,100, 98, 96,
    //  94, 92, 90, 88, 86, 84, 82, 80,
    //  78, 76, 74, 72, 70, 68, 66, 64,
    //  62, 60, 58, 56, 54, 52, 50, 48,
    //  46, 44, 42, 40, 38, 36, 34, 32,
    //  30, 28, 26, 24, 22, 20, 18, 16,
    //  14, 12, 10,  8,  6,  4,  2,  0 
    __m512i idxL = _mm512_set_epi32(0x7e7c7a78, 0x76747270,
                                    0x6e6c6a68, 0x66646260,
                                    0x5e5c5a58, 0x56545250,
                                    0x4e4c4a48, 0x46444240,
                                    0x3e3c3a38, 0x36343230,
                                    0x2e2c2a28, 0x26242220,
                                    0x1e1c1a18, 0x16141210,
                                    0x0e0c0a08, 0x06040200);
    // element order for high idx
    // 127,125,123,121,119,117,115,113,
    // 111,109,107,105,103,101, 99, 97,
    //  95, 93, 91, 89, 87, 85, 83, 81,
    //  79, 77, 75, 73, 71, 69, 67, 65,
    //  63, 61, 59, 57, 55, 53, 51, 49,
    //  47, 45, 43, 41, 39, 37, 35, 33,
    //  31, 29, 27, 25, 23, 21, 19, 17,
    //  15, 13, 11,  9,  7,  5,  3,  1 
    __m512i idxH = _mm512_set_epi32(0x7f7d7b79, 0x77757371,
                                    0x6f6d6b69, 0x67656361,
                                    0x5f5d5b59, 0x57555351,
                                    0x4f4d4b49, 0x47454341,
                                    0x3f3d3b39, 0x37353331,
                                    0x2f2d2b29, 0x27252321,
                                    0x1f1d1b19, 0x17151311,
                                    0x0f0d0b09, 0x07050301);
    l = _mm512_permutex2var_epi8(a, idxL, b);
    h = _mm512_permutex2var_epi8(a, idxH, b);
  }

#else

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h,
	Bytes<1>)
  {
    // mask is hopefully only set once if unzip is used multiple times
    /*
    __m512i mask = _mm512_set_epi8 (15,13,11,9,7,5,3,1,
				    14,12,10,8,6,4,2,0,
				    15,13,11,9,7,5,3,1,
				    14,12,10,8,6,4,2,0,
				    15,13,11,9,7,5,3,1,
				    14,12,10,8,6,4,2,0,
				    15,13,11,9,7,5,3,1,
				    14,12,10,8,6,4,2,0);
    */
    __m512i mask = _mm512_set_epi32(0x0f0d0b09, 0x07050301,
				    0x0e0c0a08, 0x06040200,
				    0x0f0d0b09, 0x07050301,
				    0x0e0c0a08, 0x06040200,
				    0x0f0d0b09, 0x07050301,
				    0x0e0c0a08, 0x06040200,
				    0x0f0d0b09, 0x07050301,
				    0x0e0c0a08, 0x06040200);
    __m512i atmp = x_mm512_shuffle_epi8(a, mask);
    __m512i btmp = x_mm512_shuffle_epi8(b, mask);
    l = _mm512_permutex2var_epi64(atmp, x_mm512_even64_mask(), btmp);
    h = _mm512_permutex2var_epi64(atmp, x_mm512_odd64_mask(),  btmp);
  }

#endif

#ifdef __AVX512BW__

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h,
	Bytes<2>)
  {
    // element order for low idx
    // 62,60,58,56,54,52,50,48,
    // 46,44,42,40,38,36,34,32,
    // 30,28,26,24,22,20,18,16,
    // 14,12,10, 8, 6, 4, 2, 0
    __m512i idxL = _mm512_set_epi32(
                     0x003e003c, 0x003a0038, 0x00360034, 0x00320030,
                     0x002e002c, 0x002a0028, 0x00260024, 0x00220020,
                     0x001e001c, 0x001a0018, 0x00160014, 0x00120010,
                     0x000e000c, 0x000a0008, 0x00060004, 0x00020000);
    // element order for high idx
    // 63,61,59,57,55,53,51,49,
    // 47,45,43,41,39,37,35,33,
    // 31,29,27,25,23,21,19,17,
    // 15,13,11, 9, 7, 5, 3, 1
    __m512i idxH = _mm512_set_epi32(
                     0x003f003d, 0x003b0039, 0x00370035, 0x00330031,
                     0x002f002d, 0x002b0029, 0x00270025, 0x00230021,
                     0x001f001d, 0x001b0019, 0x00170015, 0x00130011,
                     0x000f000d, 0x000b0009, 0x00070005, 0x00030001);
    l = _mm512_permutex2var_epi16(a, idxL, b);
    h = _mm512_permutex2var_epi16(a, idxH, b);
  }

#else

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h,
	Bytes<2>)
  {
    // mask is hopefully only set once if unzip is used multiple times
    /*
    __m512i mask = _mm512_set_epi8(15,14,11,10,7,6,3,2,
				   13,12,9,8,5,4,1,0,
				   15,14,11,10,7,6,3,2,
				   13,12,9,8,5,4,1,0,
				   15,14,11,10,7,6,3,2,
				   13,12,9,8,5,4,1,0,
				   15,14,11,10,7,6,3,2,
				   13,12,9,8,5,4,1,0);
    */
    __m512i mask = _mm512_set_epi32(0x0f0e0b0a, 0x07060302,
				    0x0d0c0908, 0x05040100,
				    0x0f0e0b0a, 0x07060302,
				    0x0d0c0908, 0x05040100,
				    0x0f0e0b0a, 0x07060302,
				    0x0d0c0908, 0x05040100,
				    0x0f0e0b0a, 0x07060302,
				    0x0d0c0908, 0x05040100);
    __m512i atmp = x_mm512_shuffle_epi8(a, mask);
    __m512i btmp = x_mm512_shuffle_epi8(b, mask);
    l = _mm512_permutex2var_epi64(atmp, x_mm512_even64_mask(), btmp);
    h = _mm512_permutex2var_epi64(atmp, x_mm512_odd64_mask(),  btmp);
  }

#endif

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h,
	Bytes<4>)
  {
    __m512i idxL = _mm512_set_epi32(30,28,26,24,22,20,18,16,
                                    14,12,10,8,6,4,2,0);
    __m512i idxH = _mm512_set_epi32(31,29,27,25,23,21,19,17,
                                    15,13,11,9,7,5,3,1);
    l = _mm512_permutex2var_epi32(a, idxL, b);
    h = _mm512_permutex2var_epi32(a, idxH, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
        const SIMDVec<T, 64> b,
        SIMDVec<T, 64> &l,
        SIMDVec<T, 64> &h,
        Bytes<8>)
  {
    __m512i idxL = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    __m512i idxH = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    l = _mm512_permutex2var_epi64(a, idxL, b);
    h = _mm512_permutex2var_epi64(a, idxH, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void 
  unzip(const SIMDVec<T, 64> a,
        const SIMDVec<T, 64> b,
        SIMDVec<T, 64> &l,
        SIMDVec<T, 64> &h,
        Bytes<16>)
  {
    // lanes:                       b2     b0   a2   a0
    __m512i idxL = _mm512_set_epi64(13,12, 9,8, 5,4, 1,0);
    // lanes:                       b3     b1     a3   a1
    __m512i idxH = _mm512_set_epi64(15,14, 11,10, 7,6, 3,2);
    l = _mm512_permutex2var_epi64(a, idxL, b);
    h = _mm512_permutex2var_epi64(a, idxH, b);
  }

  // integer version
  template <typename T>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h,
	Bytes<32>)
  {
    l = unpack(a, b, Part<0>(), Bytes<32>());
    h = unpack(a, b, Part<1>(), Bytes<32>());
  }

  // float version
  static SIMD_INLINE void
  unzip(const SIMDVec<SIMDFloat, 64> a,
	const SIMDVec<SIMDFloat, 64> b,
	SIMDVec<SIMDFloat, 64> &l,
	SIMDVec<SIMDFloat, 64> &h,
	Bytes<4>)
  {
    __m512i idxL = _mm512_set_epi32(30,28,26,24,22,20,18,16,
                                    14,12,10,8,6,4,2,0);
    __m512i idxH = _mm512_set_epi32(31,29,27,25,23,21,19,17,
                                    15,13,11,9,7,5,3,1);
    l = _mm512_permutex2var_ps(a, idxL, b);
    h = _mm512_permutex2var_ps(a, idxH, b);
  }

  // float version
  static SIMD_INLINE void
  unzip(const SIMDVec<SIMDFloat, 64> a,
	const SIMDVec<SIMDFloat, 64> b,
	SIMDVec<SIMDFloat, 64> &l,
	SIMDVec<SIMDFloat, 64> &h,
	Bytes<8>)
  {
    __m512i idxL = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    __m512i idxH = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    l = _mm512_castpd_ps(
          _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                 idxL,
                                 _mm512_castps_pd(b)));
    h = _mm512_castpd_ps(
          _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                 idxH,
                                 _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE void
  unzip(const SIMDVec<SIMDFloat, 64> a,
	const SIMDVec<SIMDFloat, 64> b,
	SIMDVec<SIMDFloat, 64> &l,
	SIMDVec<SIMDFloat, 64> &h,
	Bytes<16>)
  {
    // lanes:                       b2     b0   a2   a0
    __m512i idxL = _mm512_set_epi64(13,12, 9,8, 5,4, 1,0);
    // lanes:                       b3     b1     a3   a1
    __m512i idxH = _mm512_set_epi64(15,14, 11,10, 7,6, 3,2);
    l = _mm512_castpd_ps(
          _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                 idxL,
                                 _mm512_castps_pd(b)));
    h = _mm512_castpd_ps(
          _mm512_permutex2var_pd(_mm512_castps_pd(a),
                                 idxH,
                                 _mm512_castps_pd(b)));
  }

  // float version
  static SIMD_INLINE void
  unzip(const SIMDVec<SIMDFloat, 64> a,
	const SIMDVec<SIMDFloat, 64> b,
	SIMDVec<SIMDFloat, 64> &l,
	SIMDVec<SIMDFloat, 64> &h,
	Bytes<32>)
  {
    l = unpack(a, b, Part<0>(), Bytes<32>());
    h = unpack(a, b, Part<1>(), Bytes<32>());
  }

  // ---------------------------------------------------------------------------
  // unzip hub v
  // ---------------------------------------------------------------------------

  // hub
  template <int NUM_ELEMS, typename T>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, 64> a,
	const SIMDVec<T, 64> b,
	SIMDVec<T, 64> &l,
	SIMDVec<T, 64> &h)
  {
    return unzip(a, b, l, h, Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // ---------------------------------------------------------------------------
  // packs v
  // ---------------------------------------------------------------------------
  
  // ========== signed -> signed ==========
  
  template <>
  SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  packs(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_evenodd8x64_epi64
      (x_mm512_packs_epi16(a, b));
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDShort,64> 
  packs(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_evenodd8x64_epi64
      (x_mm512_packs_epi32(a, b));
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDShort,64>
  packs(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return packs<SIMDShort>(cvts<SIMDInt>(a), cvts<SIMDInt>(b));
  }

  // ========== signed -> unsigned ==========

  // non-avx512bw workaround
  template <>
  SIMD_INLINE SIMDVec<SIMDByte,64> 
  packs(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_evenodd8x64_epi64
      (x_mm512_packus_epi16(a, b));
  }

  // non-avx512bw workaround
  template <>
  SIMD_INLINE SIMDVec<SIMDWord,64> 
  packs(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_evenodd8x64_epi64
      (x_mm512_packus_epi32(a, b));
  }

  template <>
  SIMD_INLINE SIMDVec<SIMDWord,64>
  packs(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return packs<SIMDWord>(cvts<SIMDInt>(a), cvts<SIMDInt>(b));
  }

  // ---------------------------------------------------------------------------
  // generalized extend: no stage v
  // ---------------------------------------------------------------------------

  // from\to
  //    SB B S W I F 
  // SB  x   x   x x
  //  B    x x x x x
  //  S      x   x x
  //  W        x x x
  //  I          x x
  //  F          x x
  //
  // combinations: 
  // - signed   -> extended signed (sign extension)
  // - unsigned -> extended unsigned (zero extension)
  // - unsigned -> extended signed (zero extension)
  // (signed -> extended unsigned is not possible)

  // all types
  template <typename T>
  static SIMD_INLINE void
  extend(const SIMDVec<T,64> &vIn,
	 SIMDVec<T,64> *const vOut)
  {
    *vOut = vIn;
  }

  // ---------------------------------------------------------------------------
  // generalized extend: single stage v
  // ---------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,64> &vIn,
	 SIMDVec<SIMDShort,64> *const vOut)
  {
    vOut[0] = x_mm512_cvtepi8_epi16(x_mm512_lo_si256(vIn));
    vOut[1] = x_mm512_cvtepi8_epi16(x_mm512_hi_si256(vIn));
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDShort,64> &vIn,
	 SIMDVec<SIMDInt,64> *const vOut)
  {
    vOut[0] = _mm512_cvtepi16_epi32(x_mm512_lo_si256(vIn));
    vOut[1] = _mm512_cvtepi16_epi32(x_mm512_hi_si256(vIn));
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDShort,64> &vIn,
	 SIMDVec<SIMDFloat,64> *const vOut)
  {
    vOut[0] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(x_mm512_lo_si256(vIn)));
    vOut[1] = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(x_mm512_hi_si256(vIn)));
  }

  // unsigned -> unsigned

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,64> &vIn,
	 SIMDVec<SIMDWord,64> *const vOut)
  {
    // there's no _mm512_cvtepu8_epu16()
    SIMDVec<SIMDByte,64> zero = setzero<SIMDByte,64>();
    // 16. Jul 16 (rm): here we avoid to use generalized unpack from
    // SIMDVecExt.H
    vOut[0] = unpack(vIn, zero, Part<0>(), Bytes<1>());
    vOut[1] = unpack(vIn, zero, Part<1>(), Bytes<1>());
  }

  // unsigned -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,64> &vIn,
	 SIMDVec<SIMDShort,64> *const vOut)
  {
    vOut[0] = x_mm512_cvtepu8_epi16(x_mm512_lo_si256(vIn));
    vOut[1] = x_mm512_cvtepu8_epi16(x_mm512_hi_si256(vIn));
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDWord,64> &vIn,
	 SIMDVec<SIMDInt,64> *const vOut)
  {
    vOut[0] = _mm512_cvtepu16_epi32(x_mm512_lo_si256(vIn));
    vOut[1] = _mm512_cvtepu16_epi32(x_mm512_hi_si256(vIn));
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDWord,64> &vIn,
	 SIMDVec<SIMDFloat,64> *const vOut)
  {
    vOut[0] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(x_mm512_lo_si256(vIn)));
    vOut[1] = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(x_mm512_hi_si256(vIn)));
  }

  // ---------------------------------------------------------------------------
  // generalized extend: two stages v
  // ---------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,64> &vIn,
	 SIMDVec<SIMDInt,64> *const vOut)
  {
    __m256i vInLo256 = x_mm512_lo_si256(vIn);
    vOut[0] = _mm512_cvtepi8_epi32(x_mm256_lo128i(vInLo256));
    vOut[1] = _mm512_cvtepi8_epi32(x_mm256_hi128i(vInLo256));
    __m256i vInHi256 = x_mm512_hi_si256(vIn);
    vOut[2] = _mm512_cvtepi8_epi32(x_mm256_lo128i(vInHi256));
    vOut[3] = _mm512_cvtepi8_epi32(x_mm256_hi128i(vInHi256));
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,64> &vIn,
	 SIMDVec<SIMDFloat,64> *const vOut)
  {
    SIMDVec<SIMDInt,64> vTmp[4];
    extend(vIn, vTmp);
    for (int i = 0; i < 4; i++) vOut[i] = cvts<SIMDFloat>(vTmp[i]);
  }

  // unsigned -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,64> &vIn,
	 SIMDVec<SIMDInt,64> *const vOut)
  {
    __m256i vInLo256 = x_mm512_lo_si256(vIn);
    vOut[0] = _mm512_cvtepu8_epi32(x_mm256_lo128i(vInLo256));
    vOut[1] = _mm512_cvtepu8_epi32(x_mm256_hi128i(vInLo256));
    __m256i vInHi256 = x_mm512_hi_si256(vIn);
    vOut[2] = _mm512_cvtepu8_epi32(x_mm256_lo128i(vInHi256));
    vOut[3] = _mm512_cvtepu8_epi32(x_mm256_hi128i(vInHi256));
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,64> &vIn,
	 SIMDVec<SIMDFloat,64> *const vOut)
  {
    SIMDVec<SIMDInt,64> vTmp[4];
    extend(vIn, vTmp);
    for (int i = 0; i < 4; i++) vOut[i] = cvts<SIMDFloat>(vTmp[i]);
  }

  // ---------------------------------------------------------------------------
  // generalized extend: special case int <-> float v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDInt,64> &vIn,
	 SIMDVec<SIMDFloat,64> *const vOut)
  {
    *vOut = cvts<SIMDFloat>(vIn);
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDFloat,64> &vIn,
	 SIMDVec<SIMDInt,64> *const vOut)
  {
    *vOut = cvts<SIMDInt>(vIn);
  }

  // ---------------------------------------------------------------------------
  // srai (16/32 only) v
  // ---------------------------------------------------------------------------

  // TODO: srai: emulation of missing 8-bit versions?
  // TODO: (Hacker's Delight 2nd ed. sec. 2-7?)

#ifdef __AVX512BW__

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  srai(const SIMDVec<SIMDWord,64> &a)
  {
    return x_mm512_srai_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  srai(const SIMDVec<SIMDShort,64> &a)
  {
    return x_mm512_srai_epi16<IMM>(a);
  }

#else

  // non-avx512bw workaround
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  srai(const SIMDVec<T,64> &a)
  {
    return SIMDVec<T,64>(srai<IMM>(a.lo()),
			 srai<IMM>(a.hi()));
  }
  
#endif

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  srai(const SIMDVec<SIMDInt,64> &a)
  {
    return x_mm512_srai_epi32<IMM>(a);
  }

  // ---------------------------------------------------------------------------
  // srli v
  // ---------------------------------------------------------------------------

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDByte,64>
  srli(const SIMDVec<SIMDByte,64> &a)
  {
    return _mm512_and_si512(_mm512_set1_epi8((int8_t)(0xff >> IMM)),
			    x_mm512_srli_epi32<IMM>(a));
  }

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  srli(const SIMDVec<SIMDSignedByte,64> &a)
  {
    return _mm512_and_si512(_mm512_set1_epi8((int8_t)(0xff >> IMM)),
			    x_mm512_srli_epi32<IMM>(a));
  }

#ifdef __AVX512BW__

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  srli(const SIMDVec<SIMDWord,64> &a)
  {
    return x_mm512_srli_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  srli(const SIMDVec<SIMDShort,64> &a)
  {
    return x_mm512_srli_epi16<IMM>(a);
  }

#else

  // non-avx512bw workaround
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  srli(const SIMDVec<T,64> &a)
  {
    return SIMDVec<T,64>(srli<IMM>(a.lo()),
			 srli<IMM>(a.hi()));
  }
  
#endif

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  srli(const SIMDVec<SIMDInt,64> &a)
  {
    return x_mm512_srli_epi32<IMM>(a);
  }

  // ---------------------------------------------------------------------------
  // slli v
  // ---------------------------------------------------------------------------

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDByte,64>
  slli(const SIMDVec<SIMDByte,64> &a)
  {
    return _mm512_and_si512
      (_mm512_set1_epi8((int8_t)(uint8_t)(0xff & (0xff << IMM))),
       x_mm512_slli_epi32<IMM>(a));
  }

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,64> 
  slli(const SIMDVec<SIMDSignedByte,64> &a)
  {
    return _mm512_and_si512
      (_mm512_set1_epi8((int8_t)(uint8_t)(0xff & (0xff << IMM))),
       x_mm512_slli_epi32<IMM>(a));
  }

#ifdef __AVX512BW__

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDWord,64> 
  slli(const SIMDVec<SIMDWord,64> &a)
  {
    return x_mm512_slli_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDShort,64> 
  slli(const SIMDVec<SIMDShort,64> &a)
  {
    return x_mm512_slli_epi16<IMM>(a);
  }

#else

  // non-avx512bw workaround
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  slli(const SIMDVec<T,64> &a)
  {
    return SIMDVec<T,64>(slli<IMM>(a.lo()),
			 slli<IMM>(a.hi()));
  }
  
#endif

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  slli(const SIMDVec<SIMDInt,64> &a)
  {
    return x_mm512_slli_epi32<IMM>(a);
  }


  // 05. Aug 22 (Jonas Keller):
  // Improved implementation of hadd, hadds, hsub and hsubs,
  // implementation does not use emulation via AVX anymore.
  // SIMDByte and SIMDSignedByte are now supported as well.
  // The new implementation is faster for SIMDInt and SIMDFloat, but
  // slower for SIMDWord and SIMDShort for some reason.

  // ---------------------------------------------------------------------------
  // hadd v
  // ---------------------------------------------------------------------------

  template<typename T>
  static SIMD_INLINE SIMDVec<T,64>
  hadd(const SIMDVec<T,64> &a,
       const SIMDVec<T,64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return add(x, y);
  }

  // ---------------------------------------------------------------------------
  // hadds v
  // ---------------------------------------------------------------------------

  template<typename T>
  static SIMD_INLINE SIMDVec<T,64>
  hadds(const SIMDVec<T,64> &a,
	      const SIMDVec<T,64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return adds(x, y);
  }

  // ---------------------------------------------------------------------------
  // hsub v
  // ---------------------------------------------------------------------------

  template<typename T>
  static SIMD_INLINE SIMDVec<T,64>
  hsub(const SIMDVec<T,64> &a,
       const SIMDVec<T,64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return sub(x, y);
  }
  
  // ---------------------------------------------------------------------------
  // hsubs v
  // ---------------------------------------------------------------------------
  
  template<typename T>
  static SIMD_INLINE SIMDVec<T,64>
  hsubs(const SIMDVec<T,64> &a,
        const SIMDVec<T,64> &b)
  {
    SIMDVec<T, 64> x, y;
    unzip<1>(a, b, x, y);
    return subs(x, y);
  }

  // ---------------------------------------------------------------------------
  // extraction of element 0 v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDByte
  elem0(const SIMDVec<SIMDByte, 64> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDByte,
				_mm_cvtsi128_si32(_mm512_castsi512_si128(a)),
				0);
  }

  static SIMD_INLINE SIMDSignedByte
  elem0(const SIMDVec<SIMDSignedByte, 64> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDSignedByte,
				_mm_cvtsi128_si32(_mm512_castsi512_si128(a)),
				0);
  }

  static SIMD_INLINE SIMDWord
  elem0(const SIMDVec<SIMDWord, 64> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDWord,
				_mm_cvtsi128_si32(_mm512_castsi512_si128(a)),
				0);
  }

  static SIMD_INLINE SIMDShort
  elem0(const SIMDVec<SIMDShort, 64> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDShort,
				_mm_cvtsi128_si32(_mm512_castsi512_si128(a)),
				0);
  }

  static SIMD_INLINE SIMDInt
  elem0(const SIMDVec<SIMDInt, 64> &a)
  {
    // TODO: elem0: is conversion from return type int so SIMDInt always safe?
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
  }

  static SIMD_INLINE SIMDFloat
  elem0(const SIMDVec<SIMDFloat, 64> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDFloat,
				_mm_cvtsi128_si32
				(_mm512_castsi512_si128
				 (_mm512_castps_si512(a))),
				0);
  }

  // ---------------------------------------------------------------------------
  // alignre v
  // ---------------------------------------------------------------------------

  // Li, Hi: lanes
  // n = IMM * sizeof(T) [#bytes]
  // 
  // input: H0 H1 H2 H3
  //        L0 L1 L2 L3         NB
  // ==================         
  // n<16:  L1 L2 L3 H0   L,H   1
  //        L0 L1 L2 L3   L,H   0
  // ------------------
  // n<32:  L2 L3 H0 H1   L,H   2
  //        L1 L2 L3 H0   L,H   1
  // ------------------
  // n<48:  L3 H0 H1 H2   L,H   3
  //        L2 L3 H0 H1   L,H   2
  // ------------------
  // n<64:  H0 H1 H2 H3   L,H   4
  //        L3 H0 H1 H2   L,H   3
  // ------------------
  // n<80:  H1 H2 H3 0    H,0   1
  //        H0 H1 H2 H3   H,0   0
  // ------------------
  // n<96:  H2 H3 0  0    H,0   2
  //        H1 H2 H3 0    H,0   1
  // ------------------
  // n<112: H3 0  0  0    H,0   3
  //        H2 H3 0  0    H,0   2
  // ------------------
  // n<128: 0  0  0  0    H,0   4
  //        H3 0  0  0    H,0   3

  /*
  // cast from any type to __m512i
  static SIMD_INLINE __m512i
  x_mm512_cast_si512(__m512i a)
  {
    return a;
  }
  static SIMD_INLINE __m512i
  x_mm512_cast_si512(__m512 a)
  {
    return _mm512_castps_si512(a);
  }
  */
  
  // just a wrapper (SIMDVec -> __m512i)
  // IMM: in bytes
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  alignr_16(const SIMDVec<T, 64> &h,
	    const SIMDVec<T, 64> &l)
  {
    return reinterpret<T>
      (SIMDVec<SIMDByte,64>
       (x_mm512_alignr_epi8<IMM>
	//(x_mm512_cast_si512(h), x_mm512_cast_si512(l))));
	(reinterpret<SIMDByte>(h), reinterpret<SIMDByte>(l))));
  }

  // positive and in range
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  alignre(const SIMDVec<T, 64> &h,
	  const SIMDVec<T, 64> &l,
	  IsPosInRange<true,true>)
  {
    enum { byteShift = IMM * sizeof(T), laneShift = byteShift / 16 };
    SIMDVec<T,64> L = (byteShift < 64) ? l : h;
    SIMDVec<T,64> H = (byteShift < 64) ? h : setzero<T,64>();
    return alignr_16<(byteShift % 16)>
      (align_64_16<(laneShift % 4 + 1)>(L, H),
       align_64_16<(laneShift % 4)>(L, H));
  }

  // positive and out of range
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  alignre(const SIMDVec<T, 64> &,
	  const SIMDVec<T, 64> &,
	  IsPosInRange<true,false>)
  {
    return setzero<T,64>();
  }

  // hub:
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  alignre(const SIMDVec<T, 64> &h,
	  const SIMDVec<T, 64> &l)
  {
    return alignre<IMM>(h, l,
			IsPosInGivenRange<2*SIMDVec<T,64>::elements, IMM>());
  }

  // ---------------------------------------------------------------------------
  // srle: element-wise right shift (via alignre) v
  // ---------------------------------------------------------------------------

  // TODO: srle: solution with byte-wise shift intrinsics instead of align?

  // positive and in range
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  srle(const SIMDVec<T, 64> &a,
       IsPosInRange<true,true>)
  {
    return alignre<IMM>(setzero<T,64>(), a);
  }

  // positive and out of range
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  srle(const SIMDVec<T, 64> &,
       IsPosInRange<true,false>)
  {
    return setzero<T,64>();
  }

  // hub:
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  srle(const SIMDVec<T, 64> &a)
  {
    return srle<IMM>(a, IsPosInGivenRange<SIMDVec<T,64>::elements, IMM>());
  }

  // ---------------------------------------------------------------------------
  // slle: element-wise left shift (via alignre) v
  // ---------------------------------------------------------------------------

  // TODO: slle: solution with byte-wise shift intrinsics instead of align?
  
  // positive and in range
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  slle(const SIMDVec<T, 64> &a,
       IsPosInRange<true,true>)
  {
    return alignre<SIMDVec<T,64>::elements - IMM>(a, setzero<T,64>());
  }

  // positive and out of range
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  slle(const SIMDVec<T, 64> &,
       IsPosInRange<true,false>)
  {
    return setzero<T,64>();
  }

  // hub:
  // IMM: in elements
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,64>
  slle(const SIMDVec<T, 64> &a)
  {
    return slle<IMM>(a, IsPosInGivenRange<SIMDVec<T,64>::elements, IMM>());
  }
  
  // ---------------------------------------------------------------------------
  // swizzle v
  // ---------------------------------------------------------------------------

  // ---------- swizzle tables ----------

  // TODO: N as template parameter is short but a bit dangerous (if N
  // TODO: exceeds range 1..5)

  template <int N>
  struct SwizzleTable<N,SIMDByte,64>
  {
    SIMDVec<SIMDByte,64> mask;
    SwizzleTable() 
    { mask = x_mm512_duplicate_si128(load<16>(swizzleByteMask16_[N])); }
  };

  template <int N>
  struct SwizzleTable<N,SIMDSignedByte,64>
  {
    SIMDVec<SIMDByte,64> mask;
    SwizzleTable() 
    { mask = x_mm512_duplicate_si128(load<16>(swizzleByteMask16_[N])); }
  };

  template <int N>
  struct SwizzleTable<N,SIMDWord,64>
  {
    SIMDVec<SIMDByte,64> mask;
    SwizzleTable() 
    { mask = x_mm512_duplicate_si128(load<16>(swizzleWordMask16_[N])); }
  };

  template <int N>
  struct SwizzleTable<N,SIMDShort,64>
  {
    SIMDVec<SIMDByte,64> mask;
    SwizzleTable() 
    { mask = x_mm512_duplicate_si128(load<16>(swizzleWordMask16_[N])); }
  };

  // no mask required
  template <int N>
  struct SwizzleTable<N,SIMDInt,64>
  {
    SwizzleTable() {}
  };

  // no mask required
  template <int N>
  struct SwizzleTable<N,SIMDFloat,64>
  {
    SwizzleTable() {}
  };

  // ---------- swizzle aux functions -----------

  // alignoff is the element-wise offset (relates to size of byte)
  template <int ALIGNOFF>
  static SIMD_INLINE __m512i 
  align_shuffle_byte_512(__m512i lo, __m512i hi, __m512i mask)
  {
    return x_mm512_shuffle_epi8(x_mm512_alignr_epi8<ALIGNOFF>(hi, lo), mask);
  }

  // alignoff is the element-wise offset (relates to size of word)
  template <int ALIGNOFF>
  static SIMD_INLINE __m512i 
  align_shuffle_word_512(__m512i lo, __m512i hi, __m512i mask)
  {
    return x_mm512_shuffle_epi8(x_mm512_alignr_epi8<2*ALIGNOFF>(hi, lo), mask);
  }

  // ---------- swizzle (AoS to SoA) ----------

  // -------------------- n = 1 --------------------

  // all types
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<1, T, 64> &,
	  SIMDVec<T, 64> *const,
	  Integer<1>,
	  TypeIsIntSize<T>)
  {
    // v remains unchanged
  }

  // -------------------- n = 2 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<2, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<2>,
	  IsIntSize<true,1>)
  {
    SIMDVec<T, 64> vs[2];
    swizzle_64_16<2>(v, vs);
    __m512i s[2];
    for (int j = 0; j < 2; j++)
      s[j] = x_mm512_shuffle_epi8(vs[j], t.mask);
    v[0] = _mm512_unpacklo_epi64(s[0], s[1]);
    v[1] = _mm512_unpackhi_epi64(s[0], s[1]);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<2, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<2>,
	  IsIntSize<true,2>)
  {
    SIMDVec<T, 64> vs[2];
    swizzle_64_16<2>(v, vs);
    __m512i s[2];
    for (int j = 0; j < 2; j++)
      s[j] = x_mm512_shuffle_epi8(vs[j], t.mask);
    v[0] = _mm512_unpacklo_epi64(s[0], s[1]);
    v[1] = _mm512_unpackhi_epi64(s[0], s[1]);
  }

  // SIMDInt
  // TODO: swizzle<2,SIMDInt,...>: which version is faster?
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<2, T, 64> &,
	  SIMDVec<T, 64> *const v,
	  Integer<2>,
	  IsIntSize<true,4>)
  {
#if 0
    SIMDVec<T, 64> vs[2];
    swizzle_64_16<2>(v, vs);  
    __m512i s[2];
    s[0] = _mm512_shuffle_epi32(vs[0], _MM_SHUFFLE(3,1,2,0));
    s[1] = _mm512_shuffle_epi32(vs[1], _MM_SHUFFLE(3,1,2,0));
    v[0] = _mm512_unpacklo_epi64(s[0], s[1]);
    v[1] = _mm512_unpackhi_epi64(s[0], s[1]);
#else
    SIMDVec<T, 64> vs[2];
    swizzle_64_16<2>(v, vs);
    __m512 v0tmp = _mm512_castsi512_ps(vs[0]);
    __m512 v1tmp = _mm512_castsi512_ps(vs[1]);
    v[0] = _mm512_castps_si512(_mm512_shuffle_ps(v0tmp, v1tmp, 
						 _MM_SHUFFLE(2,0,2,0)));
    v[1] = _mm512_castps_si512(_mm512_shuffle_ps(v0tmp, v1tmp, 
						 _MM_SHUFFLE(3,1,3,1)));
#endif
  }

  // SIMDFloat
  // same code as for SIMDInt
  static SIMD_INLINE void
  swizzle(const SwizzleTable<2, SIMDFloat, 64> &,
	  SIMDVec<SIMDFloat, 64> *const v,
	  Integer<2>,
	  IsIntSize<false,4>)
  {
    SIMDVec<SIMDFloat, 64> vs[2];
    swizzle_64_16<2>(v, vs);
    __m512 v0tmp = vs[0];
    __m512 v1tmp = vs[1];
    v[0] = _mm512_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(2,0,2,0));
    v[1] = _mm512_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(3,1,3,1));
  }

  // -------------------- n = 3 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<3, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<3>,
	  IsIntSize<true,1>)
  {
    SIMDVec<T ,64> vs[3];
    swizzle_64_16<3>(v, vs);
    __m512i s0 = align_shuffle_byte_512<0> (vs[0], vs[1], t.mask);
    __m512i s1 = align_shuffle_byte_512<12>(vs[0], vs[1], t.mask);
    __m512i s2 = align_shuffle_byte_512<8> (vs[1], vs[2], t.mask);
    __m512i s3 = align_shuffle_byte_512<4> (vs[2], vs[0], t.mask);
    /* s3: v[0] is a dummy */
    __m512i l01 = _mm512_unpacklo_epi32(s0, s1);
    __m512i h01 = _mm512_unpackhi_epi32(s0, s1);
    __m512i l23 = _mm512_unpacklo_epi32(s2, s3);
    __m512i h23 = _mm512_unpackhi_epi32(s2, s3);
    v[0] = _mm512_unpacklo_epi64(l01, l23);
    v[1] = _mm512_unpackhi_epi64(l01, l23);
    v[2] = _mm512_unpacklo_epi64(h01, h23);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<3, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<3>,
	  IsIntSize<true,2>)
  {
    SIMDVec<T, 64> vs[3];
    swizzle_64_16<3>(v, vs);
    __m512i s0 = align_shuffle_word_512<0>(vs[0], vs[1], t.mask);
    __m512i s1 = align_shuffle_word_512<6>(vs[0], vs[1], t.mask);
    __m512i s2 = align_shuffle_word_512<4>(vs[1], vs[2], t.mask);
    __m512i s3 = align_shuffle_word_512<2>(vs[2], vs[0], t.mask);
    // s3: v[0] is a dummy
    __m512i l01 = _mm512_unpacklo_epi32(s0, s1);
    __m512i h01 = _mm512_unpackhi_epi32(s0, s1);
    __m512i l23 = _mm512_unpacklo_epi32(s2, s3);
    __m512i h23 = _mm512_unpackhi_epi32(s2, s3);
    v[0] = _mm512_unpacklo_epi64(l01, l23);
    v[1] = _mm512_unpackhi_epi64(l01, l23);
    v[2] = _mm512_unpacklo_epi64(h01, h23);
  }


  // SIMDInt
  // from Stan Melax: "3D Vector Normalization..."
  // https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<3, T, 64> &,
	  SIMDVec<T, 64> *const v,
	  Integer<3>,
	  IsIntSize<true,4>)
  {
    SIMDVec<T, 64> vs[3];
    swizzle_64_16<3>(v, vs);
    __m512 x0y0z0x1 = _mm512_castsi512_ps(vs[0]);
    __m512 y1z1x2y2 = _mm512_castsi512_ps(vs[1]);
    __m512 z2x3y3z3 = _mm512_castsi512_ps(vs[2]);
    __m512 x2y2x3y3 = _mm512_shuffle_ps(y1z1x2y2, z2x3y3z3, 
					_MM_SHUFFLE(2,1,3,2));
    __m512 y0z0y1z1 = _mm512_shuffle_ps(x0y0z0x1, y1z1x2y2, 
					_MM_SHUFFLE(1,0,2,1));
    // x0x1x2x3
    v[0] = _mm512_castps_si512(_mm512_shuffle_ps(x0y0z0x1, x2y2x3y3, 
						 _MM_SHUFFLE(2,0,3,0)));
    // y0y1y2y3
    v[1] = _mm512_castps_si512(_mm512_shuffle_ps(y0z0y1z1, x2y2x3y3, 
						 _MM_SHUFFLE(3,1,2,0)));
    // z0z1z2z3
    v[2] = _mm512_castps_si512(_mm512_shuffle_ps(y0z0y1z1, z2x3y3z3, 
						 _MM_SHUFFLE(3,0,3,1)));
  }

  // SIMDFloat
  // from Stan Melax: "3D Vector Normalization..."
  // https://software.intel.com/en-us/articles/3d-vector-normalization-using-512-bit-intel-advanced-vector-extensions-intel-avx
  // same code as for SIMDInt
  static SIMD_INLINE void
  swizzle(const SwizzleTable<3, SIMDFloat, 64> &,
	  SIMDVec<SIMDFloat, 64> *const v,
	  Integer<3>,
	  IsIntSize<false,4>)
  {
    SIMDVec<SIMDFloat, 64> vs[3];
    swizzle_64_16<3>(v, vs);
    // x0y0z0x1 = v[0]
    // y1z1x2y2 = v[1]
    // z2x3y3z3 = v[2]
    __m512 x2y2x3y3 = _mm512_shuffle_ps(vs[1], vs[2], _MM_SHUFFLE(2,1,3,2));
    __m512 y0z0y1z1 = _mm512_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(1,0,2,1));
    // x0x1x2x3
    v[0] = _mm512_shuffle_ps(vs[0], x2y2x3y3, _MM_SHUFFLE(2,0,3,0));
    // y0y1y2y3
    v[1] = _mm512_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3,1,2,0));
    // z0z1z2z3
    v[2] = _mm512_shuffle_ps(y0z0y1z1, vs[2], _MM_SHUFFLE(3,0,3,1));
  }

  // -------------------- n = 4 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<4, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<4>,
	  IsIntSize<true,1>)
  {
    SIMDVec<T, 64> vs[4];
    swizzle_64_16<4>(v, vs);
    __m512i s[4];
    for (int j = 0; j < 4; j++)
      s[j] = x_mm512_shuffle_epi8(vs[j], t.mask);
    __m512i l01 = _mm512_unpacklo_epi32(s[0], s[1]);
    __m512i h01 = _mm512_unpackhi_epi32(s[0], s[1]);
    __m512i l23 = _mm512_unpacklo_epi32(s[2], s[3]);
    __m512i h23 = _mm512_unpackhi_epi32(s[2], s[3]);
    v[0] = _mm512_unpacklo_epi64(l01, l23);
    v[1] = _mm512_unpackhi_epi64(l01, l23);
    v[2] = _mm512_unpacklo_epi64(h01, h23);
    v[3] = _mm512_unpackhi_epi64(h01, h23);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<4, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<4>,
	  IsIntSize<true,2>)
  {
    SIMDVec<T, 64> vs[4];
    swizzle_64_16<4>(v, vs);
    __m512i s[4];
    for (int j = 0; j < 4; j++)
      s[j] = x_mm512_shuffle_epi8(vs[j], t.mask);
    __m512i l01 = _mm512_unpacklo_epi32(s[0], s[1]);
    __m512i h01 = _mm512_unpackhi_epi32(s[0], s[1]);
    __m512i l23 = _mm512_unpacklo_epi32(s[2], s[3]);
    __m512i h23 = _mm512_unpackhi_epi32(s[2], s[3]);
    v[0] = _mm512_unpacklo_epi64(l01, l23);
    v[1] = _mm512_unpackhi_epi64(l01, l23);
    v[2] = _mm512_unpacklo_epi64(h01, h23);
    v[3] = _mm512_unpackhi_epi64(h01, h23);
  }

  // SIMDInt
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<4, T, 64> &,
	  SIMDVec<T, 64> *const v,
	  Integer<4>,
	  IsIntSize<true,4>)
  {
    SIMDVec<T, 64> vs[4];
    swizzle_64_16<4>(v, vs);
    __m512i s[4];
    s[0] = _mm512_unpacklo_epi32(vs[0], vs[1]);
    s[1] = _mm512_unpackhi_epi32(vs[0], vs[1]);
    s[2] = _mm512_unpacklo_epi32(vs[2], vs[3]);
    s[3] = _mm512_unpackhi_epi32(vs[2], vs[3]);
    v[0] = _mm512_unpacklo_epi64(s[0], s[2]);
    v[1] = _mm512_unpackhi_epi64(s[0], s[2]);
    v[2] = _mm512_unpacklo_epi64(s[1], s[3]);
    v[3] = _mm512_unpackhi_epi64(s[1], s[3]);
  }

  // SIMDFloat
  static SIMD_INLINE void
  swizzle(const SwizzleTable<4, SIMDFloat, 64> &,
	  SIMDVec<SIMDFloat, 64> *const v,
	  Integer<4>,
	  IsIntSize<false,4>)
  {
    SIMDVec<SIMDFloat, 64> vs[4];
    swizzle_64_16<4>(v, vs);
    __m512 s[4];
    s[0] = _mm512_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(1,0,1,0));
    s[1] = _mm512_shuffle_ps(vs[0], vs[1], _MM_SHUFFLE(3,2,3,2));
    s[2] = _mm512_shuffle_ps(vs[2], vs[3], _MM_SHUFFLE(1,0,1,0));
    s[3] = _mm512_shuffle_ps(vs[2], vs[3], _MM_SHUFFLE(3,2,3,2));
    v[0] = _mm512_shuffle_ps(s[0], s[2], _MM_SHUFFLE(2,0,2,0));
    v[1] = _mm512_shuffle_ps(s[0], s[2], _MM_SHUFFLE(3,1,3,1));
    v[2] = _mm512_shuffle_ps(s[1], s[3], _MM_SHUFFLE(2,0,2,0));
    v[3] = _mm512_shuffle_ps(s[1], s[3], _MM_SHUFFLE(3,1,3,1));
  }

  // -------------------- n = 5 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<5, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<5>,
	  IsIntSize<true,1>)
  {
    SIMDVec<T, 64> vs[5];
    swizzle_64_16<5>(v, vs);
    __m512i s0 = align_shuffle_byte_512<0> (vs[0], vs[1], t.mask);
    __m512i s1 = align_shuffle_byte_512<10>(vs[0], vs[1], t.mask);
    __m512i s2 = align_shuffle_byte_512<4> (vs[1], vs[2], t.mask);
    __m512i s3 = align_shuffle_byte_512<14>(vs[1], vs[2], t.mask);
    __m512i s4 = align_shuffle_byte_512<8> (vs[2], vs[3], t.mask);
    __m512i s5 = align_shuffle_byte_512<2> (vs[3], vs[4], t.mask);
    __m512i s6 = align_shuffle_byte_512<12>(vs[3], vs[4], t.mask);
    __m512i s7 = align_shuffle_byte_512<6> (vs[4], vs[0], t.mask);
    /* s7: v[0] is a dummy */
    __m512i l01 = x_mm512_unpacklo_epi16(s0, s1);
    __m512i h01 = x_mm512_unpackhi_epi16(s0, s1);
    __m512i l23 = x_mm512_unpacklo_epi16(s2, s3);
    __m512i h23 = x_mm512_unpackhi_epi16(s2, s3);
    __m512i l45 = x_mm512_unpacklo_epi16(s4, s5);
    __m512i h45 = x_mm512_unpackhi_epi16(s4, s5);
    __m512i l67 = x_mm512_unpacklo_epi16(s6, s7);
    __m512i h67 = x_mm512_unpackhi_epi16(s6, s7);
    __m512i ll01l23 = _mm512_unpacklo_epi32(l01, l23);
    __m512i hl01l23 = _mm512_unpackhi_epi32(l01, l23);
    __m512i ll45l67 = _mm512_unpacklo_epi32(l45, l67);
    __m512i hl45l67 = _mm512_unpackhi_epi32(l45, l67);
    __m512i lh01h23 = _mm512_unpacklo_epi32(h01, h23);
    __m512i lh45h67 = _mm512_unpacklo_epi32(h45, h67);
    v[0] = _mm512_unpacklo_epi64(ll01l23, ll45l67);
    v[1] = _mm512_unpackhi_epi64(ll01l23, ll45l67);
    v[2] = _mm512_unpacklo_epi64(hl01l23, hl45l67);
    v[3] = _mm512_unpackhi_epi64(hl01l23, hl45l67);
    v[4] = _mm512_unpacklo_epi64(lh01h23, lh45h67);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<5, T, 64> &t,
	  SIMDVec<T, 64> *const v,
	  Integer<5>,
	  IsIntSize<true,2>)
  {
    SIMDVec<T, 64> vs[5];
    swizzle_64_16<5>(v, vs);
    __m512i s0 = align_shuffle_word_512<0>(vs[0], vs[1], t.mask);
    __m512i s1 = align_shuffle_word_512<3>(vs[0], vs[1], t.mask);
    __m512i s2 = align_shuffle_word_512<2>(vs[1], vs[2], t.mask);
    __m512i s3 = align_shuffle_word_512<5>(vs[1], vs[2], t.mask);
    __m512i s4 = align_shuffle_word_512<4>(vs[2], vs[3], t.mask);
    __m512i s5 = align_shuffle_word_512<7>(vs[2], vs[3], t.mask);
    __m512i s6 = align_shuffle_word_512<6>(vs[3], vs[4], t.mask);
    __m512i s7 = align_shuffle_word_512<1>(vs[4], vs[0], t.mask);
    // s7: v[0] is a dummy
    __m512i l02 = _mm512_unpacklo_epi32(s0, s2);
    __m512i h02 = _mm512_unpackhi_epi32(s0, s2);
    __m512i l13 = _mm512_unpacklo_epi32(s1, s3);
    __m512i l46 = _mm512_unpacklo_epi32(s4, s6);
    __m512i h46 = _mm512_unpackhi_epi32(s4, s6);
    __m512i l57 = _mm512_unpacklo_epi32(s5, s7);
    v[0] = _mm512_unpacklo_epi64(l02, l46);
    v[1] = _mm512_unpackhi_epi64(l02, l46);
    v[2] = _mm512_unpacklo_epi64(h02, h46);
    v[3] = _mm512_unpacklo_epi64(l13, l57);
    v[4] = _mm512_unpackhi_epi64(l13, l57);
  }

  // SIMDInt
  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<5, T, 64> &,
	  SIMDVec<T, 64> *const v,
	  Integer<5>,
	  IsIntSize<true,4>)
  {
    SIMDVec<T, 64> vs[5];
    swizzle_64_16<5>(v, vs);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    // v[0]: 0 1 2 3
    // v[1]: 4 x x x
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                   x x x   x
    // 5 6 7 8
    __m512i s2 = x_mm512_alignr_epi8<4>(vs[2], vs[1]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                             x  x  x    x
    // 9 x x x
    __m512i s3 = x_mm512_alignr_epi8<4>(vs[3], vs[2]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                x  x    x  x
    // 10 11 12 13
    __m512i s4 = x_mm512_alignr_epi8<8>(vs[3], vs[2]); 
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                              x  x    x  x
    // 14 x x x
    __m512i s5 = x_mm512_alignr_epi8<8>(vs[4], vs[3]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                 X    X  X  X
    // 15 16 17 18
    __m512i s6 = x_mm512_alignr_epi8<12>(vs[4], vs[3]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                               X X X X
    // 19 x x x
    __m512i s7 = x_mm512_alignr_epi8<12>(vs[0], vs[4]);
    // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
    __m512i l02 = _mm512_unpacklo_epi32(vs[0], s2);
    __m512i h02 = _mm512_unpackhi_epi32(vs[0], s2);
    // 4 x x x / 9 x x x -> 4 9 x x
    __m512i l13 = _mm512_unpacklo_epi32(vs[1], s3);
    // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
    __m512i l46 = _mm512_unpacklo_epi32(s4, s6);
    __m512i h46 = _mm512_unpackhi_epi32(s4, s6);
    // 14 x x x / 19 x x x -> 14 19 x x
    __m512i l57 = _mm512_unpacklo_epi32(s5, s7);
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    v[0] = _mm512_unpacklo_epi64(l02, l46);
    v[1] = _mm512_unpackhi_epi64(l02, l46);
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    v[2] = _mm512_unpacklo_epi64(h02, h46);
    v[3] = _mm512_unpackhi_epi64(h02, h46);
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    v[4] = _mm512_unpacklo_epi64(l13, l57);
  }

  // SIMDFloat
  // same code as for SIMDInt, modified
  static SIMD_INLINE void
  swizzle(const SwizzleTable<5, SIMDFloat, 64> &,
	  SIMDVec<SIMDFloat, 64> *const v,
	  Integer<5>,
	  IsIntSize<false,4>)
  {
    SIMDVec<SIMDFloat, 64> vs[5];
    swizzle_64_16<5>(v, vs);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    // v[0]: 0 1 2 3
    __m512i v0 = _mm512_castps_si512(vs[0]);
    // v[1]: 4 5 6 7
    __m512i v1 = _mm512_castps_si512(vs[1]);
    // v[2]: 8 9 10 11
    __m512i v2 = _mm512_castps_si512(vs[2]);
    // v[3]: 12 13 14 15
    __m512i v3 = _mm512_castps_si512(vs[3]);
    // v[4]: 16 17 18 19
    __m512i v4 = _mm512_castps_si512(vs[4]);
    // s0:  0 1 2 3
    __m512 s0 = vs[0];
    // s1:  4 x x x
    __m512 s1 = vs[1];
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                   x x x   x
    // 5 6 7 8
    __m512 s2 = _mm512_castsi512_ps(x_mm512_alignr_epi8<4>(v2, v1));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                             x  x  x    x
    // 9 x x x
    __m512 s3 = _mm512_castsi512_ps(x_mm512_alignr_epi8<4>(v3, v2));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                x  x    x  x
    // 10 11 12 13
    __m512 s4 = _mm512_castsi512_ps(x_mm512_alignr_epi8<8>(v3, v2)); 
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                              x  x    x  x
    // 14 x x x
    __m512 s5 = _mm512_castsi512_ps(x_mm512_alignr_epi8<8>(v4, v3));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                 X    X  X  X
    // 15 16 17 18
    __m512 s6 = _mm512_castsi512_ps(x_mm512_alignr_epi8<12>(v4, v3));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                               X X X X
    // 19 x x x
    __m512 s7 = _mm512_castsi512_ps(x_mm512_alignr_epi8<12>(v0, v4));
    // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
    __m512 l02 = _mm512_unpacklo_ps(s0, s2);
    __m512 h02 = _mm512_unpackhi_ps(s0, s2);
    // 4 x x x / 9 x x x -> 4 9 x x
    __m512 l13 = _mm512_unpacklo_ps(s1, s3);
    // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
    __m512 l46 = _mm512_unpacklo_ps(s4, s6);
    __m512 h46 = _mm512_unpackhi_ps(s4, s6);
    // 14 x x x / 19 x x x -> 14 19 x x
    __m512 l57 = _mm512_unpacklo_ps(s5, s7);
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    // v[0] = x_mm512_movelh_ps(l02, l46);
    // v[1] = x_mm512_movehl_ps(l46, l02);
    v[0] = x_mm512_unpacklo_2ps(l02, l46);
    v[1] = x_mm512_unpackhi_2ps(l02, l46);
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    // v[2] = x_mm512_movelh_ps(h02, h46);
    // v[3] = x_mm512_movehl_ps(h46, h02);
    v[2] = x_mm512_unpacklo_2ps(h02, h46);
    v[3] = x_mm512_unpackhi_2ps(h02, h46);
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    // v[4] = x_mm512_movelh_ps(l13, l57);
    v[4] = x_mm512_unpacklo_2ps(l13, l57);
  }

  // ---------------------------------------------------------------------------
  // compare < v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  cmplt(const SIMDVec<SIMDByte,64> &a,
	const SIMDVec<SIMDByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmplt_epu8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  cmplt(const SIMDVec<SIMDSignedByte,64> &a,
	const SIMDVec<SIMDSignedByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  cmplt(const SIMDVec<SIMDWord,64> &a,
	const SIMDVec<SIMDWord,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmplt_epu16_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  cmplt(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmplt_epi16_mask(a, b));
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  cmplt(const SIMDVec<T,64> &a,
	const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(cmplt(a.lo(), b.lo()),
			 cmplt(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  cmplt(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_movm_epi32(_mm512_cmplt_epi32_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  cmplt(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_castsi512_ps
      (x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_LT_OS)));
  }

  // ---------------------------------------------------------------------------
  // compare <= v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  cmple(const SIMDVec<SIMDByte,64> &a,
	const SIMDVec<SIMDByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmple_epu8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  cmple(const SIMDVec<SIMDSignedByte,64> &a,
	const SIMDVec<SIMDSignedByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmple_epi8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  cmple(const SIMDVec<SIMDWord,64> &a,
	const SIMDVec<SIMDWord,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmple_epu16_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  cmple(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmple_epi16_mask(a, b));
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  cmple(const SIMDVec<T,64> &a,
	const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(cmple(a.lo(), b.lo()),
			 cmple(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  cmple(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_movm_epi32(_mm512_cmple_epi32_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  cmple(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_castsi512_ps
      (x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_LE_OS)));
  }

  // ---------------------------------------------------------------------------
  // compare == v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  cmpeq(const SIMDVec<SIMDByte,64> &a,
	const SIMDVec<SIMDByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpeq_epu8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  cmpeq(const SIMDVec<SIMDSignedByte,64> &a,
	const SIMDVec<SIMDSignedByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpeq_epi8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  cmpeq(const SIMDVec<SIMDWord,64> &a,
	const SIMDVec<SIMDWord,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpeq_epu16_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  cmpeq(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpeq_epi16_mask(a, b));
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  cmpeq(const SIMDVec<T,64> &a,
	const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(cmpeq(a.lo(), b.lo()),
			 cmpeq(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  cmpeq(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_movm_epi32(_mm512_cmpeq_epi32_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  cmpeq(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_castsi512_ps
      (x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ)));
  }

  // ---------------------------------------------------------------------------
  // compare > v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  cmpgt(const SIMDVec<SIMDByte,64> &a,
	const SIMDVec<SIMDByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpgt_epu8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  cmpgt(const SIMDVec<SIMDSignedByte,64> &a,
	const SIMDVec<SIMDSignedByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpgt_epi8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  cmpgt(const SIMDVec<SIMDWord,64> &a,
	const SIMDVec<SIMDWord,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpgt_epu16_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  cmpgt(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpgt_epi16_mask(a, b));
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  cmpgt(const SIMDVec<T,64> &a,
	const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(cmpgt(a.lo(), b.lo()),
			 cmpgt(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  cmpgt(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_movm_epi32(_mm512_cmpgt_epi32_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  cmpgt(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_castsi512_ps
      (x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_GT_OS)));
  }

  // ---------------------------------------------------------------------------
  // compare >= v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  cmpge(const SIMDVec<SIMDByte,64> &a,
	const SIMDVec<SIMDByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpge_epu8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  cmpge(const SIMDVec<SIMDSignedByte,64> &a,
	const SIMDVec<SIMDSignedByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpge_epi8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  cmpge(const SIMDVec<SIMDWord,64> &a,
	const SIMDVec<SIMDWord,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpge_epu16_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  cmpge(const SIMDVec<SIMDShort,64> &a,
	const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpge_epi16_mask(a, b));
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  cmpge(const SIMDVec<T,64> &a,
	const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(cmpge(a.lo(), b.lo()),
			 cmpge(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  cmpge(const SIMDVec<SIMDInt,64> &a,
	const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_movm_epi32(_mm512_cmpge_epi32_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  cmpge(const SIMDVec<SIMDFloat,64> &a,
	const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_castsi512_ps
      (x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_GE_OS)));
  }

  // ---------------------------------------------------------------------------
  // compare != v
  // ---------------------------------------------------------------------------

  // https://stackoverflow.com/questions/48099006/
  // different-semantic-of-comparison-intrinsic-instructions-in-avx512

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  cmpneq(const SIMDVec<SIMDByte,64> &a,
	 const SIMDVec<SIMDByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpneq_epu8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  cmpneq(const SIMDVec<SIMDSignedByte,64> &a,
	 const SIMDVec<SIMDSignedByte,64> &b)
  {
    return x_mm512_movm_epi8(_mm512_cmpneq_epi8_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  cmpneq(const SIMDVec<SIMDWord,64> &a,
	 const SIMDVec<SIMDWord,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpneq_epu16_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,64>
  cmpneq(const SIMDVec<SIMDShort,64> &a,
	 const SIMDVec<SIMDShort,64> &b)
  {
    return x_mm512_movm_epi16(_mm512_cmpneq_epi16_mask(a, b));
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  cmpneq(const SIMDVec<T,64> &a,
	 const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(cmpneq(a.lo(), b.lo()),
			 cmpneq(a.hi(), b.hi()));
  }
  
#endif

  static SIMD_INLINE SIMDVec<SIMDInt,64>
  cmpneq(const SIMDVec<SIMDInt,64> &a,
	 const SIMDVec<SIMDInt,64> &b)
  {
    return x_mm512_movm_epi32(_mm512_cmpneq_epi32_mask(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  cmpneq(const SIMDVec<SIMDFloat,64> &a,
	 const SIMDVec<SIMDFloat,64> &b)
  {
    return _mm512_castsi512_ps
      (x_mm512_movm_epi32(_mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ)));
  }

  // ---------------------------------------------------------------------------
  // ifelse v
  // ---------------------------------------------------------------------------

  // NOTE: only works if cond elements are all 1-bits or all 0-bits
  // TODO: is there a more efficient solution without AVX512BW?
  
#ifdef __AVX512BW__
  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  ifelse(const SIMDVec<T,64> &cond,
	 const SIMDVec<T,64> &trueVal,
	 const SIMDVec<T,64> &falseVal)
  {
    // cond -> __mask64
    return x_mm512_mask_blend_epi8
      (_mm512_movepi8_mask(cond),
       falseVal,
       trueVal);
  }
#else
  // non-avx512bw workaround
  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  ifelse(const SIMDVec<T,64> &cond,
	 const SIMDVec<T,64> &trueVal,
	 const SIMDVec<T,64> &falseVal)
  {
    return _mm512_castps_si512
      (x_mm512_or_ps(x_mm512_and_ps(_mm512_castsi512_ps(cond), 
				    _mm512_castsi512_ps(trueVal)),
		     x_mm512_andnot_ps(_mm512_castsi512_ps(cond), 
				       _mm512_castsi512_ps(falseVal))));
  }
#endif
  
  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  ifelse(const SIMDVec<SIMDFloat,64> &cond,
	 const SIMDVec<SIMDFloat,64> &trueVal,
	 const SIMDVec<SIMDFloat,64> &falseVal)
  {
#ifdef __AVX512DQ__
    return _mm512_mask_blend_ps
      (_mm512_movepi32_mask(reinterpret<SIMDInt>(cond)),
       falseVal, trueVal);
#else
    return reinterpret<SIMDFloat>
      (ifelse(reinterpret<SIMDInt>(cond),
	      reinterpret<SIMDInt>(trueVal),
	      reinterpret<SIMDInt>(falseVal)));
#endif
  }

  // ---------------------------------------------------------------------------
  // and v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  and(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return _mm512_and_si512(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  and(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return x_mm512_and_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // or v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  or(const SIMDVec<T,64> &a,
     const SIMDVec<T,64> &b)
  {
    return _mm512_or_si512(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  or(const SIMDVec<SIMDFloat,64> &a,
     const SIMDVec<SIMDFloat,64> &b)
  {
    return x_mm512_or_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // andnot v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  andnot(const SIMDVec<T,64> &a,
	 const SIMDVec<T,64> &b)
  {
    return _mm512_andnot_si512(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  andnot(const SIMDVec<SIMDFloat,64> &a,
	 const SIMDVec<SIMDFloat,64> &b)
  {
    return x_mm512_andnot_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // xor v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  xor(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return _mm512_xor_si512(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  xor(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    return x_mm512_xor_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // not v
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  not(const SIMDVec<T,64> &a)
  {
    return x_mm512_not_si512(a);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  not(const SIMDVec<SIMDFloat,64> &a)
  {
    return x_mm512_not_ps(a);
  }

  // ---------------------------------------------------------------------------
  // avg: average with rounding down v
  // ---------------------------------------------------------------------------

#ifdef __AVX512BW__

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  avg(const SIMDVec<SIMDByte,64> &a,
      const SIMDVec<SIMDByte,64> &b)
  {
    return _mm512_avg_epu8(a, b);
  }

  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  avg(const SIMDVec<SIMDSignedByte,64> &a,
      const SIMDVec<SIMDSignedByte,64> &b)
  {
    // from Agner Fog's VCL vectori128.h
    __m512i signbit = _mm512_set1_epi32(0x80808080);
    __m512i a1      = _mm512_xor_si512(a, signbit); // add 0x80
    __m512i b1      = _mm512_xor_si512(b, signbit); // add 0x80
    __m512i m1      = _mm512_avg_epu8(a1, b1);	    // unsigned avg
    return  _mm512_xor_si512(m1, signbit);	    // sub 0x80
  }
  
  static SIMD_INLINE SIMDVec<SIMDWord,64>
  avg(const SIMDVec<SIMDWord,64> &a,
      const SIMDVec<SIMDWord,64> &b)
  {
    return _mm512_avg_epu16(a, b);
  }

  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDShort,64>
  avg(const SIMDVec<SIMDShort,64> &a,
      const SIMDVec<SIMDShort,64> &b)
  {
    // from Agner Fog's VCL vectori128.h
    __m512i signbit = _mm512_set1_epi32(0x80008000);
    __m512i a1      = _mm512_xor_si512(a, signbit); // add 0x8000
    __m512i b1      = _mm512_xor_si512(b, signbit); // add 0x8000
    __m512i m1      = _mm512_avg_epu16(a1, b1);	    // unsigned avg
    return  _mm512_xor_si512(m1, signbit);	    // sub 0x8000
  }

#else

  // non-avx512bw workaround
  template <typename T>
  static SIMD_INLINE SIMDVec<T,64>
  avg(const SIMDVec<T,64> &a,
      const SIMDVec<T,64> &b)
  {
    return SIMDVec<T,64>(avg(a.lo(), b.lo()),
			 avg(a.hi(), b.hi()));
  }

#endif

  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  avg(const SIMDVec<SIMDInt,64> &a,
      const SIMDVec<SIMDInt,64> &b)
  {
    SIMDVec<SIMDInt,64> one = set1<SIMDInt,64>(1), as, bs, lsb;
    lsb = and(or(a, b), one);
    as = srai<1>(a);
    bs = srai<1>(b);
    return add(lsb, add(as, bs));
  }

  // NOTE: SIMDFloat version doesn't round!
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  avg(const SIMDVec<SIMDFloat,64> &a,
      const SIMDVec<SIMDFloat,64> &b)
  {
    __m512 half = _mm512_set1_ps(0.5f);
    return _mm512_mul_ps(_mm512_add_ps(a, b), half);
  }

  // ---------------------------------------------------------------------------
  // test_all_zeros v
  // ---------------------------------------------------------------------------

  static SIMD_INLINE int
  x_mm512_mask2int(__mmask16 k1)
  {
    // not supported by gcc 5.4
    // return _mm512_mask2int(k1);
    // workaround from https://gcc.gnu.org/ml/gcc-patches/2017-04/msg00374.html
    return (int) k1;
  }
  
  // all integer versions
  template <typename T>
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<T,64> &a)
  {
    // m[i] = (a[i] & a[i] != 0) = (a[i] != 0)
    __mmask16 m = _mm512_test_epi32_mask(a, a);
    return (x_mm512_mask2int(m) == 0);
  }

  // float version
  // note: contrary to IEEE 754, this function considers -0.0f to be negative
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<SIMDFloat,64> &a)
  {
    return test_all_zeros(reinterpret<SIMDInt>(a));
  }
  
  // ---------------------------------------------------------------------------
  // test_all_ones v
  // ---------------------------------------------------------------------------

  // description of testn intrinsics was not clear, chosen other way
  // note: contrary to IEEE 754, this function considers -0.0f to be negative
  template <typename T>
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<T,64> &a)
  {
    return test_all_zeros(not(a));
  }

  // ---------------------------------------------------------------------------
  // reverse
  // ---------------------------------------------------------------------------

  // also used for intra-lane shuffle (uses only lower 4 bit)
  const SIMDByte reverse_mask_1[64] SIMD_ATTR_ALIGNED(64) = {
    63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,
    47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32,
    31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0
  };

  const SIMDByte reverse_mask_2[64] SIMD_ATTR_ALIGNED(64) = {
    62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 49,
    46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33,
    30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17,
    14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1
  };
  
  const SIMDInt reverse_mask_4[16] SIMD_ATTR_ALIGNED(64) = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
  };

  static SIMD_INLINE SIMDVec<SIMDByte,64>
  reverse(const SIMDVec<SIMDByte,64> &a)
  {
#ifdef __AVX512VBMI__
    return _mm512_permutexvar_epi8(load<64>(reverse_mask_1), a);
#else
    SIMDVec<SIMDByte,64> r =
      x_mm512_shuffle_epi8(a, load<64>(reverse_mask_1));
    return permute_64_16_1<3,2,1,0>(r);
#endif
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,64>
  reverse(const SIMDVec<SIMDSignedByte,64> &a)
  {
#ifdef __AVX512VBMI__
    return _mm512_permutexvar_epi8(load<64>(reverse_mask_1), a);
#else
    SIMDVec<SIMDSignedByte,64> r =
      x_mm512_shuffle_epi8(a, load<64>(reverse_mask_1));
    return permute_64_16_1<3,2,1,0>(r);
#endif
  }

  static SIMD_INLINE SIMDVec<SIMDWord,64>
  reverse(const SIMDVec<SIMDWord,64> &a)
  {
#ifdef __AVX512VBMI__
    return _mm512_permutexvar_epi8(load<64>(reverse_mask_2), a);    
#else
    SIMDVec<SIMDWord,64> r =
      x_mm512_shuffle_epi8(a, load<64>(reverse_mask_2));
    return permute_64_16_1<3, 2, 1, 0>(r);
#endif
  }
  
  static SIMD_INLINE SIMDVec<SIMDShort,64>
  reverse(const SIMDVec<SIMDShort,64> &a)
  {
#ifdef __AVX512VBMI__
    return _mm512_permutexvar_epi8(load<64>(reverse_mask_2), a);    
#else
    SIMDVec<SIMDShort,64> r =
      x_mm512_shuffle_epi8(a, load<64>(reverse_mask_2));
    return permute_64_16_1<3, 2, 1, 0>(r);
#endif
  }
  
  static SIMD_INLINE SIMDVec<SIMDInt,64>
  reverse(const SIMDVec<SIMDInt,64> &a)
  {
    // AVX512F
    return _mm512_permutexvar_epi32(load<64>(reverse_mask_4), a);
  }

  // float version, slightly changed int version
  static SIMD_INLINE SIMDVec<SIMDFloat,64>
  reverse(const SIMDVec<SIMDFloat,64> &a)
  {
    // AVX512F
    return _mm512_permutexvar_ps(load<64>(reverse_mask_4), a);
  }


  // ---------------------------------------------------------------------------
  // movemask
  // ---------------------------------------------------------------------------

  // 27. Aug 22 (Jonas Keller): added movemask functions

#ifdef __AVX512BW__
  // TODO: workaround for movemask for 8 and 16 bits if AVX512BW is not available

  static SIMD_INLINE uint64_t
  movemask(const SIMDVec<SIMDByte,64> &a)
  {
    return _cvtmask64_u64(_mm512_movepi8_mask(a));
  }

  static SIMD_INLINE uint64_t
  movemask(const SIMDVec<SIMDSignedByte,64> &a)
  {
    return _cvtmask64_u64(_mm512_movepi8_mask(a));
  }

  static SIMD_INLINE uint64_t
  movemask(const SIMDVec<SIMDShort,64> &a)
  {
    return _cvtmask32_u32(_mm512_movepi16_mask(a));
  }

  static SIMD_INLINE uint64_t
  movemask(const SIMDVec<SIMDWord,64> &a)
  {
    return _cvtmask32_u32(_mm512_movepi16_mask(a));
  }
#endif // __AVX512BW__

  static SIMD_INLINE uint64_t
  movemask(const SIMDVec<SIMDInt,64> &a)
  {
    const __m512i mask = _mm512_set1_epi32(uint32_t(0x80000000));
    return _cvtmask16_u32(_mm512_test_epi32_mask(a, mask));
  }

  static SIMD_INLINE uint64_t
  movemask(const SIMDVec<SIMDFloat,64> &a)
  {
    const __m512i mask = _mm512_set1_epi32(0x80000000);
    return _cvtmask16_u32(_mm512_test_epi32_mask(_mm512_castps_si512(a), mask));
  }
}

#endif // __AVX__

#endif // _SIMD_VEC_INTEL_64_H_
