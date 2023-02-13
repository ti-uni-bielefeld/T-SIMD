// ===========================================================================
// 
// SIMDVecBaseImplIntel16.H --
// encapsulation for SSE Intel vector extensions
// inspired by Agner Fog's C++ Vector Class Library
// http://www.agner.org/optimize/#vectorclass
// (VCL License: GNU General Public License Version 3,
//  http://www.gnu.org/licenses/gpl-3.0.en.html)
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

#ifndef _SIMD_VEC_BASE_IMPL_INTEL_16_H_
#define _SIMD_VEC_BASE_IMPL_INTEL_16_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "ssse3_compat.H"

#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>


#ifdef __SSE2__

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
  //
  // - we could improve performance by using 128-bit instructions from
  //   AVX512-VL (e.g. permute instructions); at the moment the idea is that
  //   typically the widest vector width is used, so if AVX512 is available,
  //   SSE would only rarely be used
  //
  // ===========================================================================

  // ===========================================================================
  // SIMDVec integer instantiation for SSE
  // ===========================================================================

  // partial specialization for SIMD_WIDTH = 16
  template <typename T>
  class SIMDVec<T, 16>
  {
  public:
    typedef T Type;
    __m128i xmm;
    enum { elements = 16 / sizeof(T), bytes = 16 };
    // shorter version:
    enum { elems = elements };
    SIMDVec() {}
    SIMDVec(const __m128i &x) { xmm = x; }
    SIMDVec& operator=(const __m128i &x) { xmm = x; return *this; }
    operator __m128i() const { return xmm; }
    // 29. Nov 22 (Jonas Keller):
    // defined operators new and delete to ensure proper alignment, since
    // the default new and delete are not guaranteed to do so before C++17
    void *operator new(size_t size)
    { return simd_aligned_malloc(bytes, size); }
    void operator delete(void *p)
    { simd_aligned_free(p); }
    void *operator new[](size_t size)
    { return simd_aligned_malloc(bytes, size); }
    void operator delete[](void *p)
    { simd_aligned_free(p); }
  };

  // ===========================================================================
  // SIMDVec float specialization for SSE
  // ===========================================================================

  template <>
  class SIMDVec<SIMDFloat, 16>
  {
  public:
    typedef SIMDFloat Type;
    __m128 xmm;
    enum { elements = 16 / sizeof(SIMDFloat), bytes = 16 };
    // shorter version:
    enum { elems = elements };
    SIMDVec() {}
    SIMDVec(const __m128 &x) { xmm = x; }
    SIMDVec& operator=(const __m128 &x) { xmm = x; return *this; }
    operator __m128() const { return xmm; }
    // 29. Nov 22 (Jonas Keller):
    // defined operators new and delete to ensure proper alignment, since
    // the default new and delete are not guaranteed to do so before C++17
    void *operator new(size_t size)
    { return simd_aligned_malloc(bytes, size); }
    void operator delete(void *p)
    { simd_aligned_free(p); }
    void *operator new[](size_t size)
    { return simd_aligned_malloc(bytes, size); }
    void operator delete[](void *p)
    { simd_aligned_free(p); }
  };

namespace internal {
namespace base {

  // ===========================================================================
  // auxiliary functions
  // ===========================================================================

  // These functions either wrap SSE intrinsics (e.g. to handle
  // immediate arguments as template parameter), or switch between
  // implementations with different SSE* extensions, or provide
  // altered or additional functionality.
  // Only for use in wrapper functinos!

  // ---------------------------------------------------------------------------
  // min / max
  // ---------------------------------------------------------------------------

  static SIMD_INLINE __m128i
  x_mm_min_epi8(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_min_epi8(a, b);
#else
    // from Agner Fog's VCL vectori128.h
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit);            // add 0x80
    __m128i b1      = _mm_xor_si128(b, signbit);            // add 0x80
    __m128i m1      = _mm_min_epu8(a1, b1);                 // unsigned min
    return  _mm_xor_si128(m1, signbit);                     // sub 0x80
#endif
  }
  
  static SIMD_INLINE __m128i
  x_mm_min_epu16(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_min_epu16(a, b);
#else
    // from Agner Fog's VCL vectori128.h
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a, signbit);            // add 0x8000
    __m128i b1      = _mm_xor_si128(b, signbit);            // add 0x8000
    __m128i m1      = _mm_min_epi16(a1, b1);                // signed min
    return  _mm_xor_si128(m1, signbit);                     // sub 0x8000
#endif
  }

  static SIMD_INLINE __m128i
  x_mm_min_epi32(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_min_epi32(a, b);
#else
    // from Agner Fog's VCL vectori128.h (modified)
    __m128i gt = _mm_cmpgt_epi32(a, b);
    return _mm_or_si128(_mm_and_si128(gt, b), _mm_andnot_si128(gt, a));
#endif
  }

  static SIMD_INLINE __m128i
  x_mm_max_epi8(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_max_epi8(a, b);
#else
    // from Agner Fog's VCL vectori128.h
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit);            // add 0x80
    __m128i b1      = _mm_xor_si128(b, signbit);            // add 0x80
    __m128i m1      = _mm_max_epu8(a1, b1);                 // unsigned max
    return  _mm_xor_si128(m1, signbit);                     // sub 0x80
#endif
  }

  static SIMD_INLINE __m128i
  x_mm_max_epu16(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_max_epu16(a, b);
#else
    // from Agner Fog's VCL vectori128.h
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a,signbit);            // add 0x8000
    __m128i b1      = _mm_xor_si128(b,signbit);            // add 0x8000
    __m128i m1      = _mm_max_epi16(a1,b1);                // signed max
    return  _mm_xor_si128(m1,signbit);                     // sub 0x8000
#endif
  }
  
  static SIMD_INLINE __m128i
  x_mm_max_epi32(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_max_epi32(a, b);
#else
    // from Agner Fog's VCL vectori128.h
    __m128i gt = _mm_cmpgt_epi32(a, b);
    return _mm_or_si128(_mm_and_si128(gt, a), _mm_andnot_si128(gt, b));
#endif
  }
  
  // ---------------------------------------------------------------------------
  // packus
  // ---------------------------------------------------------------------------

  static SIMD_INLINE __m128i
  x_mm_packus_epi32(__m128i a, __m128i b)
  {
#ifdef __SSE4_1__
    return _mm_packus_epi32(a, b);
#else
    // mask for lower 16 bit
    __m128i mask = _mm_set1_epi32(0x0000ffff);
    // a >= 0 ? asat = a : asat = 0
    // 23. Nov 17 (rm): 32->31
    __m128i asat = _mm_andnot_si128(_mm_srai_epi32(a, 31), a);
    // cmp/or is used to restrict number to 16 bit
    // srai/slli is used for sign extension of 16 bit number,
    // makes signed saturation (in packs) a no-op, see
    // http://stackoverflow.com/questions/12118910/
    // converting-float-vector-to-16-bit-int-without-saturating
    // e.g.
    // a = 0xffffffff (-1)  -> asat  = 0x00000000 
    //                      -> cmpgt = 0x00000000
    //                      -> slli  = 0x00000000
    //                      -> or    = 0x00000000
    //                      -> srai  = 0x00000000
    //                      -> packs = 0x0000
    // a = 0x7fffffff (>=0) -> asat  = 0x7fffffff
    //                      -> cmpgt = 0xffffffff
    //                      -> slli  = 0xffff0000
    //                      -> or    = 0xffffffff
    //                      -> srai  = 0xffffffff
    //                      -> packs = 0xffff
    // a = 0x0000ffff (>=0) -> asat  = 0x0000ffff
    //                      -> cmpgt = 0x00000000
    //                      -> slli  = 0xffff0000
    //                      -> or    = 0xffff0000
    //                      -> srai  = 0xffffffff
    //                      -> packs = 0xffff
    // a = 0x0000fffe (>=0) -> asat  = 0x0000fffe
    //                      -> cmpgt = 0x00000000
    //                      -> slli  = 0xfffe0000
    //                      -> or    = 0xfffe0000
    //                      -> srai  = 0xfffffffe
    //                      -> packs = 0xfffe
    // a = 0x00007fff (>=0) -> asat  = 0x00007fff
    //                      -> cmpgt = 0x00000000
    //                      -> slli  = 0x7fff0000
    //                      -> or    = 0x7fff0000
    //                      -> srai  = 0x00007fff
    //                      -> packs = 0x7fff
    asat = _mm_srai_epi32(_mm_or_si128(_mm_slli_epi32(asat, 16),
				       _mm_cmpgt_epi32(asat, mask)), 16);
    // same for b
    // 23. Nov 17 (rm): 32->31
    __m128i bsat = _mm_andnot_si128(_mm_srai_epi32(b, 31), b);
    bsat = _mm_srai_epi32(_mm_or_si128(_mm_slli_epi32(bsat, 16),
				       _mm_cmpgt_epi32(bsat, mask)), 16);
    return _mm_packs_epi32(asat, bsat);
#endif
  }

  // ---------------------------------------------------------------------------
  // byte-wise shifts
  // ---------------------------------------------------------------------------

  // positive and in range: apply shift
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_srli_si128(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_srli_si128(a, IMM);
  }
  
  // positive and out of range: return zero vector
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_srli_si128(__m128i, IsPosInRange<true, false>)
  {
    return _mm_setzero_si128();
  }
 
  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_si128(__m128i a)
  {
    return x_mm_srli_si128<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // positive and in range: apply shift
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_slli_si128(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_slli_si128(a, IMM);
  }
  
  // positive and out of range: return zero vector
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_slli_si128(__m128i, IsPosInRange<true, false>)
  {
    return _mm_setzero_si128();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_si128(__m128i a)
  {
    return x_mm_slli_si128<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // ---------------------------------------------------------------------------
  // element-wise shifts
  // ---------------------------------------------------------------------------
  
  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_epi16(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_slli_epi16(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_epi16(__m128i, IsPosInRange<true, false>)
  {
    return _mm_setzero_si128();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_epi16(__m128i a)
  {
    return x_mm_slli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_epi16(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_srli_epi16(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_epi16(__m128i, IsPosInRange<true, false>)
  {
    return _mm_setzero_si128();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_epi16(__m128i a)
  {
    return x_mm_srli_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_epi32(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_slli_epi32(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_epi32(__m128i, IsPosInRange<true, false>)
  {
    return _mm_setzero_si128();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_slli_epi32(__m128i a)
  {
    return x_mm_slli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_epi32(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_srli_epi32(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_epi32(__m128i, IsPosInRange<true, false>)
  {
    return _mm_setzero_si128();
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srli_epi32(__m128i a)
  {
    return x_mm_srli_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // 16. Oct 22 (Jonas Keller): added x_mm_srai_epi8

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi8(__m128i a, IsPosInRange<true, true>)
  {
    __m128i odd = _mm_srai_epi16(a, IMM);
    __m128i even = _mm_srai_epi16(_mm_slli_epi16(a, 8), IMM + 8);
    __m128i odd_masked = _mm_and_si128(odd, _mm_set1_epi16((int16_t)0xFF00));
    __m128i even_masked = _mm_and_si128(even, _mm_set1_epi16(0x00FF));
    return _mm_or_si128(odd_masked, even_masked);
  }

  // positive and out of range: maximal shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi8(__m128i a, IsPosInRange<true, false>)
  {
    // result should be all ones if a is negative, all zeros otherwise
    return _mm_cmplt_epi8(a, _mm_setzero_si128());
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi8(__m128i a)
  {
    return x_mm_srai_epi8<IMM>(a, IsPosInGivenRange<7, IMM>());
  }

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi16(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_srai_epi16(a, IMM);
  }

  // positive and out of range: maximal shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi16(__m128i a, IsPosInRange<true, false>)
  {
    return _mm_srai_epi16(a, 15);
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi16(__m128i a)
  {
    return x_mm_srai_epi16<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // positive and in range: shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi32(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_srai_epi32(a, IMM);
  }

  // positive and out of range: maximal shift
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi32(__m128i a, IsPosInRange<true, false>)
  {
    return _mm_srai_epi32(a, 31);
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_srai_epi32(__m128i a)
  {
    return x_mm_srai_epi32<IMM>(a, IsPosInGivenRange<32, IMM>());
  }

  // ---------------------------------------------------------------------------
  // element-wise shift where count does not need to be constant
  // ---------------------------------------------------------------------------

  // 19. Dec 22 (Jonas Keller): added x_mm_sra_epi8

  static SIMD_INLINE __m128i
  x_mm_sra_epi8(__m128i a, uint8_t count)
  {
    // there is no _mm_sra_epi8 intrinsic
    if (count >= 8) {
      // result should be all ones if a is negative, all zeros otherwise
      return _mm_cmplt_epi8(a, _mm_setzero_si128());
    }
    __m128i odd = _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
    __m128i even =
        _mm_sra_epi16(_mm_slli_epi16(a, 8), _mm_cvtsi32_si128(count + 8));
    __m128i odd_masked = _mm_and_si128(odd, _mm_set1_epi16((int16_t)0xFF00));
    __m128i even_masked = _mm_and_si128(even, _mm_set1_epi16(0x00FF));
    return _mm_or_si128(odd_masked, even_masked);
  }

  // ---------------------------------------------------------------------------
  // extract
  // ---------------------------------------------------------------------------

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi16(__m128i a, IsPosInRange<true, true>)
  {
    return _mm_extract_epi16(a, IMM);
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi16(__m128i, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi16(__m128i a)
  {
    // 8 * epi16
    return x_mm_extract_epi16<IMM>(a, IsPosInGivenRange<8, IMM>());
  }

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi8(__m128i a, IsPosInRange<true, true>)
  {
#ifdef __SSE4_1__
    return _mm_extract_epi8(a, IMM);
#else
    return (((IMM & 0x1) == 0) ?
            x_mm_extract_epi16<(IMM>>1)>(a) & 0xff :
	    x_mm_extract_epi16<(IMM>>1)>(_mm_srli_epi16(a, 8)));
#endif
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi8(__m128i, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi8(__m128i a)
  {
    // 16 * epi8
    return x_mm_extract_epi8<IMM>(a, IsPosInGivenRange<16, IMM>());
  }

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi32(__m128i a, IsPosInRange<true, true>)
  {
    // TODO: extract: is conversion from return type int to SIMDInt always safe?
#ifdef __SSE4_1__
    return _mm_extract_epi32(a, IMM);
#else
    // n << 2 = 4 * n
    return _mm_cvtsi128_si32(x_mm_srli_si128<(IMM<<2)>(a));
#endif
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi32(__m128i, IsPosInRange<true, false>)
  {
    return 0;
  }

  // hub
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_epi32(__m128i a)
  {
    // 4 * epi32
    return x_mm_extract_epi32<IMM>(a, IsPosInGivenRange<4, IMM>());
  }

  // positive and in range: extract
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_ps(__m128 a, IsPosInRange<true, true>)
  {  
#ifdef __SSE4_1__
    return _mm_extract_ps(a, IMM);
#else
    // IMM << 2 = 4 * IMM
    return _mm_cvtsi128_si32(x_mm_srli_si128<(IMM<<2)>(_mm_castps_si128(a)));
#endif
  }

  // positive and out of range: return zero
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_ps(__m128, IsPosInRange<true, false>)
  {  
    return 0;
  }

  // hub: yes, really returns an int!
  template <int IMM>
  static SIMD_INLINE int
  x_mm_extract_ps(__m128 a)
  {  
    // 4 * ps
    return x_mm_extract_ps<IMM>(a, IsPosInGivenRange<4, IMM>());
  }

  // ---------------------------------------------------------------------------
  // alignr
  // ---------------------------------------------------------------------------

  // positive, zero (not non-zero), and in range: return l
  // (this case was introduced for swizzle functions)
  template <int IMM>
  static SIMD_INLINE __m128i
  x_mm_alignr_epi8(__m128i, __m128i l, 
		   IsPosNonZeroInRange<true, false, true>)
  {
    return l;
  }
  
  // positive, non-zero, and in range: run align
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_alignr_epi8(__m128i h, __m128i l, 
		   IsPosNonZeroInRange<true, true, true>)
  {
    return _mm_alignr_epi8(h, l, IMM);	
  }

  // positive, non-zero and out of range: return zero vector
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_alignr_epi8(__m128i, __m128i, 
		   IsPosNonZeroInRange<true, true, false>)
  {
    return _mm_setzero_si128();	
  }

  // hub
  template <int IMM>
  static SIMD_INLINE __m128i 
  x_mm_alignr_epi8(__m128i h, __m128i l)
  {
    // IMM < 2 * 16
    return x_mm_alignr_epi8<IMM>(h, l, 
				 IsPosNonZeroInGivenRange<32, IMM>());	
  }

  // ---------------------------------------------------------------------------
  // not (missing from instruction set)
  // ---------------------------------------------------------------------------

  // from Agner Fog's VCL vectori128.h operator ~
  static SIMD_INLINE __m128i 
  x_mm_not_si128(__m128i a)
  {
    return _mm_xor_si128(a, _mm_set1_epi32(-1));
  }

  // from Agner Fog's VCL vectorf128.h operator ~
  static SIMD_INLINE __m128 
  x_mm_not_ps(__m128 a)
  {
    return _mm_xor_ps(a, _mm_castsi128_ps(_mm_set1_epi32(-1)));
  }

  // ---------------------------------------------------------------------------
  // abs
  // ---------------------------------------------------------------------------
  
  static SIMD_INLINE __m128
  x_mm_abs_ps(__m128 a)
  {
    // there's no _mm_abs_ps, we have to emulated it:
    // -0.0F is 0x8000000, 0x7fffffff by andnot, sign bit is cleared
    return _mm_andnot_ps(_mm_set1_ps(-0.0F), a);
  }

  // ---------------------------------------------------------------------------
  // test functions
  // ---------------------------------------------------------------------------

#ifdef __SSE4_1__
  static SIMD_INLINE int 
  x_mm_test_all_zeros(__m128i a)
  {
    // 10. Oct 22 (Jonas Keller):
    // replaced unnecessary "_mm_cmpeq_epi8(a, a)" with "a"
    // return _mm_test_all_zeros(a, _mm_cmpeq_epi8(a, a));
    return _mm_test_all_zeros(a, a);
  }
#else
  static SIMD_INLINE int 
  x_mm_test_all_zeros(__m128i a)
  {
    return 
      (_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_setzero_si128(), a)) == 0xffff);
  }
#endif

#ifdef __SSE4_1__
  static SIMD_INLINE int 
  x_mm_test_all_ones(__m128i a)
  {
    return _mm_test_all_ones(a);
  }
#else
  static SIMD_INLINE int 
  x_mm_test_all_ones(__m128i a)
  {
    __m128i zero = _mm_setzero_si128();
    __m128i ones = _mm_cmpeq_epi8(zero, zero);
    return _mm_movemask_epi8(_mm_cmpeq_epi8(ones, a)) == 0xffff;
  }
#endif

  // ---------------------------------------------------------------------------
  // ifelse
  // ---------------------------------------------------------------------------

#ifdef __SSE4_1__
  static SIMD_INLINE __m128i
  x_mm_ifelsei(__m128i cond, __m128i trueVal, __m128i falseVal)
  {
    return _mm_blendv_epi8(falseVal, trueVal, cond);
  }
  static SIMD_INLINE __m128
  x_mm_ifelsef(__m128 cond, __m128 trueVal, __m128 falseVal)
  {
    return _mm_blendv_ps(falseVal, trueVal, cond);
  }
#else
  static SIMD_INLINE __m128i
  x_mm_ifelsei(__m128i cond, __m128i trueVal, __m128i falseVal)
  {
    return _mm_or_si128(_mm_and_si128(cond, trueVal),
			_mm_andnot_si128(cond, falseVal));
  }
  // emulation of _mm_blendv_ps
  static SIMD_INLINE __m128
  x_mm_ifelsef(__m128 cond, __m128 trueVal, __m128 falseVal)
  {
    return _mm_or_ps(_mm_and_ps(cond, trueVal),
		     _mm_andnot_ps(cond, falseVal));
  }
#endif

  // ###########################################################################
  // ###########################################################################
  // ###########################################################################

  // ===========================================================================
  // SIMDVec function template specialization or overloading for SSE
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // reinterpretation casts
  // ---------------------------------------------------------------------------

  // between all integer types
  template <typename Tdst, typename Tsrc>
  static SIMD_INLINE SIMDVec<Tdst,16>
  reinterpret(const SIMDVec<Tsrc,16>& vec, OutputType<Tdst>)
  {
    // 26. Nov 22 (Jonas Keller): reinterpret_cast is technically undefined
    // behavior, so just rewrapping the vector register in a new SIMDVec instead
    //return reinterpret_cast<const SIMDVec<Tdst,16>&>(vec);
    return SIMDVec<Tdst,16>(vec.xmm);
  }

  // from float to any integer type
  template <typename Tdst>
  static SIMD_INLINE SIMDVec<Tdst,16>
  reinterpret(const SIMDVec<SIMDFloat,16>& vec, OutputType<Tdst>)
  {
    return _mm_castps_si128(vec);
  }

  // from any integer type to float
  template <typename Tsrc>
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  reinterpret(const SIMDVec<Tsrc,16>& vec, OutputType<SIMDFloat>)
  {
    return _mm_castsi128_ps(vec);
  }

  /*
  // between identical types
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  reinterpret(const SIMDVec<T,16>& vec, OutputType<T>)
  {
    return vec;
  }
  */

  // BUGFIX:  1. Aug 17 (rm): remove code above, added this one
  // between float and float
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  reinterpret(const SIMDVec<SIMDFloat,16>& vec, OutputType<SIMDFloat>)
  {
    return vec;
  }

  // ---------------------------------------------------------------------------
  // convert (without changes in the number of of elements)
  // ---------------------------------------------------------------------------

  // conversion with saturation; we wanted to have a fast solution that
  // doesn't trigger the overflow which results in a negative two's
  // complement result ("invalid int32": 0x80000000); therefore we clamp
  // the positive values at the maximal positive float which is
  // convertible to int32 without overflow (0x7fffffbf = 2147483520);
  // negative values cannot overflow (they are clamped to invalid int
  // which is the most negative int32)
  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  cvts(const SIMDVec<SIMDFloat,16> &a, OutputType<SIMDInt>)
  {
    // TODO: analyze much more complex solution for cvts at
    // TODO: http://stackoverflow.com/questions/9157373/
    // TODO: most-efficient-way-to-convert-vector-of-float-to-vector-of-uint32
    // NOTE: float->int: rounding is affected by MXCSR rounding control bits!
    __m128 clip = _mm_set1_ps(MAX_POS_FLOAT_CONVERTIBLE_TO_INT32);
    return _mm_cvtps_epi32(_mm_min_ps(clip, a));
  }

  // saturation is not necessary in this case
  static SIMD_INLINE SIMDVec<SIMDFloat,16> 
  cvts(const SIMDVec<SIMDInt,16> &a, OutputType<SIMDFloat>)
  {
    return _mm_cvtepi32_ps(a);
  }

  // ---------------------------------------------------------------------------
  // setzero
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  setzero(OutputType<SIMDByte>, Integer<16>)
  { 
    return _mm_setzero_si128(); 
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  setzero(OutputType<SIMDSignedByte>, Integer<16>)
  {
    return _mm_setzero_si128();
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  setzero(OutputType<SIMDWord>, Integer<16>)
  {
    return _mm_setzero_si128();
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  setzero(OutputType<SIMDShort>, Integer<16>)
  {
    return _mm_setzero_si128();
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  setzero(OutputType<SIMDInt>, Integer<16>)
  {
    return _mm_setzero_si128();
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  setzero(OutputType<SIMDFloat>, Integer<16>)
  {
    return _mm_setzero_ps();
  }

  // ---------------------------------------------------------------------------
  // set1
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  set1(SIMDByte a, Integer<16>)
  {
    return _mm_set1_epi8(a);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  set1(SIMDSignedByte a, Integer<16>)
  {
    return _mm_set1_epi8(a);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  set1(SIMDWord a, Integer<16>)
  {
    return _mm_set1_epi16(a);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  set1(SIMDShort a, Integer<16>)
  {
    return _mm_set1_epi16(a);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  set1(SIMDInt a, Integer<16>)
  {
    return _mm_set1_epi32(a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  set1(SIMDFloat a, Integer<16>)
  {
    return _mm_set1_ps(a);
  }

  // ---------------------------------------------------------------------------
  // load
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  load(const SIMDByte *const p, Integer<16>)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    return _mm_load_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  load(const SIMDSignedByte *const p, Integer<16>)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    return _mm_load_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  load(const SIMDWord *const p, Integer<16>)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    return _mm_load_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  load(const SIMDShort *const p, Integer<16>)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    return _mm_load_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  load(const SIMDInt *const p, Integer<16>)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    return _mm_load_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  load(const SIMDFloat *const p, Integer<16>)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    return _mm_load_ps(p);
  }

  // ---------------------------------------------------------------------------
  // loadu
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  loadu(const SIMDByte *const p, Integer<16>)
  {
    return _mm_loadu_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  loadu(const SIMDSignedByte *const p, Integer<16>)
  {
    return _mm_loadu_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  loadu(const SIMDWord *const p, Integer<16>)
  {
    return _mm_loadu_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  loadu(const SIMDShort *const p, Integer<16>)
  {
    return _mm_loadu_si128((__m128i*) p);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  loadu(const SIMDInt *const p, Integer<16>)
  {
    return _mm_loadu_si128((__m128i*) p);
  }
 
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  loadu(const SIMDFloat *const p, Integer<16>)
  {
    return _mm_loadu_ps(p);
  }

  // ---------------------------------------------------------------------------
  // store
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE void 
  store(T *const p, const SIMDVec<T,16> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    _mm_store_si128((__m128i*) p, a);
  }

  // float version
  static SIMD_INLINE void
  store(SIMDFloat *const p, const SIMDVec<SIMDFloat,16> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    _mm_store_ps(p, a);
  }

  // ---------------------------------------------------------------------------
  // storeu
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE void 
  storeu(T *const p, const SIMDVec<T,16> &a)
  {
    _mm_storeu_si128((__m128i*) p, a);
  }

  // float version
  static SIMD_INLINE void
  storeu(SIMDFloat *const p, const SIMDVec<SIMDFloat,16> &a)
  {
    _mm_storeu_ps(p, a);
  }

  // ---------------------------------------------------------------------------
  // stream_store
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE void
  stream_store(T *const p, const SIMDVec<T,16> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    _mm_stream_si128((__m128i*) p, a);
  }

  // float version
  static SIMD_INLINE void
  stream_store(SIMDFloat *const p, const SIMDVec<SIMDFloat,16> &a)
  {
#ifdef SIMD_ALIGN_CHK
    // SSE load and store instructions need alignment to 16 byte
    // (lower 4 bit need to be zero)
    assert((((uintptr_t) p) & 0xf) == 0);
#endif
    _mm_stream_ps(p, a);
  }

  // ---------------------------------------------------------------------------
  // fences (defined only here and not in SIMDVec32.H)
  // ---------------------------------------------------------------------------

  static SIMD_INLINE void
  lfence()
  {
    _mm_lfence();
  }

  static SIMD_INLINE void
  sfence()
  {
    _mm_sfence();
  }

  static SIMD_INLINE void
  mfence()
  {
    _mm_mfence();
  }

  // ---------------------------------------------------------------------------
  // extract: with template parameter for immediate argument
  // ---------------------------------------------------------------------------

  template <int IMM>
  static SIMD_INLINE SIMDByte
  extract(const SIMDVec<SIMDByte,16> &a)
  {
    return x_mm_extract_epi8<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDSignedByte
  extract(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return x_mm_extract_epi8<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDWord
  extract(const SIMDVec<SIMDWord,16> &a)
  {
    return x_mm_extract_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDShort
  extract(const SIMDVec<SIMDShort,16> &a)
  {
    return x_mm_extract_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDInt
  extract(const SIMDVec<SIMDInt,16> &a)
  {
    return x_mm_extract_epi32<IMM>(a);
  }
    
  template <int IMM>
  static SIMD_INLINE SIMDFloat
  extract(const SIMDVec<SIMDFloat,16> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDFloat,x_mm_extract_ps<IMM>(a),0)
  }

  // ---------------------------------------------------------------------------
  // add
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  add(const SIMDVec<SIMDByte,16> &a,
      const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_add_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  add(const SIMDVec<SIMDSignedByte,16> &a,
      const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_add_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  add(const SIMDVec<SIMDWord,16> &a,
      const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_add_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  add(const SIMDVec<SIMDShort,16> &a,
      const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_add_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  add(const SIMDVec<SIMDInt,16> &a,
      const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_add_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  add(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_add_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // adds (integer: signed, unsigned; 8/16 only, 32 without saturation)
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  adds(const SIMDVec<SIMDByte,16> &a,
       const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_adds_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  adds(const SIMDVec<SIMDSignedByte,16> &a,
       const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_adds_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  adds(const SIMDVec<SIMDWord,16> &a,
       const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_adds_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  adds(const SIMDVec<SIMDShort,16> &a,
       const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_adds_epi16(a, b);
  }

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  adds(const SIMDVec<SIMDInt,16> &a,
       const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_add_epi32(a, b);
  }

  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  adds(const SIMDVec<SIMDFloat,16> &a,
       const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_add_ps(a, b);
  }
#endif

  // ---------------------------------------------------------------------------
  // sub
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  sub(const SIMDVec<SIMDByte,16> &a,
      const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_sub_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  sub(const SIMDVec<SIMDSignedByte,16> &a,
      const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_sub_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  sub(const SIMDVec<SIMDWord,16> &a,
      const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_sub_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  sub(const SIMDVec<SIMDShort,16> &a,
      const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_sub_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  sub(const SIMDVec<SIMDInt,16> &a,
      const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_sub_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  sub(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_sub_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // subs (integer: signed, unsigned; 8/16 only, 32 without saturation)
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  subs(const SIMDVec<SIMDByte,16> &a,
       const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_subs_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  subs(const SIMDVec<SIMDSignedByte,16> &a,
       const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_subs_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  subs(const SIMDVec<SIMDWord,16> &a,
       const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_subs_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  subs(const SIMDVec<SIMDShort,16> &a,
       const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_subs_epi16(a, b);
  }

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  subs(const SIMDVec<SIMDInt,16> &a,
       const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_sub_epi32(a, b);
  }

  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  subs(const SIMDVec<SIMDFloat,16> &a,
       const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_sub_ps(a, b);
  }
#endif

  // ---------------------------------------------------------------------------
  // neg (negate = two's complement or unary minus), only signed types
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  neg(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return _mm_sub_epi8(_mm_setzero_si128(), a);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  neg(const SIMDVec<SIMDShort,16> &a)
  {
    return _mm_sub_epi16(_mm_setzero_si128(), a);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  neg(const SIMDVec<SIMDInt,16> &a)
  {
    return _mm_sub_epi32(_mm_setzero_si128(), a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  neg(const SIMDVec<SIMDFloat,16> &a)
  {
    return _mm_sub_ps(_mm_setzero_ps(), a);
  }

  // ---------------------------------------------------------------------------
  // min
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  min(const SIMDVec<SIMDByte,16> &a,
      const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_min_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  min(const SIMDVec<SIMDSignedByte,16> &a,
      const SIMDVec<SIMDSignedByte,16> &b)
  {
    return x_mm_min_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  min(const SIMDVec<SIMDWord,16> &a,
      const SIMDVec<SIMDWord,16> &b)
  {
    return x_mm_min_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  min(const SIMDVec<SIMDShort,16> &a,
      const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_min_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  min(const SIMDVec<SIMDInt,16> &a,
      const SIMDVec<SIMDInt,16> &b)
  {
    return x_mm_min_epi32(a, b);
  }

  // there is an unsigned version of min for 32 bit but we currently
  // don't have an element type for it

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  min(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_min_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // max
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  max(const SIMDVec<SIMDByte,16> &a,
      const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_max_epu8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  max(const SIMDVec<SIMDSignedByte,16> &a,
      const SIMDVec<SIMDSignedByte,16> &b)
  {
    return x_mm_max_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  max(const SIMDVec<SIMDWord,16> &a,
      const SIMDVec<SIMDWord,16> &b)
  {
    return x_mm_max_epu16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  max(const SIMDVec<SIMDShort,16> &a,
      const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_max_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  max(const SIMDVec<SIMDInt,16> &a,
      const SIMDVec<SIMDInt,16> &b)
  {
    return x_mm_max_epi32(a, b);
  }

  // there is an unsigned version of max for 32 bit but we currently
  // don't have an element type for it

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  max(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_max_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // mul, div
  // ---------------------------------------------------------------------------

  // TODO: add mul/div versions for int types? or make special versions of mul
  // TODO: and div where the result is scaled?

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  mul(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_mul_ps(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  div(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_div_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // ceil, floor, round, truncate
  // ---------------------------------------------------------------------------

  // TODO: add versions of ceil/floor/round for int types?
  // TODO: import complex workaround for non-SSE4.1 from Agner Fog's VCL?

  // NOTE: behavior for workarounds differs for results of -0.0f and +0.0f

  // work-arounds for round, truncate, floor, and ceil all check whether
  // rounding is necessary (or whether float is an integer anyhow), this also
  // prevents range excess when converting numbers to integer

  // workarounds for floor and ceil:
  // https://en.wikipedia.org/wiki/Floor_and_ceiling_functions
  //
  // floor, ceil:
  //                 floor(x), x >= 0
  // truncate(x) = {
  //                 ceil(x), x < 0
  // 
  // floor(x) = ceil(x)  - (x in Z ? 0 : 1)
  // ceil(x)  = floor(x) + (x in Z ? 0 : 1)

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  ceil(const SIMDVec<SIMDFloat,16> &a)
  {
#ifdef __SSE4_1__
    return _mm_ceil_ps(a);
#else
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    __m128 limit = _mm_set1_ps(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
    // bool mask: true if a is negative
    __m128 isNeg = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), 31));
    // truncated result (for |a| < limit)
    __m128 aTrunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
    // check if a is an integer
    __m128 isNotInt = _mm_cmpneq_ps(a, aTrunc);
    // constant 1.0
    __m128 one = _mm_set1_ps(1.0f);
    // mask which is 1.0f for non-negative non-integer values, 0.0f otherwise
    __m128 oneMask = _mm_and_ps(_mm_andnot_ps(isNeg, isNotInt), one);
    // if non-negative, trunc computes floor, to turn it into ceil we
    // add 1 if aTrunc is non-integer
    aTrunc = _mm_add_ps(aTrunc, oneMask);
    // select result (a or aTrunc)
    return x_mm_ifelsef(noRndReq, a, aTrunc);
#endif
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  floor(const SIMDVec<SIMDFloat,16> &a)
  {
#ifdef __SSE4_1__
    return _mm_floor_ps(a);
#else
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    __m128 limit = _mm_set1_ps(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
    // bool mask: true if a is negative
    __m128 isNeg = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), 31));
    // truncated result (for |a| < limit)
    __m128 aTrunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
    // check if a is an integer
    __m128 isNotInt = _mm_cmpneq_ps(a, aTrunc);
    // constant 1.0
    __m128 one = _mm_set1_ps(1.0f);
    // mask which is 1.0f for negative non-integer values, 0.0f otherwise
    __m128 oneMask = _mm_and_ps(_mm_and_ps(isNeg, isNotInt), one);
    // if negative, trunc computes ceil, to turn it into floor we sub
    // 1 if aTrunc is non-integer
    aTrunc = _mm_sub_ps(aTrunc, oneMask);
    // select result (a or aTrunc)
    return x_mm_ifelsef(noRndReq, a, aTrunc);
#endif
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  round(const SIMDVec<SIMDFloat,16> &a)
  {
#ifdef __SSE4_1__
    // old: use _MM_SET_ROUNDING_MODE to adjust rounding direction
    // return _mm_round_ps(a, _MM_FROUND_CUR_DIRECTION);
    // new  4. Aug 16 (rm): round to nearest, and suppress exceptions
    return _mm_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#else
    // NOTE: only works if rounding mode is default (rnd. to nearest (even))
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    __m128 limit = _mm_set1_ps(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
    // rounded result (here rounded according to current rounding mode)
    // (for |a| < limit)
    __m128 aRnd = _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
    // select result
    return x_mm_ifelsef(noRndReq, a, aRnd);
#endif
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  truncate(const SIMDVec<SIMDFloat,16> &a)
  {
#ifdef __SSE4_1__
    return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    __m128 limit = _mm_set1_ps(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    __m128 noRndReq = _mm_cmpge_ps(x_mm_abs_ps(a), limit);
    // truncated result (for |a| < limit) (cvtTps!)
    __m128 aTrunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(a));
    // select result
    return x_mm_ifelsef(noRndReq, a, aTrunc);
#endif
  }

  // ---------------------------------------------------------------------------
  // elementary mathematical functions
  // ---------------------------------------------------------------------------

  // estimate of a reciprocal
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  rcp(const SIMDVec<SIMDFloat,16> &a)
  {
    return _mm_rcp_ps(a);
  }

  // estimate of a reverse square root
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  rsqrt(const SIMDVec<SIMDFloat,16> &a)
  {
    return _mm_rsqrt_ps(a);
  }

  // square root
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  sqrt(const SIMDVec<SIMDFloat,16> &a)
  {
    return _mm_sqrt_ps(a);
  }

  // ---------------------------------------------------------------------------
  // abs (integer: signed only)
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  abs(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return _mm_abs_epi8(a);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  abs(const SIMDVec<SIMDShort,16> &a)
  {
    return _mm_abs_epi16(a);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  abs(const SIMDVec<SIMDInt,16> &a)
  {
    return _mm_abs_epi32(a);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  abs(const SIMDVec<SIMDFloat,16> &a)
  {
    return x_mm_abs_ps(a);
  }

  // ---------------------------------------------------------------------------
  // unpacklo
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<0>,
	 Bytes<1>)
  {
    return _mm_unpacklo_epi8(a, b);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<0>,
	 Bytes<2>)
  {
    return _mm_unpacklo_epi16(a, b);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<0>,
	 Bytes<4>)
  {
    return _mm_unpacklo_epi32(a, b);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<0>,
	 Bytes<8>)
  {
    return _mm_unpacklo_epi64(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  unpack(const SIMDVec<SIMDFloat,16> &a,
	 const SIMDVec<SIMDFloat,16> &b,
	 Part<0>,
	 Bytes<4>)
  {
    return _mm_unpacklo_ps(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  unpack(const SIMDVec<SIMDFloat,16> &a,
	 const SIMDVec<SIMDFloat,16> &b,
	 Part<0>,
	 Bytes<8>)
  {
    // this moves two lower floats from a and b
    return _mm_movelh_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // unpackhi
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<1>,
	 Bytes<1>)
  {
    return _mm_unpackhi_epi8(a, b);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<1>,
	 Bytes<2>)
  {
    return _mm_unpackhi_epi16(a, b);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<1>,
	 Bytes<4>)
  {
    return _mm_unpackhi_epi32(a, b);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack(const SIMDVec<T, 16> &a,
	 const SIMDVec<T, 16> &b,
	 Part<1>,
	 Bytes<8>)
  {
    return _mm_unpackhi_epi64(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  unpack(const SIMDVec<SIMDFloat,16> &a,
	 const SIMDVec<SIMDFloat,16> &b,
	 Part<1>,
	 Bytes<4>)
  {
    return _mm_unpackhi_ps(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  unpack(const SIMDVec<SIMDFloat,16> &a,
	 const SIMDVec<SIMDFloat,16> &b,
	 Part<1>,
	 Bytes<8>)
  {
    // this moves two upper floats from a and b
    // order b, a
    return _mm_movehl_ps(b, a);
  }

  // contributed by Adam Marschall

  // 16-byte-lane oriented unpack: for 16 bytes same as generalized unpack
  // unpack blocks of NUM_ELEMS elements of type T
  // PART=0: low half of input vectors,
  // PART=1: high half of input vectors
  template <int PART, int NUM_ELEMS, typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  unpack16(const SIMDVec<T, 16> &a,
	   const SIMDVec<T, 16> &b)
  {
    return unpack(a, b, Part<PART>(), Bytes<NUM_ELEMS * sizeof(T)>());
  }

  // ---------------------------------------------------------------------------
  // extract 128-bit lane as SIMDVec<T, 16>, does nothing for 16 bytes
  // ---------------------------------------------------------------------------

  // contributed by Adam Marschall

  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  extractLane(const SIMDVec<T, 16> &a)
  {
    return a;
  }

  // ---------------------------------------------------------------------------
  // zip (two unpacks similar to ARM NEON vzip, but for different NUM_ELEMS)
  // ---------------------------------------------------------------------------

  // a, b are passed by-value to avoid problems with identical input/output args.

  // here we can directly map zip to unpack<PART,NUM_ELEMS,T>
  template <int NUM_ELEMS, typename T>
  static SIMD_INLINE void
  zip(const SIMDVec<T, 16> a,
      const SIMDVec<T, 16> b,
      SIMDVec<T, 16> &l,
      SIMDVec<T, 16> &h)
  {
    l = unpack<0, NUM_ELEMS>(a, b);
    h = unpack<1, NUM_ELEMS>(a, b);
  }

  // ---------------------------------------------------------------------------
  // zip16 hub  (16-byte-lane oriented zip): for 16 bytes same as zip
  // ---------------------------------------------------------------------------

  // contributed by Adam Marschall

  // a, b are passed by-value to avoid problems with identical
  // input/output args.

  template <int NUM_ELEMS, typename T>
  static SIMD_INLINE void
  zip16(const SIMDVec<T, 16> a,
        const SIMDVec<T, 16> b,
        SIMDVec<T, 16> &l,
        SIMDVec<T, 16> &h)
  {
    zip<NUM_ELEMS, T>(a, b, l, h);
  }

  // ---------------------------------------------------------------------------
  // unzip (similar to ARM NEON vuzp, but for different NUM_ELEMS)
  // ---------------------------------------------------------------------------
  
  // solutions by Peter Cordes and Starvin Marvin:
  // stackoverflow.com/q/45376193/3852630 and 
  // stackoverflow.com/a/45385216/3852630 and
  // stackoverflow.com/q/20504618/3852630

  // all integer versions
  template <typename T>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, 16> a,
	const SIMDVec<T, 16> b,
	SIMDVec<T, 16> &l,
	SIMDVec<T, 16> &h,
	Bytes<1>)
  {
    // mask is hopefully only set once if unzip is used multiple times
    __m128i mask = _mm_set_epi8(15,13,11,9,7,5,3,1,
				14,12,10,8,6,4,2,0);
    __m128i atmp = _mm_shuffle_epi8(a, mask);
    __m128i btmp = _mm_shuffle_epi8(b, mask);
    l = _mm_unpacklo_epi64(atmp, btmp);
    h = _mm_unpackhi_epi64(atmp, btmp);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, 16> a,
	const SIMDVec<T, 16> b,
	SIMDVec<T, 16> &l,
	SIMDVec<T, 16> &h,
	Bytes<2>)
  {
    // mask is hopefully only set once if unzip is used multiple times
    __m128i mask = _mm_set_epi8(15,14,11,10,7,6,3,2,
				13,12,9,8,5,4,1,0);
    __m128i atmp = _mm_shuffle_epi8(a, mask);
    __m128i btmp = _mm_shuffle_epi8(b, mask);
    l = _mm_unpacklo_epi64(atmp, btmp);
    h = _mm_unpackhi_epi64(atmp, btmp);
  }

  // all integer versions
  template <typename T>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, 16> a,
	const SIMDVec<T, 16> b,
	SIMDVec<T, 16> &l,
	SIMDVec<T, 16> &h,
	Bytes<4>)
  {
    __m128 aps = _mm_castsi128_ps(a);
    __m128 bps = _mm_castsi128_ps(b);
    l = _mm_castps_si128(_mm_shuffle_ps(aps, bps, _MM_SHUFFLE(2,0,2,0)));
    h = _mm_castps_si128(_mm_shuffle_ps(aps, bps, _MM_SHUFFLE(3,1,3,1)));
  }

  // all types
  template <typename T>
  static SIMD_INLINE void
  unzip(const SIMDVec<T, 16> a,
	const SIMDVec<T, 16> b,
	SIMDVec<T, 16> &l,
	SIMDVec<T, 16> &h,
	Bytes<8>)
  {
    l = unpack(a, b, Part<0>(), Bytes<8>());
    h = unpack(a, b, Part<1>(), Bytes<8>());
  } 

  // SIMDFloat
  static SIMD_INLINE void
  unzip(const SIMDVec<SIMDFloat, 16> a,
	const SIMDVec<SIMDFloat, 16> b,
	SIMDVec<SIMDFloat, 16> &l,
	SIMDVec<SIMDFloat, 16> &h,
	Bytes<4>)
  {
    l = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));
    h = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3,1,3,1));
  }

  // ---------------------------------------------------------------------------
  // packs
  // ---------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  packs(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b,
  OutputType<SIMDSignedByte>)
  {
    return _mm_packs_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  packs(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b,
  OutputType<SIMDShort>)
  {
    return _mm_packs_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  packs(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b,
  OutputType<SIMDShort>)
  {
    return packs(cvts(a, OutputType<SIMDInt>()),
                 cvts(b, OutputType<SIMDInt>()),
                 OutputType<SIMDShort>());
  }

  // signed -> unsigned

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  packs(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b,
  OutputType<SIMDByte>)
  {
    return _mm_packus_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  packs(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b,
  OutputType<SIMDWord>)
  {
    return x_mm_packus_epi32(a, b);
  }   

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  packs(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b,
  OutputType<SIMDWord>)
  {
    return packs(cvts(a, OutputType<SIMDInt>()),
                 cvts(b, OutputType<SIMDInt>()),
                 OutputType<SIMDWord>());
  }

  // ---------------------------------------------------------------------------
  // generalized extend: no stage
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
  extend(const SIMDVec<T,16> &vIn,
	 SIMDVec<T,16> *const vOut)
  {
    *vOut = vIn;
  }

  // ---------------------------------------------------------------------------
  // generalized extend: single stage
  // ---------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,16> &vIn,
	 SIMDVec<SIMDShort,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi8_epi16(vIn);
    vOut[1] = _mm_cvtepi8_epi16(_mm_srli_si128(vIn, 8));
#else
    vOut[0] = _mm_srai_epi16(_mm_unpacklo_epi8(vIn, vIn), 8);
    vOut[1] = _mm_srai_epi16(_mm_unpackhi_epi8(vIn, vIn), 8);
#endif
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDShort,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi16_epi32(vIn);
    vOut[1] = _mm_cvtepi16_epi32(_mm_srli_si128(vIn, 8));
#else
    vOut[0] = _mm_srai_epi32(_mm_unpacklo_epi16(vIn, vIn), 16);
    vOut[1] = _mm_srai_epi32(_mm_unpackhi_epi16(vIn, vIn), 16);
#endif
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDShort,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(vIn));
    vOut[1] = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_srli_si128(vIn, 8)));
#else
    vOut[0] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(vIn, vIn), 16));
    vOut[1] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(vIn, vIn), 16));
#endif
  }

  // unsigned -> unsigned

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDWord,16> *const vOut)
  {
    // there's no _mm_cvtepu8_epu16()
    __m128i zero = _mm_setzero_si128();
    vOut[0] = _mm_unpacklo_epi8(vIn, zero);
    vOut[1] = _mm_unpackhi_epi8(vIn, zero);
  }

  // unsigned -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDShort,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepu8_epi16(vIn);
    vOut[1] = _mm_cvtepu8_epi16(_mm_srli_si128(vIn, 8));
#else
    __m128i zero = _mm_setzero_si128();
    vOut[0] = _mm_unpacklo_epi8(vIn, zero);
    vOut[1] = _mm_unpackhi_epi8(vIn, zero);
#endif
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDWord,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepu16_epi32(vIn);
    vOut[1] = _mm_cvtepu16_epi32(_mm_srli_si128(vIn, 8));
#else
    __m128i zero = _mm_setzero_si128();
    vOut[0] = _mm_unpacklo_epi16(vIn, zero);
    vOut[1] = _mm_unpackhi_epi16(vIn, zero);
#endif
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDWord,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(vIn));
    vOut[1] = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_srli_si128(vIn, 8)));
#else
    __m128i zero = _mm_setzero_si128();
    vOut[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(vIn, zero));
    vOut[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(vIn, zero));
#endif
  }

  // ---------------------------------------------------------------------------
  // generalized extend: two stages
  // ---------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi8_epi32(vIn);
    vOut[1] = _mm_cvtepi8_epi32(_mm_srli_si128(vIn, 4));
    vOut[2] = _mm_cvtepi8_epi32(_mm_srli_si128(vIn, 8));
    vOut[3] = _mm_cvtepi8_epi32(_mm_srli_si128(vIn, 12));
#else
    __m128i lo8 = _mm_unpacklo_epi8(vIn, vIn);
    vOut[0] = _mm_srai_epi32(_mm_unpacklo_epi16(lo8, lo8), 24);
    vOut[1] = _mm_srai_epi32(_mm_unpackhi_epi16(lo8, lo8), 24);
    __m128i hi8 = _mm_unpackhi_epi8(vIn, vIn);
    vOut[2] = _mm_srai_epi32(_mm_unpacklo_epi16(hi8, hi8), 24);
    vOut[3] = _mm_srai_epi32(_mm_unpackhi_epi16(hi8, hi8), 24);
#endif
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(vIn));
    vOut[1] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 4)));
    vOut[2] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 8)));
    vOut[3] = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(vIn, 12)));
#else
    __m128i lo8 = _mm_unpacklo_epi8(vIn, vIn);
    vOut[0] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(lo8, lo8), 24));
    vOut[1] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(lo8, lo8), 24));
    __m128i hi8 = _mm_unpackhi_epi8(vIn, vIn);
    vOut[2] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(hi8, hi8), 24));
    vOut[3] = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(hi8, hi8), 24));
#endif
  }

  // unsigned -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepu8_epi32(vIn);
    vOut[1] = _mm_cvtepu8_epi32(_mm_srli_si128(vIn, 4));
    vOut[2] = _mm_cvtepu8_epi32(_mm_srli_si128(vIn, 8));
    vOut[3] = _mm_cvtepu8_epi32(_mm_srli_si128(vIn, 12));
#else
    __m128i zero = _mm_setzero_si128();
    __m128i lo8 = _mm_unpacklo_epi8(vIn, zero);
    vOut[0] = _mm_unpacklo_epi16(lo8, zero);
    vOut[1] = _mm_unpackhi_epi16(lo8, zero);
    __m128i hi8 = _mm_unpackhi_epi8(vIn, zero);
    vOut[2] = _mm_unpacklo_epi16(hi8, zero);
    vOut[3] = _mm_unpackhi_epi16(hi8, zero);
#endif
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
#ifdef __SSE4_1__
    vOut[0] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(vIn));
    vOut[1] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 4)));
    vOut[2] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 8)));
    vOut[3] = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(vIn, 12)));
#else
    __m128i zero = _mm_setzero_si128();
    __m128i lo8 = _mm_unpacklo_epi8(vIn, zero); 
    vOut[0] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo8, zero));
    vOut[1] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo8, zero));
    __m128i hi8 = _mm_unpackhi_epi8(vIn, zero);
    vOut[2] = _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi8, zero));
    vOut[3] = _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi8, zero));
#endif
  }

  // ---------------------------------------------------------------------------
  // generalized extend: special case int <-> float
  // ---------------------------------------------------------------------------

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDInt,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
    *vOut = cvts(vIn, OutputType<SIMDFloat>());
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDFloat,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
    *vOut = cvts(vIn, OutputType<SIMDInt>());
  }

  // ---------------------------------------------------------------------------
  // srai
  // ---------------------------------------------------------------------------

  // 16. Oct 22 (Jonas Keller): added missing Byte and SignedByte versions

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  srai(const SIMDVec<SIMDByte,16> &a)
  {
    return x_mm_srai_epi8<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  srai(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return x_mm_srai_epi8<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  srai(const SIMDVec<SIMDWord,16> &a)
  {
    return x_mm_srai_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  srai(const SIMDVec<SIMDShort,16> &a)
  {
    return x_mm_srai_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  srai(const SIMDVec<SIMDInt,16> &a)
  {
    return x_mm_srai_epi32<IMM>(a);
  }

  // ---------------------------------------------------------------------------
  // srli
  // ---------------------------------------------------------------------------

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDByte,16>
  srli(const SIMDVec<SIMDByte,16> &a)
  {
    return _mm_and_si128(_mm_set1_epi8((int8_t)(0xff >> IMM)),
			 x_mm_srli_epi32<IMM>(a));
  }

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  srli(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return _mm_and_si128(_mm_set1_epi8((int8_t)(0xff >> IMM)),
			 x_mm_srli_epi32<IMM>(a));
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  srli(const SIMDVec<SIMDWord,16> &a)
  {
    return x_mm_srli_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  srli(const SIMDVec<SIMDShort,16> &a)
  {
    return x_mm_srli_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  srli(const SIMDVec<SIMDInt,16> &a)
  {
    return x_mm_srli_epi32<IMM>(a);
  }

  // ---------------------------------------------------------------------------
  // slli
  // ---------------------------------------------------------------------------

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDByte,16>
  slli(const SIMDVec<SIMDByte,16> &a)
  {
    return _mm_and_si128
      (_mm_set1_epi8((int8_t)(uint8_t)(0xff & (0xff << IMM))),
       x_mm_slli_epi32<IMM>(a));
  }

  // https://github.com/grumpos/spu_intrin/blob/master/src/sse_extensions.h
  // License: not specified
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  slli(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return _mm_and_si128
      (_mm_set1_epi8((int8_t)(uint8_t)(0xff & (0xff << IMM))),
       x_mm_slli_epi32<IMM>(a));
  }
  
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  slli(const SIMDVec<SIMDWord,16> &a)
  {
    return x_mm_slli_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  slli(const SIMDVec<SIMDShort,16> &a)
  {
    return x_mm_slli_epi16<IMM>(a);
  }

  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  slli(const SIMDVec<SIMDInt,16> &a)
  {
    return x_mm_slli_epi32<IMM>(a);
  }


  // 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

  // ---------------------------------------------------------------------------
  // sra
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  sra(const SIMDVec<SIMDByte,16> &a, const uint8_t count)
  {
    return x_mm_sra_epi8(a, count);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  sra(const SIMDVec<SIMDSignedByte,16> &a, const uint8_t count)
  {
    return x_mm_sra_epi8(a, count);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  sra(const SIMDVec<SIMDWord,16> &a, const uint8_t count)
  {
    return _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  sra(const SIMDVec<SIMDShort,16> &a, const uint8_t count)
  {
    return _mm_sra_epi16(a, _mm_cvtsi32_si128(count));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  sra(const SIMDVec<SIMDInt,16> &a, const uint8_t count)
  {
    return _mm_sra_epi32(a, _mm_cvtsi32_si128(count));
  }

  // ---------------------------------------------------------------------------
  // srl
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  srl(const SIMDVec<SIMDByte,16> &a, const uint8_t count)
  {
    return _mm_and_si128(_mm_srl_epi16(a, _mm_cvtsi32_si128(count)),
                         _mm_set1_epi8((int8_t)(uint8_t)(0xff >> count)));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  srl(const SIMDVec<SIMDSignedByte,16> &a, const uint8_t count)
  {
    return _mm_and_si128(_mm_srl_epi16(a, _mm_cvtsi32_si128(count)),
                         _mm_set1_epi8((int8_t)(uint8_t)(0xff >> count)));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  srl(const SIMDVec<SIMDWord,16> &a, const uint8_t count)
  {
    return _mm_srl_epi16(a, _mm_cvtsi32_si128(count));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  srl(const SIMDVec<SIMDShort,16> &a, const uint8_t count)
  {
    return _mm_srl_epi16(a, _mm_cvtsi32_si128(count));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  srl(const SIMDVec<SIMDInt,16> &a, const uint8_t count)
  {
    return _mm_srl_epi32(a, _mm_cvtsi32_si128(count));
  }

  // ---------------------------------------------------------------------------
  // sll
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  sll(const SIMDVec<SIMDByte,16> &a, const uint8_t count)
  {
    return _mm_and_si128(
        _mm_sll_epi16(a, _mm_cvtsi32_si128(count)),
        _mm_set1_epi8((int8_t)(uint8_t)(0xff & (0xff << count))));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  sll(const SIMDVec<SIMDSignedByte,16> &a, const uint8_t count)
  {
    return _mm_and_si128(
        _mm_sll_epi16(a, _mm_cvtsi32_si128(count)),
        _mm_set1_epi8((int8_t)(uint8_t)(0xff & (0xff << count))));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  sll(const SIMDVec<SIMDWord,16> &a, const uint8_t count)
  {
    return _mm_sll_epi16(a, _mm_cvtsi32_si128(count));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  sll(const SIMDVec<SIMDShort,16> &a, const uint8_t count)
  {
    return _mm_sll_epi16(a, _mm_cvtsi32_si128(count));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  sll(const SIMDVec<SIMDInt,16> &a, const uint8_t count)
  {
    return _mm_sll_epi32(a, _mm_cvtsi32_si128(count));
  }


  // 19. Sep 22 (Jonas Keller):
  // added SIMDByte and SIMDSignedByte versions of hadd, hadds, hsub and hsubs
  // added SIMDWord version of hadds and hsubs

  // ---------------------------------------------------------------------------
  // hadd
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  hadd(const SIMDVec<SIMDByte,16> &a,
       const SIMDVec<SIMDByte,16> &b)
  {
    SIMDVec<SIMDByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDByte)>());
    return add(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  hadd(const SIMDVec<SIMDSignedByte,16> &a,
       const SIMDVec<SIMDSignedByte,16> &b)
  {
    SIMDVec<SIMDSignedByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDSignedByte)>());
    return add(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  hadd(const SIMDVec<SIMDWord,16> &a,
       const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_hadd_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  hadd(const SIMDVec<SIMDShort,16> &a,
       const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_hadd_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  hadd(const SIMDVec<SIMDInt,16> &a,
       const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_hadd_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  hadd(const SIMDVec<SIMDFloat,16> &a,
       const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_hadd_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // hadds
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  hadds(const SIMDVec<SIMDByte,16> &a,
        const SIMDVec<SIMDByte,16> &b)
  {
    SIMDVec<SIMDByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDByte)>());
    return adds(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  hadds(const SIMDVec<SIMDSignedByte,16> &a,
        const SIMDVec<SIMDSignedByte,16> &b)
  {
    SIMDVec<SIMDSignedByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDSignedByte)>());
    return adds(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  hadds(const SIMDVec<SIMDWord,16> &a,
        const SIMDVec<SIMDWord,16> &b)
  {
    SIMDVec<SIMDWord, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDWord)>());
    return adds(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  hadds(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_hadds_epi16(a, b);
  }

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  hadds(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_hadd_epi32(a, b);
  }

  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  hadds(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_hadd_ps(a, b);
  }
#endif

  // ---------------------------------------------------------------------------
  // hsub
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  hsub(const SIMDVec<SIMDByte,16> &a,
       const SIMDVec<SIMDByte,16> &b)
  {
    SIMDVec<SIMDByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDByte)>());
    return sub(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  hsub(const SIMDVec<SIMDSignedByte,16> &a,
       const SIMDVec<SIMDSignedByte,16> &b)
  {
    SIMDVec<SIMDSignedByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDSignedByte)>());
    return sub(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  hsub(const SIMDVec<SIMDWord,16> &a,
       const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_hsub_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  hsub(const SIMDVec<SIMDShort,16> &a,
       const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_hsub_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  hsub(const SIMDVec<SIMDInt,16> &a,
       const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_hsub_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  hsub(const SIMDVec<SIMDFloat,16> &a,
       const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_hsub_ps(a, b);
  }
  
  // ---------------------------------------------------------------------------
  // hsubs
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  hsubs(const SIMDVec<SIMDByte,16> &a,
        const SIMDVec<SIMDByte,16> &b)
  {
    SIMDVec<SIMDByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDByte)>());
    return subs(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  hsubs(const SIMDVec<SIMDSignedByte,16> &a,
        const SIMDVec<SIMDSignedByte,16> &b)
  {
    SIMDVec<SIMDSignedByte, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDSignedByte)>());
    return subs(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  hsubs(const SIMDVec<SIMDWord,16> &a,
        const SIMDVec<SIMDWord,16> &b)
  {
    SIMDVec<SIMDWord, 16> x, y;
    unzip(a, b, x, y, Bytes<sizeof(SIMDWord)>());
    return subs(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  hsubs(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_hsubs_epi16(a, b);
  }

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  hsubs(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_hsub_epi32(a, b);
  }

  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  hsubs(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_hsub_ps(a, b);
  }
#endif

  // ---------------------------------------------------------------------------
  // element-wise shift right
  // ---------------------------------------------------------------------------

  // all integer versions
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,16>
  srle(const SIMDVec<T, 16> &a)
  {
    return x_mm_srli_si128<IMM * sizeof(T)>(a);
  }

  // float version
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  srle(const SIMDVec<SIMDFloat, 16> &a)
  {
    return _mm_castsi128_ps
      (x_mm_srli_si128<IMM * sizeof(SIMDFloat)>(_mm_castps_si128(a)));
  }

  // ---------------------------------------------------------------------------
  // element-wise shift left
  // ---------------------------------------------------------------------------

  // all integer versions
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,16>
  slle(const SIMDVec<T, 16> &a)
  {
    return x_mm_slli_si128<IMM * sizeof(T)>(a);
  }

  // float version
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  slle(const SIMDVec<SIMDFloat, 16> &a)
  {
    return _mm_castsi128_ps
      (x_mm_slli_si128<IMM * sizeof(SIMDFloat)>(_mm_castps_si128(a)));
  }

  // ---------------------------------------------------------------------------
  // extraction of element 0
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDByte
  elem0(const SIMDVec<SIMDByte, 16> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDByte,_mm_cvtsi128_si32(a),0);
  }

  static SIMD_INLINE SIMDSignedByte
  elem0(const SIMDVec<SIMDSignedByte, 16> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDSignedByte,_mm_cvtsi128_si32(a),0);
  }

  static SIMD_INLINE SIMDWord
  elem0(const SIMDVec<SIMDWord, 16> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDWord,_mm_cvtsi128_si32(a),0);
  }

  static SIMD_INLINE SIMDShort
  elem0(const SIMDVec<SIMDShort, 16> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDShort,_mm_cvtsi128_si32(a),0);
  }

  static SIMD_INLINE SIMDInt
  elem0(const SIMDVec<SIMDInt, 16> &a)
  {
    // TODO: elem0: is conversion from return type int so SIMDInt always safe?
    return _mm_cvtsi128_si32(a);
  }

  static SIMD_INLINE SIMDFloat
  elem0(const SIMDVec<SIMDFloat, 16> &a)
  {
    SIMD_RETURN_REINTERPRET_INT(SIMDFloat,
				_mm_cvtsi128_si32(_mm_castps_si128(a)),0);
  }

  // ---------------------------------------------------------------------------
  // alignre
  // ---------------------------------------------------------------------------

  // all integer versions
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  alignre(const SIMDVec<T, 16> &h,
	  const SIMDVec<T, 16> &l)
  {
    return x_mm_alignr_epi8<IMM * sizeof(T)>(h, l);
  }

  // float version
  template <int IMM>
  static SIMD_INLINE SIMDVec<SIMDFloat, 16>
  alignre(const SIMDVec<SIMDFloat, 16> &h,
	  const SIMDVec<SIMDFloat, 16> &l)
  {
    return 
      _mm_castsi128_ps
      (x_mm_alignr_epi8<IMM * sizeof(SIMDFloat)>
       (_mm_castps_si128(h), _mm_castps_si128(l)));
  }

  // ---------------------------------------------------------------------------
  // swizzle
  // ---------------------------------------------------------------------------

  // swizzle masks (only for 8 and 16 bit element types)

  // [masks generated from ~/texte/Talks/SSE/swizzle.c]

  // SIMDByte, SIMDSignedByte
  
  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<2>, Integer<1>)
  {
    return _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
  }
  
  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<3>, Integer<1>)
  {
    return _mm_setr_epi8(0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, -1, -1, -1, -1);
  }

  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<4>, Integer<1>)
  {
    return _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
  }

  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<5>, Integer<1>)
  {
    return _mm_setr_epi8(0, 5, 1, 6, 2, 7, 3, 8, 4, 9, -1, -1, -1, -1, -1, -1);
  }

  // SIMDWord, SIMDShort

  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<2>, Integer<2>)
  {
    return _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
  }

  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<3>, Integer<2>)
  {
    return _mm_setr_epi8(0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11, -1, -1, -1, -1);
  }

  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<4>, Integer<2>)
  {
    return _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
  }

  static SIMD_INLINE __m128i
  get_swizzle_mask(Integer<5>, Integer<2>)
  {
    return _mm_setr_epi8(0, 1, 10, 11, 2, 3, 12, 13, 4, 5, 14, 15, -1, -1, -1,
                         -1);
  }

  // hub
  template <int N, typename T>
  static SIMD_INLINE __m128i 
  get_swizzle_mask() {
    return get_swizzle_mask(Integer<N>(), Integer<sizeof(T)>());
  }

  // ---------- swizzle aux functions -----------

  // alignoff is the element-wise offset (relates to size of byte)
  template <int ALIGNOFF>
  static SIMD_INLINE __m128i 
  align_shuffle_byte_128(__m128i lo, __m128i hi, __m128i mask)
  {
    return _mm_shuffle_epi8(x_mm_alignr_epi8<ALIGNOFF>(hi, lo), mask);
  }

  // alignoff is the element-wise offset (relates to size of word)
  template <int ALIGNOFF>
  static SIMD_INLINE __m128i 
  align_shuffle_word_128(__m128i lo, __m128i hi, __m128i mask)
  {
    return _mm_shuffle_epi8(x_mm_alignr_epi8<2*ALIGNOFF>(hi, lo), mask);
  }

  // ---------- swizzle (AoS to SoA) ----------

  // -------------------- n = 1 --------------------

  // all types
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const,
	  Integer<1>,
	  TypeIsIntSize<T>)
  {
    // v remains unchanged
  }

  // -------------------- n = 2 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<2>,
	  IsIntSize<true,1>)
  {
    __m128i s[2];
    for (int j = 0; j < 2; j++)
      s[j] = _mm_shuffle_epi8(v[j], get_swizzle_mask<2, T>());
    v[0] = _mm_unpacklo_epi64(s[0], s[1]);
    v[1] = _mm_unpackhi_epi64(s[0], s[1]);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<2>,
	  IsIntSize<true,2>)
  {
    __m128i s[2];
    for (int j = 0; j < 2; j++)
      s[j] = _mm_shuffle_epi8(v[j], get_swizzle_mask<2, T>());
    v[0] = _mm_unpacklo_epi64(s[0], s[1]);
    v[1] = _mm_unpackhi_epi64(s[0], s[1]);
  }

  // SIMDInt
  // TODO: swizzle<2,SIMDInt,...>: which version is faster?
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<2>,
	  IsIntSize<true,4>)
  {
#if 0
    __m128i s[2];
    s[0] = _mm_shuffle_epi32(v[0], _MM_SHUFFLE(3,1,2,0));
    s[1] = _mm_shuffle_epi32(v[1], _MM_SHUFFLE(3,1,2,0));
    v[0] = _mm_unpacklo_epi64(s[0], s[1]);
    v[1] = _mm_unpackhi_epi64(s[0], s[1]);
#else
    __m128 v0tmp = _mm_castsi128_ps(v[0]);
    __m128 v1tmp = _mm_castsi128_ps(v[1]);
    v[0] = _mm_castps_si128(_mm_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(2,0,2,0)));
    v[1] = _mm_castps_si128(_mm_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(3,1,3,1)));
#endif
  }

  // SIMDFloat
  // same code as for SIMDInt
  static SIMD_INLINE void
  swizzle(SIMDVec<SIMDFloat, 16> *const v,
	  Integer<2>,
	  IsIntSize<false,4>)
  {
    __m128 v0tmp = v[0];
    __m128 v1tmp = v[1];
    v[0] = _mm_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(2,0,2,0));
    v[1] = _mm_shuffle_ps(v0tmp, v1tmp, _MM_SHUFFLE(3,1,3,1));
  }

  // -------------------- n = 3 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<3>,
	  IsIntSize<true,1>)
  {
    __m128i mask = get_swizzle_mask<3, T>();
    __m128i s0 = align_shuffle_byte_128<0> (v[0], v[1], mask);
    __m128i s1 = align_shuffle_byte_128<12>(v[0], v[1], mask);
    __m128i s2 = align_shuffle_byte_128<8> (v[1], v[2], mask);
    __m128i s3 = align_shuffle_byte_128<4> (v[2], v[0], mask);
    /* s3: v[0] is a dummy */
    __m128i l01 = _mm_unpacklo_epi32(s0, s1);
    __m128i h01 = _mm_unpackhi_epi32(s0, s1);
    __m128i l23 = _mm_unpacklo_epi32(s2, s3);
    __m128i h23 = _mm_unpackhi_epi32(s2, s3);
    v[0] = _mm_unpacklo_epi64(l01, l23);
    v[1] = _mm_unpackhi_epi64(l01, l23);
    v[2] = _mm_unpacklo_epi64(h01, h23);
  }

  // SIMDWord, SIMDShorst
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<3>,
	  IsIntSize<true,2>)
  {
    __m128i mask = get_swizzle_mask<3, T>();
    __m128i s0 = align_shuffle_word_128<0>(v[0], v[1], mask);
    __m128i s1 = align_shuffle_word_128<6>(v[0], v[1], mask);
    __m128i s2 = align_shuffle_word_128<4>(v[1], v[2], mask);
    __m128i s3 = align_shuffle_word_128<2>(v[2], v[0], mask);
    // s3: v[0] is a dummy
    __m128i l01 = _mm_unpacklo_epi32(s0, s1);
    __m128i h01 = _mm_unpackhi_epi32(s0, s1);
    __m128i l23 = _mm_unpacklo_epi32(s2, s3);
    __m128i h23 = _mm_unpackhi_epi32(s2, s3);
    v[0] = _mm_unpacklo_epi64(l01, l23);
    v[1] = _mm_unpackhi_epi64(l01, l23);
    v[2] = _mm_unpacklo_epi64(h01, h23);
  }

  // SIMDInt
  // from Stan Melax: "3D Vector Normalization..."
  // https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<3>,
	  IsIntSize<true,4>)
  {
    __m128 x0y0z0x1 = _mm_castsi128_ps(v[0]);
    __m128 y1z1x2y2 = _mm_castsi128_ps(v[1]);
    __m128 z2x3y3z3 = _mm_castsi128_ps(v[2]);
    __m128 x2y2x3y3 = _mm_shuffle_ps(y1z1x2y2, z2x3y3z3, _MM_SHUFFLE(2,1,3,2));
    __m128 y0z0y1z1 = _mm_shuffle_ps(x0y0z0x1, y1z1x2y2, _MM_SHUFFLE(1,0,2,1));
    // x0x1x2x3
    v[0] = _mm_castps_si128(_mm_shuffle_ps(x0y0z0x1, x2y2x3y3, 
					   _MM_SHUFFLE(2,0,3,0)));
    // y0y1y2y3
    v[1] = _mm_castps_si128(_mm_shuffle_ps(y0z0y1z1, x2y2x3y3, 
					   _MM_SHUFFLE(3,1,2,0)));
    // z0z1z2z3
    v[2] = _mm_castps_si128(_mm_shuffle_ps(y0z0y1z1, z2x3y3z3, 
					   _MM_SHUFFLE(3,0,3,1)));
  }

  // SIMDFloat
  // from Stan Melax: "3D Vector Normalization..."
  // https://software.intel.com/en-us/articles/3d-vector-normalization-using-256-bit-intel-advanced-vector-extensions-intel-avx
  // same code as for SIMDInt
  static SIMD_INLINE void
  swizzle(SIMDVec<SIMDFloat, 16> *const v,
	  Integer<3>,
	  IsIntSize<false,4>)
  {
    // x0y0z0x1 = v[0]
    // y1z1x2y2 = v[1]
    // z2x3y3z3 = v[2]
    __m128 x2y2x3y3 = _mm_shuffle_ps(v[1], v[2], _MM_SHUFFLE(2,1,3,2));
    __m128 y0z0y1z1 = _mm_shuffle_ps(v[0], v[1], _MM_SHUFFLE(1,0,2,1));
    // x0x1x2x3
    v[0] = _mm_shuffle_ps(v[0], x2y2x3y3, _MM_SHUFFLE(2,0,3,0));
    // y0y1y2y3
    v[1] = _mm_shuffle_ps(y0z0y1z1, x2y2x3y3, _MM_SHUFFLE(3,1,2,0));
    // z0z1z2z3
    v[2] = _mm_shuffle_ps(y0z0y1z1, v[2], _MM_SHUFFLE(3,0,3,1));
  }

  // -------------------- n = 4 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<4>,
	  IsIntSize<true,1>)
  {
    __m128i mask = get_swizzle_mask<4, T>();
    __m128i s[4];
    for (int j = 0; j < 4; j++)
      s[j] = _mm_shuffle_epi8(v[j], mask);
    __m128i l01 = _mm_unpacklo_epi32(s[0], s[1]);
    __m128i h01 = _mm_unpackhi_epi32(s[0], s[1]);
    __m128i l23 = _mm_unpacklo_epi32(s[2], s[3]);
    __m128i h23 = _mm_unpackhi_epi32(s[2], s[3]);
    v[0] = _mm_unpacklo_epi64(l01, l23);
    v[1] = _mm_unpackhi_epi64(l01, l23);
    v[2] = _mm_unpacklo_epi64(h01, h23);
    v[3] = _mm_unpackhi_epi64(h01, h23);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<4>,
	  IsIntSize<true,2>)
  {
    __m128i mask = get_swizzle_mask<4, T>();
    __m128i s[4];
    for (int j = 0; j < 4; j++)
      s[j] = _mm_shuffle_epi8(v[j], mask);
    __m128i l01 = _mm_unpacklo_epi32(s[0], s[1]);
    __m128i h01 = _mm_unpackhi_epi32(s[0], s[1]);
    __m128i l23 = _mm_unpacklo_epi32(s[2], s[3]);
    __m128i h23 = _mm_unpackhi_epi32(s[2], s[3]);
    v[0] = _mm_unpacklo_epi64(l01, l23);
    v[1] = _mm_unpackhi_epi64(l01, l23);
    v[2] = _mm_unpacklo_epi64(h01, h23);
    v[3] = _mm_unpackhi_epi64(h01, h23);
  }

  // SIMDInt
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<4>,
	  IsIntSize<true,4>)
  {
    __m128i s[4];
    s[0] = _mm_unpacklo_epi32(v[0], v[1]);
    s[1] = _mm_unpackhi_epi32(v[0], v[1]);
    s[2] = _mm_unpacklo_epi32(v[2], v[3]);
    s[3] = _mm_unpackhi_epi32(v[2], v[3]);
    v[0] = _mm_unpacklo_epi64(s[0], s[2]);
    v[1] = _mm_unpackhi_epi64(s[0], s[2]);
    v[2] = _mm_unpacklo_epi64(s[1], s[3]);
    v[3] = _mm_unpackhi_epi64(s[1], s[3]);
  }

  // SIMDFloat
  static SIMD_INLINE void
  swizzle(SIMDVec<SIMDFloat, 16> *const v,
	  Integer<4>,
	  IsIntSize<false,4>)
  {
    __m128 s[4];
    s[0] = _mm_shuffle_ps(v[0], v[1], _MM_SHUFFLE(1,0,1,0));
    s[1] = _mm_shuffle_ps(v[0], v[1], _MM_SHUFFLE(3,2,3,2));
    s[2] = _mm_shuffle_ps(v[2], v[3], _MM_SHUFFLE(1,0,1,0));
    s[3] = _mm_shuffle_ps(v[2], v[3], _MM_SHUFFLE(3,2,3,2));
    v[0] = _mm_shuffle_ps(s[0], s[2], _MM_SHUFFLE(2,0,2,0));
    v[1] = _mm_shuffle_ps(s[0], s[2], _MM_SHUFFLE(3,1,3,1));
    v[2] = _mm_shuffle_ps(s[1], s[3], _MM_SHUFFLE(2,0,2,0));
    v[3] = _mm_shuffle_ps(s[1], s[3], _MM_SHUFFLE(3,1,3,1));
  }

  // -------------------- n = 5 --------------------

  // SIMDByte, SIMDSignedByte
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<5>,
	  IsIntSize<true,1>)
  {
    __m128i mask = get_swizzle_mask<5, T>();
    __m128i s0 = align_shuffle_byte_128<0> (v[0], v[1], mask);
    __m128i s1 = align_shuffle_byte_128<10>(v[0], v[1], mask);
    __m128i s2 = align_shuffle_byte_128<4> (v[1], v[2], mask);
    __m128i s3 = align_shuffle_byte_128<14>(v[1], v[2], mask);
    __m128i s4 = align_shuffle_byte_128<8> (v[2], v[3], mask);
    __m128i s5 = align_shuffle_byte_128<2> (v[3], v[4], mask);
    __m128i s6 = align_shuffle_byte_128<12>(v[3], v[4], mask);
    __m128i s7 = align_shuffle_byte_128<6> (v[4], v[0], mask);
    /* s7: v[0] is a dummy */
    __m128i l01 = _mm_unpacklo_epi16(s0, s1);
    __m128i h01 = _mm_unpackhi_epi16(s0, s1);
    __m128i l23 = _mm_unpacklo_epi16(s2, s3);
    __m128i h23 = _mm_unpackhi_epi16(s2, s3);
    __m128i l45 = _mm_unpacklo_epi16(s4, s5);
    __m128i h45 = _mm_unpackhi_epi16(s4, s5);
    __m128i l67 = _mm_unpacklo_epi16(s6, s7);
    __m128i h67 = _mm_unpackhi_epi16(s6, s7);
    __m128i ll01l23 = _mm_unpacklo_epi32(l01, l23);
    __m128i hl01l23 = _mm_unpackhi_epi32(l01, l23);
    __m128i ll45l67 = _mm_unpacklo_epi32(l45, l67);
    __m128i hl45l67 = _mm_unpackhi_epi32(l45, l67);
    __m128i lh01h23 = _mm_unpacklo_epi32(h01, h23);
    __m128i lh45h67 = _mm_unpacklo_epi32(h45, h67);
    v[0] = _mm_unpacklo_epi64(ll01l23, ll45l67);
    v[1] = _mm_unpackhi_epi64(ll01l23, ll45l67);
    v[2] = _mm_unpacklo_epi64(hl01l23, hl45l67);
    v[3] = _mm_unpackhi_epi64(hl01l23, hl45l67);
    v[4] = _mm_unpacklo_epi64(lh01h23, lh45h67);
  }

  // SIMDWord, SIMDShort
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<5>,
	  IsIntSize<true,2>)
  {
    __m128i mask = get_swizzle_mask<5, T>();
    __m128i s0 = align_shuffle_word_128<0>(v[0], v[1], mask);
    __m128i s1 = align_shuffle_word_128<3>(v[0], v[1], mask);
    __m128i s2 = align_shuffle_word_128<2>(v[1], v[2], mask);
    __m128i s3 = align_shuffle_word_128<5>(v[1], v[2], mask);
    __m128i s4 = align_shuffle_word_128<4>(v[2], v[3], mask);
    __m128i s5 = align_shuffle_word_128<7>(v[2], v[3], mask);
    __m128i s6 = align_shuffle_word_128<6>(v[3], v[4], mask);
    __m128i s7 = align_shuffle_word_128<1>(v[4], v[0], mask);
    // s7: v[0] is a dummy
    __m128i l02 = _mm_unpacklo_epi32(s0, s2);
    __m128i h02 = _mm_unpackhi_epi32(s0, s2);
    __m128i l13 = _mm_unpacklo_epi32(s1, s3);
    __m128i l46 = _mm_unpacklo_epi32(s4, s6);
    __m128i h46 = _mm_unpackhi_epi32(s4, s6);
    __m128i l57 = _mm_unpacklo_epi32(s5, s7);
    v[0] = _mm_unpacklo_epi64(l02, l46);
    v[1] = _mm_unpackhi_epi64(l02, l46);
    v[2] = _mm_unpacklo_epi64(h02, h46);
    v[3] = _mm_unpacklo_epi64(l13, l57);
    v[4] = _mm_unpackhi_epi64(l13, l57);
  }

  // SIMDInt
  template <typename T>
  static SIMD_INLINE void
  swizzle(SIMDVec<T, 16> *const v,
	  Integer<5>,
	  IsIntSize<true,4>)
  {
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    // v[0]: 0 1 2 3
    // v[1]: 4 x x x
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                   x x x   x
    // 5 6 7 8
    __m128i s2 = x_mm_alignr_epi8<4>(v[2], v[1]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                             x  x  x    x
    // 9 x x x
    __m128i s3 = x_mm_alignr_epi8<4>(v[3], v[2]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                x  x    x  x
    // 10 11 12 13
    __m128i s4 = x_mm_alignr_epi8<8>(v[3], v[2]); 
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                              x  x    x  x
    // 14 x x x
    __m128i s5 = x_mm_alignr_epi8<8>(v[4], v[3]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                 X    X  X  X
    // 15 16 17 18
    __m128i s6 = x_mm_alignr_epi8<12>(v[4], v[3]);
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                               X X X X
    // 19 x x x
    __m128i s7 = x_mm_alignr_epi8<12>(v[0], v[4]);
    // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
    __m128i l02 = _mm_unpacklo_epi32(v[0], s2);
    __m128i h02 = _mm_unpackhi_epi32(v[0], s2);
    // 4 x x x / 9 x x x -> 4 9 x x
    __m128i l13 = _mm_unpacklo_epi32(v[1], s3);
    // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
    __m128i l46 = _mm_unpacklo_epi32(s4, s6);
    __m128i h46 = _mm_unpackhi_epi32(s4, s6);
    // 14 x x x / 19 x x x -> 14 19 x x
    __m128i l57 = _mm_unpacklo_epi32(s5, s7);
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    v[0] = _mm_unpacklo_epi64(l02, l46);
    v[1] = _mm_unpackhi_epi64(l02, l46);
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    v[2] = _mm_unpacklo_epi64(h02, h46);
    v[3] = _mm_unpackhi_epi64(h02, h46);
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    v[4] = _mm_unpacklo_epi64(l13, l57);
  }

  // SIMDFloat
  // same code as for SIMDInt, modified
  static SIMD_INLINE void
  swizzle(SIMDVec<SIMDFloat, 16> *const v,
	  Integer<5>,
	  IsIntSize<false,4>)
  {
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    // v[0]: 0 1 2 3
    __m128i v0 = _mm_castps_si128(v[0]);
    // v[1]: 4 5 6 7
    __m128i v1 = _mm_castps_si128(v[1]);
    // v[2]: 8 9 10 11
    __m128i v2 = _mm_castps_si128(v[2]);
    // v[3]: 12 13 14 15
    __m128i v3 = _mm_castps_si128(v[3]);
    // v[4]: 16 17 18 19
    __m128i v4 = _mm_castps_si128(v[4]);
    // s0:  0 1 2 3
    __m128 s0 = v[0];
    // s1:  4 x x x
    __m128 s1 = v[1];
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                   x x x   x
    // 5 6 7 8
    __m128 s2 = _mm_castsi128_ps(x_mm_alignr_epi8<4>(v2, v1));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                             x  x  x    x
    // 9 x x x
    __m128 s3 = _mm_castsi128_ps(x_mm_alignr_epi8<4>(v3, v2));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                x  x    x  x
    // 10 11 12 13
    __m128 s4 = _mm_castsi128_ps(x_mm_alignr_epi8<8>(v3, v2)); 
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                              x  x    x  x
    // 14 x x x
    __m128 s5 = _mm_castsi128_ps(x_mm_alignr_epi8<8>(v4, v3));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                 X    X  X  X
    // 15 16 17 18
    __m128 s6 = _mm_castsi128_ps(x_mm_alignr_epi8<12>(v4, v3));
    // v:    0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19
    //                                                               X X X X
    // 19 x x x
    __m128 s7 = _mm_castsi128_ps(x_mm_alignr_epi8<12>(v0, v4));
    // 0 1 2 3 / 5 6 7 8 -> 0 5 1 6 / 2 7 3 8
    __m128 l02 = _mm_unpacklo_ps(s0, s2);
    __m128 h02 = _mm_unpackhi_ps(s0, s2);
    // 4 x x x / 9 x x x -> 4 9 x x
    __m128 l13 = _mm_unpacklo_ps(s1, s3);
    // 10 11 12 13 / 15 16 17 18 -> 10 15 11 13 / 12 17 13 18
    __m128 l46 = _mm_unpacklo_ps(s4, s6);
    __m128 h46 = _mm_unpackhi_ps(s4, s6);
    // 14 x x x / 19 x x x -> 14 19 x x
    __m128 l57 = _mm_unpacklo_ps(s5, s7);
    // 0 5 1 6 / 10 15 11 13 -> 0 5 10 15 / 1 6 11 16
    v[0] = _mm_movelh_ps(l02, l46);
    v[1] = _mm_movehl_ps(l46, l02);
    // 2 7 3 8 / 12 17 13 18 -> 2 7 12 17 / 3 8 13 18
    v[2] = _mm_movelh_ps(h02, h46);
    v[3] = _mm_movehl_ps(h46, h02);
    // 4 9 x x / 14 19 x x -> 4 9 14 19
    v[4] = _mm_movelh_ps(l13, l57);
  }

  // ---------------------------------------------------------------------------
  // compare <
  // ---------------------------------------------------------------------------

  // http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
  static SIMD_INLINE SIMDVec<SIMDByte,16>
  cmplt(const SIMDVec<SIMDByte,16> &a,
	const SIMDVec<SIMDByte,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x80
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x80
    return _mm_cmplt_epi8(a1, b1);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  cmplt(const SIMDVec<SIMDSignedByte,16> &a,
	const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_cmplt_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  cmplt(const SIMDVec<SIMDWord,16> &a,
	const SIMDVec<SIMDWord,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x8000
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x8000
    return _mm_cmplt_epi16(a1, b1);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  cmplt(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_cmplt_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  cmplt(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_cmplt_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  cmplt(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_cmplt_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // compare <=
  // ---------------------------------------------------------------------------

  // http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
  static SIMD_INLINE SIMDVec<SIMDByte,16>
  cmple(const SIMDVec<SIMDByte,16> &a,
	const SIMDVec<SIMDByte,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x80
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x80
    return _mm_or_si128(_mm_cmplt_epi8(a1, b1), _mm_cmpeq_epi8(a1, b1));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  cmple(const SIMDVec<SIMDSignedByte,16> &a,
	const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_or_si128(_mm_cmplt_epi8(a, b), _mm_cmpeq_epi8(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  cmple(const SIMDVec<SIMDWord,16> &a,
	const SIMDVec<SIMDWord,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x8000
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x8000
    return _mm_or_si128(_mm_cmplt_epi16(a1, b1), _mm_cmpeq_epi16(a1, b1));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  cmple(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_or_si128(_mm_cmplt_epi16(a, b), _mm_cmpeq_epi16(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  cmple(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_or_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  cmple(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_cmple_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // compare ==
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  cmpeq(const SIMDVec<SIMDByte,16> &a,
	const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_cmpeq_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  cmpeq(const SIMDVec<SIMDSignedByte,16> &a,
	const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_cmpeq_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  cmpeq(const SIMDVec<SIMDWord,16> &a,
	const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_cmpeq_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  cmpeq(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_cmpeq_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  cmpeq(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_cmpeq_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  cmpeq(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_cmpeq_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // compare >
  // ---------------------------------------------------------------------------

  // http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
  static SIMD_INLINE SIMDVec<SIMDByte,16>
  cmpgt(const SIMDVec<SIMDByte,16> &a,
	const SIMDVec<SIMDByte,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x80
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x80
    return _mm_cmpgt_epi8(a1, b1);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  cmpgt(const SIMDVec<SIMDSignedByte,16> &a,
	const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_cmpgt_epi8(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  cmpgt(const SIMDVec<SIMDWord,16> &a,
	const SIMDVec<SIMDWord,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x8000
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x8000
    return _mm_cmpgt_epi16(a1, b1);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  cmpgt(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_cmpgt_epi16(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  cmpgt(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_cmpgt_epi32(a, b);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  cmpgt(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_cmpgt_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // compare >=
  // ---------------------------------------------------------------------------

  // http://stackoverflow.com/questions/16204663/sse-compare-packed-unsigned-bytes
  static SIMD_INLINE SIMDVec<SIMDByte,16>
  cmpge(const SIMDVec<SIMDByte,16> &a,
	const SIMDVec<SIMDByte,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x80
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x80
    return _mm_or_si128(_mm_cmpgt_epi8(a1, b1), _mm_cmpeq_epi8(a1, b1));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  cmpge(const SIMDVec<SIMDSignedByte,16> &a,
	const SIMDVec<SIMDSignedByte,16> &b)
  {
    return _mm_or_si128(_mm_cmpgt_epi8(a, b), _mm_cmpeq_epi8(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  cmpge(const SIMDVec<SIMDWord,16> &a,
	const SIMDVec<SIMDWord,16> &b)
  {
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a, signbit);            // sub 0x8000
    __m128i b1      = _mm_xor_si128(b, signbit);            // sub 0x8000
    return _mm_or_si128(_mm_cmpgt_epi16(a1, b1), _mm_cmpeq_epi16(a1, b1));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  cmpge(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return _mm_or_si128(_mm_cmpgt_epi16(a, b), _mm_cmpeq_epi16(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  cmpge(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return _mm_or_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  cmpge(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_cmpge_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // compare !=
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  cmpneq(const SIMDVec<SIMDByte,16> &a,
	 const SIMDVec<SIMDByte,16> &b)
  {
    return x_mm_not_si128(_mm_cmpeq_epi8(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  cmpneq(const SIMDVec<SIMDSignedByte,16> &a,
	 const SIMDVec<SIMDSignedByte,16> &b)
  {
    return x_mm_not_si128(_mm_cmpeq_epi8(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  cmpneq(const SIMDVec<SIMDWord,16> &a,
	 const SIMDVec<SIMDWord,16> &b)
  {
    return x_mm_not_si128(_mm_cmpeq_epi16(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  cmpneq(const SIMDVec<SIMDShort,16> &a,
	 const SIMDVec<SIMDShort,16> &b)
  {
    return x_mm_not_si128(_mm_cmpeq_epi16(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  cmpneq(const SIMDVec<SIMDInt,16> &a,
	 const SIMDVec<SIMDInt,16> &b)
  {
    return x_mm_not_si128(_mm_cmpeq_epi32(a, b));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  cmpneq(const SIMDVec<SIMDFloat,16> &a,
	 const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_cmpneq_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // ifelse
  // ---------------------------------------------------------------------------

  // elements of cond must be all 1's or all 0's (blendv just tests top
  // bit in each byte, but work-around needs this)

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  ifelse(const SIMDVec<T,16> &cond,
	 const SIMDVec<T,16> &trueVal,
	 const SIMDVec<T,16> &falseVal)
  {
    return x_mm_ifelsei(cond, trueVal, falseVal);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  ifelse(const SIMDVec<SIMDFloat,16> &cond,
	 const SIMDVec<SIMDFloat,16> &trueVal,
	 const SIMDVec<SIMDFloat,16> &falseVal)
  {
    return x_mm_ifelsef(cond, trueVal, falseVal);
  }

  // ---------------------------------------------------------------------------
  // and
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  and_(const SIMDVec<T,16> &a,
      const SIMDVec<T,16> &b)
  {
    return _mm_and_si128(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  and_(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_and_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // or
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  or_(const SIMDVec<T,16> &a,
     const SIMDVec<T,16> &b)
  {
    return _mm_or_si128(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  or_(const SIMDVec<SIMDFloat,16> &a,
     const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_or_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // andnot
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  andnot(const SIMDVec<T,16> &a,
	 const SIMDVec<T,16> &b)
  {
    return _mm_andnot_si128(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  andnot(const SIMDVec<SIMDFloat,16> &a,
	 const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_andnot_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // xor
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  xor_(const SIMDVec<T,16> &a,
      const SIMDVec<T,16> &b)
  {
    return _mm_xor_si128(a, b);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  xor_(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return _mm_xor_ps(a, b);
  }

  // ---------------------------------------------------------------------------
  // not
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  not_(const SIMDVec<T,16> &a)
  {
    return x_mm_not_si128(a);
  }

  // float version
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  not_(const SIMDVec<SIMDFloat,16> &a)
  {
    return x_mm_not_ps(a);
  }

  // ---------------------------------------------------------------------------
  // avg: average with rounding up
  // ---------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  avg(const SIMDVec<SIMDByte,16> &a,
      const SIMDVec<SIMDByte,16> &b)
  {
    return _mm_avg_epu8(a, b);
  }

  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  avg(const SIMDVec<SIMDSignedByte,16> &a,
      const SIMDVec<SIMDSignedByte,16> &b)
  {
    // from Agner Fog's VCL vectori128.h
    __m128i signbit = _mm_set1_epi32(0x80808080);
    __m128i a1      = _mm_xor_si128(a, signbit); // add 0x80
    __m128i b1      = _mm_xor_si128(b, signbit); // add 0x80
    __m128i m1      = _mm_avg_epu8(a1, b1);	 // unsigned avg
    return  _mm_xor_si128(m1, signbit);		 // sub 0x80
  }
  
  static SIMD_INLINE SIMDVec<SIMDWord,16>
  avg(const SIMDVec<SIMDWord,16> &a,
      const SIMDVec<SIMDWord,16> &b)
  {
    return _mm_avg_epu16(a, b);
  }

  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDShort,16>
  avg(const SIMDVec<SIMDShort,16> &a,
      const SIMDVec<SIMDShort,16> &b)
  {
    // from Agner Fog's VCL vectori128.h
    __m128i signbit = _mm_set1_epi32(0x80008000);
    __m128i a1      = _mm_xor_si128(a, signbit); // add 0x8000
    __m128i b1      = _mm_xor_si128(b, signbit); // add 0x8000
    __m128i m1      = _mm_avg_epu16(a1, b1);	 // unsigned avg
    return  _mm_xor_si128(m1, signbit);		 // sub 0x8000
  }

  // Paul R at
  // http://stackoverflow.com/questions/12152640/signed-16-bit-sse-average
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  avg(const SIMDVec<SIMDInt,16> &a,
      const SIMDVec<SIMDInt,16> &b)
  {
    SIMDVec<SIMDInt,16> one = set1(SIMDInt(1), Integer<16>()), as, bs, lsb;
    lsb = and_(or_(a, b), one);
    as = srai<1>(a);
    bs = srai<1>(b);
    return add(lsb, add(as, bs));
  }

  // NOTE: SIMDFloat version doesn't round!
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  avg(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    __m128 half = _mm_set1_ps(0.5f);
    return _mm_mul_ps(_mm_add_ps(a, b), half);
  }

  // ---------------------------------------------------------------------------
  // test_all_zeros
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<T,16> &a)
  {
    return x_mm_test_all_zeros(a);
  }


  // float version
  static SIMD_INLINE int
  test_all_zeros(const SIMDVec<SIMDFloat,16> &a)
  {
    return x_mm_test_all_zeros(_mm_castps_si128(a));
  }

  // ---------------------------------------------------------------------------
  // test_all_ones
  // ---------------------------------------------------------------------------

  // all integer versions
  template <typename T>
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<T,16> &a)
  {
    return x_mm_test_all_ones(a);
  }

  // float version
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<SIMDFloat,16> &a)
  {
    return x_mm_test_all_ones(_mm_castps_si128(a));
  }

  // ---------------------------------------------------------------------------
  // reverse
  // ---------------------------------------------------------------------------
  
  // All reverse operations below are courtesy of Yannick Sander.
  // modified

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  reverse(const SIMDVec<SIMDByte,16> &a)
  {
    
    const __m128i mask =
      _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    
    // supported by sse3 and higher
    // compat available
    // no faster instruction available in newer instruction sets
    return _mm_shuffle_epi8(a, mask);
  }
  
  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  reverse(const SIMDVec<SIMDSignedByte,16> &a)
  {
    
    const __m128i mask =
      _mm_set_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    
    return _mm_shuffle_epi8(a, mask);
  }
  
  static SIMD_INLINE SIMDVec<SIMDShort,16>
  reverse(const SIMDVec<SIMDShort,16> &a)
  {
    const __m128i mask =
      _mm_set_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    
    return _mm_shuffle_epi8(a, mask);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  reverse(const SIMDVec<SIMDWord,16> &a)
  {
    const __m128i mask =
      _mm_set_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    
    return _mm_shuffle_epi8(a, mask);
  }
  
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  reverse(const SIMDVec<SIMDInt,16> &a)
  {
    return _mm_shuffle_epi32(a, _MM_SHUFFLE(0,1,2,3));
  }
  
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  reverse(const SIMDVec<SIMDFloat,16> &a)
  {
    return _mm_shuffle_ps(a, a, _MM_SHUFFLE(0,1,2,3));
  }
  

  // ---------------------------------------------------------------------------
  // msb2int
  // ---------------------------------------------------------------------------

  // 26. Aug 22 (Jonas Keller): added msb2int functions

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDByte,16> &a)
  {
    return _mm_movemask_epi8(a);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDSignedByte,16> &a)
  {
    return _mm_movemask_epi8(a);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDShort,16> &a)
  {
    // move the upper bytes of the 8 shorts to the lower 8 bytes of the vector
    // and set the upper 8 bytes of to 0, so that _mm_movemask_epi8
    // can be used to extract the upper bit of each short
    const __m128i mask =
        _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
    const __m128i shuffled = _mm_shuffle_epi8(a, mask);
    return _mm_movemask_epi8(shuffled);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDWord,16> &a)
  {
    // move the upper bytes of the 8 words to the lower 8 bytes of the vector
    // and set the upper 8 bytes of to 0, so that _mm_movemask_epi8
    // can be used to extract the upper bit of each word
    const __m128i mask =
        _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
    const __m128i shuffled = _mm_shuffle_epi8(a, mask);
    return _mm_movemask_epi8(shuffled);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDInt,16> &a)
  {
    return _mm_movemask_ps(_mm_castsi128_ps(a));
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDFloat,16> &a)
  {
    return _mm_movemask_ps(a);
  }

  // ---------------------------------------------------------------------------
  // int2msb
  // ---------------------------------------------------------------------------

  // 06. Oct 22 (Jonas Keller): added int2msb functions

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  int2msb(const uint64_t a, OutputType<SIMDByte>, Integer<16>)
  {
    // ssse3 version from https://stackoverflow.com/a/72899629
#ifdef __SSSE3__
    __m128i shuffleIndeces = _mm_set_epi64x(0x0101010101010101, 0);
    __m128i aVec = _mm_shuffle_epi8(_mm_cvtsi32_si128(a), shuffleIndeces);
#else
    __m128i maskLo = _mm_set_epi64x(0, 0xffffffffffffffff);
    __m128i aLo = _mm_and_si128(maskLo, _mm_set1_epi8(a)) ;
    __m128i aHi = _mm_andnot_si128(maskLo, _mm_set1_epi8(a >> 8));
    __m128i aVec = _mm_or_si128(aLo, aHi);
#endif
    __m128i sel = _mm_set1_epi64x(0x8040201008040201);
    __m128i selected = _mm_and_si128(aVec, sel);
    __m128i result = _mm_cmpeq_epi8(selected, sel);
    return _mm_and_si128(result, _mm_set1_epi8((int8_t)0x80));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  int2msb(const uint64_t a, OutputType<SIMDSignedByte>, Integer<16>)
  {
    return reinterpret(int2msb(a, OutputType<SIMDByte>(), Integer<16>()),
                       OutputType<SIMDSignedByte>());
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  int2msb(const uint64_t a, OutputType<SIMDShort>, Integer<16>)
  {
    __m128i aVec = _mm_set1_epi16(a);
    __m128i sel = _mm_set_epi16(0x0080, 0x0040, 0x0020, 0x0010,
                                0x0008, 0x0004, 0x0002, 0x0001);
    __m128i selected = _mm_and_si128(aVec, sel);
    __m128i result = _mm_cmpeq_epi16(selected, sel);
    return _mm_and_si128(result, _mm_set1_epi16((int16_t)0x8000));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  int2msb(const uint64_t a, OutputType<SIMDWord>, Integer<16>)
  {
    return reinterpret(int2msb(a, OutputType<SIMDShort>(), Integer<16>()),
                       OutputType<SIMDWord>());
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  int2msb(const uint64_t a, OutputType<SIMDInt>, Integer<16>)
  {
    __m128i aVec = _mm_set1_epi32(a);
    __m128i sel = _mm_set_epi32(0x00000008, 0x00000004, 0x00000002, 0x00000001);
    __m128i selected = _mm_and_si128(aVec, sel);
    __m128i result = _mm_cmpeq_epi32(selected, sel);
    return _mm_and_si128(result, _mm_set1_epi32(0x80000000));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  int2msb(const uint64_t a, OutputType<SIMDFloat>, Integer<16>)
  {
    return reinterpret(int2msb(a, OutputType<SIMDInt>(), Integer<16>()),
                       OutputType<SIMDFloat>());
  }

  // ---------------------------------------------------------------------------
  // int2bits
  // ---------------------------------------------------------------------------

  // 09. Oct 22 (Jonas Keller): added int2bits functions

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  int2bits(const uint64_t a, OutputType<SIMDByte>, Integer<16>)
  {
    // ssse3 version from https://stackoverflow.com/a/72899629
#ifdef __SSSE3__
    __m128i shuffleIndeces = _mm_set_epi64x(0x0101010101010101, 0);
    __m128i aVec = _mm_shuffle_epi8(_mm_cvtsi32_si128(a), shuffleIndeces);
#else
    __m128i maskLo = _mm_set_epi64x(0, 0xffffffffffffffff);
    __m128i aLo = _mm_and_si128(maskLo, _mm_set1_epi8(a)) ;
    __m128i aHi = _mm_andnot_si128(maskLo, _mm_set1_epi8(a >> 8));
    __m128i aVec = _mm_or_si128(aLo, aHi);
#endif
    __m128i sel = _mm_set1_epi64x(0x8040201008040201);
    __m128i selected = _mm_and_si128(aVec, sel);
    return _mm_cmpeq_epi8(selected, sel);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  int2bits(const uint64_t a, OutputType<SIMDSignedByte>, Integer<16>)
  {
    return reinterpret(int2bits(a, OutputType<SIMDByte>(), Integer<16>()),
                       OutputType<SIMDSignedByte>());
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  int2bits(const uint64_t a, OutputType<SIMDShort>, Integer<16>)
  {
    __m128i aVec = _mm_set1_epi16(a);
    __m128i sel = _mm_set_epi16(0x0080, 0x0040, 0x0020, 0x0010,
                                0x0008, 0x0004, 0x0002, 0x0001);
    __m128i selected = _mm_and_si128(aVec, sel);
    return _mm_cmpeq_epi16(selected, sel);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  int2bits(const uint64_t a, OutputType<SIMDWord>, Integer<16>)
  {
    return reinterpret(int2bits(a, OutputType<SIMDShort>(), Integer<16>()),
                       OutputType<SIMDWord>());
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  int2bits(const uint64_t a, OutputType<SIMDInt>, Integer<16>)
  {
    __m128i aVec = _mm_set1_epi32(a);
    __m128i sel = _mm_set_epi32(0x00000008, 0x00000004, 0x00000002, 0x00000001);
    __m128i selected = _mm_and_si128(aVec, sel);
    return _mm_cmpeq_epi32(selected, sel);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  int2bits(const uint64_t a, OutputType<SIMDFloat>, Integer<16>)
  {
    return reinterpret(int2bits(a, OutputType<SIMDInt>(), Integer<16>()),
                       OutputType<SIMDFloat>());
  }

  // ---------------------------------------------------------------------------
  // iota
  // ---------------------------------------------------------------------------

  // 30. Jan 23 (Jonas Keller): added iota

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  iota(OutputType<SIMDByte>, Integer<16>)
  {
    return _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  iota(OutputType<SIMDSignedByte>, Integer<16>)
  {
    return _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  iota(OutputType<SIMDShort>, Integer<16>)
  {
    return _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  iota(OutputType<SIMDWord>, Integer<16>)
  {
    return _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  iota(OutputType<SIMDInt>, Integer<16>)
  {
    return _mm_set_epi32(3, 2, 1, 0);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  iota(OutputType<SIMDFloat>, Integer<16>)
  {
    return _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
  }

} // namespace base
} // namespace internal
} // namespace ns_simd

#endif // __SSE2__

#endif // _SIMD_VEC_BASE_IMPL_INTEL_16_H_
