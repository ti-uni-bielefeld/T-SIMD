// ===========================================================================
// 
// ssse3_compat.H --
// compatibility code for CPUs without SSE3 or SSSE3
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

#ifndef _SSSE3_COMPAT_H_
#define _SSSE3_COMPAT_H_

#include "SIMDDefs.H"
#include "SIMDIntrinsIntel.H"

#include <math.h>
#include <stdint.h>

// SSE3/SSSE3 emulation for very old CPUs (very inefficient sequential code!)

// replacement for SSE3 instructions
// _mm_hadd_ps
// _mm_hsub_ps
//
// replacement for SSSE3 instructions
// _mm_abs_epi8
// _mm_abs_epi16
// _mm_abs_epi32
// _mm_alignr_epi8
// _mm_hadd_epi16
// _mm_hadd_epi32
// _mm_hadds_epi16
// _mm_hsub_epi16
// _mm_hsub_epi32
// _mm_hsubs_epi16
// _mm_shuffle_epi8
// _mm_sign_epi16

#ifdef __SSE2__

namespace ns_simd {

// ===========================================================================
// SSE3 replacements
// ===========================================================================

#ifndef __SSE3__

  // #warning "SSE3 intrinsics are replaced by slow sequential implementations"

  static inline __m128 _mm_hadd_ps(__m128 a, __m128 b)
  {
    float atmp[4] SIMD_ATTR_ALIGNED(16);
    float btmp[4] SIMD_ATTR_ALIGNED(16);
    float tmp[4] SIMD_ATTR_ALIGNED(16);
    _mm_store_ps(atmp, a);
    _mm_store_ps(btmp, b);
    tmp[0] = atmp[1] + atmp[0];
    tmp[1] = atmp[3] + atmp[2];
    tmp[2] = btmp[1] + btmp[0];
    tmp[3] = btmp[3] + btmp[2];
    return _mm_load_ps(tmp);
  }

  static inline __m128 _mm_hsub_ps(__m128 a, __m128 b)
  {
    float atmp[4] SIMD_ATTR_ALIGNED(16);
    float btmp[4] SIMD_ATTR_ALIGNED(16);
    float tmp[4] SIMD_ATTR_ALIGNED(16);
    _mm_store_ps(atmp, a);
    _mm_store_ps(btmp, b);
    tmp[0] = atmp[0] - atmp[1];
    tmp[1] = atmp[2] - atmp[3];
    tmp[2] = btmp[0] - btmp[1];
    tmp[3] = btmp[2] - btmp[3];
    return _mm_load_ps(tmp);
  }

#endif

  // ===========================================================================
  // SSE3 replacements
  // ===========================================================================

#ifndef __SSSE3__

  // #warning "SSSE3 intrinsics are replaced by slow sequential implementations"

  static inline __m128i _mm_abs_epi8(__m128i a)
  {
    int i;
    int8_t tmp[16] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) tmp, a);
    for (i = 0; i < 16; i++)
      tmp[i] = std::abs(tmp[i]);
    return _mm_load_si128((__m128i*) tmp);
  }

  static inline __m128i _mm_abs_epi16(__m128i a)
  {
    int i;
    int16_t tmp[8] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) tmp, a);
    for (i = 0; i < 8; i++)
      tmp[i] = std::abs(tmp[i]);
    return _mm_load_si128((__m128i*) tmp);
  }

  static inline __m128i _mm_abs_epi32(__m128i a)
  {
    int i;
    int32_t tmp[4] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) tmp, a);
    for (i = 0; i < 4; i++)
      tmp[i] = std::abs(tmp[i]);
    return _mm_load_si128((__m128i*) tmp);
  }

  // 24. Jul 18 (rm): strange, definitions from tmmintrin.h are
  // included even if ssse3 is not available;
  // gcc: for undefined __OPTIMIZE__, the macro _mm_alignr_epi8
  // clashes with the definition below;
  // clang: _mm_alignr_epi8 always defined as macro, regardless of
  // __OPTIMIZE__; also clashes with the definition below
  // solution: undefine the macro if it is defined
#ifdef _mm_alignr_epi8
#undef _mm_alignr_epi8
#endif
  
  // 23. Sep 15 (rm): fixed several bugs
  static inline __m128i _mm_alignr_epi8(__m128i a, __m128i b, int n)
  {
    int8_t abtmp[32] SIMD_ATTR_ALIGNED(16);
    int8_t rtmp[16] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) abtmp, b);
    _mm_store_si128((__m128i*) (abtmp + 16), a);
    int i, j;
    for (j = 0, i = n; j < 16; i++, j++)
      if ((i >= 0) && (i < 32))
	rtmp[j] = abtmp[i];
      else
	rtmp[j] = 0;
    return _mm_load_si128((__m128i*) rtmp);
  }

  static inline __m128i _mm_hadd_epi16(__m128i a, __m128i b)
  {
    int16_t atmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t btmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t tmp[8] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    tmp[0] = atmp[1] + atmp[0];
    tmp[1] = atmp[3] + atmp[2];
    tmp[2] = atmp[5] + atmp[4];
    tmp[3] = atmp[7] + atmp[6];
    tmp[4] = btmp[1] + btmp[0];
    tmp[5] = btmp[3] + btmp[2];
    tmp[6] = btmp[5] + btmp[4];
    tmp[7] = btmp[7] + btmp[6];
    return _mm_load_si128((__m128i*) tmp);
  }

  static inline __m128i _mm_hsub_epi16(__m128i a, __m128i b)
  {
    int16_t atmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t btmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t tmp[8] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    tmp[0] = atmp[0] - atmp[1];
    tmp[1] = atmp[2] - atmp[3];
    tmp[2] = atmp[4] - atmp[5];
    tmp[3] = atmp[6] - atmp[7];
    tmp[4] = btmp[0] - btmp[1];
    tmp[5] = btmp[2] - btmp[3];
    tmp[6] = btmp[4] - btmp[5];
    tmp[7] = btmp[6] - btmp[7];
    return _mm_load_si128((__m128i*) tmp);
  }

  static inline __m128i _mm_hadd_epi32(__m128i a, __m128i b)
  {
    int32_t atmp[4] SIMD_ATTR_ALIGNED(16);
    int32_t btmp[4] SIMD_ATTR_ALIGNED(16);
    int32_t tmp[4] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    tmp[0] = atmp[1] + atmp[0];
    tmp[1] = atmp[3] + atmp[2];
    tmp[2] = btmp[1] + btmp[0];
    tmp[3] = btmp[3] + btmp[2];
    return _mm_load_si128((__m128i*) tmp); 
  }

  static inline __m128i _mm_hsub_epi32(__m128i a, __m128i b)
  {
    int32_t atmp[4] SIMD_ATTR_ALIGNED(16);
    int32_t btmp[4] SIMD_ATTR_ALIGNED(16);
    int32_t tmp[4] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    tmp[0] = atmp[0] - atmp[1];
    tmp[1] = atmp[2] - atmp[3];
    tmp[2] = btmp[0] - btmp[1];
    tmp[3] = btmp[2] - btmp[3];
    return _mm_load_si128((__m128i*) tmp); 
  }

  static inline int16_t adds16(int16_t a, int16_t b)
  {
    int32_t s = int32_t(a) + int32_t(b);
    return (s < -0x8000) ? -0x8000 : (s > 0x7fff) ? 0x7fff : s;
  }

  static inline __m128i _mm_hadds_epi16(__m128i a, __m128i b)
  {
    int16_t atmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t btmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t tmp[8] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    tmp[0] = adds16(atmp[1], atmp[0]);
    tmp[1] = adds16(atmp[3], atmp[2]);
    tmp[2] = adds16(atmp[5], atmp[4]);
    tmp[3] = adds16(atmp[7], atmp[6]);
    tmp[4] = adds16(btmp[1], btmp[0]);
    tmp[5] = adds16(btmp[3], btmp[2]);
    tmp[6] = adds16(btmp[5], btmp[4]);
    tmp[7] = adds16(btmp[7], btmp[6]);
    return _mm_load_si128((__m128i*) tmp);
  }

  static inline int16_t subs16(int16_t a, int16_t b)
  {
    int32_t s = int32_t(a) - int32_t(b);
    return (s < -0x8000) ? -0x8000 : (s > 0x7fff) ? 0x7fff : s;
  }

  // 12. Aug 16 (rm): fixed bug: adds16->subs16
  static inline __m128i _mm_hsubs_epi16(__m128i a, __m128i b)
  {
    int16_t atmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t btmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t tmp[8] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    tmp[0] = subs16(atmp[0], atmp[1]);
    tmp[1] = subs16(atmp[2], atmp[3]);
    tmp[2] = subs16(atmp[4], atmp[5]);
    tmp[3] = subs16(atmp[6], atmp[7]);
    tmp[4] = subs16(btmp[0], btmp[1]);
    tmp[5] = subs16(btmp[2], btmp[3]);
    tmp[6] = subs16(btmp[4], btmp[5]);
    tmp[7] = subs16(btmp[6], btmp[7]);
    return _mm_load_si128((__m128i*) tmp);
  }

  static inline __m128i _mm_shuffle_epi8(__m128i a, __m128i mask)
  {
    int i;
    uint8_t atmp[16] SIMD_ATTR_ALIGNED(16);
    uint8_t masktmp[16] SIMD_ATTR_ALIGNED(16);
    uint8_t rtmp[16] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) masktmp, mask);
    for (i = 0; i < 16; i++)
      rtmp[i] = (masktmp[i] & 0x80) ? 0 : atmp[masktmp[i] & 0x0f];
    return _mm_load_si128((__m128i*) rtmp);
  }

  //  1. Oct 14 (rm): added
  static inline __m128i _mm_sign_epi16(__m128i a, __m128i b)
  {
    int i;
    int16_t atmp[8] SIMD_ATTR_ALIGNED(16);
    int16_t btmp[8] SIMD_ATTR_ALIGNED(16);
    _mm_store_si128((__m128i*) atmp, a);
    _mm_store_si128((__m128i*) btmp, b);
    for (i = 0; i < 8; i++)
      if (btmp[i] < 0)
	atmp[i] = -atmp[i];
    return _mm_load_si128((__m128i*) atmp);
  }

#endif

}

#endif // __SSE2__

#endif // _SSSE3_COMPAT_H_
