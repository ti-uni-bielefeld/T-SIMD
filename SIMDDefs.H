// ===========================================================================
// 
// SIMDDefs.H --
// encapsulates compiler-specific constructs
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

#ifndef _SIMD_DEFS_H_
#define _SIMD_DEFS_H_

// TODO: implement and test for other compilers
// TODO: static inline on primary function templates useful?
// TODO: (static not allowed on function template specializations)
// TODO: MS world: inline __forceinline?

// ---------------------------------------------------------------------------
// preprocessor symbols
// ---------------------------------------------------------------------------

// SIMD_STRICT_SATURATION:
// if defined, non-saturated work-arounds for saturated functions are
// not included (typically for SIMDInt and SIMDFloat, e.g. in adds, hadds)

// SIMD_ALIGN_CHK:
// if defined, alignment checks of pointers are added in aligned load
// and store functions (if necessary on given architecture)

// ---------------------------------------------------------------------------
// g++
// ---------------------------------------------------------------------------

// TODO: icc and clang++ also define __GNUC__: make this one specific for g++?
// 30. Aug 22 (Jonas Keller): clang++ on windows does not define __GNUC__
//#ifdef __GNUC__
#if defined(__GNUC__) || defined(__clang__)
#define SIMD_DEFS_DEFINED
#define SIMD_INLINE inline __attribute__ ((always_inline))
#define SIMD_ATTR_ALIGNED(ALIGN) __attribute__ ((aligned(ALIGN)))
#define SIMD_ATTR_PACKED_STRUCT struct __attribute__ ((__packed__))
#define SIMD_FULL_MEMBARRIER __sync_synchronize()
#endif

// ---------------------------------------------------------------------------
// icc
// ---------------------------------------------------------------------------

// thanks to Wolfram Schenck for icc tests
#ifdef __INTEL_COMPILER
#define SIMD_DEFS_DEFINED
#define SIMD_INLINE inline __attribute__ ((always_inline))
#define SIMD_ATTR_ALIGNED(ALIGN) __attribute__ ((aligned(ALIGN)))
#define SIMD_ATTR_PACKED_STRUCT struct __attribute__ ((__packed__))
#define SIMD_FULL_MEMBARRIER __sync_synchronize()
#endif

// ---------------------------------------------------------------------------
// compiler-independent stuff
// ---------------------------------------------------------------------------

#ifndef SIMD_DEFS_DEFINED
#error "attribute definitions missing for this compiler"
#endif

// C++ 11 support?
#if __cplusplus >= 201103L
#define SIMD_CONSTEXPR constexpr
#else
#define SIMD_CONSTEXPR const
#endif

// for reinterpretation of memory
// code like this leads to complaints about type-punning with -O3:
// return *(SIMDFloat*) &i;
#if 0
// ========== solution with union ==========
// may not be supported by all compilers
// writing to an element of a union and reading a different element is
// undefined in C++ but seems to be supported by many compilers
// http://cellperformance.beyond3d.com/articles/2006/06/
//   understanding-strict-aliasing.html
// http://dbp-consulting.com/tutorials/StrictAliasing.html
#define SIMD_RETURN_REINTERPRET_INT(NEWTYPE,SOURCE,RETELEM)		\
  union { SIMDInt i; NEWTYPE v[sizeof(SIMDInt)/sizeof(NEWTYPE)]; } u;	\
  u.i = SOURCE;								\
  return u.v[RETELEM];
#else
// ========== solution with memcpy ==========
// recommended, hopefully optimized away
#define SIMD_RETURN_REINTERPRET_INT(NEWTYPE,SOURCE,RETELEM)		\
  NEWTYPE v[sizeof(SIMDInt)/sizeof(NEWTYPE)];				\
  SIMDInt i = SOURCE;							\
  memcpy(v, &i, sizeof(v));						\
  return v[RETELEM];
#endif

// macros for handling multiples of a power of 2
#define SIMD_IS_MULTIPLE_OF(X,POWER_OF_2) \
  (((X) & ((POWER_OF_2) - 1)) == 0)
#define SIMD_PREV_MULTIPLE_OF(X,POWER_OF_2) \
  ((X) & (~((POWER_OF_2) - 1)))
#define SIMD_NEXT_MULTIPLE_OF(X,POWER_OF_2) \
  (SIMD_PREV_MULTIPLE_OF((X),(POWER_OF_2)) + (POWER_OF_2))

// macros to compute how many of NDATA elements of a data vector
// can be held in SIMD vectors of NSIMD (a power of two!) elements
// FLOOR: all SIMD vectors are full, but there may be excess elements
// CEIL:  SIMD vectors hold all elements, but some at the end are unused
#define SIMD_ELEMENTS_FLOOR(NSIMD,NDATA) ((NDATA) & ~((NSIMD)-1))
#define SIMD_ELEMENTS_CEIL(NSIMD,NDATA) (((NDATA)+((NSIMD)-1)) & ~((NSIMD)-1))

#endif
