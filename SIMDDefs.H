// ===========================================================================
//
// SIMDDefs.H --
// encapsulates compiler- and architecture-specific definitions and constructs
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

#ifndef SIMD_DEFS_H_
#define SIMD_DEFS_H_

#include <iostream>

// TODO: implement and test for other compilers
// TODO: static inline on primary function templates useful?
// TODO: (static not allowed on function template specializations)
// TODO: MS world: inline __forceinline?

// ---------------------------------------------------------------------------
// preprocessor symbols
// ---------------------------------------------------------------------------

// NOTE: these symbols should be defined before including any T-SIMD header file
//       (e.g. by passing them to the compiler via the command line) otherwise
//       they might not have the desired effect

// SIMD_ALIGN_CHK:
// if defined, alignment checks of pointers are added in aligned load
// and store functions (if necessary on given architecture)

// MAX_SIMD_WIDTH:
// if defined, NATIVE_SIMD_WIDTH will be at most to this value
// must be at least 16, as smaller vector widths are not supported

// ---------------------------------------------------------------------------
// architecture-specific definitions
// ---------------------------------------------------------------------------

#if defined(__i386__) || defined(__x86_64__)
#define SIMDVEC_INTEL_ENABLE 1

#ifdef __AVX512F__
// this can be used to check whether 64-byte support is available
#define _SIMD_VEC_64_AVAIL_
#ifdef __AVX512BW__
// this can be used to check whether full 64-byte support is available
#define _SIMD_VEC_64_FULL_AVAIL_
#endif
#endif

#ifdef __AVX__
// this can be used to check whether 32-byte support is available
#define _SIMD_VEC_32_AVAIL_
#ifdef __AVX2__
// this can be used to check whether full 32-byte support is available
#define _SIMD_VEC_32_FULL_AVAIL_
#endif
#endif

#ifdef __SSE2__
#define _SIMD_VEC_16_AVAIL_
#endif

// determine NATIVE_SIMD_WIDTH
#ifdef _SIMD_VEC_64_AVAIL_
#define NATIVE_SIMD_WIDTH 64
#elif defined(_SIMD_VEC_32_AVAIL_)
#define NATIVE_SIMD_WIDTH 32
#else
#define NATIVE_SIMD_WIDTH 16
#endif

#endif // defined(__i386__) || defined(__x86_64__)

// 31. Mar 22 (rm): gcc on ARM doesn't define __arm__
// #if defined(__arm__) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
// 20. May 23 (Jonas Keller): apparently, some gcc versions in ARM *only* define
// __arm__, so added checks for more possible defines to cover as many compilers
// as possible
// #if defined(__ARM_NEON__) || defined(__ARM_NEON)
#if defined(__arm__) || defined(__aarch64__) || defined(__ARM_NEON__) ||       \
  defined(__ARM_NEON) || defined(_M_ARM) || defined(_M_ARM64)
#define SIMDVEC_NEON_ENABLE 1

#define _SIMD_VEC_16_AVAIL_

// determine NATIVE_SIMD_WIDTH
#define NATIVE_SIMD_WIDTH 16

#endif // arm defines

// 20. May 23 (Jonas Keller):
// added error message if no SIMD support was detected
#ifndef _SIMD_VEC_16_AVAIL_
#error "no SIMD support detected"
#endif

// set NATIVE_SIMD_WIDTH to be at most MAX_SIMD_WIDTH
#ifdef MAX_SIMD_WIDTH
#if NATIVE_SIMD_WIDTH > MAX_SIMD_WIDTH
#undef NATIVE_SIMD_WIDTH
#if MAX_SIMD_WIDTH >= 64
#define NATIVE_SIMD_WIDTH 64
#elif MAX_SIMD_WIDTH >= 32
#define NATIVE_SIMD_WIDTH 32
#elif MAX_SIMD_WIDTH >= 16
#define NATIVE_SIMD_WIDTH 16
#else
#error "MAX_SIMD_WIDTH must be at least 16"
#endif
#endif
#endif

// ---------------------------------------------------------------------------
// g++, clang++ and icc
// ---------------------------------------------------------------------------

// thanks to Wolfram Schenck for icc tests
// 30. Aug 22 (Jonas Keller): clang++ on windows does not define __GNUC__
// #ifdef __GNUC__
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define SIMD_DEFS_DEFINED
#define SIMD_INLINE              inline __attribute__((always_inline))
#define SIMD_ATTR_ALIGNED(ALIGN) __attribute__((aligned(ALIGN)))
#define SIMD_ATTR_PACKED_STRUCT  struct __attribute__((__packed__))
#define SIMD_FULL_MEMBARRIER     __sync_synchronize()
#endif

// TODO: MSVC?

// ---------------------------------------------------------------------------
// compiler-independent stuff
// ---------------------------------------------------------------------------

#ifndef SIMD_DEFS_DEFINED
#error "attribute definitions missing for this compiler"
#endif

// macro to set the default value SIMD_WIDTH template parameter to
// NATIVE_WIDTH if NATIVE_WIDTH is defined
#ifdef NATIVE_SIMD_WIDTH
#define SIMD_WIDTH_DEFAULT_NATIVE SIMD_WIDTH = NATIVE_SIMD_WIDTH
#else
#define SIMD_WIDTH_DEFAULT_NATIVE SIMD_WIDTH
#endif

// for reinterpretation of memory
// code like this leads to complaints about type-punning with -O3:
// return *(Float*) &i;
#if 0
// ========== solution with union ==========
// may not be supported by all compilers
// writing to an element of a union and reading a different element is
// undefined in C++ but seems to be supported by many compilers
// http://cellperformance.beyond3d.com/articles/2006/06/
//   understanding-strict-aliasing.html
// http://dbp-consulting.com/tutorials/StrictAliasing.html
#define SIMD_RETURN_REINTERPRET_INT(NEWTYPE, SOURCE, RETELEM)                  \
  union {                                                                      \
    Int i;                                                                     \
    NEWTYPE v[sizeof(Int) / sizeof(NEWTYPE)];                                  \
  } u;                                                                         \
  u.i = SOURCE;                                                                \
  return u.v[RETELEM];
#else
// ========== solution with memcpy ==========
// recommended, hopefully optimized away
#define SIMD_RETURN_REINTERPRET_INT(NEWTYPE, SOURCE, RETELEM)                  \
  NEWTYPE v[sizeof(Int) / sizeof(NEWTYPE)];                                    \
  Int i = SOURCE;                                                              \
  memcpy(v, &i, sizeof(v));                                                    \
  return v[RETELEM];
#endif

// macros for handling multiples of a power of 2
#define SIMD_IS_MULTIPLE_OF(X, POWER_OF_2)   (((X) & ((POWER_OF_2) -1)) == 0)
#define SIMD_PREV_MULTIPLE_OF(X, POWER_OF_2) ((X) & (~((POWER_OF_2) -1)))
#define SIMD_NEXT_MULTIPLE_OF(X, POWER_OF_2)                                   \
  (SIMD_PREV_MULTIPLE_OF((X), (POWER_OF_2)) + (POWER_OF_2))

// macros to compute how many of NDATA elements of a data vector
// can be held in SIMD vectors of NSIMD (a power of two!) elements
// FLOOR: all SIMD vectors are full, but there may be excess elements
// CEIL:  SIMD vectors hold all elements, but some at the end are unused
#define SIMD_ELEMENTS_FLOOR(NSIMD, NDATA) ((NDATA) & ~((NSIMD) -1))
#define SIMD_ELEMENTS_CEIL(NSIMD, NDATA)                                       \
  (((NDATA) + ((NSIMD) -1)) & ~((NSIMD) -1))

// macros to compute how many SIMD vectors with NSIMD elements each
// are required to hold NDATA elements
// FLOOR: number of completely filled SIMD vectors (excess elements may exist)
// CEIL: number of SIMD vectors required to hold all elements (some unused)
#define SIMD_VECTORS_FLOOR(NSIMD, NDATA) (unsigned(NDATA) / unsigned(NSIMD))
#define SIMD_VECTORS_CEIL(NSIMD, NDATA)                                        \
  ((unsigned(NDATA) + (unsigned(NSIMD) - 1u)) / unsigned(NSIMD))

// 29. Mar 23 (Jonas Keller): added SIMD_ENABLE_IF macro

// macros to make enable_if SFINAE more convenient and readable
#define SIMD_ENABLE_IF(EXPR) typename = typename std::enable_if<(EXPR)>::type

// 08. Apr 23 (Jonas Keller): added SIMD_CHECK_ALIGNMENT macro

// macro to check alignment of a pointer
// does nothing if SIMD_ALIGN_CHK is not defined
// does not use assert() because assert() may be disabled (e.g. in release
// builds)

#ifdef SIMD_ALIGN_CHK
#define SIMD_CHECK_ALIGNMENT(PTR, ALIGN)                                       \
  if (reinterpret_cast<uintptr_t>(PTR) % (ALIGN) != 0) {                       \
    ::std::cerr << "SIMD_CHECK_ALIGNMENT: " << __FILE__ << ":" << __LINE__     \
                << ": " << __func__ << ": " << #PTR << " = " << (PTR)          \
                << " is not aligned to " << (ALIGN) << " bytes"                \
                << ::std::endl;                                                \
    ::std::abort();                                                            \
  }
#else
#define SIMD_CHECK_ALIGNMENT(PTR, ALIGN) ((void) 0)
#endif

#endif
