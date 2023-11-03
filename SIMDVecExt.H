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

// 22. Jan 23 (Jonas Keller): moved internal implementations into internal
// namespace

// 03. Mar 23 (Jonas Keller): removed hsub and hsubs extensions, as they do not
// do something useful

// 09. Mar 23 (Jonas Keller): added doxygen documentation

// 03. Aug 23 (Jonas Keller): renamed all swizzle2, unswizzle and transpose
// versions to sequential names <base_name>_a, <base_name>_b, <base_name>_c,
// etc. and moved them into the internal namespace and added a hub function
// <base_name> that wraps the fastest version

// 13. May 23 (Jonas Keller): added Double support

#pragma once
#ifndef SIMD_VEC_EXT_H_
#define SIMD_VEC_EXT_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecExtTransposeAutogen.H"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>

namespace simd {
namespace internal {
namespace ext {
// https://stackoverflow.com/questions/23781506/compile-time-computing-of-number-of-bits-needed-to-encode-n-different-states
template <typename T>
static constexpr SIMD_INLINE T floorlog2(T x)
{
  static_assert(std::is_integral<T>::value, "");
  return x == 1 ? 0 : 1 + floorlog2(x >> 1);
}
} // namespace ext
} // namespace internal

// determine NATIVE_SIMD_REG_COUNT
// https://stackoverflow.com/questions/62419256/how-can-i-determine-how-many-avx-registers-my-processor-has
/// @cond
#ifdef __x86_64__
#ifdef __AVX512VL__
#define NATIVE_SIMD_REG_COUNT 32
#else
#define NATIVE_SIMD_REG_COUNT 16
#endif
#else
#define NATIVE_SIMD_REG_COUNT 8
#endif
/// @endcond

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
// // Float: map -0.0f to 0.0f
// static SIMD_INLINE Float
// treatZero(Float in)
// {
//   return (in == -0.0f) ? 0.0f : in;
// }

/**
 * @addtogroup group_print
 * @{
 */

/**
 * @brief Writes the formatted elements of a Vec to a file.
 *
 * The elements are formatted using the format string and written to
 * the file using the fprintf function of the C standard library in the order
 * they are stored in memory.
 *
 * @param f file to write to
 * @param format format string Must be a valid format string to print a single
 * element of the Vec using the fprintf function of the C standard library
 * @param vec Vec to print
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void fprint(FILE *f, const char *format,
                               const Vec<T, SIMD_WIDTH> &vec)
{
  // buffer
  // 19. Jul 16 (rm)
  const auto elems = Vec<T, SIMD_WIDTH>::elems; // SIMD_WIDTH/sizeof(T)
  // T buf[SIMD_WIDTH];
  T buf[elems];
  // store vector (unaligned, not time-critical)
  storeu(buf, vec);
  // print elements of vector to f
  for (size_t i = 0; i < elems; i++)
    // 04. Aug 22 (Jonas Keller):
    // removed mapping from -0.0f to 0.0f,
    // for debugging you want to see -0.0f
    // fprintf(f, format, treatZero(buf[i]));
    fprintf(f, format, buf[i]);
}

/**
 * @brief Writes the formatted elements of a Vec to stdout.
 *
 * Equivalent to calling fprint(FILE *f, const char *format,
 * const Vec<T,SIMD_WIDTH> &vec) with stdout as the file.
 *
 * @param format format string Must be a valid format string to print a single
 * element of the Vec using the fprintf function of the C standard library
 * @param vec Vec to print
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void print(const char *format, const Vec<T, SIMD_WIDTH> &vec)
{
  fprint(stdout, format, vec);
}

/**
 * @brief Writes the formatted elements of a Vec to a file separated
 * by a separator string.
 *
 * Equivalent to calling fprint(FILE *f, const char *format,
 * const Vec<T,SIMD_WIDTH> &vec) with a format string
 * consisting of the format string and the separator string.
 *
 * @param f file to write to
 * @param format format string Must be a valid format string to print a single
 * element of the Vec using the fprintf function of the C standard library
 * @param separator separator string
 * @param vec Vec to print
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void fprint(FILE *f, const char *format,
                               const char *separator,
                               const Vec<T, SIMD_WIDTH> &vec)
{
  // 09. Jan 23 (Jonas Keller): used std::string instead of strcpy and strcat
  // to avoid potential buffer overflows
  // char fmtSep[256];
  // strcat(strcpy(fmtSep, format), separator);
  std::string fmtSep = std::string(format) + std::string(separator);
  fprint(f, fmtSep.c_str(), vec);
}

/**
 * @brief Writes the formatted elements of a Vec to stdout separated
 * by a separator string.
 *
 * Equivalent to calling fprint(FILE *f, const char *format,
 * const char *separator, const Vec<T,SIMD_WIDTH> &vec) with stdout as
 * the file.
 *
 * @param format format string Must be a valid format string to print a single
 * element of the Vec using the fprintf function of the C standard library
 * @param separator separator string
 * @param vec Vec to print
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void print(const char *format, const char *separator,
                              const Vec<T, SIMD_WIDTH> &vec)
{
  fprint(stdout, format, separator, vec);
}

/** @} */

// ===========================================================================
// multi-vector store and load
// ===========================================================================

/**
 * @ingroup group_memory_load
 * @brief Loads multiple successive Vec's from aligned memory.
 *
 * The memory location must be aligned to the SIMD_WIDTH.
 *
 * @param[in] p pointer to the aligned memory location to load from
 * @param[out] inVecs array of Vec's to store the loaded values in
 * @param numInVecs number of Vec's to load
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void load(const T *const p, Vec<T, SIMD_WIDTH> inVecs[],
                             size_t numInVecs)
{
  for (size_t i = 0; i < numInVecs; i++) {
    inVecs[i] = load<SIMD_WIDTH>(&p[i * Vec<T, SIMD_WIDTH>::elems]);
  }
}

/**
 * @ingroup group_memory_load
 * @brief Loads multiple successive Vec's from unaligned memory.
 *
 * In contrast to load(const T *const, Vec<T, SIMD_WIDTH>[], size_t), the
 * memory location does not need to be aligned to any boundary.
 *
 * @param[in] p pointer to the memory location to load from
 * @param[out] inVecs array of Vec's to store the loaded values in
 * @param numInVecs number of Vec's to load
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void loadu(const T *const p, Vec<T, SIMD_WIDTH> inVecs[],
                              size_t numInVecs)
{
  for (size_t i = 0; i < numInVecs; i++) {
    inVecs[i] = loadu<SIMD_WIDTH>(&p[i * Vec<T, SIMD_WIDTH>::elems]);
  }
}

/**
 * @ingroup group_memory_store
 * @brief Stores multiple successive Vec's to aligned memory.
 *
 * The memory location must be aligned to the SIMD_WIDTH.
 *
 * @param[out] p pointer to the aligned memory location to store to
 * @param[in] outVecs array of Vec's to store
 * @param numOutVecs number of Vec's to store
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void store(T *const p, const Vec<T, SIMD_WIDTH> outVecs[],
                              size_t numOutVecs)
{
  for (size_t i = 0; i < numOutVecs; i++) {
    store(&p[i * Vec<T, SIMD_WIDTH>::elems], outVecs[i]);
  }
}

/**
 * @ingroup group_memory_store
 * @brief Stores multiple successive Vec's to unaligned memory.
 *
 * In contrast to store(T *const, const Vec<T, SIMD_WIDTH>[], size_t), the
 * memory location does not need to be aligned to any boundary.
 *
 * @param[out] p pointer to the memory location to store to
 * @param[in] outVecs array of Vec's to store
 * @param numOutVecs number of Vec's to store
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void storeu(T *const p, const Vec<T, SIMD_WIDTH> outVecs[],
                               size_t numOutVecs)
{
  for (size_t i = 0; i < numOutVecs; i++) {
    storeu(&p[i * Vec<T, SIMD_WIDTH>::elems], outVecs[i]);
  }
}

/**
 * @ingroup group_memory_store
 * @brief Stores a single Vec multiple times to aligned memory.
 *
 * The memory location must be aligned to the SIMD_WIDTH.
 *
 * @param[out] p pointer to the aligned memory location to store to
 * @param[in] outVec Vec to store
 * @param numOutVecs number of times to store the Vec
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void store(T *const p, const Vec<T, SIMD_WIDTH> &outVec,
                              size_t numOutVecs)
{
  for (size_t i = 0; i < numOutVecs; i++) {
    store(&p[i * Vec<T, SIMD_WIDTH>::elems], outVec);
  }
}

/**
 * @ingroup group_memory_store
 * @brief Stores a single Vec multiple times to unaligned memory.
 *
 * In contrast to store(T *const p, const Vec<T, SIMD_WIDTH> &outVec,
 * size_t numOutVecs), the memory location does not need to be aligned to
 * any boundary.
 *
 * @param[out] p pointer to the memory location to store to
 * @param[in] outVec Vec to store
 * @param numOutVecs number of times to store the Vec
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void storeu(T *const p, const Vec<T, SIMD_WIDTH> &outVec,
                               size_t numOutVecs)
{
  for (size_t i = 0; i < numOutVecs; i++) {
    storeu(&p[i * Vec<T, SIMD_WIDTH>::elems], outVec);
  }
}

// -------------------- different store functions ----------------------------
namespace internal {
namespace ext {
template <typename T, size_t SIMD_WIDTH>
struct Store
{
  static SIMD_INLINE void _store(T *const p, const Vec<T, SIMD_WIDTH> &outVec)
  {
    return store(p, outVec);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct StoreU
{
  static SIMD_INLINE void _store(T *const p, const Vec<T, SIMD_WIDTH> &outVec)
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
// always_inline ‘void storeu16(..) [..]’: recursive inlining
// ---------------------------------------------------------------------------

template <template <typename, size_t> class Store, typename T,
          size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW, size_t STORE_STOP,
          size_t STORE_WIDTH, size_t SRC_OFF, size_t DST_OFF>
struct Store16
{
  static SIMD_INLINE void _store16(
    T *const p, const Vec<T, SIMD_WIDTH> outVecs[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("STORE_WIDTH=%d, SRC_OFFSET=%d, DST_OFFSET=%d\n",
    // STORE_WIDTH, SRC_OFFSET, DST_OFFSET);
    Store16<Store, T, SIMD_WIDTH, NUMROWS, ROW, STORE_STOP, STORE_WIDTH / 2,
            SRC_OFF, 2 * DST_OFF>::_store16(p, outVecs);
    Store16<Store, T, SIMD_WIDTH, NUMROWS, ROW, STORE_STOP, STORE_WIDTH / 2,
            SRC_OFF + SIMD_WIDTH / STORE_WIDTH,
            2 * DST_OFF + STORE_STOP>::_store16(p, outVecs);
  }
};

template <template <typename, size_t> class Store, typename T,
          size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW, size_t STORE_STOP,
          size_t SRC_OFF, size_t DST_OFF>
struct Store16<Store, T, SIMD_WIDTH, NUMROWS, ROW, STORE_STOP, 16, SRC_OFF,
               DST_OFF>
{
  static constexpr auto STEP = SIMD_WIDTH / 16;
  static constexpr auto VO   = SRC_OFF + ROW * STEP;
  static constexpr auto OFF  = (DST_OFF + ROW) * NUMROWS;

  static SIMD_INLINE void _store16(
    T *const p, const Vec<T, SIMD_WIDTH> outVecs[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("VO=%d\n", OFF=%d\n", VO, OFF);
    Store<T, SIMD_WIDTH>::_store(p + OFF, outVecs[VO]);
    Store16<Store, T, SIMD_WIDTH, NUMROWS, ROW + 1, STORE_STOP, 16, SRC_OFF,
            DST_OFF>::_store16(p, outVecs);
  }
};

template <template <typename, size_t> class Store, typename T,
          size_t SIMD_WIDTH, size_t NUMROWS, size_t STORE_STOP, size_t SRC_OFF,
          size_t DST_OFF>
struct Store16<Store, T, SIMD_WIDTH, NUMROWS, STORE_STOP, STORE_STOP, 16,
               SRC_OFF, DST_OFF>
{
  static SIMD_INLINE void _store16(
    T *const, const Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems])
  {}
};

// -------------------- store16 functions ------------------------------------

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void store16(
  T *const p, const Vec<T, SIMD_WIDTH> outVecs[Vec<T, SIMD_WIDTH>::elems])
{
  const auto numRows   = SIMD_WIDTH / sizeof(T);
  const auto storeStop = 16 / sizeof(T);
  internal::ext::Store16<internal::ext::Store, T, SIMD_WIDTH, numRows, 0,
                         storeStop, SIMD_WIDTH, 0, 0>::_store16(p, outVecs);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void storeu16(
  T *const p, const Vec<T, SIMD_WIDTH> outVecs[Vec<T, SIMD_WIDTH>::elems])
{
  const auto numRows   = SIMD_WIDTH / sizeof(T);
  const auto storeStop = 16 / sizeof(T);
  internal::ext::Store16<internal::ext::StoreU, T, SIMD_WIDTH, numRows, 0,
                         storeStop, SIMD_WIDTH, 0, 0>::_store16(p, outVecs);
}

} // namespace ext
} // namespace internal

// ===========================================================================
// copy (load and store)
// ===========================================================================

/**
 * @addtogroup group_memory
 * @{
 */

/**
 * @brief Copies a single Vec from one aligned memory location to
 * another aligned memory location.
 *
 * Both memory locations must be aligned to the SIMD_WIDTH.
 *
 * @param[in] src pointer to the aligned source memory location
 * @param[out] dst pointer to the aligned destination memory location
 */
template <size_t SIMD_WIDTH, typename T>
static SIMD_INLINE void load_store(const T *const src, T *const dst)
{
  Vec<T, SIMD_WIDTH> copy = load<SIMD_WIDTH>(src);
  store(dst, copy);
}

/**
 * @brief Copies a single Vec from one unaligned memory location
 * to another aligned memory location.
 *
 * The destination memory location must be aligned to the SIMD_WIDTH, the
 * source memory location does not have to be aligned to any boundary.
 *
 * @param[in] src pointer to the unaligned source memory location
 * @param[out] dst pointer to the aligned destination memory location
 */
template <size_t SIMD_WIDTH, typename T>
static SIMD_INLINE void loadu_store(const T *const src, T *const dst)
{
  Vec<T, SIMD_WIDTH> copy = loadu<SIMD_WIDTH>(src);
  store(dst, copy);
}

/**
 * @brief Copies a single Vec from one aligned memory location to
 * another unaligned memory location.
 *
 * The source memory location must be aligned to the SIMD_WIDTH, the
 * destination memory location does not have to be aligned to any boundary.
 *
 * @param[in] src pointer to the aligned source memory location
 * @param[out] dst pointer to the unaligned destination memory location
 */
template <size_t SIMD_WIDTH, typename T>
static SIMD_INLINE void load_storeu(const T *const src, T *const dst)
{
  Vec<T, SIMD_WIDTH> copy = load<SIMD_WIDTH>(src);
  storeu(dst, copy);
}

/**
 * @brief Copies a single Vec from one unaligned memory location
 * to another unaligned memory location.
 *
 * Both memory locations do not have to be aligned to any boundary.
 *
 * @param[in] src pointer to the unaligned source memory location
 * @param[out] dst pointer to the unaligned destination memory location
 */
template <size_t SIMD_WIDTH, typename T>
static SIMD_INLINE void loadu_storeu(const T *const src, T *const dst)
{
  Vec<T, SIMD_WIDTH> copy = loadu<SIMD_WIDTH>(src);
  storeu(dst, copy);
}

/** @} */

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

namespace internal {
namespace ext {

// no stage

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> packs(const Vec<T, SIMD_WIDTH> a[1],
                                            OutputType<T>, Compression<1>)
{
  return a[0];
}

// single stage

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(const Vec<Tin, SIMD_WIDTH> a[2],
                                               OutputType<Tout>, Compression<2>)
{
  return packs<Tout>(a[0], a[1]);
}

// two stages

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(const Vec<Tin, SIMD_WIDTH> a[4],
                                               OutputType<Tout>, Compression<4>)
{
  // always via Short
  return packs<Tout>(packs<Short>(a[0], a[1]), packs<Short>(a[2], a[3]));
}

// two stages from Double

template <typename Tout, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(
  const Vec<Double, SIMD_WIDTH> a[4], OutputType<Tout>, Compression<4>)
{
  // always via Int
  return packs<Tout>(packs<Int>(a[0], a[1]), packs<Int>(a[2], a[3]));
}

// three stages

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(const Vec<Tin, SIMD_WIDTH> a[8],
                                               OutputType<Tout>, Compression<8>)
{
  // always via Short
  // assuming Tin is Double:
  static_assert(std::is_same<Tin, Double>::value,
                "currently only supported for Tin == Double");
  return packs<Tout>(packs<Short>(a), packs<Short>(a + 4));
}

// special cases: int <-> float, long <-> double

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(const Vec<Tin, SIMD_WIDTH> a[1],
                                               OutputType<Tout>, Compression<1>)
{
  static_assert(sizeof(Tin) == sizeof(Tout), "");
  static_assert(std::is_floating_point<Tin>::value !=
                  std::is_floating_point<Tout>::value,
                "");
  return cvts<Tout>(a[0]);
}

} // namespace ext
} // namespace internal

// generalized version of packs: includes multistage packing

/**
 * @ingroup group_type_conversion
 * @brief Packs multiple Vec's into a single Vec by converting the
 * elements into smaller or equally sized types.
 *
 * TODO: allowed types
 *
 * In contrast to packs(const Vec<Tin, SIMD_WIDTH>&, const Vec<Tin,
 * SIMD_WIDTH>&), this function can handle multistage packing, i.e. the
 * number of input can be different from 2.
 *
 * @sa packs(const Vec<Tin, SIMD_WIDTH>&, const Vec<Tin,
 * SIMD_WIDTH>&)
 *
 * @tparam Tout type of the resulting Vec
 * @tparam Tin type of the input Vec's
 * @param[in] a array of input Vec's
 * @return Vec with the packed elements
 */
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(
  const Vec<Tin, SIMD_WIDTH> a[sizeof(Tin) / sizeof(Tout)])
{
  return internal::ext::packs(
    a, internal::OutputType<Tout>(),
    internal::Compression<sizeof(Tin) / sizeof(Tout)>());
}

// ===========================================================================
// generalized convert (using extend and packs)
// ===========================================================================

// here we distinguish between the three cases (type size comparison)
// using the CompareTypes mechanisms (from SIMDTypes.H);
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

namespace internal {
namespace ext {
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void convert(
  CompareLess, const Vec<Tin, SIMD_WIDTH> inVecs[sizeof(Tin) / sizeof(Tout)],
  Vec<Tout, SIMD_WIDTH> outVecs[1])
{
  outVecs[0] = packs<Tout>(inVecs);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void convert(CompareEqual,
                                const Vec<Tin, SIMD_WIDTH> inVecs[1],
                                Vec<Tout, SIMD_WIDTH> outVecs[1])
{
  extend(inVecs[0], outVecs);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void convert(
  CompareGreater, const Vec<Tin, SIMD_WIDTH> inVecs[1],
  Vec<Tout, SIMD_WIDTH> outVecs[sizeof(Tout) / sizeof(Tin)])
{
  extend(inVecs[0], outVecs);
}
} // namespace ext
} // namespace internal

/**
 * @ingroup group_type_conversion
 * @brief Converts (potentially multiple) Vec's between different types.
 *
 * Combines extend() and packs().
 *
 * TODO: allowed types
 *
 * If the input and output types are of the same size, both the input and
 * output are just one Vec. If the types are of different sizes, the input or
 * output consists of multiple Vec's, such that the number of input elements
 * is equal to the number of output elements.
 *
 * @sa extend()
 * @sa packs()
 *
 * @tparam Tout element type of the output Vec's
 * @tparam Tin element type of the input Vec's
 * @param[in] inVecs input Vec's
 * @param[out] outVecs output Vec's
 */
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void convert(
  const Vec<Tin, SIMD_WIDTH> inVecs[numInVecs<Tout, Tin>()],
  Vec<Tout, SIMD_WIDTH> outVecs[numOutVecs<Tout, Tin>()])
{
  internal::ext::convert(internal::CompareTypes<Tout, Tin>(), inVecs, outVecs);
}

// ===========================================================================
// float-based operations on arbitrary input and output types
// ===========================================================================

// internal helper functions for float-based operations:
namespace internal {
namespace ext {
template <typename Tin, typename Tout, typename Tfloat>
static constexpr SIMD_INLINE size_t numCalcVecs()
{
  static_assert(numInVecs<Tout, Tin>() * sizeof(Tfloat) / sizeof(Tin) ==
                  numOutVecs<Tout, Tin>() * sizeof(Tfloat) / sizeof(Tout),
                "numCalcVecs() must be equal for input and output");
  return numInVecs<Tout, Tin>() * sizeof(Tfloat) / sizeof(Tin);
}

template <typename Tin, typename Tout, typename Tfloat, size_t SIMD_WIDTH>
static SIMD_INLINE void extendInToFloat(
  const Vec<Tin, SIMD_WIDTH> inVecs[numInVecs<Tout, Tin>()],
  Vec<Tfloat, SIMD_WIDTH> floatVecs[numCalcVecs<Tout, Tin, Tfloat>()])
{
  for (size_t i = 0; i < numInVecs<Tout, Tin>(); ++i) {
    extend(inVecs[i], &floatVecs[i * sizeof(Tfloat) / sizeof(Tin)]);
  }
}

template <typename Tin, typename Tout, typename Tfloat, size_t SIMD_WIDTH>
static SIMD_INLINE void packsOutFromFloat(
  const Vec<Tfloat, SIMD_WIDTH> floatVecs[numCalcVecs<Tout, Tin, Tfloat>()],
  Vec<Tout, SIMD_WIDTH> outVecs[numOutVecs<Tout, Tin>()])
{
  for (size_t i = 0; i < numOutVecs<Tout, Tin>(); ++i) {
    outVecs[i] = packs<Tout>(&floatVecs[i * sizeof(Tfloat) / sizeof(Tout)]);
  }
}
} // namespace ext
} // namespace internal

/**
 * @addtogroup group_fops
 * @{
 */

// ---------------------------------------------------------------------------
// divide then multiply with float constant in float arithmetic
// ---------------------------------------------------------------------------

// TODO: fdivmul: better fmuldiv = first multiply then divide?

// 15. Mar 23 (Jonas Keller): fused the three cases

/**
 * @brief Divides Vec's element-wise, then multiplies with a constant factor in
 * floating point arithmetic.
 *
 * @tparam Tfloat the floating point type to perform the calculation in.
 * Defaults to @ref BigEnoughFloat<Tout, Tin>
 * @param[in] vecsNum numerator Vec's
 * @param[in] vecsDenom denominator Vec's
 * @param fac factor to multiply with
 * @param[out] vecsOut output Vec's
 * @sa numInVecs(), numOutVecs()
 */
template <typename Tout, typename Tin,
          typename Tfloat = BigEnoughFloat<Tout, Tin>, size_t SIMD_WIDTH>
static SIMD_INLINE void fdivmul(
  const Vec<Tin, SIMD_WIDTH> vecsNum[numInVecs<Tout, Tin>()],
  const Vec<Tin, SIMD_WIDTH> vecsDenom[numInVecs<Tout, Tin>()],
  dont_deduce<Tfloat> fac,
  Vec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  static_assert(sizeof(Tin) <= sizeof(Tfloat), "");
  static_assert(sizeof(Tout) <= sizeof(Tfloat), "");
  constexpr auto nFloatVecs = internal::ext::numCalcVecs<Tout, Tin, Tfloat>();
  const auto facVec         = set1<Tfloat, SIMD_WIDTH>(fac);
  Vec<Tfloat, SIMD_WIDTH> numF[nFloatVecs], denomF[nFloatVecs],
    resF[nFloatVecs];
  internal::ext::extendInToFloat<Tin, Tout>(vecsNum, numF);
  internal::ext::extendInToFloat<Tin, Tout>(vecsDenom, denomF);
  for (size_t i = 0; i < nFloatVecs; i++) {
    resF[i] = mul(div(numF[i], denomF[i]), facVec);
  }
  internal::ext::packsOutFromFloat<Tin, Tout>(resF, vecsOut);
}

// ---------------------------------------------------------------------------
// divide, apply multidimensional sigmoid and then multiply with float
// constant in float arithmetic (derived from fdivmul)
// sigmoid(x) = ((y(x,a)/(1-y(x,a)**4)**0.25))+1)/2)
// y(x,a) = sum_d(a*w[d]*(x[d]-w0[d]))
// a = -0.433 from fitting this to 1/(1+exp(y(x,1))
// ---------------------------------------------------------------------------

namespace internal {
namespace ext {
template <size_t DIM, size_t NVEC, typename Tout, typename Tin,
          size_t SIMD_WIDTH>
static SIMD_INLINE void fdivMsigmoidmul(
  CompareLess, const Vec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
  const Vec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC], const double w[DIM],
  const double w0[DIM], double fac, Vec<Tout, SIMD_WIDTH> vecsOut[1])
{
  const auto nIn   = sizeof(Tin) / sizeof(Tout);
  const auto fanIn = sizeof(Float) / sizeof(Tin);
  const auto facF  = set1<Float, SIMD_WIDTH>(fac / 2.0);
  const auto oneF  = set1<Float, SIMD_WIDTH>(1.0f);
  Vec<Float, SIMD_WIDTH> wF[DIM], w0F[DIM], numF[DIM][fanIn],
    denomF[DIM][fanIn], resF[nIn * fanIn];
  for (size_t d = 0; d < DIM; d++) {
    wF[d]  = set1<Float, SIMD_WIDTH>(-0.433 * w[d]);
    w0F[d] = set1<Float, SIMD_WIDTH>(w0[d]);
  }
  // i: index of input vector
  // j: index of extended input vector
  // k: index of output vectors
  // TODO: sometimes i < nIn does not work with -O2 is always true?
  for (size_t i = 0, k = 0; i < nIn; i++) {
    for (size_t d = 0; d < DIM; d++) {
      extend(vecsNum[d][i], numF[d]);
      extend(vecsDenom[d][i], denomF[d]);
    }
    for (size_t j = 0; j < fanIn; j++, k++) {
      auto yF = setzero<Float, SIMD_WIDTH>();
      for (size_t d = 0; d < DIM; d++) {
        yF = add(yF, mul(wF[d], sub(div(numF[d][j], denomF[d][j]), w0F[d])));
      }
      auto y4F = mul(yF, yF);
      y4F      = mul(y4F, y4F);
      resF[k]  = mul(add(div(yF, sqrt(sqrt(add(oneF, y4F)))), oneF), facF);
    }
  }
  vecsOut[0] = packs<Tout>(resF);
}

template <size_t DIM, size_t NVEC, typename Tout, typename Tin,
          size_t SIMD_WIDTH>
static SIMD_INLINE void fdivMsigmoidmul(
  CompareGreater, const Vec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
  const Vec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC], const double w[DIM],
  const double w0[DIM], double fac,
  Vec<Tout, SIMD_WIDTH> vecsOut[sizeof(Tout) / sizeof(Tin)])
{
  const auto nOut   = sizeof(Tout) / sizeof(Tin);
  const auto fanOut = sizeof(Float) / sizeof(Tout);
  const auto facF   = set1<Float, SIMD_WIDTH>(fac / 2.0);
  const auto oneF   = set1<Float, SIMD_WIDTH>(1.0f);
  Vec<Float, SIMD_WIDTH> wF[DIM], w0F[DIM], numF[DIM][nOut * fanOut],
    denomF[DIM][nOut * fanOut], resF[fanOut];
  for (size_t d = 0; d < DIM; d++) {
    wF[d]  = set1<Float, SIMD_WIDTH>(-0.433 * w[d]);
    w0F[d] = set1<Float, SIMD_WIDTH>(w0[d]);
    extend(*vecsNum[d], numF[d]);
    extend(*vecsDenom[d], denomF[d]);
  }
  // i: index of output vector
  // j: index of partial output vectors
  // k: index of input vector
  for (size_t i = 0, k = 0; i < nOut; i++) {
    for (size_t j = 0; j < fanOut; j++, k++) {
      auto yF = setzero<Float, SIMD_WIDTH>();
      for (size_t d = 0; d < DIM; d++) {
        yF = add(yF, mul(wF[d], sub(div(numF[d][k], denomF[d][k]), w0F[d])));
      }
      auto y4F = mul(yF, yF);
      y4F      = mul(y4F, y4F);
      resF[j]  = mul(add(div(yF, sqrt(sqrt(add(oneF, y4F)))), oneF), facF);
    }
    vecsOut[i] = packs<Tout>(resF);
  }
}

template <size_t DIM, size_t NVEC, typename Tout, typename Tin,
          size_t SIMD_WIDTH>
static SIMD_INLINE void fdivMsigmoidmul(
  CompareEqual, const Vec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
  const Vec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC], const double w[DIM],
  const double w0[DIM], double fac, Vec<Tout, SIMD_WIDTH> vecsOut[1])
{
  const auto fanInOut = sizeof(Float) / sizeof(Tin);
  const auto facF     = set1<Float, SIMD_WIDTH>(fac / 2.0);
  const auto oneF     = set1<Float, SIMD_WIDTH>(1.0f);
  Vec<Float, SIMD_WIDTH> wF[DIM], w0F[DIM], numF[DIM][fanInOut],
    denomF[DIM][fanInOut], resF[fanInOut];
  for (size_t d = 0; d < DIM; d++) {
    wF[d]  = set1<Float, SIMD_WIDTH>(-0.433 * w[d]);
    w0F[d] = set1<Float, SIMD_WIDTH>(w0[d]);
    extend(*vecsNum[d], numF[d]);
    extend(*vecsDenom[d], denomF[d]);
  }
  // j: index of extended input/output vector
  for (size_t j = 0; j < fanInOut; j++) {
    auto yF = setzero<Float, SIMD_WIDTH>();
    for (size_t d = 0; d < DIM; d++) {
      yF = add(yF, mul(wF[d], sub(div(numF[d][j], denomF[d][j]), w0F[d])));
    }
    auto y4F = mul(yF, yF);
    y4F      = mul(y4F, y4F);
    resF[j]  = mul(add(div(yF, sqrt(sqrt(add(oneF, y4F)))), oneF), facF);
  }
  vecsOut[0] = packs<Tout>(resF);
}
} // namespace ext
} // namespace internal

/**
 * @brief Special function used in MinWarping.
 *
 * @param[in] vecsNum Numerator vectors.
 * @param[in] vecsDenom Denominator vectors.
 * @param[in] w Weights.
 * @param[in] w0 Weights.
 * @param fac Factor.
 * @param[out] vecsOut Output vectors.
 */
template <size_t DIM, size_t NVEC, typename Tout, typename Tin,
          size_t SIMD_WIDTH>
static SIMD_INLINE void fdivMsigmoidmul(
  const Vec<Tin, SIMD_WIDTH> vecsNum[DIM][NVEC],
  const Vec<Tin, SIMD_WIDTH> vecsDenom[DIM][NVEC], const double w[DIM],
  const double w0[DIM], double fac,
  Vec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  static_assert(sizeof(Tin) <= sizeof(Float), "");
  static_assert(sizeof(Tout) <= sizeof(Float), "");
  internal::ext::fdivMsigmoidmul<DIM, NVEC>(internal::CompareTypes<Tout, Tin>(),
                                            vecsNum, vecsDenom, w, w0, fac,
                                            vecsOut);
}

// ---------------------------------------------------------------------------
// multiply with float constant in float arithmetic
// ---------------------------------------------------------------------------

// 15. Mar 23 (Jonas Keller): fused the three cases

/**
 * @brief Multiplies Vec's element-wise with a floating point constant in
 * floating point arithmetic.
 *
 * @tparam Tfloat the floating point type to perform the calculation in.
 * Defaults to @ref BigEnoughFloat<Tout, Tin>
 * @param[in] vecsIn input Vec's
 * @param fac factor
 * @param[out] vecsOut output Vec's
 */
template <typename Tout, typename Tin,
          typename Tfloat = BigEnoughFloat<Tout, Tin>, size_t SIMD_WIDTH>
static SIMD_INLINE void fmul(
  const Vec<Tin, SIMD_WIDTH> vecsIn[numInVecs<Tout, Tin>()],
  dont_deduce<Tfloat> fac,
  Vec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  static_assert(sizeof(Tin) <= sizeof(Tfloat), "");
  static_assert(sizeof(Tout) <= sizeof(Tfloat), "");
  constexpr auto nFloatVecs = internal::ext::numCalcVecs<Tout, Tin, Tfloat>();
  const auto facVec         = set1<Tfloat, SIMD_WIDTH>(fac);
  Vec<Tfloat, SIMD_WIDTH> inF[nFloatVecs], resF[nFloatVecs];
  internal::ext::extendInToFloat<Tin, Tout>(vecsIn, inF);
  for (size_t i = 0; i < nFloatVecs; i++) { resF[i] = mul(inF[i], facVec); }
  internal::ext::packsOutFromFloat<Tin, Tout>(resF, vecsOut);
}

// ---------------------------------------------------------------------------
// add then multiply with float constant in float arithmetic
// ---------------------------------------------------------------------------

// 15. Mar 23 (Jonas Keller): fused the three cases

/**
 * @brief Adds a floating point constant to the elements of Vec's, then
 * multiplies with a floating point constant in floating point arithmetic
 *
 * @tparam Tfloat the floating point type to perform the calculation in.
 * Defaults to @ref BigEnoughFloat<Tout, Tin>
 * @param[in] vecsIn input Vec's
 * @param off float constant to add
 * @param fac float constant to multiply with
 * @param[out] vecsOut output Vec's
 */
template <typename Tout, typename Tin,
          typename Tfloat = BigEnoughFloat<Tout, Tin>, size_t SIMD_WIDTH>
static SIMD_INLINE void faddmul(
  const Vec<Tin, SIMD_WIDTH> vecsIn[numInVecs<Tout, Tin>()],
  dont_deduce<Tfloat> off, dont_deduce<Tfloat> fac,
  Vec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  static_assert(sizeof(Tin) <= sizeof(Tfloat), "");
  static_assert(sizeof(Tout) <= sizeof(Tfloat), "");
  constexpr auto nFloatVecs = internal::ext::numCalcVecs<Tout, Tin, Tfloat>();
  const auto offVec         = set1<Tfloat, SIMD_WIDTH>(off);
  const auto facVec         = set1<Tfloat, SIMD_WIDTH>(fac);
  Vec<Tfloat, SIMD_WIDTH> inF[nFloatVecs], resF[nFloatVecs];
  internal::ext::extendInToFloat<Tin, Tout>(vecsIn, inF);
  for (size_t i = 0; i < nFloatVecs; i++) {
    resF[i] = mul(add(inF[i], offVec), facVec);
  }
  internal::ext::packsOutFromFloat<Tin, Tout>(resF, vecsOut);
}

// ---------------------------------------------------------------------------
// multiply then add with float constant in float arithmetic
// ---------------------------------------------------------------------------

// better for conversion of zero-centered data to unsigned pixel format

// 15. Mar 23 (Jonas Keller): fused the three cases

/**
 * @brief Multiplies the elements of Vec's with a floating point constant, then
 * adds a floating point constant in floating point arithmetic
 *
 * @tparam Tfloat the floating point type to perform the calculation in.
 * Defaults to @ref BigEnoughFloat<Tout, Tin>
 * @param[in] vecsIn input Vec's
 * @param fac float constant to multiply with
 * @param off float constant to add
 * @param[out] vecsOut output Vec's
 */
template <typename Tout, typename Tin,
          typename Tfloat = BigEnoughFloat<Tout, Tin>, size_t SIMD_WIDTH>
static SIMD_INLINE void fmuladd(
  const Vec<Tin, SIMD_WIDTH> vecsIn[numInVecs<Tout, Tin>()],
  dont_deduce<Tfloat> fac, dont_deduce<Tfloat> off,
  Vec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  static_assert(sizeof(Tin) <= sizeof(Tfloat), "");
  static_assert(sizeof(Tout) <= sizeof(Tfloat), "");
  constexpr auto nFloatVecs = internal::ext::numCalcVecs<Tout, Tin, Tfloat>();
  const auto facVec         = set1<Tfloat, SIMD_WIDTH>(fac);
  const auto offVec         = set1<Tfloat, SIMD_WIDTH>(off);
  Vec<Tfloat, SIMD_WIDTH> inF[nFloatVecs], resF[nFloatVecs];
  internal::ext::extendInToFloat<Tin, Tout>(vecsIn, inF);
  for (size_t i = 0; i < nFloatVecs; i++) {
    resF[i] = add(mul(inF[i], facVec), offVec);
  }
  internal::ext::packsOutFromFloat<Tin, Tout>(resF, vecsOut);
}

// ---------------------------------------------------------------------------
// multiply with float constant in float arithmetic
// ---------------------------------------------------------------------------

// 15. Mar 23 (Jonas Keller): fused the three cases

// fac * [v2 + w * (v1 - v2)] = fac * [w * v1 + (1 - w) * v2], w in [0,1]
// w: weight factor (in [0,1])
// fac: scale factor

/**
 * @brief Linearly interpolates Vec's element-wise with a constant weight and
 * then scales by a constant factor in floating point arithmetic.
 *
 * The result is calculated as:
 * <tt>out = fac * (w * v1 + (1 - w) * v2)</tt>
 * (implemented as <tt>out = fac * (v2 + w * (v1 - v2))</tt>).
 *
 * @tparam Tfloat the floating point type to perform the calculation in.
 * Defaults to @ref BigEnoughFloat<Tout, Tin>
 * @param[in] vecsIn1 first input Vec's
 * @param[in] vecsIn2 second input Vec's
 * @param w interpolation weight
 * @param fac scaling factor
 * @param[out] vecsOut output Vec's
 */
template <typename Tout, typename Tin,
          typename Tfloat = BigEnoughFloat<Tout, Tin>, size_t SIMD_WIDTH>
static SIMD_INLINE void fwaddmul(
  const Vec<Tin, SIMD_WIDTH> vecsIn1[numInVecs<Tout, Tin>()],
  const Vec<Tin, SIMD_WIDTH> vecsIn2[numInVecs<Tout, Tin>()],
  dont_deduce<Tfloat> w, dont_deduce<Tfloat> fac,
  Vec<Tout, SIMD_WIDTH> vecsOut[numOutVecs<Tout, Tin>()])
{
  static_assert(sizeof(Tin) <= sizeof(Tfloat), "");
  static_assert(sizeof(Tout) <= sizeof(Tfloat), "");
  constexpr auto nFloatVecs = internal::ext::numCalcVecs<Tout, Tin, Tfloat>();
  const auto wVec           = set1<Tfloat, SIMD_WIDTH>(w);
  const auto facVec         = set1<Tfloat, SIMD_WIDTH>(fac);
  Vec<Tfloat, SIMD_WIDTH> inF1[nFloatVecs], inF2[nFloatVecs], resF[nFloatVecs];
  internal::ext::extendInToFloat<Tin, Tout>(vecsIn1, inF1);
  internal::ext::extendInToFloat<Tin, Tout>(vecsIn2, inF2);
  for (size_t i = 0; i < nFloatVecs; i++) {
    resF[i] = mul(facVec, add(inF2[i], mul(wVec, sub(inF1[i], inF2[i]))));
  }
  internal::ext::packsOutFromFloat<Tin, Tout>(resF, vecsOut);
}

/** @} */

/**
 * @addtogroup group_horizontal
 * @{
 */

// ===========================================================================
// horizontal add/adds/sub/subs: generic form for multiple vector inputs
// ===========================================================================

// TODO: is there an easy way to implement multivec horizontal min/max?
// TODO: (Hackers delight: min/max via doz = hsubs?)

namespace internal {
namespace ext {
// primary template
// num: number of elements processed
// i0, i1: indices of lowest elements of block
template <typename T, size_t SIMD_WIDTH, size_t num, size_t i0, size_t i1>
struct Horizontal
{
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadd(
    const Vec<T, SIMD_WIDTH> v[Vec<T, SIMD_WIDTH>::elems])
  {
    return hadd(Horizontal<T, SIMD_WIDTH, num / 2, i0, i0 + num / 4>::_hadd(v),
                Horizontal<T, SIMD_WIDTH, num / 2, i1, i1 + num / 4>::_hadd(v));
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadds(
    const Vec<T, SIMD_WIDTH> v[Vec<T, SIMD_WIDTH>::elems])
  {
    return hadds(
      Horizontal<T, SIMD_WIDTH, num / 2, i0, i0 + num / 4>::_hadds(v),
      Horizontal<T, SIMD_WIDTH, num / 2, i1, i1 + num / 4>::_hadds(v));
  }
};

// partial specialization to end the recursion
template <typename T, size_t SIMD_WIDTH, size_t i0, size_t i1>
struct Horizontal<T, SIMD_WIDTH, 2, i0, i1>
{
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadd(
    const Vec<T, SIMD_WIDTH> v[Vec<T, SIMD_WIDTH>::elems])
  {
    return hadd(v[i0], v[i1]);
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadds(
    const Vec<T, SIMD_WIDTH> v[Vec<T, SIMD_WIDTH>::elems])
  {
    return hadds(v[i0], v[i1]);
  }
};
} // namespace ext
} // namespace internal

// function template

/**
 * @brief Sums the elements of multiple Vec's independently and returns a
 * Vec with the results.
 *
 * @param[in] v array of Vec's to be summed. The number of Vec's must be equal
 * to the number of elements in a Vec
 * @return Vec with the results of the horizontal sums
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hadd(
  const Vec<T, SIMD_WIDTH> v[Vec<T, SIMD_WIDTH>::elems])
{
  return internal::ext::Horizontal<T, SIMD_WIDTH, Vec<T, SIMD_WIDTH>::elems, 0,
                                   (Vec<T, SIMD_WIDTH>::elems) / 2>::_hadd(v);
}

// function template

/**
 * @brief Sums the elements of multiple Vec's independently using
 * saturated arithmetic and returns a Vec with the results.
 *
 * @note Does not use saturated arithmetic with floating point types.
 *
 * @param[in] v array of Vec's to be summed
 * @return Vec with the results of the saturated horizontal sums
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hadds(
  const Vec<T, SIMD_WIDTH> v[Vec<T, SIMD_WIDTH>::elems])
{
  return internal::ext::Horizontal<T, SIMD_WIDTH, Vec<T, SIMD_WIDTH>::elems, 0,
                                   (Vec<T, SIMD_WIDTH>::elems) / 2>::_hadds(v);
}

// ===========================================================================
// horizontal operations (generic form for single vector input)
// ===========================================================================

// these operations are not fully parallel!

// example: SIMD_WIDTH = 16, T = float
// extract<0>(Horizontal1<float,16,2>::_hadd(v));
//       u = Horizontal1<float,16,1>::_hadd(v);
//           hadd(v, v)
//       hadd(u, u)

namespace internal {
namespace ext {
template <typename T, size_t SIMD_WIDTH, size_t NUM>
struct Horizontal1
{
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadd(const Vec<T, SIMD_WIDTH> &v)
  {
    Vec<T, SIMD_WIDTH> u = Horizontal1<T, SIMD_WIDTH, NUM / 2>::_hadd(v);
    return hadd(u, u);
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadds(const Vec<T, SIMD_WIDTH> &v)
  {
    Vec<T, SIMD_WIDTH> u = Horizontal1<T, SIMD_WIDTH, NUM / 2>::_hadds(v);
    return hadds(u, u);
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hmin(const Vec<T, SIMD_WIDTH> &v)
  {
    return Horizontal1<T, SIMD_WIDTH, NUM / 2>::_hmin(min(srle<NUM>(v), v));
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hmax(const Vec<T, SIMD_WIDTH> &v)
  {
    return Horizontal1<T, SIMD_WIDTH, NUM / 2>::_hmax(max(srle<NUM>(v), v));
  }
};

template <typename T, size_t SIMD_WIDTH>
struct Horizontal1<T, SIMD_WIDTH, 1>
{
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadd(const Vec<T, SIMD_WIDTH> &v)
  {
    return hadd(v, v);
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hadds(const Vec<T, SIMD_WIDTH> &v)
  {
    return hadds(v, v);
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hmin(const Vec<T, SIMD_WIDTH> &v)
  {
    return min(srle<1>(v), v);
  }

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _hmax(const Vec<T, SIMD_WIDTH> &v)
  {
    return max(srle<1>(v), v);
  }
};
} // namespace ext
} // namespace internal

/**
 * @brief Adds all elements of a Vec.
 *
 * @param v input Vec
 * @return sum of all elements of the Vec
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE T hadd(const Vec<T, SIMD_WIDTH> &v)
{
  return extract<0>(
    internal::ext::Horizontal1<T, SIMD_WIDTH,
                               SIMD_WIDTH / sizeof(T) / 2>::_hadd(v));
}

/**
 * @brief Adds all elements of a Vec using saturated arithmetic.
 *
 * @note Does not use saturated arithmetic with floating point types.
 *
 * @param v Vec to add all elements of
 * @return saturated sum of all elements of the Vec
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE T hadds(const Vec<T, SIMD_WIDTH> &v)
{
  return extract<0>(
    internal::ext::Horizontal1<T, SIMD_WIDTH,
                               SIMD_WIDTH / sizeof(T) / 2>::_hadds(v));
}

/**
 * @brief Calculates the minimum of all elements of a Vec.
 *
 * @param v Vec to calculate the minimum of all elements of
 * @return minimum of all elements of the Vec
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE T hmin(const Vec<T, SIMD_WIDTH> &v)
{
  return extract<0>(
    internal::ext::Horizontal1<T, SIMD_WIDTH,
                               SIMD_WIDTH / sizeof(T) / 2>::_hmin(v));
}

/**
 * @brief Calculates the maximum of all elements of a Vec.
 *
 * @param v Vec to calculate the maximum of all elements of
 * @return maximum of all elements of the Vec
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE T hmax(const Vec<T, SIMD_WIDTH> &v)
{
  return extract<0>(
    internal::ext::Horizontal1<T, SIMD_WIDTH,
                               SIMD_WIDTH / sizeof(T) / 2>::_hmax(v));
}

/** @} */

// ===========================================================================
// iterative horizontal accumulation
// ===========================================================================

/**
 * @addtogroup group_iter_hor_acc
 * @{
 */

// 04. Aug 23 (Jonas Keller): added classes for iterative horizontal
// accumulation

/**
 * @brief Iterative horizontal accumulator. Calculates the horizontal
 * accumulation of multiple (Vec<T, SIMD_WIDTH>::elems) Vec's into a single Vec
 * in parallel with the Vec's to be accumulated pushed one by one.
 *
 * @tparam HOp horizontal accumulation operation, one of HAdd, HAdds, HMin, HMax
 */
template <class HOp, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
struct HAcc
{
private:
  size_t count    = 0;
  size_t stackTop = 0;
  Vec<T, SIMD_WIDTH> stack[internal::ext::floorlog2(Vec<T, SIMD_WIDTH>::elems)];

public:
  /**
   * @brief Checks if the horizontal accumulation is empty, i.e. if no Vec has
   * been pushed yet.
   *
   * @return whether the horizontal accumulation is empty
   */
  bool isEmpty() const { return count == 0; }

  /**
   * @brief Checks if the horizontal accumulation is done.
   *
   * @return whether the horizontal accumulation is done
   */
  bool isDone() const { return count == Vec<T, SIMD_WIDTH>::elems; }

  /**
   * @brief Pushes the next Vec to be horizontally accumulated. Does nothing if
   * the horizontal accumulation is already done.
   *
   * @param v Vec to be pushed
   */
  void push(const Vec<T, SIMD_WIDTH> &v)
  {
    if (isDone()) { return; }
    auto acc = v;
    for (size_t i = 0; count & (1 << i); i++) {
      stackTop--;
      acc = HOp::apply(stack[stackTop], acc);
    }
    stack[stackTop] = acc;
    stackTop++;
    count++;
  }

  /**
   * @brief Finishes the horizontal accumulation by pushing neutral values until
   * the horizontal accumulation is done.
   */
  void finish()
  {
    while (!isDone()) {
      push(set1<T, SIMD_WIDTH>(HOp::template neutralValue<T>()));
    }
  }

  /**
   * @brief Gets the result of the horizontal accumulation. Finishes the
   * horizontal accumulation if it is not done yet.
   *
   * @return result of the horizontal accumulation
   */
  Vec<T, SIMD_WIDTH> get()
  {
    finish();
    return stack[0];
  }

  /**
   * @brief Resets the horizontal accumulation.
   */
  void reset()
  {
    count    = 0;
    stackTop = 0;
  }
};

/**
 * @brief Iterative horizontal accumulator with store of the result. Calculates
 * the horizontal accumulation of multiple Vec's in parallel with the Vec's to
 * be accumulated pushed one by one. Stores the result of the horizontal
 * accumulation every Vec<T, SIMD_WIDTH>::elems Vec's into memory.
 *
 * @tparam HOp horizontal accumulation operation, one of HAdd, HAdds, HMin, HMax
 */
template <class HOp, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
class HAccStore
{
private:
  T *const ptr;
  size_t index = 0;
  HAcc<HOp, T, SIMD_WIDTH> hacc;

public:
  /**
   * @brief Constructs a new HAccStore object.
   *
   * @param ptr pointer to the memory to store the result of the horizontal
   * accumulation
   */
  HAccStore(T *const ptr) : ptr(ptr) {}

  ~HAccStore() { finish(); }

  /**
   * @brief Pushes the next Vec to be horizontally accumulated. Stores the
   * result of the horizontal accumulation every Vec<T, SIMD_WIDTH>::elems Vec's
   * into memory.
   *
   * @param v Vec to be pushed
   */
  void push(Vec<T, SIMD_WIDTH> v)
  {
    hacc.push(v);
    if (hacc.isDone()) {
      // TODO: aligned/unaligned?
      storeu(&ptr[index], hacc.get());
      index += Vec<T, SIMD_WIDTH>::elems;
      hacc.reset();
    }
  }

  /**
   * @brief Finishes the horizontal accumulation and stores the result of the
   * horizontal accumulation into memory.
   */
  void finish()
  {
    // if hacc is not empty, we have to finish and store;
    // if hacc was reset after last push, nothing happens
    if (hacc.isEmpty()) return;

    hacc.finish();
    storeu(&ptr[index], hacc.get());
    index = 0;
    hacc.reset();
  }
};

/**
 * @brief Horizontal addition class for iterative horizontal accumulation.
 * @sa HAcc, HAccStore
 */
struct HAdd
{
  template <typename T, size_t SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &a,
                                              const Vec<T, SIMD_WIDTH> &b)
  {
    return hadd(a, b);
  }

  template <typename T>
  static SIMD_INLINE T neutralValue()
  {
    return T(0);
  }
};

/**
 * @brief Horizontal saturated addition class for iterative horizontal
 * accumulation.
 * @sa HAcc, HAccStore
 */
struct HAdds
{
  template <typename T, size_t SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &a,
                                              const Vec<T, SIMD_WIDTH> &b)
  {
    return hadds(a, b);
  }

  template <typename T>
  static SIMD_INLINE T neutralValue()
  {
    return T(0);
  }
};

/**
 * @brief Horizontal minimum class for iterative horizontal accumulation.
 * @sa HAcc, HAccStore
 */
struct HMin
{
  template <typename T, size_t SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &a,
                                              const Vec<T, SIMD_WIDTH> &b)
  {
    return min(a, b);
  }

  template <typename T>
  static SIMD_INLINE T neutralValue()
  {
    return std::numeric_limits<T>::max();
  }
};

/**
 * @brief Horizontal maximum class for iterative horizontal accumulation.
 * @sa HAcc, HAccStore
 */
struct HMax
{
  template <typename T, size_t SIMD_WIDTH>
  static SIMD_INLINE Vec<T, SIMD_WIDTH> apply(const Vec<T, SIMD_WIDTH> &a,
                                              const Vec<T, SIMD_WIDTH> &b)
  {
    return max(a, b);
  }

  template <typename T>
  static SIMD_INLINE T neutralValue()
  {
    return std::numeric_limits<T>::lowest();
  }
};

/** @} */

/**
 * @addtogroup group_arithmetic
 * @{
 */

// ===========================================================================
// avgru: synonym f. average with rounding up
// ===========================================================================

// this is just a synonym for avg which is compatible with the auxiliary avgrd

/**
 * @copybrief avg(const Vec<T, SIMD_WIDTH> &,
 * const Vec<T, SIMD_WIDTH> &)
 *
 * Equivalent to avg(const Vec<T, SIMD_WIDTH> &, const
 * Vec<T, SIMD_WIDTH> &).
 *
 * @copydetails avg(const Vec<T, SIMD_WIDTH> &,
 * const Vec<T, SIMD_WIDTH> &)
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> avgru(const Vec<T, SIMD_WIDTH> &a,
                                            const Vec<T, SIMD_WIDTH> &b)
{
  return avg(a, b);
}

// ===========================================================================
// avgrd: average with rounding down
// ===========================================================================

// 30. Jul 17 (rm): removed unnecessary tag dispatching for avgrd()

namespace internal {
namespace ext {
// int types
template <typename T, size_t SIMD_WIDTH,
          SIMD_ENABLE_IF(std::is_integral<T>::value)>
static SIMD_INLINE Vec<T, SIMD_WIDTH> avgrd(const Vec<T, SIMD_WIDTH> &a,
                                            const Vec<T, SIMD_WIDTH> &b)
{
  Vec<T, SIMD_WIDTH> one = set1<T, SIMD_WIDTH>(1), as, bs, lsb;
  lsb                    = bit_and(bit_and(a, b), one);
  as                     = div2rd(a);
  bs                     = div2rd(b);
  return add(lsb, add(as, bs));
}

// NOTE: no rounding for floating-point types
template <typename T, size_t SIMD_WIDTH,
          SIMD_ENABLE_IF(std::is_floating_point<T>::value), typename = void>
static SIMD_INLINE Vec<T, SIMD_WIDTH> avgrd(const Vec<T, SIMD_WIDTH> &a,
                                            const Vec<T, SIMD_WIDTH> &b)
{
  return mul(add(a, b), set1<T, SIMD_WIDTH>(0.5));
}
} // namespace ext
} // namespace internal

/**
 * @brief Computes the average of the elements of two Vecs, rounding down.
 *
 * @param a first Vec
 * @param b second Vec
 * @return Vec containing the rounded down average of the elements of the two
 * input Vec's
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> avgrd(const Vec<T, SIMD_WIDTH> &a,
                                            const Vec<T, SIMD_WIDTH> &b)
{
  return internal::ext::avgrd(a, b);
}

// ===========================================================================
// div2r0: integer div. by 2 with round to 0 (for integers)
// ===========================================================================

namespace internal {
namespace ext {
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Byte, SIMD_WIDTH> div2r0(const Vec<Byte, SIMD_WIDTH> &a)
{
  return srli<1>(a);
}

// 16. Oct 22 (Jonas Keller): added missing version for SignedByte
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<SignedByte, SIMD_WIDTH> div2r0(
  const Vec<SignedByte, SIMD_WIDTH> &a)
{
  // add 1 if number is negative
  return srai<1>(add(a, srli<7>(a)));
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Word, SIMD_WIDTH> div2r0(const Vec<Word, SIMD_WIDTH> &a)
{
  return srli<1>(a);
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Short, SIMD_WIDTH> div2r0(
  const Vec<Short, SIMD_WIDTH> &a)
{
  // add 1 if number is negative
  return srai<1>(add(a, srli<15>(a)));
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Int, SIMD_WIDTH> div2r0(const Vec<Int, SIMD_WIDTH> &a)
{
  // add 1 if number is negative
  return srai<1>(add(a, srli<31>(a)));
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Long, SIMD_WIDTH> div2r0(const Vec<Long, SIMD_WIDTH> &a)
{
  // add 1 if number is negative
  return srai<1>(add(a, srli<63>(a)));
}

// NOTE: no rounding for float
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH> div2r0(
  const Vec<Float, SIMD_WIDTH> &a)
{
  return mul(set1<Float, SIMD_WIDTH>(0.5f), a);
}

// NOTE: no rounding for double
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Double, SIMD_WIDTH> div2r0(
  const Vec<Double, SIMD_WIDTH> &a)
{
  return mul(set1<Double, SIMD_WIDTH>(0.5), a);
}
} // namespace ext
} // namespace internal

/**
 * @brief Divides all elements of a Vec by 2 and rounds the result to 0.
 *
 * Only rounds the result to 0 for integer types. For floating point types
 * the result is not rounded.
 *
 * @param a input Vec
 * @return result of the division
 *
 * @sa div2rd()
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> div2r0(const Vec<T, SIMD_WIDTH> &a)
{
  return internal::ext::div2r0(a);
}

// ===========================================================================
// div2rd: integer division by two with rounding down (for integers)
// ===========================================================================

namespace internal {
namespace ext {
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Byte, SIMD_WIDTH> div2rd(const Vec<Byte, SIMD_WIDTH> &a)
{
  return srli<1>(a);
}

// 16. Oct 22 (Jonas Keller): added missing version for SignedByte
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<SignedByte, SIMD_WIDTH> div2rd(
  const Vec<SignedByte, SIMD_WIDTH> &a)
{
  return srai<1>(a);
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Word, SIMD_WIDTH> div2rd(const Vec<Word, SIMD_WIDTH> &a)
{
  return srli<1>(a);
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Short, SIMD_WIDTH> div2rd(
  const Vec<Short, SIMD_WIDTH> &a)
{
  return srai<1>(a);
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Int, SIMD_WIDTH> div2rd(const Vec<Int, SIMD_WIDTH> &a)
{
  return srai<1>(a);
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Long, SIMD_WIDTH> div2rd(const Vec<Long, SIMD_WIDTH> &a)
{
  return srai<1>(a);
}

// NOTE: no rounding for float
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Float, SIMD_WIDTH> div2rd(
  const Vec<Float, SIMD_WIDTH> &a)
{
  return mul(set1<Float, SIMD_WIDTH>(0.5f), a);
}

// NOTE: no rounding for double
template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Double, SIMD_WIDTH> div2rd(
  const Vec<Double, SIMD_WIDTH> &a)
{
  return mul(set1<Double, SIMD_WIDTH>(0.5), a);
}
} // namespace ext
} // namespace internal

/**
 * @brief Divides all elements of a Vec by 2 and rounds down the result.
 *
 * Only rounds down the result for integer types. For floating point types
 * the result is not rounded.
 *
 * @param a input Vec
 * @return result of the division
 *
 * @sa div2r0()
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> div2rd(const Vec<T, SIMD_WIDTH> &a)
{
  return internal::ext::div2rd(a);
}

// ===========================================================================
// sign function (Float and Double only)
// ===========================================================================

// contributed by Benedikt Volkmer
// negate a, where b is negative
// note: contrary to IEEE 754, this function considers -0.0f to be negative

/**
 * @brief Negates the elements of a Vec of floating-point numbers where the
 * corresponding element of a second Vec of floating-point numbers is negative.
 * @note Contrary to IEEE 754, this function considers -0.0 to be negative.
 *
 * @param a Vec of floating-point numbers to be negated
 * @param b Vec of floating-point numbers that determines which elements of a
 * are negated
 * @return resulting Vec of floating-point numbers
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> sign(const Vec<T, SIMD_WIDTH> &a,
                                           const Vec<T, SIMD_WIDTH> &b)
{
  static_assert(std::is_floating_point<T>::value,
                "sign() is only available for floating-point types");
  // -0.0F aka. 0x80000000 aka. 1000...b
  return bit_xor(a, bit_and(set1<T, SIMD_WIDTH>(T(-0.0)), b));
}

// ===========================================================================
// absDiff function
// ===========================================================================

// contributed by Benedikt Volkmer
// 23. Mar 22 (rm): removed SFINAE enable_if construct
// (not compatible with C++98)
// Computes elementwise absolute difference of vectors

namespace internal {
namespace ext {

// Use these overloads of the function template if Type is unsigned

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Byte, SIMD_WIDTH> absDiff(
  const Vec<Byte, SIMD_WIDTH> &v1, const Vec<Byte, SIMD_WIDTH> &v2)
{
  // Trick working around non-existing abs() for unsigned Type
  return bit_or(subs(v1, v2), subs(v2, v1));
}

template <size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Word, SIMD_WIDTH> absDiff(
  const Vec<Word, SIMD_WIDTH> &v1, const Vec<Word, SIMD_WIDTH> &v2)
{
  // Trick working around non-existing abs() for unsigned Type
  return bit_or(subs(v1, v2), subs(v2, v1));
}

// Use these overloads of the function template if Type is signed

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> absDiff(const Vec<T, SIMD_WIDTH> &v1,
                                              const Vec<T, SIMD_WIDTH> &v2)
{
  static_assert(std::is_signed<T>::value, "");
  return abs(sub(v1, v2));
}
} // namespace ext
} // namespace internal

/**
 * @brief Computes the absolute difference of the elements of two Vec's.
 *
 * @param v1 first Vec
 * @param v2 second Vec
 * @return Vec containing the absolute difference of the elements of the two
 * input Vec's
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> absDiff(const Vec<T, SIMD_WIDTH> &v1,
                                              const Vec<T, SIMD_WIDTH> &v2)
{
  return internal::ext::absDiff(v1, v2);
}

/** @} */

namespace internal {
namespace ext {

// ===========================================================================
// transpose
// ===========================================================================

// -------------------- different unpack functions ---------------------------

template <size_t PART, size_t NUM_ELEMS, typename T, size_t SIMD_WIDTH>
struct Unpack
{
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _unpack(const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
  {
    return unpack<PART, NUM_ELEMS>(a, b);
  }
};

template <size_t PART, size_t NUM_ELEMS, typename T, size_t SIMD_WIDTH>
struct Unpack16
{
  static SIMD_INLINE Vec<T, SIMD_WIDTH> _unpack(const Vec<T, SIMD_WIDTH> &a,
                                                const Vec<T, SIMD_WIDTH> &b)
  {
    return unpack16<PART, NUM_ELEMS>(a, b);
  }
};

// ------------------------ transpose a single row ---------------------------

// primary template
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH,
          // INDEX: index of first input element to unpack
          // NLOHI: low/high unpack selector index
          // ELEMS: number of elements to unpack in this stage
          size_t INDEX, size_t NLOHI, size_t ELEMS>
struct Transpose1
{
  static constexpr auto PART = (NLOHI & 0x01);
  static constexpr auto NEXT = (NLOHI >> 1);
  static constexpr auto LIDX = INDEX;
  static constexpr auto RIDX = INDEX + ELEMS;
  static constexpr auto HALF = ELEMS / 2;

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _transpose1(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("_transpose1("
    //       "INDEX=%d NLOHI=%d ELEMS=%d PART=%d LIDX=%d RIDX=%d HALF=%d)\n",
    //       INDEX, NLOHI, ELEMS, PART, LIDX, RIDX, HALF);
    // TODO: T,SIMD_WIDTH necessary or can it be deduced from arguments?
    return Unpack<PART, ELEMS, T, SIMD_WIDTH>::_unpack(
      Transpose1<Unpack, T, SIMD_WIDTH, LIDX, NEXT, HALF>::_transpose1(inRows),
      Transpose1<Unpack, T, SIMD_WIDTH, RIDX, NEXT, HALF>::_transpose1(inRows));
  }
};

// partial specialization to end the iteration (ELEMS=1)
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t INDEX, size_t NLOHI>
struct Transpose1<Unpack, T, SIMD_WIDTH, INDEX, NLOHI, 1>
{
  static constexpr auto PART = (NLOHI & 0x01);

  static SIMD_INLINE Vec<T, SIMD_WIDTH> _transpose1(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("_transpose1(INDEX=%d NLOHI=%d *ELEMS=%d PART=%d)\n",
    // 	   INDEX, NLOHI, 1, PART);
    // TODO: T,SIMD_WIDTH necessary or can it be deduced from arguments?
    return Unpack<PART, 1, T, SIMD_WIDTH>::_unpack(inRows[INDEX],
                                                   inRows[INDEX + 1]);
  }
};

// ----------------------- transpose multiple rows --------------------------

// primary template
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH,
          // NUMROWS: total number of rows
          // NUM_TRANSPOSE_ROWS: number of rows to transpose
          // ROW: index of row to transpose
          size_t NUMROWS, size_t NUM_TRANSPOSE_ROWS, size_t ROW>
struct Transpose
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[NUM_TRANSPOSE_ROWS])
  {
    // printf("\n_transpose(NUMROWS=%d,ROW=%d)\n", NUMROWS, ROW);
    // transpose single row with index ROW
    outRows[ROW] =
      // INDEX=0, NLOWHI=ROW, ELEMS=NUMROWS/2
      Transpose1<Unpack, T, SIMD_WIDTH, 0, ROW, NUMROWS / 2>::_transpose1(
        inRows);
    // transpose next row
    // NUMROWS=NUMROWS, ROW=ROW+1
    Transpose<Unpack, T, SIMD_WIDTH, NUMROWS, NUM_TRANSPOSE_ROWS,
              ROW + 1>::_transpose(inRows, outRows);
  }
};

// partial specialization to end the iteration
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t NUMROWS, size_t NUM_TRANSPOSE_ROWS>
struct Transpose<Unpack, T, SIMD_WIDTH, NUMROWS, NUM_TRANSPOSE_ROWS,
                 NUM_TRANSPOSE_ROWS>
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH>[NUM_TRANSPOSE_ROWS])
  {}
};

// function template: partial transpose
template <size_t NUM_TRANSPOSE_ROWS, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_a_partial(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[NUM_TRANSPOSE_ROWS])
{
  Transpose<Unpack, T, SIMD_WIDTH,
            // NUMROWS, NUM_TRANSPOSE_ROWS, ROW
            SIMD_WIDTH / sizeof(T), NUM_TRANSPOSE_ROWS, 0>::_transpose(inRows,
                                                                       outRows);
}

// function template: full transpose
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_a(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose_a_partial<SIMD_WIDTH / sizeof(T)>(inRows, outRows);
}

// ===========================================================================
// copy matrix
// ===========================================================================

// primary template
template <typename T, size_t SIMD_WIDTH, size_t ROW, size_t ROW_STOP>
struct CopyMatrix
{
  static_assert(ROW < ROW_STOP, "ROW must be less than ROW_STOP");

  static SIMD_INLINE void _copy(Vec<T, SIMD_WIDTH> v[ROW_STOP],
                                Vec<T, SIMD_WIDTH> v2[ROW_STOP])
  {
    v2[ROW] = v[ROW];
    CopyMatrix<T, SIMD_WIDTH, ROW + 1, ROW_STOP>::_copy(v, v2);
  }
};

// partial specialization to end the iteration
template <typename T, size_t SIMD_WIDTH, size_t ROW_STOP>
struct CopyMatrix<T, SIMD_WIDTH, ROW_STOP, ROW_STOP>
{
  static SIMD_INLINE void _copy(Vec<T, SIMD_WIDTH>[ROW_STOP],
                                Vec<T, SIMD_WIDTH>[ROW_STOP])
  {}
};

// ===========================================================================
// Transpose Post-Process
// ===========================================================================

// ------------------------ transpose post-process 16 ------------------------
//            Used to post-process transposed matrix using unpack16

// primary template
template <typename T, size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW,
          size_t ROW_STOP, size_t TRANSPOSE_WIDTH, size_t SRC_OFF,
          size_t DST_OFF>
struct TransposePostprocess16
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("%s", "\nTransposePostprocess16\n");
    // printf("TRANSPOSE_WIDTH=%d\n", TRANSPOSE_WIDTH);
    TransposePostprocess16<T, SIMD_WIDTH, NUMROWS, ROW, ROW_STOP,
                           TRANSPOSE_WIDTH / 2, SRC_OFF,
                           2 * DST_OFF>::_transpose(inRows, outRows);
    TransposePostprocess16<T, SIMD_WIDTH, NUMROWS, ROW, ROW_STOP,
                           TRANSPOSE_WIDTH / 2,
                           SRC_OFF + SIMD_WIDTH / TRANSPOSE_WIDTH,
                           2 * DST_OFF + ROW_STOP>::_transpose(inRows, outRows);
  }
};

// partial specialization
template <typename T, size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW,
          size_t ROW_STOP, size_t SRC_OFF, size_t DST_OFF>
struct TransposePostprocess16<T, SIMD_WIDTH, NUMROWS, ROW, ROW_STOP, 16,
                              SRC_OFF, DST_OFF>
{
  static constexpr auto STEP    = SIMD_WIDTH / 16;
  static constexpr auto SRC_ROW = SRC_OFF + ROW * STEP;
  static constexpr auto DST_ROW = DST_OFF + ROW;

  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("%s", "\nTransposePostprocess16\n");
    // printf("TRANSPOSE_WIDTH=%d\n", 16);
    // printf("SRC_ROW=%d DST_ROW=%d\n", SRC_ROW, DST_ROW);
    outRows[DST_ROW] = inRows[SRC_ROW];
    TransposePostprocess16<T, SIMD_WIDTH, NUMROWS, ROW + 1, ROW_STOP, 16,
                           SRC_OFF, DST_OFF>::_transpose(inRows, outRows);
  }
};

// partial specialization to end the iteration
template <typename T, size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW_STOP,
          size_t SRC_OFF, size_t DST_OFF>
struct TransposePostprocess16<T, SIMD_WIDTH, NUMROWS, ROW_STOP, ROW_STOP, 16,
                              SRC_OFF, DST_OFF>
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems])
  {}
};

// ------------------------ transpose post-process hub -----------------------

// primary template
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH>
struct TransposePostprocess
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems])
  {}
};

// partial specialization to post-process Transpose<Unpack16>
template <typename T, size_t SIMD_WIDTH>
struct TransposePostprocess<Unpack16, T, SIMD_WIDTH>
{
  static constexpr auto NUMROWS  = SIMD_WIDTH / sizeof(T);
  static constexpr auto ROW_STOP = 16 / sizeof(T);

  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("%s", "\nTransposePostprocess\n");
    // printf("SIMD_WIDTH=%d TYPE=%s\n", SIMD_WIDTH, TypeInfo<T>::name());
    TransposePostprocess16<T, SIMD_WIDTH, NUMROWS, 0, ROW_STOP, SIMD_WIDTH, 0,
                           0>::_transpose(inRows, outRows);
  }
};

// ===========================================================================
// transpose_b: Transpose<Unpack16> + post-process
// ===========================================================================

// function template: full transpose
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_b(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  Vec<T, SIMD_WIDTH> tempRows[Vec<T, SIMD_WIDTH>::elements];
  Transpose<Unpack16, T, SIMD_WIDTH, SIMD_WIDTH / sizeof(T),
            SIMD_WIDTH / sizeof(T), 0>::_transpose(inRows, tempRows);
  TransposePostprocess<Unpack16, T, SIMD_WIDTH>::_transpose(tempRows, outRows);
}

// ===========================================================================
// transpose_c: Transpose<Unpack16> - needs store16
// ===========================================================================

// function template: full transpose (includes store16)
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_c(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  Vec<T, SIMD_WIDTH> tempOutRows[Vec<T, SIMD_WIDTH>::elems];

  Transpose<Unpack16, T, SIMD_WIDTH, SIMD_WIDTH / sizeof(T),
            SIMD_WIDTH / sizeof(T), 0>::_transpose(inRows, tempOutRows);

  // post-process with store16 ...
  const auto N = SIMD_WIDTH / sizeof(T);
  T outArray[N * N];
  storeu16(outArray, tempOutRows);
  // ... and load to outRows
  loadu(outArray, outRows, N);
}

// ===========================================================================
// Transpose16: Template Class to transpose multiple rows with integrated
// Unpack16 post-process
// Uses Transpose1 to transpose single rows.
// ===========================================================================

// ----------------------- transpose multiple rows --------------------------

// primary template
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH,
          // NUMROWS: total number of rows
          // ROW: index of row to transpose
          size_t NUMROWS, size_t ROW, size_t ROW_STOP, size_t TRANSPOSE_WIDTH,
          size_t SRC_OFF, size_t DST_OFF>
struct Transpose16
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
  {
    Transpose16<Unpack, T, SIMD_WIDTH, NUMROWS, ROW, ROW_STOP,
                TRANSPOSE_WIDTH / 2, SRC_OFF, 2 * DST_OFF>::_transpose(inRows,
                                                                       outRows);
    Transpose16<Unpack, T, SIMD_WIDTH, NUMROWS, ROW, ROW_STOP,
                TRANSPOSE_WIDTH / 2, SRC_OFF + SIMD_WIDTH / TRANSPOSE_WIDTH,
                2 * DST_OFF + ROW_STOP>::_transpose(inRows, outRows);
  }
};

// partial specialization to end first iteration
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW, size_t ROW_STOP,
          size_t SRC_OFF, size_t DST_OFF>
struct Transpose16<Unpack, T, SIMD_WIDTH, NUMROWS, ROW, ROW_STOP, 16, SRC_OFF,
                   DST_OFF>
{
  static constexpr auto STEP    = SIMD_WIDTH / 16;
  static constexpr auto SRC_ROW = SRC_OFF + ROW * STEP;
  static constexpr auto DST_ROW = DST_OFF + ROW;

  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("\n_transpose_b(SRC=%d,DST=%d)", SRC_ROW, DST_ROW);
    // printf("\n   ROW=%d,OFF=%d,STEP=%d", ROW, TRANSPOSE_OFFSET, STEP);
    // transpose single row with index SRC_ROW
    outRows[DST_ROW] = Transpose1<Unpack, T, SIMD_WIDTH,
                                  // INDEX=0, NLOWHI=SRC_ROW, ELEMS=NUMROWS/2
                                  0, SRC_ROW, NUMROWS / 2>::_transpose1(inRows);
    // transpose next row
    // NUMROWS=NUMROWS, ROW=ROW+1
    Transpose16<Unpack, T, SIMD_WIDTH, NUMROWS, ROW + 1, ROW_STOP, 16, SRC_OFF,
                DST_OFF>::_transpose(inRows, outRows);
  }
};

// partial specialization to end the iteration
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t NUMROWS, size_t ROW_STOP, size_t SRC_OFF,
          size_t DST_OFF>
struct Transpose16<Unpack, T, SIMD_WIDTH, NUMROWS, ROW_STOP, ROW_STOP, 16,
                   SRC_OFF, DST_OFF>
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH>[Vec<T, SIMD_WIDTH>::elems])
  {}
};

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_d(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  Transpose16<Unpack16, T, SIMD_WIDTH,
              // NUMROWS, ROW, ROW_STOP
              SIMD_WIDTH / sizeof(T), 0, 16 / sizeof(T),
              // TRANSPOSE_WIDTH, SRC_OFF, DST_OFF
              SIMD_WIDTH, 0, 0>::_transpose(inRows, outRows);
}

// ===========================================================================
// swizzle2_a (deinterleave)
// ===========================================================================

// generalized from Marat Dukhan's solution referred to at
// https://stackoverflow.com/a/15377386/3852630
// takes 2*N input elements

// TODO: swizzling chunks of multiple elements (useful?)
// TODO: could be possible by starting loop at sizeof(T) and
// TODO: using zip<NUM_ELEMS>

// FINALBLKSIZE template argument is required since function is also
// used for transpose_e
template <size_t N, size_t FINALBLKSIZE, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2_a(Vec<T, SIMD_WIDTH> v[2 * N])
{
  Vec<T, SIMD_WIDTH> v2[2 * N];
  for (size_t blkSize = 1; blkSize <= FINALBLKSIZE; blkSize *= 2) {
    // zip
    for (size_t src = 0, dst = 0; src < N; src++, dst += 2)
      zip<1>(v[src], v[src + N], v2[dst], v2[dst + 1]);
    // copy result back to v
    // TODO: swizzle2_a: check code produced by compiler for copying
    for (size_t i = 0; i < 2 * N; i++) v[i] = v2[i];
  }
}

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2_a(Vec<T, SIMD_WIDTH> v[2 * N])
{
  swizzle2_a<N, Vec<T, SIMD_WIDTH>::elements>(v);
}

// ===========================================================================
// transpose_e
// ===========================================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_e(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  constexpr auto num = Vec<T, SIMD_WIDTH>::elements;
  for (size_t i = 0; i < num; i++) outRows[i] = inRows[i];
  swizzle2_a<num / 2, num / 2>(outRows);
}

// ===========================================================================
// swizzle2_b (deinterleave)
// ===========================================================================

// contributed by Adam Marschall

// generalized from Marat Dukhan's solution referred to at
// https://stackoverflow.com/a/15377386/3852630
// takes 2*N input elements

// FINALBLKSIZE template argument is required since function is also
// used for transpose_f
template <size_t N, size_t FINALBLKSIZE, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2_b(Vec<T, SIMD_WIDTH> v[2 * N])
{
  Vec<T, SIMD_WIDTH> v2[2 * N];
  const auto origReps  = floorlog2(FINALBLKSIZE) + 1;
  const auto finalReps = origReps / 2;
  // printf("origReps=%d finalReps=%d\n", origReps, finalReps);

  for (size_t rep = 0; rep < finalReps; rep++) {
    // zip there ...
    for (size_t src = 0, dst = 0; src < N; src++, dst += 2)
      zip<1>(v[src], v[src + N], v2[dst], v2[dst + 1]);

    // ... and zip back again
    for (size_t src = 0, dst = 0; src < N; src++, dst += 2)
      zip<1>(v2[src], v2[src + N], v[dst], v[dst + 1]);
  }

  // skip post-amble in case of even origReps
  if (origReps % 2 == 0) return;

  // zip there ...
  for (size_t src = 0, dst = 0; src < N; src++, dst += 2)
    zip<1>(v[src], v[src + N], v2[dst], v2[dst + 1]);

  // ...and copy back again
  for (size_t i = 0; i < 2 * N; i++) v[i] = v2[i];
}

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2_b(Vec<T, SIMD_WIDTH> v[2 * N])
{
  swizzle2_b<N, Vec<T, SIMD_WIDTH>::elements>(v);
}

// ===========================================================================
// transpose_f
// ===========================================================================

// contributed by Adam Marschall

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_f(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  const auto elems = Vec<T, SIMD_WIDTH>::elements;
  for (size_t i = 0; i < elems; i++) outRows[i] = inRows[i];
  swizzle2_b<elems / 2, elems / 2>(outRows);
}

// ===========================================================================
// Swizzle2 meta template
// ===========================================================================

// contributed by Adam Marschall

// -------------------- different zip functions ------------------------------

template <size_t NUM_ELEMS, typename T, size_t SIMD_WIDTH>
struct Zip
{
  static SIMD_INLINE void _zip(Vec<T, SIMD_WIDTH> a, Vec<T, SIMD_WIDTH> b,
                               Vec<T, SIMD_WIDTH> &l, Vec<T, SIMD_WIDTH> &h)
  {
    zip<NUM_ELEMS, T>(a, b, l, h);
  }
};

template <size_t NUM_ELEMS, typename T, size_t SIMD_WIDTH>
struct Zip16
{
  static SIMD_INLINE void _zip(Vec<T, SIMD_WIDTH> a, Vec<T, SIMD_WIDTH> b,
                               Vec<T, SIMD_WIDTH> &l, Vec<T, SIMD_WIDTH> &h)
  {
    zip16<NUM_ELEMS, T>(a, b, l, h);
  }
};

// ------------------------ swizzle matrix once ------------------------------

// primary template
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N, size_t SRC, size_t DST>
struct Swizzle2Once
{
  static constexpr auto SRC2 = SRC + N;
  static constexpr auto DST2 = DST + 1;

  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                   Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    // printf("%s\n", "SwizzleOnce");
    // printf("  SRC=%d, SRC2=%d, DST=%d, DS2T=%d\n", SRC, SRC2, DST, DST2);
    Zip<1, T, SIMD_WIDTH>::_zip(v[SRC], v[SRC2], v2[DST], v2[DST2]);
    Swizzle2Once<Zip, T, SIMD_WIDTH, N, SRC + 1, DST + 2>::_swizzle(v, v2);
  }
};

// partial specialization to end the iteration
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N, size_t DST>
struct Swizzle2Once<Zip, T, SIMD_WIDTH, N, N, DST>
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH>[2 * N] /*v*/,
                                   Vec<T, SIMD_WIDTH>[2 * N] /*v2*/)
  {
    // for (size_t i = 0; i < 2 * N; i++) {
    //   print("%4d", v2[i]);
    //   puts("");
    // }
  }
};

// ------------------------ swizzle matrix multiple times --------------------

// primary template
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N, size_t REP, size_t FINAL_REPS,
          size_t ODD>
struct Swizzle2Multiple
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                   Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    // printf("%s\n", "SwizzleMultiple");
    // printf("  REP=%d, FINAL_REPS=%d\n", REP, FINAL_REPS);
    Swizzle2Once<Zip, T, SIMD_WIDTH, N, 0, 0>::_swizzle(v, v2);
    Swizzle2Once<Zip, T, SIMD_WIDTH, N, 0, 0>::_swizzle(v2, v);
    Swizzle2Multiple<Zip, T, SIMD_WIDTH, N, REP + 1, FINAL_REPS, ODD>::_swizzle(
      v, v2);
  }
};

// partial specialization to end the iteration without swizzle post-amble
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N, size_t FINAL_REPS, size_t ODD>
struct Swizzle2Multiple<Zip, T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, ODD>
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH>[2 * N],
                                   Vec<T, SIMD_WIDTH>[2 * N])
  {}
};

// partial specialization to end the iteration with swizzle post-amble
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N, size_t FINAL_REPS>
struct Swizzle2Multiple<Zip, T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, 1>
{
  static constexpr auto ROW_STOP = 2 * N;

  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                   Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    // printf("%s\n", "SwizzlePostamble");
    Swizzle2Once<Zip, T, SIMD_WIDTH, N, 0, 0>::_swizzle(v, v2);
    CopyMatrix<T, SIMD_WIDTH, 0, ROW_STOP>::_copy(v2, v);
  }
};

// ------------------------ swizzle main meta template -----------------------

// generalized from Marat Dukhan's solution referred to at
// https://stackoverflow.com/a/15377386/3852630

// primary template
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N, size_t FINALBLKSIZE>
struct Swizzle2
{
  static constexpr auto ORIG_REPS  = floorlog2(FINALBLKSIZE) + 1;
  static constexpr auto FINAL_REPS = ORIG_REPS / 2;
  static constexpr auto ODD        = (ORIG_REPS & 0x01);

  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N])
  {
    Vec<T, SIMD_WIDTH> v2[2 * N];
    // printf("%s\n", "Swizzle");
    // printf("  N=%d, FINALBLKSIZE=%d\n", N, FINALBLKSIZE);
    // printf("  ORIG_REPS=%d, FINAL_REPS=%d, ODD=%d\n", ORIG_REPS, FINAL_REPS,
    // ODD);
    Swizzle2Multiple<Zip, T, SIMD_WIDTH, N, 0, FINAL_REPS, ODD>::_swizzle(v,
                                                                          v2);
  }
};

// ===========================================================================
// swizzle2_c wrapper function
// ===========================================================================

// 15. Oct 22 (Jonas Keller): added swizzle2_c wrapper function

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2_c(Vec<T, SIMD_WIDTH> v[2 * N])
{
  Swizzle2<Zip, T, SIMD_WIDTH, N, Vec<T, SIMD_WIDTH>::elements>::_swizzle(v);
}

// ===========================================================================
// Unswizzle
// ===========================================================================

// 15. Oct 22 (Jonas Keller): added Unswizzle classes

// Note: Unlike the Swizzle2 classes, the Unswizzle classes do not have a
// template-template parameter for the Zip class.
// In the Swizzle2 classes, the Zip template parameter is used to choose
// between the zip and zip16 functions, which is needed by the Transpose_g
// classes. The Unswizzle classes are not used by Transpose_g, so the Zip
// template parameter is not needed.

template <typename T, size_t SIMD_WIDTH, size_t N, size_t SRC, size_t DST>
struct UnswizzleOnce
{
  static SIMD_INLINE void _unswizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                     Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    unzip<1, T>(v[SRC], v[SRC + 1], v2[DST], v2[DST + N]);
    UnswizzleOnce<T, SIMD_WIDTH, N, SRC + 2, DST + 1>::_unswizzle(v, v2);
  }
};

// partial specialization to end the iteration
template <typename T, size_t SIMD_WIDTH, size_t N, size_t SRC>
struct UnswizzleOnce<T, SIMD_WIDTH, N, SRC, N>
{
  static SIMD_INLINE void _unswizzle(Vec<T, SIMD_WIDTH>[2 * N],
                                     Vec<T, SIMD_WIDTH>[2 * N])
  {}
};

template <typename T, size_t SIMD_WIDTH, size_t N, size_t REP,
          size_t FINAL_REPS, size_t ODD>
struct UnswizzleMultiple
{
  static SIMD_INLINE void _unswizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                     Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    UnswizzleOnce<T, SIMD_WIDTH, N, 0, 0>::_unswizzle(v, v2);
    UnswizzleOnce<T, SIMD_WIDTH, N, 0, 0>::_unswizzle(v2, v);
    UnswizzleMultiple<T, SIMD_WIDTH, N, REP + 1, FINAL_REPS, ODD>::_unswizzle(
      v, v2);
  }
};

// partial specialization to end the iteration without unswizzle post-amble
template <typename T, size_t SIMD_WIDTH, size_t N, size_t FINAL_REPS,
          size_t ODD>
struct UnswizzleMultiple<T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, ODD>
{
  static SIMD_INLINE void _unswizzle(Vec<T, SIMD_WIDTH>[2 * N],
                                     Vec<T, SIMD_WIDTH>[2 * N])
  {}
};

// partial specialization to end the iteration with unswizzle post-amble
template <typename T, size_t SIMD_WIDTH, size_t N, size_t FINAL_REPS>
struct UnswizzleMultiple<T, SIMD_WIDTH, N, FINAL_REPS, FINAL_REPS, 1>
{
  static SIMD_INLINE void _unswizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                     Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    UnswizzleOnce<T, SIMD_WIDTH, N, 0, 0>::_unswizzle(v, v2);
    CopyMatrix<T, SIMD_WIDTH, 0, 2 * N>::_copy(v2, v);
  }
};

// ------------------------ unswizzle main meta template ---------------------

template <typename T, size_t SIMD_WIDTH, size_t N>
struct Unswizzle
{
  static constexpr auto FINALBLKSIZE = Vec<T, SIMD_WIDTH>::elements;
  static constexpr auto ORIG_REPS    = floorlog2(FINALBLKSIZE) + 1;
  static constexpr auto FINAL_REPS   = ORIG_REPS / 2;
  static constexpr auto ODD          = (ORIG_REPS & 0x01);

  static SIMD_INLINE void _unswizzle(Vec<T, SIMD_WIDTH> v[2 * N])
  {
    Vec<T, SIMD_WIDTH> v2[2 * N];
    UnswizzleMultiple<T, SIMD_WIDTH, N, 0, FINAL_REPS, ODD>::_unswizzle(v, v2);
  }
};

// ===========================================================================
// unswizzle_b wrapper function
// ===========================================================================

// 15. Oct 22 (Jonas Keller): added unswizzle_b wrapper function

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void unswizzle_b(Vec<T, SIMD_WIDTH> v[2 * N])
{
  Unswizzle<T, SIMD_WIDTH, N>::_unswizzle(v);
}

// ===========================================================================
// transpose_g Swizzle2<Zip>
// ===========================================================================

// contributed by Adam Marschall

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_g(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  const auto elems = Vec<T, SIMD_WIDTH>::elements;
  for (size_t i = 0; i < elems; i++) outRows[i] = inRows[i];
  Swizzle2<Zip, T, SIMD_WIDTH, elems / 2, elems / 2>::_swizzle(outRows);
}

// ===========================================================================
// transpose_h: Swizzle2<Zip16> + Swizzle post-process
// ===========================================================================

// contributed by Adam Marschall

// ------------------------ swizzle post-process 16 once ---------------------

// primary template
template <typename T, size_t SIMD_WIDTH, size_t N, size_t SRC, size_t DST,
          size_t LANE_ELEMS>
struct Swizzle2Postprocess16Once
{
  static constexpr auto SRC2 = SRC + 1;
  static constexpr auto DST2 = DST + N;

  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                   Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    // printf("%s\n", "SwizzlePostprocess16Once");
    // printf("  SRC=%d, SRC2=%d, DST=%d, DS2T=%d\n", SRC, SRC2, DST, DST2);
    Zip16<LANE_ELEMS, T, SIMD_WIDTH>::_zip(v[SRC], v[SRC2], v2[DST], v2[DST2]);
    Swizzle2Postprocess16Once<T, SIMD_WIDTH, N, SRC + 2, DST + 1,
                              LANE_ELEMS>::_swizzle(v, v2);
  }
};

// partial specialization to end the iteration
template <typename T, size_t SIMD_WIDTH, size_t N, size_t SRC,
          size_t LANE_ELEMS>
struct Swizzle2Postprocess16Once<T, SIMD_WIDTH, N, SRC, N, LANE_ELEMS>
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH>[2 * N] /*v*/,
                                   Vec<T, SIMD_WIDTH>[2 * N] /*v2*/)
  {
    // for (size_t i = 0; i < 2 * N; i++) {
    //   print("%4d", v2[i]);
    //   puts("");
    // }
  }
};

// ------------------------ swizzle post-process 16 --------------------------

// primary template
template <typename T, size_t SIMD_WIDTH, size_t N, size_t LANE_ELEMS,
          size_t REP, size_t FINAL_REPS, size_t ODD>
struct Swizzle2Postprocess16
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                   Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    // printf("%s\n", "SwizzlePostprocess16");
    // printf("  REP=%d, FINAL_REPS=%d, LANE_ELEMS=%d\n", REP, FINAL_REPS,
    // LANE_ELEMS);
    Swizzle2Postprocess16Once<T, SIMD_WIDTH, N, 0, 0, LANE_ELEMS>::_swizzle(v,
                                                                            v2);
    Swizzle2Postprocess16Once<T, SIMD_WIDTH, N, 0, 0, LANE_ELEMS * 2>::_swizzle(
      v2, v);
    Swizzle2Postprocess16<T, SIMD_WIDTH, N, LANE_ELEMS * 4, REP + 1, FINAL_REPS,
                          ODD>::_swizzle(v, v2);
  }
};

// partial specialization to end the iteration without post-process post-amble
template <typename T, size_t SIMD_WIDTH, size_t N, size_t LANE_ELEMS,
          size_t FINAL_REPS, size_t ODD>
struct Swizzle2Postprocess16<T, SIMD_WIDTH, N, LANE_ELEMS, FINAL_REPS,
                             FINAL_REPS, ODD>
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH>[2 * N],
                                   Vec<T, SIMD_WIDTH>[2 * N])
  {}
};

// partial specialization to end the iteration with post-process post-amble
template <typename T, size_t SIMD_WIDTH, size_t N, size_t LANE_ELEMS,
          size_t FINAL_REPS>
struct Swizzle2Postprocess16<T, SIMD_WIDTH, N, LANE_ELEMS, FINAL_REPS,
                             FINAL_REPS, 1>
{
  static constexpr auto ROW_STOP = 2 * N;

  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N],
                                   Vec<T, SIMD_WIDTH> v2[2 * N])
  {
    Swizzle2Postprocess16Once<T, SIMD_WIDTH, N, 0, 0, LANE_ELEMS>::_swizzle(v,
                                                                            v2);
    CopyMatrix<T, SIMD_WIDTH, 0, ROW_STOP>::_copy(v2, v);
  }
};

// ------------------------ swizzle post-process hub -------------------------

// primary template
template <template <size_t, typename, size_t> class Zip, typename T,
          size_t SIMD_WIDTH, size_t N>
struct Swizzle2Postprocess
{
  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH>[2 * N]) {}
};

// partial specialization to post-process Swizzle<Zip16>
template <typename T, size_t SIMD_WIDTH, size_t N>
struct Swizzle2Postprocess<Zip16, T, SIMD_WIDTH, N>
{
  static constexpr auto ORIG_REPS  = floorlog2(SIMD_WIDTH) - 4;
  static constexpr auto FINAL_REPS = ORIG_REPS / 2;
  static constexpr auto ODD        = (ORIG_REPS & 0x01);
  static constexpr auto LANE_ELEMS = 16 / sizeof(T);

  static SIMD_INLINE void _swizzle(Vec<T, SIMD_WIDTH> v[2 * N])
  {
    Vec<T, SIMD_WIDTH> v2[2 * N];
    Swizzle2Postprocess16<T, SIMD_WIDTH, N, LANE_ELEMS, 0, FINAL_REPS,
                          ODD>::_swizzle(v, v2);
  }
};

// ------------------------ transpose_h function call -------------------------

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_h(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  const auto elems = Vec<T, SIMD_WIDTH>::elements;
  for (size_t i = 0; i < elems; i++) outRows[i] = inRows[i];
  Swizzle2<Zip16, T, SIMD_WIDTH, elems / 2, elems / 2>::_swizzle(outRows);
  Swizzle2Postprocess<Zip16, T, SIMD_WIDTH, elems / 2>::_swizzle(outRows);
}

// ===========================================================================
// transpose_i: register-count based transpose
// ===========================================================================

// contributed by Adam Marschall

// primary template: unpack repetition
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t PROCESS_ROW, size_t PROCESS_ROWS,
          size_t UNPACK_ELEMS, size_t UNPACK_REP, size_t UNPACK_REPS,
          size_t SUB_BASE, size_t SUB>
struct TransposeRcUnpackSingle
{
  static constexpr auto UNPACK_PART =
    (PROCESS_ROW >> (UNPACK_REPS - UNPACK_REP - 1)) & 0x01;
  static constexpr auto UNPACK_PART_NEXT =
    ((PROCESS_ROW + 1) >> (UNPACK_REPS - UNPACK_REP - 1)) & 0x01;
  static constexpr auto SRC1 = (PROCESS_ROW - SUB) * 2;
  static constexpr auto SRC2 = (PROCESS_ROW - SUB) * 2 + 1;

  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH> inRows[PROCESS_ROWS],
    Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    // printf("%2d <- Unpack<%d, %d, %s, %d>(%2d, %2d) SUB_BASE: %d, SUB: %d\n",
    //    PROCESS_ROW, UNPACK_PART, UNPACK_ELEMS,
    //    TypeInfo<T>::name(), SIMD_WIDTH, SRC1, SRC2, SUB_BASE, SUB);
    outRows[PROCESS_ROW] =
      Unpack<UNPACK_PART, UNPACK_ELEMS, T, SIMD_WIDTH>::_unpack(inRows[SRC1],
                                                                inRows[SRC2]);
    TransposeRcUnpackSingle<
      Unpack, T, SIMD_WIDTH, PROCESS_ROW + 1, PROCESS_ROWS, UNPACK_ELEMS,
      UNPACK_REP, UNPACK_REPS, SUB_BASE,
      SUB + (UNPACK_PART_NEXT == 1 && (PROCESS_ROW + 1) % SUB_BASE == 0 ?
               SUB_BASE :
               0)>::_transpose(inRows, outRows);
  }
};

// partial specialisation to end iteration PROCESS_REP
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t PROCESS_ROWS, size_t UNPACK_ELEMS,
          size_t UNPACK_REP, size_t UNPACK_REPS, size_t SUB_BASE, size_t SUB>
struct TransposeRcUnpackSingle<Unpack, T, SIMD_WIDTH, PROCESS_ROWS,
                               PROCESS_ROWS, UNPACK_ELEMS, UNPACK_REP,
                               UNPACK_REPS, SUB_BASE, SUB>
{
  static SIMD_INLINE void _transpose(
    const Vec<T, SIMD_WIDTH>[PROCESS_ROWS] /*inRows*/,
    Vec<T, SIMD_WIDTH>[PROCESS_ROWS] /*outRows*/)
  {
    // printf("%2d\n", PROCESS_ROWS);
    // for (size_t i = 0; i < PROCESS_ROWS; i++) {
    //   print("%5d", outRows[i]);
    //   puts("");
    // }
    // puts("");
  }
};

// ---------------------------------------------------------------------------

// primary template: unpack repetition
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t UNPACK_REP, size_t UNPACK_REPS,
          size_t PROCESS_ROWS, size_t UNPACK_ELEMS, size_t SUB_BASE,
          size_t UNPACK_ODD>
struct TransposeRcUnpackMultiple
{
  static SIMD_INLINE void _transpose(Vec<T, SIMD_WIDTH> inRows[PROCESS_ROWS],
                                     Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    // printf("\nTransposeRcUnpackMultiple %2d %s  %d/%d\n",
    //        SIMD_WIDTH, TypeInfo<T>::name(), UNPACK_REP+1, UNPACK_REPS);
    TransposeRcUnpackSingle<Unpack, T, SIMD_WIDTH, 0, PROCESS_ROWS,
                            UNPACK_ELEMS, UNPACK_REP, UNPACK_REPS, SUB_BASE,
                            0>::_transpose(inRows, outRows);
    TransposeRcUnpackMultiple<Unpack, T, SIMD_WIDTH, UNPACK_REP + 1,
                              UNPACK_REPS, PROCESS_ROWS, UNPACK_ELEMS * 2,
                              SUB_BASE / 2, UNPACK_ODD>::_transpose(outRows,
                                                                    inRows);
  }
};

// partial specialisation to end iteration UNPACK_REP
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t UNPACK_REPS, size_t PROCESS_ROWS,
          size_t UNPACK_ELEMS, size_t SUB_BASE, size_t UNPACK_ODD>
struct TransposeRcUnpackMultiple<Unpack, T, SIMD_WIDTH, UNPACK_REPS,
                                 UNPACK_REPS, PROCESS_ROWS, UNPACK_ELEMS,
                                 SUB_BASE, UNPACK_ODD>
{
  static SIMD_INLINE void _transpose(Vec<T, SIMD_WIDTH>[PROCESS_ROWS],
                                     Vec<T, SIMD_WIDTH>[PROCESS_ROWS])
  {}
};

// partial specialisation to end iteration UNPACK_REP
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t UNPACK_REPS, size_t PROCESS_ROWS,
          size_t UNPACK_ELEMS, size_t SUB_BASE>
struct TransposeRcUnpackMultiple<Unpack, T, SIMD_WIDTH, UNPACK_REPS,
                                 UNPACK_REPS, PROCESS_ROWS, UNPACK_ELEMS,
                                 SUB_BASE, 0>
{
  static SIMD_INLINE void _transpose(Vec<T, SIMD_WIDTH> inRows[PROCESS_ROWS],
                                     Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    // printf("\nTransposeRcUnpackMultiple %2d %s  %d/%d Copy Matrix\n",
    //        SIMD_WIDTH, TypeInfo<T>::name(), UNPACK_REPS, UNPACK_REPS);
    CopyMatrix<T, SIMD_WIDTH, 0, PROCESS_ROWS>::_copy(inRows, outRows);
  }
};

// ---------------------------------------------------------------------------

// primary template: store all registers lane-wise
template <typename T, size_t SIMD_WIDTH, size_t PROCESS_REP,
          size_t PROCESS_REPS, size_t PROCESS_ROWS, size_t UNPACK_REPS,
          size_t STORE_OFF, size_t VO, size_t LANE>
struct TransposeRcStoreLane
{
  static constexpr auto VEC_ELEMS_OUT = Vec<T, SIMD_WIDTH>::elems;

  static SIMD_INLINE void _store(
    T outArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    storeu(outArray + STORE_OFF, extractLane<LANE>(outRows[VO]));
    TransposeRcStoreLane<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS, PROCESS_ROWS,
                         UNPACK_REPS, STORE_OFF + PROCESS_ROWS * VEC_ELEMS_OUT,
                         VO, LANE + 1>::_store(outArray, outRows);
  }
};

// partial specialisation to end iteration LANE=PROCESS_REPS
template <typename T, size_t SIMD_WIDTH, size_t PROCESS_REP,
          size_t PROCESS_REPS, size_t PROCESS_ROWS, size_t UNPACK_REPS,
          size_t STORE_OFF, size_t VO>
struct TransposeRcStoreLane<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS,
                            PROCESS_ROWS, UNPACK_REPS, STORE_OFF, VO,
                            PROCESS_REPS>
{
  static SIMD_INLINE void _store(
    T[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH>[PROCESS_ROWS])
  {}
};

// ---------------------------------------------------------------------------

// primary template: store all registers lane-wise
template <typename T, size_t SIMD_WIDTH, size_t PROCESS_REP,
          size_t PROCESS_REPS, size_t PROCESS_ROWS, size_t UNPACK_REPS,
          size_t STORE_OFF, size_t VO>
struct TransposeRcStoreLanes
{
  static constexpr auto VEC_ELEMS_OUT = Vec<T, SIMD_WIDTH>::elems;

  static SIMD_INLINE void _store(
    T outArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    TransposeRcStoreLane<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS, PROCESS_ROWS,
                         UNPACK_REPS, STORE_OFF, VO, 0>::_store(outArray,
                                                                outRows);
    TransposeRcStoreLanes<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS,
                          PROCESS_ROWS, UNPACK_REPS, STORE_OFF + VEC_ELEMS_OUT,
                          VO + 1>::_store(outArray, outRows);
  }
};

// partial specialisation to end iteration VO=PROCESS_ROWS
template <typename T, size_t SIMD_WIDTH, size_t PROCESS_REP,
          size_t PROCESS_REPS, size_t PROCESS_ROWS, size_t UNPACK_REPS,
          size_t STORE_OFF>
struct TransposeRcStoreLanes<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS,
                             PROCESS_ROWS, UNPACK_REPS, STORE_OFF, PROCESS_ROWS>
{
  static SIMD_INLINE void _store(
    T[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH>[PROCESS_ROWS])
  {}
};

// ---------------------------------------------------------------------------

// primary template: store hub
// decides whether to store directly (Store16) or to store lane-wise
template <typename T, size_t SIMD_WIDTH, size_t PROCESS_REP,
          size_t PROCESS_REPS, size_t PROCESS_ROWS, size_t UNPACK_REPS>
struct TransposeRcStore
{
  static constexpr auto ELEMS_PER_LANE = 16 / sizeof(T);
  static constexpr auto STORE_OFF      = PROCESS_REP * ELEMS_PER_LANE;

  static SIMD_INLINE void _store(
    T outArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    TransposeRcStoreLanes<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS,
                          PROCESS_ROWS, UNPACK_REPS, STORE_OFF,
                          0>::_store(outArray, outRows);
  }
};

template <typename T, size_t SIMD_WIDTH, size_t PROCESS_REP,
          size_t PROCESS_ROWS, size_t UNPACK_REPS>
struct TransposeRcStore<T, SIMD_WIDTH, PROCESS_REP, 1, PROCESS_ROWS,
                        UNPACK_REPS>
{
  static SIMD_INLINE void _store(
    T outArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS])
  {
    // printf("\nStore16 PROCESS_ROWS=%d\n", PROCESS_ROWS);
    Store16<Store, T, SIMD_WIDTH, PROCESS_ROWS, 0, 16 / sizeof(T), SIMD_WIDTH,
            0, 0>::_store16(outArray, outRows);
  }
};

// ---------------------------------------------------------------------------

// primary template: main repetition
// loads, transposes, stores chunk of matrix
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t PROCESS_REP, size_t PROCESS_REPS,
          size_t PROCESS_ROWS, size_t UNPACK_REPS, size_t UNPACK_ODD>
struct TransposeRcRep
{
  static constexpr auto LOAD_OFF =
    PROCESS_REP * PROCESS_ROWS * SIMD_WIDTH / sizeof(T);
  static constexpr auto SUB_BASE = 1 << (floorlog2(PROCESS_ROWS) - 1);

  static SIMD_INLINE void _transpose(
    const T inArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    T outArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("\nTransposeRcRep %2d %s  %d/%d\n",
    //         SIMD_WIDTH, TypeInfo<T>::name(), PROCESS_REP+1,
    //         PROCESS_REPS);
    Vec<T, SIMD_WIDTH> inRows[PROCESS_ROWS];
    Vec<T, SIMD_WIDTH> outRows[PROCESS_ROWS];
    load(inArray + LOAD_OFF, inRows, PROCESS_ROWS);
    // for (size_t i = 0; i < PROCESS_ROWS; i++) {
    //   print("%5d", inRows[i]);
    //   puts("");
    // }
    // puts("");
    TransposeRcUnpackMultiple<Unpack, T, SIMD_WIDTH, 0, UNPACK_REPS,
                              PROCESS_ROWS, 1, SUB_BASE,
                              UNPACK_ODD>::_transpose(inRows, outRows);
    // for (size_t i = 0; i < PROCESS_ROWS; i++) {
    //   print("%5d", outRows[i]);
    //   puts("");
    // }
    // puts("");
    TransposeRcStore<T, SIMD_WIDTH, PROCESS_REP, PROCESS_REPS, PROCESS_ROWS,
                     UNPACK_REPS>::_store(outArray, outRows);
    TransposeRcRep<Unpack, T, SIMD_WIDTH, PROCESS_REP + 1, PROCESS_REPS,
                   PROCESS_ROWS, UNPACK_REPS, UNPACK_ODD>::_transpose(inArray,
                                                                      outArray);
  }
};

// partial specialisation to end iteration PROCESS_REP
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH, size_t PROCESS_REPS, size_t PROCESS_ROWS,
          size_t UNPACK_REPS, size_t UNPACK_ODD>
struct TransposeRcRep<Unpack, T, SIMD_WIDTH, PROCESS_REPS, PROCESS_REPS,
                      PROCESS_ROWS, UNPACK_REPS, UNPACK_ODD>
{
  static SIMD_INLINE void _transpose(
    const T[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    T[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems])
  {}
};

// ---------------------------------------------------------------------------

// primary template: main entrance
template <template <size_t, size_t, typename, size_t> class Unpack, typename T,
          size_t SIMD_WIDTH>
struct TransposeRc
{
  static constexpr auto SIMD_REGS = NATIVE_SIMD_REG_COUNT / 2;
  static constexpr auto NUM_ROWS  = SIMD_WIDTH / sizeof(T);
  static constexpr auto PROCESS_REPS =
    NUM_ROWS > SIMD_REGS ? SIMD_WIDTH / 16 : 1;
  static constexpr auto PROCESS_ROWS = NUM_ROWS / PROCESS_REPS;
  static constexpr auto UNPACK_REPS =
    PROCESS_REPS == 1 ? floorlog2(PROCESS_ROWS) : floorlog2(16 / sizeof(T));
  static constexpr auto UNPACK_ODD = (UNPACK_REPS & 0x01);

  static SIMD_INLINE void _transpose(
    const T inArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems],
    T outArray[Vec<T, SIMD_WIDTH>::elems * Vec<T, SIMD_WIDTH>::elems])
  {
    // printf("TransposeRc Process Rows: %d \n", PROCESS_ROWS);
    TransposeRcRep<Unpack, T, SIMD_WIDTH, 0, PROCESS_REPS, PROCESS_ROWS,
                   UNPACK_REPS, UNPACK_ODD>::_transpose(inArray, outArray);
    // printf("%s","\n");
  }
};

// ---------------------------------------------------------------------------

// function template: full transpose
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose_i(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  // 20. Sep 22 (Jonas Keller):
  // use simd_aligned_malloc for inArray and outArray
  // and free them at the end of the function
  // 30. Jul 23 (Jonas Keller):
  // put inArray and outArray on the stack instead of heap to avoid allocation
  // and possibly allow for better compiler optimisation
  const auto N = Vec<T, SIMD_WIDTH>::elements;
  T inArray[N * N] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  T outArray[N * N] SIMD_ATTR_ALIGNED(SIMD_WIDTH);
  store(inArray, inRows, N);
  TransposeRc<Unpack16, T, SIMD_WIDTH>::_transpose(inArray, outArray);
  load(outArray, outRows, N);
}

// ===========================================================================
// unswizzle_a (interleave)
// ===========================================================================

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void unswizzle_a(Vec<T, SIMD_WIDTH> v[2 * N])
{
  const auto finalBlkSize = Vec<T, SIMD_WIDTH>::elements;
  Vec<T, SIMD_WIDTH> v2[2 * N];
  for (size_t blkSize = 1; blkSize <= finalBlkSize; blkSize *= 2) {
    // zip
    for (size_t dst = 0, src = 0; dst < N; dst++, src += 2)
      unzip<1>(v[src], v[src + 1], v2[dst], v2[dst + N]);
    // copy result back to v
    // TODO: unswizzle_a: check code produced by compiler for copying
    for (size_t i = 0; i < 2 * N; i++) v[i] = v2[i];
  }
}

} // namespace ext
} // namespace internal

/**
 * @ingroup group_swizzle
 * @brief Swizzle/de-interleave/convert from AoS to SoA multiple Vec's in-place.
 *
 * This function swizzles/de-interleaves/converts from AoS (Array of Structs) to
 * SoA (Struct of Arrays) multiple Vec's in-place.
 *
 * In contrast to swizzle(), this function takes double the number of Vec's as
 * input and might be faster.
 *
 * <h4>Example:</h4>
 * Example for a swizzle distance of 3 with 6 Vec's of 8 elements each:
 *
 * input stream (structures indicated by curly brackets):
 * @code
 * {0 1 2} {3 4 5} {6 7 8} {9 10 11} ... {45 46 47}
 * @endcode
 * input vectors:
 * @code
 * v[0] =  0  1  2  3  4  5  6  7
 * v[1] =  8  9 10 11 12 13 14 15
 * v[2] = 16 17 18 19 20 21 22 23
 * v[3] = 24 25 26 27 28 29 30 31
 * v[4] = 32 33 34 35 36 37 38 39
 * v[5] = 40 41 42 43 44 45 46 47
 * @endcode
 * output vectors:
 * @code
 * v[0] =  0  6 12 18 24 30 36 42
 * v[1] =  1  7 13 19 25 31 37 43
 * v[2] =  2  8 14 20 26 32 38 44
 * v[3] =  3  9 15 21 27 33 39 45
 * v[4] =  4 10 16 22 28 34 40 46
 * v[5] =  5 11 17 23 29 35 41 47
 * @endcode
 *
 * @tparam N swizzle distance, must be between 1 and 5
 * @param[in,out] v array of Vec's to swizzle
 *
 * @sa swizzle(): swizzles half the number of Vec's as this function
 * @sa unswizzle()
 */
template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2(Vec<T, SIMD_WIDTH> v[2 * N])
{
  // uncomment fastest version
  // internal::ext::swizzle2_a<N>(v);
  // internal::ext::swizzle2_b<N>(v);
  internal::ext::swizzle2_c<N>(v);
}

/**
 * @ingroup group_swizzle
 * @brief Unswizzle/interleave/convert from SoA to AoS multiple Vec's in-place.
 *
 * This function unswizzles/interleaves/converts from SoA (Struct of Arrays) to
 * AoS (Array of Structs) multiple Vec's in-place.
 *
 * <h4>Example:</h4>
 * Example for an unswizzle distance of 3 with 6 Vec's of 8 elements each:
 *
 * input vectors:
 * @code
 * v[0] =  0  6 12 18 24 30 36 42
 * v[1] =  1  7 13 19 25 31 37 43
 * v[2] =  2  8 14 20 26 32 38 44
 * v[3] =  3  9 15 21 27 33 39 45
 * v[4] =  4 10 16 22 28 34 40 46
 * v[5] =  5 11 17 23 29 35 41 47
 * @endcode
 * output vectors:
 * @code
 * v[0] =  0  1  2  3  4  5  6  7
 * v[1] =  8  9 10 11 12 13 14 15
 * v[2] = 16 17 18 19 20 21 22 23
 * v[3] = 24 25 26 27 28 29 30 31
 * v[4] = 32 33 34 35 36 37 38 39
 * v[5] = 40 41 42 43 44 45 46 47
 * @endcode
 *
 * @tparam N unswizzle distance
 * @param[in,out] v array of Vec's to unswizzle
 *
 * @sa swizzle(), swizzle2(): %swizzle functions
 */
template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void unswizzle(Vec<T, SIMD_WIDTH> v[2 * N])
{
  // uncomment fastest version
  // internal::ext::unswizzle_a<N>(v);
  internal::ext::unswizzle_b<N>(v);
}

/**
 * @ingroup group_reordering
 * @brief Transposes a matrix held in an array of Vec's.
 *
 * The matrix must be given as an array of Vec's which are the rows of
 * the matrix. The number of rows must be equal to the number of elements
 * in a Vec (<tt>SIMD_WIDTH/sizeof(T)</tt>), i.e. the matrix must be
 * square.
 *
 * @param[in] inRows array of Vec's holding the matrix to be transposed
 * @param[out] outRows array of Vec's where the transposed matrix is stored
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  // uncomment fastest version

  // 06. Sep 23 (Jonas Keller):
  // added transpose1inplc, transpose2inplc, transpose1inplcLane and
  // transpose2inplcLane and switched to transpose1inplcLane

  // internal::ext::transpose_a(inRows, outRows);
  // internal::ext::transpose_b(inRows, outRows);
  // internal::ext::transpose_c(inRows, outRows);
  // internal::ext::transpose_d(inRows, outRows);
  // internal::ext::transpose_e(inRows, outRows);
  // internal::ext::transpose_f(inRows, outRows);
  // internal::ext::transpose_g(inRows, outRows);
  // internal::ext::transpose_h(inRows, outRows);
  // internal::ext::transpose_i(inRows, outRows);
  // internal::ext::transpose1inplc(inRows, outRows);
  // internal::ext::transpose2inplc(inRows, outRows);
  internal::ext::transpose1inplcLane(inRows, outRows);
  // internal::ext::transpose2inplcLane(inRows, outRows);
}

// ===========================================================================
// setones: set all bits to 1
// ===========================================================================

/**
 * @ingroup group_init
 * @brief Sets all bits of a Vec to 1.
 *
 * @return Vec with all bits set to 1
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setones()
{
  Vec<T, SIMD_WIDTH> zero = setzero<T, SIMD_WIDTH>();
  return cmpeq(zero, zero);
}

// ===========================================================================
// setmin / setmax: set all elements min./max. value of type without set1()
// setunity: set all elements to +1
// setnegunity: set all elements to -1
// ===========================================================================

namespace internal {
namespace ext {
// ---------- signed int ----------

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmin(IsIntIsSigned<true, true>)
{
  return slli<8 * sizeof(T) - 1>(setones<T, SIMD_WIDTH>());
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmax(IsIntIsSigned<true, true>)
{
  return srli<1>(setones<T, SIMD_WIDTH>());
}

// only for signed int
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setnegunity(IsIntIsSigned<true, true>)
{
  return setones<T, SIMD_WIDTH>();
}

// ---------- unsigned int ----------

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmin(IsIntIsSigned<true, false>)
{
  return setzero<T, SIMD_WIDTH>();
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmax(IsIntIsSigned<true, false>)
{
  return setones<T, SIMD_WIDTH>();
}

// ----------- int ------------

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setunity(IsFloatingPoint<false>)
{
  return srli<8 * sizeof(T) - 1>(setones<T, SIMD_WIDTH>());
}

// ----------- float ----------

// here we need set1()

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmin(IsIntIsSigned<false, true>)
{
  return set1<T, SIMD_WIDTH>(TypeInfo<T>::min());
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmax(IsIntIsSigned<false, true>)
{
  return set1<T, SIMD_WIDTH>(TypeInfo<T>::max());
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setunity(IsFloatingPoint<true>)
{
  return set1<T, SIMD_WIDTH>(T(1.0));
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setnegunity(IsIntIsSigned<false, true>)
{
  return set1<T, SIMD_WIDTH>(T(-1.0));
}
} // namespace ext
} // namespace internal

// ---------- hubs ----------

/**
 * @ingroup group_init
 * @brief Sets all elements of a Vec to the minimum value of the element
 * type.
 *
 * @return Vec with all elements set to the minimum value of the element type
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmin()
{
  return internal::ext::setmin<T, SIMD_WIDTH>(internal::TypeIsIntIsSigned<T>());
}

/**
 * @ingroup group_init
 * @brief Sets all elements of a Vec to the maximum value of the element
 * type.
 *
 * @return Vec with all elements set to the maximum value of the element type
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setmax()
{
  return internal::ext::setmax<T, SIMD_WIDTH>(internal::TypeIsIntIsSigned<T>());
}

/**
 * @ingroup group_init
 * @brief Sets all elements of a Vec to the value 1.
 *
 * @return Vec with all elements set to the value 1
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setunity()
{
  return internal::ext::setunity<T, SIMD_WIDTH>(
    internal::TypeIsFloatingPoint<T>());
}

/**
 * @ingroup group_init
 * @brief Sets all elements of a Vec to the value -1.
 *
 * Only available for signed integer and floating point types.
 *
 * @return Vec with all elements set to the value -1
 */
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> setnegunity()
{
  return internal::ext::setnegunity<T, SIMD_WIDTH>(
    internal::TypeIsIntIsSigned<T>());
}

// ===========================================================================
// bitonic sort
// ===========================================================================

/**
 * @addtogroup group_simd_sort
 * @{
 */

// code contributed by Lukas Schiermeier and Moritz Breipohl, modified

namespace internal {
namespace ext {
// compare-and-swap
template <SortSlope SLOPE, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
struct Cas;

// specialization for DESCENDING
template <typename T, size_t SIMD_WIDTH>
struct Cas<SortSlope::DESCENDING, T, SIMD_WIDTH>
{
  static void compareAndSwap(Vec<T, SIMD_WIDTH> &a, Vec<T, SIMD_WIDTH> &b)
  {
    Vec<T, SIMD_WIDTH> temp = min(a, b);
    a                       = max(a, b);
    b                       = temp;
  }
};

// specialization for ASCENDING
template <typename T, size_t SIMD_WIDTH>
struct Cas<SortSlope::ASCENDING, T, SIMD_WIDTH>
{
  static void compareAndSwap(Vec<T, SIMD_WIDTH> &a, Vec<T, SIMD_WIDTH> &b)
  {
    Vec<T, SIMD_WIDTH> temp = max(a, b);
    a                       = min(a, b);
    b                       = temp;
  }
};

// in-place sorting of multiple arbitrary vectors;
// transVecs have to be transposed vectors (same number of elements
// as in Vec), are still transposed afterwards
template <SortSlope SLOPE, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
static SIMD_INLINE void bitonicSortTransposed(
  Vec<T, SIMD_WIDTH> transVecs[Vec<T, SIMD_WIDTH>::elems])
{
  constexpr auto numVecs = Vec<T, SIMD_WIDTH>::elements;
  /* Dependent Loops */
  for (size_t blkSize = 2; blkSize <= numVecs; blkSize *= 2) {
    /*
     * Bitonic Core
     * Independent Loops
     */
    for (size_t blkStart = 0; blkStart < numVecs; blkStart += blkSize) {
      size_t halfBlk      = blkSize / 2;
      size_t leftCounter  = blkStart;
      size_t rightCounter = blkStart + (blkSize - 1);
      /* Independent Loops */
      for (size_t i = 0; i < halfBlk; i++) {
        Cas<SLOPE, T, SIMD_WIDTH>::compareAndSwap(transVecs[leftCounter],
                                                  transVecs[rightCounter]);
        leftCounter++;
        rightCounter--;
      }
      /*
       * This loop is skipped for blkSize < 4
       * Builds the second half of the bitonic core.
       *
       * Dependent Loops
       */
      for (size_t step = blkSize / 4; step > 0; step /= 2) {
        /* Independent Loops */
        for (size_t jump = 0; jump < blkSize; jump += step * 2) {
          leftCounter  = blkStart + jump;
          rightCounter = blkStart + jump + step;
          /* Independent Loops */
          for (size_t k = 0; k < step; k++) {
            Cas<SLOPE, T, SIMD_WIDTH>::compareAndSwap(transVecs[leftCounter],
                                                      transVecs[rightCounter]);
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
// number of elements as in Vec), are still transposed
// afterwards
template <SortSlope SLOPE, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
static SIMD_INLINE void bitonicSortReducedTransposed(
  Vec<T, SIMD_WIDTH> transVecs[Vec<T, SIMD_WIDTH>::elems])
{
  constexpr auto numVecs = Vec<T, SIMD_WIDTH>::elements;
  for (size_t step = numVecs / 2; step > 0; step /= 2) {
    /* Independent Loops */
    for (size_t jump = 0; jump < numVecs; jump += step * 2) {
      size_t leftCounter  = jump;
      size_t rightCounter = jump + step;
      /* Independent Loops */
      for (size_t k = 0; k < step; k++) {
        Cas<SLOPE, T, SIMD_WIDTH>::compareAndSwap(transVecs[leftCounter],
                                                  transVecs[rightCounter]);
        leftCounter++;
        rightCounter++;
      }
    }
  }
}
} // namespace ext
} // namespace internal

/**
 * @brief Sorts multiple Vec's independently using the bitonic sort algorithm.
 *
 * @tparam SLOPE direction to sort in (SortSlope::ASCENDING or
 * SortSlope::DESCENDING)
 * @param[in, out] vecs array of Vec's to sort
 */
template <SortSlope SLOPE, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
static SIMD_INLINE void bitonicSort(
  Vec<T, SIMD_WIDTH> vecs[Vec<T, SIMD_WIDTH>::elems])
{
  Vec<T, SIMD_WIDTH> transVecs[Vec<T, SIMD_WIDTH>::elements];
  transpose(vecs, transVecs);
  internal::ext::bitonicSortTransposed<SLOPE>(transVecs);
  transpose(transVecs, vecs);
}

namespace internal {
namespace ext {
// second vector is reversed and fused with first vector
// we don't have to reverse b after the compare-swap since it is
// bitonic
template <SortSlope SLOPE, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
static SIMD_INLINE void bitonicFusion(Vec<T, SIMD_WIDTH> &a,
                                      Vec<T, SIMD_WIDTH> &b)
{
  b = reverse(b);
  Cas<SLOPE, T, SIMD_WIDTH>::compareAndSwap(a, b);
}
} // namespace ext
} // namespace internal

// given sorted vectors as inputs, it fuses each consecutive pair
// such it is completely sorted over the pair

/**
 * @brief Fuses consecutive pairs of sorted Vec's such that the pair is sorted
 * over the two vectors.
 *
 * @tparam SLOPE direction to sort in (SortSlope::ASCENDING or
 * SortSlope::DESCENDING)
 * @param vecs array of Vec's to sort (Vec's must be sorted individually)
 */
template <SortSlope SLOPE, typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
static SIMD_INLINE void bitonicSortSortedPairs(
  Vec<T, SIMD_WIDTH> vecs[Vec<T, SIMD_WIDTH>::elems])
{
  Vec<T, SIMD_WIDTH> transVecs[Vec<T, SIMD_WIDTH>::elements];
  // second vector of each pair is reversed and fused with first vector
  for (size_t i = 0; i < Vec<T, SIMD_WIDTH>::elements; i += 2)
    internal::ext::bitonicFusion<SLOPE>(vecs[i], vecs[i + 1]);
  transpose(vecs, transVecs);
  internal::ext::bitonicSortReducedTransposed<SLOPE>(transVecs);
  transpose(transVecs, vecs);
}

/** @} */

// ===========================================================================
// operators
// ===========================================================================

// C++ Coding Standards p.49 (item 27)

#define SIMDVEC_BINOPEQ(OP, FCT)                                               \
  template <typename T, size_t SIMD_WIDTH>                                     \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> OP(Vec<T, SIMD_WIDTH> &a,              \
                                           const Vec<T, SIMD_WIDTH> &b)        \
  {                                                                            \
    a = FCT(a, b);                                                             \
    return a;                                                                  \
  }

#define SIMDVEC_BINOP(OP, FCT)                                                 \
  template <typename T, size_t SIMD_WIDTH>                                     \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> OP(const Vec<T, SIMD_WIDTH> &a,        \
                                           const Vec<T, SIMD_WIDTH> &b)        \
  {                                                                            \
    return FCT(a, b);                                                          \
  }

#define SIMDVEC_UNOP(OP, FCT)                                                  \
  template <typename T, size_t SIMD_WIDTH>                                     \
  static SIMD_INLINE Vec<T, SIMD_WIDTH> OP(const Vec<T, SIMD_WIDTH> &a)        \
  {                                                                            \
    return FCT(a);                                                             \
  }

// limitations:
// - mul, div only for Float
// - neg only for signed types

/**
 * @addtogroup group_operators
 * @{
 */

/// @brief Addition operator. Maps to adds(). @sa adds()
SIMDVEC_BINOP(operator+, adds)
/// @brief Subtraction operator. Maps to subs(). @sa subs()
SIMDVEC_BINOP(operator-, subs)
/// @brief Multiplication operator. Maps to mul(). @sa mul()
SIMDVEC_BINOP(operator*, mul)
/// @brief Division operator. Maps to div(). @sa div()
SIMDVEC_BINOP(operator/, div)
/// @brief Bitwise AND operator. Maps to bit_and(). @sa bit_and()
SIMDVEC_BINOP(operator&, bit_and)
/// @brief Bitwise OR operator. Maps to bit_or(). @sa bit_or()
SIMDVEC_BINOP(operator|, bit_or)
/// @brief Bitwise XOR operator. Maps to bit_xor(). @sa bit_xor()
SIMDVEC_BINOP(operator^, bit_xor)

/// @brief Addition assignment operator. Maps to adds(). @sa adds()
SIMDVEC_BINOPEQ(operator+=, adds)
/// @brief Subtraction assignment operator. Maps to subs(). @sa subs()
SIMDVEC_BINOPEQ(operator-=, subs)
/// @brief Multiplication assignment operator. Maps to mul(). @sa mul()
SIMDVEC_BINOPEQ(operator*=, mul)
/// @brief Division assignment operator. Maps to div(). @sa div()
SIMDVEC_BINOPEQ(operator/=, div)
/// @brief Bitwise AND assignment operator. Maps to bit_and(). @sa bit_and()
SIMDVEC_BINOPEQ(operator&=, bit_and)
/// @brief Bitwise OR assignment operator. Maps to bit_or(). @sa bit_or()
SIMDVEC_BINOPEQ(operator|=, bit_or)
/// @brief Bitwise XOR assignment operator. Maps to bit_xor(). @sa bit_xor()
SIMDVEC_BINOPEQ(operator^=, bit_xor)

/// @brief Greater than operator. Maps to cmpgt(). @sa cmpgt()
SIMDVEC_BINOP(operator>, cmpgt)
/// @brief Greater than or equal operator. Maps to cmpge(). @sa cmpge()
SIMDVEC_BINOP(operator>=, cmpge)
/// @brief Equal to operator. Maps to cmpeq(). @sa cmpeq()
SIMDVEC_BINOP(operator==, cmpeq)
/// @brief Not equal to operator. Maps to cmpneq(). @sa cmpneq()
SIMDVEC_BINOP(operator!=, cmpneq)
/// @brief Less than or equal operator. Maps to cmple(). @sa cmple()
SIMDVEC_BINOP(operator<=, cmple)
/// @brief Less than operator. Maps to cmplt(). @sa cmplt()
SIMDVEC_BINOP(operator<, cmplt)

/// @brief Negation operator. Maps to neg(). @sa neg()
SIMDVEC_UNOP(operator-, neg)
/// @brief Bitwise NOT operator. Maps to bit_not(). @sa bit_not()
SIMDVEC_UNOP(operator~, bit_not)

/** @} */
} // namespace simd

#endif
