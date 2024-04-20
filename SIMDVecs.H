// ===========================================================================
//
// Vecs.H --
// multiple Vec in a template class plus some functions
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

#pragma once
#ifndef SIMD_VECS_H_
#define SIMD_VECS_H_

#include "SIMDDefs.H"
#include "SIMDVec.H"
#include "SIMDVecBase.H"
#include "SIMDVecExt.H"

// exclude from doxygen
// TODO: include in doxygen documentation?

// exclude from doxygen (until endcond)
/// @cond

namespace simd {

// ===========================================================================
// Vecs: for functions operating on multiple Vec
// ===========================================================================

// class to compute number of input and output vectors of conversion functions
// (convert, packs, extend, f*)
template <typename Tout, typename Tin>
struct NumVecs
{
  static constexpr size_t in =
    (sizeof(Tout) < sizeof(Tin)) ? (sizeof(Tin) / sizeof(Tout)) : 1;
  static constexpr size_t out =
    (sizeof(Tout) > sizeof(Tin)) ? (sizeof(Tout) / sizeof(Tin)) : 1;
};

// multiple Vec

template <size_t NUM, typename T, size_t SIMD_WIDTH>
struct Vecs
{
  static constexpr size_t vectors  = NUM;
  static constexpr size_t elements = NUM * Vec<T, SIMD_WIDTH>::elements;
  static constexpr size_t bytes    = NUM * Vec<T, SIMD_WIDTH>::bytes;
  // shorter version:
  static constexpr size_t vecs  = vectors;
  static constexpr size_t elems = elements;

  Vec<T, SIMD_WIDTH> vec[NUM];
};

// wrapper functions working on Vecs

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void convert(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &inVecs,
  Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &outVecs)
{
  convert(inVecs.vec, outVecs.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void fdivmul(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsNum,
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsDenom, double fac,
  Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &vecsOut)
{
  fdivmul(vecsNum.vec, vecsDenom.vec, fac, vecsOut.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void fmul(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsIn, double fac,
  Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &vecsOut)
{
  fmul(vecsIn.vec, fac, vecsOut.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void faddmul(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsIn, double off,
  double fac, Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &vecsOut)
{
  faddmul(vecsIn.vec, off, fac, vecsOut.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void fmuladd(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsIn, double fac,
  double off, Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &vecsOut)
{
  fmuladd(vecsIn.vec, fac, off, vecsOut.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void fwaddmul(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsIn1,
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &vecsIn2, double w,
  double fac, Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &vecsOut)
{
  fwaddmul(vecsIn1.vec, vecsIn2.vec, w, fac, vecsOut.vec);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void load(const T *const p, Vecs<NUM, T, SIMD_WIDTH> &inVecs)
{
  load(p, inVecs.vec, inVecs.vectors);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void loadu(const T *const p,
                              Vecs<NUM, T, SIMD_WIDTH> &inVecs)
{
  loadu(p, inVecs.vec, inVecs.vectors);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void store(T *const p,
                              const Vecs<NUM, T, SIMD_WIDTH> &outVecs)
{
  store(p, outVecs.vec, outVecs.vectors);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void storeu(T *const p,
                               const Vecs<NUM, T, SIMD_WIDTH> &outVecs)
{
  storeu(p, outVecs.vec, outVecs.vectors);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void store16(T *const p,
                                const Vecs<NUM, T, SIMD_WIDTH> &outVecs)
{
  store16(p, outVecs.vec);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void storeu16(T *const p,
                                 const Vecs<NUM, T, SIMD_WIDTH> &outVecs)
{
  storeu16(p, outVecs.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<Tout, SIMD_WIDTH> packs(
  const Vecs<NumVecs<Tout, Tin>::in, Tin, SIMD_WIDTH> &a)
{
  return packs<Tout>(a.vec);
}

template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE void extend(
  const Vec<Tin, SIMD_WIDTH> &vIn,
  Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> &vOut)
{
  extend(vIn, vOut.vec);
}

// with Vecs as return value
template <typename Tout, typename Tin, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> extend(
  const Vec<Tin, SIMD_WIDTH> &vIn)
{
  Vecs<NumVecs<Tout, Tin>::out, Tout, SIMD_WIDTH> vOut;
  extend(vIn, vOut);
  return vOut;
}

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle(Vecs<N, T, SIMD_WIDTH> &v)
{
  swizzle<N>(v.vec);
}

// inRows passed by-value to allow in-place transpose
// 30. Sep 22 (rm): was called transpose1, moved back to transpose
template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose(
  const Vecs<(SIMD_WIDTH / sizeof(T)), T, SIMD_WIDTH> inRows,
  Vecs<(SIMD_WIDTH / sizeof(T)), T, SIMD_WIDTH> &outRows)
{
  transpose(inRows.vec, outRows.vec);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose(
  Vecs<(SIMD_WIDTH / sizeof(T)), T, SIMD_WIDTH> &rows)
{
  transpose(rows.vec);
}

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void swizzle2(Vecs<2 * N, T, SIMD_WIDTH> &v)
{
  swizzle2(v.vec);
}

template <size_t N, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void unswizzle(Vecs<2 * N, T, SIMD_WIDTH> &v)
{
  unswizzle(v.vec);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hadd(
  const Vecs<(SIMD_WIDTH / sizeof(T)), T, SIMD_WIDTH> &v)
{
  return hadd(v.vec);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vec<T, SIMD_WIDTH> hadds(
  const Vecs<(SIMD_WIDTH / sizeof(T)), T, SIMD_WIDTH> &v)
{
  return hadds(v.vec);
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> add(
  const Vecs<NUM, T, SIMD_WIDTH> &a, const Vecs<NUM, T, SIMD_WIDTH> &b)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = add(a.vec[v], b.vec[v]);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> adds(
  const Vecs<NUM, T, SIMD_WIDTH> &a, const Vecs<NUM, T, SIMD_WIDTH> &b)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = adds(a.vec[v], b.vec[v]);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> sub(
  const Vecs<NUM, T, SIMD_WIDTH> &a, const Vecs<NUM, T, SIMD_WIDTH> &b)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = sub(a.vec[v], b.vec[v]);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> subs(
  const Vecs<NUM, T, SIMD_WIDTH> &a, const Vecs<NUM, T, SIMD_WIDTH> &b)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = subs(a.vec[v], b.vec[v]);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> min(
  const Vecs<NUM, T, SIMD_WIDTH> &a, const Vecs<NUM, T, SIMD_WIDTH> &b)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = min(a.vec[v], b.vec[v]);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> max(
  const Vecs<NUM, T, SIMD_WIDTH> &a, const Vecs<NUM, T, SIMD_WIDTH> &b)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = max(a.vec[v], b.vec[v]);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> setzero()
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = setzero<T, SIMD_WIDTH>();
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void setzero(Vecs<NUM, T, SIMD_WIDTH> &res)
{
  for (size_t v = 0; v < NUM; v++) res.vec[v] = setzero<T, SIMD_WIDTH>();
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE Vecs<NUM, T, SIMD_WIDTH> set1(T a)
{
  Vecs<NUM, T, SIMD_WIDTH> res;
  for (size_t v = 0; v < NUM; v++) res.vec[v] = set1<T, SIMD_WIDTH>(a);
  return res;
}

template <size_t NUM, typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void set1(Vecs<NUM, T, SIMD_WIDTH> &res, T a)
{
  for (size_t v = 0; v < NUM; v++) res.vec[v] = set1<T, SIMD_WIDTH>(a);
}

// TODO: add more Vecs functions if needed

} // namespace simd

/// @endcond

#endif
