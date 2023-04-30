// ===========================================================================
//
// SIMDVecAutoTestWrapper.H --
// class wrapper, for both Vec and SerialVec, used for tests
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

// 22. Jan 23 (Jonas Keller): moved auto test related stuff into auto_test
// namespace

#ifndef SIMD_VEC_AUTO_TEST_WRAPPER_H_
#define SIMD_VEC_AUTO_TEST_WRAPPER_H_

#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"
#include "SIMDVecAutoTestRandom.H"
#include "SIMDVecAutoTestSerial.H"
#include "SIMDVecBase.H"
#include "SIMDVecExt.H"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// - template parameters T and SIMD_WIDTH are included in class
//   definition and therefore have to be explicitly provided (no
//   deduction possible), this allows members or methods to use these
//   parameters (e.g. maxI)
// - apply() has a template-template parameter such that both Vec
//   and SerialVec can be passed (at least if the function has at
//   least one argument of Vec or SerialVec)

namespace simd {
namespace auto_test {

// -------------------------------------------------------------------------
// aux. functions
// -------------------------------------------------------------------------

// 13. Mar 23 (Jonas Keller): used C++ string concatenation instead of
// sprintf for theses functions

template <typename T, int SIMD_WIDTH>
static std::string t2s(const std::string &fct)
{
  return fct + "<" + TypeInfo<T>::name() + "," + std::to_string(SIMD_WIDTH) +
         ">";
}

template <int I, typename T, int SIMD_WIDTH>
static std::string ti2s(const std::string &fct)
{
  return fct + "<" + std::to_string(I) + "," + TypeInfo<T>::name() + "," +
         std::to_string(SIMD_WIDTH) + ">";
}

template <int I, int J, typename T, int SIMD_WIDTH>
static std::string tii2s(const std::string &fct)
{
  return fct + "<" + std::to_string(I) + "," + std::to_string(J) + "," +
         TypeInfo<T>::name() + "," + std::to_string(SIMD_WIDTH) + ">";
}

template <typename Tout, typename T, int SIMD_WIDTH>
static std::string tt2s(const std::string &fct)
{
  return fct + "<" + TypeInfo<Tout>::name() + "," + TypeInfo<T>::name() + "," +
         std::to_string(SIMD_WIDTH) + ">";
}

// -------------------------------------------------------------------------
// wrapper
// -------------------------------------------------------------------------

template <typename Tout, typename T, int SIMD_WIDTH>
struct Reinterpret
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, T, SIMD_WIDTH>("reinterpret");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<Tout, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return reinterpret<Tout>(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Setzero
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("setzero");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _setzero<T, SIMD_WIDTH>();
  }
};

template <typename T, int SIMD_WIDTH>
struct Set1
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("set1"); }

  static SIMD_INLINE T randomizeInputScalar() { return getRandom<T>(); }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(T a)
  {
    return Deductor<VEC>::template _set1<T, SIMD_WIDTH>(a);
  }
};

template <int I, typename T, int SIMD_WIDTH>
struct Extract
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("extract");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = SIMD_WIDTH / sizeof(T) + 1;
  static constexpr int nextI = I + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE T apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return extract<I>(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Add
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("add"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return add(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Adds
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("adds"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return adds(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Sub
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("sub"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return sub(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Subs
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("subs"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return subs(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Neg
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("neg"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return neg(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Mul
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("mul"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return mul(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Div
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("div"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return div(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Ceil
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("ceil"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return ceil(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Floor
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("floor"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return floor(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Round
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("round"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return round(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Truncate
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("truncate");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return truncate(a);
  }
};

// 02. Oct 22 (Jonas Keller): added Rcp and Rsqrt

template <typename T, int SIMD_WIDTH>
struct Rcp
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("rcp"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return rcp(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Rsqrt
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("rsqrt"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return rsqrt(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Sqrt
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("sqrt"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeNonNegative();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return sqrt(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Min
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("min"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return min(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Max
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("max"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return max(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Abs
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("abs"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return abs(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Sign
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("sign"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return sign(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct AbsDiff
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("absDiff");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return absDiff(a, b);
  }
};

template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
struct Unpack
{
  static SIMD_INLINE std::string name()
  {
    return tii2s<PART, NUM_ELEMS, T, SIMD_WIDTH>("unpack");
  }

  // PART (I) switches between 0 and 1
  static constexpr int minI  = 0;
  static constexpr int maxI  = 1;
  static constexpr int nextI = 1 - PART;
  // NUM_ELEMS (J) is doubled if PART == 1
  static constexpr int minJ  = 1;
  static constexpr int maxJ  = SIMD_WIDTH / sizeof(T);
  static constexpr int nextJ = PART ? NUM_ELEMS * 2 : NUM_ELEMS;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return unpack<PART, NUM_ELEMS>(a, b);
  }
};

// 10. Oct 22 (Jonas Keller): added Unpack16
template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
struct Unpack16
{
  static SIMD_INLINE std::string name()
  {
    return tii2s<PART, NUM_ELEMS, T, SIMD_WIDTH>("unpack16");
  }

  // PART (I) switches between 0 and 1
  static constexpr int minI  = 0;
  static constexpr int maxI  = 1;
  static constexpr int nextI = 1 - PART;
  // NUM_ELEMS (J) is doubled if PART == 1
  static constexpr int minJ  = 1;
  static constexpr int maxJ  = 16 / sizeof(T);
  static constexpr int nextJ = PART ? NUM_ELEMS * 2 : NUM_ELEMS;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return unpack16<PART, NUM_ELEMS>(a, b);
  }
};

// 10. Oct 22 (Jonas Keller): added ExtractLane
// extracts a 128-bit lane
template <int IMM, typename T, int SIMD_WIDTH>
struct ExtractLane
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("extractLane");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = SIMD_WIDTH / 16 - 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, 16> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return extractLane<IMM>(a);
  }
};

template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
struct Zip
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<NUM_ELEMS, T, SIMD_WIDTH>("zip");
  }

  static constexpr int minI  = 1;
  static constexpr int maxI  = SIMD_WIDTH / sizeof(T);
  static constexpr int nextI = NUM_ELEMS * 2;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(const VEC<T, SIMD_WIDTH> a,
                                const VEC<T, SIMD_WIDTH> b,
                                VEC<T, SIMD_WIDTH> &c, VEC<T, SIMD_WIDTH> &d)
  {
    zip<NUM_ELEMS>(a, b, c, d);
  }
};

// 10. Oct 22 (Jonas Keller): added Zip16
template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
struct Zip16
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<NUM_ELEMS, T, SIMD_WIDTH>("zip16");
  }

  static constexpr int minI  = 1;
  static constexpr int maxI  = 16 / sizeof(T);
  static constexpr int nextI = NUM_ELEMS * 2;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(const VEC<T, SIMD_WIDTH> a,
                                const VEC<T, SIMD_WIDTH> b,
                                VEC<T, SIMD_WIDTH> &c, VEC<T, SIMD_WIDTH> &d)
  {
    zip16<NUM_ELEMS>(a, b, c, d);
  }
};

template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
struct Unzip
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<NUM_ELEMS, T, SIMD_WIDTH>("unzip");
  }

  static constexpr int minI  = 1;
  static constexpr int maxI  = SIMD_WIDTH / sizeof(T);
  static constexpr int nextI = NUM_ELEMS * 2;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(const VEC<T, SIMD_WIDTH> a,
                                const VEC<T, SIMD_WIDTH> b,
                                VEC<T, SIMD_WIDTH> &c, VEC<T, SIMD_WIDTH> &d)
  {
    unzip<NUM_ELEMS>(a, b, c, d);
  }
};

template <typename Tout, typename T, int SIMD_WIDTH>
struct Packs
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, T, SIMD_WIDTH>("packs");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<Tout, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                                 const VEC<T, SIMD_WIDTH> &b)
  {
    return packs<Tout>(a, b);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Extend
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("extend");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<Tin, SIMD_WIDTH> &a,
    VEC<Tout, SIMD_WIDTH> b[sizeof(Tout) / sizeof(Tin)])
  {
    extend(a, b);
  }
};

template <int IMM, typename T, int SIMD_WIDTH>
struct Srai
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("srai");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = sizeof(T) * 8 + 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return srai<IMM>(a);
  }
};

template <int IMM, typename T, int SIMD_WIDTH>
struct Srli
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("srli");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = sizeof(T) * 8 + 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return srli<IMM>(a);
  }
};

template <int IMM, typename T, int SIMD_WIDTH>
struct Slli
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("slli");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = sizeof(T) * 8 + 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return slli<IMM>(a);
  }
};

// 19. Dec 22 (Jonas Keller): added Sra, Srl and Sll wrappers

template <typename T, int SIMD_WIDTH>
struct Sra
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("sra"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  static SIMD_INLINE uint8_t randomizeInputScalar()
  {
    return getRandomOrSpecialValue<uint8_t>();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const uint8_t count)
  {
    return sra(a, count);
  }
};

template <typename T, int SIMD_WIDTH>
struct Srl
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("srl"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  static SIMD_INLINE uint8_t randomizeInputScalar()
  {
    return getRandomOrSpecialValue<uint8_t>();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const uint8_t count)
  {
    return srl(a, count);
  }
};

template <typename T, int SIMD_WIDTH>
struct Sll
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("sll"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  static SIMD_INLINE uint8_t randomizeInputScalar()
  {
    return getRandomOrSpecialValue<uint8_t>();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const uint8_t count)
  {
    return sll(a, count);
  }
};

template <typename Tout, typename T, int SIMD_WIDTH>
struct Cvts
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, T, SIMD_WIDTH>("cvts");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<Tout, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return cvts<Tout>(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Hadd
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hadd"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return hadd(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Hadds
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hadds"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return hadds(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Hsub
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hsub"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return hsub(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Hsubs
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hsubs"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return hsubs(a, b);
  }
};

template <int IMM, typename T, int SIMD_WIDTH>
struct Srle
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("srle");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = SIMD_WIDTH / sizeof(T) + 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return srle<IMM>(a);
  }
};

template <int IMM, typename T, int SIMD_WIDTH>
struct Slle
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("slle");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = SIMD_WIDTH / sizeof(T) + 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return slle<IMM>(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Elem0
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("elem0"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE T apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return elem0(a);
  }
};

template <int IMM, typename T, int SIMD_WIDTH>
struct Alignre
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<IMM, T, SIMD_WIDTH>("alignre");
  }

  static constexpr int minI  = 0;
  static constexpr int maxI  = 2 * (SIMD_WIDTH / sizeof(T)) + 1;
  static constexpr int nextI = IMM + 1;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return alignre<IMM>(a, b);
  }
};

template <int I, typename T, int SIMD_WIDTH>
struct Swizzle
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("swizzle");
  }

  static constexpr int minI    = 1;
  static constexpr int maxI    = 6;
  static constexpr int nextI   = I + 1;
  static constexpr int numVecs = I;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> v[I])
  {
    swizzle<I>(v);
  }
};

template <int I, typename T, int SIMD_WIDTH>
struct Swizzle2
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("swizzle2");
  }

  static constexpr int minI = 1;
  // TODO: is there an upper limit for maxI in swizzle2?
  static constexpr int maxI    = Vec<T, SIMD_WIDTH>::elems;
  static constexpr int nextI   = I + 1;
  static constexpr int numVecs = 2 * I;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> v[I])
  {
    swizzle2<I>(v);
  }
};

template <int I, typename T, int SIMD_WIDTH>
struct Unswizzle2
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("unswizzle2");
  }

  static constexpr int minI = 1;
  // TODO: is there an upper limit for maxI in unswizzle2?
  static constexpr int maxI    = Vec<T, SIMD_WIDTH>::elems;
  static constexpr int nextI   = I + 1;
  static constexpr int numVecs = 2 * I;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> v[2 * I])
  {
    unswizzle2<I>(v);
  }
};

// 07. Feb 23 (Jonas Keller): added Swizzle3
template <int I, typename T, int SIMD_WIDTH>
struct Swizzle3
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("swizzle3");
  }

  static constexpr int minI = 1;
  // TODO: is there an upper limit for maxI in swizzle3?
  static constexpr int maxI    = Vec<T, SIMD_WIDTH>::elems;
  static constexpr int nextI   = I + 1;
  static constexpr int numVecs = 2 * I;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> v[2 * I])
  {
    swizzle3<I>(v);
  }
};

// 15. Oct 22 (Jonas Keller): added _Swizzle4
template <int I, typename T, int SIMD_WIDTH>
struct Swizzle4
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("swizzle4");
  }

  static constexpr int minI = 1;
  // TODO: is there an upper limit for maxI in swizzle4?
  static constexpr int maxI    = Vec<T, SIMD_WIDTH>::elems;
  static constexpr int nextI   = I + 1;
  static constexpr int numVecs = 2 * I;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> v[2 * I])
  {
    swizzle4<I>(v);
  }
};

// 15. Oct 22 (Jonas Keller): added _Unswizzle4
template <int I, typename T, int SIMD_WIDTH>
struct Unswizzle4
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<I, T, SIMD_WIDTH>("unswizzle4");
  }

  static constexpr int minI = 1;
  // TODO: is there an upper limit for maxI in unswizzle4?
  static constexpr int maxI    = Vec<T, SIMD_WIDTH>::elems;
  static constexpr int nextI   = I + 1;
  static constexpr int numVecs = 2 * I;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> v[2 * I])
  {
    unswizzle4<I>(v);
  }
};

template <typename T, int SIMD_WIDTH>
struct Cmplt
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("cmplt"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return cmplt(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Cmple
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("cmple"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return cmple(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Cmpeq
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("cmpeq"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return cmpeq(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Cmpge
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("cmpge"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return cmpge(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Cmpgt
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("cmpgt"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return cmpgt(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Cmpneq
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("cmpneq"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return cmpneq(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct And
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("and"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return and_(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Or
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("or"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return or_(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Andnot
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("andnot"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return andnot(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Xor
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("xor"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return xor_(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Not
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("not"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return not_(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Avg
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("avg"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return avg(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Test_all_zeros
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("test_all_zeros");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    if (rand() & 0x01)
      vec.randomize();
    else
      vec.setzero();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE int apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return test_all_zeros(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Test_all_ones
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("test_all_ones");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    if (rand() & 0x01)
      vec.randomize();
    else
      vec.setones();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE int apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return test_all_ones(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Reverse
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("reverse");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return reverse(a);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Convert
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("convert");
  }

  static constexpr int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  static constexpr int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  static constexpr int numOutVecs  = ::simd::numOutVecs<Tout, Tin>();
  static constexpr int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;

  static SIMD_INLINE void randomizeInput(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(const VEC<Tin, SIMD_WIDTH> inVecs[numInVecs],
                                VEC<Tout, SIMD_WIDTH> outVecs[numOutVecs])
  {
    convert(inVecs, outVecs);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Fdivmul
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("fdivmul");
  }

  static constexpr int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  static constexpr int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  static constexpr int numOutVecs  = ::simd::numOutVecs<Tout, Tin>();
  static constexpr int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;

  static SIMD_INLINE Float randomizeInputFloat()
  {
    return getRandomOrSpecialValueRanges<Float>();
  }

  static SIMD_INLINE void randomizeInput1(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  static SIMD_INLINE void randomizeInput2(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomizeRangesNonZero();
  }

  template <template <typename, int> class VEC>
                                double fac,
  static SIMD_INLINE void apply(const VEC<Tin, SIMD_WIDTH> vecsNum[numInVecs],
                                const VEC<Tin, SIMD_WIDTH> vecsDenom[numInVecs],
                                VEC<Tout, SIMD_WIDTH> vecsOut[numOutVecs])
  {
    fdivmul(vecsNum, vecsDenom, fac, vecsOut);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Fmul
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("fmul");
  }

  static constexpr int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  static constexpr int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  static constexpr int numOutVecs  = ::simd::numOutVecs<Tout, Tin>();
  static constexpr int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;

  static SIMD_INLINE Float randomizeInputFloat()
  {
    return getRandomOrSpecialValueRanges<Float>();
  }

  // 03. Aug 22 (Jonas Keller):
  // include zeros in the randomization of the input vectors
  static SIMD_INLINE void randomizeInput(SerialVec<Tin, SIMD_WIDTH> &vec)
  // { vec.randomizeRangesNonZero(); }
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
                                double fac,
  static SIMD_INLINE void apply(const VEC<Tin, SIMD_WIDTH> vecsIn[numInVecs],
                                VEC<Tout, SIMD_WIDTH> vecsOut[numOutVecs])
  {
    fmul(vecsIn, fac, vecsOut);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Faddmul
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("faddmul");
  }

  static constexpr int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  static constexpr int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  static constexpr int numOutVecs  = ::simd::numOutVecs<Tout, Tin>();
  static constexpr int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;

  static SIMD_INLINE Float randomizeInputFloat()
  {
    return getRandomOrSpecialValueRanges<Float>();
  }

  // 03. Aug 22 (Jonas Keller):
  // include zeros in the randomization of the input vectors
  static SIMD_INLINE void randomizeInput(SerialVec<Tin, SIMD_WIDTH> &vec)
  // { vec.randomizeRangesNonZero(); }
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
                                double off, double fac,
  static SIMD_INLINE void apply(const VEC<Tin, SIMD_WIDTH> vecsIn[numInVecs],
                                VEC<Tout, SIMD_WIDTH> vecsOut[numOutVecs])
  {
    faddmul(vecsIn, off, fac, vecsOut);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Fmuladd
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("fmuladd");
  }

  static constexpr int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  static constexpr int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  static constexpr int numOutVecs  = ::simd::numOutVecs<Tout, Tin>();
  static constexpr int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;

  static SIMD_INLINE Float randomizeInputFloat()
  {
    return getRandomOrSpecialValueRanges<Float>();
  }

  // 03. Aug 22 (Jonas Keller):
  // include zeros in the randomization of the input vectors
  static SIMD_INLINE void randomizeInput(SerialVec<Tin, SIMD_WIDTH> &vec)
  // { vec.randomizeRangesNonZero(); }
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
                                double off, double fac,
  static SIMD_INLINE void apply(const VEC<Tin, SIMD_WIDTH> vecsIn[numInVecs],
                                VEC<Tout, SIMD_WIDTH> vecsOut[numOutVecs])
  {
    fmuladd(vecsIn, fac, off, vecsOut);
  }
};

template <typename Tout, typename Tin, int SIMD_WIDTH>
struct Fwaddmul
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("fwaddmul");
  }

  static constexpr int numInVecs   = ::simd::numInVecs<Tout, Tin>();
  static constexpr int numInElems  = Vec<Tin, SIMD_WIDTH>::elems;
  static constexpr int numOutVecs  = ::simd::numOutVecs<Tout, Tin>();
  static constexpr int numOutElems = Vec<Tout, SIMD_WIDTH>::elems;

  static SIMD_INLINE Float randomizeInputFloat()
  {
    return getRandomOrSpecialValueRanges<Float>();
  }

  static SIMD_INLINE void randomizeInput1(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  static SIMD_INLINE void randomizeInput2(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
                                double w, double fac,
  static SIMD_INLINE void apply(const VEC<Tin, SIMD_WIDTH> vecsIn1[numInVecs],
                                const VEC<Tin, SIMD_WIDTH> vecsIn2[numInVecs],
                                VEC<Tout, SIMD_WIDTH> vecsOut[numOutVecs])
  {
    fwaddmul(vecsIn1, vecsIn2, w, fac, vecsOut);
  }
};

template <typename T, int SIMD_WIDTH>
struct HaddMulti
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hadd"); }

  // number of input elements
  static constexpr int nInputs = SerialVec<T, SIMD_WIDTH>::elements;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(
    const VEC<T, SIMD_WIDTH> a[nInputs])
  {
    return hadd(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct HaddsMulti
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hadds"); }

  // number of input elements
  static constexpr int nInputs = SerialVec<T, SIMD_WIDTH>::elements;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(
    const VEC<T, SIMD_WIDTH> a[nInputs])
  {
    return hadds(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct HaddSingle
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hadd"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE T apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return hadd(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct HaddsSingle
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hadds"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE T apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return hadds(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct HminSingle
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hmin"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE T apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return hmin(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct HmaxSingle
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("hmax"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE T apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return hmax(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose2
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose2");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose2(inRows, outRows);
  }
};

template <int SLOPE, typename T, int SIMD_WIDTH>
struct BitonicSort
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<SLOPE, T, SIMD_WIDTH>("bitonicSort");
  }

  static constexpr int minI    = 0;
  static constexpr int maxI    = 2;
  static constexpr int nextI   = SLOPE + 1;
  static constexpr int numVecs = SerialVec<T, SIMD_WIDTH>::elements;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> vecs[numVecs])
  {
    bitonicSort<static_cast<SortSlope>(SLOPE)>(vecs);
  }
};

template <int SLOPE, typename T, int SIMD_WIDTH>
struct BitonicSort2
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<SLOPE, T, SIMD_WIDTH>("bitonicSort2");
  }

  static constexpr int minI    = 0;
  static constexpr int maxI    = 2;
  static constexpr int nextI   = SLOPE + 1;
  static constexpr int numVecs = SerialVec<T, SIMD_WIDTH>::elements;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> vecs[numVecs])
  {
    bitonicSort2<static_cast<SortSlope>(SLOPE)>(vecs);
  }
};

template <int SLOPE, typename T, int SIMD_WIDTH>
struct BitonicSortSortedPairs
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<SLOPE, T, SIMD_WIDTH>("bitonicSortSortedPairs");
  }

  static constexpr int minI    = 0;
  static constexpr int maxI    = 2;
  static constexpr int nextI   = SLOPE + 1;
  static constexpr int numVecs = SerialVec<T, SIMD_WIDTH>::elements;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeSorted(static_cast<SortSlope>(SLOPE));
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> vecs[numVecs])
  {
    bitonicSortSortedPairs<static_cast<SortSlope>(SLOPE)>(vecs);
  }
};

template <int SLOPE, typename T, int SIMD_WIDTH>
struct BitonicSortSortedPairs2
{
  static SIMD_INLINE std::string name()
  {
    return ti2s<SLOPE, T, SIMD_WIDTH>("bitonicSortSortedPairs2");
  }

  static constexpr int minI    = 0;
  static constexpr int maxI    = 2;
  static constexpr int nextI   = SLOPE + 1;
  static constexpr int numVecs = SerialVec<T, SIMD_WIDTH>::elements;

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomizeSorted(static_cast<SortSlope>(SLOPE));
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(VEC<T, SIMD_WIDTH> vecs[numVecs])
  {
    bitonicSortSortedPairs2<static_cast<SortSlope>(SLOPE)>(vecs);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose0
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose0");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose0(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose16
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose16");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose16(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose3
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose3");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose3(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose4
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose4");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose4(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose5
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose5");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose5(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose1_16
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose1_16");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose1_16_t(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Transpose6
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("transpose6");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE void apply(
    const VEC<T, SIMD_WIDTH> inRows[VEC<T, SIMD_WIDTH>::elements],
    VEC<T, SIMD_WIDTH> outRows[VEC<T, SIMD_WIDTH>::elements])
  {
    transpose6_t(inRows, outRows);
  }
};

template <typename T, int SIMD_WIDTH>
struct Avgru
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("avgru"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return avgru(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Avgrd
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("avgrd"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return avgrd(a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Setones
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("setones");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _setones<T, SIMD_WIDTH>();
  }
};

template <typename T, int SIMD_WIDTH>
struct Setmin
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("setmin"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _setmin<T, SIMD_WIDTH>();
  }
};

template <typename T, int SIMD_WIDTH>
struct Setmax
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("setmax"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _setmax<T, SIMD_WIDTH>();
  }
};

template <typename T, int SIMD_WIDTH>
struct Setnegunity
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("setnegunity");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _setnegunity<T, SIMD_WIDTH>();
  }
};

template <typename T, int SIMD_WIDTH>
struct Setunity
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("setunity");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _setunity<T, SIMD_WIDTH>();
  }
};

// TODO: generate random inputs which are often also small
template <typename Tout, typename Tin, int SIMD_WIDTH>
struct PacksMulti
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tout, Tin, SIMD_WIDTH>("packs");
  }

  // number of input elements
  static constexpr int nInputs = sizeof(Tin) / sizeof(Tout);

  static SIMD_INLINE void randomizeInput(SerialVec<Tin, SIMD_WIDTH> &vec)
  {
    vec.randomizeRanges();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<Tout, SIMD_WIDTH> apply(
    const VEC<Tin, SIMD_WIDTH> a[nInputs])
  {
    return packs<Tout>(a);
  }
};

template <typename Tcond, typename T, int SIMD_WIDTH>
struct Ifelse
{
  static SIMD_INLINE std::string name()
  {
    return tt2s<Tcond, T, SIMD_WIDTH>("ifelse");
  }

  static SIMD_INLINE void randomizeCond(SerialVec<Tcond, SIMD_WIDTH> &vec)
  {
    vec.randomizeCond();
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<Tcond, SIMD_WIDTH> &m,
                                              const VEC<T, SIMD_WIDTH> &a,
                                              const VEC<T, SIMD_WIDTH> &b)
  {
    return ifelse(m, a, b);
  }
};

template <typename T, int SIMD_WIDTH>
struct Div2r0
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("div2r0"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return div2r0(a);
  }
};

template <typename T, int SIMD_WIDTH>
struct Div2rd
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("div2rd"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return div2rd(a);
  }
};

// 27. Aug 22 (Jonas Keller): added msb2int

template <typename T, int SIMD_WIDTH>
struct Msb2int
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("msb2int");
  }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE uint64_t apply(const VEC<T, SIMD_WIDTH> &a)
  {
    return msb2int(a);
  }
};

// 06. Oct 22 (Jonas Keller): added int2msb

template <typename T, int SIMD_WIDTH>
struct Int2msb
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("int2msb");
  }

  static SIMD_INLINE uint64_t randomizeInputScalar()
  {
    return getRandom<uint64_t>();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const uint64_t &a)
  {
    return Deductor<VEC>::template _int2msb<T, SIMD_WIDTH>(a);
  }
};

// 09. Oct 22 (Jonas Keller): added int2bits

template <typename T, int SIMD_WIDTH>
struct Int2bits
{
  static SIMD_INLINE std::string name()
  {
    return t2s<T, SIMD_WIDTH>("int2bits");
  }

  static SIMD_INLINE uint64_t randomizeInputScalar()
  {
    return getRandom<uint64_t>();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply(const uint64_t &a)
  {
    return Deductor<VEC>::template _int2bits<T, SIMD_WIDTH>(a);
  }
};

// 07. Feb 23 (Jonas Keller): added Iota

template <typename T, int SIMD_WIDTH>
struct Iota
{
  static SIMD_INLINE std::string name() { return t2s<T, SIMD_WIDTH>("iota"); }

  static SIMD_INLINE void randomizeInput(SerialVec<T, SIMD_WIDTH> &vec)
  {
    vec.randomize();
  }

  template <template <typename, int> class VEC>
  static SIMD_INLINE VEC<T, SIMD_WIDTH> apply()
  {
    return Deductor<VEC>::template _iota<T, SIMD_WIDTH>();
  }
};

} // namespace auto_test
} // namespace simd

#endif // SIMD_VEC_AUTO_TEST_WRAPPER_H_
