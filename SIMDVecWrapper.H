// ===========================================================================
// 
// SIMDVecWrapper.H --
// class wrapper, for both SIMDVec and SIMDSerialVec, used for tests
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

#ifndef _SIMD_VEC_WRAPPER_
#define _SIMD_VEC_WRAPPER_

#include <string>
#include <stdio.h>
#include <algorithm>
#include <functional>
#include "SIMDDefs.H"
#include "SIMDTypes.H"
#include "SIMDSerialVec.H"

// - template parameters T and SIMD_WIDTH are included in class
//   definition and therefore have to be explicitly provided (no
//   deduction possible), this allows members or methods to use these
//   parameters (e.g. maxI)
// - apply() has a template-template parameter such that both SIMDVec
//   and SIMDSerialVec can be passed (at least if the function has at
//   least one argument of SIMDVec or SIMDSerialVec)

namespace ns_simd {

  // -------------------------------------------------------------------------
  // aux. functions
  // -------------------------------------------------------------------------

  template <typename T, int SIMD_WIDTH>
  static std::string t2s(const char *fct)
  {
    char res[256];
    sprintf(res, "%s<%s,%d>", 
	    fct, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return std::string(res);
  }

  template <int I, typename T, int SIMD_WIDTH>
  static std::string ti2s(const char* fct)
  {
    char res[256];
    sprintf(res, "%s<%d,%s,%d>", 
	    fct, I, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return std::string(res);
  }

  template <int I, int J, typename T, int SIMD_WIDTH>
  static std::string tii2s(const char* fct)
  {
    char res[256];
    sprintf(res, "%s<%d,%d,%s,%d>", 
	    fct, I, J, SIMDTypeInfo<T>::name(), SIMD_WIDTH);
    return std::string(res);
  }
  
  template <typename Tout, typename T, int SIMD_WIDTH>
  static std::string tt2s(const char *fct)
  {
    char res[256];
    sprintf(res, "%s<%s,%s,%d>", 
	    fct, SIMDTypeInfo<Tout>::name(), SIMDTypeInfo<T>::name(), 
	    SIMD_WIDTH);
    return std::string(res);    
  }

  // -------------------------------------------------------------------------
  // wrapper
  // -------------------------------------------------------------------------

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Reinterpret
  {
    static SIMD_INLINE std::string name() 
    { return tt2s<Tout,T,SIMD_WIDTH>("reinterpret"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<Tout,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return reinterpret<Tout>(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Setzero
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("setzero"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply()
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_setzero();
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Set1
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("set1"); }

    static SIMD_INLINE T
    randomizeInputScalar()
    { return getRandom<T>(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(T a)
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_set1(a);
    }
  };

  template <int I, typename T, int SIMD_WIDTH>
  struct Extract
  {
    static SIMD_INLINE std::string name() 
    { return ti2s<I,T,SIMD_WIDTH>("extract"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = SIMD_WIDTH / sizeof(T) + 1;
    static SIMD_CONSTEXPR int nextI = I + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return extract<I>(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Add
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("add"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return add(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Adds
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("adds"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return adds(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Sub
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("sub"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return sub(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Subs
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("subs"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return subs(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Neg
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("neg"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return neg(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Mul
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("mul"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return mul(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Div
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("div"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return div(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Ceil
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("ceil"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return ceil(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Floor
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("floor"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return floor(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Round
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("round"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return round(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Truncate
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("truncate"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return truncate(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Sqrt
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("sqrt"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeNonNegative(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return sqrt(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Min
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("min"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return min(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Max
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("max"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return max(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Abs
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("abs"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return abs(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Sign
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("sign"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return sign(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct AbsDiff
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("absDiff"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return absDiff(a, b);
    }
  };

  // SIMDVecExt.H has a template class Unpack, so we add an underscore
  template <int PART, int NUM_ELEMS, typename T, int SIMD_WIDTH>
  struct Unpack_
  {
    static SIMD_INLINE std::string name()
    { return tii2s<PART,NUM_ELEMS,T,SIMD_WIDTH>("unpack"); }

    // PART (I) switches between 0 and 1
    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = 1;
    static SIMD_CONSTEXPR int nextI = 1 - PART;
    // NUM_ELEMS (J) is doubled if PART == 1
    static SIMD_CONSTEXPR int minJ = 1;
    static SIMD_CONSTEXPR int maxJ = SIMD_WIDTH / sizeof(T);
    static SIMD_CONSTEXPR int nextJ = PART ? NUM_ELEMS * 2 : NUM_ELEMS;
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T, SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a,
	  const VEC<T,SIMD_WIDTH> &b)
    {
      return unpack<PART,NUM_ELEMS>(a, b);
    }
  };

  // SIMDVecExt.H has a template class Zip, so we add an underscore
  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  struct Zip_
  {
    static SIMD_INLINE std::string name()
    { return ti2s<NUM_ELEMS,T,SIMD_WIDTH>("zip"); }

    static SIMD_CONSTEXPR int minI = 1;
    static SIMD_CONSTEXPR int maxI = SIMD_WIDTH / sizeof(T);
    static SIMD_CONSTEXPR int nextI = NUM_ELEMS * 2;
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> a,
	  const VEC<T,SIMD_WIDTH> b,
	  VEC<T,SIMD_WIDTH> &c,
	  VEC<T,SIMD_WIDTH> &d)
    {
      zip<NUM_ELEMS>(a, b, c, d);
    }
  };

  template <int NUM_ELEMS, typename T, int SIMD_WIDTH>
  struct Unzip
  {
    static SIMD_INLINE std::string name()
    { return ti2s<NUM_ELEMS,T,SIMD_WIDTH>("unzip"); }

    static SIMD_CONSTEXPR int minI = 1;
    static SIMD_CONSTEXPR int maxI = SIMD_WIDTH / sizeof(T);
    static SIMD_CONSTEXPR int nextI = NUM_ELEMS * 2;
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> a,
	  const VEC<T,SIMD_WIDTH> b,
	  VEC<T,SIMD_WIDTH> &c,
	  VEC<T,SIMD_WIDTH> &d)
    {
      unzip<NUM_ELEMS>(a, b, c, d);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Packs
  {
    static SIMD_INLINE std::string name() 
    { return tt2s<Tout,T,SIMD_WIDTH>("packs"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRanges(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<Tout,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return packs<Tout>(a, b);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Extend
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("extend"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> &a,
	  VEC<Tout,SIMD_WIDTH> *const b)
    {
      extend(a, b);
    }
  };

  template <int IMM, typename T, int SIMD_WIDTH>
  struct Srai
  {
    static SIMD_INLINE std::string name()
    { return ti2s<IMM,T,SIMD_WIDTH>("srai"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = sizeof(T) * 8 + 1;
    static SIMD_CONSTEXPR int nextI = IMM + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return srai<IMM>(a);
    }
  };

  template <int IMM, typename T, int SIMD_WIDTH>
  struct Srli
  {
    static SIMD_INLINE std::string name()
    { return ti2s<IMM,T,SIMD_WIDTH>("srli"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = sizeof(T) * 8 + 1;
    static SIMD_CONSTEXPR int nextI = IMM + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return srli<IMM>(a);
    }
  };

  template <int IMM, typename T, int SIMD_WIDTH>
  struct Slli
  {
    static SIMD_INLINE std::string name()
    { return ti2s<IMM,T,SIMD_WIDTH>("slli"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = sizeof(T) * 8 + 1;
    static SIMD_CONSTEXPR int nextI = IMM + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return slli<IMM>(a);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Cvts
  {
    static SIMD_INLINE std::string name() 
    { return tt2s<Tout,T,SIMD_WIDTH>("cvts"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<Tout,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return cvts<Tout>(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Hadd
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hadd"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return hadd(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Hadds
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hadds"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return hadds(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Hsub
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hsub"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return hsub(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Hsubs
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hsubs"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return hsubs(a, b);
    }
  };

  template <int IMM, typename T, int SIMD_WIDTH>
  struct Srle
  {
    static SIMD_INLINE std::string name()
    { return ti2s<IMM,T,SIMD_WIDTH>("srle"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = SIMD_WIDTH / sizeof(T) + 1;
    static SIMD_CONSTEXPR int nextI = IMM + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return srle<IMM>(a);
    }
  };

  template <int IMM, typename T, int SIMD_WIDTH>
  struct Slle
  {
    static SIMD_INLINE std::string name()
    { return ti2s<IMM,T,SIMD_WIDTH>("slle"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = SIMD_WIDTH / sizeof(T) + 1;
    static SIMD_CONSTEXPR int nextI = IMM + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return slle<IMM>(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Elem0
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("elem0"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return elem0(a);
    }
  };

  template <int IMM, typename T, int SIMD_WIDTH>
  struct Alignre
  {
    static SIMD_INLINE std::string name()
    { return ti2s<IMM,T,SIMD_WIDTH>("alignre"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = 2 * (SIMD_WIDTH / sizeof(T)) + 1;
    static SIMD_CONSTEXPR int nextI = IMM + 1;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a,
	  const VEC<T,SIMD_WIDTH> &b)
    {
      return alignre<IMM>(a, b);
    }
  };

  template <int I, typename T, int SIMD_WIDTH>
  struct _Swizzle
  {
    static SIMD_INLINE std::string name()
    { return ti2s<I,T,SIMD_WIDTH>("swizzle"); }

    static SIMD_CONSTEXPR int minI = 1;
    static SIMD_CONSTEXPR int maxI = 6;
    static SIMD_CONSTEXPR int nextI = I + 1;
    static SIMD_CONSTEXPR int numVecs = I;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const v)
    {
      _swizzle<I>(v);
    }    
  };

  template <int I, typename T, int SIMD_WIDTH>
  struct _Swizzle2
  {
    static SIMD_INLINE std::string name()
    { return ti2s<I,T,SIMD_WIDTH>("swizzle2"); }

    static SIMD_CONSTEXPR int minI = 1;
    // TODO: is there an upper limit for maxI in swizzle2?
    static SIMD_CONSTEXPR int maxI = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int nextI = I + 1;
    static SIMD_CONSTEXPR int numVecs = 2 * I;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const v)
    {
      _swizzle2<I>(v);
    }    
  };

  template <int I, typename T, int SIMD_WIDTH>
  struct _Unswizzle2
  {
    static SIMD_INLINE std::string name()
    { return ti2s<I,T,SIMD_WIDTH>("unswizzle2"); }

    static SIMD_CONSTEXPR int minI = 1;
    // TODO: is there an upper limit for maxI in unswizzle2?
    static SIMD_CONSTEXPR int maxI = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int nextI = I + 1;
    static SIMD_CONSTEXPR int numVecs = 2 * I;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const v)
    {
      _unswizzle2<I>(v);
    }    
  };

  template <typename T, int SIMD_WIDTH>
  struct Cmplt
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("cmplt"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return cmplt(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Cmple
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("cmple"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return cmple(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Cmpeq
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("cmpeq"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return cmpeq(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Cmpge
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("cmpge"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return cmpge(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Cmpgt
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("cmpgt"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return cmpgt(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Cmpneq
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("cmpneq"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return cmpneq(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct And
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("and"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return and(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Or
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("or"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return or(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Andnot
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("andnot"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return andnot(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Xor
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("xor"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return xor(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Not
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("not"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return not(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Avg
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("avg"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return avg(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Test_all_zeros
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("test_all_zeros"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { 
      if (rand() & 0x01)
	vec.randomize(); 
      else
	vec.setzero();
    }

    template <template <typename, int> class VEC>
    static SIMD_INLINE int
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return test_all_zeros(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Test_all_ones
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("test_all_ones"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { 
      if (rand() & 0x01) 
	vec.randomize(); 
      else
	vec.setones();
    }

    template <template <typename, int> class VEC>
    static SIMD_INLINE int
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return test_all_ones(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Reverse
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("reverse"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return reverse(a);
    }
  };
  
  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Convert
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("convert"); }

    static SIMD_CONSTEXPR int numInVecs = NUM_INPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numInElems = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int numOutVecs = NUM_OUTPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRanges(); }
    
    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inVecs, 
	  VEC<Tout,SIMD_WIDTH> *const outVecs)
    {
      convert(inVecs, outVecs);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Fdivmul
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("fdivmul"); }

    static SIMD_CONSTEXPR int numInVecs = NUM_INPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numInElems = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int numOutVecs = NUM_OUTPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);

    static SIMD_INLINE SIMDFloat
    randomizeInputFloat()
    { return getRandomRanges<SIMDFloat>(); }

    static SIMD_INLINE void
    randomizeInput1(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRanges(); }
    
    static SIMD_INLINE void
    randomizeInput2(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRangesNonZero(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const vecsNum, 
	  const VEC<T,SIMD_WIDTH> *const vecsDenom, 
	  double fac,
	  VEC<Tout,SIMD_WIDTH> *const vecsOut)
    {
      fdivmul(vecsNum, vecsDenom, fac, vecsOut);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Fmul
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("fmul"); }

    static SIMD_CONSTEXPR int numInVecs = NUM_INPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numInElems = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int numOutVecs = NUM_OUTPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);

    static SIMD_INLINE SIMDFloat
    randomizeInputFloat()
    { return getRandomRanges<SIMDFloat>(); }

    // 03. Aug 22 (Jonas Keller):
    // include zeros in the randomization of the input vectors
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    // { vec.randomizeRangesNonZero(); }
    { vec.randomizeRanges(); }
    
    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const vecsIn, 
	  double fac,
	  VEC<Tout,SIMD_WIDTH> *const vecsOut)
    {
      fmul(vecsIn, fac, vecsOut);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Faddmul
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("faddmul"); }

    static SIMD_CONSTEXPR int numInVecs = NUM_INPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numInElems = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int numOutVecs = NUM_OUTPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);

    static SIMD_INLINE SIMDFloat
    randomizeInputFloat()
    { return getRandomRanges<SIMDFloat>(); }

    // 03. Aug 22 (Jonas Keller):
    // include zeros in the randomization of the input vectors
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    // { vec.randomizeRangesNonZero(); }
    { vec.randomizeRanges(); }
    
    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const vecsIn, 
	  double off,
	  double fac,
	  VEC<Tout,SIMD_WIDTH> *const vecsOut)
    {
      faddmul(vecsIn, off, fac, vecsOut);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Fmuladd
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("fmuladd"); }

    static SIMD_CONSTEXPR int numInVecs = NUM_INPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numInElems = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int numOutVecs = NUM_OUTPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);

    static SIMD_INLINE SIMDFloat
    randomizeInputFloat()
    { return getRandomRanges<SIMDFloat>(); }

    // 03. Aug 22 (Jonas Keller):
    // include zeros in the randomization of the input vectors
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    // { vec.randomizeRangesNonZero(); }
    { vec.randomizeRanges(); }
    
    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const vecsIn, 
	  double off,
	  double fac,
	  VEC<Tout,SIMD_WIDTH> *const vecsOut)
    {
      fmuladd(vecsIn, fac, off, vecsOut);
    }
  };

  template <typename Tout, typename T, int SIMD_WIDTH>
  struct Fwaddmul
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tout,T,SIMD_WIDTH>("fwaddmul"); }

    static SIMD_CONSTEXPR int numInVecs = NUM_INPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numInElems = NUM_SIMDVEC_ELEMENTS(T, SIMD_WIDTH);
    static SIMD_CONSTEXPR int numOutVecs = NUM_OUTPUT_SIMDVECS(Tout, T);
    static SIMD_CONSTEXPR int numOutElems = NUM_SIMDVEC_ELEMENTS(Tout, SIMD_WIDTH);

    static SIMD_INLINE SIMDFloat
    randomizeInputFloat()
    { return getRandomRanges<SIMDFloat>(); }

    static SIMD_INLINE void
    randomizeInput1(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRanges(); }
    
    static SIMD_INLINE void
    randomizeInput2(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRanges(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const vecsIn1, 
	  const VEC<T,SIMD_WIDTH> *const vecsIn2, 
	  double w,
	  double fac,
	  VEC<Tout,SIMD_WIDTH> *const vecsOut)
    {
      fwaddmul(vecsIn1, vecsIn2, w, fac, vecsOut);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HaddMulti
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("hadd"); }

    // number of input elements
    static SIMD_CONSTEXPR int nInputs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> *const a)
    {
      return hadd(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HaddsMulti
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("hadds"); }

    // number of input elements
    static SIMD_CONSTEXPR int nInputs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> *const a)
    {
      return hadds(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HsubMulti
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("hsub"); }

    // number of input elements
    static SIMD_CONSTEXPR int nInputs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> *const a)
    {
      return hsub(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HsubsMulti
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("hsubs"); }

    // number of input elements
    static SIMD_CONSTEXPR int nInputs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> *const a)
    {
      return hsubs(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HaddSingle
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hadd"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return hadd(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HaddsSingle
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hadds"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return hadds(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HsubSingle
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hsub"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return hsub(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HsubsSingle
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hsubs"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return hsubs(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HminSingle
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hmin"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return hmin(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct HmaxSingle
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("hmax"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE T
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return hmax(a);
    }
  };

  // SIMDVecExt.H has a template class Transpose, so we add an underscore
  template <typename T, int SIMD_WIDTH>
  struct Transpose_
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("transpose"); }
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
	  VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose2_
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("transpose2"); }
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
	  VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose2(inRows, outRows);
    }
  };

  template <int SLOPE, typename T, int SIMD_WIDTH>
  struct BitonicSort
  {
    static SIMD_INLINE std::string name() 
    { return ti2s<SLOPE,T,SIMD_WIDTH>("bitonicSort"); }

    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = 2;
    static SIMD_CONSTEXPR int nextI = SLOPE + 1;
    static SIMD_CONSTEXPR int numVecs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const vecs)
    {
      bitonicSort<SLOPE>(vecs);
    }
  };

  template <int SLOPE, typename T, int SIMD_WIDTH>
  struct BitonicSort2
  {
    static SIMD_INLINE std::string name() 
    { return ti2s<SLOPE,T,SIMD_WIDTH>("bitonicSort2"); }
    
    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = 2;
    static SIMD_CONSTEXPR int nextI = SLOPE + 1;
    static SIMD_CONSTEXPR int numVecs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const vecs)
    {
      bitonicSort2<SLOPE>(vecs);
    }
  };
  
  template <int SLOPE, typename T, int SIMD_WIDTH>
  struct BitonicSortSortedPairs
  {
    static SIMD_INLINE std::string name() 
    { return ti2s<SLOPE,T,SIMD_WIDTH>("bitonicSortSortedPairs"); }
    
    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = 2;
    static SIMD_CONSTEXPR int nextI = SLOPE + 1;
    static SIMD_CONSTEXPR int numVecs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeSorted(SLOPE); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const vecs)
    {
      bitonicSortSortedPairs<SLOPE>(vecs);
    }
  };

  template <int SLOPE, typename T, int SIMD_WIDTH>
  struct BitonicSortSortedPairs2
  {
    static SIMD_INLINE std::string name() 
    { return ti2s<SLOPE,T,SIMD_WIDTH>("bitonicSortSortedPairs2"); }
    
    static SIMD_CONSTEXPR int minI = 0;
    static SIMD_CONSTEXPR int maxI = 2;
    static SIMD_CONSTEXPR int nextI = SLOPE + 1;
    static SIMD_CONSTEXPR int numVecs = SIMDSerialVec<T,SIMD_WIDTH>::elements;

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeSorted(SLOPE) ;}

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(VEC<T,SIMD_WIDTH> *const vecs)
    {
      bitonicSortSortedPairs2<SLOPE>(vecs);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose0_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose0"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose0(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose16_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose16"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose16(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose3_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose3"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose3(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose4_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose4"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose4(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose5_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose5"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose5(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose1_16_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose1_16"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose1_16_t(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Transpose6_
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("transpose6"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE void
    apply(const VEC<T,SIMD_WIDTH> *const inRows,
          VEC<T,SIMD_WIDTH> *const outRows)
    {
      transpose6_t(inRows, outRows);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Avgru
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("avgru"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return avgru(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Avgrd
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("avgrd"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a, const VEC<T,SIMD_WIDTH> &b)
    {
      return avgrd(a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Setones
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("setones"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply()
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_setones();
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Setmin
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("setmin"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply()
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_setmin();
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Setmax
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("setmax"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply()
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_setmax();
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Setnegunity
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("setnegunity"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply()
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_setnegunity();
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Setunity
  {
    static SIMD_INLINE std::string name()
    { return t2s<T,SIMD_WIDTH>("setunity"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply()
    {
      return Deductor<T,SIMD_WIDTH,VEC>::_setunity();
    }
  };

  // TODO: generate random inputs which are often also small
  template <typename Tout, typename T, int SIMD_WIDTH>
  struct PacksMulti
  {
    static SIMD_INLINE std::string name() 
    { return tt2s<Tout,T,SIMD_WIDTH>("packs"); }

    // number of input elements
    static SIMD_CONSTEXPR int nInputs = sizeof(T) / sizeof(Tout);
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomizeRanges(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<Tout,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> *const a)
    {
      return packs<Tout>(a);
    }
  };

  template <typename Tcond, typename T, int SIMD_WIDTH>
  struct Ifelse
  {
    static SIMD_INLINE std::string name()
    { return tt2s<Tcond,T,SIMD_WIDTH>("ifelse"); }

    static SIMD_INLINE void
    randomizeCond(SIMDSerialVec<Tcond,SIMD_WIDTH> &vec)
    { vec.randomizeCond(); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<Tcond,SIMD_WIDTH> &m,
	  const VEC<T,SIMD_WIDTH> &a,
	  const VEC<T,SIMD_WIDTH> &b)
    {
      return ifelse(m, a, b);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Div2r0
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("div2r0"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return div2r0(a);
    }
  };

  template <typename T, int SIMD_WIDTH>
  struct Div2rd
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("div2rd"); }

    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }

    template <template <typename, int> class VEC>
    static SIMD_INLINE VEC<T,SIMD_WIDTH>
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return div2rd(a);
    }
  };

  // 27. Aug 22 (Jonas Keller): added movemask

  template <typename T, int SIMD_WIDTH>
  struct Movemask
  {
    static SIMD_INLINE std::string name() 
    { return t2s<T,SIMD_WIDTH>("movemask"); }
    
    static SIMD_INLINE void
    randomizeInput(SIMDSerialVec<T,SIMD_WIDTH> &vec)
    { vec.randomize(); }
    
    template <template <typename, int> class VEC>
    static SIMD_INLINE uint64_t
    apply(const VEC<T,SIMD_WIDTH> &a)
    {
      return movemask(a);
    }
  };

} // namespace

#endif
