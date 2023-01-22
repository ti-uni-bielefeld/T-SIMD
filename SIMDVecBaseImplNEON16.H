// ===========================================================================
// 
// SIMDVecBaseImplNEON16.H --
// encapsulation for ARM NEON vector extension
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

// NOTES:
//
// echo | gcc -E -dM -mcpu=cortex-a9 -mfpu=neon - | more
// echo | arm-linux-gnueabihf-gcc -E -dM -mcpu=cortex-a15 -mfpu=neon - | more
//
// -mfpu=neon
// -mfpu=neon-fp16
//
// GCC 4.9:
// GCC now supports Cortex-A12 and the Cortex-R7 through the
// -mcpu=cortex-a12 and -mcpu=cortex-r7 options.
// 
// GCC now has tuning for the Cortex-A57 and Cortex-A53 through the
// -mcpu=cortex-a57 and -mcpu=cortex-a53 options.
// 
// Initial big.LITTLE tuning support for the combination of Cortex-A57
// and Cortex-A53 was added through the -mcpu=cortex-a57.cortex-a53
// option. Similar support was added for the combination of Cortex-A15
// and Cortex-A7 through the -mcpu=cortex-a15.cortex-a7 option.

#ifndef _SIMD_VEC_BASE_IMPL_NEON_16_H_
#define _SIMD_VEC_BASE_IMPL_NEON_16_H_

#include "SIMDAlloc.H"
#include "SIMDDefs.H"
#include "SIMDIntrinsNEON.H"
#include "SIMDTypes.H"
#include "SIMDVec.H"

#include <algorithm>
#include <stdint.h>

#ifdef SIMDVEC_NEON_ENABLE

namespace ns_simd {
namespace internal {
namespace base {
  // =========================================================================
  // type templates
  // =========================================================================

  // -------------------------------------------------------------------------
  // default vector type collection
  // -------------------------------------------------------------------------

  template <typename T> struct SIMDVecNeon;
  
#define SIMDVEC_NEON(T,NEON_T,NUM,NUM64)			\
  template <> struct SIMDVecNeon<T>				\
  {								\
    typedef NEON_T ## _t ScalarType;				\
    typedef NEON_T ## x ## NUM ## _t Type;			\
    typedef NEON_T ## x ## NUM64 ## _t Type64;  		\
    typedef NEON_T ## x ## NUM ## x2_t TypeX2;	     	        \
  };

  SIMDVEC_NEON(SIMDByte,uint8,16,8)
  SIMDVEC_NEON(SIMDSignedByte,int8,16,8)
  SIMDVEC_NEON(SIMDWord,uint16,8,4)
  SIMDVEC_NEON(SIMDShort,int16,8,4)
  SIMDVEC_NEON(SIMDInt,int32,4,2)
  SIMDVEC_NEON(SIMDFloat,float32,4,2)

  // -------------------------------------------------------------------------
  // single element array type (for generalizations)
  // -------------------------------------------------------------------------

  // defined as the other array types in arm_neon.h
  // but with type conversion constructor and operator=
#define SIMDVEC_NEON_64X1(NEON_T)					\
  typedef struct NEON_T ## x1_t						\
  {									\
    NEON_T ## _t val[1];						\
    NEON_T ## x1_t() {}							\
    NEON_T ## x1_t(const NEON_T ## _t &x) { val[0] = x; }		\
    NEON_T ## x1_t& operator=(const NEON_T ## _t &x)			\
      { val[0] = x; return *this;}					\
  } NEON_T ## x1_t;

  SIMDVEC_NEON_64X1(uint8x8)
  SIMDVEC_NEON_64X1(int8x8)
  SIMDVEC_NEON_64X1(uint16x4)
  SIMDVEC_NEON_64X1(int16x4)
  SIMDVEC_NEON_64X1(int32x2)
  SIMDVEC_NEON_64X1(float32x2)

  // -------------------------------------------------------------------------
  // 64bit array type collection
  // -------------------------------------------------------------------------

  template <int N, typename T> struct SIMDVecNeonArray64;
  
#define SIMDVEC_NEON_ARRAY64(NUM,T,NEON_T)		\
  template <> struct SIMDVecNeonArray64<NUM,T>		\
  {							\
    typedef NEON_T ## x ## NUM ## _t Type;		\
    typedef NEON_T ## _t ValType;			\
  };

#define SIMDVEC_NEON_ARRAY64_ALLNUM(T,NEON_T)	\
  SIMDVEC_NEON_ARRAY64(1,T,NEON_T)		\
  SIMDVEC_NEON_ARRAY64(2,T,NEON_T)		\
  SIMDVEC_NEON_ARRAY64(3,T,NEON_T)		\
  SIMDVEC_NEON_ARRAY64(4,T,NEON_T)

  SIMDVEC_NEON_ARRAY64_ALLNUM(SIMDByte,uint8x8)
  SIMDVEC_NEON_ARRAY64_ALLNUM(SIMDSignedByte,int8x8)
  SIMDVEC_NEON_ARRAY64_ALLNUM(SIMDWord,uint16x4)
  SIMDVEC_NEON_ARRAY64_ALLNUM(SIMDShort,int16x4)
  SIMDVEC_NEON_ARRAY64_ALLNUM(SIMDInt,int32x2)
  SIMDVEC_NEON_ARRAY64_ALLNUM(SIMDFloat,float32x2)

} // namespace base
} // namespace internal

  // =========================================================================
  // SIMDVec instantiation for NEON
  // =========================================================================

  template <typename T>
  class SIMDVec<T,16>
  {
  public:
    typedef T Type;
    typedef typename internal::base::SIMDVecNeon<T>::Type RegType;
    RegType reg;
    enum { elements = 16 / sizeof(T), bytes = 16 };
    // shorter version:
    enum { elems = elements };
    SIMDVec() {}
    SIMDVec(const RegType &x) { reg = x; }
    SIMDVec& operator=(const RegType &x) 
    { reg = x; return *this; }
    operator RegType() const { return reg; }
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

  // =========================================================================
  // auxiliary NEON functions
  // =========================================================================
 
  // -------------------------------------------------------------------------
  // vreinterpret[q] with same input and output type (not avail. as intrinsic)
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_VREINTERPRET_SAME(T,NEON_SUF)			\
  static SIMD_INLINE typename internal::base::SIMDVecNeon<T>::Type			\
  vreinterpretq_ ## NEON_SUF ## _ ## NEON_SUF				\
  (typename internal::base::SIMDVecNeon<T>::Type a)					\
  {									\
    return a;								\
  }									\
  static SIMD_INLINE typename internal::base::SIMDVecNeon<T>::Type64			\
  vreinterpret_ ## NEON_SUF ## _ ## NEON_SUF				\
  (typename internal::base::SIMDVecNeon<T>::Type64 a)					\
  {									\
    return a;								\
  }
  
  SIMDVEC_NEON_VREINTERPRET_SAME(SIMDByte,u8)		
  SIMDVEC_NEON_VREINTERPRET_SAME(SIMDSignedByte,s8)	
  SIMDVEC_NEON_VREINTERPRET_SAME(SIMDWord,u16)		
  SIMDVEC_NEON_VREINTERPRET_SAME(SIMDShort,s16)	
  SIMDVEC_NEON_VREINTERPRET_SAME(SIMDInt,s32)		
  SIMDVEC_NEON_VREINTERPRET_SAME(SIMDFloat,f32)	

namespace internal {
namespace base {

  // -------------------------------------------------------------------------
  // replace a NEON function by a different NEON function
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_BINARY_REPLACE(T,NEON_FCT,NEON_REPL_FCT,NEON_SUF)	\
  static SIMD_INLINE typename SIMDVecNeon<T>::Type			\
  NEON_FCT ## _ ## NEON_SUF(typename SIMDVecNeon<T>::Type a,		\
			    typename SIMDVecNeon<T>::Type b)		\
  {									\
    return NEON_REPL_FCT ## _ ## NEON_SUF (a, b);			\
  }

  // -------------------------------------------------------------------------
  // implement float operation by int operation
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_BINARY_FLOAT_BY_INT(NEON_FCT)			\
  static SIMD_INLINE float32x4_t					\
  NEON_FCT ## _f32(float32x4_t a, float32x4_t b)			\
  {									\
    return vreinterpretq_f32_s32(NEON_FCT ## _s32			\
				 (vreinterpretq_s32_f32(a),		\
				  vreinterpretq_s32_f32(b)));		\
  }

  // -------------------------------------------------------------------------
  // implement float operation by int operation (64 bit width)
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_BINARY_FLOAT_BY_INT_64(NEON_FCT)			\
  static SIMD_INLINE float32x2_t					\
  NEON_FCT ## _f32(float32x2_t a, float32x2_t b)			\
  {									\
    return vreinterpret_f32_s32(NEON_FCT ## _s32			\
				(vreinterpret_s32_f32(a),		\
				 vreinterpret_s32_f32(b)));		\
  }

  // =========================================================================
  // macros
  // =========================================================================
 
  // -------------------------------------------------------------------------
  // reinterpret
  // -------------------------------------------------------------------------
  
  // wrapper for vreinterpretq
#define SIMDVEC_NEON_REINTERP(TDST,NEON_TDST,TSRC,NEON_TSRC)		\
  static SIMD_INLINE SIMDVec<TDST,16>					\
  reinterpret(const SIMDVec<TSRC,16>& vec, OutputType<TDST>)		\
  {									\
    return vreinterpretq_ ## NEON_TDST ## _ ## NEON_TSRC(vec);		\
  }									
  
  // wrapper for all dst types and same source type
#define SIMDVEC_NEON_REINTERP_ALLDST(TSRC,NEON_TSRC)			\
  SIMDVEC_NEON_REINTERP(SIMDByte,u8,TSRC,NEON_TSRC)			\
  SIMDVEC_NEON_REINTERP(SIMDSignedByte,s8,TSRC,NEON_TSRC)		\
  SIMDVEC_NEON_REINTERP(SIMDWord,u16,TSRC,NEON_TSRC)			\
  SIMDVEC_NEON_REINTERP(SIMDShort,s16,TSRC,NEON_TSRC)			\
  SIMDVEC_NEON_REINTERP(SIMDInt,s32,TSRC,NEON_TSRC)			\
  SIMDVEC_NEON_REINTERP(SIMDFloat,f32,TSRC,NEON_TSRC)

  // wrapper for all dst and src types
#define SIMDVEC_NEON_REINTERP_ALL()					\
  SIMDVEC_NEON_REINTERP_ALLDST(SIMDByte,u8)				\
  SIMDVEC_NEON_REINTERP_ALLDST(SIMDSignedByte,s8)			\
  SIMDVEC_NEON_REINTERP_ALLDST(SIMDWord,u16)				\
  SIMDVEC_NEON_REINTERP_ALLDST(SIMDShort,s16)				\
  SIMDVEC_NEON_REINTERP_ALLDST(SIMDInt,s32)				\
  SIMDVEC_NEON_REINTERP_ALLDST(SIMDFloat,f32)

  // -------------------------------------------------------------------------
  // binary functions (same input and output type)
  // -------------------------------------------------------------------------

  // wrapper for arbitrary binary function
#define SIMDVEC_NEON_BINARY(FCT,TYPE,NEON_FCT,NEON_SUF)			\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a,					\
      const SIMDVec<TYPE,16> &b)					\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (a, b);				\
  }

  // wrapper for arbitary binary function for all types
#define SIMDVEC_NEON_BINARY_ALL(FCT,NEON_FCT)		\
  SIMDVEC_NEON_BINARY(FCT,SIMDByte,NEON_FCT,u8)		\
  SIMDVEC_NEON_BINARY(FCT,SIMDSignedByte,NEON_FCT,s8)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDWord,NEON_FCT,u16)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDShort,NEON_FCT,s16)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDInt,NEON_FCT,s32)		\
  SIMDVEC_NEON_BINARY(FCT,SIMDFloat,NEON_FCT,f32)

  // wrapper for arbitary binary function for all integer types
#define SIMDVEC_NEON_BINARY_ALLINT(FCT,NEON_FCT)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDByte,NEON_FCT,u8)		\
  SIMDVEC_NEON_BINARY(FCT,SIMDSignedByte,NEON_FCT,s8)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDWord,NEON_FCT,u16)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDShort,NEON_FCT,s16)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDInt,NEON_FCT,s32)

  // -------------------------------------------------------------------------
  // binary functions (with reinterpret (from unsigned) to avoid type change)
  // -------------------------------------------------------------------------
  
  // wrapper for arbitrary binary function (with reinterpret...)
#define SIMDVEC_NEON_BINARY_REINTERP(FCT,TYPE,NEON_FCT,NEON_SUF,NEON_USUF) \
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a,					\
      const SIMDVec<TYPE,16> &b)					\
  {									\
    return vreinterpretq_ ## NEON_SUF ## _ ## NEON_USUF			\
      (NEON_FCT ## _ ## NEON_SUF (a, b));				\
  }
  
  // wrapper for arbitary binary function for all types (with reinterpret...)
#define SIMDVEC_NEON_BINARY_REINTERP_ALL(FCT,NEON_FCT)			\
  SIMDVEC_NEON_BINARY(FCT,SIMDByte,NEON_FCT,u8)				\
  SIMDVEC_NEON_BINARY_REINTERP(FCT,SIMDSignedByte,NEON_FCT,s8,u8)	\
  SIMDVEC_NEON_BINARY(FCT,SIMDWord,NEON_FCT,u16)			\
  SIMDVEC_NEON_BINARY_REINTERP(FCT,SIMDShort,NEON_FCT,s16,u16)		\
  SIMDVEC_NEON_BINARY_REINTERP(FCT,SIMDInt,NEON_FCT,s32,u32)		\
  SIMDVEC_NEON_BINARY_REINTERP(FCT,SIMDFloat,NEON_FCT,f32,u32)
  
  // -------------------------------------------------------------------------
  // unary functions
  // -------------------------------------------------------------------------
  
#define SIMDVEC_NEON_UNARY(FCT,TYPE,NEON_FCT,NEON_SUF)	\
  static SIMD_INLINE SIMDVec<TYPE,16>			\
  FCT(const SIMDVec<TYPE,16> &a)			\
  {							\
    return NEON_FCT ## _ ## NEON_SUF (a);		\
  }
  
  // wrapper for arbitary unary function for all types
#define SIMDVEC_NEON_UNARY_ALL(FCT,NEON_FCT)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDByte,NEON_FCT,u8)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDSignedByte,NEON_FCT,s8)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDWord,NEON_FCT,u16)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDShort,NEON_FCT,s16)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDInt,NEON_FCT,s32)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDFloat,NEON_FCT,f32)
  
  // wrapper for arbitary unary function for all types
#define SIMDVEC_NEON_UNARY_ALLSIGNED(FCT,NEON_FCT)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDSignedByte,NEON_FCT,s8)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDShort,NEON_FCT,s16)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDInt,NEON_FCT,s32)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDFloat,NEON_FCT,f32)

  // wrapper for arbitary unary function for all types
#define SIMDVEC_NEON_UNARY_ALLINT(FCT,NEON_FCT)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDByte,NEON_FCT,u8)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDSignedByte,NEON_FCT,s8)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDWord,NEON_FCT,u16)		\
  SIMDVEC_NEON_UNARY(FCT,SIMDShort,NEON_FCT,s16)	\
  SIMDVEC_NEON_UNARY(FCT,SIMDInt,NEON_FCT,s32)

  // -------------------------------------------------------------------------
  // template function with scalar input
  // -------------------------------------------------------------------------
  
#define SIMDVEC_NEON_TMPLT_SCALAR(FCT,TYPE,NEON_FCT,NEON_SUF)		\
  static SIMD_INLINE SIMDVec<TYPE,16>						\
  FCT(TYPE a, Integer<16>)								\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (a);				\
  }
  
#define SIMDVEC_NEON_TMPLT_SCALAR_ALL(FCT,NEON_FCT)			\
  SIMDVEC_NEON_TMPLT_SCALAR(FCT,SIMDByte,NEON_FCT,u8)			\
  SIMDVEC_NEON_TMPLT_SCALAR(FCT,SIMDSignedByte,NEON_FCT,s8)		\
  SIMDVEC_NEON_TMPLT_SCALAR(FCT,SIMDWord,NEON_FCT,u16)			\
  SIMDVEC_NEON_TMPLT_SCALAR(FCT,SIMDShort,NEON_FCT,s16)			\
  SIMDVEC_NEON_TMPLT_SCALAR(FCT,SIMDInt,NEON_FCT,s32)			\
  SIMDVEC_NEON_TMPLT_SCALAR(FCT,SIMDFloat,NEON_FCT,f32)

  // -------------------------------------------------------------------------
  // template function with pointer input
  // -------------------------------------------------------------------------
  
#define SIMDVEC_NEON_TMPLT_PTR(FCT,TYPE,NEON_FCT,NEON_SUF)		\
  static SIMD_INLINE SIMDVec<TYPE,16>						\
  FCT(const TYPE *const p, Integer<16>)						\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (p);				\
  }
  
#define SIMDVEC_NEON_TMPLT_PTR_ALL(FCT,NEON_FCT)			\
  SIMDVEC_NEON_TMPLT_PTR(FCT,SIMDByte,NEON_FCT,u8)			\
  SIMDVEC_NEON_TMPLT_PTR(FCT,SIMDSignedByte,NEON_FCT,s8)		\
  SIMDVEC_NEON_TMPLT_PTR(FCT,SIMDWord,NEON_FCT,u16)			\
  SIMDVEC_NEON_TMPLT_PTR(FCT,SIMDShort,NEON_FCT,s16)			\
  SIMDVEC_NEON_TMPLT_PTR(FCT,SIMDInt,NEON_FCT,s32)			\
  SIMDVEC_NEON_TMPLT_PTR(FCT,SIMDFloat,NEON_FCT,f32)
  
  // -------------------------------------------------------------------------
  // template function without input variable but immediate parameter
  // -------------------------------------------------------------------------
  
#define SIMDVEC_NEON_TMPLT_IMM(FCT,TYPE,NEON_FCT,NEON_SUF,NEON_ARG)	\
  static SIMD_INLINE SIMDVec<TYPE,16>						\
  FCT(OutputType<TYPE>, Integer<16>)									\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (TYPE(NEON_ARG));			\
  }
  
#define SIMDVEC_NEON_TMPLT_IMM_ALL(FCT,NEON_FCT,NEON_ARG)		\
  SIMDVEC_NEON_TMPLT_IMM(FCT,SIMDByte,NEON_FCT,u8,NEON_ARG)		\
  SIMDVEC_NEON_TMPLT_IMM(FCT,SIMDSignedByte,NEON_FCT,s8,NEON_ARG)	\
  SIMDVEC_NEON_TMPLT_IMM(FCT,SIMDWord,NEON_FCT,u16,NEON_ARG)		\
  SIMDVEC_NEON_TMPLT_IMM(FCT,SIMDShort,NEON_FCT,s16,NEON_ARG)		\
  SIMDVEC_NEON_TMPLT_IMM(FCT,SIMDInt,NEON_FCT,s32,NEON_ARG)		\
  SIMDVEC_NEON_TMPLT_IMM(FCT,SIMDFloat,NEON_FCT,f32,NEON_ARG)
  
  // -------------------------------------------------------------------------
  // function with pointer and vector input
  // -------------------------------------------------------------------------
  
#define SIMDVEC_NEON_PTR_VEC(FCT,TYPE,NEON_FCT,NEON_SUF)		\
  static SIMD_INLINE void						\
  FCT(TYPE *const p, const SIMDVec<TYPE,16> &a)				\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (p, a);				\
  }
  
#define SIMDVEC_NEON_PTR_VEC_ALL(FCT,NEON_FCT)			\
  SIMDVEC_NEON_PTR_VEC(FCT,SIMDByte,NEON_FCT,u8)		\
  SIMDVEC_NEON_PTR_VEC(FCT,SIMDSignedByte,NEON_FCT,s8)		\
  SIMDVEC_NEON_PTR_VEC(FCT,SIMDWord,NEON_FCT,u16)		\
  SIMDVEC_NEON_PTR_VEC(FCT,SIMDShort,NEON_FCT,s16)		\
  SIMDVEC_NEON_PTR_VEC(FCT,SIMDInt,NEON_FCT,s32)		\
  SIMDVEC_NEON_PTR_VEC(FCT,SIMDFloat,NEON_FCT,f32)
  
  // -------------------------------------------------------------------------
  // template function with scalar result and vector + immed. tmpl. argument
  // -------------------------------------------------------------------------
  
#define SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,TYPE,NEON_FCT,NEON_SUF,RANGE)	\
  template <int IMM>							\
  static SIMD_INLINE TYPE						\
  FCT(const SIMDVec<TYPE,16> &a, IsPosInRange<true, true>)		\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (a, IMM);				\
  }									\
  template <int IMM>							\
  static SIMD_INLINE TYPE						\
  FCT(const SIMDVec<TYPE,16> &, IsPosInRange<true, false>)		\
  {									\
    return TYPE(0);							\
  }									\
  template <int IMM>							\
  static SIMD_INLINE TYPE						\
  FCT(const SIMDVec<TYPE,16> &a)					\
  {									\
    return FCT<IMM>(a, IsPosInGivenRange<RANGE, IMM>());		\
  }
  
  // RANGE parameter relates to no. of elements
#define SIMDVEC_NEON_SCALAR_VEC_IMM_ALL(FCT,NEON_FCT)			\
  SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,SIMDByte,NEON_FCT,u8,16)		\
  SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,SIMDSignedByte,NEON_FCT,s8,16)	\
  SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,SIMDWord,NEON_FCT,u16,8)		\
  SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,SIMDShort,NEON_FCT,s16,8)		\
  SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,SIMDInt,NEON_FCT,s32,4)		\
  SIMDVEC_NEON_SCALAR_VEC_IMM(FCT,SIMDFloat,NEON_FCT,f32,4)
  
  // -------------------------------------------------------------------------
  // template function with vector result and vector + immediate argument
  // -------------------------------------------------------------------------

  // this is rather specific for shift instructions

  // it was necessary to introduce a special case IMM == 0, since this
  // is not allowed for the shift intrinsics (just returns the
  // argument); since the ARM docs aren't clear in this point, we also
  // treat the case IMM == no-of-bits as special case (in two
  // versions: one using FCT on RANGE-1, the other setting result to
  // zero)

#define SIMDVEC_NEON_VEC_VEC_IMM(FCT,TYPE,NEON_FCT,NEON_SUF,RANGE)	\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a, IsPosNonZeroInRange<true, true, true>)	\
  {									\
    return NEON_FCT ## _ ## NEON_SUF (a, IMM);				\
  }									\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a, IsPosNonZeroInRange<true, false, true>) \
  {									\
    return a;								\
  }									\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a)					\
  {									\
    return FCT<IMM>(a, IsPosNonZeroInGivenRange<RANGE, IMM>());		\
  }

  // out-of-range implemented with FCT of RANGE-1
#define SIMDVEC_NEON_VEC_VEC_IMM_OORFCT(FCT,TYPE,NEON_FCT,NEON_SUF,RANGE) \
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a, IsPosNonZeroInRange<true, true, false>) \
  {									\
    return NEON_FCT ## _ ## NEON_SUF (a, (RANGE - 1));			\
  }									\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,TYPE,NEON_FCT,NEON_SUF,RANGE)
  
  // out-of-range implemented with set-to-zero
#define SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(FCT,TYPE,NEON_FCT,NEON_SUF,RANGE) \
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &, IsPosNonZeroInRange<true, true, false>) \
  {									\
    return vmovq_n_ ## NEON_SUF (TYPE(0));				\
  }									\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,TYPE,NEON_FCT,NEON_SUF,RANGE)

  /*  
  // not used at the moment
  // TODO: two OOR versions would be required if this is used at some point
#define SIMDVEC_NEON_VEC_VEC_IMM_ALL(FCT,NEON_FCT)			\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,SIMDByte,NEON_FCT,u8,8)			\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,SIMDSignedByte,NEON_FCT,s8,8)		\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,SIMDWord,NEON_FCT,u16,16)		\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,SIMDShort,NEON_FCT,s16, 16)		\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,SIMDInt,NEON_FCT,s32,32)			\
  SIMDVEC_NEON_VEC_VEC_IMM(FCT,SIMDFloat,NEON_FCT,f32,32)
  */

#define SIMDVEC_NEON_VEC_VEC_IMM_REINTER(FCT,TYPE,NFCT,NSUF,NSUF2,RANGE) \
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a, IsPosNonZeroInRange<true, true, true>)	\
  {									\
    return vreinterpretq_ ## NSUF ## _ ## NSUF2				\
      (NFCT ## _ ## NSUF2						\
       (vreinterpretq_ ## NSUF2 ## _ ## NSUF (a), IMM));		\
  }									\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a, IsPosNonZeroInRange<true, false, true>) \
  {									\
    return a;								\
  }									\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a)					\
  {									\
    return FCT<IMM>(a, IsPosNonZeroInGivenRange<RANGE, IMM>());		\
  }

#define SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORFCT(FCT,TYPE,NFCT,NSUF,NSUF2,RANGE) \
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a, IsPosNonZeroInRange<true, true, false>) \
  {									\
    return vreinterpretq_ ## NSUF ## _ ## NSUF2				\
      (NFCT ## _ ## NSUF2						\
       (vreinterpretq_ ## NSUF2 ## _ ## NSUF (a), (RANGE-1)));		\
  }									\
  SIMDVEC_NEON_VEC_VEC_IMM_REINTER(FCT,TYPE,NFCT,NSUF,NSUF2,RANGE)


#define SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORZERO(FCT,TYPE,NFCT,NSUF,NSUF2,RANGE) \
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &, IsPosNonZeroInRange<true, true, false>) \
  {									\
    return vmovq_n_ ## NSUF (TYPE(0));					\
  }									\
  SIMDVEC_NEON_VEC_VEC_IMM_REINTER(FCT,TYPE,NFCT,NSUF,NSUF2,RANGE)
  
  // -------------------------------------------------------------------------
  // alignre
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_ALIGNRE_0(TYPE,NEON_SUF)				\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  alignre								\
  (const SIMDVec<TYPE,16> &,						\
   const SIMDVec<TYPE,16> &l,						\
   Range<true,true,0,SIMDVec<TYPE,16>::elements>)			\
  {									\
    return l;								\
  }
  
#define SIMDVEC_NEON_ALIGNRE_0_E(TYPE,NEON_SUF)				\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  alignre								\
  (const SIMDVec<TYPE,16> &h,						\
   const SIMDVec<TYPE,16> &l,						\
   Range<true,false,0,SIMDVec<TYPE,16>::elements>)			\
  {									\
    return vextq_ ## NEON_SUF (l, h, IMM);				\
  }
  
#define SIMDVEC_NEON_ALIGNRE_E(TYPE,NEON_SUF)				\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  alignre								\
  (const SIMDVec<TYPE,16> &h,						\
   const SIMDVec<TYPE,16> &,						\
   Range<true,true,SIMDVec<TYPE,16>::elements,2*SIMDVec<TYPE,16>::elements>) \
  {									\
    return h;								\
  }

#define SIMDVEC_NEON_ALIGNRE_E_2E(TYPE,NEON_SUF)			\
  template <int IMM>							\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  alignre								\
  (const SIMDVec<TYPE,16> &h,						\
   const SIMDVec<TYPE,16> &,						\
   Range<true,false,SIMDVec<TYPE,16>::elements,2*SIMDVec<TYPE,16>::elements>) \
  {									\
    const int imm = IMM - SIMDVec<TYPE,16>::elements;				\
    return vextq_ ## NEON_SUF (h, vmovq_n_ ## NEON_SUF (TYPE(0)),	imm);	\
  }
  
#define SIMDVEC_NEON_ALIGNRE_OTHER(TYPE,NEON_SUF)			\
  template <int IMM, bool AT_LL, int LL_INCL, int UL_EXCL>		\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  alignre								\
  (const SIMDVec<TYPE,16> &,						\
   const SIMDVec<TYPE,16> &,						\
   Range<true,AT_LL, LL_INCL, UL_EXCL>)					\
  {									\
    return vmovq_n_ ## NEON_SUF(TYPE(0));				\
  }

#define SIMDVEC_NEON_ALIGNRE_HUB()					\
  template <int IMM, typename T>					\
  static SIMD_INLINE SIMDVec<T,16>					\
  alignre								\
  (const SIMDVec<T,16> &h,						\
   const SIMDVec<T,16> &l)						\
  {									\
    return alignre<IMM>							\
      (h, l, SizeRange<IMM,SIMDVec<T,16>::elements>());			\
  }
  
#define SIMDVEC_NEON_ALIGNRE(TYPE,NEON_SUF)				\
  SIMDVEC_NEON_ALIGNRE_0(TYPE,NEON_SUF)					\
  SIMDVEC_NEON_ALIGNRE_0_E(TYPE,NEON_SUF)				\
  SIMDVEC_NEON_ALIGNRE_E(TYPE,NEON_SUF)					\
  SIMDVEC_NEON_ALIGNRE_E_2E(TYPE,NEON_SUF)				\
  SIMDVEC_NEON_ALIGNRE_OTHER(TYPE,NEON_SUF)

#define SIMDVEC_NEON_ALIGNRE_ALL()		\
  SIMDVEC_NEON_ALIGNRE(SIMDByte,u8)		\
  SIMDVEC_NEON_ALIGNRE(SIMDSignedByte,s8)	\
  SIMDVEC_NEON_ALIGNRE(SIMDWord,u16)		\
  SIMDVEC_NEON_ALIGNRE(SIMDShort,s16)		\
  SIMDVEC_NEON_ALIGNRE(SIMDInt,s32)		\
  SIMDVEC_NEON_ALIGNRE(SIMDFloat,f32)		\
  SIMDVEC_NEON_ALIGNRE_HUB()
  
  // -------------------------------------------------------------------------
  // unpack
  // -------------------------------------------------------------------------

  // TODO: unpack is inefficient here since vzipq does both unpacklo and
  // TODO: unpackhi but only half of the result is used

  // via cast to larger datatype
#define SIMDVEC_NEON_UNPACK(TYPE,PART,BYTES,NEON_SUF,NEON_SUF2)	\
  static SIMD_INLINE SIMDVec<TYPE, 16>					\
  unpack(const SIMDVec<TYPE, 16> &a,					\
	 const SIMDVec<TYPE, 16> &b,					\
	 Part<PART>,							\
	 Bytes<BYTES>)						\
  {									\
    return								\
      vreinterpretq_ ## NEON_SUF ## _ ## NEON_SUF2			\
      ((vzipq_ ## NEON_SUF2						\
	(vreinterpretq_ ## NEON_SUF2 ## _ ## NEON_SUF (a),		\
	 vreinterpretq_ ## NEON_SUF2 ## _ ## NEON_SUF (b))).val[PART]);	\
  }
  
#define SIMDVEC_NEON_UNPACK_PART01(TYPE,BYTES,NEON_SUF,NEON_SUF2)	\
  SIMDVEC_NEON_UNPACK(TYPE,0,BYTES,NEON_SUF,NEON_SUF2)		\
  SIMDVEC_NEON_UNPACK(TYPE,1,BYTES,NEON_SUF,NEON_SUF2)
  
  // via extraction of low or high halfs
  // (NOTE: PART and BYTES are needed in argument list)
#define SIMDVEC_NEON_UNPACK_HALFS(TYPE,PART,BYTES,LOHI,NEON_SUF)	\
  static SIMD_INLINE SIMDVec<TYPE, 16>					\
  unpack(const SIMDVec<TYPE, 16> &a,					\
	 const SIMDVec<TYPE, 16> &b,					\
	 Part<PART>,							\
	 Bytes<BYTES>)						\
  {									\
    return vcombine_ ## NEON_SUF (vget_ ## LOHI ## _ ## NEON_SUF (a),	\
				  vget_ ## LOHI ## _ ## NEON_SUF (b));	\
  }
  
#define SIMDVEC_NEON_UNPACK_HALFS_PART01(TYPE,BYTES,NEON_SUF)	\
  SIMDVEC_NEON_UNPACK_HALFS(TYPE,0,BYTES,low,NEON_SUF)		\
  SIMDVEC_NEON_UNPACK_HALFS(TYPE,1,BYTES,high,NEON_SUF)
  
#define SIMDVEC_NEON_UNPACK_BYTE()			\
  SIMDVEC_NEON_UNPACK_PART01(SIMDByte,1,u8,u8)		\
  SIMDVEC_NEON_UNPACK_PART01(SIMDByte,2,u8,u16)		\
  SIMDVEC_NEON_UNPACK_PART01(SIMDByte,4,u8,u32)		\
  SIMDVEC_NEON_UNPACK_HALFS_PART01(SIMDByte,8,u8)
  
#define SIMDVEC_NEON_UNPACK_SIGNEDBYTE()			\
  SIMDVEC_NEON_UNPACK_PART01(SIMDSignedByte,1,s8,s8)		\
  SIMDVEC_NEON_UNPACK_PART01(SIMDSignedByte,2,s8,s16)		\
  SIMDVEC_NEON_UNPACK_PART01(SIMDSignedByte,4,s8,s32)		\
  SIMDVEC_NEON_UNPACK_HALFS_PART01(SIMDSignedByte,8,s8)
  
#define SIMDVEC_NEON_UNPACK_WORD()				\
  SIMDVEC_NEON_UNPACK_PART01(SIMDWord,2,u16,u16)		\
  SIMDVEC_NEON_UNPACK_PART01(SIMDWord,4,u16,u32)		\
  SIMDVEC_NEON_UNPACK_HALFS_PART01(SIMDWord,8,u16)
  
#define SIMDVEC_NEON_UNPACK_SHORT()				\
  SIMDVEC_NEON_UNPACK_PART01(SIMDShort,2,s16,s16)		\
  SIMDVEC_NEON_UNPACK_PART01(SIMDShort,4,s16,s32)		\
  SIMDVEC_NEON_UNPACK_HALFS_PART01(SIMDShort,8,s16)
  
#define SIMDVEC_NEON_UNPACK_INT()			\
  SIMDVEC_NEON_UNPACK_PART01(SIMDInt,4,s32,s32)		\
  SIMDVEC_NEON_UNPACK_HALFS_PART01(SIMDInt,8,s32)
  
#define SIMDVEC_NEON_UNPACK_FLOAT()			\
  SIMDVEC_NEON_UNPACK_PART01(SIMDFloat,4,f32,f32)	\
  SIMDVEC_NEON_UNPACK_HALFS_PART01(SIMDFloat,8,f32)

#define SIMDVEC_NEON_UNPACK_ALL()		\
  SIMDVEC_NEON_UNPACK_BYTE()			\
  SIMDVEC_NEON_UNPACK_SIGNEDBYTE()		\
  SIMDVEC_NEON_UNPACK_WORD()			\
  SIMDVEC_NEON_UNPACK_SHORT()			\
  SIMDVEC_NEON_UNPACK_INT()			\
  SIMDVEC_NEON_UNPACK_FLOAT()

  // -------------------------------------------------------------------------
  // zip
  // -------------------------------------------------------------------------

  // a, b passed by-value to avoid problems with identical input/output args.

  // via cast to larger datatype
#define SIMDVEC_NEON_ZIP(TYPE,NUM_ELEMS,NEON_SUF,NEON_SUF2,NEONX2_2)	\
  static SIMD_INLINE void						\
  zip(const SIMDVec<TYPE, 16> a,					\
      const SIMDVec<TYPE, 16> b,					\
      SIMDVec<TYPE, 16> &c,						\
      SIMDVec<TYPE, 16> &d,						\
      Elements<NUM_ELEMS>)						\
  {									\
    NEONX2_2 res;							\
    res = vzipq_ ## NEON_SUF2						\
      (vreinterpretq_ ## NEON_SUF2 ## _ ## NEON_SUF (a),		\
       vreinterpretq_ ## NEON_SUF2 ## _ ## NEON_SUF (b));		\
    c = vreinterpretq_ ## NEON_SUF ## _ ## NEON_SUF2 (res.val[0]);	\
    d = vreinterpretq_ ## NEON_SUF ## _ ## NEON_SUF2 (res.val[1]);	\
  }
    
  // via extraction of low or high halfs
  // (NOTE: NUM_ELEMS is needed in argument list)
#define SIMDVEC_NEON_ZIP_HALFS(TYPE,NUM_ELEMS,NEON_SUF)			\
  static SIMD_INLINE void						\
  zip(const SIMDVec<TYPE, 16> a,					\
      const SIMDVec<TYPE, 16> b,					\
      SIMDVec<TYPE, 16> &c,						\
      SIMDVec<TYPE, 16> &d,						\
      Elements<NUM_ELEMS>)						\
  {									\
    c = vcombine_ ## NEON_SUF (vget_low_ ## NEON_SUF (a),		\
			       vget_low_ ## NEON_SUF (b));		\
    d = vcombine_ ## NEON_SUF (vget_high_ ## NEON_SUF (a),		\
			       vget_high_ ## NEON_SUF (b));		\
  }
    
  // hub template function
#define SIMDVEC_NEON_ZIP_HUB()						\
  template <int NUM_ELEMS, typename T>					\
  static SIMD_INLINE void						\
  zip(const SIMDVec<T, 16> a,						\
      const SIMDVec<T, 16> b,						\
      SIMDVec<T, 16> &c,						\
      SIMDVec<T, 16> &d)						\
  {									\
    return zip(a, b, c, d, Elements<NUM_ELEMS>());			\
  }
  
#define SIMDVEC_NEON_ZIP_BYTE()						\
  SIMDVEC_NEON_ZIP(SIMDByte,1,u8,u8,uint8x16x2_t)			\
  SIMDVEC_NEON_ZIP(SIMDByte,2,u8,u16,uint16x8x2_t)			\
  SIMDVEC_NEON_ZIP(SIMDByte,4,u8,u32,uint32x4x2_t)			\
  SIMDVEC_NEON_ZIP_HALFS(SIMDByte,8,u8)
  
#define SIMDVEC_NEON_ZIP_SIGNEDBYTE()					\
  SIMDVEC_NEON_ZIP(SIMDSignedByte,1,s8,s8,int8x16x2_t)			\
  SIMDVEC_NEON_ZIP(SIMDSignedByte,2,s8,s16,int16x8x2_t)			\
  SIMDVEC_NEON_ZIP(SIMDSignedByte,4,s8,s32,int32x4x2_t)			\
  SIMDVEC_NEON_ZIP_HALFS(SIMDSignedByte,8,s8)
  
#define SIMDVEC_NEON_ZIP_WORD()						\
  SIMDVEC_NEON_ZIP(SIMDWord,1,u16,u16,uint16x8x2_t)			\
  SIMDVEC_NEON_ZIP(SIMDWord,2,u16,u32,uint32x4x2_t)			\
  SIMDVEC_NEON_ZIP_HALFS(SIMDWord,4,u16)
  
#define SIMDVEC_NEON_ZIP_SHORT()					\
  SIMDVEC_NEON_ZIP(SIMDShort,1,s16,s16,int16x8x2_t)			\
  SIMDVEC_NEON_ZIP(SIMDShort,2,s16,s32,int32x4x2_t)			\
  SIMDVEC_NEON_ZIP_HALFS(SIMDShort,4,s16)
  
#define SIMDVEC_NEON_ZIP_INT()					\
  SIMDVEC_NEON_ZIP(SIMDInt,1,s32,s32,int32x4x2_t)		\
  SIMDVEC_NEON_ZIP_HALFS(SIMDInt,2,s32)
  
#define SIMDVEC_NEON_ZIP_FLOAT()				\
  SIMDVEC_NEON_ZIP(SIMDFloat,1,f32,f32,float32x4x2_t)		\
  SIMDVEC_NEON_ZIP_HALFS(SIMDFloat,2,f32)
  
#define SIMDVEC_NEON_ZIP_ALL()			\
  SIMDVEC_NEON_ZIP_BYTE()			\
  SIMDVEC_NEON_ZIP_SIGNEDBYTE()			\
  SIMDVEC_NEON_ZIP_WORD()			\
  SIMDVEC_NEON_ZIP_SHORT()			\
  SIMDVEC_NEON_ZIP_INT()			\
  SIMDVEC_NEON_ZIP_FLOAT()			\
  SIMDVEC_NEON_ZIP_HUB()
  
  // -------------------------------------------------------------------------
  // unzip
  // -------------------------------------------------------------------------

  // a, b passed by-value to avoid problems with identical input/output args.

  // via cast to larger datatype
#define SIMDVEC_NEON_UNZIP(TYPE,BYTES,NEON_SUF,NEON_SUF2,NEONX2_2)	\
  static SIMD_INLINE void						\
  unzip(const SIMDVec<TYPE, 16> a,					\
	const SIMDVec<TYPE, 16> b,					\
	SIMDVec<TYPE, 16> &c,						\
	SIMDVec<TYPE, 16> &d,						\
	Bytes<BYTES>)						\
  {									\
    NEONX2_2 res;							\
    res = vuzpq_ ## NEON_SUF2						\
      (vreinterpretq_ ## NEON_SUF2 ## _ ## NEON_SUF (a),		\
       vreinterpretq_ ## NEON_SUF2 ## _ ## NEON_SUF (b));		\
    c = vreinterpretq_ ## NEON_SUF ## _ ## NEON_SUF2 (res.val[0]);	\
    d = vreinterpretq_ ## NEON_SUF ## _ ## NEON_SUF2 (res.val[1]);	\
  }
    
  // via extraction of low or high halfs
  // (NOTE: BYTES is needed in argument list)
#define SIMDVEC_NEON_UNZIP_HALFS(TYPE,BYTES,NEON_SUF)		\
  static SIMD_INLINE void						\
  unzip(const SIMDVec<TYPE, 16> a,					\
	const SIMDVec<TYPE, 16> b,					\
	SIMDVec<TYPE, 16> &c,						\
	SIMDVec<TYPE, 16> &d,						\
	Bytes<BYTES>)						\
  {									\
    c = vcombine_ ## NEON_SUF (vget_low_ ## NEON_SUF (a),		\
			       vget_low_ ## NEON_SUF (b));		\
    d = vcombine_ ## NEON_SUF (vget_high_ ## NEON_SUF (a),		\
			       vget_high_ ## NEON_SUF (b));		\
  }
  
#define SIMDVEC_NEON_UNZIP_BYTE()					\
  SIMDVEC_NEON_UNZIP(SIMDByte,1,u8,u8,uint8x16x2_t)			\
  SIMDVEC_NEON_UNZIP(SIMDByte,2,u8,u16,uint16x8x2_t)			\
  SIMDVEC_NEON_UNZIP(SIMDByte,4,u8,u32,uint32x4x2_t)			\
  SIMDVEC_NEON_UNZIP_HALFS(SIMDByte,8,u8)
  
#define SIMDVEC_NEON_UNZIP_SIGNEDBYTE()					\
  SIMDVEC_NEON_UNZIP(SIMDSignedByte,1,s8,s8,int8x16x2_t)		\
  SIMDVEC_NEON_UNZIP(SIMDSignedByte,2,s8,s16,int16x8x2_t)		\
  SIMDVEC_NEON_UNZIP(SIMDSignedByte,4,s8,s32,int32x4x2_t)		\
  SIMDVEC_NEON_UNZIP_HALFS(SIMDSignedByte,8,s8)
  
#define SIMDVEC_NEON_UNZIP_WORD()					\
  SIMDVEC_NEON_UNZIP(SIMDWord,2,u16,u16,uint16x8x2_t)			\
  SIMDVEC_NEON_UNZIP(SIMDWord,4,u16,u32,uint32x4x2_t)			\
  SIMDVEC_NEON_UNZIP_HALFS(SIMDWord,8,u16)
  
#define SIMDVEC_NEON_UNZIP_SHORT()					\
  SIMDVEC_NEON_UNZIP(SIMDShort,2,s16,s16,int16x8x2_t)			\
  SIMDVEC_NEON_UNZIP(SIMDShort,4,s16,s32,int32x4x2_t)			\
  SIMDVEC_NEON_UNZIP_HALFS(SIMDShort,8,s16)
  
#define SIMDVEC_NEON_UNZIP_INT()				\
  SIMDVEC_NEON_UNZIP(SIMDInt,4,s32,s32,int32x4x2_t)		\
  SIMDVEC_NEON_UNZIP_HALFS(SIMDInt,8,s32)
  
#define SIMDVEC_NEON_UNZIP_FLOAT()				\
  SIMDVEC_NEON_UNZIP(SIMDFloat,4,f32,f32,float32x4x2_t)		\
  SIMDVEC_NEON_UNZIP_HALFS(SIMDFloat,8,f32)
  
#define SIMDVEC_NEON_UNZIP_ALL()			\
  SIMDVEC_NEON_UNZIP_BYTE()				\
  SIMDVEC_NEON_UNZIP_SIGNEDBYTE()			\
  SIMDVEC_NEON_UNZIP_WORD()				\
  SIMDVEC_NEON_UNZIP_SHORT()				\
  SIMDVEC_NEON_UNZIP_INT()				\
  SIMDVEC_NEON_UNZIP_FLOAT()
  
  // #########################################################################
  // #########################################################################
  // #########################################################################
  
  // =========================================================================
  // SIMDVec function instantiations or overloading for NEON
  // =========================================================================

  // -------------------------------------------------------------------------
  // reinterpretation casts
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_REINTERP_ALL()

  // -------------------------------------------------------------------------
  // convert (without changes in the number of of elements)
  // -------------------------------------------------------------------------

  // conversion seems to be saturated in all cases (specified by the
  // rounding mode):
  // http://stackoverflow.com/questions/24546927/
  //  behavior-of-arm-neon-float-integer-conversion-with-overflow

  // saturated
  // TODO: rounding in cvts (float->int)? +0.5?
  // TODO: (NOT the same behavior as in SIMDVecBaseImplIntel16.H
  // TODO:  float->int always uses round towards zero = trunc?)
  // TODO: cvts: should we saturate in the same way as for Intel?
  // TODO: (Intel saturates to max. float which is convertible to int,
  // TODO:  NEON saturates to 0x7fffffff)
  static SIMD_INLINE SIMDVec<SIMDInt,16> 
  cvts(const SIMDVec<SIMDFloat,16> &a, OutputType<SIMDInt>)
  {
    return vcvtq_s32_f32(a);
  }

  // saturation is not necessary in this case
  static SIMD_INLINE SIMDVec<SIMDFloat,16> 
  cvts(const SIMDVec<SIMDInt,16> &a, OutputType<SIMDFloat>)
  {
    return vcvtq_f32_s32(a);
  }

  // -------------------------------------------------------------------------
  // setzero
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_TMPLT_IMM_ALL(setzero,vmovq_n,0)
  
  // -------------------------------------------------------------------------
  // set1
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_TMPLT_SCALAR_ALL(set1,vdupq_n)

  // -------------------------------------------------------------------------
  // load
  // -------------------------------------------------------------------------
  
  SIMDVEC_NEON_TMPLT_PTR_ALL(load,vld1q)

  // -------------------------------------------------------------------------
  // loadu
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_TMPLT_PTR_ALL(loadu,vld1q)

  // -------------------------------------------------------------------------
  // store
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_PTR_VEC_ALL(store,vst1q)

  // -------------------------------------------------------------------------
  // storeu
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_PTR_VEC_ALL(storeu,vst1q)

  // -------------------------------------------------------------------------
  // stream_store
  // -------------------------------------------------------------------------

  // TODO: is there anything like _mm_stream_* for NEON?
  SIMDVEC_NEON_PTR_VEC_ALL(stream_store,vst1q)

  // -------------------------------------------------------------------------
  // fences
  // -------------------------------------------------------------------------

  // http://infocenter.arm.com/help/
  //   index.jsp?topic=/com.arm.doc.faqs/ka14552.html
  // TODO: is this portable to clang?

  // NOTE: implemented as full barrier
  static SIMD_INLINE void
  lfence()
  {
    SIMD_FULL_MEMBARRIER;
  }

  // NOTE: implemented as full barrier
  static SIMD_INLINE void
  sfence()
  {
    SIMD_FULL_MEMBARRIER;
  }

  // NOTE: implemented as full barrier
  static SIMD_INLINE void
  mfence()
  {
    SIMD_FULL_MEMBARRIER;
  }

  // -------------------------------------------------------------------------
  // extract: with template parameter for immediate argument
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_SCALAR_VEC_IMM_ALL(extract,vgetq_lane)

  // -------------------------------------------------------------------------
  // add
  // -------------------------------------------------------------------------
  
  SIMDVEC_NEON_BINARY_ALL(add,vaddq)

  // -------------------------------------------------------------------------
  // adds
  // -------------------------------------------------------------------------

#ifndef SIMD_STRICT_SATURATION
  // float NOT saturated
  SIMDVEC_NEON_BINARY_REPLACE(SIMDFloat,vqaddq,vaddq,f32)
  SIMDVEC_NEON_BINARY_ALL(adds,vqaddq)
#else
  // without float
  SIMDVEC_NEON_BINARY_ALLINT(adds,vqaddq)
#endif

  // -------------------------------------------------------------------------
  // sub
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_ALL(sub,vsubq)

  // -------------------------------------------------------------------------
  // subs
  // -------------------------------------------------------------------------

#ifndef SIMD_STRICT_SATURATION
  // float NOT saturated
  SIMDVEC_NEON_BINARY_REPLACE(SIMDFloat,vqsubq,vsubq,f32)
  SIMDVEC_NEON_BINARY_ALL(subs,vqsubq)
#else
  // without float
  SIMDVEC_NEON_BINARY_ALLINT(subs,vqsubq)
#endif

  // -------------------------------------------------------------------------
  // neg (negate = two's complement or unary minus), only signed types
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_UNARY_ALLSIGNED(neg,vnegq)  

  // -------------------------------------------------------------------------
  // min
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_ALL(min,vminq)

  // -------------------------------------------------------------------------
  // max
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_ALL(max,vmaxq)

  // -------------------------------------------------------------------------
  // mul, div
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY(mul,SIMDFloat,vmulq,f32)

#define SIMDVEC_NEON_DIV_REFINE_STEPS 2

  // adapted from Jens Froemmer's Ba thesis (2014)
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  div(const SIMDVec<SIMDFloat,16> &num,
      const SIMDVec<SIMDFloat,16> &denom)
  {
    // get estimate of reciprocal of denom
    float32x4_t reciprocal = vrecpeq_f32(denom);
    // refince estimate using Newton-Raphson steps
    for (int i = 0; i < SIMDVEC_NEON_DIV_REFINE_STEPS; i++) 
      reciprocal = vmulq_f32(vrecpsq_f32(denom, reciprocal), reciprocal);
    // num * (1.0 / denom)
    return vmulq_f32(num, reciprocal);
  }

  // -------------------------------------------------------------------------
  // ceil, floor, round, truncate
  // -------------------------------------------------------------------------

  // http://www.rowleydownload.co.uk/arm/documentation/gnu/gcc/
  //   ARM-NEON-Intrinsics.html
  // vrnd, only some architectures, see arm_neon.h

#if __ARM_ARCH >= 8

  // 10. Apr 19 (rm): BINARY->UNARY, qp -> pq etc., still not tested
  SIMDVEC_NEON_UNARY(ceil,SIMDFloat,vrndpq,f32)
  SIMDVEC_NEON_UNARY(floor,SIMDFloat,vrndmq,f32)
  SIMDVEC_NEON_UNARY(round,SIMDFloat,vrndnq,f32)
  SIMDVEC_NEON_UNARY(truncate,SIMDFloat,vrndq,f32)

#else
  
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  truncate(const SIMDVec<SIMDFloat,16> &a)
  {
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    float32x4_t limit = vmovq_n_f32(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    uint32x4_t noRndReq = vcgeq_f32(vabsq_f32(a), limit);
    // truncated result (for |a| < limit)
    float32x4_t aTrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
    // select result
    return vbslq_f32(noRndReq, a, aTrunc);
  }

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
  floor(const SIMDVec<SIMDFloat,16> &a)
  {
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    float32x4_t limit = vmovq_n_f32(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    uint32x4_t noRndReq = vcgeq_f32(vabsq_f32(a), limit);
    // bool mask: true if a is negative
    uint32x4_t isNeg = 
      vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(a), 31));
    // truncated result (for |a| < limit)
    float32x4_t aTrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
    // check if a is an integer
    uint32x4_t isNotInt = vmvnq_u32(vceqq_f32(a, aTrunc));
    // constant 1.0
    float32x4_t one = vmovq_n_f32(1.0f);
    // mask which is 1.0f for negative non-integer values, 0.0f otherwise
    float32x4_t oneMask =
      vreinterpretq_f32_u32(vandq_u32(vandq_u32(isNeg, isNotInt), 
				     vreinterpretq_u32_f32(one)));
    // if negative, trunc computes ceil, to turn it into floor we sub
    // 1 if aTrunc is non-integer
    aTrunc = vsubq_f32(aTrunc, oneMask);
    // select result (a or aTrunc)
    return vbslq_f32(noRndReq, a, aTrunc);
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  ceil(const SIMDVec<SIMDFloat,16> &a)
  {
    // if e>=23, floating point number represents an integer, 2^23 = 8388608
    float32x4_t limit = vmovq_n_f32(8388608.f);
    // bool mask: no rounding required if abs(a) >= limit
    uint32x4_t noRndReq = vcgeq_f32(vabsq_f32(a), limit);
    // bool mask: true if a is negative
    uint32x4_t isNotNeg = 
      vmvnq_u32(vreinterpretq_u32_s32
		(vshrq_n_s32(vreinterpretq_s32_f32(a), 31)));
    // truncated result (for |a| < limit)
    float32x4_t aTrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
    // check if a is an integer
    uint32x4_t isNotInt = vmvnq_u32(vceqq_f32(a, aTrunc));
    // constant 1.0
    float32x4_t one = vmovq_n_f32(1.0f);
    // mask which is 1.0f for non-negative non-integer values, 0.0f otherwise
    float32x4_t oneMask =
      vreinterpretq_f32_u32(vandq_u32(vandq_u32(isNotNeg, isNotInt), 
				     vreinterpretq_u32_f32(one)));
    // if non-negative, trunc computes floor, to turn it into ceil we
    // add 1 if aTrunc is non-integer
    aTrunc = vaddq_f32(aTrunc, oneMask);
    // select result (a or aTrunc)
    return vbslq_f32(noRndReq, a, aTrunc);
  }

  // NOTE: rounds ties (*.5) towards infinity, different from Intel
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  round(const SIMDVec<SIMDFloat,16> &a)
  {
    return floor(add(a, set1(SIMDFloat(0.5f), Integer<16>())));
  }

#endif

  // -------------------------------------------------------------------------
  // elementary mathematical functions
  // -------------------------------------------------------------------------

  // estimate of a reciprocal
  SIMDVEC_NEON_UNARY(rcp,SIMDFloat,vrecpeq,f32)

  // estimate of a reverse square root
  SIMDVEC_NEON_UNARY(rsqrt,SIMDFloat,vrsqrteq,f32)

#define SIMDVEC_NEON_SQRT_REFINE_STEPS 2
  
  // square root (may not be very efficient)
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  sqrt(const SIMDVec<SIMDFloat,16> &a)
  {
    // vector with 0s, vector with 1s
    float32x4_t zero = vmovq_n_f32(0.0f), one = vmovq_n_f32(1.0f);
    // check for 0 to avoid div-by-0 (should also cover -0.0f)
    uint32x4_t isZero = vceqq_f32(a, zero);
    // avoid inf in rev. sqrt, replace 0 by 1
    float32x4_t as = vbslq_f32(isZero, one, a);
    // get estimate of reciprocal sqrt
    float32x4_t rSqrt = vrsqrteq_f32(as);
    // refine estimate using Newton-Raphson steps
    for (int i = 0; i < SIMDVEC_NEON_SQRT_REFINE_STEPS; i++)
      rSqrt = vmulq_f32(vrsqrtsq_f32(as, vmulq_f32(rSqrt, rSqrt)), rSqrt);
    // sqrt(a) = a * (1.0 / sqrt(a))
    float32x4_t res = vmulq_f32(as, rSqrt);
    // select result
    return vbslq_f32(isZero, zero, res);
  }

  // -------------------------------------------------------------------------
  // abs (integer: signed only)
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_UNARY_ALLSIGNED(abs,vabsq)  

  // -------------------------------------------------------------------------
  // unpack
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_UNPACK_ALL()

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

  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T, 16>
  extractLane(const SIMDVec<T, 16> &a)
  {
    return a;
  }

  // -------------------------------------------------------------------------
  // zip
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_ZIP_ALL()

  // ---------------------------------------------------------------------------
  // zip16 hub  (16-byte-lane oriented zip): for 16 bytes same as zip
  // ---------------------------------------------------------------------------

  // a, b are passed by-value to avoid problems with identical input/output args.

  template <int NUM_ELEMS, typename T>
  static SIMD_INLINE void
  zip16(const SIMDVec<T, 16> a,
        const SIMDVec<T, 16> b,
        SIMDVec<T, 16> &l,
        SIMDVec<T, 16> &h)
  {
    zip<NUM_ELEMS, T>(a, b, l, h);
  }

  // -------------------------------------------------------------------------
  // unzip
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_UNZIP_ALL()

  // ---------------------------------------------------------------------------
  // packs
  // ---------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16> 
  packs(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b,
  OutputType<SIMDSignedByte>)
  {
    return vcombine_s8(vqmovn_s16(a), vqmovn_s16(b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16> 
  packs(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b,
  OutputType<SIMDShort>)
  {
    return vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  packs(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b,
  OutputType<SIMDShort>)
  {
    return packs(cvts(a, OutputType<SIMDInt>()), cvts(b, OutputType<SIMDInt>()), OutputType<SIMDShort>());
  }

  // signed -> unsigned

  static SIMD_INLINE SIMDVec<SIMDByte,16> 
  packs(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b,
  OutputType<SIMDByte>)
  {
    return  vcombine_u8(vqmovun_s16(a), vqmovun_s16(b));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16> 
  packs(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b,
  OutputType<SIMDWord>)
  {
    return vcombine_u16(vqmovun_s32(a), vqmovun_s32(b));
  }   

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  packs(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b,
  OutputType<SIMDWord>)
  {
    return packs(cvts(a, OutputType<SIMDInt>()), cvts(b, OutputType<SIMDInt>()), OutputType<SIMDWord>());
  }

  // -------------------------------------------------------------------------
  // generalized extend: no stage
  // -------------------------------------------------------------------------

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

  // -------------------------------------------------------------------------
  // generalized extend: single stage
  // -------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,16> &vIn,
	 SIMDVec<SIMDShort,16> *const vOut)
  {
    vOut[0] = vmovl_s8(vget_low_s8(vIn));
    vOut[1] = vmovl_s8(vget_high_s8(vIn));
   }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDShort,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
    vOut[0] = vmovl_s16(vget_low_s16(vIn));
    vOut[1] = vmovl_s16(vget_high_s16(vIn));
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDShort,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
    vOut[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vIn)));
    vOut[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vIn)));
  }

  // unsigned -> unsigned

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDWord,16> *const vOut)
  {
    vOut[0] = vmovl_u8(vget_low_u8(vIn));
    vOut[1] = vmovl_u8(vget_high_u8(vIn));
  }

  // unsigned -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDShort,16> *const vOut)
  {
    vOut[0] = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vIn)));
    vOut[1] = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vIn)));
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDWord,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
    vOut[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vIn)));
    vOut[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vIn)));
  }

  static SIMD_INLINE void 
  extend(const SIMDVec<SIMDWord,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
    vOut[0] = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vIn)));
    vOut[1] = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vIn)));
  }

  // -------------------------------------------------------------------------
  // generalized extend: two stages
  // -------------------------------------------------------------------------

  // signed -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
    SIMDVec<SIMDShort,16> vShort[2];
    extend(vIn, vShort);
    extend(vShort[0], vOut);
    extend(vShort[1], vOut + 2);
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDSignedByte,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
    SIMDVec<SIMDShort,16> vShort[2];
    extend(vIn, vShort);
    extend(vShort[0], vOut);
    extend(vShort[1], vOut + 2);
  }

  // unsigned -> signed

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDInt,16> *const vOut)
  {
    SIMDVec<SIMDShort,16> vShort[2];
    extend(vIn, vShort);
    extend(vShort[0], vOut);
    extend(vShort[1], vOut + 2);
  }

  static SIMD_INLINE void
  extend(const SIMDVec<SIMDByte,16> &vIn,
	 SIMDVec<SIMDFloat,16> *const vOut)
  {
    SIMDVec<SIMDShort,16> vShort[2];
    extend(vIn, vShort);
    extend(vShort[0], vOut);
    extend(vShort[1], vOut + 2);
  }

  // -------------------------------------------------------------------------
  // generalized extend: special case int <-> float
  // -------------------------------------------------------------------------

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

  // -------------------------------------------------------------------------
  // srai
  // -------------------------------------------------------------------------

  // requires cast of unsigned types to signed!
  // http://stackoverflow.com/questions/18784988/
  //   neon-intrinsic-for-arithmetic-shift
  // out-of-range case handled with FCT=srai

  // 13. Nov 22 (Jonas Keller):
  // added missing Byte and SignedByte versions of srai

  SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORFCT(srai,SIMDByte,vshrq_n,u8,s8,8)
  SIMDVEC_NEON_VEC_VEC_IMM_OORFCT(srai,SIMDSignedByte,vshrq_n,s8,8)
  SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORFCT(srai,SIMDWord,vshrq_n,u16,s16,16)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORFCT(srai,SIMDShort,vshrq_n,s16,16)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORFCT(srai,SIMDInt,vshrq_n,s32,32)  

  // -------------------------------------------------------------------------
  // srli
  // -------------------------------------------------------------------------

  // requires cast of signed types to unsigned!
  // http://stackoverflow.com/questions/18784988/
  //   neon-intrinsic-for-arithmetic-shift
  // out-of-range case handled with set-to-zero

  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(srli,SIMDByte,vshrq_n,u8,8)  
  SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORZERO(srli,SIMDSignedByte,vshrq_n,s8,u8,8)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(srli,SIMDWord,vshrq_n,u16,16)  
  SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORZERO(srli,SIMDShort,vshrq_n,s16,u16,16)  
  SIMDVEC_NEON_VEC_VEC_IMM_REINTER_OORZERO(srli,SIMDInt,vshrq_n,s32,u32,32)  

  // -------------------------------------------------------------------------
  // slli
  // -------------------------------------------------------------------------

  // out-of-range case handled with set-to-zero
  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(slli,SIMDByte,vshlq_n,u8,8)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(slli,SIMDSignedByte,vshlq_n,s8,8)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(slli,SIMDWord,vshlq_n,u16,16)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(slli,SIMDShort,vshlq_n,s16,16)  
  SIMDVEC_NEON_VEC_VEC_IMM_OORZERO(slli,SIMDInt,vshlq_n,s32,32)  


  // 19. Dec 22 (Jonas Keller): added sra, srl and sll functions

  // -------------------------------------------------------------------------
  // sra
  // -------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  sra(const SIMDVec<SIMDByte,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vreinterpretq_u8_s8(
        vshrq_n_s8(vreinterpretq_s8_u8(a), std::min(count, uint8_t(8))));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  sra(const SIMDVec<SIMDSignedByte,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshrq_n_s8(a, std::min(count, uint8_t(8)));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  sra(const SIMDVec<SIMDWord,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vreinterpretq_u16_s16(
        vshrq_n_s16(vreinterpretq_s16_u16(a), std::min(count, uint8_t(16))));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  sra(const SIMDVec<SIMDShort,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshrq_n_s16(a, std::min(count, uint8_t(16)));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  sra(const SIMDVec<SIMDInt,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshrq_n_s32(a, std::min(count, uint8_t(32)));
  }

  // -------------------------------------------------------------------------
  // srl
  // -------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  srl(const SIMDVec<SIMDByte,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshrq_n_u8(a, std::min(count, uint8_t(8)));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  srl(const SIMDVec<SIMDSignedByte,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vreinterpretq_s8_u8(
        vshrq_n_u8(vreinterpretq_u8_s8(a), std::min(count, uint8_t(8))));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  srl(const SIMDVec<SIMDWord,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshrq_n_u16(a, std::min(count, uint8_t(16)));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  srl(const SIMDVec<SIMDShort,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vreinterpretq_s16_u16(
        vshrq_n_u16(vreinterpretq_u16_s16(a), std::min(count, uint8_t(16))));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  srl(const SIMDVec<SIMDInt,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vreinterpretq_s32_u32(
        vshrq_n_u32(vreinterpretq_u32_s32(a), std::min(count, uint8_t(32))));
  }

  // -------------------------------------------------------------------------
  // sll
  // -------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  sll(const SIMDVec<SIMDByte,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshlq_n_u8(a, std::min(count, uint8_t(8)));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  sll(const SIMDVec<SIMDSignedByte,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshlq_n_s8(a, std::min(count, uint8_t(8)));
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  sll(const SIMDVec<SIMDWord,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshlq_n_u16(a, std::min(count, uint8_t(16)));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  sll(const SIMDVec<SIMDShort,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshlq_n_s16(a, std::min(count, uint8_t(16)));
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  sll(const SIMDVec<SIMDInt,16> &a, const uint8_t count)
  {
    if (count == 0) {
      // is this necessary? what does vshrq_n do for count==0?
      return a;
    }
    return vshlq_n_s32(a, std::min(count, uint8_t(32)));
  }

  // 26. Sep 22 (Jonas Keller):
  // added SIMDByte and SIMDSignedByte versions of hadd, hadds, hsub and hsubs
  // added SIMDWord version of hadds and hsubs

  // -------------------------------------------------------------------------
  // hadd
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_HADD(TYPE,NEON_SUF)				\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  hadd(const SIMDVec<TYPE,16> &a,					\
       const SIMDVec<TYPE,16> &b)					\
  {									\
    return vcombine_ ## NEON_SUF					\
      (vpadd_ ## NEON_SUF (vget_low_ ## NEON_SUF (a),			\
			   vget_high_ ## NEON_SUF (a)),			\
       vpadd_ ## NEON_SUF (vget_low_ ## NEON_SUF (b),			\
			   vget_high_ ## NEON_SUF (b)));		\
  }
  
  SIMDVEC_NEON_HADD(SIMDByte, u8)
  SIMDVEC_NEON_HADD(SIMDSignedByte, s8)
  SIMDVEC_NEON_HADD(SIMDWord,u16)
  SIMDVEC_NEON_HADD(SIMDShort,s16)
  SIMDVEC_NEON_HADD(SIMDInt,s32)
  SIMDVEC_NEON_HADD(SIMDFloat,f32)

  // -------------------------------------------------------------------------
  // hadds
  // -------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  hadds(const SIMDVec<SIMDByte,16> &a,
       const SIMDVec<SIMDByte,16> &b)
  {
    SIMDVec<SIMDByte, 16> x, y;
    unzip(a, b, x, y, Bytes<1>());
    return adds(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  hadds(const SIMDVec<SIMDSignedByte,16> &a,
       const SIMDVec<SIMDSignedByte,16> &b)
  {
    SIMDVec<SIMDSignedByte, 16> x, y;
    unzip(a, b, x, y, Bytes<1>());
    return adds(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  hadds(const SIMDVec<SIMDWord,16> &a,
        const SIMDVec<SIMDWord,16> &b)
  {
    SIMDVec<SIMDWord, 16> x, y;
    unzip(a, b, x, y, Bytes<2>());
    return adds(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  hadds(const SIMDVec<SIMDShort,16> &a,
	const SIMDVec<SIMDShort,16> &b)
  {
    return vcombine_s16(vqmovn_s32(vpaddlq_s16(a)),
			vqmovn_s32(vpaddlq_s16(b)));
  }

  // in contrast to SIMDVecIntel*, this IS SATURATED!
  static SIMD_INLINE SIMDVec<SIMDInt,16>
  hadds(const SIMDVec<SIMDInt,16> &a,
	const SIMDVec<SIMDInt,16> &b)
  {
    return vcombine_s32(vqmovn_s64(vpaddlq_s32(a)),
			vqmovn_s64(vpaddlq_s32(b)));
  }

#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  hadds(const SIMDVec<SIMDFloat,16> &a,
	const SIMDVec<SIMDFloat,16> &b)
  {
    return hadd(a, b);
  }
#endif

  // -------------------------------------------------------------------------
  // hsub
  // -------------------------------------------------------------------------

  // via vuzp (unzip)
#define SIMDVEC_NEON_HOR_VIA_UZP(FCT,TYPE,NEON_FCT,NEON_SUF)		\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  FCT(const SIMDVec<TYPE,16> &a,					\
      const SIMDVec<TYPE,16> &b)					\
  {									\
    SIMDVecNeon<TYPE>::TypeX2 unzip;					\
    unzip = vuzpq_ ## NEON_SUF (a, b);					\
    return NEON_FCT ## _ ## NEON_SUF (unzip.val[0], unzip.val[1]);	\
  }
  
  SIMDVEC_NEON_HOR_VIA_UZP(hsub,SIMDByte,vsubq,u8)
  SIMDVEC_NEON_HOR_VIA_UZP(hsub,SIMDSignedByte,vsubq,s8)
  SIMDVEC_NEON_HOR_VIA_UZP(hsub,SIMDWord,vsubq,u16)
  SIMDVEC_NEON_HOR_VIA_UZP(hsub,SIMDShort,vsubq,s16)
  SIMDVEC_NEON_HOR_VIA_UZP(hsub,SIMDInt,vsubq,s32)
  SIMDVEC_NEON_HOR_VIA_UZP(hsub,SIMDFloat,vsubq,f32)

  // -------------------------------------------------------------------------
  // hsubs
  // -------------------------------------------------------------------------

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  hsubs(const SIMDVec<SIMDByte,16> &a,
        const SIMDVec<SIMDByte,16> &b)
  {
    SIMDVec<SIMDByte, 16> x, y;
    unzip(a, b, x, y, Bytes<1>());
    return subs(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  hsubs(const SIMDVec<SIMDSignedByte,16> &a,
        const SIMDVec<SIMDSignedByte,16> &b)
  {
    SIMDVec<SIMDSignedByte, 16> x, y;
    unzip(a, b, x, y, Bytes<1>());
    return subs(x, y);
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  hsubs(const SIMDVec<SIMDWord,16> &a,
        const SIMDVec<SIMDWord,16> &b)
  {
    SIMDVec<SIMDWord, 16> x, y;
    unzip(a, b, x, y, Bytes<2>());
    return subs(x, y);
  }

  SIMDVEC_NEON_HOR_VIA_UZP(hsubs,SIMDShort,vqsubq,s16)
  SIMDVEC_NEON_HOR_VIA_UZP(hsubs,SIMDInt,vqsubq,s32)
#ifndef SIMD_STRICT_SATURATION
  // NOT SATURATED!
  SIMDVEC_NEON_HOR_VIA_UZP(hsubs,SIMDFloat,vsubq,f32)
#endif

  // -------------------------------------------------------------------------
  // alignre (moved above srle, slle)
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_ALIGNRE_ALL()

  // -------------------------------------------------------------------------
  // element-wise shift right
  // -------------------------------------------------------------------------
  
  // all types, done via alignre
  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,16>
  srle(const SIMDVec<T,16> &a)
  {
    return alignre<IMM>(setzero(OutputType<T>(), Integer<16>()), a);
  }

  // -------------------------------------------------------------------------
  // element-wise shift left
  // -------------------------------------------------------------------------

  // all types, done via alignre
  // we have to check the range since alignre doesn't accept negative IMM

  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,16>
  slle(const SIMDVec<T,16> &a, IsPosInRange<true, true>)
  {
    return alignre<SIMDVec<T,16>::elements - IMM>(a, setzero(OutputType<T>(), Integer<16>()));
  }

  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,16>
  slle(const SIMDVec<T,16> &, IsPosInRange<true, false>)
  {
    return setzero(OutputType<T>(), Integer<16>());
  }

  template <int IMM, typename T>
  static SIMD_INLINE SIMDVec<T,16>
  slle(const SIMDVec<T,16> &a)
  {
    return slle<IMM>(a, IsPosInGivenRange<SIMDVec<T,16>::elements,IMM>());
  }

  // -------------------------------------------------------------------------
  // extraction of element 0
  // -------------------------------------------------------------------------

  // all types, done via extract
  template <typename T>
  static SIMD_INLINE T
  elem0(const SIMDVec<T,16> &a)
  {
    return extract<0>(a);
  }

  // -------------------------------------------------------------------------
  // swizzle
  // -------------------------------------------------------------------------

  // this mask is rearranged using vldN (n=1..4,sizeof=*)
  static const SIMDByte swizzleMask16[32] SIMD_ATTR_ALIGNED(16) =
    { 0,  1,  2,  3,  4,  5,  6,  7,  
      8,  9,  10, 11, 12, 13, 14, 15, 
      16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 26, 27, 28, 29, 30, 31};

  // n=5,sizeof=1, 99=invalid (mapped to 0 by vtbl3)
  static const SIMDByte swizzleMask16_5_1[48] SIMD_ATTR_ALIGNED(16) =
  { 0,  5,  10, 15, 1,  6,  11, 16,
    2,  7,  12, 17, 3,  8,  13, 18,
    4,  9,  14, 19, 99, 99, 99, 99,
    20, 25, 30, 35, 21, 26, 31, 36, 
    22, 27, 32, 37, 23, 28, 33, 38, 
    24, 29, 34, 39, 99, 99, 99, 99};

  // n=5,sizeof=2, 99=invalid (mapped to 0 by vtbl3)
  static const SIMDByte swizzleMask16_5_2[48] SIMD_ATTR_ALIGNED(16) =
  { 0,  1,  10, 11, 2,  3,  12, 13,
    4,  5,  14, 15, 6,  7,  16, 17,
    8,  9,  18, 19, 99, 99, 99, 99,
    20, 21, 30, 31, 22, 23, 32, 33, 
    24, 25, 34, 35, 26, 27, 36, 37,
    28, 29, 38, 39, 99, 99, 99, 99};

  // n=5,sizeof=4, 99=invalid (mapped to 0 by vtbl3)
  static const SIMDByte swizzleMask16_5_4[48] SIMD_ATTR_ALIGNED(16) =
  { 0,  1,  2,  3,  4,  5,  6,  7,
    8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 99, 99, 99, 99,
    20, 21, 22, 23, 24, 25, 26, 27, 
    28, 29, 30, 31, 32, 33, 34, 35, 
    36, 37, 38, 39, 99, 99, 99, 99};
  
  // n=5,sizeof=index
  static const SIMDByte* swizzleMask16_5[5] =
    { NULL, swizzleMask16_5_1, swizzleMask16_5_2, NULL, swizzleMask16_5_4 };

} // namespace base
} // namespace internal
  
  // ---------- swizzle tables ----------

  // n = 2, 3, 4
#define SIMDVEC_NEON_SWIZZLE_TABLE(NUM,T,NEON_SUF)			\
  template <> struct SwizzleTable<NUM,T,16>				\
  {									\
    typedef internal::base::SIMDVecNeon<T>::ScalarType ScalarType;			\
    typedef internal::base::SIMDVecNeonArray64<NUM,T>::Type TableLoadType;		\
    typedef internal::base::SIMDVecNeonArray64<NUM,SIMDByte>::Type TableType;		\
    typedef internal::base::SIMDVecNeon<SIMDByte>::Type64 TableValType;			\
    TableType table;							\
    SwizzleTable()							\
    {									\
      TableLoadType loadTable =						\
	vld ## NUM ## _ ## NEON_SUF((ScalarType*) internal::base::swizzleMask16);	\
      for (int i = 0; i < NUM; i++)					\
        table.val[i] = vreinterpret_u8_ ## NEON_SUF(loadTable.val[i]);	\
    }									\
    void print(FILE *f = stdout)					\
    {									\
      internal::base::SIMDVecNeon<SIMDByte>::ScalarType buf[8] SIMD_ATTR_ALIGNED(8);	\
      for (int i = 0; i < NUM; i++) {					\
	vst1_u8(buf, table.val[i]);					\
	for (int j = 0; j < 8; j++)					\
	  fprintf(f, "%2d ", buf[j]);					\
	fprintf(f, "\n");						\
      }									\
    }									\
  };

  // n = 5
  template <typename T> struct SwizzleTable<5,T,16>
  {
    typedef typename internal::base::SIMDVecNeon<SIMDByte>::ScalarType ScalarType;
    typedef typename internal::base::SIMDVecNeonArray64<3,SIMDByte>::Type TableType;
    typedef typename internal::base::SIMDVecNeon<SIMDByte>::Type64 TableValType;	
    // two tables for n=3
    TableType table[2];
    SwizzleTable() 
    { 
      ScalarType* mask = (ScalarType*) internal::base::swizzleMask16_5[sizeof(T)]; 
      for (int i = 0, off = 0; i < 3; i++, off += 8) {
	// first half (applied to vectors 0,1,2)
	table[0].val[i] = vld1_u8(mask + off);
	// second half (applied to vectors 2,3,4), index-16, 99-16=invalid
	table[1].val[i] = vsub_u8(vld1_u8(mask + 24 + off), vmov_n_u8(16));
      }
    }
    void print(FILE *f = stdout)
    {
      internal::base::SIMDVecNeon<SIMDByte>::ScalarType buf[8] SIMD_ATTR_ALIGNED(8);
      for (int j = 0; j < 2; j++)
	for (int i = 0; i < 3; i++) {				     
	  vst1_u8(buf, table[j].val[i]);				     
	  for (int j = 0; j < 8; j++)				     
	    fprintf(f, "%2d ", buf[j]);				     
	  fprintf(f, "\n");					     
	}								     
    }
  };

#define SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(T,NEON_SUF)			\
  SIMDVEC_NEON_SWIZZLE_TABLE(1,T,NEON_SUF)				\
  SIMDVEC_NEON_SWIZZLE_TABLE(2,T,NEON_SUF)				\
  SIMDVEC_NEON_SWIZZLE_TABLE(3,T,NEON_SUF)				\
  SIMDVEC_NEON_SWIZZLE_TABLE(4,T,NEON_SUF)

  SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(SIMDByte,u8)
  SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(SIMDSignedByte,s8)
  SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(SIMDWord,u16)
  SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(SIMDShort,s16)
  SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(SIMDInt,s32)
  SIMDVEC_NEON_SWIZZLE_TABLE_ALLNUM(SIMDFloat,f32)

namespace internal {
namespace base {

  // ---------- swizzle (AoS to SoA) ----------
  
  // e.g. NUM==3:
  //
  // i     0        0          0        1          1        1  
  // j     0        1          2        0          1        2
  // k     0        1          2        3          4        5
  // k>>1  0        0          1        1          2        2
  //     | v0                | v1                | v2                |
  //     | vu0.val0 vu0.val1 | vu0.val2 vu1.val0 | vu1.val1 vu1.val2 |
  //     | vu0                         | vu1                         | 
  // j=0 | t.table.val0                | t.table.val0                | -> v0
  // j=1 | t.table.val1                | t.table.val1                | -> v1
  // j=2 | t.table.val2                | t.table.val2                | -> v2
  // i     0                             1

#define SIMDVEC_NEON_SWIZZLE(NUM)					\
  template <typename T>							\
  static SIMD_INLINE void						\
  swizzle(const SwizzleTable<NUM, T, 16> &t,				\
	  SIMDVec<T, 16> *const v,					\
	  Integer<NUM>,							\
	  TypeIsIntSize<T>)						\
  {									\
    typename SwizzleTable<NUM, T, 16>::TableType vu[2];			\
    for (int i = 0, k = 0; i < 2; i++)					\
      for (int j = 0; j < NUM; j++, k++) {				\
	SIMDVec<SIMDByte,16> vb = reinterpret(v[k >> 1], OutputType<SIMDByte>());	\
	vu[i].val[j] = (k & 1) ? vget_high_u8(vb) : vget_low_u8(vb);	\
      }									\
    typename SwizzleTable<NUM, T, 16>::TableValType ru[2];		\
    for (int j = 0; j < NUM; j++) {					\
      for (int i = 0; i < 2; i++)					\
	ru[i] = vtbl ## NUM ## _u8 (vu[i], t.table.val[j]);		\
      v[j] = reinterpret						\
	(SIMDVec<SIMDByte,16>(vcombine_u8(ru[0], ru[1])), OutputType<T>());		\
    }									\
  }
  
  // -------------------- n = 1 --------------------

  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<1, T, 16> &,
	  SIMDVec<T, 16> *const,
	  Integer<1>,
	  TypeIsIntSize<T>)
  {
    // v remains unchanged
  }

  // -------------------- n = 2 --------------------

  SIMDVEC_NEON_SWIZZLE(2)

  // -------------------- n = 3 --------------------

  SIMDVEC_NEON_SWIZZLE(3)

  // -------------------- n = 4 --------------------

  SIMDVEC_NEON_SWIZZLE(4)

  // -------------------- n = 5 --------------------

  template <typename T>
  static SIMD_INLINE void
  swizzle(const SwizzleTable<5, T, 16> &t,
	  SIMDVec<T, 16> *const v,
	  Integer<5>,
	  TypeIsIntSize<T>)
  {
    //     |  v0l  v0h  |  v1l  v1h  |  v2l  v2h  |  v3l  v3h  |  v4l  v4h  |
    // i=0:
    // k:      0    1       2               
    // j:      0    1       2    
    //     | vu0.0 v0.1  vu0.2|
    // i=1:
    // k:                   2    3       4  
    // j:                   0    1       2 
    //                  |vu1.0 vu1.1   vu1.2|
    // i=2:
    // k:                                     5       6    7             
    // j:                                     0       1    2
    //                                      |vu2.0  v2.1 vu2.2|
    // i=3:
    // k:                                                  7       8    9
    // j:                                                  0       1    2
    //                                                  |vu3.0  vu3.1 vu3.2 |
    //
    //       n=0:                             n=1:
    //       i=0:         i=1:                i=0:        i=1:
    //       k=0:         k=1:                k=2:        k=3:
    // j=0:
    //     | t.table[0].val[0]|             | t.table[0].val[0]|     
    //                  | t.table[1].val[0] |           | t.table[1].val[0] | 
    // j=1:
    //     | t.table[0].val[1]|             | t.table[0].val[1]|
    //                  | t.table[1].val[1] |           | t.table[1].val[1] |
    // j=2:
    //     | t.table[0].val[2]|             | t.table[0].val[2]|
    //                  | t.table[1].val[2] |           | t.table[1].val[2] | 

    typename SwizzleTable<5, T, 16>::TableType vu[4];
    // input half-vector index starts at k0
    int k0[4] = {0, 2, 5, 7};
    for (int i = 0; i < 4; i++)
      for (int j = 0, k = k0[i]; j < 3; j++, k++) {				
	SIMDVec<SIMDByte,16> vb = reinterpret(v[k >> 1], OutputType<SIMDByte>());	
	vu[i].val[j] = (k & 1) ? vget_high_u8(vb) : vget_low_u8(vb);	
      }
    typename SIMDVecNeon<SIMDByte>::Type64 r[2][3][3];
    // n: left/right half of input
    // k: index of vu
    for (int n = 0, k = 0; n < 2; n++)
      // i: left/right half of half input
      for (int i = 0; i < 2; i++, k++)
	// j: different 3-tables
	for (int j = 0; j < 3; j++)
	  // apply table
	  r[n][i][j] = vtbl3_u8(vu[k], t.table[i].val[j]);
    // zip 4-byte blocks together
    int32x2x2_t z[2][3];
    for (int n = 0; n < 2; n++)
      for (int j = 0; j < 3; j++)
	z[n][j] = vzip_s32(vreinterpret_s32_u8(r[n][0][j]),
			   vreinterpret_s32_u8(r[n][1][j]));
    // combine left and right halfs
    for (int j = 0, k = 0; j < 3; j++)
      for (int lh = 0; lh < 2; lh++) {
	v[k] = reinterpret(SIMDVec<SIMDInt,16>
			      (vcombine_s32(z[0][j].val[lh], z[1][j].val[lh])), OutputType<T>());
	if (++k == 5) break;
      }
  }

  // -------------------------------------------------------------------------
  // compare <
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_REINTERP_ALL(cmplt,vcltq)

  // -------------------------------------------------------------------------
  // compare <=
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_REINTERP_ALL(cmple,vcleq)

  // -------------------------------------------------------------------------
  // compare ==
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_REINTERP_ALL(cmpeq,vceqq)

  // -------------------------------------------------------------------------
  // compare >
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_REINTERP_ALL(cmpgt,vcgtq)

  // -------------------------------------------------------------------------
  // compare >=
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_REINTERP_ALL(cmpge,vcgeq)

  // -------------------------------------------------------------------------
  // compare !=
  // -------------------------------------------------------------------------

#define SIMDVEC_NEON_CMPNEQ(TYPE,NEON_SUF,NEON_USUF)			\
  static SIMD_INLINE SIMDVec<TYPE,16>					\
  cmpneq(const SIMDVec<TYPE,16> &a,					\
	 const SIMDVec<TYPE,16> &b)					\
  {									\
    return vreinterpretq_ ## NEON_SUF ## _ ## NEON_USUF			\
      (vmvnq_ ## NEON_USUF (vceqq_ ## NEON_SUF (a, b)));		\
  }
  
  SIMDVEC_NEON_CMPNEQ(SIMDByte,u8,u8)
  SIMDVEC_NEON_CMPNEQ(SIMDSignedByte,s8,u8)
  SIMDVEC_NEON_CMPNEQ(SIMDWord,u16,u16)
  SIMDVEC_NEON_CMPNEQ(SIMDShort,s16,u16)
  SIMDVEC_NEON_CMPNEQ(SIMDInt,s32,u32)
  SIMDVEC_NEON_CMPNEQ(SIMDFloat,f32,u32)

  // -------------------------------------------------------------------------
  // ifelse
  // -------------------------------------------------------------------------

  // vbslq, unsigned mask
#define SIMDVEC_NEON_IFELSE(T,NEON_SUF,NEON_USUF)		\
  static SIMD_INLINE SIMDVec<T,16>				\
  ifelse(const SIMDVec<T,16> &cond,				\
	 const SIMDVec<T,16> &trueVal,				\
	 const SIMDVec<T,16> &falseVal)				\
  {								\
    return vbslq_ ## NEON_SUF					\
      (vreinterpretq_ ## NEON_USUF ## _ ## NEON_SUF(cond),	\
       trueVal,							\
       falseVal);						\
  }
  
  SIMDVEC_NEON_IFELSE(SIMDByte,u8,u8)
  SIMDVEC_NEON_IFELSE(SIMDSignedByte,s8,u8)
  SIMDVEC_NEON_IFELSE(SIMDWord,u16,u16)
  SIMDVEC_NEON_IFELSE(SIMDShort,s16,u16)
  SIMDVEC_NEON_IFELSE(SIMDInt,s32,u32)
  SIMDVEC_NEON_IFELSE(SIMDFloat,f32,u32)

  // -------------------------------------------------------------------------
  // and
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_FLOAT_BY_INT(vandq)
  SIMDVEC_NEON_BINARY_ALL(and,vandq)

  // -------------------------------------------------------------------------
  // or
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_FLOAT_BY_INT(vorrq)
  SIMDVEC_NEON_BINARY_ALL(or,vorrq)

  // -------------------------------------------------------------------------
  // andnot
  // -------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE SIMDVec<T,16>
  andnot(const SIMDVec<T,16> &a,
	 const SIMDVec<T,16> &b)
  {
    return and(not(a), b);
  }

  // -------------------------------------------------------------------------
  // xor
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_FLOAT_BY_INT(veorq)
  SIMDVEC_NEON_BINARY_ALL(xor,veorq)

  // -------------------------------------------------------------------------
  // not
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_UNARY_ALLINT(not,vmvnq)

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  not(const SIMDVec<SIMDFloat,16> &a)
  {
    return vreinterpretq_f32_s32(vmvnq_s32(vreinterpretq_s32_f32(a)));
  }

  // -------------------------------------------------------------------------
  // avg: average with rounding up
  // -------------------------------------------------------------------------

  SIMDVEC_NEON_BINARY_ALLINT(avg,vrhaddq)

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  avg(const SIMDVec<SIMDFloat,16> &a,
      const SIMDVec<SIMDFloat,16> &b)
  {
    return vmulq_n_f32(vaddq_f32(a, b), 0.5f);
  }

  // -------------------------------------------------------------------------
  // test_all_zeros
  // -------------------------------------------------------------------------

  // from solution suggested by Henri Ylitie 
  // http://stackoverflow.com/questions/15389539/
  //   fastest-way-to-test-a-128-bit-neon-register-
  //   for-a-value-of-0-using-intrinsics

  SIMDVEC_NEON_BINARY_FLOAT_BY_INT_64(vorr)

  // vpmax has to operate on unsigned (u32), otherwise 0 could be
  // the max. of a pair even though the other value is non-zero (neg.)
#define SIMDVEC_NEON_TESTALLZEROS(T,NEON_SUF)				\
  static SIMD_INLINE int						\
  test_all_zeros(const SIMDVec<T,16> &a)				\
  {									\
    uint32x4_t au = vreinterpretq_u32_ ## NEON_SUF (a.reg);		\
    uint32x2_t tmp = vorr_u32(vget_low_u32(au), vget_high_u32(au));	\
    return !(vget_lane_u32(vpmax_u32(tmp, tmp), 0));			\
  }

  SIMDVEC_NEON_TESTALLZEROS(SIMDByte,u8)
  SIMDVEC_NEON_TESTALLZEROS(SIMDSignedByte,s8)
  SIMDVEC_NEON_TESTALLZEROS(SIMDWord,u16)
  SIMDVEC_NEON_TESTALLZEROS(SIMDShort,s16)
  SIMDVEC_NEON_TESTALLZEROS(SIMDInt,s32)
  SIMDVEC_NEON_TESTALLZEROS(SIMDFloat,f32)

  // -------------------------------------------------------------------------
  // test_all_ones
  // -------------------------------------------------------------------------

  template <typename T>
  static SIMD_INLINE int
  test_all_ones(const SIMDVec<T,16> &a)
  {
    return test_all_zeros(not(a));
  }

  // -------------------------------------------------------------------------
  // reverse
  // -------------------------------------------------------------------------

  // https://stackoverflow.com/questions/18760784/
  //         reverse-vector-order-in-arm-neon-intrinsics
  
#define SIMDVEC_NEON_REVERSE(T,NEON_SUF)				\
  static SIMD_INLINE SIMDVec<T,16>					\
  reverse(const SIMDVec<T,16> &a)					\
  {									\
    SIMDVecNeon<T>::Type t = vrev64q_ ## NEON_SUF (a);			\
    return vcombine_ ## NEON_SUF					\
      (vget_high_ ## NEON_SUF (t), vget_low_ ## NEON_SUF (t));		\
  } 
  
  SIMDVEC_NEON_REVERSE(SIMDByte,u8)
  SIMDVEC_NEON_REVERSE(SIMDSignedByte,s8)
  SIMDVEC_NEON_REVERSE(SIMDWord,u16)
  SIMDVEC_NEON_REVERSE(SIMDShort,s16)
  SIMDVEC_NEON_REVERSE(SIMDInt,s32)
  SIMDVEC_NEON_REVERSE(SIMDFloat,f32)
  

  // ---------------------------------------------------------------------------
  // msb2int
  // ---------------------------------------------------------------------------

  // 17. Sep 22 (Jonas Keller): added msb2int functions

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDByte, 16> &a)
  {
    // from: https://stackoverflow.com/a/58381188/8461272

    // Example input (half scale):
    // 0x89 FF 1D C0 00 10 99 33

    // Shift out everything but the sign bits
    // 0x01 01 00 01 00 00 01 00
    uint8x16_t high_bits = vshrq_n_u8(a, 7);

    // Merge the even lanes together with vsra. The '??' bytes are garbage.
    // vsri could also be used, but it is slightly slower on aarch64.
    // 0x??03 ??02 ??00 ??01
    uint16x8_t paired16 = vsraq_n_u16(vreinterpretq_u16_u8(high_bits),
                                      vreinterpretq_u16_u8(high_bits), 7);
    // Repeat with wider lanes.
    // 0x??????0B ??????04
    uint32x4_t paired32 = vsraq_n_u32(vreinterpretq_u32_u16(paired16),
                                      vreinterpretq_u32_u16(paired16), 14);
    // 0x??????????????4B
    uint64x2_t paired64 = vsraq_n_u64(vreinterpretq_u64_u32(paired32),
                                      vreinterpretq_u64_u32(paired32), 28);
    // Extract the low 8 bits from each lane and join.
    // 0x4B
    return vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 0) |
           ((int)vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 8) << 8);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDSignedByte,16> &a)
  {
    // the same as msb2int(SIMDVec<SIMDByte,16>)
    return msb2int(reinterpret(a, OutputType<SIMDByte>()));
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDWord,16> &a)
  {
    // analogous to msb2int(SIMDVec<SIMDByte,16>)
    // idea from: https://stackoverflow.com/a/58381188/8461272

    // Shift out everything but the sign bits
    uint16x8_t high_bits = vshrq_n_u16(a, 15);

    // Merge the even lanes together with vsra. The '??' bytes are garbage.
    uint32x4_t paired32 = vsraq_n_u32(vreinterpretq_u32_u16(high_bits),
                                      vreinterpretq_u32_u16(high_bits), 15);
    // Repeat with wider lanes.
    uint64x2_t paired64 = vsraq_n_u64(vreinterpretq_u64_u32(paired32),
                                      vreinterpretq_u64_u32(paired32), 30);
    // Extract the low 4 bits from each lane and join.
    return (vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 0) & 0xf) |
           (vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 8) << 4);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDShort,16> &a)
  {
    // the same as msb2int(SIMDVec<SIMDWord,16>)
    return msb2int(reinterpret(a, OutputType<SIMDWord>()));
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDInt,16> &a)
  {
    // analogous to msb2int(SIMDVec<SIMDByte,16>)
    // idea from: https://stackoverflow.com/a/58381188/8461272

    // Shift out everything but the sign bits
    uint32x4_t high_bits = vshrq_n_u32(vreinterpretq_u32_s32(a), 31);

    // Merge the even lanes together with vsra. The '??' bytes are garbage.
    uint64x2_t paired64 = vsraq_n_u64(vreinterpretq_u64_u32(high_bits),
                                      vreinterpretq_u64_u32(high_bits), 31);
    // Extract the low 2 bits from each lane and join.
    return (vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 0) & 0x3) |
           ((vgetq_lane_u8(vreinterpretq_u8_u64(paired64), 8) & 0x3) << 2);
  }

  static SIMD_INLINE uint64_t
  msb2int(const SIMDVec<SIMDFloat,16> &a)
  {
    // the same as msb2int(SIMDVec<SIMDInt,16>)
    return msb2int(reinterpret(a, OutputType<SIMDInt>()));
  }

  // ---------------------------------------------------------------------------
  // int2msb
  // ---------------------------------------------------------------------------

  // 06. Oct 22 (Jonas Keller): added int2msb functions

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  int2msb(const uint64_t a, OutputType<SIMDByte>, Integer<16>)
  {
    uint8x8_t aVecLo = vdup_n_u8(a & 0xff);
    uint8x8_t aVecHi = vdup_n_u8((a >> 8) & 0xff);
    uint8x16_t aVec = vcombine_u8(aVecLo, aVecHi);
    // shift the bits to the msb
    int8x16_t shiftAmounts = {7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0};
    uint8x16_t shifted = vshlq_u8(aVec, shiftAmounts);
    return vandq_u8(shifted, vdupq_n_u8(0x80));
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  int2msb(const uint64_t a, OutputType<SIMDSignedByte>, Integer<16>)
  {
    return reinterpret(int2msb(a, OutputType<SIMDByte>(), Integer<16>()), OutputType<SIMDSignedByte>());
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  int2msb(const uint64_t a, OutputType<SIMDWord>, Integer<16>)
  {
    uint16x8_t aVec = vdupq_n_u16(a & 0xff);
    // shift the bits to the msb
    int16x8_t shiftAmounts = {15, 14, 13, 12, 11, 10, 9, 8};
    uint16x8_t shifted = vshlq_u16(aVec, shiftAmounts);
    return vandq_u16(shifted, vdupq_n_u16(0x8000));
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  int2msb(const uint64_t a, OutputType<SIMDShort>, Integer<16>)
  {
    return reinterpret(int2msb(a, OutputType<SIMDWord>(), Integer<16>()), OutputType<SIMDShort>());
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  int2msb(const uint64_t a, OutputType<SIMDInt>, Integer<16>)
  {
    int32x4_t aVec = vdupq_n_s32(a & 0xf);
    // shift the bits to the msb
    int32x4_t shiftAmounts = {31, 30, 29, 28};
    int32x4_t shifted = vshlq_s32(aVec, shiftAmounts);
    return vandq_s32(shifted, vdupq_n_s32(0x80000000));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  int2msb(const uint64_t a, OutputType<SIMDFloat>, Integer<16>)
  {
    return reinterpret(int2msb(a, OutputType<SIMDInt>(), Integer<16>()), OutputType<SIMDFloat>());
  }

  // ---------------------------------------------------------------------------
  // int2bits
  // ---------------------------------------------------------------------------

  // 09. Oct 22 (Jonas Keller): added int2bits functions

  static SIMD_INLINE SIMDVec<SIMDByte,16>
  int2bits(const uint64_t a, OutputType<SIMDByte>, Integer<16>)
  {
    uint8x8_t aVecLo = vdup_n_u8(a & 0xff);
    uint8x8_t aVecHi = vdup_n_u8((a >> 8) & 0xff);
    uint8x16_t aVec = vcombine_u8(aVecLo, aVecHi);
    uint8x16_t sel = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    return vtstq_u8(aVec, sel);
  }

  static SIMD_INLINE SIMDVec<SIMDSignedByte,16>
  int2bits(const uint64_t a, OutputType<SIMDSignedByte>, Integer<16>)
  {
    return reinterpret(int2bits(a, OutputType<SIMDByte>(), Integer<16>()), OutputType<SIMDSignedByte>());
  }

  static SIMD_INLINE SIMDVec<SIMDWord,16>
  int2bits(const uint64_t a, OutputType<SIMDWord>, Integer<16>)
  {
    uint16x8_t aVec = vdupq_n_u16(a & 0xff);
    uint16x8_t sel = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    return vtstq_u16(aVec, sel);
  }

  static SIMD_INLINE SIMDVec<SIMDShort,16>
  int2bits(const uint64_t a, OutputType<SIMDShort>, Integer<16>)
  {
    return reinterpret(int2bits(a, OutputType<SIMDWord>(), Integer<16>()), OutputType<SIMDShort>());
  }

  static SIMD_INLINE SIMDVec<SIMDInt,16>
  int2bits(const uint64_t a, OutputType<SIMDInt>, Integer<16>)
  {
    int32x4_t aVec = vdupq_n_s32(a & 0xf);
    int32x4_t sel = {0x01, 0x02, 0x04, 0x08};
    return vreinterpretq_s32_u32(vtstq_s32(aVec, sel));
  }

  static SIMD_INLINE SIMDVec<SIMDFloat,16>
  int2bits(const uint64_t a, OutputType<SIMDFloat>, Integer<16>)
  {
    return reinterpret(int2bits(a, OutputType<SIMDInt>(), Integer<16>()), OutputType<SIMDFloat>());
  }
} // namespace base
} // namespace internal
} // namespace ns_simd

#endif // SIMDVEC_NEON_ENABLE

#endif // _SIMD_VEC_BASE_IMPL_NEON_16_H_
