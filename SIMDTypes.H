// ===========================================================================
// 
// SIMDTypes.H --
// some basic types used for SIMD programming
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

#ifndef _SIMD_TYPES_H_
#define _SIMD_TYPES_H_

#include "SIMDDefs.H"

#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

namespace ns_simd {
  
  // ===========================================================================
  // data types
  // ===========================================================================

  // these types indicate that data is processed by SIMD vector instructions;
  // constants are copied from stdint.h so that we don't have to fiddle
  // with __STDC_LIMIT_MACROS

  // unsigned byte
  typedef uint8_t SIMDByte;
#define SIMDBYTE_MAX (255)
#define SIMDBYTE_TRUE (255)
  typedef SIMDByte* SIMDBytePtr;

  // signed byte
  typedef int8_t SIMDSignedByte;
#define SIMDSIGNEDBYTE_MIN (-128)
#define SIMDSIGNEDBYTE_MAX (127)
#define SIMDSIGNEDBYTE_TRUE (-1)
  typedef SIMDSignedByte* SIMDSignedBytePtr;

  // unsigned word (16 bit)
  typedef uint16_t SIMDWord;	
#define SIMDWORD_MAX (65535)
#define SIMDWORD_TRUE (65535)
  typedef SIMDWord* SIMDWordPtr;

  // signed word (16 bit)
  typedef int16_t SIMDShort;
#define SIMDSHORT_MIN (-32768)
#define SIMDSHORT_MAX (32767)
#define SIMDSHORT_TRUE (-1)
  typedef SIMDShort* SIMDShortPtr;

  // signed int (32 bit)
  typedef int32_t SIMDInt;
#define SIMDINT_MIN (-2147483647-1)
#define SIMDINT_MAX (2147483647)
#define SIMDINT_TRUE (-1)
  typedef SIMDInt* SIMDIntPtr;

  // float
  typedef float SIMDFloat;
#define SIMDFLOAT_MIN (-3.402823466E+38F)
#define SIMDFLOAT_MAX (3.402823466E+38F)
  typedef SIMDFloat* SIMDFloatPtr;

  // 0x7fffff80
#define MAX_POS_FLOAT_CONVERTIBLE_TO_INT32 2147483520.0f

  // ===========================================================================
  // generic compile-time check (Alexandrescu "Modern C++ Design" p.25)
  // ===========================================================================

  template<bool> struct SIMDStaticAssert;
  template<> struct SIMDStaticAssert<true> {};
#define SIMD_STATIC_ASSERT(EXPR) (SIMDStaticAssert<(EXPR) != 0>())

  // ===========================================================================
  // size comparison of types (for compile-time check)
  // ===========================================================================

  // not elegant, but probably produces a well-readable error message

  // left\right
  //    SB B S W I F 
  // SB  x x x x x x
  //  B  x x x x x x
  //  S      x x x x
  //  W      x x x x
  //  I          x x
  //  F          x x

  template <typename Tleft, typename Tright>
  static SIMD_INLINE void typeSizeLEQ();

  template <> SIMD_INLINE void typeSizeLEQ<SIMDSignedByte,SIMDSignedByte>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDSignedByte,SIMDByte>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDSignedByte,SIMDShort>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDSignedByte,SIMDWord>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDSignedByte,SIMDInt>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDSignedByte,SIMDFloat>() {}

  template <> SIMD_INLINE void typeSizeLEQ<SIMDByte,SIMDSignedByte>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDByte,SIMDByte>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDByte,SIMDShort>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDByte,SIMDWord>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDByte,SIMDInt>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDByte,SIMDFloat>() {}

  template <> SIMD_INLINE void typeSizeLEQ<SIMDShort,SIMDShort>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDShort,SIMDWord>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDShort,SIMDInt>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDShort,SIMDFloat>() {}

  template <> SIMD_INLINE void typeSizeLEQ<SIMDWord,SIMDShort>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDWord,SIMDWord>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDWord,SIMDInt>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDWord,SIMDFloat>() {}

  template <> SIMD_INLINE void typeSizeLEQ<SIMDInt,SIMDInt>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDInt,SIMDFloat>() {}

  template <> SIMD_INLINE void typeSizeLEQ<SIMDFloat,SIMDInt>() {}
  template <> SIMD_INLINE void typeSizeLEQ<SIMDFloat,SIMDFloat>() {}

  // ===========================================================================
  // signed/unsigned (for compile-time check)
  // ===========================================================================

  template <typename T> static SIMD_INLINE void signedType();
  template <> SIMD_INLINE void signedType<SIMDSignedByte>() {}
  template <> SIMD_INLINE void signedType<SIMDShort>() {}
  template <> SIMD_INLINE void signedType<SIMDInt>() {}
  template <> SIMD_INLINE void signedType<SIMDFloat>() {}

  template <typename T> static SIMD_INLINE void unsignedType();
  template <> SIMD_INLINE void unsignedType<SIMDByte>() {}
  template <> SIMD_INLINE void unsignedType<SIMDWord>() {}

  // ===========================================================================
  // integer/float (for compile-time check)
  // ===========================================================================

  template <typename T> static SIMD_INLINE void integerType();
  template <> SIMD_INLINE void integerType<SIMDByte>() {}
  template <> SIMD_INLINE void integerType<SIMDSignedByte>() {}
  template <> SIMD_INLINE void integerType<SIMDWord>() {}
  template <> SIMD_INLINE void integerType<SIMDShort>() {}
  template <> SIMD_INLINE void integerType<SIMDInt>() {}

  template <typename T> static SIMD_INLINE void floatingPointType();
  template <> SIMD_INLINE void floatingPointType<SIMDFloat>() {}

  // ===========================================================================
  // SIMDTypeInfo
  // ===========================================================================

  // NOTE: min() and max() are functions since "floating-point literals are not
  // allowed in constant expressions according to -pedantic

  // defaultFormat() returns a full format specifier with % and space, can be
  // used for tests

  // 16. Nov 22 (Jonas Keller): added specialValues() for testing

  template <typename T>
  class SIMDTypeInfo;

  template <>
  class SIMDTypeInfo<SIMDByte>
  {
  public:
    static SIMD_INLINE const char* name() { return "SIMDByte"; };
    static SIMD_INLINE const char* format() { return "u"; };
    static SIMD_INLINE const char* defaultFormat() { return "%u "; };
    static SIMD_INLINE const char* hexFormat() { return "%02x "; }
    static SIMD_CONSTEXPR bool isSigned = false;
    static SIMD_CONSTEXPR bool isInteger = true;
    static SIMD_CONSTEXPR bool isFloatingPoint = false;
    static SIMD_INLINE SIMDByte min() { return 0; }
    static SIMD_INLINE SIMDByte max() { return SIMDBYTE_MAX; }
    static SIMD_INLINE SIMDByte trueval() { return SIMDBYTE_TRUE; }
    // special values mainly used for testing edge cases
    static std::vector<SIMDByte> specialValues() {
      SIMDByte v[] = {0, 1, SIMDBYTE_MAX / 2, SIMDBYTE_MAX - 1, SIMDBYTE_MAX};
      const size_t n = sizeof(v) / sizeof(SIMDByte);
      return std::vector<SIMDByte>(v, v + n);
    }
    typedef SIMDWord NextLargerType;
    typedef SIMDByte UnsignedType;
    typedef SIMDSignedByte SignedType;
  };

  template <>
  class SIMDTypeInfo<SIMDSignedByte>
  {
  public:
    static SIMD_INLINE const char* name() { return "SIMDSignedByte"; };
    static SIMD_INLINE const char* format() { return "d"; };
    static SIMD_INLINE const char* defaultFormat() { return "%d "; };
    static SIMD_INLINE const char* hexFormat() { return "%02x "; }
    static SIMD_CONSTEXPR bool isSigned = true;
    static SIMD_CONSTEXPR bool isInteger = true;
    static SIMD_CONSTEXPR bool isFloatingPoint = false;
    static SIMD_INLINE SIMDSignedByte min() { return SIMDSIGNEDBYTE_MIN; }
    static SIMD_INLINE SIMDSignedByte max() { return SIMDSIGNEDBYTE_MAX; }
    static SIMD_INLINE SIMDSignedByte trueval() { return SIMDSIGNEDBYTE_TRUE; }
    // special values mainly used for testing edge cases
    static std::vector<SIMDSignedByte> specialValues() {
      SIMDSignedByte v[] = {
        SIMDSIGNEDBYTE_MIN, SIMDSIGNEDBYTE_MIN+1, SIMDSIGNEDBYTE_MIN/2,
        -1, 0, 1,
        SIMDSIGNEDBYTE_MAX/2, SIMDSIGNEDBYTE_MAX-1, SIMDSIGNEDBYTE_MAX
      };
      const size_t n = sizeof(v) / sizeof(SIMDSignedByte);
      return std::vector<SIMDSignedByte>(v, v + n);
    }
    typedef SIMDShort NextLargerType;
    typedef SIMDByte UnsignedType;
    typedef SIMDSignedByte SignedType;
  };

  template <>
  class SIMDTypeInfo<SIMDWord>
  {
  public:
    static SIMD_INLINE const char* name() { return "SIMDWord"; };
    static SIMD_INLINE const char* format() { return "u"; };
    static SIMD_INLINE const char* defaultFormat() { return "%u "; };
    static SIMD_INLINE const char* hexFormat() { return "%04x "; }
    static SIMD_CONSTEXPR bool isSigned = false;
    static SIMD_CONSTEXPR bool isInteger = true;
    static SIMD_CONSTEXPR bool isFloatingPoint = false;
    static SIMD_INLINE SIMDWord min() { return 0; }
    static SIMD_INLINE SIMDWord max() { return SIMDWORD_MAX; }
    static SIMD_INLINE SIMDWord trueval() { return SIMDWORD_TRUE; }
    // special values mainly used for testing edge cases
    static std::vector<SIMDWord> specialValues() {
      SIMDWord v[] = {0, 1, SIMDWORD_MAX / 2, SIMDWORD_MAX - 1, SIMDWORD_MAX};
      const size_t n = sizeof(v) / sizeof(SIMDWord);
      return std::vector<SIMDWord>(v, v + n);
    }
    typedef SIMDInt NextLargerType; // no larger unsigned type, use SIMDInt
    typedef SIMDWord UnsignedType;
    typedef SIMDShort SignedType;
  };

  template <>
  class SIMDTypeInfo<SIMDShort>
  {
  public:
    static SIMD_INLINE const char* name() { return "SIMDShort"; };
    static SIMD_INLINE const char* format() { return "d"; };
    static SIMD_INLINE const char* defaultFormat() { return "%d "; };
    static SIMD_INLINE const char* hexFormat() { return "%04x "; }
    static SIMD_CONSTEXPR bool isSigned = true;
    static SIMD_CONSTEXPR bool isInteger = true;
    static SIMD_CONSTEXPR bool isFloatingPoint = false;
    static SIMD_INLINE SIMDShort min() { return SIMDSHORT_MIN; }
    static SIMD_INLINE SIMDShort max() { return SIMDSHORT_MAX; }
    static SIMD_INLINE SIMDShort trueval() { return SIMDSHORT_TRUE; }
    // special values mainly used for testing edge cases
    static std::vector<SIMDShort> specialValues() {
      SIMDShort v[] = {
        SIMDSHORT_MIN, SIMDSHORT_MIN+1, SIMDSHORT_MIN/2,
        -1, 0, 1,
        SIMDSHORT_MAX/2, SIMDSHORT_MAX-1, SIMDSHORT_MAX
      };
      const size_t n = sizeof(v) / sizeof(SIMDShort);
      return std::vector<SIMDShort>(v, v + n);
    }
    typedef SIMDInt NextLargerType;
    typedef SIMDWord UnsignedType;
    typedef SIMDShort SignedType;
  };

  template <>
  class SIMDTypeInfo<SIMDInt>
  {
  public:
    static SIMD_INLINE const char* name() { return "SIMDInt"; };
    static SIMD_INLINE const char* format() { return "d"; };
    static SIMD_INLINE const char* defaultFormat() { return "%d "; };
    static SIMD_INLINE const char* hexFormat() { return "%08x "; }
    static SIMD_CONSTEXPR bool isSigned = true;
    static SIMD_CONSTEXPR bool isInteger = true;
    static SIMD_CONSTEXPR bool isFloatingPoint = false;
    static SIMD_INLINE SIMDInt min() { return SIMDINT_MIN; }
    static SIMD_INLINE SIMDInt max() { return SIMDINT_MAX; }
    static SIMD_INLINE SIMDInt trueval() { return SIMDINT_TRUE; }
    // special values mainly used for testing edge cases
    static std::vector<SIMDInt> specialValues() {
      SIMDInt v[] = {
        SIMDINT_MIN, SIMDINT_MIN+1, SIMDINT_MIN/2,
        -1, 0, 1,
        SIMDINT_MAX/2, SIMDINT_MAX-1, SIMDINT_MAX
      };
      const size_t n = sizeof(v) / sizeof(SIMDInt);
      return std::vector<SIMDInt>(v, v + n);
    }
    typedef SIMDInt NextLargerType; // no larger integer type than SIMDInt
    typedef uint32_t UnsignedType; // not a SIMD type
    typedef SIMDInt SignedType;
  };

  template <>
  class SIMDTypeInfo<SIMDFloat>
  {
  public:
    static SIMD_INLINE const char* name() { return "SIMDFloat"; };
    static SIMD_INLINE const char* format() { return "g"; };
    static SIMD_INLINE const char* defaultFormat() { return "%g "; };
    static SIMD_CONSTEXPR bool isSigned = true;
    static SIMD_CONSTEXPR bool isInteger = false;
    static SIMD_CONSTEXPR bool isFloatingPoint = true;
    static SIMD_INLINE SIMDFloat min() { return SIMDFLOAT_MIN; }
    static SIMD_INLINE SIMDFloat max() { return SIMDFLOAT_MAX; }
    // very inconvenient: how can I specify a float constant with all bits 1?
    static SIMD_INLINE SIMDFloat trueval() 
    { 
      SIMDInt i = SIMDINT_TRUE; 
      SIMDFloat f;
      memcpy(&f, &i, sizeof(f));
      return f; 
    }
    // special values mainly used for testing edge cases
    static std::vector<SIMDFloat> specialValues() {
      SIMDFloat v[] = {
        SIMDFLOAT_MIN, SIMDFLOAT_MIN/2, SIMDFLOAT_MAX/2, SIMDFLOAT_MAX,
        -1.0f, -0.5f, -0.0f, 0.0f, 0.5f, 1.0f,
        MAX_POS_FLOAT_CONVERTIBLE_TO_INT32,
        MAX_POS_FLOAT_CONVERTIBLE_TO_INT32+1
      };
      const size_t n = sizeof(v) / sizeof(SIMDFloat);
      return std::vector<SIMDFloat>(v, v + n);
    }
    typedef SIMDFloat NextLargerType; // no larger float type than SIMDFloat
  };

  // ===========================================================================
  // formatting
  // ===========================================================================

  // note that for T=SIMDFloat, SIMDTypeInfo<T>::format() returns "g",
  // for which precision encodes the number of significant digits,
  // not the number of fractional digits
  template <typename T>
  class SIMDFormat
  {
  public:
    char format[256];
    
    SIMDFormat(int fieldWidth = -1, int precision = -1)
    {
      char fieldWidthStr[16], precisionStr[16];
      if (fieldWidth >= 0) sprintf(fieldWidthStr, "%d", fieldWidth);
      else strcpy(fieldWidthStr, "");
      if (precision >= 0) sprintf(precisionStr, ".%d", precision);
      else strcpy(precisionStr, "");
      if (SIMDTypeInfo<T>::isInteger)
	// integer format, precision is ignored
	sprintf(format, "%%%s%s", 
		fieldWidthStr, SIMDTypeInfo<T>::format());
      else
	// float format, precision is used
	sprintf(format, "%%%s%s%s", 
		fieldWidthStr, precisionStr, SIMDTypeInfo<T>::format());
    }
  };

  // a crude way to format SIMD* types as decimal number, can be used in
  // fprintf (use %s format specification)
  template <typename T>
  class SIMDDecimal
  {
  public:
    char str[256];

    SIMDDecimal(T value, int fieldWidth = -1, int precision = -1)
    {
      sprintf(str, SIMDFormat<T>(fieldWidth, precision).format, value);
    }
  };

  // ===========================================================================
  // tag dispatching
  // ===========================================================================

  // int2type trick from 
  // Andrei Alexandrescu: Modern C++ Design (Addison Wesley)

  template <bool isFloatingPoint> 
  struct IsFloatingPoint {};

  template <typename T>
  struct TypeIsFloatingPoint : 
    public IsFloatingPoint<SIMDTypeInfo<T>::isFloatingPoint> {};

  template <bool IS_INT, int SIZE>
  struct IsIntSize {};

  template <typename T>
  struct TypeIsIntSize : 
    public IsIntSize<SIMDTypeInfo<T>::isInteger, sizeof(T)> {};

  template <bool IS_INT, bool IS_SIGNED>
  struct IsIntIsSigned {};

  template <typename T>
  struct TypeIsIntIsSigned :
    public IsIntIsSigned<SIMDTypeInfo<T>::isInteger, 
			 SIMDTypeInfo<T>::isSigned> {};

  template <int N>
  struct Integer {};

  template <int N>
  struct Part {};

  template <int N>
  struct Elements {};

  template <int N>
  struct Bytes {};

  template <bool IS_POSITIVE,
	    bool AT_LOWER_LIMIT, 
	    int LOWER_LIMIT_INCLUSIVE, 
	    int UPPER_LIMIT_EXCLUSIVE>
  struct Range {};
  
  // 24. Nov 17 (rm): now includes check for IMM >= 0 to reveal
  // negative template parameters
  template <int IMM, int SIZE>
  struct SizeRange :
    public Range<(IMM >= 0),
                 SIMD_IS_MULTIPLE_OF(IMM,SIZE),
		 SIMD_PREV_MULTIPLE_OF(IMM,SIZE),
		 SIMD_NEXT_MULTIPLE_OF(IMM,SIZE)> {};

  template <int N>
  struct Compression {};

  template <int N>
  struct Expansion {};

  template <typename T>
  struct InputType {};

  template <typename T>
  struct OutputType {};

  // is positive and in a range
  template <bool pos, bool inRange>
  struct IsPosInRange {};

  // is positive and in given range [0, RANGE)
  template <int RANGE, int INDEX>
  struct IsPosInGivenRange :
    public IsPosInRange< (INDEX >= 0), (INDEX < RANGE) > {};

  // is positive and non-zero and in a range
  template <bool pos, bool nonZero, bool inRange>
  struct IsPosNonZeroInRange {};

  // is positive and non-zero and in a given range
  template <int RANGE, int INDEX>
  struct IsPosNonZeroInGivenRange :
    public IsPosNonZeroInRange< (INDEX >= 0),(INDEX != 0),(INDEX < RANGE) > {};

  // is positive and in bit range of given type
  template <typename T, int BIT>
  struct IsPosInBitRange :
    public IsPosInGivenRange< (sizeof(T) * 8), BIT > {};

} // namespace

#endif
