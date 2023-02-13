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

#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace ns_simd {
/**
 * @defgroup types Basic SIMD Types
 * @brief Basic types used for SIMD programming.
 * @{
 */

// ===========================================================================
// data types
// ===========================================================================

// these types indicate that data is processed by SIMD vector instructions;
// constants are copied from stdint.h so that we don't have to fiddle
// with __STDC_LIMIT_MACROS

/**
 * @name Basic element types
 * @{
 */
typedef uint8_t SIMDByte;      ///< Unsigned 8-bit integer
typedef int8_t SIMDSignedByte; ///< Signed 8-bit integer
typedef uint16_t SIMDWord;     ///< Unsigned 16-bit integer
typedef int16_t SIMDShort;     ///< Signed 16-bit integer
typedef int32_t SIMDInt;       ///< Signed 32-bit integer
typedef float SIMDFloat; ///< Single-precision floating point number (32-bit)
/** @} */

/**
 * @name Pointer types
 * @{
 */
typedef SIMDByte *SIMDBytePtr;             ///< Pointer to @ref SIMDByte
typedef SIMDSignedByte *SIMDSignedBytePtr; ///< Pointer to @ref SIMDSignedByte
typedef SIMDWord *SIMDWordPtr;             ///< Pointer to @ref SIMDWord
typedef SIMDShort *SIMDShortPtr;           ///< Pointer to @ref SIMDShort
typedef SIMDInt *SIMDIntPtr;               ///< Pointer to @ref SIMDInt
typedef SIMDFloat *SIMDFloatPtr;           ///< Pointer to @ref SIMDFloat
/** @} */

/// @cond
#define SIMDBYTE_MAX        (255)
#define SIMDBYTE_TRUE       (255)
#define SIMDSIGNEDBYTE_MIN  (-128)
#define SIMDSIGNEDBYTE_MAX  (127)
#define SIMDSIGNEDBYTE_TRUE (-1)
#define SIMDWORD_MAX        (65535)
#define SIMDWORD_TRUE       (65535)
#define SIMDSHORT_MIN       (-32768)
#define SIMDSHORT_MAX       (32767)
#define SIMDSHORT_TRUE      (-1)
#define SIMDINT_MIN         (-2147483647 - 1)
#define SIMDINT_MAX         (2147483647)
#define SIMDINT_TRUE        (-1)
#define SIMDFLOAT_MIN       (-3.402823466E+38F)
#define SIMDFLOAT_MAX       (3.402823466E+38F)
// 0x7fffff80
#define MAX_POS_FLOAT_CONVERTIBLE_TO_INT32 2147483520.0f
/// @endcond

// ===========================================================================
// generic compile-time check (Alexandrescu "Modern C++ Design" p.25)
// ===========================================================================

#if DOXYGEN
/**
 * @brief Compile-time assertion.
 *
 * This macro can be used to check a condition at compile time.
 *
 * @param EXPR The condition to check.
 */
#define SIMD_STATIC_ASSERT(EXPR)
/**
 * @brief Compile-time assertion with message.
 *
 * This macro can be used to check a condition at compile time with a
 * message.
 *
 * @param EXPR The condition to check.
 * @param MSG A message to be displayed if the condition is false.
 */
#define SIMD_STATIC_ASSERT_MSG(EXPR, MSG)

#else // DOXYGEN

#if __cplusplus >= 201703L // at least C++17
#define SIMD_STATIC_ASSERT(EXPR)          static_assert(EXPR)
#define SIMD_STATIC_ASSERT_MSG(EXPR, MSG) static_assert(EXPR, MSG)
#else // older C++
#define SIMD_STATIC_ASSERT(EXPR)          static_assert(EXPR, "")
#define SIMD_STATIC_ASSERT_MSG(EXPR, MSG) static_assert(EXPR, MSG)
#endif

#endif // DOXYGEN

// ===========================================================================
// SIMDTypeInfo
// ===========================================================================

// NOTE: min() and max() are functions since "floating-point literals are not
// allowed in constant expressions according to -pedantic

// defaultFormat() returns a full format specifier with % and space, can be
// used for tests

namespace internal {
namespace types {
template <typename T>
class SIMDTypeInfo;

template <>
class SIMDTypeInfo<SIMDByte>
{
public:
  static SIMD_INLINE const char *name() { return "SIMDByte"; };
  static SIMD_INLINE const char *format() { return "u"; };
  static SIMD_INLINE const char *defaultFormat() { return "%u "; };
  static SIMD_INLINE const char *hexFormat() { return "%02x "; }
  static constexpr bool isSigned        = false;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static SIMD_INLINE SIMDByte min() { return 0; }
  static SIMD_INLINE SIMDByte max() { return SIMDBYTE_MAX; }
  static SIMD_INLINE SIMDByte trueval() { return SIMDBYTE_TRUE; }
  typedef SIMDWord NextLargerType;
  typedef SIMDByte UnsignedType;
  typedef SIMDSignedByte SignedType;
};

template <>
class SIMDTypeInfo<SIMDSignedByte>
{
public:
  static SIMD_INLINE const char *name() { return "SIMDSignedByte"; };
  static SIMD_INLINE const char *format() { return "d"; };
  static SIMD_INLINE const char *defaultFormat() { return "%d "; };
  static SIMD_INLINE const char *hexFormat() { return "%02x "; }
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static SIMD_INLINE SIMDSignedByte min() { return SIMDSIGNEDBYTE_MIN; }
  static SIMD_INLINE SIMDSignedByte max() { return SIMDSIGNEDBYTE_MAX; }
  static SIMD_INLINE SIMDSignedByte trueval() { return SIMDSIGNEDBYTE_TRUE; }
  typedef SIMDShort NextLargerType;
  typedef SIMDByte UnsignedType;
  typedef SIMDSignedByte SignedType;
};

template <>
class SIMDTypeInfo<SIMDWord>
{
public:
  static SIMD_INLINE const char *name() { return "SIMDWord"; };
  static SIMD_INLINE const char *format() { return "u"; };
  static SIMD_INLINE const char *defaultFormat() { return "%u "; };
  static SIMD_INLINE const char *hexFormat() { return "%04x "; }
  static constexpr bool isSigned        = false;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static SIMD_INLINE SIMDWord min() { return 0; }
  static SIMD_INLINE SIMDWord max() { return SIMDWORD_MAX; }
  static SIMD_INLINE SIMDWord trueval() { return SIMDWORD_TRUE; }
  typedef SIMDInt NextLargerType; // no larger unsigned type, use SIMDInt
  typedef SIMDWord UnsignedType;
  typedef SIMDShort SignedType;
};

template <>
class SIMDTypeInfo<SIMDShort>
{
public:
  static SIMD_INLINE const char *name() { return "SIMDShort"; };
  static SIMD_INLINE const char *format() { return "d"; };
  static SIMD_INLINE const char *defaultFormat() { return "%d "; };
  static SIMD_INLINE const char *hexFormat() { return "%04x "; }
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static SIMD_INLINE SIMDShort min() { return SIMDSHORT_MIN; }
  static SIMD_INLINE SIMDShort max() { return SIMDSHORT_MAX; }
  static SIMD_INLINE SIMDShort trueval() { return SIMDSHORT_TRUE; }
  typedef SIMDInt NextLargerType;
  typedef SIMDWord UnsignedType;
  typedef SIMDShort SignedType;
};

template <>
class SIMDTypeInfo<SIMDInt>
{
public:
  static SIMD_INLINE const char *name() { return "SIMDInt"; };
  static SIMD_INLINE const char *format() { return "d"; };
  static SIMD_INLINE const char *defaultFormat() { return "%d "; };
  static SIMD_INLINE const char *hexFormat() { return "%08x "; }
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static SIMD_INLINE SIMDInt min() { return SIMDINT_MIN; }
  static SIMD_INLINE SIMDInt max() { return SIMDINT_MAX; }
  static SIMD_INLINE SIMDInt trueval() { return SIMDINT_TRUE; }
  typedef SIMDInt NextLargerType; // no larger integer type than SIMDInt
  typedef uint32_t UnsignedType;  // not a SIMD type
  typedef SIMDInt SignedType;
};

template <>
class SIMDTypeInfo<SIMDFloat>
{
public:
  static SIMD_INLINE const char *name() { return "SIMDFloat"; };
  static SIMD_INLINE const char *format() { return "g"; };
  static SIMD_INLINE const char *defaultFormat() { return "%g "; };
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = false;
  static constexpr bool isFloatingPoint = true;
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
  typedef SIMDFloat NextLargerType; // no larger float type than SIMDFloat
  typedef SIMDFloat UnsignedType;   // no unsigned float type
  typedef SIMDFloat SignedType;
};
} // namespace types
} // namespace internal

/**
 * @brief Type information for SIMD types
 */
template <typename T>
class SIMDTypeInfo
{
public:
  /// @brief Returns the name of the type (e.g. "SIMDInt" for SIMDInt)
  static SIMD_INLINE const char *name()
  {
    return internal::types::SIMDTypeInfo<T>::name();
  };
  /// @brief Returns the format string for printf (e.g. "d" for SIMDInt)
  static SIMD_INLINE const char *format()
  {
    return internal::types::SIMDTypeInfo<T>::format();
  };
  /// @brief Returns the default format string for printf (e.g. "%d " for
  /// SIMDInt)
  static SIMD_INLINE const char *defaultFormat()
  {
    return internal::types::SIMDTypeInfo<T>::defaultFormat();
  };
  /// @brief Returns the hex format string for printf (e.g. "%08x " for SIMDInt)
  static SIMD_INLINE const char *hexFormat()
  {
    return internal::types::SIMDTypeInfo<T>::hexFormat();
  }
  /// @brief Whether the type is signed
  static constexpr bool isSigned = internal::types::SIMDTypeInfo<T>::isSigned;
  /// @brief Whether the type is an integer
  static constexpr bool isInteger = internal::types::SIMDTypeInfo<T>::isInteger;
  /// @brief Whether the type is a floating point type
  static constexpr bool isFloatingPoint =
    internal::types::SIMDTypeInfo<T>::isFloatingPoint;
  /// @brief Returns the minimum value of the type
  static SIMD_INLINE T min() { return internal::types::SIMDTypeInfo<T>::min(); }
  /// @brief Returns the maximum value of the type
  static SIMD_INLINE T max() { return internal::types::SIMDTypeInfo<T>::max(); }
  /// @brief Returns a value where all bits are 1
  static SIMD_INLINE T trueval()
  {
    return internal::types::SIMDTypeInfo<T>::trueval();
  }
  /// @brief The next larger type (e.g. SIMDWord for SIMDByte), or the same
  /// type if there is no larger type
  typedef
    typename internal::types::SIMDTypeInfo<T>::NextLargerType NextLargerType;
  /// @brief The unsigned type (e.g. SIMDByte for SIMDSignedByte), or the same
  /// type if there is no unsigned type
  typedef typename internal::types::SIMDTypeInfo<T>::UnsignedType UnsignedType;
  /// @brief The signed type (e.g. SIMDSignedByte for SIMDByte), or the same
  /// type if there is no signed type
  typedef typename internal::types::SIMDTypeInfo<T>::SignedType SignedType;
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
    if (fieldWidth >= 0)
      sprintf(fieldWidthStr, "%d", fieldWidth);
    else
      strcpy(fieldWidthStr, "");
    if (precision >= 0)
      sprintf(precisionStr, ".%d", precision);
    else
      strcpy(precisionStr, "");
    if (SIMDTypeInfo<T>::isInteger)
      // integer format, precision is ignored
      sprintf(format, "%%%s%s", fieldWidthStr, SIMDTypeInfo<T>::format());
    else
      // float format, precision is used
      sprintf(format, "%%%s%s%s", fieldWidthStr, precisionStr,
              SIMDTypeInfo<T>::format());
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

/** @} */ // end of group types

namespace internal {
// ===========================================================================
// tag dispatching
// ===========================================================================

// int2type trick from
// Andrei Alexandrescu: Modern C++ Design (Addison Wesley)

template <bool isFloatingPoint>
struct IsFloatingPoint
{};

template <typename T>
struct TypeIsFloatingPoint
  : public IsFloatingPoint<SIMDTypeInfo<T>::isFloatingPoint>
{};

template <bool IS_INT, int SIZE>
struct IsIntSize
{};

template <typename T>
struct TypeIsIntSize : public IsIntSize<SIMDTypeInfo<T>::isInteger, sizeof(T)>
{};

template <bool IS_INT, bool IS_SIGNED>
struct IsIntIsSigned
{};

template <typename T>
struct TypeIsIntIsSigned
  : public IsIntIsSigned<SIMDTypeInfo<T>::isInteger, SIMDTypeInfo<T>::isSigned>
{};

template <int N>
struct Integer
{};

template <int N>
struct Part
{};

template <int N>
struct Elements
{};

template <int N>
struct Bytes
{};

template <bool IS_POSITIVE, bool AT_LOWER_LIMIT, int LOWER_LIMIT_INCLUSIVE,
          int UPPER_LIMIT_EXCLUSIVE>
struct Range
{};

// 24. Nov 17 (rm): now includes check for IMM >= 0 to reveal
// negative template parameters
template <int IMM, int SIZE>
struct SizeRange
  : public Range<(IMM >= 0), SIMD_IS_MULTIPLE_OF(IMM, SIZE),
                 SIMD_PREV_MULTIPLE_OF(IMM, SIZE),
                 SIMD_NEXT_MULTIPLE_OF(IMM, SIZE)>
{};

template <int N>
struct Compression
{};

template <int N>
struct Expansion
{};

template <typename T>
struct InputType
{};

template <typename T>
struct OutputType
{};

// is positive and in a range
template <bool pos, bool inRange>
struct IsPosInRange
{};

// is positive and in given range [0, RANGE)
template <int RANGE, int INDEX>
struct IsPosInGivenRange : public IsPosInRange<(INDEX >= 0), (INDEX < RANGE)>
{};

// is positive and non-zero and in a range
template <bool pos, bool nonZero, bool inRange>
struct IsPosNonZeroInRange
{};

// is positive and non-zero and in a given range
template <int RANGE, int INDEX>
struct IsPosNonZeroInGivenRange
  : public IsPosNonZeroInRange<(INDEX >= 0), (INDEX != 0), (INDEX < RANGE)>
{};

// is positive and in bit range of given type
template <typename T, int BIT>
struct IsPosInBitRange : public IsPosInGivenRange<(sizeof(T) * 8), BIT>
{};

// 12. Jan 23 (Jonas Keller): move CompareTypes to here from Compare2Type.H

// int2type trick from
// Andrei Alexandrescu: Modern C++ Design (Addison Wesley),
// adapted for type comparisons

// empty struct
// each unique combination of less, equal, greater declares a unique type
template <bool less, bool equal, bool greater>
struct Compare
{};

// structs derived from Compare for the special case of sizeof comparisons

// use with CompareLess, CompareEqual, CompareGreater
template <typename Tout, typename Tin>
struct CompareTypes
  : Compare<(sizeof(Tout) < sizeof(Tin)), (sizeof(Tout) == sizeof(Tin)),
            (sizeof(Tout) > sizeof(Tin))>
{};

// use with CompareLessOrEqual, CompareGreater
template <typename Tout, typename Tin>
struct CompareTypesLessOrEqual
  : Compare<(sizeof(Tout) <= sizeof(Tin)), (sizeof(Tout) <= sizeof(Tin)),
            (sizeof(Tout) > sizeof(Tin))>
{};

// use with CompareLess, CompareGreaterOrEqual
template <typename Tout, typename Tin>
struct CompareTypesGreaterOrEqual
  : Compare<(sizeof(Tout) < sizeof(Tin)), (sizeof(Tout) >= sizeof(Tin)),
            (sizeof(Tout) >= sizeof(Tin))>
{};

typedef Compare<true, false, false> CompareLess;
typedef Compare<true, true, false> CompareLessOrEqual;
typedef Compare<false, true, false> CompareEqual;
typedef Compare<false, true, true> CompareGreaterOrEqual;
typedef Compare<false, false, true> CompareGreater;
} // namespace internal

} // namespace ns_simd

#endif
