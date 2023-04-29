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

// 02. Mar 23 (Jonas Keller): added doxygen documentation

#ifndef SIMD_TYPES_H_
#define SIMD_TYPES_H_

#include "SIMDDefs.H"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <type_traits>

namespace simd {

// ===========================================================================
// data types
// ===========================================================================

// these types indicate that data is processed by SIMD vector instructions;
// constants are copied from stdint.h so that we don't have to fiddle
// with __STDC_LIMIT_MACROS

/**
 * @addtogroup group_element_types
 * @{
 */
using Byte       = uint8_t;  ///< Unsigned 8-bit integer
using SignedByte = int8_t;   ///< Signed 8-bit integer
using Word       = uint16_t; ///< Unsigned 16-bit integer
using Short      = int16_t;  ///< Signed 16-bit integer
using Int        = int32_t;  ///< Signed 32-bit integer
using Float      = float; ///< Single-precision floating point number (32-bit)
/** @} */

/// @cond
// 30. Mar 23 (Jonas Keller): replaced the magic numbers in these macros with
// std::numeric_limits<T>::max() and std::numeric_limits<T>::lowest()
// and also added SIMDFLOAT_TRUE using an immediately invoked lambda and
// std::memcpy

#define SIMDBYTE_TRUE       (::simd::Byte(~0))
#define SIMDSIGNEDBYTE_TRUE (::simd::SignedByte(~0))
#define SIMDWORD_TRUE       (::simd::Word(~0))
#define SIMDSHORT_TRUE      (::simd::Short(~0))
#define SIMDINT_TRUE        (::simd::Int(~0))
// use an immediately invoked lambda to be able to use memcpy
#define SIMDFLOAT_TRUE                                                         \
  ([]() -> ::simd::Float {                                                     \
    static_assert(sizeof(::simd::Float) == sizeof(::simd::Int),                \
                  "Float and Int must have the same size");                    \
    ::simd::Int i = ~(::simd::Int(0));                                         \
    ::simd::Float f;                                                           \
    ::std::memcpy(&f, &i, sizeof(::simd::Float));                              \
    return f;                                                                  \
  }())

#define SIMDBYTE_MIN       (::std::numeric_limits<Byte>::lowest())
#define SIMDSIGNEDBYTE_MIN (::std::numeric_limits<SignedByte>::lowest())
#define SIMDWORD_MIN       (::std::numeric_limits<Word>::lowest())
#define SIMDSHORT_MIN      (::std::numeric_limits<Short>::lowest())
#define SIMDINT_MIN        (::std::numeric_limits<Int>::lowest())
#define SIMDFLOAT_MIN      (::std::numeric_limits<Float>::lowest())

#define SIMDBYTE_MAX       (::std::numeric_limits<Byte>::max())
#define SIMDSIGNEDBYTE_MAX (::std::numeric_limits<SignedByte>::max())
#define SIMDWORD_MAX       (::std::numeric_limits<Word>::max())
#define SIMDSHORT_MAX      (::std::numeric_limits<Short>::max())
#define SIMDINT_MAX        (::std::numeric_limits<Int>::max())
#define SIMDFLOAT_MAX      (::std::numeric_limits<Float>::max())
// 0x7fffff80
#define MAX_POS_FLOAT_CONVERTIBLE_TO_INT32 2147483520.0f
/// @endcond

// 28. Feb 23 (Jonas Keller): added SortSlope enum

/**
 * @ingroup group_types
 * @brief Used to indicate the direction of a sort function.
 */
enum class SortSlope { ASCENDING = 0, DESCENDING = 1 };

#if DOXYGEN
/**
 * @brief Compile-time assertion.
 *
 * This macro can be used to check a condition at compile time.
 *
 * @param EXPR condition to check
 */
#define SIMD_STATIC_ASSERT(EXPR)

#else // DOXYGEN

#if __cplusplus >= 201703L // at least C++17
#define SIMD_STATIC_ASSERT(EXPR) static_assert(EXPR)
#else // older C++
#define SIMD_STATIC_ASSERT(EXPR) static_assert(EXPR, "")
#endif

#endif // DOXYGEN

// ===========================================================================
// TypeInfo
// ===========================================================================

// NOTE: min() and max() are functions since "floating-point literals are not
// allowed in constant expressions according to -pedantic

// defaultFormat() returns a full format specifier with % and space, can be
// used for tests

/**
 * @addtogroup group_element_types
 * @{
 */

// 27. Jan 23 (Jonas Keller): added wrapper class for TypeInfo for doxygen
// documentation
// 30. Mar 23 (Jonas Keller): deprecated isSigned, isInteger, isFloatingPoint,
// min() and max() in TypeInfo in favor of the corresponding functions in
// the std library and used those functions in the implementation of
// TypeInfo

namespace internal {
namespace types {
template <typename T>
struct TypeInfo;

template <>
struct TypeInfo<Byte>
{
  static constexpr SIMD_INLINE const char *name() { return "Byte"; }
  static constexpr SIMD_INLINE const char *format() { return "u"; }
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%u "; }
  static constexpr SIMD_INLINE const char *hexFormat() { return "%02x "; }
  static constexpr SIMD_INLINE Byte trueval() { return SIMDBYTE_TRUE; }
  using NextLargerType = Word;
  using UnsignedType   = Byte;
  using SignedType     = SignedByte;
};

template <>
struct TypeInfo<SignedByte>
{
  static constexpr SIMD_INLINE const char *name() { return "SignedByte"; }
  static constexpr SIMD_INLINE const char *format() { return "d"; }
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%d "; }
  static constexpr SIMD_INLINE const char *hexFormat() { return "%02x "; }
  static constexpr SIMD_INLINE SignedByte trueval()
  {
    return SIMDSIGNEDBYTE_TRUE;
  }
  using NextLargerType = Short;
  using UnsignedType   = Byte;
  using SignedType     = SignedByte;
};

template <>
struct TypeInfo<Word>
{
  static constexpr SIMD_INLINE const char *name() { return "Word"; }
  static constexpr SIMD_INLINE const char *format() { return "u"; }
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%u "; }
  static constexpr SIMD_INLINE const char *hexFormat() { return "%04x "; }
  static constexpr SIMD_INLINE Word trueval() { return SIMDWORD_TRUE; }
  using NextLargerType = Int; // no larger unsigned type, use Int
  using UnsignedType   = Word;
  using SignedType     = Short;
};

template <>
struct TypeInfo<Short>
{
  static constexpr SIMD_INLINE const char *name() { return "Short"; }
  static constexpr SIMD_INLINE const char *format() { return "d"; }
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%d "; }
  static constexpr SIMD_INLINE const char *hexFormat() { return "%04x "; }
  static constexpr SIMD_INLINE Short trueval() { return SIMDSHORT_TRUE; }
  using NextLargerType = Int;
  using UnsignedType   = Word;
  using SignedType     = Short;
};

template <>
struct TypeInfo<Int>
{
  static constexpr SIMD_INLINE const char *name() { return "Int"; }
  static constexpr SIMD_INLINE const char *format() { return "d"; }
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%d "; }
  static constexpr SIMD_INLINE const char *hexFormat() { return "%08x "; }
  static constexpr SIMD_INLINE Int trueval() { return SIMDINT_TRUE; }
  using NextLargerType = Int;      // no larger integer type than Int
  using UnsignedType   = uint32_t; // not a SIMD type
  using SignedType     = Int;
};

template <>
struct TypeInfo<Float>
{
  static constexpr SIMD_INLINE const char *name() { return "Float"; }
  static constexpr SIMD_INLINE const char *format() { return ".9g"; }
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%.9g "; }
  static SIMD_INLINE Float trueval() { return SIMDFLOAT_TRUE; }
  using NextLargerType = Float; // no larger float type than Float
  using UnsignedType   = Float; // no unsigned float type
  using SignedType     = Float;
};
} // namespace types
} // namespace internal

/**
 * @brief Type information for SIMD types
 */
template <typename T>
struct TypeInfo
{
  /// @brief Returns the name of the type (e.g. "Int" for Int)
  static constexpr SIMD_INLINE const char *name()
  {
    return internal::types::TypeInfo<T>::name();
  };
  /// @brief Returns the format string for printf (e.g. "d" for Int)
  static constexpr SIMD_INLINE const char *format()
  {
    return internal::types::TypeInfo<T>::format();
  };
  /// @brief Returns the default format string for printf (e.g. "%d " for
  /// Int)
  static constexpr SIMD_INLINE const char *defaultFormat()
  {
    return internal::types::TypeInfo<T>::defaultFormat();
  };
  /// @brief Returns the hex format string for printf (e.g. "%08x " for Int)
  static constexpr SIMD_INLINE const char *hexFormat()
  {
    return internal::types::TypeInfo<T>::hexFormat();
  }
  /**
   * @brief Whether the type is signed
   * @deprecated Use std::is_signed<T>::value or (if C++17 is available)
   * std::is_signed_v<T> instead
   */
  static constexpr bool isSigned = std::is_signed<T>::value;
  /**
   * @brief Whether the type is an integer
   * @deprecated Use std::is_integral<T>::value or (if C++17 is available)
   * std::is_integral_v<T> instead
   */
  static constexpr bool isInteger = std::is_integral<T>::value;
  /**
   * @brief Whether the type is a floating point type
   * @deprecated Use std::is_floating_point<T>::value or (if C++17 is
   * available) std::is_floating_point_v<T> instead
   */
  static constexpr bool isFloatingPoint = std::is_floating_point<T>::value;
  /**
   * @brief Returns the minimum value of the type
   * @deprecated Use std::numeric_limits<T>::lowest() instead
   */
  static constexpr SIMD_INLINE T min()
  {
    return std::numeric_limits<T>::lowest();
  }
  /**
   * @brief Returns the maximum value of the type
   * @deprecated Use std::numeric_limits<T>::max() instead
   */
  static constexpr SIMD_INLINE T max() { return std::numeric_limits<T>::max(); }
  /// @brief Returns a value where all bits are 1
  static constexpr SIMD_INLINE T trueval()
  {
    return internal::types::TypeInfo<T>::trueval();
  }
  /// @brief The next larger type (e.g. Word for Byte), or the same
  /// type if there is no larger type
  using NextLargerType = typename internal::types::TypeInfo<T>::NextLargerType;
  /// @brief The unsigned type (e.g. Byte for SignedByte), or the same
  /// type if there is no unsigned type
  using UnsignedType = typename internal::types::TypeInfo<T>::UnsignedType;
  /// @brief The signed type (e.g. SignedByte for Byte), or the same
  /// type if there is no signed type
  using SignedType = typename internal::types::TypeInfo<T>::SignedType;
};

// ===========================================================================
// formatting
// ===========================================================================

// note that for T=Float, TypeInfo<T>::format() returns "g",
// for which precision encodes the number of significant digits,
// not the number of fractional digits
/**
 * @brief Class for generating format strings for printf for SIMD types
 *
 * @tparam T type for which to generate a format string
 */
template <typename T>
struct Format
{
  char format[256]; ///< The generated format string

  /**
   * @brief Constructor
   *
   * @param fieldWidth field width (e.g. 4 for "%4d")
   * @param precision precision (e.g. 2 for "%.2f")
   */
  Format(int fieldWidth = -1, int precision = -1)
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
    if (TypeInfo<T>::isInteger)
      // integer format, precision is ignored
      sprintf(format, "%%%s%s", fieldWidthStr, TypeInfo<T>::format());
    else
      // float format, precision is used
      sprintf(format, "%%%s%s%s", fieldWidthStr, precisionStr,
              TypeInfo<T>::format());
  }
};

// a crude way to format SIMD* types as decimal number, can be used in
// fprintf (use %s format specification)
/**
 * @brief Class for formatting SIMD types as decimal numbers
 *
 * @tparam T type to format
 */
template <typename T>
struct Decimal
{
  char str[256]; ///< The formatted string

  /**
   * @brief Constructor
   *
   * @param value value to format
   * @param fieldWidth field width (e.g. 4 for "%4d")
   * @param precision precision (e.g. 2 for "%.2f")
   */
  Decimal(T value, int fieldWidth = -1, int precision = -1)
  {
    sprintf(str, Format<T>(fieldWidth, precision).format, value);
  }
};

/** @} */

// 22. Jan 23 (Jonas Keller): moved tag dispatching classes into internal
// namespace

namespace internal {
// ===========================================================================
// tag dispatching
// ===========================================================================

// int2type trick from
// Andrei Alexandrescu: Modern C++ Design (Addison Wesley)

// 08. Apr 23 (Jonas Keller): removed IsIntSize; it is no longer used

template <bool isFloatingPoint>
struct IsFloatingPoint
{};

template <typename T>
struct TypeIsFloatingPoint
  : public IsFloatingPoint<TypeInfo<T>::isFloatingPoint>
{};

template <bool IS_INT, bool IS_SIGNED>
struct IsIntIsSigned
{};

template <typename T>
struct TypeIsIntIsSigned
  : public IsIntIsSigned<TypeInfo<T>::isInteger, TypeInfo<T>::isSigned>
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

using CompareLess           = Compare<true, false, false>;
using CompareLessOrEqual    = Compare<true, true, false>;
using CompareEqual          = Compare<false, true, false>;
using CompareGreaterOrEqual = Compare<false, true, true>;
using CompareGreater        = Compare<false, false, true>;
} // namespace internal

} // namespace simd

#endif
