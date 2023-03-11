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

#include <stdint.h>
#include <stdio.h>
#include <string.h>
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
typedef uint8_t Byte;      ///< Unsigned 8-bit integer
typedef int8_t SignedByte; ///< Signed 8-bit integer
typedef uint16_t Word;     ///< Unsigned 16-bit integer
typedef int16_t Short;     ///< Signed 16-bit integer
typedef int32_t Int;       ///< Signed 32-bit integer
typedef float Float;       ///< Single-precision floating point number (32-bit)
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

namespace internal {
namespace types {
template <typename T>
class TypeInfo;

template <>
class TypeInfo<Byte>
{
public:
  static constexpr SIMD_INLINE const char *name() { return "Byte"; };
  static constexpr SIMD_INLINE const char *format() { return "u"; };
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%u "; };
  static constexpr SIMD_INLINE const char *hexFormat() { return "%02x "; }
  static constexpr bool isSigned        = false;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static constexpr SIMD_INLINE Byte min() { return 0; }
  static constexpr SIMD_INLINE Byte max() { return SIMDBYTE_MAX; }
  static constexpr SIMD_INLINE Byte trueval() { return SIMDBYTE_TRUE; }
  typedef Word NextLargerType;
  typedef Byte UnsignedType;
  typedef SignedByte SignedType;
};

template <>
class TypeInfo<SignedByte>
{
public:
  static constexpr SIMD_INLINE const char *name() { return "SignedByte"; };
  static constexpr SIMD_INLINE const char *format() { return "d"; };
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%d "; };
  static constexpr SIMD_INLINE const char *hexFormat() { return "%02x "; }
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static constexpr SIMD_INLINE SignedByte min() { return SIMDSIGNEDBYTE_MIN; }
  static constexpr SIMD_INLINE SignedByte max() { return SIMDSIGNEDBYTE_MAX; }
  static constexpr SIMD_INLINE SignedByte trueval()
  {
    return SIMDSIGNEDBYTE_TRUE;
  }
  typedef Short NextLargerType;
  typedef Byte UnsignedType;
  typedef SignedByte SignedType;
};

template <>
class TypeInfo<Word>
{
public:
  static constexpr SIMD_INLINE const char *name() { return "Word"; };
  static constexpr SIMD_INLINE const char *format() { return "u"; };
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%u "; };
  static constexpr SIMD_INLINE const char *hexFormat() { return "%04x "; }
  static constexpr bool isSigned        = false;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static constexpr SIMD_INLINE Word min() { return 0; }
  static constexpr SIMD_INLINE Word max() { return SIMDWORD_MAX; }
  static constexpr SIMD_INLINE Word trueval() { return SIMDWORD_TRUE; }
  typedef Int NextLargerType; // no larger unsigned type, use Int
  typedef Word UnsignedType;
  typedef Short SignedType;
};

template <>
class TypeInfo<Short>
{
public:
  static constexpr SIMD_INLINE const char *name() { return "Short"; };
  static constexpr SIMD_INLINE const char *format() { return "d"; };
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%d "; };
  static constexpr SIMD_INLINE const char *hexFormat() { return "%04x "; }
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static constexpr SIMD_INLINE Short min() { return SIMDSHORT_MIN; }
  static constexpr SIMD_INLINE Short max() { return SIMDSHORT_MAX; }
  static constexpr SIMD_INLINE Short trueval() { return SIMDSHORT_TRUE; }
  typedef Int NextLargerType;
  typedef Word UnsignedType;
  typedef Short SignedType;
};

template <>
class TypeInfo<Int>
{
public:
  static constexpr SIMD_INLINE const char *name() { return "Int"; };
  static constexpr SIMD_INLINE const char *format() { return "d"; };
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%d "; };
  static constexpr SIMD_INLINE const char *hexFormat() { return "%08x "; }
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = true;
  static constexpr bool isFloatingPoint = false;
  static constexpr SIMD_INLINE Int min() { return SIMDINT_MIN; }
  static constexpr SIMD_INLINE Int max() { return SIMDINT_MAX; }
  static constexpr SIMD_INLINE Int trueval() { return SIMDINT_TRUE; }
  typedef Int NextLargerType;    // no larger integer type than Int
  typedef uint32_t UnsignedType; // not a SIMD type
  typedef Int SignedType;
};

template <>
class TypeInfo<Float>
{
public:
  static constexpr SIMD_INLINE const char *name() { return "Float"; };
  static constexpr SIMD_INLINE const char *format() { return "g"; };
  static constexpr SIMD_INLINE const char *defaultFormat() { return "%g "; };
  static constexpr bool isSigned        = true;
  static constexpr bool isInteger       = false;
  static constexpr bool isFloatingPoint = true;
  static constexpr const char *test     = "st";
  static constexpr SIMD_INLINE Float min() { return SIMDFLOAT_MIN; }
  static constexpr SIMD_INLINE Float max() { return SIMDFLOAT_MAX; }
  // very inconvenient: how can I specify a float constant with all bits 1?
  static SIMD_INLINE Float trueval()
  {
    Int i = SIMDINT_TRUE;
    Float f;
    memcpy(&f, &i, sizeof(f));
    return f;
  }
  typedef Float NextLargerType; // no larger float type than Float
  typedef Float UnsignedType;   // no unsigned float type
  typedef Float SignedType;
};
} // namespace types
} // namespace internal

/**
 * @brief Type information for SIMD types
 */
template <typename T>
class TypeInfo
{
public:
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
  /// @brief Whether the type is signed
  static constexpr bool isSigned = internal::types::TypeInfo<T>::isSigned;
  /// @brief Whether the type is an integer
  static constexpr bool isInteger = internal::types::TypeInfo<T>::isInteger;
  /// @brief Whether the type is a floating point type
  static constexpr bool isFloatingPoint =
    internal::types::TypeInfo<T>::isFloatingPoint;
  /// @brief Returns the minimum value of the type
  static constexpr SIMD_INLINE T min()
  {
    return internal::types::TypeInfo<T>::min();
  }
  /// @brief Returns the maximum value of the type
  static constexpr SIMD_INLINE T max()
  {
    return internal::types::TypeInfo<T>::max();
  }
  /// @brief Returns a value where all bits are 1
  static constexpr SIMD_INLINE T trueval()
  {
    return internal::types::TypeInfo<T>::trueval();
  }
  /// @brief The next larger type (e.g. Word for Byte), or the same
  /// type if there is no larger type
  typedef typename internal::types::TypeInfo<T>::NextLargerType NextLargerType;
  /// @brief The unsigned type (e.g. Byte for SignedByte), or the same
  /// type if there is no unsigned type
  typedef typename internal::types::TypeInfo<T>::UnsignedType UnsignedType;
  /// @brief The signed type (e.g. SignedByte for Byte), or the same
  /// type if there is no signed type
  typedef typename internal::types::TypeInfo<T>::SignedType SignedType;
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
class Format
{
public:
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
class Decimal
{
public:
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

template <bool isFloatingPoint>
struct IsFloatingPoint
{};

template <typename T>
struct TypeIsFloatingPoint
  : public IsFloatingPoint<TypeInfo<T>::isFloatingPoint>
{};

template <bool IS_INT, int SIZE>
struct IsIntSize
{};

template <typename T>
struct TypeIsIntSize : public IsIntSize<TypeInfo<T>::isInteger, sizeof(T)>
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

typedef Compare<true, false, false> CompareLess;
typedef Compare<true, true, false> CompareLessOrEqual;
typedef Compare<false, true, false> CompareEqual;
typedef Compare<false, true, true> CompareGreaterOrEqual;
typedef Compare<false, false, true> CompareGreater;
} // namespace internal

} // namespace simd

#endif
