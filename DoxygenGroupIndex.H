// ===========================================================================
//
// DoxygenGroupIndex.H --
// File to define group structure and order for doxygen documentation of the
// T-SIMD library.
// This has to be the first file in the "INPUT" field in the file "Doxyfile",
// so that the groups appear in the order defined here.
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
// (C) Jonas Keller, Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

/**
 * @defgroup group_types SIMD Types and Classes
 * @brief Types and classes for SIMD programming.
 * @{
 *
 * @defgroup group_element_types Element Types
 * @brief Types for the elements of Vec's.
 *
 * @}
 *
 * @defgroup group_init Initialization
 * @brief Functions for initializing Vec's.
 *
 * @defgroup group_memory Memory
 * @brief Functions related to memory.
 * @sa @ref group_mask_memory
 * @{
 *
 * @defgroup group_memory_load Load
 * @brief Functions for loading Vec's from memory.
 * @sa @ref group_mask_memory
 *
 * @defgroup group_memory_store Store
 * @brief Functions for storing Vec's to memory.
 * @sa @ref group_mask_memory
 *
 * @}
 *
 * @defgroup group_arithmetic Arithmetic
 * @brief Arithmetic functions on Vec's, such as addition, multiplication,
 * etc.
 * @sa @ref group_mask_arithmetic
 *
 * @defgroup group_math_operations Mathematical Operations
 * @brief Mathematical operations on Vec's, such as min, max, rounding,
 * etc.
 * @sa @ref group_mask_math_operations
 *
 * @defgroup group_math_functions Mathematical Functions
 * @brief Mathematical functions on Vec's, such as sqrt, rcp,
 * etc.
 * @sa @ref group_mask_math_functions
 *
 * @defgroup group_horizontal Horizontal Functions
 * @brief Horizontal Functions
 * @sa @ref group_mask_horizontal
 * @{
 *
 * @defgroup group_iter_hor_acc Iterative Horizontal Accumulation
 * @brief Classes for iterative horizontal accumulation.
 *
 * @}
 *
 * @defgroup group_logic Logic
 * @brief Logical functions on Vec's.
 * @sa @ref group_mask_logic
 *
 * @defgroup group_shift Bitwise Shifts
 * @brief Bitwise shift functions on Vec's.
 * @sa @ref group_mask_shift
 *
 * @defgroup group_cmp Comparisons
 * @brief Functions for comparing elements of Vec's.
 * @sa @ref group_mask_cmp
 *
 * @defgroup group_mask Masked Functions
 * @brief Masked versions of functions.
 *
 * Most functions have two masked versions:
 * - A masked version that takes a source Vec and a Mask as the first two
 * arguments, followed by the remaining arguments.<br> These functions will
 * replace the elements of the result of the normal version of the function with
 * the elements of the source Vec, where the Mask is false.<br> These functions
 * are prefixed with "mask_".
 * - A zero-masked version that takes a Mask as the first argument, followed by
 * the remaining arguments.<br> These functions will replace the elements of the
 * result of the normal version of the function with zero, where the Mask is
 * false.<br> These functions are prefixed with "maskz_".
 * @{
 *
 * @defgroup group_mask_functions Pure Mask Functions
 * @brief Functions that operate on masks themselves.
 *
 * @defgroup group_mask_memory Masked Memory
 * @brief Functions for masked loading and storing Vec's from and to
 * memory.
 * @sa @ref group_memory
 *
 * @defgroup group_mask_arithmetic Masked Arithmetic
 * @brief Masked arithmetic functions on Vec's, like addition,
 * multiplication, etc.
 * @sa @ref group_arithmetic
 *
 * @defgroup group_mask_math_operations Masked Mathematical Operations
 * @brief Masked mathematical operations on Vec's, such as min, max,
 * rounding, etc.
 * @sa @ref group_math_operations
 *
 * @defgroup group_mask_math_functions Masked Mathematical Functions
 * @brief Masked mathematical functions on Vec's, such as sqrt,
 * rcp, etc.
 * @sa @ref group_math_functions
 *
 * @defgroup group_mask_horizontal Masked Horizontal Functions
 * @brief Masked horizontal functions on Vec's.
 * @sa @ref group_horizontal
 *
 * @defgroup group_mask_logic Masked Logic
 * @brief Masked logical functions on Vec's.
 * @sa @ref group_logic
 *
 * @defgroup group_mask_shift Masked Bitwise Shifts
 * @brief Masked bitwise shift functions on Vec's.
 * @sa @ref group_shift
 *
 * @defgroup group_mask_cmp Masked Comparisons
 * @brief Functions for masked comparing elements of Vec's.
 * @sa @ref group_cmp
 *
 * @defgroup group_other_mask Other Masked Functions
 * @brief Other masked functions.
 *
 * @}
 *
 * @defgroup group_reordering Reordering
 * @brief Functions on Vec's for reordering elements and other related
 * functions.
 * @{
 *
 * @defgroup group_swizzle Swizzling
 * @brief Swizzle/deinterleave functions on Vec's.
 *
 * @defgroup group_transpose Transpose
 * @brief Functions for transposing a matrix held in an array of Vec's.
 *
 * @defgroup group_elementwise_shift Elementwise Shifting
 * @brief Functions for shifting a Vec elementwise.
 *
 * @defgroup group_extract Extraction
 * @brief Functions for extracting elements or lanes from Vec's.
 *
 * @defgroup group_zip_unpack Zip and Unpack
 * @brief Zip, unzip and unpack functions.
 *
 * @}
 *
 * @defgroup group_type_conversion Type Conversions
 * @brief Functions for converting Vec's between different types.
 *
 * @defgroup group_fops Floating point based Operations
 * @brief Operations that are based on floating point arithmetic, but can be
 * applied to arbitrary input and output types.
 *
 * These functions take an input of arbitrary type, convert it to a floating
 * point type, perform an operation in floating point arithmetic, and convert
 * the result to an output of arbitrary type.
 *
 * Similar to convert(), if the input and output types are of the same size,
 * both the inputs and outputs consist of one Vec. If the types are of
 * different sizes, the input or output consists of multiple Vec's, such that
 * the number of input elements is equal to the number of output elements.
 *
 * @defgroup group_operators Operators
 * @brief Operators for Vec's
 *
 * @defgroup group_simd_sort Sorting
 * @brief Functions for sorting Vec's.
 *
 * @defgroup group_print Print
 * @brief Functions to print the elements of a Vec.
 *
 * @defgroup group_time_measurements Time Measurement
 * @brief Functions for measuring time.
 *
 * These may be useful to measure the time spent for a section of code.
 *
 * Example:
 * @code
 * struct timespec start = getTimeSpecMonotonic();
 * // do something
 * struct timespec end = getTimeSpecMonotonic();
 * long int diff_ns = diffTimeSpecNsec(end, start);
 * std::cout << "Time spent: " << diff_ns << "ns" << std::endl;
 * @endcode
 *
 * @defgroup group_aliases Back. Comp. Aliases
 * @brief Aliases for SIMD types for backward compatibility
 */
