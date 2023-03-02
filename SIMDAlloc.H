// ===========================================================================
//
// SIMDAlloc.H --
// allocation code
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

/**
 * @file SIMDAlloc.H
 * @brief Aligned memory allocation and deallocation.
 *
 * This file contains functions for aligned memory allocation and deallocation.
 *
 * This file is standalone, i.e. it can also be used independently of T-SIMD.
 *
 * @author Ralf Möller
 * @author Jonas Keller
 */

#ifndef _SIMDALLOC_H_
#define _SIMDALLOC_H_

// 30. Aug 22 (Jonas Keller): added simd_aligned_malloc and simd_aligned_free

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

/**
 * @brief Aligned memory allocation.
 *
 * This function allocates a block of memory of size bytes, aligned to the
 * specified alignment.
 *
 * The allocated memory must be freed with simd_aligned_free().
 *
 * @param alignment alignment of the memory block in bytes
 * @param size size of the memory block in bytes
 * @return pointer to the allocated memory block
 */
void *simd_aligned_malloc(size_t alignment, size_t size)
{
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void *ptr = NULL;
  if (posix_memalign(&ptr, alignment, size) != 0) { return NULL; }
  return ptr;
#endif
}

/**
 * @brief Aligned memory deallocation.
 *
 * This function frees a block of memory that was allocated with
 * simd_aligned_malloc().
 *
 * @param ptr pointer to the memory block to be freed
 */
void simd_aligned_free(void *ptr)
{
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

#endif // _SIMDALLOC_H_
