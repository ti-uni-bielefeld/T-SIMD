// ===========================================================================
//
// SIMDVecAll.H --
// common include file for SIMD*.H and SIMDVec*.H
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
 * @file SIMDVecAll.H
 * @brief Main file to include for using the SIMD library when C++11 is not
 * available.
 *
 * Use tsimd.H instead if C++11 is available.
 */

#ifndef _SIMD_VEC_ALL_H_
#define _SIMD_VEC_ALL_H_

// base-level templates/functions
#include "SIMDVecBase.H"
// generic templates for extension commands
#include "SIMDVecExt.H"
// SIMDVecs: multiple SIMDVec in template class
#include "SIMDVecs.H"
// templates/functions for masked operations
#include "SIMDVecMask.H"
// aligned memory allocation
#include "SIMDAlloc.H"

#endif
