// ===========================================================================
//
// Main include file for using the T-SIMD library.
//
// This source code file is part of the following software:
//
//    - the low-level C++ template SIMD library
//    - the SIMD implementation of the MinWarping and the 2D-Warping methods
//      for local visual homing.
//
// The software is provided based on the accompanying license agreement in the
// file LICENSE.md.
// The software is provided "as is" without any warranty by the licensor and
// without any liability of the licensor, and the software may not be
// distributed by the licensee; see the license agreement for details.
//
// (C) Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

/**
 * @file tsimd.H
 * @brief Main include file for using the T-SIMD library.
 */

#pragma once
#ifndef T_SIMD_H_
#define T_SIMD_H_

// base-level templates/functions
#include "tsimd/base.H"
// generic templates for extension commands
#include "tsimd/ext.H"
// Vecs: multiple Vec in template class
#include "tsimd/vecs.H"
// templates/functions for masked operations
#include "tsimd/mask.H"
// aligned memory allocation
#include "tsimd/alloc.H"
// backward compatibility
#include "tsimd/backward_compat.H"

#endif
