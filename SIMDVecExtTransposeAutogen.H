// ===========================================================================
//
// SIMDVecExtTransposeAutogen.H --
// auto-generated transpose functions with in-place processing
// DO NOT EDIT!!!
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

#pragma once
#ifndef SIMD_VEC_EXT_TRANSPOSE_AUTOGEN_H_
#define SIMD_VEC_EXT_TRANSPOSE_AUTOGEN_H_

#include "SIMDVecBase.H"

#include <utility>

namespace simd {
namespace internal {
namespace ext {
// ==========================================================
// transpose1inplc
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<2>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<4>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  std::swap(outRows[1], outRows[2]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  std::swap(outRows[1], outRows[4]);
  std::swap(outRows[3], outRows[6]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  std::swap(outRows[1], outRows[8]);
  std::swap(outRows[2], outRows[4]);
  std::swap(outRows[3], outRows[12]);
  std::swap(outRows[5], outRows[10]);
  std::swap(outRows[7], outRows[14]);
  std::swap(outRows[11], outRows[13]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<32>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip<1>(inRows[16], inRows[17], outRows[16], outRows[17]);
  zip<1>(inRows[18], inRows[19], outRows[18], outRows[19]);
  zip<1>(inRows[20], inRows[21], outRows[20], outRows[21]);
  zip<1>(inRows[22], inRows[23], outRows[22], outRows[23]);
  zip<1>(inRows[24], inRows[25], outRows[24], outRows[25]);
  zip<1>(inRows[26], inRows[27], outRows[26], outRows[27]);
  zip<1>(inRows[28], inRows[29], outRows[28], outRows[29]);
  zip<1>(inRows[30], inRows[31], outRows[30], outRows[31]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<4>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip<4>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip<4>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip<4>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip<4>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip<4>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip<4>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip<4>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip<8>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip<8>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip<8>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip<8>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip<8>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip<8>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip<8>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip<8>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip<16>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip<16>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip<16>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip<16>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip<16>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip<16>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip<16>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip<16>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip<16>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip<16>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip<16>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip<16>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip<16>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip<16>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip<16>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip<16>(outRows[15], outRows[31], outRows[15], outRows[31]);
  std::swap(outRows[1], outRows[16]);
  std::swap(outRows[2], outRows[8]);
  std::swap(outRows[3], outRows[24]);
  std::swap(outRows[5], outRows[20]);
  std::swap(outRows[6], outRows[12]);
  std::swap(outRows[7], outRows[28]);
  std::swap(outRows[9], outRows[18]);
  std::swap(outRows[11], outRows[26]);
  std::swap(outRows[13], outRows[22]);
  std::swap(outRows[15], outRows[30]);
  std::swap(outRows[19], outRows[25]);
  std::swap(outRows[23], outRows[29]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<64>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip<1>(inRows[16], inRows[17], outRows[16], outRows[17]);
  zip<1>(inRows[18], inRows[19], outRows[18], outRows[19]);
  zip<1>(inRows[20], inRows[21], outRows[20], outRows[21]);
  zip<1>(inRows[22], inRows[23], outRows[22], outRows[23]);
  zip<1>(inRows[24], inRows[25], outRows[24], outRows[25]);
  zip<1>(inRows[26], inRows[27], outRows[26], outRows[27]);
  zip<1>(inRows[28], inRows[29], outRows[28], outRows[29]);
  zip<1>(inRows[30], inRows[31], outRows[30], outRows[31]);
  zip<1>(inRows[32], inRows[33], outRows[32], outRows[33]);
  zip<1>(inRows[34], inRows[35], outRows[34], outRows[35]);
  zip<1>(inRows[36], inRows[37], outRows[36], outRows[37]);
  zip<1>(inRows[38], inRows[39], outRows[38], outRows[39]);
  zip<1>(inRows[40], inRows[41], outRows[40], outRows[41]);
  zip<1>(inRows[42], inRows[43], outRows[42], outRows[43]);
  zip<1>(inRows[44], inRows[45], outRows[44], outRows[45]);
  zip<1>(inRows[46], inRows[47], outRows[46], outRows[47]);
  zip<1>(inRows[48], inRows[49], outRows[48], outRows[49]);
  zip<1>(inRows[50], inRows[51], outRows[50], outRows[51]);
  zip<1>(inRows[52], inRows[53], outRows[52], outRows[53]);
  zip<1>(inRows[54], inRows[55], outRows[54], outRows[55]);
  zip<1>(inRows[56], inRows[57], outRows[56], outRows[57]);
  zip<1>(inRows[58], inRows[59], outRows[58], outRows[59]);
  zip<1>(inRows[60], inRows[61], outRows[60], outRows[61]);
  zip<1>(inRows[62], inRows[63], outRows[62], outRows[63]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip<2>(outRows[32], outRows[34], outRows[32], outRows[34]);
  zip<2>(outRows[33], outRows[35], outRows[33], outRows[35]);
  zip<2>(outRows[36], outRows[38], outRows[36], outRows[38]);
  zip<2>(outRows[37], outRows[39], outRows[37], outRows[39]);
  zip<2>(outRows[40], outRows[42], outRows[40], outRows[42]);
  zip<2>(outRows[41], outRows[43], outRows[41], outRows[43]);
  zip<2>(outRows[44], outRows[46], outRows[44], outRows[46]);
  zip<2>(outRows[45], outRows[47], outRows[45], outRows[47]);
  zip<2>(outRows[48], outRows[50], outRows[48], outRows[50]);
  zip<2>(outRows[49], outRows[51], outRows[49], outRows[51]);
  zip<2>(outRows[52], outRows[54], outRows[52], outRows[54]);
  zip<2>(outRows[53], outRows[55], outRows[53], outRows[55]);
  zip<2>(outRows[56], outRows[58], outRows[56], outRows[58]);
  zip<2>(outRows[57], outRows[59], outRows[57], outRows[59]);
  zip<2>(outRows[60], outRows[62], outRows[60], outRows[62]);
  zip<2>(outRows[61], outRows[63], outRows[61], outRows[63]);
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<4>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip<4>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip<4>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip<4>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip<4>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip<4>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip<4>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip<4>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip<4>(outRows[32], outRows[36], outRows[32], outRows[36]);
  zip<4>(outRows[34], outRows[38], outRows[34], outRows[38]);
  zip<4>(outRows[33], outRows[37], outRows[33], outRows[37]);
  zip<4>(outRows[35], outRows[39], outRows[35], outRows[39]);
  zip<4>(outRows[40], outRows[44], outRows[40], outRows[44]);
  zip<4>(outRows[42], outRows[46], outRows[42], outRows[46]);
  zip<4>(outRows[41], outRows[45], outRows[41], outRows[45]);
  zip<4>(outRows[43], outRows[47], outRows[43], outRows[47]);
  zip<4>(outRows[48], outRows[52], outRows[48], outRows[52]);
  zip<4>(outRows[50], outRows[54], outRows[50], outRows[54]);
  zip<4>(outRows[49], outRows[53], outRows[49], outRows[53]);
  zip<4>(outRows[51], outRows[55], outRows[51], outRows[55]);
  zip<4>(outRows[56], outRows[60], outRows[56], outRows[60]);
  zip<4>(outRows[58], outRows[62], outRows[58], outRows[62]);
  zip<4>(outRows[57], outRows[61], outRows[57], outRows[61]);
  zip<4>(outRows[59], outRows[63], outRows[59], outRows[63]);
  zip<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip<8>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip<8>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip<8>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip<8>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip<8>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip<8>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip<8>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip<8>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip<8>(outRows[32], outRows[40], outRows[32], outRows[40]);
  zip<8>(outRows[36], outRows[44], outRows[36], outRows[44]);
  zip<8>(outRows[34], outRows[42], outRows[34], outRows[42]);
  zip<8>(outRows[38], outRows[46], outRows[38], outRows[46]);
  zip<8>(outRows[33], outRows[41], outRows[33], outRows[41]);
  zip<8>(outRows[37], outRows[45], outRows[37], outRows[45]);
  zip<8>(outRows[35], outRows[43], outRows[35], outRows[43]);
  zip<8>(outRows[39], outRows[47], outRows[39], outRows[47]);
  zip<8>(outRows[48], outRows[56], outRows[48], outRows[56]);
  zip<8>(outRows[52], outRows[60], outRows[52], outRows[60]);
  zip<8>(outRows[50], outRows[58], outRows[50], outRows[58]);
  zip<8>(outRows[54], outRows[62], outRows[54], outRows[62]);
  zip<8>(outRows[49], outRows[57], outRows[49], outRows[57]);
  zip<8>(outRows[53], outRows[61], outRows[53], outRows[61]);
  zip<8>(outRows[51], outRows[59], outRows[51], outRows[59]);
  zip<8>(outRows[55], outRows[63], outRows[55], outRows[63]);
  zip<16>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip<16>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip<16>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip<16>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip<16>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip<16>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip<16>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip<16>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip<16>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip<16>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip<16>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip<16>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip<16>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip<16>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip<16>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip<16>(outRows[15], outRows[31], outRows[15], outRows[31]);
  zip<16>(outRows[32], outRows[48], outRows[32], outRows[48]);
  zip<16>(outRows[40], outRows[56], outRows[40], outRows[56]);
  zip<16>(outRows[36], outRows[52], outRows[36], outRows[52]);
  zip<16>(outRows[44], outRows[60], outRows[44], outRows[60]);
  zip<16>(outRows[34], outRows[50], outRows[34], outRows[50]);
  zip<16>(outRows[42], outRows[58], outRows[42], outRows[58]);
  zip<16>(outRows[38], outRows[54], outRows[38], outRows[54]);
  zip<16>(outRows[46], outRows[62], outRows[46], outRows[62]);
  zip<16>(outRows[33], outRows[49], outRows[33], outRows[49]);
  zip<16>(outRows[41], outRows[57], outRows[41], outRows[57]);
  zip<16>(outRows[37], outRows[53], outRows[37], outRows[53]);
  zip<16>(outRows[45], outRows[61], outRows[45], outRows[61]);
  zip<16>(outRows[35], outRows[51], outRows[35], outRows[51]);
  zip<16>(outRows[43], outRows[59], outRows[43], outRows[59]);
  zip<16>(outRows[39], outRows[55], outRows[39], outRows[55]);
  zip<16>(outRows[47], outRows[63], outRows[47], outRows[63]);
  zip<32>(outRows[0], outRows[32], outRows[0], outRows[32]);
  zip<32>(outRows[16], outRows[48], outRows[16], outRows[48]);
  zip<32>(outRows[8], outRows[40], outRows[8], outRows[40]);
  zip<32>(outRows[24], outRows[56], outRows[24], outRows[56]);
  zip<32>(outRows[4], outRows[36], outRows[4], outRows[36]);
  zip<32>(outRows[20], outRows[52], outRows[20], outRows[52]);
  zip<32>(outRows[12], outRows[44], outRows[12], outRows[44]);
  zip<32>(outRows[28], outRows[60], outRows[28], outRows[60]);
  zip<32>(outRows[2], outRows[34], outRows[2], outRows[34]);
  zip<32>(outRows[18], outRows[50], outRows[18], outRows[50]);
  zip<32>(outRows[10], outRows[42], outRows[10], outRows[42]);
  zip<32>(outRows[26], outRows[58], outRows[26], outRows[58]);
  zip<32>(outRows[6], outRows[38], outRows[6], outRows[38]);
  zip<32>(outRows[22], outRows[54], outRows[22], outRows[54]);
  zip<32>(outRows[14], outRows[46], outRows[14], outRows[46]);
  zip<32>(outRows[30], outRows[62], outRows[30], outRows[62]);
  zip<32>(outRows[1], outRows[33], outRows[1], outRows[33]);
  zip<32>(outRows[17], outRows[49], outRows[17], outRows[49]);
  zip<32>(outRows[9], outRows[41], outRows[9], outRows[41]);
  zip<32>(outRows[25], outRows[57], outRows[25], outRows[57]);
  zip<32>(outRows[5], outRows[37], outRows[5], outRows[37]);
  zip<32>(outRows[21], outRows[53], outRows[21], outRows[53]);
  zip<32>(outRows[13], outRows[45], outRows[13], outRows[45]);
  zip<32>(outRows[29], outRows[61], outRows[29], outRows[61]);
  zip<32>(outRows[3], outRows[35], outRows[3], outRows[35]);
  zip<32>(outRows[19], outRows[51], outRows[19], outRows[51]);
  zip<32>(outRows[11], outRows[43], outRows[11], outRows[43]);
  zip<32>(outRows[27], outRows[59], outRows[27], outRows[59]);
  zip<32>(outRows[7], outRows[39], outRows[7], outRows[39]);
  zip<32>(outRows[23], outRows[55], outRows[23], outRows[55]);
  zip<32>(outRows[15], outRows[47], outRows[15], outRows[47]);
  zip<32>(outRows[31], outRows[63], outRows[31], outRows[63]);
  std::swap(outRows[1], outRows[32]);
  std::swap(outRows[2], outRows[16]);
  std::swap(outRows[3], outRows[48]);
  std::swap(outRows[4], outRows[8]);
  std::swap(outRows[5], outRows[40]);
  std::swap(outRows[6], outRows[24]);
  std::swap(outRows[7], outRows[56]);
  std::swap(outRows[9], outRows[36]);
  std::swap(outRows[10], outRows[20]);
  std::swap(outRows[11], outRows[52]);
  std::swap(outRows[13], outRows[44]);
  std::swap(outRows[14], outRows[28]);
  std::swap(outRows[15], outRows[60]);
  std::swap(outRows[17], outRows[34]);
  std::swap(outRows[19], outRows[50]);
  std::swap(outRows[21], outRows[42]);
  std::swap(outRows[22], outRows[26]);
  std::swap(outRows[23], outRows[58]);
  std::swap(outRows[25], outRows[38]);
  std::swap(outRows[27], outRows[54]);
  std::swap(outRows[29], outRows[46]);
  std::swap(outRows[31], outRows[62]);
  std::swap(outRows[35], outRows[49]);
  std::swap(outRows[37], outRows[41]);
  std::swap(outRows[39], outRows[57]);
  std::swap(outRows[43], outRows[53]);
  std::swap(outRows[47], outRows[61]);
  std::swap(outRows[55], outRows[59]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose1inplc(inRows, outRows, Elements<Vec<T, SIMD_WIDTH>::elements>());
}

// ==========================================================
// transpose1inplcLane
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>,
  Bytes<16>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip16<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip16<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip16<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip16<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip16<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip16<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip16<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip16<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip16<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip16<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  std::swap(outRows[1], outRows[8]);
  std::swap(outRows[2], outRows[4]);
  std::swap(outRows[3], outRows[12]);
  std::swap(outRows[5], outRows[10]);
  std::swap(outRows[7], outRows[14]);
  std::swap(outRows[11], outRows[13]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<16>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  std::swap(outRows[1], outRows[4]);
  std::swap(outRows[3], outRows[6]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<16>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  std::swap(outRows[1], outRows[2]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<2>, Bytes<16>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<32>,
  Bytes<32>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip16<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip16<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip16<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip16<1>(inRows[16], inRows[17], outRows[16], outRows[17]);
  zip16<1>(inRows[18], inRows[19], outRows[18], outRows[19]);
  zip16<1>(inRows[20], inRows[21], outRows[20], outRows[21]);
  zip16<1>(inRows[22], inRows[23], outRows[22], outRows[23]);
  zip16<1>(inRows[24], inRows[25], outRows[24], outRows[25]);
  zip16<1>(inRows[26], inRows[27], outRows[26], outRows[27]);
  zip16<1>(inRows[28], inRows[29], outRows[28], outRows[29]);
  zip16<1>(inRows[30], inRows[31], outRows[30], outRows[31]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip16<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip16<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip16<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip16<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip16<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip16<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip16<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip16<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<4>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip16<4>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip16<4>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip16<4>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip16<4>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip16<4>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip16<4>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip16<4>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip16<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip16<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip16<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip16<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip16<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip16<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip16<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip16<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip16<8>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip16<8>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip16<8>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip16<8>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip16<8>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip16<8>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip16<8>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip16<8>(outRows[23], outRows[31], outRows[23], outRows[31]);
  std::swap(outRows[1], outRows[8]);
  std::swap(outRows[2], outRows[4]);
  std::swap(outRows[3], outRows[12]);
  std::swap(outRows[5], outRows[10]);
  std::swap(outRows[7], outRows[14]);
  std::swap(outRows[11], outRows[13]);
  std::swap(outRows[17], outRows[24]);
  std::swap(outRows[18], outRows[20]);
  std::swap(outRows[19], outRows[28]);
  std::swap(outRows[21], outRows[26]);
  std::swap(outRows[23], outRows[30]);
  std::swap(outRows[27], outRows[29]);
  // correction steps follow below (if required)
  zip<16>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip<16>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip<16>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip<16>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip<16>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip<16>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip<16>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip<16>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip<16>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip<16>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip<16>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip<16>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip<16>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip<16>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip<16>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip<16>(outRows[15], outRows[31], outRows[15], outRows[31]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>,
  Bytes<32>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip16<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip16<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip16<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  std::swap(outRows[1], outRows[4]);
  std::swap(outRows[3], outRows[6]);
  std::swap(outRows[9], outRows[12]);
  std::swap(outRows[11], outRows[14]);
  // correction steps follow below (if required)
  zip<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<32>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  std::swap(outRows[1], outRows[2]);
  std::swap(outRows[5], outRows[6]);
  // correction steps follow below (if required)
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<32>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  // correction steps follow below (if required)
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<64>,
  Bytes<64>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip16<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip16<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip16<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip16<1>(inRows[16], inRows[17], outRows[16], outRows[17]);
  zip16<1>(inRows[18], inRows[19], outRows[18], outRows[19]);
  zip16<1>(inRows[20], inRows[21], outRows[20], outRows[21]);
  zip16<1>(inRows[22], inRows[23], outRows[22], outRows[23]);
  zip16<1>(inRows[24], inRows[25], outRows[24], outRows[25]);
  zip16<1>(inRows[26], inRows[27], outRows[26], outRows[27]);
  zip16<1>(inRows[28], inRows[29], outRows[28], outRows[29]);
  zip16<1>(inRows[30], inRows[31], outRows[30], outRows[31]);
  zip16<1>(inRows[32], inRows[33], outRows[32], outRows[33]);
  zip16<1>(inRows[34], inRows[35], outRows[34], outRows[35]);
  zip16<1>(inRows[36], inRows[37], outRows[36], outRows[37]);
  zip16<1>(inRows[38], inRows[39], outRows[38], outRows[39]);
  zip16<1>(inRows[40], inRows[41], outRows[40], outRows[41]);
  zip16<1>(inRows[42], inRows[43], outRows[42], outRows[43]);
  zip16<1>(inRows[44], inRows[45], outRows[44], outRows[45]);
  zip16<1>(inRows[46], inRows[47], outRows[46], outRows[47]);
  zip16<1>(inRows[48], inRows[49], outRows[48], outRows[49]);
  zip16<1>(inRows[50], inRows[51], outRows[50], outRows[51]);
  zip16<1>(inRows[52], inRows[53], outRows[52], outRows[53]);
  zip16<1>(inRows[54], inRows[55], outRows[54], outRows[55]);
  zip16<1>(inRows[56], inRows[57], outRows[56], outRows[57]);
  zip16<1>(inRows[58], inRows[59], outRows[58], outRows[59]);
  zip16<1>(inRows[60], inRows[61], outRows[60], outRows[61]);
  zip16<1>(inRows[62], inRows[63], outRows[62], outRows[63]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip16<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip16<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip16<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip16<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip16<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip16<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip16<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip16<2>(outRows[32], outRows[34], outRows[32], outRows[34]);
  zip16<2>(outRows[33], outRows[35], outRows[33], outRows[35]);
  zip16<2>(outRows[36], outRows[38], outRows[36], outRows[38]);
  zip16<2>(outRows[37], outRows[39], outRows[37], outRows[39]);
  zip16<2>(outRows[40], outRows[42], outRows[40], outRows[42]);
  zip16<2>(outRows[41], outRows[43], outRows[41], outRows[43]);
  zip16<2>(outRows[44], outRows[46], outRows[44], outRows[46]);
  zip16<2>(outRows[45], outRows[47], outRows[45], outRows[47]);
  zip16<2>(outRows[48], outRows[50], outRows[48], outRows[50]);
  zip16<2>(outRows[49], outRows[51], outRows[49], outRows[51]);
  zip16<2>(outRows[52], outRows[54], outRows[52], outRows[54]);
  zip16<2>(outRows[53], outRows[55], outRows[53], outRows[55]);
  zip16<2>(outRows[56], outRows[58], outRows[56], outRows[58]);
  zip16<2>(outRows[57], outRows[59], outRows[57], outRows[59]);
  zip16<2>(outRows[60], outRows[62], outRows[60], outRows[62]);
  zip16<2>(outRows[61], outRows[63], outRows[61], outRows[63]);
  zip16<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<4>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip16<4>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip16<4>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip16<4>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip16<4>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip16<4>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip16<4>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip16<4>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip16<4>(outRows[32], outRows[36], outRows[32], outRows[36]);
  zip16<4>(outRows[34], outRows[38], outRows[34], outRows[38]);
  zip16<4>(outRows[33], outRows[37], outRows[33], outRows[37]);
  zip16<4>(outRows[35], outRows[39], outRows[35], outRows[39]);
  zip16<4>(outRows[40], outRows[44], outRows[40], outRows[44]);
  zip16<4>(outRows[42], outRows[46], outRows[42], outRows[46]);
  zip16<4>(outRows[41], outRows[45], outRows[41], outRows[45]);
  zip16<4>(outRows[43], outRows[47], outRows[43], outRows[47]);
  zip16<4>(outRows[48], outRows[52], outRows[48], outRows[52]);
  zip16<4>(outRows[50], outRows[54], outRows[50], outRows[54]);
  zip16<4>(outRows[49], outRows[53], outRows[49], outRows[53]);
  zip16<4>(outRows[51], outRows[55], outRows[51], outRows[55]);
  zip16<4>(outRows[56], outRows[60], outRows[56], outRows[60]);
  zip16<4>(outRows[58], outRows[62], outRows[58], outRows[62]);
  zip16<4>(outRows[57], outRows[61], outRows[57], outRows[61]);
  zip16<4>(outRows[59], outRows[63], outRows[59], outRows[63]);
  zip16<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip16<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip16<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip16<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip16<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip16<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip16<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip16<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip16<8>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip16<8>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip16<8>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip16<8>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip16<8>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip16<8>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip16<8>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip16<8>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip16<8>(outRows[32], outRows[40], outRows[32], outRows[40]);
  zip16<8>(outRows[36], outRows[44], outRows[36], outRows[44]);
  zip16<8>(outRows[34], outRows[42], outRows[34], outRows[42]);
  zip16<8>(outRows[38], outRows[46], outRows[38], outRows[46]);
  zip16<8>(outRows[33], outRows[41], outRows[33], outRows[41]);
  zip16<8>(outRows[37], outRows[45], outRows[37], outRows[45]);
  zip16<8>(outRows[35], outRows[43], outRows[35], outRows[43]);
  zip16<8>(outRows[39], outRows[47], outRows[39], outRows[47]);
  zip16<8>(outRows[48], outRows[56], outRows[48], outRows[56]);
  zip16<8>(outRows[52], outRows[60], outRows[52], outRows[60]);
  zip16<8>(outRows[50], outRows[58], outRows[50], outRows[58]);
  zip16<8>(outRows[54], outRows[62], outRows[54], outRows[62]);
  zip16<8>(outRows[49], outRows[57], outRows[49], outRows[57]);
  zip16<8>(outRows[53], outRows[61], outRows[53], outRows[61]);
  zip16<8>(outRows[51], outRows[59], outRows[51], outRows[59]);
  zip16<8>(outRows[55], outRows[63], outRows[55], outRows[63]);
  std::swap(outRows[1], outRows[8]);
  std::swap(outRows[2], outRows[4]);
  std::swap(outRows[3], outRows[12]);
  std::swap(outRows[5], outRows[10]);
  std::swap(outRows[7], outRows[14]);
  std::swap(outRows[11], outRows[13]);
  std::swap(outRows[17], outRows[24]);
  std::swap(outRows[18], outRows[20]);
  std::swap(outRows[19], outRows[28]);
  std::swap(outRows[21], outRows[26]);
  std::swap(outRows[23], outRows[30]);
  std::swap(outRows[27], outRows[29]);
  std::swap(outRows[33], outRows[40]);
  std::swap(outRows[34], outRows[36]);
  std::swap(outRows[35], outRows[44]);
  std::swap(outRows[37], outRows[42]);
  std::swap(outRows[39], outRows[46]);
  std::swap(outRows[43], outRows[45]);
  std::swap(outRows[49], outRows[56]);
  std::swap(outRows[50], outRows[52]);
  std::swap(outRows[51], outRows[60]);
  std::swap(outRows[53], outRows[58]);
  std::swap(outRows[55], outRows[62]);
  std::swap(outRows[59], outRows[61]);
  // correction steps follow below (if required)
  zip<16>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip<16>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip<16>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip<16>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip<16>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip<16>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip<16>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip<16>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip<16>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip<16>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip<16>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip<16>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip<16>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip<16>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip<16>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip<16>(outRows[15], outRows[31], outRows[15], outRows[31]);
  zip<16>(outRows[32], outRows[48], outRows[32], outRows[48]);
  zip<16>(outRows[33], outRows[49], outRows[33], outRows[49]);
  zip<16>(outRows[34], outRows[50], outRows[34], outRows[50]);
  zip<16>(outRows[35], outRows[51], outRows[35], outRows[51]);
  zip<16>(outRows[36], outRows[52], outRows[36], outRows[52]);
  zip<16>(outRows[37], outRows[53], outRows[37], outRows[53]);
  zip<16>(outRows[38], outRows[54], outRows[38], outRows[54]);
  zip<16>(outRows[39], outRows[55], outRows[39], outRows[55]);
  zip<16>(outRows[40], outRows[56], outRows[40], outRows[56]);
  zip<16>(outRows[41], outRows[57], outRows[41], outRows[57]);
  zip<16>(outRows[42], outRows[58], outRows[42], outRows[58]);
  zip<16>(outRows[43], outRows[59], outRows[43], outRows[59]);
  zip<16>(outRows[44], outRows[60], outRows[44], outRows[60]);
  zip<16>(outRows[45], outRows[61], outRows[45], outRows[61]);
  zip<16>(outRows[46], outRows[62], outRows[46], outRows[62]);
  zip<16>(outRows[47], outRows[63], outRows[47], outRows[63]);
  zip<32>(outRows[0], outRows[32], outRows[0], outRows[32]);
  zip<32>(outRows[1], outRows[33], outRows[1], outRows[33]);
  zip<32>(outRows[2], outRows[34], outRows[2], outRows[34]);
  zip<32>(outRows[3], outRows[35], outRows[3], outRows[35]);
  zip<32>(outRows[4], outRows[36], outRows[4], outRows[36]);
  zip<32>(outRows[5], outRows[37], outRows[5], outRows[37]);
  zip<32>(outRows[6], outRows[38], outRows[6], outRows[38]);
  zip<32>(outRows[7], outRows[39], outRows[7], outRows[39]);
  zip<32>(outRows[8], outRows[40], outRows[8], outRows[40]);
  zip<32>(outRows[9], outRows[41], outRows[9], outRows[41]);
  zip<32>(outRows[10], outRows[42], outRows[10], outRows[42]);
  zip<32>(outRows[11], outRows[43], outRows[11], outRows[43]);
  zip<32>(outRows[12], outRows[44], outRows[12], outRows[44]);
  zip<32>(outRows[13], outRows[45], outRows[13], outRows[45]);
  zip<32>(outRows[14], outRows[46], outRows[14], outRows[46]);
  zip<32>(outRows[15], outRows[47], outRows[15], outRows[47]);
  zip<32>(outRows[16], outRows[48], outRows[16], outRows[48]);
  zip<32>(outRows[17], outRows[49], outRows[17], outRows[49]);
  zip<32>(outRows[18], outRows[50], outRows[18], outRows[50]);
  zip<32>(outRows[19], outRows[51], outRows[19], outRows[51]);
  zip<32>(outRows[20], outRows[52], outRows[20], outRows[52]);
  zip<32>(outRows[21], outRows[53], outRows[21], outRows[53]);
  zip<32>(outRows[22], outRows[54], outRows[22], outRows[54]);
  zip<32>(outRows[23], outRows[55], outRows[23], outRows[55]);
  zip<32>(outRows[24], outRows[56], outRows[24], outRows[56]);
  zip<32>(outRows[25], outRows[57], outRows[25], outRows[57]);
  zip<32>(outRows[26], outRows[58], outRows[26], outRows[58]);
  zip<32>(outRows[27], outRows[59], outRows[27], outRows[59]);
  zip<32>(outRows[28], outRows[60], outRows[28], outRows[60]);
  zip<32>(outRows[29], outRows[61], outRows[29], outRows[61]);
  zip<32>(outRows[30], outRows[62], outRows[30], outRows[62]);
  zip<32>(outRows[31], outRows[63], outRows[31], outRows[63]);
  std::swap(outRows[16], outRows[32]);
  std::swap(outRows[17], outRows[33]);
  std::swap(outRows[18], outRows[34]);
  std::swap(outRows[19], outRows[35]);
  std::swap(outRows[20], outRows[36]);
  std::swap(outRows[21], outRows[37]);
  std::swap(outRows[22], outRows[38]);
  std::swap(outRows[23], outRows[39]);
  std::swap(outRows[24], outRows[40]);
  std::swap(outRows[25], outRows[41]);
  std::swap(outRows[26], outRows[42]);
  std::swap(outRows[27], outRows[43]);
  std::swap(outRows[28], outRows[44]);
  std::swap(outRows[29], outRows[45]);
  std::swap(outRows[30], outRows[46]);
  std::swap(outRows[31], outRows[47]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<32>,
  Bytes<64>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip16<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip16<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip16<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip16<1>(inRows[16], inRows[17], outRows[16], outRows[17]);
  zip16<1>(inRows[18], inRows[19], outRows[18], outRows[19]);
  zip16<1>(inRows[20], inRows[21], outRows[20], outRows[21]);
  zip16<1>(inRows[22], inRows[23], outRows[22], outRows[23]);
  zip16<1>(inRows[24], inRows[25], outRows[24], outRows[25]);
  zip16<1>(inRows[26], inRows[27], outRows[26], outRows[27]);
  zip16<1>(inRows[28], inRows[29], outRows[28], outRows[29]);
  zip16<1>(inRows[30], inRows[31], outRows[30], outRows[31]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip16<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip16<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip16<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip16<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip16<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip16<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip16<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip16<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<4>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip16<4>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip16<4>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip16<4>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip16<4>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip16<4>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip16<4>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip16<4>(outRows[27], outRows[31], outRows[27], outRows[31]);
  std::swap(outRows[1], outRows[4]);
  std::swap(outRows[3], outRows[6]);
  std::swap(outRows[9], outRows[12]);
  std::swap(outRows[11], outRows[14]);
  std::swap(outRows[17], outRows[20]);
  std::swap(outRows[19], outRows[22]);
  std::swap(outRows[25], outRows[28]);
  std::swap(outRows[27], outRows[30]);
  // correction steps follow below (if required)
  zip<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip<8>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip<8>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip<8>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip<8>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip<8>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip<8>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip<8>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip<8>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip<16>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip<16>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip<16>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip<16>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip<16>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip<16>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip<16>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip<16>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip<16>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip<16>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip<16>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip<16>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip<16>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip<16>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip<16>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip<16>(outRows[15], outRows[31], outRows[15], outRows[31]);
  std::swap(outRows[8], outRows[16]);
  std::swap(outRows[9], outRows[17]);
  std::swap(outRows[10], outRows[18]);
  std::swap(outRows[11], outRows[19]);
  std::swap(outRows[12], outRows[20]);
  std::swap(outRows[13], outRows[21]);
  std::swap(outRows[14], outRows[22]);
  std::swap(outRows[15], outRows[23]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>,
  Bytes<64>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  zip16<1>(inRows[8], inRows[9], outRows[8], outRows[9]);
  zip16<1>(inRows[10], inRows[11], outRows[10], outRows[11]);
  zip16<1>(inRows[12], inRows[13], outRows[12], outRows[13]);
  zip16<1>(inRows[14], inRows[15], outRows[14], outRows[15]);
  zip16<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  std::swap(outRows[1], outRows[2]);
  std::swap(outRows[5], outRows[6]);
  std::swap(outRows[9], outRows[10]);
  std::swap(outRows[13], outRows[14]);
  // correction steps follow below (if required)
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<4>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<4>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<4>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<4>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<8>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<8>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<8>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<8>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<8>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<8>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<8>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<8>(outRows[7], outRows[15], outRows[7], outRows[15]);
  std::swap(outRows[4], outRows[8]);
  std::swap(outRows[5], outRows[9]);
  std::swap(outRows[6], outRows[10]);
  std::swap(outRows[7], outRows[11]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<64>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  zip16<1>(inRows[2], inRows[3], outRows[2], outRows[3]);
  zip16<1>(inRows[4], inRows[5], outRows[4], outRows[5]);
  zip16<1>(inRows[6], inRows[7], outRows[6], outRows[7]);
  // correction steps follow below (if required)
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<4>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<4>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<4>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<4>(outRows[3], outRows[7], outRows[3], outRows[7]);
  std::swap(outRows[2], outRows[4]);
  std::swap(outRows[3], outRows[5]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose1inplcLane(inRows, outRows, Elements<Vec<T, SIMD_WIDTH>::elements>(),
                      Bytes<SIMD_WIDTH>());
}

// ==========================================================
// transpose2inplc
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<2>)
{
  zip<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<4>)
{
  zip<1>(inRows[0], inRows[2], outRows[0], outRows[2]);
  zip<1>(inRows[1], inRows[3], outRows[1], outRows[3]);
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>)
{
  zip<1>(inRows[0], inRows[4], outRows[0], outRows[4]);
  zip<1>(inRows[1], inRows[5], outRows[1], outRows[5]);
  zip<1>(inRows[2], inRows[6], outRows[2], outRows[6]);
  zip<1>(inRows[3], inRows[7], outRows[3], outRows[7]);
  zip<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>)
{
  zip<1>(inRows[0], inRows[8], outRows[0], outRows[8]);
  zip<1>(inRows[1], inRows[9], outRows[1], outRows[9]);
  zip<1>(inRows[2], inRows[10], outRows[2], outRows[10]);
  zip<1>(inRows[3], inRows[11], outRows[3], outRows[11]);
  zip<1>(inRows[4], inRows[12], outRows[4], outRows[12]);
  zip<1>(inRows[5], inRows[13], outRows[5], outRows[13]);
  zip<1>(inRows[6], inRows[14], outRows[6], outRows[14]);
  zip<1>(inRows[7], inRows[15], outRows[7], outRows[15]);
  zip<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<1>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<1>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<1>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<1>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<32>)
{
  zip<1>(inRows[0], inRows[16], outRows[0], outRows[16]);
  zip<1>(inRows[1], inRows[17], outRows[1], outRows[17]);
  zip<1>(inRows[2], inRows[18], outRows[2], outRows[18]);
  zip<1>(inRows[3], inRows[19], outRows[3], outRows[19]);
  zip<1>(inRows[4], inRows[20], outRows[4], outRows[20]);
  zip<1>(inRows[5], inRows[21], outRows[5], outRows[21]);
  zip<1>(inRows[6], inRows[22], outRows[6], outRows[22]);
  zip<1>(inRows[7], inRows[23], outRows[7], outRows[23]);
  zip<1>(inRows[8], inRows[24], outRows[8], outRows[24]);
  zip<1>(inRows[9], inRows[25], outRows[9], outRows[25]);
  zip<1>(inRows[10], inRows[26], outRows[10], outRows[26]);
  zip<1>(inRows[11], inRows[27], outRows[11], outRows[27]);
  zip<1>(inRows[12], inRows[28], outRows[12], outRows[28]);
  zip<1>(inRows[13], inRows[29], outRows[13], outRows[29]);
  zip<1>(inRows[14], inRows[30], outRows[14], outRows[30]);
  zip<1>(inRows[15], inRows[31], outRows[15], outRows[31]);
  zip<1>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<1>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip<1>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<1>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip<1>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<1>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip<1>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<1>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip<1>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<1>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip<1>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<1>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip<1>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<1>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip<1>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip<1>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<1>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip<1>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<1>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip<1>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<1>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip<1>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<1>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip<1>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<1>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<1>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<1>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip<1>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip<1>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip<1>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<1>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<1>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<1>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip<1>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip<1>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip<1>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  zip<1>(outRows[16], outRows[17], outRows[16], outRows[17]);
  zip<1>(outRows[18], outRows[19], outRows[18], outRows[19]);
  zip<1>(outRows[20], outRows[21], outRows[20], outRows[21]);
  zip<1>(outRows[22], outRows[23], outRows[22], outRows[23]);
  zip<1>(outRows[24], outRows[25], outRows[24], outRows[25]);
  zip<1>(outRows[26], outRows[27], outRows[26], outRows[27]);
  zip<1>(outRows[28], outRows[29], outRows[28], outRows[29]);
  zip<1>(outRows[30], outRows[31], outRows[30], outRows[31]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<64>)
{
  zip<1>(inRows[0], inRows[32], outRows[0], outRows[32]);
  zip<1>(inRows[1], inRows[33], outRows[1], outRows[33]);
  zip<1>(inRows[2], inRows[34], outRows[2], outRows[34]);
  zip<1>(inRows[3], inRows[35], outRows[3], outRows[35]);
  zip<1>(inRows[4], inRows[36], outRows[4], outRows[36]);
  zip<1>(inRows[5], inRows[37], outRows[5], outRows[37]);
  zip<1>(inRows[6], inRows[38], outRows[6], outRows[38]);
  zip<1>(inRows[7], inRows[39], outRows[7], outRows[39]);
  zip<1>(inRows[8], inRows[40], outRows[8], outRows[40]);
  zip<1>(inRows[9], inRows[41], outRows[9], outRows[41]);
  zip<1>(inRows[10], inRows[42], outRows[10], outRows[42]);
  zip<1>(inRows[11], inRows[43], outRows[11], outRows[43]);
  zip<1>(inRows[12], inRows[44], outRows[12], outRows[44]);
  zip<1>(inRows[13], inRows[45], outRows[13], outRows[45]);
  zip<1>(inRows[14], inRows[46], outRows[14], outRows[46]);
  zip<1>(inRows[15], inRows[47], outRows[15], outRows[47]);
  zip<1>(inRows[16], inRows[48], outRows[16], outRows[48]);
  zip<1>(inRows[17], inRows[49], outRows[17], outRows[49]);
  zip<1>(inRows[18], inRows[50], outRows[18], outRows[50]);
  zip<1>(inRows[19], inRows[51], outRows[19], outRows[51]);
  zip<1>(inRows[20], inRows[52], outRows[20], outRows[52]);
  zip<1>(inRows[21], inRows[53], outRows[21], outRows[53]);
  zip<1>(inRows[22], inRows[54], outRows[22], outRows[54]);
  zip<1>(inRows[23], inRows[55], outRows[23], outRows[55]);
  zip<1>(inRows[24], inRows[56], outRows[24], outRows[56]);
  zip<1>(inRows[25], inRows[57], outRows[25], outRows[57]);
  zip<1>(inRows[26], inRows[58], outRows[26], outRows[58]);
  zip<1>(inRows[27], inRows[59], outRows[27], outRows[59]);
  zip<1>(inRows[28], inRows[60], outRows[28], outRows[60]);
  zip<1>(inRows[29], inRows[61], outRows[29], outRows[61]);
  zip<1>(inRows[30], inRows[62], outRows[30], outRows[62]);
  zip<1>(inRows[31], inRows[63], outRows[31], outRows[63]);
  zip<1>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip<1>(outRows[32], outRows[48], outRows[32], outRows[48]);
  zip<1>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip<1>(outRows[33], outRows[49], outRows[33], outRows[49]);
  zip<1>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip<1>(outRows[34], outRows[50], outRows[34], outRows[50]);
  zip<1>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip<1>(outRows[35], outRows[51], outRows[35], outRows[51]);
  zip<1>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip<1>(outRows[36], outRows[52], outRows[36], outRows[52]);
  zip<1>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip<1>(outRows[37], outRows[53], outRows[37], outRows[53]);
  zip<1>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip<1>(outRows[38], outRows[54], outRows[38], outRows[54]);
  zip<1>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip<1>(outRows[39], outRows[55], outRows[39], outRows[55]);
  zip<1>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip<1>(outRows[40], outRows[56], outRows[40], outRows[56]);
  zip<1>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip<1>(outRows[41], outRows[57], outRows[41], outRows[57]);
  zip<1>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip<1>(outRows[42], outRows[58], outRows[42], outRows[58]);
  zip<1>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip<1>(outRows[43], outRows[59], outRows[43], outRows[59]);
  zip<1>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip<1>(outRows[44], outRows[60], outRows[44], outRows[60]);
  zip<1>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip<1>(outRows[45], outRows[61], outRows[45], outRows[61]);
  zip<1>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip<1>(outRows[46], outRows[62], outRows[46], outRows[62]);
  zip<1>(outRows[15], outRows[31], outRows[15], outRows[31]);
  zip<1>(outRows[47], outRows[63], outRows[47], outRows[63]);
  zip<1>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip<1>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip<1>(outRows[32], outRows[40], outRows[32], outRows[40]);
  zip<1>(outRows[48], outRows[56], outRows[48], outRows[56]);
  zip<1>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip<1>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip<1>(outRows[33], outRows[41], outRows[33], outRows[41]);
  zip<1>(outRows[49], outRows[57], outRows[49], outRows[57]);
  zip<1>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip<1>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip<1>(outRows[34], outRows[42], outRows[34], outRows[42]);
  zip<1>(outRows[50], outRows[58], outRows[50], outRows[58]);
  zip<1>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip<1>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip<1>(outRows[35], outRows[43], outRows[35], outRows[43]);
  zip<1>(outRows[51], outRows[59], outRows[51], outRows[59]);
  zip<1>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip<1>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip<1>(outRows[36], outRows[44], outRows[36], outRows[44]);
  zip<1>(outRows[52], outRows[60], outRows[52], outRows[60]);
  zip<1>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip<1>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip<1>(outRows[37], outRows[45], outRows[37], outRows[45]);
  zip<1>(outRows[53], outRows[61], outRows[53], outRows[61]);
  zip<1>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip<1>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip<1>(outRows[38], outRows[46], outRows[38], outRows[46]);
  zip<1>(outRows[54], outRows[62], outRows[54], outRows[62]);
  zip<1>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip<1>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip<1>(outRows[39], outRows[47], outRows[39], outRows[47]);
  zip<1>(outRows[55], outRows[63], outRows[55], outRows[63]);
  zip<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip<1>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip<1>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip<1>(outRows[32], outRows[36], outRows[32], outRows[36]);
  zip<1>(outRows[40], outRows[44], outRows[40], outRows[44]);
  zip<1>(outRows[48], outRows[52], outRows[48], outRows[52]);
  zip<1>(outRows[56], outRows[60], outRows[56], outRows[60]);
  zip<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip<1>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip<1>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip<1>(outRows[33], outRows[37], outRows[33], outRows[37]);
  zip<1>(outRows[41], outRows[45], outRows[41], outRows[45]);
  zip<1>(outRows[49], outRows[53], outRows[49], outRows[53]);
  zip<1>(outRows[57], outRows[61], outRows[57], outRows[61]);
  zip<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip<1>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip<1>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip<1>(outRows[34], outRows[38], outRows[34], outRows[38]);
  zip<1>(outRows[42], outRows[46], outRows[42], outRows[46]);
  zip<1>(outRows[50], outRows[54], outRows[50], outRows[54]);
  zip<1>(outRows[58], outRows[62], outRows[58], outRows[62]);
  zip<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip<1>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip<1>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip<1>(outRows[35], outRows[39], outRows[35], outRows[39]);
  zip<1>(outRows[43], outRows[47], outRows[43], outRows[47]);
  zip<1>(outRows[51], outRows[55], outRows[51], outRows[55]);
  zip<1>(outRows[59], outRows[63], outRows[59], outRows[63]);
  zip<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<1>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<1>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<1>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip<1>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip<1>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip<1>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip<1>(outRows[32], outRows[34], outRows[32], outRows[34]);
  zip<1>(outRows[36], outRows[38], outRows[36], outRows[38]);
  zip<1>(outRows[40], outRows[42], outRows[40], outRows[42]);
  zip<1>(outRows[44], outRows[46], outRows[44], outRows[46]);
  zip<1>(outRows[48], outRows[50], outRows[48], outRows[50]);
  zip<1>(outRows[52], outRows[54], outRows[52], outRows[54]);
  zip<1>(outRows[56], outRows[58], outRows[56], outRows[58]);
  zip<1>(outRows[60], outRows[62], outRows[60], outRows[62]);
  zip<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<1>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<1>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<1>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip<1>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip<1>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip<1>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip<1>(outRows[33], outRows[35], outRows[33], outRows[35]);
  zip<1>(outRows[37], outRows[39], outRows[37], outRows[39]);
  zip<1>(outRows[41], outRows[43], outRows[41], outRows[43]);
  zip<1>(outRows[45], outRows[47], outRows[45], outRows[47]);
  zip<1>(outRows[49], outRows[51], outRows[49], outRows[51]);
  zip<1>(outRows[53], outRows[55], outRows[53], outRows[55]);
  zip<1>(outRows[57], outRows[59], outRows[57], outRows[59]);
  zip<1>(outRows[61], outRows[63], outRows[61], outRows[63]);
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  zip<1>(outRows[16], outRows[17], outRows[16], outRows[17]);
  zip<1>(outRows[18], outRows[19], outRows[18], outRows[19]);
  zip<1>(outRows[20], outRows[21], outRows[20], outRows[21]);
  zip<1>(outRows[22], outRows[23], outRows[22], outRows[23]);
  zip<1>(outRows[24], outRows[25], outRows[24], outRows[25]);
  zip<1>(outRows[26], outRows[27], outRows[26], outRows[27]);
  zip<1>(outRows[28], outRows[29], outRows[28], outRows[29]);
  zip<1>(outRows[30], outRows[31], outRows[30], outRows[31]);
  zip<1>(outRows[32], outRows[33], outRows[32], outRows[33]);
  zip<1>(outRows[34], outRows[35], outRows[34], outRows[35]);
  zip<1>(outRows[36], outRows[37], outRows[36], outRows[37]);
  zip<1>(outRows[38], outRows[39], outRows[38], outRows[39]);
  zip<1>(outRows[40], outRows[41], outRows[40], outRows[41]);
  zip<1>(outRows[42], outRows[43], outRows[42], outRows[43]);
  zip<1>(outRows[44], outRows[45], outRows[44], outRows[45]);
  zip<1>(outRows[46], outRows[47], outRows[46], outRows[47]);
  zip<1>(outRows[48], outRows[49], outRows[48], outRows[49]);
  zip<1>(outRows[50], outRows[51], outRows[50], outRows[51]);
  zip<1>(outRows[52], outRows[53], outRows[52], outRows[53]);
  zip<1>(outRows[54], outRows[55], outRows[54], outRows[55]);
  zip<1>(outRows[56], outRows[57], outRows[56], outRows[57]);
  zip<1>(outRows[58], outRows[59], outRows[58], outRows[59]);
  zip<1>(outRows[60], outRows[61], outRows[60], outRows[61]);
  zip<1>(outRows[62], outRows[63], outRows[62], outRows[63]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose2inplc(inRows, outRows, Elements<Vec<T, SIMD_WIDTH>::elements>());
}

// ==========================================================
// transpose2inplcLane
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>,
  Bytes<16>)
{
  zip16<1>(inRows[0], inRows[8], outRows[0], outRows[8]);
  zip16<1>(inRows[1], inRows[9], outRows[1], outRows[9]);
  zip16<1>(inRows[2], inRows[10], outRows[2], outRows[10]);
  zip16<1>(inRows[3], inRows[11], outRows[3], outRows[11]);
  zip16<1>(inRows[4], inRows[12], outRows[4], outRows[12]);
  zip16<1>(inRows[5], inRows[13], outRows[5], outRows[13]);
  zip16<1>(inRows[6], inRows[14], outRows[6], outRows[14]);
  zip16<1>(inRows[7], inRows[15], outRows[7], outRows[15]);
  zip16<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<1>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<1>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<1>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<1>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip16<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip16<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip16<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip16<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip16<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip16<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip16<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<16>)
{
  zip16<1>(inRows[0], inRows[4], outRows[0], outRows[4]);
  zip16<1>(inRows[1], inRows[5], outRows[1], outRows[5]);
  zip16<1>(inRows[2], inRows[6], outRows[2], outRows[6]);
  zip16<1>(inRows[3], inRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip16<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip16<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip16<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<16>)
{
  zip16<1>(inRows[0], inRows[2], outRows[0], outRows[2]);
  zip16<1>(inRows[1], inRows[3], outRows[1], outRows[3]);
  zip16<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip16<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<2>, Bytes<16>)
{
  zip16<1>(inRows[0], inRows[1], outRows[0], outRows[1]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<32>,
  Bytes<32>)
{
  zip16<1>(inRows[0], inRows[16], outRows[0], outRows[16]);
  zip16<1>(inRows[1], inRows[17], outRows[1], outRows[17]);
  zip16<1>(inRows[2], inRows[18], outRows[2], outRows[18]);
  zip16<1>(inRows[3], inRows[19], outRows[3], outRows[19]);
  zip16<1>(inRows[4], inRows[20], outRows[4], outRows[20]);
  zip16<1>(inRows[5], inRows[21], outRows[5], outRows[21]);
  zip16<1>(inRows[6], inRows[22], outRows[6], outRows[22]);
  zip16<1>(inRows[7], inRows[23], outRows[7], outRows[23]);
  zip16<1>(inRows[8], inRows[24], outRows[8], outRows[24]);
  zip16<1>(inRows[9], inRows[25], outRows[9], outRows[25]);
  zip16<1>(inRows[10], inRows[26], outRows[10], outRows[26]);
  zip16<1>(inRows[11], inRows[27], outRows[11], outRows[27]);
  zip16<1>(inRows[12], inRows[28], outRows[12], outRows[28]);
  zip16<1>(inRows[13], inRows[29], outRows[13], outRows[29]);
  zip16<1>(inRows[14], inRows[30], outRows[14], outRows[30]);
  zip16<1>(inRows[15], inRows[31], outRows[15], outRows[31]);
  zip16<1>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip16<1>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip16<1>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip16<1>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip16<1>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip16<1>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip16<1>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip16<1>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip16<1>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip16<1>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip16<1>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip16<1>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip16<1>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip16<1>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip16<1>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip16<1>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip16<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<1>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip16<1>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip16<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<1>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip16<1>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip16<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<1>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip16<1>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip16<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<1>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip16<1>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip16<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<1>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<1>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<1>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip16<1>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip16<1>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip16<1>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip16<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<1>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<1>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip16<1>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip16<1>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip16<1>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip16<1>(outRows[29], outRows[31], outRows[29], outRows[31]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  zip<1>(outRows[16], outRows[17], outRows[16], outRows[17]);
  zip<1>(outRows[18], outRows[19], outRows[18], outRows[19]);
  zip<1>(outRows[20], outRows[21], outRows[20], outRows[21]);
  zip<1>(outRows[22], outRows[23], outRows[22], outRows[23]);
  zip<1>(outRows[24], outRows[25], outRows[24], outRows[25]);
  zip<1>(outRows[26], outRows[27], outRows[26], outRows[27]);
  zip<1>(outRows[28], outRows[29], outRows[28], outRows[29]);
  zip<1>(outRows[30], outRows[31], outRows[30], outRows[31]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[2];
    outRows[2]               = outRows[4];
    outRows[4]               = outRows[8];
    outRows[8]               = outRows[16];
    outRows[16]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[6];
    outRows[6]               = outRows[12];
    outRows[12]              = outRows[24];
    outRows[24]              = outRows[17];
    outRows[17]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[5];
    outRows[5]               = outRows[10];
    outRows[10]              = outRows[20];
    outRows[20]              = outRows[9];
    outRows[9]               = outRows[18];
    outRows[18]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[7];
    outRows[7]               = outRows[14];
    outRows[14]              = outRows[28];
    outRows[28]              = outRows[25];
    outRows[25]              = outRows[19];
    outRows[19]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[11];
    outRows[11]              = outRows[22];
    outRows[22]              = outRows[13];
    outRows[13]              = outRows[26];
    outRows[26]              = outRows[21];
    outRows[21]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[15];
    outRows[15]              = outRows[30];
    outRows[30]              = outRows[29];
    outRows[29]              = outRows[27];
    outRows[27]              = outRows[23];
    outRows[23]              = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>,
  Bytes<32>)
{
  zip16<1>(inRows[0], inRows[8], outRows[0], outRows[8]);
  zip16<1>(inRows[1], inRows[9], outRows[1], outRows[9]);
  zip16<1>(inRows[2], inRows[10], outRows[2], outRows[10]);
  zip16<1>(inRows[3], inRows[11], outRows[3], outRows[11]);
  zip16<1>(inRows[4], inRows[12], outRows[4], outRows[12]);
  zip16<1>(inRows[5], inRows[13], outRows[5], outRows[13]);
  zip16<1>(inRows[6], inRows[14], outRows[6], outRows[14]);
  zip16<1>(inRows[7], inRows[15], outRows[7], outRows[15]);
  zip16<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<1>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip16<1>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip16<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip16<1>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip16<1>(outRows[13], outRows[15], outRows[13], outRows[15]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[2];
    outRows[2]               = outRows[4];
    outRows[4]               = outRows[8];
    outRows[8]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[6];
    outRows[6]               = outRows[12];
    outRows[12]              = outRows[9];
    outRows[9]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[5];
    outRows[5]               = outRows[10];
    outRows[10]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[7];
    outRows[7]               = outRows[14];
    outRows[14]              = outRows[13];
    outRows[13]              = outRows[11];
    outRows[11]              = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<32>)
{
  zip16<1>(inRows[0], inRows[4], outRows[0], outRows[4]);
  zip16<1>(inRows[1], inRows[5], outRows[1], outRows[5]);
  zip16<1>(inRows[2], inRows[6], outRows[2], outRows[6]);
  zip16<1>(inRows[3], inRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip16<1>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip16<1>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip16<1>(outRows[5], outRows[7], outRows[5], outRows[7]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[2];
    outRows[2]               = outRows[4];
    outRows[4]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[6];
    outRows[6]               = outRows[5];
    outRows[5]               = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<32>)
{
  zip16<1>(inRows[0], inRows[2], outRows[0], outRows[2]);
  zip16<1>(inRows[1], inRows[3], outRows[1], outRows[3]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[2];
    outRows[2]               = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<64>,
  Bytes<64>)
{
  zip16<1>(inRows[0], inRows[32], outRows[0], outRows[32]);
  zip16<1>(inRows[1], inRows[33], outRows[1], outRows[33]);
  zip16<1>(inRows[2], inRows[34], outRows[2], outRows[34]);
  zip16<1>(inRows[3], inRows[35], outRows[3], outRows[35]);
  zip16<1>(inRows[4], inRows[36], outRows[4], outRows[36]);
  zip16<1>(inRows[5], inRows[37], outRows[5], outRows[37]);
  zip16<1>(inRows[6], inRows[38], outRows[6], outRows[38]);
  zip16<1>(inRows[7], inRows[39], outRows[7], outRows[39]);
  zip16<1>(inRows[8], inRows[40], outRows[8], outRows[40]);
  zip16<1>(inRows[9], inRows[41], outRows[9], outRows[41]);
  zip16<1>(inRows[10], inRows[42], outRows[10], outRows[42]);
  zip16<1>(inRows[11], inRows[43], outRows[11], outRows[43]);
  zip16<1>(inRows[12], inRows[44], outRows[12], outRows[44]);
  zip16<1>(inRows[13], inRows[45], outRows[13], outRows[45]);
  zip16<1>(inRows[14], inRows[46], outRows[14], outRows[46]);
  zip16<1>(inRows[15], inRows[47], outRows[15], outRows[47]);
  zip16<1>(inRows[16], inRows[48], outRows[16], outRows[48]);
  zip16<1>(inRows[17], inRows[49], outRows[17], outRows[49]);
  zip16<1>(inRows[18], inRows[50], outRows[18], outRows[50]);
  zip16<1>(inRows[19], inRows[51], outRows[19], outRows[51]);
  zip16<1>(inRows[20], inRows[52], outRows[20], outRows[52]);
  zip16<1>(inRows[21], inRows[53], outRows[21], outRows[53]);
  zip16<1>(inRows[22], inRows[54], outRows[22], outRows[54]);
  zip16<1>(inRows[23], inRows[55], outRows[23], outRows[55]);
  zip16<1>(inRows[24], inRows[56], outRows[24], outRows[56]);
  zip16<1>(inRows[25], inRows[57], outRows[25], outRows[57]);
  zip16<1>(inRows[26], inRows[58], outRows[26], outRows[58]);
  zip16<1>(inRows[27], inRows[59], outRows[27], outRows[59]);
  zip16<1>(inRows[28], inRows[60], outRows[28], outRows[60]);
  zip16<1>(inRows[29], inRows[61], outRows[29], outRows[61]);
  zip16<1>(inRows[30], inRows[62], outRows[30], outRows[62]);
  zip16<1>(inRows[31], inRows[63], outRows[31], outRows[63]);
  zip16<1>(outRows[0], outRows[16], outRows[0], outRows[16]);
  zip16<1>(outRows[32], outRows[48], outRows[32], outRows[48]);
  zip16<1>(outRows[1], outRows[17], outRows[1], outRows[17]);
  zip16<1>(outRows[33], outRows[49], outRows[33], outRows[49]);
  zip16<1>(outRows[2], outRows[18], outRows[2], outRows[18]);
  zip16<1>(outRows[34], outRows[50], outRows[34], outRows[50]);
  zip16<1>(outRows[3], outRows[19], outRows[3], outRows[19]);
  zip16<1>(outRows[35], outRows[51], outRows[35], outRows[51]);
  zip16<1>(outRows[4], outRows[20], outRows[4], outRows[20]);
  zip16<1>(outRows[36], outRows[52], outRows[36], outRows[52]);
  zip16<1>(outRows[5], outRows[21], outRows[5], outRows[21]);
  zip16<1>(outRows[37], outRows[53], outRows[37], outRows[53]);
  zip16<1>(outRows[6], outRows[22], outRows[6], outRows[22]);
  zip16<1>(outRows[38], outRows[54], outRows[38], outRows[54]);
  zip16<1>(outRows[7], outRows[23], outRows[7], outRows[23]);
  zip16<1>(outRows[39], outRows[55], outRows[39], outRows[55]);
  zip16<1>(outRows[8], outRows[24], outRows[8], outRows[24]);
  zip16<1>(outRows[40], outRows[56], outRows[40], outRows[56]);
  zip16<1>(outRows[9], outRows[25], outRows[9], outRows[25]);
  zip16<1>(outRows[41], outRows[57], outRows[41], outRows[57]);
  zip16<1>(outRows[10], outRows[26], outRows[10], outRows[26]);
  zip16<1>(outRows[42], outRows[58], outRows[42], outRows[58]);
  zip16<1>(outRows[11], outRows[27], outRows[11], outRows[27]);
  zip16<1>(outRows[43], outRows[59], outRows[43], outRows[59]);
  zip16<1>(outRows[12], outRows[28], outRows[12], outRows[28]);
  zip16<1>(outRows[44], outRows[60], outRows[44], outRows[60]);
  zip16<1>(outRows[13], outRows[29], outRows[13], outRows[29]);
  zip16<1>(outRows[45], outRows[61], outRows[45], outRows[61]);
  zip16<1>(outRows[14], outRows[30], outRows[14], outRows[30]);
  zip16<1>(outRows[46], outRows[62], outRows[46], outRows[62]);
  zip16<1>(outRows[15], outRows[31], outRows[15], outRows[31]);
  zip16<1>(outRows[47], outRows[63], outRows[47], outRows[63]);
  zip16<1>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip16<1>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip16<1>(outRows[32], outRows[40], outRows[32], outRows[40]);
  zip16<1>(outRows[48], outRows[56], outRows[48], outRows[56]);
  zip16<1>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip16<1>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip16<1>(outRows[33], outRows[41], outRows[33], outRows[41]);
  zip16<1>(outRows[49], outRows[57], outRows[49], outRows[57]);
  zip16<1>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip16<1>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip16<1>(outRows[34], outRows[42], outRows[34], outRows[42]);
  zip16<1>(outRows[50], outRows[58], outRows[50], outRows[58]);
  zip16<1>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip16<1>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip16<1>(outRows[35], outRows[43], outRows[35], outRows[43]);
  zip16<1>(outRows[51], outRows[59], outRows[51], outRows[59]);
  zip16<1>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip16<1>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip16<1>(outRows[36], outRows[44], outRows[36], outRows[44]);
  zip16<1>(outRows[52], outRows[60], outRows[52], outRows[60]);
  zip16<1>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip16<1>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip16<1>(outRows[37], outRows[45], outRows[37], outRows[45]);
  zip16<1>(outRows[53], outRows[61], outRows[53], outRows[61]);
  zip16<1>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip16<1>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip16<1>(outRows[38], outRows[46], outRows[38], outRows[46]);
  zip16<1>(outRows[54], outRows[62], outRows[54], outRows[62]);
  zip16<1>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip16<1>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip16<1>(outRows[39], outRows[47], outRows[39], outRows[47]);
  zip16<1>(outRows[55], outRows[63], outRows[55], outRows[63]);
  zip16<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<1>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip16<1>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip16<1>(outRows[32], outRows[36], outRows[32], outRows[36]);
  zip16<1>(outRows[40], outRows[44], outRows[40], outRows[44]);
  zip16<1>(outRows[48], outRows[52], outRows[48], outRows[52]);
  zip16<1>(outRows[56], outRows[60], outRows[56], outRows[60]);
  zip16<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<1>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip16<1>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip16<1>(outRows[33], outRows[37], outRows[33], outRows[37]);
  zip16<1>(outRows[41], outRows[45], outRows[41], outRows[45]);
  zip16<1>(outRows[49], outRows[53], outRows[49], outRows[53]);
  zip16<1>(outRows[57], outRows[61], outRows[57], outRows[61]);
  zip16<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<1>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip16<1>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip16<1>(outRows[34], outRows[38], outRows[34], outRows[38]);
  zip16<1>(outRows[42], outRows[46], outRows[42], outRows[46]);
  zip16<1>(outRows[50], outRows[54], outRows[50], outRows[54]);
  zip16<1>(outRows[58], outRows[62], outRows[58], outRows[62]);
  zip16<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<1>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip16<1>(outRows[27], outRows[31], outRows[27], outRows[31]);
  zip16<1>(outRows[35], outRows[39], outRows[35], outRows[39]);
  zip16<1>(outRows[43], outRows[47], outRows[43], outRows[47]);
  zip16<1>(outRows[51], outRows[55], outRows[51], outRows[55]);
  zip16<1>(outRows[59], outRows[63], outRows[59], outRows[63]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  zip<1>(outRows[16], outRows[17], outRows[16], outRows[17]);
  zip<1>(outRows[18], outRows[19], outRows[18], outRows[19]);
  zip<1>(outRows[20], outRows[21], outRows[20], outRows[21]);
  zip<1>(outRows[22], outRows[23], outRows[22], outRows[23]);
  zip<1>(outRows[24], outRows[25], outRows[24], outRows[25]);
  zip<1>(outRows[26], outRows[27], outRows[26], outRows[27]);
  zip<1>(outRows[28], outRows[29], outRows[28], outRows[29]);
  zip<1>(outRows[30], outRows[31], outRows[30], outRows[31]);
  zip<1>(outRows[32], outRows[33], outRows[32], outRows[33]);
  zip<1>(outRows[34], outRows[35], outRows[34], outRows[35]);
  zip<1>(outRows[36], outRows[37], outRows[36], outRows[37]);
  zip<1>(outRows[38], outRows[39], outRows[38], outRows[39]);
  zip<1>(outRows[40], outRows[41], outRows[40], outRows[41]);
  zip<1>(outRows[42], outRows[43], outRows[42], outRows[43]);
  zip<1>(outRows[44], outRows[45], outRows[44], outRows[45]);
  zip<1>(outRows[46], outRows[47], outRows[46], outRows[47]);
  zip<1>(outRows[48], outRows[49], outRows[48], outRows[49]);
  zip<1>(outRows[50], outRows[51], outRows[50], outRows[51]);
  zip<1>(outRows[52], outRows[53], outRows[52], outRows[53]);
  zip<1>(outRows[54], outRows[55], outRows[54], outRows[55]);
  zip<1>(outRows[56], outRows[57], outRows[56], outRows[57]);
  zip<1>(outRows[58], outRows[59], outRows[58], outRows[59]);
  zip<1>(outRows[60], outRows[61], outRows[60], outRows[61]);
  zip<1>(outRows[62], outRows[63], outRows[62], outRows[63]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip<2>(outRows[32], outRows[34], outRows[32], outRows[34]);
  zip<2>(outRows[36], outRows[38], outRows[36], outRows[38]);
  zip<2>(outRows[40], outRows[42], outRows[40], outRows[42]);
  zip<2>(outRows[44], outRows[46], outRows[44], outRows[46]);
  zip<2>(outRows[48], outRows[50], outRows[48], outRows[50]);
  zip<2>(outRows[52], outRows[54], outRows[52], outRows[54]);
  zip<2>(outRows[56], outRows[58], outRows[56], outRows[58]);
  zip<2>(outRows[60], outRows[62], outRows[60], outRows[62]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  zip<2>(outRows[33], outRows[35], outRows[33], outRows[35]);
  zip<2>(outRows[37], outRows[39], outRows[37], outRows[39]);
  zip<2>(outRows[41], outRows[43], outRows[41], outRows[43]);
  zip<2>(outRows[45], outRows[47], outRows[45], outRows[47]);
  zip<2>(outRows[49], outRows[51], outRows[49], outRows[51]);
  zip<2>(outRows[53], outRows[55], outRows[53], outRows[55]);
  zip<2>(outRows[57], outRows[59], outRows[57], outRows[59]);
  zip<2>(outRows[61], outRows[63], outRows[61], outRows[63]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[4];
    outRows[4]               = outRows[16];
    outRows[16]              = outRows[2];
    outRows[2]               = outRows[8];
    outRows[8]               = outRows[32];
    outRows[32]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[12];
    outRows[12]              = outRows[48];
    outRows[48]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[5];
    outRows[5]               = outRows[20];
    outRows[20]              = outRows[18];
    outRows[18]              = outRows[10];
    outRows[10]              = outRows[40];
    outRows[40]              = outRows[33];
    outRows[33]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[6];
    outRows[6]               = outRows[24];
    outRows[24]              = outRows[34];
    outRows[34]              = outRows[9];
    outRows[9]               = outRows[36];
    outRows[36]              = outRows[17];
    outRows[17]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[7];
    outRows[7]               = outRows[28];
    outRows[28]              = outRows[50];
    outRows[50]              = outRows[11];
    outRows[11]              = outRows[44];
    outRows[44]              = outRows[49];
    outRows[49]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[13];
    outRows[13]              = outRows[52];
    outRows[52]              = outRows[19];
    outRows[19]              = outRows[14];
    outRows[14]              = outRows[56];
    outRows[56]              = outRows[35];
    outRows[35]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[15];
    outRows[15]              = outRows[60];
    outRows[60]              = outRows[51];
    outRows[51]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[21];
    outRows[21]              = outRows[22];
    outRows[22]              = outRows[26];
    outRows[26]              = outRows[42];
    outRows[42]              = outRows[41];
    outRows[41]              = outRows[37];
    outRows[37]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[23];
    outRows[23]              = outRows[30];
    outRows[30]              = outRows[58];
    outRows[58]              = outRows[43];
    outRows[43]              = outRows[45];
    outRows[45]              = outRows[53];
    outRows[53]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[25];
    outRows[25]              = outRows[38];
    outRows[38]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[27];
    outRows[27]              = outRows[46];
    outRows[46]              = outRows[57];
    outRows[57]              = outRows[39];
    outRows[39]              = outRows[29];
    outRows[29]              = outRows[54];
    outRows[54]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[31];
    outRows[31]              = outRows[62];
    outRows[62]              = outRows[59];
    outRows[59]              = outRows[47];
    outRows[47]              = outRows[61];
    outRows[61]              = outRows[55];
    outRows[55]              = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<32>,
  Bytes<64>)
{
  zip16<1>(inRows[0], inRows[16], outRows[0], outRows[16]);
  zip16<1>(inRows[1], inRows[17], outRows[1], outRows[17]);
  zip16<1>(inRows[2], inRows[18], outRows[2], outRows[18]);
  zip16<1>(inRows[3], inRows[19], outRows[3], outRows[19]);
  zip16<1>(inRows[4], inRows[20], outRows[4], outRows[20]);
  zip16<1>(inRows[5], inRows[21], outRows[5], outRows[21]);
  zip16<1>(inRows[6], inRows[22], outRows[6], outRows[22]);
  zip16<1>(inRows[7], inRows[23], outRows[7], outRows[23]);
  zip16<1>(inRows[8], inRows[24], outRows[8], outRows[24]);
  zip16<1>(inRows[9], inRows[25], outRows[9], outRows[25]);
  zip16<1>(inRows[10], inRows[26], outRows[10], outRows[26]);
  zip16<1>(inRows[11], inRows[27], outRows[11], outRows[27]);
  zip16<1>(inRows[12], inRows[28], outRows[12], outRows[28]);
  zip16<1>(inRows[13], inRows[29], outRows[13], outRows[29]);
  zip16<1>(inRows[14], inRows[30], outRows[14], outRows[30]);
  zip16<1>(inRows[15], inRows[31], outRows[15], outRows[31]);
  zip16<1>(outRows[0], outRows[8], outRows[0], outRows[8]);
  zip16<1>(outRows[16], outRows[24], outRows[16], outRows[24]);
  zip16<1>(outRows[1], outRows[9], outRows[1], outRows[9]);
  zip16<1>(outRows[17], outRows[25], outRows[17], outRows[25]);
  zip16<1>(outRows[2], outRows[10], outRows[2], outRows[10]);
  zip16<1>(outRows[18], outRows[26], outRows[18], outRows[26]);
  zip16<1>(outRows[3], outRows[11], outRows[3], outRows[11]);
  zip16<1>(outRows[19], outRows[27], outRows[19], outRows[27]);
  zip16<1>(outRows[4], outRows[12], outRows[4], outRows[12]);
  zip16<1>(outRows[20], outRows[28], outRows[20], outRows[28]);
  zip16<1>(outRows[5], outRows[13], outRows[5], outRows[13]);
  zip16<1>(outRows[21], outRows[29], outRows[21], outRows[29]);
  zip16<1>(outRows[6], outRows[14], outRows[6], outRows[14]);
  zip16<1>(outRows[22], outRows[30], outRows[22], outRows[30]);
  zip16<1>(outRows[7], outRows[15], outRows[7], outRows[15]);
  zip16<1>(outRows[23], outRows[31], outRows[23], outRows[31]);
  zip16<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<1>(outRows[16], outRows[20], outRows[16], outRows[20]);
  zip16<1>(outRows[24], outRows[28], outRows[24], outRows[28]);
  zip16<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<1>(outRows[17], outRows[21], outRows[17], outRows[21]);
  zip16<1>(outRows[25], outRows[29], outRows[25], outRows[29]);
  zip16<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<1>(outRows[18], outRows[22], outRows[18], outRows[22]);
  zip16<1>(outRows[26], outRows[30], outRows[26], outRows[30]);
  zip16<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  zip16<1>(outRows[19], outRows[23], outRows[19], outRows[23]);
  zip16<1>(outRows[27], outRows[31], outRows[27], outRows[31]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  zip<1>(outRows[16], outRows[17], outRows[16], outRows[17]);
  zip<1>(outRows[18], outRows[19], outRows[18], outRows[19]);
  zip<1>(outRows[20], outRows[21], outRows[20], outRows[21]);
  zip<1>(outRows[22], outRows[23], outRows[22], outRows[23]);
  zip<1>(outRows[24], outRows[25], outRows[24], outRows[25]);
  zip<1>(outRows[26], outRows[27], outRows[26], outRows[27]);
  zip<1>(outRows[28], outRows[29], outRows[28], outRows[29]);
  zip<1>(outRows[30], outRows[31], outRows[30], outRows[31]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<2>(outRows[16], outRows[18], outRows[16], outRows[18]);
  zip<2>(outRows[20], outRows[22], outRows[20], outRows[22]);
  zip<2>(outRows[24], outRows[26], outRows[24], outRows[26]);
  zip<2>(outRows[28], outRows[30], outRows[28], outRows[30]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  zip<2>(outRows[17], outRows[19], outRows[17], outRows[19]);
  zip<2>(outRows[21], outRows[23], outRows[21], outRows[23]);
  zip<2>(outRows[25], outRows[27], outRows[25], outRows[27]);
  zip<2>(outRows[29], outRows[31], outRows[29], outRows[31]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[4];
    outRows[4]               = outRows[16];
    outRows[16]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[2];
    outRows[2]               = outRows[8];
    outRows[8]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[12];
    outRows[12]              = outRows[18];
    outRows[18]              = outRows[9];
    outRows[9]               = outRows[6];
    outRows[6]               = outRows[24];
    outRows[24]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[5];
    outRows[5]               = outRows[20];
    outRows[20]              = outRows[17];
    outRows[17]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[7];
    outRows[7]               = outRows[28];
    outRows[28]              = outRows[19];
    outRows[19]              = outRows[13];
    outRows[13]              = outRows[22];
    outRows[22]              = outRows[25];
    outRows[25]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[11];
    outRows[11]              = outRows[14];
    outRows[14]              = outRows[26];
    outRows[26]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[15];
    outRows[15]              = outRows[30];
    outRows[30]              = outRows[27];
    outRows[27]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[23];
    outRows[23]              = outRows[29];
    outRows[29]              = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<16>,
  Bytes<64>)
{
  zip16<1>(inRows[0], inRows[8], outRows[0], outRows[8]);
  zip16<1>(inRows[1], inRows[9], outRows[1], outRows[9]);
  zip16<1>(inRows[2], inRows[10], outRows[2], outRows[10]);
  zip16<1>(inRows[3], inRows[11], outRows[3], outRows[11]);
  zip16<1>(inRows[4], inRows[12], outRows[4], outRows[12]);
  zip16<1>(inRows[5], inRows[13], outRows[5], outRows[13]);
  zip16<1>(inRows[6], inRows[14], outRows[6], outRows[14]);
  zip16<1>(inRows[7], inRows[15], outRows[7], outRows[15]);
  zip16<1>(outRows[0], outRows[4], outRows[0], outRows[4]);
  zip16<1>(outRows[8], outRows[12], outRows[8], outRows[12]);
  zip16<1>(outRows[1], outRows[5], outRows[1], outRows[5]);
  zip16<1>(outRows[9], outRows[13], outRows[9], outRows[13]);
  zip16<1>(outRows[2], outRows[6], outRows[2], outRows[6]);
  zip16<1>(outRows[10], outRows[14], outRows[10], outRows[14]);
  zip16<1>(outRows[3], outRows[7], outRows[3], outRows[7]);
  zip16<1>(outRows[11], outRows[15], outRows[11], outRows[15]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<1>(outRows[8], outRows[9], outRows[8], outRows[9]);
  zip<1>(outRows[10], outRows[11], outRows[10], outRows[11]);
  zip<1>(outRows[12], outRows[13], outRows[12], outRows[13]);
  zip<1>(outRows[14], outRows[15], outRows[14], outRows[15]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[8], outRows[10], outRows[8], outRows[10]);
  zip<2>(outRows[12], outRows[14], outRows[12], outRows[14]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  zip<2>(outRows[9], outRows[11], outRows[9], outRows[11]);
  zip<2>(outRows[13], outRows[15], outRows[13], outRows[15]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[4];
    outRows[4]               = outRows[2];
    outRows[2]               = outRows[8];
    outRows[8]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[12];
    outRows[12]              = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[5];
    outRows[5]               = outRows[6];
    outRows[6]               = outRows[10];
    outRows[10]              = outRows[9];
    outRows[9]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[7];
    outRows[7]               = outRows[14];
    outRows[14]              = outRows[11];
    outRows[11]              = outRows[13];
    outRows[13]              = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<64>)
{
  zip16<1>(inRows[0], inRows[4], outRows[0], outRows[4]);
  zip16<1>(inRows[1], inRows[5], outRows[1], outRows[5]);
  zip16<1>(inRows[2], inRows[6], outRows[2], outRows[6]);
  zip16<1>(inRows[3], inRows[7], outRows[3], outRows[7]);
  // correction steps follow below (if required)
  zip<1>(outRows[0], outRows[1], outRows[0], outRows[1]);
  zip<1>(outRows[2], outRows[3], outRows[2], outRows[3]);
  zip<1>(outRows[4], outRows[5], outRows[4], outRows[5]);
  zip<1>(outRows[6], outRows[7], outRows[6], outRows[7]);
  zip<2>(outRows[0], outRows[2], outRows[0], outRows[2]);
  zip<2>(outRows[4], outRows[6], outRows[4], outRows[6]);
  zip<2>(outRows[1], outRows[3], outRows[1], outRows[3]);
  zip<2>(outRows[5], outRows[7], outRows[5], outRows[7]);
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[1];
    outRows[1]               = outRows[4];
    outRows[4]               = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = outRows[3];
    outRows[3]               = outRows[6];
    outRows[6]               = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
  Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose2inplcLane(inRows, outRows, Elements<Vec<T, SIMD_WIDTH>::elements>(),
                      Bytes<SIMD_WIDTH>());
}

// ==========================================================
// transpose1inplc (1-argument version)
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<2>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<4>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  std::swap(rows[1], rows[2]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
  std::swap(rows[1], rows[4]);
  std::swap(rows[3], rows[6]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<2>(rows[8], rows[10], rows[8], rows[10]);
  zip<2>(rows[9], rows[11], rows[9], rows[11]);
  zip<2>(rows[12], rows[14], rows[12], rows[14]);
  zip<2>(rows[13], rows[15], rows[13], rows[15]);
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
  zip<4>(rows[8], rows[12], rows[8], rows[12]);
  zip<4>(rows[10], rows[14], rows[10], rows[14]);
  zip<4>(rows[9], rows[13], rows[9], rows[13]);
  zip<4>(rows[11], rows[15], rows[11], rows[15]);
  zip<8>(rows[0], rows[8], rows[0], rows[8]);
  zip<8>(rows[4], rows[12], rows[4], rows[12]);
  zip<8>(rows[2], rows[10], rows[2], rows[10]);
  zip<8>(rows[6], rows[14], rows[6], rows[14]);
  zip<8>(rows[1], rows[9], rows[1], rows[9]);
  zip<8>(rows[5], rows[13], rows[5], rows[13]);
  zip<8>(rows[3], rows[11], rows[3], rows[11]);
  zip<8>(rows[7], rows[15], rows[7], rows[15]);
  std::swap(rows[1], rows[8]);
  std::swap(rows[2], rows[4]);
  std::swap(rows[3], rows[12]);
  std::swap(rows[5], rows[10]);
  std::swap(rows[7], rows[14]);
  std::swap(rows[11], rows[13]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<32>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<2>(rows[8], rows[10], rows[8], rows[10]);
  zip<2>(rows[9], rows[11], rows[9], rows[11]);
  zip<2>(rows[12], rows[14], rows[12], rows[14]);
  zip<2>(rows[13], rows[15], rows[13], rows[15]);
  zip<2>(rows[16], rows[18], rows[16], rows[18]);
  zip<2>(rows[17], rows[19], rows[17], rows[19]);
  zip<2>(rows[20], rows[22], rows[20], rows[22]);
  zip<2>(rows[21], rows[23], rows[21], rows[23]);
  zip<2>(rows[24], rows[26], rows[24], rows[26]);
  zip<2>(rows[25], rows[27], rows[25], rows[27]);
  zip<2>(rows[28], rows[30], rows[28], rows[30]);
  zip<2>(rows[29], rows[31], rows[29], rows[31]);
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
  zip<4>(rows[8], rows[12], rows[8], rows[12]);
  zip<4>(rows[10], rows[14], rows[10], rows[14]);
  zip<4>(rows[9], rows[13], rows[9], rows[13]);
  zip<4>(rows[11], rows[15], rows[11], rows[15]);
  zip<4>(rows[16], rows[20], rows[16], rows[20]);
  zip<4>(rows[18], rows[22], rows[18], rows[22]);
  zip<4>(rows[17], rows[21], rows[17], rows[21]);
  zip<4>(rows[19], rows[23], rows[19], rows[23]);
  zip<4>(rows[24], rows[28], rows[24], rows[28]);
  zip<4>(rows[26], rows[30], rows[26], rows[30]);
  zip<4>(rows[25], rows[29], rows[25], rows[29]);
  zip<4>(rows[27], rows[31], rows[27], rows[31]);
  zip<8>(rows[0], rows[8], rows[0], rows[8]);
  zip<8>(rows[4], rows[12], rows[4], rows[12]);
  zip<8>(rows[2], rows[10], rows[2], rows[10]);
  zip<8>(rows[6], rows[14], rows[6], rows[14]);
  zip<8>(rows[1], rows[9], rows[1], rows[9]);
  zip<8>(rows[5], rows[13], rows[5], rows[13]);
  zip<8>(rows[3], rows[11], rows[3], rows[11]);
  zip<8>(rows[7], rows[15], rows[7], rows[15]);
  zip<8>(rows[16], rows[24], rows[16], rows[24]);
  zip<8>(rows[20], rows[28], rows[20], rows[28]);
  zip<8>(rows[18], rows[26], rows[18], rows[26]);
  zip<8>(rows[22], rows[30], rows[22], rows[30]);
  zip<8>(rows[17], rows[25], rows[17], rows[25]);
  zip<8>(rows[21], rows[29], rows[21], rows[29]);
  zip<8>(rows[19], rows[27], rows[19], rows[27]);
  zip<8>(rows[23], rows[31], rows[23], rows[31]);
  zip<16>(rows[0], rows[16], rows[0], rows[16]);
  zip<16>(rows[8], rows[24], rows[8], rows[24]);
  zip<16>(rows[4], rows[20], rows[4], rows[20]);
  zip<16>(rows[12], rows[28], rows[12], rows[28]);
  zip<16>(rows[2], rows[18], rows[2], rows[18]);
  zip<16>(rows[10], rows[26], rows[10], rows[26]);
  zip<16>(rows[6], rows[22], rows[6], rows[22]);
  zip<16>(rows[14], rows[30], rows[14], rows[30]);
  zip<16>(rows[1], rows[17], rows[1], rows[17]);
  zip<16>(rows[9], rows[25], rows[9], rows[25]);
  zip<16>(rows[5], rows[21], rows[5], rows[21]);
  zip<16>(rows[13], rows[29], rows[13], rows[29]);
  zip<16>(rows[3], rows[19], rows[3], rows[19]);
  zip<16>(rows[11], rows[27], rows[11], rows[27]);
  zip<16>(rows[7], rows[23], rows[7], rows[23]);
  zip<16>(rows[15], rows[31], rows[15], rows[31]);
  std::swap(rows[1], rows[16]);
  std::swap(rows[2], rows[8]);
  std::swap(rows[3], rows[24]);
  std::swap(rows[5], rows[20]);
  std::swap(rows[6], rows[12]);
  std::swap(rows[7], rows[28]);
  std::swap(rows[9], rows[18]);
  std::swap(rows[11], rows[26]);
  std::swap(rows[13], rows[22]);
  std::swap(rows[15], rows[30]);
  std::swap(rows[19], rows[25]);
  std::swap(rows[23], rows[29]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<64>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
  zip<1>(rows[32], rows[33], rows[32], rows[33]);
  zip<1>(rows[34], rows[35], rows[34], rows[35]);
  zip<1>(rows[36], rows[37], rows[36], rows[37]);
  zip<1>(rows[38], rows[39], rows[38], rows[39]);
  zip<1>(rows[40], rows[41], rows[40], rows[41]);
  zip<1>(rows[42], rows[43], rows[42], rows[43]);
  zip<1>(rows[44], rows[45], rows[44], rows[45]);
  zip<1>(rows[46], rows[47], rows[46], rows[47]);
  zip<1>(rows[48], rows[49], rows[48], rows[49]);
  zip<1>(rows[50], rows[51], rows[50], rows[51]);
  zip<1>(rows[52], rows[53], rows[52], rows[53]);
  zip<1>(rows[54], rows[55], rows[54], rows[55]);
  zip<1>(rows[56], rows[57], rows[56], rows[57]);
  zip<1>(rows[58], rows[59], rows[58], rows[59]);
  zip<1>(rows[60], rows[61], rows[60], rows[61]);
  zip<1>(rows[62], rows[63], rows[62], rows[63]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<2>(rows[8], rows[10], rows[8], rows[10]);
  zip<2>(rows[9], rows[11], rows[9], rows[11]);
  zip<2>(rows[12], rows[14], rows[12], rows[14]);
  zip<2>(rows[13], rows[15], rows[13], rows[15]);
  zip<2>(rows[16], rows[18], rows[16], rows[18]);
  zip<2>(rows[17], rows[19], rows[17], rows[19]);
  zip<2>(rows[20], rows[22], rows[20], rows[22]);
  zip<2>(rows[21], rows[23], rows[21], rows[23]);
  zip<2>(rows[24], rows[26], rows[24], rows[26]);
  zip<2>(rows[25], rows[27], rows[25], rows[27]);
  zip<2>(rows[28], rows[30], rows[28], rows[30]);
  zip<2>(rows[29], rows[31], rows[29], rows[31]);
  zip<2>(rows[32], rows[34], rows[32], rows[34]);
  zip<2>(rows[33], rows[35], rows[33], rows[35]);
  zip<2>(rows[36], rows[38], rows[36], rows[38]);
  zip<2>(rows[37], rows[39], rows[37], rows[39]);
  zip<2>(rows[40], rows[42], rows[40], rows[42]);
  zip<2>(rows[41], rows[43], rows[41], rows[43]);
  zip<2>(rows[44], rows[46], rows[44], rows[46]);
  zip<2>(rows[45], rows[47], rows[45], rows[47]);
  zip<2>(rows[48], rows[50], rows[48], rows[50]);
  zip<2>(rows[49], rows[51], rows[49], rows[51]);
  zip<2>(rows[52], rows[54], rows[52], rows[54]);
  zip<2>(rows[53], rows[55], rows[53], rows[55]);
  zip<2>(rows[56], rows[58], rows[56], rows[58]);
  zip<2>(rows[57], rows[59], rows[57], rows[59]);
  zip<2>(rows[60], rows[62], rows[60], rows[62]);
  zip<2>(rows[61], rows[63], rows[61], rows[63]);
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
  zip<4>(rows[8], rows[12], rows[8], rows[12]);
  zip<4>(rows[10], rows[14], rows[10], rows[14]);
  zip<4>(rows[9], rows[13], rows[9], rows[13]);
  zip<4>(rows[11], rows[15], rows[11], rows[15]);
  zip<4>(rows[16], rows[20], rows[16], rows[20]);
  zip<4>(rows[18], rows[22], rows[18], rows[22]);
  zip<4>(rows[17], rows[21], rows[17], rows[21]);
  zip<4>(rows[19], rows[23], rows[19], rows[23]);
  zip<4>(rows[24], rows[28], rows[24], rows[28]);
  zip<4>(rows[26], rows[30], rows[26], rows[30]);
  zip<4>(rows[25], rows[29], rows[25], rows[29]);
  zip<4>(rows[27], rows[31], rows[27], rows[31]);
  zip<4>(rows[32], rows[36], rows[32], rows[36]);
  zip<4>(rows[34], rows[38], rows[34], rows[38]);
  zip<4>(rows[33], rows[37], rows[33], rows[37]);
  zip<4>(rows[35], rows[39], rows[35], rows[39]);
  zip<4>(rows[40], rows[44], rows[40], rows[44]);
  zip<4>(rows[42], rows[46], rows[42], rows[46]);
  zip<4>(rows[41], rows[45], rows[41], rows[45]);
  zip<4>(rows[43], rows[47], rows[43], rows[47]);
  zip<4>(rows[48], rows[52], rows[48], rows[52]);
  zip<4>(rows[50], rows[54], rows[50], rows[54]);
  zip<4>(rows[49], rows[53], rows[49], rows[53]);
  zip<4>(rows[51], rows[55], rows[51], rows[55]);
  zip<4>(rows[56], rows[60], rows[56], rows[60]);
  zip<4>(rows[58], rows[62], rows[58], rows[62]);
  zip<4>(rows[57], rows[61], rows[57], rows[61]);
  zip<4>(rows[59], rows[63], rows[59], rows[63]);
  zip<8>(rows[0], rows[8], rows[0], rows[8]);
  zip<8>(rows[4], rows[12], rows[4], rows[12]);
  zip<8>(rows[2], rows[10], rows[2], rows[10]);
  zip<8>(rows[6], rows[14], rows[6], rows[14]);
  zip<8>(rows[1], rows[9], rows[1], rows[9]);
  zip<8>(rows[5], rows[13], rows[5], rows[13]);
  zip<8>(rows[3], rows[11], rows[3], rows[11]);
  zip<8>(rows[7], rows[15], rows[7], rows[15]);
  zip<8>(rows[16], rows[24], rows[16], rows[24]);
  zip<8>(rows[20], rows[28], rows[20], rows[28]);
  zip<8>(rows[18], rows[26], rows[18], rows[26]);
  zip<8>(rows[22], rows[30], rows[22], rows[30]);
  zip<8>(rows[17], rows[25], rows[17], rows[25]);
  zip<8>(rows[21], rows[29], rows[21], rows[29]);
  zip<8>(rows[19], rows[27], rows[19], rows[27]);
  zip<8>(rows[23], rows[31], rows[23], rows[31]);
  zip<8>(rows[32], rows[40], rows[32], rows[40]);
  zip<8>(rows[36], rows[44], rows[36], rows[44]);
  zip<8>(rows[34], rows[42], rows[34], rows[42]);
  zip<8>(rows[38], rows[46], rows[38], rows[46]);
  zip<8>(rows[33], rows[41], rows[33], rows[41]);
  zip<8>(rows[37], rows[45], rows[37], rows[45]);
  zip<8>(rows[35], rows[43], rows[35], rows[43]);
  zip<8>(rows[39], rows[47], rows[39], rows[47]);
  zip<8>(rows[48], rows[56], rows[48], rows[56]);
  zip<8>(rows[52], rows[60], rows[52], rows[60]);
  zip<8>(rows[50], rows[58], rows[50], rows[58]);
  zip<8>(rows[54], rows[62], rows[54], rows[62]);
  zip<8>(rows[49], rows[57], rows[49], rows[57]);
  zip<8>(rows[53], rows[61], rows[53], rows[61]);
  zip<8>(rows[51], rows[59], rows[51], rows[59]);
  zip<8>(rows[55], rows[63], rows[55], rows[63]);
  zip<16>(rows[0], rows[16], rows[0], rows[16]);
  zip<16>(rows[8], rows[24], rows[8], rows[24]);
  zip<16>(rows[4], rows[20], rows[4], rows[20]);
  zip<16>(rows[12], rows[28], rows[12], rows[28]);
  zip<16>(rows[2], rows[18], rows[2], rows[18]);
  zip<16>(rows[10], rows[26], rows[10], rows[26]);
  zip<16>(rows[6], rows[22], rows[6], rows[22]);
  zip<16>(rows[14], rows[30], rows[14], rows[30]);
  zip<16>(rows[1], rows[17], rows[1], rows[17]);
  zip<16>(rows[9], rows[25], rows[9], rows[25]);
  zip<16>(rows[5], rows[21], rows[5], rows[21]);
  zip<16>(rows[13], rows[29], rows[13], rows[29]);
  zip<16>(rows[3], rows[19], rows[3], rows[19]);
  zip<16>(rows[11], rows[27], rows[11], rows[27]);
  zip<16>(rows[7], rows[23], rows[7], rows[23]);
  zip<16>(rows[15], rows[31], rows[15], rows[31]);
  zip<16>(rows[32], rows[48], rows[32], rows[48]);
  zip<16>(rows[40], rows[56], rows[40], rows[56]);
  zip<16>(rows[36], rows[52], rows[36], rows[52]);
  zip<16>(rows[44], rows[60], rows[44], rows[60]);
  zip<16>(rows[34], rows[50], rows[34], rows[50]);
  zip<16>(rows[42], rows[58], rows[42], rows[58]);
  zip<16>(rows[38], rows[54], rows[38], rows[54]);
  zip<16>(rows[46], rows[62], rows[46], rows[62]);
  zip<16>(rows[33], rows[49], rows[33], rows[49]);
  zip<16>(rows[41], rows[57], rows[41], rows[57]);
  zip<16>(rows[37], rows[53], rows[37], rows[53]);
  zip<16>(rows[45], rows[61], rows[45], rows[61]);
  zip<16>(rows[35], rows[51], rows[35], rows[51]);
  zip<16>(rows[43], rows[59], rows[43], rows[59]);
  zip<16>(rows[39], rows[55], rows[39], rows[55]);
  zip<16>(rows[47], rows[63], rows[47], rows[63]);
  zip<32>(rows[0], rows[32], rows[0], rows[32]);
  zip<32>(rows[16], rows[48], rows[16], rows[48]);
  zip<32>(rows[8], rows[40], rows[8], rows[40]);
  zip<32>(rows[24], rows[56], rows[24], rows[56]);
  zip<32>(rows[4], rows[36], rows[4], rows[36]);
  zip<32>(rows[20], rows[52], rows[20], rows[52]);
  zip<32>(rows[12], rows[44], rows[12], rows[44]);
  zip<32>(rows[28], rows[60], rows[28], rows[60]);
  zip<32>(rows[2], rows[34], rows[2], rows[34]);
  zip<32>(rows[18], rows[50], rows[18], rows[50]);
  zip<32>(rows[10], rows[42], rows[10], rows[42]);
  zip<32>(rows[26], rows[58], rows[26], rows[58]);
  zip<32>(rows[6], rows[38], rows[6], rows[38]);
  zip<32>(rows[22], rows[54], rows[22], rows[54]);
  zip<32>(rows[14], rows[46], rows[14], rows[46]);
  zip<32>(rows[30], rows[62], rows[30], rows[62]);
  zip<32>(rows[1], rows[33], rows[1], rows[33]);
  zip<32>(rows[17], rows[49], rows[17], rows[49]);
  zip<32>(rows[9], rows[41], rows[9], rows[41]);
  zip<32>(rows[25], rows[57], rows[25], rows[57]);
  zip<32>(rows[5], rows[37], rows[5], rows[37]);
  zip<32>(rows[21], rows[53], rows[21], rows[53]);
  zip<32>(rows[13], rows[45], rows[13], rows[45]);
  zip<32>(rows[29], rows[61], rows[29], rows[61]);
  zip<32>(rows[3], rows[35], rows[3], rows[35]);
  zip<32>(rows[19], rows[51], rows[19], rows[51]);
  zip<32>(rows[11], rows[43], rows[11], rows[43]);
  zip<32>(rows[27], rows[59], rows[27], rows[59]);
  zip<32>(rows[7], rows[39], rows[7], rows[39]);
  zip<32>(rows[23], rows[55], rows[23], rows[55]);
  zip<32>(rows[15], rows[47], rows[15], rows[47]);
  zip<32>(rows[31], rows[63], rows[31], rows[63]);
  std::swap(rows[1], rows[32]);
  std::swap(rows[2], rows[16]);
  std::swap(rows[3], rows[48]);
  std::swap(rows[4], rows[8]);
  std::swap(rows[5], rows[40]);
  std::swap(rows[6], rows[24]);
  std::swap(rows[7], rows[56]);
  std::swap(rows[9], rows[36]);
  std::swap(rows[10], rows[20]);
  std::swap(rows[11], rows[52]);
  std::swap(rows[13], rows[44]);
  std::swap(rows[14], rows[28]);
  std::swap(rows[15], rows[60]);
  std::swap(rows[17], rows[34]);
  std::swap(rows[19], rows[50]);
  std::swap(rows[21], rows[42]);
  std::swap(rows[22], rows[26]);
  std::swap(rows[23], rows[58]);
  std::swap(rows[25], rows[38]);
  std::swap(rows[27], rows[54]);
  std::swap(rows[29], rows[46]);
  std::swap(rows[31], rows[62]);
  std::swap(rows[35], rows[49]);
  std::swap(rows[37], rows[41]);
  std::swap(rows[39], rows[57]);
  std::swap(rows[43], rows[53]);
  std::swap(rows[47], rows[61]);
  std::swap(rows[55], rows[59]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose1inplc(rows, Elements<Vec<T, SIMD_WIDTH>::elements>());
}

// ==========================================================
// transpose1inplcLane (1-argument version)
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>, Bytes<16>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<2>(rows[8], rows[10], rows[8], rows[10]);
  zip16<2>(rows[9], rows[11], rows[9], rows[11]);
  zip16<2>(rows[12], rows[14], rows[12], rows[14]);
  zip16<2>(rows[13], rows[15], rows[13], rows[15]);
  zip16<4>(rows[0], rows[4], rows[0], rows[4]);
  zip16<4>(rows[2], rows[6], rows[2], rows[6]);
  zip16<4>(rows[1], rows[5], rows[1], rows[5]);
  zip16<4>(rows[3], rows[7], rows[3], rows[7]);
  zip16<4>(rows[8], rows[12], rows[8], rows[12]);
  zip16<4>(rows[10], rows[14], rows[10], rows[14]);
  zip16<4>(rows[9], rows[13], rows[9], rows[13]);
  zip16<4>(rows[11], rows[15], rows[11], rows[15]);
  zip16<8>(rows[0], rows[8], rows[0], rows[8]);
  zip16<8>(rows[4], rows[12], rows[4], rows[12]);
  zip16<8>(rows[2], rows[10], rows[2], rows[10]);
  zip16<8>(rows[6], rows[14], rows[6], rows[14]);
  zip16<8>(rows[1], rows[9], rows[1], rows[9]);
  zip16<8>(rows[5], rows[13], rows[5], rows[13]);
  zip16<8>(rows[3], rows[11], rows[3], rows[11]);
  zip16<8>(rows[7], rows[15], rows[7], rows[15]);
  std::swap(rows[1], rows[8]);
  std::swap(rows[2], rows[4]);
  std::swap(rows[3], rows[12]);
  std::swap(rows[5], rows[10]);
  std::swap(rows[7], rows[14]);
  std::swap(rows[11], rows[13]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<16>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<4>(rows[0], rows[4], rows[0], rows[4]);
  zip16<4>(rows[2], rows[6], rows[2], rows[6]);
  zip16<4>(rows[1], rows[5], rows[1], rows[5]);
  zip16<4>(rows[3], rows[7], rows[3], rows[7]);
  std::swap(rows[1], rows[4]);
  std::swap(rows[3], rows[6]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<16>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  std::swap(rows[1], rows[2]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<2>, Bytes<16>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<32>, Bytes<32>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  zip16<1>(rows[16], rows[17], rows[16], rows[17]);
  zip16<1>(rows[18], rows[19], rows[18], rows[19]);
  zip16<1>(rows[20], rows[21], rows[20], rows[21]);
  zip16<1>(rows[22], rows[23], rows[22], rows[23]);
  zip16<1>(rows[24], rows[25], rows[24], rows[25]);
  zip16<1>(rows[26], rows[27], rows[26], rows[27]);
  zip16<1>(rows[28], rows[29], rows[28], rows[29]);
  zip16<1>(rows[30], rows[31], rows[30], rows[31]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<2>(rows[8], rows[10], rows[8], rows[10]);
  zip16<2>(rows[9], rows[11], rows[9], rows[11]);
  zip16<2>(rows[12], rows[14], rows[12], rows[14]);
  zip16<2>(rows[13], rows[15], rows[13], rows[15]);
  zip16<2>(rows[16], rows[18], rows[16], rows[18]);
  zip16<2>(rows[17], rows[19], rows[17], rows[19]);
  zip16<2>(rows[20], rows[22], rows[20], rows[22]);
  zip16<2>(rows[21], rows[23], rows[21], rows[23]);
  zip16<2>(rows[24], rows[26], rows[24], rows[26]);
  zip16<2>(rows[25], rows[27], rows[25], rows[27]);
  zip16<2>(rows[28], rows[30], rows[28], rows[30]);
  zip16<2>(rows[29], rows[31], rows[29], rows[31]);
  zip16<4>(rows[0], rows[4], rows[0], rows[4]);
  zip16<4>(rows[2], rows[6], rows[2], rows[6]);
  zip16<4>(rows[1], rows[5], rows[1], rows[5]);
  zip16<4>(rows[3], rows[7], rows[3], rows[7]);
  zip16<4>(rows[8], rows[12], rows[8], rows[12]);
  zip16<4>(rows[10], rows[14], rows[10], rows[14]);
  zip16<4>(rows[9], rows[13], rows[9], rows[13]);
  zip16<4>(rows[11], rows[15], rows[11], rows[15]);
  zip16<4>(rows[16], rows[20], rows[16], rows[20]);
  zip16<4>(rows[18], rows[22], rows[18], rows[22]);
  zip16<4>(rows[17], rows[21], rows[17], rows[21]);
  zip16<4>(rows[19], rows[23], rows[19], rows[23]);
  zip16<4>(rows[24], rows[28], rows[24], rows[28]);
  zip16<4>(rows[26], rows[30], rows[26], rows[30]);
  zip16<4>(rows[25], rows[29], rows[25], rows[29]);
  zip16<4>(rows[27], rows[31], rows[27], rows[31]);
  zip16<8>(rows[0], rows[8], rows[0], rows[8]);
  zip16<8>(rows[4], rows[12], rows[4], rows[12]);
  zip16<8>(rows[2], rows[10], rows[2], rows[10]);
  zip16<8>(rows[6], rows[14], rows[6], rows[14]);
  zip16<8>(rows[1], rows[9], rows[1], rows[9]);
  zip16<8>(rows[5], rows[13], rows[5], rows[13]);
  zip16<8>(rows[3], rows[11], rows[3], rows[11]);
  zip16<8>(rows[7], rows[15], rows[7], rows[15]);
  zip16<8>(rows[16], rows[24], rows[16], rows[24]);
  zip16<8>(rows[20], rows[28], rows[20], rows[28]);
  zip16<8>(rows[18], rows[26], rows[18], rows[26]);
  zip16<8>(rows[22], rows[30], rows[22], rows[30]);
  zip16<8>(rows[17], rows[25], rows[17], rows[25]);
  zip16<8>(rows[21], rows[29], rows[21], rows[29]);
  zip16<8>(rows[19], rows[27], rows[19], rows[27]);
  zip16<8>(rows[23], rows[31], rows[23], rows[31]);
  std::swap(rows[1], rows[8]);
  std::swap(rows[2], rows[4]);
  std::swap(rows[3], rows[12]);
  std::swap(rows[5], rows[10]);
  std::swap(rows[7], rows[14]);
  std::swap(rows[11], rows[13]);
  std::swap(rows[17], rows[24]);
  std::swap(rows[18], rows[20]);
  std::swap(rows[19], rows[28]);
  std::swap(rows[21], rows[26]);
  std::swap(rows[23], rows[30]);
  std::swap(rows[27], rows[29]);
  // correction steps follow below (if required)
  zip<16>(rows[0], rows[16], rows[0], rows[16]);
  zip<16>(rows[1], rows[17], rows[1], rows[17]);
  zip<16>(rows[2], rows[18], rows[2], rows[18]);
  zip<16>(rows[3], rows[19], rows[3], rows[19]);
  zip<16>(rows[4], rows[20], rows[4], rows[20]);
  zip<16>(rows[5], rows[21], rows[5], rows[21]);
  zip<16>(rows[6], rows[22], rows[6], rows[22]);
  zip<16>(rows[7], rows[23], rows[7], rows[23]);
  zip<16>(rows[8], rows[24], rows[8], rows[24]);
  zip<16>(rows[9], rows[25], rows[9], rows[25]);
  zip<16>(rows[10], rows[26], rows[10], rows[26]);
  zip<16>(rows[11], rows[27], rows[11], rows[27]);
  zip<16>(rows[12], rows[28], rows[12], rows[28]);
  zip<16>(rows[13], rows[29], rows[13], rows[29]);
  zip<16>(rows[14], rows[30], rows[14], rows[30]);
  zip<16>(rows[15], rows[31], rows[15], rows[31]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>, Bytes<32>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<2>(rows[8], rows[10], rows[8], rows[10]);
  zip16<2>(rows[9], rows[11], rows[9], rows[11]);
  zip16<2>(rows[12], rows[14], rows[12], rows[14]);
  zip16<2>(rows[13], rows[15], rows[13], rows[15]);
  zip16<4>(rows[0], rows[4], rows[0], rows[4]);
  zip16<4>(rows[2], rows[6], rows[2], rows[6]);
  zip16<4>(rows[1], rows[5], rows[1], rows[5]);
  zip16<4>(rows[3], rows[7], rows[3], rows[7]);
  zip16<4>(rows[8], rows[12], rows[8], rows[12]);
  zip16<4>(rows[10], rows[14], rows[10], rows[14]);
  zip16<4>(rows[9], rows[13], rows[9], rows[13]);
  zip16<4>(rows[11], rows[15], rows[11], rows[15]);
  std::swap(rows[1], rows[4]);
  std::swap(rows[3], rows[6]);
  std::swap(rows[9], rows[12]);
  std::swap(rows[11], rows[14]);
  // correction steps follow below (if required)
  zip<8>(rows[0], rows[8], rows[0], rows[8]);
  zip<8>(rows[1], rows[9], rows[1], rows[9]);
  zip<8>(rows[2], rows[10], rows[2], rows[10]);
  zip<8>(rows[3], rows[11], rows[3], rows[11]);
  zip<8>(rows[4], rows[12], rows[4], rows[12]);
  zip<8>(rows[5], rows[13], rows[5], rows[13]);
  zip<8>(rows[6], rows[14], rows[6], rows[14]);
  zip<8>(rows[7], rows[15], rows[7], rows[15]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<32>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  std::swap(rows[1], rows[2]);
  std::swap(rows[5], rows[6]);
  // correction steps follow below (if required)
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<32>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  // correction steps follow below (if required)
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<64>, Bytes<64>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  zip16<1>(rows[16], rows[17], rows[16], rows[17]);
  zip16<1>(rows[18], rows[19], rows[18], rows[19]);
  zip16<1>(rows[20], rows[21], rows[20], rows[21]);
  zip16<1>(rows[22], rows[23], rows[22], rows[23]);
  zip16<1>(rows[24], rows[25], rows[24], rows[25]);
  zip16<1>(rows[26], rows[27], rows[26], rows[27]);
  zip16<1>(rows[28], rows[29], rows[28], rows[29]);
  zip16<1>(rows[30], rows[31], rows[30], rows[31]);
  zip16<1>(rows[32], rows[33], rows[32], rows[33]);
  zip16<1>(rows[34], rows[35], rows[34], rows[35]);
  zip16<1>(rows[36], rows[37], rows[36], rows[37]);
  zip16<1>(rows[38], rows[39], rows[38], rows[39]);
  zip16<1>(rows[40], rows[41], rows[40], rows[41]);
  zip16<1>(rows[42], rows[43], rows[42], rows[43]);
  zip16<1>(rows[44], rows[45], rows[44], rows[45]);
  zip16<1>(rows[46], rows[47], rows[46], rows[47]);
  zip16<1>(rows[48], rows[49], rows[48], rows[49]);
  zip16<1>(rows[50], rows[51], rows[50], rows[51]);
  zip16<1>(rows[52], rows[53], rows[52], rows[53]);
  zip16<1>(rows[54], rows[55], rows[54], rows[55]);
  zip16<1>(rows[56], rows[57], rows[56], rows[57]);
  zip16<1>(rows[58], rows[59], rows[58], rows[59]);
  zip16<1>(rows[60], rows[61], rows[60], rows[61]);
  zip16<1>(rows[62], rows[63], rows[62], rows[63]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<2>(rows[8], rows[10], rows[8], rows[10]);
  zip16<2>(rows[9], rows[11], rows[9], rows[11]);
  zip16<2>(rows[12], rows[14], rows[12], rows[14]);
  zip16<2>(rows[13], rows[15], rows[13], rows[15]);
  zip16<2>(rows[16], rows[18], rows[16], rows[18]);
  zip16<2>(rows[17], rows[19], rows[17], rows[19]);
  zip16<2>(rows[20], rows[22], rows[20], rows[22]);
  zip16<2>(rows[21], rows[23], rows[21], rows[23]);
  zip16<2>(rows[24], rows[26], rows[24], rows[26]);
  zip16<2>(rows[25], rows[27], rows[25], rows[27]);
  zip16<2>(rows[28], rows[30], rows[28], rows[30]);
  zip16<2>(rows[29], rows[31], rows[29], rows[31]);
  zip16<2>(rows[32], rows[34], rows[32], rows[34]);
  zip16<2>(rows[33], rows[35], rows[33], rows[35]);
  zip16<2>(rows[36], rows[38], rows[36], rows[38]);
  zip16<2>(rows[37], rows[39], rows[37], rows[39]);
  zip16<2>(rows[40], rows[42], rows[40], rows[42]);
  zip16<2>(rows[41], rows[43], rows[41], rows[43]);
  zip16<2>(rows[44], rows[46], rows[44], rows[46]);
  zip16<2>(rows[45], rows[47], rows[45], rows[47]);
  zip16<2>(rows[48], rows[50], rows[48], rows[50]);
  zip16<2>(rows[49], rows[51], rows[49], rows[51]);
  zip16<2>(rows[52], rows[54], rows[52], rows[54]);
  zip16<2>(rows[53], rows[55], rows[53], rows[55]);
  zip16<2>(rows[56], rows[58], rows[56], rows[58]);
  zip16<2>(rows[57], rows[59], rows[57], rows[59]);
  zip16<2>(rows[60], rows[62], rows[60], rows[62]);
  zip16<2>(rows[61], rows[63], rows[61], rows[63]);
  zip16<4>(rows[0], rows[4], rows[0], rows[4]);
  zip16<4>(rows[2], rows[6], rows[2], rows[6]);
  zip16<4>(rows[1], rows[5], rows[1], rows[5]);
  zip16<4>(rows[3], rows[7], rows[3], rows[7]);
  zip16<4>(rows[8], rows[12], rows[8], rows[12]);
  zip16<4>(rows[10], rows[14], rows[10], rows[14]);
  zip16<4>(rows[9], rows[13], rows[9], rows[13]);
  zip16<4>(rows[11], rows[15], rows[11], rows[15]);
  zip16<4>(rows[16], rows[20], rows[16], rows[20]);
  zip16<4>(rows[18], rows[22], rows[18], rows[22]);
  zip16<4>(rows[17], rows[21], rows[17], rows[21]);
  zip16<4>(rows[19], rows[23], rows[19], rows[23]);
  zip16<4>(rows[24], rows[28], rows[24], rows[28]);
  zip16<4>(rows[26], rows[30], rows[26], rows[30]);
  zip16<4>(rows[25], rows[29], rows[25], rows[29]);
  zip16<4>(rows[27], rows[31], rows[27], rows[31]);
  zip16<4>(rows[32], rows[36], rows[32], rows[36]);
  zip16<4>(rows[34], rows[38], rows[34], rows[38]);
  zip16<4>(rows[33], rows[37], rows[33], rows[37]);
  zip16<4>(rows[35], rows[39], rows[35], rows[39]);
  zip16<4>(rows[40], rows[44], rows[40], rows[44]);
  zip16<4>(rows[42], rows[46], rows[42], rows[46]);
  zip16<4>(rows[41], rows[45], rows[41], rows[45]);
  zip16<4>(rows[43], rows[47], rows[43], rows[47]);
  zip16<4>(rows[48], rows[52], rows[48], rows[52]);
  zip16<4>(rows[50], rows[54], rows[50], rows[54]);
  zip16<4>(rows[49], rows[53], rows[49], rows[53]);
  zip16<4>(rows[51], rows[55], rows[51], rows[55]);
  zip16<4>(rows[56], rows[60], rows[56], rows[60]);
  zip16<4>(rows[58], rows[62], rows[58], rows[62]);
  zip16<4>(rows[57], rows[61], rows[57], rows[61]);
  zip16<4>(rows[59], rows[63], rows[59], rows[63]);
  zip16<8>(rows[0], rows[8], rows[0], rows[8]);
  zip16<8>(rows[4], rows[12], rows[4], rows[12]);
  zip16<8>(rows[2], rows[10], rows[2], rows[10]);
  zip16<8>(rows[6], rows[14], rows[6], rows[14]);
  zip16<8>(rows[1], rows[9], rows[1], rows[9]);
  zip16<8>(rows[5], rows[13], rows[5], rows[13]);
  zip16<8>(rows[3], rows[11], rows[3], rows[11]);
  zip16<8>(rows[7], rows[15], rows[7], rows[15]);
  zip16<8>(rows[16], rows[24], rows[16], rows[24]);
  zip16<8>(rows[20], rows[28], rows[20], rows[28]);
  zip16<8>(rows[18], rows[26], rows[18], rows[26]);
  zip16<8>(rows[22], rows[30], rows[22], rows[30]);
  zip16<8>(rows[17], rows[25], rows[17], rows[25]);
  zip16<8>(rows[21], rows[29], rows[21], rows[29]);
  zip16<8>(rows[19], rows[27], rows[19], rows[27]);
  zip16<8>(rows[23], rows[31], rows[23], rows[31]);
  zip16<8>(rows[32], rows[40], rows[32], rows[40]);
  zip16<8>(rows[36], rows[44], rows[36], rows[44]);
  zip16<8>(rows[34], rows[42], rows[34], rows[42]);
  zip16<8>(rows[38], rows[46], rows[38], rows[46]);
  zip16<8>(rows[33], rows[41], rows[33], rows[41]);
  zip16<8>(rows[37], rows[45], rows[37], rows[45]);
  zip16<8>(rows[35], rows[43], rows[35], rows[43]);
  zip16<8>(rows[39], rows[47], rows[39], rows[47]);
  zip16<8>(rows[48], rows[56], rows[48], rows[56]);
  zip16<8>(rows[52], rows[60], rows[52], rows[60]);
  zip16<8>(rows[50], rows[58], rows[50], rows[58]);
  zip16<8>(rows[54], rows[62], rows[54], rows[62]);
  zip16<8>(rows[49], rows[57], rows[49], rows[57]);
  zip16<8>(rows[53], rows[61], rows[53], rows[61]);
  zip16<8>(rows[51], rows[59], rows[51], rows[59]);
  zip16<8>(rows[55], rows[63], rows[55], rows[63]);
  std::swap(rows[1], rows[8]);
  std::swap(rows[2], rows[4]);
  std::swap(rows[3], rows[12]);
  std::swap(rows[5], rows[10]);
  std::swap(rows[7], rows[14]);
  std::swap(rows[11], rows[13]);
  std::swap(rows[17], rows[24]);
  std::swap(rows[18], rows[20]);
  std::swap(rows[19], rows[28]);
  std::swap(rows[21], rows[26]);
  std::swap(rows[23], rows[30]);
  std::swap(rows[27], rows[29]);
  std::swap(rows[33], rows[40]);
  std::swap(rows[34], rows[36]);
  std::swap(rows[35], rows[44]);
  std::swap(rows[37], rows[42]);
  std::swap(rows[39], rows[46]);
  std::swap(rows[43], rows[45]);
  std::swap(rows[49], rows[56]);
  std::swap(rows[50], rows[52]);
  std::swap(rows[51], rows[60]);
  std::swap(rows[53], rows[58]);
  std::swap(rows[55], rows[62]);
  std::swap(rows[59], rows[61]);
  // correction steps follow below (if required)
  zip<16>(rows[0], rows[16], rows[0], rows[16]);
  zip<16>(rows[1], rows[17], rows[1], rows[17]);
  zip<16>(rows[2], rows[18], rows[2], rows[18]);
  zip<16>(rows[3], rows[19], rows[3], rows[19]);
  zip<16>(rows[4], rows[20], rows[4], rows[20]);
  zip<16>(rows[5], rows[21], rows[5], rows[21]);
  zip<16>(rows[6], rows[22], rows[6], rows[22]);
  zip<16>(rows[7], rows[23], rows[7], rows[23]);
  zip<16>(rows[8], rows[24], rows[8], rows[24]);
  zip<16>(rows[9], rows[25], rows[9], rows[25]);
  zip<16>(rows[10], rows[26], rows[10], rows[26]);
  zip<16>(rows[11], rows[27], rows[11], rows[27]);
  zip<16>(rows[12], rows[28], rows[12], rows[28]);
  zip<16>(rows[13], rows[29], rows[13], rows[29]);
  zip<16>(rows[14], rows[30], rows[14], rows[30]);
  zip<16>(rows[15], rows[31], rows[15], rows[31]);
  zip<16>(rows[32], rows[48], rows[32], rows[48]);
  zip<16>(rows[33], rows[49], rows[33], rows[49]);
  zip<16>(rows[34], rows[50], rows[34], rows[50]);
  zip<16>(rows[35], rows[51], rows[35], rows[51]);
  zip<16>(rows[36], rows[52], rows[36], rows[52]);
  zip<16>(rows[37], rows[53], rows[37], rows[53]);
  zip<16>(rows[38], rows[54], rows[38], rows[54]);
  zip<16>(rows[39], rows[55], rows[39], rows[55]);
  zip<16>(rows[40], rows[56], rows[40], rows[56]);
  zip<16>(rows[41], rows[57], rows[41], rows[57]);
  zip<16>(rows[42], rows[58], rows[42], rows[58]);
  zip<16>(rows[43], rows[59], rows[43], rows[59]);
  zip<16>(rows[44], rows[60], rows[44], rows[60]);
  zip<16>(rows[45], rows[61], rows[45], rows[61]);
  zip<16>(rows[46], rows[62], rows[46], rows[62]);
  zip<16>(rows[47], rows[63], rows[47], rows[63]);
  zip<32>(rows[0], rows[32], rows[0], rows[32]);
  zip<32>(rows[1], rows[33], rows[1], rows[33]);
  zip<32>(rows[2], rows[34], rows[2], rows[34]);
  zip<32>(rows[3], rows[35], rows[3], rows[35]);
  zip<32>(rows[4], rows[36], rows[4], rows[36]);
  zip<32>(rows[5], rows[37], rows[5], rows[37]);
  zip<32>(rows[6], rows[38], rows[6], rows[38]);
  zip<32>(rows[7], rows[39], rows[7], rows[39]);
  zip<32>(rows[8], rows[40], rows[8], rows[40]);
  zip<32>(rows[9], rows[41], rows[9], rows[41]);
  zip<32>(rows[10], rows[42], rows[10], rows[42]);
  zip<32>(rows[11], rows[43], rows[11], rows[43]);
  zip<32>(rows[12], rows[44], rows[12], rows[44]);
  zip<32>(rows[13], rows[45], rows[13], rows[45]);
  zip<32>(rows[14], rows[46], rows[14], rows[46]);
  zip<32>(rows[15], rows[47], rows[15], rows[47]);
  zip<32>(rows[16], rows[48], rows[16], rows[48]);
  zip<32>(rows[17], rows[49], rows[17], rows[49]);
  zip<32>(rows[18], rows[50], rows[18], rows[50]);
  zip<32>(rows[19], rows[51], rows[19], rows[51]);
  zip<32>(rows[20], rows[52], rows[20], rows[52]);
  zip<32>(rows[21], rows[53], rows[21], rows[53]);
  zip<32>(rows[22], rows[54], rows[22], rows[54]);
  zip<32>(rows[23], rows[55], rows[23], rows[55]);
  zip<32>(rows[24], rows[56], rows[24], rows[56]);
  zip<32>(rows[25], rows[57], rows[25], rows[57]);
  zip<32>(rows[26], rows[58], rows[26], rows[58]);
  zip<32>(rows[27], rows[59], rows[27], rows[59]);
  zip<32>(rows[28], rows[60], rows[28], rows[60]);
  zip<32>(rows[29], rows[61], rows[29], rows[61]);
  zip<32>(rows[30], rows[62], rows[30], rows[62]);
  zip<32>(rows[31], rows[63], rows[31], rows[63]);
  std::swap(rows[16], rows[32]);
  std::swap(rows[17], rows[33]);
  std::swap(rows[18], rows[34]);
  std::swap(rows[19], rows[35]);
  std::swap(rows[20], rows[36]);
  std::swap(rows[21], rows[37]);
  std::swap(rows[22], rows[38]);
  std::swap(rows[23], rows[39]);
  std::swap(rows[24], rows[40]);
  std::swap(rows[25], rows[41]);
  std::swap(rows[26], rows[42]);
  std::swap(rows[27], rows[43]);
  std::swap(rows[28], rows[44]);
  std::swap(rows[29], rows[45]);
  std::swap(rows[30], rows[46]);
  std::swap(rows[31], rows[47]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<32>, Bytes<64>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  zip16<1>(rows[16], rows[17], rows[16], rows[17]);
  zip16<1>(rows[18], rows[19], rows[18], rows[19]);
  zip16<1>(rows[20], rows[21], rows[20], rows[21]);
  zip16<1>(rows[22], rows[23], rows[22], rows[23]);
  zip16<1>(rows[24], rows[25], rows[24], rows[25]);
  zip16<1>(rows[26], rows[27], rows[26], rows[27]);
  zip16<1>(rows[28], rows[29], rows[28], rows[29]);
  zip16<1>(rows[30], rows[31], rows[30], rows[31]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<2>(rows[8], rows[10], rows[8], rows[10]);
  zip16<2>(rows[9], rows[11], rows[9], rows[11]);
  zip16<2>(rows[12], rows[14], rows[12], rows[14]);
  zip16<2>(rows[13], rows[15], rows[13], rows[15]);
  zip16<2>(rows[16], rows[18], rows[16], rows[18]);
  zip16<2>(rows[17], rows[19], rows[17], rows[19]);
  zip16<2>(rows[20], rows[22], rows[20], rows[22]);
  zip16<2>(rows[21], rows[23], rows[21], rows[23]);
  zip16<2>(rows[24], rows[26], rows[24], rows[26]);
  zip16<2>(rows[25], rows[27], rows[25], rows[27]);
  zip16<2>(rows[28], rows[30], rows[28], rows[30]);
  zip16<2>(rows[29], rows[31], rows[29], rows[31]);
  zip16<4>(rows[0], rows[4], rows[0], rows[4]);
  zip16<4>(rows[2], rows[6], rows[2], rows[6]);
  zip16<4>(rows[1], rows[5], rows[1], rows[5]);
  zip16<4>(rows[3], rows[7], rows[3], rows[7]);
  zip16<4>(rows[8], rows[12], rows[8], rows[12]);
  zip16<4>(rows[10], rows[14], rows[10], rows[14]);
  zip16<4>(rows[9], rows[13], rows[9], rows[13]);
  zip16<4>(rows[11], rows[15], rows[11], rows[15]);
  zip16<4>(rows[16], rows[20], rows[16], rows[20]);
  zip16<4>(rows[18], rows[22], rows[18], rows[22]);
  zip16<4>(rows[17], rows[21], rows[17], rows[21]);
  zip16<4>(rows[19], rows[23], rows[19], rows[23]);
  zip16<4>(rows[24], rows[28], rows[24], rows[28]);
  zip16<4>(rows[26], rows[30], rows[26], rows[30]);
  zip16<4>(rows[25], rows[29], rows[25], rows[29]);
  zip16<4>(rows[27], rows[31], rows[27], rows[31]);
  std::swap(rows[1], rows[4]);
  std::swap(rows[3], rows[6]);
  std::swap(rows[9], rows[12]);
  std::swap(rows[11], rows[14]);
  std::swap(rows[17], rows[20]);
  std::swap(rows[19], rows[22]);
  std::swap(rows[25], rows[28]);
  std::swap(rows[27], rows[30]);
  // correction steps follow below (if required)
  zip<8>(rows[0], rows[8], rows[0], rows[8]);
  zip<8>(rows[1], rows[9], rows[1], rows[9]);
  zip<8>(rows[2], rows[10], rows[2], rows[10]);
  zip<8>(rows[3], rows[11], rows[3], rows[11]);
  zip<8>(rows[4], rows[12], rows[4], rows[12]);
  zip<8>(rows[5], rows[13], rows[5], rows[13]);
  zip<8>(rows[6], rows[14], rows[6], rows[14]);
  zip<8>(rows[7], rows[15], rows[7], rows[15]);
  zip<8>(rows[16], rows[24], rows[16], rows[24]);
  zip<8>(rows[17], rows[25], rows[17], rows[25]);
  zip<8>(rows[18], rows[26], rows[18], rows[26]);
  zip<8>(rows[19], rows[27], rows[19], rows[27]);
  zip<8>(rows[20], rows[28], rows[20], rows[28]);
  zip<8>(rows[21], rows[29], rows[21], rows[29]);
  zip<8>(rows[22], rows[30], rows[22], rows[30]);
  zip<8>(rows[23], rows[31], rows[23], rows[31]);
  zip<16>(rows[0], rows[16], rows[0], rows[16]);
  zip<16>(rows[1], rows[17], rows[1], rows[17]);
  zip<16>(rows[2], rows[18], rows[2], rows[18]);
  zip<16>(rows[3], rows[19], rows[3], rows[19]);
  zip<16>(rows[4], rows[20], rows[4], rows[20]);
  zip<16>(rows[5], rows[21], rows[5], rows[21]);
  zip<16>(rows[6], rows[22], rows[6], rows[22]);
  zip<16>(rows[7], rows[23], rows[7], rows[23]);
  zip<16>(rows[8], rows[24], rows[8], rows[24]);
  zip<16>(rows[9], rows[25], rows[9], rows[25]);
  zip<16>(rows[10], rows[26], rows[10], rows[26]);
  zip<16>(rows[11], rows[27], rows[11], rows[27]);
  zip<16>(rows[12], rows[28], rows[12], rows[28]);
  zip<16>(rows[13], rows[29], rows[13], rows[29]);
  zip<16>(rows[14], rows[30], rows[14], rows[30]);
  zip<16>(rows[15], rows[31], rows[15], rows[31]);
  std::swap(rows[8], rows[16]);
  std::swap(rows[9], rows[17]);
  std::swap(rows[10], rows[18]);
  std::swap(rows[11], rows[19]);
  std::swap(rows[12], rows[20]);
  std::swap(rows[13], rows[21]);
  std::swap(rows[14], rows[22]);
  std::swap(rows[15], rows[23]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>, Bytes<64>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  zip16<2>(rows[0], rows[2], rows[0], rows[2]);
  zip16<2>(rows[1], rows[3], rows[1], rows[3]);
  zip16<2>(rows[4], rows[6], rows[4], rows[6]);
  zip16<2>(rows[5], rows[7], rows[5], rows[7]);
  zip16<2>(rows[8], rows[10], rows[8], rows[10]);
  zip16<2>(rows[9], rows[11], rows[9], rows[11]);
  zip16<2>(rows[12], rows[14], rows[12], rows[14]);
  zip16<2>(rows[13], rows[15], rows[13], rows[15]);
  std::swap(rows[1], rows[2]);
  std::swap(rows[5], rows[6]);
  std::swap(rows[9], rows[10]);
  std::swap(rows[13], rows[14]);
  // correction steps follow below (if required)
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
  zip<4>(rows[8], rows[12], rows[8], rows[12]);
  zip<4>(rows[9], rows[13], rows[9], rows[13]);
  zip<4>(rows[10], rows[14], rows[10], rows[14]);
  zip<4>(rows[11], rows[15], rows[11], rows[15]);
  zip<8>(rows[0], rows[8], rows[0], rows[8]);
  zip<8>(rows[1], rows[9], rows[1], rows[9]);
  zip<8>(rows[2], rows[10], rows[2], rows[10]);
  zip<8>(rows[3], rows[11], rows[3], rows[11]);
  zip<8>(rows[4], rows[12], rows[4], rows[12]);
  zip<8>(rows[5], rows[13], rows[5], rows[13]);
  zip<8>(rows[6], rows[14], rows[6], rows[14]);
  zip<8>(rows[7], rows[15], rows[7], rows[15]);
  std::swap(rows[4], rows[8]);
  std::swap(rows[5], rows[9]);
  std::swap(rows[6], rows[10]);
  std::swap(rows[7], rows[11]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<64>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  // correction steps follow below (if required)
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<4>(rows[0], rows[4], rows[0], rows[4]);
  zip<4>(rows[1], rows[5], rows[1], rows[5]);
  zip<4>(rows[2], rows[6], rows[2], rows[6]);
  zip<4>(rows[3], rows[7], rows[3], rows[7]);
  std::swap(rows[2], rows[4]);
  std::swap(rows[3], rows[5]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose1inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose1inplcLane(rows, Elements<Vec<T, SIMD_WIDTH>::elements>(),
                      Bytes<SIMD_WIDTH>());
}

// ==========================================================
// transpose2inplc (1-argument version)
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<2>)
{
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<4>)
{
  zip<1>(rows[0], rows[2], rows[0], rows[2]);
  zip<1>(rows[1], rows[3], rows[1], rows[3]);
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>)
{
  zip<1>(rows[0], rows[4], rows[0], rows[4]);
  zip<1>(rows[1], rows[5], rows[1], rows[5]);
  zip<1>(rows[2], rows[6], rows[2], rows[6]);
  zip<1>(rows[3], rows[7], rows[3], rows[7]);
  zip<1>(rows[0], rows[2], rows[0], rows[2]);
  zip<1>(rows[4], rows[6], rows[4], rows[6]);
  zip<1>(rows[1], rows[3], rows[1], rows[3]);
  zip<1>(rows[5], rows[7], rows[5], rows[7]);
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>)
{
  zip<1>(rows[0], rows[8], rows[0], rows[8]);
  zip<1>(rows[1], rows[9], rows[1], rows[9]);
  zip<1>(rows[2], rows[10], rows[2], rows[10]);
  zip<1>(rows[3], rows[11], rows[3], rows[11]);
  zip<1>(rows[4], rows[12], rows[4], rows[12]);
  zip<1>(rows[5], rows[13], rows[5], rows[13]);
  zip<1>(rows[6], rows[14], rows[6], rows[14]);
  zip<1>(rows[7], rows[15], rows[7], rows[15]);
  zip<1>(rows[0], rows[4], rows[0], rows[4]);
  zip<1>(rows[8], rows[12], rows[8], rows[12]);
  zip<1>(rows[1], rows[5], rows[1], rows[5]);
  zip<1>(rows[9], rows[13], rows[9], rows[13]);
  zip<1>(rows[2], rows[6], rows[2], rows[6]);
  zip<1>(rows[10], rows[14], rows[10], rows[14]);
  zip<1>(rows[3], rows[7], rows[3], rows[7]);
  zip<1>(rows[11], rows[15], rows[11], rows[15]);
  zip<1>(rows[0], rows[2], rows[0], rows[2]);
  zip<1>(rows[4], rows[6], rows[4], rows[6]);
  zip<1>(rows[8], rows[10], rows[8], rows[10]);
  zip<1>(rows[12], rows[14], rows[12], rows[14]);
  zip<1>(rows[1], rows[3], rows[1], rows[3]);
  zip<1>(rows[5], rows[7], rows[5], rows[7]);
  zip<1>(rows[9], rows[11], rows[9], rows[11]);
  zip<1>(rows[13], rows[15], rows[13], rows[15]);
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<32>)
{
  zip<1>(rows[0], rows[16], rows[0], rows[16]);
  zip<1>(rows[1], rows[17], rows[1], rows[17]);
  zip<1>(rows[2], rows[18], rows[2], rows[18]);
  zip<1>(rows[3], rows[19], rows[3], rows[19]);
  zip<1>(rows[4], rows[20], rows[4], rows[20]);
  zip<1>(rows[5], rows[21], rows[5], rows[21]);
  zip<1>(rows[6], rows[22], rows[6], rows[22]);
  zip<1>(rows[7], rows[23], rows[7], rows[23]);
  zip<1>(rows[8], rows[24], rows[8], rows[24]);
  zip<1>(rows[9], rows[25], rows[9], rows[25]);
  zip<1>(rows[10], rows[26], rows[10], rows[26]);
  zip<1>(rows[11], rows[27], rows[11], rows[27]);
  zip<1>(rows[12], rows[28], rows[12], rows[28]);
  zip<1>(rows[13], rows[29], rows[13], rows[29]);
  zip<1>(rows[14], rows[30], rows[14], rows[30]);
  zip<1>(rows[15], rows[31], rows[15], rows[31]);
  zip<1>(rows[0], rows[8], rows[0], rows[8]);
  zip<1>(rows[16], rows[24], rows[16], rows[24]);
  zip<1>(rows[1], rows[9], rows[1], rows[9]);
  zip<1>(rows[17], rows[25], rows[17], rows[25]);
  zip<1>(rows[2], rows[10], rows[2], rows[10]);
  zip<1>(rows[18], rows[26], rows[18], rows[26]);
  zip<1>(rows[3], rows[11], rows[3], rows[11]);
  zip<1>(rows[19], rows[27], rows[19], rows[27]);
  zip<1>(rows[4], rows[12], rows[4], rows[12]);
  zip<1>(rows[20], rows[28], rows[20], rows[28]);
  zip<1>(rows[5], rows[13], rows[5], rows[13]);
  zip<1>(rows[21], rows[29], rows[21], rows[29]);
  zip<1>(rows[6], rows[14], rows[6], rows[14]);
  zip<1>(rows[22], rows[30], rows[22], rows[30]);
  zip<1>(rows[7], rows[15], rows[7], rows[15]);
  zip<1>(rows[23], rows[31], rows[23], rows[31]);
  zip<1>(rows[0], rows[4], rows[0], rows[4]);
  zip<1>(rows[8], rows[12], rows[8], rows[12]);
  zip<1>(rows[16], rows[20], rows[16], rows[20]);
  zip<1>(rows[24], rows[28], rows[24], rows[28]);
  zip<1>(rows[1], rows[5], rows[1], rows[5]);
  zip<1>(rows[9], rows[13], rows[9], rows[13]);
  zip<1>(rows[17], rows[21], rows[17], rows[21]);
  zip<1>(rows[25], rows[29], rows[25], rows[29]);
  zip<1>(rows[2], rows[6], rows[2], rows[6]);
  zip<1>(rows[10], rows[14], rows[10], rows[14]);
  zip<1>(rows[18], rows[22], rows[18], rows[22]);
  zip<1>(rows[26], rows[30], rows[26], rows[30]);
  zip<1>(rows[3], rows[7], rows[3], rows[7]);
  zip<1>(rows[11], rows[15], rows[11], rows[15]);
  zip<1>(rows[19], rows[23], rows[19], rows[23]);
  zip<1>(rows[27], rows[31], rows[27], rows[31]);
  zip<1>(rows[0], rows[2], rows[0], rows[2]);
  zip<1>(rows[4], rows[6], rows[4], rows[6]);
  zip<1>(rows[8], rows[10], rows[8], rows[10]);
  zip<1>(rows[12], rows[14], rows[12], rows[14]);
  zip<1>(rows[16], rows[18], rows[16], rows[18]);
  zip<1>(rows[20], rows[22], rows[20], rows[22]);
  zip<1>(rows[24], rows[26], rows[24], rows[26]);
  zip<1>(rows[28], rows[30], rows[28], rows[30]);
  zip<1>(rows[1], rows[3], rows[1], rows[3]);
  zip<1>(rows[5], rows[7], rows[5], rows[7]);
  zip<1>(rows[9], rows[11], rows[9], rows[11]);
  zip<1>(rows[13], rows[15], rows[13], rows[15]);
  zip<1>(rows[17], rows[19], rows[17], rows[19]);
  zip<1>(rows[21], rows[23], rows[21], rows[23]);
  zip<1>(rows[25], rows[27], rows[25], rows[27]);
  zip<1>(rows[29], rows[31], rows[29], rows[31]);
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<64>)
{
  zip<1>(rows[0], rows[32], rows[0], rows[32]);
  zip<1>(rows[1], rows[33], rows[1], rows[33]);
  zip<1>(rows[2], rows[34], rows[2], rows[34]);
  zip<1>(rows[3], rows[35], rows[3], rows[35]);
  zip<1>(rows[4], rows[36], rows[4], rows[36]);
  zip<1>(rows[5], rows[37], rows[5], rows[37]);
  zip<1>(rows[6], rows[38], rows[6], rows[38]);
  zip<1>(rows[7], rows[39], rows[7], rows[39]);
  zip<1>(rows[8], rows[40], rows[8], rows[40]);
  zip<1>(rows[9], rows[41], rows[9], rows[41]);
  zip<1>(rows[10], rows[42], rows[10], rows[42]);
  zip<1>(rows[11], rows[43], rows[11], rows[43]);
  zip<1>(rows[12], rows[44], rows[12], rows[44]);
  zip<1>(rows[13], rows[45], rows[13], rows[45]);
  zip<1>(rows[14], rows[46], rows[14], rows[46]);
  zip<1>(rows[15], rows[47], rows[15], rows[47]);
  zip<1>(rows[16], rows[48], rows[16], rows[48]);
  zip<1>(rows[17], rows[49], rows[17], rows[49]);
  zip<1>(rows[18], rows[50], rows[18], rows[50]);
  zip<1>(rows[19], rows[51], rows[19], rows[51]);
  zip<1>(rows[20], rows[52], rows[20], rows[52]);
  zip<1>(rows[21], rows[53], rows[21], rows[53]);
  zip<1>(rows[22], rows[54], rows[22], rows[54]);
  zip<1>(rows[23], rows[55], rows[23], rows[55]);
  zip<1>(rows[24], rows[56], rows[24], rows[56]);
  zip<1>(rows[25], rows[57], rows[25], rows[57]);
  zip<1>(rows[26], rows[58], rows[26], rows[58]);
  zip<1>(rows[27], rows[59], rows[27], rows[59]);
  zip<1>(rows[28], rows[60], rows[28], rows[60]);
  zip<1>(rows[29], rows[61], rows[29], rows[61]);
  zip<1>(rows[30], rows[62], rows[30], rows[62]);
  zip<1>(rows[31], rows[63], rows[31], rows[63]);
  zip<1>(rows[0], rows[16], rows[0], rows[16]);
  zip<1>(rows[32], rows[48], rows[32], rows[48]);
  zip<1>(rows[1], rows[17], rows[1], rows[17]);
  zip<1>(rows[33], rows[49], rows[33], rows[49]);
  zip<1>(rows[2], rows[18], rows[2], rows[18]);
  zip<1>(rows[34], rows[50], rows[34], rows[50]);
  zip<1>(rows[3], rows[19], rows[3], rows[19]);
  zip<1>(rows[35], rows[51], rows[35], rows[51]);
  zip<1>(rows[4], rows[20], rows[4], rows[20]);
  zip<1>(rows[36], rows[52], rows[36], rows[52]);
  zip<1>(rows[5], rows[21], rows[5], rows[21]);
  zip<1>(rows[37], rows[53], rows[37], rows[53]);
  zip<1>(rows[6], rows[22], rows[6], rows[22]);
  zip<1>(rows[38], rows[54], rows[38], rows[54]);
  zip<1>(rows[7], rows[23], rows[7], rows[23]);
  zip<1>(rows[39], rows[55], rows[39], rows[55]);
  zip<1>(rows[8], rows[24], rows[8], rows[24]);
  zip<1>(rows[40], rows[56], rows[40], rows[56]);
  zip<1>(rows[9], rows[25], rows[9], rows[25]);
  zip<1>(rows[41], rows[57], rows[41], rows[57]);
  zip<1>(rows[10], rows[26], rows[10], rows[26]);
  zip<1>(rows[42], rows[58], rows[42], rows[58]);
  zip<1>(rows[11], rows[27], rows[11], rows[27]);
  zip<1>(rows[43], rows[59], rows[43], rows[59]);
  zip<1>(rows[12], rows[28], rows[12], rows[28]);
  zip<1>(rows[44], rows[60], rows[44], rows[60]);
  zip<1>(rows[13], rows[29], rows[13], rows[29]);
  zip<1>(rows[45], rows[61], rows[45], rows[61]);
  zip<1>(rows[14], rows[30], rows[14], rows[30]);
  zip<1>(rows[46], rows[62], rows[46], rows[62]);
  zip<1>(rows[15], rows[31], rows[15], rows[31]);
  zip<1>(rows[47], rows[63], rows[47], rows[63]);
  zip<1>(rows[0], rows[8], rows[0], rows[8]);
  zip<1>(rows[16], rows[24], rows[16], rows[24]);
  zip<1>(rows[32], rows[40], rows[32], rows[40]);
  zip<1>(rows[48], rows[56], rows[48], rows[56]);
  zip<1>(rows[1], rows[9], rows[1], rows[9]);
  zip<1>(rows[17], rows[25], rows[17], rows[25]);
  zip<1>(rows[33], rows[41], rows[33], rows[41]);
  zip<1>(rows[49], rows[57], rows[49], rows[57]);
  zip<1>(rows[2], rows[10], rows[2], rows[10]);
  zip<1>(rows[18], rows[26], rows[18], rows[26]);
  zip<1>(rows[34], rows[42], rows[34], rows[42]);
  zip<1>(rows[50], rows[58], rows[50], rows[58]);
  zip<1>(rows[3], rows[11], rows[3], rows[11]);
  zip<1>(rows[19], rows[27], rows[19], rows[27]);
  zip<1>(rows[35], rows[43], rows[35], rows[43]);
  zip<1>(rows[51], rows[59], rows[51], rows[59]);
  zip<1>(rows[4], rows[12], rows[4], rows[12]);
  zip<1>(rows[20], rows[28], rows[20], rows[28]);
  zip<1>(rows[36], rows[44], rows[36], rows[44]);
  zip<1>(rows[52], rows[60], rows[52], rows[60]);
  zip<1>(rows[5], rows[13], rows[5], rows[13]);
  zip<1>(rows[21], rows[29], rows[21], rows[29]);
  zip<1>(rows[37], rows[45], rows[37], rows[45]);
  zip<1>(rows[53], rows[61], rows[53], rows[61]);
  zip<1>(rows[6], rows[14], rows[6], rows[14]);
  zip<1>(rows[22], rows[30], rows[22], rows[30]);
  zip<1>(rows[38], rows[46], rows[38], rows[46]);
  zip<1>(rows[54], rows[62], rows[54], rows[62]);
  zip<1>(rows[7], rows[15], rows[7], rows[15]);
  zip<1>(rows[23], rows[31], rows[23], rows[31]);
  zip<1>(rows[39], rows[47], rows[39], rows[47]);
  zip<1>(rows[55], rows[63], rows[55], rows[63]);
  zip<1>(rows[0], rows[4], rows[0], rows[4]);
  zip<1>(rows[8], rows[12], rows[8], rows[12]);
  zip<1>(rows[16], rows[20], rows[16], rows[20]);
  zip<1>(rows[24], rows[28], rows[24], rows[28]);
  zip<1>(rows[32], rows[36], rows[32], rows[36]);
  zip<1>(rows[40], rows[44], rows[40], rows[44]);
  zip<1>(rows[48], rows[52], rows[48], rows[52]);
  zip<1>(rows[56], rows[60], rows[56], rows[60]);
  zip<1>(rows[1], rows[5], rows[1], rows[5]);
  zip<1>(rows[9], rows[13], rows[9], rows[13]);
  zip<1>(rows[17], rows[21], rows[17], rows[21]);
  zip<1>(rows[25], rows[29], rows[25], rows[29]);
  zip<1>(rows[33], rows[37], rows[33], rows[37]);
  zip<1>(rows[41], rows[45], rows[41], rows[45]);
  zip<1>(rows[49], rows[53], rows[49], rows[53]);
  zip<1>(rows[57], rows[61], rows[57], rows[61]);
  zip<1>(rows[2], rows[6], rows[2], rows[6]);
  zip<1>(rows[10], rows[14], rows[10], rows[14]);
  zip<1>(rows[18], rows[22], rows[18], rows[22]);
  zip<1>(rows[26], rows[30], rows[26], rows[30]);
  zip<1>(rows[34], rows[38], rows[34], rows[38]);
  zip<1>(rows[42], rows[46], rows[42], rows[46]);
  zip<1>(rows[50], rows[54], rows[50], rows[54]);
  zip<1>(rows[58], rows[62], rows[58], rows[62]);
  zip<1>(rows[3], rows[7], rows[3], rows[7]);
  zip<1>(rows[11], rows[15], rows[11], rows[15]);
  zip<1>(rows[19], rows[23], rows[19], rows[23]);
  zip<1>(rows[27], rows[31], rows[27], rows[31]);
  zip<1>(rows[35], rows[39], rows[35], rows[39]);
  zip<1>(rows[43], rows[47], rows[43], rows[47]);
  zip<1>(rows[51], rows[55], rows[51], rows[55]);
  zip<1>(rows[59], rows[63], rows[59], rows[63]);
  zip<1>(rows[0], rows[2], rows[0], rows[2]);
  zip<1>(rows[4], rows[6], rows[4], rows[6]);
  zip<1>(rows[8], rows[10], rows[8], rows[10]);
  zip<1>(rows[12], rows[14], rows[12], rows[14]);
  zip<1>(rows[16], rows[18], rows[16], rows[18]);
  zip<1>(rows[20], rows[22], rows[20], rows[22]);
  zip<1>(rows[24], rows[26], rows[24], rows[26]);
  zip<1>(rows[28], rows[30], rows[28], rows[30]);
  zip<1>(rows[32], rows[34], rows[32], rows[34]);
  zip<1>(rows[36], rows[38], rows[36], rows[38]);
  zip<1>(rows[40], rows[42], rows[40], rows[42]);
  zip<1>(rows[44], rows[46], rows[44], rows[46]);
  zip<1>(rows[48], rows[50], rows[48], rows[50]);
  zip<1>(rows[52], rows[54], rows[52], rows[54]);
  zip<1>(rows[56], rows[58], rows[56], rows[58]);
  zip<1>(rows[60], rows[62], rows[60], rows[62]);
  zip<1>(rows[1], rows[3], rows[1], rows[3]);
  zip<1>(rows[5], rows[7], rows[5], rows[7]);
  zip<1>(rows[9], rows[11], rows[9], rows[11]);
  zip<1>(rows[13], rows[15], rows[13], rows[15]);
  zip<1>(rows[17], rows[19], rows[17], rows[19]);
  zip<1>(rows[21], rows[23], rows[21], rows[23]);
  zip<1>(rows[25], rows[27], rows[25], rows[27]);
  zip<1>(rows[29], rows[31], rows[29], rows[31]);
  zip<1>(rows[33], rows[35], rows[33], rows[35]);
  zip<1>(rows[37], rows[39], rows[37], rows[39]);
  zip<1>(rows[41], rows[43], rows[41], rows[43]);
  zip<1>(rows[45], rows[47], rows[45], rows[47]);
  zip<1>(rows[49], rows[51], rows[49], rows[51]);
  zip<1>(rows[53], rows[55], rows[53], rows[55]);
  zip<1>(rows[57], rows[59], rows[57], rows[59]);
  zip<1>(rows[61], rows[63], rows[61], rows[63]);
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
  zip<1>(rows[32], rows[33], rows[32], rows[33]);
  zip<1>(rows[34], rows[35], rows[34], rows[35]);
  zip<1>(rows[36], rows[37], rows[36], rows[37]);
  zip<1>(rows[38], rows[39], rows[38], rows[39]);
  zip<1>(rows[40], rows[41], rows[40], rows[41]);
  zip<1>(rows[42], rows[43], rows[42], rows[43]);
  zip<1>(rows[44], rows[45], rows[44], rows[45]);
  zip<1>(rows[46], rows[47], rows[46], rows[47]);
  zip<1>(rows[48], rows[49], rows[48], rows[49]);
  zip<1>(rows[50], rows[51], rows[50], rows[51]);
  zip<1>(rows[52], rows[53], rows[52], rows[53]);
  zip<1>(rows[54], rows[55], rows[54], rows[55]);
  zip<1>(rows[56], rows[57], rows[56], rows[57]);
  zip<1>(rows[58], rows[59], rows[58], rows[59]);
  zip<1>(rows[60], rows[61], rows[60], rows[61]);
  zip<1>(rows[62], rows[63], rows[62], rows[63]);
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplc(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose2inplc(rows, Elements<Vec<T, SIMD_WIDTH>::elements>());
}

// ==========================================================
// transpose2inplcLane (1-argument version)
// ==========================================================

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>, Bytes<16>)
{
  zip16<1>(rows[0], rows[8], rows[0], rows[8]);
  zip16<1>(rows[1], rows[9], rows[1], rows[9]);
  zip16<1>(rows[2], rows[10], rows[2], rows[10]);
  zip16<1>(rows[3], rows[11], rows[3], rows[11]);
  zip16<1>(rows[4], rows[12], rows[4], rows[12]);
  zip16<1>(rows[5], rows[13], rows[5], rows[13]);
  zip16<1>(rows[6], rows[14], rows[6], rows[14]);
  zip16<1>(rows[7], rows[15], rows[7], rows[15]);
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[8], rows[12], rows[8], rows[12]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[9], rows[13], rows[9], rows[13]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[10], rows[14], rows[10], rows[14]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[11], rows[15], rows[11], rows[15]);
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[4], rows[6], rows[4], rows[6]);
  zip16<1>(rows[8], rows[10], rows[8], rows[10]);
  zip16<1>(rows[12], rows[14], rows[12], rows[14]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  zip16<1>(rows[5], rows[7], rows[5], rows[7]);
  zip16<1>(rows[9], rows[11], rows[9], rows[11]);
  zip16<1>(rows[13], rows[15], rows[13], rows[15]);
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  zip16<1>(rows[8], rows[9], rows[8], rows[9]);
  zip16<1>(rows[10], rows[11], rows[10], rows[11]);
  zip16<1>(rows[12], rows[13], rows[12], rows[13]);
  zip16<1>(rows[14], rows[15], rows[14], rows[15]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<16>)
{
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[4], rows[6], rows[4], rows[6]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  zip16<1>(rows[5], rows[7], rows[5], rows[7]);
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  zip16<1>(rows[4], rows[5], rows[4], rows[5]);
  zip16<1>(rows[6], rows[7], rows[6], rows[7]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<16>)
{
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  zip16<1>(rows[2], rows[3], rows[2], rows[3]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<2>, Bytes<16>)
{
  zip16<1>(rows[0], rows[1], rows[0], rows[1]);
  // correction steps follow below (if required)
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<32>, Bytes<32>)
{
  zip16<1>(rows[0], rows[16], rows[0], rows[16]);
  zip16<1>(rows[1], rows[17], rows[1], rows[17]);
  zip16<1>(rows[2], rows[18], rows[2], rows[18]);
  zip16<1>(rows[3], rows[19], rows[3], rows[19]);
  zip16<1>(rows[4], rows[20], rows[4], rows[20]);
  zip16<1>(rows[5], rows[21], rows[5], rows[21]);
  zip16<1>(rows[6], rows[22], rows[6], rows[22]);
  zip16<1>(rows[7], rows[23], rows[7], rows[23]);
  zip16<1>(rows[8], rows[24], rows[8], rows[24]);
  zip16<1>(rows[9], rows[25], rows[9], rows[25]);
  zip16<1>(rows[10], rows[26], rows[10], rows[26]);
  zip16<1>(rows[11], rows[27], rows[11], rows[27]);
  zip16<1>(rows[12], rows[28], rows[12], rows[28]);
  zip16<1>(rows[13], rows[29], rows[13], rows[29]);
  zip16<1>(rows[14], rows[30], rows[14], rows[30]);
  zip16<1>(rows[15], rows[31], rows[15], rows[31]);
  zip16<1>(rows[0], rows[8], rows[0], rows[8]);
  zip16<1>(rows[16], rows[24], rows[16], rows[24]);
  zip16<1>(rows[1], rows[9], rows[1], rows[9]);
  zip16<1>(rows[17], rows[25], rows[17], rows[25]);
  zip16<1>(rows[2], rows[10], rows[2], rows[10]);
  zip16<1>(rows[18], rows[26], rows[18], rows[26]);
  zip16<1>(rows[3], rows[11], rows[3], rows[11]);
  zip16<1>(rows[19], rows[27], rows[19], rows[27]);
  zip16<1>(rows[4], rows[12], rows[4], rows[12]);
  zip16<1>(rows[20], rows[28], rows[20], rows[28]);
  zip16<1>(rows[5], rows[13], rows[5], rows[13]);
  zip16<1>(rows[21], rows[29], rows[21], rows[29]);
  zip16<1>(rows[6], rows[14], rows[6], rows[14]);
  zip16<1>(rows[22], rows[30], rows[22], rows[30]);
  zip16<1>(rows[7], rows[15], rows[7], rows[15]);
  zip16<1>(rows[23], rows[31], rows[23], rows[31]);
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[8], rows[12], rows[8], rows[12]);
  zip16<1>(rows[16], rows[20], rows[16], rows[20]);
  zip16<1>(rows[24], rows[28], rows[24], rows[28]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[9], rows[13], rows[9], rows[13]);
  zip16<1>(rows[17], rows[21], rows[17], rows[21]);
  zip16<1>(rows[25], rows[29], rows[25], rows[29]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[10], rows[14], rows[10], rows[14]);
  zip16<1>(rows[18], rows[22], rows[18], rows[22]);
  zip16<1>(rows[26], rows[30], rows[26], rows[30]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[11], rows[15], rows[11], rows[15]);
  zip16<1>(rows[19], rows[23], rows[19], rows[23]);
  zip16<1>(rows[27], rows[31], rows[27], rows[31]);
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[4], rows[6], rows[4], rows[6]);
  zip16<1>(rows[8], rows[10], rows[8], rows[10]);
  zip16<1>(rows[12], rows[14], rows[12], rows[14]);
  zip16<1>(rows[16], rows[18], rows[16], rows[18]);
  zip16<1>(rows[20], rows[22], rows[20], rows[22]);
  zip16<1>(rows[24], rows[26], rows[24], rows[26]);
  zip16<1>(rows[28], rows[30], rows[28], rows[30]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  zip16<1>(rows[5], rows[7], rows[5], rows[7]);
  zip16<1>(rows[9], rows[11], rows[9], rows[11]);
  zip16<1>(rows[13], rows[15], rows[13], rows[15]);
  zip16<1>(rows[17], rows[19], rows[17], rows[19]);
  zip16<1>(rows[21], rows[23], rows[21], rows[23]);
  zip16<1>(rows[25], rows[27], rows[25], rows[27]);
  zip16<1>(rows[29], rows[31], rows[29], rows[31]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[2];
    rows[2]                  = rows[4];
    rows[4]                  = rows[8];
    rows[8]                  = rows[16];
    rows[16]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[6];
    rows[6]                  = rows[12];
    rows[12]                 = rows[24];
    rows[24]                 = rows[17];
    rows[17]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[5];
    rows[5]                  = rows[10];
    rows[10]                 = rows[20];
    rows[20]                 = rows[9];
    rows[9]                  = rows[18];
    rows[18]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[7];
    rows[7]                  = rows[14];
    rows[14]                 = rows[28];
    rows[28]                 = rows[25];
    rows[25]                 = rows[19];
    rows[19]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[11];
    rows[11]                 = rows[22];
    rows[22]                 = rows[13];
    rows[13]                 = rows[26];
    rows[26]                 = rows[21];
    rows[21]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[15];
    rows[15]                 = rows[30];
    rows[30]                 = rows[29];
    rows[29]                 = rows[27];
    rows[27]                 = rows[23];
    rows[23]                 = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>, Bytes<32>)
{
  zip16<1>(rows[0], rows[8], rows[0], rows[8]);
  zip16<1>(rows[1], rows[9], rows[1], rows[9]);
  zip16<1>(rows[2], rows[10], rows[2], rows[10]);
  zip16<1>(rows[3], rows[11], rows[3], rows[11]);
  zip16<1>(rows[4], rows[12], rows[4], rows[12]);
  zip16<1>(rows[5], rows[13], rows[5], rows[13]);
  zip16<1>(rows[6], rows[14], rows[6], rows[14]);
  zip16<1>(rows[7], rows[15], rows[7], rows[15]);
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[8], rows[12], rows[8], rows[12]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[9], rows[13], rows[9], rows[13]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[10], rows[14], rows[10], rows[14]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[11], rows[15], rows[11], rows[15]);
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[4], rows[6], rows[4], rows[6]);
  zip16<1>(rows[8], rows[10], rows[8], rows[10]);
  zip16<1>(rows[12], rows[14], rows[12], rows[14]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  zip16<1>(rows[5], rows[7], rows[5], rows[7]);
  zip16<1>(rows[9], rows[11], rows[9], rows[11]);
  zip16<1>(rows[13], rows[15], rows[13], rows[15]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[2];
    rows[2]                  = rows[4];
    rows[4]                  = rows[8];
    rows[8]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[6];
    rows[6]                  = rows[12];
    rows[12]                 = rows[9];
    rows[9]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[5];
    rows[5]                  = rows[10];
    rows[10]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[7];
    rows[7]                  = rows[14];
    rows[14]                 = rows[13];
    rows[13]                 = rows[11];
    rows[11]                 = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<32>)
{
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[4], rows[6], rows[4], rows[6]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  zip16<1>(rows[5], rows[7], rows[5], rows[7]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[2];
    rows[2]                  = rows[4];
    rows[4]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[6];
    rows[6]                  = rows[5];
    rows[5]                  = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<4>, Bytes<32>)
{
  zip16<1>(rows[0], rows[2], rows[0], rows[2]);
  zip16<1>(rows[1], rows[3], rows[1], rows[3]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[2];
    rows[2]                  = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<64>, Bytes<64>)
{
  zip16<1>(rows[0], rows[32], rows[0], rows[32]);
  zip16<1>(rows[1], rows[33], rows[1], rows[33]);
  zip16<1>(rows[2], rows[34], rows[2], rows[34]);
  zip16<1>(rows[3], rows[35], rows[3], rows[35]);
  zip16<1>(rows[4], rows[36], rows[4], rows[36]);
  zip16<1>(rows[5], rows[37], rows[5], rows[37]);
  zip16<1>(rows[6], rows[38], rows[6], rows[38]);
  zip16<1>(rows[7], rows[39], rows[7], rows[39]);
  zip16<1>(rows[8], rows[40], rows[8], rows[40]);
  zip16<1>(rows[9], rows[41], rows[9], rows[41]);
  zip16<1>(rows[10], rows[42], rows[10], rows[42]);
  zip16<1>(rows[11], rows[43], rows[11], rows[43]);
  zip16<1>(rows[12], rows[44], rows[12], rows[44]);
  zip16<1>(rows[13], rows[45], rows[13], rows[45]);
  zip16<1>(rows[14], rows[46], rows[14], rows[46]);
  zip16<1>(rows[15], rows[47], rows[15], rows[47]);
  zip16<1>(rows[16], rows[48], rows[16], rows[48]);
  zip16<1>(rows[17], rows[49], rows[17], rows[49]);
  zip16<1>(rows[18], rows[50], rows[18], rows[50]);
  zip16<1>(rows[19], rows[51], rows[19], rows[51]);
  zip16<1>(rows[20], rows[52], rows[20], rows[52]);
  zip16<1>(rows[21], rows[53], rows[21], rows[53]);
  zip16<1>(rows[22], rows[54], rows[22], rows[54]);
  zip16<1>(rows[23], rows[55], rows[23], rows[55]);
  zip16<1>(rows[24], rows[56], rows[24], rows[56]);
  zip16<1>(rows[25], rows[57], rows[25], rows[57]);
  zip16<1>(rows[26], rows[58], rows[26], rows[58]);
  zip16<1>(rows[27], rows[59], rows[27], rows[59]);
  zip16<1>(rows[28], rows[60], rows[28], rows[60]);
  zip16<1>(rows[29], rows[61], rows[29], rows[61]);
  zip16<1>(rows[30], rows[62], rows[30], rows[62]);
  zip16<1>(rows[31], rows[63], rows[31], rows[63]);
  zip16<1>(rows[0], rows[16], rows[0], rows[16]);
  zip16<1>(rows[32], rows[48], rows[32], rows[48]);
  zip16<1>(rows[1], rows[17], rows[1], rows[17]);
  zip16<1>(rows[33], rows[49], rows[33], rows[49]);
  zip16<1>(rows[2], rows[18], rows[2], rows[18]);
  zip16<1>(rows[34], rows[50], rows[34], rows[50]);
  zip16<1>(rows[3], rows[19], rows[3], rows[19]);
  zip16<1>(rows[35], rows[51], rows[35], rows[51]);
  zip16<1>(rows[4], rows[20], rows[4], rows[20]);
  zip16<1>(rows[36], rows[52], rows[36], rows[52]);
  zip16<1>(rows[5], rows[21], rows[5], rows[21]);
  zip16<1>(rows[37], rows[53], rows[37], rows[53]);
  zip16<1>(rows[6], rows[22], rows[6], rows[22]);
  zip16<1>(rows[38], rows[54], rows[38], rows[54]);
  zip16<1>(rows[7], rows[23], rows[7], rows[23]);
  zip16<1>(rows[39], rows[55], rows[39], rows[55]);
  zip16<1>(rows[8], rows[24], rows[8], rows[24]);
  zip16<1>(rows[40], rows[56], rows[40], rows[56]);
  zip16<1>(rows[9], rows[25], rows[9], rows[25]);
  zip16<1>(rows[41], rows[57], rows[41], rows[57]);
  zip16<1>(rows[10], rows[26], rows[10], rows[26]);
  zip16<1>(rows[42], rows[58], rows[42], rows[58]);
  zip16<1>(rows[11], rows[27], rows[11], rows[27]);
  zip16<1>(rows[43], rows[59], rows[43], rows[59]);
  zip16<1>(rows[12], rows[28], rows[12], rows[28]);
  zip16<1>(rows[44], rows[60], rows[44], rows[60]);
  zip16<1>(rows[13], rows[29], rows[13], rows[29]);
  zip16<1>(rows[45], rows[61], rows[45], rows[61]);
  zip16<1>(rows[14], rows[30], rows[14], rows[30]);
  zip16<1>(rows[46], rows[62], rows[46], rows[62]);
  zip16<1>(rows[15], rows[31], rows[15], rows[31]);
  zip16<1>(rows[47], rows[63], rows[47], rows[63]);
  zip16<1>(rows[0], rows[8], rows[0], rows[8]);
  zip16<1>(rows[16], rows[24], rows[16], rows[24]);
  zip16<1>(rows[32], rows[40], rows[32], rows[40]);
  zip16<1>(rows[48], rows[56], rows[48], rows[56]);
  zip16<1>(rows[1], rows[9], rows[1], rows[9]);
  zip16<1>(rows[17], rows[25], rows[17], rows[25]);
  zip16<1>(rows[33], rows[41], rows[33], rows[41]);
  zip16<1>(rows[49], rows[57], rows[49], rows[57]);
  zip16<1>(rows[2], rows[10], rows[2], rows[10]);
  zip16<1>(rows[18], rows[26], rows[18], rows[26]);
  zip16<1>(rows[34], rows[42], rows[34], rows[42]);
  zip16<1>(rows[50], rows[58], rows[50], rows[58]);
  zip16<1>(rows[3], rows[11], rows[3], rows[11]);
  zip16<1>(rows[19], rows[27], rows[19], rows[27]);
  zip16<1>(rows[35], rows[43], rows[35], rows[43]);
  zip16<1>(rows[51], rows[59], rows[51], rows[59]);
  zip16<1>(rows[4], rows[12], rows[4], rows[12]);
  zip16<1>(rows[20], rows[28], rows[20], rows[28]);
  zip16<1>(rows[36], rows[44], rows[36], rows[44]);
  zip16<1>(rows[52], rows[60], rows[52], rows[60]);
  zip16<1>(rows[5], rows[13], rows[5], rows[13]);
  zip16<1>(rows[21], rows[29], rows[21], rows[29]);
  zip16<1>(rows[37], rows[45], rows[37], rows[45]);
  zip16<1>(rows[53], rows[61], rows[53], rows[61]);
  zip16<1>(rows[6], rows[14], rows[6], rows[14]);
  zip16<1>(rows[22], rows[30], rows[22], rows[30]);
  zip16<1>(rows[38], rows[46], rows[38], rows[46]);
  zip16<1>(rows[54], rows[62], rows[54], rows[62]);
  zip16<1>(rows[7], rows[15], rows[7], rows[15]);
  zip16<1>(rows[23], rows[31], rows[23], rows[31]);
  zip16<1>(rows[39], rows[47], rows[39], rows[47]);
  zip16<1>(rows[55], rows[63], rows[55], rows[63]);
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[8], rows[12], rows[8], rows[12]);
  zip16<1>(rows[16], rows[20], rows[16], rows[20]);
  zip16<1>(rows[24], rows[28], rows[24], rows[28]);
  zip16<1>(rows[32], rows[36], rows[32], rows[36]);
  zip16<1>(rows[40], rows[44], rows[40], rows[44]);
  zip16<1>(rows[48], rows[52], rows[48], rows[52]);
  zip16<1>(rows[56], rows[60], rows[56], rows[60]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[9], rows[13], rows[9], rows[13]);
  zip16<1>(rows[17], rows[21], rows[17], rows[21]);
  zip16<1>(rows[25], rows[29], rows[25], rows[29]);
  zip16<1>(rows[33], rows[37], rows[33], rows[37]);
  zip16<1>(rows[41], rows[45], rows[41], rows[45]);
  zip16<1>(rows[49], rows[53], rows[49], rows[53]);
  zip16<1>(rows[57], rows[61], rows[57], rows[61]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[10], rows[14], rows[10], rows[14]);
  zip16<1>(rows[18], rows[22], rows[18], rows[22]);
  zip16<1>(rows[26], rows[30], rows[26], rows[30]);
  zip16<1>(rows[34], rows[38], rows[34], rows[38]);
  zip16<1>(rows[42], rows[46], rows[42], rows[46]);
  zip16<1>(rows[50], rows[54], rows[50], rows[54]);
  zip16<1>(rows[58], rows[62], rows[58], rows[62]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[11], rows[15], rows[11], rows[15]);
  zip16<1>(rows[19], rows[23], rows[19], rows[23]);
  zip16<1>(rows[27], rows[31], rows[27], rows[31]);
  zip16<1>(rows[35], rows[39], rows[35], rows[39]);
  zip16<1>(rows[43], rows[47], rows[43], rows[47]);
  zip16<1>(rows[51], rows[55], rows[51], rows[55]);
  zip16<1>(rows[59], rows[63], rows[59], rows[63]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
  zip<1>(rows[32], rows[33], rows[32], rows[33]);
  zip<1>(rows[34], rows[35], rows[34], rows[35]);
  zip<1>(rows[36], rows[37], rows[36], rows[37]);
  zip<1>(rows[38], rows[39], rows[38], rows[39]);
  zip<1>(rows[40], rows[41], rows[40], rows[41]);
  zip<1>(rows[42], rows[43], rows[42], rows[43]);
  zip<1>(rows[44], rows[45], rows[44], rows[45]);
  zip<1>(rows[46], rows[47], rows[46], rows[47]);
  zip<1>(rows[48], rows[49], rows[48], rows[49]);
  zip<1>(rows[50], rows[51], rows[50], rows[51]);
  zip<1>(rows[52], rows[53], rows[52], rows[53]);
  zip<1>(rows[54], rows[55], rows[54], rows[55]);
  zip<1>(rows[56], rows[57], rows[56], rows[57]);
  zip<1>(rows[58], rows[59], rows[58], rows[59]);
  zip<1>(rows[60], rows[61], rows[60], rows[61]);
  zip<1>(rows[62], rows[63], rows[62], rows[63]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[8], rows[10], rows[8], rows[10]);
  zip<2>(rows[12], rows[14], rows[12], rows[14]);
  zip<2>(rows[16], rows[18], rows[16], rows[18]);
  zip<2>(rows[20], rows[22], rows[20], rows[22]);
  zip<2>(rows[24], rows[26], rows[24], rows[26]);
  zip<2>(rows[28], rows[30], rows[28], rows[30]);
  zip<2>(rows[32], rows[34], rows[32], rows[34]);
  zip<2>(rows[36], rows[38], rows[36], rows[38]);
  zip<2>(rows[40], rows[42], rows[40], rows[42]);
  zip<2>(rows[44], rows[46], rows[44], rows[46]);
  zip<2>(rows[48], rows[50], rows[48], rows[50]);
  zip<2>(rows[52], rows[54], rows[52], rows[54]);
  zip<2>(rows[56], rows[58], rows[56], rows[58]);
  zip<2>(rows[60], rows[62], rows[60], rows[62]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<2>(rows[9], rows[11], rows[9], rows[11]);
  zip<2>(rows[13], rows[15], rows[13], rows[15]);
  zip<2>(rows[17], rows[19], rows[17], rows[19]);
  zip<2>(rows[21], rows[23], rows[21], rows[23]);
  zip<2>(rows[25], rows[27], rows[25], rows[27]);
  zip<2>(rows[29], rows[31], rows[29], rows[31]);
  zip<2>(rows[33], rows[35], rows[33], rows[35]);
  zip<2>(rows[37], rows[39], rows[37], rows[39]);
  zip<2>(rows[41], rows[43], rows[41], rows[43]);
  zip<2>(rows[45], rows[47], rows[45], rows[47]);
  zip<2>(rows[49], rows[51], rows[49], rows[51]);
  zip<2>(rows[53], rows[55], rows[53], rows[55]);
  zip<2>(rows[57], rows[59], rows[57], rows[59]);
  zip<2>(rows[61], rows[63], rows[61], rows[63]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[4];
    rows[4]                  = rows[16];
    rows[16]                 = rows[2];
    rows[2]                  = rows[8];
    rows[8]                  = rows[32];
    rows[32]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[12];
    rows[12]                 = rows[48];
    rows[48]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[5];
    rows[5]                  = rows[20];
    rows[20]                 = rows[18];
    rows[18]                 = rows[10];
    rows[10]                 = rows[40];
    rows[40]                 = rows[33];
    rows[33]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[6];
    rows[6]                  = rows[24];
    rows[24]                 = rows[34];
    rows[34]                 = rows[9];
    rows[9]                  = rows[36];
    rows[36]                 = rows[17];
    rows[17]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[7];
    rows[7]                  = rows[28];
    rows[28]                 = rows[50];
    rows[50]                 = rows[11];
    rows[11]                 = rows[44];
    rows[44]                 = rows[49];
    rows[49]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[13];
    rows[13]                 = rows[52];
    rows[52]                 = rows[19];
    rows[19]                 = rows[14];
    rows[14]                 = rows[56];
    rows[56]                 = rows[35];
    rows[35]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[15];
    rows[15]                 = rows[60];
    rows[60]                 = rows[51];
    rows[51]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[21];
    rows[21]                 = rows[22];
    rows[22]                 = rows[26];
    rows[26]                 = rows[42];
    rows[42]                 = rows[41];
    rows[41]                 = rows[37];
    rows[37]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[23];
    rows[23]                 = rows[30];
    rows[30]                 = rows[58];
    rows[58]                 = rows[43];
    rows[43]                 = rows[45];
    rows[45]                 = rows[53];
    rows[53]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[25];
    rows[25]                 = rows[38];
    rows[38]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[27];
    rows[27]                 = rows[46];
    rows[46]                 = rows[57];
    rows[57]                 = rows[39];
    rows[39]                 = rows[29];
    rows[29]                 = rows[54];
    rows[54]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[31];
    rows[31]                 = rows[62];
    rows[62]                 = rows[59];
    rows[59]                 = rows[47];
    rows[47]                 = rows[61];
    rows[61]                 = rows[55];
    rows[55]                 = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<32>, Bytes<64>)
{
  zip16<1>(rows[0], rows[16], rows[0], rows[16]);
  zip16<1>(rows[1], rows[17], rows[1], rows[17]);
  zip16<1>(rows[2], rows[18], rows[2], rows[18]);
  zip16<1>(rows[3], rows[19], rows[3], rows[19]);
  zip16<1>(rows[4], rows[20], rows[4], rows[20]);
  zip16<1>(rows[5], rows[21], rows[5], rows[21]);
  zip16<1>(rows[6], rows[22], rows[6], rows[22]);
  zip16<1>(rows[7], rows[23], rows[7], rows[23]);
  zip16<1>(rows[8], rows[24], rows[8], rows[24]);
  zip16<1>(rows[9], rows[25], rows[9], rows[25]);
  zip16<1>(rows[10], rows[26], rows[10], rows[26]);
  zip16<1>(rows[11], rows[27], rows[11], rows[27]);
  zip16<1>(rows[12], rows[28], rows[12], rows[28]);
  zip16<1>(rows[13], rows[29], rows[13], rows[29]);
  zip16<1>(rows[14], rows[30], rows[14], rows[30]);
  zip16<1>(rows[15], rows[31], rows[15], rows[31]);
  zip16<1>(rows[0], rows[8], rows[0], rows[8]);
  zip16<1>(rows[16], rows[24], rows[16], rows[24]);
  zip16<1>(rows[1], rows[9], rows[1], rows[9]);
  zip16<1>(rows[17], rows[25], rows[17], rows[25]);
  zip16<1>(rows[2], rows[10], rows[2], rows[10]);
  zip16<1>(rows[18], rows[26], rows[18], rows[26]);
  zip16<1>(rows[3], rows[11], rows[3], rows[11]);
  zip16<1>(rows[19], rows[27], rows[19], rows[27]);
  zip16<1>(rows[4], rows[12], rows[4], rows[12]);
  zip16<1>(rows[20], rows[28], rows[20], rows[28]);
  zip16<1>(rows[5], rows[13], rows[5], rows[13]);
  zip16<1>(rows[21], rows[29], rows[21], rows[29]);
  zip16<1>(rows[6], rows[14], rows[6], rows[14]);
  zip16<1>(rows[22], rows[30], rows[22], rows[30]);
  zip16<1>(rows[7], rows[15], rows[7], rows[15]);
  zip16<1>(rows[23], rows[31], rows[23], rows[31]);
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[8], rows[12], rows[8], rows[12]);
  zip16<1>(rows[16], rows[20], rows[16], rows[20]);
  zip16<1>(rows[24], rows[28], rows[24], rows[28]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[9], rows[13], rows[9], rows[13]);
  zip16<1>(rows[17], rows[21], rows[17], rows[21]);
  zip16<1>(rows[25], rows[29], rows[25], rows[29]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[10], rows[14], rows[10], rows[14]);
  zip16<1>(rows[18], rows[22], rows[18], rows[22]);
  zip16<1>(rows[26], rows[30], rows[26], rows[30]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[11], rows[15], rows[11], rows[15]);
  zip16<1>(rows[19], rows[23], rows[19], rows[23]);
  zip16<1>(rows[27], rows[31], rows[27], rows[31]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<1>(rows[16], rows[17], rows[16], rows[17]);
  zip<1>(rows[18], rows[19], rows[18], rows[19]);
  zip<1>(rows[20], rows[21], rows[20], rows[21]);
  zip<1>(rows[22], rows[23], rows[22], rows[23]);
  zip<1>(rows[24], rows[25], rows[24], rows[25]);
  zip<1>(rows[26], rows[27], rows[26], rows[27]);
  zip<1>(rows[28], rows[29], rows[28], rows[29]);
  zip<1>(rows[30], rows[31], rows[30], rows[31]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[8], rows[10], rows[8], rows[10]);
  zip<2>(rows[12], rows[14], rows[12], rows[14]);
  zip<2>(rows[16], rows[18], rows[16], rows[18]);
  zip<2>(rows[20], rows[22], rows[20], rows[22]);
  zip<2>(rows[24], rows[26], rows[24], rows[26]);
  zip<2>(rows[28], rows[30], rows[28], rows[30]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<2>(rows[9], rows[11], rows[9], rows[11]);
  zip<2>(rows[13], rows[15], rows[13], rows[15]);
  zip<2>(rows[17], rows[19], rows[17], rows[19]);
  zip<2>(rows[21], rows[23], rows[21], rows[23]);
  zip<2>(rows[25], rows[27], rows[25], rows[27]);
  zip<2>(rows[29], rows[31], rows[29], rows[31]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[4];
    rows[4]                  = rows[16];
    rows[16]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[2];
    rows[2]                  = rows[8];
    rows[8]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[12];
    rows[12]                 = rows[18];
    rows[18]                 = rows[9];
    rows[9]                  = rows[6];
    rows[6]                  = rows[24];
    rows[24]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[5];
    rows[5]                  = rows[20];
    rows[20]                 = rows[17];
    rows[17]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[7];
    rows[7]                  = rows[28];
    rows[28]                 = rows[19];
    rows[19]                 = rows[13];
    rows[13]                 = rows[22];
    rows[22]                 = rows[25];
    rows[25]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[11];
    rows[11]                 = rows[14];
    rows[14]                 = rows[26];
    rows[26]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[15];
    rows[15]                 = rows[30];
    rows[30]                 = rows[27];
    rows[27]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[23];
    rows[23]                 = rows[29];
    rows[29]                 = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<16>, Bytes<64>)
{
  zip16<1>(rows[0], rows[8], rows[0], rows[8]);
  zip16<1>(rows[1], rows[9], rows[1], rows[9]);
  zip16<1>(rows[2], rows[10], rows[2], rows[10]);
  zip16<1>(rows[3], rows[11], rows[3], rows[11]);
  zip16<1>(rows[4], rows[12], rows[4], rows[12]);
  zip16<1>(rows[5], rows[13], rows[5], rows[13]);
  zip16<1>(rows[6], rows[14], rows[6], rows[14]);
  zip16<1>(rows[7], rows[15], rows[7], rows[15]);
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[8], rows[12], rows[8], rows[12]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[9], rows[13], rows[9], rows[13]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[10], rows[14], rows[10], rows[14]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  zip16<1>(rows[11], rows[15], rows[11], rows[15]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<1>(rows[8], rows[9], rows[8], rows[9]);
  zip<1>(rows[10], rows[11], rows[10], rows[11]);
  zip<1>(rows[12], rows[13], rows[12], rows[13]);
  zip<1>(rows[14], rows[15], rows[14], rows[15]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[8], rows[10], rows[8], rows[10]);
  zip<2>(rows[12], rows[14], rows[12], rows[14]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  zip<2>(rows[9], rows[11], rows[9], rows[11]);
  zip<2>(rows[13], rows[15], rows[13], rows[15]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[4];
    rows[4]                  = rows[2];
    rows[2]                  = rows[8];
    rows[8]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[12];
    rows[12]                 = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[5];
    rows[5]                  = rows[6];
    rows[6]                  = rows[10];
    rows[10]                 = rows[9];
    rows[9]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[7];
    rows[7]                  = rows[14];
    rows[14]                 = rows[11];
    rows[11]                 = rows[13];
    rows[13]                 = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems], Elements<8>, Bytes<64>)
{
  zip16<1>(rows[0], rows[4], rows[0], rows[4]);
  zip16<1>(rows[1], rows[5], rows[1], rows[5]);
  zip16<1>(rows[2], rows[6], rows[2], rows[6]);
  zip16<1>(rows[3], rows[7], rows[3], rows[7]);
  // correction steps follow below (if required)
  zip<1>(rows[0], rows[1], rows[0], rows[1]);
  zip<1>(rows[2], rows[3], rows[2], rows[3]);
  zip<1>(rows[4], rows[5], rows[4], rows[5]);
  zip<1>(rows[6], rows[7], rows[6], rows[7]);
  zip<2>(rows[0], rows[2], rows[0], rows[2]);
  zip<2>(rows[4], rows[6], rows[4], rows[6]);
  zip<2>(rows[1], rows[3], rows[1], rows[3]);
  zip<2>(rows[5], rows[7], rows[5], rows[7]);
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[1];
    rows[1]                  = rows[4];
    rows[4]                  = vec_v;
  }
  {
    Vec<T, SIMD_WIDTH> vec_v = rows[3];
    rows[3]                  = rows[6];
    rows[6]                  = vec_v;
  }
}

template <typename T, size_t SIMD_WIDTH>
static SIMD_INLINE void transpose2inplcLane(
  Vec<T, SIMD_WIDTH> rows[Vec<T, SIMD_WIDTH>::elems])
{
  transpose2inplcLane(rows, Elements<Vec<T, SIMD_WIDTH>::elements>(),
                      Bytes<SIMD_WIDTH>());
}

} // namespace ext
} // namespace internal
} // namespace simd

#endif // SIMD_VEC_EXT_TRANSPOSE_AUTOGEN_H_
