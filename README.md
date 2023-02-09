# T-SIMD


\tableofcontents

# Introduction

**T-SIMD** is a low-level C++ template SIMD library which wraps built-in vector data types and built-in vector intrincics in template classes and template functions or overloaded functions, respectively.

Templates parameters are the element data type of the vectors and the vector width in bytes (e.g. 16 for SSE* and NEON, 32 for AVX/AVX2). This makes it possible to flexibly change the data type and the vector instruction set for entire portions of the code. Moreover, many implementation details at the intrinsics level are hidden by **T-SIMD**. SSE*, AVX/AVX2, AVX-512, and ARM NEON vector instruction sets are currently supported.

# Quick Start

First, download the latest release from  <https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html> (section "Software Download"). Unpack the archive and copy the contents of the folder starting with "CODE" into your project.

To use the **T-SIMD** library, simply include the header file `tsimd.H` if at least C++11 is available or `SIMDVecAll.H` if C++11 is not available.

# Example

The following example shows how to use **T-SIMD** to compute the sum of two vectors
and print the result to the standard output.

```cpp
#include "tsimd.H"
#include <stdio.h>

int main() {
  simd::Vec<simd::Float> a = simd::setzero<simd::Float>();
  simd::Vec<simd::Float> b = simd::setzero<simd::Float>();
  simd::Vec<simd::Float> sum = simd::add(a, b);
  simd::print("%g ", c);
  puts("");
  return 0;
}
```
