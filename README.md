# T-SIMD

# Introduction

**T-SIMD** is a low-level C++ template SIMD library which wraps built-in vector data types and built-in vector intrincics in template classes and template functions or overloaded functions, respectively.

Templates parameters are the element data type of the vectors and the vector width in bytes (e.g. 16 for SSE* and NEON, 32 for AVX/AVX2). This makes it possible to flexibly change the data type and the vector instruction set for entire portions of the code. Moreover, many implementation details at the intrinsics level are hidden by **T-SIMD**. SSE*, AVX/AVX2, AVX-512, and ARM NEON vector instruction sets are currently supported. Please note the [differences on different architectures](@ref differences-on-different-architectures).

# Compiler and C++ Standard Support/Requirements

**T-SIMD** requires at least C++11.

**T-SIMD** has been tested with gcc and clang on Linux.

Using clang is strongly recommended, as it is much faster and uses much less memory than gcc when compiling **T-SIMD** code.

# Quick Start

First, download the latest release from

<https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html>

(section "Software Download").

Either

- download the single header library version `tsimd_sh.H` and include it in your project or
- download the full archive, unpack it and copy the contents of the folder starting with "CODE" into your project.

To use the **T-SIMD** library, simply include the single-header file `tsimd_sh.H` or the `tsimd.H` header from the archive.

# Example

The following example shows how to use **T-SIMD** to compute the sum of two vectors
and print the result to the standard output.

```cpp
#include "tsimd.H" // or "tsimd_sh.H" if you use the single-header version
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

# Vector width

Almost all functions and classes in **T-SIMD** are templated on the vector width (`SIMD_WIDTH`). This can be explicitly specified to force a specific vector width, or it can be left unspecified to automatically use the maximum vector width available on the target architecture.
To limit the maximum vector width that **T-SIMD** will automatically use, see the section [Preprocessor Definitions](@ref preprocessor-definitions).

# Preprocessor Definitions {#preprocessor-definitions}

The following preprocessor definitions can be defined to change the behavior of **T-SIMD** (**WARNING**: These definitions **must** be defined before including `tsimd.H` or any other **T-SIMD** header file):

- `MAX_SIMD_WIDTH` : This macro can be used to limit the maximum vector width that **T-SIMD** will use automatically. Must be an integer of at least 16.<br>
**Note**: While **T-SIMD** will not use vectors wider than `MAX_SIMD_WIDTH` (except when a width is explicitly specified), other libraries or the compiler may still use wider vectors, e.g. when auto-vectorizing code.
- `SIMD_ALIGN_CHK` : If this macro is defined, **T-SIMD** will check whether the alignment of the data pointers passed to the functions is correct before executing the functions. If the alignment is incorrect, the location of the error will be printed to the standard error output and the program will be aborted using `std::abort()`. This is useful for debugging purposes.
- `SIMDVEC_SANDBOX` : If this macro is defined, **T-SIMD** functions will not execute their intended operations, but instead will print what function was called to the standard output. This is useful for debugging purposes.

# Support for 64-Bit Data Types {#support-for-64-bit-data-types}

**T-SIMD** only supports 64-bit data types on architectures that support 64-bit data types in their vector instruction sets. At the moment, the only architecture that does not support 64-bit data types is ARMv7 or older. 

Whether 64-bit data types are supported can be determined by checking whether the `SIMD_64BIT_TYPES` macro is defined.

# Differences on Different Architectures {#differences-on-different-architectures}

**T-SIMD** tries to behave as similar as possible on different architectures. However, there are some differences that are due to the different instruction sets and the different implementations of the intrinsics. The following list gives an overview of the differences:

- As mentioned in the section [Support for 64-Bit Data Types](@ref support-for-64-bit-data-types), 64-bit data types are not supported on ARMv7 or older.
- Denormalized floating-point numbers are flushed to zero on ARMv7. This is not the case on all other supported architectures.
- The `simd::div` and `simd::sqrt` functions are approximations on ARM in contrast to the exact results on all other supported architectures.
- Converting from floating-point to integer values results in slightly different results on ARM.

There may be other slight differences that are not listed here.

# Author and Contributors

**T-SIMD** was originally written by [Ralf Möller](http://www.ti.uni-bielefeld.de/html/people/moeller/) (moeller@ti.uni-bielefeld.de).

Jonas Keller substantially improved and extended the **T-SIMD** library and added the documentation.

Other contributors are: Markus Vieth, Adam Marschall, Jan-Lukas Wolf, Lukas Schiermeier, Moritz Breipohl, and Ebba Stina Siebold.

Their specific contributions are mentioned in the source files. Thanks a lot to all contributors!

# Contact and Bug Reports

Please report any bugs or errors found to [Ralf Möller](http://www.ti.uni-bielefeld.de/html/people/moeller/) (moeller@ti.uni-bielefeld.de).
