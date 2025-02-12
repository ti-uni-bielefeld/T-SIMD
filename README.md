# T-SIMD

**T-SIMD** is a header-only low-level C++ template SIMD library which wraps built-in vector data types and built-in vector intrinsics in template classes and template functions or overloaded functions, respectively.

Templates parameters are the element data type of the vectors and the vector width in bytes (e.g. 16 for SSE* and NEON, 32 for AVX/AVX2). This makes it possible to flexibly change the data type and the vector instruction set for entire portions of the code. Moreover, many implementation details at the intrinsics level are hidden by **T-SIMD**. SSE*, AVX/AVX2, AVX-512, and ARM NEON vector instruction sets are currently supported. In addition, higher-level functions like transposition are provided.

## Documentation

The documentation of **T-SIMD** is available [here](https://ti-uni-bielefeld.github.io/T-SIMD/).

For documentation of functions and classes, see the [Library Reference](https://ti-uni-bielefeld.github.io/T-SIMD/topics.html).

For further information on the **T-SIMD** library and the SIMD implementation of Warping methods, including related papers, videos, and image databases see https://www.ti.uni-bielefeld.de/html/people/moeller/tsimd_warpingsimd.html.

## Download and Usage

Download the latest release of **T-SIMD** from the [GitHub releases page](https://github.com/ti-uni-bielefeld/T-SIMD/releases).

The easiest way to use **T-SIMD** is to download and include the single-header version `%tsimd.H` in your project.

Alternatively, you can download the full source code and include the `%src/lib/tsimd.H` header file.

## Example

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

## Additional information

### Compiler and C++ Standard

**T-SIMD** requires at least C++11.

**T-SIMD** has been tested with gcc and clang on Linux.

Using clang is strongly recommended, as it is much faster and uses much less memory than gcc when compiling **T-SIMD** code.

### Vector width

Almost all functions and classes in **T-SIMD** are templated on the vector width (`SIMD_WIDTH`). This can be explicitly specified to force a specific vector width, or it can be left unspecified to automatically use the maximum vector width available on the target architecture.
To limit the maximum vector width that **T-SIMD** will automatically use, see the information in the macro `MAX_SIMD_WIDTH` in the section [Preprocessor Definitions](#preprocessor-definitions).

### Preprocessor Definitions

The following preprocessor definitions can be defined to change the behavior of **T-SIMD** (**WARNING**: These definitions **must** be defined before including `%tsimd.H` or any other **T-SIMD** header file):

- `MAX_SIMD_WIDTH` : This macro can be used to limit the maximum vector width that **T-SIMD** will use automatically. Must be an integer of at least 16.<br>
**Note**: While **T-SIMD** will not use vectors wider than `MAX_SIMD_WIDTH` (except when a width is explicitly specified), other libraries or the compiler may still use wider vectors, e.g. when auto-vectorizing code.
- `SIMD_ALIGN_CHK` : If this macro is defined, **T-SIMD** will check whether the alignment of the data pointers passed to the functions is correct before executing the functions. If the alignment is incorrect, the location of the error will be printed to the standard error output and the program will be aborted using `std::abort()`. This is useful for debugging purposes.
- `SIMDVEC_SANDBOX` : If this macro is defined, **T-SIMD** functions will not execute their intended operations, but instead will print what function was called to the standard output. This is useful for debugging purposes.

### Support for 64-Bit Data Types

**T-SIMD** only supports 64-bit data types on architectures that support 64-bit data types in their vector instruction sets. At the moment, the only architecture that does not support 64-bit data types is ARMv7 or older. 

Whether 64-bit data types are supported can be determined by checking whether the `SIMD_64BIT_TYPES` macro is defined.

### Differences on Different Architectures

**T-SIMD** tries to behave as similar as possible on different architectures. However, there are some differences that are due to the different instruction sets and the different implementations of the intrinsics. The following list gives an overview of the differences:

- As mentioned in the section [Support for 64-Bit Data Types](#support-for-64-bit-data-types), 64-bit data types are not supported on ARMv7 or older.
- Denormalized floating-point numbers are flushed to zero on ARMv7. This is not the case on all other supported architectures.
- The `simd::div` and `simd::sqrt` functions are approximations on ARM in contrast to the exact results on all other supported architectures.
- Converting from floating-point to integer values results in slightly different results on ARM.

There may be other slight differences that are not listed here.

## License

This software is distributed based on a specific **license agreement**, please see the file [LICENSE.md](LICENSE.md).

## Authors and Contributors

**T-SIMD** was originally written by [Prof. Dr.-Ing. Ralf Möller](http://www.ti.uni-bielefeld.de/html/people/moeller/) (moeller@ti.uni-bielefeld.de).

[Jonas Keller](https://github.com/jonicho) substantially improved and extended the **T-SIMD** library, added the documentation and set up the GitHub repository.

Other contributors are: Markus Vieth, Adam Marschall, Jan-Lukas Wolf, Lukas Schiermeier, Moritz Breipohl, and Ebba Stina Siebold.

Their specific contributions are mentioned in the source files. Thanks a lot to all contributors!

## Contact and Bug Reports

If you find any bugs or errors in **T-SIMD**, please [open an issue on GitHub](https://github.com/ti-uni-bielefeld/T-SIMD/issues).

Alternatively, or for other inquiries, contact [Prof. Dr.-Ing. Ralf Möller](http://www.ti.uni-bielefeld.de/html/people/moeller/) (moeller@ti.uni-bielefeld.de).
