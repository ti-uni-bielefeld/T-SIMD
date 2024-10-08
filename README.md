# T-SIMD

**T-SIMD** is a header-only low-level C++ template SIMD library which wraps built-in vector data types and built-in vector intrinsics in template classes and template functions or overloaded functions, respectively.

Templates parameters are the element data type of the vectors and the vector width in bytes (e.g. 16 for SSE* and NEON, 32 for AVX/AVX2). This makes it possible to flexibly change the data type and the vector instruction set for entire portions of the code. Moreover, many implementation details at the intrinsics level are hidden by **T-SIMD**. SSE*, AVX/AVX2, AVX-512, and ARM NEON vector instruction sets are currently supported. In addition, higher-level functions like transposition are provided.

# Documentation

The documentation of **T-SIMD** is available [here](https://ti-uni-bielefeld.github.io/T-SIMD/).

Alternatively, you can generate the documentation yourself by cloning the repository, running `make doc` in the root directory and then opening the file `doc_html/index.html` in a web browser.

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

# Usage

For information on how to use **T-SIMD**, please refer to the [documentation](https://ti-uni-bielefeld.github.io/T-SIMD/).

# License

This software is distributed based on a specific **license agreement**, please see the file [LICENSE.md](LICENSE.md).

# Author and Contributors

**T-SIMD** was originally written by [Ralf Möller](http://www.ti.uni-bielefeld.de/html/people/moeller/) (moeller@ti.uni-bielefeld.de).

Jonas Keller substantially improved and extended the **T-SIMD** library and added the documentation.

Other contributors are: Markus Vieth, Adam Marschall, Jan-Lukas Wolf, Lukas Schiermeier, Moritz Breipohl, and Ebba Stina Siebold.

Their specific contributions are mentioned in the source files. Thanks a lot to all contributors!

# Contact and Bug Reports

If you find any bugs or errors in **T-SIMD**, please [open an issue on GitHub](https://github.com/ti-uni-bielefeld/T-SIMD/issues).

Alternatively, or for other inquiries, contact [Ralf Möller](http://www.ti.uni-bielefeld.de/html/people/moeller/) (moeller@ti.uni-bielefeld.de).
