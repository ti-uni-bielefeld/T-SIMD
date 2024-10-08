# Notes for Library Developers

This page contains information for developers contributing to the T-SIMD library.

## Overview over the project structure

The project is organized into the following directory structure:

- `src/`: all C++ source files
  - `lib/`: all header files of the library
    - `tsimd.H`: main include file for the T-SIMD library
    - `tsimd/`: all header files of the T-SIMD library
      - `autogen/`: automatically generated header files
      - `intel/`: intel-specific implementations
      - `neon/`: ARM NEON-specific implementations
      - `sandbox/`: sandbox implementations for testing
  - `test/`: test programs
    - `autotest/`: test programs for the automatic tests
- `doc/`: configuration file for generating the documentation with `doxygen`
- `scripts/`: various scripts

All build artifacts are placed in the `build/` directory which is listed in the `.gitignore` file to avoid accidentally committing build artifacts.

The whole library consists only of C++ header files, since it is a template library.

The header files are structured in such a way that every header file directly includes other header files it needs. This way, the order in which headers are included does not matter, enabling the code formatter to sort the includes.

### Namespaces

The library uses namespaces for code organization. The main namespace is `simd` where all code for the library is located.

#### Internal Code

All internal code is contained in the namespace `simd::internal`, such that all code intended for use by the user of the library is directly contained in the namespace `simd` (not in any sub-namespace). This forms the public interface of the library.

The main purposes of this are:

- Avoiding users of the library accidentally using internal code that might change in the future or is unintuitive to use.
- Avoiding cluttering suggestions for auto-completion with internal code. (Typing `simd::` and then using auto-completion only shows the public interface this way.)
- Making the separation between public interface and internal implementation clearer.

The internal namespace is further subdivided into the following sub-namespaces for better organization:

- `simd::internal::base`: Contains internal code for base-level functions and classes.
- `simd::internal::ext`: Contains internal code for extension functions built on top of the base-level functions and classes.
- `simd::internal::mask`: Contains internal code for mask functions and classes.
- `simd::internal::auto_test`: Contains code for the tests. This code is not used by the library itself but only by the test programs.

### Implementations for Different Architectures

Choosing different implementations is achieved the following way. Take the base level functions, for example:

The file `base.H` does not contain any implementations, but simply wrapper functions (which are contained directly in the `simd` namespace), that forward to the implementation functions (which are contained in the `simd::internal` namespace). For each supported architecture, there is a separate directory containing header files for those implementations. For example, the file `intel/base_impl_intel32.H` contains the implementation for the 32-byte SIMD instruction set extensions of Intel CPUs.

There is a special case for the masked functions. Implementations for different architectures are handled the same way, but there is an additional file `mask_impl_emu.H` containing emulated implementations for masked functions which are chosen by automatic overload resolution when there isn't a more specific implementation available.

There is also a special "sandbox" implementation for testing (which is activated by defining the macro `SIMDVEC_SANDBOX`), which does not contain any actual implementations, but only prints the function calls to the console.

Other functions than the base level and masked functions also have a similar interface wrapper layer with implementations in the `simd::internal` namespace but without separate implementation files, since they are not architecture-specific.

### Macro "`SIMD_WIDTH_DEFAULT_NATIVE`"

In order to give the `SIMD_WIDTH` template parameter a default value of `NATIVE_SIMD_WIDTH` if `NATIVE_SIMD_WIDTH` is defined but not cause an error if `NATIVE_SIMD_WIDTH` is not defined, the macro `SIMD_WIDTH_DEFAULT_NATIVE` is used. This macro is defined as `SIMD_WIDTH = NATIVE_SIMD_WIDTH` if `NATIVE_SIMD_WIDTH` is defined and as `SIMD_WIDTH` otherwise. It is used in the following way:

```cpp
template <typename T, size_t SIMD_WIDTH_DEFAULT_NATIVE>
struct MySIMDStruct {
  // code using the SIMD_WIDTH template parameter
};
```

## Formatting

The code is formatted with `clang-format` version 16 or newer (https://clang.llvm.org/docs/ClangFormat.html, https://releases.llvm.org/download.html) using the configuration file `.clang-format`.

`clang-format` can be installed with the following commands for example:

```shell
$ pip install clang-format
```
or
```shell
$ pip3 install clang-format
```

To format the code, run

```shell
$ make format
```

This will format all files in the project ending in `.C` or `.H`.

The code should ideally be formatted after every change and before making a new release to keep a consistent code style.

## Generating the documentation

Generating the documentation requires `doxygen` 1.9.8 or later (https://www.doxygen.nl).

To generate the documentation, run

```shell
$ make doc
```

The documentation will be in the folder `doc_html`.
To view the generated documentation, open the file `doc_html/index.html` in a web browser.

The configuration file for documentation generation is contained in the file `doc/Doxyfile` (this is also where the files to be included in the documentation are specified).

## Generating the single-header include file

Generating the single-header include file requires `quom` which is written in Python and can be installed with

```shell
$ pip3 install --user quom
```

for example
(see https://github.com/Viatorus/quom).

Once `quom` is installed, run

```shell
$ make single-header
```

to generate the single-header include file `build/tsimd_sh.H` which can now be distributed and included in projects on its own.

## Testing

To test the library, compile the test programs with

```shell
$ make autotest
```

and run them:

```shell
$ ./build/src/test/autotest/test0
$ ./build/src/test/autotest/test1
$ ./build/src/test/autotest/testM
$ ./build/src/test/autotest/test_load_store
```

### Testing different configurations automatically

To automatically run the tests for different architectures, compilers and other configurations, use the `autotest.py` script. This script uses the [Intel® Software Development Emulator](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) for testing the T-SIMD library with architecture features that are not available on the current CPU.

The script compiles and runs the tests for different configurations in parallel, so it is recommended to run the script on a machine with many cores and a lot of RAM.

The script can be run with

```shell
$ scripts/autotest.py
```

The results of the tests are written into a directory starting with `autotest_` followed by the hostname of the machine and the current date and time.

