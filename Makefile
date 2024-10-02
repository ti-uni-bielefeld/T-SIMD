#===========================================================================
#
# Makefile --
# Makefile for WarpingSIMD demonstrator package
#
# This file is part of the following software:
# 
#    - the low-level C++ template SIMD library
#    - the SIMD implementation of the MinWarping and the 2D-Warping methods 
#      for local visual homing.
# 
# The software is provided based on the accompanying license agreement in the
# file LICENSE.md.
# The software is provided "as is" without any warranty by the licensor and
# without any liability of the licensor, and the software may not be
# distributed by the licensee; see the license agreement for details.
# 
# (C) Ralf Möller
#     Computer Engineering
#     Faculty of Technology
#     Bielefeld University
#     www.ti.uni-bielefeld.de
#
#===========================================================================

# 01. Feb 23 (Jonas Keller): added syntax only option (see below)

# see notes below marked ARM for compilation on ARM architecture

#===========================================================================
# binaries
#===========================================================================

default_binaries = $(patsubst %.C,%,$(wildcard src/*.C)) $(patsubst %.C,%,$(wildcard src/test/*.C))
autotest_binaries = $(patsubst %.C,%,$(wildcard src/test/autotest/*.C))

binaries = $(default_binaries) $(autotest_binaries)

depend_files = $(addsuffix .d,$(binaries))

tsimd_single_header_file = tsimd_sh.H

objects      = $(addsuffix .o,$(binaries))

#===========================================================================
# definitions and flags
#===========================================================================

# compiler ?= clang++-10
compiler ?= g++

# - rt library is needed for clock_gettime
# - pthread may be needed on some systems
# (you may have to remove -lrt on Mac)
libraries = -lm -lrt # -lpthread

# optimization flags
# -funroll-loops is faster in both min-warping phases
optflags ?= -O3 -funroll-loops

# play in template sandbox only: -DSIMDVEC_SANDBOX
sandbox_defines ?= # -DSIMDVEC_SANDBOX

# set this to something != 0 to only check syntax and not compile
syntax_only ?= 0

# set this to something != 0 to compile statically
static_bins ?= 0

# user defines
# - needed if we use constants in stdint.h: -D__STDC_LIMIT_MACROS
# - save assembly code: -masm=intel -save-temps -fverbose-asm
# - profiling: -pg for gprof
# - valgrind: -mssse3 instead of -march=native for valgrind --tool=callgrind
# - test different vector sets: -msse2, -msse3, -mssse3 -march=native
# - activate SIMDVec load/store alignment check: -DSIMD_ALIGN_CHK
# - use at least -std=c++11
# - for tilt-search programs, -std=c++17 is required, now used

# for compilation on/for ARM:
# - keep -march=native or use a specific architecture, e.g. -march=armv7-a
#   or use specific CPU, e.g. -mcpu=cortex-a15
# - replace -mfpmath=sse with -mfpu=neon or remove it, some compilers/architectures
#   require it, some don't
# - crosscompilation and emulation may require static_bins to be set to
#   something other than 0 (see above) if dynamic libraries are not available
# - may require -lpthread in libraries above

userdefs_arch     = -march=native -mfpmath=sse
userdefs_avx512   = -mavx512f -mavx512bw -mavx512dq -mavx512vl -mpopcnt
userdefs_c        = -ggdb -fno-var-tracking
userdefs_cppstd   = -std=c++17
userdefs_cpp      =
# 18. Nov 23 (Jonas Keller): added more warnings
# recommended by:
# https://github.com/cpp-best-practices/cppbestpractices/blob/master/02-Use_the_Tools_Available.md#gcc--clang
# (also has explanations for the flags)
userdefs_warnings = -Wall -Wextra -pedantic -Wpedantic \
	-Wnon-virtual-dtor -Wcast-align -Wunused -Woverloaded-virtual \
	-Wsign-compare -Wmisleading-indentation -Wformat=2 \
	-Wimplicit-fallthrough \
	-Wno-format-nonliteral -Wno-cast-align

# 20. Sep 22 (Jonas Keller): use secure template overloads on Windows and disable
# warnings about unsafe functions
# see: https://learn.microsoft.com/en-us/cpp/c-runtime-library/security-features-in-the-crt
crt_sec_features_flags = -D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1 \
	-D_CRT_SECURE_NO_WARNINGS

ifeq ($(syntax_only),0)
syntax_defines =
else
syntax_defines = -fsyntax-only
endif

ifeq ($(static_bins),0)
static_bins_flags =
else
static_bins_flags = -static
endif

# all flags (split into c and cpp (c++)) to avoid preprocessor warnings
flags_c        = $(userdefs_c) $(userdefs_warnings) $(optflags) $(sandbox_defines) $(EXTRA_DEFINES) $(syntax_defines) $(static_bins_flags)
flags_cppstd  ?= $(userdefs_cppstd)
flags_cpp      = $(flags_c) $(userdefs_cpp) $(flags_cppstd) $(crt_sec_features_flags)
flags_arch    ?= $(userdefs_arch)
flags_depends  = -MMD -MP
flags_includes = -I./src/lib

# os dependent definitions
ifeq ($(OS),Windows_NT) 
RM = del /Q /F
RMDIR_R = rmdir /S /Q
RMDIR = rmdir /Q
MKDIR = mkdir
NULL = NUL
else
RM = rm -f
RMDIR_R = rm -rf
RMDIR = rmdir --ignore-fail-on-non-empty
MKDIR = mkdir -p
NULL = /dev/null
endif

build_dir ?= .

.PHONY: all_default
all_default: $(addprefix $(build_dir)/,$(default_binaries))
	@echo "use 'make autotest' to compile $(autotest_binaries)"
	@echo "  requires long compilation time and may run out of memory,"
	@echo "  compilation may take much longer on g++ than on clang++"

.PHONY: autotest
autotest: $(addprefix $(build_dir)/,$(autotest_binaries))

.PHONY: all
all: all_default autotest

ifneq ($(build_dir),.)
.PHONY: $(binaries)
$(binaries): %: $(build_dir)/%
endif


#===========================================================================
# compiler
#===========================================================================

# https://www.gnu.org/software/make/manual/make.html#Static-Pattern

$(addprefix $(build_dir)/,$(objects)): $(build_dir)/%.o: %.C
	@echo compiling default $@ from $<
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_depends) $(flags_arch) $(flags_cpp) $(flags_includes) -c $< -o $@

#===========================================================================
# linker
#===========================================================================

# https://www.gnu.org/software/make/manual/make.html#Static-Pattern

ifeq ($(syntax_only),0)
$(addprefix $(build_dir)/,$(binaries)): $(build_dir)/%: $(build_dir)/%.o
	@echo linking $@ from $< $(libraries)
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_arch) $(flags_cpp) -o $@ $< $(libraries)
else
$(addprefix $(build_dir)/,$(binaries)): $(build_dir)/%: $(build_dir)/%.o ;
endif

#===========================================================================
# other rules
#===========================================================================

all_build_files = $(objects) $(binaries) $(depend_files) \
		$(addsuffix .exe,$(binaries)) \
		$(addsuffix .ilk,$(binaries)) $(addsuffix .pdb,$(binaries)) \
		$(tsimd_single_header_file)

.PHONY: clean
clean:
ifneq ($(wildcard $(build_dir)),)
	@echo deleting all binaries, all objects, all dependency files, all .exe files, all .ilk files, all .pdb files, all backup files, all temporary files, single header file and doc_html/ documentation directory
	@$(RM) $(addprefix $(build_dir)/,$(all_build_files)) $(build_dir)/*~ $(build_dir)/*.tmp >$(NULL) 2>&1
	@$(RMDIR_R) $(build_dir)/doc_html >$(NULL) 2>&1
ifneq ($(build_dir),.)
	@$(RMDIR) $(build_dir) >$(NULL) 2>&1
endif
else
	@echo build directory \"$(build_dir)\" does not exist, nothing to clean
endif

.PHONY: info
info:
	@echo "vector extensions"
	@echo | $(compiler) -xc++ -dM -E - $(flags_arch) $(flags_c) \
		| grep -e SSE -e AVX -e POPCNT | sort
	@echo "compiler:          " $(compiler)
	@echo "flags_c:           " $(flags_c)
	@echo "flags_cpp:         " $(flags_cpp)
	@echo "flags_arch:        " $(flags_arch)
	@echo "default_binaries:  " $(default_binaries)
	@echo "autotest_binaries: " $(autotest_binaries)
	@echo "binaries:          " $(binaries)
	@echo "objects:           " $(objects)

# for compileAndTest
.PHONY: flags-info
flags-info:
	@echo | $(compiler) -dM -E - $(flags_arch) \
		$(optflags) $(EXTRA_DEFINES) \
		| grep -e SSE -e AVX -e POPCNT | sort

# for compileAndTest to be compatible with PROG system
# echo -n is to prevent make from printing "nothing to be done"
.PHONY: platform_dirs dep
platform_dirs:
	@echo -n
dep:
	@echo -n

# 10. Feb 23 (Jonas Keller): added format rule
# 10. Jun 24 (rm): this creates new files instead of the linked ones:
# @clang-format -i *.C *.H
.PHONY: format
format:
	@echo "formatting all .C and .H files"
	@./formatAllHC .

# 02. Mar 23 (Jonas Keller): added documenation rule
# 20. Nov 23 (Jonas Keller): added check for doxygen version
min_doxygen_version = 1.9.8
actual_doxygen_version = $(shell doxygen --version | grep -io "[0-9][0-9a-z.-]*" | head -n1)
# this has unnecessary many aliases, but why not
.PHONY: doc docs docu documentation doxygen dox doxy
doc docs docu documentation doxygen dox doxy:
	@printf '%s\n%s\n' "$(min_doxygen_version)" "$(actual_doxygen_version)" \
		| sort -V -C \
		|| (echo "Error: Doxygen version $(min_doxygen_version) or higher is required (found: $(actual_doxygen_version))" \
			; exit 1)
	@echo "generating documentation"
	@$(MKDIR) $(build_dir)
	@( cat doc/Doxyfile ; echo "HTML_OUTPUT=$(build_dir)/doc_html" ) | doxygen -

# 04. Mar 23 (Jonas Keller): added rule for generating single header file
# uses quom (https://github.com/Viatorus/quom)
.PHONY: single-header
single-header: gen-transpose-autogen
	@echo "generating tsimd single header file"
	@./generateSingleHeader.sh $(tsimd_single_header_file)
	@echo "single header written to $(tsimd_single_header_file)"

# 06. Sep 23 (Jonas Keller): added rule for autogenerating transpose functions
transpose_autogen_file = src/lib/tsimd/autogen/ext_transpose.H
.PHONY: gen-transpose-autogen
gen-transpose-autogen: check-clang-format-version
	@echo "generating $(transpose_autogen_file)"
	@./transpose_inplace_autogen.tcl | clang-format > $(transpose_autogen_file)

-include $(addprefix $(build_dir)/,$(depend_files))
