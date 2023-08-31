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
# The software is provided based on the accompanying license agreement
# in the file LICENSE or LICENSE.doc. The software is provided "as is"
# without any warranty by the licensor and without any liability of the
# licensor, and the software may not be distributed by the licensee; see
# the license agreement for details.
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

tsimd_binaries = tsimdtest nativetest simdvectest simdmasktest styleExamples \
	hacctest
warping_binaries = warpingSIMDTest warpingSIMDSmallTest \
	BaselineDemo IdealScale Tilt TiltSearchDemo

# binaries (target all)
# contains only binaries with existing source files in the distribution
both_binaries = $(tsimd_binaries) $(warping_binaries)
both_sources =  $(addsuffix .C,$(both_binaries))
existing_sources = $(wildcard $(both_sources))
binaries = $(patsubst %.C,%,$(existing_sources))

# binaries autotest (target autotest)
# these are assumed to always exists in the distribution
autotest_binaries = simdvecautotest0 simdvecautotest1 simdvecautotestM simdvecautotest_load_store

# architecture-specific binaries (target archspec)
# these are assumed to always exists in the distribution
archspec_binaries = simdRadixSortGeneric

# all non archspec binaries
default_binaries = $(binaries) $(autotest_binaries)

# all binaries
# (ARM: remove $(archspec_binaries))
all_binaries = $(default_binaries) $(archspec_binaries)

# dependency files
depend_files = $(addsuffix .d,$(all_binaries))

# single header tsimd library file
tsimd_single_header_file = tsimd_sh.H
# minimized single header tsimd library file
tsimd_single_header_file_min = tsimd_sh_min.H

#===========================================================================
# object files
#===========================================================================

all_objects      = $(addsuffix .o,$(all_binaries))
default_objects  = $(addsuffix .o,$(default_binaries))
archspec_objects = $(addsuffix .o,$(archspec_binaries))

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
# - crosscompilation and emulation may require -static in userdefs_c, if dynamic
#   libraries are not available
# - may require -lpthread in libraries above

userdefs_arch   = -mavx512bw -mavx512dq -mfpmath=sse
userdefs_avx512 = -mavx512f -mavx512bw -mavx512dq -mavx512vl -mpopcnt
userdefs_c      = -Wall -Wextra -Wpedantic -ggdb -fno-var-tracking
userdefs_cppstd = -std=c++17
userdefs_cpp    =

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

# all flags (split into c and cpp (c++)) to avoid preprocessor warnings
flags_c        = $(userdefs_c) $(optflags) $(sandbox_defines) $(EXTRA_DEFINES) $(syntax_defines)
flags_cppstd  ?= $(userdefs_cppstd)
flags_cpp      = $(flags_c) $(userdefs_cpp) $(flags_cppstd) $(crt_sec_features_flags)
flags_arch    ?= $(userdefs_arch)
flags_depends  = -MMD -MP
# use flags_avx512 on a non-avx512 machine (just to compile, won't run
# on this machine), and flags_arch on a avx512 machine
flags_archspec = $(userdefs_avx512) -pthread

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

.PHONY: all
all: $(addprefix $(build_dir)/,$(binaries))
	@echo "use 'make autotest' to compile $(autotest_binaries)"
	@echo "  requires long compilation time and may run out of memory,"
	@echo "  compilation may take much longer on g++ than on clang++"
	@echo "use 'make archspec' to compile $(archspec_binaries)"
	@echo "  please select flags_archspec for your machine"

.PHONY: all_tsimd
all_tsimd: $(addprefix $(build_dir)/,$(tsimd_binaries))

.PHONY: autotest
autotest: $(addprefix $(build_dir)/,$(autotest_binaries))

.PHONY: archspec
archspec: $(addprefix $(build_dir)/,$(archspec_binaries))

ifneq ($(build_dir),.)
.PHONY: $(binaries)
$(binaries): %: $(build_dir)/%
.PHONY: $(autotest_binaries)
$(autotest_binaries): %: $(build_dir)/%
.PHONY: $(archspec_binaries)
$(archspec_binaries): %: $(build_dir)/%
endif


#===========================================================================
# compiler
#===========================================================================

# https://www.gnu.org/software/make/manual/make.html#Static-Pattern

$(addprefix $(build_dir)/,$(default_objects)): $(build_dir)/%.o: %.C
	@echo compiling default $@ from $<
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_depends) $(flags_arch) $(flags_cpp) -c $< -o $@

$(addprefix $(build_dir)/,$(archspec_objects)): $(build_dir)/%.o: %.C
	@echo compiling archspec $@ from $<
	@echo please select flags_archspec for your machine
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_depends) $(flags_archspec) $(flags_cpp) -c $< -o $@

#===========================================================================
# linker
#===========================================================================

# https://www.gnu.org/software/make/manual/make.html#Static-Pattern

ifeq ($(syntax_only),0)
$(addprefix $(build_dir)/,$(binaries)): $(build_dir)/%: $(build_dir)/%.o
	@echo linking $@ from $< $(libraries)
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_arch) $(flags_cpp) -o $@ $< $(libraries)

$(addprefix $(build_dir)/,$(autotest_binaries)): $(build_dir)/%: $(build_dir)/%.o
	@echo linking $@ from $< $(libraries)
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_arch) $(flags_cpp) -o $@ $< $(libraries)

$(addprefix $(build_dir)/,$(archspec_binaries)): $(build_dir)/%: $(build_dir)/%.o
	@echo linking $@ from $< $(libraries)
	@$(MKDIR) $(build_dir)
	@$(compiler) $(flags_archspec) $(flags_cpp) -o $@ $< $(libraries)
else
$(addprefix $(build_dir)/,$(binaries)): $(build_dir)/%: $(build_dir)/%.o ;
$(addprefix $(build_dir)/,$(autotest_binaries)): $(build_dir)/%: $(build_dir)/%.o ;
$(addprefix $(build_dir)/,$(archspec_binaries)): $(build_dir)/%: $(build_dir)/%.o ;
endif

#===========================================================================
# other rules
#===========================================================================

all_build_files = $(all_objects) $(all_binaries) $(depend_files) \
		$(addsuffix .exe,$(all_binaries)) \
		$(addsuffix .ilk,$(all_binaries)) $(addsuffix .pdb,$(all_binaries)) \
		$(tsimd_single_header_file) $(tsimd_single_header_file_min)

.PHONY: clean
clean:
ifneq ($(wildcard $(build_dir)),)
	@echo deleting all binaries, all objects, all dependency files, all .exe files, all .ilk files, all .pdb files, backup files, single header file and html/ documentation directory
	@$(RM) $(addprefix $(build_dir)/,$(all_build_files)) $(build_dir)/*~ >$(NULL) 2>&1
	@$(RMDIR_R) $(build_dir)/doc_html >$(NULL) 2>&1
ifneq ($(build_dir),.)
	@$(RMDIR) $(build_dir) >$(NULL) 2>&1
endif
else
	@echo build directory \"$(build_dir)\" does not exist, nothing to clean
endif

.PHONY: info
info:
	@echo "vector extensions default"
	@echo | $(compiler) -dM -E - $(flags_arch) $(flags_c) \
		| grep -e SSE -e AVX -e POPCNT | sort
	@echo "vector extensions archspec"
	@echo | $(compiler) -dM -E - $(flags_archspec) $(flags_c) \
		| grep -e SSE -e AVX -e POPCNT | sort
	@echo "compiler:          " $(compiler)
	@echo "flags_c:           " $(flags_c)
	@echo "flags_cpp:         " $(flags_cpp)
	@echo "flags_arch:        " $(flags_arch)
	@echo "flags_archspec:    " $(flags_archspec)
	@echo "binaries:          " $(binaries)
	@echo "autotest_binaries: " $(autotest_binaries)
	@echo "archspec_binaries: " $(archspec_binaries)
	@echo "default_binaries:  " $(default_binaries)
	@echo "all_binaries:      " $(all_binaries)
	@echo "archspec_objects:  " $(archspec_objects)
	@echo "default_objects    " $(default_objects)
	@echo "all_objects:       " $(all_objects)

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
.PHONY: format
format:
	@echo "formatting all .C and .H files"
	@clang-format -i *.C *.H

# 02. Mar 23 (Jonas Keller): added documenation rule
# this has unnecessary many aliases, but why not
.PHONY: doc docs docu documentation doxygen dox doxy
doc docs docu documentation doxygen dox doxy:
	@echo "generating documentation"
	@$(MKDIR) $(build_dir)
	@( cat Doxyfile ; echo "HTML_OUTPUT=$(build_dir)/doc_html" ) | doxygen -

# 04. Mar 23 (Jonas Keller): added rule for generating single header file
# uses quom (https://github.com/Viatorus/quom)
.PHONY: single-header
single-header:
	@echo "generating tsimd single header file"
	@./generateSingleHeader.sh $(tsimd_single_header_file)
	@echo "single header written to $(tsimd_single_header_file)"

# 31. Aug 23 (Jonas Keller): added rule for generating a minified single header file
# uses quom (https://github.com/Viatorus/quom)
.PHONY: single-header-min
single-header-min:
	@echo "generating minified tsimd single header file"
	@./generateSingleHeader.sh $(tsimd_single_header_file_min) -m
	@echo "minified single header written to $(tsimd_single_header_file_min)"

-include $(addprefix $(build_dir)/,$(depend_files))
