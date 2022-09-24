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

# see notes below marked ARM for compilation on ARM architecture

#===========================================================================
# binaries
#===========================================================================

tsimd_binaries = tsimdtest nativetest simdvectest simdmasktest styleExamples
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
autotest_binaries = simdvecautotest0 simdvecautotest1 simdvecautotestM

# architecture-specific binaries (target archspec)
# these are assumed to always exists in the distribution
archspec_binaries = simdRadixSortGeneric

# all non archspec binaries
default_binaries = $(binaries) $(autotest_binaries)

# all binaries
# (ARM: remove $(archspec_binaries))
all_binaries = $(default_binaries) $(archspec_binaries)

#===========================================================================
# object files
#===========================================================================

all_objects      = $(addsuffix .o,$(all_binaries))
default_objects  = $(addsuffix .o,$(default_binaries))
archspec_objects = $(addsuffix .o,$(archspec_binaries))

#===========================================================================
# definitions and flags
#===========================================================================

# compiler = clang++-10
compiler = g++

# - rt library is needed for clock_gettime
# (you may have to remove -lrt on Mac)
libraries = -lm -lrt

# optimization flags
# -funroll-loops is faster in both min-warping phases
optflags = -O3 -funroll-loops

# user defines
# - needed if we use constants in stdint.h: -D__STDC_LIMIT_MACROS
# - save assembly code: -masm=intel -save-temps -fverbose-asm
# - profiling: -pg for gprof
# - valgrind: -mssse3 instead of -march=native for valgrind --tool=callgrind
# - test different vector sets: -msse2, -msse3, -mssse3 -march=native
# - activate SIMDVec load/store alignment check: -DSIMD_ALIGN_CHK
# - play in template sandbox only: -DSIMDVEC_SANDBOX
# - use -mfpu=neon for NEON version (instead of -pfmath=sse)
# - use -mcpu=cortex-a15 for specific ARM CPU (better use -march=native though)
# - use -std=c++98 or -std=c++11 (tsimdtest requires c++11)
# - for tilt-search programs, -std=c++17 is required, now used

# (ARM: remove -mfpmath=sse)
userdefs_arch   = -march=native
userdefs_avx512 = -mavx512f -mavx512bw -mavx512dq -mavx512vl -mpopcnt
userdefs_c      = -Wall -Wextra -Wpedantic -mfpmath=sse -ggdb -fno-var-tracking
userdefs_cpp    = -fno-operator-names -std=c++17

# 20. Sep 22 (Jonas Keller): use secure template overloads on Windows
# see: https://learn.microsoft.com/en-us/cpp/c-runtime-library/security-features-in-the-crt
crt_sec_features_flags = -D_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1 \
	-D_CRT_SECURE_NO_WARNINGS

# all flags (split into c and cpp (c++)) to avoid preprocessor warnings
flags_c        = $(userdefs_c) $(optflags) $(EXTRA_DEFINES)
flags_cpp      = $(userdefs_c) $(userdefs_cpp) $(optflags) $(EXTRA_DEFINES) \
	$(crt_sec_features_flags)
flags_arch     = $(userdefs_arch)
# use flags_avx512 on a non-avx512 machine (just to compile, won't run
# on this machine), and flags_arch on a avx512 machine
flags_archspec = $(userdefs_avx512) -pthread

# dependencies
depend = Makefile.depend

# use del instead of rm on Windows
ifeq ($(OS),Windows_NT) 
RM = del /Q /F
else
RM = rm -f
endif

# C++ files (for dependencies)
all_cpp_files      = $(addsuffix .C,$(all_binaries))
archspec_cpp_files = $(addsuffix .C,$(archspec_binaries))
default_cpp_files  = $(addsuffix .C,$(default_binaries))

.PHONY: all
all: $(binaries)
	@echo "use 'make autotest' to compile $(autotest_binaries)"
	@echo "  requires long compilation time and may run out of memory,"
	@echo "  compilation may take much longer on g++ than on clang++"
	@echo "use 'make archspec' to compile $(archspec_binaries)"
	@echo "  please select flags_archspec for your machine"

.PHONY: autotest
autotest: $(autotest_binaries)

.PHONY: archspec
archspec: $(archspec_binaries)

#===========================================================================
# compiler
#===========================================================================

# https://www.gnu.org/software/make/manual/make.html#Static-Pattern

$(default_objects): %.o: %.C
	@echo compiling default $@ from $<
	@$(compiler) $(flags_arch) $(flags_cpp) -c $< -o $@

$(archspec_objects): %.o: %.C
	@echo compiling archspec $@ from $<
	@echo please select flags_archspec for your machine
	@$(compiler) $(flags_archspec) $(flags_cpp) -c $< -o $@

#===========================================================================
# linker
#===========================================================================

# https://www.gnu.org/software/make/manual/make.html#Static-Pattern

$(binaries): %: %.o
	@echo linking $@ from $< $(libraries)
	@$(compiler) $(flags_arch) $(flags_cpp) -o $@ $< $(libraries)

$(autotest_binaries): %: %.o
	@echo linking $@ from $< $(libraries)
	@$(compiler) $(flags_arch) $(flags_cpp) -o $@ $< $(libraries)

$(archspec_binaries): %: %.o
	@echo linking $@ from $< $(libraries)
	@$(compiler) $(flags_archspec) $(flags_cpp) -o $@ $< $(libraries)

#===========================================================================
# other rules
#===========================================================================

.PHONY: clean
clean:
	@echo deleting all binaries, all objects, all .exe files, all .ilk files, all .pdb files, backup files and dependency file $(depend)
	@$(RM) $(all_objects) $(all_binaries) $(addsuffix .exe,$(all_binaries)) \
		$(addsuffix .ilk,$(all_binaries)) $(addsuffix .pdb,$(all_binaries)) \
		*~ $(depend) >nul 2>&1

# (ARM: remove second compiler invocation for archspec_cpp_files)
.PHONY: dep
dep: 
	@echo generating dependency file $(depend)
	@$(compiler) -M $(flags_arch) $(flags_cpp) $(default_cpp_files) \
		> $(depend)
	@$(compiler) -M $(flags_archspec) $(flags_cpp) $(archspec_cpp_files) \
		>> $(depend)

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
	@echo "all_cpp_files:     " $(all_cpp_files)

ifeq ($(depend),$(wildcard $(depend)))
include $(depend)
else
$(info warping: missing $(depend), make dep)
endif
