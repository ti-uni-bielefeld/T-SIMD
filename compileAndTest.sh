#!/usr/bin/env tcsh
# ===========================================================================
# 
# compileAndTest --
# compile with different flags and perform test runs
# 
# Ralf Moeller
# 
#    Copyright (C) 2016
#    Computer Engineering Group
#    Faculty of Technology
#    Bielefeld University
#    www.ti.uni-bielefeld.de
#
# ===========================================================================

date

set doitCompile = 1
set doitAlsoCompileAutoTest = 1
set doitRun = 1
set doitAlsoRunAutoTest = 1

# only one of these:
set useGccClang = 1
set useICC = 0

set sandbox_test = 1
set opt_test = 1
set opt_arch_test = 1
set comparison = 1
set autotest_errors = 1
set cppstd_test = 1
set default_compilation = 1

set VMK_JOBS = 16

if ($useGccClang) then
    # g++, clang++
    # set compiler = g++
    # set compiler = clang++-10
    set arch_info_list = (\
	 "sse2,-msse2" \
	 "sse3,-msse3" \
	 "ssse3,-mssse3" \
	 "sse4.1,-msse4.1" \
	 "sse4.2,-msse4.2" \
	 "avx,-mavx" \
	 "avx2,-mavx2" \
	 "AVX512F,-mavx512f" \
	 "AVX512DQ,-mavx512dq" \
	 "AVX512BW,-mavx512bw" \
	 "AVX512BWDQ,-mavx512bw,-mavx512dq" \
	 "AVX512VBMI2,-mavx512bw,-mavx512dq,-mavx512vbmi,-mavx512vbmi2" \
	 "native,-march=native" \
    )
endif

if ($useICC) then
    # icc
    # 29. Jul 18 (rm): couldn't find option such that only AVX512F or
    # AVX512F+BW are available
    # TODO: vbmi2 version is still missing
    # set compiler = icc
    set arch_info_list = (\
		      "sse2,-msse2" \
		      "sse3,-msse3" \
		      "ssse3,-mssse3" \
		      "sse4.1,-msse4.1" \
		      "sse4.2,-msse4.2" \
		      "avx,-mavx" \
		      "avx2,-march=skylake" \
		      "AVX512BWDQ,-march=skylake-avx512" \
		      "native,-march=native")
endif

vset
make platform_dirs

mkdir -p COMPILE_AND_TEST

if ($sandbox_test) then
  foreach sandbox_defines ("" "-DSIMDVEC_SANDBOX")
    echo "================= SANDBOX_DEFINES $sandbox_defines ================="
    if ($doitCompile) then
	make clean
	make dep
	# make SANDBOX_DEFINES="$sandbox_defines" conf-info
	make -j ${VMK_JOBS} SANDBOX_DEFINES="$sandbox_defines" all
	if ($doitAlsoCompileAutoTest) then
	    make -j 1 SANDBOX_DEFINES="$sandbox_defines" autotest
	endif
    endif
    if ($doitRun) then
	rehash
	simdvectest > COMPILE_AND_TEST/sandbox_vec
	# can't be used in sandbox mode at the moment
	# simdmasktest > COMPILE_AND_TEST/sandbox_mask
    endif
  end
endif

if ($opt_test) then
  foreach opt_flags ("-O0" "-O3 -funroll-loops")
    echo "==================== OPT_FLAGS $opt_flags ===================="
    if ($doitCompile) then
	make clean
	make dep
	# make OPT_FLAGS="$opt_flags" conf-info
	make -j ${VMK_JOBS} OPT_FLAGS="$opt_flags" all
	if ($doitAlsoCompileAutoTest) then
	    make -j 1 OPT_FLAGS="$opt_flags" autotest
	endif
	rehash
    endif
  end
endif

if ($opt_arch_test) then
  # uppercase letters for AVX512 to simplify comparison below
  foreach arch_info ($arch_info_list)
    # https://stackoverflow.com/questions/7735160/how-do-i-split-a-string-in-csh
    set arch_list = ($arch_info:as/,/ /)
    set arch = $arch_list[1]
    set arch_defines = "$arch_list[2-]"
    # echo "arch_info = $arch_info"
    # echo "arch = $arch"
    # echo "arch_defines = $arch_defines"
    foreach opt_flags ("-O0" "-O3 -funroll-loops")
      echo "======== ARCH_DEFINES $arch_defines OPT_FLAGS $opt_flags ========"
      if ($doitCompile) then
	make clean
	make dep
	# make ARCH_DEFINES="$arch_defines" conf-info
	make -j ${VMK_JOBS} \
	    ARCH_DEFINES="$arch_defines" OPT_FLAGS="$opt_flags" all
	if ($doitAlsoCompileAutoTest) then
	    make -j 1 \
		ARCH_DEFINES="$arch_defines" OPT_FLAGS="$opt_flags" autotest
	endif
      endif
    end
    # tests only for last compilation above
    echo "running tests for $arch"
    if ($doitRun) then
      rehash
      make -j 1 flags-info ARCH_DEFINES="$arch_defines" > \
	    COMPILE_AND_TEST/flags.$arch
      simdvectest > COMPILE_AND_TEST/simdvectest.$arch
      simdmasktest > COMPILE_AND_TEST/simdmasktest.$arch
      if ($doitAlsoRunAutoTest) then
	  simdvecautotest0 > COMPILE_AND_TEST/simdvecautotest.$arch
	  simdvecautotest1 >> COMPILE_AND_TEST/simdvecautotest.$arch
      endif
      nativetest > COMPILE_AND_TEST/nativetest.$arch
    endif
  end
endif

if ($comparison) then
  echo "==================== comparison simdvectest ===================="
  cd COMPILE_AND_TEST
  foreach basearch ("sse" "avx" "AVX512")
    foreach simdvectest1 (`ls -1 simdvectest.${basearch}*`)
      foreach simdvectest2 (`ls -1 simdvectest.${basearch}*`)
        if ($doitRun) then
	    set diffres = `diff $simdvectest1 $simdvectest2`
	    if ( $%diffres != 0 ) then
		echo "$simdvectest1 $simdvectest2 differ"		
	    endif
	endif
      end
    end
  end
  cd ..
endif

if ($autotest_errors) then
    echo "================= autotest errors ================"
    if ($doitRun && $doitAlsoRunAutoTest) then
	grep complete COMPILE_AND_TEST/simdvecautotest.*
	echo "----------------------------------------------"
	grep error COMPILE_AND_TEST/simdvecautotest.*
    endif
endif

if ($cppstd_test) then
  foreach cppstd ("c++98" "c++11")
    set cppstd_defines = "-std=$cppstd"
    echo "================= CPPSTD_DEFINES $cppstd_defines ================="
    if ($doitCompile) then
	make clean
	make dep
	# make CPPSTD_DEFINES="$cppstd_defines" conf-info
	make -j ${VMK_JOBS} CPPSTD_DEFINES="$cppstd_defines" all-std-indep
	if ($doitAlsoCompileAutoTest) then
	    make -j ${VMK_JOBS} CPPSTD_DEFINES="$cppstd_defines" autotest
	endif
	rehash
    endif
  end
endif

if ($default_compilation) then
    echo "================== default compilation ================="
    if ($doitCompile) then
	make clean
	make dep
	make -j ${VMK_JOBS} all
	if ($doitAlsoCompileAutoTest) then
	    make -j 1 autotest
	endif
    endif
endif

date
