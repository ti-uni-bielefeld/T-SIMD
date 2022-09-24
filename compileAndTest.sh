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
set errors = 1
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

if ($doitAlsoCompileAutoTest) then
  set autotest_target = "autotest"
else
  set autotest_target = ""
endif

# vset # TODO: what is this?

# make platform_dirs

mkdir -p COMPILE_AND_TEST

if ($sandbox_test) then
  mkdir -p COMPILE_AND_TEST/sandbox
  foreach sandbox_defines ("" "-DSIMDVEC_SANDBOX")
    echo "================= sandbox_defines $sandbox_defines ================="
    if ($doitCompile) then
      make clean
      # make sandbox_defines="$sandbox_defines" conf-info
      make -j ${VMK_JOBS} sandbox_defines="$sandbox_defines" all $autotest_target >& COMPILE_AND_TEST/sandbox/compile_${sandbox_defines}.log
    endif
    if ($doitRun) then
      rehash
      ./simdvectest >& "COMPILE_AND_TEST/sandbox/run_simdvectest_$sandbox_defines.log" &
      # can't be used in sandbox mode at the moment
      ./simdmasktest >& "COMPILE_AND_TEST/sandbox/run_simdmasktest_$sandbox_defines.log" &
      wait
    endif
  end
endif

if ($opt_test) then
  mkdir -p COMPILE_AND_TEST/opt
  foreach opt_flags ("-O3 -funroll-loops")# "-O0"
    echo "==================== optflags $opt_flags ===================="
    if ($doitCompile) then
      make clean
      # make optflags="$opt_flags" conf-info
      make -j ${VMK_JOBS} optflags="$opt_flags" all $autotest_target >& COMPILE_AND_TEST/opt/compile_${opt_flags}.log
      rehash
    endif
  end
endif

if ($opt_arch_test) then
  mkdir -p COMPILE_AND_TEST/opt_arch
  # uppercase letters for AVX512 to simplify comparison below
  foreach arch_info ($arch_info_list)
    # https://stackoverflow.com/questions/7735160/how-do-i-split-a-string-in-csh
    set arch_list = ($arch_info:as/,/ /)
    set arch = $arch_list[1]
    set arch_defines = "$arch_list[2-]"
    mkdir -p COMPILE_AND_TEST/opt_arch/$arch
    # echo "arch_info = $arch_info"
    # echo "arch = $arch"
    # echo "arch_defines = $arch_defines"
    foreach opt_flags ("-O3 -funroll-loops") #"-O0"
      echo "======== flags_arch $arch_defines optflags $opt_flags ========"
      if ($doitCompile) then
        make clean
        # make flags_arch="$arch_defines" conf-info
        make -j ${VMK_JOBS} \
          flags_arch="$arch_defines" optflags="$opt_flags" all $autotest_target \
          >& "COMPILE_AND_TEST/opt_arch/$arch/compile_$opt_flags.log"
      endif
    end
    # tests only for last compilation above
    echo "running tests for $arch"
    if ($doitRun) then
      rehash
      make -j ${VMK_JOBS} flags-info flags_arch="$arch_defines" >& \
	      COMPILE_AND_TEST/opt_arch/$arch/flags_info.log
      ./simdvectest >& COMPILE_AND_TEST/opt_arch/$arch/run_simdvectest.log &
      ./simdmasktest >& COMPILE_AND_TEST/opt_arch/$arch/run_simdmasktest.log &
      if ($doitAlsoRunAutoTest) then
        ./simdvecautotest0 "" 10000 >& COMPILE_AND_TEST/opt_arch/$arch/run_simdvecautotest0.log &
        ./simdvecautotest1 "" 1000 >& COMPILE_AND_TEST/opt_arch/$arch/run_simdvecautotest1.log &
        ./simdvecautotestM "" 10000 >& COMPILE_AND_TEST/opt_arch/$arch/run_simdvecautotestM.log &
      endif
      ./nativetest >& COMPILE_AND_TEST/opt_arch/$arch/run_nativetest.log &
      wait
    endif
  end
endif

if ($comparison) then
  echo "==================== comparison simdvectest ===================="
  mkdir -p COMPILE_AND_TEST/comparison
  cd COMPILE_AND_TEST/opt_arch
  foreach simdvectest1 (`ls -1d *`)
    foreach simdvectest2 (`ls -1d *`)
      if ($doitRun) then
        set diffres = `diff $simdvectest1/run_simdvectest.log $simdvectest2/run_simdvectest.log`
        if ( $%diffres != 0 ) then
          echo "$simdvectest1/run_simdvectest.log and $simdvectest2/run_simdvectest.log differ" \
            >> ../comparison/diff_simdvectest.log
        endif
      endif
    end
  end
  cd ../..
endif

if ($cppstd_test) then
  mkdir -p COMPILE_AND_TEST/cppstd
  foreach cppstd ("c++98" "c++11")
    set cppstd_defines = "-std=$cppstd"
    echo "================= flags_cppstd $cppstd_defines ================="
    if ($doitCompile) then
      make clean
      # make flags_cppstd="$cppstd_defines" conf-info
      make -j ${VMK_JOBS} flags_cppstd="$cppstd_defines" all $autotest_target >& COMPILE_AND_TEST/cppstd/compile_${cppstd}.log
      rehash
    endif
  end
endif

if ($default_compilation) then
  mkdir -p COMPILE_AND_TEST/default
  echo "================== default compilation ================="
  if ($doitCompile) then
    make clean
    make -j ${VMK_JOBS} all $autotest_target >& COMPILE_AND_TEST/default/compile.log
  endif
endif

set where_there_errors = 0

if ($errors) then
  echo "================= errors ================"
  grep -nri --exclude={errors,warnings,problems}.log error COMPILE_AND_TEST >& COMPILE_AND_TEST/errors.log
  if ( -s COMPILE_AND_TEST/errors.log ) then
    echo "There are errors! See COMPILE_AND_TEST/errors.log"
    set where_there_errors = 1
  endif
  grep -nri --exclude={errors,warnings,problems}.log warning COMPILE_AND_TEST >& COMPILE_AND_TEST/warnings.log
  if ( -s COMPILE_AND_TEST/warnings.log ) then
    echo "There are warnings! See COMPILE_AND_TEST/warnings.log"
    set where_there_errors = 1
  endif
  grep -nri --exclude={errors,warnings,problems}.log problem COMPILE_AND_TEST >& COMPILE_AND_TEST/problems.log
  if ( -s COMPILE_AND_TEST/problems.log ) then
    echo "There are problems! See COMPILE_AND_TEST/problems.log"
    set where_there_errors = 1
  endif
endif

make clean

date

exit $where_there_errors
