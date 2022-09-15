#!/usr/bin/tcsh -f
# -f means: don't read .tcshrc (resets search path)
# ===========================================================================
# 
# runWarpingSIMDTest --
# runs warpingSIMDTest and visualizes two home-vector plots
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
# ===========================================================================

set path = ( . $path )

# set base = "lab_quarterres"
# # set base = "4walls_1"
# set ss = "day"
# set cvList = "day"
# set ssAuto = "day"
# set cvAuto = "day"
# set suffix = "288"
# set bw = "0.00"
# set ssx = 5
# set ssy = 4

if (1) then
    # database base name
    set base = "living3"
    # database extension for snapshots
    set ss = "day"
    # cvList
    set cvList = "day night"
    # for autoscaling
    set ssAuto = "day"
    set cvAuto = "night"
    # database suffix
    # set suffix = "Hh1152sh"
    # set suffix = "Hh576sh"
    set suffix = "Hh384sh"
    # set suffix = "Hh288sh"
    # set suffix = "Hh144sh"
    # relative cutoff frequency of Butterworth filter
    set bw = "0.10"
    # set bw = "0.05"
    # snapshot position (coordinates correspond to those given in image names)
    set ssx = 5
    set ssy = 4
else
    # database base name
    set base = "grid1"
    # database extension for snapshots
    set ss = "day"
    # cvList
    set cvList = "day night"
    # for autoscaling
    set ssAuto = "day"
    set cvAuto = "night"
    # database suffix
    set suffix = "h384shb"
    # relative cutoff frequency of Butterworth filter
    set bw = "0.20"
    # set bw = "0.10"
    # set bw = "0.05"
    # snapshot position (coordinates correspond to those given in image names)
    set ssx = 5
    set ssy = 4
endif

# change parameters:
# ------------------
#
# e.g.
# setenv firstPhase 1
# setenv partialSearch 1
# setenv nAlpha 48
# setenv nPsi 48
#
# for Hh1152sh:
# setenv nAlpha 384
# setenv nPsi 384
# setenv postScale 50.0
#
# for Hh576sh
# setenv nAlpha 192
# setenv nPsi 192
# setenv postScale 100.0
#
# setenv firstPhase 21
# setenv sigmoidW -8.23537952,-4.57185675,-3.85960976
# setenv sigmoidW0 0.76350367,0.653014,0.463149
#
# setenv firstPhase 22
# setenv sigmoidW -12.2601
# setenv sigmoidW0 0.7637

# for Hh384sh
setenv nAlpha 128
setenv nPsi 128

mkdir -p TEST

#  4. Jun 18 (rm): automatically adjust scaling
set scaling = (`warpingSIMDTest p $base $ssAuto $cvAuto $suffix $bw |& grep SCALING`)
setenv pixelScale $scaling[6]
setenv postScale $scaling[8]
echo "# base = $base, pixelScale = $pixelScale, postScale = $postScale"

# go through list of database extensions for current views
foreach cv ($cvList)
    # run warpingSIMDTest in mode "s"
    # echo "s $base $ss $cv $suffix $bw $ssx $ssy"
    warpingSIMDTest s $base $ss $cv $suffix $bw $ssx $ssy
    # generate same gnuplot filename as done by warpingSIMDTest
    set gpFile = \
      "./TEST/homevectors_${base}_${ss}_${cv}_${suffix}_${bw}_${ssx}_${ssy}.gp"
    set gpTrueFile = \
      "./TEST/truehomevectors_${base}_${ss}_${cv}_${suffix}_${bw}_${ssx}_${ssy}.gp"
    # plot filename
    set gpPlotFile = \
      "./TEST/plot_homevectors_${base}_${ss}_${cv}_${suffix}_${bw}_${ssx}_${ssy}.gp"
    # 21. Mar 16 (rm): gnuplot with -p (persist) option fails in version 4.4
    # works in version 4.6, therefore not used any longer
    echo \
      set title '"'$base $ss $cv $suffix $bw'"' ';' \
      plot '"'$gpFile'"' i 0 w p t '""' ',' \
           '""' i 1 ls 7 lw 3 ',' \
	   '"'$gpTrueFile'"' i 1 ls 2 ';' \
      pause -1 '"'Press ET'"' | \
      cat "homevectors.gp" - > $gpPlotFile
      gnuplot $gpPlotFile
end
