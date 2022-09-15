#!/usr/bin/tcsh -f
# -f means: don't read .tcshrc (resets search path)
# ===========================================================================
# 
# runPlaceRecognitionTest --
# runs warpingSIMDTest and visualizes two compass match plots
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

# TODO: automatic computation of pixelScale?

set path = ( . $path )

# database base name
set base = "living3"
# database extension for snapshots
set ss = "day"
# database suffix
# set suffix = "Hh1152sh"
# set suffix = "Hh576sh"
# set suffix = "Hh384sh"
set suffix = "Hh288sh"
# set suffix = "Hh144sh"
# relative cutoff frequency of Butterworth filter
set bw = "0.10"
# set bw = "0.05"
# snapshot position (coordinates correspond to those given in image names)
set ssx = 5
set ssy = 4

# change parameters:
# ------------------

setenv nScalePlanes 1

mkdir -p TEST
# go through list of database extensions for current views
foreach cv ("day" "night")
    # run warpingSIMDTest in mode "s"
    echo "l $base $ss $cv $suffix $bw $ssx $ssy"
    warpingSIMDTest l $base $ss $cv $suffix $bw $ssx $ssy
    # generate same gnuplot filename as done by warpingSIMDTest
    set gpFile = \
      "./TEST/dissim_${base}_${ss}_${cv}_${suffix}_${bw}_${ssx}_${ssy}.gp"
    # plot filename
    set gpPlotFile = \
      "./TEST/plot_dissim_${base}_${ss}_${cv}_${suffix}_${bw}_${ssx}_${ssy}.gp"
    # 21. Mar 16 (rm): gnuplot with -p (persist) option fails in version 4.4
    # works in version 4.6, therefore not used any longer
    echo \
      set title '"'$base $ss $cv $suffix $bw'"' ';' \
      splot '"'$gpFile'"' index 1 with linespoints palette pointsize 2 pointtype 7 title '""', '""' index 0 with points pointsize 2 pointtype 7 title '""' ';' \
      pause -1 '"'Press ET'"' \
       > $gpPlotFile
      gnuplot $gpPlotFile
end
