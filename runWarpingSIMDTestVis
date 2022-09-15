#!/usr/bin/tcsh -f
# -f means: don't  read .tcshrc (resets search path)
# ===========================================================================
# 
# runWarpingSIMDTestVis --
# runs warpingSIMDTest and visualizes results and intermediate data as images
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

# database base name
set base = "living3"
# database extension for snapshots
set ss = "day"
# database extensions for current view
set cvList = "day night"
# for autoscaling
set ssAuto = "day"
set cvAuto = "night"
# database suffix
# set suffix = "Hh288sh"
set suffix = "Hh384sh"
# relative cutoff frequency of Butterworth filter
set bw = "0.10"
# snapshot position (coordinates correspond to those given in image names)
set ssx = 5
set ssy = 4
# current view position (coordinates correspond to those given in image names)
set cvx = 8
set cvy = 2

# change parameters:
# setenv firstPhase 1
# setenv partialSearch 1
#
# setenv firstPhase 21
# setenv sigmoidW -8.23537952,-4.57185675,-3.85960976
# setenv sigmoidW0 0.76350367,0.653014,0.463149
#
# setenv firstPhase 22
# setenv sigmoidW -12.2601
# setenv sigmoidW0 0.7637

# for Hh384sh:
setenv nAlpha 128
setenv nPsi 128

mkdir -p TEST

#  29. Jul 18 (rm): automatically adjust scaling
set scaling = (`warpingSIMDTest p $base $ssAuto $cvAuto $suffix $bw |& grep SCALING`)
setenv pixelScale $scaling[6]
setenv postScale $scaling[8]
echo "# base = $base, pixelScale = $pixelScale, postScale = $postScale"

# go through list of database extensions for current views ("day" "night")
foreach cv ($cvList)
    # run warpingSIMDTest in mode "s"
    echo "v $base $ss $cv $suffix $bw $ssx $ssy $cvx $cvy"
    warpingSIMDTest v $base $ss $cv $suffix $bw $ssx $ssy $cvx $cvy
end
