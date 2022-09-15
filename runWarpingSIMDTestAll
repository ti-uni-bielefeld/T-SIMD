#!/usr/bin/tcsh -f
# -f means: don't read .tcshrc (resets search path)
# ===========================================================================
# 
# runWarpingSIMDTestAll --
# runs warpingSIMDTest in 'a' mode
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

# dryRun=1: don't compute home vectors
set dryRun = 0

# allMode: a or A
set allMode = "A"
# database base names list
set baseList = "living1"
# set baseList = "living1 living2 living3 living4"
# sscvList
set sscvList = "day night"
# for autoscaling
set ssAuto = "day"
set cvAuto = "night"
# database suffix
# set suffix = "Hh288sh"
set suffix = "Hh384sh"
# relative cutoff frequency of Butterworth filter
set bw = "0.10"
# grid step (1 for each ss/cv)
set gridStep = 2
# set gridStep = 1
# box radius (large number for "all")
set boxRad = 1000
# set boxRad = 3

# change parameters:
# setenv firstPhase 1
# setenv partialSearch 1
#
# for Hh384sh
setenv nAlpha 128
setenv nPsi 128

mkdir -p TEST
# print base parameter (pixelScale, postScale changed below!), 
# add comment char
warpingSIMDTest |& grep -e PARAMETER | sed 's/^/#\ /'
echo "\n"
# go through list of database extensions for current views
foreach base ($baseList)
    #  9. May 18 (rm): automatically adjust scaling
    set scaling = (`warpingSIMDTest p $base $ssAuto $cvAuto $suffix $bw |& grep SCALING`)
    setenv pixelScale $scaling[6]
    setenv postScale $scaling[8]
    echo "# base = $base, pixelScale = $pixelScale, postScale = $postScale"
    foreach ss ($sscvList)
	foreach cv ($sscvList)
	    # run warpingSIMDTest in $allMode "a" or "A"
	    echo "# $allMode $base $ss $cv $suffix $bw $gridStep $boxRad $pixelScale $postScale"
	    # -e ALL -e DBINFO
	    if ( !($dryRun) ) then
		warpingSIMDTest $allMode $base $ss $cv $suffix $bw \
		    $gridStep $boxRad \
		    |& grep -e RESULTS
	    endif
	end
    end
    echo "\n"
end
