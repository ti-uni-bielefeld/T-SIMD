#!/usr/bin/tcsh -f
# -f means: don't read .tcshrc (resets search path)
# ===========================================================================
# 
# runWarpingSIMDTestBW --
# runs warpingSIMDTest in 'a' mode for varying bw
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

# database base names list
set baseList = "living1 living2 living3 living4"
# set baseList = "living3"
# database extension for snapshots
set ss = "day"
# database extension for current views
set cvList = ("day" "night")
# set cvList = ("day")
# set cvList = ("night")
# for autoscaling
set ssAuto = "day"
set cvAuto = "night"
# database suffix
# set suffix = "Hh288sh"
set suffix = "Hh384sh"
# butterworth parameter
# set bwList = (0.01 0.02 0.03 0.04 0.05 0.07 0.10 0.14 0.19) 
set bwList = (0.05 0.07 0.10 0.14 0.19) 
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


# go through list of database extensions for current views
foreach cv ($cvList)
    foreach base ($baseList)
	foreach bw ($bwList)
	    # 29. Jul 18 (rm): automatically adjust scaling
	    set scaling = (`warpingSIMDTest p $base $ssAuto $cvAuto $suffix $bw |& grep SCALING`)
	    setenv pixelScale $scaling[6]
	    setenv postScale $scaling[8]
	    echo "# base = $base, pixelScale = $pixelScale, postScale = $postScale"
	    # run warpingSIMDTest in mode "a"
	    echo "a $base $ss $cv $suffix $bw $gridStep $boxRad"
	    # -e ALL -e DBINFO 
	    warpingSIMDTest a $base $ss $cv $suffix $bw $gridStep $boxRad \
		|& grep -e RESULTS
	end
	echo ""
    end
end
