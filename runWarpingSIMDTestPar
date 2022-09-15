#!/usr/bin/tcsh -f
# -f means: don't  read .tcshrc (resets search path)
# ===========================================================================
# 
# runWarpingSIMDTestPar --
# runs warpingSIMDTest in p mode (compute scale parameters)
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

# database suffix
# set suffix = "Hh1152sh"
# set suffix = "Hh576sh"
set suffix = "Hh384sh"
# set suffix = "Hh288sh"
# relative cutoff frequency of Butterworth filter
set bw = "0.10"

# for Hh384sh
setenv nAlpha 128
setenv nPsi 128

foreach distMeas (0 1 2 3 4 5 6 7)
    setenv firstPhase $distMeas
    foreach base ("living1" "living2" "living3" "living4")
	warpingSIMDTest p $base day night $suffix $bw
    end
end

