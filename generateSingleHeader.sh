#!/bin/sh
# ===========================================================================
# 
# generateSingleHeader.sh --
# Generates a single header file from the T-SIMD library.
# 
# This source code file is part of the following software:
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
# (C) Jonas Keller, Ralf Möller
#     Computer Engineering
#     Faculty of Technology
#     Bielefeld University
#     www.ti.uni-bielefeld.de
# 
# ===========================================================================

# exit on error
set -e

# check if single header file name is given
if [ -z "$1" ]; then
    echo "Usage: $0 <single header file name> [-m]"
    echo "  -m: minify single header file (remove comments and superfluous spaces)"
    exit 1
fi

# get single header file name
sh_name=$1

# check if quom is installed
if ! which quom 2> /dev/null > /dev/null; then
    echo "ERROR: quom is not installed, install with \"pip3 install --user quom\" (see https://github.com/Viatorus/quom)"
    exit 1
fi

# write header comment to single header file
cat <<EOF > $sh_name
// ===========================================================================
//
// $sh_name --
// Single header version of the T-SIMD library.
//
// This file is part of the following software:
//
//    - the low-level C++ template SIMD library
//    - the SIMD implementation of the MinWarping and the 2D-Warping methods
//      for local visual homing.
//
// The software is provided based on the accompanying license agreement
// in the file LICENSE or LICENSE.doc. The software is provided "as is"
// without any warranty by the licensor and without any liability of the
// licensor, and the software may not be distributed by the licensee; see
// the license agreement for details.
//
// (C) Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================
EOF

# write LICENSE to single header file with comment
cat <<EOF >> $sh_name

// ===========================================================================
// LICENSE
// ===========================================================================
/*
EOF
cat LICENSE >> $sh_name
cat <<EOF >> $sh_name
*/
// ===========================================================================
// end of LICENSE
// ===========================================================================

EOF

# generate single header file
quom tsimd.H $sh_name.tmp

# if -m option is given
if [ "$2" = "-m" ]; then
    # write minified single header file
    cat $sh_name.tmp |
    # remove backslash and line break at the end of a line
    sed ':a;N;$!ba;s/\\\n//g' |
    # remove all comments
    gcc -fpreprocessed -dD -E -P - 2>/dev/null |
    # remove all empty lines
    sed '/^\s*$/d' |
    # remove superfluous spaces
    sed 's/  */ /g' |
    # remove spaces at beginning of line
    sed 's/^ //' |
    # remove spaces at end of line
    sed 's/ $//' >> $sh_name
else
    # write single header file
    cat $sh_name.tmp >> $sh_name
fi

# remove temporary file
rm $sh_name.tmp
