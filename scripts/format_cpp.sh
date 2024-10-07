#!/usr/bin/env bash
#===========================================================================
#
# runs clang-format on all .H and .C files in the given directory
#
# This file is part of the following software:
# 
#    - the low-level C++ template SIMD library
#    - the SIMD implementation of the MinWarping and the 2D-Warping methods 
#      for local visual homing.
# 
# The software is provided based on the accompanying license agreement in the
# file LICENSE.md.
# The software is provided "as is" without any warranty by the licensor and
# without any liability of the licensor, and the software may not be
# distributed by the licensee; see the license agreement for details.
# 
# (C) Ralf Möller
#     Computer Engineering
#     Faculty of Technology
#     Bielefeld University
#     www.ti.uni-bielefeld.de
#
#===========================================================================

if [ $# -ne 1 ]; then
        echo "Usage: $0 <directory/file>"
        exit 1
fi

if ! which clang-format > /dev/null; then
        echo "Error: clang-format not found"
        exit 1
fi

REQUIRED_CLANG_FORMAT_VERSION=16
if [[ $(clang-format --version | grep -oP '(?<=version )\d+') -lt $REQUIRED_CLANG_FORMAT_VERSION ]]; then
        echo "Error: clang-format version $REQUIRED_CLANG_FORMAT_VERSION or higher required (found $(clang-format --version))"
        exit 1
fi

find $1 -type f \( -name "*.H" -o -name "*.C" \) | while read -r src; do
        # "clang-format -i" can break links, so we format into a temporary file
        # and then write the new content back
        tmpSrc="${src}.tmp"
        clang-format "$src" > "$tmpSrc"
        if ! cmp -s "$src" "$tmpSrc"; then
                cp -f "$tmpSrc" "$src"
                echo "$src was re-formatted"
        fi
        rm -f "$tmpSrc"
done
