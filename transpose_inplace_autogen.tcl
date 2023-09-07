#!/usr/bin/env tclsh
# ===========================================================================
#
# transpose_inplace_autogen.tcl --
# auto-generate transpose functions with in-place processing
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
# (C) Ralf Möller
#     Computer Engineering
#     Faculty of Technology
#     Bielefeld University
#     www.ti.uni-bielefeld.de
#
# ===========================================================================

# TODO: write directly to .H file?
# TODO: doxygen information?

# from byte (1) to (currently) double (8)
set bytesPerElementList {1 2 4 8}
# currently SSE, AVX, AVX-512
set bytesList {16 32 64}
# list of element numbers
set elementsList {2 4 8 16 32 64}

# ===========================================================================
# C++ output of template headers etc.
# ===========================================================================

set includeGuardStart \
{#pragma once
#ifndef SIMD_VEC_EXT_TRANSPOSE_AUTOGEN_H_
#define SIMD_VEC_EXT_TRANSPOSE_AUTOGEN_H_
}
set includeGuardEnd \
    "\n#endif // SIMD_VEC_EXT_TRANSPOSE_AUTOGEN_H_"

set includes \
    "\#include \"SIMDVecBase.H\"\n\n\#include <utility>\n"

set simdStart \
    "namespace simd \{"
set simdEnd \
    "\}"

set internalExtStart \
    "namespace internal \{\nnamespace ext \{"
set internalExtEnd \
    "\}\}"

set header \
{// ===========================================================================
//
// transpose_inplace_autogen.H --
// auto-generated transpose functions with in-place processing
// DO NOT EDIT!!!
//
// This source code file is part of the following software:
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
}

# ---------------------------------------------------------------------------
# for full transpose
# ---------------------------------------------------------------------------

# wrap in internal::ext
set transposeCoreTemplateHead {
    template <typename T, int SIMD_WIDTH>
    static SIMD_INLINE void %s
    (const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
     Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems],
     Elements<%d>)}

# wrap in internal::ext
# TODO: namespace ok?
set transposeHubTemplate {
    template <typename T, int SIMD_WIDTH>
    static SIMD_INLINE void %s
    (const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
     Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
    {
        %s
        (inRows, outRows, Elements<Vec<T, SIMD_WIDTH>::elements>());
    }
}

# ---------------------------------------------------------------------------
# for lane-based transpose
# ---------------------------------------------------------------------------

# wrap in internal::ext
set transposeLaneCoreTemplateHead {
    template <typename T, int SIMD_WIDTH>
    static SIMD_INLINE void %s
    (const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
     Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems],
     Elements<%d>,
     Bytes<%d>)}

# wrap in internal::ext
# TODO: namespace ok?
set transposeLaneHubTemplate {
    template <typename T, int SIMD_WIDTH>
    static SIMD_INLINE void %s
    (const Vec<T, SIMD_WIDTH> inRows[Vec<T, SIMD_WIDTH>::elems],
     Vec<T, SIMD_WIDTH> outRows[Vec<T, SIMD_WIDTH>::elems])
    {
        %s
        (inRows, outRows,
         Elements<Vec<T, SIMD_WIDTH>::elements>(),
         Bytes<SIMD_WIDTH>());
    }
}

# ===========================================================================
# counter
# ===========================================================================

proc resetCounters {} {
    set ::numZip16 0
    set ::numZip 0
    set ::numMove 0
}

proc storeCounters {infoString} {
    lappend ::counterInfoList \
        "$infoString: zip16 $::numZip16 zip $::numZip move $::numMove"
}

proc resetCounterInfo {} {
    set ::counterInfoList [list]
}
    
proc writeCounterInfo {fileName} {
    set f [open $fileName "w"]
    foreach counterInfo $::counterInfoList {
        puts $f $counterInfo
    }
    close $f
}

# ===========================================================================
# auto-generation: transpose1fullGenerator
# ===========================================================================

# in-place with post-processing, uses full-length zip

# from swizzle_experiments.tcl: proc transpose1inplace, without lanes argument

proc transpose1fullGenerator {elements} {
    set n $elements
    set n2 [expr $n / 2]
    
    for {set v 0} {$v < $n} {incr v} {
        set isAt($v) $v
    }
    # vs is overwritten after first round
    set vs "inRows"
    set vd "outRows"
    for {set span 1; set parts $n2} \
        {$span <= $n2} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {
            
            set dst1 0
            set partStart 0
            for {set part 0} \
                {$part < $parts} \
                {incr part; set partStart [expr $partStart + 2 * $span]} {
                    
                    set opsInPart $span
                    set src1 $partStart
                    for {set op 0; set src1 $partStart} \
                        {$op < $opsInPart} \
                        {incr op; incr src1; incr dst1 2} {
                            
                            set src2 [expr $src1 + $span]
                            set dst2 [expr $dst1 + 1]
                            set atSrc1 $isAt($src1)
                            set atSrc2 $isAt($src2)
                            puts -nonewline "        "
                            puts "zip<$span>($vs\[$atSrc1\], $vs\[$atSrc2\],\
                                             $vd\[$atSrc1\], $vd\[$atSrc2\]);"
                            incr ::numZip
                            # remember where it should be
                            set newIsAt($dst1) $atSrc1
                            set newIsAt($dst2) $atSrc2
                        }
                }
            set vs "outRows"
            array set isAt [array get newIsAt]
        }
    # post-sorting
    # we try swapping and raise an error if it doesn't work
    for {set v 0} {$v < $n} {incr v} {
        set swapped($v) 0
    }
    # we leave out element 0 and n-1
    for {set v 1} {$v < $n - 1} {incr v} {
        set u $isAt($v)
        # is a swap required? and don't swap twice!
        if {($u != $v) && !$swapped($v)} {
            # do we have a swap partner?
            if {$isAt($u) == $v} {
                puts -nonewline "        "
                puts "std::swap($vd\[$v\], $vd\[$u\]);"
                incr ::numMove 3
                # enought to mark later u (we never reach v again)
                set swapped($u) 1
            } else {
                puts "!!!!! ERROR: NO SWAP PARTNER !!!!!"
                exit
            }
        }
    }
}

# ===========================================================================
# auto-generation: transpose1laneGenerator
# ===========================================================================

# in-place with post-processing, leaves out last steps of processing
# depending on number of lanes, uses lane-oriented zip16

# from swizzle_experiments.tcl: proc transpose1inplace

proc transpose1laneGenerator {elements bytes} {
    set n $elements
    set n2 [expr $n / 2]

    set lanes [expr $bytes / 16]
    if {$lanes < 1} {
        puts "!!!!! ERROR: $bytes BYTES CANNOT FILL $lanes LANES !!!!!"
        exit
    }
    for {set v 0} {$v < $n} {incr v} {
        set isAt($v) $v
    }
    # vs is overwritten after first round
    set vs "inRows"
    set vd "outRows"
    for {set span 1; set parts $n2} \
        {$span <= $n2 / $lanes} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set dst1 0
            set partStart 0
            for {set part 0} \
                {$part < $parts} \
                {incr part; set partStart [expr $partStart + 2 * $span]} {
                    
                    set opsInPart $span
                    set src1 $partStart
                    for {set op 0; set src1 $partStart} \
                        {$op < $opsInPart} \
                        {incr op; incr src1; incr dst1 2} {
                            
                            set src2 [expr $src1 + $span]
                            set dst2 [expr $dst1 + 1]
                            set atSrc1 $isAt($src1)
                            set atSrc2 $isAt($src2)
                            puts -nonewline "        "
                            puts "zip16<$span>($vs\[$atSrc1\], $vs\[$atSrc2\],\
                                               $vd\[$atSrc1\], $vd\[$atSrc2\]);"
                            incr ::numZip16
                            # remember where it should be
                            set newIsAt($dst1) $atSrc1
                            set newIsAt($dst2) $atSrc2
                        }
                }
            set vs "outRows"
            array set isAt [array get newIsAt]
        }
    # post-sorting
    # we try swapping and raise an error if it doesn't work
    for {set v 0} {$v < $n} {incr v} {
        set swapped($v) 0
    }
    # we leave out element 0 and n-1
    for {set v 1} {$v < $n - 1} {incr v} {
        set u $isAt($v)
        # is a swap required? and don't swap twice!
        if {($u != $v) && !$swapped($v)} {
            # do we have a swap partner?
            if {$isAt($u) == $v} {
                puts -nonewline "        "
                puts "std::swap($vd\[$v\], $vd\[$u\]);"
                incr ::numMove 3
                # enought to mark later u (we never reach v again)
                set swapped($u) 1
            } else {
                puts "!!!!! ERROR: NO SWAP PARTNER !!!!!"
                exit
            }
        }
    }
}

# ===========================================================================
# blockTranspose1Generator
# ===========================================================================

# block-wise transposition, in-place with post-processing, uses
# full-length block zip

proc blockTranspose1Generator {elements bytes} {
    set n $elements
    set lanes [expr $bytes / 16]
    if {$lanes < 1} {
        puts "!!!!! ERROR: $bytes BYTES CANNOT FILL $lanes LANES !!!!!"
        exit
    }
    set N $lanes
    set N2 [expr $N / 2]
    set vecsPerLane [expr $n / $lanes]

    for {set b 0} {$b < $N} {incr b} {
        set isAt($b) $b
    }
    # we stay in-place the entire time
    set vs "outRows"
    set vd "outRows"
    for {set span 1; set parts $N2} \
        {$span <= $N2} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set dst1 0
            set partStart 0
            for {set part 0} \
                {$part < $parts} \
                {incr part; set partStart [expr $partStart + 2 * $span]} {
                    
                    set opsInPart $span
                    set src1 $partStart
                    for {set op 0; set src1 $partStart} \
                        {$op < $opsInPart} \
                        {incr op; incr src1; incr dst1 2} {
                            
                            set src2 [expr $src1 + $span]
                            set dst2 [expr $dst1 + 1]
                            set atSrc1 $isAt($src1)
                            set atSrc2 $isAt($src2)
                            # here we zip entire lane blocks
                            set vecAtSrc1 [expr $atSrc1 * $vecsPerLane]
                            set vecAtSrc2 [expr $atSrc2 * $vecsPerLane]
                            set vecSpan [expr $span * $vecsPerLane]
                            for {set v 0} {$v < $vecsPerLane} {incr v} {
                                puts -nonewline "        "
                                puts "zip<$vecSpan>($vs\[$vecAtSrc1\],\
                                                    $vs\[$vecAtSrc2\],\
                                                    $vd\[$vecAtSrc1\],\
                                                    $vd\[$vecAtSrc2\]);"
                                incr ::numZip
                                # in-place
                                incr vecAtSrc1
                                incr vecAtSrc2
                            }
                            # remember where it should be
                            set newIsAt($dst1) $atSrc1
                            set newIsAt($dst2) $atSrc2
                        }
                }
            array set isAt [array get newIsAt]
        }
    # post-sorting
    # we try swapping and raise an error if it doesn't work
    for {set b 0} {$b < $N} {incr b} {
        set swapped($b) 0
    }
    # we leave out element 0 and n-1
    for {set vb 1} {$vb < $N - 1} {incr vb} {
        set ub $isAt($vb)
        # is a swap required? and don't swap twice!
        if {($ub != $vb) && !$swapped($vb)} {
            # do we have a swap partner?
            if {$isAt($ub) == $vb} {
                set uVec [expr $ub * $vecsPerLane]
                set vVec [expr $vb * $vecsPerLane]
                for {set i 0} {$i < $vecsPerLane} {incr i} {
                    puts -nonewline "        "
                    puts "std::swap($vd\[$vVec\], $vd\[$uVec\]);"
                    incr ::numMove 3
                    incr uVec
                    incr vVec
                }
                # enought to mark later u (we never reach v again)
                set swapped($ub) 1
            } else {
                puts "!!!!! ERROR: NO SWAP PARTNER !!!!!"
                exit
            }
        }
    }
}

# ===========================================================================
# transpose2fullGenerator
# ===========================================================================

# from swizzle2inplace and transpose2inplace in swizzle_experiments.tcl
# in-place, full-length zip, no early stop (lanes not used)

proc transpose2fullGenerator {elements} {
    set numVecs $elements
    # special case of swizzle2inplace (as called by transpose2inplace)
    set N [expr $numVecs / 2]
    set finalBlkSize $N
    
    for {set v 0} {$v < $numVecs} {incr v} {
        set isAt($v) $v
    }
    # vs is overwritten after first round
    set vs "inRows"
    set vd "outRows"
    for {set blkSize 1} {$blkSize <= $finalBlkSize} \
        {set blkSize [expr $blkSize * 2]} {
            for {set src 0; set dst 0} {$src < $N} {incr src; incr dst 2} {
                set src1 $src
                set src2 [expr $src + $N]
                set dst1 $dst
                set dst2 [expr $dst + 1]
                set atSrc1 $isAt($src1)
                set atSrc2 $isAt($src2)
                puts -nonewline "        "
                puts "zip<1>($vs\[$atSrc1\], $vs\[$atSrc2\],\
                             $vd\[$atSrc1\], $vd\[$atSrc2\]);"
                incr ::numZip
                # remember where it should be
                set newIsAt($dst1) $atSrc1
                set newIsAt($dst2) $atSrc2
            }
            set vs "outRows"
            array set isAt [array get newIsAt]
        }
    # no post-sorting is required (swizzle2inplace for numVec=2^x)
}

# ===========================================================================
# transpose2laneGenerator
# ===========================================================================

# same as transpose2Generator, but with zip16
# in-place, lane-oriented zip, but no early stopping
# bytes argument not required here (lanes not computed),
# but for post-processing
# TODO: could be fused with transpose2fullGenerator

proc transpose2laneGeneratorA {elements bytes} {
    set numVecs $elements
    # special case of swizzle2inplace (as called by transpose2inplace)
    set N [expr $numVecs / 2]
    set finalBlkSize $N
    
    for {set v 0} {$v < $numVecs} {incr v} {
        set isAt($v) $v
    }
    # vs is overwritten after first round
    set vs "inRows"
    set vd "outRows"
    for {set blkSize 1} {$blkSize <= $finalBlkSize} \
        {set blkSize [expr $blkSize * 2]} {
            for {set src 0; set dst 0} {$src < $N} {incr src; incr dst 2} {
                set src1 $src
                set src2 [expr $src + $N]
                set dst1 $dst
                set dst2 [expr $dst + 1]
                set atSrc1 $isAt($src1)
                set atSrc2 $isAt($src2)
                puts -nonewline "        "
                puts "zip16<1>($vs\[$atSrc1\], $vs\[$atSrc2\],\
                               $vd\[$atSrc1\], $vd\[$atSrc2\]);"
                incr ::numZip16
                # remember where it should be
                set newIsAt($dst1) $atSrc1
                set newIsAt($dst2) $atSrc2
            }
            set vs "outRows"
            array set isAt [array get newIsAt]
        }
    # no post-sorting is required (swizzle2inplace for numVec=2^x)
}

# ===========================================================================
# blockTranspose2GeneratorA
# ===========================================================================

# from swizzle_experiments.tcl: laneTranspose2blkInplace
# in-place with post-processing, uses full length block zip
# (in contrast to transpose2inplace)

proc blockTranspose2GeneratorA {elements bytes} {
    set n $elements
    set n2 [expr  $n / 2]
    set lanes [expr $bytes / 16]
    if {$lanes < 1} {
        puts "!!!!! ERROR: $bytes BYTES CANNOT FILL $lanes LANES !!!!!"
        exit
    }
    set vecsPerLane [expr $n / $lanes]
    for {set v 0} {$v < $n} {incr v} {
        set isAt($v) $v
    }
    # we stay in-place the entire time
    set vs "outRows"
    set vd "outRows"
    # size of zip blk
    for {set blkSz $vecsPerLane; set dstSpan $n2} \
        {$blkSz < $n} \
        {set blkSz [expr 2 * $blkSz]; set dstSpan [expr $dstSpan / 2]} {

            set parts $blkSz
            set partSize [expr $n / $parts]
            
            set dst1 0
            for {set part 0; set partStart 0} \
                {$part < $parts} \
                {incr part; set partStart [expr $partStart + $partSize]} {

                    set opsInPart [expr $partSize / 2]
                    
                    for {set op 0; set src1 $partStart} \
                        {$op < $opsInPart} \
                        {incr op; incr src1 2} {
                            
                            set src2 [expr $src1 + 1]
                            set dst2 [expr $dst1 + $dstSpan]

                            set atSrc1 $isAt($src1)
                            set atSrc2 $isAt($src2)

                            puts -nonewline "        "
                            puts "zip<$blkSz>($vs\[$atSrc1\],\
                                              $vs\[$atSrc2\],\
                                              $vd\[$atSrc1\],\
                                              $vd\[$atSrc2\]);"
                            incr ::numZip
                            
                            # remember where it should be
                            set newIsAt($dst1) $atSrc1
                            set newIsAt($dst2) $atSrc2

                            # TODO: the structure is complex, this was
                            # TODO: found empirically, is there a
                            # TODO: better way, like another variable that
                            # TODO: dst1 could be derived from?
                            incr dst1
                            if {($dst1 % $dstSpan) == 0} {
                                set dst1 [expr $dst1 + $dstSpan]
                            }
                        }
                }
            array set isAt [array get newIsAt]
        }
    # post-sorting: repeated ring-swap
    for {set v 0} {$v < $n} {incr v} {
        set swapped($v) 0
    }
    # we leave out element 0 and n-1
    for {set v 1} {$v < $n - 1} {incr v} {
        set u $isAt($v)
        # is a swap required? and don't swap twice!
        if {($u != $v) && !$swapped($v)} {
            # remember the first one, this one is overwritten
            puts            "        \{"
            puts -nonewline "          "
            puts "Vec<T, SIMD_WIDTH> vec_v = outRows\[$v\];"
            incr ::numMove
            # go through cycle
            set w $v
            while {1} {
                set swapped($w) 1
                set u $isAt($w)
                puts -nonewline "          "
                if {$u == $v} {
                    puts "outRows\[$w\] = vec_v;"
                } else {
                    puts "outRows\[$w\] = outRows\[$u\];"
                }
                incr ::numMove
                set w $u
                if {$w == $v} break;
            }
            puts            "        \}"
        }
    }
}

# ===========================================================================
# transpose2laneGeneratorB
# ===========================================================================

# same as transpose2Generator, but with zip16
# in-place, lane-oriented zip, with early stopping

proc transpose2laneGeneratorB {elements bytes} {
    set numVecs $elements
    # special case of swizzle2inplace (as called by transpose2inplace)
    set N [expr $numVecs / 2]
    set finalBlkSize $N
    set lanes [expr $bytes / 16]
    if {$lanes < 1} {
        puts "!!!!! ERROR: $bytes BYTES CANNOT FILL $lanes LANES !!!!!"
        exit
    }
    
    for {set v 0} {$v < $numVecs} {incr v} {
        set isAt($v) $v
    }
    # vs is overwritten after first round
    set vs "inRows"
    set vd "outRows"
    # difference to transpose2laneGeneratorA: / $lanes (early stopping)
    for {set blkSize 1} {$blkSize <= $finalBlkSize / $lanes} \
        {set blkSize [expr $blkSize * 2]} {
            for {set src 0; set dst 0} {$src < $N} {incr src; incr dst 2} {
                set src1 $src
                set src2 [expr $src + $N]
                set dst1 $dst
                set dst2 [expr $dst + 1]
                set atSrc1 $isAt($src1)
                set atSrc2 $isAt($src2)
                puts -nonewline "        "
                puts "zip16<1>($vs\[$atSrc1\], $vs\[$atSrc2\],\
                               $vd\[$atSrc1\], $vd\[$atSrc2\]);"
                incr ::numZip16
                # remember where it should be
                set newIsAt($dst1) $atSrc1
                set newIsAt($dst2) $atSrc2
            }
            set vs "outRows"
            array set isAt [array get newIsAt]
        }
    # no post-sorting is required (swizzle2inplace for numVec=2^x)
}

# ===========================================================================
# blockTranspose2GeneratorB
# ===========================================================================

# from swizzle_experiments.tcl: laneTranspose2inplace
# in-place with post-processing, uses full length block zip
# (in contrast to transpose2inplace)

proc blockTranspose2GeneratorB {elements bytes} {
    set n $elements
    set n2 [expr  $n / 2]
    set lanes [expr $bytes / 16]
    if {$lanes < 1} {
        puts "!!!!! ERROR: $bytes BYTES CANNOT FILL $lanes LANES !!!!!"
        exit
    }
    set vecsPerLane [expr $n / $lanes]
    for {set v 0} {$v < $n} {incr v} {
        set isAt($v) $v
    }
    # we stay in-place the entire time
    set vs "outRows"
    set vd "outRows"
    
    for {set numParts $vecsPerLane; set blkSz 1; set dstSpan $n2} \
        {$numParts < $n} \
        {set numParts [expr $numParts * 2]; \
             set dstSpan [expr $dstSpan / 2]; \
             set blkSz [expr $blkSz * 2]} {

                 set partSize [expr $n / $numParts]
                 set dst1 0

                 for {set part 0; set partStart 0} \
                     {$part < $numParts} \
                     {incr part; incr partStart $partSize} {

                         for {set op 0; set src1 $partStart} \
                             {$op < $partSize / 2} \
                             {incr op; incr src1 2} {

                                 set src2 [expr $src1 + 1]
                                 set dst2 [expr $dst1 + $dstSpan]

                                 set atSrc1 $isAt($src1)
                                 set atSrc2 $isAt($src2)

                                 puts -nonewline "        "
                                 puts "zip<$blkSz>($vs\[$atSrc1\],\
                                                   $vs\[$atSrc2\],\
                                                   $vd\[$atSrc1\],\
                                                   $vd\[$atSrc2\]);"
                                 incr ::numZip
 
                                 # remember where it should be
                                 set newIsAt($dst1) $atSrc1
                                 set newIsAt($dst2) $atSrc2

                                 # TODO: the structure is complex, this was
                                 # TODO: found empirically, is there a
                                 # TODO: better way, like another variable that
                                 # TODO: dst1 could be derived from?
                                 incr dst1
                                 if {($dst1 % $dstSpan) == 0} {
                                     set dst1 [expr $dst1 + $dstSpan]
                                 }
                             }
                     }
                 array set isAt [array get newIsAt]
             }
    # post-sorting: repeated ring-swap
    for {set v 0} {$v < $n} {incr v} {
        set swapped($v) 0
    }
    # we leave out element 0 and n-1
    for {set v 1} {$v < $n - 1} {incr v} {
        set u $isAt($v)
        # is a swap required? and don't swap twice!
        if {($u != $v) && !$swapped($v)} {
            # remember the first one, this one is overwritten
            puts            "        \{"
            puts -nonewline "          "
            puts "Vec<T, SIMD_WIDTH> vec_v = outRows\[$v\];"
            incr ::numMove
            # go through cycle
            set w $v
            while {1} {
                set swapped($w) 1
                set u $isAt($w)
                puts -nonewline "          "
                if {$u == $v} {
                    puts "outRows\[$w\] = vec_v;"
                } else {
                    puts "outRows\[$w\] = outRows\[$u\];"
                }
                incr ::numMove
                set w $u
                if {$w == $v} break;
            }
            puts            "        \}"
        }
    }
}

# ===========================================================================
# auto-generation of entire function
# ===========================================================================

proc transposeCoreAutoGen {funcName elements \
                               transposeGenerator} {
    resetCounters
    puts "    [format $::transposeCoreTemplateHead $funcName $elements]"
    puts "    \{"
    puts -nonewline [$transposeGenerator $elements]
    puts "    \}"
    storeCounters "$funcName elements=$elements"
}

proc transposeLaneCoreAutoGen {funcName elements bytes \
                                   transposeLaneGenerator \
                                   blockTransposeGenerator} {
    resetCounters
    puts "    [format $::transposeLaneCoreTemplateHead $funcName \
                      $elements $bytes]"
    puts "    \{"
    puts -nonewline [$transposeLaneGenerator $elements $bytes]
    puts "        // correction steps follow below (if required)"
    puts -nonewline [$blockTransposeGenerator $elements $bytes]
    puts "    \}"
    storeCounters "$funcName elements=$elements bytes=$bytes"
}

# ===========================================================================
# main
# ===========================================================================

resetCounterInfo

puts $header
puts $includeGuardStart
puts $includes
puts "$simdStart"
puts "$internalExtStart"

puts "    // =========================================================="
puts "    // transpose1inplc"
puts "    // =========================================================="
foreach elements $elementsList {
    transposeCoreAutoGen "transpose1inplc" $elements \
        transpose1fullGenerator
}
puts [format $transposeHubTemplate \
          "transpose1inplc" "transpose1inplc"]

puts "    // =========================================================="
puts "    // transpose1inplcLane"
puts "    // =========================================================="
foreach bytes $bytesList {
    foreach bytesPerElement $bytesPerElementList {
        set elements [expr $bytes / $bytesPerElement]
        transposeLaneCoreAutoGen "transpose1inplcLane" $elements $bytes \
            transpose1laneGenerator blockTranspose1Generator
    }
}
puts [format $transposeLaneHubTemplate \
          "transpose1inplcLane" "transpose1inplcLane"]

puts "    // =========================================================="
puts "    // transpose2inplc"
puts "    // =========================================================="
foreach elements $elementsList {
    transposeCoreAutoGen "transpose2inplc" $elements \
        transpose2fullGenerator
}
puts [format $transposeHubTemplate \
          "transpose2inplc" "transpose2inplc"]

if {0} {
    # not as efficient as version below, more ops in first phase
    puts "    // =========================================================="
    puts "    // transpose2inplcLaneA"
    puts "    // =========================================================="
    foreach bytes $bytesList {
        foreach bytesPerElement $bytesPerElementList {
            set elements [expr $bytes / $bytesPerElement]
            transposeLaneCoreAutoGen "transpose2inplcLaneA" $elements $bytes \
                transpose2laneGeneratorA blockTranspose2GeneratorA
        }
    }
    puts [format $transposeLaneHubTemplate \
              "transpose2inplcLaneA" "transpose2inplcLaneA"]
}

puts "    // =========================================================="
puts "    // transpose2inplcLane"
puts "    // =========================================================="
foreach bytes $bytesList {
    foreach bytesPerElement $bytesPerElementList {
        set elements [expr $bytes / $bytesPerElement]
        transposeLaneCoreAutoGen "transpose2inplcLane" $elements $bytes \
            transpose2laneGeneratorB blockTranspose2GeneratorB
    }
}
puts [format $transposeLaneHubTemplate \
          "transpose2inplcLane" "transpose2inplcLane"]

puts "$internalExtEnd"
puts "$simdEnd"
puts $includeGuardEnd

writeCounterInfo "transpose_inplace_autogen_stat.txt"
