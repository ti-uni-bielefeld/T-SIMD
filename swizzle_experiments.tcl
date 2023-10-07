#!/usr/bin/env tclsh
# ===========================================================================
#
# swizzle_experiments.tcl --
# program with tcl-based experiments, was used to design
# auto-generation code of in-place swizzle and transpose functions
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

# ===========================================================================
# for debugging
# ===========================================================================

proc printVecsArray {arrayName n} {
    upvar $arrayName a
    for {set v 0} {$v < $n} {incr v} {
        puts -nonewline "\[ "
        for {set i 0} {$i < $n} {incr i} {
            puts -nonewline "[format %3d [lindex $a($v) $i]] "
        }
        puts "\]"
    }
    puts ""
}
    
# ===========================================================================
# zip functions
# ===========================================================================

# factors: 2^n
proc splitFactor {factor vec} {
    set lf [expr [llength $vec] / $factor]
    set r [list]
    set start 0
    set end [expr $start + $lf - 1]
    for {set i 0} {$i < $factor} {incr i} {
        lappend r [lrange $vec $start $end]
        incr start $lf
        incr end $lf
    }
    return $r
}

proc fullZip {avec bvec} {
    set r [list]
    foreach a $avec b $bvec {
        lappend r $a $b
    }
    return [splitFactor 2 $r]
}

proc zipLanes {lanes avec bvec} {
    #puts "avec $avec"
    #puts "bvec $bvec"
    set avecs [splitFactor $lanes $avec]
    set bvecs [splitFactor $lanes $bvec]
    #puts "avecs $avecs"
    #puts "bvecs $bvecs"
    set lvecs [list]
    set hvecs [list]
    foreach a $avecs b $bvecs {
        lassign [fullZip $a $b] l h
        # puts "$a , $b -> $l , $h"
        lappend lvecs {*}$l
        lappend hvecs {*}$h
    }
    return [list $lvecs $hvecs]
}

proc zip_1Lanes {avec bvec} {
    return [zipLanes 1 $avec $bvec]
}

proc zip_2Lanes {avec bvec} {
    return [zipLanes 2 $avec $bvec]
}

proc zip_4Lanes {avec bvec} {
    return [zipLanes 4 $avec $bvec]
}

proc zip_8Lanes {avec bvec} {
    return [zipLanes 8 $avec $bvec]
}

# ===========================================================================
# block-zip functions
# ===========================================================================

proc makeBlock {blkSz vec} {
    set blkVec [list]
    set l [llength $vec]
    for {set i 0} {$i < $l} {incr i $blkSz} {
        set partVec [list]
        for {set j 0} {$j < $blkSz} {incr j} {
            lappend partVec [lindex $vec [expr $i + $j]]
        }
        lappend blkVec $partVec
    }
    return $blkVec
}

proc fullZipBlk {blkSz avec bvec} {
    set ablk [makeBlock $blkSz $avec]
    set bblk [makeBlock $blkSz $bvec]
    set r [list]
    foreach a $ablk b $bblk {
        lappend r {*}$a {*}$b
    }
    return [splitFactor 2 $r]
}

proc zipBlkLanes {blkSz lanes avec bvec} {
    #puts "blkSz $blkSz lanes $lanes"
    #puts "avec $avec"
    #puts "bvec $bvec"
    set avecs [splitFactor $lanes $avec]
    set bvecs [splitFactor $lanes $bvec]
    #puts "avecs $avecs"
    #puts "bvecs $bvecs"
    set lvecs [list]
    set hvecs [list]
    foreach a $avecs b $bvecs {
        lassign [fullZipBlk $blkSz $a $b] l h
        #puts "a $a"
        #puts "b $b"
        #puts "l $l"
        #puts "h $h"
        lappend lvecs {*}$l
        lappend hvecs {*}$h
    }
    return [list $lvecs $hvecs]
}

proc zipBlk_1Lanes {blkSz avec bvec} {
    return [zipBlkLanes $blkSz 1 $avec $bvec]
}

proc zipBlk_2Lanes {blkSz avec bvec} {
    return [zipBlkLanes $blkSz 2 $avec $bvec]
}

proc zipBlk_4Lanes {blkSz avec bvec} {
    return [zipBlkLanes $blkSz 4 $avec $bvec]
}

proc zipBlk_8Lanes {blkSz avec bvec} {
    return [zipBlkLanes $blkSz 8 $avec $bvec]
}

# ===========================================================================
# swizzle2: with copying back, list version
# ===========================================================================

# TODO: what was the effect of restricting the number of loop through
# TODO: parameter lanes?

proc swizzle2 {zipFct lanes N finalBlkSize vecs} {
    puts "swizzle2 zipFct=$zipFct lanes=$lanes N=$N finalBlkSize=$finalBlkSize" 
    set elems [llength [lindex $vecs 0]]
    # puts "swizzle2 N = $N finalBlkSize = $finalBlkSize elems = $elems"
    for {set blkSize 1} {$blkSize <= $finalBlkSize / $lanes} \
        {set blkSize [expr $blkSize * 2]} {
            # puts $vecs
            set newVecs [list]
            for {set src 0} {$src < $N} {incr src} {
                set r [$zipFct \
                           [lindex $vecs $src] [lindex $vecs [expr $src + $N]]]
                lappend newVecs {*}$r
            }
            set vecs $newVecs
        }
    # puts $vecs
    # puts "-------------------"
    return $vecs
}

# ===========================================================================
# swizzle2idx: with copying back, index version
# ===========================================================================

proc swizzle2idx {zipFct lanes N finalBlkSize vecs} {
    puts "swizzle2idx zipFct=$zipFct lanes=$lanes N=$N finalBlkSize=$finalBlkSize" 
    set numVecs [llength $vecs]
    set elems [llength [lindex $vecs 0]]
    # puts "swizzle2idx N = $N finalBlkSize = $finalBlkSize elems = $elems"
    for {set v 0} {$v < $numVecs} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
    }
    for {set blkSize 1} {$blkSize <= $finalBlkSize / $lanes} \
        {set blkSize [expr $blkSize * 2]} {
            for {set src 0; set dst 0} {$src < $N} {incr src; incr dst 2} {
                set r [$zipFct $vecsArray($src) $vecsArray([expr $src + $N])]
                set newVecsArray($dst) [lindex $r 0]
                set newVecsArray([expr $dst + 1]) [lindex $r 1]
            }
            array set vecsArray [array get newVecsArray]
            # printVecsArray newVecsArray $numVecs
        }
    set vecs [list]
    for {set v 0} {$v < $numVecs} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs
}

# ===========================================================================
# swizzle2inplace: inplace version (with indices)
# ===========================================================================

# https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/

proc isPowerOfTwo {x} {
    return [expr !($x & ($x - 1))]
}

proc swizzle2inplace {zipFct lanes N finalBlkSize vecs} {
    puts "swizzle2inplace zipFct=$zipFct lanes=$lanes N=$N finalBlkSize=$finalBlkSize" 
    set numVecs [llength $vecs]
    set elems [llength [lindex $vecs 0]]
    for {set v 0} {$v < $numVecs} {incr v} {
        set isAt($v) $v
        set vecsArray($v) [lindex $vecs $v]
    }
    for {set blkSize 1} {$blkSize <= $finalBlkSize / $lanes} \
        {set blkSize [expr $blkSize * 2]} {
            for {set src 0; set dst 0} {$src < $N} {incr src; incr dst 2} {
                set src1 $src
                set src2 [expr $src + $N]
                set dst1 $dst
                set dst2 [expr $dst + 1]
                set atSrc1 $isAt($src1)
                set atSrc2 $isAt($src2)
                set r [$zipFct $vecsArray($atSrc1) $vecsArray($atSrc2)]
                # in-place
                set vecsArray($atSrc1) [lindex $r 0]
                set vecsArray($atSrc2) [lindex $r 1]
                # remember where it should be
                set newIsAt($dst1) $atSrc1
                set newIsAt($dst2) $atSrc2
            }
            array set isAt [array get newIsAt]
            if {1} {
                puts -nonewline "isAt: "
                for {set v 0} {$v < $numVecs} {incr v} {
                    puts -nonewline "[format %2d $isAt($v)] "
                }
                puts ""
            }
            # printVecsArray vecsArray $numVecs
        }
    if {1} {
        puts -nonewline "idx:  "
        for {set v 0} {$v < $numVecs} {incr v} {
            puts -nonewline "[format %2d $v] "
        }
        puts ""
    }
    set vecs [list]
    if {0} {
        for {set v 0} {$v < $numVecs} {incr v} {
            lappend vecs $vecsArray($isAt($v))
        }
    }
    # we can leave out first and last;
    # for all powers of 2 we don't need to do anything
    # TODO: swap operation for in-place? (probably not possible in general)
    if {1} {
        set numVecsM1 [expr $numVecs - 1]
        if {![isPowerOfTwo $N]} {
            lappend vecs $vecsArray(0)
            for {set v 1} {$v < $numVecsM1} {incr v} {
                lappend vecs $vecsArray($isAt($v))
            }
            lappend vecs $vecsArray($numVecsM1)
        } else {
            # only required in this Tcl implementation
            for {set v 0} {$v < $numVecs} {incr v} {
                lappend vecs $vecsArray($v)
            }
        }
    }
    return $vecs
}

# ===========================================================================
# tranpose2idx: based on swizzle2idx
# transpose2inplace: based on swizzle2inplace
# ===========================================================================

proc transpose2idx {zipFct lanes vecs} {
    puts "transpose2inplace zipFct=$zipFct lanes=$lanes" 
    set n [llength $vecs]
    set N [expr $n / 2]
    return [swizzle2idx $zipFct $lanes $N $N $vecs]
}

proc transpose2inplace {zipFct lanes vecs} {
    puts "transpose2inplace zipFct=$zipFct lanes=$lanes" 
    set n [llength $vecs]
    set N [expr $n / 2]
    return [swizzle2inplace $zipFct $lanes $N $N $vecs]
}

# ===========================================================================
# transpose1: with copy-back step, list version
# ===========================================================================

# has the advantage that with lane-oriented zip as zipFct we just have
# to reorganize the vector order without accessing parts of vectors
# (as required with transpose2)

# lanes argument used to restrict the number of iterations
proc transpose1 {zipBlkFct lanes vecs} {
    puts "transpose1 zipBlkFct=$zipBlkFct lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr $n / 2]

    for {set span 1; set parts $n2} \
        {$span <= $n2 / $lanes} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set newVecs [list]
            puts "span=$span parts=$parts"
            set partStart 0
            for {set part 0} \
                {$part < $parts} \
                {incr part; set partStart [expr $partStart + 2 * $span]} {
                    
                    set opsInPart $span
                    set src1 $partStart
                    for {set op 0; set src1 $partStart} \
                        {$op < $opsInPart} \
                        {incr op; incr src1} {
                            
                            set src2 [expr $src1 + $span]
                            puts -nonewline "($src1,$src2) "
                            set r [$zipBlkFct $span \
                                       [lindex $vecs $src1] \
                                       [lindex $vecs $src2]]
                            lappend newVecs {*}$r
                        }
                }
            puts ""
            set vecs $newVecs
        }
    return $vecs    
}

# ===========================================================================
# transpose1idx: with copy-back step, index version
# ===========================================================================

# lanes argument used to restrict the number of iterations
proc transpose1idx {zipBlkFct lanes vecs} {
    puts "transpose1idx zipBlkFct=$zipBlkFct lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr $n / 2]
    puts "n=$n"
    
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
    }
    for {set span 1; set parts $n2} \
        {$span <= $n2 / $lanes} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set dst1 0
            puts "span=$span parts=$parts"
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
                            puts -nonewline "($src1,$src2)->($dst1,$dst2) "
                            set r [$zipBlkFct $span \
                                       $vecsArray($src1) \
                                       $vecsArray($src2)]
                            set newVecsArray($dst1) [lindex $r 0]
                            set newVecsArray($dst2) [lindex $r 1]
                        }
                }
            puts ""
            array set vecsArray [array get newVecsArray]
            # printVecsArray vecsArray $n
        }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs    
}

# ===========================================================================
# transpose1inplace: inplace, index version
# ===========================================================================

proc swapVecsInArray {arrayName index1 index2} {
    upvar $arrayName a
    set tmp $a($index1)
    set a($index1) $a($index2)
    set a($index2) $tmp
}

# lanes argument used to restrict the number of iterations
proc transpose1inplace {zipBlkFct lanes vecs} {
    puts "transpose1inplace zipBlkFct=$zipBlkFct lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr $n / 2]
    puts "n=$n"
    
    for {set v 0} {$v < $n} {incr v} {
        set isAt($v) $v
        set vecsArray($v) [lindex $vecs $v]
    }
    for {set span 1; set parts $n2} \
        {$span <= $n2 / $lanes} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set dst1 0
            puts "span=$span parts=$parts"
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
                            # puts -nonewline "($src1,$src2)->($dst1,$dst2) "
                            puts -nonewline "($atSrc1,$atSrc2) "
                            set r [$zipBlkFct $span \
                                       $vecsArray($atSrc1) \
                                       $vecsArray($atSrc2)]
                            # in-place
                            set vecsArray($atSrc1) [lindex $r 0]
                            set vecsArray($atSrc2) [lindex $r 1]
                            # remember where it should be
                            set newIsAt($dst1) $atSrc1
                            set newIsAt($dst2) $atSrc2
                        }
                }
            puts ""
            array set isAt [array get newIsAt]
            if {1} {
                puts -nonewline "isAt: "
                for {set v 0} {$v < $n} {incr v} {
                    puts -nonewline "[format %2d $isAt($v)] "
                }
                puts ""
            }
            if {1} {
                puts -nonewline "idx:  "
                for {set v 0} {$v < $n} {incr v} {
                    puts -nonewline "[format %2d $v] "
                }
                puts ""
            }
        }
    # post-sorting
    # we try swapping and raise an error if it doesn't work
    if {1} {
        for {set v 0} {$v < $n} {incr v} {
            set swapped($v) 0
        }
        puts -nonewline "swap: "
        # we leave out element 0 and n-1
        for {set v 1} {$v < $n - 1} {incr v} {
            set u $isAt($v)
            # is a swap required? and don't swap twice!
            if {($u != $v) && !$swapped($v)} {
                # do we have a swap partner?
                if {$isAt($u) == $v} {
                    puts -nonewline "($v,$u) "
                    swapVecsInArray vecsArray $u $v
                    # enought to mark later u (we never reach v again)
                    set swapped($u) 1
                } else {
                    puts "!!!!! ERROR: NO SWAP PARTNER !!!!!"
                    exit
                }
            }
        }
        puts ""
    }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs    
}

# ===========================================================================
# laneTranspose1idx
# ===========================================================================

# for post-processing of transpose1 with lane-oriented unpacks
# assumption: always needs full block transpose
# as transpose1idx, but for blocks

# lanes argument is here used to organize the block-wise processing,
# not to restrict the number of iterations

proc laneTranspose1idx {lanes vecs} {
    puts "laneTranspose1idx lanes=$lanes" 
    set n [llength $vecs]
    set N $lanes
    set N2 [expr $N / 2]
    set vecsPerLane [expr $n / $lanes]
    puts "n=$n N=$N vecsPerLane=$vecsPerLane"
    
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
    }
    for {set span 1; set parts $N2} \
        {$span <= $N2} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set dst1 0
            puts "span=$span parts=$parts"
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
                            puts -nonewline "($src1,$src2)->($dst1,$dst2) "
                            # here we zip entire lane blocks
                            set vecSrc1 [expr $src1 * $vecsPerLane]
                            set vecSrc2 [expr $src2 * $vecsPerLane]
                            set vecDst1 [expr $dst1 * $vecsPerLane]
                            set vecDst2 [expr $dst2 * $vecsPerLane]
                            set vecSpan [expr $span * $vecsPerLane]
                            for {set v 0} {$v < $vecsPerLane} {incr v} {
                                set r [fullZipBlk $vecSpan \
                                           $vecsArray($vecSrc1) \
                                           $vecsArray($vecSrc2)]
                                set newVecsArray($vecDst1) [lindex $r 0]
                                set newVecsArray($vecDst2) [lindex $r 1]
                                incr vecSrc1
                                incr vecSrc2
                                incr vecDst1
                                incr vecDst2
                            }
                        }
                }
            puts ""
            array set vecsArray [array get newVecsArray]
            # printVecsArray vecsArray $n
        }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs    
}

# ===========================================================================
# laneTranspose1inplace
# ===========================================================================

proc laneTranspose1inplace {lanes vecs} {
    puts "laneTranspose1inplace lanes=$lanes" 
    set n [llength $vecs]
    set N $lanes
    set N2 [expr $N / 2]
    set vecsPerLane [expr $n / $lanes]
    puts "n=$n N=$N vecsPerLane=$vecsPerLane"
    
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
    }
    for {set b 0} {$b < $N} {incr b} {
        set isAt($b) $b
    }
    for {set span 1; set parts $N2} \
        {$span <= $N2} \
        {set span [expr 2 * $span]; set parts [expr $parts / 2]} {

            set dst1 0
            puts "span=$span parts=$parts"
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
                            # puts -nonewline "($src1,$src2)->($dst1,$dst2) "
                            puts -nonewline "($atSrc1,$atSrc2) "
                            # here we zip entire lane blocks
                            set vecAtSrc1 [expr $atSrc1 * $vecsPerLane]
                            set vecAtSrc2 [expr $atSrc2 * $vecsPerLane]
                            set vecSpan [expr $span * $vecsPerLane]
                            for {set v 0} {$v < $vecsPerLane} {incr v} {
                                set r [fullZipBlk $vecSpan \
                                           $vecsArray($vecAtSrc1) \
                                           $vecsArray($vecAtSrc2)]
                                # in-place
                                set vecsArray($vecAtSrc1) [lindex $r 0]
                                set vecsArray($vecAtSrc2) [lindex $r 1]
                                incr vecAtSrc1
                                incr vecAtSrc2
                            }
                            # remember where it should be
                            set newIsAt($dst1) $atSrc1
                            set newIsAt($dst2) $atSrc2
                        }
                }
            puts ""
            array set isAt [array get newIsAt]
            # printVecsArray vecsArray $n
            if {1} {
                puts -nonewline "isAt: "
                for {set b 0} {$b < $N} {incr b} {
                    puts -nonewline "[format %2d $isAt($b)] "
                }
                puts ""
            }
        }
    if {1} {
        puts -nonewline "idx:  "
        for {set b 0} {$b < $N} {incr b} {
            puts -nonewline "[format %2d $b] "
        }
        puts ""
    }
    # post-sorting
    # we try swapping and raise an error if it doesn't work
    if {1} {
        for {set b 0} {$b < $n} {incr b} {
            set swapped($b) 0
        }
        puts -nonewline "swap: "
        # we leave out element 0 and n-1
        for {set vb 1} {$vb < $N - 1} {incr vb} {
            set ub $isAt($vb)
            # is a swap required? and don't swap twice!
            if {($ub != $vb) && !$swapped($vb)} {
                # do we have a swap partner?
                if {$isAt($ub) == $vb} {
                    puts -nonewline "($vb,$ub) "
                    set uVec [expr $ub * $vecsPerLane]
                    set vVec [expr $vb * $vecsPerLane]
                    for {set i 0} {$i < $vecsPerLane} {incr i} {
                        swapVecsInArray vecsArray $uVec $vVec
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
        puts ""
    }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs    
}

# ===========================================================================
# laneTranspose2blkIdx
# ===========================================================================

# for use without early termination in laneTranspose2*
# uses zip<blkSz>

proc laneTranspose2blkIdx {lanes vecs} {
    puts "laneTranspose2blkIdx lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr  $n / 2]
    set vecsPerLane [expr $n / $lanes]
    puts "n=$n vecsPerLane=$vecsPerLane"
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
    }

    # size of zip blk
    for {set blkSz $vecsPerLane; set dstSpan $n2} \
        {$blkSz < $n} \
        {set blkSz [expr 2 * $blkSz]; set dstSpan [expr $dstSpan / 2]} {

            set parts $blkSz
            set partSize [expr $n / $parts]
            puts "blkSz=$blkSz parts=$parts partSize=$partSize dstSpan=$dstSpan"

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

                            puts "part=$part partStart=$partStart op=$op \
                                  ($src1,$src2)->($dst1,$dst2) "
                            flush stdout
                            set r [fullZipBlk $blkSz \
                                       $vecsArray($src1) \
                                       $vecsArray($src2)]
                            set newVecsArray($dst1) [lindex $r 0]
                            set newVecsArray($dst2) [lindex $r 1]

                            # TODO: the structure is complex, this was
                            # TODO: found empirically, it there a
                            # TODO: better way, like another variable that
                            # TODO: dst1 could be derived from?
                            incr dst1
                            if {($dst1 % $dstSpan) == 0} {
                                set dst1 [expr $dst1 + $dstSpan]
                            }
                        }
                }
            array set vecsArray [array get newVecsArray]
            printVecsArray vecsArray $n
        }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs    

}
                                  
# ===========================================================================
# laneTranspose2blkInplace
# ===========================================================================

# for use without early termination in laneTranspose2*
# uses zip<blkSz>

proc laneTranspose2blkInplace {lanes vecs} {
    puts "laneTranspose2blkIdx lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr  $n / 2]
    set vecsPerLane [expr $n / $lanes]
    puts "n=$n vecsPerLane=$vecsPerLane"
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
        set isAt($v) $v
    }
    # size of zip blk
    for {set blkSz $vecsPerLane; set dstSpan $n2} \
        {$blkSz < $n} \
        {set blkSz [expr 2 * $blkSz]; set dstSpan [expr $dstSpan / 2]} {

            set parts $blkSz
            set partSize [expr $n / $parts]
            puts "blkSz=$blkSz parts=$parts partSize=$partSize dstSpan=$dstSpan"

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
                           
                            #puts "part=$part partStart=$partStart op=$op \
                            #      ($src1,$src2)->($dst1,$dst2) "

                            puts -nonewline "($atSrc1,$atSrc2) "
                            flush stdout
                            
                            set r [fullZipBlk $blkSz \
                                       $vecsArray($atSrc1) \
                                       $vecsArray($atSrc2)]
                            # in-place
                            set vecsArray($atSrc1) [lindex $r 0]
                            set vecsArray($atSrc2) [lindex $r 1]

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
            puts ""
            array set isAt [array get newIsAt]
            if {1} {
                puts -nonewline "isAt: "
                for {set v 0} {$v < $n} {incr v} {
                    puts -nonewline "[format %2d $isAt($v)] "
                }
                puts ""
            }
            # printVecsArray vecsArray $n
        }
    if {1} {
        puts -nonewline "idx:  "
        for {set v 0} {$v < $n} {incr v} {
            puts -nonewline "[format %2d $v] "
        }
        puts ""
    }
    # post-sorting: repeated ring-swap
    if {1} {
        for {set v 0} {$v < $n} {incr v} {
            set swapped($v) 0
        }
        puts -nonewline "swap: "
        # we leave out element 0 and n-1
        for {set v 1} {$v < $n - 1} {incr v} {
            set u $isAt($v)
            # is a swap required? and don't swap twice!
            if {($u != $v) && !$swapped($v)} {
                # remember the first one, this one is overwritten
                set vec_v $vecsArray($v)
                puts -nonewline "rem($v), "
                # go through cycle
                set w $v
                while {1} {
                    set swapped($w) 1
                    set u $isAt($w)
                    puts -nonewline "$w<=$u "
                    if {$u == $v} {
                        set vecsArray($w) $vec_v
                    } else {
                        set vecsArray($w) $vecsArray($u)
                    }
                    set w $u
                    if {$w == $v} break;
                }
                puts ""
            }
        }
    }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    return $vecs
}
                                  
# ===========================================================================
# laneTranspose2idx
# ===========================================================================

# for use with early termination in laneTranspose2*
# uses zip<blkSz>

proc laneTranspose2idx {lanes vecs} {
    
    puts "laneTranspose2idx lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr  $n / 2]
    set vecsPerLane [expr $n / $lanes]
    puts "n=$n vecsPerLane=$vecsPerLane"
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
    }

    for {set numParts $vecsPerLane; set blkSz 1; set dstSpan $n2} \
        {$numParts < $n} \
        {set numParts [expr $numParts * 2]; \
             set dstSpan [expr $dstSpan / 2]; \
             set blkSz [expr $blkSz * 2]} {

                 set partSize [expr $n / $numParts]
                 puts "numParts=$numParts partSize=$partSize\
                       blkSz=$blkSz dstSpan=$dstSpan"
                 set dst1 0

                 for {set part 0; set partStart 0} \
                     {$part < $numParts} \
                     {incr part; incr partStart $partSize} {

                         set src1 $partStart

                         for {set op 0} \
                             {$op < $partSize / 2} \
                             {incr op} {

                                 set src2 [expr $src1 + 1]
                                 set dst2 [expr $dst1 + $dstSpan]
                                 puts "part $part blkSz $blkSz\
                                  ($src1,$src2)->($dst1,$dst2) "
                                 set r [fullZipBlk $blkSz \
                                            $vecsArray($src1) \
                                            $vecsArray($src2)]
                                 set newVecsArray($dst1) [lindex $r 0]
                                 set newVecsArray($dst2) [lindex $r 1]
                                 incr src1 2
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
                 array set vecsArray [array get newVecsArray]
                 printVecsArray vecsArray $n
             }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    
    return $vecs
}

# ===========================================================================
# laneTranspose2inplace
# ===========================================================================

# for use with early termination in laneTranspose2*
# uses zip<blkSz>

proc laneTranspose2inplace {lanes vecs} {
    
    puts "laneTranspose2inplace lanes=$lanes" 
    set n [llength $vecs]
    set n2 [expr  $n / 2]
    set vecsPerLane [expr $n / $lanes]
    puts "n=$n vecsPerLane=$vecsPerLane"
    for {set v 0} {$v < $n} {incr v} {
        set vecsArray($v) [lindex $vecs $v]
        set isAt($v) $v
    }

    for {set numParts $vecsPerLane; set blkSz 1; set dstSpan $n2} \
        {$numParts < $n} \
        {set numParts [expr $numParts * 2]; \
             set dstSpan [expr $dstSpan / 2]; \
             set blkSz [expr $blkSz * 2]} {

                 set partSize [expr $n / $numParts]
                 puts "numParts=$numParts partSize=$partSize\
                       blkSz=$blkSz dstSpan=$dstSpan"
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

                                 puts -nonewline "($atSrc1,$atSrc2) "

                                 set r [fullZipBlk $blkSz \
                                            $vecsArray($atSrc1) \
                                            $vecsArray($atSrc2)]
                                 # in-place
                                 set vecsArray($atSrc1) [lindex $r 0]
                                 set vecsArray($atSrc2) [lindex $r 1]

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
                 puts ""
                 array set isAt [array get newIsAt]
                 if {1} {
                     puts -nonewline "isAt: "
                     for {set v 0} {$v < $n} {incr v} {
                         puts -nonewline "[format %2d $isAt($v)] "
                     }
                     puts ""
                 }
                 # printVecsArray vecsArray $n
             }
    if {1} {
        puts -nonewline "idx:  "
        for {set v 0} {$v < $n} {incr v} {
            puts -nonewline "[format %2d $v] "
        }
        puts ""
    }
    # post-sorting: repeated ring-swap
    if {1} {
        for {set v 0} {$v < $n} {incr v} {
            set swapped($v) 0
        }
        puts -nonewline "swap: "
        # we leave out element 0 and n-1
        for {set v 1} {$v < $n - 1} {incr v} {
            set u $isAt($v)
            # is a swap required? and don't swap twice!
            if {($u != $v) && !$swapped($v)} {
                # remember the first one, this one is overwritten
                set vec_v $vecsArray($v)
                puts -nonewline "rem($v), "
                # go through cycle
                set w $v
                while {1} {
                    set swapped($w) 1
                    set u $isAt($w)
                    puts -nonewline "$w<=$u "
                    if {$u == $v} {
                        set vecsArray($w) $vec_v
                    } else {
                        set vecsArray($w) $vecsArray($u)
                    }
                    set w $u
                    if {$w == $v} break;
                }
                puts ""
            }
        }
    }
    set vecs [list]
    for {set v 0} {$v < $n} {incr v} {
        lappend vecs $vecsArray($v)
    }
    
    return $vecs
}

# ===========================================================================
# generation of test vectors
# ===========================================================================

proc genTestVector {elems start} {
    set vec [list]
    for {set i 0} {$i < $elems} {incr i} {
        lappend vec [expr $start + $i]
    }
    return $vec
}

proc genTestVectors {N elems} {
    set vecs [list]
    for {set start 0} {$start < [expr 2 * $N * $elems]} {incr start $elems} {
        lappend vecs [genTestVector $elems $start]
    }
    return $vecs
}

# ===========================================================================
# output functions
# ===========================================================================

proc putsVec {vec} {
    puts -nonewline "\{ "
    foreach e $vec {
        puts -nonewline "[format %3d $e] "
    }
    puts -nonewline "\} "
}

proc puts2 {vecs} {
    set i 0
    foreach vec $vecs {
        putsVec $vec
        incr i
        if {$i % 2 == 0} {
            puts ""
        }
    }
}

proc puts1 {vecs} {
    foreach vec $vecs {
        putsVec $vec
        puts ""
    }
}

# ===========================================================================
# check functions
# ===========================================================================

proc checkTransposed {mat matT} {
    set n [llength $mat]
    for {set i 0} {$i < $n} {incr i} {
        for {set j 0} {$j < $n} {incr j} {
            set e [lindex [lindex $mat $i] $j]
            set eT [lindex [lindex $matT $j] $i]
            if {$e != $eT} {
                puts "!!! TRANSPOSE ERROR AT ($i,$j): $e != $eT !!!"
                return
            }
        }
    }
    puts "VVVVVVVVVVVVVVV transpose ok VVVVVVVVVVVVVV"
}

# ===========================================================================
# "main"
# ===========================================================================

if {0} {
    # ========== test splitFactor ==========
    # set v { \
                0 1 2 3 4 5 6 7 \
            }
    set v { \
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
                16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 \
            }
    
    puts $v
    puts [splitFactor 1 $v]
    puts [splitFactor 2 $v]
    puts [splitFactor 4 $v]
    exit
}

if {0} {
    # ========== test genTestVectors ==========
    puts [genTestVectors 5 4]
    puts [genTestVectors 1 4]
    exit
}

proc tsimdPuts {f str spaces varL varH} {
    puts $f "${spaces}printf(\"${str}\");" 
    puts $f "${spaces}print(\"%d \", $varL);"
    puts $f "${spaces}printf(\"| \");"
    puts $f "${spaces}print(\"%d \", $varH);"
    puts $f "${spaces}puts(\"\");"
}

proc puts2v {str vecs} {
    puts -nonewline "$str"
    lassign $vecs vl vh
    foreach e $vl {
        puts -nonewline "$e "
    }
    puts -nonewline "| "
    foreach e $vh {
        puts -nonewline "$e "
    }    
    puts ""
}

# SIMD_WIDTH SIZE lanes elements elementsPerLane
# 16         1    1     16       16
# 16         2    1     8        8
# 16         4    1     4        4
# 32         1    2     32       16
# 32         2    2     16       8
# 32         4    2     8        4
# 64         1    4     64       16
# 64         2    4     32       8
# 64         4    4     16       4

if {0} {
    # ========== full test of zip functions ==========
    set f [open "ziptest.C" "w"]
    # T-SIMD preamble
    puts $f "// don't edit, this file was auto-generated by swizzle_test.tcl!"
    puts $f "\#include \"tsimd.H\""
    puts $f "using namespace simd;"
    puts $f "int main()"
    puts $f "\{"
    foreach SIMD_WIDTH {16 32 64} {
        set lanes [expr $SIMD_WIDTH / 16]
        foreach DATA_TYPE {Byte Word Int} SIZE {1 2 4} {
            set elements [expr $SIMD_WIDTH / $SIZE]
            set elementsPerLane [expr $elements / $lanes]
            puts $f "  \{"
            # vector type info
            set vecType "Vec<$DATA_TYPE,$SIMD_WIDTH>" 
            # generate input vectors tcl
            set input [genTestVectors 1 $elements]
            lassign $input inL inH
            # generate input vectors T-SIMD
            puts $f "    Vec<$DATA_TYPE,$SIMD_WIDTH> inL, inH, outL, outH;"
            puts $f "    constexpr size_t elems"
            puts $f "      = Vec<$DATA_TYPE,$SIMD_WIDTH>::elements;"
            puts $f "    $DATA_TYPE inLbuf\[elems\], inHbuf\[elems\];"
            puts $f "    for (size_t i = 0; i < elems; i++) \{"
            puts $f "      inLbuf\[i\] = i;"
            puts $f "      inHbuf\[i\] = i + elems;"
            puts $f "    \}"
            puts $f "    inL = loadu<$SIMD_WIDTH>(inLbuf);"
            puts $f "    inH = loadu<$SIMD_WIDTH>(inHbuf);"
            # print input tcl
            puts2v "input $vecType " $input
            # print input T-SIMD
            tsimdPuts $f "input $vecType " "    " inL inH
            # fullZip tcl
            puts2v "zip<1> $vecType " [fullZip $inL $inH]
            # fullZip T-SIMD
            puts $f "    zip<1>(inL, inH, outL, outH);"
            tsimdPuts $f "zip<1> $vecType " "    " outL outH
            # zipLanes tcl
            puts2v "zip16<1> $vecType " [zipLanes $lanes $inL $inH]
            # zipLanes T-SIMD
            puts $f "    zip16<1>(inL, inH, outL, outH);"
            tsimdPuts $f "zip16<1> $vecType " "    " outL outH
            for {set blkSz 1} {$blkSz < $elementsPerLane} {incr blkSz $blkSz} {
                # fullZipBlk tcl
                puts2v "zip<$blkSz> $vecType " [fullZipBlk $blkSz $inL $inH]
                # fullZipBlk C++
                puts $f "    zip<$blkSz>(inL, inH, outL, outH);"
                tsimdPuts $f "zip<$blkSz> $vecType " "    " outL outH
                # zipBlkLanes tcl
                puts2v "zip16<$blkSz> $vecType " \
                    [zipBlkLanes $blkSz $lanes $inL $inH]
                # zipBlkLanes T-SIMD
                puts $f "    zip16<$blkSz>(inL, inH, outL, outH);"
                tsimdPuts $f "zip16<$blkSz> $vecType " "    " outL outH
            }
            puts $f "  \}"
        }
    }
    puts $f "\}"
    close $f
}

if {0} {
    # ========== test zipLanes ==========
    set input4 [genTestVectors 1 4]
    set input8 [genTestVectors 1 8]
    set input16 [genTestVectors 1 16]
    set inputs [list $input4 $input8 $input16]

    #puts [zipBlkLanes 4 2 {*}$input16]
    #exit
    
    # puts $inputs
    foreach input $inputs {
        puts "input:\n$input"
        set len [llength [lindex $input 0]]
        puts "len: $len"
        puts ""
        # <
        for {set lanes 1} {$lanes <= $len} {set lanes [expr 2 * $lanes]} {
            puts "*** lanes [format "%2d" $lanes] ***"
            puts [zipLanes $lanes {*}$input]
            set blks [expr $len / $lanes]
            for {set blkSz 1} {$blkSz < $blks} {set blkSz [expr 2 * $blkSz]} {
                puts "blkSz [format "%2d" $blkSz]"
                puts [zipBlkLanes $blkSz $lanes {*}$input]
            }
            puts ""
        }
        puts "-----------\n"
    }
    exit
}

if {0} {
    # ========== test swizzle2* ==========
    set input34 [genTestVectors 3 4]
    
    puts2 [swizzle2 fullZip 1 3 4 $input34]
    puts2 [swizzle2idx fullZip 1 3 4 $input34]
    puts2 [swizzle2inplace fullZip 1 3 4 $input34]

    set input44 [genTestVectors 4 4]
    puts2 [swizzle2inplace fullZip 1 4 4 $input44]
    
    set input54 [genTestVectors 5 4]
    puts2 [swizzle2inplace fullZip 1 5 4 $input54]
    exit
}

set mat4 [genTestVectors 2 4]
set mat8 [genTestVectors 4 8]
set mat16 [genTestVectors 8 16]
set mat32 [genTestVectors 16 32]
set mat64 [genTestVectors 32 64]
set mat128 [genTestVectors 64 128]

set allMats [list $mat4 $mat8 $mat16 $mat32 $mat64 $mat128]
set mats [list $mat4 $mat8 $mat16]

if {0} {
    # ========== test transpose2idx ==========
    set mat $mat16
    puts1 $mat
    puts "-------------"
    set matT [transpose2idx fullZip 1 $mat]
    puts $matT
}

# observation: different structure of intermediate result with
# lane-zip, not clear yet
if {0} {
    # ========== test transpose2inplace ==========
    foreach mat $mats {
        puts1 $mat
        set matT [transpose2inplace fullZip 1 $mat]
        puts1 $matT
        checkTransposed $mat $matT
        puts "--------------------"
    }
    foreach mat $mats {
        puts1 $mat
        set matT [transpose2inplace zip_2Lanes 2 $mat]
        puts1 $matT
        # checkTransposed $mat $matT
        puts "--------------------"
    }
    foreach mat $mats {
        puts1 $mat
        set matT [transpose2inplace zip_4Lanes 4 $mat]
        puts1 $matT
        # checkTransposed $mat $matT
        puts "--------------------"
    }
    
}

if {0} {
    # ========== test transpose1idx ==========
    set mat $mat16
    puts1 $mat
    puts "-------------"
    set matT [transpose1idx fullZipBlk 1 $mat]
    puts1 $matT
}

if {0} {
    # ========== test transpose1 ==========
    foreach mat $mats {
        puts1 $mat
        set matT [transpose1 zipBlk_1Lanes 1 $mat]
        puts1 $matT
        checkTransposed $mat $matT
        puts "--------------------"
    }
    foreach mat $mats {
        puts1 $mat
        set matT [transpose1 zipBlk_2Lanes 2 $mat]
        puts1 $matT
        # checkTransposed $mat $matT
        puts "--------------------"
    }
    foreach mat $mats {
        puts1 $mat
        set matT [transpose1 zipBlk_4Lanes 4 $mat]
        puts1 $matT
        # checkTransposed $mat $matT
        puts "--------------------"
    }

}

if {0} {
    # ========== test transpose2inplace mit lane-zip  ==========
    set mat $mat16
    puts1 $mat
    set matT [transpose2inplace zip_4Lanes 4 $mat]
    puts1 $matT
}

if {0} {
    # ========== test transpose2inplace mit lane-zip  ==========
    set mat $mat16
    puts1 $mat
    set matT [transpose2inplace zip_1Lanes 1 $mat]
    puts1 $matT
    set matT [transpose2inplace zip_2Lanes 2 $mat]
    puts1 $matT
    set matT [transpose2inplace zip_4Lanes 4 $mat]
    puts1 $matT
}

if {0} {
    # ========== test transpose1 mit lane-zip  ==========
    set mat $mat16
    puts1 $mat
    puts1 [transpose1 zipBlk_1Lanes 1 $mat]
    puts "--------"
    puts1 [transpose1 zipBlk_2Lanes 2 $mat]
    puts "--------"
    puts1 [transpose1 zipBlk_4Lanes 4 $mat]
}

if {0} {
    # ========== test transpose1* ==========
    set mat $mat8
    
    puts1 [transpose1 fullZipBlk 1 $mat]
    puts "--------"
    puts1 [transpose1idx fullZipBlk 1 $mat]
    puts "--------"
    puts1 [transpose1inplace fullZipBlk 1 $mat]
}

if {0} {
    # ========== test transpose1inplace ==========
    # in-place swapping seems to work up to 128x128 elements
    foreach mat $allMats {
        puts1 $mat
        set matT [transpose1inplace fullZipBlk 1 $mat]
        puts1 $matT
        checkTransposed $mat $matT
        puts "--------------------"
    }
}

if {0} {
    # ========== test transpose1inplace mit lane-zip  ==========
    set mat $mat16
    puts1 $mat
    puts "--------------"
    foreach lanes {2 4 8} {
        set matT1 [transpose1inplace zipBlk_${lanes}Lanes ${lanes} $mat]
        puts1 $matT1
        puts "--------------"
        set matT2 [laneTranspose1inplace $lanes $matT1]
        puts1 $matT2
        checkTransposed $mat $matT2
        puts "=============="
    }
}

# * not clear: is it better to stop early and continue to use zip<1>
#   or to proceed and use zip<larger blocks>?
if {0} {
    # ========== test transpose2inplace mit lane-zip  ==========
    set mat $mat32
    puts1 $mat
    puts "--------------"
    foreach lanes {2 4 8} {
        # we don't stop early in transpose2inplace, required distances
        # between elements is only achieved after 
        set matT1 [transpose2inplace zip_${lanes}Lanes 1 $mat]
        puts1 $matT1
        puts "--------------"
        set matT2 [laneTranspose2blkInplace $lanes $matT1]
        puts1 $matT2
        checkTransposed $mat $matT2
        puts "=============="
    }
}

if {1} {
    # ========== test transpose2inplace mit lane-zip  ==========
    set mat $mat32
    puts1 $mat
    puts "--------------"
    foreach lanes {2 4 8} {
        # we stop early in transpose2inplace
        set matT1 [transpose2inplace zip_${lanes}Lanes $lanes $mat]
        puts1 $matT1
        puts "--------------"
        set matT2 [laneTranspose2inplace $lanes $matT1]
        puts1 $matT2
        checkTransposed $mat $matT2
        puts "=============="
    }
}

# transpose1inplace:
# (+) first part can be terminated early
# (-) requires swap operations


# transpose2inplace
# (-) first part can't be terminated early (but see above)
# (+) doesn't require swap operations
