// ===========================================================================
// 
// Compare2Type.H --
// structs for comparison which can be used with the int2type trick
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

#ifndef _COMPARE2TYPE_
#define _COMPARE2TYPE_

namespace ns_simd {

  // int2type trick from 
  // Andrei Alexandrescu: Modern C++ Design (Addison Wesley),
  // adapted for type comparisons

  // empty struct
  // each unique combination of less, equal, greater declares a unique type
  template <bool less, 
	    bool equal, 
	    bool greater>
  struct Compare
  {
  };

  // structs derived from Compare for the special case of sizeof comparisons

  // use with CompareLess, CompareEqual, CompareGreater
  template <typename Tout, typename Tin>
  struct CompareTypes : 
    Compare<(sizeof(Tout) < sizeof(Tin)),
    (sizeof(Tout) == sizeof(Tin)),
    (sizeof(Tout) > sizeof(Tin))>
  {
  };

  // use with CompareLessOrEqual, CompareGreater
  template <typename Tout, typename Tin>
  struct CompareTypesLessOrEqual : 
    Compare<(sizeof(Tout) <= sizeof(Tin)),
    (sizeof(Tout) <= sizeof(Tin)),
    (sizeof(Tout) > sizeof(Tin))>
  {
  };

  // use with CompareLess, CompareGreaterOrEqual
  template <typename Tout, typename Tin>
  struct CompareTypesGreaterOrEqual :
    Compare<(sizeof(Tout) < sizeof(Tin)),
    (sizeof(Tout) >= sizeof(Tin)),
	    (sizeof(Tout) >= sizeof(Tin))>
  {
  };

  typedef Compare<true,false,false> CompareLess;
  typedef Compare<true,true,false>  CompareLessOrEqual;
  typedef Compare<false,true,false> CompareEqual;
  typedef Compare<false,true,true>  CompareGreaterOrEqual;
  typedef Compare<false,false,true> CompareGreater;

}

#endif
