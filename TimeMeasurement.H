// ===========================================================================
// 
// TimeMeasurement.H --
// functions for time measurement
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

// https://
// en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns#Factory

#ifndef _TIME_MEASUREMENT_H_
#define _TIME_MEASUREMENT_H_

#include <time.h>
// 20. Apr 18 (rm): thanks to Benedikt Volkmer
#include <assert.h>

namespace ns_simd {

  /* long long not supported everywhere
  signed long long 
  getTimeUsec()
  {
    signed long long sec, nsec;
    struct timespec ts;
    assert(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts));
    sec = ts.tv_sec;
    nsec = ts.tv_nsec;
    return 1000000LL * sec + nsec / 1000LL;
  }
  */

   struct timespec getTimeSpec()
   {
     struct timespec ts;
     assert(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts));
     return ts;
   }
  
  struct timespec getTimeSpecMonotonic()
  {
    struct timespec ts;
    assert(!clock_gettime(CLOCK_MONOTONIC, &ts));
    return ts;
  }
  
  // https://www.gnu.org/
  //   software/libc/manual/html_node/Calculating-Elapsed-Time.html
  // adapted, computes x - y for timespec instead of timeval
  int
  timespec_subtract(const struct timespec &xx, 
		    const struct timespec &yy,
		    struct timespec &result)
  {
    struct timespec x = xx, y = yy;
    /* Perform the carry for the later subtraction by updating y. */
    if (x.tv_nsec < y.tv_nsec) {
      int nsec = (y.tv_nsec - x.tv_nsec) / 1000000000L + 1;
      y.tv_nsec -= 1000000000L * nsec;
      y.tv_sec += nsec;
    }
    if (x.tv_nsec - y.tv_nsec > 1000000000L) {
      int nsec = (x.tv_nsec - y.tv_nsec) / 1000000000L;
      y.tv_nsec += 1000000000L * nsec;
      y.tv_sec -= nsec;
    }
    /* Compute the time remaining to wait. tv_nsec is certainly positive. */
    result.tv_sec = x.tv_sec - y.tv_sec;
    result.tv_nsec = x.tv_nsec - y.tv_nsec;
    /* Return 1 if result is negative. */
    return x.tv_sec < y.tv_sec;
  }
  
  // convert to us (intended for results of timespec_subtract!)
  double
  timespec_usec(const struct timespec &x)
  {
    return double(1E6 * x.tv_sec + x.tv_nsec / 1E3);
  }
  
  double
  timeSpecDiffUsec(const struct timespec &x,
		   const struct timespec &y)
  {
    struct timespec diff;
    timespec_subtract(x, y, diff);
    return timespec_usec(diff);
  }
  
} // namespace


#endif
 
