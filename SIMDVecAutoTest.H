// ===========================================================================
// 
// SIMDVecAutoTest.H --
// code for auto-test (comparison of SIMDVec and SIMDSerialVec behavior)
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

// https://stackoverflow.com/questions/15651488/
// how-to-pass-a-template-function-in-a-template-argument-list 
//
// ... you cannot pass a function template as a template
// argument. From 14.3.3: A template-argument for a template
// template-parameter shall be the name of a class template or an
// alias template, expressed as id-expression.

// 01. Aug 22 (Jonas Keller):
// removed SIMD_INLINE from all test functions, speeds up compilation by a
// factor of at least 6 with clang and there is no reason to inline the test
// functions

#ifndef _SIMD_VEC_AUTO_TEST_H_
#define _SIMD_VEC_AUTO_TEST_H_

#include "SIMDSerialVec.H"
#include "SIMDVecWrapper.H"

namespace ns_simd {

  // how many error cases should be printed to stdout
#define PRINT_ERRORS 1

  // ===========================================================================
  // aux. functions and macros
  // ===========================================================================

  void printInfo(const std::string &fct, const std::string &args)
  {
    printf("testing %s(%s)\n", fct.c_str(), args.c_str()); fflush(stdout);
  }

  void printError(const std::string &fct, const std::string &args)
  {
    printf("*** error in %s(%s)\n", fct.c_str(), args.c_str()); fflush(stdout);
  }

  void printErrorStats(int errors, int trials)
  {
    if (errors > 0) printf("---------- %d errors / %d trials ----------\n",  
			   errors, trials);
  }

  // #define EXIT break
#define EXIT
  // #define EXIT exit(-1)

#define PRINT(FORMAT,VEC)					\
  {								\
    printf(#VEC "\t= "); print(FORMAT, VEC.getSIMDVec());	\
    puts("");  fflush(stdout);					\
  }

#define PRINTVAL(FORMAT,VAL)			\
  {						\
    printf(#VAL "\t= "); printf(FORMAT, VAL);	\
    puts(""); fflush(stdout);			\
  }

  // ===========================================================================
  // tests for different type combinations: no additional template argument
  // ===========================================================================

  template <template <typename, int, template <typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <typename, int> class FCT, typename... Ts>
  struct TestList;

  template <template <typename, int, template <typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <typename, int> class FCT, typename T>
  struct TestList<Tester, SIMD_WIDTH, FCT, T> {
    static void test(int repeats, const std::string &pattern) {
      Tester<T, SIMD_WIDTH, FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, template <typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <typename, int> class FCT, typename T,
            typename... Ts>
  struct TestList<Tester, SIMD_WIDTH, FCT, T, Ts...> {
    static void test(int repeats, const std::string &pattern) {
      Tester<T, SIMD_WIDTH, FCT>::test(repeats, pattern);
      TestList<Tester, SIMD_WIDTH, FCT, Ts...>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <typename, int> class> class Tester,
	    int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct TestSignedInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <typename, int> class FCT>
  struct TestUnsignedInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <typename, int> class FCT>
  struct TestFloat
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };
  
  template <template <typename, int, 
		      template <typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <typename, int> class FCT>
  struct TestInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestSignedInt<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestUnsignedInt<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };
      
  template <template <typename, int, 
		      template <typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <typename, int> class FCT>
  struct TestSigned
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestSignedInt<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestFloat<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <typename, int> class FCT>
  struct TestAll
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestInt<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestFloat<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // tests for different type combinations: one additional int template argument
  // ===========================================================================

  template <template <typename, int, template <int, typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <int, typename, int> class FCT,
            typename... Ts>
  struct TestListTI;

  template <template <typename, int, template <int, typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <int, typename, int> class FCT, typename T>
  struct TestListTI<Tester, SIMD_WIDTH, FCT, T> {
    static void test(int repeats, const std::string &pattern) {
      Tester<T, SIMD_WIDTH, FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, template <int, typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <int, typename, int> class FCT, typename T,
            typename... Ts>
  struct TestListTI<Tester, SIMD_WIDTH, FCT, T, Ts...> {
    static void test(int repeats, const std::string &pattern) {
      Tester<T, SIMD_WIDTH, FCT>::test(repeats, pattern);
      TestListTI<Tester, SIMD_WIDTH, FCT, Ts...>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, typename, int> class> class Tester,
	    int SIMD_WIDTH,
	    template <int, typename, int> class FCT>
  struct TestSignedIntTI
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, typename, int> class FCT>
  struct TestUnsignedIntTI
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, typename, int> class FCT>
  struct TestFloatTI
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };
  
  template <template <typename, int, 
		      template <int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, typename, int> class FCT>
  struct TestIntTI
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestSignedIntTI<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestUnsignedIntTI<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };
      
  template <template <typename, int, 
		      template <int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, typename, int> class FCT>
  struct TestSignedTI
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestSignedIntTI<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestFloatTI<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, typename, int> class FCT>
  struct TestAllTI
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestIntTI<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestFloatTI<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // tests for different type combinations: two additional int template args.
  // ===========================================================================

  template <template <typename, int, template <int,int, typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <int, int, typename, int> class FCT,
            typename... Ts>
  struct TestListTII;

  template <template <typename, int, template <int,int, typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <int, int, typename, int> class FCT, typename T>
  struct TestListTII<Tester, SIMD_WIDTH, FCT, T> {
    static void test(int repeats, const std::string &pattern) {
      Tester<T, SIMD_WIDTH, FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, template <int,int, typename, int> class>
            class Tester,
            int SIMD_WIDTH, template <int, int, typename, int> class FCT, typename T,
            typename... Ts>
  struct TestListTII<Tester, SIMD_WIDTH, FCT, T, Ts...> {
    static void test(int repeats, const std::string &pattern) {
      Tester<T, SIMD_WIDTH, FCT>::test(repeats, pattern);
      TestListTII<Tester, SIMD_WIDTH, FCT, Ts...>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, int, typename, int> class> class Tester,
	    int SIMD_WIDTH,
	    template <int, int, typename, int> class FCT>
  struct TestSignedIntTII
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, int, typename, int> class FCT>
  struct TestUnsignedIntTII
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, int, typename, int> class FCT>
  struct TestFloatTII
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };
  
  template <template <typename, int, 
		      template <int, int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, int, typename, int> class FCT>
  struct TestIntTII
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestSignedIntTII<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestUnsignedIntTII<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };
      
  template <template <typename, int, 
		      template <int, int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, int, typename, int> class FCT>
  struct TestSignedTII
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestSignedIntTII<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestFloatTII<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  template <template <typename, int, 
		      template <int, int, typename, int> class> class Tester,
	    int SIMD_WIDTH, 
	    template <int, int, typename, int> class FCT>
  struct TestAllTII
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      TestIntTII<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
      TestFloatTII<Tester,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test for different type combinations: one additional template type arg.
  // ===========================================================================

  template <template <typename, typename, int,
		      template <typename, typename, int> class> class Tester,
	    int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct TestAllTT
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      Tester<SIMDByte,SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDSignedByte,SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDFloat,SIMDByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDByte,SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDSignedByte,SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDFloat,SIMDSignedByte,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDByte,SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDSignedByte,SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDFloat,SIMDWord,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDByte,SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDSignedByte,SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDFloat,SIMDShort,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDByte,SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDSignedByte,SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDFloat,SIMDInt,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDByte,SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDSignedByte,SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDWord,SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDShort,SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDInt,SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
      Tester<SIMDFloat,SIMDFloat,SIMD_WIDTH,FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of nullary commands
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct Nullary
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "";
      SIMDSerialVec<T,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	// repeats don't make sense here
	for (int i = 0; i < repeats; i++, trials++) {
	  cs = FCT<T,SIMD_WIDTH>::template apply<SIMDSerialVec>();
	  cp = FCT<T,SIMD_WIDTH>::template apply<SIMDVec>();
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of unary commands
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct Unary
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v";
      SIMDSerialVec<T,SIMD_WIDTH> a, cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<T,SIMD_WIDTH>::randomizeInput(a);
	  cs = FCT<T,SIMD_WIDTH>::apply(a);
	  cp = FCT<T,SIMD_WIDTH>::apply(a.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of commands with scalar input and vector result
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct Scalar
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "s";
      T a;
      SIMDSerialVec<T,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  a = FCT<T,SIMD_WIDTH>::randomizeInputScalar();
	  cs = FCT<T,SIMD_WIDTH>::template apply<SIMDSerialVec>(a);
	  cp = FCT<T,SIMD_WIDTH>::template apply<SIMDVec>(a);
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINTVAL(fmt,a);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of unary commands with scalar result
  // ===========================================================================

  // vector if type T, scalar result of type TRES
  template <typename TRES, typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct UnaryScalar
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v";
      SIMDSerialVec<T,SIMD_WIDTH> a;
      TRES cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtRes = SIMDTypeInfo<TRES>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<T,SIMD_WIDTH>::randomizeInput(a);
	  cs = FCT<T,SIMD_WIDTH>::apply(a);
	  cp = FCT<T,SIMD_WIDTH>::apply(a.getSIMDVec());
	  if (!sameScalar(cs, cp) /*cs != cp*/) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINTVAL(fmtRes,cs);
	      PRINTVAL(fmtRes,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct UnaryScalarT : public UnaryScalar<T,T,SIMD_WIDTH,FCT> 
  {
  };

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct UnaryScalarInt : public UnaryScalar<int,T,SIMD_WIDTH,FCT> 
  {
  };

  // ===========================================================================
  // test of unary commands with single type template parameter
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct UnaryTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v";
      SIMDSerialVec<T,SIMD_WIDTH> a;
      SIMDSerialVec<Tout,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<Tout,T,SIMD_WIDTH>::randomizeInput(a);
	  cs = FCT<Tout,T,SIMD_WIDTH>::apply(a);
	  cp = FCT<Tout,T,SIMD_WIDTH>::apply(a.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmtOut,cs);
	      PRINT(fmtOut,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of unary commands with single int template parameter and scalar res.
  // ===========================================================================

  template <typename T, int SIMD_WIDTH, int I, int N,
	    template <int, typename, int> class FCT>
  struct UnaryTemplateIntScalar
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v";
      SIMDSerialVec<T,SIMD_WIDTH> a;
      T cp, cs;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<I,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<I,T,SIMD_WIDTH>::randomizeInput(a);
	  cs = FCT<I,T,SIMD_WIDTH>::apply(a);
	  cp = FCT<I,T,SIMD_WIDTH>::apply(a.getSIMDVec());
	  if (!sameScalar(cs, cp) /*cs != cp*/) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINTVAL(fmt,cs);
	      PRINTVAL(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
      UnaryTemplateIntScalar<T, SIMD_WIDTH, 
			     FCT<I,T,SIMD_WIDTH>::nextI, 
			     N, FCT>::test(repeats, pattern);
    }
  };

  // end of recursion at I = maxI
  template <typename T, int SIMD_WIDTH, int N,
	    template <int, typename, int> class FCT>
  struct UnaryTemplateIntScalar<T,SIMD_WIDTH,N,N,FCT>
  {
    static void
    test(int, const std::string&)
    {
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <int, typename, int> class FCT>
  struct UnaryTemplateIntScalarMinToMax
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      UnaryTemplateIntScalar<T, SIMD_WIDTH, 
			     FCT<0,T,SIMD_WIDTH>::minI, 
			     FCT<0,T,SIMD_WIDTH>::maxI, 
			     FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of unary commands with single int template parameter and vector res.
  // ===========================================================================

  template <typename T, int SIMD_WIDTH, int I, int N,
	    template <int, typename, int> class FCT>
  struct UnaryTemplateInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v";
      SIMDSerialVec<T,SIMD_WIDTH> a, cp, cs;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<I,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<I,T,SIMD_WIDTH>::randomizeInput(a);
	  cs = FCT<I,T,SIMD_WIDTH>::apply(a);
	  cp = FCT<I,T,SIMD_WIDTH>::apply(a.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
      UnaryTemplateInt<T, SIMD_WIDTH, 
		       FCT<I,T,SIMD_WIDTH>::nextI, 
		       N, FCT>::test(repeats, pattern);
    }
  };

  // end of recursion at I = maxI
  template <typename T, int SIMD_WIDTH, int N,
	    template <int, typename, int> class FCT>
  struct UnaryTemplateInt<T,SIMD_WIDTH,N,N,FCT>
  {
    static void
    test(int, const std::string&)
    {
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <int, typename, int> class FCT>
  struct UnaryTemplateIntMinToMax
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      UnaryTemplateInt<T, SIMD_WIDTH,
		       FCT<0,T,SIMD_WIDTH>::minI, 
		       FCT<0,T,SIMD_WIDTH>::maxI, 
		       FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of binary commands
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct Binary
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,v";
      SIMDSerialVec<T,SIMD_WIDTH> a, b, cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<T,SIMD_WIDTH>::randomizeInput(b);
	  cs = FCT<T,SIMD_WIDTH>::apply(a, b);
	  cp = FCT<T,SIMD_WIDTH>::apply(a.getSIMDVec(), b.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,b);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of binary commands with single type template parameter
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct BinaryTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,v";
      SIMDSerialVec<T,SIMD_WIDTH> a, b;
      SIMDSerialVec<Tout,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<Tout,T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<Tout,T,SIMD_WIDTH>::randomizeInput(b);
	  cs = FCT<Tout,T,SIMD_WIDTH>::apply(a, b);
	  cp = FCT<Tout,T,SIMD_WIDTH>::apply(a.getSIMDVec(), b.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,b);
	      PRINT(fmtOut,cs);
	      PRINT(fmtOut,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of commands with unary in and array out w. single type template param.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct UnaryArrayTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,*v";
      const int nOutVecs = sizeof(Tout) / sizeof(T);
      SIMDSerialVec<T,SIMD_WIDTH> a;
      SIMDSerialVec<Tout,SIMD_WIDTH> cs[nOutVecs], cp[nOutVecs];
      SIMDVec<Tout,SIMD_WIDTH> _cp[nOutVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<Tout,T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<Tout,T,SIMD_WIDTH>::apply(a, cs);
	  FCT<Tout,T,SIMD_WIDTH>::apply(a.getSIMDVec(), _cp);
	  bool foundError = false;
	  for (int io = 0; io < nOutVecs; io++) {
	    cp[io] = _cp[io];
	    if (cs[io] != cp[io])
	      foundError = true;
	  }
	  if (foundError) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      for (int io = 0; io < nOutVecs; io++) {
		PRINT(fmtOut,cs[io]);
		PRINT(fmtOut,cp[io]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of commands with array in and vector out
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct Array
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v";
      const int nInVecs = FCT<T,SIMD_WIDTH>::nInputs;
      SIMDSerialVec<T,SIMD_WIDTH> a[nInVecs];
      SIMDVec<T,SIMD_WIDTH> _a[nInVecs];
      SIMDSerialVec<T,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  for (int j = 0; j < nInVecs; j++) {
	    FCT<T,SIMD_WIDTH>::randomizeInput(a[j]);
	    _a[j] = a[j].getSIMDVec();
	  }
	  cs = FCT<T,SIMD_WIDTH>::apply(a);
	  cp = FCT<T,SIMD_WIDTH>::apply(_a);
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      for (int j = 0; j < nInVecs; j++)
		PRINT(fmt,a[j]);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of commands with array in and vector out w. single templ. type param.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct ArrayTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v";
      const int nInVecs = FCT<Tout,T,SIMD_WIDTH>::nInputs;
      SIMDSerialVec<T,SIMD_WIDTH> a[nInVecs];
      SIMDVec<T,SIMD_WIDTH> _a[nInVecs];
      SIMDSerialVec<Tout,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  for (int j = 0; j < nInVecs; j++) {
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput(a[j]);
	    _a[j] = a[j].getSIMDVec();
	  }
	  cs = FCT<Tout,T,SIMD_WIDTH>::apply(a);
	  cp = FCT<Tout,T,SIMD_WIDTH>::apply(_a);
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      for (int j = 0; j < nInVecs; j++)
		PRINT(fmt,a[j]);
	      PRINT(fmtOut,cs);
	      PRINT(fmtOut,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of void commands with array in and array out
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    template <typename, int> class FCT>
  struct ArrayArrayVoid
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v,*v";
      const int n = SIMD_WIDTH / sizeof(T);
      SIMDSerialVec<T,SIMD_WIDTH> as[n], cs[n], cp[n];
      SIMDVec<T,SIMD_WIDTH> _ap[n], _cp[n];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  for (int j = 0; j < n; j++) {
	    FCT<T,SIMD_WIDTH>::randomizeInput(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	  }
	  // apply function
	  FCT<T,SIMD_WIDTH>::apply(as, cs);
	  FCT<T,SIMD_WIDTH>::apply(_ap, _cp);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < n; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    cp[j] = _cp[j];
	    if (cs[j] != cp[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input copy
	      for (int j = 0; j < n; j++)
		PRINT(fmt,as[j]);
	      // output
	      for (int j = 0; j < n; j++) {
		PRINT(fmt,cs[j]);
		PRINT(fmt,cp[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of void commands with array in and array out and single templ. type p.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct ArrayArrayVoidTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v,*v";
      const int numInVecs = FCT<Tout,T,SIMD_WIDTH>::numInVecs;
      const int numOutVecs = FCT<Tout,T,SIMD_WIDTH>::numOutVecs;
      SIMDSerialVec<T,SIMD_WIDTH> as[numInVecs];
      SIMDSerialVec<Tout, SIMD_WIDTH> cs[numOutVecs], cp[numOutVecs];
      SIMDVec<T,SIMD_WIDTH> _ap[numInVecs];
      SIMDVec<Tout,SIMD_WIDTH> _cp[numOutVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  for (int j = 0; j < numInVecs; j++) {
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	  }
	  // apply function
	  FCT<Tout,T,SIMD_WIDTH>::apply(as, cs);
	  FCT<Tout,T,SIMD_WIDTH>::apply(_ap, _cp);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < numOutVecs; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    cp[j] = _cp[j];
	    if (cs[j] != cp[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input
	      for (int j = 0; j < numInVecs; j++)
		PRINT(fmt,as[j]);
	      // output
	      for (int j = 0; j < numOutVecs; j++) {
		PRINT(fmtOut,cs[j]);
		PRINT(fmtOut,cp[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of void cmds. w. 1 array and float in, array out, single templ. type p.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct ArrayDoubleArrayVoidTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v,s,*v";
      const int numInVecs = FCT<Tout,T,SIMD_WIDTH>::numInVecs;
      const int numOutVecs = FCT<Tout,T,SIMD_WIDTH>::numOutVecs;
      SIMDSerialVec<T,SIMD_WIDTH> as[numInVecs];
      SIMDSerialVec<Tout, SIMD_WIDTH> cs[numOutVecs], cp[numOutVecs];
      SIMDVec<T,SIMD_WIDTH> _ap[numInVecs];
      SIMDVec<Tout,SIMD_WIDTH> _cp[numOutVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  double d = FCT<Tout,T,SIMD_WIDTH>::randomizeInputFloat();
	  for (int j = 0; j < numInVecs; j++) {
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	  }
	  // apply function
	  FCT<Tout,T,SIMD_WIDTH>::apply(as, d, cs);
	  FCT<Tout,T,SIMD_WIDTH>::apply(_ap, d, _cp);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < numOutVecs; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    cp[j] = _cp[j];
	    if (cs[j] != cp[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input
	      printf("d = %g\n", d);
	      for (int j = 0; j < numInVecs; j++)
		PRINT(fmt,as[j]);
	      // output
	      for (int j = 0; j < numOutVecs; j++) {
		PRINT(fmtOut,cs[j]);
		PRINT(fmtOut,cp[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test void cmds. w. 1 array and 2 float in, array out, single templ. type p.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct ArrayDoubleDoubleArrayVoidTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v,s,s,*v";
      const int numInVecs = FCT<Tout,T,SIMD_WIDTH>::numInVecs;
      const int numOutVecs = FCT<Tout,T,SIMD_WIDTH>::numOutVecs;
      SIMDSerialVec<T,SIMD_WIDTH> as[numInVecs];
      SIMDSerialVec<Tout, SIMD_WIDTH> cs[numOutVecs], cp[numOutVecs];
      SIMDVec<T,SIMD_WIDTH> _ap[numInVecs];
      SIMDVec<Tout,SIMD_WIDTH> _cp[numOutVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  double d1 = FCT<Tout,T,SIMD_WIDTH>::randomizeInputFloat();
	  double d2 = FCT<Tout,T,SIMD_WIDTH>::randomizeInputFloat();
	  for (int j = 0; j < numInVecs; j++) {
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	  }
	  // apply function
	  FCT<Tout,T,SIMD_WIDTH>::apply(as, d1, d2, cs);
	  FCT<Tout,T,SIMD_WIDTH>::apply(_ap, d1, d2, _cp);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < numOutVecs; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    cp[j] = _cp[j];
	    if (cs[j] != cp[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input
	      printf("d1 = %g, d2 = %g\n", d1, d2);
	      for (int j = 0; j < numInVecs; j++)
		PRINT(fmt,as[j]);
	      // output
	      for (int j = 0; j < numOutVecs; j++) {
		PRINT(fmtOut,cs[j]);
		PRINT(fmtOut,cp[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of void cmds. w. 2 arrays and float in, array out, single templ. type p.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct ArrayArrayDoubleArrayVoidTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v,*v,s,*v";
      const int numInVecs = FCT<Tout,T,SIMD_WIDTH>::numInVecs;
      const int numOutVecs = FCT<Tout,T,SIMD_WIDTH>::numOutVecs;
      SIMDSerialVec<T,SIMD_WIDTH> as[numInVecs], bs[numInVecs];
      SIMDSerialVec<Tout, SIMD_WIDTH> cs[numOutVecs], cp[numOutVecs];
      SIMDVec<T,SIMD_WIDTH> _ap[numInVecs], _bp[numInVecs];
      SIMDVec<Tout,SIMD_WIDTH> _cp[numOutVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  double d = FCT<Tout,T,SIMD_WIDTH>::randomizeInputFloat();
	  for (int j = 0; j < numInVecs; j++) {
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput1(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput2(bs[j]);
	    _bp[j] = bs[j].getSIMDVec();
	  }
	  // apply function
	  FCT<Tout,T,SIMD_WIDTH>::apply(as, bs, d, cs);
	  FCT<Tout,T,SIMD_WIDTH>::apply(_ap, _bp, d, _cp);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < numOutVecs; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    cp[j] = _cp[j];
	    if (cs[j] != cp[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input
	      printf("d = %g\n", d);
	      for (int j = 0; j < numInVecs; j++) {
		PRINT(fmt,as[j]);
		PRINT(fmt,bs[j]);
	      }
	      // output
	      for (int j = 0; j < numOutVecs; j++) {
		PRINT(fmtOut,cs[j]);
		PRINT(fmtOut,cp[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test void cmds. w. 2 arrays and 2 float in, array out, single templ. type p.
  // ===========================================================================

  template <typename Tout, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct ArrayArrayDoubleDoubleArrayVoidTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v,*v,s,s,*v";
      const int numInVecs = FCT<Tout,T,SIMD_WIDTH>::numInVecs;
      const int numOutVecs = FCT<Tout,T,SIMD_WIDTH>::numOutVecs;
      SIMDSerialVec<T,SIMD_WIDTH> as[numInVecs], bs[numInVecs];
      SIMDSerialVec<Tout, SIMD_WIDTH> cs[numOutVecs], cp[numOutVecs];
      SIMDVec<T,SIMD_WIDTH> _ap[numInVecs], _bp[numInVecs];
      SIMDVec<Tout,SIMD_WIDTH> _cp[numOutVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtOut = SIMDTypeInfo<Tout>::defaultFormat();
      std::string name = FCT<Tout,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  double d1 = FCT<Tout,T,SIMD_WIDTH>::randomizeInputFloat();
	  double d2 = FCT<Tout,T,SIMD_WIDTH>::randomizeInputFloat();
	  for (int j = 0; j < numInVecs; j++) {
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput1(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	    FCT<Tout,T,SIMD_WIDTH>::randomizeInput2(bs[j]);
	    _bp[j] = bs[j].getSIMDVec();
	  }
	  // apply function
	  FCT<Tout,T,SIMD_WIDTH>::apply(as, bs, d1, d2, cs);
	  FCT<Tout,T,SIMD_WIDTH>::apply(_ap, _bp, d1, d2, _cp);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < numOutVecs; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    cp[j] = _cp[j];
	    if (cs[j] != cp[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input
	      printf("d1 = %g, d2 = %g\n", d1, d2);
	      for (int j = 0; j < numInVecs; j++) {
		PRINT(fmt,as[j]);
		PRINT(fmt,bs[j]);
	      }
	      // output
	      for (int j = 0; j < numOutVecs; j++) {
		PRINT(fmtOut,cs[j]);
		PRINT(fmtOut,cp[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

  // ===========================================================================
  // test of void commands with array in/out and single int template parameters
  // ===========================================================================

  template <typename T, int SIMD_WIDTH, int I, int N,
	    template <int, typename, int> class FCT>
  struct ArrayVoidTemplateInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "*v";
      const int numVecs = FCT<I,T,SIMD_WIDTH>::numVecs;
      SIMDSerialVec<T,SIMD_WIDTH> as[numVecs], ap[numVecs], a[numVecs];
      SIMDVec<T,SIMD_WIDTH> _ap[numVecs];
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<I,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  for (int j = 0; j < numVecs; j++) {
	    FCT<N,T,SIMD_WIDTH>::randomizeInput(as[j]);
	    _ap[j] = as[j].getSIMDVec();
	    // make a copy of the input
	    a[j] = as[j];
	  }
	  // apply function (overwrites results)
	  FCT<I,T,SIMD_WIDTH>::apply(as);
	  FCT<I,T,SIMD_WIDTH>::apply(_ap);
	  // compare results
	  bool equal = true;
	  for (int j = 0; j < numVecs; j++) {
	    // assign SIMDVec -> SIMDSerialVec
	    ap[j] = _ap[j];
	    if (as[j] != ap[j])	    
	      equal = false;
	  }
	  if (!equal) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      // input copy
	      for (int j = 0; j < numVecs; j++)
		PRINT(fmt,a[j]);
	      // output
	      for (int j = 0; j < numVecs; j++) {
		PRINT(fmt,as[j]);
		PRINT(fmt,ap[j]);
	      }
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
      ArrayVoidTemplateInt<T, SIMD_WIDTH,
			   FCT<I,T,SIMD_WIDTH>::nextI,
			   N, FCT>::test(repeats, pattern);
    }
  };

  template <typename T, int SIMD_WIDTH, int N,
	    template <int, typename, int> class FCT>
  struct ArrayVoidTemplateInt<T,SIMD_WIDTH,N,N,FCT>
  {
    static void
    test(int, const std::string&)
    {
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <int, typename, int> class FCT>
  struct ArrayVoidTemplateIntMinToMax
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      ArrayVoidTemplateInt<T, SIMD_WIDTH,
			   FCT<0,T,SIMD_WIDTH>::minI,
			   FCT<0,T,SIMD_WIDTH>::maxI,
			   FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of binary commands with single int template parameter and vector res.
  // ===========================================================================

  template <typename T, int SIMD_WIDTH, int I, int N,
	    template <int, typename, int> class FCT>
  struct BinaryTemplateInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,v";
      SIMDSerialVec<T,SIMD_WIDTH> a, b, cp, cs;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<I,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<I,T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<I,T,SIMD_WIDTH>::randomizeInput(b);
	  cs = FCT<I,T,SIMD_WIDTH>::apply(a, b);
	  cp = FCT<I,T,SIMD_WIDTH>::apply(a.getSIMDVec(), b.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,b);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
      BinaryTemplateInt<T, SIMD_WIDTH,
			FCT<I,T,SIMD_WIDTH>::nextI, 
			N, FCT>::test(repeats, pattern);
    }
  };

  // end of recursion at I = maxI
  template <typename T, int SIMD_WIDTH, int N,
	    template <int, typename, int> class FCT>
  struct BinaryTemplateInt<T,SIMD_WIDTH,N,N,FCT>
  {
    static void
    test(int, const std::string&)
    {
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <int, typename, int> class FCT>
  struct BinaryTemplateIntMinToMax
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      BinaryTemplateInt<T, SIMD_WIDTH, 
			FCT<0,T,SIMD_WIDTH>::minI, 
			FCT<0,T,SIMD_WIDTH>::maxI, 
			FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of binary commands with two vector outputs and int template arg.
  // ===========================================================================

  template <typename T, int SIMD_WIDTH,
	    int I, int IMAX,
	    template <int, typename, int> class FCT>
  struct BinaryBinaryTemplateInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,v,v&,v&";
      SIMDSerialVec<T,SIMD_WIDTH> a, b, cs, cp, ds, dp;
      SIMDVec<T,SIMD_WIDTH> _cp, _dp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<I,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<I,T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<I,T,SIMD_WIDTH>::randomizeInput(b);
	  FCT<I,T,SIMD_WIDTH>::apply(a, b, cs, ds);
	  FCT<I,T,SIMD_WIDTH>::apply(a.getSIMDVec(), b.getSIMDVec(), _cp, _dp);
	  cp = _cp;
	  dp = _dp;
	  if ((cs != cp) || (ds != dp)) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,b);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	      PRINT(fmt,ds);
	      PRINT(fmt,dp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
      BinaryBinaryTemplateInt<T, SIMD_WIDTH,
			      FCT<I,T,SIMD_WIDTH>::nextI, IMAX,
			      FCT>::test(repeats, pattern);
    }
  };

  // end of recursion at I = IMAX
  template <typename T, int SIMD_WIDTH, int IMAX,
	    template <int, typename, int> class FCT>
  struct BinaryBinaryTemplateInt<T,SIMD_WIDTH,IMAX,IMAX,FCT>
  {
    static void
    test(int, const std::string&)
    {
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <int, typename, int> class FCT>
  struct BinaryBinaryTemplateIntMinToMax
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      BinaryBinaryTemplateInt<T, SIMD_WIDTH, 
			      FCT<0,T,SIMD_WIDTH>::minI, 
			      FCT<0,T,SIMD_WIDTH>::maxI, 
			      FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of binary commands with two int template parameters and vector res.
  // ===========================================================================

  template <typename T, int SIMD_WIDTH, 
	    int I, int IMAX,
	    int J, int JMAX,
	    template <int, int, typename, int> class FCT>
  struct BinaryTemplateIntInt
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,v";
      SIMDSerialVec<T,SIMD_WIDTH> a, b, cp, cs;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      std::string name = FCT<I,J,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<I,J,T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<I,J,T,SIMD_WIDTH>::randomizeInput(b);
	  cs = FCT<I,J,T,SIMD_WIDTH>::apply(a, b);
	  cp = FCT<I,J,T,SIMD_WIDTH>::apply(a.getSIMDVec(), b.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmt,a);
	      PRINT(fmt,b);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
      BinaryTemplateIntInt<T, SIMD_WIDTH,
			   FCT<I,J,T,SIMD_WIDTH>::nextI, IMAX, 
			   FCT<I,J,T,SIMD_WIDTH>::nextJ, JMAX,
			   FCT>::test(repeats, pattern);
    }
  };

  // end of recursion at J = JMAX for arbitrary I
  // assumes that I changes fastest and J changes slowest
  template <typename T, int SIMD_WIDTH, int I, int IMAX, int JMAX,
	    template <int, int, typename, int> class FCT>
  struct BinaryTemplateIntInt<T,SIMD_WIDTH,I,IMAX,JMAX,JMAX,FCT>
  {
    static void
    test(int, const std::string&)
    {
    }
  };

  template <typename T, int SIMD_WIDTH,
	    template <int, int, typename, int> class FCT>
  struct BinaryTemplateIntIntMinToMax
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      BinaryTemplateIntInt<T, SIMD_WIDTH, 
			   FCT<0,0,T,SIMD_WIDTH>::minI, 
			   FCT<0,0,T,SIMD_WIDTH>::maxI, 
			   FCT<0,0,T,SIMD_WIDTH>::minJ,
			   FCT<0,0,T,SIMD_WIDTH>::maxJ,
			   FCT>::test(repeats, pattern);
    }
  };

  // ===========================================================================
  // test of ternary commands with condition arg. & single type template param.
  // ===========================================================================

  template <typename Tcond, typename T, int SIMD_WIDTH,
	    template <typename, typename, int> class FCT>
  struct TernaryConditionTemplateType
  {
    static void
    test(int repeats, const std::string &pattern)
    {
      std::string args = "v,v,v";
      SIMDSerialVec<T,SIMD_WIDTH> a, b;
      SIMDSerialVec<Tcond,SIMD_WIDTH> cond;
      SIMDSerialVec<T,SIMD_WIDTH> cs, cp;
      const char *fmt = SIMDTypeInfo<T>::defaultFormat();
      const char *fmtCond = SIMDTypeInfo<Tcond>::defaultFormat();
      std::string name = FCT<Tcond,T,SIMD_WIDTH>::name();
      if (name.find(pattern) != std::string::npos) {
	printInfo(name, args);
	int errors = 0, trials = 0;
	for (int i = 0; i < repeats; i++, trials++) {
	  FCT<Tcond,T,SIMD_WIDTH>::randomizeCond(cond);
	  FCT<Tcond,T,SIMD_WIDTH>::randomizeInput(a);
	  FCT<Tcond,T,SIMD_WIDTH>::randomizeInput(b);
	  cs = FCT<Tcond,T,SIMD_WIDTH>::apply(cond, a, b);
	  cp = FCT<Tcond,T,SIMD_WIDTH>::apply(cond.getSIMDVec(), 
					      a.getSIMDVec(), b.getSIMDVec());
	  if (cs != cp) {
	    errors++;
	    if (errors <= PRINT_ERRORS) {
	      printError(name, args);
	      PRINT(fmtCond,cond);
	      PRINT(fmt,a);
	      PRINT(fmt,b);
	      PRINT(fmt,cs);
	      PRINT(fmt,cp);
	    }
	    EXIT;
	  }
	}
	printErrorStats(errors, trials);
      }
    }
  };

} // namespace

#endif
