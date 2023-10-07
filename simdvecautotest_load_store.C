// ===========================================================================
//
// simdvecautotest_load_store.C --
// automatic test of Vec load and store operations
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
// (C) Jonas Keller, Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

#include "SIMDVecAll.H"
#include "SIMDVecAutoTestSerial.H"
#include "SIMDVecAutoTestSerialMask.H"

#include <algorithm>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <sys/mman.h>
#include <unistd.h>

using namespace simd;
using namespace auto_test;

size_t getPageSize()
{
  // Buffer page size so that we don't have to call sysconf() every time.
  // PAGESIZE must not be less than 1 according to POSIX, so we can use 0 as
  // indicator that the page size has not been set yet.
  static size_t page_size = 0;
  if (page_size == 0) {
    page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  }
  return page_size;
}

/**
 * Two subsequent pages, the second one is protected against read/write access
 * so that a load/store operation to the second page will cause a segmentation
 * fault.
 */
struct TestPages
{
  TestPages()
  {
    // allocate two subsequent pages
    page1 = mmap(nullptr, 2 * getPageSize(), PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page1 == MAP_FAILED) {
      perror("ERROR: mmap");
      std::exit(EXIT_FAILURE);
    }

    // protect second page against read/write access
    page2 = static_cast<char *>(page1) + getPageSize();
    if (mprotect(page2, getPageSize(), PROT_NONE) == -1) {
      perror("ERROR: mprotect");
      std::exit(EXIT_FAILURE);
    }
  }

  ~TestPages()
  {
    // deallocate pages
    if (munmap(page1, 2 * getPageSize()) == -1) {
      perror("ERROR: munmap");
      std::exit(EXIT_FAILURE);
    }
  }

  void *getPage1() const { return page1; }
  void *getPage2() const { return page2; }
  size_t getPageSize() const { return ::getPageSize(); }

private:
  void *page1;
  void *page2;
};

void fillPageWithRandomData(void *page)
{
  std::generate_n(static_cast<char *>(page), getPageSize(),
                  std::bind(std::rand));
}

template <typename T, size_t SIMD_WIDTH>
SerialVec<T, SIMD_WIDTH> getRandomSerialVec()
{
  SerialVec<T, SIMD_WIDTH> sVec;
  sVec.randomize();
  return sVec;
}

template <typename T, size_t SIMD_WIDTH>
SerialMask<T, SIMD_WIDTH> getRandomSerialMask()
{
  SerialMask<T, SIMD_WIDTH> sMask;
  sMask.randomize();
  return sMask;
}

template <size_t SIMD_WIDTH>
size_t getRandomPageOffset(const size_t alignment = 1)
{
  // get random number between 0 and getPageSize() - SIMD_WIDTH - 1 with
  // alignment
  const size_t preAlignment =
    static_cast<size_t>(std::rand()) % (getPageSize() - SIMD_WIDTH);
  return (preAlignment / alignment) * alignment;
}

template <typename T, size_t SIMD_WIDTH>
size_t getRandomPageOffsetWithMask(const SerialMask<T, SIMD_WIDTH> &mask,
                                   const size_t alignment = 1)
{
  size_t numElems = mask.bits;
  while (numElems >= 1 && !mask[numElems - 1]) { --numElems; }

  const size_t numBytes = numElems * sizeof(T);

  // get random number between 0 and getPageSize() - numBytes - 1 with
  // alignment
  const size_t preAlignment =
    static_cast<size_t>(std::rand()) % (getPageSize() - numBytes);
  return (preAlignment / alignment) * alignment;
}

void printPageDiff(const void *const pageSerial, const void *const pageSIMD)
{
  const auto page1Bytes = static_cast<const unsigned char *>(pageSerial);
  const auto page2Bytes = static_cast<const unsigned char *>(pageSIMD);
  for (size_t i = 0; i < getPageSize(); ++i) {
    if (page1Bytes[i] != page2Bytes[i]) {
      std::cout << "Difference at byte " << i << std::endl;
      std::cout << "Page Serial:" << std::endl;
      for (size_t j = i - 10; j < i + 10; ++j) {
        std::cout << std::hex << static_cast<uint>(page1Bytes[j]) << " ";
      }
      std::cout << std::endl;
      std::cout << "Page SIMD:" << std::endl;
      for (size_t j = i - 10; j < i + 10; ++j) {
        std::cout << std::hex << static_cast<uint>(page2Bytes[j]) << " ";
      }
      std::cout << std::endl;
      break;
    }
  }
  std::cout << "Pages are equal." << std::endl;
}

template <typename T, size_t SIMD_WIDTH>
void printVec(const SerialVec<T, SIMD_WIDTH> &vec)
{
  for (size_t i = 0; i < vec.elems; ++i) { std::cout << vec[i] << " "; }
  std::cout << std::endl;
}

template <typename T, size_t SIMD_WIDTH>
void printMask(const SerialMask<T, SIMD_WIDTH> &mask)
{
  for (size_t i = 0; i < mask.bits; ++i) { std::cout << mask[i] << " "; }
  std::cout << std::endl;
}

template <typename T, size_t SIMD_WIDTH>
std::string makeName(const std::string name)
{
  return name + "<" + std::string(TypeInfo<T>::name()) + "," +
         std::to_string(SIMD_WIDTH) + ">";
}

template <typename T, size_t SIMD_WIDTH>
struct Store
{
  static constexpr long alignment = Vec<T, SIMD_WIDTH>::bytes;
  static std::string name() { return makeName<T, SIMD_WIDTH>("Store"); }
  static void store(T *const dst, const Vec<T, SIMD_WIDTH> &src)
  {
    ::simd::store(dst, src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct StoreU
{
  static constexpr long alignment = 1;
  static std::string name() { return makeName<T, SIMD_WIDTH>("StoreU"); }
  static void store(T *const dst, const Vec<T, SIMD_WIDTH> &src)
  {
    ::simd::storeu(dst, src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct StreamStore
{
  static constexpr long alignment = Vec<T, SIMD_WIDTH>::bytes;
  static std::string name() { return makeName<T, SIMD_WIDTH>("StreamStore"); }
  static void store(T *const dst, const Vec<T, SIMD_WIDTH> &src)
  {
    ::simd::stream_store(dst, src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct MaskStore
{
  static constexpr long alignment = Vec<T, SIMD_WIDTH>::bytes;
  static std::string name() { return makeName<T, SIMD_WIDTH>("MaskStore"); }
  static void mask_store(T *const dst, const Mask<T, SIMD_WIDTH> &mask,
                         const Vec<T, SIMD_WIDTH> &src)
  {
    ::simd::mask_store(dst, mask, src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct MaskStoreU
{
  static constexpr long alignment = 1;
  static std::string name() { return makeName<T, SIMD_WIDTH>("MaskStoreU"); }
  static void mask_store(T *const dst, const Mask<T, SIMD_WIDTH> &mask,
                         const Vec<T, SIMD_WIDTH> &src)
  {
    ::simd::mask_storeu(dst, mask, src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct Load
{
  static constexpr long alignment = Vec<T, SIMD_WIDTH>::bytes;
  static std::string name() { return makeName<T, SIMD_WIDTH>("Load"); }
  static Vec<T, SIMD_WIDTH> load(const T *const src)
  {
    return ::simd::load(src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct LoadU
{
  static constexpr long alignment = 1;
  static std::string name() { return makeName<T, SIMD_WIDTH>("LoadU"); }
  static Vec<T, SIMD_WIDTH> load(const T *const src)
  {
    return ::simd::loadu(src);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct MaskLoad
{
  static constexpr long alignment = Vec<T, SIMD_WIDTH>::bytes;
  static std::string name() { return makeName<T, SIMD_WIDTH>("MaskLoad"); }
  static Vec<T, SIMD_WIDTH> mask_load(const Vec<T, SIMD_WIDTH> &src,
                                      const Mask<T, SIMD_WIDTH> &mask,
                                      const T *const ptr)
  {
    return ::simd::mask_load(src, mask, ptr);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct MaskLoadU
{
  static constexpr long alignment = 1;
  static std::string name() { return makeName<T, SIMD_WIDTH>("MaskLoadU"); }
  static Vec<T, SIMD_WIDTH> mask_load(const Vec<T, SIMD_WIDTH> &src,
                                      const Mask<T, SIMD_WIDTH> &mask,
                                      const T *const ptr)
  {
    return ::simd::mask_loadu(src, mask, ptr);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct MaskZLoad
{
  static constexpr long alignment = Vec<T, SIMD_WIDTH>::bytes;
  static std::string name() { return makeName<T, SIMD_WIDTH>("MaskZLoad"); }
  static Vec<T, SIMD_WIDTH> maskz_load(const Mask<T, SIMD_WIDTH> &mask,
                                       const T *const ptr)
  {
    return ::simd::maskz_load(mask, ptr);
  }
};

template <typename T, size_t SIMD_WIDTH>
struct MaskZLoadU
{
  static constexpr long alignment = 1;
  static std::string name() { return makeName<T, SIMD_WIDTH>("MaskZLoadU"); }
  static Vec<T, SIMD_WIDTH> maskz_load(const Mask<T, SIMD_WIDTH> &mask,
                                       const T *const ptr)
  {
    return ::simd::maskz_loadu(mask, ptr);
  }
};

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT>
struct TestStore
{
  static void test(const size_t reps)
  {
    std::cout << "Testing " << FCT<T, SIMD_WIDTH>::name() << std::endl;
    const TestPages testPagesSerial, testPagesSIMD;
    for (size_t i = 0; i < reps; i++) {
      fillPageWithRandomData(testPagesSerial.getPage1());
      std::memcpy(testPagesSIMD.getPage1(), testPagesSerial.getPage1(),
                  getPageSize());

      const auto input = getRandomSerialVec<T, SIMD_WIDTH>();

      const auto pageOffset =
        getRandomPageOffset<SIMD_WIDTH>(FCT<T, SIMD_WIDTH>::alignment);

      void *const storeLocSerial =
        static_cast<char *>(testPagesSerial.getPage1()) + pageOffset;
      void *const storeLocSIMD =
        static_cast<char *>(testPagesSIMD.getPage1()) + pageOffset;

      // write input to page1 at pageOffset
      for (size_t i = 0; i < input.elems; ++i) {
        static_cast<T *>(storeLocSerial)[i] = input[i];
      }
      FCT<T, SIMD_WIDTH>::store(static_cast<T *>(storeLocSIMD), input.getVec());

      // compare both first pages
      if (std::memcmp(testPagesSerial.getPage1(), testPagesSIMD.getPage1(),
                      getPageSize()) != 0) {
        std::cout << "ERROR: " << FCT<T, SIMD_WIDTH>::name() << " failed"
                  << std::endl;
        printPageDiff(testPagesSerial.getPage1(), testPagesSIMD.getPage1());
        std::cout << "Input:" << std::endl;
        printVec(input);
        std::cout << "Offset: " << pageOffset << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
};

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT>
struct TestMaskStore
{
  static void test(const size_t reps)
  {
    std::cout << "Testing " << FCT<T, SIMD_WIDTH>::name() << std::endl;
    const TestPages testPagesSerial, testPagesSIMD;
    for (size_t i = 0; i < reps; i++) {
      fillPageWithRandomData(testPagesSerial.getPage1());
      std::memcpy(testPagesSIMD.getPage1(), testPagesSerial.getPage1(),
                  getPageSize());

      const auto input = getRandomSerialVec<T, SIMD_WIDTH>();
      const auto mask  = getRandomSerialMask<T, SIMD_WIDTH>();

      const auto pageOffset =
        getRandomPageOffsetWithMask(mask, FCT<T, SIMD_WIDTH>::alignment);

      void *const storeLocSerial =
        static_cast<char *>(testPagesSerial.getPage1()) + pageOffset;
      void *const storeLocSIMD =
        static_cast<char *>(testPagesSIMD.getPage1()) + pageOffset;

      // write input to page1 at pageOffset
      for (size_t i = 0; i < input.elems; ++i) {
        if (mask[i]) { static_cast<T *>(storeLocSerial)[i] = input[i]; }
      }
      FCT<T, SIMD_WIDTH>::mask_store(static_cast<T *>(storeLocSIMD),
                                     mask.getMask(), input.getVec());

      // compare both first pages
      if (std::memcmp(testPagesSerial.getPage1(), testPagesSIMD.getPage1(),
                      getPageSize()) != 0) {
        std::cout << "ERROR: " << FCT<T, SIMD_WIDTH>::name() << " failed"
                  << std::endl;
        printPageDiff(testPagesSerial.getPage1(), testPagesSIMD.getPage1());
        std::cout << "Input:" << std::endl;
        printVec(input);
        std::cout << "Mask:" << std::endl;
        printMask(mask);
        std::cout << "Offset: " << pageOffset << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
};

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT>
struct TestLoad
{
  static void test(const size_t reps)
  {
    std::cout << "Testing " << FCT<T, SIMD_WIDTH>::name() << std::endl;
    const TestPages testPagesSerial, testPagesSIMD;
    for (size_t i = 0; i < reps; i++) {
      fillPageWithRandomData(testPagesSerial.getPage1());
      std::memcpy(testPagesSIMD.getPage1(), testPagesSerial.getPage1(),
                  getPageSize());

      const auto pageOffset =
        getRandomPageOffset<SIMD_WIDTH>(FCT<T, SIMD_WIDTH>::alignment);

      const void *const loadLocSerial =
        static_cast<const char *>(testPagesSerial.getPage1()) + pageOffset;
      const void *const loadLocSIMD =
        static_cast<const char *>(testPagesSIMD.getPage1()) + pageOffset;

      // load from page1 at pageOffset
      SerialVec<T, SIMD_WIDTH> vecSerial;
      for (size_t i = 0; i < vecSerial.elems; ++i) {
        // volatile to prevent compiler from optimizing, since some versions of
        // gcc erroneously optimize this with an aligned load despite
        // loadLocSerial not necessarily being aligned
        vecSerial[i] = static_cast<volatile const T *>(loadLocSerial)[i];
      }
      const auto vecSIMD = SerialVec<T, SIMD_WIDTH>::fromVec(
        FCT<T, SIMD_WIDTH>::load(static_cast<const T *>(loadLocSIMD)));

      if (std::memcmp(vecSerial.vec, vecSIMD.vec, vecSerial.elems) != 0) {
        std::cout << "ERROR: " << FCT<T, SIMD_WIDTH>::name() << " failed"
                  << std::endl;
        printPageDiff(testPagesSerial.getPage1(), testPagesSIMD.getPage1());
        std::cout << "Serial:" << std::endl;
        printVec(vecSerial);
        std::cout << "SIMD:" << std::endl;
        printVec(vecSIMD);
        std::cout << "Offset: " << pageOffset << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
};

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT>
struct TestMaskLoad
{
  static void test(const size_t reps)
  {
    std::cout << "Testing " << FCT<T, SIMD_WIDTH>::name() << std::endl;
    const TestPages testPagesSerial, testPagesSIMD;
    for (size_t i = 0; i < reps; i++) {
      fillPageWithRandomData(testPagesSerial.getPage1());
      std::memcpy(testPagesSIMD.getPage1(), testPagesSerial.getPage1(),
                  getPageSize());

      const auto mask = getRandomSerialMask<T, SIMD_WIDTH>();
      const auto src  = getRandomSerialVec<T, SIMD_WIDTH>();

      const auto pageOffset =
        getRandomPageOffsetWithMask(mask, FCT<T, SIMD_WIDTH>::alignment);

      const void *const loadLocSerial =
        static_cast<const char *>(testPagesSerial.getPage1()) + pageOffset;
      const void *const loadLocSIMD =
        static_cast<const char *>(testPagesSIMD.getPage1()) + pageOffset;

      // load from page1 at pageOffset
      SerialVec<T, SIMD_WIDTH> vecSerial;
      for (size_t i = 0; i < vecSerial.elems; ++i) {
        // volatile to prevent compiler from optimizing, since some versions of
        // gcc erroneously optimize this with an aligned load despite
        // loadLocSerial not necessarily being aligned
        vecSerial[i] =
          mask[i] ? static_cast<volatile const T *>(loadLocSerial)[i] : src[i];
      }
      const auto vecSIMD =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<T, SIMD_WIDTH>::mask_load(
          src.getVec(), mask.getMask(), static_cast<const T *>(loadLocSIMD)));

      if (std::memcmp(vecSerial.vec, vecSIMD.vec, vecSerial.elems) != 0) {
        std::cout << "ERROR: " << FCT<T, SIMD_WIDTH>::name() << " failed"
                  << std::endl;
        printPageDiff(testPagesSerial.getPage1(), testPagesSIMD.getPage1());
        std::cout << "Serial:" << std::endl;
        printVec(vecSerial);
        std::cout << "SIMD:" << std::endl;
        printVec(vecSIMD);
        std::cout << "Mask:" << std::endl;
        printMask(mask);
        std::cout << "Src:" << std::endl;
        printVec(src);
        std::cout << "Offset: " << pageOffset << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
};

template <typename T, size_t SIMD_WIDTH, template <typename, size_t> class FCT>
struct TestMaskZLoad
{
  static void test(const size_t reps)
  {
    std::cout << "Testing " << FCT<T, SIMD_WIDTH>::name() << std::endl;
    const TestPages testPagesSerial, testPagesSIMD;
    for (size_t i = 0; i < reps; i++) {
      fillPageWithRandomData(testPagesSerial.getPage1());
      std::memcpy(testPagesSIMD.getPage1(), testPagesSerial.getPage1(),
                  getPageSize());

      const auto mask = getRandomSerialMask<T, SIMD_WIDTH>();

      const auto pageOffset =
        getRandomPageOffsetWithMask(mask, FCT<T, SIMD_WIDTH>::alignment);

      const void *const loadLocSerial =
        static_cast<const char *>(testPagesSerial.getPage1()) + pageOffset;
      const void *const loadLocSIMD =
        static_cast<const char *>(testPagesSIMD.getPage1()) + pageOffset;

      // load from page1 at pageOffset
      SerialVec<T, SIMD_WIDTH> vecSerial;
      for (size_t i = 0; i < vecSerial.elems; ++i) {
        // volatile to prevent compiler from optimizing, since some versions of
        // gcc erroneously optimize this with an aligned load despite
        // loadLocSerial not necessarily being aligned
        vecSerial[i] =
          mask[i] ? static_cast<volatile const T *>(loadLocSerial)[i] : 0;
      }
      const auto vecSIMD =
        SerialVec<T, SIMD_WIDTH>::fromVec(FCT<T, SIMD_WIDTH>::maskz_load(
          mask.getMask(), static_cast<const T *>(loadLocSIMD)));

      if (std::memcmp(vecSerial.vec, vecSIMD.vec, vecSerial.elems) != 0) {
        std::cout << "ERROR: " << FCT<T, SIMD_WIDTH>::name() << " failed"
                  << std::endl;
        printPageDiff(testPagesSerial.getPage1(), testPagesSIMD.getPage1());
        std::cout << "Mask:" << std::endl;
        printMask(mask);
        std::cout << "Serial:" << std::endl;
        printVec(vecSerial);
        std::cout << "SIMD:" << std::endl;
        printVec(vecSIMD);
        std::cout << "Offset: " << pageOffset << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
};

template <size_t SIMD_WIDTH,
          template <typename, size_t, template <typename, size_t> class>
          class TEST,
          template <typename, size_t> class FCT>
void testAll(const size_t reps)
{
  TEST<Byte, SIMD_WIDTH, FCT>::test(reps);
  TEST<SignedByte, SIMD_WIDTH, FCT>::test(reps);
  TEST<Short, SIMD_WIDTH, FCT>::test(reps);
  TEST<Word, SIMD_WIDTH, FCT>::test(reps);
  TEST<Int, SIMD_WIDTH, FCT>::test(reps);
  TEST<Float, SIMD_WIDTH, FCT>::test(reps);
#ifdef SIMD_DOUBLE
  TEST<Double, SIMD_WIDTH, FCT>::test(reps);
#endif
}

const auto SW = NATIVE_SIMD_WIDTH;

int main(int argc, char *argv[])
{
  // number of repetitions
  const size_t reps =
    argc >= 2 ? static_cast<size_t>(std::atol(argv[1])) : 10000;

  // seed random number generator
  std::srand(static_cast<unsigned int>(time(nullptr)));

  // set up signal handler for SIGSEGV
  {
    struct sigaction sa;
    sa.sa_handler = [](int) {
      std::cout << "ERROR: Segmentation fault occurred." << std::endl;
      std::exit(EXIT_FAILURE);
    };
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;

    if (sigaction(SIGSEGV, &sa, nullptr) == -1) {
      perror("ERROR: sigaction");
      std::exit(EXIT_FAILURE);
    }
  }

  testAll<SW, TestStore, Store>(reps);
  testAll<SW, TestStore, StoreU>(reps);
  testAll<SW, TestStore, StreamStore>(reps);
  testAll<SW, TestMaskStore, MaskStore>(reps);
  testAll<SW, TestMaskStore, MaskStoreU>(reps);
  testAll<SW, TestLoad, Load>(reps);
  testAll<SW, TestLoad, LoadU>(reps);
  testAll<SW, TestMaskLoad, MaskLoad>(reps);
  testAll<SW, TestMaskLoad, MaskLoadU>(reps);
  testAll<SW, TestMaskZLoad, MaskZLoad>(reps);
  testAll<SW, TestMaskZLoad, MaskZLoadU>(reps);

  std::cout << "simdvecautotest_load_store complete" << std::endl;
  return 0;
}
