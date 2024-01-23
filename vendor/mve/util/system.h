/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_SYSTEM_HEADER
#define UTIL_SYSTEM_HEADER

#include <ctime>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <vector>
#include <istream>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN
UTIL_SYSTEM_NAMESPACE_BEGIN

/*
 * ------------------------- Sleep functions--------------------------
 */

/** Sleeps the given amount of milli seconds. */
void sleep (std::size_t msec);
/** Sleeps the given amount of seconds. */
void sleep_sec (float secs);

/*
 * ------------------------- Random functions-------------------------
 */

/** Initializes the random number generator. */
void rand_init (void);
/** Initializes the random number generator with a given seed. */
void rand_seed (int seed);
/** Returns a floating point random number in [0, 1]. */
float rand_float (void);
/** Returns a random number in [0, 2^31]. */
int rand_int (void);

/*
 * ---------------------- Signals / Application ----------------------
 */

/** Prints the application name and date and time of the build. */
void
print_build_timestamp (char const* application_name);

/** Prints the application name and the given date and time strings. */
void
print_build_timestamp (char const* application_name,
    char const* date, char const* time);

/** Registers signal SIGSEGV (segmentation fault) handler. */
void register_segfault_handler (void);

/** Handles signal SIGSEGV (segmentation fault) printing a stack trace. */
void signal_segfault_handler (int code);

/** Prints a stack trace. */
void print_stack_trace (void);

/*
 * ----------------------- Endian conversions ------------------------
 */

/** Swaps little/big endianess of the operand. */
template <int N>
inline void
byte_swap (char* data);

/** Little endian to host order conversion. */
template <typename T>
inline T
letoh (T const& x);

/** Big endian to host order conversion. */
template <typename T>
inline T
betoh (T const& x);

/** Reads little endian according to host order conversion. */
template <typename T>
inline T
read_binary_little_endian(std::istream* stream);

/* ---------------------------------------------------------------- */

inline void
sleep (std::size_t msec)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(msec));
}

inline void
sleep_sec (float secs)
{
    sleep((std::size_t)(secs * 1000.0f));
}

inline void
rand_init (void)
{
    std::srand(std::time(nullptr));
}

inline void
rand_seed (int seed)
{
    std::srand(seed);
}

inline float
rand_float (void)
{
    return (float)std::rand() / (float)RAND_MAX;
}

inline int
rand_int (void)
{
    return std::rand();
}

inline void
print_build_timestamp (char const* application_name)
{
    print_build_timestamp(application_name, __DATE__, __TIME__);
}

/*
 * Determine host byte-order.
 * Windows: "All versions of windows run little-endian, period."
 * http://social.msdn.microsoft.com/Forums/en-US/windowsmobiledev/thread/04c92ef9-e38e-415f-8958-ec9f7c196fd3
 */
#if defined(_WIN32) || defined(__CYGWIN__)
#   define HOST_BYTEORDER_LE
#endif

/*
 * The only Apple architectures with big-endian are PowerPC 32 and 64 bit.
 */
#if defined(__APPLE__)
#   if defined(__ppc__) || defined(__ppc64__)
#       define HOST_BYTEORDER_BE
#   else
#       define HOST_BYTEORDER_LE
#   endif
#endif

#if defined(__linux__)
#   include <endian.h> // part of glibc 2.9
#   if __BYTE_ORDER == __LITTLE_ENDIAN
#       define HOST_BYTEORDER_LE
#   else
#       define HOST_BYTEORDER_BE
#   endif
#endif

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#   include <machine/endian.h>
#   if BYTE_ORDER == LITTLE_ENDIAN
#       define HOST_BYTEORDER_LE
#   else
#       define HOST_BYTEORDER_BE
#   endif
#endif

#if defined(__SunOS)
#   include <sys/byteorder.h>
#   if defined(_LITTLE_ENDIAN)
#       define HOST_BYTEORDER_LE
#   else
#       define HOST_BYTEORDER_BE
#   endif
#endif

template <>
inline void
byte_swap<2> (char* data)
{
    std::swap(data[0], data[1]);
}

template <>
inline void
byte_swap<4> (char* data)
{
    std::swap(data[0], data[3]);
    std::swap(data[1], data[2]);
}

template <>
inline void
byte_swap<8> (char* data)
{
    std::swap(data[0], data[7]);
    std::swap(data[1], data[6]);
    std::swap(data[2], data[5]);
    std::swap(data[3], data[4]);
}

#if defined(HOST_BYTEORDER_LE) && defined(HOST_BYTEORDER_BE)
#   error "Host endianess can not be both LE and BE!"
#elif defined(HOST_BYTEORDER_LE)

template <typename T>
inline T
letoh (T const& x)
{
    return x;
}

template <typename T>
inline T
betoh (T const& x)
{
    T copy(x);
    byte_swap<sizeof(T)>(reinterpret_cast<char*>(&copy));
    return copy;
}

#elif defined(HOST_BYTEORDER_BE)

template <typename T>
inline T
letoh (T const& x)
{
    T copy(x);
    byte_swap<sizeof(T)>(reinterpret_cast<char*>(&copy));
    return copy;
}

template <typename T>
inline T
betoh (T const& x)
{
    return x;
}

#else
#   error "Couldn't determine host endianess!"
#endif

template <typename T>
inline T
read_binary_little_endian(std::istream* stream)
{
    T data_little_endian;
    stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
    return letoh(data_little_endian);
}

UTIL_SYSTEM_NAMESPACE_END
UTIL_NAMESPACE_END

#endif /* UTIL_SYSTEM_HEADER */
