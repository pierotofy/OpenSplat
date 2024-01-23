/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_DEFINES_HEADER
#define UTIL_DEFINES_HEADER

#define UTIL_NAMESPACE_BEGIN namespace util {
#define UTIL_NAMESPACE_END }

#define UTIL_FS_NAMESPACE_BEGIN namespace fs {
#define UTIL_FS_NAMESPACE_END }

#define UTIL_STRING_NAMESPACE_BEGIN namespace string {
#define UTIL_STRING_NAMESPACE_END }

#define UTIL_SYSTEM_NAMESPACE_BEGIN namespace system {
#define UTIL_SYSTEM_NAMESPACE_END }

#ifndef STD_NAMESPACE_BEGIN
#   define STD_NAMESPACE_BEGIN namespace std {
#   define STD_NAMESPACE_END }
#endif

/** Parser, tokenizer, timer, smart pointer, threads, etc. */
UTIL_NAMESPACE_BEGIN
/** Cross-platform file system functions. */
UTIL_FS_NAMESPACE_BEGIN UTIL_FS_NAMESPACE_END
/** String conversions and helper functions. */
UTIL_STRING_NAMESPACE_BEGIN UTIL_STRING_NAMESPACE_END
/** Cross-platform operating system related functions. */
UTIL_SYSTEM_NAMESPACE_BEGIN UTIL_SYSTEM_NAMESPACE_END
UTIL_NAMESPACE_END

#endif /* UTIL_DEFINES_HEADER */
