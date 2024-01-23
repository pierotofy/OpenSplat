/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_STRING_HEADER
#define UTIL_STRING_HEADER

#include <sstream>
#include <string>
#include <iomanip>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN
UTIL_STRING_NAMESPACE_BEGIN

/** From arbitrary types to string conversion. **/
template <typename T>
std::string get (T const& value);

/** Returns string with 'digits' of fixed precision (fills with zeros). */
template <typename T>
std::string get_fixed (T const& value, int digits);

/** Returns string with 'digits' of precision. */
template <typename T>
std::string get_digits (T const& value, int digits);

/** Returns a string filled to the left to a length of 'width' chars. */
template <typename T>
std::string get_filled (T const& value, int width, char fill = '0');

/** From string to other types conversions. */
template <typename T>
T convert (std::string const& str, bool strict_conversion = true);

/** String representation for types. */
template <typename T>
char const* for_type (void);

/**
 * Returns the byte size of given type string (e.g. 1 for "uint8").
 * If the type is unknown, the function returns 0.
 */
int size_for_type_string (std::string const& typestring);

/** Inserts 'delim' every 'spacing' characters from the right, in-place. */
void punctate (std::string* input, char delim = ',', std::size_t spacing = 3);

/** Inserts 'delim' every 'spacing' characters from the right. */
std::string punctated (std::string const& input,
    char delim = ',', std::size_t spacing = 3);

/** Clips whitespaces from the front and end of the string, in-place. */
void clip_whitespaces (std::string* str);

/** Clips whitespaces from the front and end of the string. */
std::string clipped_whitespaces (std::string const& str);

/** Clips newlines from the end of the string, in-place. */
void clip_newlines (std::string* str);

/** Clips newlines from the end of the string. */
std::string clipped_newlines (std::string const& str);

/** Inserts line breaks on word boundaries to limit lines to 'width' chars. */
std::string wordwrap (char const* str, int width);

/**
 * Reduces string size by inserting "..." at the end (type = 0),
 * in the middle (type = 1) or at the beginning  (type = 2).
 */
std::string ellipsize (std::string const& in, std::size_t chars, int type = 0);

/** Replaces several whitespaces with a single blank, in-place. */
void normalize (std::string* str);

/** Replaces several whitespaces with a single blank. */
std::string normalized (std::string const& str);

/** Returns the leftmost 'chars' characters of 'str'. */
std::string left (std::string const& str, std::size_t chars);

/** Returns the rightmost 'chars' characters of 'str'. */
std::string right (std::string const& str, std::size_t chars);

/** Returns a lower-case version of the string. */
std::string lowercase (std::string const& str);

/** Returns an upper-case version of the string. */
std::string uppercase (std::string const& str);

/** Returns a string with a human readable byte size, e.g. 9.3 MB. */
std::string get_size_string (std::size_t size);

/* ---------------------------------------------------------------- */

template <typename T>
inline std::string
get (T const& value)
{
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <>
inline std::string
get (std::string const& value)
{
    return value;
}

template <typename T>
inline std::string
get_fixed (T const& value, int digits)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(digits) << value;
    return ss.str();
}

template <typename T>
inline std::string
get_digits (T const& value, int digits)
{
    std::stringstream ss;
    ss << std::setprecision(digits) << value;
    return ss.str();
}

template <typename T>
inline std::string
get_filled (T const& value, int width, char fill)
{
    std::stringstream ss;
    ss << std::setw(width) << std::setfill(fill) << value;
    return ss.str();
}

template <typename T>
inline T
convert (std::string const& str, bool strict_conversion)
{
    std::stringstream ss(str);
    T ret = T();
    ss >> ret;
    if (strict_conversion && (!ss.eof() || ss.fail()))
        throw std::invalid_argument("Invalid string conversion: " + str);
    return ret;
}

template <typename T>
inline char const*
for_type (void)
{
    return "unknown";
}

template <>
inline char const*
for_type<int8_t> (void)
{
    return "sint8";
}

#ifdef __GNUC__
/* Note: 'char' is neither recognized as 'int8_t' nor 'uint8_t'. We assume
 * that 'char' is singed, which is not always true:
 * http://www.network-theory.co.uk/docs/gccintro/gccintro_71.html
 */
template <>
inline char const*
for_type<char> (void)
{
    return "sint8";
}
#endif

template <>
inline char const*
for_type<int16_t> (void)
{
    return "sint16";
}

template <>
inline char const*
for_type<int32_t> (void)
{
    return "sint32";
}

template <>
inline char const*
for_type<int64_t> (void)
{
    return "sint64";
}

template <>
inline char const*
for_type<uint8_t> (void)
{
    return "uint8";
}

template <>
inline char const*
for_type<uint16_t> (void)
{
    return "uint16";
}

template <>
inline char const*
for_type<uint32_t> (void)
{
    return "uint32";
}

template <>
inline char const*
for_type<uint64_t> (void)
{
    return "uint64";
}

template <>
inline char const*
for_type<float> (void)
{
    return "float";
}

template <>
inline char const*
for_type<double> (void)
{
    return "double";
}

inline int
size_for_type_string (std::string const& typestring)
{
    if (typestring == "sint8" || typestring == "uint8")
        return 1;
    else if (typestring == "sint16" || typestring == "uint16")
        return 2;
    else if (typestring == "sint32" || typestring == "uint32")
        return 4;
    else if (typestring == "sint64" || typestring == "uint64")
        return 8;
    else if (typestring == "float")
        return sizeof(float);
    else if (typestring == "double")
        return sizeof(double);
    else
        return 0;
}

inline void
punctate (std::string* str, char delim, std::size_t spacing)
{
    if (str->size() <= spacing || spacing == 0)
        return;

    std::size_t pos = str->size() - 1;
    std::size_t cnt = 0;
    while (pos > 0)
    {
        cnt += 1;
        if (cnt == spacing)
        {
            str->insert(str->begin() + pos, delim);
            cnt = 0;
        }
        pos -= 1;
    }
}

inline std::string
punctated (std::string const& input, char delim, std::size_t spacing)
{
    std::string ret(input);
    punctate(&ret, delim, spacing);
    return ret;
}

inline void
clip_whitespaces (std::string* str)
{
    // TODO: Use str->back() and str->front() once C++11 is standard.
    while (!str->empty() && (*str->rbegin() == ' ' || *str->rbegin() == '\t'))
        str->resize(str->size() - 1);
    while (!str->empty() && (*str->begin() == ' ' || *str->begin() == '\t'))
        str->erase(str->begin());
}

inline std::string
clipped_whitespaces (std::string const& str)
{
    std::string ret(str);
    clip_whitespaces(&ret);
    return ret;
}

inline void
clip_newlines (std::string* str)
{
    while (!str->empty() && (*str->rbegin() == '\r' || *str->rbegin() == '\n'))
        str->resize(str->size() - 1);
}


inline std::string
clipped_newlines (std::string const& str)
{
    std::string ret(str);
    clip_newlines(&ret);
    return ret;
}

inline std::string
wordwrap (char const* str, int width)
{
    if (str == nullptr)
        return std::string();
    if (width <= 0)
        return str;

    int spaceleft = width;
    bool firstword = true;
    std::string out;
    for (int i = 0, word = 0; true; ++i)
    {
        char c(str[i]);
        bool softbreak = (c == ' ' || c == '\t' || c == '\0' || c == '\n');

        if (softbreak)
        {
            if (word > spaceleft)
            {
                if (!firstword)
                    out.append(1, '\n');
                out.append(str + i - word, word);
                spaceleft = width - word - 1;
            }
            else
            {
                if (!firstword)
                    out.append(1, ' ');
                out.append(str + i - word, word);
                spaceleft -= word + 1;
            }
            firstword = false;

            word = 0;
            if (c == '\n')
            {
                out.append(1, '\n');
                firstword = true;
                word = 0;
                spaceleft = width;
            }
        }
        else
        {
            word += 1;
        }

        if (str[i] == '\0')
            break;
    }

    return out;
}

inline std::string
ellipsize (std::string const& str, std::size_t chars, int type)
{
    if (str.size() <= chars)
        return str;

    switch (type)
    {
        case 0:
            return str.substr(0, chars - 3) + "...";
        case 1:
            return str.substr(0, chars / 2 - 1) + "..."
                + str.substr(str.size() - chars / 2 + 1);
        case 2:
            return "..." + str.substr(str.size() - chars + 3);
        default:
            break;
    }
    return str;
}

inline void
normalize (std::string* str)
{
  std::size_t iter = 0;
  bool was_whitespace = false;
  while (iter < str->size())
  {
    if (str->at(iter) == '\t')
      str->at(iter) = ' ';

    if (str->at(iter) == ' ')
    {
      if (was_whitespace)
      {
        str->erase(str->begin() + iter);
        was_whitespace = true;
        continue;
      }
      was_whitespace = true;
    }
    else
    {
      was_whitespace = false;
    }

    iter += 1;
  }
}

inline std::string
normalized (std::string const& str)
{
    std::string ret(str);
    normalize(&ret);
    return ret;
}

inline std::string
left (std::string const& str, std::size_t chars)
{
    return str.substr(0, std::min(str.size(), chars));
}

inline std::string
right (std::string const& str, std::size_t chars)
{
    return str.substr(str.size() - std::min(str.size(), chars));
}

inline std::string
lowercase (std::string const& str)
{
    std::string ret(str);
    for (std::size_t i = 0; i < str.size(); ++i)
        if (ret[i] >= 0x41 && ret[i] <= 0x5a)
            ret[i] += 0x20;
    return ret;
}

inline std::string
uppercase (std::string const& str)
{
    std::string ret(str);
    for (std::size_t i = 0; i < str.size(); ++i)
        if (ret[i] >= 0x61 && ret[i] <= 0x7a)
            ret[i] -= 0x20;
    return ret;
}

inline std::string
get_size_string (std::size_t size)
{
    std::string size_str;
    double size_flt;

    if (size < 1000)            /* 1000 * 1024^0 */
    {
        size_flt = (double)size;
        size_str = " B";
    }
    else if (size < 1024000)    /* 1000 * 1024^1 */
    {
        size_flt = size / 1024.0;
        size_str = " KB";
    }
    else if (size < 1048576000) /* 1000 * 1024^2 */
    {
        size_flt = size / 1048576.0;
        size_str = " MB";
    }
    else
    {
        size_flt = size / 1073741824.0;
        size_str = " GB";
    }

    int digits = 1;
    if (size_flt >= 10.0)
        digits = 0;

    return util::string::get_fixed(size_flt, digits) + size_str;
}

UTIL_STRING_NAMESPACE_END
UTIL_NAMESPACE_END

#endif /* UTIL_STRING_HEADER */
