/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_EXCEPTION_HEADER
#define UTIL_EXCEPTION_HEADER

#include <string>
#include <stdexcept>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN

/**
 * Universal, simple exception class.
 */
class Exception : public std::exception, public std::string
{
public:
    Exception (void) throw()
    { }

    Exception (std::string const& msg) throw() : std::string(msg)
    { }

    Exception (std::string const& msg, char const* msg2) throw()
        : std::string(msg)
    { this->append(msg2); }

    Exception (std::string const& msg, std::string const& msg2) throw()
        : std::string(msg)
    { this->append(msg2); }

    virtual ~Exception (void) throw()
    { }

    virtual const char* what (void) const throw()
    { return this->c_str(); }
};

/* ---------------------------------------------------------------- */

/**
 * Exception class for file exceptions with additional filename.
 */
class FileException : public Exception
{
public:
    FileException(std::string const& filename, std::string const& msg) throw()
        : Exception(msg), filename(filename)
    { }

    FileException(std::string const& filename, char const* msg) throw()
        : Exception(msg), filename(filename)
    { }

    virtual ~FileException (void) throw()
    { }

public:
    std::string filename;
};

UTIL_NAMESPACE_END

#endif /* UTIL_EXCEPTION_HEADER */
