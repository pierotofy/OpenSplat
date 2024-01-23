/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_LOGGING_HEADER
#define UTIL_LOGGING_HEADER

#include <iostream>
#include <ostream>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN

class Logging
{
public:
    enum LogLevel
    {
        LOG_ERROR,
        LOG_WARNING,
        LOG_INFO,
        LOG_VERBOSE,
        LOG_DEBUG
    };

    struct NullStream : public std::ostream
    {
        NullStream (void) : std::ostream(nullptr) {}
        NullStream (NullStream const&) =delete;
        template <typename T>
        NullStream& operator<< (T const& arg);
    };

public:
    Logging (void);
    Logging (LogLevel max_level);
    void set_max_level (LogLevel max_level);

    std::ostream& log (LogLevel log_level) const;
    std::ostream& error (void) const;
    std::ostream& warning (void) const;
    std::ostream& info (void) const;
    std::ostream& verbose (void) const;
    std::ostream& debug (void) const;

private:
    LogLevel max_level;
    NullStream nullstream;
};

/* ------------------------ Implementation ------------------------ */

template <typename T>
inline Logging::NullStream&
Logging::NullStream::operator<< (T const& /*arg*/)
{
    return *this;
}

inline
Logging::Logging (void)
    : max_level(LOG_INFO)
{
}

inline
Logging::Logging (LogLevel max_level)
    : max_level(max_level)
{
}

inline void
Logging::set_max_level (LogLevel max_level)
{
    this->max_level = max_level;
}

inline std::ostream&
Logging::log (LogLevel log_level) const
{
    if (log_level > this->max_level)
        return const_cast<NullStream&>(this->nullstream);
    return (log_level == LOG_ERROR) ? std::cerr : std::cout;
}

inline std::ostream&
Logging::error (void) const
{
    return this->log(LOG_ERROR);
}

inline std::ostream&
Logging::warning (void) const
{
    return this->log(LOG_WARNING);
}

inline std::ostream&
Logging::info (void) const
{
    return this->log(LOG_INFO);
}

inline std::ostream&
Logging::verbose (void) const
{
    return this->log(LOG_VERBOSE);
}

inline std::ostream&
Logging::debug (void) const
{
    return this->log(LOG_DEBUG);
}

UTIL_NAMESPACE_END

#endif  /* UTIL_LOGGING_HEADER */

