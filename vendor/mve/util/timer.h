/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 *
 * - WallTimer: A real world "user" time timer.
 *   Useful for timings displayed to the user.
 * - ClockTimer: A timer that measures execution times.
 *   Useful for measuring performance of the application because it excludes
 *   I/O preemtion and runs faster when multiple threads are running.
 */
#ifndef UTIL_TIMER_HEADER
#define UTIL_TIMER_HEADER

#include <chrono>
#include <ctime>

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN

/**
 * Cross-platform high-resolution real-time timer.
 * This implementation returns milli seconds as smallest unit.
 */
class WallTimer
{
private:
    typedef std::chrono::high_resolution_clock TimerClock;
    typedef std::chrono::time_point<TimerClock> TimerTimePoint;
    typedef std::chrono::duration<double, std::milli> TimerDurationMs;

    TimerTimePoint start;

public:
    WallTimer (void);
    void reset (void);

    /** Returns the milli seconds since last reset. */
    std::size_t get_elapsed (void) const;

    /** Returns the seconds since last reset. */
    float get_elapsed_sec (void) const;
};

/* ---------------------------------------------------------------- */

/**
 * Simple timer class to take execution times. The reported
 * float values are in seconds, the integer values are in milli seconds.
 * The functions that provide milli seconds should be preferred. The
 * precision of this timer is limited (~10ms, depending on the system).
 *
 * This class should not be used for timings that rely on the
 * actual real world time but rather for computational timings.
 * The timings here are pure processing times which vary from
 * real time if the application is scheduled (e.g. when using I/O)
 * or sleeps or uses several threads.
 */
class ClockTimer
{
private:
    std::size_t start;

public:
    ClockTimer (void);
    void reset (void);

    static float now_sec (void);
    static std::size_t now (void);

    std::size_t get_elapsed (void) const;
    float get_elapsed_sec (void) const;
};

/* ---------------------------------------------------------------- */

inline
WallTimer::WallTimer (void)
{
    this->reset();
}

inline void
WallTimer::reset (void)
{
    this->start = TimerClock::now();
}

inline std::size_t
WallTimer::get_elapsed (void) const
{
    TimerDurationMs diff = TimerClock::now() - this->start;
    return diff.count();
}

inline float
WallTimer::get_elapsed_sec (void) const
{
    return static_cast<float>(this->get_elapsed()) / 1000.0f;
}

/* ---------------------------------------------------------------- */

inline
ClockTimer::ClockTimer (void)
{
    this->reset();
}

inline void
ClockTimer::reset (void)
{
    this->start = ClockTimer::now();
}

inline float
ClockTimer::now_sec (void)
{
    return (float)std::clock() / (float)CLOCKS_PER_SEC;
}

inline std::size_t
ClockTimer::now (void)
{
    return ((std::size_t)(std::clock()) * 1000) / (std::size_t)CLOCKS_PER_SEC;
}

inline float
ClockTimer::get_elapsed_sec (void) const
{
    return (1.0f / 1000.0f) * (float)this->get_elapsed();
}

inline std::size_t
ClockTimer::get_elapsed (void) const
{
    return ClockTimer::now() - start;
}

UTIL_NAMESPACE_END

#endif /* UTIL_TIMER_HEADER */
