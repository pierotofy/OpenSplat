/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_FRAME_TIMER_HEADER
#define UTIL_FRAME_TIMER_HEADER

#include <ctime>

#include "util/defines.h"
#include "util/timer.h"
#include "util/system.h"

#define FRAME_TIMER_BUSY_WAITING 0

UTIL_NAMESPACE_BEGIN

/**
 * A timer class for frame-based applications.
 * With the get_time() method, the class provides a constant
 * time for each frame. A call to next_frame() updates this time.
 * By setting a max FPS rate, next_frame() also limits the frames
 * by sleeping. The code works with real-world time rather than
 * with CPU time or execution time (like std::clock()).
 */
class FrameTimer
{
public:
    FrameTimer (void);

    /** Sets the desired FPS. Zero disables frame limit. */
    void set_max_fps (std::size_t fps);

    /** Returns the maximum frames per second. */
    std::size_t get_max_fps (void) const;

    /**
     * Returns the current time in seconds since program start.
     * This time is constant over the whole frame.
     */
    float get_time_sec (void) const;

    /**
     * Returns the current time in milli seconds since program start.
     * This time is constant over the whole frame.
     */
    std::size_t get_time (void) const;

    /** Returns the amount of calls to nextFrame. */
    std::size_t get_frame_count (void) const;

    /** Called to update the current time and limit FPS. */
    void next_frame (void);

private:
    std::size_t now (void) const;
    std::size_t delay (std::size_t ms) const;

private:
    std::size_t cur_time; // Current time
    std::size_t last_time; // Last time

    std::size_t max_fps; // Maximum frames per second
    std::size_t frame_count; // Amount of calls to nextFrame

    WallTimer timer; // High-resolution timer
};

/* ---------------------------------------------------------------- */

inline
FrameTimer::FrameTimer (void)
{
    this->cur_time = now();
    this->last_time = this->cur_time;
    this->max_fps = 60;
    this->frame_count = 0;
}

inline void
FrameTimer::set_max_fps (std::size_t fps)
{
    this->max_fps = fps;
}

inline std::size_t
FrameTimer::get_max_fps (void) const
{
    return this->max_fps;
}

inline std::size_t
FrameTimer::get_time (void) const
{
    return this->cur_time;
}

inline float
FrameTimer::get_time_sec (void) const
{
    return (1.0f / 1000.0f) * (float)this->cur_time;
}

inline std::size_t
FrameTimer::get_frame_count (void) const
{
    return this->frame_count;
}

inline void
FrameTimer::next_frame (void)
{
    this->last_time = this->cur_time;
    this->cur_time = this->now();

    if (this->max_fps != 0)
    {
        std::size_t diff = this->cur_time - this->last_time;
        std::size_t frame_ms = 1000 / this->max_fps;

        if (diff < frame_ms)
            this->cur_time = delay(frame_ms - diff);
    }

    this->frame_count += 1;
}

inline std::size_t
FrameTimer::now (void) const
{
    return this->timer.get_elapsed();
}

inline std::size_t
FrameTimer::delay (std::size_t ms) const
{
    std::size_t curTime = this->cur_time;

#if FRAME_TIMER_BUSY_WAITING
    std::size_t endTime = cur_time + ms;
    while (curTime < endTime)
        curTime = now();
#else
    system::sleep(ms);
    curTime = now();
#endif

    return curTime;
}

UTIL_NAMESPACE_END

#endif /* UTIL_FRAME_TIMER_HEADER */
