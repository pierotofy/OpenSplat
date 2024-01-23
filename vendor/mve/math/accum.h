/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_ACCUM_HEADER
#define MATH_ACCUM_HEADER

#include "math/functions.h"
#include "math/defines.h"

MATH_NAMESPACE_BEGIN

/**
 * Accumulator class that operates on arbitrary types.
 *
 * The idea behind this is that values can be accumulated even for
 * basic integral types, such as unsigned chars, where accumulation
 * would normally quickly cause overflows or rounding errors.
 *
 * Accumulation of arbitrary types is handled by specializations of
 * the 'Accum' class. For example, unsigned char values are all
 * internally converted to float to achieve accurate results.
 *
 * Note: This class currently supports:
 *   - Floating point types
 *   - unsigned char (through specialization)
 */
template <typename T>
class Accum
{
public:
    T v;
    float w;

public:
    /** Leaves internal value uninitialized. */
    Accum (void);

    /** Initializes the internal value (usually to zero). */
    Accum (T const& init);

    /** Adds the weighted given value to the internal value. */
    void add (T const& value, float weight);

    /** Subtracts the weighted given value from the internal value. */
    void sub (T const& value, float weight);

    /**
     * Returns a normalized version of the internal value,
     * i.e. dividing the internal value by the given weight.
     * The internal value is not changed by this operation.
     */
    T normalized (float weight) const;

    /**
     * Returns a normalized version of the internal value,
     * i.e. dividing the internal value by the internal weight,
     * which is the cumulative weight from the 'add' calls.
     */
    T normalized (void) const;
};

/* ------------------------- Implementation ----------------------- */

template <typename T>
inline
Accum<T>::Accum (void)
    : w(0.0f)
{
}

template <typename T>
inline
Accum<T>::Accum (T const& init)
    : v(init), w(0.0f)
{
}

template <typename T>
inline void
Accum<T>::add (T const& value, float weight)
{
    this->v += value * weight;
    this->w += weight;
}

template <typename T>
inline void
Accum<T>::sub (T const& value, float weight)
{
    this->v -= value * weight;
    this->w -= weight;
}

template <typename T>
inline T
Accum<T>::normalized (float weight) const
{
    return this->v / weight;
}

template <typename T>
inline T
Accum<T>::normalized (void) const
{
    return this->v / this->w;
}

/* ------------------------- Specialization ----------------------- */

template <>
class Accum<unsigned char>
{
public:
    float v;
    float w;

public:
    Accum (void);
    Accum (unsigned char const& init);
    void add (unsigned char const& value, float weight);
    void sub (unsigned char const& value, float weight);
    unsigned char normalized (float weight) const;
    unsigned char normalized (void) const;
};

/* ------------------------- Implementation ----------------------- */

inline
Accum<unsigned char>::Accum (void)
    : w(0.0f)
{
}

inline
Accum<unsigned char>::Accum (unsigned char const& init)
    : v(init), w(0.0f)
{
}

inline void
Accum<unsigned char>::add (unsigned char const& value, float weight)
{
    this->v += static_cast<float>(value) * weight;
    this->w += weight;
}

inline void
Accum<unsigned char>::sub (unsigned char const& value, float weight)
{
    this->v -= static_cast<float>(value) * weight;
    this->w -= weight;
}

inline unsigned char
Accum<unsigned char>::normalized (float weight) const
{
    return static_cast<unsigned char>(math::round(this->v / weight));
}

inline unsigned char
Accum<unsigned char>::normalized (void) const
{
    return static_cast<unsigned char>(math::round(this->v / this->w));
}

MATH_NAMESPACE_END

#endif /* MATH_ACCUM_HEADER */
