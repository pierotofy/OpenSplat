/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef UTIL_BEZIERCURVE_HEADER
#define	UTIL_BEZIERCURVE_HEADER

#include <vector>
#include <stdexcept>

#include "math/defines.h"

MATH_NAMESPACE_BEGIN

/**
 * A class for defining and evaluating Bezier curves in all dimensions.
 */
template <class T>
class BezierCurve
{
public:
    BezierCurve (void);
    virtual ~BezierCurve (void);

    /**
     * Appends a new end point or control point to the curve.
     * The first and the last point are end points, the points in
     * between are control points. The degree of the polynomial is then
     * the total amount of points minus one.
     */
    void append_point (T const& p);

    /** Removes all control points, resetting the curve. */
    void clear (void);

    /** Returns the amount of control points in the vector. */
    std::size_t size (void) const;

    /** Returns the control point at the given index. */
    T const& operator[] (std::size_t index) const;

    /** Evaluates the Bezier curve at position t in [0, 1]. */
    T evaluate (float t) const;

private:
    typedef std::vector<T> ControlPointVector;
    ControlPointVector cp; // The control points
};

/* ------------------------------------------------------------ */

template <class T>
inline
BezierCurve<T>::BezierCurve (void)
{
}

template <class T>
inline
BezierCurve<T>::~BezierCurve (void)
{
}

template <class T>
inline void
BezierCurve<T>::append_point (T const& p)
{
    this->cp.push_back(p);
}

template <class T>
inline T
BezierCurve<T>::evaluate (float t) const
{
    if (this->cp.size() < 2)
        throw std::domain_error("Curve must have at least two end points!");

    /*
     * Implementation of the de Casteljau algorithm.
     * Complexity: O(d^2) for polynomial degree d = size(cp) - 1.
     *
     * 1. Create copy of the control point vector.
     * 2. Simplify using linear interpolation.
     * 3. Return the value of the last simplification step.
     */

    t = std::max(0.0f, std::min(1.0f, t));
    ControlPointVector cpr(this->cp);
    while (cpr.size() > 1)
    {
        for (std::size_t i = 0; i < cpr.size() - 1; ++i)
            cpr[i] = cpr[i] * (1.0f - t) + cpr[i+1] * t;
        cpr.pop_back();
    }

    return cpr.front();
}

template <class T>
inline void
BezierCurve<T>::clear (void)
{
    this->cp.clear();
}


template <class T>
inline std::size_t
BezierCurve<T>::size (void) const
{
    return this->cp.size();
}

template <class T>
inline T const&
BezierCurve<T>::operator[] (std::size_t index) const
{
    return this->cp[index];
}

MATH_NAMESPACE_END

#endif/* UTIL_BEZIERCURVE_HEADER */
