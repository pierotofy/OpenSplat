/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_LINE_HEADER
#define MATH_LINE_HEADER

#include "math/defines.h"
#include "math/vector.h"

MATH_NAMESPACE_BEGIN

template <class T> class Line3;
typedef Line3<float> Line3f;
typedef Line3<double> Line3d;

/**
 * Class that represents a line using a point and a vector.
 * The normal is expected to have unit length.
 */
template <class T>
class Line3
{
public:
    /** Creates an uninitialized line. */
    Line3 (void);

    /** Creates a line with normal n and distance d from the origin. */
    Line3 (math::Vector<T, 3> const& d, math::Vector<T, 3> const& p);

public:
    math::Vector<T, 3> d;
    math::Vector<T, 3> p;
};

/* ---------------------------------------------------------------- */

template <class T>
inline
Line3<T>::Line3 (void)
{
}

template <class T>
inline
Line3<T>::Line3 (math::Vector<T, 3> const& d, math::Vector<T, 3> const& p)
    : d(d), p(p)
{
}

MATH_NAMESPACE_END

#endif /* MATH_LINE_HEADER */
