/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_PLANE_HEADER
#define MATH_PLANE_HEADER

#include "math/defines.h"
#include "math/vector.h"

MATH_NAMESPACE_BEGIN

template <class T> class Plane3;
typedef Plane3<float> Plane3f;
typedef Plane3<double> Plane3d;

/**
 * Class that represents a plane in hesse form.
 * This type of plane allows efficient calculation of orthogonal
 * distances. The normal is expected to have unit length.
 */
template <class T>
class Plane3
{
public:
    typedef Vector<T, 3> Vec3T;

public:
    /** Creates an uninitialized plane. */
    Plane3 (void);

    /** Creates a plane with normal n and distance d from the origin. */
    Plane3 (Vec3T const& n, T const& d);

    /** Creates a plane containing p with normal n. */
    Plane3 (Vec3T const& n, Vec3T const& p);

    /** Creates the plane from three points. */
    Plane3 (Vec3T const& p1, Vec3T const& p2, Vec3T const& p3);

    /** Returns the signed distance from a point to the plane. */
    T point_dist (Vec3T const& p) const;

    /** Flips the orientation of the plane. */
    Plane3<T>& invert (void);

    /** Returns plane with flipped orientation. */
    Plane3<T> inverted (void) const;

public:
    Vec3T n;
    T d;
};

/* ---------------------------------------------------------------- */

template <class T>
inline
Plane3<T>::Plane3 (void)
{
}

template <class T>
inline
Plane3<T>::Plane3 (Vec3T const& n, T const& d)
    : n(n), d(d)
{
}

template <class T>
inline
Plane3<T>::Plane3 (Vec3T const& n, Vec3T const& p)
    : n(n), d(p.dot(n))
{
}

template <class T>
inline
Plane3<T>::Plane3 (Vec3T const& p1, Vec3T const& p2, Vec3T const& p3)
{
    n = (p2 - p1).cross(p3 - p1).normalize();
    d = p1.dot(n);
}

template <class T>
inline T
Plane3<T>::point_dist (Vec3T const& p) const
{
    return p.dot(n) - d;
}

template <class T>
Plane3<T>&
Plane3<T>::invert (void)
{
    n = -n;
    d = -d;
    return *this;
}

template <class T>
inline Plane3<T>
Plane3<T>::inverted (void) const
{
    return Plane3<T>(-n, -d);
}

MATH_NAMESPACE_END

#endif /* MATH_PLANE_HEADER */
