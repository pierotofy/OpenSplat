/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_GEOMETRY_HEADER
#define MATH_GEOMETRY_HEADER

#include "math/defines.h"
#include "math/vector.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"

MATH_NAMESPACE_BEGIN
MATH_GEOM_NAMESPACE_BEGIN

/**
 * Returns the center of the circumsphere defined by the four given vertices.
 * The vertices are expected to be in general position.
 * http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
 */
template <typename T>
math::Vector<T,3>
circumsphere_center (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d);

/**
 * Returns the circumsphere radius of the sphere defined by a, b, c, d.
 * Involves one square root.
 */
template <typename T>
float
circumsphere_radius (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d);

/**
 * Tests whether vertex 'p' is contained in the circumsphere defined
 * by the four vertices a,b,c,d. The function returns a positive value
 * if the point is contained in the sphere, and a negative value otherwise.
 * The returned valued is actually the different between the squared
 * radius of the sphere minus the squared distance of the point to the
 * circumsphere center.
 * http://www.cgafaq.info/wiki/Tetrahedron_Circumsphere
 */
template <typename T>
float
circumsphere_test (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d, math::Vector<T,3> const& p);

/**
 * Returns the insphere radius of the tetrahedron defined by a, b, c, d.
 * Involves four square roots.
 */
template <typename T>
float
insphere_radius (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d);

/**
 * Calculates the area of the given triangle. Depending on the
 * orientation of the triangle, this area may be negative.
 */
template <typename T>
T
triangle_area (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c);

/**
 * Calculates the volume of the given tetraheron. Depending on the
 * orientation of the tetrahedron, this volume may be negative.
 */
template <typename T>
T
tetrahedron_volume (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d);

/**
 * Calculates the orientation of the given tetrahedron. The orientation
 * is given by the sign of the returned value. If the returned value is
 * near zero or zero, the tetrahedron is degenerated.
 */
template <typename T>
T
tetrahedron_orientation (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d);

/**
 * Calculates the barycentric coordinates of point 'p' with respect
 * to the tetrahedron given by vertices a,b,c,d.
 */
template <typename T>
math::Vector<T,3>
tetrahedron_bary (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d, math::Vector<T,3> const& p);

/**
 * Tests whether four points are coplanar. Involves two square roots!
 * This is done by calculating a "normalized" version of the triple product
 * of three edges such that the resulting value is the cosine of an angle,
 * which is zero if the points are coplanar.
 */
template <typename T>
bool
points_coplanar (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d, T const& cos_angle);

MATH_GEOM_NAMESPACE_END
MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN
MATH_GEOM_NAMESPACE_BEGIN

template <typename T>
math::Vector<T,3>
circumsphere_center (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d)
{
    math::Vector<T,3> ba(b - a);
    math::Vector<T,3> ca(c - a);
    math::Vector<T,3> da(d - a);
    math::Vector<T,3> x1(ba.cross(ca));
    math::Vector<T,3> x2(da.cross(ba));
    math::Vector<T,3> x3(ca.cross(da));

    math::Vector<T,3> ret = x1 * da.square_norm() + x2 * ca.square_norm()
        + x3 * ba.square_norm();
    ret = a + ret / (ba.dot(x3) * T(2));

    return ret;
}

/* ---------------------------------------------------------------- */

template <typename T>
float
circumsphere_radius (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d)
{
    math::Vector<T,3> ba(b - a);
    math::Vector<T,3> ca(c - a);
    math::Vector<T,3> da(d - a);
    T vol6 = std::abs(ba.dot(ca.cross(da)));
    return (ba.dot(ba) * ca.cross(da) + ca.dot(ca) * da.cross(ba)
        + da.dot(da) * ba.cross(ca)).norm() / (vol6 * T(2));
}

/* ---------------------------------------------------------------- */

template <typename T>
float
insphere_radius (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d)
{
    math::Vector<T,3> va(b - a);
    math::Vector<T,3> vb(c - a);
    math::Vector<T,3> vc(d - a);
    T vol6 = std::abs(va.dot(vb.cross(vc)));
    return vol6 / (vb.cross(vc).norm() + vc.cross(va).norm()
        + va.cross(vb).norm()
        + (vb.cross(vc) + vc.cross(va) + va.cross(vb)).norm());
}

/* ---------------------------------------------------------------- */

template <typename T>
float
circumsphere_test (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d, math::Vector<T,3> const& p)
{
    math::Vector<T,3> ba(b - a);
    math::Vector<T,3> ca(c - a);
    math::Vector<T,3> da(d - a);
    math::Vector<T,3> x1(ba.cross(ca));
    math::Vector<T,3> x2(da.cross(ba));
    math::Vector<T,3> x3(ca.cross(da));

    math::Vector<T,3> num(x1 * da.square_norm() + x2 * ca.square_norm()
        + x3 * ba.square_norm());
    T denom = ba.dot(x3) * T(2);

    T square_radius = num.square_norm() / (denom * denom);
    T point_sdist = (p - (a + num / denom)).square_norm();
    return square_radius - point_sdist;
}

/* ---------------------------------------------------------------- */

template <typename T>
T
triangle_area (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c)
{
    return (b - a).cross(c - a).norm() / T(2);
}

/* ---------------------------------------------------------------- */

template <typename T>
inline T
tetrahedron_volume (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d)
{
#if 1
    /* Efficient calculation using determinat identity from
     * http://mathworld.wolfram.com/DeterminantIdentities.html
     */
    return (c - a).dot((b - a).cross(d - c)) / T(6);
#else
    /* Calculation using generic matrix determinant. */
    math::Vector<T,3> v[3] = { a - d, b - d, c - d };
    math::Matrix<T,3,3> m;
    for (int i = 0; i < 9; ++i)
        m[i] = v[i % 3][i / 3];
    return matrix_determinant(m) / T(6);
#endif
}

/* ---------------------------------------------------------------- */

template <typename T>
inline T
tetrahedron_orientation (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d)
{
    return (c - a).dot((b - a).cross(d - c));
}

/* ---------------------------------------------------------------- */

template <typename T>
math::Vector<T,3>
tetrahedron_bary (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d, math::Vector<T,3> const& p)
{
#if 0
    /* Calculation using the areas of sub-tetrahedra. */
    /* This version appears to be slower. */
    T tet = tetrahedron_volume(a, b, c, d);
    return math::Vector<T,3>(tetrahedron_volume(p, b, c, d) / tet,
        tetrahedron_volume(p, a, d, c) / tet,
        tetrahedron_volume(p, a, b, d) / tet);
#else
    /* Calculation using inverse M as in x = M*b <=> b = (M^-1)*x. */
    math::Vector<T,3> v[3] = { a - d, b - d, c - d };
    math::Matrix<T,3,3> m;
    for (int i = 0; i < 9; ++i)
        m[i] = v[i % 3][i / 3];
    return matrix_inverse(m) * (p - d);
#endif
}

/* ---------------------------------------------------------------- */

template <typename T>
bool
points_coplanar (math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    math::Vector<T,3> const& d, T const& cos_angle)
{
    /*
     * http://www.gamedev.net/community/forums/topic.asp?topic_id=408739
     *
     * To test if four points P0, P1, P2, P3 are coplanar, you can
     * create three vectors: V0 = P1-P0, V1 = P2-P0, V2 = P3-P0.
     * Then compute the triple scalar product: Dot(Cross(V0,V1), V2).
     * If the result is near 0, then they might be coplanar. However,
     * they could be collinear. To check for that, do Cross(V0,V1) and
     * Cross(V0,V2). If both results are near zero, then they're collinear.
     */
    math::Vector<T,3> e1(b - a);
    math::Vector<T,3> e2(c - a);
    math::Vector<T,3> e3(d - a);
    T x = std::abs(e1.normalized().dot(e2.cross(e3).normalized()));
    return x <= cos_angle;
}

MATH_GEOM_NAMESPACE_END
MATH_NAMESPACE_END

#endif /* MATH_GEOMETRY_HEADER */
