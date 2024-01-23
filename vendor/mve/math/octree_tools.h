/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_OCTREETOOLS_HEADER
#define MATH_OCTREETOOLS_HEADER

#include <limits>

#include "math/defines.h"
#include "math/vector.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"
#include "math/functions.h"

MATH_NAMESPACE_BEGIN
MATH_GEOM_NAMESPACE_BEGIN

/**
 * Returns true if the given triangle and box overlap, false otherwise.
 * The triangle is given by vertices 'a', 'b' and 'c' and the
 * axis-aligned box is given by box center 'boxcenter' and box
 * half sizes aligned with the axis 'halfsize'.
 */
template <typename T>
bool
triangle_box_overlap (math::Vector<T,3> const& boxcenter,
    math::Vector<T,3> const& boxhalfsize, math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c);

/**
 * Returns true if the given plane (in Hesse normal form) and a box
 * in the origin given by 'boxhalfsizes' overlap, false otherwise.
 */
template <typename T>
bool
plane_box_overlap (math::Vector<T,3> const& normal,
    math::Vector<T,3> const& pos,
    math::Vector<T,3> const& boxhalfsize);

/**
 * Returns true if the given ray intersects with the given
 * axis-aligned bounding box, false otherwise.
 */
template <typename T>
bool
ray_box_overlap (math::Vector<T,3> const& origin, math::Vector<T,3> const& dir,
    math::Vector<T,3> const& box_min, math::Vector<T,3> const& box_max);

/**
 * Intersects the given ray with the given triangle and returns the 't'
 * parameter of the intersection point wrt the ray. If the ray does
 * not hit the triangle, 't'=0 is returned. The algorithm optionally
 * returns the barycentric coordinates in 'bary' if 'bary' is not null.
 */
template <typename T>
T
ray_triangle_intersect (math::Vector<T,3> const& origin,
    math::Vector<T,3> const& dir, math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    float* bary = 0);

/**
 * Returns true if the given boxes overlap, false otherwise.
 * Defined for boxes in arbitrary dimension.
 */
template <typename T, int N>
bool
box_box_overlap (math::Vector<T,N> const& b1min, math::Vector<T,N> const& b1max,
    math::Vector<T,N> const& b2min, math::Vector<T,N> const& b2max);

/**
 * Intersects the given rays with each other. It returns t1 and t2
 * with the minimal distance between (t1*d1+p1) and (t2*d2+p2).
 */
template <typename T>
math::Vector<T,2>
ray_ray_intersect (math::Vector<T,3> const& p1, math::Vector<T,3> const& d1,
    math::Vector<T,3> const& p2, math::Vector<T,3> const& d2);

/**
 * Returns true if the given point overlaps with the axis-aligned box.
 * Defined for points and boxes in arbitrary dimension.
 */
template <typename T, int N>
bool
point_box_overlap (math::Vector<T, N> const& point,
    math::Vector<T, N> const& aabb_min, math::Vector<T, N> const& aabb_max);

/* ------------------------ Implementation ------------------------ */

#define AXISTEST_X01(a, b, fa, fb) { \
    T p0 = a * v[0][1] - b * v[0][2]; \
    T p2 = a * v[2][1] - b * v[2][2]; \
    T min = std::min(p0, p2); T max = std::max(p0, p2); \
    T rad = fa * boxhalfsize[1] + fb * boxhalfsize[2]; \
    if (min > rad || max < -rad) return false; }

#define AXISTEST_X2(a, b, fa, fb) { \
    T p0 = a * v[0][1] - b * v[0][2]; \
    T p1 = a * v[1][1] - b * v[1][2]; \
    T min = std::min(p0, p1); T max = std::max(p0, p1); \
    T rad = fa * boxhalfsize[1] + fb * boxhalfsize[2]; \
    if (min > rad || max < -rad) return false; }

#define AXISTEST_Y02(a, b, fa, fb) { \
    T p0 = -a * v[0][0] + b * v[0][2]; \
    T p2 = -a * v[2][0] + b * v[2][2]; \
    T min = std::min(p0, p2); T max = std::max(p0, p2); \
    T rad = fa * boxhalfsize[0] + fb * boxhalfsize[2]; \
    if (min > rad || max < -rad) return false; }

#define AXISTEST_Y1(a, b, fa, fb) { \
    T p0 = -a * v[0][0] + b * v[0][2]; \
    T p1 = -a * v[1][0] + b * v[1][2]; \
    T min = std::min(p0, p1); T max = std::max(p0, p1); \
    T rad = fa * boxhalfsize[0] + fb * boxhalfsize[2]; \
    if (min > rad || max < -rad) return false; }

#define AXISTEST_Z12(a, b, fa, fb) { \
    T p1 = a * v[1][0] - b * v[1][1]; \
    T p2 = a * v[2][0] - b * v[2][1]; \
    T min = std::min(p1, p2); T max = std::max(p1, p2); \
    T rad = fa * boxhalfsize[0] + fb * boxhalfsize[1]; \
    if (min > rad || max < -rad) return false; }

#define AXISTEST_Z0(a, b, fa, fb) { \
    T p0 = a * v[0][0] - b * v[0][1]; \
    T p1 = a * v[1][0] - b * v[1][1]; \
    T min = std::min(p0, p1); T max = std::max(p0, p1); \
    T rad = fa * boxhalfsize[0] + fb * boxhalfsize[1]; \
    if (min > rad || max < -rad) return false; }

template <typename T>
bool
triangle_box_overlap (math::Vector<T,3> const& boxcenter,
    math::Vector<T,3> const& boxhalfsize, math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c)
{
    math::Vector<T,3> v[3] = { a - boxcenter, b - boxcenter, c - boxcenter };
    math::Vector<T,3> e[3] = { v[1] - v[0], v[2] - v[1], v[0] - v[2] };

    {
        T abs[3] = { std::abs(e[0][0]), std::abs(e[0][1]), std::abs(e[0][2]) };
        AXISTEST_X01(e[0][2], e[0][1], abs[2], abs[1])
        AXISTEST_Y02(e[0][2], e[0][0], abs[2], abs[0])
        AXISTEST_Z12(e[0][1], e[0][0], abs[1], abs[0])
    }

    {
        T abs[3] = { std::abs(e[1][0]), std::abs(e[1][1]), std::abs(e[1][2]) };
        AXISTEST_X01(e[1][2], e[1][1], abs[2], abs[1])
        AXISTEST_Y02(e[1][2], e[1][0], abs[2], abs[0])
        AXISTEST_Z0 (e[1][1], e[1][0], abs[1], abs[0])
    }

    {
        T abs[3] = { std::abs(e[2][0]), std::abs(e[2][1]), std::abs(e[2][2]) };
        AXISTEST_X2 (e[2][2], e[2][1], abs[2], abs[1])
        AXISTEST_Y1 (e[2][2], e[2][0], abs[2], abs[0])
        AXISTEST_Z12(e[2][1], e[2][0], abs[1], abs[0])
    }

    for (int i = 0; i < 3; ++i)
    {
        T min = math::min(v[0][i], v[1][i], v[2][i]);
        T max = math::max(v[0][i], v[1][i], v[2][i]);
        if (min > boxhalfsize[i] || max < -boxhalfsize[i])
            return false;
    }

    math::Vector<T,3> normal(e[0].cross(e[1])); // Not normalized
    if (!plane_box_overlap(normal, v[0], boxhalfsize))
        return false;

    return true;
}

/* ---------------------------------------------------------------- */

template <typename T>
bool
plane_box_overlap (math::Vector<T,3> const& normal,
    math::Vector<T,3> const& pos,
    math::Vector<T,3> const& boxhalfsize)
{
    math::Vector<T,3> vmin, vmax;
    for (int q = 0; q < 3; ++q)
    {
        if (normal[q] > (T)0)
        {
            vmin[q] = -boxhalfsize[q] - pos[q];
            vmax[q] = boxhalfsize[q] - pos[q];
        }
        else
        {
            vmin[q] = boxhalfsize[q] - pos[q];
            vmax[q] = -boxhalfsize[q] - pos[q];
        }
    }

    if (normal.dot(vmin) > (T)0)
        return false;
    if (normal.dot(vmax) >= (T)0)
        return true;

    return false;
}

/* ---------------------------------------------------------------- */

/*
 * Ray-box intersection using IEEE numerical properties to ensure that the
 * test is both robust and efficient, as described in:
 *
 *      Amy Williams, Steve Barrus, R. Keith Morley, and Peter Shirley
 *      "An Efficient and Robust Ray-Box Intersection Algorithm"
 *      Journal of graphics tools, 10(1):49-54, 2005
 *
 */

template <typename T>
bool
ray_box_overlap (math::Vector<T,3> const& origin, math::Vector<T,3> const& dir,
    math::Vector<T,3> const& box_min, math::Vector<T,3> const& box_max)
{
    math::Vector<T,3> box[2] = { box_min, box_max };
    T idir[3] = { (T)1 / dir[0], (T)1 / dir[1], (T)1 / dir[2] };
    bool sign[3] = { idir[0] < (T)0, idir[1] < (T)0, idir[2] < (T)0 };

    float tmin = (box[sign[0]][0] - origin[0]) * idir[0];
    float tmax = (box[1 - sign[0]][0] - origin[0]) * idir[0];
    float tymin = (box[sign[1]][1] - origin[1]) * idir[1];
    float tymax = (box[1 - sign[1]][1] - origin[1]) * idir[1];

    if (tmin > tymax || tymin > tmax)
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (box[sign[2]][2] - origin[2]) * idir[2];
    float tzmax = (box[1 - sign[2]][2] - origin[2]) * idir[2];

    if (tmin > tzmax || tzmin > tmax)
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;

    T const t0 = (T)0;
    T const t1 = (T)1/(T)0;
    return tmin < t1 && tmax > t0;
}

/* ---------------------------------------------------------------- */

template <typename T>
T
ray_triangle_intersect (math::Vector<T,3> const& origin,
    math::Vector<T,3> const& dir, math::Vector<T,3> const& a,
    math::Vector<T,3> const& b, math::Vector<T,3> const& c,
    float* bary)
{
    /* find vectors for two edges sharing vert0 */
    math::Vector<T,3> edge1 = b - a;
    math::Vector<T,3> edge2 = c - a;

    /* begin calculating determinant - also used to calculate U parameter */
    math::Vector<T,3> pvec = dir.cross(edge2);

    /* if determinant is near zero, ray lies in plane of triangle */
    T det = edge1.dot(pvec);
    if (MATH_FLOAT_EQ(det, T(0)))
        return T(0);
    T inv_det = T(1) / det;

    /* calculate distance from vert0 to ray origin */
    math::Vector<T,3> tvec = origin - a;

    /* calculate U parameter and test bounds */
    T u = tvec.dot(pvec) * inv_det;
    if (u < T(0) || u > T(1))
        return (T)0;

    /* prepare to test V parameter */
    math::Vector<T,3> qvec = tvec.cross(edge1);

    /* calculate V parameter and test bounds */
    T v = dir.dot(qvec) * inv_det;
    if (v < T(0) || u + v > T(1))
        return T(0);

    /* calculate t, ray intersects triangle */
    T t = edge2.dot(qvec) * inv_det;

    /* Since "0" is used to indicate failure, return minimum value. */
    if (t == T(0))
        return std::numeric_limits<T>::min();

    if (bary != 0)
    {
        bary[0] = u;
        bary[1] = v;
    }

    return t;
}

/* ---------------------------------------------------------------- */

template <typename T, int N>
bool
box_box_overlap (math::Vector<T,N> const& b1min, math::Vector<T,N> const& b1max,
    math::Vector<T,N> const& b2min, math::Vector<T,N> const& b2max)
{
    for (int i = 0; i < N; ++i)
        if (b1min[i] > b2max[i] || b1max[i] < b2min[i])
            return false;
    return true;
}

/* ---------------------------------------------------------------- */

template <typename T>
math::Vector<T,2>
ray_ray_intersect (math::Vector<T,3> const& p1, math::Vector<T,3> const& d1,
    math::Vector<T,3> const& p2, math::Vector<T,3> const& d2)
{
    math::Vector<T,3> dx = d1.cross(d2);
    math::Matrix<T,3,3> m1;
    m1[0] = p2[0] - p1[0]; m1[1] = d2[0]; m1[2] = dx[0];
    m1[3] = p2[1] - p1[1]; m1[4] = d2[1]; m1[5] = dx[1];
    m1[6] = p2[2] - p1[2]; m1[7] = d2[2]; m1[8] = dx[2];

    math::Matrix<T,3,3> m2;
    m2[0] = p2[0] - p1[0]; m2[1] = d1[0]; m2[2] = dx[0];
    m2[3] = p2[1] - p1[1]; m2[4] = d1[1]; m2[5] = dx[1];
    m2[6] = p2[2] - p1[2]; m2[7] = d1[2]; m2[8] = dx[2];

    math::Vector<T,2> ret(matrix_determinant(m1) / dx.square_norm(),
        matrix_determinant(m2) / dx.square_norm());
    return ret;
}

/* ---------------------------------------------------------------- */

template <typename T, int N>
bool
point_box_overlap (math::Vector<T, N> const& point,
    math::Vector<T, N> const& aabb_min, math::Vector<T, N> const& aabb_max)
{
    for (int i = 0; i < N; ++i)
        if (point[i] < aabb_min[i] || point[i] > aabb_max[i])
            return false;
    return true;
}

MATH_GEOM_NAMESPACE_END
MATH_NAMESPACE_END

#endif /* MATH_OCTREETOOLS_HEADER */
