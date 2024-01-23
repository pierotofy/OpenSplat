/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_BSPLINE_HEADER
#define MATH_BSPLINE_HEADER

#include <vector>

#include "math/defines.h"
#include "math/matrix.h"

MATH_NAMESPACE_BEGIN

/**
 * Implementation of non-uniform B-Spline curves according to
 *
 *     http://en.wikipedia.org/wiki/B-spline
 *
 * A B-Spline of degree n is defined by at least n+1 control points.
 * Let p be the number of control points, the spline is composed of p-n
 * segments. The knot vector contains m = p+n+1 real values and defines
 * the length of the segments with respect to t. The first and last n
 * values in the knot vector are "extra knots" that specify Bezier end
 * conditions. The remaining p-n+1 knot values define the segment lengths.
 *
 * The spline is evaluated in a simple manner using the de Boor algorithm.
 * The current implementation is inefficient for large p.
 */
template <class V, class T = float>
class BSpline
{
public:
    typedef std::vector<T> KnotVector;
    typedef std::vector<V> PointVector;

public:
    BSpline (void);

    /** Returns whether there are no points in this spline. */
    bool empty (void) const;

    /** Sets the degree of spline segments. */
    void set_degree (int degree);
    /** Returns the degree of the spline segments. */
    int get_degree (void) const;

    /** Reserves space for points and the knot vector. */
    void reserve (std::size_t n_points);

    /** Adds a point to the control point vector. */
    void add_point (V const& p);
    /** Adds a knot to the knot vector. */
    void add_knot (T const& t);
    /** Initializes the knot vector to be uniform. */
    void uniform_knots (T const& min, T const& max);
    /** Scales the knots such that evaluation is valid in [min, max]. */
    void scale_knots (T const& min, T const& max);

    /** Returns the point vector. */
    PointVector const& get_points (void) const;
    /** Returns the knot vector. */
    KnotVector const& get_knots (void) const;

    /** Evalutes the B-Spline. */
    V evaluate (T const& t) const;

    /** Transforms the B-Spline. */
    void transform (math::Matrix4f const& transf);

private:
    /** De Boor algorithm to evaluate the basis polynomials. */
    T deboor (int i, int k, T const& x) const;

private:
    int n; ///< the degree of the polynom
    KnotVector knots; ///< knot vector with m entries
    PointVector points; ///< m-n-1 control points
};

MATH_NAMESPACE_END

/* ----------------------- Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <class V, class T>
inline
BSpline<V,T>::BSpline (void)
    : n(3)
{
}

template <class V, class T>
bool
BSpline<V,T>::empty (void) const
{
    return this->points.empty();
}

template <class V, class T>
inline void
BSpline<V,T>::set_degree (int degree)
{
    this->n = degree;
}

template <class V, class T>
inline int
BSpline<V,T>::get_degree (void) const
{
    return this->n;
}

template <class V, class T>
inline void
BSpline<V,T>::reserve (std::size_t n_points)
{
    this->points.reserve(n_points);
    this->knots.reserve(n_points + n + 1);
}

template <class V, class T>
inline void
BSpline<V,T>::add_point (V const& p)
{
    this->points.push_back(p);
}

template <class V, class T>
inline void
BSpline<V,T>::add_knot (T const& t)
{
    this->knots.push_back(t);
}

template <class V, class T>
inline void
BSpline<V,T>::uniform_knots (T const& min, T const& max)
{
    T width = max - min;
    int n_knots = this->points.size() + this->n + 1;
    int segments = this->points.size() - this->n;
    this->knots.clear();
    this->knots.reserve(n_knots);

    for (int i = 0; i < this->n; ++i)
        this->knots.push_back(min);
    for (int i = 0; i < segments + 1; ++i)
        this->knots.push_back(min + T(i) * width / T(segments));
    for (int i = 0; i < this->n - 1; ++i)
        this->knots.push_back(max);
    if (this->n > 0)
        this->knots.push_back(max + T(1));

#if 0
    std::cout << "Made knots: ";
    for (std::size_t i = 0; i < this->knots.size(); ++i)
        std::cout << this->knots[i] << " ";
    std::cout << std::endl;
#endif
}


template <class V, class T>
inline void
BSpline<V,T>::scale_knots (T const& min, T const& max)
{
    T first = this->knots[this->n];
    T const& last = this->knots[this->knots.size() - this->n - 1];
    T scale = (max - min) / (last - first);
    for (std::size_t i = 0; i < this->knots.size(); ++i)
        this->knots[i] = (this->knots[i] - first) * scale;
}

template <class V, class T>
inline typename BSpline<V,T>::PointVector const&
BSpline<V,T>::get_points (void) const
{
    return this->points;
}

template <class V, class T>
inline typename BSpline<V,T>::KnotVector const&
BSpline<V,T>::get_knots (void) const
{
    return this->knots;
}

template <class V, class T>
inline V
BSpline<V,T>::evaluate (T const& t) const
{
    /* FIXME inefficient */
    V p = this->points[0] * this->deboor(0, this->n, t);
    for (std::size_t i = 1; i < this->points.size(); ++i)
        p += this->points[i] * this->deboor(i, this->n, t);
    return p;
}

template <class V, class T>
inline T
BSpline<V,T>::deboor (int i, int k, T const& x) const
{
    if (k == 0)
    {
        //FIXME
        return (x >= this->knots[i] && x < this->knots[i+1]) ? T(1) : T(0);
        //return (x == this->knots[i] || (x > this->knots[i] && x < this->knots[i+1])) ? T(1) : T(0);
    }
    float d1 = this->knots[i+k] - this->knots[i];
    float d2 = this->knots[i+k+1] - this->knots[i+1];
    T v1 = d1 > T(0) ? (x - this->knots[i]) / d1 : T(0);
    T v2 = d2 > T(0) ? (this->knots[i+k+1] - x) / d2 : T(0);
    return v1 * deboor(i, k-1, x) + v2 * deboor(i+1, k-1, x);
}

template <class V, class T>
inline void
BSpline<V,T>::transform (math::Matrix4f const& transf)
{
    for (std::size_t i = 0; i < points.size(); ++i)
        points[i] = transf.mult(points[i], 1.0f);
}

MATH_NAMESPACE_END

#endif /* MATH_BSPLINE_HEADER */
