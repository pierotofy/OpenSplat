/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_VECTOR_HEADER
#define MATH_VECTOR_HEADER

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <ostream>

#include "math/defines.h"
#include "math/algo.h"

MATH_NAMESPACE_BEGIN

/*
 * Vector type definitions for convenience.
 */
template <typename T, int N> class Vector;
typedef Vector<float,1> Vec1f;
typedef Vector<float,2> Vec2f;
typedef Vector<float,3> Vec3f;
typedef Vector<float,4> Vec4f;
typedef Vector<float,5> Vec5f;
typedef Vector<float,6> Vec6f;
typedef Vector<float,64> Vec64f;
typedef Vector<float,128> Vec128f;
typedef Vector<double,1> Vec1d;
typedef Vector<double,2> Vec2d;
typedef Vector<double,3> Vec3d;
typedef Vector<double,4> Vec4d;
typedef Vector<double,5> Vec5d;
typedef Vector<double,6> Vec6d;
typedef Vector<int,1> Vec1i;
typedef Vector<int,2> Vec2i;
typedef Vector<int,3> Vec3i;
typedef Vector<int,4> Vec4i;
typedef Vector<int,5> Vec5i;
typedef Vector<int,6> Vec6i;
typedef Vector<unsigned int,1> Vec1ui;
typedef Vector<unsigned int,2> Vec2ui;
typedef Vector<unsigned int,3> Vec3ui;
typedef Vector<unsigned int,4> Vec4ui;
typedef Vector<unsigned int,5> Vec5ui;
typedef Vector<unsigned int,6> Vec6ui;
typedef Vector<char,1> Vec1c;
typedef Vector<char,2> Vec2c;
typedef Vector<char,3> Vec3c;
typedef Vector<char,4> Vec4c;
typedef Vector<char,5> Vec5c;
typedef Vector<char,6> Vec6c;
typedef Vector<unsigned char,1> Vec1uc;
typedef Vector<unsigned char,2> Vec2uc;
typedef Vector<unsigned char,3> Vec3uc;
typedef Vector<unsigned char,4> Vec4uc;
typedef Vector<unsigned char,5> Vec5uc;
typedef Vector<unsigned char,6> Vec6uc;
typedef Vector<short,64> Vec64s;
typedef Vector<unsigned short,128> Vec128us;
typedef Vector<std::size_t,1> Vec1st;
typedef Vector<std::size_t,2> Vec2st;
typedef Vector<std::size_t,3> Vec3st;
typedef Vector<std::size_t,4> Vec4st;
typedef Vector<std::size_t,5> Vec5st;
typedef Vector<std::size_t,6> Vec6st;
typedef Vector<unsigned short,1> Vec1us;
typedef Vector<unsigned short,2> Vec2us;
typedef Vector<unsigned short,3> Vec3us;
typedef Vector<unsigned short,4> Vec4us;
typedef Vector<unsigned short,5> Vec5us;
typedef Vector<unsigned short,6> Vec6us;

/**
 * Vector class for arbitrary dimensions and types.
 */
template <typename T, int N>
class Vector
{
public:
    typedef T ValueType;

    static int constexpr dim = N;

    /* ------------------------ Constructors ---------------------- */

    /** Default ctor. */
    Vector (void);
    /** Ctor taking a pointer to initialize values. */
    explicit Vector (T const* values);
    /** Ctor that initializes ALL elements. */
    explicit Vector (T const& value);
    /** Ctor that initializes the first two elements. */
    Vector (T const& v1, T const& v2);
    /** Ctor that initializes the first three elements. */
    Vector (T const& v1, T const& v2, T const& v3);
    /** Ctor that initializes the first four elements. */
    Vector (T const& v1, T const& v2, T const& v3, T const& v4);

    /** Ctor that takes a smaller vector and one element. */
    Vector (Vector<T,N-1> const& other, T const& v1);

    /** Copy ctor from vector of same type. */
    Vector (Vector<T,N> const& other);
    /** Copy ctor from vector of different type. */
    template <typename O>
    Vector (Vector<O,N> const& other);

    /** Ctor taking a pointer from different type to initialize values. */
    template <typename O>
    explicit Vector (O const* values);

    /* ------------------------- Management ----------------------- */

    /** Fills all vector elements with the given value. */
    Vector<T,N>& fill (T const& value);

    /** Copies values from the given pointer. */
    Vector<T,N>& copy (T const* values, int num = N);

    /** Returns the smallest element in the vector. */
    T minimum (void) const;

    /** Returns the largest element in the vector. */
    T maximum (void) const;

    /** Returns the sum of all elements. */
    T sum (void) const;

    /** Returns the sum of the absolute values of all elements. */
    T abs_sum (void) const;

    /** Returns the product of all elements. */
    T product (void) const;

    // TODO: central projection (4D -> 3D)

    /* ---------------------- Unary operators --------------------- */

    /** Computes the norm (length) of the vector. */
    T norm (void) const;
    /** Computes the squared norm of the vector (much cheaper). */
    T square_norm (void) const;

    /** Normalizes self and returns reference to self. */
    Vector<T,N>& normalize (void);
    /** Returns a normalized copy of self. */
    Vector<T,N> normalized (void) const;

    /** Component-wise absolute-value on self, returns self. */
    Vector<T,N>& abs_value (void);
    /** Returns a component-wise absolute-value copy of self. */
    Vector<T,N> abs_valued (void) const;

    /** Component-wise negation on self, returns self. */
    Vector<T,N>& negate (void);
    /** Returns a component-wise negation on copy of self. */
    Vector<T,N> negated (void) const;

    /** Sorts the elements of the vector into ascending order. */
    Vector<T,N>& sort_asc (void);
    /** Sorts the elements of the vector into descending order. */
    Vector<T,N>& sort_desc (void);
    /** Returns a sorted vector into ascending order. */
    Vector<T,N> sorted_asc (void) const;
    /** Returns a sorted vector into descending order. */
    Vector<T,N> sorted_desc (void) const;

    /** Applies a for-each functor to all values. */
    template <typename F>
    Vector<T,N>& apply_for_each (F functor);
    /** Applies a for-each functor to a copy of the vector. */
    template <typename F>
    Vector<T,N> applied_for_each (F functor) const;

    // TODO: component-wise max, min, mean?

    /* --------------------- Binary operators --------------------- */

    /** Dot (or scalar) product between self and another vector. */
    T dot (Vector<T,N> const& other) const;

    /** Cross product between this and another vector. */
    Vector<T,N> cross (Vector<T,N> const& other) const;

    /** Component-wise multiplication with another vector. */
    Vector<T,N> cw_mult (Vector<T,N> const& other) const;

    /** Component-wise division with another vector. */
    Vector<T,N> cw_div (Vector<T,N> const& other) const;

    /** Component-wise similarity using epsilon checks. */
    bool is_similar (Vector<T,N> const& other, T const& epsilon) const;

    /* --------------------- Value iterators ---------------------- */

    T* begin (void);
    T const* begin (void) const;
    T* end (void);
    T const* end (void) const;

    /* --------------------- Object operators --------------------- */

    /** Dereference operator to value array. */
    T* operator* (void);
    /** Const dereference operator to value array. */
    T const* operator* (void) const;

    /** Element access operator. */
    T& operator[] (int index);
    /** Const element access operator. */
    T const& operator[] (int index) const;

    /** Element access operator. */
    T& operator() (int index);
    /** Const element access operator. */
    T const& operator() (int index) const;

    /** Comparison operator. */
    bool operator== (Vector<T,N> const& rhs) const;
    /** Comparison operator. */
    bool operator!= (Vector<T,N> const& rhs) const;

    /** Assignment operator. */
    Vector<T,N>& operator= (Vector<T,N> const& rhs);
    /** Assignment operator from different type. */
    template <typename O>
    Vector<T,N>& operator= (Vector<O,N> const& rhs);

    /** Component-wise negation. */
    Vector<T,N> operator- (void) const;

    /** Self-substraction with other vector. */
    Vector<T,N>& operator-= (Vector<T,N> const& rhs);
    /** Substraction with other vector. */
    Vector<T,N> operator- (Vector<T,N> const& rhs) const;
    /** Self-addition with other vector. */
    Vector<T,N>& operator+= (Vector<T,N> const& rhs);
    /** Addition with other vector. */
    Vector<T,N> operator+ (Vector<T,N> const& rhs) const;

    /** Component-wise self-substraction with scalar. */
    Vector<T,N>& operator-= (T const& rhs);
    /** Component-wise substraction with scalar. */
    Vector<T,N> operator- (T const& rhs) const;
    /** Component-wise self-addition with scalar. */
    Vector<T,N>& operator+= (T const& rhs);
    /** Component-wise addition with scalar. */
    Vector<T,N> operator+ (T const& rhs) const;
    /** Component-wise self-multiplication with scalar. */
    Vector<T,N>& operator*= (T const& rhs);
    /** Component-wise multiplication with scalar. */
    Vector<T,N> operator* (T const& rhs) const;
    /** Component-wise self-division by scalar. */
    Vector<T,N>& operator/= (T const& rhs);
    /** Component-wise division by scalar. */
    Vector<T,N> operator/ (T const& rhs) const;

protected:
    T v[N];
};

MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

MATH_NAMESPACE_BEGIN

template <typename T, int N>
int constexpr Vector<T,N>::dim;

template <typename T, int N>
inline
Vector<T,N>::Vector (void)
{
}

template <typename T, int N>
inline
Vector<T,N>::Vector (T const* values)
{
    std::copy(values, values + N, v);
}

template <typename T, int N>
inline
Vector<T,N>::Vector (T const& value)
{
    fill(value);
}

template <typename T, int N>
inline
Vector<T,N>::Vector (T const& v1, T const& v2)
{
    v[0] = v1; v[1] = v2;
}

template <typename T, int N>
inline
Vector<T,N>::Vector (T const& v1, T const& v2, T const& v3)
{
    v[0] = v1; v[1] = v2; v[2] = v3;
}

template <typename T, int N>
inline
Vector<T,N>::Vector (T const& v1, T const& v2, T const& v3, T const& v4)
{
    v[0] = v1; v[1] = v2; v[2] = v3; v[3] = v4;
}

template <typename T, int N>
inline
Vector<T,N>::Vector (Vector<T,N-1> const& other, T const& v1)
{
    std::copy(*other, *other + N-1, v);
    v[N-1] = v1;
}

template <typename T, int N>
inline
Vector<T,N>::Vector (Vector<T,N> const& other)
{
    std::copy(*other, *other + N, v);
}

template <typename T, int N>
template <typename O>
inline
Vector<T,N>::Vector (Vector<O,N> const& other)
{
    std::copy(*other, *other + N, v);
}

template <typename T, int N>
template <typename O>
inline
Vector<T,N>::Vector (O const* values)
{
    std::copy(values, values + N, v);
}

/* ------------------------- Vector helpers ----------------------- */

/** Cross product template for partial specialization. */
template <typename T, int N>
inline Vector<T,N>
cross_product (Vector<T,N> const& /*v1*/, Vector<T,N> const& /*v2*/)
{
    throw std::invalid_argument("Only defined for 3-vectors");
}

/** Cross product function for 3-vectors of any type. */
template <typename T>
inline Vector<T,3>
cross_product (Vector<T,3> const& v1, Vector<T,3> const& v2)
{
    return Vector<T,3>(v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]);
}

/* ---------------------------- Management ------------------------ */

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::fill (T const& value)
{
    std::fill(v, v + N, value);
    return *this;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::copy (T const* values, int num)
{
    std::copy(values, values + num, this->v);
    return *this;
}

template <typename T, int N>
inline T
Vector<T,N>::minimum (void) const
{
    return *std::min_element(v, v + N);
}

template <typename T, int N>
inline T
Vector<T,N>::maximum (void) const
{
    return *std::max_element(v, v + N);
}

template <typename T, int N>
inline T
Vector<T,N>::sum (void) const
{
    return std::accumulate(v, v + N, T(0), std::plus<T>());
}

template <typename T, int N>
inline T
Vector<T,N>::abs_sum (void) const
{
    return std::accumulate(v, v + N, T(0), &algo::accum_absolute_sum<T>);
}

template <typename T, int N>
inline T
Vector<T,N>::product (void) const
{
    return std::accumulate(v, v + N, T(1), std::multiplies<T>());
}

/* ------------------------- Unary operators ---------------------- */

template <typename T, int N>
inline T
Vector<T,N>::norm (void) const
{
    return std::sqrt(square_norm());
}

template <typename T, int N>
inline T
Vector<T,N>::square_norm (void) const
{
    return std::accumulate(v, v + N, T(0), &algo::accum_squared_sum<T>);
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::normalize (void)
{
    std::for_each(v, v + N, algo::foreach_divide_by_const<T>(norm()));
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::normalized (void) const
{
    return Vector<T,N>(*this).normalize();
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::abs_value (void)
{
    std::for_each(v, v + N, &algo::foreach_absolute_value<T>);
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::abs_valued (void) const
{
    return Vector<T,N>(*this).abs_value();
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::negate (void)
{
    std::for_each(v, v + N, &algo::foreach_negate_value<T>);
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::negated (void) const
{
    return Vector<T,N>(*this).negate();
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::sort_asc (void)
{
    std::sort(v, v + N, std::less<T>());
    return *this;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::sort_desc (void)
{
    std::sort(v, v + N, std::greater<T>());
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::sorted_asc (void) const
{
    return Vector<T,N>(*this).sort_asc();
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::sorted_desc (void) const
{
    return Vector<T,N>(*this).sort_desc();
}

template <typename T, int N>
template <typename F>
inline Vector<T,N>&
Vector<T,N>::apply_for_each (F functor)
{
    std::for_each(v, v + N, functor);
    return *this;
}

template <typename T, int N>
template <typename F>
inline Vector<T,N>
Vector<T,N>::applied_for_each (F functor) const
{
    return Vector<T,N>(*this).apply_for_each(functor);
}

/* ------------------------ Binary operators ---------------------- */

template <typename T, int N>
inline T
Vector<T,N>::dot (Vector<T,N> const& other) const
{
    return std::inner_product(v, v + N, *other, T(0));
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::cross (Vector<T,N> const& other) const
{
    return cross_product(*this, other);
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::cw_mult (Vector<T,N> const& other) const
{
    Vector<T,N> ret;
    std::transform(v, v + N, other.v, ret.v, std::multiplies<T>());
    return ret;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::cw_div (Vector<T,N> const& other) const
{
    Vector<T,N> ret;
    std::transform(v, v + N, other.v, ret.v, std::divides<T>());
    return ret;
}

template <typename T, int N>
inline bool
Vector<T,N>::is_similar (Vector<T,N> const& other, T const& eps) const
{
    return std::equal(v, v + N, *other, algo::predicate_epsilon_equal<T>(eps));
}

/* ------------------------ Value iterators ----------------------- */

template <typename T, int N>
inline T*
Vector<T,N>::begin (void)
{
    return v;
}

template <typename T, int N>
inline T const*
Vector<T,N>::begin (void) const
{
    return v;
}

template <typename T, int N>
inline T*
Vector<T,N>::end (void)
{
    return v + N;
}

template <typename T, int N>
inline T const*
Vector<T,N>::end (void) const
{
    return v + N;
}

/* ------------------------ Object operators ---------------------- */

template <typename T, int N>
inline T*
Vector<T,N>::operator* (void)
{
    return v;
}

template <typename T, int N>
inline T const*
Vector<T,N>::operator* (void) const
{
    return v;
}

template <typename T, int N>
inline T&
Vector<T,N>::operator[] (int index)
{
    return v[index];
}

template <typename T, int N>
inline T const&
Vector<T,N>::operator[] (int index) const
{
    return v[index];
}

template <typename T, int N>
inline T&
Vector<T,N>::operator() (int index)
{
    return v[index];
}

template <typename T, int N>
inline T const&
Vector<T,N>::operator() (int index) const
{
    return v[index];
}

template <typename T, int N>
inline bool
Vector<T,N>::operator== (Vector<T,N> const& rhs) const
{
    return std::equal(v, v + N, *rhs);
}

template <typename T, int N>
inline bool
Vector<T,N>::operator!= (Vector<T,N> const& rhs) const
{
    return !std::equal(v, v + N, *rhs);
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator= (Vector<T,N> const& rhs)
{
    std::copy(*rhs, *rhs + N, v);
    return *this;
}

template <typename T, int N>
template <typename O>
inline Vector<T,N>&
Vector<T,N>::operator= (Vector<O,N> const& rhs)
{
    std::copy(*rhs, *rhs + N, v);
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator- (void) const
{
    return negated();
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator-= (Vector<T,N> const& rhs)
{
    std::transform(v, v + N, *rhs, v, std::minus<T>());
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator- (Vector<T,N> const& rhs) const
{
    return Vector<T,N>(*this) -= rhs;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator+= (Vector<T,N> const& rhs)
{
    std::transform(v, v + N, *rhs, v, std::plus<T>());
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator+ (Vector<T,N> const& rhs) const
{
    return Vector<T,N>(*this) += rhs;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator-= (T const& rhs)
{
    std::for_each(v, v + N, algo::foreach_substraction_with_const<T>(rhs));
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator- (T const& rhs) const
{
    return Vector<T,N>(*this) -= rhs;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator+= (T const& rhs)
{
    std::for_each(v, v + N, algo::foreach_addition_with_const<T>(rhs));
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator+ (T const& rhs) const
{
    return Vector<T,N>(*this) += rhs;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator*= (T const& rhs)
{
    std::for_each(v, v + N, algo::foreach_multiply_with_const<T>(rhs));
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator* (T const& rhs) const
{
    return Vector<T,N>(*this) *= rhs;
}

template <typename T, int N>
inline Vector<T,N>&
Vector<T,N>::operator/= (T const& rhs)
{
    std::for_each(v, v + N, algo::foreach_divide_by_const<T>(rhs));
    return *this;
}

template <typename T, int N>
inline Vector<T,N>
Vector<T,N>::operator/ (T const& rhs) const
{
    return Vector<T,N>(*this) /= rhs;
}

/* ------------------------- Vector tools ------------------------- */

/** Scalar-vector multiplication. */
template <typename T, int N>
inline Vector<T,N>
operator* (T const& s, Vector<T,N> const& v)
{
    return v * s;
}

/** Scalar-vector addition. */
template <typename T, int N>
inline Vector<T,N>
operator+ (T const& s, Vector<T,N> const& v)
{
    return v + s;
}

/** Scalar-vector substraction. */
template <typename T, int N>
inline Vector<T,N>
operator- (T const& s, Vector<T,N> const& v)
{
    return -v + s;
}

/** Tests if any of the vector values is NaN. */
template <typename T, int N>
inline bool
isnan (Vector<T,N> const& v)
{
    return std::any_of(v.begin(), v.end(),
        [](T const& value) { return std::isnan(value); });
}

MATH_NAMESPACE_END

/* --------------------- Ouput stream adapter --------------------- */

#include <ostream>

MATH_NAMESPACE_BEGIN

/** Serializing a vector to an output stream. */
template <typename T, int N>
inline std::ostream&
operator<< (std::ostream& os, Vector<T,N> const& v)
{
    for (int i = 0; i < N - 1; ++i)
        os << v[i] << " ";
    os << v[N-1];
    return os;
}

MATH_NAMESPACE_END

#endif /* MATH_VECTOR_HEADER */
