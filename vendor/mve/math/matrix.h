/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_MATRIX_HEADER
#define MATH_MATRIX_HEADER

#include <algorithm>
#include <functional>
#include <utility>
#include <numeric>
#include <ostream>

#include "math/defines.h"
#include "math/algo.h"
#include "math/vector.h"

MATH_NAMESPACE_BEGIN

/*
 * Vector type definitions for convenience.
 */
template <typename T, int N, int M> class Matrix;
typedef Matrix<float, 2, 2> Matrix2f;
typedef Matrix<float, 3, 3> Matrix3f;
typedef Matrix<float, 4, 4> Matrix4f;
typedef Matrix<double, 2, 2> Matrix2d;
typedef Matrix<double, 3, 3> Matrix3d;
typedef Matrix<double, 4, 4> Matrix4d;
typedef Matrix<int, 2, 2> Matrix2i;
typedef Matrix<int, 3, 3> Matrix3i;
typedef Matrix<int, 4, 4> Matrix4i;
typedef Matrix<unsigned int, 2, 2> Matrix2ui;
typedef Matrix<unsigned int, 3, 3> Matrix3ui;
typedef Matrix<unsigned int, 4, 4> Matrix4ui;

/**
 * Matrix class for arbitrary dimensions and types.
 * This class uses conventions from mathematics:
 *
 * Type: Matrix<T,ROWS,COLS>
 * Access: M(row, col)
 *
 * E.g., a matrix with 3 rows and 4 columns has type Matrix<T,3,4>.
 * The lower right element is accessed with M(2,3);
 */
template <typename T, int N, int M>
class Matrix
{
public:
    typedef T ValueType;

    static int constexpr rows = N;
    static int constexpr cols = M;

    /* ------------------------ Constructors ---------------------- */

    /** Default ctor that leaves values uninitialized. */
    Matrix (void);
    /** Ctor taking a pointer to initialize values. */
    explicit Matrix (T const* values);
    /** Ctor that initializes ALL elements. */
    explicit Matrix (T const& value);

    /** Copy ctor from matrix of same type. */
    Matrix (Matrix<T,N,M> const& other);
    /** Copy ctor from matrix of different type. */
    template <typename O>
    Matrix (Matrix<O,N,M> const& other);

    /* ------------------------- Management ----------------------- */

    /** Fills all vector elements with the given value. */
    Matrix<T,N,M>& fill (T const& value);

    /** Tests if the matrix is square. */
    bool is_square (void) const;

    /** Returns a row of the matrix as vector. */
    Vector<T,M> row (int index) const;
    /** Returns a column of the matrix as vector. */
    Vector<T,N> col (int index) const;

    /** Returns the smallest element in the matrix. */
    T minimum (void) const;
    /** Returns the largest element in the matrix. */
    T maximum (void) const;

    /** Stacks this (left) and another matrix (right) horizontally. */
    template <int O>
    Matrix<T,N,M+O> hstack (Matrix<T,N,O> const& other) const;

    /** Stacks this (top) and another matrix (bottom) vertically. */
    template <int O>
    Matrix<T,N+O,M> vstack (Matrix<T,O,M> const& other) const;

    /** Stacks this matrix (left) and another vector (right) horizontally. */
    Matrix<T,N,M+1> hstack (Vector<T,N> const& other) const;

    /** Stacks this matrix (top) and another vector (bottom) vertically. */
    Matrix<T,N+1,M> vstack (Vector<T,M> const& other) const;

    /** Returns a new matrix with the specified row deleted. */
    Matrix<T,N-1,M> delete_row (int index) const;

    /** Returns a new matrix with the specified column deleted. */
    Matrix<T,N,M-1> delete_col (int index) const;

    /* ---------------------- Unary operators --------------------- */

    /** Component-wise negation on self, returns self. */
    Matrix<T,N,M>& negate (void);
    /** Returns a component-wise negation on copy of self. */
    Matrix<T,N,M> negated (void) const;

    /** Transpose the current matrix. Only available for square matrices. */
    Matrix<T,M,N>& transpose (void);
    /** Returns a transposed copy of self by treating rows as columns. */
    Matrix<T,M,N> transposed (void) const;

    /* --------------------- Binary operators --------------------- */

    /** Matrix with matrix multiplication. */
    template <int U>
    Matrix<T,N,U> mult (Matrix<T,M,U> const& rhs) const;

    /** Matrix with vector multiplication. */
    Vector<T,N> mult (Vector<T,M> const& rhs) const;

    /** Matrix with smaller vector multiplication. */
    Vector<T,N-1> mult (Vector<T,M-1> const& rhs, T const& v) const;

    /** Component-wise similarity using epsilon checks. */
    bool is_similar (Matrix<T,N,M> const& other, T const& epsilon) const;

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
    T& operator() (int row, int col);
    /** Const element access operator. */
    T const& operator() (int row, int col) const;

    /** Element linear access operator. */
    T& operator[] (unsigned int i);
    /** Const element linear access operator. */
    T const& operator[] (unsigned int i) const;

    /** Comparison operator. */
    bool operator== (Matrix<T,N,M> const& rhs) const;
    /** Comparison operator. */
    bool operator!= (Matrix<T,N,M> const& rhs) const;

    /** Assignment operator. */
    Matrix<T,N,M>& operator= (Matrix<T,N,M> const& rhs);
    /** Assignment operator from different type. */
    template <typename O>
    Matrix<T,N,M>& operator= (Matrix<O,N,M> const& rhs);

    /** Component-wise negation. */
    Matrix<T,N,M> operator- (void) const;

    /** Self-substraction with other matrix. */
    Matrix<T,N,M>& operator-= (Matrix<T,N,M> const& rhs);
    /** Substraction with other matrix. */
    Matrix<T,N,M> operator- (Matrix<T,N,M> const& rhs) const;
    /** Self-addition with other matrix. */
    Matrix<T,N,M>& operator+= (Matrix<T,N,M> const& rhs);
    /** Addition with other matrix. */
    Matrix<T,N,M> operator+ (Matrix<T,N,M> const& rhs) const;

    /** Multiplication with other matrix. */
    template <int U>
    Matrix<T,N,U> operator* (Matrix<T,M,U> const& rhs) const;
    /** Multiplication with other vector. */
    Vector<T,N> operator* (Vector<T,M> const& rhs) const;

    /** Component-wise self-substraction with scalar. */
    Matrix<T,N,M>& operator-= (T const& rhs);
    /** Component-wise substraction with scalar. */
    Matrix<T,N,M> operator- (T const& rhs) const;
    /** Component-wise self-addition with scalar. */
    Matrix<T,N,M>& operator+= (T const& rhs);
    /** Component-wise addition with scalar. */
    Matrix<T,N,M> operator+ (T const& rhs) const;
    /** Component-wise self-multiplication with scalar. */
    Matrix<T,N,M>& operator*= (T const& rhs);
    /** Component-wise multiplication with scalar. */
    Matrix<T,N,M> operator* (T const& rhs) const;
    /** Component-wise self-division by scalar. */
    Matrix<T,N,M>& operator/= (T const& rhs);
    /** Component-wise division by scalar. */
    Matrix<T,N,M> operator/ (T const& rhs) const;

protected:
    T m[N * M];
};

MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

#include <algorithm>

MATH_NAMESPACE_BEGIN

template <typename T, int N, int M>
int constexpr Matrix<T,N,M>::rows;

template <typename T, int N, int M>
int constexpr Matrix<T,N,M>::cols;

template <typename T, int N, int M>
inline
Matrix<T,N,M>::Matrix (void)
{
}

template <typename T, int N, int M>
inline
Matrix<T,N,M>::Matrix (T const* values)
{
    std::copy(values, values + N * M, m);
}

template <typename T, int N, int M>
inline
Matrix<T,N,M>::Matrix (T const& value)
{
    fill(value);
}

template <typename T, int N, int M>
inline
Matrix<T,N,M>::Matrix (Matrix<T,N,M> const& other)
{
    std::copy(*other, *other + M * N, m);
}

template <typename T, int N, int M>
template <typename O>
inline
Matrix<T,N,M>::Matrix (Matrix<O,N,M> const& other)
{
    std::copy(*other, *other + M * N, m);
}

/* ------------------------- Matrix helpers ----------------------- */

template <typename T, int N, int M>
inline bool
matrix_is_square (Matrix<T,N,M> const& /*m*/)
{
    return false;
}

template <typename T, int N>
inline bool
matrix_is_square (Matrix<T,N,N> const& /*m*/)
{
    return true;
}

template <typename T, int N>
inline Matrix<T,N,N>&
matrix_inplace_transpose (Matrix<T,N,N>& matrix)
{
    for (int i = 0; i < matrix.rows; ++i)
        for (int j = i + 1; j < matrix.cols; ++j)
            std::swap(matrix(i, j), matrix(j, i));
    return matrix;
}

/* ---------------------------- Management ------------------------ */

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::fill (T const& value)
{
    std::fill(m, m + N * M, value);
    return *this;
}

template <typename T, int N, int M>
inline bool
Matrix<T,N,M>::is_square (void) const
{
    return matrix_is_square(*this);
}

template <typename T, int N, int M>
inline T
Matrix<T,N,M>::minimum (void) const
{
    return *std::min_element(m, m + N * M);
}

template <typename T, int N, int M>
inline T
Matrix<T,N,M>::maximum (void) const
{
    return *std::max_element(m, m + N * M);
}

template <typename T, int N, int M>
inline Vector<T,M>
Matrix<T,N,M>::row (int index) const
{
    return Vector<T,M>(m + index * M);
}

template <typename T, int N, int M>
inline Vector<T,N>
Matrix<T,N,M>::col (int index) const
{
    typedef algo::InterleavedIter<T,M> RowIter;
    Vector<T,N> ret;
    std::copy_n(RowIter(m + index), N, *ret);
    return ret;
}

template <typename T, int N, int M>
template <int O>
inline Matrix<T,N,M+O>
Matrix<T,N,M>::hstack (Matrix<T,N,O> const& other) const
{
    math::Matrix<T,N,M+O> ret;
    T const* in1 = m;
    T const* in2 = *other;
    for (T* out = *ret; in1 < m + M*N; in1 += M, in2 += O, out += O+M)
    {
        std::copy(in1, in1 + M, out);
        std::copy(in2, in2 + O, out + M);
    }
    return ret;
}

template <typename T, int N, int M>
template <int O>
inline Matrix<T,N+O,M>
Matrix<T,N,M>::vstack (Matrix<T,O,M> const& other) const
{
    Matrix<T,N+O,M> ret;
    std::copy(m, m + M*N, *ret);
    std::copy(*other, *other + O*M, *ret + M*N);
    return ret;
}

template <typename T, int N, int M>
inline Matrix<T,N,M+1>
Matrix<T,N,M>::hstack (Vector<T,N> const& other) const
{
    Matrix<T,N,M+1> ret;
    T const* in1 = m;
    T const* in2 = *other;
    for (T* out = *ret; in1 < m + M*N; in1 += M, in2 += 1, out += M+1)
    {
        std::copy(in1, in1 + M, out);
        std::copy(in2, in2 + 1, out + M);
    }
    return ret;
}

template <typename T, int N, int M>
inline Matrix<T,N+1,M>
Matrix<T,N,M>::vstack (Vector<T,M> const& other) const
{
    Matrix<T,N+1,M> ret;
    std::copy(m, m + M*N, *ret);
    std::copy(*other, *other + M, *ret + M*N);
    return ret;
}

template <typename T, int N, int M>
inline Matrix<T,N-1,M>
Matrix<T,N,M>::delete_row (int index) const
{
    Matrix<T,N-1,M> ret;
    T const* in_ptr = this->begin();
    T* out_ptr = ret.begin();
    for (int i = 0; i < N; ++i, in_ptr += M)
        if (i != index)
        {
            std::copy(in_ptr, in_ptr + M, out_ptr);
            out_ptr += M;
        }
    return ret;
}

template <typename T, int N, int M>
inline Matrix<T,N,M-1>
Matrix<T,N,M>::delete_col (int index) const
{
    Matrix<T,N,M-1> ret;
    T const* in_ptr = this->begin();
    T* out_ptr = ret.begin();
    for (int i = 0; i < M*N; ++i, ++in_ptr)
        if (i % M != index)
        {
            *out_ptr = *in_ptr;
            out_ptr += 1;
        }
    return ret;
}

/* ------------------------- Unary operators ---------------------- */

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::negate (void)
{
    std::for_each(m, m + N*M, &algo::foreach_negate_value<T>);
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::negated (void) const
{
    return Matrix<T,N,M>(*this).negate();
}

template <typename T, int N, int M>
inline Matrix<T,M,N>&
Matrix<T,N,M>::transpose (void)
{
    return matrix_inplace_transpose(*this);
}

template <typename T, int N, int M>
inline Matrix<T,M,N>
Matrix<T,N,M>::transposed (void) const
{
    Matrix<T,M,N> ret;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < M; ++j)
            ret(j,i) = (*this)(i,j);
    return ret;
}

/* ------------------------ Binary operators ---------------------- */

template <typename T, int N, int M>
template <int U>
inline Matrix<T,N,U>
Matrix<T,N,M>::mult (Matrix<T,M,U> const& rhs) const
{
    typedef algo::InterleavedIter<T,U> RowIter;
    Matrix<T,N,U> ret;
    for (int i = 0; i < ret.rows; ++i)
        for (int j = 0; j < ret.cols; ++j)
            ret(i,j) = std::inner_product(m + M * i,
                m + M * i + M, RowIter(*rhs + j), T(0));
    return ret;
}

template <typename T, int N, int M>
inline Vector<T,N>
Matrix<T,N,M>::mult (Vector<T,M> const& rhs) const
{
    Vector<T,N> ret;
    for (int i = 0; i < N; ++i)
        ret[i] = std::inner_product(m + M * i, m + M * i + M, *rhs, T(0));
    return ret;
}

template <typename T, int N, int M>
inline Vector<T,N-1>
Matrix<T,N,M>::mult (Vector<T,M-1> const& rhs, T const& v) const
{
    Vector<T,N-1> ret;
    for (int i = 0; i < N-1; ++i)
        ret[i] = std::inner_product(m + M * i, m + M * i + M - 1, *rhs, T(0))
            + v * m[M * i + M - 1];
    return ret;
}

template <typename T, int N, int M>
inline bool
Matrix<T,N,M>::is_similar (Matrix<T,N,M> const& other, T const& eps) const
{
    return std::equal(m, m + N * M, *other,
        algo::predicate_epsilon_equal<T>(eps));
}

/* ------------------------ Value iterators ----------------------- */

template <typename T, int N, int M>
inline T*
Matrix<T,N,M>::begin (void)
{
    return m;
}

template <typename T, int N, int M>
inline T const*
Matrix<T,N,M>::begin (void) const
{
    return m;
}

template <typename T, int N, int M>
inline T*
Matrix<T,N,M>::end (void)
{
    return m + N * M;
}

template <typename T, int N, int M>
inline T const*
Matrix<T,N,M>::end (void) const
{
    return m + N * M;
}

/* ------------------------ Object operators ---------------------- */

template <typename T, int N, int M>
inline T*
Matrix<T,N,M>::operator* (void)
{
    return m;
}

template <typename T, int N, int M>
inline T const*
Matrix<T,N,M>::operator* (void) const
{
    return m;
}

template <typename T, int N, int M>
inline T&
Matrix<T,N,M>::operator() (int row, int col)
{
    return m[row * M + col];
}

template <typename T, int N, int M>
inline T const&
Matrix<T,N,M>::operator() (int row, int col) const
{
    return m[row * M + col];
}
template <typename T, int N, int M>
inline T&
Matrix<T,N,M>::operator[] (unsigned int i)
{
    return m[i];
}

template <typename T, int N, int M>
inline T const&
Matrix<T,N,M>::operator[] (unsigned int i) const
{
    return m[i];
}

template <typename T, int N, int M>
inline bool
Matrix<T,N,M>::operator== (Matrix<T,N,M> const& rhs) const
{
    return std::equal(m, m + N * M, *rhs);
}

template <typename T, int N, int M>
inline bool
Matrix<T,N,M>::operator!= (Matrix<T,N,M> const& rhs) const
{
    return !std::equal(m, m + N * M, *rhs);
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator= (Matrix<T,N,M> const& rhs)
{
    std::copy(*rhs, *rhs + N * M, m);
    return *this;
}

template <typename T, int N, int M>
template <typename O>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator= (Matrix<O,N,M> const& rhs)
{
    std::copy(*rhs, *rhs + N * M, m);
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator- (void) const
{
    return negated();
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator-= (Matrix<T,N,M> const& rhs)
{
    std::transform(m, m + N * M, *rhs, m, std::minus<T>());
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator- (Matrix<T,N,M> const& rhs) const
{
    return Matrix<T,N,M>(*this) -= rhs;
}

template <typename T, int N, int M>
template <int U>
inline Matrix<T,N,U>
Matrix<T,N,M>::operator* (Matrix<T,M,U> const& rhs) const
{
    return mult(rhs);
}

template <typename T, int N, int M>
inline Vector<T,N>
Matrix<T,N,M>::operator* (Vector<T,M> const& rhs) const
{
    return mult(rhs);
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator+= (Matrix<T,N,M> const& rhs)
{
    std::transform(m, m + N * M, *rhs, m, std::plus<T>());
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator+ (Matrix<T,N,M> const& rhs) const
{
    return Matrix<T,N,M>(*this) += rhs;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator-= (T const& rhs)
{
    std::for_each(m, m + N * M, algo::foreach_substraction_with_const<T>(rhs));
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator- (T const& rhs) const
{
    return Matrix<T,N,M>(*this) -= rhs;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator+= (T const& rhs)
{
    std::for_each(m, m + N * M, algo::foreach_addition_with_const<T>(rhs));
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator+ (T const& rhs) const
{
    return Matrix<T,N,M>(*this) += rhs;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator*= (T const& rhs)
{
    std::for_each(m, m + N * M, algo::foreach_multiply_with_const<T>(rhs));
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator* (T const& rhs) const
{
    return Matrix<T,N,M>(*this) *= rhs;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>&
Matrix<T,N,M>::operator/= (T const& rhs)
{
    std::for_each(m, m + N * M, algo::foreach_divide_by_const<T>(rhs));
    return *this;
}

template <typename T, int N, int M>
inline Matrix<T,N,M>
Matrix<T,N,M>::operator/ (T const& rhs) const
{
    return Matrix<T,N,M>(*this) /= rhs;
}

MATH_NAMESPACE_END

/* --------------------- Ouput stream adapter --------------------- */

#include <ostream>

MATH_NAMESPACE_BEGIN

/** Serializing a vector to an output stream. */
template <typename T, int N, int M>
inline std::ostream&
operator<< (std::ostream& os, Matrix<T,N,M> const& m)
{
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            os << m(i,j) << (j == m.cols - 1 ? "\n" : " ");
    return os;
}

MATH_NAMESPACE_END

#endif /* MATH_MATRIX_HEADER */
