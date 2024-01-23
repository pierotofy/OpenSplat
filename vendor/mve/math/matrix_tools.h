/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_MATRIX_TOOLS_HEADER
#define MATH_MATRIX_TOOLS_HEADER

#include <algorithm>

#include "math/defines.h"
#include "math/matrix.h"
#include "math/vector.h"

MATH_NAMESPACE_BEGIN

/**
 * Creates a symmetric projection matrix as used in OpenGL.
 * Values in the frustum are mapped to the unit cube. The frustum
 * near plane is defined by z-near, right and top values, the far-plane
 * is located at z-far.
 */
template <typename T>
Matrix<T,4,4>
matrix_gl_projection (T const& znear, T const& zfar,
    T const& top, T const& right);

/**
 * Creates a symmetric inverse projection matrix as used in OpenGL.
 */
template <typename T>
Matrix<T,4,4>
matrix_inverse_gl_projection (T const& znear, T const& zfar,
    T const& top, T const& right);

/**
 * Creates a view transformation matrix for camera parameters given
 * as camera position, normalized viewing direction, and normalized
 * up-vector.
 */
template <typename T>
Matrix<T,4,4>
matrix_viewtrans (Vector<T,3> const& campos,
    Vector<T,3> const& viewdir, Vector<T,3> const& upvec);

/**
 * Creates an inverse view transformation matrix.
 */
template <typename T>
Matrix<T,4,4>
matrix_inverse_viewtrans (Vector<T,3> const& campos,
    Vector<T,3> const& viewdir, Vector<T,3> const& upvec);

/**
 * Inverts a transformation matrix.
 */
template <typename T>
Matrix<T,4,4>
matrix_invert_trans (Matrix<T,4,4> const& mat);

/**
 * Sets the given square matrix to the identity matrix.
 * The function returns a reference to the given matrix.
 */
template <typename T, int N>
Matrix<T,N,N>&
matrix_set_identity (Matrix<T,N,N>* mat);

/**
 * Sets the given square matrix of dimension 'n' to the identity matrix.
 * The function returns the argument pointer.
 */
template <typename T>
T*
matrix_set_identity (T* mat, int n);

/**
 * Returns true if and only if the given matrix is the identity matrix.
 */
template <typename T, int N>
bool
matrix_is_identity (Matrix<T,N,N> const& mat, T const& epsilon = T(0));

/**
 * Returns a diagonal matrix from the given vector.
 */
template <typename T, int N>
Matrix<T,N,N>
matrix_from_diagonal (math::Vector<T,N> const& v);

/**
 * Sets the diagonal elements of the given matrix.
 */
template <typename T, int N>
Matrix<T,N,N>&
matrix_set_diagonal (Matrix<T,N,N>& mat, T const* diag);

/**
 * Checks whether the input matrix is a diagonal matrix. This is done by
 * testing if all non-diagonal entries are zero (up to some epsilon).
 */
template <typename T>
bool
matrix_is_diagonal (T* const mat, int rows, int cols, T const& epsilon = T(0));

/**
 * Returns the diagonal elements of the matrix as a vector.
 */
template <typename T, int N>
Vector<T,N>
matrix_get_diagonal (Matrix<T,N,N> const& mat);

/**
 * Calculates the trace of the given matrix.
 */
template <typename T, int N>
T
matrix_trace(math::Matrix<T, N, N> const& mat);

/**
 * Calculates the determinant of the given matrix.
 * This is specialized for 1x1, 2x2, 3x3 and 4x4 matrices only.
 */
template <typename T, int N>
T
matrix_determinant (Matrix<T,N,N> const& mat);

/**
 * Calculates the inverse of the given matrix.
 * This is specialized for 1x1, 2x2, 3x3 and 4x4 matrices only.
 */
template <typename T, int N>
Matrix<T,N,N>
matrix_inverse (Matrix<T,N,N> const& mat);

/**
 * Calculates the inverse of the given matrix given its determinant.
 * This is specialized for 1x1, 2x2 and 3x3 matrices only.
 */
template <typename T, int N>
Matrix<T,N,N>
matrix_inverse (Matrix<T,N,N> const& mat, T const& det);

/**
 * Computes the 3x3 rotation matrix from axis and angle notation.
 */
template <typename T>
Matrix<T,3,3>
matrix_rotation_from_axis_angle (Vector<T,3> const& axis, T const& angle);

/**
 * In-place transpose of a dynamically sized dense matrix.
 * The resulting matrix has number of rows and columns exchanged.
 */
template <typename T>
void
matrix_transpose (T const* mat, int rows, int cols);

/**
 * Matrix multiplication of dynamically sized dense matrices.
 * R = A * B where A is MxN, B is NxL and R is MxL. Compexity: O(n*n*n).
 */
template <typename T>
void
matrix_multiply (T const* mat_a, int rows_a, int cols_a,
    T const* mat_b, int cols_b, T* mat_res);

/**
 * Matrix multiplication of the transposed with itself.
 * This computes for a given matrix A the product R = A^T * A.
 * The resulting matrix is of size cols times cols.
 */
template <typename T>
void
matrix_transpose_multiply (T const* mat_a, int rows, int cols, T* mat_res);

/**
 * Swaps the rows r1 and r2 of matrix mat with dimension rows, cols.
 */
template <typename T>
void
matrix_swap_rows (T* mat, int rows, int cols, int r1, int r2);

/**
 * Swaps the columns c1 and c2 of matrix mat with dimension rows, cols.
 */
template <typename T>
void
matrix_swap_columns (T* const mat, int rows, int cols, int c1, int c2);

/**
 * Rotates the entries of the given matrix by 180 degrees in-place.
 */
template <typename T, int N>
void
matrix_rotate_180_inplace (Matrix<T, N, N>* mat_a);

/**
 * Rotates the entries of the given matrix by 180 degrees.
 */
template <typename T, int N>
Matrix<T, N, N>
matrix_rotate_180 (Matrix<T, N, N> const& mat_a);

MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <typename T>
Matrix<T,4,4>
matrix_gl_projection (T const& znear, T const& zfar,
    T const& top, T const& right)
{
    Matrix<T,4,4> proj(0.0f);
    proj(0,0) = znear / right;
    proj(1,1) = znear / top;
    proj(2,2) = -(zfar + znear) / (zfar - znear);
    proj(2,3) = T(-2) * zfar * znear / (zfar - znear);
    proj(3,2) = -1;

    return proj;
}

template <typename T>
Matrix<T,4,4>
matrix_inverse_gl_projection (T const& znear, T const& zfar,
    T const& top, T const& right)
{
    Matrix<T,4,4> iproj(0.0f);
    iproj(0,0) = right / znear;
    iproj(1,1) = top / znear;
    iproj(2,3) = -1;
    iproj(3,2) = (zfar - znear) / (T(-2) * zfar * znear);
    iproj(3,3) = -(zfar + znear) / (T(-2) * zfar * znear);

    return iproj;
}

template <typename T>
Matrix<T,4,4>
matrix_viewtrans (Vector<T,3> const& campos,
    Vector<T,3> const& viewdir, Vector<T,3> const& upvec)
{
    /* Normalize x in case upvec is not perpendicular to viewdir. */
    Vector<T,3> z(-viewdir);
    Vector<T,3> x(upvec.cross(z).normalized());
    Vector<T,3> y(z.cross(x));

    Matrix<T,4,4> m;
    m(0,0) = x[0]; m(0,1) = x[1]; m(0,2) = x[2]; m(0,3) = 0.0f;
    m(1,0) = y[0]; m(1,1) = y[1]; m(1,2) = y[2]; m(1,3) = 0.0f;
    m(2,0) = z[0]; m(2,1) = z[1]; m(2,2) = z[2]; m(2,3) = 0.0f;
    m(3,0) = 0.0f; m(3,1) = 0.0f; m(3,2) = 0.0f; m(3,3) = 1.0f;

    Vector<T,3> t(-campos);
    m(0,3) = m(0,0) * t[0] + m(0,1) * t[1] + m(0,2) * t[2];
    m(1,3) = m(1,0) * t[0] + m(1,1) * t[1] + m(1,2) * t[2];
    m(2,3) = m(2,0) * t[0] + m(2,1) * t[1] + m(2,2) * t[2];

    return m;
}

template <typename T>
Matrix<T,4,4>
matrix_inverse_viewtrans (Vector<T,3> const& campos,
    Vector<T,3> const& viewdir, Vector<T,3> const& upvec)
{
    Vector<T,3> z(-viewdir);
    Vector<T,3> x(upvec.cross(z).normalized());
    Vector<T,3> y(z.cross(x));

    Matrix<T,4,4> m;
    m(0,0) = x[0]; m(0,1) = y[0]; m(0,2) = z[0]; m(0,3) = campos[0];
    m(1,0) = x[1]; m(1,1) = y[1]; m(1,2) = z[1]; m(1,3) = campos[1];
    m(2,0) = x[2]; m(2,1) = y[2]; m(2,2) = z[2]; m(2,3) = campos[2];
    m(3,0) = 0.0f; m(3,1) = 0.0f; m(3,2) = 0.0f; m(3,3) = 1.0f;

    return m;
}

template <typename T>
Matrix<T,4,4>
matrix_invert_trans (Matrix<T,4,4> const& mat)
{
    Matrix<T,4,4> ret(0.0f);
    /* Transpose rotation. */
    ret[0] = mat[0]; ret[1] = mat[4]; ret[2] = mat[8];
    ret[4] = mat[1]; ret[5] = mat[5]; ret[6] = mat[9];
    ret[8] = mat[2]; ret[9] = mat[6]; ret[10] = mat[10];
    /* Invert translation. */
    ret[3]  = -(ret[0] * mat[3] + ret[1] * mat[7] + ret[2] * mat[11]);
    ret[7]  = -(ret[4] * mat[3] + ret[5] * mat[7] + ret[6] * mat[11]);
    ret[11] = -(ret[8] * mat[3] + ret[9] * mat[7] + ret[10] * mat[11]);
    ret[15] = 1.0f;

    return ret;
}

template <typename T, int N>
Matrix<T,N,N>&
matrix_set_identity (Matrix<T,N,N>* mat)
{
    mat->fill(T(0));
    for (int i = 0; i < N * N; i += N + 1)
        (*mat)[i] = T(1);
    return *mat;
}

template <typename T>
T*
matrix_set_identity (T* mat, int n)
{
    int const len = n * n;
    std::fill(mat, mat + len, T(0));
    for (int i = 0; i < len; i += n + 1)
        mat[i] = T(1);
    return mat;
}

template <typename T, int N>
bool
matrix_is_identity (Matrix<T,N,N> const& mat, T const& epsilon)
{
    for (int y = 0, i = 0; y < N; ++y)
        for (int x = 0; x < N; ++x, ++i)
            if ((x == y && !MATH_EPSILON_EQ(mat[i], T(1), epsilon))
                || (x != y && !MATH_EPSILON_EQ(mat[i], T(0), epsilon)))
                return false;
    return true;
}

template <typename T, int N>
Matrix<T,N,N>
matrix_from_diagonal (math::Vector<T,N> const& v)
{
    Matrix<T,N,N> mat;
    std::fill(*mat, *mat + N*N, T(0));
    for (int i = 0, j = 0; i < N*N; i += N+1, j += 1)
        mat[i] = v[j];
    return mat;
}

template <typename T, int N>
Matrix<T,N,N>&
matrix_set_diagonal (Matrix<T,N,N>& mat, T const* diag)
{
    for (int i = 0, j = 0; i < N*N; i += N+1, j += 1)
        mat[i] = diag[j];
    return mat;
}

template <typename T, int N>
Vector<T,N>
matrix_get_diagonal (Matrix<T,N,N> const& mat)
{
    Vector<T,N> diag;
    for (int i = 0, j = 0; i < N*N; i += N+1, j += 1)
        diag[j] = mat[i];
    return diag;
}

template <typename T, int N>
inline T
matrix_trace(math::Matrix<T, N, N> const& mat)
{
    T ret(0.0);
    for (int i = 0; i < N * N; i += N + 1)
        ret += mat[i];
    return ret;
}

template <typename T>
inline T
matrix_determinant (Matrix<T,1,1> const& mat)
{
    return mat[0];
}

template <typename T>
inline T
matrix_determinant (Matrix<T,2,2> const& mat)
{
    return mat[0] * mat[3] - mat[1] * mat [2];
}

template <typename T>
inline T
matrix_determinant (Matrix<T,3,3> const& m)
{
    return m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7]
         - m[2] * m[4] * m[6] - m[1] * m[3] * m[8] - m[0] * m[5] * m[7];
}

template <typename T>
T
matrix_determinant (Matrix<T,4,4> const& m)
{
    return m[0] * (m[5] * m[10] * m[15] - m[5] * m[11] * m[14]
        - m[9] * m[6] * m[15] + m[9] * m[7] * m[14]
        + m[13] * m[6] * m[11] - m[13] * m[7] * m[10])
        //
        + m[1] * (-m[4] * m[10] * m[15] + m[4] * m[11] * m[14]
        + m[8] * m[6] * m[15] - m[8] * m[7] * m[14]
        - m[12] * m[6] * m[11] + m[12] * m[7] * m[10])
        //
        + m[2] * (m[4] * m[9] * m[15] - m[4] * m[11] * m[13]
        - m[8] * m[5] * m[15] + m[8] * m[7] * m[13]
        + m[12] * m[5] * m[11] - m[12] * m[7] * m[9])
        //
        + m[3] * (-m[4] * m[9] * m[14] + m[4] * m[10] * m[13]
        + m[8] * m[5] * m[14] - m[8] * m[6] * m[13]
        - m[12] * m[5] * m[10] + m[12] * m[6] * m[9]);
}

template <typename T>
inline Matrix<T,1,1>
matrix_inverse (Matrix<T,1,1> const& mat)
{
    return matrix_inverse(mat, matrix_determinant(mat));
}

template <typename T>
inline Matrix<T,2,2>
matrix_inverse (Matrix<T,2,2> const& mat)
{
    return matrix_inverse(mat, matrix_determinant(mat));
}

template <typename T>
inline Matrix<T,3,3>
matrix_inverse (Matrix<T,3,3> const& m)
{
    return matrix_inverse(m, matrix_determinant(m));
}

template <typename T>
Matrix<T,1,1>
matrix_inverse (Matrix<T,1,1> const& /*mat*/, T const& det)
{
    Matrix<T,1,1> ret(T(1));
    return ret / det;
}

template <typename T>
Matrix<T,2,2>
matrix_inverse (Matrix<T,2,2> const& mat, T const& det)
{
    Matrix<T,2,2> ret;
    ret[0] =  mat[3]; ret[1] = -mat[1];
    ret[2] = -mat[2]; ret[3] =  mat[0];
    return ret / det;
}

template <typename T>
Matrix<T,3,3>
matrix_inverse (Matrix<T,3,3> const& m, T const& det)
{
    Matrix<T,3,3> ret;
    ret[0] = m[4] * m[8] - m[5] * m[7];
    ret[1] = m[2] * m[7] - m[1] * m[8];
    ret[2] = m[1] * m[5] - m[2] * m[4];
    ret[3] = m[5] * m[6] - m[3] * m[8];
    ret[4] = m[0] * m[8] - m[2] * m[6];
    ret[5] = m[2] * m[3] - m[0] * m[5];
    ret[6] = m[3] * m[7] - m[4] * m[6];
    ret[7] = m[1] * m[6] - m[0] * m[7];
    ret[8] = m[0] * m[4] - m[1] * m[3];
    return ret / det;
}

template <typename T>
Matrix<T,4,4>
matrix_inverse (Matrix<T,4,4> const& m)
{
    Matrix<T,4,4> ret;

    ret[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15]
        + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    ret[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15]
        - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    ret[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15]
        + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    ret[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11]
        - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

    ret[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15]
        - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    ret[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15]
        + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    ret[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15]
        - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    ret[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11]
        + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

    ret[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15]
        + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    ret[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15]
        - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    ret[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15]
        + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    ret[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11]
        - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

    ret[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14]
        - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    ret[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14]
        + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    ret[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14]
        - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    ret[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10]
        + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    T det = m[0] * ret[0] + m[1] * ret[4] + m[2] * ret[8] + m[3] * ret[12];
    return ret / det;
}

template <typename T>
Matrix<T,3,3>
matrix_rotation_from_axis_angle (Vector<T,3> const& axis, T const& angle)
{
    /*
     * http://en.wikipedia.org/wiki/Rotation_matrix
     *     #Rotation_matrix_from_axis_and_angle
     */
    T const ca = std::cos(angle);
    T const sa = std::sin(angle);
    T const omca = T(1) - ca;

    math::Matrix<T, 3, 3> rot;
    rot[0] = ca + MATH_POW2(axis[0]) * omca;
    rot[1] = axis[0] * axis[1] * omca - axis[2] * sa;
    rot[2] = axis[0] * axis[2] * omca + axis[1] * sa;

    rot[3] = axis[1] * axis[0] * omca + axis[2] * sa;
    rot[4] = ca + MATH_POW2(axis[1]) * omca;
    rot[5] = axis[1] * axis[2] * omca - axis[0] * sa;

    rot[6] = axis[2] * axis[0] * omca - axis[1] * sa;
    rot[7] = axis[2] * axis[1] * omca + axis[0] * sa;
    rot[8] = ca + MATH_POW2(axis[2]) * omca;

    return rot;
}

template <typename T>
void
matrix_transpose (T* mat, int rows, int cols)
{
    /* Create a temporary copy of the matrix. */
    T* tmp = new T[rows * cols];
    std::copy(mat, mat + rows * cols, tmp);

    /* Transpose matrix elements. */
    for (int iter = 0, col = 0; col < cols; ++col)
        for (int row = 0; row < rows; ++row, ++iter)
            mat[iter] = tmp[row * cols + col];

    delete[] tmp;
}

template <typename T>
void
matrix_multiply (T const* mat_a, int rows_a, int cols_a,
    T const* mat_b, int cols_b, T* mat_res)
{
    std::fill(mat_res, mat_res + rows_a * cols_b, T(0));
    for (int j = 0; j < cols_b; ++j)
    {
        for (int i = 0; i < rows_a; ++i)
        {
            int const ica = i * cols_a;
            int const icb = i * cols_b;
            for (int k = 0; k < cols_a; ++k)
                mat_res[icb + j] += mat_a[ica + k] * mat_b[k * cols_b + j];
        }
    }
}

template <typename T>
void
matrix_transpose_multiply (T const* mat_a, int rows, int cols, T* mat_res)
{
    std::fill(mat_res, mat_res + cols * cols, T(0));

    T const* A_trans_iter = mat_a;
    T const* A_row = mat_a;
    for (int ri = 0; ri < rows; ++ri, A_row += cols)
    {
        T* R_iter = mat_res;
        for (int c1 = 0; c1 < cols; ++c1, ++A_trans_iter)
            for (int c2 = 0; c2 < cols; ++c2, ++R_iter)
                (*R_iter) += A_row[c2] * (*A_trans_iter);
    }
}

template <typename T>
bool
matrix_is_diagonal (T* const mat, int rows, int cols, T const& epsilon)
{
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < y && x < cols; ++x)
            if (!MATH_EPSILON_EQ(T(0), mat[y * cols + x], epsilon))
                return false;
        for (int x = y + 1; x < cols; ++x)
            if (!MATH_EPSILON_EQ(T(0), mat[y * cols + x], epsilon))
                return false;
    }
    return true;
}

template <typename T>
void
matrix_swap_columns (T* mat, int rows, int cols, int c1, int c2)
{
    for (int i = 0; i < rows; ++i, c1 += cols, c2 += cols)
        std::swap(mat[c1], mat[c2]);
}

template <typename T>
void
matrix_swap_rows (T* mat, int /*rows*/, int cols, int r1, int r2)
{
    r1 = cols * r1;
    r2 = cols * r2;
    for (int i = 0; i < cols; ++i, ++r1, ++r2)
        std::swap(mat[r1], mat[r2]);
}

template <typename T, int N>
void
matrix_rotate_180_inplace (Matrix<T, N, N>* mat_a)
{
    for (int i = 0, j = N * N - 1; i < j; ++i, --j)
        std::swap((*mat_a)[i], (*mat_a)[j]);
}

template <typename T, int N>
Matrix<T, N, N>
matrix_rotate_180 (Matrix<T, N, N> const& mat_a)
{
    Matrix<T, N, N> ret = mat_a;
    matrix_rotate_180_inplace(&ret);
    return  ret;
}

MATH_NAMESPACE_END

#endif /* MATH_MATRIX_TOOLS_HEADER */
