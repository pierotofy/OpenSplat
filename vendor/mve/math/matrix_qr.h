/*
 * Copyright (C) 2015, Daniel Thuerck, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_MATRIX_QR_HEADER
#define MATH_MATRIX_QR_HEADER

#include <vector>
#include <algorithm>
#include <iostream>

#include "math/defines.h"
#include "math/matrix.h"

MATH_NAMESPACE_BEGIN

/**
 * Calculates a QR decomposition for a given matrix A. A is MxN,
 * Q is MxM and R is MxN. Uses Givens algorithm for computation.
 *
 * Reference:
 * - "Matrix Computations" by Gloub and Loan, 3rd Edition, page 227.
 */
template <typename T>
void
matrix_qr (T const* mat_a, int rows, int cols,
    T* mat_q, T* mat_r, T const& epsilon = T(1e-12));

/**
 * Matrix QR decomposition for compile-time fixed-size matrices. The
 * implementation uses the dynamic-size matrices interface in the background.
 */
template <typename T, int M, int N>
void
matrix_qr (Matrix<T, M, N> const& mat_a, Matrix<T, M, M>* mat_q,
    Matrix<T, M, N>* mat_r, T const& epsilon = T(1e-12));

MATH_NAMESPACE_END

/* ------------------------- QR Internals ------------------------- */

MATH_NAMESPACE_BEGIN
MATH_INTERNAL_NAMESPACE_BEGIN

/**
 * Calculates the Givens rotation coefficients c and s by solving [alpha beta]
 * [c s;-c s] = [sqrt(alpha^2+beta^2) 0].
 */
template <typename T>
void
matrix_givens_rotation (T const& alpha, T const& beta,
    T* givens_c, T* givens_s, T const& epsilon)
{
    if (MATH_EPSILON_EQ(beta, T(0), epsilon))
    {
        *givens_c = T(1);
        *givens_s = T(0);
        return;
    }

    if (std::abs(beta) > std::abs(alpha))
    {
        T tao = -alpha / beta;
        *givens_s = T(1) / std::sqrt(T(1) + tao * tao);
        *givens_c = *givens_s * tao;
    }
    else
    {
        T tao = -beta / alpha;
        *givens_c = T(1) / std::sqrt(T(1) + tao * tao);
        *givens_s = *givens_c * tao;
    }
}

/**
 * Applies a Givens rotation for columns (givens_i, givens_k)
 * by only rotating the required set of columns in-place.
 */
template <typename T>
void
matrix_apply_givens_column (T* mat, int rows, int cols, int givens_i,
    int givens_k, T const& givens_c, T const& givens_s)
{
    for (int j = 0; j < rows; ++j)
    {
        T const tao1 = mat[j * cols + givens_i];
        T const tao2 = mat[j * cols + givens_k];
        mat[j * cols + givens_i] = givens_c * tao1 - givens_s * tao2;
        mat[j * cols + givens_k] = givens_s * tao1 + givens_c * tao2;
    }
}

/**
 * Applies a transposed Givens rotation for rows (givens_i, givens_k)
 * by only rotating the required set of rows in-place.
 */
template <typename T>
void
matrix_apply_givens_row (T* mat, int /*rows*/, int cols, int givens_i,
    int givens_k, T const& givens_c, T const& givens_s)
{
    for (int j = 0; j < cols; ++j)
    {
        T const tao1 = mat[givens_i * cols + j];
        T const tao2 = mat[givens_k * cols + j];
        mat[givens_i * cols + j] = givens_c * tao1 - givens_s * tao2;
        mat[givens_k * cols + j] = givens_s * tao1 + givens_c * tao2;
    }
}

MATH_INTERNAL_NAMESPACE_END
MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <typename T>
void
matrix_qr (T const* mat_a, int rows, int cols,
    T* mat_q, T* mat_r, T const& epsilon)
{
    /* Prepare Q and R: Copy A to R, init Q's diagonal. */
    std::copy(mat_a, mat_a + rows * cols, mat_r);
    std::fill(mat_q, mat_q + rows * rows, T(0));
    for (int i = 0; i < rows; ++i)
        mat_q[i * rows + i] = T(1);

    std::vector<T> matrix_subset(2 * (cols + 1), T(0));
    for (int j = 0; j < cols; ++j)
    {
        for (int i = rows - 1; i > j; --i)
        {
            T givens_c, givens_s;
            internal::matrix_givens_rotation(mat_r[(i - 1) * cols + j],
                mat_r[i * cols + j], &givens_c, &givens_s, epsilon);

            int submatrix_width = cols - j;
            for (int k = 0; k < submatrix_width; ++k)
            {
                matrix_subset[0 * submatrix_width + k] =
                    mat_r[(i - 1) * cols + (j + k)];
                matrix_subset[1 * submatrix_width + k] =
                    mat_r[i * cols + (j + k)];
            }

            /* Only apply Givens rotation on a subblock of the matrix. */
            for (int k = 0; k < (cols - j); ++k)
            {
                mat_r[(i - 1) * cols + (j + k)] =
                    givens_c * matrix_subset[0 * submatrix_width + k] -
                    givens_s * matrix_subset[1 * submatrix_width + k];
                mat_r[i * cols + (j + k)] =
                    givens_s * matrix_subset[0 * submatrix_width + k] +
                    givens_c * matrix_subset[1 * submatrix_width + k];
            }

            /* Construct Q matrix by repeated Givens rotations. */
            internal::matrix_apply_givens_column(mat_q, rows, rows, i - 1, i,
                givens_c, givens_s);
        }
    }
}

template <typename T, int M, int N>
void
matrix_qr (Matrix<T, M, N> const& mat_a, Matrix<T, M, M>* mat_q,
    Matrix<T, M, N>* mat_r, T const& epsilon)
{
    matrix_qr(mat_a.begin(), M, N, mat_q->begin(), mat_r->begin(), epsilon);
}

MATH_NAMESPACE_END

#endif /* MATH_MATRIX_QR_HEADER */
