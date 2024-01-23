/*
 * Copyright (C) 2015, Daniel Thuerck, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 *
 * The matrix formats for this implementation are exemplary visualized:
 *
 * A A   U U
 * A A = U U * S S * V V
 * A A   U U   S S   V V
 *
 *                 S S S   V V V
 * A A A = U U U * S S S * V V V
 * A A A   U U U   S S S   V V V
 */

#ifndef MATH_MATRIX_SVD_HEADER
#define MATH_MATRIX_SVD_HEADER

#include <vector>

#include "math/defines.h"
#include "math/matrix.h"
#include "math/matrix_tools.h"
#include "math/matrix_qr.h"

#define MATH_SVD_DEFAULT_ZERO_THRESHOLD 1e-12

MATH_NAMESPACE_BEGIN

/**
 * SVD for dynamic-size matrices A of size MxN (M rows, N columns).
 * The function decomposes input matrix A such that A = USV^T where
 * A is MxN, U is MxN, S is a N-vector and V is NxN.
 * Any of U, S or V can be null, however, this does not save operations.
 *
 * Usually, M >= N, i.e. the input matrix has more rows than columns.
 * If M > 5/3 N, QR decomposition is used to do an economy SVD after Chan
 * that saves some operations. This SVD also handles the case where M < N. In
 * this case, zero rows are internally added to A until A is a square matrix.
 *
 * References:
 * - "Matrix Computations" by Gloub and Loan (page 455, algo 8.6.2, [GK-SVD])
 * - "An Improved Algorithm for Computing the SVD" by Chan (1987) [R-SVD].
 */
template <typename T>
void
matrix_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v,
    T const& epsilon = T(MATH_SVD_DEFAULT_ZERO_THRESHOLD));

/**
 * SVD for compile-time fixed-size matrices. The implementation of this
 * function uses the dynamic-size matrices interface in the background.
 * Any of the results can be null, however, this does not save operations.
 */
template <typename T, int M, int N>
void
matrix_svd (Matrix<T, M, N> const& mat_a, Matrix<T, M, N>* mat_u,
    Matrix<T, N, N>* mat_s, Matrix<T, N, N>* mat_v,
    T const& epsilon = T(MATH_SVD_DEFAULT_ZERO_THRESHOLD));

/**
 * Computes the Moore–Penrose pseudoinverse of matrix A using the SVD.
 * Let the SVD of A be A = USV*, then the pseudoinverse is A' = VS'U*.
 * The inverse S' of S is obtained by taking the reciprocal of non-zero
 * diagonal elements, leaving zeros (up to the epsilon) in place.
 */
template <typename T, int M, int N>
void
matrix_pseudo_inverse (Matrix<T, M, N> const& A,
    Matrix<T, N, M>* result,
    T const& epsilon = T(MATH_SVD_DEFAULT_ZERO_THRESHOLD));

MATH_NAMESPACE_END

/* ------------------------ SVD Internals ------------------------- */

MATH_NAMESPACE_BEGIN
MATH_INTERNAL_NAMESPACE_BEGIN

/**
 * Checks whether the lower-right square sub-matrix of size KxK is enclosed by
 * zeros (up to some epsilon) within a square matrix of size MxM. This check
 * is SVD specific and probably not very useful for other code.
 * Note: K must be smaller than or equal to M.
 */
template <typename T>
bool
matrix_is_submatrix_zero_enclosed (T const* mat, int m, int k, T const& epsilon)
{
    int const j = m - k - 1;
    if (j < 0)
        return true;
    for (int i = m - k; i < m; ++i)
        if (!MATH_EPSILON_EQ(mat[j * m + i], T(0), epsilon)
            || !MATH_EPSILON_EQ(mat[i * m + j], T(0), epsilon))
            return false;
    return true;
}

/**
 * Checks whether the super-diagonal (above the diagonal) of a MxN matrix
 * does not contain zeros up to some epsilon.
 */
template <typename T>
bool
matrix_is_superdiagonal_nonzero (T const* mat,
    int rows, int cols, T const& epsilon)
{
    int const n = std::min(rows, cols) - 1;
    for (int i = 0; i < n; ++i)
        if (MATH_EPSILON_EQ(T(0), mat[i * cols + i + 1], epsilon))
            return false;
    return true;
}

/**
 * Returns the larger eigenvalue of the given 2x2 matrix. The eigenvalues of
 * the matrix are assumed to be non-complex and a negative root set to zero.
 */
template <typename T>
void
matrix_2x2_eigenvalues (T const* mat, T* smaller_ev, T* larger_ev)
{
    /* For matrix [a b; c d] solve (a+d) / 2 + sqrt((a+d)^2 / 4 - ad + bc). */
    T const& a = mat[0];
    T const& b = mat[1];
    T const& c = mat[2];
    T const& d = mat[3];

    T x = MATH_POW2(a + d) / T(4) - a * d + b * c;
    x = (x > T(0) ? std::sqrt(x) : T(0));
    *smaller_ev = (a + d) / T(2) - x;
    *larger_ev = (a + d) / T(2) + x;
}

/**
 * Creates a householder transformation vector and the coefficient for
 * householder matrix creation. As input, the function uses a column-frame
 * in a given matrix, i.e.  mat(subset_row_start:subset_row_end,
 * subset_col).
 */
template <typename T>
void
matrix_householder_vector (T const* input, int length,
    T* vector, T* beta, T const& epsilon, T const& norm_factor)
{
    T sigma(0);
    for (int i = 1; i < length; ++i)
        sigma += MATH_POW2(input[i] / norm_factor);

    vector[0] = T(1);
    for (int i = 1; i < length; ++i)
        vector[i] = input[i] / norm_factor;

    if (MATH_EPSILON_EQ(sigma, T(0), epsilon))
    {
        *beta = T(0);
        return;
    }

    T first = input[0] / norm_factor;
    T mu = std::sqrt(MATH_POW2(first) + sigma);
    if (first < epsilon)
        vector[0] = first - mu;
    else
        vector[0] = -sigma / (first + mu);

    first = vector[0];
    *beta = T(2) * MATH_POW2(first) / (sigma + MATH_POW2(first));
    for (int i = 0; i < length; ++i)
        vector[i] /= first;
}

/**
 * Given a Householder vector and beta coefficient, this function creates
 * a transformation matrix to apply the Householder Transformation by simple
 * matrix multiplication.
 */
template <typename T>
void
matrix_householder_matrix (T const* vector, int length, T const beta,
    T* matrix)
{
    std::fill(matrix, matrix + length * length, T(0));
    for (int i = 0; i < length; ++i)
    {
        matrix[i * length + i] = T(1);
    }

    for (int i = 0; i < length; ++i)
    {
        for (int j = 0; j < length; ++j)
        {
            matrix[i * length + j] -= beta * vector[i] * vector[j];
        }
    }
}

/**
 * Applies a given householder matrix to a frame in a given matrix with
 * offset (offset_rows, offset_cols).
 */
template <typename T>
void
matrix_apply_householder_matrix (T* mat_a, int rows, int cols,
    T const* house_mat, int house_length, int offset_rows, int offset_cols)
{
    /* Save block from old matrix that will be modified. */
    int house_length_n = house_length - (rows - cols);
    std::vector<T> rhs(house_length * house_length_n);
    for (int i = 0; i < house_length; ++i)
    {
        for (int j = 0; j < house_length_n; ++j)
        {
            rhs[i * house_length_n + j] = mat_a[(offset_rows + i) *
                cols + (offset_cols + j)];
        }
    }

    /* Multiply block matrices. */
    for (int i = 0; i < (rows - offset_rows); ++i)
    {
        for (int j = 0; j < (cols - offset_cols); ++j)
        {
            T current(0);
            for (int k = 0; k < house_length; ++k)
            {
                current += (house_mat[i * house_length + k] *
                    rhs[k * house_length_n + j]);
            }
            mat_a[(offset_rows + i) * cols + (offset_cols + j)] = current;
        }
    }
}

/**
 * Bidiagonalizes a given MxN matrix, resulting in a MxN matrix U,
 * a bidiagonal MxN matrix B and a NxN matrix V.
 *
 * Reference: "Matrix Computations" by Golub and Loan, 3rd edition,
 * from page 252 (algorithm 5.4.2).
 */
template <typename T>
void
matrix_bidiagonalize (T const* mat_a, int rows, int cols, T* mat_u,
    T* mat_b, T* mat_v, T const& epsilon)
{
    /* Initialize U and V with identity matrices. */
    matrix_set_identity(mat_u, rows);
    matrix_set_identity(mat_v, cols);

    /* Copy mat_a into mat_b. */
    std::copy(mat_a, mat_a + rows * cols, mat_b);

    int const steps = (rows == cols) ? (cols - 1) : cols;
    for (int k = 0; k < steps; ++k)
    {
        int const sub_length = rows - k;
        std::vector<T> buffer(sub_length // input_vec
            + sub_length // house_vec
            + sub_length * sub_length // house_mat
            + rows * rows // update_u
            + rows * rows); // mat_u_tmp
        T* input_vec = &buffer[0];
        T* house_vec = input_vec + sub_length;
        T* house_mat = house_vec + sub_length;
        T house_beta;
        T* update_u = house_mat + sub_length * sub_length;
        T* mat_u_tmp = update_u + rows * rows;

        for (int i = 0; i < sub_length; ++i)
            input_vec[i] = mat_b[(k + i) * cols + k];

        matrix_householder_vector(input_vec, sub_length, house_vec,
            &house_beta, epsilon, T(1));
        matrix_householder_matrix(house_vec, sub_length,
            house_beta, house_mat);
        matrix_apply_householder_matrix(mat_b, rows, cols,
            house_mat, sub_length, k, k);

        for (int i = k + 1; i < rows; ++i)
            mat_b[i * cols + k] = T(0);

        /* Construct U update matrix and update U. */
        std::fill(update_u, update_u + rows * rows, T(0));
        for (int i = 0; i < k; ++i)
            update_u[i * rows + i] = T(1);
        for (int i = 0; i < sub_length; ++i)
        {
            for (int j = 0; j < sub_length; ++j)
            {
                update_u[(k + i) * rows + (k + j)] =
                    house_mat[i * sub_length + j];
            }
        }

        /* Copy matrix U for multiplication. */
        std::copy(mat_u, mat_u + rows * rows, mat_u_tmp);
        matrix_multiply(mat_u_tmp, rows, rows, update_u, rows, mat_u);

        if (k <= cols - 3)
        {
            /* Normalization constant for numerical stability. */
            T norm(0);
            for (int i = k + 1; i < cols; ++i)
                norm += mat_b[k * cols + i];
            if (MATH_EPSILON_EQ(norm, T(0), epsilon))
                norm = T(1);

            int const inner_sub_length = cols - (k + 1);
            int const slice_rows = rows - k;
            int const slice_cols = cols - k - 1;
            std::vector<T> buffer2(inner_sub_length // inner_input_vec
                + inner_sub_length // inner_house_vec
                + inner_sub_length * inner_sub_length // inner_house_mat
                + slice_rows * slice_cols // mat_b_res
                + slice_rows * slice_cols // mat_b_tmp
                + cols * cols // update_v
                + cols * cols); // mat_v_tmp

            T* inner_input_vec = &buffer2[0];
            T* inner_house_vec = inner_input_vec + inner_sub_length;
            T* inner_house_mat = inner_house_vec + inner_sub_length;
            T inner_house_beta;
            T* mat_b_res = inner_house_mat  + inner_sub_length * inner_sub_length;
            T* mat_b_tmp = mat_b_res + slice_rows * slice_cols;
            T* update_v = mat_b_tmp + slice_rows * slice_cols;
            T* mat_v_tmp = update_v + cols * cols;

            for (int i = 0; i < cols - k - 1; ++i)
                inner_input_vec[i] = mat_b[k * cols + (k + 1 + i)];

            matrix_householder_vector(inner_input_vec, inner_sub_length,
                inner_house_vec, &inner_house_beta, epsilon, norm);
            matrix_householder_matrix(inner_house_vec, inner_sub_length,
                inner_house_beta, inner_house_mat);

            /* Cut out mat_b(k:m, (k+1):n). */
            for (int i = 0; i < slice_rows; ++i)
            {
                for (int j = 0; j < slice_cols; ++j)
                {
                    mat_b_tmp[i * slice_cols + j] =
                        mat_b[(k + i) * cols + (k + 1 + j)];
                }
            }
            matrix_multiply(mat_b_tmp, slice_rows, slice_cols,
                inner_house_mat, inner_sub_length, mat_b_res);

            /* Write frame back into mat_b. */
            for (int i = 0; i < slice_rows; ++i)
            {
                for (int j = 0; j < slice_cols; ++j)
                {
                    mat_b[(k + i) * cols + (k + 1 + j)] =
                        mat_b_res[i * slice_cols + j];
                }
            }

            for (int i = k + 2; i < cols; ++i)
                mat_b[k * cols + i] = T(0);

            std::fill(update_v, update_v + cols * cols, T(0));
            for (int i = 0; i < k + 1; ++i)
                update_v[i * cols + i] = T(1);
            for (int i = 0; i < inner_sub_length; ++i)
            {
                for (int j = 0; j < inner_sub_length; ++j)
                {
                    update_v[(k + i + 1) * cols + (k + j + 1)] =
                        inner_house_mat[i * inner_sub_length + j];
                }
            }

            /* Copy matrix v for multiplication. */
            std::copy(mat_v, mat_v + cols * cols, mat_v_tmp);
            matrix_multiply(mat_v_tmp, cols, cols, update_v, cols, mat_v);
        }
    }
}

/**
 * Single step in the [GK-SVD] method.
 */
template <typename T>
void
matrix_gk_svd_step (int rows, int cols, T* mat_b, T* mat_q, T* mat_p,
    int p, int q, T const& epsilon)
{
    int const slice_length = cols - q - p;
    int const mat_sizes = slice_length * slice_length;
    std::vector<T> buffer(3 * mat_sizes);
    T* mat_b22 = &buffer[0];
    T* mat_b22_t = mat_b22 + mat_sizes;
    T* mat_tmp = mat_b22_t + mat_sizes;

    for (int i = 0; i < slice_length; ++i)
        for (int j = 0; j < slice_length; ++j)
            mat_b22[i * slice_length + j] = mat_b[(p + i) *  cols + (p + j)];
    for (int i = 0; i < slice_length; ++i)
        for (int j = 0; j < slice_length; ++j)
            mat_b22_t[i * slice_length + j] = mat_b22[j * slice_length + i];

    /* Slice outer product gives covariance matrix. */
    matrix_multiply(mat_b22, slice_length, slice_length, mat_b22_t,
        slice_length, mat_tmp);

    T mat_c[2 * 2];
    mat_c[0] = mat_tmp[(slice_length - 2) * slice_length + (slice_length - 2)];
    mat_c[1] = mat_tmp[(slice_length - 2) * slice_length + (slice_length - 1)];
    mat_c[2] = mat_tmp[(slice_length - 1) * slice_length + (slice_length - 2)];
    mat_c[3] = mat_tmp[(slice_length - 1) * slice_length + (slice_length - 1)];

    /* Use eigenvalue that is closer to the lower right entry of the slice. */
    T eig_1, eig_2;
    matrix_2x2_eigenvalues(mat_c, &eig_1, &eig_2);

    T diff1 = std::abs(mat_c[3] - eig_1);
    T diff2 = std::abs(mat_c[3] - eig_2);
    T mu = (diff1 < diff2) ? eig_1 : eig_2;

    /* Zero another entry bz applyting givens rotations. */
    int k = p;
    T alpha = mat_b[k * cols + k] * mat_b[k * cols + k] - mu;
    T beta = mat_b[k * cols + k] * mat_b[k * cols + (k + 1)];

    for (int k = p; k < cols - q - 1; ++k)
    {
        T givens_c, givens_s;
        matrix_givens_rotation(alpha, beta, &givens_c, &givens_s, epsilon);
        matrix_apply_givens_column(mat_b, cols, cols, k, k + 1, givens_c,
            givens_s);
        matrix_apply_givens_column(mat_p, cols, cols, k, k + 1, givens_c,
            givens_s);

        alpha = mat_b[k * cols + k];
        beta = mat_b[(k + 1) * cols + k];
        matrix_givens_rotation(alpha, beta, &givens_c, &givens_s, epsilon);
        internal::matrix_apply_givens_row(mat_b, cols, cols, k, k + 1,
            givens_c, givens_s);
        internal::matrix_apply_givens_column(mat_q, rows, cols, k, k + 1,
            givens_c, givens_s);

        if (k < (cols - q - 2))
        {
            alpha = mat_b[k * cols + (k + 1)];
            beta = mat_b[k * cols + (k + 2)];
        }
    }
}

template <typename T>
void
matrix_svd_clear_super_entry(int rows, int cols, T* mat_b, T* mat_q,
    int row_index, T const& epsilon)
{
    for (int i = row_index + 1; i < cols; ++i)
    {
        if (MATH_EPSILON_EQ(mat_b[row_index * cols + i], T(0), epsilon))
        {
            mat_b[row_index * cols + i] = T(0);
            break;
        }

        T norm = MATH_POW2(mat_b[row_index * cols + i])
            + MATH_POW2(mat_b[i * cols + i]);
        norm = std::sqrt(norm) * MATH_SIGN(mat_b[i * cols + i]);

        T givens_c = mat_b[i * cols + i] / norm;
        T givens_s = mat_b[row_index * cols + i] / norm;
        matrix_apply_givens_row(mat_b, cols, cols, row_index,
            i, givens_c, givens_s);
        matrix_apply_givens_column(mat_q, rows, cols, row_index,
            i, givens_c, givens_s);
    }
}

/**
 * Implementation of the [GK-SVD] method.
 */
template <typename T>
void
matrix_gk_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allocate memory for temp matrices. */
    int const mat_q_full_size = rows * rows;
    int const mat_b_full_size = rows * cols;
    int const mat_p_size = cols * cols;
    int const mat_q_size = rows * cols;
    int const mat_b_size = cols * cols;

    std::vector<T> buffer(mat_q_full_size + mat_b_full_size
        + mat_p_size + mat_q_size + mat_b_size);
    T* mat_q_full = &buffer[0];
    T* mat_b_full = mat_q_full + mat_q_full_size;
    T* mat_p = mat_b_full + mat_b_full_size;
    T* mat_q = mat_p + mat_p_size;
    T* mat_b = mat_q + mat_q_size;

    matrix_bidiagonalize(mat_a, rows, cols,
        mat_q_full, mat_b_full, mat_p, epsilon);

    /* Extract smaller matrices. */
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            mat_q[i * cols + j] = mat_q_full[i * rows + j];
        }
    }
    std::copy(mat_b_full, mat_b_full + cols * cols, mat_b);

    /* Avoid infinite loops and exit after maximum number of iterations. */
    int const max_iterations = rows * cols;
    int iteration = 0;
    while (iteration < max_iterations)
    {
        iteration += 1;

        /* Enforce exact zeros for numerical stability. */
        for (int i = 0; i < cols * cols; ++i)
        {
            T const& entry = mat_b[i];
            if (MATH_EPSILON_EQ(entry, T(0), epsilon))
                mat_b[i] = T(0);
        }

        /* GK 2a. */
        for (int i = 0; i < (cols - 1); ++i)
        {
            if (std::abs(mat_b[i * cols + (i + 1)]) <= epsilon *
                std::abs(mat_b[i * cols + i] + mat_b[(i + 1) * cols + (i + 1)]))
            {
                mat_b[i * cols + (i + 1)] = T(0);
            }
        }

        /* GK 2b. */
        /* Select q such that b33 is diagonal and blocked by zeros. */
        int q = 0;
        for (int k = 0; k < cols; ++k)
        {
            int const slice_len = k + 1;
            std::vector<T> mat_b33(slice_len * slice_len);
            for (int i = 0; i < slice_len; ++i)
            {
                for (int j = 0; j < slice_len; ++j)
                {
                    mat_b33[i * slice_len + j]
                        = mat_b[(cols - k - 1 + i) * cols + (cols - k - 1 + j)];
                }
            }

            if (matrix_is_diagonal(&mat_b33[0], slice_len, slice_len, epsilon))
            {
                if (k < cols - 1)
                {
                    if (matrix_is_submatrix_zero_enclosed(
                        mat_b, cols, k + 1, epsilon))
                    {
                        q = k + 1;
                    }
                }
                else
                {
                    q = k + 1;
                }
            }
        }

        /* Select z := n-p-q such that B22 has no zero superdiagonal entry. */
        int z = 0;
        T* mat_b22_tmp = new T[(cols - q) * (cols - q)];
        for (int k = 0; k < (cols - q); ++k)
        {
            int const slice_len = k + 1;
            for (int i = 0; i < slice_len; ++i)
            {
                for (int j = 0; j < slice_len; ++j)
                {
                    mat_b22_tmp[i * slice_len + j]
                        = mat_b[(cols - q - k - 1 + i)
                        * cols + (cols - q - k - 1 + j)];
                }
            }
            if (matrix_is_superdiagonal_nonzero(
                mat_b22_tmp, slice_len, slice_len, epsilon))
            {
                z = k + 1;
            }
        }
        delete[] mat_b22_tmp;

        int const p = cols - q - z;

        /* GK 2c. */
        if (q == cols)
            break;

        bool diagonal_non_zero = true;
        int nz = 0;
        for (nz = p; nz < (cols - q - 1); ++nz)
        {
            if (MATH_EPSILON_EQ(mat_b[nz * cols + nz], T(0), epsilon))
            {
                diagonal_non_zero = false;
                mat_b[nz * cols + nz] = T(0);
                break;
            }
        }

        if (diagonal_non_zero)
            matrix_gk_svd_step(rows, cols, mat_b, mat_q, mat_p, p, q, epsilon);
        else
            matrix_svd_clear_super_entry(rows, cols, mat_b, mat_q, nz, epsilon);
    }

    /* Create resulting matrices and vector from temporary entities. */
    std::copy(mat_q, mat_q + rows * cols, mat_u);
    std::copy(mat_p, mat_p + cols * cols, mat_v);
    for (int i = 0; i < cols; ++i)
        vec_s[i] = mat_b[i * cols + i];

    /* Correct signs. */
    for (int i = 0; i < cols; ++i)
    {
        if (vec_s[i] < epsilon)
        {
            vec_s[i] = -vec_s[i];
            for (int j = 0; j < rows; ++j)
            {
                int index = j * cols + i;
                mat_u[index] = -mat_u[index];
            }
        }
    }
}

/**
 * Implementation of the [R-SVD] method, uses [GK-SVD] as solver
 * for the reduced problem.
 */
template <typename T>
void
matrix_r_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allocate memory for temp matrices. */
    int const mat_q_size = rows * rows;
    int const mat_r_size = rows * cols;
    int const mat_u_tmp_size = rows * cols;
    std::vector<T> buffer(mat_q_size + mat_r_size + mat_u_tmp_size);
    T* mat_q = &buffer[0];
    T* mat_r = mat_q + mat_q_size;
    T* mat_u_tmp = mat_r + mat_r_size;

    matrix_qr(mat_a, rows, cols, mat_q, mat_r, epsilon);
    matrix_gk_svd(mat_r, cols, cols, mat_u_tmp, vec_s, mat_v, epsilon);
    std::fill(mat_u_tmp + cols * cols, mat_u_tmp + rows * cols, T(0));

    /* Adapt U for big matrices. */
    matrix_multiply(mat_q, rows, rows, mat_u_tmp, cols, mat_u);
}

/**
 * Returns the index of the largest eigenvalue. If all eigenvalues are
 * zero, -1 is returned.
 */
template <typename T>
int
find_largest_ev_index (T const* values, int length)
{
    T largest = T(0);
    int index = -1;
    for (int i = 0; i < length; ++i)
        if (values[i] > largest)
        {
            largest = values[i];
            index = i;
        }
    return index;
}

MATH_INTERNAL_NAMESPACE_END
MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <typename T>
void
matrix_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allow for null result matrices. */
    std::vector<T> mat_u_tmp;
    std::vector<T> vec_s_tmp;
    if (vec_s == nullptr)
    {
        vec_s_tmp.resize(cols);
        vec_s = &vec_s_tmp[0];
    }
    std::vector<T> mat_v_tmp;
    if (mat_v == nullptr)
    {
        mat_v_tmp.resize(cols * cols);
        mat_v = &mat_v_tmp[0];
    }

    /*
     * Handle two cases: The regular case, where M >= N (rows >= cols), and
     * the irregular case, where M < N (rows < cols). In the latter one,
     * zero rows are appended to A until A is a square matrix.
     */
    if (rows >= cols)
    {
        /* Allow for null result U matrix. */
        if (mat_u == nullptr)
        {
            mat_u_tmp.resize(rows * cols);
            mat_u = &mat_u_tmp[0];
        }

        /* Perform economy SVD if rows > 5/3 cols to save some operations. */
        if (rows >= 5 * cols / 3)
        {
            internal::matrix_r_svd(mat_a, rows, cols,
                mat_u, vec_s, mat_v, epsilon);
        }
        else
        {
            internal::matrix_gk_svd(mat_a, rows, cols,
                mat_u, vec_s, mat_v, epsilon);
        }
    }
    else
    {
        /* Append zero rows to A until A is square. */
        std::vector<T> mat_a_tmp(cols * cols, T(0));
        std::copy(mat_a, mat_a + cols * rows, &mat_a_tmp[0]);

        /* Temporarily resize U, allow for null result matrices. */
        mat_u_tmp.resize(cols * cols);
        internal::matrix_gk_svd(&mat_a_tmp[0], cols, cols,
            &mat_u_tmp[0], vec_s, mat_v, epsilon);

        /* Copy the result back to U leaving out the last rows. */
        if (mat_u != nullptr)
            std::copy(&mat_u_tmp[0], &mat_u_tmp[0] + rows * cols, mat_u);
        else
            mat_u = &mat_u_tmp[0];
    }

    /* Sort the eigenvalues in S and adapt the columns of U and V. */
    for (int i = 0; i < cols; ++i)
    {
        int pos = internal::find_largest_ev_index(vec_s + i, cols - i);
        if (pos < 0)
            break;
        if (pos == 0)
            continue;
        std::swap(vec_s[i], vec_s[i + pos]);
        matrix_swap_columns(mat_u, rows, cols, i, i + pos);
        matrix_swap_columns(mat_v, cols, cols, i, i + pos);
    }
}

#if 0
template <typename T>
void
matrix_svd (T const* mat_a, int rows, int cols,
    T* mat_u, T* vec_s, T* mat_v, T const& epsilon)
{
    /* Allow for null result matrices. */
    std::vector<T> mat_u_tmp;
    std::vector<T> vec_s_tmp;
    std::vector<T> mat_v_tmp;
    if (mat_u == nullptr)
    {
        mat_u_tmp.resize(rows * cols);
        mat_u = &mat_u_tmp[0];
    }
    if (vec_s == nullptr)
    {
        vec_s_tmp.resize(cols);
        vec_s = &vec_s_tmp[0];
    }
    if (mat_v == nullptr)
    {
        mat_v_tmp.resize(cols * cols);
        mat_v = &mat_v_tmp[0];
    }

    /*
     * The SVD can only handle systems where rows > cols. Thus, in case
     * of cols > rows, the input matrix is transposed and the resulting
     * solution converted to the desired solution:
     *
     *   A* = (U S V*)* = V S* U* = V S U*
     *
     * In order to output same-sized matrices for every problem, many zero
     * rows and columns can appear in systems with cols > rows.
     */
    std::vector<T> mat_a_tmp;
    bool was_transposed = false;
    if (cols > rows)
    {
        mat_a_tmp.resize(rows * cols);
        std::copy(mat_a, mat_a + rows * cols, mat_a_tmp.begin());
        matrix_transpose(&mat_a_tmp[0], rows, cols);
        std::swap(rows, cols);
        std::swap(mat_u, mat_v);
        mat_a = &mat_a_tmp[0];
        was_transposed = true;
    }

    /* Perform economy SVD if rows > 5/3 cols to save some operations. */
    if (rows >= 5 * cols / 3)
    {
        internal::matrix_r_svd(mat_a, rows, cols,
            mat_u, vec_s, mat_v, epsilon);
    }
    else
    {
        internal::matrix_gk_svd(mat_a, rows, cols,
            mat_u, vec_s, mat_v, epsilon);
    }

    if (was_transposed)
    {
        std::swap(rows, cols);
        std::swap(mat_u, mat_v);

        /* Fix S by appending zeros. */
        for (int i = rows; i < cols; ++i)
            vec_s[i] = T(0);

        /* Fix U by reshaping the matrix. */
        for (int i = rows * rows - 1, y = rows - 1; y >= 0; --y)
        {
            for (int x = rows - 1; x >= 0; --x, --i)
                mat_u[y * cols + x] = mat_u[i];
            for (int x = rows; x < cols; ++x)
                mat_u[y * cols + x] = T(0);
        }

        /* Fix V by reshaping the matrix. */
        for (int i = cols * rows - 1, y = cols - 1; y >= 0; --y)
        {
            for (int x = rows - 1; x >= 0; --x, --i)
                mat_v[y * cols + x] = mat_v[i];
            for (int x = rows; x < cols; ++x)
                mat_v[y * cols + x] = T(0);
        }
    }

    /* Sort the eigenvalues in S and adapt the columns of U and V. */
    for (int i = 0; i < cols; ++i)
    {
        int pos = internal::find_largest_ev_index(vec_s + i, cols - i);
        if (pos < 0)
            break;
        if (pos == 0)
            continue;
        std::swap(vec_s[i], vec_s[i + pos]);
        matrix_swap_columns(mat_u, rows, cols, i, i + pos);
        matrix_swap_columns(mat_v, cols, cols, i, i + pos);
    }
}
#endif

template <typename T, int M, int N>
void
matrix_svd (Matrix<T, M, N> const& mat_a, Matrix<T, M, N>* mat_u,
    Matrix<T, N, N>* mat_s, Matrix<T, N, N>* mat_v, T const& epsilon)
{
    T* mat_u_ptr = mat_u ? mat_u->begin() : nullptr;
    T* mat_s_ptr = mat_s ? mat_s->begin() : nullptr;
    T* mat_v_ptr = mat_v ? mat_v->begin() : nullptr;

    matrix_svd<T>(mat_a.begin(), M, N,
        mat_u_ptr, mat_s_ptr, mat_v_ptr, epsilon);

    /* Swap elements of S into place. */
    if (mat_s_ptr)
    {
        std::fill(mat_s_ptr + N, mat_s_ptr + N * N, T(0));
        for (int x = 1, i = N + 1; x < N; ++x, i += N + 1)
            std::swap(mat_s_ptr[x], mat_s_ptr[i]);
    }
}

template <typename T, int M, int N>
void
matrix_pseudo_inverse (Matrix<T, M, N> const& A,
    Matrix<T, N, M>* result, T const& epsilon)
{
    Matrix<T, M, N> U;
    Matrix<T, N, N> S;
    Matrix<T, N, N> V;
    matrix_svd(A, &U, &S, &V, epsilon);

    /* Invert diagonal of S. */
    for (int i = 0; i < N; ++i)
        if (MATH_EPSILON_EQ(S(i, i), T(0), epsilon))
            S(i, i) = T(0);
        else
            S(i, i) = T(1) / S(i, i);

    *result = V * S * U.transposed();
}

MATH_NAMESPACE_END

#endif /* MATH_MATRIX_SVD_HEADER */
