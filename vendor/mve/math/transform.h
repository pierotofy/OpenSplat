/*
 * Copyright (C) 2015, Simon Fuhrmann, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_TRANSFORM_HEADER
#define MATH_TRANSFORM_HEADER

#include "math/matrix.h"
#include "math/matrix_svd.h"

MATH_NAMESPACE_BEGIN

/*
 * Determines the transformation between two lists of corresponding points.
 * Minimizing sum_i (R * s * p0[i] + t - p1[i])^2
 */
template <typename T, int N>
bool
determine_transform(std::vector<math::Vector<T, N>> const& p0,
    std::vector<math::Vector<T, N>> const& p1,
    math::Matrix<T, N, N>* rot, T* scale, math::Vector<T, N>* trans);

MATH_NAMESPACE_END

/* ------------------------ Implementation ------------------------ */

MATH_NAMESPACE_BEGIN

template <typename T, int N>
bool
determine_transform(std::vector<math::Vector<T, N>> const& p0,
    std::vector<math::Vector<T, N>> const& p1,
    math::Matrix<T, N, N>* rot, T* scale, math::Vector<T, N>* trans)
{
    if (p0.size() != p1.size())
        throw std::invalid_argument("Dimension size mismatch");

    std::size_t num_correspondences = p0.size();
    if (num_correspondences < 3)
        throw std::invalid_argument("At least three correspondences required");

    /* Calculate centroids. */
    math::Vector<T, N> c0(0.0), c1(0.0);
    for (std::size_t i = 0; i < num_correspondences; ++i)
    {
        c0 += p0[i];
        c1 += p1[i];
    }
    c0 /= static_cast<T>(num_correspondences);
    c1 /= static_cast<T>(num_correspondences);

    /* Calculate covariance and variance. */
    T sigma2(0.0);
    math::Matrix<T, N, N> cov(0.0);
    for (std::size_t i = 0; i < num_correspondences; ++i)
    {
        math::Vector<T, N> pc0 = p0[i] - c0;
        math::Vector<T, N> pc1 = p1[i] - c1;
        cov += math::Matrix<T, N, 1>(pc0.begin()) *
            math::Matrix<T, 1, N>(pc1.begin());
        sigma2 += pc0.square_norm();
    }
    cov /= static_cast<T>(num_correspondences);
    sigma2 /= static_cast<T>(num_correspondences);

    /* Determine rotation and scale. */
    math::Matrix<T, N, N> U, S, V;
    math::matrix_svd(cov, &U, &S, &V);
    if (S(N - 1, N - 1) < T(MATH_SVD_DEFAULT_ZERO_THRESHOLD))
        return false;

    math::Matrix<T, N, N> R = V * U.transposed();
    T s = math::matrix_trace(S) / sigma2;

    /* Handle improper rotations (reflections). */
    if (math::matrix_determinant(R) < T(0.0))
    {
        math::Matrix<T, N, N> F;
        math::matrix_set_identity(&F);
        F(N - 1, N - 1) = T(-1.0);
        s = math::matrix_trace(S * F) / sigma2;
        R = V * F * U.transposed();
    }

    if (rot != nullptr)
        *rot = R;

    if (scale != nullptr)
        *scale = s;

    if (trans != nullptr)
        *trans = c1 + (-R * s * c0);

    return true;
}

MATH_NAMESPACE_END

#endif /* MATH_TRANSFORM_HEADER */
