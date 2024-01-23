/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_PERMUTE_HEADER
#define MATH_PERMUTE_HEADER

#include <vector>
#include <stdexcept>

#include "math/defines.h"

MATH_NAMESPACE_BEGIN
MATH_ALGO_NAMESPACE_BEGIN

/**
 * This function permutes a vector of elements v using a permutation
 * given by vector p, calculating v' = p(v). Vector p can be seen as
 * a mapping from old indices to new indices,
 *
 *   v'_p[i] = v_i  e.g.  v = [a, b, c], p = [1, 2, 0], v' = [c, a, b].
 *
 * which is better called a index-based relocation of the elements.
 * Each element is copied two times, from the original vector to
 * a temporary variable, and back to the vector using cycles in the
 * permutation.
 *
 * "permute_reloc" and "permute_math" are inverse to each other.
 */
template <class V, class P>
void
permute_reloc (std::vector<V>& v, std::vector<P> const& p);

/**
 * This function permutes a vector of elements v using a permutation
 * given by vector p, calculating v' = p(v). In this algorithm, p is
 * more mathematically defined and computationally is more efficient:
 *
 *   v'_i = v_p[i]  e.g.  v = [a, b, c], p = [1, 2, 0], v' = [b, c, a]
 *
 * Each element is copied only once inside the vector, and elements at
 * the beginning of a cycle are copied twice.
 *
 * "permute_reloc" and "permute_math" are inverse to each other.
 */
template <class V, class P>
void
permute_math (std::vector<V>& v, std::vector<P> const& p);

/* --------------------------- Implementation --------------------- */

template <class V, class P>
void
permute_reloc (std::vector<V>& v, std::vector<P> const& p)
{
    if (v.size() != p.size())
        throw std::invalid_argument("Vector length does not match");

    if (v.empty())
        return;

    std::vector<bool> visited(v.size(), false);
    std::size_t i = 0;
    std::size_t seek = 0;

    do
    {
        /* Permute a cycle using index-based relocation permutation. */
        V tmp[2];
        bool idx = false;
        tmp[idx] = v[i];
        while (!visited[i])
        {
            tmp[!idx] = v[p[i]];
            v[p[i]] = tmp[idx];
            idx = !idx;
            visited[i] = true;
            i = p[i];
        }

        /* Seek for an unvisited element. */
        i = MATH_MAX_SIZE_T;
        for (; seek < visited.size(); ++seek)
        if (!visited[seek])
        {
            i = seek;
            break;
        }
    }
    while (i != MATH_MAX_SIZE_T);
}

/* ---------------------------------------------------------------- */

template <class V, class P>
void
permute_math (std::vector<V>& v, std::vector<P> const& p)
{
    if (v.size() != p.size())
        throw std::invalid_argument("Vector length does not match");

    if (v.empty())
        return;

    std::vector<bool> visited(v.size(), false);
    std::size_t i = 0;
    std::size_t seek = 0;
    do
    {
        /* Permute a cycle using mathematical permutation. */
        visited[i] = true;
        if (i != p[i])
        {
            V tmp = v[i];
            while (!visited[p[i]])
            {
                v[i] = v[p[i]];
                visited[p[i]] = true;
                i = p[i];
            }
            v[i] = tmp;
        }

        /* Seek for an unvisited element. */
        i = MATH_MAX_SIZE_T;
        for (; seek < visited.size(); ++seek)
            if (!visited[seek])
            {
                i = seek;
                break;
            }
    }
    while (i != MATH_MAX_SIZE_T);
}

MATH_ALGO_NAMESPACE_END
MATH_NAMESPACE_END

#endif /* MATH_PERMUTE_HEADER */
