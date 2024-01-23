/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MATH_ALGO_HEADER
#define MATH_ALGO_HEADER

#include <cmath>
#include <vector>
#include <iterator>
#include <stdexcept>

#include "math/defines.h"

MATH_NAMESPACE_BEGIN
MATH_ALGO_NAMESPACE_BEGIN

/* ---------------------------- Algorithms ------------------------ */

/**
 * Algorithm that returns the ID (starting from zero at element 'first')
 * of the smallest element in range [first, last[.
 */
template <typename FwdIter>
std::size_t
min_element_id (FwdIter first, FwdIter last)
{
    if (first == last)
        throw std::invalid_argument("Invalid range");

    FwdIter lowest = first;
    std::size_t lowest_id = 0;

    for (std::size_t cnt = 1; ++first != last; ++cnt)
    {
        if (*first < *lowest)
        {
            lowest_id = cnt;
            lowest = first;
        }
    }

    return lowest_id;
}

/**
 * Algorithm that returns the ID (starting from zero at element 'first')
 * of the largest element in range [first, last[.
 */
template <typename FwdIter>
std::size_t
max_element_id (FwdIter first, FwdIter last)
{
    if (first == last)
        throw std::invalid_argument("Invalid range");

    FwdIter largest = first;
    std::size_t largest_id = 0;

    for (std::size_t cnt = 1; ++first != last; ++cnt)
    {
        if (*largest < *first)
        {
            largest_id = cnt;
            largest = first;
        }
    }

    return largest_id;
}

/**
 * Algorithm that finds the value corresponding to a key in sorted vector
 * of key-value pairs. If the key does not exist, null is returned.
 */
template <typename Key, typename Value>
Value const*
binary_search (std::vector<std::pair<Key, Value> > const& vec, Key const& key)
{
    std::size_t range1 = 0;
    std::size_t range2 = vec.size();
    while (range1 != range2)
    {
        std::size_t pos = (range1 + range2) / 2;
        if (key < vec[pos].first)
            range2 = pos;
        else if (vec[pos].first < key)
            range1 = pos + 1;
        else
            return &vec[pos].second;
    }

    return nullptr;
}

/* ---------------------- Generator functors ---------------------- */

template <typename T>
struct IncrementGenerator
{
    T state;

    IncrementGenerator (T const& init = T(0)) : state(init)
    {}

    T operator() (void)
    { return this->state++; }
};

/* ------------------- Misc: predicates, iterators, ... ----------- */

/** Squared sum accumulator. */
template <typename T>
inline T
accum_squared_sum (T const& init, T const& next)
{
    return init + next * next;
}

/** Absolute sum accumulator. */
template <typename T>
inline T
accum_absolute_sum (T const& init, T const& next)
{
    return init + std::abs(next);
}

/** Epsilon comparator predicate. */
template <typename T>
struct predicate_epsilon_equal
{
    T eps;
    predicate_epsilon_equal (T const& eps) : eps(eps) {}
    bool operator() (T const& v1, T const& v2)
    { return (v1 - eps <= v2 && v2 <= v1 + eps); }
};

/** Iterator that advances 'S' elements of type T. */
template <typename T, int S>
struct InterleavedIter : public std::iterator<std::input_iterator_tag, T>
{
    T const* pos;
    InterleavedIter (T const* pos) : pos(pos) {}
    InterleavedIter& operator++ (void) { pos += S; return *this; }
    T const& operator* (void) const { return *pos; }
    bool operator== (InterleavedIter<T,S> const& other) const
    { return pos == other.pos; }
    bool operator!= (InterleavedIter<T,S> const& other) const
    { return pos != other.pos; }
};

/* --------------------------- Vector tools ----------------------- */

/**
 * Erases all elements from 'vector' that are marked with 'true'
 * in 'delete_list'. The remaining elements are kept in order but relocated
 * to another position in the vector.
 */
template <typename T>
void
vector_clean (std::vector<bool> const& delete_list, std::vector<T>* vector)
{
    typename std::vector<T>::iterator vr = vector->begin();
    typename std::vector<T>::iterator vw = vector->begin();
    typename std::vector<bool>::const_iterator dr = delete_list.begin();

    while (vr != vector->end() && dr != delete_list.end())
    {
        if (*dr++)
        {
            vr++;
            continue;
        }
        if (vw != vr)
            *vw = *vr;
        vw++;
        vr++;
    }
    vector->erase(vw, vector->end());
}

/* ------------------------------ Misc ---------------------------- */

/**
 * Returns the kernel region (x1,y1) to (x2,y2) for a kernel of size ks
 * for image of size (width, height) and for center pixel (cx,cy).
 * The kernel size ks is the half-size, i.e. 2*ks+1 is full kernel size.
 * Values x2 and y2 are inclusive, i.e. belong to the kernel.
 */
template <typename T>
inline void
kernel_region (T const& cx, T const& cy, T const& ks,
    T const& width, T const& height, T* x1, T* x2, T* y1, T* y2)
{
    *x1 = (cx > ks ? cx - ks : T(0));
    *x2 = (cx + ks > width - T(1) ? width - T(1) : cx + ks);
    *y1 = (cy > ks ? cy - ks : T(0));
    *y2 = (cy + ks > height - T(1) ? height - T(1) : cy + ks);
}

template <typename T>
inline void
sort_values (T* a, T* b, T* c)
{
    if (*b < *a)
        std::swap(*a, *b);
    if (*c < *b)
        std::swap(*b, *c);
    if (*b < *a)
        std::swap(*b, *a);
}

/* ------------------------ for-each functors --------------------- */

/** for-each functor: multiplies operand with constant factor. */
template <typename T>
struct foreach_multiply_with_const
{
    T value;
    foreach_multiply_with_const (T const& value) : value(value) {}
    void operator() (T& val) { val *= value; }
};

/** for-each functor: divides operand by constant divisor. */
template <typename T>
struct foreach_divide_by_const
{
    T div;
    foreach_divide_by_const (T const& div) : div(div) {}
    void operator() (T& val) { val /= div; }
};

/** for-each functor: adds a constant value to operand. */
template <typename T>
struct foreach_addition_with_const
{
    T value;
    foreach_addition_with_const (T const& value) : value(value) {}
    void operator() (T& val) { val += value; }
};

/** for-each functor: substracts a constant value to operand. */
template <typename T>
struct foreach_substraction_with_const
{
    T value;
    foreach_substraction_with_const (T const& value) : value(value) {}
    void operator() (T& val) { val -= value; }
};

/** for-each functor: raises each operand to the power of constant value. */
template <typename T>
struct foreach_constant_power
{
    T value;
    foreach_constant_power (T const& value) : value(value) {}
    void operator() (T& val) { val = std::pow(val, value); }
};

/** for-each functor: matrix-vector multiplication. */
template <typename M, typename V>
struct foreach_matrix_mult
{
    M mat;
    foreach_matrix_mult (M const& matrix) : mat(matrix) {}
    void operator() (V& vec) { vec = mat * vec; }
};

/** for-each functor: applies absolute value to operand. */
template <typename T>
inline void
foreach_absolute_value (T& val)
{
    val = std::abs(val);
}

/** for-each functor: negates the operand. */
template <typename T>
inline void
foreach_negate_value (T& val)
{
    val = -val;
}

/** for-each functor: inverts floating point values with 1/value. */
template <typename T>
inline void
foreach_invert_value (T& val)
{
    val = T(1) / val;
}

/** for-each functor: applies floor operation to the operand. */
template <typename T>
inline void
foreach_floor (T& val)
{
    val = std::floor(val);
}

/** for-each functor: applies ceil operation to the operand. */
template <typename T>
inline void
foreach_ceil (T& val)
{
    val = std::ceil(val);
}

/** for-each functor: applies rounding to the operand. */
template <typename T>
inline void
foreach_round (T& val)
{
    val = round(val);
}

MATH_ALGO_NAMESPACE_END
MATH_NAMESPACE_END

#endif /* MATH_ALGO_HEADER */
