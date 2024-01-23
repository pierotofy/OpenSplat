/*
 * Copyright (C) 2015, Benjamin Richter, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_ALIGNED_ALLOCATOR_HEADER
#define MVE_ALIGNED_ALLOCATOR_HEADER

#include <memory>
#include <vector>
#include <limits>
#include <cstdlib>

#ifdef _WIN32
#   include <malloc.h>
#endif

#include "util/defines.h"

UTIL_NAMESPACE_BEGIN

/**
 * Implements the STL allocator interface for aligned memory allocation.
 */
template <typename T, size_t alignment>
struct AlignedAllocator
{
public:
    typedef T value_type;
    typedef T *pointer;
    typedef T const *const_pointer;
    typedef T &reference;
    typedef T const &const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    template<class U>
    struct rebind
    {
        typedef AlignedAllocator<U, alignment> other;
    };

public:
    AlignedAllocator (void);

    template <class T_other, size_t alignment_other>
    AlignedAllocator(AlignedAllocator<T_other, alignment_other> const&);

    pointer allocate (size_type n);
    void deallocate (pointer p, size_type /*n*/);
    size_type max_size (void) const;
    void construct (pointer p, const_reference other);
    void destroy (pointer p);

    template <class T_other, size_t alignment_other>
    bool operator== (AlignedAllocator<T_other, alignment_other> const&) const;

    template <class T_other, size_t alignment_other>
    bool operator!= (AlignedAllocator<T_other, alignment_other> const&) const;
};

/* ------------------------ Implementation ------------------------ */

template <typename T, size_t alignment>
inline
AlignedAllocator<T, alignment>::AlignedAllocator (void)
{
}

template <typename T, size_t alignment>
template <class T_other, size_t alignment_other>
inline
AlignedAllocator<T, alignment>::AlignedAllocator
    (AlignedAllocator<T_other, alignment_other> const& /*other*/)
{
}

template <typename T, size_t alignment>
inline typename AlignedAllocator<T, alignment>::pointer
AlignedAllocator<T, alignment>::allocate (size_type n)
{
    if (n > this->max_size())
      throw std::bad_alloc();
    size_t size = n * sizeof(T);
    pointer p = nullptr;
#ifdef _WIN32
    p = reinterpret_cast<pointer>(::_aligned_malloc(size, alignment));
    if (p == nullptr)
        throw std::bad_alloc();
#else
    if (::posix_memalign(&reinterpret_cast<void*&>(p), alignment, size))
        throw std::bad_alloc();
#endif
    return p;
}

template <typename T, size_t alignment>
inline void
AlignedAllocator<T, alignment>::deallocate(pointer p, size_type /*n*/)
{
#ifdef _WIN32
    ::_aligned_free(p);
#else
    ::free(p);
#endif
}

template <typename T, size_t alignment>
inline typename AlignedAllocator<T, alignment>::size_type
AlignedAllocator<T, alignment>::max_size (void) const
{
    return std::numeric_limits<size_t>::max() / sizeof(T);
}

template <typename T, size_t alignment>
inline void
AlignedAllocator<T, alignment>::construct (pointer p, const_reference other)
{
    ::new((void*)p) T(other);
}

template <typename T, size_t alignment>
inline void
AlignedAllocator<T, alignment>::destroy (pointer p)
{
    p->~T();
}

template <typename T, size_t alignment>
template <class T_other, size_t alignment_other>
inline bool
AlignedAllocator<T, alignment>::operator==
    (AlignedAllocator<T_other, alignment_other> const&) const
{
    return true;
}

template <typename T, size_t alignment>
template <class T_other, size_t alignment_other>
inline bool
AlignedAllocator<T, alignment>::operator!=
    (AlignedAllocator<T_other, alignment_other> const&) const
{
    return false;
}

UTIL_NAMESPACE_END

#endif // MVE_ALIGNED_ALLOCATOR_HEADER
