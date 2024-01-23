/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_IMAGE_HEADER
#define MVE_IMAGE_HEADER

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "math/functions.h"
#include "mve/defines.h"
#include "mve/image_base.h"

MVE_NAMESPACE_BEGIN

template <typename T> class Image;
typedef Image<uint8_t> ByteImage;
typedef Image<uint16_t> RawImage;
typedef Image<char> CharImage;
typedef Image<float> FloatImage;
typedef Image<double> DoubleImage;
typedef Image<int> IntImage;

/**
 * Multi-channel image class of arbitrary but homogenous data type.
 * Image data is interleaved, i.e. "RGBRGB...", not planar "RR..GG..BB..".
 */
template <typename T>
class Image : public TypedImageBase<T>
{
public:
    typedef std::shared_ptr<Image<T> > Ptr;
    typedef std::shared_ptr<Image<T> const> ConstPtr;
    typedef std::vector<T> ImageData;
    typedef T ValueType;

public:
    /** Default constructor creates an empty image. */
    Image (void) = default;

    /** Allocating constructor. */
    Image (int64_t width, int64_t height, int64_t channels);

    /** Copy constructor. */
    Image (Image<T> const& other);

    /** Smart pointer image constructor. */
    static Ptr create (void);
    /** Allocating smart pointer image constructor. */
    static Ptr create (int64_t width, int64_t height, int64_t channels);
    /** Smart pointer image copy constructor. */
    static Ptr create (Image<T> const& other);

    /** Duplicates the image. */
    Ptr duplicate (void) const;

    /** Fills every pixel of the image with the given color. */
    void fill_color (T const* color);

    /** Adds 'amount' channels to the back with default value 'value'. */
    void add_channels (int64_t amount, T const& value = T(0));
    /** Swaps channels 'c1' and 'c2'. */
    void swap_channels (int64_t c1, int64_t c2);
    /** Copies channel from src to dest. "-1" for dest creates new channel. */
    void copy_channel (int64_t src, int64_t dest);
    /** Deletes a channel from the image. */
    void delete_channel (int64_t channel);

    /** Linear indexing of image data. */
    T const& at (int64_t index) const;
    /** Linear indexing of channel data. */
    T const& at (int64_t index, int64_t channel) const;
    /** 2D indexing of image data, more expensive. */
    T const& at (int64_t x, int64_t y, int64_t channel) const;

    /** Linear indexing of image data. */
    T& at (int64_t index);
    /** Linear indexing of channel data. */
    T& at (int64_t index, int64_t channel);
    /** 2D indexing of image data, more expensive. */
    T& at (int64_t x, int64_t y, int64_t channel);

    /** Linear interpolation (more expensive) for a single color channel. */
    T linear_at (float x, float y, int64_t channel) const;

    /**
     * Linear interpolation (more expensive) for all color channels.
     * The method generates one value for each color channel and places
     * the result in the buffer provided by 'px'.
     */
    void linear_at (float x, float y, T* px) const;

    // TODO operators for data access (operators & refptr?)
    T& operator[] (int64_t index);
    T const& operator[] (int64_t index) const;

    T const& operator() (int64_t index) const;
    T const& operator() (int64_t index, int64_t channel) const;
    T const& operator() (int64_t x, int64_t y, int64_t channel) const;
    T& operator() (int64_t index);
    T& operator() (int64_t index, int64_t channel);
    T& operator() (int64_t x, int64_t y, int64_t channel);
};

MVE_NAMESPACE_END

/* ---------------------------------------------------------------- */

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/**
 * Creates an image instance for a given type.
 */
ImageBase::Ptr
create_for_type (ImageType type, int64_t width, int64_t height, int64_t chans);

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

/* ------------------------- Implementation ----------------------- */

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

inline ImageBase::Ptr
create_for_type (ImageType type, int64_t width, int64_t height, int64_t chans)
{
    switch (type)
    {
        case IMAGE_TYPE_UINT8:
            return Image<uint8_t>::create(width, height, chans);
        case IMAGE_TYPE_UINT16:
            return Image<uint16_t>::create(width, height, chans);
        case IMAGE_TYPE_UINT32:
            return Image<uint32_t>::create(width, height, chans);
        case IMAGE_TYPE_UINT64:
            return Image<uint64_t>::create(width, height, chans);
        case IMAGE_TYPE_SINT8:
            return Image<int8_t>::create(width, height, chans);
        case IMAGE_TYPE_SINT16:
            return Image<int16_t>::create(width, height, chans);
        case IMAGE_TYPE_SINT32:
            return Image<int32_t>::create(width, height, chans);
        case IMAGE_TYPE_SINT64:
            return Image<int64_t>::create(width, height, chans);
        case IMAGE_TYPE_FLOAT:
            return Image<float>::create(width, height, chans);
        case IMAGE_TYPE_DOUBLE:
            return Image<double>::create(width, height, chans);
        default:
            break;
    }

    return ImageBase::Ptr(nullptr);
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

/* ------------------------- Implementation ----------------------- */

MVE_NAMESPACE_BEGIN

template <typename T>
inline
Image<T>::Image (int64_t width, int64_t height, int64_t channels)
{
    this->allocate(width, height, channels);
}

template <typename T>
inline
Image<T>::Image (Image<T> const& other)
    : TypedImageBase<T>(other)
{
}

template <typename T>
inline typename Image<T>::Ptr
Image<T>::create (void)
{
    return Ptr(new Image<T>());
}

template <typename T>
inline typename Image<T>::Ptr
Image<T>::create (int64_t width, int64_t height, int64_t channels)
{
    return Ptr(new Image<T>(width, height, channels));
}

template <typename T>
inline typename Image<T>::Ptr
Image<T>::create (Image<T> const& other)
{
    return Ptr(new Image<T>(other));
}

template <typename T>
inline typename Image<T>::Ptr
Image<T>::duplicate (void) const
{
    return Ptr(new Image<T>(*this));
}

template <typename T>
inline void
Image<T>::fill_color (T const* color)
{
    for (T* ptr = this->begin(); ptr != this->end(); ptr += this->c)
        std::copy_n(color, this->c, ptr);
}

template <typename T>
/*inline*/ void
Image<T>::add_channels (int64_t num_channels, T const& value)
{
    if (num_channels <= 0 || !this->valid())
        return;

    std::vector<T> tmp(this->w * this->h * (this->c + num_channels));
    typename std::vector<T>::iterator dest_ptr = tmp.end();
    typename std::vector<T>::const_iterator src_ptr = this->data.end();
    const int64_t pixels = this->get_pixel_amount();
    for (int64_t p = 0; p < pixels; ++p)
    {
        for (int64_t i = 0; i < num_channels; ++i)
            *(--dest_ptr) = value;
        for (int64_t i = 0; i < this->c; ++i)
            *(--dest_ptr) = *(--src_ptr);
    }

    this->c += num_channels;
    std::swap(this->data, tmp);
}

template <typename T>
void
Image<T>::swap_channels (int64_t c1, int64_t c2)
{
    if (!this->valid() || c1 == c2
        || c1 >= this->channels() || c2 >= this->channels())
        return;

    T* iter1 = &this->data[0] + c1;
    T* iter2 = &this->data[0] + c2;
    int64_t pixels = this->get_pixel_amount();
    for (int64_t i = 0; i < pixels; ++i, iter1 += this->c, iter2 += this->c)
        std::swap(*iter1, *iter2);
}

template <typename T>
void
Image<T>::copy_channel (int64_t src, int64_t dest)
{
    if (!this->valid() || src == dest)
        return;

    if (dest < 0)
    {
        dest = this->channels();
        this->add_channels(1);
    }

    T const* src_iter = &this->data[0] + src;
    T* dst_iter = &this->data[0] + dest;
    int64_t pixels = this->get_pixel_amount();
    for (int64_t i = 0; i < pixels;
        ++i, src_iter += this->c, dst_iter += this->c)
    {
        *dst_iter = *src_iter;
    }
}

template <typename T>
void
Image<T>::delete_channel (int64_t chan)
{
    if (chan < 0 || chan >= this->channels())
        return;

    typename std::vector<T>::iterator src_iter = this->data.begin();
    typename std::vector<T>::iterator dst_iter = this->data.begin();
    for (int64_t i = 0; src_iter != this->data.end(); ++i)
    {
        if (i % this->c == chan)
            src_iter++;
        else
            *(dst_iter++) = *(src_iter++);
    }
    this->resize(this->width(), this->height(), this->channels() - 1);
}

template <typename T>
inline T const&
Image<T>::at (int64_t index) const
{
    return this->data[index];
}

template <typename T>
inline T const&
Image<T>::at (int64_t index, int64_t channel) const
{
    int64_t off = index * this->channels() + channel;
    return this->data[off];
}

template <typename T>
inline T const&
Image<T>::at (int64_t x, int64_t y, int64_t channel) const
{
    int64_t off = channel + this->channels() * (x + y * this->width());
    return this->data[off];
}

template <typename T>
inline T&
Image<T>::at (int64_t index)
{
    return this->data[index];
}

template <typename T>
inline T&
Image<T>::at (int64_t index, int64_t channel)
{
    int64_t off = index * this->channels() + channel;
    return this->data[off];
}

template <typename T>
inline T&
Image<T>::at (int64_t x, int64_t y, int64_t channel)
{
    int64_t off = channel + this->channels() * (x + y * this->width());
    return this->data[off];
}

template <typename T>
inline T&
Image<T>::operator[] (int64_t index)
{
    return this->data[index];
}

template <typename T>
inline T const&
Image<T>::operator[] (int64_t index) const
{
    return this->data[index];
}

template <typename T>
inline T const&
Image<T>::operator() (int64_t index) const
{
    return this->at(index);
}

template <typename T>
inline T const&
Image<T>::operator() (int64_t index, int64_t channel) const
{
    return this->at(index, channel);
}

template <typename T>
inline T const&
Image<T>::operator() (int64_t x, int64_t y, int64_t channel) const
{
    return this->at(x, y, channel);
}

template <typename T>
inline T&
Image<T>::operator() (int64_t index)
{
    return this->at(index);
}

template <typename T>
inline T&
Image<T>::operator() (int64_t index, int64_t channel)
{
    return this->at(index, channel);
}

template <typename T>
inline T&
Image<T>::operator() (int64_t x, int64_t y, int64_t channel)
{
    return this->at(x, y, channel);
}

template <typename T>
T
Image<T>::linear_at (float x, float y, int64_t channel) const
{
    x = std::max(0.0f, std::min(static_cast<float>(this->w - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(this->h - 1), y));

    int64_t const floor_x = static_cast<int64_t>(x);
    int64_t const floor_y = static_cast<int64_t>(y);
    int64_t const floor_xp1 = std::min(floor_x + 1, this->w - 1);
    int64_t const floor_yp1 = std::min(floor_y + 1, this->h - 1);

    float const w1 = x - static_cast<float>(floor_x);
    float const w0 = 1.0f - w1;
    float const w3 = y - static_cast<float>(floor_y);
    float const w2 = 1.0f - w3;

    int64_t const rowstride = this->w * this->c;
    int64_t const row1 = floor_y * rowstride;
    int64_t const row2 = floor_yp1 * rowstride;
    int64_t const col1 = floor_x * this->c;
    int64_t const col2 = floor_xp1 * this->c;

    return math::interpolate<T>
        (this->at(row1 + col1 + channel), this->at(row1 + col2 + channel),
        this->at(row2 + col1 + channel), this->at(row2 + col2 + channel),
        w0 * w2, w1 * w2, w0 * w3, w1 * w3);
}

template <typename T>
void
Image<T>::linear_at (float x, float y, T* px) const
{
    x = std::max(0.0f, std::min(static_cast<float>(this->w - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(this->h - 1), y));

    int64_t const floor_x = static_cast<int64_t>(x);
    int64_t const floor_y = static_cast<int64_t>(y);
    int64_t const floor_xp1 = std::min(floor_x + 1, this->w - 1);
    int64_t const floor_yp1 = std::min(floor_y + 1, this->h - 1);

    float const w1 = x - static_cast<float>(floor_x);
    float const w0 = 1.0f - w1;
    float const w3 = y - static_cast<float>(floor_y);
    float const w2 = 1.0f - w3;

    int64_t const rowstride = this->w * this->c;
    int64_t const row1 = floor_y * rowstride;
    int64_t const row2 = floor_yp1 * rowstride;
    int64_t const col1 = floor_x * this->c;
    int64_t const col2 = floor_xp1 * this->c;

    /* Copy interpolated channel values to output buffer. */
    for (int64_t cc = 0; cc < this->c; ++cc)
    {
        px[cc] = math::interpolate<T>
            (this->at(row1 + col1 + cc), this->at(row1 + col2 + cc),
            this->at(row2 + col1 + cc), this->at(row2 + col2 + cc),
            w0 * w2, w1 * w2, w0 * w3, w1 * w3);
    }
}

MVE_NAMESPACE_END

/* ---------------------------------------------------------------- */

STD_NAMESPACE_BEGIN

/** Specialization of std::swap for efficient image swapping. */
template <class T>
inline void
swap (mve::Image<T>& a, mve::Image<T>& b)
{
    a.swap(b);
}

STD_NAMESPACE_END

#endif /* MVE_IMAGE_HEADER */
