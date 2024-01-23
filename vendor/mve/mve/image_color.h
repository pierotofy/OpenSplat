/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_IMAGE_COLOR_HEADER
#define MVE_IMAGE_COLOR_HEADER

#include <algorithm>

#include "math/functions.h"
#include "mve/image.h"
#include "mve/defines.h"

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/**
 * Applies an in-place color conversion to the given image. The conversion
 * is performed by applying the given conversion functor to every pixel.
 * The converter functor can either be a regular function which takes
 * a pointer to the image pixel, or a conversion object which overloads
 * operator() and takes a pointer to the image pixel.
 *
 * The conversion function is expected to process pixels with 3 channels,
 * and this function checks that images have at least 3 channels.
 *
 * Note that color conversion with byte images is supported but not
 * recommended. Color conversion often produces values outside the usual
 * range and clamping occurs. Due to rounding and clampling, back and forth
 * conversion (A -> B -> A') is often very unstable (A != A').
 */
template <typename T, typename FUNCTOR>
void
color_convert (typename Image<T>::Ptr image, FUNCTOR& converter);

/* ---------------------------- Functors -------------------------- */

/**
 * Converts linear sRGB values RGB into XYZ (CIE 1931) according to
 * http://www.w3.org/Graphics/Color/sRGB
 *
 *   X    0.4124 0.3576 0.1805   R
 *   Y =  0.2126 0.7152 0.0722 * G
 *   Z    0.0193 0.1192 0.9505   B
 *
 * Warning: Conversion of byte images is supported, but clamping can occur.
 */
template <typename T>
void
color_srgb_to_xyz (T* values);

/**
 * Converts XYZ into linear sRGB values RGB according to
 * http://www.w3.org/Graphics/Color/sRGB
 *
 *   R    3.2410 -1.5374 -0.4986   X
 *   G = -0.9692  1.8760  0.0416 * Y
 *   B    0.0556 -0.2040  1.0570   Z
 *
 * Warning: Conversion of byte images is supported, but clamping can occur.
 */
template <typename T>
void
color_xyz_to_srgb (T* values);

/**
 * Converts xyY colors to XYZ (CIE 1931) coordinates according to
 * http://www.brucelindbloom.com/index.html?Eqn_xyY_to_XYZ.html
 *
 *   X = x * Y / y
 *   Y = Y
 *   Z = (1 - x - y) * Y / y
 *
 * Warning: Conversion of byte images is supported, but clamping occurs.
 * Also note that back and forth conversion with byte images is inaccurate.
 */
template <typename T>
void
color_xyy_to_xyz (T* values);

/**
 * Converts XYZ colors to xyY coordinates according to
 * http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_xyY.html
 *
 *    x = X / (X + Y + Z)
 *    y = Y / (X + Y + Z)
 *    Y = Y
 *
 * Warning: Conversion of byte images is supported, but clamping occurs.
 * Also note that back and forth conversion with byte images is inaccurate.
 */
template <typename T>
void
color_xyz_to_xyy (T* values);

/**
 * Converts an image from RGB to YCbCr color space according to
 * http://en.wikipedia.org/wiki/YCbCr
 *
 *   Y     0.299000  0.587000  0.114000   R   0.0
 *   Cb = -0.168736 -0.331264  0.500000 * G + 0.5
 *   Cr    0.500000 -0.418688 -0.081312   B   0.5
 *
 * Works with float, double and byte images.
 */
template <typename T>
void
color_rgb_to_ycbcr (T* values);

/**
 * Converts an image from YCbCr to RGB color space according to
 * http://en.wikipedia.org/wiki/YCbCr
 *
 *    R   1.00  0.00000  1.40200   ( Y  - 0.0 )
 *    G = 1.00 -0.34414 -0.71414 * ( Cb - 0.5 )
 *    B   1.00  1.77200  0.00000   ( Cr - 0.5 )
 *
 * Works with float, double and byte images.
 */
template <typename T>
void
color_ycbcr_to_rgb (T* values);

/* ------------------------- Implementation ----------------------- */

template <typename T, typename FUNCTOR>
void
color_convert (typename Image<T>::Ptr image, FUNCTOR& converter)
{
    int64_t const channels = image->channels();
    if (channels != 3)
        throw std::invalid_argument("Only 3-channel images supported");

    for (T* ptr = image->begin(); ptr != image->end(); ptr += channels)
        converter(ptr);
}

template <typename T>
void
color_srgb_to_xyz (T* v)
{
    T out[3];
    out[0] = v[0] * T(0.4124) + v[1] * T(0.3576) + v[2] * T(0.1805);
    out[1] = v[0] * T(0.2126) + v[1] * T(0.7152) + v[2] * T(0.0722);
    out[2] = v[0] * T(0.0193) + v[1] * T(0.1192) + v[2] * T(0.9505);
    std::copy_n(out, 3, v);
}

template <>
inline void
color_srgb_to_xyz<uint8_t> (uint8_t* v)
{
    double out[3];
    out[0] = v[0] * 0.4124 + v[1] * 0.3576 + v[2] * 0.1805;
    out[1] = v[0] * 0.2126 + v[1] * 0.7152 + v[2] * 0.0722;
    out[2] = v[0] * 0.0193 + v[1] * 0.1192 + v[2] * 0.9505;
    v[0] = std::max(0.0, std::min(255.0, math::round(out[0])));
    v[1] = std::max(0.0, std::min(255.0, math::round(out[1])));
    v[2] = std::max(0.0, std::min(255.0, math::round(out[2])));
}

template <typename T>
void
color_xyz_to_srgb (T* v)
{
    T out[3];
    out[0] = v[0] * T( 3.2410) + v[1] * T(-1.5374) + v[2] * T(-0.4986);
    out[1] = v[0] * T(-0.9692) + v[1] * T( 1.8760) + v[2] * T( 0.0416);
    out[2] = v[0] * T( 0.0556) + v[1] * T(-0.2040) + v[2] * T( 1.0570);
    std::copy_n(out, 3, v);
}

template <>
inline void
color_xyz_to_srgb<uint8_t> (uint8_t* v)
{
    double out[3];
    out[0] = v[0] *  3.2410 + v[1] * -1.5374 + v[2] * -0.4986;
    out[1] = v[0] * -0.9692 + v[1] *  1.8760 + v[2] *  0.0416;
    out[2] = v[0] *  0.0556 + v[1] * -0.2040 + v[2] *  1.0570;
    v[0] = std::max(0.0, std::min(255.0, math::round(out[0])));
    v[1] = std::max(0.0, std::min(255.0, math::round(out[1])));
    v[2] = std::max(0.0, std::min(255.0, math::round(out[2])));
}

template <typename T>
void
color_xyy_to_xyz (T* v)
{
    if (v[1] == T(0))
    {
        v[0] = T(0);
        v[1] = T(0);
        v[2] = T(0);
    }
    else
    {
        T const ratio = v[2] / v[1];
        T out[3];
        out[0] = v[0] * ratio;
        out[1] = v[2];
        out[2] = (T(1) - v[0] - v[1]) * ratio;
        std::copy_n(out, 3, v);
    }
}

template <>
inline void
color_xyy_to_xyz<uint8_t> (uint8_t* v)
{
    if (v[1] == 0)
    {
        v[0] = 0;
        v[1] = 0;
        v[2] = 0;
    }
    else
    {
        double const ratio = v[2] / static_cast<double>(v[1]);
        double out[3];
        out[0] = v[0] * ratio;
        out[1] = v[2];
        out[2] = (255 - v[0] - v[1]) * ratio;
        v[0] = std::max(0.0, std::min(255.0, math::round(out[0])));
        v[1] = std::max(0.0, std::min(255.0, math::round(out[1])));
        v[2] = std::max(0.0, std::min(255.0, math::round(out[2])));
    }
}

template <typename T>
void
color_xyz_to_xyy (T* v)
{
    T const sum = v[0] + v[1] + v[2];
    if (sum == T(0))
    {
        v[0] = T(0);
        v[1] = T(0);
        v[2] = T(0);
    }
    else
    {
        T out[3];
        out[0] = v[0] / sum;
        out[1] = v[1] / sum;
        out[2] = v[1];
        std::copy_n(out, 3, v);
    }
}

template <>
inline void
color_xyz_to_xyy<uint8_t> (uint8_t* v)
{
    if (v[0] == 0 && v[1] == 0 && v[2] == 0)
    {
        v[0] = 0;
        v[1] = 0;
        v[2] = 0;
    }
    else
    {
        double const sum = v[0] + v[1] + v[2];
        double out[3];
        out[0] = 255.0 * v[0] / sum;
        out[1] = 255.0 * v[1] / sum;
        out[2] = static_cast<double>(v[1]);
        v[0] = std::max(0.0, std::min(255.0, math::round(out[0])));
        v[1] = std::max(0.0, std::min(255.0, math::round(out[1])));
        v[2] = std::max(0.0, std::min(255.0, math::round(out[2])));
    }
}

template <typename T>
void
color_rgb_to_ycbcr (T* v)
{
    T out[3];
    out[0] = v[0] * T(0.299) + v[1] * T(0.587) + v[2] * T(0.114);
    out[1] = v[0] * T(-0.168736) + v[1] * T(-0.331264) + v[2] * T(0.5) + T(0.5);
    out[2] = v[0] * T(0.5) + v[1] * T(-0.418688) + v[2] * T(-0.081312) + T(0.5);
    std::copy_n(out, 3, v);
}

template <>
inline void
color_rgb_to_ycbcr<uint8_t> (uint8_t* v)
{
    double out[3];
    out[0] = v[0] * 0.299     + v[1] * 0.587     + v[2] * 0.114     + 0.0;
    out[1] = v[0] * -0.168736 + v[1] * -0.331264 + v[2] * 0.5       + 128.0;
    out[2] = v[0] * 0.5       + v[1] * -0.418688 + v[2] * -0.081312 + 128.0;
    v[0] = std::max(0.0, std::min(255.0, math::round(out[0])));
    v[1] = std::max(0.0, std::min(255.0, math::round(out[1])));
    v[2] = std::max(0.0, std::min(255.0, math::round(out[2])));
}

template <typename T>
void
color_ycbcr_to_rgb (T* v)
{
    v[1] = v[1] - T(0.5);
    v[2] = v[2] - T(0.5);

    T out[3];
    out[0] = v[0] * T(1) + v[1] * T(0) + v[2] * T(1.402);
    out[1] = v[0] * T(1) + v[1] * T(-0.34414) + v[2] * T(-0.71414);
    out[2] = v[0] * T(1) + v[1] * T(1.772) + v[2] * T(0);
    std::copy_n(out, 3, v);
}

template <>
inline void
color_ycbcr_to_rgb<uint8_t> (uint8_t* v)
{
    double out[3];
    out[0] = v[0]                            + 1.402   * (v[2] - 128.0);
    out[1] = v[0] - 0.34414 * (v[1] - 128.0) - 0.71414 * (v[2] - 128.0);
    out[2] = v[0] + 1.772   * (v[1] - 128.0);
    v[0] = std::max(0.0, std::min(255.0, math::round(out[0])));
    v[1] = std::max(0.0, std::min(255.0, math::round(out[1])));
    v[2] = std::max(0.0, std::min(255.0, math::round(out[2])));
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

#endif /* MVE_IMAGE_COLOR_HEADER */
