/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <algorithm>

#include "mve/camera.h"
#include "mve/image_tools.h"

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/*
 * ----------------------- Image conversion -----------------------
 */

FloatImage::Ptr
byte_to_float_image (ByteImage::ConstPtr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    FloatImage::Ptr img = FloatImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        float value = (float)image->at(i) / 255.0f;
        img->at(i) = std::min(1.0f, std::max(0.0f, value));
    }
    return img;
}

/* ---------------------------------------------------------------- */

DoubleImage::Ptr
byte_to_double_image (ByteImage::ConstPtr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    DoubleImage::Ptr img = DoubleImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        double value = static_cast<double>(image->at(i)) / 255.0;
        img->at(i) = std::min(1.0, std::max(0.0, value));
    }
    return img;
}

/* ---------------------------------------------------------------- */

ByteImage::Ptr
float_to_byte_image (FloatImage::ConstPtr image, float vmin, float vmax)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    ByteImage::Ptr img = ByteImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        float value = std::min(vmax, std::max(vmin, image->at(i)));
        value = 255.0f * (value - vmin) / (vmax - vmin);
        img->at(i) = static_cast<uint8_t>(value + 0.5f);
    }
    return img;
}

/* ---------------------------------------------------------------- */

ByteImage::Ptr
double_to_byte_image (DoubleImage::ConstPtr image, double vmin, double vmax)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    ByteImage::Ptr img = ByteImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        double value = std::min(vmax, std::max(vmin, image->at(i)));
        value = 255.0 * (value - vmin) / (vmax - vmin);
        img->at(i) = static_cast<uint8_t>(value + 0.5);
    }
    return img;
}

/* ---------------------------------------------------------------- */

ByteImage::Ptr
int_to_byte_image (IntImage::ConstPtr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    ByteImage::Ptr img = ByteImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        img->at(i) = math::clamp(std::abs(image->at(i)), 0, 255);
    }
    return img;
}

/* ---------------------------------------------------------------- */

ByteImage::Ptr
raw_to_byte_image (RawImage::ConstPtr image, uint16_t vmin, uint16_t vmax)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    ByteImage::Ptr img = ByteImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        uint16_t value = std::min(vmax, std::max(vmin, image->at(i)));
        value = 255.0 * static_cast<double>(value - vmin)
            / static_cast<double>(vmax - vmin);
        img->at(i) = static_cast<uint8_t>(value + 0.5);
    }
    return img;
}

/* ---------------------------------------------------------------- */

FloatImage::Ptr
raw_to_float_image (RawImage::ConstPtr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    FloatImage::Ptr img = FloatImage::create();
    img->allocate(image->width(), image->height(), image->channels());
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
    {
        float const value = static_cast<float>(image->at(i)) / 65535.0f;
        img->at(i) = std::min(1.0f, std::max(0.0f, value));
    }
    return img;
}

/* ---------------------------------------------------------------- */

void
float_image_normalize (FloatImage::Ptr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    float vmin, vmax;
    find_min_max_value<float>(image, &vmin, &vmax);
    if (vmin >= vmax)
    {
        image->fill(0.0f);
        return;
    }
    for (float* ptr = image->begin(); ptr != image->end(); ++ptr)
        *ptr = (*ptr - vmin) / (vmax - vmin);
}

/* ---------------------------------------------------------------- */

void
gamma_correct (ByteImage::Ptr image, float power)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    uint8_t lookup[256];
    for (int64_t i = 0; i < 256; ++i)
        lookup[i] = static_cast<uint8_t>(std::pow(i / 255.0f, power)
            * 255.0f + 0.5f);
    for (int64_t i = 0; i < image->get_value_amount(); ++i)
        image->at(i) = lookup[image->at(i)];
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END
