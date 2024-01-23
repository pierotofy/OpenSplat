/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_IMAGE_TOOLS_HEADER
#define MVE_IMAGE_TOOLS_HEADER

#include <iostream>
#include <limits>
#include <complex>
#include <type_traits>

#include "util/exception.h"
#include "math/accum.h"
#include "math/functions.h"
#include "mve/defines.h"
#include "mve/image.h"

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/*
 * ----------------------- Image conversions ---------------------
 */

/**
 * Converts a given byte image to a float image.
 * This is done by scaling from [0, 255] to [0, 1].
 */
FloatImage::Ptr
byte_to_float_image (ByteImage::ConstPtr image);

/** Converts a given byte image to a double image.
 * This is done by scaling from [0, 255] to [0, 1].
 */
DoubleImage::Ptr
byte_to_double_image (ByteImage::ConstPtr image);

/**
 * Converts a given float image to a byte image.
 * This is done by clamping values to [vmin, vmax] and scaling to [0, 255].
 */
ByteImage::Ptr
float_to_byte_image (FloatImage::ConstPtr image,
    float vmin = 0.0f, float vmax = 1.0f);

/**
 * Converts a given double image to a byte image.
 * This is done by clamping values to [vmin, vmax] and scaling to [0, 255].
 */
ByteImage::Ptr
double_to_byte_image (DoubleImage::ConstPtr image,
    double vmin = 0.0, double vmax = 1.0);

/**
 * Convertes a given int image to a byte image.
 * Conversion is done by clamping absolute values.
 */
ByteImage::Ptr
int_to_byte_image (IntImage::ConstPtr image);

/**
 * Converts a given raw image to a byte image.
 * This is done by clamping values to [vmin, vmax] and scaling to [0, 255].
 */
ByteImage::Ptr
raw_to_byte_image (RawImage::ConstPtr image,
    uint16_t vmin = 0, uint16_t vmax = 65535);

/**
 * Converts a given raw image to a float image.
 * This is done by scaling from [0, 65535] to [0, 1].
 */
FloatImage::Ptr
raw_to_float_image(RawImage::ConstPtr image);

/**
 * Generic conversion between image types without scaling or clamping.
 * This is useful to convert between float and double.
 */
template <typename SRC, typename DST>
typename Image<DST>::Ptr
type_to_type_image (typename Image<SRC>::ConstPtr image);

/**
 * Finds the smallest and largest value in the given image.
 */
template <typename T>
void
find_min_max_value (typename mve::Image<T>::ConstPtr image, T* vmin, T* vmax);

/**
 * Normalizes a float image IN-PLACE such that all values are [0, 1].
 * This is done by first finding the largest value in the image
 * and dividing all image values by the largest value.
 */
void
float_image_normalize (FloatImage::Ptr image);

/*
 * ----------------------- Image undistortion ---------------------
 */

/**
 * Undistorts the input image given the two undistortion parameters.
 * If both distortion parameters are equal, undistortion has no effect.
 * This distortion model is used by Microsoft's Photosynther and is
 * independent of the focal length.
 */
template <typename T>
typename Image<T>::Ptr
image_undistort_msps (typename Image<T>::ConstPtr img, double k0, double k1);

/**
 * Undistorts the input image given the focal length of the image and
 * two undistortion parameters. The distortion model is computed as:
 * rd(r) = 1 + k_2 r^2 + k_4 r^4  with  r^2 = x^2 + y^2. The focal length
 * is expected to be in unit format. This distortion model is used by MVE
 * and Noah's bundler.
 */
template <typename T>
typename Image<T>::Ptr
image_undistort_k2k4 (typename Image<T>::ConstPtr img,
    double focal_length, double k2, double k4);

/**
 * Undistorts the input image given the focal length of the image and
 * a single distortion parameter. If the distortion parameter is 0, the
 * undistortion has no effect. The focal length is expected to be in
 * unit format. This distortion model is used by VisualSfM.
 */
template <typename T>
typename Image<T>::Ptr
image_undistort_vsfm (typename Image<T>::ConstPtr img,
    double focal_length, double k1);

/*
 * ------------------- Image scaling and cropping -----------------
 */

/**
 * Returns a sub-image by cropping against a rectangular region.
 * Region may exceed the input image dimensions, new pixel values
 * are initialized with the given color.
 */
template <typename T>
typename Image<T>::Ptr
crop (typename Image<T>::ConstPtr image, int64_t width, int64_t height,
    int64_t left, int64_t top, T const* fill_color);

/** Rescale interpolation type. */
enum RescaleInterpolation
{
    RESCALE_NEAREST,
    RESCALE_LINEAR,
    RESCALE_GAUSSIAN ///< Not suited for byte images
    //RESCALE_CUBIC // Not implemented
};

/**
 * Returns a rescaled version of 'image' with dimensions 'width'
 * times 'height' using 'interp' for value interpolation. Set one of
 * 'width' or 'height' to 0 to keep aspect ratio. Mipmap reduction
 * is applied if the image is rescaled with size factor < 1/2.
 */
template <typename T>
typename Image<T>::Ptr
rescale (typename Image<T>::ConstPtr image, RescaleInterpolation interp,
    int64_t width, int64_t height);

/**
 * Returns a rescaled version of image, scaled by factor 1/2, by grouping
 * blocks of 2x2 pixel into one pixel in the new image. If the image size
 * is odd, the new size is computed as new_size = (old_size + 1) / 2.
 */
template <typename T>
typename Image<T>::Ptr
rescale_half_size (typename Image<T>::ConstPtr image);

/**
 * Returns a rescaled version of the image, scaled with a gaussian
 * approximation by factor 1/2. Note that the kernel size if fixed to
 * 4x4 pixels, so sigma shouldn't be much larger than 1. The default sigma
 * is sqrt(1.0^2 - 0.5^2) =~ 0.866 to double the inherent sigma of 0.5.
 */
template <typename T>
typename Image<T>::Ptr
rescale_half_size_gaussian (typename Image<T>::ConstPtr image,
    float sigma = 0.866025403784439f);

/**
 * Returns a rescaled version of the image by subsampling every second
 * column and row. Useful if the original image already has appropriate blur.
 */
template <typename T>
typename Image<T>::Ptr
rescale_half_size_subsample (typename Image<T>::ConstPtr image);

/**
 * Returns a rescaled version of the image, upscaled with linear
 * interpolation by factor 2. In this version, only interpolated values
 * are used to construct the new image. This is eliminates some original
 * information but preserves the dimension/appearance of the image.
 */
template <typename T>
typename Image<T>::Ptr
rescale_double_size (typename Image<T>::ConstPtr img);

/**
 * Returns a rescaled version of the image, upscaled with linear
 * interpolation. Every second row and column is directly taken.
 * This technique preserves all information (i.e. the original image
 * can be recovered using rescale_half_size_subsample), but introduces
 * a half pixel shift towards the upper left corner.
 */
template <typename T>
typename Image<T>::Ptr
rescale_double_size_supersample (typename Image<T>::ConstPtr img);

/**
 * Rescales image 'in' using nearest neighbor. The new images is scaled to
 * the dimensions of 'out', placing the result in 'out'.
 * When downsampling, requires 'in' to be at appropriate mipmap level
 * to avoid aliasing artifacts.
 */
template <typename T>
void
rescale_nearest (typename Image<T>::ConstPtr in, typename Image<T>::Ptr out);

/**
 * Rescales image 'in' using linear interpolation. The new images is
 * scaled to the dimensions of 'out', placing the result in 'out'.
 * When downsampling, requires 'in' to be at appropriate mipmap level
 * to avoid aliasing artifacts.
 */
template <typename T>
void
rescale_linear (typename Image<T>::ConstPtr in, typename Image<T>::Ptr out);

/**
 * Rescales image 'in' using a gaussian kernel mask. The new image is
 * scaled to the dimension of 'out', placing the result in 'out'.
 * A smaller sigma factor produces more crisp results but aliased results,
 * whereas a larger sigma factor produces smoother but blurred results.
 * Warning: This function is terribly slow due to naive implementation.
 */
template <typename T>
void
rescale_gaussian (typename Image<T>::ConstPtr in,
    typename Image<T>::Ptr out, float sigma_factor = 1.0f);

/*
 * ------------------------- Image blurring --------------------------
 */

/**
 * Blurs the image using a gaussian convolution kernel.
 * The implementation exploits kernel separability.
 */
template <typename T>
typename Image<T>::Ptr
blur_gaussian (typename Image<T>::ConstPtr in, float sigma);

/**
 * Blurs the image using a box filter of integer size 'ks'.
 * The implementaion is separated, and much faster than Gaussian blur,
 * but of inferior blur quality (usual box filter artifacts).
 */
template <typename T>
typename Image<T>::Ptr
blur_boxfilter (typename Image<T>::ConstPtr in, int ks);

/*
 * ------------------- Image rotation and flipping -------------------
 */

/** Image rotation type. */
enum RotateType
{
    ROTATE_CCW, ///< Counter-clock wise rotation
    ROTATE_CW, ///< Clock wise rotation
    ROTATE_180, ///< 180 degree rotation
    ROTATE_SWAP ///< Exchanges x- and y-axis
};

/**
 * Returns a rotated copy of the given image either rotated clock wise,
 * counter-clock wise, with 180 degree or with swapped x- and y-axis.
 * Swapping x- and y-axis is not a real rotation but rather transposing.
 * Note: In-place not supported because CW, CCW rotation and swap changes
 * image dimensions. An in-place version of ROTATE_180 is available using
 * in-place flipping in both directions.
 */
template <typename T>
typename Image<T>::Ptr
rotate (typename Image<T>::ConstPtr image, RotateType type);

/**
 * Returns an image created by rotating the input image by the given amount
 * of degrees (in radian) in clock-wise direction. The size of the output
 * image is the same as the input image, pixels may be rotated outside
 * the view port and new pixels are filled with the provided fill color.
 */
template <typename T>
typename Image<T>::Ptr
rotate (typename Image<T>::ConstPtr image, float angle, T const* fill_color);

/** Image flipping type. */
enum FlipType
{
    FLIP_NONE = 0,
    FLIP_HORIZONTAL = 1 << 0,
    FLIP_VERTICAL = 1 << 1,
    FLIP_BOTH = FLIP_HORIZONTAL | FLIP_VERTICAL
};

/**
 * Flips the given image either horizontally, vertically or both IN-PLACE.
 */
template <typename T>
void
flip (typename Image<T>::Ptr image, FlipType type);

/*
 * ----------------------- Image desaturation ------------------------
 */

/** Desaturaturation type. */
enum DesaturateType
{
    DESATURATE_MAXIMUM, ///< max(R,G,B)
    DESATURATE_LIGHTNESS, ///< (max(R,G,B) + min(R,G,B)) * 1/2
    DESATURATE_LUMINOSITY, ///< 0.21 * R + 0.72 * G + 0.07 * B
    DESATURATE_LUMINANCE, ///< 0.30 * R + 0.59 * G + 0.11 * B
    DESATURATE_AVERAGE ///< (R + G + B) * 1/3
};

/**
 * Desaturates an RGB or RGBA image to G or GA respectively.
 * A new image is returned, the original image is untouched.
 *
 * From http://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
 *
 * Maximum = max(R,G,B)
 * Lightness = 1/2 * (max(R,G,B) + min(R,G,B))
 * Luminosity = 0.21 * R + 0.72 * G + 0.07 * B
 * Luminance = 0.30 * R + 0.59 * G + 0.11 * B
 * Average Brightness = 1/3 * (R + G + B)
 */
template <typename T>
typename Image<T>::Ptr
desaturate (typename Image<T>::ConstPtr image, DesaturateType type);

/**
 * Expands a gray image (one or two channels) to an RGB or RGBA image.
 */
template <typename T>
typename Image<T>::Ptr
expand_grayscale (typename Image<T>::ConstPtr image);

/** Reduce alpha: Reduces RGBA or GA images to RGB or G images. */
template <typename T>
void
reduce_alpha (typename mve::Image<T>::Ptr img); // TODO Blend color?

/* ------------------------- Edge detection ----------------------- */

/**
 * Implementation of the Sobel operator.
 * For details, see http://en.wikipedia.org/wiki/Sobel_operator
 * For byte images, the operation can lead to clipped values.
 * Likewise for floating point images, it leads to values >1.
 */
template <typename T>
typename mve::Image<T>::Ptr
sobel_edge (typename mve::Image<T>::ConstPtr img);

/* ------------------------- Miscellaneous ------------------------ */

/**
 * Subtracts two images to create the signed difference between the values.
 * This does not work for unsigned image types, use image difference instead.
 */
template <typename T>
typename Image<T>::Ptr
subtract (typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2);

/**
 * Creates a difference image by computing the absolute difference per value.
 * This works for unsigned image types but discards the sign.
 * The function requires 'operator<' to be defined on the image value type.
 */
template <typename T>
typename Image<T>::Ptr
difference (typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2);

/**
 * Applies gamma correction to float/double images (in-place).
 * To obtain color values from linear intensities, use 1/2.2 as exponent.
 * To remove gamma correction from an image, use 2.2 as exponent.
 */
template <typename T>
void
gamma_correct (typename Image<T>::Ptr image, T const& power);

/**
 * Applies fast gamma correction to byte image using a lookup table.
 * Note that alpha channels are not treated as such and are also corrected!
 */
void
gamma_correct (ByteImage::Ptr image, float power);

/**
 * Applies gamma correction to float/double images (in-place) with linear
 * RGB values in range [0, 1] to nonlinear R'G'B' values according to
 * http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html:
 *
 *   X' = 12.92 * X                   if X <= 0.0031308
 *   X' = 1.055 * X^(1/2.4) - 0.055   otherwise
 *
 * TODO: Implement overloading for integer image types.
 */
template <typename T>
void
gamma_correct_srgb (typename Image<T>::Ptr image);

/**
 * Applies inverse gamma correction to float/double (in-place) images with
 * nonlinear R'G'B' values in the range [0, 1] to linear sRGB values according
 * to http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html:
 *
 *   X = X' / 12.92                     if X' <= 0.04045
 *   X = ((X' + 0.055) / (1.055))^2.4   otherwise
 *
 * TODO: Implement overloading for integer image types.
 */
template <typename T>
void
gamma_correct_inv_srgb (typename Image<T>::Ptr image);

/**
 * Calculates the integral image (or summed area table) for the input image.
 * The integral image is computed channel-wise, i.e. the output image has
 * the same amount of channels as the input image.
 */
template <typename T_IN, typename T_OUT>
typename Image<T_OUT>::Ptr
integral_image (typename Image<T_IN>::ConstPtr image);

/**
 * Sums over the rectangle defined by A=(x1,y1) and B=(x2,y2) on the given
 * SAT for channel cc. This is efficiently calculated as B + A - C - D
 * where C and D are the other two points of the rectange. The points
 * A and B are considered to be INSIDE the rectangle.
 */
template <typename T>
T
integral_image_area (typename Image<T>::ConstPtr sat,
    int64_t x1, int64_t y1, int64_t x2, int64_t y2, int64_t cc = 0);

/**
 * Creates a thumbnail of the given size by first rescaling the image
 * and then cropping to fill the thumbnail.
 */
template <typename T>
typename Image<T>::Ptr
create_thumbnail (typename Image<T>::ConstPtr image,
    int64_t thumb_width, int64_t thumb_height);

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

/* ------------------------- Implementation ----------------------- */

#include <cmath>
#include <stdexcept>

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

template <typename SRC, typename DST>
typename Image<DST>::Ptr
type_to_type_image (typename Image<SRC>::ConstPtr image)
{
    typename Image<DST>::Ptr out = Image<DST>::create();
    out->allocate(image->width(), image->height(), image->channels());
    int64_t size = image->get_value_amount();
    SRC const* src_buf = image->get_data_pointer();
    DST* dst_buf = out->get_data_pointer();
    std::copy_n(src_buf, size, dst_buf);
    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
void
find_min_max_value (typename mve::Image<T>::ConstPtr image, T* vmin, T* vmax)
{
    *vmin = std::numeric_limits<T>::max();
    *vmax = std::numeric_limits<T>::is_signed
        ? -std::numeric_limits<T>::max()
        : std::numeric_limits<T>::min();

    for (T const* ptr = image->begin(); ptr != image->end(); ++ptr)
    {
        if (*ptr < *vmin)
            *vmin = *ptr;
        if (*ptr > *vmax)
            *vmax = *ptr;
    }
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rescale (typename Image<T>::ConstPtr img, RescaleInterpolation interp,
    int64_t width, int64_t height)
{
    if (width == 0 && height == 0)
        throw std::invalid_argument("Invalid size request");

    /* Duplicate input image if width and height match image size. */
    if (width == img->width() && height == img->height())
        return Image<T>::create(*img);

    /* Keep aspect ratio if one of width or height is 0. */
    if (width == 0)
        width = height * img->width() / img->height();
    else if (height == 0)
        height = width * img->height() / img->width();

    /* Scale down to an appropriate mipmap level for resizing. */
    if (interp == RESCALE_NEAREST || interp == RESCALE_LINEAR)
    {
        while (2 * width <= img->width() && 2 * height <= img->height())
            img = rescale_half_size<T>(img);
    }

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(width, height, img->channels());

    switch (interp)
    {
        case RESCALE_NEAREST:
            rescale_nearest<T>(img, out);
            break;
        case RESCALE_LINEAR:
            rescale_linear<T>(img, out);
            break;
#if 0
        case RESCALE_CUBIC:
            rescale_cubic(img, out);
            break;
#endif
        case RESCALE_GAUSSIAN:
            rescale_gaussian<T>(img, out);
            break;

        default:
            throw std::invalid_argument("Invalid interpolation type");
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rescale_half_size (typename Image<T>::ConstPtr img)
{
    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = (iw + 1) >> 1;
    int64_t const oh = (ih + 1) >> 1;

    if (iw < 2 || ih < 2)
        throw std::invalid_argument("Input image too small for half-sizing");

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow, oh, ic);

    int64_t outpos = 0;
    int64_t rowstride = iw * ic;
    for (int64_t y = 0; y < oh; ++y)
    {
        int64_t irow1 = y * 2 * rowstride;
        int64_t irow2 = irow1 + rowstride * (y * 2 + 1 < ih);

        for (int64_t x = 0; x < ow; ++x)
        {
            int64_t ipix1 = irow1 + x * 2 * ic;
            int64_t ipix2 = irow2 + x * 2 * ic;
            int64_t hasnext = (x * 2 + 1 < iw);

            for (int64_t c = 0; c < ic; ++c)
                out->at(outpos++) = math::interpolate<T>(
                    img->at(ipix1 + c), img->at(ipix1 + ic * hasnext + c),
                    img->at(ipix2 + c), img->at(ipix2 + ic * hasnext + c),
                    0.25f, 0.25f, 0.25f, 0.25f);
        }
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rescale_half_size_gaussian (typename Image<T>::ConstPtr img, float sigma)
{
    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = (iw + 1) >> 1;
    int64_t const oh = (ih + 1) >> 1;

    if (iw < 2 || ih < 2)
        throw std::invalid_argument("Invalid input image");

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow, oh, ic);

    /*
     * Weights w1 (4 center px), w2 (8 skewed px) and w3 (4 corner px).
     * Weights can be normalized by dividing with (4*w1 + 8*w2 + 4*w3).
     * Since the accumulator is used, normalization is not explicit.
     */
    float const w1 = std::exp(-0.5f / (2.0f * MATH_POW2(sigma)));
    float const w2 = std::exp(-2.5f / (2.0f * MATH_POW2(sigma)));
    float const w3 = std::exp(-4.5f / (2.0f * MATH_POW2(sigma)));

    int64_t outpos = 0;
    int64_t const rowstride = iw * ic;
    for (int64_t y = 0; y < oh; ++y)
    {
        /* Init the four row pointers. */
        int64_t const y2 = y * 2;
        T const* row[4];
        row[0] = &img->at(std::max<int64_t>(0, y2 - 1) * rowstride);
        row[1] = &img->at(y2 * rowstride);
        row[2] = &img->at(std::min(ih - 1, y2 + 1) * rowstride);
        row[3] = &img->at(std::min(ih - 1, y2 + 2) * rowstride);

        for (int64_t x = 0; x < ow; ++x)
        {
            /* Init four pixel positions for each row. */
            int64_t x2 = x * 2;
            int64_t xi[4];
            xi[0] = std::max<int64_t>(0, x2 - 1) * ic;
            xi[1] = x2 * ic;
            xi[2] = std::min(iw - 1, x2 + 1) * ic;
            xi[3] = std::min(iw - 1, x2 + 2) * ic;

            /* Accumulate 16 values in each channel. */
            for (int64_t c = 0; c < ic; ++c)
            {
                math::Accum<T> accum(T(0));
                accum.add(row[0][xi[0] + c], w3);
                accum.add(row[0][xi[1] + c], w2);
                accum.add(row[0][xi[2] + c], w2);
                accum.add(row[0][xi[3] + c], w3);

                accum.add(row[1][xi[0] + c], w2);
                accum.add(row[1][xi[1] + c], w1);
                accum.add(row[1][xi[2] + c], w1);
                accum.add(row[1][xi[3] + c], w2);

                accum.add(row[2][xi[0] + c], w2);
                accum.add(row[2][xi[1] + c], w1);
                accum.add(row[2][xi[2] + c], w1);
                accum.add(row[2][xi[3] + c], w2);

                accum.add(row[3][xi[0] + c], w3);
                accum.add(row[3][xi[1] + c], w2);
                accum.add(row[3][xi[2] + c], w2);
                accum.add(row[3][xi[3] + c], w3);

                out->at(outpos++) = accum.normalized();
            }
        }
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rescale_half_size_subsample (typename Image<T>::ConstPtr img)
{
    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = (iw + 1) >> 1;
    int64_t const oh = (ih + 1) >> 1;
    int64_t const irs = iw * ic; // input image row stride

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow, oh, ic);

    int64_t iter = 0; // Output image iterator
    for (int64_t iy = 0; iy < ih; iy += 2)
    {
        int64_t rowoff = iy * irs;
        int64_t pixoff = rowoff;
        for (int64_t ix = 0; ix < iw; ix += 2)
        {
            T const* iptr = &img->at(pixoff);
            T* optr = &out->at(iter);
            std::copy(iptr, iptr + ic, optr);
            pixoff += ic << 1;
            iter += ic;
        }
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rescale_double_size (typename Image<T>::ConstPtr img)
{
    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = iw << 1;
    int64_t const oh = ih << 1;
    int64_t const irs = iw * ic;  // input image row stride

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow, oh, ic);

    float w[4] = { 0.75f*0.75f, 0.25f*0.75f, 0.75f*0.25f, 0.25f*0.25f };

    T const* row1 = img->begin();
    T const* row2 = img->begin();

    int64_t i = 0;
    for (int64_t y = 0; y < oh; ++y)
    {
        /* Uneven row -> advance, even row -> swap. */
        if (y % 2)
            row2 = row1 + (y < oh - 1 ? irs : 0);
        else
            std::swap(row1, row2);

        T const* px[4] = { row1, row1, row2, row2 };
        for (int64_t x = 0; x < ow; ++x)
        {
            /* Uneven pixel -> advance, even pixel -> swap. */
            if (x % 2)
            {
                int64_t off = (x < ow - 1 ? ic : 0);
                px[1] = px[0] + off;
                px[3] = px[2] + off;
            }
            else
            {
                std::swap(px[0], px[1]);
                std::swap(px[2], px[3]);
            }

            for (int64_t c = 0; c < ic; ++c, ++i)
                out->at(i) = math::interpolate
                    (px[0][c], px[1][c], px[2][c], px[3][c],
                    w[0], w[1], w[2], w[3]);
        }
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rescale_double_size_supersample (typename Image<T>::ConstPtr img)
{
    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = iw << 1;
    int64_t const oh = ih << 1;

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(ow, oh, ic);

    int64_t witer = 0;
    for (int64_t y = 0; y < oh; ++y)
    {
        bool nexty = (y + 1 < oh);
        int64_t yoff[2] = { iw * (y >> 1), iw * ((y + nexty) >> 1) };
        for (int64_t x = 0; x < ow; ++x)
        {
            bool nextx = (x + 1 < ow);
            int64_t xoff[2] = { x >> 1, (x + nextx) >> 1 };
            T const* val[4] =
            {
                &img->at(yoff[0] + xoff[0], 0),
                &img->at(yoff[0] + xoff[1], 0),
                &img->at(yoff[1] + xoff[0], 0),
                &img->at(yoff[1] + xoff[1], 0)
            };

            for (int64_t c = 0; c < ic; ++c, ++witer)
                out->at(witer) = math::interpolate
                    (val[0][c], val[1][c], val[2][c], val[3][c],
                    0.25f, 0.25f, 0.25f, 0.25f);
        }
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
void
rescale_nearest (typename Image<T>::ConstPtr img, typename Image<T>::Ptr out)
{
    if (img->channels() != out->channels())
        throw std::invalid_argument("Image channel mismatch");

    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = out->width();
    int64_t const oh = out->height();

    int64_t outpos = 0;
    for (int64_t y = 0; y < oh; ++y)
    {
        float ly = ((float)y + 0.5f) * (float)ih / (float)oh;
        int64_t iy = static_cast<int64_t>(ly);
        for (int64_t x = 0; x < ow; ++x)
        {
            float lx = ((float)x + 0.5f) * (float)iw / (float)ow;
            int64_t ix = static_cast<int64_t>(lx);
            for (int64_t c = 0; c < ic; ++c)
                out->at(outpos++) = img->at(ix,iy,c);
        }
    }
}

/* ---------------------------------------------------------------- */

template <typename T>
void
rescale_linear (typename Image<T>::ConstPtr img, typename Image<T>::Ptr out)
{
    if (img->channels() != out->channels())
        throw std::invalid_argument("Image channel mismatch");

    int64_t const iw = img->width();
    int64_t const ih = img->height();
    int64_t const ic = img->channels();
    int64_t const ow = out->width();
    int64_t const oh = out->height();

    T* out_ptr = out->get_data_pointer();
    int64_t outpos = 0;
    for (int64_t y = 0; y < oh; ++y)
    {
        float fy = ((float)y + 0.5f) * (float)ih / (float)oh;
        for (int64_t x = 0; x < ow; ++x, outpos += ic)
        {
            float fx = ((float)x + 0.5f) * (float)iw / (float)ow;
            img->linear_at(fx - 0.5f, fy - 0.5f, out_ptr + outpos);
        }
    }
}

/* ---------------------------------------------------------------- */

template <typename T>
T
gaussian_kernel (typename Image<T>::ConstPtr img,
    float x, float y, int64_t c, float sigma)
{
    int64_t const width = img->width();
    int64_t const height = img->height();

    /*
     * Calculate kernel size for geometric gaussian
     * Kernel is cut off at y=1/N, x = sigma * sqrt(2 * ln N).
     *
     * For N=256: x = sigma * 3.33.
     * For N=128: x = sigma * 3.12.
     * For N=64: x = sigma * 2.884.
     * For N=32: x = sigma * 2.63.
     * For N=16: x = sigma * 2.355.
     * For N=8: x = sigma * 2.04.
     * For N=4: x = sigma * 1.67.
     */
    float ks = sigma * 2.884f;

    /* Calculate min/max kernel position. */
    float kx_min = std::floor(x - ks);
    float kx_max = std::ceil(x + ks - 1.0f);
    float ky_min = std::floor(y - ks);
    float ky_max = std::ceil(y + ks - 1.0f);

    int64_t kxi_min = static_cast<int64_t>(std::max(0.0f, kx_min));
    int64_t kxi_max =
        static_cast<int64_t>(std::min((float)width - 1.0f, kx_max));
    int64_t kyi_min = static_cast<int64_t>(std::max(0.0f, ky_min));
    int64_t kyi_max =
        static_cast<int64_t>(std::min((float)height - 1.0f, ky_max));

    /* Determine pixel weight for kernel bounaries. */
    float wx_start = kx_min > 0.0f ? kx_min + 1.0f + ks - x : 1.0f;
    float wx_end = kx_max < (float)width - 1.0f ? ks + x - kx_max : 1.0f;
    float wy_start = ky_min > 0.0f ? ky_min + 1.0f + ks - y : 1.0f;
    float wy_end = ky_max < (float)height - 1.0f ? ks + y - ky_max : 1.0f;

    /* Apply kernel. */
    math::Accum<T> accum(0);
    for (int64_t yi = kyi_min; yi <= kyi_max; ++yi)
        for (int64_t xi = kxi_min; xi <= kxi_max; ++xi)
        {
            float weight = 1.0f;
            weight *= (xi == kxi_min ? wx_start : 1.0f);
            weight *= (xi == kxi_max ? wx_end : 1.0f);
            weight *= (yi == kyi_min ? wy_start : 1.0f);
            weight *= (yi == kyi_max ? wy_end : 1.0f);
            float dx = static_cast<float>(xi) + 0.5f - x;
            float dy = static_cast<float>(yi) + 0.5f - y;
            weight *= math::gaussian_xx(dx * dx + dy * dy, sigma);

            accum.add(img->at(xi, yi, c), weight);
        }

    return accum.normalized();
}

/* ---------------------------------------------------------------- */

template <typename T>
void
rescale_gaussian (typename Image<T>::ConstPtr img,
    typename Image<T>::Ptr out, float sigma_factor)
{
    if (img->channels() != out->channels())
        throw std::invalid_argument("Image channels mismatch");

    int64_t const ow = out->width();
    int64_t const oh = out->height();
    int64_t const oc = out->channels();

    /* Choose gaussian sigma parameter according to scale factor. */
    float const scale_x = (float)img->width() / (float)ow;
    float const scale_y = (float)img->height() / (float)oh;
    float const sigma = sigma_factor * std::max(scale_x, scale_y) / 2.0f;

    /* Iterate pixels of dest image and convolute with gaussians on input. */
    int64_t i = 0;
    for (int64_t y = 0; y < oh; ++y)
        for (int64_t x = 0; x < ow; ++x, ++i)
        {
            float xf = ((float)x + 0.5f) * scale_x;
            float yf = ((float)y + 0.5f) * scale_y;
            for (int64_t c = 0; c < oc; ++c)
                out->at(i, c) = gaussian_kernel<T>(img, xf, yf, c, sigma);
        }
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
crop (typename Image<T>::ConstPtr image, int64_t width, int64_t height,
    int64_t left, int64_t top, T const* fill_color)
{
    if (width < 0 || height < 0 || !image.get())
        throw std::invalid_argument("Invalid width/height or null image given");

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(width, height, image->channels());

    int64_t const iw = image->width();
    int64_t const ih = image->height();
    int64_t const ic = image->channels();

    /* Output image with fill color if new pixels are revealed. */
    if (left < 0 || top < 0 || left + width > iw || top + height > ih)
        out->fill_color(fill_color);

    /* Check if input and output have no overlap. */
    if (left >= iw || left <= -width || top >= ih || top <= -height)
        return out;

    /* Copy horizontal overlap for each overlapping row. */
    int64_t const overlap =
        ic * (std::min(iw, left + width) - std::max<int64_t>(0, left));
    for (int64_t y = std::max<int64_t>(0, -top);
        y < std::min(height, ih - top); ++y)
    {
        int64_t lookup_y = top + y;
        if (lookup_y >= ih)
            break;
        T* out_ptr = &out->at(left < 0 ? -left : 0, y, 0);
        T const* in_ptr = &image->at(left > 0 ? left : 0, lookup_y, 0);
        std::copy(in_ptr, in_ptr + overlap, out_ptr);
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
blur_gaussian (typename Image<T>::ConstPtr in, float sigma)
{
    if (in == nullptr)
        throw std::invalid_argument("Null image given");

    /* Small sigmas result in literally no change. */
    if (MATH_EPSILON_EQ(sigma, 0.0f, 0.1f))
        return in->duplicate();

    int64_t const w = in->width();
    int64_t const h = in->height();
    int64_t const c = in->channels();
    int64_t const ks = std::ceil(sigma * 2.884f); // Cap kernel at 1/128
    std::vector<float> kernel(ks + 1);

    /* Fill kernel values. */
    for (int64_t i = 0; i < ks + 1; ++i)
        kernel[i] = math::gaussian((float)i, sigma);

#if 1 // Separated kernel implementation

    /* Convolve the image in x direction. */
    typename Image<T>::Ptr sep(Image<T>::create(w, h, c));
    int64_t px = 0;
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x, ++px)
            for (int64_t cc = 0; cc < c; ++cc)
            {
                math::Accum<T> accum(T(0));
                for (int64_t i = -ks; i <= ks; ++i)
                {
                    int64_t idx = math::clamp<int64_t>(x + i, 0, w - 1);
                    accum.add(in->at(y * w + idx, cc), kernel[std::abs(i)]);
                }
                sep->at(px, cc) = accum.normalized();
            }

    /* Convolve the image in y direction. */
    typename Image<T>::Ptr out(Image<T>::create(w, h, c));
    px = 0;
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x, ++px)
            for (int64_t cc = 0; cc < c; ++cc)
            {
                math::Accum<T> accum(T(0));
                for (int64_t i = -ks; i <= ks; ++i)
                {
                    int64_t idx = math::clamp<int64_t>(y + i, 0, h - 1);
                    accum.add(sep->at(idx * w + x, cc), kernel[std::abs(i)]);
                }
                out->at(px, cc) = accum.normalized();
            }

#else // Non-separated kernel implementation

    typename Image<T>::Ptr out(Image<T>::create(w, h, c));
    int64_t px = 0;
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x, ++px)
            for (int64_t cc = 0; cc < c; ++cc)
            {
                math::Accum<T> accum(T(0));

                for (int64_t ky = -ks; ky <= ks; ++ky)
                    for (int64_t kx = -ks; kx <= ks; ++kx)
                    {
                        int64_t idx_x = math::clamp(x + kx, 0, w - 1);
                        int64_t idx_y = math::clamp(y + ky, 0, h - 1);
                        accum.add(in->at(idx_y * w + idx_x, cc),
                            kernel[std::abs(kx)] * kernel[std::abs(ky)]);
                    }

                out->at(px, cc) = accum.normalized();
            }

#endif

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
blur_boxfilter (typename Image<T>::ConstPtr in, int ks)
{
    if (in == nullptr)
        throw std::invalid_argument("Null image given");

    int64_t w(in->width());
    int64_t h(in->height());
    int64_t c(in->channels());
    int64_t wc = w * c;

#if 1
    /* Super-fast separated kernel implementation. */
    typename Image<T>::Ptr sep(Image<T>::create(w, h, c));
    math::Accum<T>* accums = new math::Accum<T>[c];
    T const* row = &in->at(0);
    T* outrow = &sep->at(0);
    for (int64_t y = 0; y < h; ++y, row += w * c, outrow += w * c)
    {
        for (int64_t cc = 0; cc < c; ++cc) // Reset accumulators
            accums[cc] = math::Accum<T>(T(0));

        for (int i = 0; i < ks; ++i)
            for (int64_t cc = 0; cc < c; ++cc)
                accums[cc].add(row[i * c + cc], 1.0f);

        for (int64_t x = 0; x < w; ++x)
        {
            if (x + ks < w - 1)
                for (int64_t cc = 0; cc < c; ++cc)
                    accums[cc].add(row[(x + ks) * c + cc], 1.0f);
            if (x > ks)
                for (int64_t cc = 0; cc < c; ++cc)
                    accums[cc].sub(row[(x - ks - 1) * c + cc], 1.0f);
            for (int64_t cc = 0; cc < c; ++cc)
                outrow[x * c + cc] = accums[cc].normalized();
        }
    }

    /* Second filtering pass with kernel in y-direction. */
    typename Image<T>::Ptr out(Image<T>::create(w, h, c));
    T const* col = &sep->at(0);
    T* outcol = &out->at(0);
    for (int64_t x = 0; x < w; ++x, col += c, outcol += c)
    {
        for (int64_t cc = 0; cc < c; ++cc)
            accums[cc] = math::Accum<T>(T(0));

        for (int i = 0; i < ks; ++i)
            for (int64_t cc = 0; cc < c; ++cc)
                accums[cc].add(col[i * wc + cc], 1.0f);

        for (int64_t y = 0; y < h; ++y)
        {
            if (y + ks < h - 1)
                for (int64_t cc = 0; cc < c; ++cc)
                    accums[cc].add(col[(y + ks) * wc + cc], 1.0f);
            if (y > ks)
                for (int64_t cc = 0; cc < c; ++cc)
                    accums[cc].sub(col[(y - ks - 1) * wc + cc], 1.0f);
            for (int64_t cc = 0; cc < c; ++cc)
                outcol[y * wc + cc] = accums[cc].normalized();
        }
    }

    delete [] accums;
#endif


#if 0 // Slow and naive, but simple implementation
    typename Image<T>::Ptr out(Image<T>::create(w, h, c));
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x)
            for (int64_t cc = 0; cc < c; ++cc)
            {
                math::Accum<T> accum(T(0));
                for (int ky = -ks; ky <= ks; ++ky)
                    for (int kx = -ks; kx <= ks; ++kx)
                    {
                        int64_t idx_x = math::clamp(x + kx, 0, w - 1);
                        int64_t idx_y = math::clamp(y + ky, 0, h - 1);
                        accum.add(in->at(idx_x, idx_y, cc), 1.0f);
                    }
                out->at(x, y, cc) = accum.normalized();
            }
#endif

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rotate (typename Image<T>::ConstPtr image, RotateType type)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    int64_t const iw = image->width();
    int64_t const ih = image->height();
    int64_t const ic = image->channels();
    int64_t const ow = type == ROTATE_180 ? iw : ih;
    int64_t const oh = type == ROTATE_180 ? ih : iw;

    typename Image<T>::Ptr ret(Image<T>::create());
    ret->allocate(ow, oh, ic);

    int64_t idx = 0;
    for (int64_t y = 0; y < ih; ++y)
        for (int64_t x = 0; x < iw; ++x, idx += ic)
        {
            int64_t dx = x;
            int64_t dy = y;
            switch (type)
            {
                case ROTATE_180:  dx = iw - x - 1; dy = ih - y - 1; break;
                case ROTATE_CW:   dx = ih - y - 1; dy = x;          break;
                case ROTATE_CCW:  dx = y;          dy = iw - x - 1; break;
                case ROTATE_SWAP: dx = y;          dy = x;          break;
                default: break;
            }

            T const* in_pixel = &image->at(idx);
            T* out_pixel = &ret->at(dx, dy, 0);
            std::copy(in_pixel, in_pixel + ic, out_pixel);
        }

    return ret;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
rotate (typename Image<T>::ConstPtr image, float angle, T const* fill_color)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    int64_t const w = image->width();
    int64_t const h = image->height();
    int64_t const c = image->channels();
    float const w2 = static_cast<float>(w - 1) / 2.0f;
    float const h2 = static_cast<float>(h - 1) / 2.0f;
    typename Image<T>::Ptr ret = Image<T>::create(w, h, c);

    float const sin_angle = std::sin(-angle);
    float const cos_angle = std::cos(-angle);
    T* ret_ptr = ret->begin();
    for (int64_t y = 0; y < h; ++y)
        for (int64_t x = 0; x < w; ++x, ret_ptr += c)
        {
            float const sample_x =
                cos_angle * (x - w2) - sin_angle * (y - h2) + w2;
            float const sample_y =
                sin_angle * (x - w2) + cos_angle * (y - h2) + h2;
            if (sample_x < -0.5f || sample_x > w - 0.5f
                || sample_y < -0.5f || sample_y > h - 0.5f)
                std::copy(fill_color, fill_color + c, ret_ptr);
            else
                image->linear_at(sample_x, sample_y, ret_ptr);
        }
    return ret;
}

/* ---------------------------------------------------------------- */

template <typename T>
void
flip (typename Image<T>::Ptr image, FlipType type)
{
    if (type == FLIP_NONE)
        return;
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    bool const fh = type & FLIP_HORIZONTAL;
    bool const fv = type & FLIP_VERTICAL;
    int64_t const iw = image->width();
    int64_t const ih = image->height();
    int64_t const ic = image->channels();
    int64_t const max_x = fh ? iw / 2 : iw;
    int64_t const max_y = fv && !fh ? ih / 2 : ih;

    for (int64_t y = 0; y < max_y; ++y)
        for (int64_t x = 0; x < max_x; ++x)
        {
            int64_t dx = (type & FLIP_HORIZONTAL ? iw - x - 1 : x);
            int64_t dy = (type & FLIP_VERTICAL ? ih - y - 1 : y);
            T* src = &image->at(x, y, 0);
            T* dst = &image->at(dx, dy, 0);
            std::swap_ranges(src, src + ic, dst);
        }
}

/* ---------------------------------------------------------------- */

template <typename T>
inline T
desaturate_maximum (T const* v)
{
    return *std::max_element(v, v + 3);
}

template <typename T>
inline T
desaturate_lightness (T const* v)
{
    T const* max = std::max_element(v, v + 3);
    T const* min = std::min_element(v, v + 3);
    return math::interpolate(*max, *min, 0.5f, 0.5f);
}

template <typename T>
inline T
desaturate_luminosity (T const* v)
{
    return math::interpolate(v[0], v[1], v[2], 0.21f, 0.72f, 0.07f);
}

template <typename T>
inline T
desaturate_luminance (T const* v)
{
    return math::interpolate(v[0], v[1], v[2], 0.30f, 0.59f, 0.11f);
}

template <typename T>
inline T
desaturate_average (T const* v)
{
    float third(1.0f / 3.0f);
    return math::interpolate(v[0], v[1], v[2], third, third, third);
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
desaturate (typename Image<T>::ConstPtr img, DesaturateType type)
{
    if (img == nullptr)
        throw std::invalid_argument("Null image given");

    int64_t ic = img->channels();
    if (ic != 3 && ic != 4)
        throw std::invalid_argument("Image must be RGB or RGBA");

    bool has_alpha = (ic == 4);

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(img->width(), img->height(), 1 + has_alpha);

    typedef T(*DesaturateFunc)(T const*);
    DesaturateFunc func;
    switch (type)
    {
        case DESATURATE_MAXIMUM: func = desaturate_maximum<T>; break;
        case DESATURATE_LIGHTNESS: func = desaturate_lightness<T>; break;
        case DESATURATE_LUMINOSITY: func = desaturate_luminosity<T>; break;
        case DESATURATE_LUMINANCE: func = desaturate_luminance<T>; break;
        case DESATURATE_AVERAGE: func = desaturate_average<T>; break;
        default: throw std::invalid_argument("Invalid desaturate type");
    }

    int64_t outpos = 0;
    int64_t inpos = 0;
    int64_t pixels = img->get_pixel_amount();
    for (int64_t i = 0; i < pixels; ++i)
    {
        T const* v = &img->at(inpos);
        out->at(outpos) = func(v);

        if (has_alpha)
            out->at(outpos + 1) = img->at(inpos + 3);

        outpos += 1 + has_alpha;
        inpos += 3 + has_alpha;
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
expand_grayscale (typename Image<T>::ConstPtr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    int64_t const ic = image->channels();
    if (ic != 1 && ic != 2)
        throw std::invalid_argument("Image must be in G or GA");

    bool const has_alpha = (ic == 2);

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(image->width(), image->height(), 3 + has_alpha);

    int64_t pixels = image->get_pixel_amount();
    for (int64_t i = 0; i < pixels; ++i)
    {
        out->at(i, 0) = image->at(i, 0);
        out->at(i, 1) = image->at(i, 0);
        out->at(i, 2) = image->at(i, 0);
        if (has_alpha)
            out->at(i, 3) = image->at(i, 1);
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
void
reduce_alpha (typename mve::Image<T>::Ptr img)
{
    int64_t const channels = img->channels();
    if (channels != 2 && channels != 4)
        throw std::invalid_argument("Image must be in GA or RGBA");
    img->delete_channel(channels - 1);
}

/* ---------------------------------------------------------------- */

template <typename T>
typename mve::Image<T>::Ptr
sobel_edge (typename mve::Image<T>::ConstPtr img)
{
    int64_t const width = img->width();
    int64_t const height = img->height();
    int64_t const chans = img->channels();
    int64_t const row_stride = width * chans;

    double const max_value = static_cast<double>(std::numeric_limits<T>::max());
    typename mve::Image<T>::Ptr out = mve::Image<T>::create
        (width, height, chans);
    T* out_ptr = out->get_data_pointer();

    int64_t pos = 0;
    for (int64_t y = 0; y < height; ++y)
        for (int64_t x = 0; x < width; ++x, pos += chans)
        {
            if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
            {
                std::fill(out_ptr + pos, out_ptr + pos + chans, T(0));
                continue;
            }

            for (int64_t cc = 0; cc < chans; ++cc)
            {
                int64_t const i = pos + cc;
                double gx = 1.0 * (double)img->at(i + chans - row_stride)
                    - 1.0 * (double)img->at(i - chans - row_stride)
                    + 2.0 * (double)img->at(i + chans)
                    - 2.0 * (double)img->at(i - chans)
                    + 1.0 * (double)img->at(i + chans + row_stride)
                    - 1.0 * (double)img->at(i - chans + row_stride);
                double gy = 1.0 * (double)img->at(i + row_stride - chans)
                    - 1.0 * (double)img->at(i - row_stride - chans)
                    + 2.0 * (double)img->at(i + row_stride)
                    - 2.0 * (double)img->at(i - row_stride)
                    + 1.0 * (double)img->at(i + row_stride + chans)
                    - 1.0 * (double)img->at(i - row_stride + chans);
                double g = std::sqrt(gx * gx + gy * gy);
                out_ptr[i] = static_cast<T>(std::min(max_value, g));
            }
        }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
subtract (typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2)
{
    int64_t const iw = i1->width();
    int64_t const ih = i1->height();
    int64_t const ic = i1->channels();

    if (i1 == nullptr || i2 == nullptr)
        throw std::invalid_argument("Null image given");

    if (iw != i2->width() || ih != i2->height() || ic != i2->channels())
        throw std::invalid_argument("Image dimensions do not match");

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(iw, ih, ic);

    // FIXME: That needs speedup! STL algo with pointers.
    for (int64_t i = 0; i < i1->get_value_amount(); ++i)
        out->at(i) = i1->at(i) - i2->at(i);

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
difference (typename Image<T>::ConstPtr i1, typename Image<T>::ConstPtr i2)
{
    int64_t const iw = i1->width();
    int64_t const ih = i1->height();
    int64_t const ic = i1->channels();

    if (i1 == nullptr || i2 == nullptr)
        throw std::invalid_argument("Null image given");

    if (iw != i2->width() || ih != i2->height() || ic != i2->channels())
        throw std::invalid_argument("Image dimensions do not match");

    typename Image<T>::Ptr out(Image<T>::create());
    out->allocate(iw, ih, ic);

    for (int64_t i = 0; i < i1->get_value_amount(); ++i)
    {
        if (i1->at(i) < i2->at(i))
            out->at(i) = i2->at(i) - i1->at(i);
        else
            out->at(i) = i1->at(i) - i2->at(i);
    }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
void
gamma_correct (typename Image<T>::Ptr image, T const& power)
{
    math::algo::foreach_constant_power<T> f(power);
    std::for_each(image->begin(), image->end(), f);
}

/* ---------------------------------------------------------------- */

template <typename T>
void
gamma_correct_srgb (typename Image<T>::Ptr image)
{
    static_assert(std::is_floating_point<T>::value,
        "Only implemented for floating point images");
    int64_t const num_values = image->get_value_amount();
    for (int64_t i = 0; i < num_values; i++)
    {
        if (image->at(i) <= T(0.0031308))
            image->at(i) *= T(12.92);
        else
        {
            T corrected = std::pow(image->at(i), T(1.0)/T(2.4));
            image->at(i) = T(1.055) * corrected - T(0.055);
        }
    }
}

/* ---------------------------------------------------------------- */

template <typename T>
void
gamma_correct_inv_srgb (typename Image<T>::Ptr image)
{
    static_assert(std::is_floating_point<T>::value,
        "Only implemented for floating point images");
    int64_t const num_values = image->get_value_amount();
    for (int64_t i = 0; i < num_values; i++)
    {
        if (image->at(i) <= T(0.04045))
            image->at(i) /= T(12.92);
        else
        {
            T base = (image->at(i) + T(0.055)) / T(1.055);
            image->at(i) = std::pow(base, T(2.4));
        }
    }
}

/* ---------------------------------------------------------------- */

template <typename T_IN, typename T_OUT>
typename Image<T_OUT>::Ptr
integral_image (typename Image<T_IN>::ConstPtr image)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    int64_t const width = image->width();
    int64_t const height = image->height();
    int64_t const chans = image->channels();
    int64_t const row_stride = width * chans;

    typename Image<T_OUT>::Ptr ret(Image<T_OUT>::create());
    ret->allocate(width, height, chans);

    /* Input image row and destination image rows. */
    std::vector<T_OUT> zeros(row_stride, T_OUT(0));
    T_IN const* inrow = image->get_data_pointer();
    T_OUT* dest = ret->get_data_pointer();
    T_OUT* prev = &zeros[0];

    /*
     * I(x,y) = i(x,y) + I(x-1,y) + I(x,y-1) - I(x-1,y-1)
     */
    for (int64_t y = 0; y < height; ++y)
    {
        /* Calculate first pixel in row. */
        for (int64_t cc = 0; cc < chans; ++cc)
            dest[cc] = static_cast<T_OUT>(inrow[cc]) + prev[cc];
        /* Calculate all following pixels in row. */
        for (int64_t i = chans; i < row_stride; ++i)
            dest[i] = inrow[i] + prev[i] + dest[i - chans] - prev[i - chans];

        prev = dest;
        dest += row_stride;
        inrow += row_stride;
    }

    return ret;
}

/* ---------------------------------------------------------------- */

template <typename T>
T
integral_image_area (typename Image<T>::ConstPtr sat,
    int64_t x1, int64_t y1, int64_t x2, int64_t y2, int64_t cc)
{
    int64_t const row_stride = sat->width() * sat->channels();
    int64_t const channels = sat->channels();
    T ret = sat->at(y2 * row_stride + x2 * channels + cc);
    if (x1 > 0)
        ret -= sat->at(y2 * row_stride + (x1-1) * channels + cc);
    if (y1 > 0)
        ret -= sat->at((y1-1) * row_stride + x2 * channels + cc);
    if (x1 > 0 && y1 > 0)
        ret += sat->at((y1-1) * row_stride + (x1-1) * channels + cc);
    return ret;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
create_thumbnail (typename Image<T>::ConstPtr image,
    int64_t thumb_width, int64_t thumb_height)
{
    int64_t const width = image->width();
    int64_t const height = image->height();
    float const image_aspect = static_cast<float>(width) / height;
    float const thumb_aspect = static_cast<float>(thumb_width) / thumb_height;

    int64_t rescale_width, rescale_height;
    int64_t crop_left, crop_top;
    if (image_aspect > thumb_aspect)
    {
        rescale_width = std::ceil(thumb_height * image_aspect);
        rescale_height = thumb_height;
        crop_left = (rescale_width - thumb_width) / 2;
        crop_top = 0;
    }
    else
    {
        rescale_width = thumb_width;
        rescale_height = std::ceil(thumb_width / image_aspect);
        crop_left = 0;
        crop_top = (rescale_height - thumb_height) / 2;
    }

    typename mve::Image<T>::Ptr thumb = mve::image::rescale<T>(image,
        mve::image::RESCALE_LINEAR, rescale_width, rescale_height);
    thumb = mve::image::crop<T>(thumb, thumb_width, thumb_height,
        crop_left, crop_top, nullptr);

    return thumb;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
image_undistort_msps (typename Image<T>::ConstPtr img, double k0, double k1)
{
    int64_t const width = img->width();
    int64_t const height = img->height();
    int64_t const chans = img->channels();
    int64_t const D = std::max(width, height);

    double const width_half = static_cast<double>(width) / 2.0;
    double const height_half = static_cast<double>(height) / 2.0;

    typename Image<T>::Ptr out = Image<T>::create(width, height, chans);
    out->fill(T(0));
    T* out_ptr = out->get_data_pointer();

    for (int64_t y = 0; y < height; y++)
        for (int64_t x = 0; x < width; x++, out_ptr += chans)
        {
            double fx = static_cast<double>(x) - width_half;
            double fy = static_cast<double>(y) - height_half;
            double const s1 = D * D + k1 * (fx * fx + fy * fy);
            double const s2 = D * D + k0 * (fx * fx + fy * fy);
            double const factor = s1 / s2;
            fx = fx * factor + width_half;
            fy = fy * factor + height_half;

            if (fx < -0.5 || fx > width - 0.5 || fy < -0.5 || fy > height - 0.5)
                continue;
            img->linear_at(fx, fy, out_ptr);
        }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
image_undistort_k2k4 (typename Image<T>::ConstPtr img,
    double focal_length, double k2, double k4)
{
    if (img == nullptr)
        throw std::invalid_argument("Null image given");

    if (k2 == 0.0 && k4 == 0.0)
        return img->duplicate();

    int64_t const width = img->width();
    int64_t const height = img->height();
    int64_t const chans = img->channels();
    typename Image<T>::Ptr out = Image<T>::create(width, height, chans);
    out->fill(T(0));
    T* out_ptr = out->get_data_pointer();

    double const fwidth2 = static_cast<double>(width) / 2.0;
    double const fheight2 = static_cast<double>(height) / 2.0;
    double const fnorm = static_cast<double>(std::max(width, height));
    for (int64_t y = 0; y < height; ++y)
        for (int64_t x = 0; x < width; ++x, out_ptr += chans)
        {
            double const fx = (static_cast<double>(x) + 0.5 - fwidth2) / fnorm;
            double const fy = (static_cast<double>(y) + 0.5 - fheight2) / fnorm;
            double const rd = (fx * fx + fy * fy) / MATH_POW2(focal_length);
            double const rd_factor = 1.0 + k2 * rd + k4 * rd * rd;
            double const dist_x = fx * rd_factor;
            double const dist_y = fy * rd_factor;
            float const ix = dist_x * fnorm + fwidth2 - 0.5;
            float const iy = dist_y * fnorm + fheight2 - 0.5;
            if (ix < -0.5 || ix > width - 0.5 || iy < -0.5 || iy > height - 0.5)
                continue;
            img->linear_at(ix, iy, out_ptr);
        }

    return out;
}

/* ---------------------------------------------------------------- */

template <typename T>
typename Image<T>::Ptr
image_undistort_vsfm (typename Image<T>::ConstPtr img,
    double focal_length, double k1)
{
    if (img == nullptr)
        throw std::invalid_argument("Null image given");

    if (k1 == 0.0)
        return img->duplicate();

    int64_t const width = img->width();
    int64_t const height = img->height();
    int64_t const chans = img->channels();

    /*
     * The image coordinates must be normalized before the distortion
     * model is applied. The image coordinates are first centered at
     * the origin and then scaled w.r.t. the focal length in pixel.
     */
    double const norm = focal_length * std::max(width, height);
    double const width_half = static_cast<double>(width) / 2.0;
    double const height_half = static_cast<double>(height) / 2.0;

    typename Image<T>::Ptr out = Image<T>::create(width, height, chans);
    out->fill(T(0));
    T* out_ptr = out->begin();

    for (int64_t y = 0; y < height; ++y)
    {
        for (int64_t x = 0; x < width; ++x, out_ptr += chans)
        {
            double fx = (static_cast<double>(x) - width_half) / norm;
            double fy = (static_cast<double>(y) - height_half) / norm;
            if (fy == 0.0)
                fy = 1e-10;

            double const t2 = fy * fy;
            double const t3 = t2 * t2 * t2;
            double const t4 = fx * fx;
            double const t7 = k1 * (t2 + t4);

            if (k1 > 0.0)
            {
                double const t8 = 1.0 / t7;
                double const t10 = t3 / (t7 * t7);
                double const t14 = std::sqrt(t10 * (0.25 + t8 / 27.0));
                double const t15 = t2 * t8 * fy * 0.5;
                double const t17 = std::pow(t14 + t15, 1.0/3.0);
                double const t18 = t17 - t2 * t8 / (t17 * 3.0);
                fx = t18 * fx / fy;
                fy = t18;
            }
            else
            {
                double const t9 = t3 / (t7 * t7 * 4.0);
                double const t11 = t3 / (t7 * t7 * t7 * 27.0);
                std::complex<double> const t12 = t9 + t11;
                std::complex<double> const t13 = std::sqrt(t12);
                double const t14 = t2 / t7;
                double const t15 = t14 * fy * 0.5;
                std::complex<double> const t16 = t13 + t15;
                std::complex<double> const t17 = std::pow(t16, 1.0/3.0);
                std::complex<double> const t18 = (t17 + t14 / (t17 * 3.0))
                    * std::complex<double>(0.0, std::sqrt(3.0));
                std::complex<double> const t19 = -0.5 * (t17 + t18)
                    + t14 / (t17 * 6.0);
                fx = t19.real() * fx / fy;
                fy = t19.real();
            }

            fx = fx * norm + width_half;
            fy = fy * norm + height_half;

            if (fx < -0.5 || fx > width - 0.5 || fy < -0.5 || fy > height - 0.5)
                continue;
            img->linear_at(fx, fy, out_ptr);
        }
    }

    return out;
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

#endif /* MVE_IMAGE_TOOLS_HEADER */
