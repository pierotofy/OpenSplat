/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 *
 * Tiny EXIF tag extractor.
 * Original code written by Mayank Lahiri (see .cc file).
 * Original code from: http://code.google.com/p/easyexif/
 * Some docs here: http://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif.html
 * More docs: http://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
 * Even more docs: http://paulbourke.net/dataformats/tiff/
 */

#ifndef MVE_IMAGEEXIF_HEADER
#define MVE_IMAGEEXIF_HEADER

#include <iostream>
#include <string>

#include "mve/defines.h"

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/**
 * EXIF information.
 */
struct ExifInfo
{
    ExifInfo (void);

    /** Camera manufacturer. */
    std::string camera_maker;
    /** Camera model. */
    std::string camera_model;
    /** Date/time string of last modification. */
    std::string date_modified;
    /** Date/time string of original image. */
    std::string date_original;
    /** Description of the image. */
    std::string description;
    /** Software used to process the image. */
    std::string software;
    /** Copyright information. */
    std::string copyright;
    /** Artist information. */
    std::string artist;

    /** Camera ISO speed rating for the image. */
    int iso_speed;
    /** Bits per sample. */
    int bits_per_sample;

    /**
     * Orientation of the image:
     *
     *   1 = Horizontal (normal)
     *   2 = Mirror horizontal
     *   3 = Rotate 180
     *   4 = Mirror vertical
     *   5 = Mirror horizontal and rotate 270 CW
     *   6 = Rotate 90 CW
     *   7 = Mirror horizontal and rotate 90 CW
     *   8 = Rotate 270 CW
     */
    int orientation;
    /** Focal length of the image in mm, relative to sensor size. */
    float focal_length;
    /** Focal length equivalent for 35mm film. */
    float focal_length_35mm;
    /** F-number in 1/f. */
    float f_number;
    /** Image exposure time in seconds. */
    float exposure_time;
    /** Image exposure bias in F-stops. */
    float exposure_bias;
    /** Image shutter speed in seconds. */
    float shutter_speed;
    /** Flash mode (see http://tinyurl.com/o7pawes). */
    int flash_mode;
    /** EXIF image width. */
    int image_width;
    /** EXIF image height. */
    int image_height;
};

/**
 * Function to extract a (selected) EXIF tags from binary data.
 * The function accepts pure EXIF binary data as read from the JPEG file,
 * or the complete JPEG file. In the latter case, argument 'is_jpeg' needs
 * to be true.
 */
ExifInfo
exif_extract (char const* data, std::size_t len, bool is_jpeg = false);

/**
 * Prints the EXIF information to stream.
 */
void
exif_debug_print (std::ostream& stream, ExifInfo const& exif,
    bool indent = false);

/* ---------------------------------------------------------------- */

inline
ExifInfo::ExifInfo (void)
    : iso_speed(-1)
    , bits_per_sample(-1)
    , orientation(-1)
    , focal_length(-1.0f)
    , focal_length_35mm(-1.0f)
    , f_number(-1.0f)
    , exposure_time(-1.0f)
    , exposure_bias(0.0f)
    , shutter_speed(-1.0f)
    , flash_mode(-1)
    , image_width(-1)
    , image_height(-1)
{
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

#endif /* MVE_IMAGEEXIF_HEADER */
