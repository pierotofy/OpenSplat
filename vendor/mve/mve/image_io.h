/*
 * Copyright (C) 2015, Simon Fuhrmann, Jens Ackermann, Sebastian Koch
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_IMAGE_FILE_HEADER
#define MVE_IMAGE_FILE_HEADER

#include <string>

#include "mve/defines.h"
#include "mve/image.h"

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/**
 * Image meta data. Some loaders offer to retrieve only this meta data
 * and skip loading the payload.
 */
struct ImageHeaders
{
    int64_t width;
    int64_t height;
    int64_t channels;
    ImageType type;
};

/*
 * ------------------ Image loading and saving --------------------
 *
 * libpng docs: http://www.libpng.org/pub/png/libpng-manual.txt
 * libjpg docs: http://apodeline.free.fr/DOC/libjpeg/libjpeg.html
 * libtiff docs: http://www.libtiff.org/libtiff.html
 * pfm file docs: http://netpbm.sourceforge.net/doc/pfm.html
 * ppm file docs: http://en.wikipedia.org/wiki/Netpbm_format
 * more ppm file docs: http://netpbm.sourceforge.net/doc/#formats
 *
 * TODO
 * - Read and display PNG embedded "keywords" (author, comment, etc.)
 * - Fix PPM endianess
 */

/**
 * Loads an image, detecting file type.
 * May throw util::Exception.
 */
ByteImage::Ptr
load_file (std::string const& filename);

/**
 * Loads the image headers, detecting file type.
 * May throw util::Exception.
 */
ImageHeaders
load_file_headers (std::string const& filename);

/**
 * Saves a byte image to file, detecting file type.
 * May throw util::Exception.
 */
void
save_file (ByteImage::ConstPtr image, std::string const& filename);

/**
 * Saves a byte image to file, detecting file type.
 * Overloaded for non-const pointers to avoid cast ambiguities.
 * May throw util::Exception.
 */
void
save_file (ByteImage::Ptr image, std::string const& filename);

/**
 * Saves a float image to file, detecting file type.
 * May throw util::Exception.
 */
void
save_file (FloatImage::ConstPtr image, std::string const& filename);

/**
 * Saves a float image to file, detecting file type.
 * Overloaded for non-const pointers to avoid cast ambiguities.
 * May throw util::Exception.
 */
void
save_file (FloatImage::Ptr image, std::string const& filename);

/* -------------------------- PNG support ------------------------- */

#ifndef MVE_NO_PNG_SUPPORT

/**
 * Loads a PNG file.
 * PNG has 1, 2, 3 or 4 channels with gray, gray-alpha, RGB or RGBA values.
 * Conversion of 1, 2 and 4 to 8 bit is auto-applied.
 * May throw util::FileException and util::Exception.
 */
ByteImage::Ptr
load_png_file (std::string const& filename);

/**
 * Loads PNG file headers only.
 * May throw util::FileException and util::Exception.
 */
ImageHeaders
load_png_file_headers (std::string const& filename);

/**
 * Saves image data to a PNG file. Supports 1, 2, 3 and 4 channel images.
 * Valid compression levels are in [0, 9], 0 is fastest.
 * May throw util::FileException and util::Exception.
 */
void
save_png_file (ByteImage::ConstPtr image,
    std::string const& filename, int compression_level = 1);

#endif /* MVE_NO_PNG_SUPPORT */

/* ------------------------- JPEG support ------------------------- */

#ifndef MVE_NO_JPEG_SUPPORT

/**
 * Loads a JPEG file. The EXIF data blob may be loaded into 'exif'.
 * JPEGs have 1 (gray values) or 3 (RGB) channels.
 * May throw util::FileException and util::Exception.
 */
ByteImage::Ptr
load_jpg_file (std::string const& filename, std::string* exif = nullptr);

/**
 * Loads JPEG file headers only.
 * May throw util::FileException and util::Exception.
 */
ImageHeaders
load_jpg_file_headers (std::string const& filename);

/**
 * Saves image data to a JPEG file. Supports 1 and 3 channel images.
 * The quality value is in range [0, 100] from worst to best quality.
 * May throw util::FileException and util::Exception.
 */
void
save_jpg_file (ByteImage::ConstPtr image,
    std::string const& filename, int quality);

#endif /* MVE_NO_JPEG_SUPPORT */

/* ------------------------- TIFF support ------------------------- */

#ifndef MVE_NO_TIFF_SUPPORT

/**
 * Loads TIFF file headers only.
 * May throw util::FileException and util::Exception.
 */
ImageHeaders
load_tiff_file_headers (std::string const& filename);

/**
 * Loads a TIFF file.
 * May throw util::FileException and util::Exception.
 */
ByteImage::Ptr
load_tiff_file (std::string const& filename);

/**
 * Writes a TIFF to file. Supports any number of channels.
 * May throw util::FileException and util::Exception.
 */
void
save_tiff_file (ByteImage::ConstPtr image, std::string const& filename);

/**
 * Loads a 16bit TIFF file.
 * May throw util::FileException and util::Exception.
 */
RawImage::Ptr
load_tiff_16_file (std::string const& filename);

/**
 * Writes a 16bit TIFF to file. Supports any number of channels.
 * May throw util::FileException and util::Exception.
 */
void
save_tiff_16_file (RawImage::ConstPtr image, std::string const& filename);

/**
 * Loads a 32bit float TIFF file.
 * May throw util::FileException and util::Exception.
 */
FloatImage::Ptr
load_tiff_float_file (std::string const& filename);

/**
 * Writes a 32bit float TIFF to file. Supports any number of channels.
 * May throw util::FileException and util::Exception.
 */
void
save_tiff_float_file (FloatImage::ConstPtr image, std::string const& filename);

#endif /* MVE_NO_TIFF_SUPPORT */

/* -------------------------- PFM support ------------------------- */

/**
 * Loads a PFM file.
 * Only handles 1 or 3 channel images of float values. No scaling is
 * performed. May throw util::FileException and util::Exception.
 */
FloatImage::Ptr
load_pfm_file (std::string const& filename);

/**
 * Saves float image data to PFM file. Supports 1 and 3 channel images.
 * May throw util::FileException and util::Exception.
 */
void
save_pfm_file (FloatImage::ConstPtr image, std::string const& filename);

/* ------------------------- PPM16 support ------------------------ */

/**
 * Loads a 16 bit PPM file.
 * May throw util::FileException and util::Exception.
 */
RawImage::Ptr
load_ppm_16_file (std::string const& filename);

/**
 * Save a 16 bit PPM file. Supports 1 and 3 channel images.
 * May throw util::FileException and util::Exception.
 */
void
save_ppm_16_file (RawImage::ConstPtr image, std::string const& filename);

/* -------------------------- PPM support ------------------------- */

/**
 * Loads a 8 bit PPM file.
 * May throw util::FileException and util::Exception.
 */
ByteImage::Ptr
load_ppm_file (std::string const& filename);

/**
 * Writes a 8 bit PPM file. Supports 1 and 3 channel images.
 * May throw util::FileException and std::invalid_argument.
 */
void
save_ppm_file (ByteImage::ConstPtr image, std::string const& filename);

/* ------------------- Native MVE image support ------------------- */

/**
 * Loads a native MVE image. Supports arbitrary type, size and depth,
 * with a primitive, uncompressed format.
 * May throw util::FileException.
 */
ImageBase::Ptr
load_mvei_file (std::string const& filename);

/**
 * Loads the meta information for a native MVE image.
 */
ImageHeaders
load_mvei_file_headers (std::string const& filename);

/**
 * Writes a native MVE image. Supports arbitrary type, size and depth,
 * with a primitive, uncompressed format.
 * May throw util::FileException.
 */
void
save_mvei_file (ImageBase::ConstPtr image, std::string const& filename);

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

#endif /* MVE_IMAGE_FILE_HEADER */
