/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <algorithm>
#include <limits>
#include <fstream>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cerrno>

#ifndef MVE_NO_PNG_SUPPORT
#   include <png.h>
#endif

#ifndef MVE_NO_JPEG_SUPPORT
/* Include windows.h before jpeglib.h to prevent INT32 typedef collision. */
#   if defined(_WIN32)
#       include <windows.h>
#   endif
#   include <jpeglib.h>
#endif

#ifndef MVE_NO_TIFF_SUPPORT
#   include <tiff.h>
#   include <tiffio.h>
#endif

#include "math/algo.h"
#include "util/exception.h"
#include "util/strings.h"
#include "util/system.h"
#include "mve/image_io.h"

/* Loader limits for reading PPM files. */
#define PPM_MAX_PIXEL_AMOUNT (16384 * 16384) /* 2^28 */

/* The signature to identify MVEI image files and loader limits. */
#define MVEI_FILE_SIGNATURE "\211MVE_IMAGE\n"
#define MVEI_FILE_SIGNATURE_LEN 11
#define MVEI_MAX_PIXEL_AMOUNT (16384 * 16384) /* 2^28 */

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

/*
 * ------------------------ Loading and Saving -----------------------
 */

ByteImage::Ptr
load_file (std::string const& filename)
{
    try
    {
#ifndef MVE_NO_PNG_SUPPORT
        try
        { return load_png_file(filename); }
        catch (util::FileException& e) { throw; }
        catch (util::Exception& e) {}
#endif

#ifndef MVE_NO_JPEG_SUPPORT
        try
        { return load_jpg_file(filename); }
        catch (util::FileException& e) { throw; }
        catch (util::Exception& e) {}
#endif

#ifndef MVE_NO_TIFF_SUPPORT
        try
        { return load_tiff_file(filename); }
        catch (util::FileException& e) { throw; }
        catch (util::Exception& e) {}
#endif

        try
        { return load_ppm_file(filename); }
        catch (util::FileException& e) { throw; }
        catch (util::Exception& e) {}

        try
        {
            ImageHeaders header = load_mvei_file_headers(filename);
            if (header.type != IMAGE_TYPE_UINT8)
                throw util::Exception("Invalid image format");
            ImageBase::Ptr image = load_mvei_file(filename);
            return std::dynamic_pointer_cast<ByteImage>(image);
        }
        catch (util::FileException& e) { throw; }
        catch (util::Exception& e) {}
    }
    catch (util::FileException& e)
    {
        throw util::Exception(filename + ": ", e.what());
    }

    throw util::Exception(filename, ": Cannot determine image format");
}

ImageHeaders
load_file_headers (std::string const& filename)
{
    try
    {
#ifndef MVE_NO_PNG_SUPPORT
        try
        { return load_png_file_headers(filename); }
        catch (util::FileException&) { throw; }
        catch (util::Exception&) {}
#endif

#ifndef MVE_NO_JPEG_SUPPORT
        try
        { return load_jpg_file_headers(filename); }
        catch (util::FileException&) { throw; }
        catch (util::Exception&) {}
#endif

#ifndef MVE_NO_TIFF_SUPPORT
        try
        { return load_tiff_file_headers(filename); }
        catch (util::FileException&) { throw; }
        catch (util::Exception&) {}
#endif

        try
        { return load_mvei_file_headers(filename); }
        catch (util::FileException&) { throw; }
        catch (util::Exception&) {}
    }
    catch (util::FileException& e)
    {
        throw util::Exception(filename + ": ", e.what());
    }

    throw util::Exception(filename, ": Cannot determine image format");
}

void
save_file (ByteImage::ConstPtr image, std::string const& filename)
{
    using namespace util::string;
    std::string fext4 = lowercase(right(filename, 4));
    std::string fext5 = lowercase(right(filename, 5));

#ifndef MVE_NO_JPEG_SUPPORT
    if (fext4 == ".jpg" || fext5 == ".jpeg")
    {
        save_jpg_file(image, filename, 85);
        return;
    }
#endif

#ifndef MVE_NO_PNG_SUPPORT
    if (fext4 == ".png")
    {
        save_png_file(image, filename);
        return;
    }
#endif

#ifndef MVE_NO_TIFF_SUPPORT
    if (fext4 == ".tif" || fext5 == ".tiff")
    {
        save_tiff_file(image, filename);
        return;
    }
#endif

    if (fext4 == ".ppm")
    {
        save_ppm_file(image, filename);
        return;
    }

    throw util::Exception("Output filetype not supported");
}

void
save_file (ByteImage::Ptr image, std::string const& filename)
{
    save_file(ByteImage::ConstPtr(image), filename);
}

void
save_file (FloatImage::ConstPtr image, std::string const& filename)
{

    using namespace util::string;
    std::string fext4 = lowercase(right(filename, 4));
    if (fext4 == ".pfm")
    {
        save_pfm_file(image, filename);
        return;
    }

    throw util::Exception("Output filetype not supported");
}

void
save_file (FloatImage::Ptr image, std::string const& filename)
{
    save_file(FloatImage::ConstPtr(image), filename);
}

/* ---------------------------------------------------------------- */

#ifndef MVE_NO_PNG_SUPPORT

namespace
{
    void
    load_png_headers_intern (FILE* fp, ImageHeaders* headers,
        png_structp* png, png_infop* png_info)
    {
        /* Identify the PNG signature. */
        png_byte signature[8];
        if (std::fread(signature, 1, 8, fp) != 8)
        {
            std::fclose(fp);
            throw util::Exception("PNG signature could not be read");
        }
        if (png_sig_cmp(signature, 0, 8) != 0)
        {
            std::fclose(fp);
            throw util::Exception("PNG signature did not match");
        }

        /* Initialize PNG structures. */
        *png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
            nullptr, nullptr, nullptr);
        if (!*png)
        {
            std::fclose(fp);
            throw util::Exception("Out of memory");
        }

        *png_info = png_create_info_struct(*png);
        if (!*png_info)
        {
            png_destroy_read_struct(png, nullptr, nullptr);
            std::fclose(fp);
            throw util::Exception("Out of memory");
        }

        /* Init PNG file IO */
        png_init_io(*png, fp);
        png_set_sig_bytes(*png, 8);

        /* Read PNG header info. */
        png_read_info(*png, *png_info);

        headers->width = png_get_image_width(*png, *png_info);
        headers->height = png_get_image_height(*png, *png_info);
        headers->channels = png_get_channels(*png, *png_info);

        int const bit_depth = png_get_bit_depth(*png, *png_info);
        if (bit_depth <= 8)
            headers->type = IMAGE_TYPE_UINT8;
        else if (bit_depth == 16)
            headers->type = IMAGE_TYPE_UINT16;
        else
        {
            png_destroy_read_struct(png, png_info, nullptr);
            std::fclose(fp);
            throw util::Exception("PNG with unknown bit depth");
        }
    }
}

ByteImage::Ptr
load_png_file (std::string const& filename)
{
    // TODO: use throw-safe FILE* wrapper
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        throw util::FileException(filename, std::strerror(errno));

    /* Read PNG header info. */
    ImageHeaders headers;
    png_structp png = nullptr;
    png_infop png_info = nullptr;
    load_png_headers_intern(fp, &headers, &png, &png_info);

    /* Check if bit depth is valid. */
    int const bit_depth = png_get_bit_depth(png, png_info);
    if (bit_depth > 8)
    {
        png_destroy_read_struct(&png, &png_info, nullptr);
        std::fclose(fp);
        throw util::Exception("PNG with more than 8 bit");
    }

    /* Apply transformations. */
    int const color_type = png_get_color_type(png, png_info);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, png_info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    /* Update the info struct to reflect the transformations. */
    png_read_update_info(png, png_info);

    /* Create image. */
    ByteImage::Ptr image = ByteImage::create();
    image->allocate(headers.width, headers.height, headers.channels);
    ByteImage::ImageData& data = image->get_data();

    /* Setup row pointers. */
    std::vector<png_bytep> row_pointers;
    row_pointers.resize(headers.height);
    for (int i = 0; i < headers.height; ++i)
        row_pointers[i] = &data[i * headers.width * headers.channels];

    /* Read the whole PNG in memory. */
    png_read_image(png, &row_pointers[0]);

    /* Clean up. */
    png_destroy_read_struct(&png, &png_info, nullptr);
    std::fclose(fp);

    return image;
}

ImageHeaders
load_png_file_headers (std::string const& filename)
{
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        throw util::FileException(filename, std::strerror(errno));

    /* Read PNG header info. */
    ImageHeaders headers;
    png_structp png = nullptr;
    png_infop png_info = nullptr;
    load_png_headers_intern(fp, &headers, &png, &png_info);

    /* Clean up. */
    png_destroy_read_struct(&png, &png_info, nullptr);
    std::fclose(fp);

    return headers;
}

void
save_png_file (ByteImage::ConstPtr image,
    std::string const& filename, int compression_level)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    FILE *fp = std::fopen(filename.c_str(), "wb");
    if (!fp)
        throw util::FileException(filename, std::strerror(errno));

    //png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING,
    //    (png_voidp)user_error_ptr, user_error_fn, user_warning_fn);
    png_structp png_ptr = png_create_write_struct
        (PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

    if (!png_ptr)
    {
        std::fclose(fp);
        throw util::Exception("Out of memory");
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, (png_infopp)nullptr);
        std::fclose(fp);
        throw util::Exception("Out of memory");
    }

    png_init_io(png_ptr, fp);

    // void write_row_callback(png_ptr, png_uint_32 row, int pass);
    //png_set_write_status_fn(png_ptr, write_row_callback);

    /* Determine color type to be written. */
    int color_type;
    switch (image->channels())
    {
        case 1: color_type = PNG_COLOR_TYPE_GRAY; break;
        case 2: color_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
        case 3: color_type = PNG_COLOR_TYPE_RGB; break;
        case 4: color_type = PNG_COLOR_TYPE_RGB_ALPHA; break;
        default:
        {
            png_destroy_write_struct(&png_ptr, &info_ptr);
            std::fclose(fp);
            throw util::Exception("Cannot determine image color type");
        }
    }

    /* Set compression level (6 seems to be the default). */
    png_set_compression_level(png_ptr, compression_level);

    /* Write image. */
    png_set_IHDR(png_ptr, info_ptr, image->width(), image->height(),
        8 /* Bit depth */, color_type, PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    /* Setup row pointers. */
    std::vector<png_bytep> row_pointers;
    row_pointers.resize(image->height());
    ByteImage::ImageData const& data = image->get_data();
    for (int i = 0; i < image->height(); ++i)
        row_pointers[i] = const_cast<png_bytep>(
            &data[i * image->width() * image->channels()]);

    /* Setup transformations. */
    int png_transforms = PNG_TRANSFORM_IDENTITY;
    //png_transforms |= PNG_TRANSFORM_INVERT_ALPHA;

    /* Write to file. */
    png_set_rows(png_ptr, info_ptr, &row_pointers[0]);
    png_write_png(png_ptr, info_ptr, png_transforms, nullptr);
    png_write_end(png_ptr, info_ptr);

    /* Cleanup. */
    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
}

#endif /* MVE_NO_PNG_SUPPORT */

/* ---------------------------------------------------------------- */

#ifndef MVE_NO_JPEG_SUPPORT

void
jpg_error_handler (j_common_ptr /*cinfo*/)
{
    throw util::Exception("JPEG format not recognized");
}

void
jpg_message_handler (j_common_ptr /*cinfo*/, int msg_level)
{
    if (msg_level < 0)
        throw util::Exception("JPEG data corrupt");
}

ByteImage::Ptr
load_jpg_file (std::string const& filename, std::string* exif)
{
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        throw util::FileException(filename, std::strerror(errno));

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    ByteImage::Ptr image;
    try
    {
        /* Setup error handler and JPEG reader. */
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = &jpg_error_handler;
        jerr.emit_message = &jpg_message_handler;
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, fp);

        if (exif)
        {
            /* Request APP1 marker to be saved (this is the EXIF data). */
            jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xffff);
        }

        /* Read JPEG header. */
        int ret = jpeg_read_header(&cinfo, static_cast<boolean>(false));
        if (ret != JPEG_HEADER_OK)
            throw util::Exception("JPEG header not recognized");

        /* Examine JPEG markers. */
        if (exif)
        {
            jpeg_saved_marker_ptr marker = cinfo.marker_list;
            if (marker != nullptr && marker->marker == JPEG_APP0 + 1
                && marker->data_length > 6
                && std::equal(marker->data, marker->data + 6, "Exif\0\0"))
            {
                char const* data = reinterpret_cast<char const*>(marker->data);
                exif->append(data, data + marker->data_length);
            }
        }

        if (cinfo.out_color_space != JCS_GRAYSCALE
            && cinfo.out_color_space != JCS_RGB)
            throw util::Exception("Invalid JPEG color space");

        /* Create image. */
        int const width = cinfo.image_width;
        int const height = cinfo.image_height;
        int const channels = (cinfo.out_color_space == JCS_RGB ? 3 : 1);
        image = ByteImage::create(width, height, channels);
        ByteImage::ImageData& data = image->get_data();

        /* Start decompression. */
        jpeg_start_decompress(&cinfo);

        unsigned char* data_ptr = &data[0];
        while (cinfo.output_scanline < cinfo.output_height)
        {
            jpeg_read_scanlines(&cinfo, &data_ptr, 1);
            data_ptr += channels * cinfo.output_width;
        }

        /* Shutdown JPEG decompression. */
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
    }
    catch (...)
    {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw;
    }

    return image;
}

ImageHeaders
load_jpg_file_headers (std::string const& filename)
{
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (fp == nullptr)
        throw util::FileException(filename, std::strerror(errno));

    jpeg_decompress_struct cinfo;
    jpeg_error_mgr jerr;
    ImageHeaders headers;
    try
    {
        /* Setup error handler and JPEG reader. */
        cinfo.err = jpeg_std_error(&jerr);
        jerr.error_exit = &jpg_error_handler;
        jerr.emit_message = &jpg_message_handler;
        jpeg_create_decompress(&cinfo);
        jpeg_stdio_src(&cinfo, fp);

        /* Read JPEG header. */
        int ret = jpeg_read_header(&cinfo, static_cast<boolean>(false));
        if (ret != JPEG_HEADER_OK)
            throw util::Exception("JPEG header not recognized");

        if (cinfo.out_color_space != JCS_GRAYSCALE
            && cinfo.out_color_space != JCS_RGB)
            throw util::Exception("Invalid JPEG color space");

        headers.width = cinfo.image_width;
        headers.height = cinfo.image_height;
        headers.channels = (cinfo.out_color_space == JCS_RGB ? 3 : 1);
        headers.type = IMAGE_TYPE_UINT8;

        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
    }
    catch (...)
    {
        jpeg_destroy_decompress(&cinfo);
        std::fclose(fp);
        throw;
    }

    return headers;
}

// http://download.blender.org/source/chest/blender_2.03_tree/jpeg/example.c
void
save_jpg_file (ByteImage::ConstPtr image, std::string const& filename, int quality)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    if (image->channels() != 1 && image->channels() != 3)
        throw util::Exception("Invalid image color space");

    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp)
        throw util::FileException(filename, std::strerror(errno));

    jpeg_compress_struct cinfo;
    jpeg_error_mgr jerr;

    /* Setup error handler and info object. */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    /* Specify image dimensions. */
    cinfo.image_width = image->width();
    cinfo.image_height = image->height();
    cinfo.input_components = image->channels();
    cinfo.in_color_space = (image->channels() == 1 ? JCS_GRAYSCALE : JCS_RGB);

    /* Set default compression parameters. */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    ByteImage::ImageData const& data = image->get_data();
    int row_stride = image->width() * image->channels();
    while (cinfo.next_scanline < cinfo.image_height)
    {
        JSAMPROW row_pointer = const_cast<JSAMPROW>(
            &data[cinfo.next_scanline * row_stride]);
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
}

#endif /* MVE_NO_JPEG_SUPPORT */

/* ---------------------------------------------------------------- */

#ifndef MVE_NO_TIFF_SUPPORT

void
tiff_error_handler (char const* /*module*/, char const* fmt, va_list ap)
{
    char msg[2048];
    ::vsprintf(msg, fmt, ap);
    throw util::Exception(msg);
}

ImageHeaders
load_tiff_file_headers (std::string const& filename)
{
    ImageHeaders headers;
    TIFFSetWarningHandler(nullptr);
    TIFFSetErrorHandler(tiff_error_handler);

    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (tif == nullptr)
        throw util::FileException(filename, "TIFF file format not recognized");

    try
    {
        uint32_t width, height;
        uint16_t channels, bits, sampleFormat;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

        headers.width = width;
        headers.height = height;
        headers.channels = channels;
        headers.type = IMAGE_TYPE_UNKNOWN;

        if (bits == 8) {
            switch(sampleFormat) {
                case SAMPLEFORMAT_UINT:
                    headers.type = IMAGE_TYPE_UINT8;
                    break;
                case SAMPLEFORMAT_INT:
                    headers.type = IMAGE_TYPE_SINT8;
                    break;
                case SAMPLEFORMAT_IEEEFP:
                    headers.type = IMAGE_TYPE_FLOAT;
                    break;
                default:
                    break;
            }
        } else if (bits == 16) {
            switch(sampleFormat) {
                case SAMPLEFORMAT_UINT:
                    headers.type = IMAGE_TYPE_UINT16;
                    break;
                case SAMPLEFORMAT_INT:
                    headers.type = IMAGE_TYPE_SINT16;
                    break;
                case SAMPLEFORMAT_IEEEFP:
                    headers.type = IMAGE_TYPE_FLOAT;
                    break;
                default:
                    break;
            }
        } else if (bits == 32) {
            switch(sampleFormat) {
                case SAMPLEFORMAT_UINT:
                    headers.type = IMAGE_TYPE_UINT32;
                    break;
                case SAMPLEFORMAT_INT:
                    headers.type = IMAGE_TYPE_SINT32;
                    break;
                case SAMPLEFORMAT_IEEEFP:
                    headers.type = IMAGE_TYPE_FLOAT;
                    break;
                default:
                    break;
            }
        } else if (bits == 64) {
            switch(sampleFormat) {
                case SAMPLEFORMAT_UINT:
                    headers.type = IMAGE_TYPE_UINT64;
                    break;
                case SAMPLEFORMAT_INT:
                    headers.type = IMAGE_TYPE_SINT64;
                    break;
                case SAMPLEFORMAT_IEEEFP:
                    headers.type = IMAGE_TYPE_FLOAT;
                    break;
                default:
                    break;
            }
        }

        if (headers.type == IMAGE_TYPE_UNKNOWN){
            throw util::Exception("TIFF file has unsupported bits and/or sample format.");
        }

        TIFFClose(tif);
        return headers;
    }
    catch (std::exception& e)
    {
        TIFFClose(tif);
        throw;
    }
}

ByteImage::Ptr
load_tiff_file (std::string const& filename)
{
    TIFFSetWarningHandler(nullptr);
    TIFFSetErrorHandler(tiff_error_handler);

    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (!tif)
        throw util::Exception("TIFF file format not recognized");

    try
    {
        /* Read width and height from TIFF and create MVE image. */
        uint32_t width, height;
        uint16_t channels, bits;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
        if (bits != 8)
            throw util::Exception("Expected 8 bit TIFF file");
        ByteImage::Ptr image = ByteImage::create(width, height, channels);

        /* Scanline based TIFF reading. */
        uint32_t rowstride = TIFFScanlineSize(tif);
        ByteImage::ImageData& data = image->get_data();
        for (uint32_t row = 0; row < height; row++)
        {
            tdata_t row_pointer = &data[row * rowstride];
            TIFFReadScanline(tif, row_pointer, row);
        }

        TIFFClose(tif);
        return image;
    }
    catch (std::exception& e)
    {
        TIFFClose(tif);
        throw;
    }
}

void
save_tiff_file (ByteImage::ConstPtr image, std::string const& filename)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    TIFF* tif = TIFFOpen(filename.c_str(), "w");
    if (!tif)
        throw util::FileException(filename, "Unknown TIFF file error");

    uint32_t width = image->width();
    uint32_t height = image->height();
    uint32_t channels = image->channels();
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    tdata_t buffer = const_cast<uint8_t*>(image->get_data_pointer());
    int64_t ret = TIFFWriteEncodedStrip(tif, 0, buffer,
        image->get_value_amount());

    TIFFClose(tif);

    if (ret < 0)
        throw util::Exception("Error writing TIFF image");
}

RawImage::Ptr
load_tiff_16_file (std::string const& filename)
{
    if (sizeof(uint16_t) != 2)
        throw util::Exception("Need 16bit data type for TIFF image.");

    TIFFSetWarningHandler(nullptr);
    TIFFSetErrorHandler(tiff_error_handler);

    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (!tif)
        throw util::Exception("TIFF file format not recognized");

    try
    {
        /* Read width and height from TIFF and create MVE image. */
        uint32_t width, height;
        uint16_t channels, bits;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
        if (bits != 16)
            throw util::Exception("TIFF file bits per sample don't match");

        RawImage::Ptr image = Image<uint16_t>::create(width, height, channels);

        /* Scanline based TIFF reading. */
        uint32_t rowstride = TIFFScanlineSize(tif) / sizeof(uint16_t);
        Image<uint16_t>::ImageData& data = image->get_data();
        for (uint32_t row = 0; row < height; row++)
        {
            tdata_t row_pointer = &data[row * rowstride];
            TIFFReadScanline(tif, row_pointer, row);
        }

        TIFFClose(tif);
        return image;
    }
    catch (std::exception& e)
    {
        TIFFClose(tif);
        throw;
    }
}

void
save_tiff_16_file (RawImage::ConstPtr image, std::string const& filename)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    TIFF* tif = TIFFOpen(filename.c_str(), "w");
    if (!tif)
        throw util::FileException(filename, "Unknown TIFF file error");

    uint32_t width = image->width();
    uint32_t height = image->height();
    uint32_t channels = image->channels();
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(uint16_t));
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);

    tdata_t buffer = const_cast<uint16_t*>(image->get_data_pointer());
    int64_t ret = TIFFWriteEncodedStrip(tif, 0, buffer,
        image->get_value_amount() * sizeof(uint16_t));

    TIFFClose(tif);

    if (ret < 0)
        throw util::Exception("Error writing TIFF image");
}

FloatImage::Ptr
load_tiff_float_file (std::string const& filename)
{
    if (sizeof(float) != 4)
        throw util::Exception("Need 32bit data type for TIFF image.");

    TIFFSetWarningHandler(nullptr);
    TIFFSetErrorHandler(tiff_error_handler);

    TIFF* tif = TIFFOpen(filename.c_str(), "r");
    if (!tif)
        throw util::Exception("TIFF file format not recognized");

    try
    {
        /* Read width and height from TIFF and create MVE image. */
        uint32_t width, height;
        uint16_t channels, bits;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &channels);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bits);
        if (bits != 32)
            throw util::Exception("TIFF file bits per sample don't match");

        FloatImage::Ptr image = Image<float>::create(width, height, channels);

        /* Scanline based TIFF reading. */
        uint32_t rowstride = TIFFScanlineSize(tif) / sizeof(float);
        Image<float>::ImageData& data = image->get_data();
        for (uint32_t row = 0; row < height; row++)
        {
            tdata_t row_pointer = &data[row * rowstride];
            TIFFReadScanline(tif, row_pointer, row);
        }

        TIFFClose(tif);
        return image;
    }
    catch (std::exception& e)
    {
        TIFFClose(tif);
        throw;
    }
}

void
save_tiff_float_file (FloatImage::ConstPtr image, std::string const& filename)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    TIFF* tif = TIFFOpen(filename.c_str(), "w");
    if (!tif)
        throw util::FileException(filename, "Unknown TIFF file error");

    uint32_t width = image->width();
    uint32_t height = image->height();
    uint32_t channels = image->channels();
    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, channels);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8 * sizeof(float));
    TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);

    tdata_t buffer = const_cast<float*>(image->get_data_pointer());
    int64_t ret = TIFFWriteEncodedStrip(tif, 0, buffer,
        image->get_value_amount() * sizeof(float));

    TIFFClose(tif);

    if (ret < 0)
        throw util::Exception("Error writing TIFF image");
}

#endif /* MVE_NO_TIFF_SUPPORT */

/* ---------------------------------------------------------------- */

/*
 * PFM file format for float images.
 * http://netpbm.sourceforge.net/doc/pfm.html
 * FIXME: Convert to C++ I/O (std::fstream)
 */

FloatImage::Ptr
load_pfm_file (std::string const& filename)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.good())
        throw util::FileException(filename, std::strerror(errno));

    char signature[2];
    in.read(signature, 2);

    // check signature and determine channels
    int channels = 0;
    if (signature[0] == 'P' && signature[1] == 'f')
        channels = 1;
    else if (signature[0] == 'P' && signature[1] == 'F')
        channels = 3;
    else
    {
        in.close();
        throw util::Exception("PPM signature did not match");
    }

    /* Read width and height as well as max value. */
    int width = 0;
    int height = 0;
    float scale = -1.0;
    in >> width >> height >> scale;

    /* Read final whitespace character. */
    char temp;
    in.read(&temp, 1);

    /* Check image width and height. Shouldn't be too large. */
    if (width * height > PPM_MAX_PIXEL_AMOUNT)
    {
        in.close();
        throw util::Exception("Image too friggin huge");
    }

    /* Read image rows in reverse order according to PFM specification. */
    FloatImage::Ptr image = FloatImage::create(width, height, channels);
    std::size_t row_size = image->get_byte_size() / image->height();
    char* ptr = reinterpret_cast<char*>(image->end()) - row_size;
    for (int y = 0; y < height; ++y, ptr -= row_size)
        in.read(ptr, row_size);
    in.close();

    /* Handle endianess. BE if scale > 0, LE if scale < 0. */
    if (scale < 0.0f)
    {
        std::transform(image->begin(), image->end(), image->begin(),
            (float(*)(float const&))util::system::letoh<float>);
        scale = -scale;
    }
    else
    {
        std::transform(image->begin(), image->end(), image->begin(),
            (float(*)(float const&))util::system::betoh<float>);
    }

    /* Handle scale. Multiply image values if scale is not 1.0. */
    if (scale != 1.0f)
    {
        std::for_each(image->begin(), image->end(),
            math::algo::foreach_multiply_with_const<float>(scale));
    }

   return image;
}

void
save_pfm_file (FloatImage::ConstPtr image, std::string const& filename)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    std::string magic_number;
    if (image->channels() == 1)
        magic_number = "Pf";
    else if (image->channels() == 3)
        magic_number = "PF";
    else
        throw std::invalid_argument("Supports 1 and 3 channel images only");

#ifdef HOST_BYTEORDER_LE
    std::string scale = "-1.0"; // we currently don't support scales
#else
    std::string scale = "1.0"; // we currently don't support scales
#endif

    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out.good())
        throw util::FileException(filename, std::strerror(errno));

    out << magic_number << "\n";
    out << image->width() << " " << image->height() << " " << scale << "\n";

    /* Output rows in reverse order according to PFM specification. */
    std::size_t row_size = image->get_byte_size() / image->height();
    char const* ptr = reinterpret_cast<char const*>(image->end()) - row_size;
    for (int y = 0; y < image->height(); ++y, ptr -= row_size)
        out.write(ptr, row_size);
    out.close();
}

/* ---------------------------------------------------------------- */

/*
 * PPM file format for 8 and 16 bit images. The bit8 argument is set to
 * true, 8 bit PPMs with maxval set to less than 256 are loaded.
 * Otherwise, 16 bit PPMs with maxval less than 65536 are loaded.
 * http://netpbm.sourceforge.net/doc/ppm.html
 */
ImageBase::Ptr
load_ppm_file_intern (std::string const& filename, bool bit8)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.good())
        throw util::FileException(filename, std::strerror(errno));

    char signature[2];
    in.read(signature, 2);

    // check signature and determine channels
    int channels = 0;
    if (signature[0] == 'P' && signature[1] == '5')
        channels = 1;
    else if (signature[0] == 'P' && signature[1] == '6')
        channels = 3;
    else
    {
        in.close();
        throw util::Exception("PPM signature did not match");
    }

    /* Read width and height as well as max value. */
    int width = 0;
    int height = 0;
    int maxval = 0;
    in >> width >> height >> maxval;

    /* Read final whitespace character. */
    char temp;
    in.read(&temp, 1);

    /* Check image width and height. Shouldn't be too large. */
    if (width * height > PPM_MAX_PIXEL_AMOUNT)
    {
        in.close();
        throw util::Exception("Image too friggin huge");
    }

    ImageBase::Ptr ret;

    /* Max value should be in <256 for 8 bit and <65535 for 16 bit. */
    if (maxval < 256 && bit8)
    {
        /* Read image content. */
        ByteImage::Ptr image = ByteImage::create(width, height, channels);
        in.read(image->get_byte_pointer(), image->get_byte_size());
        ret = image;
    }
    else if (maxval < 65536 && !bit8)
    {
        /* Read image content, convert from big-endian to host order. */
        RawImage::Ptr image = RawImage::create(width, height, channels);
        in.read(image->get_byte_pointer(), image->get_byte_size());
        for (int i = 0; i < image->get_value_amount(); ++i)
            image->at(i) = util::system::betoh(image->at(i));
        ret = image;
    }
    else
    {
        in.close();
        throw util::Exception("PPM max value is invalid");
    }

    in.close();
    return ret;
}

RawImage::Ptr
load_ppm_16_file (std::string const& filename)
{
    return std::dynamic_pointer_cast<RawImage>
        (load_ppm_file_intern(filename, false));
}

ByteImage::Ptr
load_ppm_file (std::string const& filename)
{
    return std::dynamic_pointer_cast<ByteImage>
        (load_ppm_file_intern(filename, true));
}

void
save_ppm_file_intern (ImageBase::ConstPtr image, std::string const& filename)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    std::string magic_number;
    if (image->channels() == 1)
        magic_number = "P5";
    else if (image->channels() == 3)
        magic_number = "P6";
    else
        throw std::invalid_argument("Supports 1 and 3 channel images only");

    int maxval = 0;
    if (image->get_type() == IMAGE_TYPE_UINT8)
        maxval = 255;
    else if (image->get_type() == IMAGE_TYPE_UINT16)
        maxval = 65535;
    else
        throw std::invalid_argument("Invalid image format");

    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out.good())
        throw util::FileException(filename, std::strerror(errno));

    out << magic_number << "\n";
    out << image->width() << " " << image->height() << " " << maxval << "\n";

    if (image->get_type() == IMAGE_TYPE_UINT8)
    {
        /* Byte images can be saved as-is. */
        out.write(image->get_byte_pointer(), image->get_byte_size());
    }
    else
    {
        /* PPM is big-endian, so we need to convert 16 bit data. */
        RawImage::ConstPtr handle
            = std::dynamic_pointer_cast<RawImage const>(image);
        for (int i = 0; i < handle->get_value_amount(); ++i)
        {
            RawImage::ValueType value = util::system::betoh(handle->at(i));
            out.write((char const*)(&value), sizeof(RawImage::ValueType));
        }
    }
    out.close();
}

void
save_ppm_16_file (RawImage::ConstPtr image, std::string const& filename)
{
    save_ppm_file_intern(image, filename);
}

void
save_ppm_file (ByteImage::ConstPtr image, std::string const& filename)
{
    save_ppm_file_intern(image, filename);
}

/* ---------------------------------------------------------------- */

namespace
{
    void
    load_mvei_headers_intern (std::istream& in, ImageHeaders* headers)
    {
        char signature[MVEI_FILE_SIGNATURE_LEN];
        in.read(signature, MVEI_FILE_SIGNATURE_LEN);
        if (!std::equal(signature, signature + MVEI_FILE_SIGNATURE_LEN,
            MVEI_FILE_SIGNATURE))
            throw util::Exception("Invalid file signature");

        /* Read image headers data, */
        int32_t width, height, channels, raw_type;
        in.read(reinterpret_cast<char*>(&width), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&height), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&channels), sizeof(int32_t));
        in.read(reinterpret_cast<char*>(&raw_type), sizeof(int32_t));

        if (!in.good())
            throw util::Exception("Error reading headers");

        headers->width = width;
        headers->height = height;
        headers->channels = channels;
        headers->type = static_cast<ImageType>(raw_type);
    }
}

ImageBase::Ptr
load_mvei_file (std::string const& filename)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.good())
        throw util::FileException(filename, std::strerror(errno));

    /* Load image header data. */
    ImageHeaders headers;
    load_mvei_headers_intern(in, &headers);
    if (headers.width * headers.height > MVEI_MAX_PIXEL_AMOUNT)
        throw util::Exception("Ridiculously large image");

    /* Load image data. */
    ImageBase::Ptr image = create_for_type(headers.type,
        headers.width, headers.height, headers.channels);
    in.read(image->get_byte_pointer(), image->get_byte_size());
    if (!in.good())
        throw util::FileException(filename, std::strerror(errno));

    return image;
}

ImageHeaders
load_mvei_file_headers (std::string const& filename)
{
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.good())
        throw util::FileException(filename, std::strerror(errno));

    ImageHeaders headers;
    load_mvei_headers_intern(in, &headers);
    return headers;
}

void
save_mvei_file (ImageBase::ConstPtr image, std::string const& filename)
{
    if (image == nullptr)
        throw std::invalid_argument("Null image given");

    // Note: This is a narrowing conversion.
    int32_t width = static_cast<int32_t>(image->width());
    int32_t height = static_cast<int32_t>(image->height());
    int32_t channels = static_cast<int32_t>(image->channels());
    int32_t type = image->get_type();
    char const* data = image->get_byte_pointer();
    std::size_t size = image->get_byte_size();

    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out.good())
        throw util::FileException(filename, std::strerror(errno));

    out.write(MVEI_FILE_SIGNATURE, MVEI_FILE_SIGNATURE_LEN);
    out.write(reinterpret_cast<char const*>(&width), sizeof(int32_t));
    out.write(reinterpret_cast<char const*>(&height), sizeof(int32_t));
    out.write(reinterpret_cast<char const*>(&channels), sizeof(int32_t));
    out.write(reinterpret_cast<char const*>(&type), sizeof(int32_t));
    out.write(data, size);

    if (!out.good())
        throw util::FileException(filename, std::strerror(errno));
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END

