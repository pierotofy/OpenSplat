/*
 * This is a heavily rewritten version of the code found at:
 * http://code.google.com/p/easyexif/
 *
 * EXIF specifications can be found here:
 * http://www.cipa.jp/std/documents/e/DC-008-2012_E.pdf
 *
 * Modifications: Copyright (c) 2011 - 2014 Simon Fuhrmann
 * Original code: Copyright (c) 2010 - 2013 Mayank Lahiri <mlahiri@gmail.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * -- Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * -- Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
 * NO EVENT SHALL THE FREEBSD PROJECT OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "util/system.h"
#include "util/strings.h"
#include "mve/image_exif.h"

MVE_NAMESPACE_BEGIN
MVE_IMAGE_NAMESPACE_BEGIN

namespace
{
    uint32_t
    parse_u32 (unsigned char const* buf, bool intel)
    {
        uint32_t value;
        std::copy(buf, buf + 4, reinterpret_cast<unsigned char*>(&value));
        return intel ? util::system::letoh(value) : util::system::betoh(value);
    }

    int32_t
    parse_s32 (unsigned char const* buf, bool intel)
    {
        int32_t value;
        std::copy(buf, buf + 4, reinterpret_cast<unsigned char*>(&value));
        return intel ? util::system::letoh(value) : util::system::betoh(value);
    }

    uint16_t
    parse_u16 (unsigned char const* buf, bool intel)
    {
        uint16_t value;
        std::copy(buf, buf + 2, reinterpret_cast<unsigned char*>(&value));
        return intel ? util::system::letoh(value) : util::system::betoh(value);
    }

    float
    parse_rational_u64 (unsigned char const* buf, bool intel)
    {
        uint32_t numerator = parse_u32(buf, intel);
        uint32_t denominator = parse_u32(buf + 4, intel);
        if (denominator == 0)
            return std::numeric_limits<float>::infinity();
        return static_cast<float>(numerator) / static_cast<float>(denominator);
    }

    float
    parse_rational_s64 (unsigned char const* buf, bool intel)
    {
        int32_t numerator = parse_s32(buf, intel);
        int32_t denominator = parse_s32(buf + 4, intel);
        if (denominator == 0)
            return std::numeric_limits<float>::infinity();
        return static_cast<float>(numerator) / static_cast<float>(denominator);
    }

    void
    copy_exif_string (unsigned char const* buf, unsigned int len,
        std::string* dest)
    {
        dest->clear();
        if (len == 0)
            return;
        dest->append(reinterpret_cast<char const*>(buf), len - 1);
    }

    /* EXIF type specifications. */
    enum ExifType
    {
        EXIF_TYPE_BYTE = 1,       // 8 bit
        EXIF_TYPE_ASCII = 2,      // 8 bit character
        EXIF_TYPE_USHORT = 3,     // 16 bit
        EXIF_TYPE_ULONG = 4,      // 32 bit
        EXIF_TYPE_URATIONAL = 5,  // 64 bit
        EXIF_TYPE_SBYTE = 6,      // 8 bit
        EXIF_TYPE_UNDEF = 7,      // 8 bit (field dependent value)
        EXIF_TYPE_SSHORT = 8,     // 16 bit
        EXIF_TYPE_SLONG = 9,      // 32 bit
        EXIF_TYPE_SRATIONAL = 10, // 64 bit
        EXIF_TYPE_FLOAT = 11,     // 32 bit
        EXIF_TYPE_DOUBLE = 12     // 64 bit
    };

    bool
    ifd_is_offset (unsigned short type, unsigned int count)
    {
        return (type == EXIF_TYPE_BYTE && count > 4)
            || (type == EXIF_TYPE_ASCII && count > 4)
            || (type == EXIF_TYPE_USHORT && count > 2)
            || (type == EXIF_TYPE_ULONG && count > 1)
            || (type == EXIF_TYPE_URATIONAL && count > 0)
            || (type == EXIF_TYPE_SBYTE && count > 4)
            || (type == EXIF_TYPE_UNDEF && count > 4)
            || (type == EXIF_TYPE_SSHORT && count > 2)
            || (type == EXIF_TYPE_SLONG && count > 1)
            || (type == EXIF_TYPE_SRATIONAL && count > 0)
            || (type == EXIF_TYPE_FLOAT && count > 1)
            || (type == EXIF_TYPE_DOUBLE && count > 0);
    }

    float
    apex_time_to_exposure (float apex_time)
    {
        return 1.0f / std::pow(2.0f, apex_time);
    }
}  /* namespace */

/* ---------------------------------------------------------------- */

ExifInfo
exif_extract (char const* data, std::size_t len, bool is_jpeg)
{
    /* Data buffer. */
    unsigned char const* buf = reinterpret_cast<unsigned char const*>(data);
    /* Current offset into data buffer. */
    std::size_t offs = 0;
    /* Intel byte alignment. */
    bool align_intel = true;

    /* Prepare return structure. */
    ExifInfo result;

    /*
     * Scan for EXIF header and do a sanity check.
     * The full EXIF header with signature looks as follows:
     *
     *   2 bytes: EXIF header: 0xFFD8
     *   2 bytes: Section size
     *   6 bytes: "Exif\0\0" ASCII signature
     *   2 bytes: TIFF header (either "II" or "MM" byte alignment)
     *   2 bytes: TIFF magic: 0x2A00
     *   4 bytes: Offset to first IFD
     */
    if (is_jpeg)
    {
        /*
         * Make sanity check that this is really a JPEG file.
         * Every JPEG file starts with 0xFFD8 (and ends with 0xFFD9).
         */
        if (buf[0] != 0xFF || buf[1] != 0xD8)
            throw std::invalid_argument("Invalid JPEG signature.");

        /* Scan forward and search for the EXIF header (0xFF 0xE1). */
        for (offs = 0; offs < len - 1; offs++)
            if (buf[offs] == 0xFF && buf[offs + 1] == 0xE1)
                break;
        if (offs == len - 1)
            throw std::invalid_argument("Cannot find EXIF marker!");
        offs += 4;
    }

    /* At least 14 more bytes required for a valid EXIF. */
    if (offs + 14 > len)
        throw std::invalid_argument("EXIF data corrupt (header)");

    /* Check EXIF signature. */
    if (!std::equal(buf + offs, buf + offs + 6, "Exif\0\0"))
        throw std::invalid_argument("Cannot find EXIF signature");
    offs += 6;

    /* Get byte alignment (Intel "little endian" or Motorola "big endian"). */
    std::size_t tiff_header_offset = offs;
    if (std::equal(buf + offs, buf + offs + 2, "II"))
        align_intel = true;
    else if (std::equal(buf + offs, buf + offs + 2, "MM"))
        align_intel = false;
    else
        throw std::invalid_argument("Cannot find EXIF byte alignment");
    offs += 2;

    /* Check TIFF magic number. */
    uint16_t tiff_magic_number = parse_u16(buf + offs, align_intel);
    if (tiff_magic_number != 0x2a)
        throw std::invalid_argument("Cannot find TIFF magic bytes");
    offs += 2;

    /* Get offset and jump into first IFD (Image File Directory). */
    std::size_t first_ifd_offset = parse_u32(buf + offs, align_intel);
    offs += first_ifd_offset - 4;  // Why subtract 4?

    /*
     * The first two bytes of the IFD section contain the number of directory
     * entries in the section. Then the list of IFD entries follows. After all
     * entries, the last 4 bytes contain an offset to the next IFD.
     */
    if (offs + 2 > len)
        throw std::invalid_argument("EXIF data corrupt (IFD entries)");
    int num_entries = parse_u16(buf + offs, align_intel);
    offs += 2;
    if (num_entries < 0 || num_entries > 10000)
        throw std::invalid_argument("EXIF data corrupt (number of IFDs)");
    if (offs + 4 + 12 * num_entries > len)
        throw std::invalid_argument("EXIF data corrupt (IFD table)");

    /*
     * Each IFD entry consists of 12 bytes.
     *
     *   2 bytes: Identififes tag type (as in Tagged Image File Format)
     *   2 bytes: Field data type (byte, ASCII, short int, long int, ...)
     *   4 bytes: Number of values
     *   4 bytes: Either the value itself or an offset to the values
     *
     * While parsing the IFD entries, try to find SubIFD and GPS offsets.
     */
    std::size_t exif_sub_ifd_offset = 0;
    //unsigned int gps_sub_ifd_offset = 0;
    for (int i = 0; i < num_entries; ++i)
    {
        unsigned short tag = parse_u16(buf + offs, align_intel);
        unsigned short type = parse_u16(buf + offs + 2, align_intel);
        unsigned int ncomp = parse_u32(buf + offs + 4, align_intel);
        unsigned int coffs = parse_u32(buf + offs + 8, align_intel);

        /*
         * Compute offset to the entry value. Depending on 'type' and 'ncomp'
         * the 'coffs' variable contains the value itself or an offset.
         * In case of an offset, it is relative to the TIFF header.
         */
        std::size_t buf_off = offs + 8;
        if (ifd_is_offset(type, ncomp))
            buf_off = tiff_header_offset + coffs;
        if (buf_off + ncomp > len)
            throw std::invalid_argument("EXIF data corrupt (IFD entry)");

        switch (tag)
        {
            case 0x8769: // EXIF subIFD offset.
                exif_sub_ifd_offset = tiff_header_offset + coffs;
                break;

            //case 0x8825: // GPS IFS offset.
            //    gps_sub_ifd_offset = tiff_header_offset + coffs;
            //    break;

            case 0x102: // Bits per color sample.
                if (type == EXIF_TYPE_USHORT)
                    result.bits_per_sample = parse_u16(buf + buf_off, align_intel);
                break;

            case 0x112: // Image orientation.
                if (type == EXIF_TYPE_USHORT)
                    result.orientation = parse_u16(buf + buf_off, align_intel);
                break;

            case 0x10F: // Digicam manufacturer.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.camera_maker);
                break;

            case 0x110: // Digicam model.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.camera_model);
                break;

            case 0x132: // EXIF/TIFF date/time of image.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.date_modified);
                break;

            case 0x10E: // Image description.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.description);
                break;

            case 0x131: // Software used to process the image.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.software);
                break;

            case 0x8298: // Copyright information.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.copyright);
                break;

            case 0x013b: // Artist information.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.artist);
                break;

            default:
                break;
        }
        offs += 12;
    }

    /* Check if a SubIFD section exists. */
    if (exif_sub_ifd_offset == 0)
        return result;

    /* Parse SubIFD section. */
    offs = exif_sub_ifd_offset;
    if (offs + 2 > len)
        throw std::invalid_argument("EXIF data corrupt (SubIFD entries)");
    num_entries = parse_u16(buf + offs, align_intel);
    offs += 2;
    if (num_entries < 0 || num_entries > 10000)
        throw std::invalid_argument("EXIF data corrupt (number of SubIFDs)");
    if (offs + 4 + 12 * num_entries > len)
        throw std::invalid_argument("EXIF data corrupt (SubIFD table)");

    /* Like with IFD entries, each IFD entry consists of 12 bytes. */
    for (int j = 0; j < num_entries; j++)
    {
        unsigned short tag = parse_u16(buf + offs, align_intel);
        unsigned short type = parse_u16(buf + offs + 2, align_intel);
        unsigned int ncomp = parse_u32(buf + offs + 4, align_intel);
        unsigned int coffs = parse_u32(buf + offs + 8, align_intel);

        std::size_t buf_off = offs + 8;
        if (ifd_is_offset(type, ncomp))
            buf_off = tiff_header_offset + coffs;
        if (buf_off + ncomp > len)
            throw std::invalid_argument("EXIF data corrupt (SubIFD entry)");

        switch (tag)
        {
            case 0x9003: // Original image date/time string.
                if (type == EXIF_TYPE_ASCII)
                    copy_exif_string(buf + buf_off, ncomp, &result.date_original);
                break;

            case 0x8827: // ISO speed ratings.
                if (type == EXIF_TYPE_USHORT)
                    result.iso_speed = parse_u16(buf + buf_off, align_intel);
                break;

            case 0x920A: // Focal length in mm.
                if (type == EXIF_TYPE_URATIONAL)
                    result.focal_length = parse_rational_u64(buf + buf_off, align_intel);
                break;

            case 0xA405: // Focal length (35 mm equivalent).
                if (type == EXIF_TYPE_USHORT)
                    result.focal_length_35mm = (float)parse_u16(buf + buf_off, align_intel);
                break;

            case 0x829D: // F-stop number.
                if (type == EXIF_TYPE_URATIONAL)
                    result.f_number = parse_rational_u64(buf + buf_off, align_intel);
                break;

            case 0x829A: // Exposure time.
                if (type == EXIF_TYPE_URATIONAL)
                    result.exposure_time = parse_rational_u64(buf + buf_off, align_intel);
                break;

            case 0x9201: // Shutter speed (in APEX format).
                if (type == EXIF_TYPE_SRATIONAL)
                    result.shutter_speed = apex_time_to_exposure(parse_rational_s64(buf + buf_off, align_intel));
                break;

            case 0x9204: // Exposure bias.
                if (type == EXIF_TYPE_URATIONAL)
                    result.exposure_bias = parse_rational_u64(buf + buf_off, align_intel);
                break;

            case 0x9209: // Flash mode.
                if (type == EXIF_TYPE_USHORT)
                    result.flash_mode = parse_u16(buf + buf_off, align_intel);
                break;

            case 0xA002: // Image width.
                if (type == EXIF_TYPE_USHORT)
                    result.image_width = parse_u16(buf + buf_off, align_intel);
                if (type == EXIF_TYPE_ULONG)
                    result.image_width = parse_u32(buf + buf_off, align_intel);
                break;

            case 0xA003: // Image height.
                if (type == EXIF_TYPE_USHORT)
                    result.image_height = parse_u16(buf + buf_off, align_intel);
                if (type == EXIF_TYPE_ULONG)
                    result.image_height = parse_u32(buf + buf_off, align_intel);
                break;

            default:
                break;
        }
        offs += 12;
    }

    return result;
}

/* ---------------------------------------------------------------- */

namespace
{
    template <typename T>
    std::string
    debug_print (T const& value, std::string extra = "")
    {
        if (value < T(0))
            return "<not set>";
        else
            return util::string::get(value) + extra;
    }

    std::string
    debug_print (std::string const& value)
    {
        if (value.empty())
            return "<not set>";
        else
            return value;
    }
}

void
exif_debug_print (std::ostream& stream, ExifInfo const& exif, bool indent)
{
    int const width = indent ? 22 : 0;
    stream
        << std::setw(width) << std::left << "Camera manufacturer: "
        << debug_print(exif.camera_maker) << std::endl
        << std::setw(width) << std::left << "Camera model: "
        << debug_print(exif.camera_model) << std::endl
        << std::setw(width) << std::left << "Date (modified): "
        << debug_print(exif.date_modified) << std::endl
        << std::setw(width) << std::left << "Date (original): "
        << debug_print(exif.date_original) << std::endl
        << std::setw(width) << std::left << "Description: "
        << debug_print(exif.description) << std::endl
        << std::setw(width) << std::left << "Software: "
        << debug_print(exif.software) << std::endl
        << std::setw(width) << std::left << "Copyright info: "
        << debug_print(exif.copyright) << std::endl
        << std::setw(width) << std::left << "Artist info: "
        << debug_print(exif.artist) << std::endl

        << std::setw(width) << std::left << "ISO speed: "
        << debug_print(exif.iso_speed) << std::endl
        << std::setw(width) << std::left << "Bits per sample: "
        << debug_print(exif.bits_per_sample) << std::endl
        << std::setw(width) << std::left << "Image Orientation: "
        << debug_print(exif.orientation) << std::endl
        << std::setw(width) << std::left << "Focal length: "
        << debug_print(exif.focal_length, " mm") << std::endl
        << std::setw(width) << std::left << "Focal length (35mm): "
        << debug_print(exif.focal_length_35mm, " mm") << std::endl
        << std::setw(width) << std::left << "F-Number: "
        << debug_print(exif.f_number) << std::endl
        << std::setw(width) << std::left << "Exposure time: "
        << debug_print(exif.exposure_time, " sec") << std::endl
        << std::setw(width) << std::left << "Exposure bias: "
        << debug_print(exif.exposure_bias) << std::endl
        << std::setw(width) << std::left << "Shutter speed: "
        << debug_print(exif.shutter_speed, " sec") << std::endl
        << std::setw(width) << std::left << "Flash mode: "
        << debug_print(exif.flash_mode) << std::endl
        << std::setw(width) << std::left << "Image width: "
        << debug_print(exif.image_width, " pixel") << std::endl
        << std::setw(width) << std::left << "Image height: "
        << debug_print(exif.image_height, " pixel") << std::endl;
}

MVE_IMAGE_NAMESPACE_END
MVE_NAMESPACE_END
