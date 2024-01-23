/*
 * Copyright (C) 2015, Simon Fuhrmann
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef MVE_IMAGE_BASE_HEADER
#define MVE_IMAGE_BASE_HEADER

#include <cstdint>
#include <memory>
#include <vector>

#include "util/strings.h"
#include "mve/defines.h"

MVE_NAMESPACE_BEGIN

/**
 * Identifiers for image types.
 * WARNING: Do not change ordering or insert new types. These numbers
 * are stored in files and changing them will break compatibility.
 */
enum ImageType
{
    IMAGE_TYPE_UNKNOWN,
    /* Unsigned integer types. */
    IMAGE_TYPE_UINT8, // uint8_t, unsigned char
    IMAGE_TYPE_UINT16, // uint16_t
    IMAGE_TYPE_UINT32, // uint32_t, unsigned int
    IMAGE_TYPE_UINT64, // uint64_t
    /* Signed integer types. */
    IMAGE_TYPE_SINT8, // int8_t, char, signed char
    IMAGE_TYPE_SINT16, // int16_t
    IMAGE_TYPE_SINT32, // int32_t, int
    IMAGE_TYPE_SINT64, // int64_t
    /* Floating point types. */
    IMAGE_TYPE_FLOAT, // float
    IMAGE_TYPE_DOUBLE // double
};

/* ---------------------------------------------------------------- */

/**
 * Base class for images without type information.
 * This class basically provides width, height and channel
 * information and a framework for type information and data access.
 */
class ImageBase
{
public:
    typedef std::shared_ptr<ImageBase> Ptr;
    typedef std::shared_ptr<ImageBase const> ConstPtr;

public:
    /** Initializes members with 0. */
    ImageBase (void) = default;
    virtual ~ImageBase (void) = default;

    /** Duplicates the image. Data holders need to reimplement this. */
    virtual ImageBase::Ptr duplicate_base (void) const;

    /** Returns the width of the image. */
    int64_t width (void) const;
    /** Returns the height of the image. */
    int64_t height (void) const;
    /** Returns the amount of channels in the image. */
    int64_t channels (void) const;

    /** Returns false if one of width, height or channels is 0. */
    bool valid (void) const;

    /**
     * Re-interprets the dimensions of the image. This will fail and
     * return false if the total image size does not match the old image.
     */
    bool reinterpret (int64_t new_w, int64_t new_h, int64_t new_c);

    /** Generic byte size information. Returns 0 if not overwritten. */
    virtual std::size_t get_byte_size (void) const;
    /** Pointer to image data. Returns 0 if not overwritten. */
    virtual char const* get_byte_pointer (void) const;
    /** Pointer to image data. Returns 0 if not overwritten. */
    virtual char* get_byte_pointer (void);
    /** Value type information. Returns UNKNOWN if not overwritten. */
    virtual ImageType get_type (void) const;
    /** Returns a string representation of the image data type. */
    virtual char const* get_type_string (void) const;

    /** Returns the type for a valid type string, otherwise UNKNOWN. */
    static ImageType get_type_for_string (std::string const& type_string);

protected:
    int64_t w = 0;
    int64_t h = 0;
    int64_t c = 0;
};

/* ---------------------------------------------------------------- */

/**
 * Base class for images of arbitrary type. Image values are stored
 * in a standard STL Vector. Type information is provided. This class
 * makes no assumptions about the image structure, i.e. it provides no
 * pixel access methods.
 */
template <typename T>
class TypedImageBase : public ImageBase
{
public:
    typedef T ValueType;
    typedef std::shared_ptr<TypedImageBase<T> > Ptr;
    typedef std::shared_ptr<TypedImageBase<T> const> ConstPtr;
    typedef std::vector<T> ImageData;

public:
    /** Default constructor creates an empty image. */
    TypedImageBase (void) = default;

    /** Copy constructor duplicates another image. */
    TypedImageBase (TypedImageBase<T> const& other);

    virtual ~TypedImageBase (void) = default;

    /** Duplicates the image. Data holders need to reimplement this. */
    virtual ImageBase::Ptr duplicate_base (void) const;

    /** Allocates new image space, clearing previous content. */
    void allocate (int64_t width, int64_t height, int64_t chans);

    /**
     * Resizes the underlying image data vector.
     * Note: This leaves the existing/remaining image data unchanged.
     * Warning: If the image is shrunk, the data vector is resized but
     * may still consume the original amount of memory. Use allocate()
     * instead if the previous data is not important.
     */
    void resize (int64_t width, int64_t height, int64_t chans);

    /** Clears the image data from memory. */
    virtual void clear (void);

    /** Fills the data with a constant value. */
    void fill (T const& value);

    /** Swaps the contents of the images. */
    void swap (TypedImageBase<T>& other);

    /** Value type information by template specialization. */
    virtual ImageType get_type (void) const;
    /** Returns a string representation of the image data type. */
    char const* get_type_string (void) const;

    /** Returns the data vector for the image. */
    ImageData const& get_data (void) const;
    /** Returns the data vector for the image. */
    ImageData& get_data (void);

    /** Returns the data pointer. */
    T const* get_data_pointer (void) const;
    /** Returns the data pointer. */
    T* get_data_pointer (void);

    /** Returns data pointer to beginning. */
    T* begin (void);
    /** Returns const data pointer to beginning. */
    T const* begin (void) const;
    /** Returns data pointer to end. */
    T* end (void);
    /** Returns const data pointer to end. */
    T const* end (void) const;

    /** Returns the amount of pixels in the image (w * h). */
    int64_t get_pixel_amount (void) const;
    /** Returns the amount of values in the image (w * h * c). */
    int64_t get_value_amount (void) const;

    /** Returns the size of the image in bytes (w * h * c * BPV). */
    std::size_t get_byte_size (void) const;
    /** Returns the char pointer to the data. */
    char const* get_byte_pointer (void) const;
    /** Returns the char pointer to the data. */
    char* get_byte_pointer (void);

protected:
    ImageData data;
};

/* ================================================================ */

inline ImageBase::Ptr
ImageBase::duplicate_base (void) const
{
    return ImageBase::Ptr(new ImageBase(*this));
}

inline int64_t
ImageBase::width (void) const
{
    return this->w;
}

inline int64_t
ImageBase::height (void) const
{
    return this->h;
}

inline int64_t
ImageBase::channels (void) const
{
    return this->c;
}

inline bool
ImageBase::valid (void) const
{
    return this->w > 0 && this->h > 0 && this->c > 0;
}

inline bool
ImageBase::reinterpret (int64_t new_w, int64_t new_h, int64_t new_c)
{
    if (new_w * new_h * new_c != this->w * this->h * this->c)
        return false;

    this->w = new_w;
    this->h = new_h;
    this->c = new_c;
    return true;
}

inline std::size_t
ImageBase::get_byte_size (void) const
{
    return 0;
}

inline char const*
ImageBase::get_byte_pointer (void) const
{
    return nullptr;
}

inline char*
ImageBase::get_byte_pointer (void)
{
    return nullptr;
}

inline ImageType
ImageBase::get_type (void) const
{
    return IMAGE_TYPE_UNKNOWN;
}

inline char const*
ImageBase::get_type_string (void) const
{
    return "unknown";
}

inline ImageType
ImageBase::get_type_for_string (std::string const& type_string)
{
    if (type_string == "sint8")
        return IMAGE_TYPE_SINT8;
    else if (type_string == "sint16")
        return IMAGE_TYPE_SINT16;
    else if (type_string == "sint32")
        return IMAGE_TYPE_SINT32;
    else if (type_string == "sint64")
        return IMAGE_TYPE_SINT64;
    else if (type_string == "uint8")
        return IMAGE_TYPE_UINT8;
    else if (type_string == "uint16")
        return IMAGE_TYPE_UINT16;
    else if (type_string == "uint32")
        return IMAGE_TYPE_UINT32;
    else if (type_string == "uint64")
        return IMAGE_TYPE_UINT64;
    else if (type_string == "float")
        return IMAGE_TYPE_FLOAT;
    else if (type_string == "double")
        return IMAGE_TYPE_DOUBLE;

    return IMAGE_TYPE_UNKNOWN;
}

/* ---------------------------------------------------------------- */

template <typename T>
inline
TypedImageBase<T>::TypedImageBase (TypedImageBase<T> const& other)
    : ImageBase(other), data(other.data)
{
}

template <typename T>
inline ImageBase::Ptr
TypedImageBase<T>::duplicate_base (void) const
{
    return ImageBase::Ptr(new TypedImageBase<T>(*this));
}

template <typename T>
inline ImageType
TypedImageBase<T>::get_type (void) const
{
    return IMAGE_TYPE_UNKNOWN;
}

template <>
inline ImageType
TypedImageBase<int8_t>::get_type (void) const
{
    return IMAGE_TYPE_SINT8;
}

template <>
inline ImageType
TypedImageBase<int16_t>::get_type (void) const
{
    return IMAGE_TYPE_SINT16;
}

template <>
inline ImageType
TypedImageBase<int32_t>::get_type (void) const
{
    return IMAGE_TYPE_SINT32;
}

template <>
inline ImageType
TypedImageBase<int64_t>::get_type (void) const
{
    return IMAGE_TYPE_SINT64;
}

template <>
inline ImageType
TypedImageBase<uint8_t>::get_type (void) const
{
    return IMAGE_TYPE_UINT8;
}

template <>
inline ImageType
TypedImageBase<uint16_t>::get_type (void) const
{
    return IMAGE_TYPE_UINT16;
}

template <>
inline ImageType
TypedImageBase<uint32_t>::get_type (void) const
{
    return IMAGE_TYPE_UINT32;
}

template <>
inline ImageType
TypedImageBase<uint64_t>::get_type (void) const
{
    return IMAGE_TYPE_UINT64;
}

template <>
inline ImageType
TypedImageBase<float>::get_type (void) const
{
    return IMAGE_TYPE_FLOAT;
}

template <>
inline ImageType
TypedImageBase<double>::get_type (void) const
{
    return IMAGE_TYPE_DOUBLE;
}

template <typename T>
inline char const*
TypedImageBase<T>::get_type_string (void) const
{
    return util::string::for_type<T>();
}

template <typename T>
inline void
TypedImageBase<T>::allocate (int64_t width, int64_t height, int64_t chans)
{
    this->clear();
    this->resize(width, height, chans);
}

template <typename T>
inline void
TypedImageBase<T>::resize (int64_t width, int64_t height, int64_t chans)
{
    this->w = width;
    this->h = height;
    this->c = chans;
    this->data.resize(width * height * chans);
}

template <typename T>
inline void
TypedImageBase<T>::clear (void)
{
    this->w = 0;
    this->h = 0;
    this->c = 0;
    this->data.clear();
}

template <typename T>
inline void
TypedImageBase<T>::fill (T const& value)
{
    std::fill(this->data.begin(), this->data.end(), value);
}

template <typename T>
inline void
TypedImageBase<T>::swap (TypedImageBase<T>& other)
{
    std::swap(this->w, other.w);
    std::swap(this->h, other.h);
    std::swap(this->c, other.c);
    std::swap(this->data, other.data);
}

template <typename T>
inline typename TypedImageBase<T>::ImageData&
TypedImageBase<T>::get_data (void)
{
    return this->data;
}

template <typename T>
inline typename TypedImageBase<T>::ImageData const&
TypedImageBase<T>::get_data (void) const
{
    return this->data;
}

template <typename T>
inline T const*
TypedImageBase<T>::get_data_pointer (void) const
{
    if (this->data.empty())
        return nullptr;
    return &this->data[0];
}

template <typename T>
inline T*
TypedImageBase<T>::get_data_pointer (void)
{
    if (this->data.empty())
        return nullptr;
    return &this->data[0];
}

template <typename T>
inline T*
TypedImageBase<T>::begin (void)
{
    return this->data.empty() ? nullptr : &this->data[0];
}

template <typename T>
inline T const*
TypedImageBase<T>::begin (void) const
{
    return this->data.empty() ? nullptr : &this->data[0];
}

template <typename T>
inline T*
TypedImageBase<T>::end (void)
{
    return this->data.empty() ? nullptr : this->begin() + this->data.size();
}

template <typename T>
inline T const*
TypedImageBase<T>::end (void) const
{
    return this->data.empty() ? nullptr : this->begin() + this->data.size();
}

template <typename T>
inline int64_t
TypedImageBase<T>::get_pixel_amount(void) const
{
    return this->w * this->h;
}

template <typename T>
inline int64_t
TypedImageBase<T>::get_value_amount (void) const
{
    return this->get_pixel_amount() * this->c;
}

template <typename T>
inline std::size_t
TypedImageBase<T>::get_byte_size (void) const
{
    return this->data.size() * sizeof(T);
}

template <typename T>
inline char const*
TypedImageBase<T>::get_byte_pointer (void) const
{
    return reinterpret_cast<char const*>(this->get_data_pointer());
}

template <typename T>
inline char*
TypedImageBase<T>::get_byte_pointer (void)
{
    return reinterpret_cast<char*>(this->get_data_pointer());
}

MVE_NAMESPACE_END

/* ---------------------------------------------------------------- */

STD_NAMESPACE_BEGIN

/** Specialization of std::swap for efficient image swapping. */
template <class T>
inline void
swap (mve::TypedImageBase<T>& a, mve::TypedImageBase<T>& b)
{
    a.swap(b);
}

STD_NAMESPACE_END

#endif /* MVE_IMAGE_BASE_HEADER */
