#include "cv_utils.hpp"
#include <array>
#include <cctype>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>
#include <vector>

namespace {
std::uint16_t readBigEndian16(std::istream &stream) {
    std::array<unsigned char, 2> bytes{};
    stream.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    if (!stream)
        throw std::runtime_error("Unexpected end of image header");
    return static_cast<std::uint16_t>((bytes[0] << 8) | bytes[1]);
}

std::uint32_t readBigEndian32(const unsigned char *bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24) | (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) | static_cast<std::uint32_t>(bytes[3]);
}

std::uint16_t tiff16(const unsigned char *bytes, bool little) {
    return little ? static_cast<std::uint16_t>(bytes[0] | (bytes[1] << 8))
                  : static_cast<std::uint16_t>((bytes[0] << 8) | bytes[1]);
}

std::uint32_t tiff32(const unsigned char *bytes, bool little) {
    if (little)
        return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
               (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
    return readBigEndian32(bytes);
}

int exifOrientation(const std::vector<unsigned char> &payload) {
    if (payload.size() < 14 || std::memcmp(payload.data(), "Exif\0\0", 6) != 0)
        return 1;
    const unsigned char *tiff = payload.data() + 6;
    const std::size_t size = payload.size() - 6;
    bool little;
    if (tiff[0] == 'I' && tiff[1] == 'I')
        little = true;
    else if (tiff[0] == 'M' && tiff[1] == 'M')
        little = false;
    else
        return 1;
    if (tiff16(tiff + 2, little) != 42)
        return 1;
    std::uint32_t offset = tiff32(tiff + 4, little);
    if (offset > size - 2)
        return 1;
    std::uint16_t count = tiff16(tiff + offset, little);
    std::size_t entries = static_cast<std::size_t>(offset) + 2;
    if (count > (size - entries) / 12)
        return 1;
    for (std::uint16_t i = 0; i < count; ++i) {
        const unsigned char *entry = tiff + entries + static_cast<std::size_t>(i) * 12;
        if (tiff16(entry, little) == 0x0112 && tiff16(entry + 2, little) == 3 && tiff32(entry + 4, little) == 1) {
            int value = tiff16(entry + 8, little);
            return value >= 1 && value <= 8 ? value : 1;
        }
    }
    return 1;
}

bool jpegStartOfFrame(unsigned char marker) {
    return (marker >= 0xc0 && marker <= 0xc3) || (marker >= 0xc5 && marker <= 0xc7) ||
           (marker >= 0xc9 && marker <= 0xcb) || (marker >= 0xcd && marker <= 0xcf);
}

cv::Size jpegDimensions(std::istream &stream) {
    cv::Size dimensions;
    int orientation = 1;
    while (stream) {
        int prefix = stream.get();
        if (prefix != 0xff)
            continue;
        int marker = stream.get();
        while (marker == 0xff)
            marker = stream.get();
        if (marker < 0 || marker == 0xd9 || marker == 0xda)
            break;
        if (marker == 0x01 || (marker >= 0xd0 && marker <= 0xd8))
            continue;
        std::uint16_t length = readBigEndian16(stream);
        if (length < 2)
            break;
        if (jpegStartOfFrame(static_cast<unsigned char>(marker))) {
            if (length < 7)
                break;
            stream.get();
            int height = readBigEndian16(stream), width = readBigEndian16(stream);
            if (width > 0 && height > 0)
                dimensions = {width, height};
            stream.seekg(static_cast<std::streamoff>(length - 7), std::ios::cur);
        } else if (marker == 0xe1) {
            std::vector<unsigned char> payload(length - 2);
            stream.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(payload.size()));
            if (!stream)
                break;
            int parsedOrientation = exifOrientation(payload);
            if (parsedOrientation != 1)
                orientation = parsedOrientation;
        } else {
            stream.seekg(static_cast<std::streamoff>(length - 2), std::ios::cur);
        }
    }
    if (orientation >= 5 && orientation <= 8 && dimensions.width > 0)
        std::swap(dimensions.width, dimensions.height);
    return dimensions;
}

std::string pnmToken(std::istream &stream) {
    std::string token;
    while (stream) {
        int next = stream.peek();
        if (next == '#') {
            stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }
        if (next >= 0 && std::isspace(static_cast<unsigned char>(next))) {
            stream.get();
            continue;
        }
        break;
    }
    while (stream) {
        int next = stream.peek();
        if (next < 0 || next == '#' || std::isspace(static_cast<unsigned char>(next)))
            break;
        token.push_back(static_cast<char>(stream.get()));
    }
    return token;
}
} // namespace

cv::Mat imreadRGB(const std::string &filename) {
    cv::Mat cImg = cv::imread(filename);

    if (cImg.empty())
        throw std::runtime_error("Cannot read image " + filename + "; make sure the path is correct");

    cv::cvtColor(cImg, cImg, cv::COLOR_BGR2RGB);
    return cImg;
}

cv::Size imageDimensions(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot read image " + filename + "; make sure the path is correct");
    std::array<unsigned char, 24> header{};
    stream.read(reinterpret_cast<char *>(header.data()), header.size());
    std::streamsize count = stream.gcount();
    if (count >= 2 && header[0] == 0xff && header[1] == 0xd8) {
        stream.clear();
        stream.seekg(2);
        cv::Size size = jpegDimensions(stream);
        if (size.width > 0 && size.height > 0)
            return size;
    }
    static constexpr std::array<unsigned char, 8> pngSignature{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    if (count >= 24 && std::equal(pngSignature.begin(), pngSignature.end(), header.begin()) &&
        readBigEndian32(header.data() + 8) == 13 && header[12] == 'I' && header[13] == 'H' &&
        header[14] == 'D' && header[15] == 'R') {
        std::uint32_t width = readBigEndian32(header.data() + 16), height = readBigEndian32(header.data() + 20);
        if (width > 0 && height > 0 && width <= INT_MAX && height <= INT_MAX)
            return {static_cast<int>(width), static_cast<int>(height)};
    }
    if (count >= 2 && header[0] == 'P' && header[1] >= '1' && header[1] <= '6') {
        stream.clear();
        stream.seekg(0);
        pnmToken(stream);
        std::string width = pnmToken(stream), height = pnmToken(stream);
        if (!width.empty() && !height.empty()) {
            long long parsedWidth = std::stoll(width), parsedHeight = std::stoll(height);
            if (parsedWidth > 0 && parsedHeight > 0 && parsedWidth <= INT_MAX && parsedHeight <= INT_MAX)
                return {static_cast<int>(parsedWidth), static_cast<int>(parsedHeight)};
        }
    }
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (image.empty())
        throw std::runtime_error("Cannot read image " + filename + "; make sure the path is correct");
    return image.size();
}

cv::Mat tensorToImage(const torch::Tensor &t) {
    int h = t.sizes()[0];
    int w = t.sizes()[1];
    int c = t.sizes()[2];

    int type = CV_8UC3;
    if (c != 3)
        throw std::runtime_error("Only images with 3 channels are supported");

    cv::Mat image(h, w, type);
    torch::Tensor scaledTensor =
        t.scalar_type() == torch::kUInt8 ? t.contiguous() : (t * 255.0).toType(torch::kU8).contiguous();
    uint8_t *dataPtr = static_cast<uint8_t *>(scaledTensor.data_ptr());
    std::copy(dataPtr, dataPtr + (w * h * c), image.data);

    return image;
}

torch::Tensor imageToTensor(const cv::Mat &image) {
    torch::Tensor img = torch::from_blob(image.data, {image.rows, image.cols, image.dims + 1}, torch::kU8);
    return (img.toType(torch::kFloat32) / 255.0f);
}
