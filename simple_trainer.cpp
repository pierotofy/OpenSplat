#include <iostream>
#include <torch/torch.h>
#include <mve/image_io.h>

using namespace torch::indexing;

mve::ByteImage::Ptr tensorToImage(const torch::Tensor &t){
    int w = t.sizes()[1];
    int h = t.sizes()[0];
    int c = t.sizes()[2];

    mve::ByteImage::Ptr image = mve::ByteImage::create(w, h, c);
    uint8_t *dataPtr = static_cast<uint8_t *>((t * 255.0).toType(torch::kU8).data_ptr());
    
    std::copy(dataPtr, dataPtr + (w * h * c), image->get_data().data());

    return image;
}

int main(int argc, char **argv){
    int width = 256,
        height = 256;
    int numPoints = 100000;
    int iterations = 1000;
    float learningRate = 0.01;

    // Test image
    // Top left red
    // Bottom right blue
    torch::Tensor gtImage = torch::ones({height, width, 3});
    gtImage.index_put_({Slice(None, height / 2), Slice(None, width / 2), Slice()}, torch::tensor({1.0, 0.0, 0.0}));
    gtImage.index_put_({Slice(height / 2, None), Slice(width / 2, None), Slice()}, torch::tensor({0.0, 0.0, 1.0}));

    mve::ByteImage::Ptr image = tensorToImage(gtImage);
    mve::image::save_file(image, "test.png");
}