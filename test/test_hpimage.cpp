//
// Created by morri2wj on 3/25/24.
//

#include <gtest/gtest.h>
#include "../src/hpimage/hpimage.h"

using namespace Magick;
using namespace hpcarver;

// Testing helper.
Image *test() {
    hpcarver::init();
    return hpcarver::load_image("/nfs/home/morri2wj/research/hpcarver/img/3x4.png");
}

// Test that init works.
TEST(hpimage, init) {
    delete test();
}

TEST(hpimage, load_image) {
    Image *img = test();
    ASSERT_STREQ("PN6Magick5ImageE", typeid(img).name());
    delete img;
}

TEST(hpimage, get_pixels) {
    Image *img = test();
    size_t rows = img->rows();
    size_t cols = img->columns();

    PixelPacket *pixels = img->getPixels(0, 0, rows, cols);
    // Set the upper pixel to 0,0
    pixels->red = 0;
    pixels->blue = 0;
    pixels->green = 0;

    img->syncPixels();

    pixels = img->getPixels(0, 0, 1, 1);
    ASSERT_EQ(0, pixels->red);
    ASSERT_EQ(0, pixels->blue);
    ASSERT_EQ(0, pixels->green);
    
    delete img;
}
