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
    auto *img = test();
    ASSERT_STREQ("PN6Magick5ImageE", typeid(img).name());
    delete img;
}

TEST(hpimage, get_pixels) {
    auto *img = test();
    size_t rows = img->rows();
    size_t cols = img->columns();

    auto *pixels = img->getPixels(0, 0, rows, cols);
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

TEST(hpimage, write_image) {
    Image *img = test();

    PixelPacket *pixels = img->getPixels(0, 0, 1, 1);
    pixels->red = 0;
    pixels->blue = 0;
    pixels->green = 0;
    img->syncPixels();

    auto test = std::string("test/junk/test.png");
    write_image(img, test);
    delete img;
}

TEST(hpimage, cut_image) {
    Image *img = test();
    cut_width(img); 
    ASSERT_EQ(2, img->columns());
    
    cut_height(img);
    ASSERT_EQ(3, img->rows());

    delete img;
}

TEST(hpimage, write_then_cut) {
    Image *img = test();

    PixelPacket *pixel = img->getPixels(1, 1, 1, 1);
    pixel->red = 0;
    pixel->blue = 0;
    pixel->green = 0;
    img->syncPixels();

    cut_width(img);
    cut_height(img);
    ASSERT_EQ(2, img->columns());
    ASSERT_EQ(3, img->rows());

    pixel = img->getPixels(1, 1, 1, 1);
    ASSERT_EQ(0, pixel->blue);

    auto test = std::string("test/junk/testsmall.png");
    write_image(img, test);

    delete img;
}
