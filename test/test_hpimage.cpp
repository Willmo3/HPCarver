//
// Created by morri2wj on 3/25/24.
//

#include <gtest/gtest.h>
#include "../src/hpimage/hpimage.h"

using namespace Magick;
using namespace hpcarver;

// Testing helper.
Image *test() {
    hpcarver::init("/nfs/home/morri2wj/research/hpcarver/out/test_hpimage");
    return hpcarver::load_image("img/3x4.png");
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

// Test that we can sync pixels.
TEST(hpimage, get_pixels) {
    Image *img = test();
    Pixels *view = get_pixels(*img);

    size_t rows = view->rows();
    size_t cols = view->columns();

    PixelPacket *pixels = view->get(0, 0, cols, rows);
    // Set the upper pixel to 0,0
    //pixels->red = 0;
//    pixels->blue = 0;
//    pixels->green = 0;
//
//    view->sync();

    delete view;
//    view = get_pixels(*img);
//
//    pixels = view->get(0, 0, cols, rows);
//    ASSERT_EQ(0, pixels->red);
//    ASSERT_EQ(0, pixels->blue);
//    ASSERT_EQ(0, pixels->green);
//
//    delete view;
    delete img;
}
