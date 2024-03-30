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

TEST(hpimage, get_pixels) {
    Image *img = test();
    Pixels *pixels = get_pixels(*img);
    delete img;
    delete pixels;
}
