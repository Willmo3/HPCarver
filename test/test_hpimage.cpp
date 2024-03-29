//
// Created by morri2wj on 3/25/24.
//

#include <gtest/gtest.h>
#include "../src/hpimage/hpimage.h"

using namespace Magick;
using namespace hpcarver;

// Test that init works.
TEST(hpimage, init) {
    hpcarver::init();
}

TEST(hpimage, load_image) {
    hpcarver::init();
    hpcarver::load_image("img/3x4.png");
}
