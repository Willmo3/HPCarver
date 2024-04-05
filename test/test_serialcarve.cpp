//
// Created by morri2wj on 4/5/24.
//

#include <gtest/gtest.h>
#include "../src/hpimage/hpimage.h"
#include "../src/serial/serial.h"

using namespace Magick;
using namespace hpcarver;

// Testing helper.
Image *test() {
    hpcarver::init();
    return hpcarver::load_image("/nfs/home/morri2wj/research/hpcarver/img/3x4.png");
}

TEST(serialcarve, vertical_carve) {
    auto image = test();
    auto seam = std::vector<uint32_t>();
    seam.push_back(0);
    seam.push_back(1);
    seam.push_back(1);
    seam.push_back(2);

    serial::remove_vert_seam(*image, seam);

    auto test = std::string("test/junk/testseam.png");
    hpcarver::write_image(image, test);
    delete image;
}

