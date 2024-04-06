//
// Created by morri2wj on 4/5/24.
//

#include <gtest/gtest.h>
#include "../../HPImage/hpimage.h"
#include "../src/serial/serial.h"

// Testing helper.
hpimage::Hpimage test() {
    return hpimage::Hpimage("3x4.ppm");
}

TEST(serialcarve, vertical_carve) {
    auto image = test();
    auto seam = std::vector<uint32_t>();
    seam.push_back(0);
    seam.push_back(1);
    seam.push_back(1);
    seam.push_back(2);

    serial::remove_vert_seam(image, seam);

    image.write_image("testseam.png");
}

