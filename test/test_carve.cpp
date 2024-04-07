//
// Created by morri2wj on 4/5/24.
//

#include <gtest/gtest.h>
#include "../HPImage/hpimage.h"
#include "../src/carver.h"

// Testing helper.
hpimage::Hpimage test() {
    return hpimage::Hpimage{"3x4.ppm"};
}

TEST(carver, vertical_carve) {
    auto image = test();
    auto seam = std::vector<uint32_t>();
    seam.push_back(0);
    seam.push_back(1);
    seam.push_back(1);
    seam.push_back(2);

    carver::remove_vert_seam(image, seam);
    // Dimensions should be modified.
    ASSERT_EQ(2, image.cols());
    ASSERT_EQ(4, image.rows());
    // Now, check that each seam removal has occured as expected.
    ASSERT_EQ(153, image.get_pixel(0, 0).blue);
    ASSERT_EQ(255, image.get_pixel(1, 0).blue);
    ASSERT_EQ(255, image.get_pixel(1, 1).blue);

    ASSERT_EQ(255, image.get_pixel(1, 2).blue);

    ASSERT_EQ(153, image.get_pixel(1, 3).blue);
}

TEST(carver, horizontal_carve) {
    auto image = test();
    auto seam = std::vector<uint32_t>();
    seam.push_back(0);
    seam.push_back(1);
    seam.push_back(2);

    carver::remove_horiz_seam(image, seam);
    ASSERT_EQ(3, image.cols());
    ASSERT_EQ(3, image.rows());

    ASSERT_EQ(153, image.get_pixel(0, 0).green);
    ASSERT_EQ(203, image.get_pixel(0, 1).green);
    ASSERT_EQ(255, image.get_pixel(0, 2).green);

    ASSERT_EQ(204, image.get_pixel(1, 1).green);
    ASSERT_EQ(255, image.get_pixel(1, 2).green);

    ASSERT_EQ(255, image.get_pixel(2, 2).green);
}

TEST(carver, energy) {
    auto image = test();

    // Should wrap left, top around.
    // Difference will be (255 - 153) ** 2 * 2
    ASSERT_EQ(20808, carver::pixel_energy(image, 0, 0));

    // Now, test a more standard, central energy w/ no wrapping.
    ASSERT_EQ(52225, carver::pixel_energy(image, 1, 1));
}

TEST(carver, horiz_seam) {
    auto image = test();
    // For now, let's just make sure this bad boy doesn't break!
    auto seam = carver::horiz_seam(image);
    
    ASSERT_EQ(2, seam->at(0));
}

