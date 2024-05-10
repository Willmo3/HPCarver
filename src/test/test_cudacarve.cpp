// Special case for CUDA

#include <gtest/gtest.h>
#include "../cuda/image/cuda_image.h"
#include "../cuda/carver/cuda_energy.h"
#include "../carver/carver.h"

TEST(carver, vertical_carve) {
    auto image = hpc_cuda::CudaImage{"3x4.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto seam = std::vector<uint32_t>();
    seam.push_back(0);
    seam.push_back(1);
    seam.push_back(1);
    seam.push_back(2);

    carver.remove_vert_seam(seam);
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
    auto image = hpc_cuda::CudaImage{"3x4.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto seam = std::vector<uint32_t>();
    seam.push_back(0);
    seam.push_back(1);
    seam.push_back(2);

    carver.remove_horiz_seam(seam);
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
    auto image = hpc_cuda::CudaImage{"3x4.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    // Should wrap left, top around.
    // Difference will be (255 - 153) ** 2 * 2
    ASSERT_EQ(20808, carver.pixel_energy(0, 0));

    // Now, test a more standard, central energy w/ no wrapping.
    ASSERT_EQ(52225, carver.pixel_energy(1, 1));
}

TEST(carver, horiz_seam) {
    auto image = hpc_cuda::CudaImage{"3x4.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto seam = carver.horiz_seam();

    // The lowest energy horizontal seam should finish at index 2
    ASSERT_EQ(0, seam.at(0));
    ASSERT_EQ(0, seam.at(1));
    ASSERT_EQ(0, seam.at(2));
}

TEST(carver, vert_seam) {
    auto image = hpc_cuda::CudaImage{"3x4.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto seam = carver.vertical_seam();

    // The lowest energy vertical seam should finish at index 0.
    ASSERT_EQ(0, seam.at(0));
    ASSERT_EQ(0, seam.at(1));
    ASSERT_EQ(0, seam.at(2));
    ASSERT_EQ(0, seam.at(3));
}

TEST(carver, hard_vert_seam) {
    auto image = hpc_cuda::CudaImage{"6x5.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto seam = carver.vertical_seam();

    ASSERT_EQ(3, seam.at(0));
    ASSERT_EQ(4, seam.at(1));
    ASSERT_EQ(3, seam.at(2));
    ASSERT_EQ(2, seam.at(3));
    ASSERT_EQ(2, seam.at(4));
}

TEST(carver, hard_horiz_seam) {
    auto image = hpc_cuda::CudaImage{"6x5.ppm"};
    auto energy = hpc_cuda::CudaEnergy(image.cols(), image.rows());
    auto carver = carver::Carver(&image, &energy);

    auto seam = carver.horiz_seam();

    ASSERT_EQ(2, seam.at(0));
    ASSERT_EQ(2, seam.at(1));
    ASSERT_EQ(1, seam.at(2));
    ASSERT_EQ(2, seam.at(3));
    ASSERT_EQ(1, seam.at(4));
    ASSERT_EQ(2, seam.at(5));
}

