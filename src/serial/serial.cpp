//
// Created by morri2wj on 3/30/24.
//

#include <cassert>
#include "../serial.h"

namespace serial {

uint32_t wrap_index(int32_t index, uint32_t length);
uint32_t gradient_energy(hpimage::pixel p1, hpimage::pixel p2);

/**
 * Given an image, return the minimum energy horizontal seam
 * @param image image to process
 * @return Minimum energy horizontal seam
 */
std::vector<uint32_t> *horiz_seam(hpimage::Hpimage &image) {
    return new std::vector<uint32_t>{};
}

/**
 * Given an image, return the minimum energy vertical seam
 * @param image image to process
 * @return Minimum energy vertical seam
 */
std::vector<uint32_t> *vertical_seam(hpimage::Hpimage &image) {
    return new std::vector<uint32_t>{};
}

/**
 * Get the base energy of a single pixel.
 * This is calculated using an energy gradient approach
 * considering the differences of adjacent colors
 *
 * @param image Image to calculate energy
 * @return the energy
 */
uint32_t pixel_energy(hpimage::Hpimage &image, int32_t col, int32_t row) {
    assert(col >= 0 && col < image.cols());
    assert(row >= 0 && row < image.rows());

    uint32_t left_col = wrap_index(col - 1, image.cols());
    uint32_t right_col = wrap_index(col + 1, image.cols());
    uint32_t top_row = wrap_index(row + 1, image.rows());
    uint32_t bottom_row = wrap_index(row - 1, image.rows());

    hpimage::pixel left = image.get_pixel(left_col, row);
    hpimage::pixel right = image.get_pixel(right_col, row);
    hpimage::pixel top = image.get_pixel(col, top_row);
    hpimage::pixel bottom = image.get_pixel(col, bottom_row);

    return gradient_energy(left, right) + gradient_energy(top, bottom);
}

/**
 * Calculate the gradient energy of two pixels.
 * This is simply the energy difference of their corresponding RGB color fields
 * Squared to ensure positivity and penalize small differences.
 *
 * @param p1 First pixel to consider.
 * @param p2 Second pixel to consider.
 * @return The gradient energy.
 */
uint32_t gradient_energy(hpimage::pixel p1, hpimage::pixel p2) {
    auto energy = 0;

    auto red_diff = p1.red - p2.red;
    auto green_diff= p1.green - p2.green;
    auto blue_diff = p1.blue - p2.blue;

    // Square differences w/o importing pow fn
    energy += red_diff * red_diff;
    energy += green_diff * green_diff;
    energy += blue_diff * blue_diff;

    return energy;
}

/**
 * Calculate a wrapped index over a dimension.
 * The formula is (index + length) % length
 * So that indexes of length will wrap to 0
 * And indexes of -1 will wrap to length -1.
 *
 * @param index Base index to wrap.
 * @param length Length of dimension
 * @return The wrapped index
 */
uint32_t wrap_index(int32_t index, uint32_t length) {
    // Any negative value wraps around the other side.
    return (index + length) % length;
}

/**
 * Remove a horizontal seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_horiz_seam(hpimage::Hpimage &image, std::vector<uint32_t> &seam) {
    // Must be exactly one row to remove from each column.
    assert(seam.size() == image.cols());

    for (auto col = 0; col < image.cols(); ++col) {
        auto index = seam[col];
        assert(index >= 0 && index < image.rows());

        // Shift all pixels below this up one.
        for (auto row = index; row < image.rows() - 1; ++row) {
            hpimage::pixel below = image.get_pixel(col, row + 1);
            image.set_pixel(col, row, below);
        }
    }
    // Finally, cut the last row from the pixel.
    image.cut_row();
}

/**
 * Remove a vertical seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_vert_seam(hpimage::Hpimage &image, std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image.rows());

    // Shift every pixel after a given image over.
    // Then reduce image size by one.
    for (auto row = 0; row < image.rows(); ++row) {
        auto index = seam[row];
        assert(index >= 0 && index < image.cols());

        // Shift all pixels after this one back
        for (auto col = index; col < image.cols() - 1; ++col) {
            hpimage::pixel next = image.get_pixel(col + 1, row);
            image.set_pixel(col, row, next);
        }
    }
    // Finally, with all pixels shifted over, time to trim the image!
    image.cut_col();
}
} // End serial namespace
