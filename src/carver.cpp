/**
 * Common files for all implementations of HPCarver.
 * These are assumed to all run serially.
 *
 * Author: Will Morris
 */

#include <cassert>
#include "carver.h"

using namespace carver;

void Carver::assert_valid_dims() {
    if (image.cols() < 3) {
        std::cerr <<
                  "ERROR: Carving: minimum image width three"
                  << std::endl;
        exit(EXIT_FAILURE);
    } else if (image.rows() < 2) {
        std::cerr <<
                  "ERROR: Carving: minimum image height three"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

uint32_t Carver::wrap_index(int64_t index, uint32_t length) {
    // Any negative value wraps around the other side.
    return (index + length) % length;
}

uint32_t Carver::gradient_energy(hpimage::pixel p1, hpimage::pixel p2) {
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

uint32_t Carver::pixel_energy(uint32_t col, uint32_t row) {
    assert(col >= 0 && col < image.cols());
    assert(row >= 0 && row < image.rows());

    // Casting here to allow the standard uint32_t API for external calls
    // While avoiding underflow with internal ones.
    auto signed_col = (int64_t) col;
    auto signed_row = (int64_t) row;

    uint32_t left_col = wrap_index(signed_col - 1, image.cols());
    uint32_t right_col = wrap_index(signed_col + 1, image.cols());
    uint32_t top_row = wrap_index(signed_row + 1, image.rows());
    uint32_t bottom_row = wrap_index(signed_row - 1, image.rows());

    hpimage::pixel left = image.get_pixel(left_col, row);
    hpimage::pixel right = image.get_pixel(right_col, row);
    hpimage::pixel top = image.get_pixel(col, top_row);
    hpimage::pixel bottom = image.get_pixel(col, bottom_row);

    return gradient_energy(left, right) + gradient_energy(top, bottom);
}

// NOTE: the largest fields of energy are heap-allocated.
// So passing around stack objects isn't a huge deal.
Energy *Carver::get_energy() {
    return &energy;
}

hpimage::Hpimage *Carver::get_image() {
    return &image;
}




