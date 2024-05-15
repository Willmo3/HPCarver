/**
 * Common files for all implementations of HPCarver.
 * These are assumed to all run serially.
 *
 * Author: Will Morris
 */

#include <cassert>
#include "carver.h"

namespace carver {

Carver::Carver() {
    // Initialize image with tombstone hpimage.
    image = nullptr;
    energy = nullptr;
}

Carver::Carver(hpimage::Hpimage *image, Energy *energy) {
    // Invariant: external allocation must have initalized.
    assert(image->cols() == energy->cols());
    assert(image->rows() == energy->rows());

    this->image = image;
    this->energy = energy;

    assert_valid_dims();
}

void Carver::resize(uint32_t new_width, uint32_t new_height) {
    // Invariant: resized image must be shrunk from original.
    assert(new_width < image->cols() && new_height < image->rows());

    // Repeatedly vertically shrink it until it fits target width.
    while (image->cols() != new_width) {
        auto seam = vertical_seam();
        remove_vert_seam(seam);
    }

    // Now, repeatedly horizontally shrink until it fits target height.
    while (image->rows() != new_height) {
        auto seam = horiz_seam();
        remove_horiz_seam(seam);
    }
}

// Basic decomposition of seam removal algorithms identical regardless of threading impl.
std::vector<uint32_t> Carver::horiz_seam() {
    assert_valid_dims();
    horiz_energy();
    return Carver::min_horiz_seam();
}

std::vector<uint32_t> Carver::vertical_seam() {
    assert_valid_dims();
    vert_energy();
    return min_vert_seam();
}

// Assorted helpers

void Carver::assert_valid_dims() {
    if (image->cols() < 3) {
        std::cerr <<
                  "ERROR: Carving: minimum image width three"
                  << std::endl;
        exit(EXIT_FAILURE);
    } else if (image->rows() < 2) {
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
    auto green_diff = p1.green - p2.green;
    auto blue_diff = p1.blue - p2.blue;

    // Square differences w/o importing pow fn
    energy += red_diff * red_diff;
    energy += green_diff * green_diff;
    energy += blue_diff * blue_diff;

    return energy;
}

uint32_t Carver::pixel_energy(uint32_t col, uint32_t row) {
    assert(col < image->cols());
    assert(row < image->rows());

    // Casting here to allow the standard uint32_t API for external calls
    // While avoiding underflow with internal ones.
    auto signed_col = (int64_t) col;
    auto signed_row = (int64_t) row;

    uint32_t left_col = wrap_index(signed_col - 1, image->cols());
    uint32_t right_col = wrap_index(signed_col + 1, image->cols());
    uint32_t top_row = wrap_index(signed_row + 1, image->rows());
    uint32_t bottom_row = wrap_index(signed_row - 1, image->rows());

    hpimage::pixel left = image->get_pixel(left_col, row);
    hpimage::pixel right = image->get_pixel(right_col, row);
    hpimage::pixel top = image->get_pixel(col, top_row);
    hpimage::pixel bottom = image->get_pixel(col, bottom_row);

    return gradient_energy(left, right) + gradient_energy(top, bottom);
}

// NOTE: the largest fields of energy are heap-allocated.
// So passing around stack objects isn't a huge deal.
Energy *Carver::get_energy() {
    return energy;
}

hpimage::Hpimage *Carver::get_image() {
    return image;
}

} // End carver namespace.
