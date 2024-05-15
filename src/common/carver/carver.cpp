/**
 * Common files for all implementations of HPCarver.
 * These are assumed to all run serially.
 *
 * Author: Will Morris
 */

#include <cassert>
#include "carver.h"

namespace carver {

// ***** STATIC HELPERS ***** //

/**
 * Calculate the gradient energy of a pixel.
 * This is the difference in RGB values squared.
 *
 * @param p1 Pixel to compare
 * @param p2 Pixel to compare
 * @return Gradient energy of the two pixels.
 */
static uint32_t gradient_energy(hpimage::pixel p1, hpimage::pixel p2) {
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

/**
 * Calculate a wrapped index over a dimension.
 * The formula is (index + length) % length
 * So that indexes of length will wrap to 0
 * And indexes of -1 will wrap to length -1.
 *
 * @param index Base index to wrap. Int64_t sizing prevents underflow.
 * @param length Length of dimension
 * @return The wrapped index
 */
static uint32_t wrap_index(int64_t index, uint32_t length) {
    // Any negative value wraps around the other side.
    return (index + length) % length;
}

// Private helper to get dimensions based on col, row.
static void assert_valid_dims(uint32_t cols, uint32_t rows) {
    if (cols < 3) {
        std::cerr <<
                  "ERROR: Carving: minimum image width three"
                  << std::endl;
        exit(EXIT_FAILURE);
    } else if (rows < 2) {
        std::cerr <<
                  "ERROR: Carving: minimum image height three"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ***** CLASS IMPLS ***** //

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
    // Check whether the resized image will fit.
    carver::assert_valid_dims(new_width, new_height);

    // Repeatedly vertically shrink it until it fits target width.
    while (image->cols() != new_width) {
        vert_energy();
        auto seam = min_vert_seam();
        remove_vert_seam(seam);
    }

    // Now, repeatedly horizontally shrink until it fits target height.
    while (image->rows() != new_height) {
        horiz_energy();
        auto seam = min_horiz_seam();
        remove_horiz_seam(seam);
    }
}

// Assorted helpers
void Carver::assert_valid_dims() const {
    // Just call the static helper with the number of cols we currently have.
    carver::assert_valid_dims(image->cols(), image->rows());
}


uint32_t Carver::pixel_energy(uint32_t col, uint32_t row) const {
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
