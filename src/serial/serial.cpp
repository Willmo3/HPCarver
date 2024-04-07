//
// Created by morri2wj on 3/30/24.
//

#include <cassert>
#include "serial.h"

namespace serial {
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
