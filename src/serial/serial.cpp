//
// Created by morri2wj on 3/30/24.
//

#include <cassert>
#include "serial.h"
#include "../../HPImage/hpimage.h"

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
 * Given an image, return the minimum energy horizontal seam
 * @param image image to process
 * @return Minimum energy horizontal seam
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
    assert(seam.size() == image.num_cols());

    for (auto col = 0; col < image.num_cols(); ++col) {
        auto index = seam[col];
        assert(index >= 0 && index < image.num_rows());

        // Shift all columns below this up one.
        for (auto row = index; row < image.num_rows() - 1; ++row) {
            image.set_pixel(col, row, image.get_pixel(col, row + 1));
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
    assert(seam.size() == image.num_rows());

    // Shift every pixel after a given image over.
    // Then reduce image size by one.
    for (auto row = 0; row < image.num_rows(); ++row) {
        auto index = seam[row];
        assert(index >= 0 && index < image.num_cols());

        // Now, shift all pixels in the buffer back one.
        for (auto col = 0; col < image.num_cols() - 1; ++col) {
            image.set_pixel(col, row, image.get_pixel(col + 1, row));
        }
    }

    // Finally, with all pixels shifted over, time to trim the image!
    image.cut_col();
}
} // End serial namespace
