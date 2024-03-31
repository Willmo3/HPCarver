//
// Created by morri2wj on 3/30/24.
//

#include "serial.h"

namespace serial {
/**
 * Given an image, return the minimum energy horizontal seam
 * @param image image to process
 * @return Minimum energy horizontal seam
 */
std::vector<uint32_t> *horiz_seam(Magick::Image &image) {
    return new std::vector<uint32_t>{};
}

/**
 * Given an image, return the minimum energy horizontal seam
 * @param image image to process
 * @return Minimum energy horizontal seam
 */
std::vector<uint32_t> *vertical_seam(Magick::Image &image) {
    return new std::vector<uint32_t>{};
}

/**
 * Remove a horizontal seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_horiz_seam(Magick::Image &image, std::vector<uint32_t> &seam) {}

/**
 * Remove a vertical seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_vert_seam(Magick::Image &image, std::vector<uint32_t> &seam) {}
} // End serial namespace