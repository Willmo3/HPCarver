//
// Created by morri2wj on 3/30/24.
//

#include <cassert>
#include "serial.h"
#include "../hpimage/hpimage.h"

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
 * Given an image, return the minimum energy vertical seam
 * @param image image to process
 * @return Minimum energy vertical seam
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
void remove_horiz_seam(Magick::Image &image, std::vector<uint32_t> &seam) {
  // Must be exactly one row to remove from each column.
  assert(seam.size() == image.columns());

    for (auto col = 0; col < image.columns(); col++) {
        auto index = seam[col];
        assert(index >= 0 && index < image.rows());

        // How many pixels are below our index?
        auto num_pixels = image.rows() - index;
        Magick::PixelPacket *pixels = image.getPixels(index, col, 1, num_pixels);

        // Shift all columns below this up one.
        for (auto i = 0; i < num_pixels - 1; i++) {
            pixels[i].red = pixels[i + 1].red;
            pixels[i].blue = pixels[i + 1].blue;
            pixels[i].green = pixels[i + 1].green;
            pixels[i].opacity = pixels[i + 1].opacity;
        }
        image.syncPixels();
    }
    hpcarver::cut_height(&image);
}

/**
 * Remove a vertical seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_vert_seam(Magick::Image &image, std::vector<uint32_t> &seam) {
    // Must be exactly one column to remove from each row.
    assert(seam.size() == image.rows());

    // Shift every pixel after a given image over.
    // Then reduce image size by one.
    for (auto row = 0; row < image.rows(); row++) {
        auto index = seam[row];
        assert(index >= 0 && index < image.columns());

        // To minimize the number of pizels gotten at each attempt, only get up to index.
        auto num_pixels = image.columns() - index;
        Magick::PixelPacket *pixels = image.getPixels(index, row, num_pixels, 1);

        // Now, shift all pixels in the buffer back one.
        for (auto i = 0; i < num_pixels - 1 ; i++) {
            // For now, only copying the fields we know are relevant
            pixels[i].red = pixels[i + 1].red;
            pixels[i].blue = pixels[i + 1].blue;
            pixels[i].green = pixels[i + 1].green;
            pixels[i].opacity = pixels[i + 1].opacity;
        }
        image.syncPixels();
    }
    hpcarver::cut_width(&image);
}
} // End serial namespace
