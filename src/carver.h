//
// Created by morri2wj on 3/30/24.
//

#ifndef HPCARVER_CARVER_H
#define HPCARVER_CARVER_H

#include "../HPImage/hpimage.h"
#include <vector>

namespace carver {
/**
 * Given an image, return the minimum energy horizontal seam
 * @param image image to process
 * @return Minimum energy horizontal seam
 */
std::vector<uint32_t> *horiz_seam(hpimage::Hpimage &image);

/**
 * Given an image, return the minimum energy vertical seam
 * @param image image to process
 * @return Minimum energy vertical seam
 */
std::vector<uint32_t> *vertical_seam(hpimage::Hpimage &image);

/**
 * Get the base energy of a single pixel.
 * This is calculated using an energy gradient approach
 * considering the differences of adjacent colors
 *
 * @param image Image to calculate energy
 * @param row Row of pixel whose energy we're calculating
 * @param col Column of pixel whose energy we're calculating
 * @return the energy
 */
uint32_t pixel_energy(hpimage::Hpimage &image, int32_t col, int32_t row);

/**
 * Remove a horizontal seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_horiz_seam(hpimage::Hpimage &image, std::vector<uint32_t> &seam);

/**
 * Remove a vertical seam from the image.
 * Updates the given image object.
 *
 * @param image Image to remove seam from.
 * @param seam Seam to remove.
 */
void remove_vert_seam(hpimage::Hpimage &image, std::vector<uint32_t> &seam);
} // namespace carver

#endif //HPCARVER_CARVER_H
