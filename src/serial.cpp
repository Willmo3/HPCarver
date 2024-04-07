//
// Created by morri2wj on 3/30/24.
//

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "carver.h"
#include "energy.h"

namespace carver {

uint32_t wrap_index(int32_t index, uint32_t length);
uint32_t gradient_energy(hpimage::pixel p1, hpimage::pixel p2);

/**
 * Given an image, return the minimum energy horizontal seam
 * @param image image to process
 * @return Minimum energy horizontal seam
 */
std::vector<uint32_t> *horiz_seam(hpimage::Hpimage &image) {
    if (image.cols() < 3) {
        std::cerr <<
            "ERROR: HPCarver does not support horizontal carving on an image of less than width three"
            << std::endl;
        exit(EXIT_FAILURE);
    } else if (image.rows() < 2) {
        std::cerr <<
              "ERROR: HPCarver does not support horizontal carving on an image of less than height two"
              << std::endl;
        exit(EXIT_FAILURE);
    }

    // At the end, pick any of the pixels with minimal memoized energy. Then, find the lowest energy
    // Adjacent pixel in the next row. Repeat until the end, adding to the seam vector.

    // Return the reversed vector.

    // Generate energy matrix
    auto energy = carver::Energy(image.cols(), image.rows());
    // Horizontal seam direction: left to right.
    // Prime memo structure with base energies of first pixel column.
    for (auto row = 0; row < energy.rows(); ++row) {
        energy.set_energy(0, row, pixel_energy(image, 0, row));
    }

    // Now set energy to minimum of three neighbors.
    for (auto col = 1; col < energy.cols(); ++col) {
        for (auto row = 0; row < energy.rows(); ++row) {
            // No wrapping
            auto neighbor_energies = energy.get_left_predecessors(col, row);

            // Energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(image, col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy.set_energy(col, row, local_energy);
        }
    }

    // Now, prime the reverse traversal with the minimum above energy.
    uint32_t back_col = energy.cols() - 1;
    auto *seam = new std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy.
    // Invariant: there will be at least two rows to consider.
    uint32_t min_index = 0;
    uint32_t min_energy = energy.get_energy(back_col, 0);

    for (auto row = 1; row < energy.rows(); ++row) {
        uint32_t current_energy = energy.get_energy(back_col, row);
        if (current_energy < min_energy) {
            min_index = row;
            min_energy = current_energy;
        }
    }

    seam->push_back(min_index);
    return seam;
}

/**
 * Given an image, return the minimum energy vertical seam
 * @param image image to process
 * @return Minimum energy vertical seam
 */
std::vector<uint32_t> *vertical_seam(hpimage::Hpimage &image) {
    if (image.rows() < 3) {
        std::cerr <<
              "ERROR: HPCarver does not support vertical carving on an image of less than height three"
              << std::endl;
        exit(EXIT_FAILURE);
    } else if (image.cols() < 2) {
        std::cerr <<
              "ERROR: HPCarver does not support vertical carving on an image of less than width two"
              << std::endl;
        exit(EXIT_FAILURE);
    }

    auto energy = carver::Energy(image.cols(), image.rows());
    // Vertical seam direction: top to bottom
    // Prime memo structure with base energies of first pixel row.
    for (auto col = 0; col < energy.cols(); ++col) {
        energy.set_energy(col, 0, pixel_energy(image, col, 0));
    }

    for (auto row = 1; row < energy.rows(); ++row) {
        for (auto col = 0; col < energy.cols(); ++col) {
            // Note: no wrapping in seams!
            auto neighbor_energies = energy.get_top_predecessors(col, row);

            // energy = local energy + min(neighbors)
            uint32_t local_energy = pixel_energy(image, col, row);
            local_energy += *std::min_element(neighbor_energies.begin(), neighbor_energies.end());
            energy.set_energy(col, row, local_energy);
        }
    }

    // Now, prime the reverse traversal with the minimum above energy.
    uint32_t bottom_row = energy.rows() - 1;
    auto *seam = new std::vector<uint32_t>{};

    // Default: row 0 of the last column contains the minimum energy.
    // Invariant: there will be at least two rows to consider.
    uint32_t min_index = 0;
    uint32_t min_energy = energy.get_energy(0, bottom_row);

    for (auto col = 1; col < energy.cols(); ++col) {
        uint32_t current_energy = energy.get_energy(col, bottom_row);
        if (current_energy < min_energy) {
            min_index = col;
            min_energy = current_energy;
        }
    }

    seam->push_back(min_index);
    return seam;
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
} // End carver namespace
